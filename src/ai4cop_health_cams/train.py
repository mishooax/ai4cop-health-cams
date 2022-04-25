from typing import Callable, Mapping, Any, Optional
import argparse
import os
import pickle

import xarray as xr
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import albumentations as A

from ai4cop_health_cams.config import YAMLConfig
from ai4cop_health_cams.scalers import Log1pDataScaler, StdScaler, SimpleScaler, DataTransformer, Log1pStdScaler
from ai4cop_health_cams.trainer import Lightning_Model

from ai4cop_health_cams.wgan_gp_trainer import Lightning_WGAN_GP
from ai4cop_health_cams.rgan_trainer import Lightning_RGAN
from ai4cop_health_cams.swin_trainer import Lightning_SwinIR
from ai4cop_health_cams.unet_trainer import Lightning_UNet, Lightning_UNetAE
from ai4cop_health_cams.utils import build_generator_and_discriminator, build_unet, build_xnet, build_swin_transformer, get_datasets
from ai4cop_health_cams.logger import get_logger

LOGGER = get_logger(__name__)


def get_scalers(
    pretrained: bool = False,
    store_path: Optional[str] = None,
) -> Mapping[str, DataTransformer]:
    if pretrained:
        # read scalers used for the pre-trained generator
        LOGGER.debug("Reading pre-trained scaler objects from %s ...", store_path)
        with open(store_path, "rb") as f:
            scalers: Mapping[str, DataTransformer] = pickle.load(f)
        return scalers

    pm2p5_lr_scaler = Log1pDataScaler("pm2p5")
    pm2p5_hr_scaler = Log1pDataScaler(name="pm2p5")

    return {
        # weather
        "sp_LR": StdScaler(name="sp"),
        "u10_LR": StdScaler(name="u10"),
        "v10_LR": StdScaler(name="v10"),
        "t2m_LR": StdScaler(name="t2m"),
        "blh_LR": StdScaler(name="blh"),
        "tp_LR": Log1pDataScaler(name="tp"),
        "msl_LR": StdScaler(name="msl"),
        "d2m_LR": StdScaler(name="d2m"),
        # constant fields
        "built_frac_HR": SimpleScaler(name="built_frac"),
        "urban_frac_HR": SimpleScaler(name="urban_frac"),
        "orog_scal_HR": SimpleScaler(name="orog_scal"),
        "lsm_HR": SimpleScaler(name="lsm"),
        # chemical species: all log-scaled
        "no_LR": Log1pDataScaler(name="no"),  # µg/m3
        "no2_LR": Log1pDataScaler(name="no2"),  # µg/m3
        "nh3_LR": Log1pDataScaler(name="nh3"),  # µg/m3
        "dust_LR": Log1pDataScaler(name="dust"),  # µg/m3
        "so2_LR": Log1pDataScaler(name="so2"),  # µg/m3
        "co_LR": Log1pDataScaler(name="co"),  # µg/m3
        "nmvoc_LR": Log1pDataScaler(name="nmvoc"),  # µg/m3
        "o3_LR": Log1pDataScaler(name="o3"),  # µg/m3
        "pm10_LR": Log1pDataScaler(name="pm10"),  # µg/m3
        # "pm2p5_LR": Log1pDataScaler(name="pm2p5"),  # µg/m3
        # "pm2p5_HR": Log1pDataScaler(name="pm2p5"),  # µg/m3
        "pm2p5_LR": pm2p5_lr_scaler,  # µg/m3
        "pm2p5_HR": pm2p5_hr_scaler,  # µg/m3
    }


def build_inverse_transformation(scalers: Mapping[str, DataTransformer]) -> Callable:
    def inverse_transformation(resolution: str, data: np.ndarray) -> np.ndarray:
        return scalers[f"pm2p5_{resolution.upper()}"].inverse_transform(data)

    return inverse_transformation


def train(
    model_type: str,
    config: Mapping[str, Any],
    interactive: bool = False,
    pretrained_generator: bool = False,
) -> None:
    """Train a single-image SR model"""
    # for reproducibility of results; TODO: get rid of this later
    torch.manual_seed(0)
    LOGGER.debug("Pretrained generator: %s", pretrained_generator)

    def _get_obj_path(obj: str, start_date: str, end_date: str, stage: str) -> str:
        return os.path.join(
            config["output:basedir"],
            config[f"output:{obj}:{stage}:filename-template"].format(
                model_type=model_type, start_date=start_date, end_date=end_date
            ),
        )

    train_start_date, train_end_date = tuple(config["model:split:train"])
    store_path = _get_obj_path("scalers", train_start_date, train_end_date, "pre-trained") if pretrained_generator else None
    scalers = get_scalers(pretrained=pretrained_generator, store_path=store_path)

    augment_ops = A.Compose(
        [
            # random vertical / horizontal flip(s)
            A.Flip(p=0.5),
        ],
        additional_targets={"X_hr": "image", "Y": "image"},
    )

    ds_train, ds_valid, _ = get_datasets(scalers, config, augmentations=augment_ops, return_test_dataset=False)

    dl_train = torch.utils.data.DataLoader(
        ds_train, batch_size=config["model:batch-size:training"], num_workers=config["model:num-workers"]
    )
    dl_valid = torch.utils.data.DataLoader(
        ds_valid, batch_size=config["model:batch-size:training"], num_workers=config["model:num-workers"], shuffle=False
    )

    inverse_transform: Callable = build_inverse_transformation(scalers)

    store_path = _get_obj_path("model", train_start_date, train_end_date, "pre-trained") if pretrained_generator else None
    model = get_model(
        model_type,
        config,
        interactive,
        pretrained_generator=pretrained_generator,
        store_path=store_path,
        inverse_transform=inverse_transform,
    )

    trainer = get_trainer(model_type, config)

    # fit the model
    trainer.fit(model, train_dataloaders=dl_train, val_dataloaders=dl_valid)

    # almost done = save the model and scaler objects
    trainer.save_checkpoint(_get_obj_path("model", train_start_date, train_end_date, "trained"))

    with open(_get_obj_path("scalers", train_start_date, train_end_date, "trained"), "wb") as f:
        pickle.dump(scalers, f)

    LOGGER.debug("---- TRAINING DONE. ----")


def get_model(
    model_type: str,
    config: YAMLConfig,
    interactive: bool = False,
    pretrained_generator: bool = False,
    store_path: Optional[str] = None,
    inverse_transform: Optional[Callable] = None,
) -> Lightning_Model:
    """
    Returns a Lightning_Model wrapper for the given model type.
    Args:
        model_type: must be one of ['srgan' | 'unet' | 'xnet' | 'swin']
        config: configuration
        interactive: interactive mode (set to True when running in a Jupyter Notebook)
        pretrained_generator: use a pretrained generator model
        store_path: if pretrained_generator == True, load the generator model weights stored here
    """
    if model_type == "srgan":
        gen, disc = build_generator_and_discriminator(config, gan_type=model_type)
        if config["model:discriminator:type"] == "wasserstein":
            model = Lightning_WGAN_GP(
                gen,
                disc,
                gen_lr=config["model:generator:learn-rate"],
                disc_lr=config["model:discriminator:learn-rate"],
                gen_freq=config["model:generator:repeats"],
                disc_freq=config["model:discriminator:repeats"],
                gen_weight_decay=config["model:generator:weight-decay"],
                lambda_gp=config["model:gp-lambda"],
                lambda_l1=config["model:l1-lambda"],
                lambda_l2=config["model:l2-lambda"],
                lambda_adv=config["model:adv-lambda"],
                save_basedir=config["output:basedir"],
                interactive=interactive,
                wandb_logs=config["model:wandb"],
                inverse_transform=inverse_transform,
            )
        elif config["model:discriminator:type"] == "relativistic":
            model = Lightning_RGAN(
                gen,
                disc,
                gen_lr=config["model:generator:learn-rate"],
                disc_lr=config["model:discriminator:learn-rate"],
                gen_freq=1,
                disc_freq=1,
                gen_weight_decay=config["model:generator:weight-decay"],
                lambda_l1=config["model:l1-lambda"],
                lambda_l2=config["model:l2-lambda"],
                lambda_adv=config["model:adv-lambda"],
                save_basedir=config["output:basedir"],
                interactive=interactive,
                wandb_logs=config["model:wandb"],
                inverse_transform=inverse_transform,
            )
        else:
            LOGGER.error(
                "Unsupported SRGAN type: %s",
            )
            raise NotImplementedError
        if pretrained_generator:
            # load generator state from checkpoint
            LOGGER.debug("Reading pre-trained generator model from %s ...", store_path)
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            with open(store_path, "rb") as f:
                model.gen.load_state_dict(torch.load(f, map_location=device))
    elif model_type == "unet":
        unet = build_unet(config)
        if not unet.autoencoder:
            model = Lightning_UNet(
                unet,
                lr=config["model:unet:learn-rate"],
                weight_decay=config["model:unet:weight-decay"],
                save_basedir=config["output:basedir"],
                interactive=interactive,
                wandb_logs=config["model:wandb"],
                inverse_transform=inverse_transform,
            )
        else:
            model = Lightning_UNetAE(
                unet,
                lr=config["model:unet:learn-rate"],
                lambda_ae=config["model:unet:ae-lambda"],
                weight_decay=config["model:unet:weight-decay"],
                save_basedir=config["output:basedir"],
                interactive=interactive,
                wandb_logs=config["model:wandb"],
                inverse_transform=inverse_transform,
            )
    elif model_type == "xnet":
        xnet = build_xnet(config)
        model = Lightning_UNet(
            xnet,
            lr=config["model:optimizer:adam:learn-rate"],
            weight_decay=config["model:unet:weight-decay"],
            save_basedir=config["output:basedir"],
            interactive=interactive,
            wandb_logs=config["model:wandb"],
            inverse_transform=inverse_transform,
        )
    elif model_type == "swin":
        swin_transformer = build_swin_transformer(config)
        model = Lightning_SwinIR(
            swin_transformer,
            lr=config["model:swin:learn-rate"],
            save_basedir=config["output:basedir"],
            interactive=interactive,
            wandb_logs=config["model:wandb"],
            inverse_transform=inverse_transform,
        )
    else:
        raise RuntimeError(f"Invalid model type {model_type}. Must be one of ['srgan' | 'unet' | 'xnet' | 'swin']")

    return model


def get_trainer(model_type: str, config: YAMLConfig) -> pl.Trainer:
    """
    Returns a Lightning Trainer for the given model type.
        model_type: must be one of ['srgan' | 'unet' | 'xnet'].
    """
    # init logger
    if config["model:wandb"]:
        # use weights-and-biases
        logger = WandbLogger(project="AQ-Downscaling")
    else:
        # use tensorboard
        logger = TensorBoardLogger(config["output:logdir"])

    checkpoint_callback = ModelCheckpoint(
        dirpath=config["output:basedir"],
        filename=config["output:model:trained:filename-template"],
        monitor=f"{model_type}_val",
        verbose=False,
        save_top_k=3,
        every_n_epochs=1,
        save_weights_only=True,
        auto_insert_metric_name=False,
    )

    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch", log_momentum=True)

    earlystop_callback = EarlyStopping(
        monitor=f"{model_type}_val",
        min_delta=config["model:earlystop:min-delta"],
        patience=config["model:earlystop:patience"],
        verbose=False,
        mode=config["model:earlystop:mode"],
        check_finite=True,
    )

    if model_type == "srgan":
        trainer = pl.Trainer(
            gpus=1,
            precision=config["model:precision"],
            max_epochs=config["model:max-epochs"],
            logger=logger,
            log_every_n_steps=50,
        )
    elif model_type in ["unet", "xnet", "swin"]:
        trainer = pl.Trainer(
            gpus=1,
            precision=16,
            max_epochs=config["model:max-epochs"],
            logger=logger,
            log_every_n_steps=50,
            callbacks=[checkpoint_callback, earlystop_callback, lr_monitor_callback],
        )
    else:
        raise RuntimeError(f"Invalid model type {model_type}. Must be one of ['srgan' | 'unet' | 'xnet', | 'swin']")

    return trainer


def get_args() -> argparse.Namespace:
    """Returns a namespace containing the command line arguments"""
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument("--model", required=True, choices=["srgan", "unet", "xnet", "swin"], help="Super-resolution model")
    required_args.add_argument("--config", required=True, help="Model configuration file (YAML)")
    opt_args = parser.add_argument_group("optional arguments")
    opt_args.add_argument("--pretrained-generator", default=False, action="store_true", help="Use a pretrained SRGAN generator")
    return parser.parse_args()


def main() -> None:
    """Entry point for training."""
    args = get_args()
    config = YAMLConfig(args.config)

    if args.pretrained_generator and args.model != "srgan":
        LOGGER.error("Invalid arguments: cannot use a pre-trained generator for a model of type %s ...", args.model)
        raise RuntimeError

    if args.model in ["srgan", "unet", "xnet", "swin"]:
        train(args.model, config, interactive=False, pretrained_generator=args.pretrained_generator)
    else:
        raise NotImplementedError("Model {args.model} not implemented!")
