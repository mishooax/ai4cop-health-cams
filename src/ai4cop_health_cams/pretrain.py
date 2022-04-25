from typing import Mapping, Any, Callable, Optional
import argparse
import os
import pickle

import torch
import numpy as np
import xarray as xr
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import albumentations as A

from ai4cop_health_cams.config import YAMLConfig
from ai4cop_health_cams.scalers import Log1pDataScaler, StdScaler, SimpleScaler, DataTransformer, Log1pStdScaler

from ai4cop_health_cams.gen_trainer import Lightning_GAN_EnsembleGenerator, Lightning_GAN_Generator
from ai4cop_health_cams.utils import build_generator, get_datasets
from ai4cop_health_cams.logger import get_logger

LOGGER = get_logger(__name__)


def build_inverse_transformation(scalers: Mapping[str, DataTransformer]) -> Callable:
    def inverse_transformation(resolution: str, data: np.ndarray) -> np.ndarray:
        return scalers[f"pm2p5_{resolution.upper()}"].inverse_transform(data)

    return inverse_transformation


def pretrain(model_type: str, config: Mapping[str, Any], interactive: bool = False) -> None:
    """
    Pretrain a single-image super-resolution model.
    This can be a deterministic or a generative model (e.g. the generator of a GAN).
    """
    # for reproducibility of results; TODO: get rid of this later
    torch.manual_seed(0)

    pretrain_start_date, pretrain_end_date = tuple(config["model:split:train"])
    sum_stats_path = os.path.join(
        config["input:scalers:basedir"],
        config["input:scalers:filename-template"].format(yyyymmddhh_start=pretrain_start_date, yyyymmddhh_end=pretrain_end_date),
    )

    sum_stats: xr.Dataset = xr.load_dataset(sum_stats_path)
    pm2p5_lr_scaler = Log1pStdScaler("pm2p5", sum_stats["log1p_pm2p5_mu_lr"].values, sum_stats["log1p_pm2p5_std_lr"].values)
    pm2p5_hr_scaler = Log1pStdScaler("pm2p5", sum_stats["log1p_pm2p5_mu_hr"].values, sum_stats["log1p_pm2p5_std_hr"].values)

    scalers: Mapping[str, DataTransformer] = {
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
        "pm2p5_LR": pm2p5_lr_scaler,  # µg/m3
        "pm2p5_HR": pm2p5_hr_scaler,  # µg/m3
    }

    augment_ops = A.Compose(
        [
            # random vertical / horizontal flip(s)
            A.Flip(p=0.5),
        ],
        additional_targets={"X_hr": "image", "Y": "image"},
    )

    ds_train, ds_valid, _ = get_datasets(scalers, config, augmentations=augment_ops)

    dl_train = torch.utils.data.DataLoader(
        ds_train, batch_size=config["model:batch-size:training"], num_workers=config["model:num-workers"]
    )
    dl_valid = torch.utils.data.DataLoader(
        ds_valid, batch_size=config["model:batch-size:training"], num_workers=config["model:num-workers"], shuffle=False
    )

    inverse_transform: Callable = build_inverse_transformation(scalers)
    model: Lightning_GAN_Generator = build_generator_model(
        model_type, config, interactive=interactive, inverse_transform=inverse_transform
    )

    trainer = get_trainer(model_type, config)

    # fit the model
    trainer.fit(model, train_dataloaders=dl_train, val_dataloaders=dl_valid)

    def _get_save_path(obj: str) -> str:
        return os.path.join(
            config["output:basedir"],
            config[f"output:{obj}:pre-trained:filename-template"].format(
                model_type=model_type, start_date=pretrain_start_date, end_date=pretrain_end_date
            ),
        )

    if trainer.checkpoint_callback is not None:
        LOGGER.debug("Restoring generator state from the best checkpoint ...")
        if isinstance(model, Lightning_GAN_EnsembleGenerator):
            model = Lightning_GAN_EnsembleGenerator.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        else:
            model = Lightning_GAN_Generator.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # save the generator weights
    torch.save(model.gen.state_dict(), _get_save_path("model"))
    # trainer.save_checkpoint(_get_save_path("model"))

    with open(_get_save_path("scalers"), "wb") as f:
        pickle.dump(scalers, f)

    LOGGER.debug("---- PRETRAINING DONE. ----")


def build_generator_model(
    model_type: str, config: YAMLConfig, interactive: bool = False, inverse_transform: Optional[Callable] = None
) -> Lightning_GAN_Generator:
    """
    Returns a LightningModule wrapper for the given model type.
    Args:
        model_type: model type, must be one of ['srgan' | ...]
        config: job configuration
        interactive: interactive mode (set to True if running this inside a Jupyter notebook)
    """
    if model_type == "srgan":
        generator = build_generator(config, gan_type=model_type, ensemble_output=config["model:generator:ensemble"])
        if config["model:generator:ensemble"]:
            model = Lightning_GAN_EnsembleGenerator(
                generator,
                lr=config["model:generator:learn-rate"],
                weight_decay=config["model:generator:weight-decay"],
                lambda_l1=0.0,
                lambda_l2=0.0,
                predict_ensemble_size=config["model:generator:ensemble-size"],
                save_basedir=config["output:basedir"],
                interactive=interactive,
                wandb_logs=config["model:wandb"],
                inverse_transform=inverse_transform,
            )
        else:
            model = Lightning_GAN_Generator(
                generator,
                lr=config["model:generator:learn-rate"],
                weight_decay=config["model:generator:weight-decay"],
                lambda_l1=config["model:l1-lambda"],
                lambda_l2=config["model:l2-lambda"],
                predict_ensemble_size=1,
                save_basedir=config["output:basedir"],
                interactive=interactive,
                wandb_logs=config["model:wandb"],
                inverse_transform=inverse_transform,
            )
    else:
        LOGGER.error("Invalid model type %s. Only 'srgan' is supported at present ...", model_type)
        raise RuntimeError()

    return model


def get_trainer(model_type: str, config: YAMLConfig) -> pl.Trainer:
    """
    Returns a Lightning Trainer for the given model type.
    Args:
        model_type: model type, must be 'srgan' (other model types to be added later)
        config: job configuration
    """
    del model_type  # not used
    # init logger
    if config["model:wandb"]:
        # use weights-and-biases
        logger = WandbLogger(project="AQ-Downscaling")
    else:
        # use tensorboard
        logger = TensorBoardLogger(config["output:logdir"])

    checkpoint_callback = ModelCheckpoint(
        dirpath=config["output:basedir"],
        filename=config["output:model:pre-trained:filename-template"],
        monitor="gen_val_l1_pretrain",
        verbose=False,
        save_top_k=1,
        every_n_epochs=1,
        save_weights_only=False,
        auto_insert_metric_name=False,
    )

    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch", log_momentum=True)

    earlystop_callback = EarlyStopping(
        monitor="gen_val_l1_pretrain",
        min_delta=config["model:earlystop:min-delta"],
        patience=config["model:earlystop:patience"],
        verbose=False,
        mode=config["model:earlystop:mode"],
        check_finite=True,
    )

    trainer = pl.Trainer(
        gpus=1,
        precision=config["model:precision"],
        max_epochs=config["model:max-epochs"],
        logger=logger,
        log_every_n_steps=50,
        callbacks=[checkpoint_callback, earlystop_callback, lr_monitor_callback],
    )

    return trainer


def get_args() -> argparse.Namespace:
    """Returns a namespace containing the command line arguments"""
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument("--model", required=True, choices=["srgan"], help="Super-resolution model")
    required_args.add_argument("--config", required=True, help="Model configuration file (YAML)")
    return parser.parse_args()


def main() -> None:
    """Entry point for training."""
    args = get_args()
    config = YAMLConfig(args.config)
    if args.model in ["srgan"]:
        pretrain(args.model, config)
    else:
        raise NotImplementedError("Model {args.model} not implemented!")
