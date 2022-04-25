from typing import Tuple, Union, Optional, Mapping, List
import os

from torch import nn
from torch.utils.data import ConcatDataset
import albumentations as A

from ai4cop_health_cams.config import YAMLConfig
from ai4cop_health_cams.srgan import Discriminator, Generator
from ai4cop_health_cams.trainer import Lightning_Model
from ai4cop_health_cams.unet import UNet, XNet
from ai4cop_health_cams.swin import SwinIR
from ai4cop_health_cams.dataset import AirQualityDataset
from ai4cop_health_cams.scalers import DataTransformer
from ai4cop_health_cams.logger import get_logger
from ai4cop_health_cams.wgan_gp_trainer import Lightning_WGAN_GP
from ai4cop_health_cams.rgan_trainer import Lightning_RGAN
from ai4cop_health_cams.dataset import AirQualityDataset

LOGGER = get_logger(__name__)


def build_generator_and_discriminator(
    config: YAMLConfig,
    gan_type: str = "srgan",
) -> Tuple[nn.Module, nn.Module]:
    """Creates the generator and discriminator models"""

    num_inputs_lr = len(config["input:processed:lowres:varnames"])
    inputs_hr = config["input:processed:hires-const:varnames"]
    num_inputs_hr = len(inputs_hr) if inputs_hr is not None else 0

    if gan_type == "srgan":
        # Super-resolution GAN
        gen = Generator(
            num_inputs_lr=num_inputs_lr,
            num_outputs=config["model:generator:num-outputs"],
            num_inputs_hr=num_inputs_hr,
            activation_out=config["model:generator:activation-out"],
            spectral_norm=config["model:generator:spectral-norm"],
        )
        disc = Discriminator(
            num_inputs_lr=num_inputs_lr,
            num_inputs_hr_const=num_inputs_hr,
            num_inputs_hr=config["model:generator:num-outputs"],
            spectral_norm=config["model:discriminator:spectral-norm"],
        )
    elif gan_type == "sagan":
        # Self-attention (SR)GAN
        # HR inputs not yet supported
        gen = SelfAttentionGenerator(num_inputs=num_inputs_lr, use_noise=True)
        disc = SelfAttentionDiscriminator(num_inputs=num_inputs_lr)
    else:
        raise NotImplementedError(f"GAN type {gan_type} not implemented!")

    return gen, disc


def load_gan_model(
    gan_type: str, path_to_model: str, config: YAMLConfig, load_discriminator: bool = True, interactive: bool = False
) -> Lightning_Model:
    """
    Loads the pre-trained GAN model (wrapped into a LightningModule).
    Args:
        gan_type: type of GAN model (["srgan" | "sagan"])
        path_to_model: dir path to saved models (generator + optionally discriminator)

    """
    LOGGER.debug("Loading %s pre-trained GAN generator and discriminator models from %s ...", gan_type, path_to_model)
    gen, disc = build_generator_and_discriminator(config, gan_type=gan_type)

    LOGGER.debug("Loading trained model state ...")

    if load_discriminator:
        checkpoint_state = {"generator": gen, "discriminator": disc, "interactive": interactive}
    else:
        checkpoint_state = {"generator": gen, "interactive": interactive}

    if config["model:discriminator:type"] == "wasserstein":
        model = Lightning_WGAN_GP.load_from_checkpoint(path_to_model, **checkpoint_state)
    elif config["model:discriminator:type"] == "relativistic":
        model = Lightning_RGAN.load_from_checkpoint(path_to_model, **checkpoint_state)
    else:
        LOGGER.error("Invalid discriminator type: %s! Re-check your config file...", config["model:discriminator:type"])
        raise RuntimeError

    return model


def build_generator(
    config: YAMLConfig,
    gan_type: str = "srgan",
    ensemble_output: bool = False,
) -> nn.Module:
    """Creates a generator model"""

    num_inputs_lr = len(config["input:processed:lowres:varnames"])
    inputs_hr = config["input:processed:hires-const:varnames"]
    num_inputs_hr = len(inputs_hr) if inputs_hr is not None else 0

    num_outputs = config["model:generator:ensemble-size"] if ensemble_output else config["model:generator:num-outputs"]

    if gan_type == "srgan":
        # Super-resolution GAN
        gen = Generator(
            num_inputs_lr=num_inputs_lr,
            num_outputs=num_outputs,
            num_inputs_hr=num_inputs_hr,
            activation_out=config["model:generator:activation-out"],
            spectral_norm=config["model:generator:spectral-norm"],
        )
    else:
        raise NotImplementedError(f"GAN type {gan_type} not implemented!")

    return gen


def build_unet(
    config: YAMLConfig,
) -> nn.Module:
    """Creates the U-Net model"""

    num_inputs_lr = len(config["input:processed:lowres:varnames"])
    inputs_hr = config["input:processed:hires-const:varnames"]
    num_inputs_hr = len(inputs_hr) if inputs_hr is not None else 0
    num_outputs = 1

    return UNet(
        num_inputs_lr,
        num_outputs,
        num_inputs_hr,
        pooling=config["model:unet:pooling"],
        norm=config["model:unet:norm"],
        dropout_rate=config["model:unet:dropout-rate"],
        autoencoder=config["model:unet:autoencoder"],
    )


def build_xnet(
    config: YAMLConfig,
) -> nn.Module:
    """Creates the UNet++ model"""

    num_inputs_lr = len(config["input:processed:lowres:varnames"])
    inputs_hr = config["input:processed:hires-const:varnames"]
    num_inputs_hr = len(inputs_hr) if inputs_hr is not None else 0
    num_outputs = 1

    return XNet(
        num_inputs_lr,
        num_outputs,
        num_inputs_hr,
        pooling=config["model:unet:pooling"],
        norm=config["model:unet:norm"],
        dropout_rate=config["model:unet:dropout-rate"],
    )


def build_swin_transformer(
    config: YAMLConfig,
) -> nn.Module:
    """Creates the UNet++ model"""

    num_inputs_lr = len(config["input:processed:lowres:varnames"])
    # inputs_hr = config["input:processed:hires-const:varnames"]
    # num_inputs_hr = len(inputs_hr) if inputs_hr is not None else 0
    num_outputs = 1

    assert num_outputs == num_inputs_lr == 1, "Cannot use this with multiple inputs (yet)!"

    return SwinIR(
        img_size=(8, 8),
        input_channels=num_inputs_lr,
        window_size=config["model:swin:window-size"],
        depths=tuple(config["model:swin:depths"]),
        embed_dim=config["model:swin:embedding-dim"],
        num_heads=tuple(config["model:swin:num-heads"]),
        mlp_ratio=config["model:swin:mlp-ratio"],
        activation_out=config["model:swin:activation-out"],
    )


def get_datasets(
    scalers: Mapping[str, DataTransformer],
    config: YAMLConfig,
    augmentations: Optional[A.Compose] = None,
    return_test_dataset: bool = True,
) -> Tuple[Optional[Union[AirQualityDataset, ConcatDataset]], ...]:
    """Builds train and validation datasets from the configuration settings.
    Args:
        scalers: CAMS i/o data scalers, {varname: scaler_object}
        config: job configuration
        augmentations: data augmentation operations (to be applied only to the training set)
    Returns:
        training, validation and (optionally) test datasets
    """
    train_window: Tuple[str, str] = tuple(config["model:split:train"])
    valid_window: Tuple[str, str] = tuple(config["model:split:validation"])

    # we apply the data augmentation ops only to the training dataset
    ds_train = get_dataset("train", scalers, train_window[0], train_window[1], config, augmentations)
    ds_valid = get_dataset("valid", scalers, valid_window[0], valid_window[1], config)

    if return_test_dataset:
        test_window: Tuple[str, str] = tuple(config["model:split:test"])
        ds_test = get_dataset("test", scalers, test_window[0], test_window[1], config)
    else:
        ds_test = None

    return ds_train, ds_valid, ds_test


# def get_dataset(
#     transformers: Mapping[str, DataTransformer],
#     start_date: str,
#     end_date: str,
#     time_idx_range: Tuple[int],
#     config: YAMLConfig,
#     augmentations: Optional[A.Compose] = None,
# ) -> Union[AirQualityDataset, ConcatDataset]:
#     """Instantiates a CamsAirQualityDataset
#     Args:
#         ds_name: name of dataset ("train", "valid" or "test")
#         scalers: mapping {varname: transformer-object}
#         start_date, end_date: training interval (includes validation points)
#         time_idx_range: time index range to retrieve
#         config: job configuration
#         augmentations: (composed) data augmentation operations
#     """
#     regions: List[int] = config["input:processed:regions"]
#     if len(regions) < 1:
#         raise RuntimeError(f"List of region indices is invalid: {regions}. Check your config file.")

#     datasets: List[AirQualityDataset] = []
#     for region_index in regions:
#         ds = AirQualityDataset(
#             file_path_lr_input=os.path.join(
#                 config["input:processed:basedir"],
#                 config["input:processed:lowres:filename-template"].format(
#                     yyyymmddhh_start=start_date, yyyymmddhh_end=end_date, region_index=f"{region_index:03d}"
#                 ),
#             ),
#             lr_input_vars=config["input:processed:lowres:varnames"],
#             time_idx=time_idx_range,
#             file_path_hr_output=os.path.join(
#                 config["input:processed:basedir"],
#                 config["input:processed:hires:filename-template"].format(
#                     yyyymmddhh_start=start_date, yyyymmddhh_end=end_date, region_index=f"{region_index:03d}"
#                 ),
#             ),
#             hr_output_vars=config["input:processed:hires:varnames"],
#             hr_input_vars=config["input:processed:hires-const:varnames"],
#             file_path_hr_input=os.path.join(
#                 config["input:processed:basedir"],
#                 config["input:processed:hires-const:filename-template"].format(region_index=f"{region_index:03d}"),
#             ),
#             transformers=transformers,
#             augmentations=augmentations,
#         )
#         datasets.append(ds)

#     if len(datasets) == 1:
#         return datasets[0]
#     return ConcatDataset(datasets)


def get_dataset(
    ds_name: str,
    transformers: Mapping[str, DataTransformer],
    start_date: str,
    end_date: str,
    config: YAMLConfig,
    augmentations: Optional[A.Compose] = None,
) -> Union[AirQualityDataset, ConcatDataset]:
    """Instantiates an AirQualityDataset
    Args:
        scalers: mapping {varname: transformer-object}
        start_date, end_date: training interval (includes validation points)
        time_idx_range: time index range to retrieve
        config: job configuration
        augmentations: (composed) data augmentation operations
    """
    regions: List[int] = config["input:processed:regions"]
    if len(regions) < 1:
        raise RuntimeError(f"List of region indices is invalid: {regions}. Check your config file.")

    datasets: List[AirQualityDataset] = []
    for region_index in regions:
        ds = AirQualityDataset(
            file_path_lr_input=os.path.join(
                config["input:processed:basedir"].format(name=ds_name),
                config["input:processed:lowres:filename-template"].format(
                    yyyymmddhh_start=start_date,
                    yyyymmddhh_end=end_date,
                    region_index=f"{region_index:03d}",
                    stage=ds_name,
                ),
            ),
            lr_input_vars=config["input:processed:lowres:varnames"],
            file_path_hr_output=os.path.join(
                config["input:processed:basedir"].format(name=ds_name),
                config["input:processed:hires:filename-template"].format(
                    yyyymmddhh_start=start_date,
                    yyyymmddhh_end=end_date,
                    region_index=f"{region_index:03d}",
                    stage=ds_name,
                ),
            ),
            hr_output_vars=config["input:processed:hires:varnames"],
            hr_input_vars=config["input:processed:hires-const:varnames"],
            file_path_hr_input=os.path.join(
                config["input:processed:basedir"].format(name=ds_name),
                config["input:processed:hires-const:filename-template"].format(
                    yyyymmddhh_start=start_date,
                    yyyymmddhh_end=end_date,
                    region_index=f"{region_index:03d}",
                    stage=ds_name,
                ),
            ),
            transformers=transformers,
            augmentations=augmentations,
        )
        datasets.append(ds)

    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)
