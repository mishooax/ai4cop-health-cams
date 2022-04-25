from typing import Mapping, Any, Tuple, List
import argparse
import os
import pickle

import numpy as np
import torch
import pytorch_lightning as pl

from ai4cop_health_cams.utils import load_gan_model, get_dataset
from ai4cop_health_cams.config import YAMLConfig
from ai4cop_health_cams.dataset import AirQualityDataset
from ai4cop_health_cams.scalers import DataTransformer
from ai4cop_health_cams.logger import get_logger
from ai4cop_health_cams.plots import plot_batch_predictions

LOGGER = get_logger(__name__)


def get_test_dataset(transformers: Mapping[str, DataTransformer], config: YAMLConfig) -> AirQualityDataset:
    """Builds train and validation datasets from the configuration settings.
    Args:
        scalers: CAMS i/o data scalers, {varname: scaler_object}
        config: job configuration
    Returns:
        Test dataset
    """
    test_window: Tuple[str, str] = tuple(config["model:split:test"])
    return get_dataset("test", transformers, test_window[0], test_window[1], config)


def save_data(data: torch.Tensor, data_type: str, model_type: str, config: YAMLConfig) -> None:
    """
    Saves the predictions or observations to a file.
    Args:
        data: data tensor to save
        data_type: data type (predictions or observations)
        model_type: model type
        config: job configuration
    Returns:
        Nothing.
    """
    start_date, end_date = tuple(config["model:split:test"])
    save_path = os.path.join(
        config["output:basedir"],
        config[f"output:{data_type}:filename-template"].format(
            model_type=model_type,
            varnames="_".join(config["input:processed:hires:varnames"]),
            start_date=start_date,
            end_date=end_date,
        ),
    )
    with open(save_path, "wb") as f:
        torch.save(data, f)


def get_args() -> argparse.Namespace:
    """Returns a namespace containing the command line arguments"""
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument("--model", required=True, choices=["srgan"], help="Pre-trained super-resolution model")
    required_args.add_argument("--config", required=True, help="Model configuration file (YAML)")
    return parser.parse_args()


def predict(model_type: str, config: Mapping[str, Any], interactive: bool = False) -> None:
    """Predict single HR frames with a pre-trained model"""

    start_date, end_date = tuple(config["model:split:test"])

    def _get_save_path(obj: str) -> str:
        return os.path.join(
            config["output:basedir"],
            config[f"output:{obj}:trained:filename-template"].format(
                model_type=model_type, start_date=start_date, end_date=end_date
            ),
        )

    path_to_scalers = _get_save_path("scalers")
    path_to_model = _get_save_path("model")

    # 1/ load scalers
    with open(path_to_scalers, "rb") as f:
        scalers: Mapping[str, DataTransformer] = pickle.load(f)

    # 2/ get test dataset / dataloader
    ds_test: AirQualityDataset = get_test_dataset(scalers, config)
    dl_test = torch.utils.data.DataLoader(
        ds_test, batch_size=config["model:batch-size:inference"], num_workers=config["model:num-workers"], shuffle=False
    )

    # 3/ load models and create trainer object
    if model_type in ["srgan", "sagan"]:
        model = load_gan_model(model_type, path_to_model, config, load_discriminator=True, interactive=interactive)
    else:
        LOGGER.error("Predictions for %s haven't been implemented yet!")
        raise NotImplementedError

    # 4/ predict
    trainer = pl.Trainer(gpus=1, precision=config["model:precision"])
    predictions: List[torch.Tensor] = trainer.predict(model, dataloaders=dl_test, return_predictions=True)

    obs: List[torch.Tensor] = []

    # 5/ plot predictions
    LOGGER.debug("Plotting the results (once every N batches) ...")
    for i_batch, batch in enumerate(dl_test):
        X, Y = batch
        obs.append(Y["y"])
        if i_batch % 50 == 0:
            LOGGER.debug("Plotting predictions for batch %d of %d ...", i_batch, len(dl_test))
            lowres, truth = X["X_lr"].cpu().numpy(), Y["y"].cpu().numpy()
            num_members = predictions[i_batch].shape[-1]
            i_member = np.random.randint(num_members)
            plot_batch_predictions(lowres, truth, predictions[i_batch][..., i_member].cpu().numpy(), config, interactive)

    observations = torch.cat(obs, dim=0).cpu()
    predictions = torch.cat(predictions, dim=0).cpu()
    LOGGER.debug("Shape of ground truth tensor: %s", observations.shape)
    LOGGER.debug("Shape of predictions tensor: %s", predictions.shape)

    # 6/ inverse-transform
    for ix, var in enumerate(config["input:processed:hires:varnames"]):
        predictions[:, ix, ...] = scalers[f"{var}_HR"].inverse_transform(predictions[:, ix, ...])

    # 7/ save predictions and observations
    save_data(predictions, "predictions", model_type, config)
    save_data(observations, "observations", model_type, config)

    # 8/ calculate forecast scores
    # calculate_forecast_scores(predictions)

    LOGGER.debug("----- DONE. ------")


def main() -> None:
    """Entry point for prediction tasks."""
    args = get_args()
    config = YAMLConfig(args.config)
    predict(args.model, config)
