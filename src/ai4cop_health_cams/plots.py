from typing import Optional, Tuple, Callable
import os

import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ai4cop_health_cams.logger import get_logger
from ai4cop_health_cams.config import YAMLConfig

LOGGER = get_logger(__name__)


def plot_sample(
    X: torch.Tensor,
    y: torch.Tensor,
    y_pred: torch.Tensor,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Tuple:
    """Plots data corresponding to a single sample:
        Left: LR input
        Middle: HR prediction
        Right: HR "ground truth"
    Args:
        X, X_hr, y: batch of data, i.e. predictors (X, X_hr) and true labels (y)
        gen: generator model
        i_sample: index of the sample to plot, must be in [0, batch_size)
        vmin, vmax: min/max values used for the image plots
    Returns:
        vmin, vmax (so that they can be re-used later)
    """
    lr = X.detach().cpu().numpy()
    hr = y.detach().cpu().numpy()
    pred = y_pred.detach().cpu().numpy()

    if vmin is None:
        vmin = np.min([np.min(lr), np.min(hr)])
    if vmax is None:
        vmax = np.max([np.max(lr), np.max(hr)])

    fig, ax = plt.subplots(1, 4, figsize=(16, 5))
    im = ax[0].imshow(lr, vmin=vmin, vmax=vmax, cmap="viridis", origin="lower")
    ax[0].set_title("X")
    __get_colorbar(im, ax[0])
    im = ax[1].imshow(pred, vmin=vmin, vmax=vmax, cmap="viridis", origin="lower")
    ax[1].set_title("Prediction")
    __get_colorbar(im, ax[1])
    im = ax[2].imshow(hr, vmin=vmin, vmax=vmax, cmap="viridis", origin="lower")
    ax[2].set_title("Ground truth")
    __get_colorbar(im, ax[2])
    im = ax[3].imshow(hr - pred, cmap="bwr", origin="lower", norm=colors.TwoSlopeNorm(vcenter=0.0))
    ax[3].set_title("Truth - prediction")
    __get_colorbar(im, ax[3])

    return fig, vmin, vmax


def plot_predicted_samples(
    X_lr: np.ndarray,
    y_pred: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    ensemble_spread: Optional[np.ndarray] = None,
    i_sample: Optional[int] = None,
    inverse_transform: Optional[Callable] = None,
):
    """Plots data corresponding to a single sample:
            X_lr == LR input
            y_pred == HR prediction
            y_true == HR "ground truth" (if not None)
            y_true - y_pred (if y_true is not None)
    Args:
        ensemble spread: spread of pseudo-ensemble realizations (generative models)
        i_sample: index of the sample to plot, must be in [0, batch_size)
                  if None then all samples are plotted
        inverse_transform: function used to inverse-transform the data samples (to the original data space)
    Returns:
        The figure object handle.
    """

    num_samples = y_true.shape[0] if i_sample is None else 1
    n_plots = 2
    if y_true is not None:
        n_plots += 2
    if ensemble_spread is not None:
        n_plots += 1
    figsize = (n_plots * 4, num_samples * 4)

    fig, ax = plt.subplots(num_samples, n_plots, figsize=figsize)

    if inverse_transform is not None:
        y_pred = inverse_transform("hr", y_pred)
        X_lr = inverse_transform("lr", X_lr)
        if y_true is not None:
            y_true = inverse_transform("hr", y_true)

    if i_sample is not None:
        assert 0 <= i_sample < X_lr.shape[0], f"Invalid sample index {i_sample}!"
        X = X_lr[i_sample, 0, ...]
        yp = y_pred[i_sample, 0, ...]
        yt = y_true[i_sample, 0, ...]
        ens_spread = ensemble_spread[i_sample, 0, ...]
        __plot_single_sample(ax, X, yp, yt, ens_spread)
    else:
        for i_s in range(y_true.shape[0]):
            X = X_lr[i_s, 0, ...]
            yp = y_pred[i_s, 0, ...]
            yt = y_true[i_s, 0, ...]
            ens_spread = ensemble_spread[i_s, 0, ...]
            __plot_single_sample(ax[i_s, :], X, yp, yt, ens_spread)

    return fig


def plot_reconstructed_samples(
    X_hr: np.ndarray,
    X_hr_fake: np.ndarray,
    i_sample: Optional[int] = None,
):
    """Plots reconstructed data from a single sample:
            X_hr == HR input [orography]
            X_fake == HR reconstruction of X_hr
            X_fake - X_hr
    Args:
        i_sample: index of the sample to plot, must be in [0, batch_size)
                  if None then all samples are plotted
    Returns:
        the figure object handle
    """

    num_samples = X_hr.shape[0] if i_sample is None else 1
    n_plots = 3
    figsize = (11, num_samples * 4)

    fig, ax = plt.subplots(num_samples, n_plots, figsize=figsize)

    if i_sample is not None:
        assert 0 <= i_sample < X_hr.shape[0], f"Invalid sample index {i_sample}!"
        X = X_hr[i_sample, ...]
        Xp = X_hr_fake[i_sample, ...]
        __plot_reconstructed_sample(ax, Xp, X)
    else:
        for i_s in range(X_hr.shape[0]):
            X = X_hr[i_s, ...]
            Xp = X_hr_fake[i_s, ...]
            __plot_reconstructed_sample(ax[i_s, :], Xp, X)

    return fig


def __plot_reconstructed_sample(ax, Xp: np.ndarray, X: np.ndarray) -> None:
    """Plots a reconstructed (hi-res) input sample Xp, the truth (X) and the reconstruction error X - Xp"""
    im = ax[0].imshow(Xp, cmap="viridis", origin="lower")
    ax[0].set_title("HR reconstruction")
    __get_colorbar(im, ax[0])
    im = ax[1].imshow(X, cmap="viridis", origin="lower")
    ax[1].set_title("HR input")
    __get_colorbar(im, ax[1])
    im = ax[2].imshow(X - Xp, cmap="bwr", origin="lower", norm=colors.TwoSlopeNorm(vcenter=0.0))
    ax[2].set_title("Reconstruction error")
    __get_colorbar(im, ax[2])


def __plot_single_sample(
    ax, X: np.ndarray, yp: np.ndarray, yt: Optional[np.ndarray] = None, ens_sd: Optional[np.ndarray] = None
) -> None:
    """Plots (X, y_predicted, y_true, y_error) for a single sample."""
    if yt is not None:
        vmin, vmax = np.min(yt), np.max(yt)
    else:
        vmin, vmax = np.min(X), np.max(X)

    im = ax[0].imshow(X, vmin=vmin, vmax=vmax, cmap="viridis", origin="lower")
    ax[0].set_title("LR input")
    __get_colorbar(im, ax[0])
    im = ax[1].imshow(yp, vmin=vmin, vmax=vmax, cmap="viridis", origin="lower")
    ax[1].set_title("Prediction")
    __get_colorbar(im, ax[1])
    if yt is not None:
        im = ax[2].imshow(yt, vmin=vmin, vmax=vmax, cmap="viridis", origin="lower")
        __get_colorbar(im, ax[2])
        ax[2].set_title("Truth")
        im = ax[3].imshow(yt - yp, cmap="bwr", origin="lower", norm=colors.TwoSlopeNorm(vcenter=0.0))
        ax[3].set_title("[Truth - Prediction]")
        __get_colorbar(im, ax[3])
    if ens_sd is not None:
        im = ax[4].imshow(ens_sd, cmap="viridis", origin="lower")
        ax[4].set_title("Pseudo-ens spread")
        __get_colorbar(im, ax[4])


def __get_colorbar(im, ax) -> None:
    """Returns a custom colorbar."""
    # hide x/y-axis ticks
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis="both", which="both", length=0)
    # create colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)


def plot_batch_predictions(
    lowres: np.ndarray, truth: np.ndarray, predictions: np.ndarray, config: YAMLConfig, interactive: bool = False
) -> None:
    """Plot predictions for a batch. Set interactive == True to show the plots (e.g. in a Jupyter notebook)."""
    num_samples = min(8, predictions.shape[0])
    sample_indices = np.random.choice(predictions.shape[0], size=(num_samples,), replace=False)

    lowres = lowres[sample_indices, ...]
    truth = truth[sample_indices, ...]
    predictions = predictions[sample_indices, ...]

    save_path = os.path.join(config["output:basedir"], "plots/wgan_gp_predicted_samples.jpg")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig = plot_predicted_samples(lowres, predictions, truth)
    fig.savefig(save_path)

    if interactive:
        plt.show()
    else:
        plt.close(fig)  # cleanup
