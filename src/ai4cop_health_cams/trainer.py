from typing import Optional, Tuple, Dict
import os

import abc
from abc import ABC
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import wandb

from ai4cop_health_cams.logger import get_logger


LOGGER = get_logger(__name__)


class Lightning_Model(pl.LightningModule, ABC):
    """Abstract trainer module containing common boilerplate code. Meant to be sub-classed."""

    _VAL_PLOT_FREQ = 50  # plot a generated sample every so many validation batches

    def __init__(
        self,
        save_basedir: Optional[str] = None,
        interactive: bool = False,
        wandb_logs: bool = False,
    ) -> None:
        """
        Trainer for the WGAN-GP.
        Args:
            save_basedir: basedir for plots
            interactive: running in interactive mode, i.e. inside a Jupyter notebook
            wandb_logs: use Weights&Biases to log media objects (generated sample plots)
        """
        super().__init__()
        self.save_basedir = save_basedir
        self.interactive = interactive
        self.wandb = wandb_logs

    def _get_batch_tensors(self, batch: Tuple[Dict[str, torch.Tensor], ...]) -> Tuple[torch.Tensor, ...]:
        """Get all tensors in a batch on the correct device: X_lr, X_hr, Y."""
        X_batch, Y_batch = batch
        X_lr = X_batch["X_lr"]
        X_hr = X_batch.get("X_hr", None)
        Y = Y_batch["y"] if Y_batch is not None else None
        return X_lr, X_hr, Y

    def test_step(self, batch: Tuple[Dict, ...], batch_idx: int) -> torch.Tensor:
        return self.predict_step(batch, batch_idx)

    def _get_noise(self, X: torch.Tensor) -> torch.Tensor:
        """
        Gets a single-channel random normal noise tensor of the same type as X.
        The noise shape is the same as that of X except in the channel dimension.
        """
        bs, _, h, w = X.shape
        return torch.randn(bs, 1, h, w).type_as(X)

    def _output_figure(self, fig: Figure, tag: str = "generic_model") -> None:
        """Figure output: save to file and/or display in notebook."""
        if self.save_basedir is not None:
            save_path = os.path.join(self.save_basedir, f"plots/{tag}_epoch{self.current_epoch:03d}.jpg")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path)
            if self.wandb:
                wandb.log({"predicted_val_sample": wandb.Image(save_path)})
        if self.interactive:
            plt.show()
        else:
            plt.close(fig)  # cleanup

    @abc.abstractmethod
    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        """Forward operator (use this to generate new hi-res samples)"""
        raise NotImplementedError

    @abc.abstractmethod
    def predict_step(self, batch: Tuple[Dict, ...], batch_idx: int, dataloader_idx: Optional[int] = None) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def training_step(self, batch: Tuple[Dict, ...], batch_idx: int, optimizer_idx: int) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def validation_step(self, batch: Tuple[Dict, ...], batch_idx: int) -> Dict:
        raise NotImplementedError

    @abc.abstractmethod
    def configure_optimizers(self):
        raise NotImplementedError
