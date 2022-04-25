from typing import Optional, Tuple, Dict, Callable
import random

import torch
from torch import nn
import torch.nn.functional as F

from ai4cop_health_cams.logger import get_logger
from ai4cop_health_cams.plots import plot_predicted_samples
from ai4cop_health_cams.trainer import Lightning_Model


LOGGER = get_logger(__name__)


class Lightning_UNet(Lightning_Model):
    """Vanilla U-Net trainer."""

    _VAL_PLOT_FREQ = 50  # plot a validation sample every so many epochs

    def __init__(
        self,
        unet: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        save_basedir: Optional[str] = None,
        interactive: bool = False,
        wandb_logs: bool = False,
        inverse_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(save_basedir=save_basedir, interactive=interactive, wandb_logs=wandb_logs)
        self.unet = unet
        self.lr = lr
        self.weight_decay = weight_decay
        self.b1 = 0.5
        self.b2 = 0.999
        self.tag = "unet"
        self.inverse_transform = inverse_transform

    def forward(self, x_lr: torch.Tensor, x_hr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward operator (use this to generate new hi-res samples)"""
        return self.unet(x_lr, x_hr)

    def predict_step(self, batch: Tuple[dict], batch_idx: int, dataloader_idx: Optional[int] = None) -> torch.Tensor:
        del batch_idx, dataloader_idx  # not used
        X, X_const, _ = self._get_batch_tensors(batch)
        return self.unet(X, X_const)

    def training_step(self, batch: Tuple[Dict, ...]) -> torch.Tensor:
        """U-Net training step."""
        X, X_const, Y = self._get_batch_tensors(batch)
        Y_fake = self.unet(X, X_const)
        unet_loss = self._compute_unet_loss(Y_fake, Y)
        self.log("unet_loss", unet_loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        return unet_loss

    def _compute_unet_loss(self, y_fake: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calculate the U-Net loss."""
        return F.smooth_l1_loss(y_fake, y_true, reduction="mean", beta=1e-2)

    def configure_optimizers(self):
        unet_opt = torch.optim.Adam(self.unet.parameters(), lr=self.lr, betas=(self.b1, self.b2), weight_decay=self.weight_decay)
        unet_lrs = torch.optim.lr_scheduler.ReduceLROnPlateau(
            unet_opt,
            mode="min",
            factor=0.3,
            patience=3,
            verbose=False,
        )
        return {
            "optimizer": unet_opt,
            "lr_scheduler": {
                "scheduler": unet_lrs,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "unet_val",
                "name": "reduce_on_plateau_lr",
            },
        }

    def validation_step(self, batch: Tuple[Dict, ...], batch_idx: int) -> Dict:
        """U-Net validation step."""
        with torch.no_grad():
            X_lr, X_hr, Y = self._get_batch_tensors(batch)
            Y_fake = self.unet(X_lr, X_hr)
            unet_loss_val = self._compute_unet_loss(Y_fake, Y)

            self.log("unet_val", unet_loss_val, on_epoch=True, on_step=False, prog_bar=True, logger=True)

            if batch_idx % self._VAL_PLOT_FREQ == 0:
                sample_index = random.randrange(Y.shape[0])
                fig = plot_predicted_samples(
                    X_lr.cpu().numpy(),
                    Y_fake.cpu().numpy(),
                    Y.cpu().numpy(),
                    i_sample=sample_index,
                    inverse_transform=self.inverse_transform,
                )
                self._output_figure(fig, tag=f"{self.tag}_predicted_val_samples_batch{batch_idx:04d}")

        return {"unet_val": unet_loss_val}


class Lightning_UNetAE(Lightning_UNet):
    """
    UNet-AE, Sha et al. 2020
    https://journals.ametsoc.org/view/journals/apme/59/12/jamc-d-20-0058.1.xml
    Incorporates an unsupervised reconstruction loss for the HR orography input.
    """

    def __init__(
        self,
        unet: nn.Module,
        lambda_ae: float = 1.0,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        save_basedir: Optional[str] = None,
        interactive: bool = False,
        wandb_logs: bool = False,
        inverse_transform: Optional[Callable] = None,
    ):
        super().__init__(
            unet,
            lr=lr,
            weight_decay=weight_decay,
            save_basedir=save_basedir,
            interactive=interactive,
            wandb_logs=wandb_logs,
            inverse_transform=inverse_transform,
        )
        assert unet.autoencoder, "This Lightning module requires a UNet-AE."
        self.lambda_ae = lambda_ae
        self.tag = "unet_ae"
        self.save_hyperparameters()

    def forward(self, x_lr: torch.Tensor, x_hr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward operator (use this to generate new hi-res samples)"""
        return self.unet(x_lr, x_hr)[0]

    def training_step(self, batch: Tuple[Dict, ...]) -> torch.Tensor:
        """One training step"""
        X, X_const, Y = self._get_batch_tensors(batch)
        Y_fake, X_fake = self.unet(X, X_const)
        unet_loss = self._compute_unet_loss(Y_fake, Y, X_fake, X_const)
        return unet_loss

    def _compute_unet_loss(
        self, Y_fake: torch.Tensor, Y: torch.Tensor, X_fake: torch.Tensor, X_const: torch.Tensor
    ) -> torch.Tensor:
        """Calculates the total loss; this includes the AE reconstruction term."""

        target_loss = F.smooth_l1_loss(Y_fake, Y, reduction="mean", beta=1e-2)
        # reconstruction loss scaled by batch size
        reconstruction_loss = F.smooth_l1_loss(X_fake.squeeze(dim=1), X_const[:, 0, ...], reduction="mean", beta=1e-2)
        unet_loss = target_loss + self.lambda_ae * reconstruction_loss

        self.log("target_loss", target_loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        self.log("ae_loss", reconstruction_loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        self.log("unet_loss", unet_loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)

        return unet_loss

    def validation_step(self, batch: Tuple[Dict, ...], batch_idx: int) -> Dict:
        """One validation step."""
        del batch_idx  # not used
        X_lr, X_const, Y = self._get_batch_tensors(batch)

        with torch.no_grad():
            preds: Tuple[torch.Tensor, ...] = self.unet(X_lr, X_const)
            Y_fake, X_fake = preds
            unet_loss_val = self._compute_unet_loss(Y_fake, Y, X_fake, X_const)
            self.log("unet_val", unet_loss_val, on_epoch=True, on_step=False, prog_bar=True, logger=True)

            if batch_idx % self._VAL_PLOT_FREQ == 0:
                sample_index = random.randrange(Y.shape[0])
                fig = plot_predicted_samples(
                    X_lr.cpu().numpy(),
                    Y_fake.cpu().numpy(),
                    Y.cpu().numpy(),
                    i_sample=sample_index,
                    inverse_transform=self.inverse_transform,
                )
                self._output_figure(fig, tag=f"{self.tag}_predicted_val_samples_batch{batch_idx:04d}")

        return {"unet_val": unet_loss_val}
