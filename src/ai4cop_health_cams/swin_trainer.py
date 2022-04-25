from typing import Optional, Tuple, Dict, Callable
import random

import torch
from torch import nn
import torch.nn.functional as F

from ai4cop_health_cams.logger import get_logger
from ai4cop_health_cams.plots import plot_predicted_samples
from ai4cop_health_cams.trainer import Lightning_Model


LOGGER = get_logger(__name__)


class Lightning_SwinIR(Lightning_Model):
    """SwinIR trainer module. See https://arxiv.org/abs/2108.10257."""

    _VAL_PLOT_FREQ = 50  # plot a validation sample every so many batches

    def __init__(
        self,
        swin_ir: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        save_basedir: Optional[str] = None,
        interactive: bool = False,
        wandb_logs: bool = False,
        inverse_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(save_basedir=save_basedir, interactive=interactive, wandb_logs=wandb_logs)
        self.inverse_transform = inverse_transform
        self.swin_ir = swin_ir
        self.lr = lr
        self.weight_decay = weight_decay

        self.b1 = 0.5
        self.b2 = 0.999

        self.tag = "swin-ir"

    def forward(self, x_lr: torch.Tensor) -> torch.Tensor:
        return self.swin_ir(x_lr)

    def predict_step(self, batch: Tuple[dict], batch_idx: int, dataloader_idx: Optional[int] = None) -> torch.Tensor:
        del batch_idx, dataloader_idx  # not used
        X, _, _ = self._get_batch_tensors(batch)
        return self.swin_ir(X)

    def configure_optimizers(self):
        swin_opt = torch.optim.Adam(self.swin_ir.parameters(), lr=self.lr, betas=(self.b1, self.b2), weight_decay=self.weight_decay)
        swin_lrs = torch.optim.lr_scheduler.ReduceLROnPlateau(
            swin_opt,
            mode="min",
            factor=0.3,
            patience=3,
            verbose=False,
        )
        return {
            "optimizer": swin_opt,
            "lr_scheduler": {
                "scheduler": swin_lrs,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "swin_val",
                "name": "reduce_on_plateau_lr",
            },
        }

    def training_step(self, batch: Tuple[Dict, ...]) -> torch.Tensor:
        X, _, Y = self._get_batch_tensors(batch)
        Y_fake = self.swin_ir(X)
        swin_loss = F.l1_loss(Y_fake, Y)
        self.log("swin_loss", swin_loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        return swin_loss

    def validation_step(self, batch: Tuple[Dict, ...], batch_idx: int) -> Dict:
        """U-Net validation step."""
        with torch.no_grad():
            X_lr, _, Y = self._get_batch_tensors(batch)
            Y_fake = self.swin_ir(X_lr)
            swin_loss_val = F.l1_loss(Y_fake, Y)

            self.log("swin_val", swin_loss_val, on_epoch=True, on_step=False, prog_bar=True, logger=True)

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

        return {"swin_val": swin_loss_val}
