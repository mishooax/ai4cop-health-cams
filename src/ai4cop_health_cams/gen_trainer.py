from typing import Optional, Tuple, Dict, Callable

import random
import torch
from torch import nn

from ai4cop_health_cams.logger import get_logger
from ai4cop_health_cams.trainer import Lightning_Model
from ai4cop_health_cams.plots import plot_predicted_samples
from ai4cop_health_cams.losses import energy_score_loss

LOGGER = get_logger(__name__)


class Lightning_GAN_Generator(Lightning_Model):
    """Pre-trainer module for a generator model that is part of a GAN."""

    _VAL_PLOT_FREQ = 50  # plot a generated sample every so many validation batches
    _DEBUG = True  # if True then enable anomaly detection (this will result in slower training)

    def __init__(
        self,
        generator: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        lambda_l1: float = 1.0,
        lambda_l2: float = 0.0,
        predict_ensemble_size: int = 1,
        save_basedir: Optional[str] = None,
        interactive: bool = False,
        wandb_logs: bool = False,
        inverse_transform: Optional[Callable] = None,
    ) -> None:
        """
        Trainer for the WGAN-GP.
        Args:
            generator: generator model
            weight_decay: weight decay hyperparameter
            lambda_l1, lambda_l2: l1 / l2 loss hyperparameters
            predict_ensemble_size: number of "ensemble" members generated when making predictions
            save_basedir: basedir for plots
            interactive: running in interactive mode, i.e. inside a Jupyter notebook
            wandb_logs: use Weights&Biases to log media objects (generated sample plots)
        """
        super().__init__(save_basedir=save_basedir, interactive=interactive, wandb_logs=wandb_logs)

        self.gen = generator
        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.b1 = 0.5
        self.b2 = 0.99
        self.predict_ensemble_size = predict_ensemble_size
        self.inverse_transform = inverse_transform

        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, z: torch.Tensor, x_lr: torch.Tensor, x_hr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward operator (use this to generate new hi-res samples)"""
        return self.gen(z, x_lr, x_hr)

    def predict_step(self, batch: Tuple[Dict, ...], batch_idx: int, dataloader_idx: Optional[int] = None) -> torch.Tensor:
        del batch_idx, dataloader_idx  # not used
        X_lr, X_hr, _ = self._get_batch_tensors(batch)
        noise = self._get_noise(X_lr)
        return self.gen(noise, X_lr, X_hr)

    def _generator_step(self, batch: Tuple[Dict, ...]) -> torch.Tensor:
        """Single training step for the generator."""
        X_lr, X_hr, Y = self._get_batch_tensors(batch)

        noise = self._get_noise(X_lr)
        Y_fake = self.gen(noise, X_lr, X_hr)

        # L1 error term contributes to generator loss
        loss_gen = self._generator_loss(Y_fake, Y)
        self.log("gen_loss_pretrain", loss_gen, on_epoch=True, on_step=True, prog_bar=True, logger=True)

        return loss_gen

    def _generator_loss(self, Y_fake: torch.Tensor, Y_true: torch.Tensor) -> torch.Tensor:
        loss_gen_l1 = self.l1_loss(Y_fake, Y_true)
        loss_gen_l2 = self.l2_loss(Y_fake, Y_true)
        loss_gen = self.lambda_l1 * loss_gen_l1 + self.lambda_l2 * loss_gen_l2

        self.log("gen_l1_pretrain", loss_gen_l1, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        self.log("gen_l2_pretrain", loss_gen_l2, on_epoch=True, on_step=True, prog_bar=True, logger=True)

        return loss_gen

    def training_step(self, batch: Tuple[Dict, ...], batch_idx: int) -> torch.Tensor:
        """Implements a single training step."""
        del batch_idx  # not used
        with torch.autograd.set_detect_anomaly(self._DEBUG):
            gen_loss = self._generator_step(batch)
        return gen_loss

    def validation_step(self, batch: Tuple[Dict, ...], batch_idx: int) -> Dict:
        X_lr, X_hr, real = self._get_batch_tensors(batch)

        with torch.no_grad():
            noise = self._get_noise(X_lr)
            fake_imgs = self.gen(noise, X_lr, X_hr)
            mean_fake = fake_imgs.mean(dim=1, keepdim=True)  # pseudo-ensemble mean
            mean_std = fake_imgs.std(dim=1, keepdim=True)  # pseudo-ensemble std
            loss_gen_val = self.l1_loss(mean_fake, real)

            self.log("gen_val_l1_pretrain", loss_gen_val, on_epoch=True, on_step=False, prog_bar=True, logger=True)

            if batch_idx % self._VAL_PLOT_FREQ == 0:
                sample_index = random.randrange(real.shape[0])
                fig = plot_predicted_samples(
                    X_lr.cpu().numpy(),
                    mean_fake.cpu().numpy(),
                    real.cpu().numpy(),
                    ensemble_spread=mean_std.cpu().numpy(),
                    i_sample=sample_index,
                    inverse_transform=self.inverse_transform,
                )
                self._output_figure(fig, tag="gen_predicted_val_samples_batch{batch_idx:04d}")

        return {"gen_val": loss_gen_val}

    def configure_optimizers(self) -> Dict:
        """Configures the optimizers for the generator and discriminator."""
        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=self.lr, betas=(self.b1, self.b2), weight_decay=self.weight_decay)
        return {"optimizer": gen_opt}


class Lightning_GAN_EnsembleGenerator(Lightning_GAN_Generator):
    def __init__(
        self,
        generator: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        lambda_l1: float = 0.0,
        lambda_l2: float = 0.0,
        predict_ensemble_size: int = 8,
        save_basedir: Optional[str] = None,
        interactive: bool = False,
        wandb_logs: bool = False,
        inverse_transform: Optional[Callable] = None,
    ) -> None:
        """
        Trainer for the WGAN-GP.
        Args:
            generator: generator model
            lr: learning rate
            weight_decay: weight decay hyperparameter
            lambda_l1, lambda_l2: l1 / l2 loss hyperparameters [not used]
            predict_ensemble_size: number of "ensemble" members generated when making predictions
            save_basedir: basedir for plots
            interactive: running in interactive mode, i.e. inside a Jupyter notebook
            wandb_logs: use Weights&Biases to log media objects (generated sample plots)
            inverse_transform: inverse transformation function
        """
        super().__init__(
            generator,
            lr=lr,
            weight_decay=weight_decay,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            predict_ensemble_size=predict_ensemble_size,
            save_basedir=save_basedir,
            interactive=interactive,
            wandb_logs=wandb_logs,
            inverse_transform=inverse_transform,
        )
        if predict_ensemble_size <= 1:
            LOGGER.error(
                "Invalid ensemble size %d - 'predict_ensemble_size' must be > 1! Check your config file settings ...",
                predict_ensemble_size,
            )
            raise RuntimeError
        self.save_hyperparameters()

    def _generator_step(self, batch: Tuple[Dict, ...]) -> torch.Tensor:
        """Single training step for the generator."""
        X_lr, X_hr, Y = self._get_batch_tensors(batch)

        noise = self._get_noise(X_lr)
        Y_fake_ens = self.gen(noise, X_lr, X_hr).permute((0, 2, 3, 1))
        loss_gen = energy_score_loss(Y_fake_ens[:, None, ...], Y)
        self.log("gen_loss_energy_pretrain", loss_gen, on_epoch=True, on_step=True, prog_bar=True, logger=True)

        return loss_gen
