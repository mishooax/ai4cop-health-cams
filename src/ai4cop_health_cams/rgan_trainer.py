from typing import Optional, Tuple, Dict, Callable

import random
import torch
from torch import nn

from ai4cop_health_cams.logger import get_logger
from ai4cop_health_cams.plots import plot_predicted_samples
from ai4cop_health_cams.gan_trainer import Lightning_GAN


LOGGER = get_logger(__name__)


class Lightning_RGAN(Lightning_GAN):
    """Relativistic (average) GAN. See (Jolicoeur-Martineau, 2018) - https://arxiv.org/abs/1807.00734"""

    _VAL_PLOT_FREQ = 50  # plot a generated sample every so many validation batches

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        gen_lr: float = 5e-5,
        disc_lr: float = 5e-5,
        gen_freq: int = 1,
        disc_freq: int = 1,
        gen_weight_decay: float = 1e-5,
        lambda_l1: float = 1.0,
        lambda_l2: float = 0.0,
        lambda_adv: float = 1.0,
        predict_ensemble_size: int = 10,
        save_basedir: Optional[str] = None,
        interactive: bool = False,
        wandb_logs: bool = False,
        inverse_transform: Optional[Callable] = None,
    ) -> None:
        """
        Trainer for the WGAN-GP.
        Args:
            generator, discriminator: models
            gen_lr, disc_lr: learning rates
            gen_freq, disc_freg: optimizer frequencies (update only every so many batches)
            gen_weight_decay: generator weight decay hyperparameter
            lambda_gp: gradient penalty hyperparameter
            lambda_l1, lambda_l2, lambda_adv: l1 / l2 / adversarial loss hyperparameters for the generator model
            predict_ensemble_size: number of "ensemble" members generated when making predictions
            save_basedir: basedir for plots
            interactive: running in interactive mode, i.e. inside a Jupyter notebook
            wandb_logs: use Weights&Biases to log media objects (generated sample plots)
        """
        super().__init__(
            generator,
            discriminator,
            gen_lr=gen_lr,
            disc_lr=disc_lr,
            gen_freq=gen_freq,
            disc_freq=disc_freq,
            gen_weight_decay=gen_weight_decay,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            lambda_adv=lambda_adv,
            predict_ensemble_size=predict_ensemble_size,
            save_basedir=save_basedir,
            interactive=interactive,
            wandb_logs=wandb_logs,
        )

        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.adversarial_loss = torch.nn.BCEWithLogitsLoss()
        self.inverse_transform = inverse_transform

    def _generator_step(self, batch: Tuple[Dict, ...]) -> torch.Tensor:
        """Single training step for the generator."""
        X_lr, X_hr, Y = self._get_batch_tensors(batch)

        noise = self._get_noise(X_lr)
        gen_fake = self.gen(noise, X_lr, X_hr)

        disc_fake = self.disc(X_lr, gen_fake, X_hr).reshape(-1)
        disc_real = self.disc(X_lr, Y, X_hr).reshape(-1).detach()

        ones = torch.ones_like(disc_fake)
        zeros = torch.zeros_like(disc_real)

        # relativistic loss
        real_loss = self.adversarial_loss(disc_real - disc_fake.mean(0, keepdim=True), zeros)
        fake_loss = self.adversarial_loss(disc_fake - disc_real.mean(0, keepdim=True), ones)
        loss_gen = self.lambda_adv * (fake_loss + real_loss)

        self.log("gen_adv", loss_gen, on_epoch=True, on_step=True, prog_bar=True, logger=True)

        if self.lambda_l1 > 0.0:
            # L1 error term contributes to generator loss
            loss_gen_l1 = self.l1_loss(gen_fake, Y)
            loss_gen += loss_gen_l1
        else:
            loss_gen_l1 = self.l1_loss(gen_fake.detach(), Y)
        if self.lambda_l2 > 0.0:
            # L2 error term contributes to generator loss
            loss_gen_l2 = self.l2_loss(gen_fake, Y)
            loss_gen += loss_gen_l2
        else:
            loss_gen_l2 = self.l2_loss(gen_fake.detach(), Y)

        self.log("gen_l1", loss_gen_l1, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        self.log("gen_l2", loss_gen_l2, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        self.log("gen_loss", loss_gen, on_epoch=True, on_step=True, prog_bar=True, logger=True)

        return loss_gen

    def _discriminator_step(self, batch: Tuple[Dict, ...]) -> torch.Tensor:
        """Single training step for the discriminator."""
        X_lr, X_hr, real = self._get_batch_tensors(batch)

        # concatenate 4D noise vector along the channel (filter) dimension
        noise = self._get_noise(X_lr)

        fake = self.gen(noise, X_lr, X_hr)

        disc_real = self.disc(X_lr, real, X_hr).reshape(-1)
        disc_fake = self.disc(X_lr, fake.detach(), X_hr).reshape(-1)

        ones = torch.ones_like(disc_real)
        zeros = torch.zeros_like(disc_real)

        real_loss = self.adversarial_loss(disc_real - disc_fake.mean(0, keepdim=True), ones)
        fake_loss = self.adversarial_loss(disc_fake - disc_real.mean(0, keepdim=True), zeros)
        loss_disc = real_loss + fake_loss

        self.log("disc_loss", loss_disc, on_epoch=True, on_step=True, prog_bar=True, logger=True)

        return loss_disc

    def validation_step(self, batch: Tuple[Dict, ...], batch_idx: int) -> Dict:
        X_lr, X_hr, real = self._get_batch_tensors(batch)

        with torch.no_grad():
            noise = self._get_noise(X_lr)
            fake = self.gen(noise, X_lr, X_hr)
            loss_gen_val = self.l1_loss(fake, real)

            disc_real = self.disc(X_lr, real, X_hr).reshape(-1)
            disc_fake = self.disc(X_lr, fake.detach(), X_hr).reshape(-1)

            ones = torch.ones_like(disc_real)
            zeros = torch.zeros_like(disc_real)

            real_loss = self.adversarial_loss(disc_real - disc_fake.mean(0, keepdim=True), ones)
            fake_loss = self.adversarial_loss(disc_fake - disc_real.mean(0, keepdim=True), zeros)

            loss_disc_val = 0.5 * (real_loss + fake_loss)

            self.log("gen_val_l1", loss_gen_val, on_epoch=True, on_step=False, prog_bar=True, logger=True)
            self.log("disc_val", loss_disc_val, on_epoch=True, on_step=False, prog_bar=True, logger=True)

            if batch_idx % self._VAL_PLOT_FREQ == 0:
                sample_index = random.randrange(real.shape[0])
                fig = plot_predicted_samples(
                    X_lr.cpu().numpy(),
                    fake.cpu().numpy(),
                    real.cpu().numpy(),
                    i_sample=sample_index,
                    inverse_transform=self.inverse_transform,
                )
                self._output_figure(fig, tag="rgan_predicted_val_samples_batch{batch_idx:04d}")

        return {"gen_val": loss_gen_val, "disc_val": loss_disc_val}
