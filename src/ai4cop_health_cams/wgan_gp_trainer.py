from typing import Optional, Tuple, Dict, Callable

import random
import torch
from torch import nn


# import torchvision
from ai4cop_health_cams.gan_trainer import Lightning_GAN
from ai4cop_health_cams.logger import get_logger
from ai4cop_health_cams.plots import plot_predicted_samples


LOGGER = get_logger(__name__)


class Lightning_WGAN_GP(Lightning_GAN):
    """Wasserstein GAN with gradient penalty. Uses Pytorch Lightning."""

    _VAL_PLOT_FREQ = 50  # plot a generated sample every so many validation batches

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        gen_lr: float = 5e-5,
        disc_lr: float = 5e-5,
        gen_freq: int = 1,
        disc_freq: int = 4,
        gen_weight_decay: float = 1e-5,
        lambda_gp: float = 10.0,
        lambda_l1: float = 10.0,
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
            lambda_l1, lambda_l2: l1 / l2 loss hyperparameters
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

        self.lambda_gp = lambda_gp
        self.inverse_transform = inverse_transform

        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

    def _generator_step(self, batch: Tuple[Dict, ...]) -> torch.Tensor:
        """Single training step for the generator."""
        X_lr, X_hr, Y = self._get_batch_tensors(batch)

        noise = self._get_noise(X_lr)
        gen_fake = self.gen(noise, X_lr, X_hr)
        disc_fake = self.disc(X_lr, gen_fake, X_hr).reshape(-1)
        loss_gen = -torch.mean(disc_fake) * self.lambda_adv

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
        disc_real = self.disc(X_lr, real, X_hr)
        disc_fake = self.disc(X_lr, fake.detach(), X_hr)

        loss_disc = -torch.mean(disc_real) + torch.mean(disc_fake)

        gp = self._gradient_penalty(X_lr, real, fake.detach(), X_hr)
        # gp = self._r1_penalty(real, disc_real)

        if torch.isfinite(gp):
            loss_disc += self.lambda_gp * gp
            self.log("disc_gp", self.lambda_gp * gp, on_epoch=True, on_step=True, prog_bar=True, logger=True)

        self.log("disc_loss", loss_disc, on_epoch=True, on_step=True, prog_bar=True, logger=True)

        return loss_disc

    def validation_step(self, batch: Tuple[Dict, ...], batch_idx: int) -> Dict:
        X_lr, X_hr, real = self._get_batch_tensors(batch)

        with torch.no_grad():
            noise = self._get_noise(X_lr)
            fake = self.gen(noise, X_lr, X_hr)
            loss_gen_val = self.l1_loss(fake, real)
            disc_real = self.disc(X_lr, real, X_hr)
            disc_fake = self.disc(X_lr, fake, X_hr)
            loss_disc_val = -torch.mean(disc_real) + torch.mean(disc_fake)

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
                self._output_figure(fig, tag="wgan_gp_predicted_val_samples_batch{batch_idx:04d}")

        return {"gen_val": loss_gen_val, "disc_val": loss_disc_val}

    def _gradient_penalty(
        self, X: torch.Tensor, real: torch.Tensor, fake: torch.Tensor, X_hr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculates the gradient penalty term for the WGAN model."""
        bs = real.shape[0]
        epsilon = torch.rand(bs, 1, 1, 1).expand_as(real).type_as(real)
        interp = real * epsilon + fake * (1.0 - epsilon)
        interp = interp.requires_grad_(True)
        mixed_scores = self.disc(X, interp, X_hr)

        self.log("interp_l2", interp.norm(), on_epoch=True, on_step=True, prog_bar=True, logger=True)
        self.log("mix_scores_l2", mixed_scores.norm(), on_epoch=True, on_step=True, prog_bar=True, logger=True)

        gradient = torch.autograd.grad(
            inputs=interp,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            only_inputs=True,
        )[0]

        gradient_norm = torch.linalg.norm(gradient.view(bs, -1), ord=2, dim=-1)
        return torch.mean((gradient_norm - 1.0) ** 2)

    def _r1_penalty(self, y_real: torch.Tensor, disc_real: torch.Tensor) -> torch.Tensor:
        """Calculates the R1 penalty term for the WGAN model."""
        bs = y_real.shape[0]
        y_real = y_real.requires_grad_(True)

        gradient = torch.autograd.grad(
            inputs=y_real,
            outputs=disc_real,
            grad_outputs=torch.ones_like(disc_real),
            create_graph=True,
            only_inputs=True,
        )[0]

        gradient_norm = torch.linalg.norm(gradient.view(bs, -1), ord=2, dim=-1)
        self.log("r1_grad_norm", gradient_norm, on_epoch=True, on_step=True, prog_bar=True, logger=True)

        return torch.mean(gradient_norm ** 2)
