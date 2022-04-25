from typing import Optional, Tuple, Dict, List

import abc
import torch
from torch import nn

from ai4cop_health_cams.logger import get_logger
from ai4cop_health_cams.trainer import Lightning_Model


LOGGER = get_logger(__name__)


class Lightning_GAN(Lightning_Model):
    """Abstract GAN mode containing common boilerplate code. Meant to be sub-classed."""

    _VAL_PLOT_FREQ = 50  # plot a generated sample every so many validation batches
    _DEBUG = True  # set to True to enable anomaly detection during training

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        gen_lr: float = 5e-5,
        disc_lr: float = 5e-5,
        gen_freq: int = 1,
        disc_freq: int = 4,
        gen_weight_decay: float = 1e-5,
        lambda_l1: float = 10.0,
        lambda_l2: float = 0.0,
        lambda_adv: float = 1.0,
        predict_ensemble_size: int = 10,
        save_basedir: Optional[str] = None,
        interactive: bool = False,
        wandb_logs: bool = False,
    ) -> None:
        """
        Generic GAN model interface. Meant to be subclassed. See e.g. the RGAN and WGAN-GP modules.
        Args:
            generator, discriminator: models
            gen_lr, disc_lr: learning rates
            gen_freq, disc_freg: optimizer frequencies (update only every so many batches)
            gen_weight_decay: generator weight decay hyperparameter
            lambda_gp: gradient penalty hyperparameter
            lambda_l1, lambda_l2, lambda_adv: l1 / l2 / adversarial loss hyperparameters
            predict_ensemble_size: size of ensemble generated in predict mode
            save_basedir: basedir for plots
            interactive: running in interactive mode, i.e. inside a Jupyter notebook
            wandb_logs: use Weights&Biases to log media objects (generated sample plots)
        """
        super().__init__(save_basedir=save_basedir, interactive=interactive, wandb_logs=wandb_logs)
        self.gen = generator
        self.disc = discriminator
        self.gen_lr = gen_lr
        self.disc_lr = disc_lr
        self.disc_freq, self.gen_freq = disc_freq, gen_freq
        self.gen_weight_decay = gen_weight_decay
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.lambda_adv = lambda_adv
        self.b1 = 0.5
        self.b2 = 0.99
        self.predict_ensemble_size = predict_ensemble_size

        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

    def forward(self, z: torch.Tensor, x_lr: torch.Tensor, x_hr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward operator (use this to generate new hi-res samples)"""
        return self.gen(z, x_lr, x_hr)

    def predict_step(self, batch: Tuple[Dict, ...], batch_idx: int, dataloader_idx: Optional[int] = None) -> torch.Tensor:
        del batch_idx, dataloader_idx  # not used
        X_lr, X_hr, _ = self._get_batch_tensors(batch)
        predictions: List[torch.Tensor] = []
        for _ in range(self.predict_ensemble_size):
            # run N predictions with different noise inputs to generate a pseudo-"ensemble"
            noise = self._get_noise(X_lr)
            predictions.append(self.gen(noise, X_lr, X_hr))
        return torch.stack(predictions, dim=-1)

    @abc.abstractmethod
    def _generator_step(self, batch: Tuple[Dict, ...]) -> torch.Tensor:
        """Single training step for the generator."""
        raise NotImplementedError

    @abc.abstractmethod
    def _discriminator_step(self, batch: Tuple[Dict, ...]) -> torch.Tensor:
        """Single training step for the discriminator."""
        raise NotImplementedError

    def training_step(self, batch: Tuple[Dict, ...], batch_idx: int, optimizer_idx: int) -> torch.Tensor:
        """Implements a single training step."""
        del batch_idx  # not used
        if optimizer_idx == 0:
            # train discriminator
            with torch.autograd.set_detect_anomaly(self._DEBUG):
                disc_loss = self._discriminator_step(batch)
            return disc_loss
        if optimizer_idx == 1:
            # train generator
            with torch.autograd.set_detect_anomaly(self._DEBUG):
                gen_loss = self._generator_step(batch)
            return gen_loss
        raise RuntimeError(f"Invalid optimizer index {optimizer_idx}!")

    @abc.abstractmethod
    def validation_step(self, batch: Tuple[Dict, ...], batch_idx: int) -> Dict:
        """Single validation step"""
        raise NotImplementedError

    def configure_optimizers(self):
        """Configures the optimizers for the generator and discriminator."""
        disc_opt = torch.optim.Adam(self.disc.parameters(), lr=self.disc_lr, betas=(self.b1, self.b2))
        gen_opt = torch.optim.Adam(
            self.gen.parameters(), lr=self.gen_lr, betas=(self.b1, self.b2), weight_decay=self.gen_weight_decay
        )
        return [{"optimizer": disc_opt, "frequency": self.disc_freq}, {"optimizer": gen_opt, "frequency": self.gen_freq}]
