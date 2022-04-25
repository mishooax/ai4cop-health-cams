from typing import Optional  # , List

import torch
from torch import nn
import torch.nn.functional as F

from ai4cop_health_cams.layers import ConvBlock, ResidualBlock, UpsampleBlock
from ai4cop_health_cams.logger import get_logger

LOGGER = get_logger(__name__)


class Generator(nn.Module):

    _NUM_UPBLOCKS = 3
    _NUM_RES_BLOCKS = 1
    _KERNEL_SIZE = 3
    _PADDING = _KERNEL_SIZE // 2
    _LRU_ALPHA = 0.2
    _NUM_FILTERS = 256  # should be >= 64!

    def __init__(
        self,
        num_inputs_lr: int,
        num_outputs: int,
        num_inputs_hr: int = 0,
        activation_out: Optional[str] = None,
        spectral_norm: bool = False,
    ) -> None:

        super().__init__()
        self.activation_out = activation_out

        num_filters_lr = self._NUM_FILTERS - 1 if num_inputs_hr == 0 else self._NUM_FILTERS // 2 - 1

        self.in_block_lr = ResidualBlock(num_inputs_lr, num_filters_lr, spectral_norm=spectral_norm)

        if num_inputs_hr > 0:
            self.in_block_hr = nn.Sequential(
                *[
                    ResidualBlock(num_inputs_hr, self._NUM_FILTERS // 32, stride=2, spectral_norm=spectral_norm),
                    ResidualBlock(self._NUM_FILTERS // 32, self._NUM_FILTERS // 8, stride=2, spectral_norm=spectral_norm),
                    ResidualBlock(self._NUM_FILTERS // 8, self._NUM_FILTERS // 2, stride=2, spectral_norm=spectral_norm),
                ]
            )

        self.res_block_lr = nn.Sequential(
            *[ResidualBlock(self._NUM_FILTERS, self._NUM_FILTERS, spectral_norm=spectral_norm) for _ in range(self._NUM_RES_BLOCKS)]
        )

        self.up_blocks = nn.Sequential(
            *[
                UpsampleBlock(
                    self._NUM_FILTERS // 2 ** i,
                    self._NUM_FILTERS // 2 ** (i + 1),
                    mode="bilinear",
                    spectral_norm=spectral_norm,
                    batch_norm=False,
                )
                for i in range(self._NUM_UPBLOCKS)
            ]
        )

        # final convolutional layer
        self.conv_out = ConvBlock(
            self._NUM_FILTERS // 2 ** self._NUM_UPBLOCKS,
            num_outputs,
            activation=None,
            spectral_norm=spectral_norm,
        )

    def forward(self, z: torch.Tensor, x_lr: torch.Tensor, x_hr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Given input tensors x = [z, x_lr, x_hr], returns Generator(x).
        Args:
            z: noise vector, shape == (bs, 1, h_lowres, w_lowres)
            x_lr: low-res input, shape == (bs, n_inputs_lowres, h_lowres, w_lowres)
            x_hr: high-res input (optional), shape == (bs, n_inputs_hires, h_hires, w_hires)
        Returns
            The generator output, shape == (bs, n_outputs, h_hires, w_hires)
        """

        x_lr_conv = self.in_block_lr(x_lr)
        out = torch.cat([x_lr_conv, z], dim=1)

        if x_hr is not None:
            x_hr = self.in_block_hr(x_hr)
            out = torch.cat([out, x_hr], dim=1)

        # residual blocks (low-res output)
        out = self.res_block_lr(out)

        # downscaling (upsampling) residual blocks
        out = self.up_blocks(out)

        # final convolution layer
        out = self.conv_out(out)

        # softplus activation (optional)
        if self.activation_out == "softplus":
            out = F.softplus(out)
        elif self.activation_out == "sigmoid":
            out = torch.sigmoid(out)
        elif self.activation_out == "tanh":
            out = F.tanh(out)

        return out


class Discriminator(nn.Module):
    """Conditional GAN discriminator."""

    _LRU_ALPHA = 0.2
    _NUM_OUT_LINEAR = 64
    _NUM_FILTERS = 256
    _NUM_RES_BLOCKS = 1

    def __init__(
        self,
        num_inputs_lr: int,
        num_inputs_hr: int,
        num_inputs_hr_const: int = 0,
        spectral_norm: bool = False,
    ) -> None:
        """Initializes the discriminator object.
        Args:
            num_inputs_lr, num_inputs_hr: number of low-res / hi-res input variables
            num_inputs_hr_const: number of input hi-res fields
            num_filters: number of convolutional filters
            spectral_norm: enable spectral norms for the various layers
        """
        # Initialize object:
        super().__init__()

        ################
        # LR branch
        ################
        self.conv_in_lr = ResidualBlock(num_inputs_lr, self._NUM_FILTERS, spectral_norm=spectral_norm)

        if num_inputs_hr_const > 0:
            self.conv_in_hr_const = nn.Sequential(
                *[
                    ResidualBlock(num_inputs_hr_const, self._NUM_FILTERS // 16, stride=2, spectral_norm=spectral_norm),
                    ResidualBlock(self._NUM_FILTERS // 16, self._NUM_FILTERS // 4, stride=2, spectral_norm=spectral_norm),
                    ResidualBlock(self._NUM_FILTERS // 4, self._NUM_FILTERS, stride=2, spectral_norm=spectral_norm),
                ]
            )

            self.resblock_merge_lr = ResidualBlock(self._NUM_FILTERS * 2, self._NUM_FILTERS, spectral_norm=spectral_norm)

        self.resblocks_lr = nn.Sequential(
            *[ResidualBlock(self._NUM_FILTERS, self._NUM_FILTERS, spectral_norm=spectral_norm) for _ in range(self._NUM_RES_BLOCKS)]
        )

        # merge in (down-conv'ed) HR signal
        self.resblock_merge_hr = ResidualBlock(self._NUM_FILTERS * 2, self._NUM_FILTERS, spectral_norm=spectral_norm)

        #############
        # HR branch
        #############
        self.conv_in_hr = nn.Sequential(
            *[
                ResidualBlock(num_inputs_hr, self._NUM_FILTERS // 16, stride=2, spectral_norm=spectral_norm),
                ResidualBlock(self._NUM_FILTERS // 16, self._NUM_FILTERS // 4, stride=2, spectral_norm=spectral_norm),
                ResidualBlock(self._NUM_FILTERS // 4, self._NUM_FILTERS, stride=2, spectral_norm=spectral_norm),
            ]
        )

        self.resblocks_hr = nn.Sequential(
            *[ResidualBlock(self._NUM_FILTERS, self._NUM_FILTERS, spectral_norm=spectral_norm) for _ in range(self._NUM_RES_BLOCKS)]
        )

        # Dense layers (output)
        linear1 = nn.Linear(self._NUM_FILTERS * 2, self._NUM_OUT_LINEAR)
        linear2 = nn.Linear(self._NUM_OUT_LINEAR, num_inputs_hr)

        if spectral_norm:
            linear1 = nn.utils.spectral_norm(linear1)
            linear2 = nn.utils.spectral_norm(linear2)

        self.final_layers = nn.Sequential(
            linear1,
            nn.LeakyReLU(self._LRU_ALPHA),
            linear2,
        )

    def forward(self, x_lr: torch.Tensor, x_hr_out: torch.Tensor, x_hr_in: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Discriminator module logic.
        Calculates an output given input tensors x_lr (LR input), x_hr_out (HR output), and x_hr_in (HR input).
        Args:
            x_lr, x_hr_out, x_hr_in: input tensors (low-res, high-res fake / truth)
        Returns:
            The discriminator output, shape (batch_size, num_outputs)
        """
        x_lr = self.conv_in_lr(x_lr)

        if x_hr_in is not None:
            out = self.conv_in_hr_const(x_hr_in)
            x_lr = torch.cat([x_lr, out], dim=1)
            x_lr = self.resblock_merge_lr(x_lr)

        out_lr = self.resblocks_lr(x_lr)

        # HR
        x_hr_out = self.conv_in_hr(x_hr_out)
        out_hr = self.resblocks_hr(x_hr_out)

        # re-cat (with down-convolved HR input)
        out_lr = torch.cat([out_lr, x_hr_out], dim=1)
        out_lr = self.resblock_merge_hr(out_lr)

        # global average pooling
        out_lr = out_lr.mean([2, 3])  # (bs, NUM_FILTERS)
        out_hr = out_hr.mean([2, 3])  # (bs, NUM_FILTERS)

        # merged paths
        out = torch.cat([out_lr, out_hr], axis=1)  # (bs, NUM_FILTERS * 2)
        out = self.final_layers(out)  # (bs, num_inputs_hr)
        return out


# if __name__ == "__main__":
#     # TODO: refactor this as a unit test!
#     # test the generator
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     gen = Generator(
#         num_inputs_lr=1,
#         num_outputs=1,
#         num_inputs_hr=3,
#         activation_out="softplus",
#         spectral_norm=False,
#     ).to(device)

#     z_ = torch.rand((32, 1, 16, 16)).to(device)
#     x_lr_ = torch.rand((32, 1, 16, 16)).type_as(z_)
#     x_hr_ = torch.rand((32, 3, 128, 128)).type_as(z_)

#     y_ = gen.forward(z_, x_lr_, x_hr_)
#     LOGGER.debug("Generator output shape: %s", y_.size())

#     # test the generator
#     disc = Discriminator(
#         num_inputs_lr=1,
#         num_inputs_hr=1,
#         num_inputs_hr_const=3,
#         spectral_norm=False,
#     ).to(device)

#     x_lr_ = torch.rand((32, 1, 16, 16)).type_as(z_)
#     x_hr_ = torch.rand((32, 1, 128, 128)).type_as(z_)
#     x_hr_const_ = torch.rand((32, 3, 128, 128)).type_as(z_)

#     y_ = disc.forward(x_lr_, x_hr_, x_hr_const_)
#     LOGGER.debug("Discriminator output shape: %s", y_.size())
