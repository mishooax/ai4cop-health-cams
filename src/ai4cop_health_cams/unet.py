"""
U-Net SR model. Similar to that of Sha et al., 2020
    https://journals.ametsoc.org/view/journals/apme/59/12/jamc-d-20-0057.1.xml
    https://github.com/yingkaisha/JAMC_20_0057
"""
from typing import Optional, Tuple, Union, List
import torch
from torch import nn

from ai4cop_health_cams.layers import ConvBlock, ConvBlockStack
from ai4cop_health_cams.logger import get_logger

LOGGER = get_logger(__name__)


class UNet_Down(nn.Module):

    _NUM_STACK = 1

    def __init__(
        self,
        nf_in: int,
        nf_out: int,
        kernel_size: Tuple[int, int] = (3, 3),
        pool_size: int = 2,
        pool: bool = True,
        activation: str = "leaky_relu",
    ) -> None:
        super().__init__()
        self.pool = pool

        if self.pool:
            self.maxpool = nn.MaxPool2d(pool_size)
        else:
            self.convblock = ConvBlock(
                nf_in, nf_out, kernel_size=kernel_size, norm="batch", stride=pool_size, activation=activation
            )

        n_inputs = nf_in if self.pool else nf_out
        self.convstack = ConvBlockStack(
            self._NUM_STACK, n_inputs, nf_out, kernel_size=kernel_size, norm="batch", activation=activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Given input x, returns the output of the encoder ("left") branch of the UNet"""
        # max-pool or strided conv
        if self.pool:
            x = self.maxpool(x)
        else:
            x = self.convblock(x)
        x = self.convstack(x)
        return x


class UNet_Up(nn.Module):
    """
    Decoder (Up-)block of (Nested-)UNet.
    See Sha et al., 2020, 10.1175/JAMC-D-20-0058.1
    """

    _NUM_STACK = 1

    def __init__(
        self,
        nf_in: int,  # input channels
        nf_skip: int,  # skip-link channels
        nf_out: int,  # output channels
        kernel_size: Tuple[int, int] = (3, 3),
        pool_size: int = 2,
        activation: str = "leaky_relu",
    ) -> None:
        super().__init__()
        self.nf_in = nf_in
        self.nf_skip = nf_skip
        self.nf_out = nf_out

        self.upsample = nn.Upsample(scale_factor=pool_size, mode="bilinear", align_corners=False)

        self.convstack1 = ConvBlockStack(self._NUM_STACK, nf_in, nf_out, kernel_size, norm="batch", activation=activation)
        self.convstack2 = ConvBlockStack(
            self._NUM_STACK, nf_skip + nf_out, nf_out, kernel_size, norm="batch", activation=activation
        )

    def forward(self, x: torch.Tensor, x_skip: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """
        Given inputs x and x_skip - the residual connection(s) from the "Unet-Down" (encoder) branch,
        returns the output of the right branch of the (Nested-)UNet.
        """

        # upsample -> stacked conv2d -> concat (skip-link) -> stacked conv2d
        x = self.upsample(x)
        x = self.convstack1(x)

        if isinstance(x_skip, List):
            x = torch.cat([x] + x_skip, dim=1)
        else:
            x = torch.cat([x, x_skip], dim=1)

        assert x.shape[1] == (
            self.nf_out + self.nf_skip
        ), f"Shape mismatch before final convstack! {x.shape[1]} vs. {self.nf_out + self.nf_skip} channels."

        x = self.convstack2(x)
        return x


class UNet(nn.Module):

    _NF_LAYERS = [32, 64, 128, 256]  # [64, 128, 256, 512]

    def __init__(
        self,
        num_inputs_lr: int,
        num_outputs: int,
        num_inputs_hr: int = 0,
        kernel_size: Tuple[int, int] = (3, 3),
        pooling: bool = True,
        norm: str = "batch",
        activation: str = "leaky_relu",
        dropout_rate: float = 0.0,
        autoencoder: bool = False,
    ) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate
        self.autoencoder = autoencoder

        LOGGER.debug("Inputs: LR = %d -- HR = %d", num_inputs_lr, num_inputs_hr)
        LOGGER.debug("Outputs:  %d", num_outputs)

        num_filters_lr = self._NF_LAYERS[0] // 2 if num_inputs_hr > 0 else self._NF_LAYERS[0]

        self.upscale_in_lr = nn.Sequential(
            *[
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                ConvBlock(num_inputs_lr, num_filters_lr, kernel_size, norm=norm, activation=activation),
            ]
        )

        if num_inputs_hr > 0:
            self.convstack_in_hr = ConvBlock(
                num_inputs_hr,
                self._NF_LAYERS[0] // 2,
                kernel_size=kernel_size,
                norm=norm,
                activation=activation,
            )

        self.down1 = UNet_Down(self._NF_LAYERS[0], self._NF_LAYERS[1], kernel_size=kernel_size, pool=pooling, activation=activation)
        self.down2 = UNet_Down(self._NF_LAYERS[1], self._NF_LAYERS[2], kernel_size=kernel_size, pool=pooling, activation=activation)
        self.down3 = UNet_Down(self._NF_LAYERS[2], self._NF_LAYERS[3], kernel_size=kernel_size, pool=pooling, activation=activation)

        self.dropout = nn.Dropout2d(dropout_rate)

        self.up3 = UNet_Up(
            self._NF_LAYERS[3], self._NF_LAYERS[2], self._NF_LAYERS[2], kernel_size=kernel_size, activation=activation
        )
        self.up2 = UNet_Up(
            self._NF_LAYERS[2], self._NF_LAYERS[1], self._NF_LAYERS[1], kernel_size=kernel_size, activation=activation
        )
        self.up1 = UNet_Up(
            self._NF_LAYERS[1], self._NF_LAYERS[0], self._NF_LAYERS[0], kernel_size=kernel_size, activation=activation
        )

        self.convstack_out = ConvBlock(
            self._NF_LAYERS[0],
            num_outputs,
            kernel_size=kernel_size,
            norm=None,
            activation="softplus",
        )

        if self.autoencoder:
            self.convstack_out_ae = ConvBlock(
                self._NF_LAYERS[0],
                num_outputs,
                kernel_size=kernel_size,
                norm=None,
                activation=None,
            )

    def forward(self, x_lr: torch.Tensor, x_hr: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Args:
            x_lr: low-res inputs
            x_hr: high-res inputs (orography, land-sea mask, built area, ...)
        Returns:
            The high-resolution (downscaled) output i.e. out = UNET(x_l, x_hr).
        """
        x = self.upscale_in_lr(x_lr)

        if x_hr is not None:
            # stacked-conv2D on HR input
            x_hr = self.convstack_in_hr(x_hr)
            x = torch.cat([x, x_hr], dim=1)

        # down branch
        x_down1 = self.down1(x)
        x_down2 = self.down2(x_down1)
        x_down3 = self.down3(x_down2)

        if self.dropout_rate >= 0.0:
            x_down3 = self.dropout(x_down3)

        # up branch
        x_up3 = self.up3(x_down3, x_down2)
        x_up2 = self.up2(x_up3, x_down1)
        x_up1 = self.up1(x_up2, x)

        out = self.convstack_out(x_up1)

        if self.autoencoder:
            return (out, self.convstack_out_ae(x_up1))

        return out


class XNet(nn.Module):

    _NF_LAYERS = [64, 128, 256, 512]

    def __init__(
        self,
        num_inputs_lr: int,
        num_outputs: int,
        num_inputs_hr: int = 0,
        kernel_size: Tuple[int, int] = (3, 3),
        pooling: bool = True,
        norm: str = "batch",
        activation: str = "leaky_relu",
        dropout_rate: float = 0.0,
    ) -> None:

        super().__init__()
        self.dropout_rate = dropout_rate

        LOGGER.debug("Inputs: LR = %d -- HR = %d", num_inputs_lr, num_inputs_hr)
        LOGGER.debug("Outputs:  %d", num_outputs)

        num_filters_lr = self._NF_LAYERS[0] // 2 if num_inputs_hr > 0 else self._NF_LAYERS[0]

        self.upscale_in_lr = nn.Sequential(
            *[
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                ConvBlock(num_inputs_lr, num_filters_lr, kernel_size, norm=norm, activation=activation),
            ]
        )

        if num_inputs_hr > 0:
            self.convstack_in_hr = ConvBlock(
                num_inputs_hr,
                self._NF_LAYERS[0] // 2,
                kernel_size=kernel_size,
                norm=norm,
                activation=activation,
            )

        # downsampling ("encoder")
        self.down_21 = UNet_Down(
            self._NF_LAYERS[0], self._NF_LAYERS[1], kernel_size=kernel_size, pool=pooling, activation=activation
        )
        self.down_31 = UNet_Down(
            self._NF_LAYERS[1], self._NF_LAYERS[2], kernel_size=kernel_size, pool=pooling, activation=activation
        )
        self.down_41 = UNet_Down(
            self._NF_LAYERS[2], self._NF_LAYERS[3], kernel_size=kernel_size, pool=pooling, activation=activation
        )

        self.dropout = nn.Dropout2d(dropout_rate)

        # upsampling ("decoder")
        self.up_32 = UNet_Up(
            self._NF_LAYERS[3], self._NF_LAYERS[2], self._NF_LAYERS[2], kernel_size=kernel_size, activation=activation
        )
        self.up_22 = UNet_Up(
            self._NF_LAYERS[2], self._NF_LAYERS[1], self._NF_LAYERS[1], kernel_size=kernel_size, activation=activation
        )
        self.up_12 = UNet_Up(
            self._NF_LAYERS[1], self._NF_LAYERS[0], self._NF_LAYERS[0], kernel_size=kernel_size, activation=activation
        )
        self.up_23 = UNet_Up(
            self._NF_LAYERS[2], 2 * self._NF_LAYERS[1], self._NF_LAYERS[1], kernel_size=kernel_size, activation=activation
        )
        self.up_13 = UNet_Up(
            self._NF_LAYERS[1], 2 * self._NF_LAYERS[0], self._NF_LAYERS[0], kernel_size=kernel_size, activation=activation
        )
        self.up_14 = UNet_Up(
            self._NF_LAYERS[1], 3 * self._NF_LAYERS[0], self._NF_LAYERS[0], kernel_size=kernel_size, activation=activation
        )

        self.convstack_out = ConvBlock(
            self._NF_LAYERS[0],
            num_outputs,
            kernel_size=kernel_size,
            norm=None,
            activation="softplus",
        )

    def forward(self, x_lr: torch.Tensor, x_hr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x_lr: low-res inputs
            x_hr: high-res inputs (orography, land-sea mask, built area, ...)
        Returns:
            The high-resolution (downscaled) output i.e. out = UNET(x_l, x_hr).
        """
        x_11 = self.upscale_in_lr(x_lr)  # 32 (or 64) x 64 x 64

        if x_hr is not None:
            # stacked-conv2D on HR input
            x_hr = self.convstack_in_hr(x_hr)  # 32 x 64 x 64
            x_11 = torch.cat([x_11, x_hr], dim=1)  # 64 x 64 x 64

        # down branch
        x_21 = self.down_21(x_11)  # 128 x 32 x 32
        x_31 = self.down_31(x_21)  # 256 x 16 x 16
        x_41 = self.down_41(x_31)  # 512 x  8 x  8

        if self.dropout_rate >= 0.0:
            x_41 = self.dropout(x_41)  # 512 x 8 x 8

        # up-sampling part 1
        x_32 = self.up_32(x_41, [x_31])  # 256 x 16 x 16
        x_22 = self.up_22(x_31, [x_21])  # 128 x 32 x 32
        x_12 = self.up_12(x_21, [x_11])  #  64 x 64 x 64

        # up-sampling part 2
        x_23 = self.up_23(x_32, [x_21, x_22])  # 128 x 32 x 32
        x_13 = self.up_13(x_22, [x_11, x_12])  #  64 x 64 x 64

        # up-sampling part 3
        x_14 = self.up_14(x_23, [x_11, x_12, x_13])  # 64 x 64 x 64

        # output
        out = self.convstack_out(x_14)  # num_outputs x 64 x 64
        return out
