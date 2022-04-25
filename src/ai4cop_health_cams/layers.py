from typing import Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F

from ai4cop_health_cams.logger import get_logger

LOGGER = get_logger(__name__)


def make_conv2d(
    nf_in: int, nf_out: int, kernel_size: Tuple[int, int], stride: int = 1, padding: int = 0, spectral_norm: bool = False, **kwargs
) -> nn.Module:
    """Convenience constructor for conv2d with and without spectral norm.
    Args:
        nf_in: number of input filters (channels)
        nf_out: number of output filters (channels)
        kernel_size: kernel size
        padding: padding size
        stride: stride length
        spectral_norm: use spectral normalization [T/F]
    """
    conv2d = nn.Conv2d(nf_in, nf_out, kernel_size=kernel_size, padding=padding, stride=stride, padding_mode="reflect", **kwargs)
    if spectral_norm:
        conv2d = nn.utils.spectral_norm(conv2d)
    return conv2d


class ResidualBlock(nn.Module):
    """Residual conv-block."""

    _KERNEL = 3
    _PADDING = _KERNEL // 2
    _LRU_ALPHA = 0.2

    def __init__(self, nf_in: int, nf_out: int, stride: int = 1, spectral_norm: bool = False, batch_norm: bool = False) -> None:
        """Initializes the residual block.
        Args:
            nf_in: number of input filters
            nf_out: number of output filters
            stride: stride of 2D convolution
            spectral_norm: use spectral normalization [T/F]
            batch_norm: use batch normalization [T/F]
        """
        super().__init__()
        self.batch_norm = batch_norm

        self.activation1 = nn.LeakyReLU(self._LRU_ALPHA)
        self.conv1 = make_conv2d(
            nf_in,
            nf_out,
            kernel_size=self._KERNEL,
            padding=self._PADDING,
            stride=stride,
            spectral_norm=spectral_norm,
        )
        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(nf_out)

        self.activation2 = nn.LeakyReLU(self._LRU_ALPHA)
        self.conv2 = make_conv2d(nf_out, nf_out, kernel_size=self._KERNEL, padding=self._PADDING, spectral_norm=spectral_norm)
        if self.batch_norm:
            self.bn2 = nn.BatchNorm2d(nf_out)

        # define the skip connection
        if nf_in != nf_out:
            self.skip_path = make_conv2d(nf_in, nf_out, kernel_size=1, spectral_norm=spectral_norm)
            if stride > 1:
                self.skip_path = nn.Sequential(nn.AvgPool2d(stride), self.skip_path)
        elif stride > 1:
            self.skip_path = nn.AvgPool2d(stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Given tensor input x, returns ResidualBlock(x)."""
        out = self.activation1(x)
        out = self.conv1(out)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.activation2(out)
        out = self.conv2(out)
        if self.batch_norm:
            out = self.bn2(out)
        if hasattr(self, "skip_path"):
            x = self.skip_path(x)
        return out + x


class UpsampleBlock(nn.Module):
    """Custom upsampling block:
    input -> nn.Upsample(scale_factor=2, mode='bilinear') -> ResidualBlock -> output
    """

    def __init__(
        self,
        nf_in: int,
        nf_out: int,
        scale_factor: int = 2,
        mode: str = "nearest",
        stride: int = 1,
        spectral_norm: bool = False,
        batch_norm: bool = False,
    ) -> None:
        """Initializes the upsampling block.
        Args:
            nf_in: number of input filters
            nf_out: number of output filters
            scale_factor: scale factor
            mode: interpolation mode (nearest neighbor, bilinear, ....)
            stride: convolution stride
            spectral_norm: use spectral normalization [T/F]
            batch_norm: use batch normalization [T/F]
        """
        super().__init__()
        align_corners = None if mode == "nearest" else False
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=align_corners)
        self.resblock = ResidualBlock(nf_in, nf_out, stride=stride, spectral_norm=spectral_norm, batch_norm=batch_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Given input tensor x, returns UpsampleBlock(x)"""
        out = self.upsample(x)
        out = self.resblock(out)
        return out


class UpsampleLayer(nn.Module):
    """Upsample layer"""

    def __init__(self, scale_factor: int = 2, mode: str = "bilinear", align_corners: bool = False):
        """Initializes the upsampling layer.
        Args:
            scale_factor: upsampling scale factor
            mode: upsampling method
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.align_corners = align_corners
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Given input tensor x, returns UpsampleLayer(x)"""
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return x


class ConvBlock(nn.Module):
    """Custom convolutional block: Conv2D -> BatchNorm2d -> (Leaky)ReLU
    The latter two ops are optional and can be disabled.
    """

    _LEAKY_RELU_ALPHA = 0.2

    def __init__(
        self,
        nf_in: int,
        nf_out: int,
        kernel_size: Tuple[int] = (3, 3),
        norm: Optional[str] = None,
        stride: int = 1,
        activation: Optional[str] = None,
        padding: int = 1,
        spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        self.norm = norm
        self.activation = activation
        self.conv = make_conv2d(nf_in, nf_out, kernel_size, padding=padding, stride=stride, spectral_norm=spectral_norm)
        if self.norm == "batch":
            self.bn = nn.BatchNorm2d(nf_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Given input tensor x, calculates ConvBlock(x)"""
        x = self.conv(x)
        if self.norm == "batch":
            x = self.bn(x)
        if self.activation == "leaky_relu":
            x = F.leaky_relu(x, negative_slope=self._LEAKY_RELU_ALPHA)
        elif self.activation == "relu":
            x = F.relu(x)
        elif self.activation == "softplus":
            x = F.softplus(x)
        return x


class ResidualBlockV2(nn.Module):
    """Custom residual block:
         ConvBlock -> ConvBlock
       /                         \
    x                             + -> output
        -                         /
         AvgPool2D -> ConvBlock
    """

    def __init__(
        self, in_planes: int = 256, planes: int = 256, stride: int = 1, activation: str = "relu", norm: Optional[str] = None
    ) -> None:
        super().__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        self.activation = activation
        self.norm = norm

        shortcut_modules = []
        if self.stride > 1:
            shortcut_modules.append(nn.AvgPool2d(self.stride))
        if self.planes != self.in_planes:
            shortcut_modules.append(ConvBlock(self.in_planes, self.planes, 1, stride=1, activation=None, padding=0))
        self.shortcut = nn.Sequential(*shortcut_modules)

        self.convblock1 = ConvBlock(self.in_planes, self.planes, 3, stride=self.stride, norm=self.norm, activation=self.activation)
        self.convblock2 = ConvBlock(self.planes, self.planes, 3, stride=1, norm=self.norm, activation=self.activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Given input tensor x, returns ResidualBlockV2(x)"""
        x_in = x
        x = self.convblock1(x)
        x = self.convblock2(x)
        x_in = self.shortcut(x_in)
        x = x + x_in
        return x


class ConvBlockStack(nn.Module):
    """A stack of ConvBlocks."""

    def __init__(
        self,
        stack_num: int,
        nf_in: int,
        nf_out: int,
        kernel_size: Tuple[int] = (3, 3),
        norm: Optional[str] = None,
        stride: int = 1,
        activation: Optional[str] = None,
        spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.convblocks = nn.Sequential(
            *[
                # the input may have a different number of filters
                ConvBlock(nf_in, nf_out, kernel_size, norm, stride, activation, self.padding, spectral_norm),
                *[
                    ConvBlock(nf_out, nf_out, kernel_size, norm, stride, activation, self.padding, spectral_norm)
                    for _ in range(stack_num - 1)
                ],
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Given input tensor x, calculates ConvBlockStack(x)"""
        return self.convblocks(x)


class ChannelAttention(nn.Module):
    def __init__(self, num_channels: int, reduction: int = 16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.seq = nn.Sequential(
            *[
                nn.Conv2d(num_channels, num_channels // reduction, 1, bias=False),
                nn.ReLU(),
                nn.Conv2d(num_channels // reduction, num_channels, 1, bias=False),
            ]
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.seq(max_result)
        avg_out = self.seq(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):
    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(num_channels=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + x
