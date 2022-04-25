"""
   SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
   Copyright [2021] [Ze Liu, Modified by Jingyun Liang and Mihai Alexe]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

from typing import Tuple, Optional, Union

import math
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from ai4cop_health_cams.logger import get_logger
from ai4cop_health_cams.layers import ConvBlockStack, ResidualBlock

LOGGER = get_logger(__name__)


class MLP(nn.Module):
    """MLP module"""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    # LOGGER.debug("window_partition: input x.shape %s", x.shape)
    # LOGGER.debug("window_partition: window_size %d", window_size)
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both shifted and non-shifted windows.
    """

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        """
        Args:
            dim: Number of input channels.
            window_size: The height and width of the window.
            num_heads: Number of attention heads.
            qkv_bias: If True, add a learnable bias to query, key, value
            qk_scale: Overrides default qk scale of 1/sqrt(head_dim) if set
            attn_drop: Dropout ratio of attention weight
            proj_drop: Dropout ratio of output
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        trunc_normal_(self.relative_position_bias_table, std=0.02)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block."""

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        """
        Args:
            dim: Number of input channels.
            input_resolution: Input resulotion.
            num_heads: Number of attention heads.
            window_size: Window size.
            shift_size: Shift size for SW-MSA.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value
            qk_scale: Override default qk scale of 1 / sqrt(head_dim) if set
            drop: Dropout rate
            attn_drop: Attention dropout rate
            drop_path: Stochastic depth rate
            act_layer: Activation layer
            norm_layer: Normalization layer
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        if not 0 <= self.shift_size < self.window_size:
            raise RuntimeError(f"shift_size = {self.shift_size} must be in [0, {self.window_size})!")

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # LOGGER.debug("SwinTransformerBlock shift_size: %d", self.shift_size)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage. See Figure 2(b) of the SwinIR paper."""

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        """
        Args:
            dim (int): Number of input channels.
            input_resolution (tuple[int]): Input resolution.
            depth (int): Number of blocks.
            num_heads (int): Number of attention heads.
            window_size (int): Local window size.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
            drop (float, optional): Dropout rate. Default: 0.0
            attn_drop (float, optional): Attention dropout rate. Default: 0.0
            drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
            norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        """

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x, x_size):
        for blk in self.blocks:
            x = blk(x, x_size)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB)."""

    _KERNEL_SIZE = 3
    _STRIDE = 1
    _PADDING = _KERNEL_SIZE // 2
    _PADDING_MODE = "reflect"

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Union[float, Tuple[float, ...]] = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        img_size: int = 8,
        patch_size: int = 4,
    ) -> None:
        """
        Args:
            dim: Number of input channels.
            input_resolution: Input resolution.
            depth: Number of blocks.
            num_heads: Number of attention heads.
            window_size: Local window size.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            qkv_bias: If True, add a learnable bias to query, key, value
            qk_scale: Override default qk scale of head_dim ** -0.5 if set
            drop: Dropout rate. Default: 0.0
            attn_drop: Attention dropout rate. Default: 0.0
            drop_path: Stochastic depth rate
            norm_layer: Normalization layer
            downsampleDownsample layer at the end of the layer
            img_size: Input image size.
            patch_size: Patch size.
        """
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
        )

        self.conv = nn.Conv2d(dim, dim, self._KERNEL_SIZE, self._STRIDE, self._PADDING, padding_mode=self._PADDING_MODE)
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size, embed_dim=dim)

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        """Returns RSTB(x) given x. See Figure 2(a) in the SwinIR paper."""
        # return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x
        out = self.residual_group(x, x_size)
        out = self.patch_unembed(out, x_size)
        out = self.conv(out)
        out = self.patch_embed(out)
        return out + x


class PatchEmbed(nn.Module):
    """Image-to-Patch Embedding"""

    def __init__(self, img_size: int = 8, patch_size: int = 4, embed_dim: int = 96, norm_layer: Optional[nn.Module] = None) -> None:
        """
        Args:
            img_size: Image size
            patch_size: Patch token size
            in_chans: Number of input image channels
            embed_dim: Number of linear projection output channels
            norm_layer: Normalization layer
        """
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x-to-patch embedding: x (B, C, Ph, Pw) -> (B, C, Ph*Pw) -> (B, Ph*Pw, C)"""
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    """Image-to-Patch Unembedding"""

    def __init__(self, img_size: int = 8, patch_size: int = 4, embed_dim: int = 96) -> None:
        """
        Args:
            img_size (int): Image size.  Default: 224.
            patch_size (int): Patch token size. Default: 4.
            in_chans (int): Number of input image channels. Default: 3.
            embed_dim (int): Number of linear projection output channels. Default: 96.
            norm_layer (nn.Module, optional): Normalization layer. Default: None
        """
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        """
        Un-embed patch data x: (B, H*W, C) -> x : (B, C, H, W).
        Args:
            x: input tensor, shape (B, H*W, C)
            x_size: unpacked tensor size (H, W)
        Returns:
            Un-embedded tensor, shape (B, C, H, W).
        """
        bs = x.shape[0]
        x = x.transpose(1, 2).view(bs, self.embed_dim, x_size[0], x_size[1])
        return x


class Upsample(nn.Sequential):
    """Upsample module. Uses a sequence of PixelShuffles with upscale_factor = 2."""

    _KERNEL_SIZE = 3
    _STRIDE = 1
    _PADDING = _KERNEL_SIZE // 2
    _PADDING_MODE = "reflect"
    _UPSCALE_FACTOR = 2

    def __init__(self, scale: int, num_feat: int) -> None:
        """
        Initializes the upsample module.
        Args:
            scale: Scale factor. Supported scales: 2^n, with integer n > 0.
            num_feat: Channel number of intermediate features.
        """
        m = []
        assert (scale & (scale - 1)) == 0, f"scale {scale} is not supported. Supported scales: 2^n, n > 0."
        for _ in range(int(math.log(scale, 2))):
            m.append(
                nn.Conv2d(num_feat, 4 * num_feat, self._KERNEL_SIZE, self._STRIDE, self._PADDING, padding_mode=self._PADDING_MODE)
            )
            m.append(nn.PixelShuffle(self._UPSCALE_FACTOR))
        super().__init__(*m)


class ShallowFeatureExtractor(nn.Module):
    """Shallow feature extractor. This is the first high-level block in the SwinIR."""

    _KERNEL_SIZE = 3
    _PADDING = _KERNEL_SIZE // 2
    _PADDING_TYPE = "reflect"
    _STRIDE = 1
    _STACK_NUM = 3  # 8x SR

    def __init__(self, in_channels_lr: int, n_features: int, num_res_blocks: int = 1, in_channels_hr: int = 0) -> None:
        """
        Initializes the shallow feature extractor.
        Args:
            in_channels_lr: number of LR input variables
            n_features: number of (shallow) hidden features to extract
            num_res_blocks: number of residual blocks (after the input convolutions)
            in_channels_hr: number of HR input variables
        """
        super().__init__()

        nf_lr = n_features // 2 if in_channels_hr > 0 else n_features
        nf_hr = n_features // 2

        self.conv_in_lr = nn.Conv2d(
            in_channels_lr,
            nf_lr,
            self._KERNEL_SIZE,
            self._STRIDE,
            self._PADDING,
            padding_mode=self._PADDING_TYPE,
        )

        if in_channels_hr > 0:
            self.conv_in_hr = ConvBlockStack(
                self._STACK_NUM,  # 3 blocks, i.e. an 8x reduction in pixel counts
                in_channels_hr,
                nf_hr,
                self._KERNEL_SIZE,
                norm=None,
                stride=2,  # coarsen 2x
                activation="leaky_relu",
                spectral_norm=False,
            )

        self.resblocks = nn.Sequential(
            *[
                ResidualBlock(n_features, n_features, self._STRIDE, spectral_norm=False, batch_norm=False)
                for _ in range(num_res_blocks)
            ]
        )

    def forward(self, x_lr: torch.Tensor, x_hr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward operator. Accepts both LR and (optionally) HR inputs. Returns the "shallow" features."""
        x = self.conv_in_lr(x_lr)
        if x_hr is not None:
            x_hr = self.conv_in_hr(x_hr)
            x = torch.cat([x, x_hr], dim=1)
        x = self.resblocks(x)
        return x


class SwinIR(nn.Module):
    """SwinIR: image super-resolution using the Swin Transformer architecture."""

    _UPSCALE_FACTOR = 8
    _NF_OUT = 64
    _KERNEL_SIZE = 3
    _PADDING = _KERNEL_SIZE // 2
    _STRIDE = 1
    _PADDING_MODE = "reflect"

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 64,
        patch_size: Union[int, Tuple[int, int]] = 1,
        input_channels: int = 1,
        output_channels: int = 1,
        embed_dim: int = 128,
        depths: Tuple[int, ...] = (6, 6, 6, 6),
        num_heads: Tuple[int, ...] = (6, 6, 6, 6),
        window_size: int = 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        ape: bool = False,
        patch_norm: bool = True,
        activation_out: Optional[str] = None,
    ) -> None:
        """
        Args:
            img_size: Input image size
            patch_size: Patch size
            input_channels: Number of input image channels
            embed_dim: Patch embedding dimension
            depths: Depth of each Swin Transformer layer
            num_heads: Number of attention heads in different layers
            window_size: Window size
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            qkv_bias: If True, add a learnable bias to query, key, value
            qk_scale: Override default qk scale of head_dim ** -0.5 if set
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            drop_path_rate: Stochastic depth rate
            norm_layer: Normalization layer
            ape: If True, add absolute position embedding to the patch embedding
            patch_norm: If True, add normalization after patch embedding
            activation_out: Output activation, one of ["softplus", "sigmoid", "tanh", "None"]
        """
        super().__init__()

        self.window_size = window_size

        # 1, shallow feature extraction
        # self.conv_first = nn.Conv2d(
        #     input_channels, embed_dim, self._KERNEL_SIZE, self._STRIDE, self._PADDING, padding_mode=self._PADDING_MODE
        # )
        self.input_feature_extractor = ShallowFeatureExtractor(input_channels, embed_dim, num_res_blocks=2)

        # 2, deep feature extraction
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                img_size=img_size,
                patch_size=patch_size,
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        self.conv_after_body = nn.Conv2d(
            embed_dim, embed_dim, self._KERNEL_SIZE, self._STRIDE, self._PADDING, padding_mode=self._PADDING_MODE
        )

        # 3, high quality image reconstruction for classical SR
        self.conv_before_upsample = nn.Sequential(
            *[
                nn.Conv2d(embed_dim, self._NF_OUT, self._KERNEL_SIZE, self._STRIDE, self._PADDING, padding_mode=self._PADDING_MODE),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            ]
        )
        self.upsample = Upsample(self._UPSCALE_FACTOR, self._NF_OUT)
        self.conv_last = nn.Conv2d(
            self._NF_OUT, output_channels, self._KERNEL_SIZE, self._STRIDE, self._PADDING, padding_mode=self._PADDING_MODE
        )
        self.activation_out = activation_out

        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Custom weight initialization."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    def check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies reflection padding to the original (low-res) image.
        The amount of padding is calculated from the transformer window size.
        """
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Deep feature extraction module (see Figure 2 in the paper)"""
        x_size = (x.shape[2], x.shape[3])  # (H_LR, W_LR)

        # patch embeddings (Ph*Pw = num patches; C = embedding dimension)
        x = self.patch_embed(x)  # (B, Ph*Pw, C)

        # add absolute positional embeddings (optional) - shape (1, Ph*Pw, C)
        if self.ape:
            x = x + self.absolute_pos_embed

        # dropout
        x = self.pos_drop(x)

        # apply RSTB (residual swin-transformer block) layers
        for layer in self.layers:
            x = layer(x, x_size)

        # layernorm (optional)
        x = self.norm(x)  # (B, L, C)

        # un-embed patches
        x = self.patch_unembed(x, x_size)

        # (B, C, H_LR, W_LR)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Super-resolution module.
        Args:
            x: low-res input tensor, shape (B, C=1, H, W)
        Output:
            x: hi-res tensor, shape (B, C=1, 8*H, 8*W)
        """
        H, W = x.shape[2:]

        # apply reflection padding to the original image, if needed
        # the amount of padding depends on the window size and on H_LR, W_LR
        x = self.check_image_size(x)  # (B, C, H, W) + padding (when needed)

        # LOGGER.debug("SwinIR check_image_size(x).shape: %s", x.shape)

        # shallow feature extraction module
        # E = embedding dimension
        # x = self.conv_first(x)  # (B, E, H, W)
        x = self.input_feature_extractor(x)

        # deep feature extraction module
        y = self.forward_features(x)  # (B, E, H, W)

        # HQ image reconstruction block (pixelshuffle)
        x = self.conv_after_body(y) + x  # (B,  E,   H,   W)
        x = self.conv_before_upsample(x)  # (B, NF,   H,   W)
        y = self.upsample(x)  # (B, NF, H*8, W*8)
        x = self.conv_last(y)  # (B,  1, H*8, W*8)

        if self.activation_out == "softplus":
            x = F.softplus(x)
        elif self.activation_out == "tanh":
            x = torch.tanh(x)
        elif self.activation_out == "sigmoid":
            x = torch.sigmoid(x)

        # cut off the padding, if any
        return x[:, :, : H * self._UPSCALE_FACTOR, : W * self._UPSCALE_FACTOR]


# if __name__ == "__main__":
#     wsz = 4
#     psz = 4
#     height_, width_ = 8, 8

#     model = SwinIR(
#         img_size=(height_, width_),
#         patch_size=psz,
#         window_size=wsz,
#         depths=(4, 4, 4, 4),
#         embed_dim=256,
#         num_heads=(8, 8, 8, 8),
#         mlp_ratio=2.0,
#     )

#     print(model)

#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Trainable parameters: {trainable_params}")

#     x_: torch.Tensor = torch.randn((32, 1, height_, width_))
#     x_ = model(x_)
#     LOGGER.debug("Output shape: %s", x_.shape)
