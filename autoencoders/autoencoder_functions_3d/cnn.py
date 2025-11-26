#Implementation of an AutoencoderKL model based on the following repositories:
#https://github.com/CompVis/latent-diffusion
#https://github.com/Project-MONAI/GenerativeModels
from __future__ import annotations

import importlib.util
import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra


# To install xformers, use pip install xformers==0.0.16rc401
if importlib.util.find_spec("xformers") is not None:
    import xformers
    import xformers.ops

    has_xformers = True
else:
    xformers = None
    has_xformers = False

class ResBlock(nn.Module):
    """
    Residual block consisting of a cascade of 2 convolutions + activation + normalisation block, and a
    residual connection between input and output.

    Args:
        in_channels: input channels to the layer.
        norm_num_groups: number of groups involved for the group normalisation layer. Ensure that your number of
            channels is divisible by this number.
        norm_eps: epsilon for the normalisation.
        out_channels: number of output channels.
    """

    def __init__(
        self, in_channels: int, norm_num_groups: int, norm_eps: float, out_channels: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        self.norm1 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=norm_eps, affine=True)

        self.conv1 = nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            stride=1,
            kernel_size=3,
            padding=1,
        )
        self.norm2 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=out_channels, eps=norm_eps, affine=True)
        self.conv2 = nn.Conv3d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            stride=1,
            kernel_size=3,
            padding=1,
        )

        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv3d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                stride=1,
                kernel_size=1,
                padding=0,
            )
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        if torch.isnan(h).any():
            print("h_out is nan after conv1")
            #find indices of nan values
            exit()
        h = self.norm2(h)
        if torch.isnan(h).any():
            print("h is nan after norm2")
            exit()
        h = F.silu(h)
        if torch.isnan(h).any():
            print("h is nan after silu")
            exit()
        h = self.conv2(h)
        if torch.isnan(h).any():
            print("h is nan after conv2")
            print(h)

        #check if h is nan
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h

class Downsample(nn.Module):
    """
    Convolution-based downsampling layer.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        in_channels: number of input channels.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.pad = (0, 1) * 3 #3 is number of spatial dimensions
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            stride=2,
            kernel_size=3,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, self.pad, mode="constant", value=0.0)
        x = self.conv(x)
        return x

class Upsample(nn.Module):
    """
    Convolution-based upsampling layer.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        in_channels: number of input channels to the layer.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            stride=1,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = x.dtype
        if dtype == torch.bfloat16:
            x = x.to(torch.float32)

        x = F.interpolate(x, scale_factor=2.0, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            x = x.to(dtype)

        x = self.conv(x)
        return x

class VariationalEncoder(nn.Module):
    """
    Convolutional cascade that downsamples the image into a spatial latent space.

    Args:
        in_channels: number of input channels.
        num_channels: sequence of block output channels.
        out_channels: number of channels in the bottom layer (latent space) of the autoencoder.
        num_res_blocks: number of residual blocks (see ResBlock) per level.
        norm_num_groups: number of groups for the GroupNorm layers, num_channels must be divisible by this number.
        norm_eps: epsilon for the normalization.
        attention_levels: indicate which level from num_channels contain an attention block.
        with_nonlocal_attn: if True use non-local attention block.
    """

    def __init__(
        self,
        in_channels: int,
        num_channels,
        out_channels,
        num_res_blocks,
        norm_num_groups: int,
        norm_eps: float,
        attention_levels: Sequence[bool],
        with_nonlocal_attn: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.norm_eps = norm_eps
        self.attention_levels = attention_levels
        blocks = []
        # Initial convolution
        blocks.append(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=num_channels[0],
                stride=1,
                kernel_size=3,
                padding=1,
            )
        )
        # Residual and downsampling blocks
        output_channel = num_channels[0]
        for i in range(len(num_channels)):
            input_channel = output_channel
            output_channel = num_channels[i]
            is_final_block = i == len(num_channels) - 1

            for _ in range(self.num_res_blocks[i]):
                blocks.append(
                    ResBlock(
                        in_channels=input_channel,
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        out_channels=output_channel,
                    )
                )
                input_channel = output_channel
                if attention_levels[i]:
                    blocks.append(
                        AttentionBlock(
                            spatial_dims=3, #TODO change to not fixed
                            num_channels=input_channel,
                            norm_num_groups=norm_num_groups,
                            norm_eps=norm_eps,
                            use_flash_attention= False, #TODO change to not fixed
                        )
                    )
            if not is_final_block:
                blocks.append(Downsample(in_channels=input_channel))

        # Non-local attention block
        if with_nonlocal_attn is True:
            blocks.append(
                ResBlock(
                    in_channels=num_channels[-1],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=num_channels[-1],
                )
            )

            blocks.append(
                AttentionBlock(
                    spatial_dims=3,
                    num_channels=num_channels[-1],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    use_flash_attention=False,
                )
            )
            blocks.append(
                ResBlock(
                    in_channels=num_channels[-1],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=num_channels[-1],
                )
            )
        # Normalise and convert to latent size
        blocks.append(
            nn.GroupNorm(num_groups=norm_num_groups, num_channels=num_channels[-1], eps=norm_eps, affine=True)
        )
        blocks.append(
            nn.Conv3d(
                in_channels=num_channels[-1],
                out_channels=out_channels,
                stride=1,
                kernel_size=3,
                padding=1,
            )
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:  
            x = block(x)
        return x

class Decoder(nn.Module):
    """
    Convolutional cascade upsampling from a spatial latent space into an image space.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        num_channels: sequence of block output channels.
        in_channels: number of channels in the bottom layer (latent space) of the autoencoder.
        out_channels: number of output channels.
        num_res_blocks: number of residual blocks (see ResBlock) per level.
        norm_num_groups: number of groups for the GroupNorm layers, num_channels must be divisible by this number.
        norm_eps: epsilon for the normalization.
        attention_levels: indicate which level from num_channels contain an attention block.
        with_nonlocal_attn: if True use non-local attention block.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
    """

    def __init__(
        self,
        num_channels: Sequence[int],
        in_channels: int,
        out_channels: int,
        num_res_blocks: Sequence[int],
        norm_num_groups: int,
        norm_eps: float,
        attention_levels: Sequence[bool],
        dec_dist,
        with_nonlocal_attn: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.norm_num_groups = norm_num_groups
        self.norm_eps = norm_eps
        self.dec_dist = dec_dist
        reversed_block_out_channels = list(reversed(num_channels))

        blocks = []
        # Initial convolution
        blocks.append(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=reversed_block_out_channels[0],
                stride=1,
                kernel_size=3,
                padding=1,
            )
        )

        # Non-local attention block
        if with_nonlocal_attn is True:
            blocks.append(
                ResBlock(
                    in_channels=reversed_block_out_channels[0],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=reversed_block_out_channels[0],
                )
            )
            blocks.append(
                AttentionBlock(
                    spatial_dims=3,
                    num_channels=reversed_block_out_channels[0],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    use_flash_attention=False,
                )
            )
            blocks.append(
                ResBlock(
                    in_channels=reversed_block_out_channels[0],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=reversed_block_out_channels[0],
                )
            )

        reversed_num_res_blocks = list(reversed(num_res_blocks))
        reversed_attention_levels = list(reversed(attention_levels))
        block_out_ch = reversed_block_out_channels[0]
        for i in range(len(reversed_block_out_channels)):
            block_in_ch = block_out_ch
            block_out_ch = reversed_block_out_channels[i]
            is_final_block = i == len(num_channels) - 1

            for _ in range(reversed_num_res_blocks[i]):
                blocks.append(
                    ResBlock(
                        in_channels=block_in_ch,
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        out_channels=block_out_ch,
                    )
                )
                block_in_ch = block_out_ch

                if reversed_attention_levels[i]:
                    blocks.append(
                        AttentionBlock(
                            spatial_dims=3, #TODO change to not fixed
                            num_channels=block_in_ch,
                            norm_num_groups=norm_num_groups,
                            norm_eps=norm_eps,
                            use_flash_attention=False, #TODO change to not fixed
                        )
                    )
            if not is_final_block:
                blocks.append(Upsample(in_channels=block_in_ch))

        blocks.append(nn.GroupNorm(num_groups=norm_num_groups, num_channels=block_in_ch, eps=norm_eps, affine=True))
        blocks.append(
            nn.Conv3d(
                in_channels=block_in_ch,
                out_channels=out_channels,
                stride=1,
                kernel_size=3,
                padding=1,
            )
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = hydra.utils.instantiate(self.dec_dist, x=x)
        return x

class AttentionBlock(nn.Module):
    """
    Attention block.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        num_channels: number of input channels.
        num_head_channels: number of channels in each attention head.
        norm_num_groups: number of groups involved for the group normalisation layer. Ensure that your number of
            channels is divisible by this number.
        norm_eps: epsilon value to use for the normalisation.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
    """

    def __init__(
        self,
        spatial_dims: int,
        num_channels: int,
        num_head_channels: int | None = None,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.use_flash_attention = use_flash_attention
        self.spatial_dims = spatial_dims
        self.num_channels = num_channels

        self.num_heads = num_channels // num_head_channels if num_head_channels is not None else 1
        self.scale = 1 / math.sqrt(num_channels / self.num_heads)

        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=num_channels, eps=norm_eps, affine=True)

        self.to_q = nn.Linear(num_channels, num_channels)
        self.to_k = nn.Linear(num_channels, num_channels)
        self.to_v = nn.Linear(num_channels, num_channels)

        self.proj_attn = nn.Linear(num_channels, num_channels)

    def reshape_heads_to_batch_dim(self, x: torch.Tensor) -> torch.Tensor:
        """
        Divide hidden state dimension to the multiple attention heads and reshape their input as instances in the batch.
        """
        batch_size, seq_len, dim = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, dim // self.num_heads)
        x = x.permute(0, 2, 1, 3).reshape(batch_size * self.num_heads, seq_len, dim // self.num_heads)
        return x

    def reshape_batch_dim_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Combine the output of the attention heads back into the hidden state dimension."""
        batch_size, seq_len, dim = x.shape
        x = x.reshape(batch_size // self.num_heads, self.num_heads, seq_len, dim)
        x = x.permute(0, 2, 1, 3).reshape(batch_size // self.num_heads, seq_len, dim * self.num_heads)
        return x

    def _memory_efficient_attention_xformers(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        x = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=None)
        return x

    def _attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        attention_probs = attention_scores.softmax(dim=-1)
        x = torch.bmm(attention_probs, value)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        batch = channel = height = width = depth = -1
        if self.spatial_dims == 2:
            batch, channel, height, width = x.shape
        if self.spatial_dims == 3:
            batch, channel, height, width, depth = x.shape

        # norm
        x = self.norm(x)

        if self.spatial_dims == 2:
            x = x.view(batch, channel, height * width).transpose(1, 2)
        if self.spatial_dims == 3:
            x = x.view(batch, channel, height * width * depth).transpose(1, 2)

        # proj to q, k, v
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        # Multi-Head Attention
        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if self.use_flash_attention:
            x = self._memory_efficient_attention_xformers(query, key, value)
        else:
            x = self._attention(query, key, value)

        x = self.reshape_batch_dim_to_heads(x)
        x = x.to(query.dtype)

        if self.spatial_dims == 2:
            x = x.transpose(-1, -2).reshape(batch, channel, height, width)
        if self.spatial_dims == 3:
            x = x.transpose(-1, -2).reshape(batch, channel, height, width, depth)

        return x + residual