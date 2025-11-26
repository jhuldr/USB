# diffusion model u-net adapted from monai.generative package: 
# https://github.com/Project-MONAI/GenerativeModels

import importlib.util
import math
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from typing import Any
from collections.abc import Iterable, Sequence
# TODO
# from models.encoder import *
# To install xformers, use pip install xformers==0.0.16rc401
if importlib.util.find_spec("xformers") is not None:
    import xformers
    import xformers.ops

    has_xformers = True
else:
    xformers = None
    has_xformers = False

from numbers import Number


class GEGLU(nn.Module):
    r"""Applies the element-wise function:

    .. math::
        \text{GEGLU}(x) = x_1 * \text{Sigmoid}(x_2)

    where :math:`x_1` and :math:`x_2` are split from the input tensor along the last dimension.

    Citation: GLU Variants Improve Transformer, Noam Shazeer, 2020, https://arxiv.org/abs/2002.05202.
    Implementation from MONAI package: https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/blocks/activation.py

    Shape:
        - Input: :math:`(N, *, 2 * D)`
        - Output: :math:`(N, *, D)`, where `*` means, any number of additional dimensions
    """

    def forward(self, input: torch.Tensor):
        x, gate = input.chunk(2, dim=-1)
        return x * nn.functional.gelu(gate)


def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def issequenceiterable(obj: Any) -> bool:
    """
    Determine if the object is an iterable sequence and is not a string.
    """
    try:
        if hasattr(obj, "ndim") and obj.ndim == 0:
            return False  # a 0-d tensor is not iterable
    except Exception:
        return False
    return isinstance(obj, Iterable) and not isinstance(obj, (str, bytes))

def ensure_tuple_rep(tup: Any, dim: int) -> tuple[Any, ...]:
    """
    Returns a copy of `tup` with `dim` values by either shortened or duplicated input.

    Raises:
        ValueError: When ``tup`` is a sequence and ``tup`` length is not ``dim``.

    Examples::

        >>> ensure_tuple_rep(1, 3)
        (1, 1, 1)
        >>> ensure_tuple_rep(None, 3)
        (None, None, None)
        >>> ensure_tuple_rep('test', 3)
        ('test', 'test', 'test')
        >>> ensure_tuple_rep([1, 2, 3], 3)
        (1, 2, 3)
        >>> ensure_tuple_rep(range(3), 3)
        (0, 1, 2)
        >>> ensure_tuple_rep([1, 2], 3)
        ValueError: Sequence must have length 3, got length 2.

    """
    if isinstance(tup, torch.Tensor):
        tup = tup.detach().cpu().numpy()
    if isinstance(tup, np.ndarray):
        tup = tup.tolist()
    if not issequenceiterable(tup):
        return (tup,) * dim
    if len(tup) == dim:
        return tuple(tup)

    raise ValueError(f"Sequence must have length {dim}, got {len(tup)}.")

class MLPBlock(nn.Module):
    def __init__(
        self, hidden_size: int, mlp_dim: int, dropout_rate: float = 0.0
    ) -> None:
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        mlp_dim = mlp_dim or hidden_size

        self.linear1 = nn.Linear(hidden_size, mlp_dim * 2)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)
        self.fn = GEGLU()

        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)


    def forward(self, x):
        x = self.fn(self.linear1(x))
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.drop2(x)
        return x
    
class CrossAttention(nn.Module):
    """
    A cross attention layer.

    Args:
        query_dim: number of channels in the query.
        cross_attention_dim: number of channels in the context.
        num_attention_heads: number of heads to use for multi-head attention.
        num_head_channels: number of channels in each head.
        dropout: dropout probability to use.
        upcast_attention: if True, upcast attention operations to full precision.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int | None = None,
        num_attention_heads: int = 8,
        num_head_channels: int = 64,
        dropout: float = 0.0,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.use_flash_attention = use_flash_attention
        inner_dim = num_head_channels * num_attention_heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        self.scale = 1 / math.sqrt(num_head_channels)
        self.num_heads = num_attention_heads

        self.upcast_attention = upcast_attention

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

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
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(dtype=dtype)

        x = torch.bmm(attention_probs, value)
        return x

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        query = self.to_q(x)
        context = context if context is not None else x
        key = self.to_k(context)
        value = self.to_v(context)

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

        return self.to_out(x)


class BasicTransformerBlock(nn.Module):
    """
    A basic Transformer block.

    Args:
        num_channels: number of channels in the input and output.
        num_attention_heads: number of heads to use for multi-head attention.
        num_head_channels: number of channels in each attention head.
        dropout: dropout probability to use.
        cross_attention_dim: size of the context vector for cross attention.
        upcast_attention: if True, upcast attention operations to full precision.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
    """

    def __init__(
        self,
        num_channels: int,
        num_attention_heads: int,
        num_head_channels: int,
        dropout: float = 0.0,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=num_channels,
            num_attention_heads=num_attention_heads,
            num_head_channels=num_head_channels,
            dropout=dropout,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
        )  # is a self-attention
        self.ff = MLPBlock(hidden_size=num_channels, mlp_dim=num_channels * 4, dropout_rate=dropout)
        self.attn2 = CrossAttention(
            query_dim=num_channels,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            num_head_channels=num_head_channels,
            dropout=dropout,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
        )  # is a self-attention if context is None
        self.norm1 = nn.LayerNorm(num_channels)
        self.norm2 = nn.LayerNorm(num_channels)
        self.norm3 = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        # 1. Self-Attention
        x = self.attn1(self.norm1(x)) + x

        # 2. Cross-Attention
        x = self.attn2(self.norm2(x), context=context) + x

        # 3. Feed-forward
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data. First, project the input (aka embedding) and reshape to b, t, d. Then apply
    standard transformer action. Finally, reshape to image.

    Args:
        in_channels: number of channels in the input and output.
        num_attention_heads: number of heads to use for multi-head attention.
        num_head_channels: number of channels in each attention head.
        num_layers: number of layers of Transformer blocks to use.
        dropout: dropout probability to use.
        norm_num_groups: number of groups for the normalization.
        norm_eps: epsilon for the normalization.
        cross_attention_dim: number of context dimensions to use.
        upcast_attention: if True, upcast attention operations to full precision.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
    """

    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int,
        num_head_channels: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        inner_dim = num_attention_heads * num_head_channels

        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=norm_eps, affine=True)

        self.proj_in = nn.Conv3d(
            in_channels,
            inner_dim,
            stride=1,
            kernel_size=1,
            padding=0,
            groups=1,
            bias=True,
            dilation=1,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    num_channels=inner_dim,
                    num_attention_heads=num_attention_heads,
                    num_head_channels=num_head_channels,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    upcast_attention=upcast_attention,
                    use_flash_attention=use_flash_attention,
                )
                for _ in range(num_layers)
            ]
        )

        self.proj_out = zero_module(
            nn.Conv3d(
                in_channels,
                inner_dim,
                stride=1,
                kernel_size=1,
                padding=0,
                groups=1,
                bias=True,
                dilation=1,
            )
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        # note: if no context is given, cross-attention defaults to self-attention
        batch = channel = height = width = depth = -1
 
        batch, channel, height, width, depth = x.shape

        residual = x
        x = self.norm(x)
        x = self.proj_in(x)

        inner_dim = x.shape[1]

        x = x.permute(0, 2, 3, 4, 1).reshape(batch, height * width * depth, inner_dim)

        for block in self.transformer_blocks:
            x = block(x, context=context)

        x = x.reshape(batch, height, width, depth, inner_dim).permute(0, 4, 1, 2, 3).contiguous()

        x = self.proj_out(x)
        return x + residual


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other. Uses three q, k, v linear layers to
    compute attention.

    Args:
        num_channels: number of input channels.
        num_head_channels: number of channels in each attention head.
        norm_num_groups: number of groups involved for the group normalisation layer. Ensure that your number of
            channels is divisible by this number.
        norm_eps: epsilon value to use for the normalisation.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
    """

    def __init__(
        self,
        num_channels: int,
        num_head_channels: int | None = None,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.use_flash_attention = use_flash_attention

        self.num_channels = num_channels

        self.num_heads = num_channels // num_head_channels if num_head_channels is not None else 1
        self.scale = 1 / math.sqrt(num_channels / self.num_heads)

        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=num_channels, eps=norm_eps, affine=True)

        self.to_q = nn.Linear(num_channels, num_channels)
        self.to_k = nn.Linear(num_channels, num_channels)
        self.to_v = nn.Linear(num_channels, num_channels)

        self.proj_attn = nn.Linear(num_channels, num_channels)

    def reshape_heads_to_batch_dim(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, dim // self.num_heads)
        x = x.permute(0, 2, 1, 3).reshape(batch_size * self.num_heads, seq_len, dim // self.num_heads)
        return x

    def reshape_batch_dim_to_heads(self, x: torch.Tensor) -> torch.Tensor:
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

        batch, channel, height, width, depth = x.shape

        # norm
        x = self.norm(x)

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

        x = x.transpose(-1, -2).reshape(batch, channel, height, width, depth)

        return x + residual


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings following the implementation in Ho et al. "Denoising Diffusion Probabilistic
    Models" https://arxiv.org/abs/2006.11239.

    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element.
        embedding_dim: the dimension of the output.
        max_period: controls the minimum frequency of the embeddings.
    """
    if timesteps.ndim != 1:
        raise ValueError("Timesteps should be a 1d-array")

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    freqs = torch.exp(exponent / half_dim)

    args = timesteps[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        embedding = torch.nn.functional.pad(embedding, (0, 1, 0, 0))

    return embedding


class Downsample(nn.Module):
    """
    Downsampling layer.

    Args:
        num_channels: number of input channels.
        use_conv: if True uses conv3d instead of AvgPool3d to perform downsampling. In case that use_conv is
            False, the number of output channels must be the same as the number of input channels.
        out_channels: number of output channels.
        padding: controls the amount of implicit zero-paddings on both sides for padding number of points
            for each dimension.
    """

    def __init__(
        self, num_channels: int, use_conv: bool, out_channels: int | None = None, padding: int = 1
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.out_channels = out_channels or num_channels
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv3d(
            self.num_channels,
            self.out_channels,
            stride=2,
            kernel_size=3,
            padding=padding,
            groups=1,
            bias=True,
            dilation=1,
        )
        else:
            if self.num_channels != self.out_channels:
                raise ValueError("num_channels and out_channels must be equal when use_conv=False")
            self.op = nn.AvgPool3d(2, stride=2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        del emb
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"Input number of channels ({x.shape[1]}) is not equal to expected number of channels "
                f"({self.num_channels})"
            )
        return self.op(x)


class Upsample(nn.Module):
    """
    Upsampling layer with an optional convolution.

    Args:
        num_channels: number of input channels.
        use_conv: if True uses Conv3d.
        out_channels: number of output channels.
        padding: controls the amount of implicit zero-paddings on both sides for padding number of points for each
            dimension.
    """

    def __init__(
        self, num_channels: int, use_conv: bool, out_channels: int | None = None, padding: int = 1
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.out_channels = out_channels or num_channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv3d(
            self.num_channels,
            self.out_channels,
            stride=1,
            kernel_size=3,
            padding=padding,
            groups=1,
            bias=True,
            dilation=1,
        )
        else:
            self.conv = None

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None, cond: torch.Tensor | None = None) -> torch.Tensor:
        del emb
        del cond
        if x.shape[1] != self.num_channels:
            raise ValueError("Input channels should be equal to num_channels")

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = x.dtype
        if dtype == torch.bfloat16:
            x = x.to(torch.float32)

        x = F.interpolate(x, scale_factor=2.0, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            x = x.to(dtype)

        if self.use_conv:
            x = self.conv(x)
        return x


class ResnetBlock(nn.Module):
    """
    Residual block with timestep conditioning.

    Args:
        in_channels: number of input channels.
        temb_channels: number of timestep embedding  channels.
        out_channels: number of output channels.
        up: if True, performs upsampling.
        down: if True, performs downsampling.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
    """

    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        cond_channels: int | None = None,
        out_channels: int | None = None,
        up: bool = False,
        down: bool = False,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.channels = in_channels
        self.cond_channels = cond_channels or temb_channels
        self.emb_channels = temb_channels
        self.out_channels = out_channels or in_channels
        self.up = up
        self.down = down

        self.norm1 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=norm_eps, affine=True)
        self.nonlinearity = nn.SiLU()
        self.conv1 = nn.Conv3d(
        in_channels,
        self.out_channels,
        stride=1,
        kernel_size=3,
        padding=1,
        groups=1,
        bias=True,
        dilation=1,
        )
        self.upsample = self.downsample = None
        if self.up:
            self.upsample = Upsample(in_channels, use_conv=False)
        elif down:
            self.downsample = Downsample(in_channels, use_conv=False)

        self.time_emb_proj = nn.Linear(temb_channels, 2*self.out_channels)
        self.cond_emb_proj = nn.Linear(cond_channels, 2*self.out_channels)

        layers = []

        layers += [nn.GroupNorm(num_groups=norm_num_groups, num_channels=self.out_channels, eps=norm_eps, affine=True), nn.SiLU(), nn.Dropout(p=0), zero_module(
            nn.Conv3d(
                self.out_channels,
                self.out_channels,
                stride=1,
                kernel_size=3,
                padding=1,
                groups=1,
                bias=True,
                dilation=1,
                )
        )
        ]
        self.out_layers = nn.Sequential(*layers)
        if self.out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv3d(
                in_channels,
                self.out_channels,
                stride=1,
                kernel_size=1,
                padding=0,
                groups=1,
                bias=True,
                dilation=1,
            )
            

    def forward(self, x: torch.Tensor, emb: torch.Tensor, cond: torch.Tensor=None) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = self.nonlinearity(h)

        if self.upsample is not None:
            if h.shape[0] >= 64:
                x = x.contiguous()
                h = h.contiguous()
            x = self.upsample(x)
            h = self.upsample(h)
        elif self.downsample is not None:
            x = self.downsample(x)
            h = self.downsample(h)

        h = self.conv1(h)
        # import ipdb; ipdb.set_trace()
        temb = self.time_emb_proj(self.nonlinearity(emb))[:, :, None, None, None]
        if cond is not None:
            cond = self.cond_emb_proj(self.nonlinearity(cond))[:, :, None, None, None]

        #embed cond here? 
        h = apply_conditions(
                h=h,
                emb=temb,
                cond=cond,
                layers=self.out_layers,
                scale_bias=1,
                in_channels=self.out_channels,
                up_down_layer=None,
            )

        return self.skip_connection(x) + h

def apply_conditions(
    h,
    emb=None,
    cond=None,
    layers: nn.Sequential = None,
    scale_bias: float = 1,
    in_channels: int = 512,
    up_down_layer: nn.Module = None,
):
    """
    apply conditions on the feature maps

    Args:
        emb: time conditional (ready to scale + shift)
        cond: encoder's conditional (read to scale + shift)
    """
    two_cond = emb is not None and cond is not None

    if emb is not None:
        # adjusting shapes
        while len(emb.shape) < len(h.shape):
            emb = emb[..., None]

    if two_cond:
        # adjusting shapes
        while len(cond.shape) < len(h.shape):
            cond = cond[..., None]
        # time first
        scale_shifts = [emb, cond]
    else:
        # "cond" is not used with single cond mode
        scale_shifts = [emb]

    # support scale, shift or shift only
    for i, each in enumerate(scale_shifts):
        if each is None:
            # special case: the condition is not provided
            a = None
            b = None
        else:
            if each.shape[1] == in_channels * 2:
                a, b = torch.chunk(each, 2, dim=1)
            else:
                a = each
                b = None
        scale_shifts[i] = (a, b)

    # condition scale bias could be a list
    if isinstance(scale_bias, Number):
        biases = [scale_bias] * len(scale_shifts)
    else:
        # a list
        biases = scale_bias

    # default, the scale & shift are applied after the group norm but BEFORE SiLU
    pre_layers, post_layers = layers[0], layers[1:]

    # spilt the post layer to be able to scale up or down before conv
    # post layers will contain only the conv
    mid_layers, post_layers = post_layers[:-2], post_layers[-2:]

    h = pre_layers(h)
    # scale and shift for each condition
    for i, (scale, shift) in enumerate(scale_shifts):
        # if scale is None, it indicates that the condition is not provided
        if scale is not None:
            h = h * (biases[i] + scale)
            if shift is not None:
                h = h + shift
    h = mid_layers(h)

    # upscale or downscale if any just before the last conv
    if up_down_layer is not None:
        h = up_down_layer(h)
    h = post_layers(h)
    return h


class DownBlock(nn.Module):
    """
    Unet's down block containing resnet and downsamplers blocks.

    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        temb_channels: number of timestep embedding channels.
        num_res_blocks: number of residual blocks.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
        add_downsample: if True add downsample block.
        resblock_updown: if True use residual blocks for downsampling.
        downsample_padding: padding used in the downsampling block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        cond_channels: int | None = None,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_downsample: bool = True,
        resblock_updown: bool = False,
        downsample_padding: int = 1,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown

        resnets = []

        for i in range(num_res_blocks):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    cond_channels=cond_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            if resblock_updown:
                self.downsampler = ResnetBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    cond_channels=cond_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    down=True,
                )
            else:
                self.downsampler = Downsample(
                    num_channels=out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                    padding=downsample_padding,
                )
        else:
            self.downsampler = None

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor, context: torch.Tensor | None = None, cond: torch.Tensor | None = None, 
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        del context
        output_states = []

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb, cond)
            output_states.append(hidden_states)

        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states, temb)
            output_states.append(hidden_states)

        return hidden_states, output_states


class AttnDownBlock(nn.Module):
    """
    Unet's down block containing resnet, downsamplers and self-attention blocks.

    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        temb_channels: number of timestep embedding  channels.
        num_res_blocks: number of residual blocks.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
        add_downsample: if True add downsample block.
        resblock_updown: if True use residual blocks for downsampling.
        downsample_padding: padding used in the downsampling block.
        num_head_channels: number of channels in each attention head.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        cond_channels: int | None = None,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_downsample: bool = True,
        resblock_updown: bool = False,
        downsample_padding: int = 1,
        num_head_channels: int = 1,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown

        resnets = []
        attentions = []

        for i in range(num_res_blocks):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    cond_channels=cond_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )
            attentions.append(
                AttentionBlock(
                    num_channels=out_channels,
                    num_head_channels=num_head_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    use_flash_attention=use_flash_attention,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            if resblock_updown:
                self.downsampler = ResnetBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    cond_channels=cond_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    down=True,
                )
            else:
                self.downsampler = Downsample(
                    num_channels=out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                    padding=downsample_padding,
                )
        else:
            self.downsampler = None

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor, context: torch.Tensor | None = None, cond: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        del context
        output_states = []

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb, cond)
            hidden_states = attn(hidden_states)
            output_states.append(hidden_states)

        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states, temb, cond)
            output_states.append(hidden_states)

        return hidden_states, output_states


class CrossAttnDownBlock(nn.Module):
    """
    Unet's down block containing resnet, downsamplers and cross-attention blocks.

    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        temb_channels: number of timestep embedding channels.
        num_res_blocks: number of residual blocks.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
        add_downsample: if True add downsample block.
        resblock_updown: if True use residual blocks for downsampling.
        downsample_padding: padding used in the downsampling block.
        num_head_channels: number of channels in each attention head.
        transformer_num_layers: number of layers of Transformer blocks to use.
        cross_attention_dim: number of context dimensions to use.
        upcast_attention: if True, upcast attention operations to full precision.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
        dropout_cattn: if different from zero, this will be the dropout value for the cross-attention layers
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_downsample: bool = True,
        resblock_updown: bool = False,
        downsample_padding: int = 1,
        num_head_channels: int = 1,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
        dropout_cattn: float = 0.0,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown

        resnets = []
        attentions = []

        for i in range(num_res_blocks):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )

            attentions.append(
                SpatialTransformer(
                    in_channels=out_channels,
                    num_attention_heads=out_channels // num_head_channels,
                    num_head_channels=num_head_channels,
                    num_layers=transformer_num_layers,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    cross_attention_dim=cross_attention_dim,
                    upcast_attention=upcast_attention,
                    use_flash_attention=use_flash_attention,
                    dropout=dropout_cattn,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            if resblock_updown:
                self.downsampler = ResnetBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    down=True,
                )
            else:
                self.downsampler = Downsample(
                    num_channels=out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                    padding=downsample_padding,
                )
        else:
            self.downsampler = None

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        output_states = []

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, context=context)
            output_states.append(hidden_states)

        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states, temb)
            output_states.append(hidden_states)

        return hidden_states, output_states


class AttnMidBlock(nn.Module):
    """
    Unet's mid block containing resnet and self-attention blocks.

    Args:
        in_channels: number of input channels.
        temb_channels: number of timestep embedding channels.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
        num_head_channels: number of channels in each attention head.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
    """

    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        cond_channels: int | None = None,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        num_head_channels: int = 1,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.attention = None

        self.resnet_1 = ResnetBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            cond_channels=cond_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )
        self.attention = AttentionBlock(
            num_channels=in_channels,
            num_head_channels=num_head_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            use_flash_attention=use_flash_attention,
        )

        self.resnet_2 = ResnetBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            cond_channels=cond_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor, context: torch.Tensor | None = None, cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del context
        hidden_states = self.resnet_1(hidden_states, temb, cond)
        hidden_states = self.attention(hidden_states)
        hidden_states = self.resnet_2(hidden_states, temb, cond)

        return hidden_states


class CrossAttnMidBlock(nn.Module):
    """
    Unet's mid block containing resnet and cross-attention blocks.

    Args:
        in_channels: number of input channels.
        temb_channels: number of timestep embedding channels
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
        num_head_channels: number of channels in each attention head.
        transformer_num_layers: number of layers of Transformer blocks to use.
        cross_attention_dim: number of context dimensions to use.
        upcast_attention: if True, upcast attention operations to full precision.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
    """

    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        num_head_channels: int = 1,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
        dropout_cattn: float = 0.0,
    ) -> None:
        super().__init__()
        self.attention = None

        self.resnet_1 = ResnetBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )
        self.attention = SpatialTransformer(
            in_channels=in_channels,
            num_attention_heads=in_channels // num_head_channels,
            num_head_channels=num_head_channels,
            num_layers=transformer_num_layers,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
            dropout=dropout_cattn,
        )
        self.resnet_2 = ResnetBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        hidden_states = self.resnet_1(hidden_states, temb)
        hidden_states = self.attention(hidden_states, context=context)
        hidden_states = self.resnet_2(hidden_states, temb)

        return hidden_states


class UpBlock(nn.Module):
    """
    Unet's up block containing resnet and upsamplers blocks.

    Args:
        in_channels: number of input channels.
        prev_output_channel: number of channels from residual connection.
        out_channels: number of output channels.
        temb_channels: number of timestep embedding channels.
        num_res_blocks: number of residual blocks.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
        add_upsample: if True add downsample block.
        resblock_updown: if True use residual blocks for upsampling.
    """

    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        cond_channels: int | None = None,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_upsample: bool = True,
        resblock_updown: bool = False,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown
        resnets = []

        for i in range(num_res_blocks):
            res_skip_channels = in_channels if (i == num_res_blocks - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    cond_channels=cond_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            if resblock_updown:
                self.upsampler = ResnetBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    cond_channels=cond_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    up=True,
                )
            else:
                self.upsampler = Upsample(
                    num_channels=out_channels, use_conv=True, out_channels=out_channels
                )
        else:
            self.upsampler = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_list: list[torch.Tensor],
        temb: torch.Tensor,
        context: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del context
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_list[-1]
            res_hidden_states_list = res_hidden_states_list[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb, cond)

        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states, temb, cond)

        return hidden_states


class AttnUpBlock(nn.Module):
    """
    Unet's up block containing resnet, upsamplers, and self-attention blocks.

    Args:
        in_channels: number of input channels.
        prev_output_channel: number of channels from residual connection.
        out_channels: number of output channels.
        temb_channels: number of timestep embedding channels.
        num_res_blocks: number of residual blocks.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
        add_upsample: if True add downsample block.
        resblock_updown: if True use residual blocks for upsampling.
        num_head_channels: number of channels in each attention head.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
    """

    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        cond_channels: int | None = None,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_upsample: bool = True,
        resblock_updown: bool = False,
        num_head_channels: int = 1,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown

        resnets = []
        attentions = []

        for i in range(num_res_blocks):
            res_skip_channels = in_channels if (i == num_res_blocks - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    cond_channels=cond_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )
            attentions.append(
                AttentionBlock(
                    num_channels=out_channels,
                    num_head_channels=num_head_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    use_flash_attention=use_flash_attention,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

        if add_upsample:
            if resblock_updown:
                self.upsampler = ResnetBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    cond_channels=cond_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    up=True,
                )
            else:
                self.upsampler = Upsample(
                    num_channels=out_channels, use_conv=True, out_channels=out_channels
                )
        else:
            self.upsampler = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_list: list[torch.Tensor],
        temb: torch.Tensor,
        context: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del context
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_list[-1]
            res_hidden_states_list = res_hidden_states_list[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb, cond)
            hidden_states = attn(hidden_states)

        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states, temb, cond)

        return hidden_states


class CrossAttnUpBlock(nn.Module):
    """
    Unet's up block containing resnet, upsamplers, and self-attention blocks.

    Args:
        in_channels: number of input channels.
        prev_output_channel: number of channels from residual connection.
        out_channels: number of output channels.
        temb_channels: number of timestep embedding channels.
        num_res_blocks: number of residual blocks.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
        add_upsample: if True add downsample block.
        resblock_updown: if True use residual blocks for upsampling.
        num_head_channels: number of channels in each attention head.
        transformer_num_layers: number of layers of Transformer blocks to use.
        cross_attention_dim: number of context dimensions to use.
        upcast_attention: if True, upcast attention operations to full precision.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
        dropout_cattn: if different from zero, this will be the dropout value for the cross-attention layers
    """

    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_upsample: bool = True,
        resblock_updown: bool = False,
        num_head_channels: int = 1,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
        dropout_cattn: float = 0.0,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown

        resnets = []
        attentions = []

        for i in range(num_res_blocks):
            res_skip_channels = in_channels if (i == num_res_blocks - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )
            attentions.append(
                SpatialTransformer(
                    in_channels=out_channels,
                    num_attention_heads=out_channels // num_head_channels,
                    num_head_channels=num_head_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    num_layers=transformer_num_layers,
                    cross_attention_dim=cross_attention_dim,
                    upcast_attention=upcast_attention,
                    use_flash_attention=use_flash_attention,
                    dropout=dropout_cattn,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            if resblock_updown:
                self.upsampler = ResnetBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    up=True,
                )
            else:
                self.upsampler = Upsample(
                    num_channels=out_channels, use_conv=True, out_channels=out_channels
                )
        else:
            self.upsampler = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_list: list[torch.Tensor],
        temb: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_list[-1]
            res_hidden_states_list = res_hidden_states_list[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, context=context)

        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states, temb)

        return hidden_states


def get_down_block(
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    cond_channels: int,
    num_res_blocks: int,
    norm_num_groups: int,
    norm_eps: float,
    add_downsample: bool,
    resblock_updown: bool,
    with_attn: bool,
    with_cross_attn: bool,
    num_head_channels: int,
    transformer_num_layers: int,
    cross_attention_dim: int | None,
    upcast_attention: bool = False,
    use_flash_attention: bool = False,
    dropout_cattn: float = 0.0,
) -> nn.Module:
    if with_attn:
        return AttnDownBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            cond_channels=cond_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            add_downsample=add_downsample,
            resblock_updown=resblock_updown,
            num_head_channels=num_head_channels,
            use_flash_attention=use_flash_attention,
        )
    elif with_cross_attn:
        return CrossAttnDownBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            add_downsample=add_downsample,
            resblock_updown=resblock_updown,
            num_head_channels=num_head_channels,
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
            dropout_cattn=dropout_cattn,
        )
    else:
        return DownBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            cond_channels=cond_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            add_downsample=add_downsample,
            resblock_updown=resblock_updown,
        )


def get_mid_block(
    in_channels: int,
    temb_channels: int,
    cond_channels: int,
    norm_num_groups: int,
    norm_eps: float,
    with_conditioning: bool,
    num_head_channels: int,
    transformer_num_layers: int,
    cross_attention_dim: int | None,
    upcast_attention: bool = False,
    use_flash_attention: bool = False,
    dropout_cattn: float = 0.0,
) -> nn.Module:
    if with_conditioning:
        return CrossAttnMidBlock(
            in_channels=in_channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            num_head_channels=num_head_channels,
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
            dropout_cattn=dropout_cattn,
        )
    else:
        return AttnMidBlock(
            in_channels=in_channels,
            temb_channels=temb_channels,
            cond_channels=cond_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            num_head_channels=num_head_channels,
            use_flash_attention=use_flash_attention,
        )


def get_up_block(
    in_channels: int,
    prev_output_channel: int,
    out_channels: int,
    temb_channels: int,
    cond_channels: int,
    num_res_blocks: int,
    norm_num_groups: int,
    norm_eps: float,
    add_upsample: bool,
    resblock_updown: bool,
    with_attn: bool,
    with_cross_attn: bool,
    num_head_channels: int,
    transformer_num_layers: int,
    cross_attention_dim: int | None,
    upcast_attention: bool = False,
    use_flash_attention: bool = False,
    dropout_cattn: float = 0.0,
) -> nn.Module:
    if with_attn:
        return AttnUpBlock(
            in_channels=in_channels,
            prev_output_channel=prev_output_channel,
            out_channels=out_channels,
            temb_channels=temb_channels,
            cond_channels=cond_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            add_upsample=add_upsample,
            resblock_updown=resblock_updown,
            num_head_channels=num_head_channels,
            use_flash_attention=use_flash_attention,
        )
    elif with_cross_attn:
        return CrossAttnUpBlock(
            in_channels=in_channels,
            prev_output_channel=prev_output_channel,
            out_channels=out_channels,
            temb_channels=temb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            add_upsample=add_upsample,
            resblock_updown=resblock_updown,
            num_head_channels=num_head_channels,
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
            dropout_cattn=dropout_cattn,
        )
    else:
        return UpBlock(
            in_channels=in_channels,
            prev_output_channel=prev_output_channel,
            out_channels=out_channels,
            temb_channels=temb_channels,
            cond_channels=cond_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            add_upsample=add_upsample,
            resblock_updown=resblock_updown,
        )


class LDM_3D(nn.Module):
    """
    Unet network with timestep embedding and attention mechanisms for conditioning based on
    Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models" https://arxiv.org/abs/2112.10752
    and Pinaya et al. "Brain Imaging Generation with Latent Diffusion Models" https://arxiv.org/abs/2209.07162

    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        num_res_blocks: number of residual blocks (see ResnetBlock) per level.
        num_channels: tuple of block output channels.
        attention_levels: list of levels to add attention.
        norm_num_groups: number of groups for the normalization.
        norm_eps: epsilon for the normalization.
        resblock_updown: if True use residual blocks for up/downsampling.
        num_head_channels: number of channels in each attention head.
        with_conditioning: if True add spatial transformers to perform conditioning.
        transformer_num_layers: number of layers of Transformer blocks to use.
        cross_attention_dim: number of context dimensions to use.
        num_class_embeds: if specified (as an int), then this model will be class-conditional with `num_class_embeds`
        classes.
        upcast_attention: if True, upcast attention operations to full precision.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
        dropout_cattn: if different from zero, this will be the dropout value for the cross-attention layers
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 3, 
        num_res_blocks: Sequence[int] | int = (2, 2, 2, 2),
        num_channels: Sequence[int] = (32, 64, 64, 64),
        attention_levels: Sequence[bool] = (False, False, True, True),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        resblock_updown: bool = False,
        num_head_channels: int | Sequence[int] = 8,
        with_conditioning: bool = False,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        num_class_embeds: int | None = None,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
        dropout_cattn: float = 0.0,
        learn_sigma: bool = False,
    ) -> None:
        
        super().__init__()
        
        out_channels = out_channels * 2 if learn_sigma else out_channels

        if with_conditioning is True and cross_attention_dim is None:
            raise ValueError(
                "CondDiff expects dimension of the cross-attention conditioning (cross_attention_dim) "
                "when using with_conditioning."
            )
        if cross_attention_dim is not None and with_conditioning is False:
            raise ValueError(
                "CondDiff expects with_conditioning=True when specifying the cross_attention_dim."
            )
        if dropout_cattn > 1.0 or dropout_cattn < 0.0:
            raise ValueError("Dropout cannot be negative or >1.0!")

        # All number of channels should be multiple of num_groups
        if any((out_channel % norm_num_groups) != 0 for out_channel in num_channels):
            raise ValueError("CondDiff expects all num_channels being multiple of norm_num_groups")

        if len(num_channels) != len(attention_levels):
            raise ValueError("CondDiff expects num_channels being same size of attention_levels")

        if isinstance(num_head_channels, int):
            num_head_channels = ensure_tuple_rep(num_head_channels, len(attention_levels))

        if len(num_head_channels) != len(attention_levels):
            raise ValueError(
                "num_head_channels should have the same length as attention_levels. For the i levels without attention,"
                " i.e. `attention_level[i]=False`, the num_head_channels[i] will be ignored."
            )

        if isinstance(num_res_blocks, int):
            num_res_blocks = ensure_tuple_rep(num_res_blocks, len(num_channels))

        if len(num_res_blocks) != len(num_channels):
            raise ValueError(
                "`num_res_blocks` should be a single integer or a tuple of integers with the same length as "
                "`num_channels`."
            )

        if use_flash_attention and not has_xformers:
            raise ValueError("use_flash_attention is True but xformers is not installed.")

        if use_flash_attention is True and not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. Flash attention is only available for GPU."
            )

        self.in_channels = in_channels
        self.block_out_channels = num_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels
        self.num_head_channels = num_head_channels
        self.with_conditioning = False

        # input
        self.conv_in = nn.Conv3d(
            in_channels * 2,
            num_channels[0],
            stride=1,
            kernel_size=3,
            padding=1,
            groups=1,
            bias=True,
            dilation=1,
        )

        # time
        time_embed_dim = num_channels[0] * 4
        self.time_embed = nn.Sequential(
            nn.Linear(num_channels[0], time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # class embedding
        self.num_class_embeds = num_class_embeds
        if num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)

        # down
        self.down_blocks = nn.ModuleList([])
        output_channel = num_channels[0]
        for i in range(len(num_channels)):
            input_channel = output_channel
            output_channel = num_channels[i]
            is_final_block = i == len(num_channels) - 1

            down_block = get_down_block(
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                cond_channels=1280,
                num_res_blocks=num_res_blocks[i],
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                add_downsample=not is_final_block,
                resblock_updown=resblock_updown,
                with_attn=(attention_levels[i] and not with_conditioning),
                with_cross_attn=(attention_levels[i] and with_conditioning),
                num_head_channels=num_head_channels[i],
                transformer_num_layers=transformer_num_layers,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
                use_flash_attention=use_flash_attention,
                dropout_cattn=dropout_cattn,
            )

            self.down_blocks.append(down_block)

        # mid
        self.middle_block = get_mid_block(
            in_channels=num_channels[-1],
            temb_channels=time_embed_dim,
            cond_channels=1280,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            with_conditioning=with_conditioning,
            num_head_channels=num_head_channels[-1],
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
            dropout_cattn=dropout_cattn,
        )

        # up
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(num_channels))
        reversed_num_res_blocks = list(reversed(num_res_blocks))
        reversed_attention_levels = list(reversed(attention_levels))
        reversed_num_head_channels = list(reversed(num_head_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(reversed_block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(num_channels) - 1)]

            is_final_block = i == len(num_channels) - 1

            up_block = get_up_block(
                in_channels=input_channel,
                prev_output_channel=prev_output_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                cond_channels=1280,
                num_res_blocks=reversed_num_res_blocks[i] + 1,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                add_upsample=not is_final_block,
                resblock_updown=resblock_updown,
                with_attn=(reversed_attention_levels[i] and not with_conditioning),
                with_cross_attn=(reversed_attention_levels[i] and with_conditioning),
                num_head_channels=reversed_num_head_channels[i],
                transformer_num_layers=transformer_num_layers,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
                use_flash_attention=use_flash_attention,
                dropout_cattn=dropout_cattn,
            )

            self.up_blocks.append(up_block)

        # out
        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=norm_num_groups, num_channels=num_channels[0], eps=norm_eps, affine=True),
            nn.SiLU(),
            zero_module(
                nn.Conv3d(
                    num_channels[0],
                    out_channels,
                    stride=1,
                    kernel_size=3,
                    padding=1,
                    groups=1,
                    bias=True,
                    dilation=1,
                )
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor, 
        t: torch.Tensor,
        y: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        down_block_additional_residuals: tuple[torch.Tensor] | None = None,
        mid_block_additional_residual: torch.Tensor | None = None,
        use_fp16=False,
    ) -> torch.Tensor:
        """
        Args:
            x: input tensor (N, C, SpatialDims).
            t: timestep tensor (N,).
            y: context tensor (N, 1, ContextDim).
            class_labels: context tensor (N, ).
            down_block_additional_residuals: additional residual tensors for down blocks (N, C, FeatureMapsDims).
            mid_block_additional_residual: additional residual tensor for mid block (N, C, FeatureMapsDims).
        """

        # 1. time
        t_emb = get_timestep_embedding(t, self.block_out_channels[0])

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=x.dtype) 
        emb = self.time_embed(t_emb)
        
        # 2. class
        if self.num_class_embeds is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            class_emb = self.class_embedding(class_labels)
            class_emb = class_emb.to(dtype=x.dtype)
            emb = emb + class_emb

        # 3. initial convolution
        h = self.conv_in(torch.cat([x, cond], dim=1))

        # 4. down
        if y is not None and self.with_conditioning is False:
            raise ValueError("model should have with_conditioning = True if context is provided")
        down_block_res_samples: list[torch.Tensor] = [h]
        for downsample_block in self.down_blocks:
            h, res_samples = downsample_block(hidden_states=h, temb=emb, context=y)
            for residual in res_samples:
                down_block_res_samples.append(residual)

        # Additional residual conections for Controlnets
        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()
            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples += (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 5. mid
        h = self.middle_block(hidden_states=h, temb=emb, context=y)

        # Additional residual conections for Controlnets
        if mid_block_additional_residual is not None:
            h = h + mid_block_additional_residual

        # 6. up
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            h = upsample_block(hidden_states=h, res_hidden_states_list=res_samples, temb=emb, context=y)

        # 7. output block
        h = self.out(h)

        return h