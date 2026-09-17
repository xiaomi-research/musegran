"""Transformer building blocks: attention, RoPE, MusicRoPE, conformer, and continuous transformer."""

import logging
from functools import reduce
from typing import Callable, Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from packaging import version
from torch import nn, einsum

logger = logging.getLogger(__name__)

try:
    from flash_attn import flash_attn_func
except ImportError:
    logger.info("flash_attn not installed, disabling Flash Attention")
    flash_attn_func = None

try:
    import natten
except ImportError:
    natten = None

def checkpoint(function, *args, **kwargs):
    """Gradient checkpointing wrapper with non-reentrant default."""
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)


# Copyright (c) 2020 Phil Wang
# SPDX-License-Identifier: MIT
# Copied and modified from https://github.com/lucidrains/x-transformers/blob/main/x_transformers/attend.py

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

# positional embeddings

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.scale = dim ** -0.5
        self.max_seq_len = max_seq_len
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device
        assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}'

        if pos is None:
            pos = torch.arange(seq_len, device = device)

        if seq_start_pos is not None:
            pos = (pos - seq_start_pos[..., None]).clamp(min = 0)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return pos_emb

class ScaledSinusoidalEmbedding(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        assert (dim % 2) == 0, 'dimension must be divisible by 2'
        self.scale = nn.Parameter(torch.ones(1) * dim ** -0.5)

        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta ** -freq_seq
        self.register_buffer('inv_freq', inv_freq, persistent = False)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device

        if pos is None:
            pos = torch.arange(seq_len, device = device)

        if seq_start_pos is not None:
            pos = pos - seq_start_pos[..., None]

        emb = einsum('i, j -> i j', pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim = -1)
        return emb * self.scale

class RotaryEmbedding(nn.Module):
    """Standard rotary position embedding with optional xPos length extrapolation."""

    def __init__(
        self,
        dim,
        use_xpos = False,
        scale_base = 512,
        interpolation_factor = 1.,
        base = 10000,
        base_rescale_factor = 1.
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        assert interpolation_factor >= 1.
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer('scale', None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.register_buffer('scale', scale)

    def forward_from_seq_len(self, seq_len: int):
        """Compute RoPE frequencies for a contiguous sequence [0, seq_len)."""
        device = self.inv_freq.device

        t = torch.arange(seq_len, device = device)
        return self.forward(t)

    @torch.amp.autocast('cuda', enabled=False)
    def forward_from_batch_seq(self, batch_t: torch.Tensor):
        """Compute RoPE frequencies from per-sample position sequences [B, T]."""
        device = self.inv_freq.device

        batch_t = batch_t.to(torch.float32)

        batch_t = batch_t / self.interpolation_factor

        freqs = torch.einsum('b i , j -> b i j', batch_t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)

        if self.scale is None:
            return freqs, 1.

        seq_len = batch_t.shape[-1]
        power = (torch.arange(seq_len, device = device) - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1).unsqueeze(0)

        return freqs, scale

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, t):
        device = self.inv_freq.device

        t = t.to(torch.float32)
        t = t / self.interpolation_factor
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)

        if self.scale is None:
            return freqs, 1.

        seq_len = t.shape[-1]
        power = (torch.arange(seq_len, device = device) - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        return freqs, scale

def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

@torch.amp.autocast('cuda', enabled=False)
def apply_rotary_pos_emb(t, freqs, scale = 1):
    out_dtype = t.dtype

    # cast to float32 if necessary for numerical stability
    dtype = reduce(torch.promote_types, (t.dtype, freqs.dtype, torch.float32))
    rot_dim, seq_len = freqs.shape[-1], t.shape[-2]
    freqs, t = freqs.to(dtype), t.to(dtype)
    if freqs.ndim == 2:
        freqs = freqs[-seq_len:, :]
    elif freqs.ndim == 3:
        freqs = freqs[:, -seq_len:, :]
        if freqs.shape[0] < t.shape[0]: # for cfg infer
            freqs = torch.cat([freqs]*(t.shape[0]//freqs.shape[0]), dim=0)

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, 'b n d -> b 1 n d')

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)

    t, t_unrotated = t.to(out_dtype), t_unrotated.to(out_dtype)

    return torch.cat((t, t_unrotated), dim = -1)


class MusicRotaryEmbedding(nn.Module):
    """Hierarchical RoPE with separate frequency bases for bar, beat, and sequence positions."""

    def __init__(
        self,
        dim,
        group_order=["seq", "bar", "beat"],
        dim_split=[4, 2, 2],
        seq_base=10000,
        bar_base=5000,
        beat_base=2000,
        use_period_pos=False,
        period_bars=4,
    ):
        super().__init__()
        assert len(group_order) == len(dim_split), "group_order and dim_split must have the same length"

        self.group_order = group_order
        self.num_groups = len(group_order)
        self.period_bars = period_bars

        base_map = {"seq": seq_base, "bar": bar_base, "beat": beat_base}

        dim_scale = dim // sum(dim_split)
        self.group_dims = {
            name: split * dim_scale
            for name, split in zip(group_order, dim_split)
        }

        self.rope_modules = nn.ModuleDict({
            name: RotaryEmbedding(self.group_dims[name], base=base_map.get(name, 10000))
            for name in group_order
        })

        full_inv_freq = 1. / (seq_base ** (torch.arange(0, dim, 2).float() / dim))
        offset = 0
        for name in group_order:
            group_half_dim = self.group_dims[name] // 2
            self.register_buffer(f'fallback_inv_freq_{name}', full_inv_freq[offset:offset + group_half_dim])
            offset += group_half_dim

        self.period_pos_emb = nn.Embedding(period_bars + 1, dim * 2, padding_idx=0) if use_period_pos else None

    def _fallback_freqs(self, name, seq_len):
        """Standard RoPE fallback when no music positions are provided."""
        inv_freq = getattr(self, f'fallback_inv_freq_{name}')
        t = torch.arange(seq_len, device=inv_freq.device, dtype=torch.float32)
        freqs = torch.einsum('i, j -> i j', t, inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs, 1.

    def forward(self, seq_len, bar_pos=None, beat_pos=None, prepend_length=0):
        if prepend_length > 0:
            bar_pos = F.pad(bar_pos, (prepend_length, 0)) if bar_pos is not None else None
            beat_pos = F.pad(beat_pos, (prepend_length, 0)) if beat_pos is not None else None

        pos_map = {"seq": None, "bar": bar_pos, "beat": beat_pos}
        has_music_pos = bar_pos is not None or beat_pos is not None

        freqs_list = []
        dims_list = []
        scales_list = []

        for name in self.group_order:
            rope = self.rope_modules[name]
            pos = pos_map.get(name)

            if has_music_pos:
                if name == "seq" or pos is None:
                    freqs, scale = rope.forward_from_seq_len(seq_len)
                else:
                    freqs, scale = rope.forward_from_batch_seq(pos)
            else:
                freqs, scale = self._fallback_freqs(name, seq_len)

            freqs_list.append(freqs)
            dims_list.append(self.group_dims[name] * 2)
            scales_list.append(scale)

        period_pos_emb = None
        if self.period_pos_emb is not None and bar_pos is not None:
            period_pos = bar_pos % self.period_bars
            period_pos[(bar_pos % self.period_bars == 0) & (bar_pos != 0)] = self.period_bars
            period_pos_emb = self.period_pos_emb(period_pos.long())

        return [freqs_list, dims_list, period_pos_emb], scales_list

def apply_music_rotary_pos_emb(t, freqs_and_dim, scale=1):
    freqs_list, dims_list, period_pos_emb = freqs_and_dim

    ts = torch.split(t, dims_list, dim=-1)
    assert len(ts) == len(freqs_list), f"split count mismatch: {len(ts)} vs {len(freqs_list)}"

    t_rot = torch.cat([
        apply_rotary_pos_emb(ts[i], freqs_list[i], scale)
        for i in range(len(freqs_list))
    ], dim=-1)

    if period_pos_emb is not None:
        # period_pos_emb: [B, seq_len, dim*2]
        if period_pos_emb.ndim == 3:
            period_pos_emb = period_pos_emb.unsqueeze(1)  # [B, 1, seq_len, dim*2]
        if period_pos_emb.shape[0] != t_rot.shape[0]:  # for cfg infer
            period_pos_emb = period_pos_emb.repeat(t_rot.shape[0] // period_pos_emb.shape[0], 1, 1, 1)
        t_rot = t_rot + period_pos_emb

    return t_rot



# norms
class LayerNorm(nn.Module):
    """Bias-less LayerNorm with optional fixed scale."""

    def __init__(self, dim, bias=False, fix_scale=False):
        super().__init__()

        if fix_scale:
            self.register_buffer("gamma", torch.ones(dim))
        else:
            self.gamma = nn.Parameter(torch.ones(dim))

        if bias:
            self.beta = nn.Parameter(torch.zeros(dim))
        else:
            self.register_buffer("beta", torch.zeros(dim))


    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], weight=self.gamma, bias=self.beta)

# feedforward

class GLU(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        activation: Callable,
        use_conv = False,
        conv_kernel_size = 3,
    ):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2) if not use_conv else nn.Conv1d(dim_in, dim_out * 2, conv_kernel_size, padding = (conv_kernel_size // 2))
        self.use_conv = use_conv

    def forward(self, x):
        if self.use_conv:
            x = rearrange(x, 'b n d -> b d n')
            x = self.proj(x)
            x = rearrange(x, 'b d n -> b n d')
        else:
            x = self.proj(x)

        x, gate = x.chunk(2, dim = -1)
        return x * self.act(gate)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        mult = 4,
        no_bias = False,
        glu = True,
        use_conv = False,
        conv_kernel_size = 3,
        zero_init_output = True,
    ):
        super().__init__()
        inner_dim = int(dim * mult)

        # Default to SwiGLU

        activation = nn.SiLU()

        dim_out = dim if dim_out is None else dim_out

        if glu:
            linear_in = GLU(dim, inner_dim, activation)
        else:
            linear_in = nn.Sequential(
                Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
                nn.Linear(dim, inner_dim, bias = not no_bias) if not use_conv else nn.Conv1d(dim, inner_dim, conv_kernel_size, padding = (conv_kernel_size // 2), bias = not no_bias),
                Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
                activation
            )

        linear_out = nn.Linear(inner_dim, dim_out, bias = not no_bias) if not use_conv else nn.Conv1d(inner_dim, dim_out, conv_kernel_size, padding = (conv_kernel_size // 2), bias = not no_bias)

        # init last linear layer to 0
        if zero_init_output:
            nn.init.zeros_(linear_out.weight)
            if not no_bias:
                nn.init.zeros_(linear_out.bias)


        self.ff = nn.Sequential(
            linear_in,
            Rearrange('b d n -> b n d') if use_conv else nn.Identity(),
            linear_out,
            Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
        )

    def forward(self, x):
        return self.ff(x)

class Attention(nn.Module):
    """Multi-head attention supporting Flash Attention, SDPA, and neighborhood attention (NATTEN)."""

    def __init__(
        self,
        dim,
        dim_heads = 64,
        dim_context = None,
        causal = False,
        zero_init_output=True,
        qk_norm: Literal['l2', 'ln', 'none'] = 'none',
        natten_kernel_size = None,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.dim_heads = dim_heads
        self.causal = causal

        dim_kv = dim_context if dim_context is not None else dim

        self.num_heads = dim // dim_heads
        self.kv_heads = dim_kv // dim_heads

        if dim_context is not None:
            self.to_q = nn.Linear(dim, dim, bias=False)
            self.to_kv = nn.Linear(dim_kv, dim_kv * 2, bias=False)
        else:
            self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.to_out = nn.Linear(dim, dim, bias=False)

        if zero_init_output:
            nn.init.zeros_(self.to_out.weight)

        self.qk_norm = qk_norm

        if self.qk_norm == "ln":
            self.q_norm = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=1.0e-6)
            self.k_norm = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=1.0e-6)

        # Using 1d neighborhood attention
        self.natten_kernel_size = natten_kernel_size
        if natten_kernel_size is not None:
            return

        self.use_pt_flash = torch.cuda.is_available() and version.parse(torch.__version__) >= version.parse('2.0.0')
        self.use_fa_flash = torch.cuda.is_available() and flash_attn_func is not None

        self.sdp_kwargs = dict(
            enable_flash = True,
            enable_math = True,
            enable_mem_efficient = True
        )

    def create_causal_mask(self, q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.ones((q_len, k_len), device=device, dtype=torch.bool)
        mask = torch.tril(mask)
        return mask.unsqueeze(0).unsqueeze(0)

    def flash_attn(
            self,
            q,
            k,
            v,
            mask = None,
            causal = None
    ):
        batch, heads, q_len, _, k_len, device = *q.shape, k.shape[-2], q.device
        kv_heads = k.shape[1]

        if heads != kv_heads:
            heads_per_kv_head = heads // kv_heads
            k, v = map(lambda t: t.repeat_interleave(heads_per_kv_head, dim = 1), (k, v))

        if k.ndim == 3:
            k = rearrange(k, 'b ... -> b 1 ...').expand_as(q)

        if v.ndim == 3:
            v = rearrange(v, 'b ... -> b 1 ...').expand_as(q)

        causal = self.causal if causal is None else causal

        if q_len == 1 and causal:
            causal = False

        final_mask = mask

        if causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device=device)

            if final_mask is None:
                final_mask = causal_mask
            else:
                final_mask = final_mask & causal_mask
            causal = False

        # Handle fully-masked rows to prevent NaN from softmax
        row_is_entirely_masked = None
        if final_mask is not None:
            if final_mask.ndim == 3:
                final_mask = final_mask.unsqueeze(1).expand(batch, heads, q_len, k_len)
            elif final_mask.ndim == 2:
                final_mask = final_mask.unsqueeze(0).unsqueeze(0).expand(batch, heads, q_len, k_len)

            row_is_entirely_masked = ~final_mask.any(dim=-1)

            if row_is_entirely_masked.any():
                final_mask = final_mask.clone()
                final_mask[..., 0] = final_mask[..., 0] | row_is_entirely_masked

        with torch.backends.cuda.sdp_kernel(**self.sdp_kwargs):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = final_mask,
                is_causal = causal
            )

        if row_is_entirely_masked is not None and row_is_entirely_masked.any():
            out = out.masked_fill(row_is_entirely_masked[..., None], 0.)

        return out

    def calc_qkv_attn(self, input_mask, q, k, v, h, kv_h, causal):
        final_attn_mask = None

        if input_mask is not None:
            input_mask = rearrange(input_mask, 'b j -> b 1 1 j')
            final_attn_mask = input_mask.bool()

        n, device = q.shape[-2], q.device
        causal = self.causal if causal is None else causal

        if n == 1 and causal:
            causal = False

        if self.natten_kernel_size is not None:
            if natten is None:
                raise ImportError('natten not installed, please install natten to use neighborhood attention')

            dtype_in = q.dtype
            q, k, v = map(lambda t: t.to(torch.float32), (q, k, v))

            attn = natten.functional.natten1dqk(q, k, kernel_size = self.natten_kernel_size, dilation=1)

            if final_attn_mask is not None:
                attn = attn.masked_fill(~final_attn_mask, -torch.finfo(attn.dtype).max)

            attn = F.softmax(attn, dim=-1, dtype=torch.float32)
            out = natten.functional.natten1dav(attn, v, kernel_size = self.natten_kernel_size, dilation=1).to(dtype_in)

        elif self.use_fa_flash:
            if final_attn_mask is None:
                # Flash Attention 2 requires FP16 inputs
                fa_dtype_in = q.dtype
                q, k, v = map(lambda t: rearrange(t, 'b h n d -> b n h d').to(torch.float16), (q, k, v))

                out = flash_attn_func(q, k, v, causal = causal)

                out = rearrange(out.to(fa_dtype_in), 'b n h d -> b h n d')
            else:
                out = self.flash_attn(q, k, v, causal = causal, mask = final_attn_mask)

        elif self.use_pt_flash:
            out = self.flash_attn(q, k, v, causal = causal, mask = final_attn_mask)

        else:
            if h != kv_h:
                heads_per_kv_head = h // kv_h
                k, v = map(lambda t: t.repeat_interleave(heads_per_kv_head, dim = 1), (k, v))

            scale = 1. / (q.shape[-1] ** 0.5)
            kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'

            dots = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k) * scale
            mask_value = -torch.finfo(dots.dtype).max

            if final_attn_mask is not None:
                dots = dots.masked_fill(~final_attn_mask, mask_value)

            if causal:
                causal_mask = self.create_causal_mask(dots.shape[-2], dots.shape[-1], device=device)
                dots = dots.masked_fill(~causal_mask, mask_value)

            attn = F.softmax(dots, dim=-1, dtype=torch.float32)
            attn = attn.type(dots.dtype)

            out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)

        out = rearrange(out, ' b h n d -> b n (h d)')
        return out

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None,
        rotary_pos_emb = None,
        causal = None,
        **kwargs,
    ):
        h, kv_h, has_context = self.num_heads, self.kv_heads, context is not None

        kv_input = context if has_context else x

        if hasattr(self, 'to_q'):
            q = self.to_q(x)
            q = rearrange(q, 'b n (h d) -> b h n d', h = h)
            k, v = self.to_kv(kv_input).chunk(2, dim=-1)
            k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = kv_h), (k, v))
        else:
            q, k, v = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        if self.qk_norm == "l2":
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        elif self.qk_norm == "ln":
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rotary_pos_emb is not None and not has_context:
            freqs, _ = rotary_pos_emb
            q_dtype = q.dtype
            k_dtype = k.dtype

            q = q.to(torch.float32)
            k = k.to(torch.float32)

            if isinstance(freqs, torch.Tensor):
                freqs = freqs.to(torch.float32)
                q = apply_rotary_pos_emb(q, freqs)
                k = apply_rotary_pos_emb(k, freqs)
            else:
                q = apply_music_rotary_pos_emb(q, freqs)
                k = apply_music_rotary_pos_emb(k, freqs)

            q = q.to(q_dtype)
            k = k.to(k_dtype)

        input_mask = context_mask if has_context else mask

        out = self.calc_qkv_attn(input_mask, q, k, v, h, kv_h, causal)
        out = self.to_out(out)

        if mask is not None:
            mask = rearrange(mask, 'b n -> b n 1')
            out = out.masked_fill(~mask, 0.)

        return out

class MultiAttention(nn.Module):
    """Multi-head cross-attention with multiple context sources."""

    def __init__(
        self,
        dim,
        dim_heads = 64,
        dim_context = None,
        dim_cross = None,
        causal = False,
        zero_init_output=True,
        qk_norm: Literal['l2', 'ln', 'none'] = 'none',
        natten_kernel_size = None
    ):
        super().__init__()
        self.dim = dim
        self.dim_heads = dim_heads
        self.causal = causal

        dim_kv = dim_cross if dim_cross is not None else dim

        self.num_heads = dim // dim_heads
        self.kv_heads = dim_kv // dim_heads

        if dim_cross is not None:
            self.to_q = nn.Linear(dim, dim, bias=False)
            self.to_kv = nn.Linear(dim_kv, dim_kv * 2, bias=False)
        else:
            self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
            self.to_qkv_context = nn.Linear(dim_context, dim_context * 3, bias=False) if dim_context is not None else None

        self.to_out = nn.Linear(dim, dim, bias=False)
        self.to_out_context = nn.Linear(dim_context, dim_context, bias=False) if dim_context is not None else None

        if zero_init_output:
            nn.init.zeros_(self.to_out.weight)

        self.qk_norm = qk_norm

        if self.qk_norm == "ln":
            self.q_norm = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=1.0e-6)
            self.k_norm = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=1.0e-6)

            self.q_norm_context = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=1.0e-6) if dim_context is not None else None
            self.k_norm_context = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=1.0e-6) if dim_context is not None else None

        # Using 1d neighborhood attention
        self.natten_kernel_size = natten_kernel_size
        if natten_kernel_size is not None:
            return

        self.use_pt_flash = torch.cuda.is_available() and version.parse(torch.__version__) >= version.parse('2.0.0')

        self.use_fa_flash = torch.cuda.is_available() and flash_attn_func is not None

        self.sdp_kwargs = dict(
            enable_flash = True,
            enable_math = True,
            enable_mem_efficient = True
        )

    def flash_attn(
            self,
            q,
            k,
            v,
            mask = None,
            causal = None
    ):
        batch, heads, q_len, _, k_len, device = *q.shape, k.shape[-2], q.device
        kv_heads = k.shape[1]
        if heads != kv_heads:
            # Repeat interleave kv_heads to match q_heads
            heads_per_kv_head = heads // kv_heads
            k, v = map(lambda t: t.repeat_interleave(heads_per_kv_head, dim = 1), (k, v))

        if k.ndim == 3:
            k = rearrange(k, 'b ... -> b 1 ...').expand_as(q)

        if v.ndim == 3:
            v = rearrange(v, 'b ... -> b 1 ...').expand_as(q)

        causal = self.causal if causal is None else causal

        if q_len == 1 and causal:
            causal = False

        if mask is not None:
            assert mask.ndim == 4
            mask = mask.expand(batch, heads, q_len, k_len)

        if k_len > q_len and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device = device)
            if mask is None:
                mask = ~causal_mask
            else:
                mask = mask & ~causal_mask
            causal = False

        # manually handle causal mask, if another mask was given

        row_is_entirely_masked = None

        if mask is not None and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device = device)
            mask = mask & ~causal_mask

            # protect against an entire row being masked out

            row_is_entirely_masked = ~mask.any(dim = -1)
            mask[..., 0] = mask[..., 0] | row_is_entirely_masked

            causal = False

        with torch.backends.cuda.sdp_kernel(**self.sdp_kwargs):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                is_causal = causal
            )

        # for a row that is entirely masked out, should zero out the output of that row token

        if row_is_entirely_masked is not None:
            out = out.masked_fill(row_is_entirely_masked[..., None], 0.)

        return out

    def forward(
        self,
        x,
        context = None,
        cross_context = None,
        mask = None,
        context_mask = None,
        rotary_pos_emb = None,
        causal = None
    ):
        h, kv_h, has_context, has_cross = self.num_heads, self.kv_heads, context is not None, cross_context is not None

        kv_input = cross_context if has_cross else x

        if hasattr(self, 'to_q'):
            # Use separate linear projections for q and k/v
            q = self.to_q(x)
            q = rearrange(q, 'b n (h d) -> b h n d', h = h)

            k, v = self.to_kv(kv_input).chunk(2, dim=-1)

            k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = kv_h), (k, v))
        else:
            # Use fused linear projection
            q, k, v = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # Normalize q and k for cosine sim attention
        if self.qk_norm == "l2":
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        elif self.qk_norm == "ln":
            q = self.q_norm(q)
            k = self.k_norm(k)

        if has_context:
            # `context` projections.
            # Use fused linear projection
            c_q, c_k, c_v = self.to_qkv_context(context).chunk(3, dim=-1)
            c_q, c_k, c_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (c_q, c_k, c_v))

            # Normalize q and k for cosine sim attention
            if self.qk_norm == "l2":
                c_q = F.normalize(c_q, dim=-1)
                c_k = F.normalize(c_k, dim=-1)
            elif self.qk_norm == "ln":
                c_q = self.q_norm_context(c_q)
                c_k = self.k_norm_context(c_k)

            # attention
            q = torch.cat([c_q, q], dim=2)
            k = torch.cat([c_k, k], dim=2)
            v = torch.cat([c_v, v], dim=2)

        if rotary_pos_emb is not None and not has_cross:
            freqs, _ = rotary_pos_emb

            q_dtype = q.dtype
            k_dtype = k.dtype

            q = q.to(torch.float32)
            k = k.to(torch.float32)
            if isinstance(freqs, torch.Tensor):
                freqs = freqs.to(torch.float32)
                q = apply_rotary_pos_emb(q, freqs)
                k = apply_rotary_pos_emb(k, freqs)
            else:
                q = apply_music_rotary_pos_emb(q, freqs)
                k = apply_music_rotary_pos_emb(k, freqs)

            q = q.to(q_dtype)
            k = k.to(k_dtype)

        input_mask = context_mask

        if input_mask is None and not has_context:
            input_mask = mask

        final_attn_mask = None
        if input_mask is not None:
            final_attn_mask = rearrange(input_mask, 'b j -> b 1 1 j')

        n, device = q.shape[-2], q.device

        causal = self.causal if causal is None else causal

        if n == 1 and causal:
            causal = False

        if self.natten_kernel_size is not None:
            if natten is None:
                raise ImportError('natten not installed, please install natten to use neighborhood attention')

            dtype_in = q.dtype
            q, k, v = map(lambda t: t.to(torch.float32), (q, k, v))

            attn = natten.functional.natten1dqk(q, k, kernel_size = self.natten_kernel_size, dilation=1)

            if final_attn_mask is not None:
                attn = attn.masked_fill(final_attn_mask, -torch.finfo(attn.dtype).max)

            attn = F.softmax(attn, dim=-1, dtype=torch.float32)

            out = natten.functional.natten1dav(attn, v, kernel_size = self.natten_kernel_size, dilation=1).to(dtype_in)

        # Prioritize Flash Attention 2
        elif self.use_fa_flash:
            assert final_attn_mask is None, 'masking not yet supported for Flash Attention 2'
            # Flash Attention 2 requires FP16 inputs
            fa_dtype_in = q.dtype
            q, k, v = map(lambda t: rearrange(t, 'b h n d -> b n h d').to(torch.float16), (q, k, v))

            out = flash_attn_func(q, k, v, causal = causal)

            out = rearrange(out.to(fa_dtype_in), 'b n h d -> b h n d')

        # Fall back to PyTorch implementation
        elif self.use_pt_flash:
            out = self.flash_attn(q, k, v, causal = causal, mask = final_attn_mask)

        else:
            # Fall back to custom implementation

            if h != kv_h:
                # Repeat interleave kv_heads to match q_heads
                heads_per_kv_head = h // kv_h
                k, v = map(lambda t: t.repeat_interleave(heads_per_kv_head, dim = 1), (k, v))

            scale = 1. / (q.shape[-1] ** 0.5)

            kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'

            dots = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k) * scale

            i, j, dtype = *dots.shape[-2:], dots.dtype

            mask_value = -torch.finfo(dots.dtype).max

            if final_attn_mask is not None:
                dots = dots.masked_fill(~final_attn_mask, mask_value)

            if causal:
                causal_mask = self.create_causal_mask(i, j, device = device)
                dots = dots.masked_fill(causal_mask, mask_value)

            attn = F.softmax(dots, dim=-1, dtype=torch.float32)
            attn = attn.type(dtype)

            out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)

        # merge heads
        x = rearrange(out, ' b h n d -> b n (h d)')

        if has_context:
            context, x = (
                x[:, : context.shape[1]],
                x[:, context.shape[1] :],
            )

        x = self.to_out(x)

        if has_context:
            context = self.to_out_context(context)

        if mask is not None:
            mask = rearrange(mask, 'b n -> b n 1')
            x = x.masked_fill(~mask, 0.)

        if has_context:
            if context_mask is not None:
                context_mask = rearrange(context_mask, 'b n -> b n 1')
                context = context.masked_fill(~context_mask, 0.)

            return x, context

        else:
            return x

class ConformerModule(nn.Module):
    """Conformer-style convolution block: pointwise → GLU → depthwise → pointwise."""

    def __init__(
        self,
        dim,
        norm_kwargs = {},
    ):

        super().__init__()

        self.dim = dim

        self.in_norm = LayerNorm(dim, **norm_kwargs)
        self.pointwise_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.glu = GLU(dim, dim, nn.SiLU())
        self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size=17, groups=dim, padding=8, bias=False)
        self.mid_norm = LayerNorm(dim, **norm_kwargs)
        self.swish = nn.SiLU()
        self.pointwise_conv_2 = nn.Conv1d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.in_norm(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.pointwise_conv(x)
        x = rearrange(x, 'b d n -> b n d')
        x = self.glu(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.depthwise_conv(x)
        x = rearrange(x, 'b d n -> b n d')
        x = self.mid_norm(x)
        x = self.swish(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.pointwise_conv_2(x)
        x = rearrange(x, 'b d n -> b n d')

        return x

class TransformerBlock(nn.Module):
    """Single transformer layer: self-attn → (cross-attn) → FFN, with optional AdaLN and conformer."""

    def __init__(
            self,
            dim,
            dim_heads = 64,
            cross_attend = False,
            dim_context = None,
            global_cond_dim = None,
            causal = False,
            zero_init_branch_outputs = True,
            conformer = False,
            layer_ix = -1,
            remove_norms = False,
            attn_kwargs = {},
            ff_kwargs = {},
            norm_kwargs = {}
    ):

        super().__init__()
        self.dim = dim
        self.dim_heads = dim_heads
        self.cross_attend = cross_attend
        self.dim_context = dim_context
        self.causal = causal

        self.pre_norm = LayerNorm(dim, **norm_kwargs) if not remove_norms else nn.Identity()

        self.self_attn = Attention(
            dim,
            dim_heads = dim_heads,
            causal = causal,
            zero_init_output=zero_init_branch_outputs,
            **attn_kwargs
        )

        if cross_attend:
            self.cross_attend_norm = LayerNorm(dim, **norm_kwargs) if not remove_norms else nn.Identity()
            self.cross_attn = Attention(
                dim,
                dim_heads = dim_heads,
                dim_context = dim_context,
                causal = causal,
                zero_init_output=zero_init_branch_outputs,
                **attn_kwargs
            )

        self.ff_norm = LayerNorm(dim, **norm_kwargs) if not remove_norms else nn.Identity()
        self.ff = FeedForward(dim, zero_init_output=zero_init_branch_outputs, **ff_kwargs)

        self.layer_ix = layer_ix

        self.conformer = ConformerModule(dim, norm_kwargs=norm_kwargs) if conformer else None

        self.global_cond_dim = global_cond_dim
        self.adaln_para_num = 9 if cross_attend else 6

        if global_cond_dim is not None:
            self.to_scale_shift_gate = nn.Sequential(
                nn.LayerNorm(global_cond_dim),
                nn.SiLU(),
                nn.Linear(global_cond_dim, dim * self.adaln_para_num, bias=False)
            )

            nn.init.zeros_(self.to_scale_shift_gate[-1].weight)

    def forward(
        self,
        x,
        context = None,
        global_cond=None,
        mask = None,
        context_mask = None,
        rotary_pos_emb = None,
        **kwargs
    ):
        if self.global_cond_dim is not None and self.global_cond_dim > 0 and global_cond is not None:

            scale_shift_gates = self.to_scale_shift_gate(global_cond).unsqueeze(1).chunk(self.adaln_para_num, dim = -1)
            scale_self, shift_self, gate_self = scale_shift_gates[:3]
            scale_ff, shift_ff, gate_ff = scale_shift_gates[-3:]

            residual = x
            x = self.pre_norm(x)
            x = x * (1 + scale_self) + shift_self
            x = self.self_attn(x, mask = mask, rotary_pos_emb = rotary_pos_emb)
            x = x * torch.sigmoid(gate_self)
            x = x + residual

            if self.cross_attend and context is not None:
                scale_crs, shift_crs, gate_crs = scale_shift_gates[3:6]
                residual_crs = x
                x = self.cross_attend_norm(x)
                x = x * (1 + scale_crs) + shift_crs
                x = self.cross_attn(x, context=context, context_mask=context_mask)
                x = x * torch.sigmoid(gate_crs)
                x = residual_crs + x

            if self.conformer is not None:
                x = x + self.conformer(x)

            residual = x
            x = self.ff_norm(x)
            x = x * (1 + scale_ff) + shift_ff
            x = self.ff(x)
            x = x * torch.sigmoid(gate_ff)
            x = x + residual
        else:
            x = x + self.self_attn(self.pre_norm(x), mask = mask, rotary_pos_emb = rotary_pos_emb)

            if self.cross_attend and context is not None:
                x = x + self.cross_attn(self.cross_attend_norm(x), context=context, context_mask=context_mask)

            if self.conformer is not None:
                x = x + self.conformer(x)

            x = x + self.ff(self.ff_norm(x))

        return x

class ContinuousTransformer(nn.Module):
    """Stack of TransformerBlocks with positional encoding, projection layers, and optional SegAlign hooks."""

    def __init__(
        self,
        dim,
        depth,
        *,
        dim_in = None,
        dim_out = None,
        dim_heads = 64,
        cross_attend=False,
        cond_token_dim=None,
        global_cond_dim=None,
        causal=False,
        rotary_pos_emb=True,
        zero_init_branch_outputs=True,
        conformer=False,
        use_sinusoidal_emb=False,
        use_abs_pos_emb=False,
        abs_pos_emb_max_length=10000,
        **kwargs
        ):

        super().__init__()

        self.dim = dim
        self.depth = depth
        self.causal = causal
        self.layers = nn.ModuleList([])

        use_patch_emb = kwargs.pop('use_patch_emb', False)
        if use_patch_emb:
            patch_size = kwargs.pop('proj_patch_size', 2)
            self.project_in = nn.Sequential(
                Lambda(lambda x: x.transpose(1, 2)),  # [B, T, C] -> [B, C, T]
                nn.Conv1d(
                    in_channels=dim_in,
                    out_channels=dim,
                    kernel_size=patch_size,
                    stride=patch_size,
                    padding=0,
                ),
                Lambda(lambda x: x.transpose(1, 2)),  # [B, C, T//patch_size] -> [B, T//patch_size, C]
            )
            self.project_out = nn.Sequential(
                Lambda(lambda x: x.transpose(1, 2)),  # [B, T//patch_size, inner_dim] -> [B, inner_dim, T//patch_size]
                nn.ConvTranspose1d(
                    in_channels=dim,
                    out_channels=dim_out,
                    kernel_size=patch_size,
                    stride=patch_size,
                    padding=0,
                ),
                Lambda(lambda x: x.transpose(1, 2)),  # [B, out_channels, T] -> [B, T, out_channels]
            )
        else:
            self.project_in = nn.Linear(dim_in, dim, bias=False) if dim_in is not None else nn.Identity()
            self.project_out = nn.Linear(dim, dim_out, bias=False) if dim_out is not None else nn.Identity()

        rotary_pos = kwargs.pop('rotary_pos', 'rope')
        self.rotary_pos = rotary_pos

        bar_base = kwargs.pop('bar_base', 5000)
        beat_base = kwargs.pop('beat_base', 2000)
        dim_split = kwargs.pop('dim_split', [4,2,2])
        group_order = kwargs.pop('group_order', ["seq", "bar", "beat"])
        use_period_pos = kwargs.pop('use_period_pos', False)
        period_bars = kwargs.pop('period_bars', 4)
        if rotary_pos_emb:
            if rotary_pos == 'music_rope': # music rope
                self.rotary_pos_emb = MusicRotaryEmbedding(dim=max(dim_heads // 2, 32), group_order=group_order, bar_base=bar_base,
                                                           beat_base=beat_base, dim_split=dim_split, use_period_pos=use_period_pos, period_bars=period_bars)
            else:
                self.rotary_pos_emb = RotaryEmbedding(max(dim_heads // 2, 32))
        else:
            self.rotary_pos_emb = None

        self.use_sinusoidal_emb = use_sinusoidal_emb
        if use_sinusoidal_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(dim)

        self.use_abs_pos_emb = use_abs_pos_emb
        if use_abs_pos_emb:
            self.pos_emb = AbsolutePositionalEmbedding(dim, abs_pos_emb_max_length)

        for i in range(depth):
            self.layers.append(
                TransformerBlock(
                    dim,
                    dim_heads = dim_heads,
                    cross_attend = cross_attend,
                    dim_context = cond_token_dim,
                    global_cond_dim = global_cond_dim,
                    causal = causal,
                    zero_init_branch_outputs = zero_init_branch_outputs,
                    conformer=conformer,
                    layer_ix=i,
                    **kwargs
                )
            )

    def forward(
        self,
        x,
        mask = None,
        prepend_embeds = None,
        prepend_mask = None,
        global_cond = None,
        return_info = False,
        **kwargs
    ):
        batch, seq, device = *x.shape[:2], x.device

        info = {
            "hidden_states": [],
        }

        x = self.project_in(x)

        custom_emb = kwargs.pop('custom_emb', {})
        kwargs.pop('negative_custom_emb', None)
        if custom_emb is None:
            custom_emb = {}
        pos_seqs = {key: custom_emb[key] for key in custom_emb.keys() if key in ['bar_pos', 'beat_pos']}
        structure_pos = custom_emb.get('structure', None)

        if structure_pos is not None:
            x = x + structure_pos

        if prepend_embeds is not None:
            prepend_length, prepend_dim = prepend_embeds.shape[1:]

            assert prepend_dim == x.shape[-1], 'prepend dimension must match sequence dimension'

            x = torch.cat((prepend_embeds, x), dim = -2)

            if prepend_mask is not None or mask is not None:
                mask = mask if mask is not None else torch.ones((batch, seq), device = device, dtype = torch.bool)
                prepend_mask = prepend_mask if prepend_mask is not None else torch.ones((batch, prepend_length), device = device, dtype = torch.bool)

                mask = torch.cat((prepend_mask, mask), dim = -1)

        # Attention layers
        prepend_length = kwargs.pop('prepend_length', 0)
        if self.rotary_pos_emb is not None:
            if len(pos_seqs.keys())>0:
                if self.rotary_pos == 'music_rope':
                    bar_pos = pos_seqs.get('bar_pos')
                    beat_pos = pos_seqs.get('beat_pos')
                    rotary_pos_emb = self.rotary_pos_emb(x.shape[1], bar_pos, beat_pos, prepend_length)
                    cfg_ref_pos = bar_pos if bar_pos is not None else beat_pos
                    if cfg_ref_pos is not None and cfg_ref_pos.shape[0] < x.shape[0]:
                        repeat_factor = x.shape[0] // cfg_ref_pos.shape[0]
                        expanded_bar_pos = bar_pos.repeat(repeat_factor, 1) if bar_pos is not None else None
                        expanded_beat_pos = beat_pos.repeat(repeat_factor, 1) if beat_pos is not None else None
                        rotary_pos_emb = self.rotary_pos_emb(x.shape[1], expanded_bar_pos, expanded_beat_pos, prepend_length)
            else:
                rotary_pos_emb = self.rotary_pos_emb.forward_from_seq_len(x.shape[1])
        else:
            rotary_pos_emb = None

        if self.use_sinusoidal_emb or self.use_abs_pos_emb:
            x = x + self.pos_emb(x)

        segalign_depths = kwargs.pop('segalign_depths',None)
        inner_hidden_states = []

        for idx, layer in enumerate(self.layers):
            x = checkpoint(layer, x, rotary_pos_emb = rotary_pos_emb, global_cond=global_cond, **kwargs)

            if return_info:
                info["hidden_states"].append(x)

            if segalign_depths is not None:
                for depth in segalign_depths:
                    if idx == depth:
                        inner_hidden_states.append(x)

        x = self.project_out(x)

        if return_info and len(inner_hidden_states) > 0:
            return (x, inner_hidden_states), info

        if return_info:
            return x, info

        if len(inner_hidden_states) > 0:
            return x, inner_hidden_states

        return x
