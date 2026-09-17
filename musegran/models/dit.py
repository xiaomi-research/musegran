"""Diffusion Transformer (DiT) with multi-granularity conditioning and classifier-free guidance."""

import math
import typing as tp
import warnings

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from x_transformers import ContinuousTransformerWrapper, Encoder

from .transformer import ContinuousTransformer


class FourierFeatures(nn.Module):
    """Random Fourier feature embedding for continuous timesteps."""

    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        if out_features % 2 != 0:
            raise ValueError("out_features must be even")
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


def _pad_to_match(a: torch.Tensor, b: torch.Tensor, dim: int = 1):
    """Pad a and b along `dim` so they share the same length."""
    len_a, len_b = a.size(dim), b.size(dim)
    if len_a == len_b:
        return a, b
    max_len = max(len_a, len_b)
    ndim = a.dim()
    # F.pad uses reversed dim order: last dim first
    def _make_pad(tensor, target_len):
        cur_len = tensor.size(dim)
        if cur_len == target_len:
            return tensor
        pad = [0] * (2 * (ndim - dim - 1)) + [0, target_len - cur_len]
        return F.pad(tensor, pad)
    return _make_pad(a, max_len), _make_pad(b, max_len)


def _null_mask_like(mask: torch.Tensor) -> torch.Tensor:
    """Create a null mask (only first token True) with same shape."""
    null = torch.zeros_like(mask)
    null[:, 0] = True
    return null


class DiffusionTransformer(nn.Module):
    """DiT backbone: handles conditioning injection, SegAlign projections, and CFG at inference."""

    def __init__(
        self,
        io_channels=32,
        patch_size=1,
        embed_dim=768,
        cond_token_dim=0,
        project_cond_tokens=True,
        global_cond_dim=0,
        project_global_cond=True,
        input_concat_dim=0,
        prepend_cond_dim=0,
        depth=12,
        num_heads=8,
        transformer_type: tp.Literal["x-transformers", "continuous_transformer"] = "x-transformers",
        global_cond_type: tp.List[tp.Literal["prepend", "adaLN"]] = ["prepend"],
        use_segalign=False,
        segalign_dims=None,
        **kwargs,
    ):
        super().__init__()

        self.cond_token_dim = cond_token_dim
        self.transformer_type = transformer_type

        # Timestep embedding
        timestep_features_dim = 256
        self.timestep_features = FourierFeatures(1, timestep_features_dim)
        self.to_timestep_embed = nn.Sequential(
            nn.Linear(timestep_features_dim, embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=True),
        )

        if cond_token_dim > 0:
            cond_embed_dim = cond_token_dim if not project_cond_tokens else embed_dim
            self.to_cond_embed = nn.Sequential(
                nn.Linear(cond_token_dim, cond_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(cond_embed_dim, cond_embed_dim, bias=False),
            )
        else:
            cond_embed_dim = 0

        if isinstance(global_cond_dim, int) and global_cond_dim > 0:
            self.to_global_embed = nn.Sequential(
                nn.Linear(global_cond_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False),
            )
        elif isinstance(global_cond_dim, list):
            self.to_global_embed = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d, embed_dim, bias=False),
                    nn.SiLU(),
                    nn.Linear(embed_dim, embed_dim, bias=False),
                ) if d > 0 else nn.Identity()
                for d in global_cond_dim
            ])

        if prepend_cond_dim > 0:
            self.to_prepend_embed = nn.Sequential(
                nn.Linear(prepend_cond_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False),
            )

        self.input_concat_dim = input_concat_dim
        dim_in = io_channels + self.input_concat_dim

        if use_segalign:
            self.segalign_dims = segalign_dims
            projector_dim = 2 * embed_dim
            self.segalign_projectors = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embed_dim, projector_dim),
                    nn.SiLU(),
                    nn.Linear(projector_dim, projector_dim),
                    nn.SiLU(),
                    nn.Linear(projector_dim, dim),
                )
                for dim in self.segalign_dims
            ])

        self.patch_size = patch_size
        self.use_attn_mask = bool(kwargs.pop("use_attn_mask", False))
        self.global_cond_type = global_cond_type

        if self.transformer_type == "x-transformers":
            self.transformer = ContinuousTransformerWrapper(
                dim_in=dim_in * patch_size,
                dim_out=io_channels * patch_size,
                max_seq_len=0,
                attn_layers=Encoder(
                    dim=embed_dim,
                    depth=depth,
                    heads=num_heads,
                    attn_flash=True,
                    cross_attend=cond_token_dim > 0,
                    dim_context=None if cond_embed_dim == 0 else cond_embed_dim,
                    zero_init_branch_output=True,
                    use_abs_pos_emb=False,
                    rotary_pos_emb=True,
                    ff_swish=True,
                    ff_glu=True,
                    **kwargs,
                ),
            )
        elif self.transformer_type == "continuous_transformer":
            global_dim = embed_dim if "adaLN" in self.global_cond_type else None
            self.transformer = ContinuousTransformer(
                dim=embed_dim,
                depth=depth,
                dim_heads=embed_dim // num_heads,
                dim_in=dim_in * patch_size,
                dim_out=io_channels * patch_size,
                cross_attend=cond_token_dim > 0,
                cond_token_dim=cond_embed_dim,
                global_cond_dim=global_dim,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown transformer type: {self.transformer_type}")

        self.preprocess_conv = nn.Conv1d(dim_in, dim_in, 1, bias=False)
        nn.init.zeros_(self.preprocess_conv.weight)
        self.postprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)
        nn.init.zeros_(self.postprocess_conv.weight)

    def _forward(
        self,
        x,
        t,
        mask=None,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        input_concat_cond=None,
        global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        return_info=False,
        exit_layer_ix=None,
        segalign_depths=None,
        **kwargs,
    ):
        """Single-pass forward without CFG batching. Handles conditioning projection and transformer call."""
        if cross_attn_cond is not None:
            cross_attn_cond = self.to_cond_embed(cross_attn_cond)

        if global_embed is not None:
            if isinstance(global_embed, dict):
                global_embed = {
                    k: self.to_global_embed[idx](v)
                    for idx, (k, v) in enumerate(global_embed.items())
                }
            else:
                assert len(self.global_cond_type) == 1
                global_embed = {self.global_cond_type[0]: self.to_global_embed(global_embed)}

        prepend_inputs = None
        prepend_mask = None
        prepend_length = 0

        if prepend_cond is not None:
            prepend_inputs = self.to_prepend_embed(prepend_cond)
            if prepend_cond_mask is not None:
                prepend_mask = prepend_cond_mask

        if input_concat_cond is not None:
            if input_concat_cond.shape[2] != x.shape[2]:
                input_concat_cond = F.interpolate(input_concat_cond, (x.shape[2],), mode='nearest')
            x = torch.cat([x, input_concat_cond], dim=1)

        # Add timestep embedding to global conditioning
        timestep_embed = self.to_timestep_embed(self.timestep_features(t[:, None]))
        if global_embed is not None:
            for k in global_embed:
                global_embed[k] = global_embed[k] + timestep_embed
        else:
            global_embed = timestep_embed

        # Prepend global embed as a token if using prepend mode
        if "prepend" in self.global_cond_type:
            prepend_global = global_embed["prepend"].unsqueeze(1)
            if prepend_inputs is None:
                prepend_inputs = prepend_global
                prepend_mask = torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)
            else:
                prepend_inputs = torch.cat([prepend_inputs, prepend_global], dim=1)
                ones = torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)
                prepend_mask = torch.cat([prepend_mask, ones], dim=1)
            prepend_length = prepend_inputs.shape[1]

        x = self.preprocess_conv(x) + x
        x = rearrange(x, "b c t -> b t c")

        extra_args = {}
        if "adaLN" in self.global_cond_type:
            extra_args["global_cond"] = global_embed["adaLN"]
        if self.patch_size > 1:
            x = rearrange(x, "b (t p) c -> b t (c p)", p=self.patch_size)
        if segalign_depths is not None:
            extra_args["segalign_depths"] = segalign_depths
        if prepend_length > 1:
            extra_args["prepend_length"] = prepend_length - 1

        if self.transformer_type == "x-transformers":
            output = self.transformer(
                x, prepend_embeds=prepend_inputs, context=cross_attn_cond,
                context_mask=cross_attn_cond_mask, mask=mask, prepend_mask=prepend_mask,
                **extra_args, **kwargs,
            )
        elif self.transformer_type == "continuous_transformer":
            output = self.transformer(
                x, prepend_embeds=prepend_inputs, context=cross_attn_cond,
                context_mask=cross_attn_cond_mask, return_info=return_info,
                exit_layer_ix=exit_layer_ix, **extra_args, **kwargs,
            )
            if return_info:
                output, info = output
            if isinstance(output, tuple):
                output, inner_hidden_states = output
            else:
                inner_hidden_states = []
            if exit_layer_ix is not None:
                return (output, info) if return_info else output

        output = rearrange(output, "b t c -> b c t")[:, :, prepend_length:]

        if self.patch_size > 1:
            output = rearrange(output, "b (c p) t -> b c (t p)", p=self.patch_size)

        output = self.postprocess_conv(output) + output

        if segalign_depths is not None:
            projections = [proj(h) for proj, h in zip(self.segalign_projectors, inner_hidden_states)]
            return output, projections

        if return_info:
            return output, info

        return output

    # ------------------------------------------------------------------
    # CFG helpers
    # ------------------------------------------------------------------

    def _prepare_cross_attn_batch(self, cond, mask, neg_cond, neg_mask, n_way):
        """Batch cross-attention cond/mask for CFG (2-way or 3-way)."""
        null_embed = torch.zeros_like(cond)

        if neg_cond is not None:
            cond, neg_cond = _pad_to_match(cond, neg_cond)
            null_embed = torch.zeros_like(cond)
        else:
            neg_cond = cond if n_way == 3 else null_embed

        parts = [cond, neg_cond, null_embed] if n_way == 3 else [cond, neg_cond]
        batch_cond = torch.cat(parts, dim=0)

        batch_mask = None
        if mask is not None:
            null_mask = _null_mask_like(mask)
            if neg_mask is not None:
                mask, neg_mask = _pad_to_match(mask, neg_mask)
                null_mask = _null_mask_like(mask)
            else:
                neg_mask = mask
                null_mask = _null_mask_like(mask)
            mask_parts = [mask, neg_mask, null_mask] if n_way == 3 else [mask, neg_mask]
            batch_mask = torch.cat(mask_parts, dim=0)

        return batch_cond, batch_mask

    @staticmethod
    def _apply_cfg_rescale(cond_output, cfg_output, scale_phi):
        if scale_phi == 0.0:
            return cfg_output
        cond_std = cond_output.std(dim=1, keepdim=True)
        cfg_std = cfg_output.std(dim=1, keepdim=True)
        return scale_phi * (cfg_output * (cond_std / cfg_std)) + (1 - scale_phi) * cfg_output

    # ------------------------------------------------------------------
    # Public forward with CFG logic
    # ------------------------------------------------------------------

    def forward(
        self,
        x,
        t,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        negative_cross_attn_cond=None,
        negative_cross_attn_mask=None,
        input_concat_cond=None,
        global_embed=None,
        negative_input_concat_cond=None,
        negative_global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        cfg_scale=1.0,
        cfg_dropout_prob=0.0,
        scale_phi=0.0,
        mask=None,
        return_info=False,
        segalign_depths=None,
        **kwargs,
    ):
        """Forward with CFG: supports 2-way (pos/uncond) and 3-way (pos/neg/uncond) guidance."""
        # Convert masks to bool
        if cross_attn_cond_mask is not None:
            cross_attn_cond_mask = cross_attn_cond_mask.bool() if self.use_attn_mask else None
        if negative_cross_attn_mask is not None:
            negative_cross_attn_mask = negative_cross_attn_mask.bool() if self.use_attn_mask else None
        if prepend_cond_mask is not None:
            prepend_cond_mask = prepend_cond_mask.bool()

        # CFG dropout (training)
        if cfg_dropout_prob > 0.0:
            if cross_attn_cond is not None:
                null_embed = torch.zeros_like(cross_attn_cond)
                drop_mask = torch.bernoulli(torch.full((cross_attn_cond.shape[0], 1, 1), cfg_dropout_prob, device=cross_attn_cond.device)).bool()
                cross_attn_cond = torch.where(drop_mask, null_embed, cross_attn_cond)
            if prepend_cond is not None:
                null_embed = torch.zeros_like(prepend_cond)
                drop_mask = torch.bernoulli(torch.full((prepend_cond.shape[0], 1, 1), cfg_dropout_prob, device=prepend_cond.device)).bool()
                prepend_cond = torch.where(drop_mask, null_embed, prepend_cond)

        # No CFG — direct forward
        is_multi_scale = isinstance(cfg_scale, list)
        if not is_multi_scale and cfg_scale == 1.0:
            return self._forward(
                x, t,
                cross_attn_cond=cross_attn_cond,
                cross_attn_cond_mask=cross_attn_cond_mask,
                input_concat_cond=input_concat_cond,
                global_embed=global_embed,
                prepend_cond=prepend_cond,
                prepend_cond_mask=prepend_cond_mask,
                mask=mask,
                return_info=return_info,
                segalign_depths=segalign_depths,
                **kwargs,
            )

        # CFG inference
        n_way = 3 if is_multi_scale else 2

        if n_way == 3 and negative_cross_attn_cond is None:
            warnings.warn(
                "3-way CFG requested (cfg_scale is list) but no negative_cross_attn_cond provided. "
                "Negative branch will equal positive branch, degenerating to 2-way with scale=(scale[0]-scale[1]).",
                stacklevel=2,
            )

        # Determine unconditioned strategy for scalar CFG
        true_uncond = not is_multi_scale and (
            negative_cross_attn_cond is None
            and negative_input_concat_cond is None
            and negative_global_embed is None
        )

        # --- Batch x and t ---
        batch_x = torch.cat([x] * n_way, dim=0)
        batch_t = torch.cat([t] * n_way, dim=0)

        # --- Global embed ---
        batch_global = None
        if global_embed is not None:
            if is_multi_scale:
                neg_g = negative_global_embed if negative_global_embed is not None else global_embed
            else:
                neg_g = global_embed if true_uncond else (negative_global_embed if negative_global_embed is not None else global_embed)

            if isinstance(global_embed, dict):
                batch_global = {}
                for k in global_embed:
                    parts = [global_embed[k], neg_g[k]]
                    if n_way == 3:
                        parts.append(neg_g[k])
                    batch_global[k] = torch.cat(parts, dim=0)
            else:
                parts = [global_embed, neg_g]
                if n_way == 3:
                    parts.append(neg_g)
                batch_global = torch.cat(parts, dim=0)

        # --- Input concat cond ---
        batch_input_concat = None
        if input_concat_cond is not None:
            null_embed = torch.zeros_like(input_concat_cond)
            if is_multi_scale:
                neg_ic = negative_input_concat_cond if negative_input_concat_cond is not None else null_embed
                batch_input_concat = torch.cat([input_concat_cond, neg_ic, null_embed], dim=0)
            else:
                if true_uncond:
                    batch_input_concat = torch.cat([input_concat_cond, null_embed], dim=0)
                else:
                    neg_ic = negative_input_concat_cond if negative_input_concat_cond is not None else input_concat_cond
                    batch_input_concat = torch.cat([input_concat_cond, neg_ic], dim=0)

        # --- Cross-attention cond ---
        batch_cond, batch_cond_mask = None, None
        if cross_attn_cond is not None:
            batch_cond, batch_cond_mask = self._prepare_cross_attn_batch(
                cross_attn_cond, cross_attn_cond_mask,
                negative_cross_attn_cond, negative_cross_attn_mask, n_way,
            )

        # --- Prepend cond ---
        batch_prepend, batch_prepend_mask = None, None
        if prepend_cond is not None:
            null_embed = torch.zeros_like(prepend_cond)
            if n_way == 3:
                batch_prepend = torch.cat([prepend_cond, prepend_cond, null_embed], dim=0)
            else:
                batch_prepend = torch.cat([prepend_cond, null_embed], dim=0)
            if prepend_cond_mask is not None:
                batch_prepend_mask = torch.cat([prepend_cond_mask] * n_way, dim=0)

        # --- Mask ---
        batch_mask = torch.cat([mask] * n_way, dim=0) if mask is not None else None

        # --- Forward ---
        batch_output = self._forward(
            batch_x, batch_t,
            cross_attn_cond=batch_cond,
            cross_attn_cond_mask=batch_cond_mask,
            mask=batch_mask,
            input_concat_cond=batch_input_concat,
            global_embed=batch_global,
            prepend_cond=batch_prepend,
            prepend_cond_mask=batch_prepend_mask,
            return_info=return_info,
            segalign_depths=segalign_depths,
            **kwargs,
        )

        if segalign_depths is not None:
            batch_output, projections = batch_output
        if return_info:
            batch_output, info = batch_output

        # --- CFG formula ---
        chunks = torch.chunk(batch_output, n_way, dim=0)
        if n_way == 3:
            cond_output, neg_output, uncond_output = chunks
            cfg_output = (
                uncond_output
                + (cond_output - uncond_output) * cfg_scale[0]
                - (neg_output - uncond_output) * cfg_scale[1]
            )
        else:
            cond_output, uncond_output = chunks
            cfg_output = uncond_output + (cond_output - uncond_output) * cfg_scale

        output = self._apply_cfg_rescale(cond_output, cfg_output, scale_phi)

        if segalign_depths is not None:
            return output, projections
        if return_info:
            return output, info
        return output
