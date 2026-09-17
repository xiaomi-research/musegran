"""Conditioned diffusion model wrapper and factory."""

import typing as tp

import torch
from torch import nn
from peft import LoraConfig, get_peft_model

from .autoencoders import AutoencoderPretransform, create_autoencoder_from_config
from .conditioners import MultiConditioner, create_multi_conditioner_from_conditioning_config
from .dit import DiffusionTransformer
from ..inference.generation import generate_diffusion_cond


class ConditionedDiffusionModelWrapper(nn.Module):
    """Wraps a diffusion backbone with conditioning logic and pretransform."""

    def __init__(
            self,
            model: nn.Module,
            conditioner: MultiConditioner,
            io_channels: int,
            sample_rate: int,
            min_input_length: int,
            diffusion_objective: tp.Literal["v", "rectified_flow"] = "v",
            pretransform: tp.Optional[nn.Module] = None,
            cross_attn_cond_ids: tp.Optional[tp.List[str]] = None,
            global_cond_ids: tp.Optional[tp.List[str]] = None,
            custom_emb_ids: tp.Optional[tp.List[str]] = None,
            input_concat_ids: tp.Optional[tp.List[str]] = None,
            prepend_cond_ids: tp.Optional[tp.List[str]] = None,
            ):
        super().__init__()

        self.model = model
        self.conditioner = conditioner
        self.io_channels = io_channels
        self.sample_rate = sample_rate
        self.diffusion_objective = diffusion_objective
        self.pretransform = pretransform
        self.min_input_length = min_input_length

        self.cross_attn_cond_ids = cross_attn_cond_ids or []
        self.global_cond_ids = global_cond_ids or []
        self.custom_emb_ids = custom_emb_ids or []
        self.input_concat_ids = input_concat_ids or []
        self.prepend_cond_ids = prepend_cond_ids or []

    def _collect_cross_attn(self, keys, conditioning_tensors):
        """Gather and concatenate cross-attention inputs [B, seq, D] from keys."""
        if not keys:
            return None, None
        inputs, masks = [], []
        for key in keys:
            cond_in, cond_mask = conditioning_tensors[key]
            if cond_in.dim() == 2:
                cond_in = cond_in.unsqueeze(1)
                cond_mask = cond_mask.unsqueeze(1)
            inputs.append(cond_in)
            masks.append(cond_mask)
        return torch.cat(inputs, dim=1), torch.cat(masks, dim=1)

    def get_conditioning_inputs(self, conditioning_tensors: tp.Dict[str, tp.Any], negative=False):
        cross_attention_input, cross_attention_masks = self._collect_cross_attn(
            self.cross_attn_cond_ids, conditioning_tensors
        )

        global_cond = None
        if self.global_cond_ids:
            global_cond = torch.cat([conditioning_tensors[k][0] for k in self.global_cond_ids], dim=-1)
            if global_cond.dim() == 3:
                global_cond = global_cond.squeeze(1)

        input_concat_cond = None
        if self.input_concat_ids:
            input_concat_cond = torch.cat(
                [conditioning_tensors[k][0] for k in self.input_concat_ids], dim=1
            )

        prepend_cond = None
        prepend_cond_mask = None
        if self.prepend_cond_ids:
            prepend_conds, prepend_masks = zip(
                *[conditioning_tensors[k] for k in self.prepend_cond_ids]
            )
            prepend_cond = torch.cat(prepend_conds, dim=1)
            prepend_cond_mask = torch.cat(prepend_masks, dim=1)

        custom_embs = None
        if self.custom_emb_ids:
            custom_embs = {k: conditioning_tensors[k] for k in self.custom_emb_ids}

        prefix = "negative_" if negative else ""
        return {
            f"{prefix}cross_attn_cond": cross_attention_input,
            f"{prefix}cross_attn_mask": cross_attention_masks,
            f"{prefix}global_cond": global_cond,
            f"{prefix}custom_emb": custom_embs,
            f"{prefix}input_concat_cond": input_concat_cond,
            f"{prefix}prepend_cond": prepend_cond,
            f"{prefix}prepend_cond_mask": prepend_cond_mask,
        }

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: tp.Dict[str, tp.Any], **kwargs):
        return self.model(x, t, **self.get_conditioning_inputs(cond), **kwargs)

    def generate(self, *args, **kwargs):
        return generate_diffusion_cond(self, *args, **kwargs)


class DiTWrapper(nn.Module):
    """Wraps DiffusionTransformer with optional LoRA and parameter scaling."""

    def __init__(self, *args, **kwargs):
        super().__init__()

        self._lora_cfg = kwargs.pop('lora', None)
        self.model = DiffusionTransformer(*args, **kwargs)

        with torch.no_grad():
            for param in self.model.parameters():
                param *= 0.5

    def apply_lora(self):
        """Apply LoRA adapters. Must be called AFTER loading pretrained weights."""
        lora_cfg = self._lora_cfg
        if lora_cfg and isinstance(lora_cfg, dict) and lora_cfg.get('enable', False):
            self._setup_lora(lora_cfg)

    def _setup_lora(self, lora_cfg: dict):
        """Apply PEFT LoRA adapters to the DiT model, keeping segalign projectors fully trainable."""
        rank = int(lora_cfg.get('rank', 16))
        alpha = int(lora_cfg.get('alpha', rank))
        target_modules = lora_cfg.get('target_modules', 'all-linear')

        modules_to_save = [
            n for n, m in self.model.named_modules()
            if 'segalign_projectors' in n and not isinstance(m, nn.ModuleList)
        ] or None

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            init_lora_weights=lora_cfg.get('init_lora_weights', "gaussian"),
            target_modules=target_modules,
            modules_to_save=modules_to_save,
        )
        self.model = get_peft_model(self.model, lora_config)

    def forward(self, x, t, cross_attn_mask=None, global_cond=None,
                negative_global_cond=None, **kwargs):
        return self.model(
            x, t,
            cross_attn_cond_mask=cross_attn_mask,
            global_embed=global_cond,
            negative_global_embed=negative_global_cond,
            **kwargs)


def freeze_conditioner(conditioner: MultiConditioner):
    """Freeze entire conditioner including projection layers (for LoRA fine-tuning)."""
    conditioner.eval()
    conditioner.requires_grad_(False)


def create_diffusion_cond_from_config(config: tp.Dict[str, tp.Any]):
    """Build a ConditionedDiffusionModelWrapper from a complete model config dict."""
    model_config = config["model"]

    diffusion_config = model_config.get('diffusion')
    if diffusion_config is None:
        raise ValueError("Must specify 'diffusion' in model config")

    diffusion_model_type = diffusion_config.get('type')
    if diffusion_model_type != 'dit':
        raise ValueError(f"Only 'dit' model type is supported, got '{diffusion_model_type}'")

    diffusion_model_config = diffusion_config.get('config')
    if diffusion_model_config is None:
        raise ValueError("Must specify diffusion model config")

    conditioning_config = model_config.get('conditioning')
    conditioner = None
    if conditioning_config is not None:
        conditioning_config.setdefault("sample_rate", config.get("sample_rate", 44100))
        conditioning_config.setdefault("sample_size", config.get("sample_size"))
        conditioning_config.setdefault("downsampling_ratio", model_config.get("pretransform", {}).get("config", {}).get("downsampling_ratio", 2048))
        conditioner = create_multi_conditioner_from_conditioning_config(conditioning_config)
        if conditioning_config.get('freeze', False):
            freeze_conditioner(conditioner)

    diffusion_model = DiTWrapper(**diffusion_model_config)

    io_channels = model_config.get('io_channels')
    if io_channels is None:
        raise ValueError("Must specify 'io_channels' in model config")

    sample_rate = config.get('sample_rate')
    if sample_rate is None:
        raise ValueError("Must specify 'sample_rate' in config")

    pretransform_config = model_config.get("pretransform")
    pretransform = None
    if pretransform_config is not None:
        inner_ae = create_autoencoder_from_config({
            "sample_rate": sample_rate,
            "model": pretransform_config["config"],
        })
        pretransform = AutoencoderPretransform(
            inner_ae,
            scale=pretransform_config.get("scale", 1.0),
            model_half=pretransform_config.get("model_half", False),
            iterate_batch=pretransform_config.get("iterate_batch", False),
        )
        pretransform.enable_grad = pretransform_config.get("enable_grad", False)
        pretransform.eval().requires_grad_(pretransform.enable_grad)
        min_input_length = pretransform.downsampling_ratio
    else:
        min_input_length = 1

    min_input_length *= diffusion_model.model.patch_size

    return ConditionedDiffusionModelWrapper(
        diffusion_model,
        conditioner,
        io_channels=io_channels,
        sample_rate=sample_rate,
        min_input_length=min_input_length,
        diffusion_objective=diffusion_config.get('diffusion_objective', 'v'),
        pretransform=pretransform,
        cross_attn_cond_ids=diffusion_config.get('cross_attention_cond_ids', []),
        global_cond_ids=diffusion_config.get('global_cond_ids', []),
        custom_emb_ids=diffusion_config.get('custom_emb_ids', []),
        input_concat_ids=diffusion_config.get('input_concat_ids', []),
        prepend_cond_ids=diffusion_config.get('prepend_cond_ids', []),
    )
