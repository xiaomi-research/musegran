"""Model construction, checkpoint loading, and weight fusion utilities."""

import json
import logging
import os

import torch
import yaml
from safetensors.torch import load_file
from torch import nn
from torch.nn.utils import remove_weight_norm

logger = logging.getLogger(__name__)


def load_ckpt_state_dict(ckpt_path: str) -> dict:
    """Load a state dict from .safetensors or .ckpt/.pt file."""
    if ckpt_path.endswith(".safetensors"):
        return load_file(ckpt_path)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in state_dict:
        return state_dict["state_dict"]
    return state_dict


def copy_state_dict(model: nn.Module, state_dict: dict) -> tuple:
    """Load state_dict into model, but only for keys whose name AND shape match exactly.

    Size-mismatched keys are silently skipped (left at their random init), so
    fine-tuning from a checkpoint with a different conditioning interface works.
    """
    model_state_dict = model.state_dict()
    for key in state_dict:
        if key in model_state_dict and state_dict[key].shape == model_state_dict[key].shape:
            model_state_dict[key] = state_dict[key].data if isinstance(state_dict[key], nn.Parameter) else state_dict[key]
    model.load_state_dict(model_state_dict, strict=False)
    return [], []


def remove_weight_norm_from_model(model) -> nn.Module:
    """Remove weight normalization from all submodules (silently skips modules without it)."""
    for module in model.modules():
        if hasattr(module, "weight"):
            try:
                remove_weight_norm(module)
            except ValueError:
                pass
    return model


def create_model_from_config(model_config: dict) -> nn.Module:
    """Dispatch to autoencoder or diffusion model factory based on model_type."""
    model_type = model_config.get('model_type')
    if model_type is None:
        raise ValueError("model_type must be specified in model config")
    if model_type == 'autoencoder':
        from .autoencoders import create_autoencoder_from_config
        return create_autoencoder_from_config(model_config)
    elif model_type == 'diffusion_cond':
        from .diffusion import create_diffusion_cond_from_config
        return create_diffusion_cond_from_config(model_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_model_from_config_path(model_config_path: str) -> nn.Module:
    """Create a model from a JSON or YAML config file path."""
    with open(model_config_path) as f:
        raw = f.read()
    try:
        model_config = json.loads(raw)
    except json.JSONDecodeError:
        model_config = yaml.safe_load(raw)
    return create_model_from_config(model_config)


def _strip_prefix(state_dict, prefix='diffusion.'):
    """Strip training wrapper prefix from state_dict keys if present."""
    stripped = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    return stripped if stripped else state_dict


def _load_config(path: str) -> dict:
    with open(path) as f:
        if path.endswith(('.yaml', '.yml')):
            return yaml.safe_load(f)
        return json.load(f)


def _resolve_config_path(model_path: str, model_config_name: str = None) -> str:
    """Resolve model config file path from model directory."""
    if model_config_name is not None and (os.path.isabs(model_config_name) or os.path.exists(model_config_name)):
        return model_config_name
    if model_config_name is not None:
        return os.path.join(model_path, model_config_name)
    for name in ("model_config.yaml", "model_config.json"):
        candidate = os.path.join(model_path, name)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(model_path, "model_config.yaml")


def get_pretrained_model(
    model_path: str,
    model_name: str,
    model_config_name: str = None,
    vae_ckpt: str = None,
) -> tuple:
    """Load a pretrained model from a directory.

    Loading order:
        1. create_model_from_config() — builds model; conditioner backbones
           (Qwen/T5/GLAP/MuQ) load their own pretrained weights internally.
        2. load_state_dict(ckpt) — loads DiT weights + conditioner adapter layers
           (proj_out, null_emb) from the training checkpoint.
        3. VAE pretransform loaded from a separate checkpoint (not part of the
           training save since it's frozen).
    """
    config_path = _resolve_config_path(model_path, model_config_name)
    model_config = _load_config(config_path)

    model = create_model_from_config(model_config)

    ckpt_name = model_name if model_name is not None else "model.ckpt"
    ckpt_path = ckpt_name if os.path.isabs(ckpt_name) or os.path.exists(ckpt_name) else os.path.join(model_path, ckpt_name)
    state_dict = _strip_prefix(load_ckpt_state_dict(ckpt_path))
    result = model.load_state_dict(state_dict, strict=False)
    # Known-benign checkpoint leftovers: zero norm-bias buffers (beta) from the
    # bias-less LayerNorm and non-persistent rotary buffers (fallback_inv_freq /
    # rope_modules inv_freq) recomputed deterministically at load.
    benign = [k for k in result.unexpected_keys if k.endswith(".beta") or "fallback_inv_freq" in k or k.endswith(".inv_freq")]
    other_unexpected = [k for k in result.unexpected_keys if k not in benign]
    if other_unexpected:
        print(f"[load_state_dict] Unexpected ({len(other_unexpected)}): {other_unexpected}")
    logger.debug("Skipped %d known-benign checkpoint buffers (norm beta / rotary fallback).", len(benign))
    _INDEPENDENT_PREFIXES = ("glap_model", "segalign_projector", "qwen_", "muq_model", "pretransform", "struct-combined-text.model")
    suspicious_missing = [k for k in result.missing_keys if not any(p in k for p in _INDEPENDENT_PREFIXES)]
    if suspicious_missing:
        print(f"[load_state_dict] Missing (non-submodel, {len(suspicious_missing)}): {suspicious_missing}")

    if vae_ckpt and model.pretransform is not None:
        vae_sd = load_ckpt_state_dict(vae_ckpt)
        vae_sd = _strip_prefix(vae_sd, "autoencoder.")
        model.pretransform.load_state_dict(vae_sd, strict=False)
        logger.info("VAE loaded: %s", vae_ckpt)

    return model, model_config


def get_pretrained_model_weight_fuse(
    model_infos,
    weights=None,
    model_config_name=None,
    save_ckpt_path=None,
    vae_ckpt: str = None,
) -> tuple:
    """Load multiple checkpoints and fuse their weights via weighted average."""
    model_path = model_infos[0][0]
    config_path = _resolve_config_path(model_path, model_config_name)
    model_config = _load_config(config_path)

    model = create_model_from_config(model_config)

    state_dicts = []
    for model_path, model_name in model_infos:
        ckpt_name = model_name if model_name is not None else "model.ckpt"
        sd = _strip_prefix(load_ckpt_state_dict(os.path.join(model_path, ckpt_name)))
        state_dicts.append(sd)

    # Weighted average fusion
    fused_state_dict = {}
    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)
    else:
        assert len(weights) == len(model_infos), \
            f"weights count ({len(weights)}) != model count ({len(model_infos)})"

    for key in state_dicts[0].keys():
        if all(key in sd for sd in state_dicts):
            fused_state_dict[key] = sum(w * sd[key] for w, sd in zip(weights, state_dicts))
        else:
            logger.warning("Key missing in some models, skipping: %s", key)

    model.load_state_dict(fused_state_dict, strict=False)

    if vae_ckpt and model.pretransform is not None:
        vae_sd = load_ckpt_state_dict(vae_ckpt)
        vae_sd = _strip_prefix(vae_sd, "autoencoder.")
        model.pretransform.load_state_dict(vae_sd, strict=False)
        logger.info("VAE loaded: %s", vae_ckpt)

    if save_ckpt_path:
        os.makedirs(os.path.dirname(save_ckpt_path), exist_ok=True)
        torch.save(fused_state_dict, save_ckpt_path)
        logger.info("Fused model saved: %s", save_ckpt_path)

    return model, model_config
