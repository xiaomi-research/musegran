"""Diffusion-based audio generation."""

from __future__ import annotations

import numpy as np
import torch
import typing as tp

from .sampling import sample_k, sample_rf


def generate_diffusion_cond(
        model: "ConditionedDiffusionModelWrapper",
        steps: int = 250,
        cfg_scale: float = 6,
        conditioning: tp.Optional[tp.List[dict]] = None,
        conditioning_tensors: tp.Optional[dict] = None,
        negative_conditioning: tp.Optional[tp.List[dict]] = None,
        negative_conditioning_tensors: tp.Optional[dict] = None,
        batch_size: int = 1,
        sample_size: int = 44100 * 47,  # ~47 seconds at 44.1kHz
        seed: int = -1,
        device: str = "cuda",
        return_latents: bool = False,
        **sampler_kwargs,
        ) -> tp.Tuple[torch.Tensor, int]:
    """Generate audio from conditioning using a diffusion model.

    Args:
        model: The diffusion model to use for generation.
        steps: Number of diffusion steps.
        cfg_scale: Classifier-free guidance scale.
        conditioning: Conditioning dict to be encoded by the model's conditioner.
        conditioning_tensors: Pre-computed conditioning tensors (skips conditioner).
        negative_conditioning: Negative conditioning dict for CFG.
        batch_size: Number of samples to generate in parallel.
        sample_size: Target audio length in samples.
        seed: Random seed (-1 for random).
        device: Device for generation.
        return_latents: If True, return raw latents without decoding.
        **sampler_kwargs: Passed to the sampler (sigma_min, sigma_max, sampler_type, etc.).
            For v-prediction: sigma_min, sigma_max, sampler_type are used.
            For rectified_flow: only sigma_max is used (others are ignored).

    Returns:
        (output, seed): Generated audio tensor and the seed used.
    """

    # For latent diffusion, convert audio sample size to latent size
    if model.pretransform is not None:
        sample_size = sample_size // model.pretransform.downsampling_ratio

    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1, dtype=np.uint32)
    torch.manual_seed(seed)
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)

    # Disable reduced-precision ops for deterministic generation quality
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False

    # Encode conditioning
    assert conditioning is not None or conditioning_tensors is not None, "Must provide either conditioning or conditioning_tensors"
    if conditioning_tensors is None:
        conditioning_tensors, _ = model.conditioner(conditioning, device)
    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)

    if negative_conditioning is not None or negative_conditioning_tensors is not None:
        if negative_conditioning_tensors is None:
            negative_conditioning_tensors, _ = model.conditioner(negative_conditioning, device)
        negative_conditioning_tensors = model.get_conditioning_inputs(negative_conditioning_tensors, negative=True)
    else:
        negative_conditioning_tensors = {}

    # Cast to model dtype
    model_dtype = next(model.model.parameters()).dtype
    noise = noise.type(model_dtype)
    conditioning_inputs = {k: v.type(model_dtype) if isinstance(v, torch.Tensor) else v for k, v in conditioning_inputs.items()}

    # Sample
    common_kwargs = dict(**conditioning_inputs, **negative_conditioning_tensors, cfg_scale=cfg_scale)

    if model.diffusion_objective == "v":
        sampled = sample_k(model.model, noise, steps, **sampler_kwargs, **common_kwargs)
    elif model.diffusion_objective == "rectified_flow":
        sampled = sample_rf(model.model, noise, steps, sigma_max=sampler_kwargs.get("sigma_max", 1), **common_kwargs)

    del noise, conditioning_tensors, conditioning_inputs
    torch.cuda.empty_cache()

    # Decode latents to audio
    if model.pretransform is not None and not return_latents:
        sampled = sampled.to(next(model.pretransform.parameters()).dtype)
        sampled = model.pretransform.decode(sampled)

    return sampled, seed
