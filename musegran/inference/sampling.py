"""Sampling methods for diffusion inference.

Supports k-diffusion (v-prediction) and rectified flow objectives.
"""

import math

import torch
import k_diffusion.external as k_external
import k_diffusion.sampling as k_sampling
import k_diffusion.utils as k_utils
from tqdm import tqdm


def get_alphas_sigmas(t):
    """Return scaling factors (alpha, sigma) for a given timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


@torch.no_grad()
def sample_discrete_euler(model, x, steps, sigma_max=1, **extra_args):
    """Draws samples from a model given starting noise. Euler method for rectified flow."""
    t = torch.linspace(sigma_max, 0, steps + 1)

    for t_curr, t_prev in tqdm(zip(t[:-1], t[1:])):
        t_curr_tensor = t_curr * torch.ones(
            (x.shape[0],), dtype=x.dtype, device=x.device
        )
        dt = t_prev - t_curr
        x = x + dt * model(x, t_curr_tensor, **extra_args)

    return x


def make_cond_model_fn(model, cond_fn):
    """Wrap a model with a conditioning gradient function."""
    def cond_model_fn(x, sigma, **kwargs):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            denoised = model(x, sigma, **kwargs)
            cond_grad = cond_fn(x, sigma, denoised=denoised, **kwargs).detach()
            cond_denoised = denoised.detach() + cond_grad * k_utils.append_dims(sigma**2, x.ndim)
        return cond_denoised
    return cond_model_fn


SAMPLERS = {
    "k-heun": k_sampling.sample_heun,
    "k-lms": k_sampling.sample_lms,
    "k-dpmpp-2s-ancestral": k_sampling.sample_dpmpp_2s_ancestral,
    "k-dpm-2": k_sampling.sample_dpm_2,
    "k-dpm-fast": k_sampling.sample_dpm_fast,
    "k-dpm-adaptive": k_sampling.sample_dpm_adaptive,
    "dpmpp-2m-sde": k_sampling.sample_dpmpp_2m_sde,
    "dpmpp-3m-sde": k_sampling.sample_dpmpp_3m_sde,
}


def sample_k(
        model_fn,
        noise,
        steps=100,
        sampler_type="dpmpp-3m-sde",
        sigma_min=0.5,
        sigma_max=50,
        rho=1.0,
        device="cuda",
        callback=None,
        cond_fn=None,
        **extra_args
    ):
    """Sample using k-diffusion schedulers (v-prediction objective)."""
    denoiser = k_external.VDenoiser(model_fn)

    if cond_fn is not None:
        denoiser = make_cond_model_fn(denoiser, cond_fn)

    sigmas = k_sampling.get_sigmas_polyexponential(steps, sigma_min, sigma_max, rho, device=device)
    x = noise * sigmas[0]

    sampler_fn = SAMPLERS.get(sampler_type)
    assert sampler_fn is not None, f"Unknown sampler_type: {sampler_type}. Available: {list(SAMPLERS.keys())}"

    with torch.amp.autocast('cuda'):
        if sampler_type == "k-dpm-fast":
            return sampler_fn(denoiser, x, sigma_min, sigma_max, steps, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "k-dpm-adaptive":
            return sampler_fn(denoiser, x, sigma_min, sigma_max, rtol=0.01, atol=0.01, disable=False, callback=callback, extra_args=extra_args)
        return sampler_fn(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)


def sample_rf(
        model_fn,
        noise,
        steps=100,
        sigma_max=1,
        device="cuda",
        callback=None,
        cond_fn=None,
        **extra_args
    ):
    """Sample using discrete Euler method (rectified flow objective)."""
    if sigma_max > 1:
        sigma_max = 1

    if cond_fn is not None:
        model_fn = make_cond_model_fn(model_fn, cond_fn)

    with torch.amp.autocast('cuda'):
        return sample_discrete_euler(model_fn, noise, steps, sigma_max, **extra_args)
