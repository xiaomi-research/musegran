# Note: Parts of this package are adapted from stable-audio-tools.
# Copyright (c) 2023 Stability AI
# SPDX-License-Identifier: MIT
# Source: https://github.com/Stability-AI/stable-audio-tools
"""MuseGran: Multi-granularity conditioned music generation with latent diffusion."""

from .models.builder import (
    create_model_from_config,
    create_model_from_config_path,
    get_pretrained_model,
    get_pretrained_model_weight_fuse,
)
from .models.structplanner import (
    StructPlanner,
    StructPlannerConfig,
    spec_to_conditioning,
)

__all__ = [
    "create_model_from_config",
    "create_model_from_config_path",
    "get_pretrained_model",
    "get_pretrained_model_weight_fuse",
    "StructPlanner",
    "StructPlannerConfig",
    "spec_to_conditioning",
]
