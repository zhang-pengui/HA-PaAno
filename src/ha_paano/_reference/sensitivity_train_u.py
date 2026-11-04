"""Isolated revised-Jitter trainer for the Tuning-only sensitivity study.

Only the sample-level probability and noise multiplier are configurable. Scale
and temporal shift are deliberately absent. The formal U optimizer, objectives,
sampling rules, and schedules are delegated unchanged to train_FINAL_U_branch.
"""

from __future__ import annotations

import os

import torch

from . import train_FINAL_U_branch as _formal_train


LAST_DEBUG: dict[str, object] = {}


def _float_env(name: str) -> float:
    try:
        return float(os.environ[name])
    except KeyError as exc:
        raise RuntimeError(f"Missing required sensitivity setting: {name}") from exc


def patch_augment_uonly(x: torch.Tensor) -> torch.Tensor:
    batch_size, channels, _ = x.shape
    if channels > 1:
        return x

    probability = _float_env("PAANO_SENS_JITTER_P")
    noise_std = _float_env("PAANO_SENS_NOISE_STD")
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"Invalid jitter probability: {probability}")
    if noise_std < 0.0:
        raise ValueError(f"Invalid jitter strength: {noise_std}")

    mask = (torch.rand(batch_size, 1, 1, device=x.device) < probability).float()
    local_std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-5)
    noise = noise_std * local_std * torch.randn_like(x)
    output = mask * (x + noise) + (1.0 - mask) * x

    global LAST_DEBUG
    LAST_DEBUG = {
        "variant": "revised_jitter_only",
        "jitter_probability": probability,
        "noise_std": noise_std,
        "local_std_unbiased": False,
        "local_std_floor": 1e-5,
        "mask_shape": list(mask.shape),
        "noise_shape": list(noise.shape),
        "scale_enabled": False,
        "shift_enabled": False,
    }
    return torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)


def train_model(*args, **kwargs):
    original = _formal_train.patch_augment_uonly
    _formal_train.patch_augment_uonly = patch_augment_uonly
    try:
        return _formal_train.train_model(*args, **kwargs)
    finally:
        _formal_train.patch_augment_uonly = original

