"""Revision-only URPA transform switches.

The default (``full``) mirrors ``train_FINAL_U_branch.patch_augment_uonly``
statement-for-statement.  Other modes only disable transforms; the formal
trainer, optimizer, objectives, and evaluator remain unchanged.
"""

from __future__ import annotations

import os

import torch

from . import train_FINAL_U_branch as _formal_train


VARIANT_SWITCHES = {
    "full": (True, True, True),
    "all_off": (False, False, False),
    "scale_only": (True, False, False),
    "jitter_only": (False, True, False),
    "shift_only": (False, False, True),
}
LAST_DEBUG: dict[str, object] = {}


def active_variant() -> str:
    variant = os.environ.get("PAANO_URPA_TRANSFORM_VARIANT", "full")
    if variant not in VARIANT_SWITCHES:
        raise ValueError(f"Unsupported URPA transform variant: {variant!r}")
    return variant


def patch_augment_uonly(x: torch.Tensor) -> torch.Tensor:
    variant = active_variant()
    enable_scale, enable_jitter, enable_shift = VARIANT_SWITCHES[variant]
    batch_size, channels, _ = x.shape

    if channels > 1:
        return x

    device = x.device
    mask = (torch.rand(batch_size, 1, 1, device=device) < 0.50).float()
    x_aug = x

    if enable_scale:
        scale = 1.0 + 0.05 * torch.randn(batch_size, channels, 1, device=device)
        x_scaled = x_aug * scale
        x_aug = mask * x_scaled + (1.0 - mask) * x_aug

    if enable_jitter:
        local_std = x_aug.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-5)
        noise = 0.01 * local_std * torch.randn_like(x_aug)
        x_noisy = x_aug + noise
        x_aug = mask * x_noisy + (1.0 - mask) * x_aug

    if enable_shift:
        x_shifted = _formal_train._shift_1d_batch(x_aug, max_shift=2)
        x_aug = mask * x_shifted + (1.0 - mask) * x_aug

    global LAST_DEBUG
    LAST_DEBUG = {
        "variant": variant,
        "augmentation_probability": 0.50,
        "enable_scale": enable_scale,
        "enable_jitter": enable_jitter,
        "enable_shift": enable_shift,
        "scale_std": 0.05,
        "noise_std": 0.01,
        "shift_max": 2,
        "shared_sample_mask": True,
        "operation_order": ["scale", "jitter", "shift"],
        "jitter_local_std_from_current_input": True,
        "shift_boundary_rule": "edge replication",
    }
    return torch.nan_to_num(x_aug, nan=0.0, posinf=0.0, neginf=0.0)


def train_model(*args, **kwargs):
    original = _formal_train.patch_augment_uonly
    _formal_train.patch_augment_uonly = patch_augment_uonly
    try:
        return _formal_train.train_model(*args, **kwargs)
    finally:
        _formal_train.patch_augment_uonly = original
