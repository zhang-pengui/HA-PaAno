"""Revision-only Jitter transfer using the protected formal U trainer.

All objective, optimizer, scheduler, sampling, and iteration behavior is
delegated to train_FINAL_U_branch.  The only injected function is the frozen
revised Jitter.  For unified conditions, the model records each untouched
pre-jitter tensor so canonical CAC receives raw input rather than augmented
input.
"""

from __future__ import annotations

import os

import torch

from . import train_FINAL_U_branch as _formal_train
from . import sensitivity_train_u


LAST_DEBUG: dict[str, object] = {}


def transferred_jitter(x: torch.Tensor) -> torch.Tensor:
    """Frozen revised Jitter, extended to C>1 by bypassing only its C gate."""
    if x.shape[1] == 1:
        output = sensitivity_train_u.patch_augment_uonly(x)
        global LAST_DEBUG
        LAST_DEBUG = {**sensitivity_train_u.LAST_DEBUG, "transfer_gate_bypassed": False}
        return output

    probability = float(os.environ["PAANO_SENS_JITTER_P"])
    noise_std = float(os.environ["PAANO_SENS_NOISE_STD"])
    if not 0.0 <= probability <= 1.0 or noise_std < 0.0:
        raise ValueError("Invalid frozen Jitter settings")

    batch_size = x.shape[0]
    mask = (torch.rand(batch_size, 1, 1, device=x.device) < probability).float()
    local_std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-5)
    noise = noise_std * local_std * torch.randn_like(x)
    output = mask * (x + noise) + (1.0 - mask) * x
    LAST_DEBUG = {
        "variant": "revised_jitter_only_transferred_to_M",
        "jitter_probability": probability,
        "noise_std": noise_std,
        "local_std_axis": -1,
        "local_std_unbiased": False,
        "local_std_floor": 1e-5,
        "mask_shape": list(mask.shape),
        "noise_shape": list(noise.shape),
        "transfer_gate_bypassed": True,
        "scale_enabled": False,
        "shift_enabled": False,
    }
    return torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)


def train_model(model, *args, **kwargs):
    """Run the protected formal trainer with one temporary Jitter hook."""
    original = _formal_train.patch_augment_uonly

    def jitter_with_raw_capture(x: torch.Tensor) -> torch.Tensor:
        model.queue_raw_context(x)
        return transferred_jitter(x)

    _formal_train.patch_augment_uonly = jitter_with_raw_capture
    model.clear_raw_context_queue()
    try:
        return _formal_train.train_model(model, *args, **kwargs, use_patch_aug=True)
    finally:
        _formal_train.patch_augment_uonly = original
        model.clear_raw_context_queue()


__all__ = ["LAST_DEBUG", "transferred_jitter", "train_model"]
