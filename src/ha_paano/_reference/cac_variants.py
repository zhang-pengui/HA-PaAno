"""Isolated Channel Activity Context (CAC) variants.

The class subclasses the frozen formal HA-PaAno encoder and replaces only its
multivariate ``channel_context`` module.  The formal forward method continues
to capture ``x_raw`` before RevIN and passes it to the replacement context.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn

from .model_FINAL_HA_PaAno import PatchEncoder as FormalPatchEncoder


VALID_VARIANTS = {
    "std_only",
    "diff_only",
    "no_relative_norm",
    "no_gate",
}


class AblatedChannelActivityContext(nn.Module):
    def __init__(
        self,
        variant: str,
        embed_dim: int = 64,
        context_hidden: int = 64,
        init_alpha: float = 0.10,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if variant not in VALID_VARIANTS:
            raise ValueError(f"Unsupported CAC internal variant: {variant!r}")
        self.variant = variant
        self.embed_dim = embed_dim
        self.eps = eps
        input_dim = 4 if variant in {"std_only", "diff_only"} else 8
        self.context_mlp = nn.Sequential(
            nn.Linear(input_dim, context_hidden),
            nn.LayerNorm(context_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(context_hidden, embed_dim),
        )
        if variant != "no_gate":
            bounded_alpha = min(max(init_alpha, 1e-4), 1.0 - 1e-4)
            init_logit = math.log(bounded_alpha / (1.0 - bounded_alpha))
            self.alpha_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))

        self.collect_activity = False
        self.activity_cache: list[dict[str, torch.Tensor]] = []
        self.last_debug: dict[str, object] = {}
        self.last_raw_input: torch.Tensor | None = None

    def clear_activity_cache(self) -> None:
        self.activity_cache = []

    def get_activity_cache(self):
        return self.activity_cache

    @staticmethod
    def _four_stats(activity: torch.Tensor) -> list[torch.Tensor]:
        return [
            activity.mean(dim=1),
            activity.std(dim=1, unbiased=False),
            activity.max(dim=1).values,
            activity.min(dim=1).values,
        ]

    def _relative(self, activity: torch.Tensor) -> torch.Tensor:
        denominator = activity.mean(dim=1, keepdim=True) + self.eps
        return activity / denominator

    def forward(self, h: torch.Tensor, x_raw: torch.Tensor) -> torch.Tensor:
        if x_raw.ndim != 3:
            return h
        batch_size, channels, length = x_raw.shape
        if channels <= 1:
            return h

        ch_std = x_raw.std(dim=-1, unbiased=False)
        if length > 1:
            ch_diff = torch.abs(x_raw[:, :, 1:] - x_raw[:, :, :-1]).mean(dim=-1)
        else:
            ch_diff = torch.zeros_like(ch_std)
        ch_std = torch.nan_to_num(ch_std, nan=0.0, posinf=0.0, neginf=0.0)
        ch_diff = torch.nan_to_num(ch_diff, nan=0.0, posinf=0.0, neginf=0.0)

        use_relative = self.variant != "no_relative_norm"
        std_activity = self._relative(ch_std) if use_relative else ch_std
        diff_activity = self._relative(ch_diff) if use_relative else ch_diff
        std_activity = torch.nan_to_num(std_activity, nan=0.0, posinf=0.0, neginf=0.0)
        diff_activity = torch.nan_to_num(diff_activity, nan=0.0, posinf=0.0, neginf=0.0)

        if self.variant == "std_only":
            stat_parts = self._four_stats(std_activity)
        elif self.variant == "diff_only":
            stat_parts = self._four_stats(diff_activity)
        else:
            stat_parts = self._four_stats(std_activity) + self._four_stats(diff_activity)
        stats = torch.stack(stat_parts, dim=1)
        stats = torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)

        if os.environ.get("PAANO_INTERNAL_SMOKE") == "1":
            self.last_raw_input = x_raw.detach().clone()
        self.last_debug = {
            "raw_shape": tuple(x_raw.shape),
            "stats_shape": tuple(stats.shape),
            "context_input_dim": int(stats.shape[1]),
            "used_relative_norm": use_relative,
            "uses_learnable_gate": self.variant != "no_gate",
        }

        if self.collect_activity:
            self.activity_cache.append(
                {
                    "ch_std_rel": std_activity.detach().cpu(),
                    "ch_diff_rel": diff_activity.detach().cpu(),
                }
            )

        context = self.context_mlp(stats)
        context = torch.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
        if self.variant == "no_gate":
            h_new = h + context
        else:
            alpha = torch.sigmoid(self.alpha_logit)
            h_new = h + alpha * context
        return torch.nan_to_num(h_new, nan=0.0, posinf=0.0, neginf=0.0)


class PatchEncoder(FormalPatchEncoder):
    """Formal multivariate encoder with one requested CAC substitution."""

    def __init__(self, *args, **kwargs) -> None:
        variant = os.environ.get("PAANO_INTERNAL_VARIANT", "")
        if variant not in VALID_VARIANTS:
            raise ValueError(f"Unsupported CAC internal variant: {variant!r}")
        super().__init__(*args, **kwargs)
        if not self.is_multivariate:
            raise ValueError("CAC internal ablations are restricted to C>1 data.")
        if self.dual_pooling_head is not None:
            raise RuntimeError("CAC ablation unexpectedly enabled LSDP.")
        self.channel_context = AblatedChannelActivityContext(
            variant=variant,
            embed_dim=self.layers[-1],
            context_hidden=64,
            init_alpha=0.10,
        )
        self.internal_ablation_variant = variant

