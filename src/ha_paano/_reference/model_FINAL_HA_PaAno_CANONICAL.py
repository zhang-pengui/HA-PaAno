"""Canonical final HA-PaAno model used for submission reproducibility.

The univariate branch and the shared encoder are inherited unchanged from the
frozen HA-PaAno implementation.  The multivariate branch uses the accepted
revised CAC definition by default: channel standard deviation and mean
absolute first-difference activity are computed directly from the raw patch
captured before RevIN.  No relative normalization is applied.

This module intentionally has no environment-variable switch and no ablation
name.  Legacy Full-CAC and internal-ablation modules remain available through
their original paths for traceability.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .model_FINAL_HA_PaAno import PatchEncoder as SharedHAPaAnoPatchEncoder


class FinalChannelActivityContext(nn.Module):
    """Accepted final CAC for the multivariate HA-PaAno branch."""

    def __init__(
        self,
        embed_dim: int = 64,
        context_hidden: int = 64,
        init_alpha: float = 0.10,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.context_mlp = nn.Sequential(
            nn.Linear(8, context_hidden),
            nn.LayerNorm(context_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(context_hidden, embed_dim),
        )

        bounded_alpha = min(max(init_alpha, 1e-4), 1.0 - 1e-4)
        init_logit = math.log(bounded_alpha / (1.0 - bounded_alpha))
        self.alpha_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))

        # Disabled by default; enabled only by the diagnostic logger.
        self.collect_activity = False
        self.activity_cache: list[dict[str, torch.Tensor]] = []
        self.last_debug: dict[str, object] = {}

    def clear_activity_cache(self) -> None:
        self.activity_cache = []

    def get_activity_cache(self) -> list[dict[str, torch.Tensor]]:
        return self.activity_cache

    @staticmethod
    def _four_stats(activity: torch.Tensor) -> list[torch.Tensor]:
        return [
            activity.mean(dim=1),
            activity.std(dim=1, unbiased=False),
            activity.max(dim=1).values,
            activity.min(dim=1).values,
        ]

    def forward(self, h: torch.Tensor, x_raw: torch.Tensor) -> torch.Tensor:
        if x_raw.ndim != 3:
            return h
        batch_size, channels, length = x_raw.shape
        if channels <= 1:
            return h

        channel_std_activity = x_raw.std(dim=-1, unbiased=False)
        if length > 1:
            first_difference_activity = torch.abs(
                x_raw[:, :, 1:] - x_raw[:, :, :-1]
            ).mean(dim=-1)
        else:
            first_difference_activity = torch.zeros_like(channel_std_activity)

        channel_std_activity = torch.nan_to_num(
            channel_std_activity, nan=0.0, posinf=0.0, neginf=0.0
        )
        first_difference_activity = torch.nan_to_num(
            first_difference_activity, nan=0.0, posinf=0.0, neginf=0.0
        )

        stats = torch.stack(
            self._four_stats(channel_std_activity)
            + self._four_stats(first_difference_activity),
            dim=1,
        )
        stats = torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)

        self.last_debug = {
            "raw_shape": tuple(x_raw.shape),
            "stats_shape": tuple(stats.shape),
            "context_input_dim": int(stats.shape[1]),
            "used_relative_norm": False,
            "uses_learnable_gate": True,
        }

        if self.collect_activity:
            self.activity_cache.append(
                {
                    "channel_std_activity": channel_std_activity.detach().cpu(),
                    "first_difference_activity": first_difference_activity.detach().cpu(),
                }
            )

        context = self.context_mlp(stats)
        context = torch.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
        alpha = torch.sigmoid(self.alpha_logit)
        output = h + alpha * context
        return torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)


class PatchEncoder(SharedHAPaAnoPatchEncoder):
    """Canonical final HA-PaAno patch encoder."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.is_multivariate:
            if self.dual_pooling_head is not None:
                raise RuntimeError("The multivariate branch must not enable LSDP.")
            self.channel_context = FinalChannelActivityContext(
                embed_dim=self.layers[-1],
                context_hidden=64,
                init_alpha=0.10,
            )
        self.canonical_model_name = "HA-PaAno final"
        self.cac_definition = "raw pre-RevIN channel activity without relative normalization"


__all__ = ["FinalChannelActivityContext", "PatchEncoder"]
