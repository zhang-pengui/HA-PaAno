"""Revision-only encoders for the cross-branch and unified three-seed pilot.

The shared CNN, RevIN, projection/classification heads, LSDP, and accepted
revised CAC are imported from their protected formal sources.  This module
only controls which already-defined heads are attached to each pilot
condition.  It is never imported by the formal launchers.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from .model_baseline import PatchEncoder as BaselinePatchEncoder
from .model_FINAL_HA_PaAno import DualPoolingEmbeddingHead
from .model_FINAL_HA_PaAno_CANONICAL import FinalChannelActivityContext


CONDITIONS = {
    "m_jitter_lsdp": {"track": "M", "lsdp": True, "cac": False, "jitter": True},
    "u_cac": {"track": "U", "lsdp": False, "cac": True, "jitter": False},
    "u_jitter_lsdp_cac": {"track": "U", "lsdp": True, "cac": True, "jitter": True},
    "m_jitter_lsdp_cac": {"track": "M", "lsdp": True, "cac": True, "jitter": True},
}


class SingleChannelTransferCAC(FinalChannelActivityContext):
    """Accepted CAC with only the C<=1 early-return gate bypassed.

    For C>1 this delegates directly to the protected canonical forward.  For
    C=1 it performs the same operations in the same order, yielding the exact
    descriptor [s, 0, s, s, d, 0, d, d].
    """

    def forward(self, h: torch.Tensor, x_raw: torch.Tensor) -> torch.Tensor:
        if x_raw.ndim != 3:
            return h
        _, channels, length = x_raw.shape
        if channels > 1:
            return super().forward(h, x_raw)

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
            "single_channel_gate_bypassed": True,
            "descriptor": stats.detach().cpu(),
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


class PatchEncoder(BaselinePatchEncoder):
    """Condition-selected encoder injected into the unchanged formal entry."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        condition = os.environ.get("PAANO_CROSS_CONDITION", "")
        if condition not in CONDITIONS:
            raise RuntimeError(f"Unknown cross-branch condition: {condition!r}")
        spec = CONDITIONS[condition]
        expected_u = spec["track"] == "U"
        if expected_u != (int(kwargs.get("in_channels", args[0] if args else 1)) == 1):
            raise ValueError(f"Condition {condition} received incompatible channel count")

        self.condition = condition
        self.is_univariate = expected_u
        self.is_multivariate = not expected_u
        self.jitter_enabled = bool(spec["jitter"])
        self.dual_pooling_head: Optional[DualPoolingEmbeddingHead] = None
        if spec["lsdp"]:
            self.dual_pooling_head = DualPoolingEmbeddingHead(
                embed_dim=self.layers[-1], init_alpha=0.05
            )

        self.channel_context: Optional[FinalChannelActivityContext] = None
        if spec["cac"]:
            cac_cls = SingleChannelTransferCAC if expected_u else FinalChannelActivityContext
            self.channel_context = cac_cls(
                embed_dim=self.layers[-1], context_hidden=64, init_alpha=0.10
            )

        self._queued_raw_context: list[torch.Tensor] = []
        self.last_raw_context_debug: dict[str, object] = {}

    def queue_raw_context(self, x_raw: torch.Tensor) -> None:
        """Record each pre-jitter tensor for the next concatenated embedding call."""
        if self.channel_context is not None:
            self._queued_raw_context.append(x_raw)

    def clear_raw_context_queue(self) -> None:
        self._queued_raw_context = []

    def _resolve_raw_context(self, x: torch.Tensor) -> torch.Tensor:
        if not self._queued_raw_context:
            return x
        raw = torch.cat(self._queued_raw_context, dim=0)
        self._queued_raw_context = []
        if raw.shape != x.shape or raw.device != x.device or raw.dtype != x.dtype:
            raise RuntimeError(
                f"Queued raw context mismatch: raw={tuple(raw.shape)}/{raw.device}/{raw.dtype}, "
                f"main={tuple(x.shape)}/{x.device}/{x.dtype}"
            )
        self.last_raw_context_debug = {
            "main_shape": tuple(x.shape),
            "raw_shape": tuple(raw.shape),
            "same_storage": raw.data_ptr() == x.data_ptr(),
            "raw_context_queue_consumed": True,
        }
        return raw

    def _forward_with_raw(
        self,
        x: torch.Tensor,
        x_raw: torch.Tensor,
        return_embedding: bool = False,
        return_projection: bool = False,
    ) -> torch.Tensor:
        if self.revin is not None:
            x = self.revin.norm(x)
        for block in self.convblocks:
            x = block(x)

        if self.dual_pooling_head is not None:
            h = self.dual_pooling_head(x)
        else:
            h = self.fc_embedding(x).flatten(start_dim=1)
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

        if self.channel_context is not None:
            h = self.channel_context(h, x_raw)
        if return_embedding:
            return h
        if return_projection:
            return self.projection_head(h)
        raise ValueError("The forward method is not designed to handle classification directly.")

    def forward(self, x, return_embedding=False, return_projection=False):
        return self._forward_with_raw(x, x, return_embedding, return_projection)

    def embedding(self, x):
        x_raw = self._resolve_raw_context(x)
        return self._forward_with_raw(x, x_raw, return_embedding=True)


__all__ = ["CONDITIONS", "SingleChannelTransferCAC", "PatchEncoder"]
