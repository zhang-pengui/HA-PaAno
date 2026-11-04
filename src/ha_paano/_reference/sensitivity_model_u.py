"""Isolated U encoder with configurable effective LSDP alpha initialization."""

from __future__ import annotations

import os

from .model_FINAL_HA_PaAno import (
    DualPoolingEmbeddingHead,
    PatchEncoder as FormalHAPaAnoPatchEncoder,
)


class PatchEncoder(FormalHAPaAnoPatchEncoder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not self.is_univariate:
            raise ValueError("U sensitivity encoder requires C=1")
        requested = float(os.environ["PAANO_SENS_LSDP_ALPHA_INIT"])
        self.dual_pooling_head = DualPoolingEmbeddingHead(
            embed_dim=self.layers[-1],
            init_alpha=requested,
        )
        self.sensitivity_lsdp_alpha_init = requested
