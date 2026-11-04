"""Register the six missing seed-2000 factorial conditions.

The actual encoder, transferred Jitter, canonical LSDP, canonical CAC, and
single-channel CAC transfer are imported from the already validated
cross_branch_full_pilot_3seed wrappers.  This file changes no scientific
operation; it only adds the missing condition switch combinations.
"""

from __future__ import annotations

from . import cross_branch_model as _validated


FACTORIAL_CONDITIONS = {
    "u_jitter_cac": {"track": "U", "lsdp": False, "cac": True, "jitter": True},
    "u_lsdp_cac": {"track": "U", "lsdp": True, "cac": True, "jitter": False},
    "m_jitter": {"track": "M", "lsdp": False, "cac": False, "jitter": True},
    "m_lsdp": {"track": "M", "lsdp": True, "cac": False, "jitter": False},
    "m_jitter_cac": {"track": "M", "lsdp": False, "cac": True, "jitter": True},
    "m_lsdp_cac": {"track": "M", "lsdp": True, "cac": True, "jitter": False},
}

_validated.CONDITIONS.update(FACTORIAL_CONDITIONS)

PatchEncoder = _validated.PatchEncoder
SingleChannelTransferCAC = _validated.SingleChannelTransferCAC

__all__ = ["FACTORIAL_CONDITIONS", "PatchEncoder", "SingleChannelTransferCAC"]
