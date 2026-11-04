"""Accepted R2 Mean-only control; full CAC is unchanged when not installed."""
from __future__ import annotations
import hashlib
import pickle
import numpy as np
import torch

def rng_hash():
    import random
    state = [random.getstate(), np.random.get_state(), torch.get_rng_state().numpy().tobytes()]
    if torch.cuda.is_initialized():
        state += [x.cpu().numpy().tobytes() for x in torch.cuda.get_rng_state_all()]
    return hashlib.sha256(pickle.dumps(state, protocol=4)).hexdigest()

def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def install_mean_only(model):
    # Ported hook body; additionally fail closed on non-C multivariate models.
    from ._reference.model_FINAL_HA_PaAno_CANONICAL import PatchEncoder
    if (type(model) is not PatchEncoder or getattr(model, "is_univariate", False) or
            getattr(model, "channel_context", None) is None or
            getattr(model, "dual_pooling_head", None) is not None or
            getattr(model, "jitter_enabled", False) or
            getattr(model, "condition", None) not in (None, "C")):
        raise ValueError("Mean-only is supported only for the admitted M-C route")
    before = rng_hash()
    schema = list(model.state_dict())
    if hasattr(model, "_r2_mean_only_handle"):
        raise RuntimeError("Mean-only hook is already installed")

    def mask_before_mlp(module, inputs):
        (descriptor,) = inputs
        if descriptor.shape[-1] != 8:
            raise RuntimeError("CAC descriptor must have eight entries")
        mask = descriptor.new_tensor([1, 0, 0, 0, 1, 0, 0, 0])
        return (descriptor * mask,)

    model._r2_mean_only_handle = model.channel_context.context_mlp.register_forward_pre_hook(mask_before_mlp)
    if list(model.state_dict()) != schema or rng_hash() != before:
        model._r2_mean_only_handle.remove()
        del model._r2_mean_only_handle
        raise RuntimeError("Mean-only registration changed state schema or RNG")
    return {"keep_zero_based": [0, 4], "mask": [1, 0, 0, 0, 1, 0, 0, 0],
            "position": "full finite descriptor after raw statistics, immediately before MLP",
            "train_and_eval_same_hook": True, "new_parameters": 0, "RNG_unchanged": True}

def extract_lsdp(model):
    head = getattr(model, "dual_pooling_head", None)
    if head is None:
        return None
    before = rng_hash()
    raw = head.alpha_logit.detach().cpu().clone()
    effective = torch.sigmoid(head.alpha_logit.detach()).cpu().numpy().reshape(-1)
    if effective.size != 64 or not np.isfinite(effective).all():
        raise RuntimeError("Invalid LSDP coefficients")
    result = {"alpha_logit": raw.numpy().reshape(-1).tolist(),
              "mixing_weights": effective.tolist(),
              "capture_point": "best-restored full state, before memory bank",
              "sigmoid_compute_device": str(head.alpha_logit.device)}
    if before != rng_hash():
        raise RuntimeError("LSDP inspection changed RNG")
    return result
