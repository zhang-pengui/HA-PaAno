"""Historical R2 routes, ported without merging constructor paths.

One bind per fresh condition/seed interpreter. No model code is defined here.
Source: revision_experiments/apin_r2_formal_tuning/scripts/routes.py.
"""
from __future__ import annotations
import importlib
import os

CONDITIONS = ("B", "J", "L", "C", "JL", "JC", "LC", "JLC")
ROUTES = {
    ("U", "B"): ("model_baseline", "train_baseline", None),
    ("U", "J"): ("model_baseline", "urpa_transform_train", None),
    ("U", "L"): ("model_FINAL_HA_PaAno", "train_baseline", None),
    ("U", "C"): ("cross_branch_model", "train_baseline", "u_cac"),
    ("U", "JL"): ("sensitivity_model_u", "sensitivity_train_u", None),
    ("U", "JC"): ("factorial_model", "cross_branch_train", "u_jitter_cac"),
    ("U", "LC"): ("factorial_model", "train_baseline", "u_lsdp_cac"),
    ("U", "JLC"): ("cross_branch_model", "cross_branch_train", "u_jitter_lsdp_cac"),
    ("M", "B"): ("model_baseline", "train_baseline", None),
    ("M", "J"): ("factorial_model", "cross_branch_train", "m_jitter"),
    ("M", "L"): ("factorial_model", "train_baseline", "m_lsdp"),
    ("M", "C"): ("model_FINAL_HA_PaAno_CANONICAL", "train_FINAL_M_branch", None),
    ("M", "JL"): ("cross_branch_model", "cross_branch_train", "m_jitter_lsdp"),
    ("M", "JC"): ("factorial_model", "cross_branch_train", "m_jitter_cac"),
    ("M", "LC"): ("factorial_model", "train_baseline", "m_lsdp_cac"),
    ("M", "JLC"): ("cross_branch_model", "cross_branch_train", "m_jitter_lsdp_cac"),
}
_bound = None

def desired_env(track, condition):
    _, _, cross = ROUTES[(track, condition)]
    values = {"PAANO_SENS_JITTER_P": "0.50", "PAANO_SENS_NOISE_STD": "0.010",
              "PAANO_SENS_LSDP_ALPHA_INIT": "0.05", "PAANO_SENS_CAC_ALPHA_INIT": "0.10",
              "PAANO_URPA_TRANSFORM_VARIANT": "jitter_only"}
    if cross:
        values["PAANO_CROSS_CONDITION"] = cross
    return values

def clean_environment(env, track, condition):
    removed = {k: v for k, v in env.items() if k.startswith("PAANO_")}
    clean = {k: v for k, v in env.items() if not k.startswith("PAANO_")}
    clean.update(desired_env(track, condition))
    clean["PYTHONDONTWRITEBYTECODE"] = "1"
    return clean, removed

def bind(track, condition):
    global _bound
    if _bound is not None:
        raise RuntimeError("R2 permits one condition/seed binding per fresh interpreter")
    clean, _ = clean_environment(dict(os.environ), track, condition)
    for k in list(os.environ):
        if k.startswith("PAANO_"):
            del os.environ[k]
    os.environ.update({k: v for k, v in clean.items() if k.startswith("PAANO_")})
    model_name, train_name, _ = ROUTES[(track, condition)]
    model_module = importlib.import_module("._reference." + model_name, __package__)
    train_module = importlib.import_module("._reference." + train_name, __package__)
    _bound = (track, condition)
    return model_module, train_module, {"effective": desired_env(track, condition)}
