"""Fail-closed placeholders; only the public runner binds the frozen pathway."""

def PatchEncoder(*args, **kwargs):
    raise RuntimeError('Use scripts/run_experiment.py to bind the final U-JLC or M-C model.')

def train_model(*args, **kwargs):
    raise RuntimeError('Use scripts/run_experiment.py; unbound historical defaults are disabled.')
