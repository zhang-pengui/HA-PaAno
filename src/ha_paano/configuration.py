"""Standard-library-only configuration checks: no training or CUDA imports."""
from contextlib import contextmanager
from pathlib import Path
import csv
import hashlib
import json
import os
import re

ROOT = Path(__file__).resolve().parents[2]

def read_config(track):
    if track not in ('U', 'M'):
        raise ValueError('Track must be U or M')
    config = json.loads((ROOT/f'configs/final_{track}.json').read_text(encoding='utf-8'))
    common = {'patch_size':96, 'patch_stride':1, 'num_iters':100, 'batch_size':512,
              'lr':1e-4, 'use_revin':True, 'embedding_dim':64, 'optimizer':'AdamW',
              'weight_decay':1e-4, 'scheduler':'per_iteration_cosine_to_lr_div_10',
              'positive_radius':2, 'triplet_margin':0.1, 'memory_bank_ratio':0.1,
              'top_k':3, 'cac_alpha_init':0.1,
              'evaluation_scope':'full_series_including_training_prefix',
              'core_seeds':list(range(2000,2010)), 'track':track}
    specific = ({'pathway':'JLC', 'jitter_probability':0.5, 'jitter_noise_multiplier':0.01,
                 'pooling':'learnable_avg_max', 'lsdp_alpha_init':0.05,
                 'cac':'raw_statistics_no_relative_norm_C1_enabled'} if track == 'U' else
                {'pathway':'C', 'jitter_probability':0.0, 'jitter_noise_multiplier':0.0,
                 'pooling':'average', 'lsdp_alpha_init':None, 'cac':'raw_statistics_no_relative_norm'})
    if config != common | specific:
        raise ValueError('Frozen scientific configuration differs from the reviewed protocol')
    return config

def inspect_input(data_dir, track, scope='formal'):
    directory = Path(data_dir).resolve(strict=True)
    if not directory.is_dir():
        raise ValueError('Input must be a flat CSV directory')
    files = sorted(directory.glob('*.csv'), key=lambda path: path.name)
    if not files:
        raise ValueError('No CSV files found; datasets are not bundled')
    expected = json.loads((ROOT/f'configs/evaluation_{track}.json').read_text(encoding='utf-8'))
    if scope == 'formal' and [p.name for p in files] != expected['series']:
        raise ValueError(f'Formal file identity mismatch: expected the exact {expected["expected_series"]}-series list; use diagnostic scope only for non-benchmark smoke cases')
    records = []
    for path in files:
        boundary = re.search(r'_tr_(\d+)(?:_|\.)', path.name)
        if not boundary or int(boundary.group(1)) < 98:
            raise ValueError(f'Invalid or too-short training prefix in filename: {path.name}')
        with path.open(encoding='utf-8-sig', newline='') as handle:
            rows = csv.reader(handle)
            header = next(rows, [])
            first = next(rows, [])
        channels = len(header)-1
        if len(first) != len(header) or channels < 1:
            raise ValueError(f'Invalid header or first data row: {path.name}')
        if (track == 'U') != (channels == 1):
            raise ValueError(f'{track} track/channel mismatch: {path.name}, C={channels}')
        records.append({'file':path.name, 'channels':channels, 'training_prefix':int(boundary.group(1))})
    return directory, records

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

@contextmanager
def frozen_environment(track):
    values = {'PAANO_CROSS_CONDITION':'u_jitter_lsdp_cac', 'PAANO_SENS_JITTER_P':'0.50',
              'PAANO_SENS_NOISE_STD':'0.010', 'PAANO_INTERNAL_VARIANT':'no_relative_norm',
              'PAANO_INTERNAL_SMOKE':'0'}
    previous = {key:os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
