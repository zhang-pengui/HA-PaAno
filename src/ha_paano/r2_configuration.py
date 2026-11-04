"""R2 protocol/identity checks. Standard library only; dry runs do not train."""
from pathlib import Path
import csv
import json
import re
from .configuration import ROOT, sha256
from .r2_routes import CONDITIONS, ROUTES

CONFIG_SHA256 = {
    "r2_evaluation_M.json": "e9764924ef7886518210b2288fc1ef8f87e622b906e3f0a43c1c0085ba131641",
    "r2_eva_execution.json": "948341ef787fe1349ea7a19ba33fded72ca0a80a8bbefebd48ca2c315521eb15",
    "r2_protocol.json": "44300fb4a6d00ad6d886c234ac528cb3d1db9de099d9a8b3ee3fcde26f5f943f",
    "r2_tuning_M.json": "e710623c1884e172fc1715495fc69fdc07b5216a0f06b7cefb0c1876584d8444",
    "r2_tuning_U.json": "3490f2d159d5768565a00a65a1b4a02500bd78affcf85073e4d083b75554b668"
}

def read_frozen(name):
    path = ROOT/'configs'/name
    if name not in CONFIG_SHA256 or sha256(path) != CONFIG_SHA256[name]:
        raise ValueError("R2 frozen configuration identity mismatch: " + name)
    return json.loads(path.read_text(encoding='utf-8'))

def validate_request(track, condition, seed, split, mean_only=False):
    if (track, condition) not in ROUTES or seed not in (2000, 2001, 2002):
        raise ValueError("R2 supports the admitted U/M routes and seeds 2000–2002 only")
    if split not in ('tuning', 'evaluation'):
        raise ValueError("Unknown R2 split")
    if mean_only and (track != 'M' or condition != 'C'):
        raise ValueError("Mean-only requires M-C")
    if split == 'evaluation' and not (track == 'M' and condition == 'C' and mean_only):
        raise ValueError("R2 evaluation is limited to M Eva-180 Mean-only; use the original entry for final Eva runs")

def expanded_jobs():
    jobs = []
    for track, count in [('U', 48), ('M', 20)]:
        for condition in CONDITIONS:
            for seed in (2000, 2001, 2002):
                jobs.append({'track': track, 'split': 'tuning', 'condition': condition,
                             'seed': seed, 'mean_only': False, 'fits': count})
    for split, count in [('tuning', 20), ('evaluation', 180)]:
        for seed in (2000, 2001, 2002):
            jobs.append({'track': 'M', 'split': split, 'condition': 'C', 'seed': seed,
                         'mean_only': True, 'fits': count})
    return jobs

def plan_r2(track, condition, seed, split, data_dir, output_dir, mean_only=False):
    validate_request(track, condition, seed, split, mean_only)
    protocol = read_frozen('r2_protocol.json')
    manifest_name = f'r2_{split}_{track}.json'
    manifest = read_frozen(manifest_name)
    directory = Path(data_dir).resolve(strict=True)
    if not directory.is_dir():
        raise ValueError('Input must be a flat authorized CSV directory')
    actual = sorted(p.name for p in directory.glob('*.csv'))
    expected = [m['series'] for m in manifest['members']]
    if len(expected) != len(set(expected)) or actual != expected:
        raise ValueError(f'R2 {track}/{split} requires its exact {manifest["expected_series"]} members')
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError('Output exists; overwriting/resume is unsupported')
    if output == directory or output in directory.parents or directory in output.parents:
        raise ValueError('Input and output must be disjoint')
    # Keep the research protection used by the original entry; no hard-coded user path.
    import os
    protected = Path(os.environ.get('HA_PAANO_PROTECTED_SOURCE', str(ROOT.parent/'PaAno'))).resolve()
    if output == protected or protected in output.parents:
        raise ValueError('Output points into the protected research checkout')
    records = []
    for member in manifest['members']:
        path = directory/member['series']
        if path.stat().st_size != member['bytes'] or sha256(path) != member['sha256']:
            raise ValueError('Frozen data SHA256/size mismatch: ' + path.name)
        boundary = re.search(r'_tr_(\d+)(?:_|\.)', path.name)
        with path.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.reader(f)
            header, first = next(reader, []), next(reader, [])
        channels = len(header)-1
        if (not boundary or int(boundary.group(1)) < 98 or len(header) != len(first)
                or channels < 1 or ((track == 'U') != (channels == 1))):
            raise ValueError('Invalid R2 prefix/header/track: ' + path.name)
        if 'channels' in member and member['channels'] != channels:
            raise ValueError('Frozen channel count mismatch: ' + path.name)
        records.append({**member, 'channels': channels, 'training_prefix': int(boundary.group(1))})
    execution = read_frozen('r2_eva_execution.json') if split == 'evaluation' else None
    condition_id = 'MEAN_ONLY' if mean_only else condition
    return {'run_id': f'R2_{track}_{split}_{condition_id}_seed{seed}',
            'track': track, 'condition': condition, 'condition_id': condition_id,
            'seed': seed, 'split': split, 'mean_only': mean_only,
            'data_dir': directory.as_posix(), 'output_dir': str(output),
            'task_count': len(records), 'series': records, 'protocol': protocol,
            'config_sha256': CONFIG_SHA256['r2_protocol.json'],
            'data_manifest_sha256': CONFIG_SHA256[manifest_name],
            'execution': execution, 'training_started': False,
            'scientific_identity': 'A_HISTORICAL_PATHS_NOT_UNIFIED_RNG'}

