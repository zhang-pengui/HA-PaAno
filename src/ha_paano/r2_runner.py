"""Portable serial R2 fit/bank/score runner; does not import the internal controller."""
from __future__ import annotations
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

from .configuration import ROOT, sha256
from .r2_configuration import plan_r2

def seed_series(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def construct_training_input(path, seed, device, model_module):
    """Keep the frozen worker's input/probe/constructor order, including unused stats."""
    import numpy as np
    from ._reference.utils.data_preprocess import load_and_split_data, PatchCreator, preprocess_to_patches, find_length_rank
    seed_series(seed)
    tr, tl, te, el = load_and_split_data(Path(path).as_posix())
    train = np.array(tr, dtype=np.float32)
    train_labels = np.array(tl, dtype=np.float32)
    test = np.array(te, dtype=np.float32)
    test_labels = np.array(el, dtype=np.float32)
    if len(train) < 96 or len(test) == 0 or train.shape[0] != train_labels.shape[0]:
        raise RuntimeError('Invalid R2 training prefix/suffix; do not skip the series')
    mean = np.mean(train, axis=0, keepdims=True).astype(np.float32)
    std = np.std(train, axis=0, keepdims=True).astype(np.float32)
    std = np.where(std == 0., 1e-8, std)
    full = np.concatenate([train, test], axis=0)
    labels = np.concatenate([train_labels, test_labels], axis=0)
    sliding_input = full.reshape(-1, 1) if full.ndim == 1 else full[:, 0].reshape(-1, 1)
    window = find_length_rank(sliding_input, rank=1)
    creator = PatchCreator(L=96, s=1, random_seed=seed)
    train_loader, test_loader, _ = creator.create_dataloaders(train, full, labels, batch_size=512)
    probe, indexes = next(iter(train_loader))
    channels = probe.shape[1]
    model = model_module.PatchEncoder(in_channels=channels, use_revin=True).to(device)
    patches = preprocess_to_patches(train, patch_size=96, stride=1)
    return {'model': model, 'train_loader': train_loader, 'test_loader': test_loader,
            'patches': patches, 'labels': labels, 'window': window, 'channels': channels,
            'probe_indexes': indexes, 'train_length': len(train), 'full_length': len(full)}

def call_original_trainer(train_module, context, device):
    return train_module.train_model(context['model'], context['train_loader'], context['patches'], device,
                                    num_iter=100, pretext_step=96, lr=1e-4, see_loss=False)

def _write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)

def execute_r2(plan, device='cpu'):
    # Re-read all identities immediately before execution; no forged plan/changed input.
    verified = plan_r2(plan['track'], plan['condition'], plan['seed'], plan['split'],
                       plan['data_dir'], plan['output_dir'], plan['mean_only'])
    if verified != plan:
        raise ValueError('R2 plan changed after preflight')
    if device not in ('cpu', 'cuda'):
        raise ValueError('Device must be cpu or cuda')
    if device == 'cpu':
        if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
            raise RuntimeError('CPU R2 requires a fresh process before CUDA initialization')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    output = Path(plan['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    # All generated caches stay under the requested fresh output, not source/data.
    for key, rel in [('MPLCONFIGDIR', '.cache/matplotlib'), ('TORCH_HOME', '.cache/torch'),
                     ('XDG_CACHE_HOME', '.cache/xdg')]:
        os.environ[key] = str(output/rel)
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    receipt = {**plan, 'device': device, 'status': 'RUNNING', 'training_started': False,
               'pid': os.getpid(), 'started_at': datetime.now(timezone.utc).isoformat(), 'completed_series': []}
    _write_json(output/'run_receipt.json', receipt)
    try:
        import torch
        import numpy as np
        from .r2_routes import bind
        from .r2_controls import install_mean_only, state_hash, extract_lsdp
        from ._reference.utils.utils import create_memory_bank
        from ._reference.utils.evaluation import calculate_anomaly_scores, distribute_patch_scores_to_points
        from ._reference.utils.metrics import get_metrics
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; no automatic fallback')
        model_module, train_module, env = bind(plan['track'], plan['condition'])
        receipt['route'] = {'model': model_module.__name__, 'trainer': train_module.__name__, **env}
        receipt['source_sha256'] = {
            str(Path(m.__file__).resolve().relative_to(ROOT)): sha256(Path(m.__file__))
            for m in list(sys.modules.values()) if getattr(m, '__file__', None)
            and Path(m.__file__).resolve().is_relative_to(ROOT/'src')}
        receipt['environment'] = {'python': sys.version, 'torch': torch.__version__,
            'threads': torch.get_num_threads(), 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'thread_environment': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'LOKY_MAX_CPU_COUNT', 'JOBLIB_MULTIPROCESSING')}}
        for member in plan['series']:
            data_path = Path(plan['data_dir'])/member['series']
            if sha256(data_path) != member['sha256']:
                raise RuntimeError('Input changed after preflight')
            started = time.perf_counter()
            context = construct_training_input(data_path, plan['seed'], torch.device(device), model_module)
            model = context['model']
            if (plan['track'] == 'U') != (context['channels'] == 1):
                raise RuntimeError('R2 model/data track mismatch')
            mask_info = install_mean_only(model) if plan['mean_only'] else None
            receipt['training_started'] = True
            receipt['current_series'] = member['series']
            _write_json(output/'run_receipt.json', receipt)
            call_original_trainer(train_module, context, torch.device(device))
            if not all(torch.isfinite(t).all().item() for t in model.state_dict().values()):
                raise RuntimeError('Nonfinite trained model state')
            weights = extract_lsdp(model)
            best_hash = state_hash(model)
            bank, _ = create_memory_bank(model, context['train_loader'], torch.device(device), num_cores=.1)
            patch_scores = calculate_anomaly_scores(model, context['test_loader'], bank, device=torch.device(device), top_k=3)
            scores = distribute_patch_scores_to_points(patch_scores, patch_size=96, num_points=len(context['labels']))
            if not np.isfinite(scores).all() or scores.shape != context['labels'].shape:
                raise RuntimeError('Nonfinite/misaligned scores')
            metrics = get_metrics(scores, context['labels'], slidingWindow=context['window'], pred=None, version='opt', thre=250)
            metrics = {k: float(v) for k, v in metrics.items()}
            if not all(np.isfinite(v) for v in metrics.values()):
                raise RuntimeError('Nonfinite metrics; no COMPLETE commit')
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            if sha256(data_path) != member['sha256']:
                raise RuntimeError('Input changed during execution')
            folder = output/'series'/Path(member['series']).stem
            folder.mkdir(parents=True, exist_ok=False)
            with (folder/'result.npz').open('xb') as f:
                np.savez_compressed(f, scores=scores, labels=context['labels'])
            torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        'checkpoint_kind': 'restored_original_best_loss_full_state'}, folder/'best_model.pt')
            saved = torch.load(folder/'best_model.pt', map_location='cpu', weights_only=True)['state_dict']
            if set(saved) != set(model.state_dict()) or any(not torch.equal(saved[k], v.detach().cpu()) for k, v in model.state_dict().items()):
                raise RuntimeError('Checkpoint CPU readback mismatch')
            _write_json(folder/'metrics.json', metrics)
            _write_json(folder/'metadata.json', {
                'run_id': plan['run_id'], 'series': member['series'], 'track': plan['track'], 'split': plan['split'],
                'seed': plan['seed'], 'condition': plan['condition_id'], 'mean_only': mask_info,
                'protocol_sha256': plan['config_sha256'], 'data_sha256': member['sha256'],
                'best_state_sha256': best_hash, 'runtime_seconds': elapsed,
                'score_length': len(scores), 'label_length': len(context['labels']),
                'channels': context['channels'], 'training_prefix': context['train_length'],
                'sliding_window': int(context['window']), 'checkpoint_kind': 'restored_original_best_loss_full_state',
                'evaluation_scope': 'full sequence including training prefix', 'scientific_identity': plan['scientific_identity']})
            if weights is not None:
                _write_json(folder/'lsdp_weights.json', weights)
            hashes = {p.name: sha256(p) for p in folder.iterdir() if p.is_file()}
            _write_json(folder/'COMPLETE.json', {'status': 'COMPLETE', 'sha256': hashes})
            receipt['completed_series'].append(member['series'])
            _write_json(output/'run_receipt.json', receipt)
            print(json.dumps({'run_id': plan['run_id'], 'series': member['series'], 'completed': len(receipt['completed_series']), 'total': plan['task_count']}), flush=True)
        receipt['status'] = 'COMPLETE'
    except BaseException:
        receipt['status'] = 'FAILED'
        receipt['exception'] = traceback.format_exc()
        raise
    finally:
        receipt['finished_at'] = datetime.now(timezone.utc).isoformat()
        _write_json(output/'run_receipt.json', receipt)
