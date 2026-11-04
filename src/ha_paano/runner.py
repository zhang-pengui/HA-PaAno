"""Portable binding around the original joint fit/bank/score/evaluate pipeline."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import traceback

from .configuration import ROOT, frozen_environment, inspect_input, read_config, sha256

def plan_run(track, seed, data_dir, output_dir, scope='formal'):
    config = read_config(track)
    if seed not in config['core_seeds']:
        raise ValueError('This frozen release supports core seeds 2000 through 2009')
    directory, records = inspect_input(data_dir, track, scope)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError('Output directory already exists; overwrite/resume is intentionally unsupported')
    if output == directory or output in directory.parents or directory in output.parents:
        raise ValueError('Output and input directories must be disjoint')
    # Prevent writes to the source checkout when it is available on this machine.
    protected_source = os.environ.get('HA_PAANO_PROTECTED_SOURCE', str(ROOT.parent/'PaAno'))
    if protected_source:
        protected = Path(protected_source).resolve()
        if output == protected or protected in output.parents:
            raise ValueError('Output points into the protected research tree')
    return {'track':track, 'seed':seed, 'scope':scope, 'config':config,
            'data_dir':directory.as_posix(), 'output_dir':str(output),
            'task_count':len(records), 'series':records, 'training_started':False,
            'protocol_identity': 'filename_set_verified_content_hashes_recorded_on_execute'}

def execute_run(plan, device='cpu'):
    # CPU requests must not initialize a GPU through inherited runtime checks.
    if device == 'cpu':
        import sys
        if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
            raise RuntimeError('Run CPU jobs in a fresh process before CUDA initialization')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import torch
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no automatic fallback')
    from ._reference import pipeline
    from ._reference import cross_branch_model, cross_branch_train, cac_variants, train_FINAL_M_branch
    output = Path(plan['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    receipt_path = output/'run_receipt.json'
    receipt = dict(plan)
    receipt.update({'started_at':datetime.now(timezone.utc).isoformat(), 'device':device,
                    'status':'RUNNING', 'training_started':True,
                    'input_sha256':{row['file']:sha256(Path(plan['data_dir'])/row['file']) for row in plan['series']},
                    'config_sha256':sha256(ROOT/f'configs/final_{plan["track"]}.json')})
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    original = pipeline.PatchEncoder, pipeline.train_model
    try:
        with frozen_environment(plan['track']):
            if plan['track'] == 'U':
                pipeline.PatchEncoder, pipeline.train_model = cross_branch_model.PatchEncoder, cross_branch_train.train_model
            else:
                pipeline.PatchEncoder, pipeline.train_model = cac_variants.PatchEncoder, train_FINAL_M_branch.train_model
            config = plan['config']
            experiment = pipeline.AnomalyDetection(
                data_dir=plan['data_dir'], device=torch.device(device),
                patch_size=config['patch_size'], num_iters=config['num_iters'],
                lr=config['lr'], batch_size=config['batch_size'], random_seed=plan['seed'],
                use_revin=config['use_revin'], patch_stride=config['patch_stride'],
                save_results=True, result_dir=str(output), exp_name='results',
                save_cac_activity=False, max_files=-1)
            experiment.run()
        receipt['status'] = 'COMPLETE'
    except BaseException:
        receipt['status'] = 'FAILED'
        receipt['exception'] = traceback.format_exc()
        raise
    finally:
        pipeline.PatchEncoder, pipeline.train_model = original
        receipt['finished_at'] = datetime.now(timezone.utc).isoformat()
        receipt_path.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
