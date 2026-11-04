"""R2 exact-split preflight; --execute runs ONE condition/seed in a fresh child."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from ha_paano.r2_configuration import expanded_jobs, plan_r2
from ha_paano.r2_routes import CONDITIONS, clean_environment

def child_environment(plan, device):
    env, _ = clean_environment(dict(os.environ), plan['track'], plan['condition'])
    if device == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = '-1'
    if plan['execution']:
        policies = plan['execution']['thread_environment_by_seed']
        keys = {k for v in policies.values() for k in v}
        for key in keys:
            env.pop(key, None)
        env.update(policies[str(plan['seed'])])
    return env

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track', choices=('U','M'))
    parser.add_argument('--condition', choices=CONDITIONS)
    parser.add_argument('--seed', type=int, choices=(2000,2001,2002))
    parser.add_argument('--split', choices=('tuning','evaluation'))
    parser.add_argument('--mean-only', action='store_true', help='Only M-C; fixed 8D mask throughout fit/bank/query')
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--list-jobs', action='store_true', help='Expand the 51 tuning + 3 Eva Mean-only identities without reading data or training')
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.list_jobs:
        if args.execute or args._worker:
            parser.error('--list-jobs cannot execute training')
        jobs = expanded_jobs()
        print(json.dumps({'jobs': jobs, 'tuning_jobs': 51, 'tuning_fits': 1692, 'eva_jobs': 3, 'eva_fits': 540, 'training_started': False}, indent=2))
        return 0
    for field in ('track','condition','seed','split','data_dir','output_dir'):
        if getattr(args, field) is None:
            parser.error('--' + field.replace('_','-') + ' is required')
    if args._worker and not args.execute:
        parser.error('Internal worker requires explicit --execute')
    plan = plan_r2(args.track, args.condition, args.seed, args.split, args.data_dir, args.output_dir, args.mean_only)
    if not args._worker:
        print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return 0
    if args._worker:
        # Apply the archived thread policy before any Torch import even when the
        # internal switch is invoked directly; it is not a policy bypass.
        environment = child_environment(plan, args.device)
        os.environ.clear()
        os.environ.update(environment)
        from ha_paano.r2_runner import execute_r2
        execute_r2(plan, args.device)
        return 0
    # Never batch conditions in the current interpreter; no implicit retry or queue.
    child_args = list(sys.argv[1:] if argv is None else argv) + ['--_worker']
    completed = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), *child_args],
                               env=child_environment(plan, args.device), check=False)
    return completed.returncode

if __name__ == '__main__':
    raise SystemExit(main())
