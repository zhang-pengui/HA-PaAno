"""Default: read-only dry run. --execute explicitly fits and evaluates one seed."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from ha_paano.runner import execute_run, plan_run

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track', required=True, choices=('U','M'))
    parser.add_argument('--seed', required=True, type=int, choices=range(2000,2010), metavar='2000..2009')
    parser.add_argument('--data-dir', required=True, type=Path, help='Flat, authorized evaluation CSV directory')
    parser.add_argument('--output-dir', required=True, type=Path, help='Fresh directory, never an existing run')
    parser.add_argument('--scope', choices=('formal','diagnostic'), default='formal', help='Formal checks exact frozen filenames; diagnostic outputs are not benchmark evidence')
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    parser.add_argument('--execute', action='store_true', help='Actually fit then score/evaluate; omission does not train or write files')
    args = parser.parse_args()
    plan = plan_run(args.track, args.seed, args.data_dir, args.output_dir, args.scope)
    print(json.dumps(plan, indent=2), flush=True)
    if args.execute:
        execute_run(plan, args.device)

if __name__ == '__main__':
    main()
