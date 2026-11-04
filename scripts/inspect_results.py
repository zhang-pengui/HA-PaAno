"""Read stored metrics only. Does not fit, infer, recompute metrics or write files."""
from pathlib import Path
import argparse
import csv
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from ha_paano.configuration import sha256

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve(strict=True)
    receipt = json.loads((root/'run_receipt.json').read_text(encoding='utf-8'))
    summary = root/'results/summary.csv'
    with summary.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    expected = [row['file'] for row in receipt['series']]
    names = [row['file'] for row in rows]
    if len(set(names)) != len(names) or names != expected:
        raise ValueError('Stored summary has missing/duplicate/reordered rows; not a complete run')
    import numpy as np
    for row in rows:
        path = root/'results/npz'/(Path(row['file']).stem+'.npz')
        with np.load(path, allow_pickle=False) as data:
            scores, labels = data['scores'], data['labels']
            metrics = json.loads(str(data['metrics']))
            if scores.ndim != 1 or labels.ndim != 1 or scores.size != labels.size:
                raise ValueError(f'Score/label shape mismatch: {path.name}')
            if not np.isfinite(scores).all() or not np.isfinite(labels).all() or not all(np.isfinite(value) for value in metrics.values()):
                raise ValueError(f'Nonfinite result: {path.name}')
            mapping = {'VUS_PR':'VUS_PR','VUS_ROC':'VUS_ROC','Point_F1':'Point_F1','Range_F1':'Range_F1','AUC_PR':'AUC_PR','AUC_ROC':'AUC_ROC'}
            for column, key in mapping.items():
                if float(row[column]) != float(metrics[key]):
                    raise ValueError(f'Metric/CSV mismatch: {path.name}/{column}')
        print(json.dumps({'file':row['file'],'sha256':sha256(path),'metrics':metrics}))
    print(json.dumps({'status':receipt['status'], 'series':len(rows), 'evaluation_recomputed':False}))

if __name__ == '__main__':
    main()
