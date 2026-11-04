import os
import csv
import json
import numpy as np
import torch


def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def make_safe_name(name):
    base = os.path.basename(str(name))
    base = base.replace(".csv", "")
    base = base.replace("/", "_").replace("\\", "_")
    base = base.replace(":", "_").replace(" ", "_")
    return base


def to_numpy(x):
    if x is None:
        return None

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    return np.asarray(x)


def save_one_file_result(
    save_dir,
    file_name,
    raw_series,
    labels,
    scores,
    metrics,
    runtime,
    extra_info=None
):
    """
    保存单个 csv 文件的实验结果。
    后面画异常分数可视化图时主要用 raw_series、labels、scores。
    """

    ensure_dir(save_dir)

    safe_name = make_safe_name(file_name)
    save_path = os.path.join(save_dir, safe_name + ".npz")

    raw_series = to_numpy(raw_series)
    labels = to_numpy(labels)
    scores = to_numpy(scores)

    if labels is not None:
        labels = labels.reshape(-1)

    if scores is not None:
        scores = scores.reshape(-1)

    if metrics is None:
        metrics = {}

    if extra_info is None:
        extra_info = {}

    np.savez_compressed(
        save_path,
        file_name=str(file_name),
        raw_series=raw_series,
        labels=labels,
        scores=scores,
        metrics=json.dumps(metrics),
        runtime=float(runtime),
        extra_info=json.dumps(extra_info)
    )

    return save_path


def append_summary_csv(summary_path, row_dict):
    """
    每个文件一行，保存指标和 runtime。
    """
    ensure_dir(os.path.dirname(summary_path))

    file_exists = os.path.exists(summary_path)

    with open(summary_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row_dict)


def save_dataset_summary(exp_root, final_metrics, runtime_list):
    """
    保存整个数据集的平均结果。
    """
    ensure_dir(exp_root)

    runtime_arr = np.asarray(runtime_list, dtype=float)

    summary = {
        "final_metrics": final_metrics,
        "num_files": int(len(runtime_arr)),
        "total_runtime": float(runtime_arr.sum()) if len(runtime_arr) > 0 else 0.0,
        "avg_runtime_per_file": float(runtime_arr.mean()) if len(runtime_arr) > 0 else 0.0,
        "std_runtime_per_file": float(runtime_arr.std()) if len(runtime_arr) > 0 else 0.0
    }

    save_path = os.path.join(exp_root, "dataset_summary.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return save_path