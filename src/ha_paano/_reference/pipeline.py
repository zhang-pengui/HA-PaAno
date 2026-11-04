import os
import time
import random
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from sklearn.exceptions import UndefinedMetricWarning

from ..binding import PatchEncoder
from ..binding import train_model
from .utils.data_preprocess import *
from .utils.utils import *
from .utils.evaluation import *
from .utils.metrics import get_metrics

from .utils.result_saver import (
    ensure_dir,
    save_one_file_result,
    append_summary_csv,
    save_dataset_summary
)

from .utils.cac_activity_logger import (
    start_cac_activity_collection,
    stop_cac_activity_collection,
    save_cac_activity_for_file
)

warnings.simplefilter("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class AnomalyDetection:
    def __init__(
        self,
        data_dir,
        output_dir=None,
        patch_size=64,
        num_iters=None,
        lr=1e-4,
        batch_size=512,
        random_seed=2000,
        device=None,
        see_loss=False,
        use_revin=False,
        save_results=False,
        result_dir="saved_results",
        exp_name="default_exp",
        max_files=-1,
        save_cac_activity=False,
        patch_stride=1
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.patch_size = patch_size
        self.num_iters = num_iters
        self.lr = lr
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.see_loss = see_loss
        self.use_revin = use_revin

        # 新增：保存结果和 runtime
        self.save_results = save_results
        self.result_dir = result_dir
        self.exp_name = exp_name
        self.max_files = max_files

        # 新增：是否保存严格版 CAC 内部通道活跃度
        self.save_cac_activity = save_cac_activity
        self.patch_stride = patch_stride

    def run(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)

        self.dis_aurocs = []
        self.dis_auprcs = []
        self.dis_vuspr = []
        self.dis_vusroc = []
        self.dis_f1 = []
        self.dis_Rfl = []
        self.results_by_category = defaultdict(list)
        self.summary_rows = []

        # 新增：runtime 记录
        self.runtime_list = []

        total_csv_files = len([f for f in os.listdir(self.data_dir) if f.endswith(".csv")])

        if self.max_files > 0:
            run_csv_files = min(total_csv_files, self.max_files)
        else:
            run_csv_files = total_csv_files

        print(f"PaAno is running... (found {total_csv_files} files, will run {run_csv_files} files)")

        # 新增：保存目录
        # save_results 用于保存原来的 npz/summary；
        # save_cac_activity 用于额外保存严格版 CAC 内部通道活跃度。
        if self.save_results or self.save_cac_activity:
            self.exp_root = os.path.join(self.result_dir, self.exp_name)
            ensure_dir(self.exp_root)

            if self.save_results:
                self.npz_dir = os.path.join(self.exp_root, "npz")
                self.summary_path = os.path.join(self.exp_root, "summary.csv")
                ensure_dir(self.npz_dir)
                print(f"[Save] Results will be saved to: {self.exp_root}")
            else:
                self.npz_dir = None
                self.summary_path = None

            if self.save_cac_activity:
                self.cac_activity_dir = os.path.join(self.exp_root, "cac_activity")
                ensure_dir(self.cac_activity_dir)
                print(f"[Save] CAC internal activity will be saved to: {self.cac_activity_dir}")
            else:
                self.cac_activity_dir = None

        else:
            self.exp_root = None
            self.npz_dir = None
            self.summary_path = None
            self.cac_activity_dir = None

        current_idx = 1

        # 只记录一次模型参数量，避免每个 csv 都打印
        param_info_recorded = False
        total_params_once = None
        trainable_params_once = None
        in_channels_once = None

        csv_files = [f for f in sorted(os.listdir(self.data_dir)) if f.endswith(".csv")]

        for file_idx, file_name in enumerate(csv_files):
            if self.max_files > 0 and file_idx >= self.max_files:
                break

            file_path = os.path.join(self.data_dir, file_name)

            print(f"\033[1m══ Running on file ({current_idx}/{run_csv_files})\033[0m : {file_name}")
            current_idx += 1

            # ============================================================
            # Runtime 开始计时
            # 注意：从这里开始，包含读取数据、构造 patch、训练、memory bank、
            #       anomaly score、指标计算。
            # ============================================================
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            file_start_time = time.perf_counter()

            # 1. 读取并预处理数据
            train_data, train_labels, test_data, test_labels = load_and_split_data(file_path)

            train_data = np.array(train_data, dtype=np.float32)
            train_labels = np.array(train_labels, dtype=np.float32)
            test_data = np.array(test_data, dtype=np.float32)
            test_labels = np.array(test_labels, dtype=np.float32)

            train_mean = np.mean(train_data, axis=0, keepdims=True).astype(np.float32)
            train_std = np.std(train_data, axis=0, keepdims=True).astype(np.float32)
            train_std = np.where(train_std == 0.0, 1e-8, train_std)

            if self.use_revin is False:
                train_data = (train_data - train_mean) / train_std
                test_data = (test_data - train_mean) / train_std

            full_data = np.concatenate([train_data, test_data], axis=0)
            full_labels = np.concatenate([train_labels, test_labels], axis=0)

            if full_data.ndim == 1:
                sliding_input = full_data.reshape(-1, 1)
            else:
                sliding_input = full_data[:, 0].reshape(-1, 1)

            slidingWindow = find_length_rank(sliding_input, rank=1)

            # 2. 构造 patch loader
            patch_creator = PatchCreator(L=self.patch_size, s=1, random_seed=self.random_seed)

            train_loader, test_loader, true_test_labels = patch_creator.create_dataloaders(
                train_data,
                full_data,
                full_labels,
                batch_size=self.batch_size
            )

            xb, _ = next(iter(train_loader))
            C = xb.shape[1]
            print(f"[init] inferred in_channels = {C}")

            # 3. 建立模型
            model = PatchEncoder(in_channels=C, use_revin=self.use_revin).to(self.device)

            if not param_info_recorded:
                total_params_once = sum(p.numel() for p in model.parameters())
                trainable_params_once = sum(p.numel() for p in model.parameters() if p.requires_grad)
                in_channels_once = C
                param_info_recorded = True

            # 4. 训练模型
            train_model(
                model,
                train_loader,
                preprocess_to_patches(train_data, patch_size=self.patch_size, stride=1),
                self.device,
                num_iter=self.num_iters,
                pretext_step=self.patch_size,
                lr=self.lr,
                see_loss=self.see_loss
            )

            # 注意：
            # 原来这里有 torch.save + torch.load:
            # model_path = 'trained_encoder.pth'
            # torch.save(model.state_dict(), model_path)
            # model.load_state_dict(torch.load(model_path))
            #
            # 现在去掉。因为刚训练好的 model 已经在内存里，
            # 保存再读取没有必要，而且会影响 RunTime 统计公平性。

            # 5. 构建 memory bank
            memory_bank, _ = create_memory_bank(
                model,
                train_loader,
                self.device,
                num_cores=0.1
            )

            # ============================================================
            # CAC internal activity collection
            # 注意：
            #   1. 必须放在 memory bank 构建之后；
            #   2. 必须放在 test_loader 推理之前；
            #   3. 这样收集到的才是测试 patch anomaly scoring 阶段
            #      模型 forward 中真实使用的 ch_std_rel / ch_diff_rel。
            # ============================================================
            cac_collect_enabled = False

            if self.save_cac_activity and getattr(model, "is_multivariate", False):
                cac_collect_enabled = start_cac_activity_collection(model)
                if cac_collect_enabled:
                    print("    [CAC Logger] Started collecting internal CAC activity.")

            # 6. 计算 patch anomaly score
            all_scores = calculate_anomaly_scores(
                model,
                test_loader,
                memory_bank,
                top_k=3,
                device=self.device
            )

            # 7. patch score 分配到 point score
            dist_scores = distribute_patch_scores_to_points(
                all_scores,
                patch_size=self.patch_size,
                num_points=len(full_labels)
            )

            # ============================================================
            # 保存严格版 CAC 内部通道活跃度
            # 这里放在 dist_scores 之后，是因为保存时需要 point-level score。
            # ============================================================
            if self.save_cac_activity and cac_collect_enabled:
                cac_activity = stop_cac_activity_collection(model)

                if cac_activity is not None:
                    cac_save_path = save_cac_activity_for_file(
                        save_root=self.cac_activity_dir,
                        file_name=file_name,
                        cac_activity=cac_activity,
                        labels=full_labels,
                        raw_series=full_data,
                        scores=dist_scores,
                        patch_len=self.patch_size,
                        patch_stride=self.patch_stride,
                        extra_info={
                            "data_dir": self.data_dir,
                            "exp_name": self.exp_name,
                            "patch_size": self.patch_size,
                            "patch_stride": self.patch_stride,
                            "in_channels": int(C),
                            "note": "Internal CAC activity collected during test anomaly scoring."
                        }
                    )

                    print(f"    [CAC Logger] Saved internal CAC activity to {cac_save_path}")
                else:
                    print("    [CAC Logger] No CAC activity was collected.")

            print(f"    >> Anomaly detection completed. Calculating the score...")

            # 8. 计算指标
            results = get_metrics(
                dist_scores,
                full_labels,
                slidingWindow=slidingWindow,
                pred=None,
                version="opt",
                thre=250
            )

            # ============================================================
            # Runtime 停止计时
            # 注意：保存 npz 和 csv 的时间不算进 RunTime。
            # ============================================================
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            file_end_time = time.perf_counter()
            file_runtime = file_end_time - file_start_time
            self.runtime_list.append(file_runtime)

            print(f"    [Runtime] File runtime: {file_runtime:.4f} s")

            # 9. 收集指标
            self.dis_aurocs.append(results["AUC-ROC"])
            self.dis_auprcs.append(results["AUC-PR"])
            self.dis_vuspr.append(results["VUS-PR"])
            self.dis_vusroc.append(results["VUS-ROC"])
            self.dis_f1.append(results["Standard-F1"])
            self.dis_Rfl.append(results["R-based-F1"])

            # 文件名类别，保留原逻辑；增加保护，避免文件名不符合格式时报错
            parts = file_name.split("_")
            if len(parts) > 1:
                category = parts[1]
            else:
                category = "Unknown"

            self.results_by_category[category].append(results)

            print(f"    [Anomaly Detection Results]")
            print(
                f"    >> AUC-ROC: {results['AUC-ROC']:.4f}, "
                f"AUC-PR: {results['AUC-PR']:.4f}, "
                f"VUS-PR: {results['VUS-PR']:.4f}, "
                f"VUS-ROC: {results['VUS-ROC']:.4f}, "
                f"BestF1: {results['Standard-F1']:.4f}, "
                f"RangeF1: {results['R-based-F1']:.4f}"
            )

            metrics_dict = {
                "AUC_ROC": float(results["AUC-ROC"]),
                "AUC_PR": float(results["AUC-PR"]),
                "VUS_PR": float(results["VUS-PR"]),
                "VUS_ROC": float(results["VUS-ROC"]),
                "Point_F1": float(results["Standard-F1"]),
                "Range_F1": float(results["R-based-F1"])
            }

            # ============================================================
            # 新增：保存 npz + summary.csv
            # ============================================================
            if self.save_results:
                save_path = save_one_file_result(
                    save_dir=self.npz_dir,
                    file_name=file_name,
                    raw_series=full_data,
                    labels=full_labels,
                    scores=dist_scores,
                    metrics=metrics_dict,
                    runtime=file_runtime,
                    extra_info={
                        "data_dir": self.data_dir,
                        "patch_size": self.patch_size,
                        "num_iters": self.num_iters,
                        "batch_size": self.batch_size,
                        "lr": self.lr,
                        "seed": self.random_seed,
                        "use_revin": bool(self.use_revin),
                        "in_channels": int(C),
                        "total_params": int(total_params_once) if total_params_once is not None else None,
                        "trainable_params": int(trainable_params_once) if trainable_params_once is not None else None
                    }
                )

                append_summary_csv(
                    self.summary_path,
                    {
                        "file": file_name,
                        "Category": category,
                        "runtime": file_runtime,
                        "AUC_ROC": metrics_dict["AUC_ROC"],
                        "AUC_PR": metrics_dict["AUC_PR"],
                        "VUS_PR": metrics_dict["VUS_PR"],
                        "VUS_ROC": metrics_dict["VUS_ROC"],
                        "Point_F1": metrics_dict["Point_F1"],
                        "Range_F1": metrics_dict["Range_F1"],
                        "npz_path": save_path
                    }
                )

                print(f"    >> Saved npz result to {save_path}")

            # ============================================================
            # 保留你原来的 output_dir 保存方式
            # 这个可以不用；如果你想只用新保存逻辑，运行时不要传 --output_dir 即可。
            # ============================================================
            if self.output_dir:
                row = {
                    "file": file_name,
                    "Category": category,
                    "AUC-ROC": results["AUC-ROC"],
                    "AUC-PR": results["AUC-PR"],
                    "VUS-PR": results["VUS-PR"],
                    "VUS-ROC": results["VUS-ROC"],
                    "BestF1": results["Standard-F1"],
                    "RangeF1": results["R-based-F1"],
                    "runtime": file_runtime
                }
                self.summary_rows.append(row)

                os.makedirs(self.output_dir, exist_ok=True)
                summary_path = os.path.join(self.output_dir, "summary_metrics.csv")
                write_header = not os.path.exists(summary_path)
                pd.DataFrame([row]).to_csv(summary_path, mode="a", header=write_header, index=False)

                scores_dir = os.path.join(self.output_dir, "Filewise_scores")
                os.makedirs(scores_dir, exist_ok=True)
                output_file_path = os.path.join(scores_dir, f"{file_name}_output.csv")

                df = pd.DataFrame({
                    "True Labels": full_labels,
                    "Anomaly scores": dist_scores,
                })
                df.to_csv(output_file_path, index=False)

                print(f"    >> Saved results to {output_file_path}")

        # ============================================================
        # 10. 数据集平均指标
        # ============================================================
        dist_auroc, dist_auprc, dist_vuspr, dist_vusroc, dist_F1, dist_RF1 = map(
            np.mean,
            [
                self.dis_aurocs,
                self.dis_auprcs,
                self.dis_vuspr,
                self.dis_vusroc,
                self.dis_f1,
                self.dis_Rfl
            ]
        )

        print(
            f"PaAno's Averaged Final Results: "
            f"AUROC={dist_auroc:.4f}, "
            f"AUPRC={dist_auprc:.4f}, "
            f"VUSPR={dist_vuspr:.4f}, "
            f"VUSROC={dist_vusroc:.4f}, "
            f"F1-Score={dist_F1:.4f}, "
            f"RangeF1 = {dist_RF1:.4f} "
        )

        if param_info_recorded:
            print(
                f"Model Info: in_channels={in_channels_once}, "
                f"Total params={total_params_once:,}, "
                f"Trainable params={trainable_params_once:,}"
            )

        # ============================================================
        # 11. 保存整个数据集 summary
        # ============================================================
        if self.save_results:
            final_metrics = {
                "AUROC": float(dist_auroc),
                "AUPRC": float(dist_auprc),
                "VUSPR": float(dist_vuspr),
                "VUSROC": float(dist_vusroc),
                "F1_Score": float(dist_F1),
                "RangeF1": float(dist_RF1)
            }

            dataset_summary_path = save_dataset_summary(
                exp_root=self.exp_root,
                final_metrics=final_metrics,
                runtime_list=self.runtime_list
            )

            total_runtime = float(np.sum(self.runtime_list)) if len(self.runtime_list) > 0 else 0.0
            avg_runtime = float(np.mean(self.runtime_list)) if len(self.runtime_list) > 0 else 0.0
            std_runtime = float(np.std(self.runtime_list)) if len(self.runtime_list) > 0 else 0.0

            print("=" * 60)
            print("[Runtime Summary]")
            print(f"Experiment: {self.exp_name}")
            print(f"Total runtime: {total_runtime:.4f} s")
            print(f"Average runtime per file: {avg_runtime:.4f} s")
            print(f"Std runtime per file: {std_runtime:.4f} s")
            print(f"Saved dataset summary to: {dataset_summary_path}")
            print("=" * 60)

        # ============================================================
        # 12. 保留你原来的 output_dir 汇总
        # ============================================================
        if self.output_dir and self.summary_rows:
            summary_df = pd.DataFrame(self.summary_rows)

            summary_df = pd.concat([
                summary_df,
                pd.DataFrame([{
                    "file": "AVERAGE",
                    "Category": "AVERAGE",
                    "AUC-ROC": float(dist_auroc),
                    "AUC-PR": float(dist_auprc),
                    "VUS-PR": float(dist_vuspr),
                    "VUS-ROC": float(dist_vusroc),
                    "BestF1": float(dist_F1),
                    "RangeF1": float(dist_RF1),
                    "runtime": float(np.mean(self.runtime_list)) if len(self.runtime_list) > 0 else 0.0
                }])
            ], ignore_index=True)

            summary_path = os.path.join(self.output_dir, "summary_metrics.csv")
            summary_df.to_csv(summary_path, index=False)

            cat_df = (
                summary_df[summary_df["file"] != "AVERAGE"]
                .groupby("Category")
                .agg(
                    N=("file", "count"),
                    AUC_ROC=("AUC-ROC", "mean"),
                    AUC_PR=("AUC-PR", "mean"),
                    VUS_PR=("VUS-PR", "mean"),
                    VUS_ROC=("VUS-ROC", "mean"),
                    BestF1=("BestF1", "mean"),
                    RangeF1=("RangeF1", "mean"),
                    runtime=("runtime", "mean"),
                )
                .reset_index()
            )

            cat_df.to_csv(os.path.join(self.output_dir, "categorical_metrics.csv"), index=False)
            print(f">> Saved summary metrics to {summary_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run PaAno Anomaly Detection")

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--num_iters", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--see_loss", dest="see_loss", action="store_true")
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--use_revin", action="store_true", help="Use RevIN")

    # 新增参数
    parser.add_argument("--save_results", action="store_true",
                        help="Save raw series, labels, anomaly scores, metrics and runtime.")

    parser.add_argument("--result_dir", type=str, default="saved_results",
                        help="Root directory for saved results.")

    parser.add_argument("--exp_name", type=str, default="default_exp",
                        help="Experiment name, such as U_base, U_final, M_base, M_final.")

    parser.add_argument("--max_files", type=int, default=-1,
                        help="Only run first N csv files for debugging. -1 means run all files.")

    parser.add_argument("--save_cac_activity", action="store_true",
                        help="Save internal CAC channel activity maps during test inference.")

    parser.add_argument("--patch_stride", type=int, default=1,
                        help="Patch stride used to map CAC patch activity to time index.")

    args = parser.parse_args()

    experiment = AnomalyDetection(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        num_iters=args.num_iters,
        batch_size=args.batch_size,
        lr=args.lr,
        see_loss=args.see_loss,
        random_seed=args.seed,
        use_revin=args.use_revin,
        save_results=args.save_results,
        result_dir=args.result_dir,
        exp_name=args.exp_name,
        max_files=args.max_files,
        save_cac_activity=args.save_cac_activity,
        patch_stride=args.patch_stride
    )

    experiment.run()