# utils/cac_activity_logger.py
import os
import numpy as np
import torch


def ensure_dir(path):
    """
    确保目录存在。
    """
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def make_safe_name(file_name):
    """
    把 csv 文件名转换成安全的 npz 文件名。
    """
    base = os.path.basename(str(file_name))
    base = base.replace(".csv", "")
    base = base.replace("/", "_").replace("\\", "_")
    base = base.replace(":", "_").replace(" ", "_")
    return base


def to_numpy(x):
    """
    Tensor / list / ndarray 统一转 numpy。
    """
    if x is None:
        return None

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    return np.asarray(x)


def get_core_model(model):
    """
    兼容 DataParallel / 普通模型。
    """
    return model.module if hasattr(model, "module") else model


def get_cac_context(model):
    """
    从 PatchEncoder 中取出 ChannelActivityContext。
    如果当前模型是单变量分支，或者没有 CAC，则返回 None。
    """
    core = get_core_model(model)

    if not hasattr(core, "channel_context"):
        return None

    return core.channel_context


def has_cac_context(model):
    """
    判断模型是否包含 CAC。
    """
    ctx = get_cac_context(model)
    return ctx is not None


def start_cac_activity_collection(model):
    """
    开始收集 CAC 内部通道活跃度。

    调用位置：
        在测试集 test patches forward 之前调用。

    返回：
        True  : 成功开启收集
        False : 当前模型没有 CAC，通常是单变量模型
    """
    ctx = get_cac_context(model)

    if ctx is None:
        return False

    if not hasattr(ctx, "clear_activity_cache"):
        raise AttributeError(
            "当前 channel_context 没有 clear_activity_cache()。"
            "请确认 model.py 中的 ChannelActivityContext 已经加入 CAC logger 代码。"
        )

    if not hasattr(ctx, "collect_activity"):
        raise AttributeError(
            "当前 channel_context 没有 collect_activity 属性。"
            "请确认 model.py 中的 ChannelActivityContext 已经加入 CAC logger 代码。"
        )

    ctx.clear_activity_cache()
    ctx.collect_activity = True

    return True


def stop_cac_activity_collection(model):
    """
    停止收集 CAC 内部通道活跃度，并返回拼接后的结果。

    返回：
        None:
            没有收集到数据，或当前模型无 CAC。

        dict:
            {
                "ch_std_rel":  ndarray, shape = (num_forward_patches, C),
                "ch_diff_rel": ndarray, shape = (num_forward_patches, C)
            }
    """
    ctx = get_cac_context(model)

    if ctx is None:
        return None

    if not hasattr(ctx, "collect_activity"):
        return None

    ctx.collect_activity = False

    if not hasattr(ctx, "get_activity_cache"):
        return None

    records = ctx.get_activity_cache()

    if records is None or len(records) == 0:
        return None

    std_list = []
    diff_list = []

    for r in records:
        if "ch_std_rel" not in r or "ch_diff_rel" not in r:
            continue

        std_list.append(r["ch_std_rel"])
        diff_list.append(r["ch_diff_rel"])

    if len(std_list) == 0 or len(diff_list) == 0:
        return None

    ch_std_rel = torch.cat(std_list, dim=0).detach().cpu().numpy()
    ch_diff_rel = torch.cat(diff_list, dim=0).detach().cpu().numpy()

    # 清空缓存，避免影响后续文件
    ctx.clear_activity_cache()

    return {
        "ch_std_rel": ch_std_rel,
        "ch_diff_rel": ch_diff_rel
    }


def build_patch_centers(num_patches, label_len, patch_len=None, patch_stride=1):
    """
    构造每个 patch 对应的时间中心点。

    对于你的 PaAno 流程，如果 test_patches 是 stride=1 滑窗，
    那第 i 个 patch 的中心位置通常是：

        center_i = i * patch_stride + patch_len // 2

    如果 patch_len 不知道，则退化为线性映射。
    """
    num_patches = int(num_patches)
    label_len = int(label_len)

    if num_patches <= 0:
        return np.asarray([], dtype=float)

    if label_len <= 0:
        return np.arange(num_patches, dtype=float)

    if patch_len is not None and int(patch_len) > 0:
        patch_len = int(patch_len)
        patch_stride = int(patch_stride)

        centers = np.arange(num_patches, dtype=float) * patch_stride + patch_len // 2
        centers = np.clip(centers, 0, label_len - 1)
        return centers

    # 兜底：不知道 patch_len 时，用线性映射保证可以画图
    return np.linspace(0, label_len - 1, num_patches)


def save_cac_activity_npz(
    save_path,
    file_name,
    ch_std_rel,
    ch_diff_rel,
    labels,
    raw_series=None,
    scores=None,
    patch_len=None,
    patch_stride=1,
    extra_info=None
):
    """
    保存严格版 CAC 内部通道活跃度。

    保存内容：
        file_name:
            当前 csv 文件名

        ch_std_rel:
            模型 forward 中实际计算得到的通道相对波动强度
            shape = (num_patches, C)

        ch_diff_rel:
            模型 forward 中实际计算得到的通道相对变化强度
            shape = (num_patches, C)

        centers:
            每个 patch 对应的时间中心点
            shape = (num_patches,)

        labels:
            point-level label
            shape = (T,)

        raw_series:
            原始或标准化后的时间序列，用于第一行可视化

        scores:
            HA-PaAno point-level anomaly score

        patch_len / patch_stride:
            用于解释 centers 的参数
    """
    ensure_dir(os.path.dirname(save_path))

    ch_std_rel = to_numpy(ch_std_rel)
    ch_diff_rel = to_numpy(ch_diff_rel)
    labels = to_numpy(labels)
    raw_series = to_numpy(raw_series)
    scores = to_numpy(scores)

    if labels is None:
        raise ValueError("labels 不能为空。")

    labels = labels.reshape(-1)

    if ch_std_rel is None or ch_diff_rel is None:
        raise ValueError("ch_std_rel 和 ch_diff_rel 不能为空。")

    if ch_std_rel.ndim != 2:
        raise ValueError(f"ch_std_rel 必须是 2D，当前 shape={ch_std_rel.shape}")

    if ch_diff_rel.ndim != 2:
        raise ValueError(f"ch_diff_rel 必须是 2D，当前 shape={ch_diff_rel.shape}")

    if ch_std_rel.shape != ch_diff_rel.shape:
        raise ValueError(
            f"ch_std_rel 和 ch_diff_rel 形状不一致："
            f"{ch_std_rel.shape} vs {ch_diff_rel.shape}"
        )

    num_patches = ch_std_rel.shape[0]
    centers = build_patch_centers(
        num_patches=num_patches,
        label_len=len(labels),
        patch_len=patch_len,
        patch_stride=patch_stride
    )

    if scores is not None:
        scores = scores.reshape(-1)

    if extra_info is None:
        extra_info = {}

    # np.savez_compressed 对 dict 不如 json 稳定；
    # 这里保存为字符串，后续需要时再解析。
    extra_info_str = str(extra_info)

    np.savez_compressed(
        save_path,
        file_name=str(file_name),
        ch_std_rel=ch_std_rel,
        ch_diff_rel=ch_diff_rel,
        centers=centers,
        labels=labels,
        raw_series=raw_series if raw_series is not None else np.array([]),
        scores=scores if scores is not None else np.array([]),
        patch_len=-1 if patch_len is None else int(patch_len),
        patch_stride=int(patch_stride),
        extra_info=extra_info_str
    )

    return save_path


def save_cac_activity_for_file(
    save_root,
    file_name,
    cac_activity,
    labels,
    raw_series=None,
    scores=None,
    patch_len=None,
    patch_stride=1,
    extra_info=None
):
    """
    更方便的封装：根据 file_name 自动生成保存路径。

    保存路径：
        save_root / <safe_file_name>_cac_internal.npz
    """
    if cac_activity is None:
        return None

    if "ch_std_rel" not in cac_activity or "ch_diff_rel" not in cac_activity:
        raise ValueError("cac_activity 必须包含 ch_std_rel 和 ch_diff_rel。")

    ensure_dir(save_root)

    safe_name = make_safe_name(file_name)
    save_path = os.path.join(save_root, safe_name + "_cac_internal.npz")

    return save_cac_activity_npz(
        save_path=save_path,
        file_name=file_name,
        ch_std_rel=cac_activity["ch_std_rel"],
        ch_diff_rel=cac_activity["ch_diff_rel"],
        labels=labels,
        raw_series=raw_series,
        scores=scores,
        patch_len=patch_len,
        patch_stride=patch_stride,
        extra_info=extra_info
    )