# model_FINAL_HA_PaAno.py
import math
import torch
import torch.nn as nn
from .utils.utils import RevIN1d


class DualPoolingEmbeddingHead(nn.Module):
    """
    E12-UOnly: Dual-Pooling Embedding Head

    仅用于 C == 1 的单变量序列。
    AvgPool 保留整体趋势，MaxPool 补充局部显著响应。
    """

    def __init__(self, embed_dim, init_alpha=0.05):
        super(DualPoolingEmbeddingHead, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.max_pool = nn.AdaptiveMaxPool1d(output_size=1)

        init_alpha = min(max(init_alpha, 1e-4), 1.0 - 1e-4)
        init_logit = math.log(init_alpha / (1.0 - init_alpha))

        self.alpha_logit = nn.Parameter(
            torch.full((1, embed_dim), init_logit, dtype=torch.float32)
        )

    def forward(self, x):
        """
        x: Conv feature map, shape = (B, D, L)
        """

        h_avg = self.avg_pool(x).flatten(start_dim=1)
        h_max = self.max_pool(x).flatten(start_dim=1)

        h_avg = torch.nan_to_num(h_avg, nan=0.0, posinf=0.0, neginf=0.0)
        h_max = torch.nan_to_num(h_max, nan=0.0, posinf=0.0, neginf=0.0)

        alpha = torch.sigmoid(self.alpha_logit)

        h = h_avg + alpha * (h_max - h_avg)
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

        return h


class ChannelActivityContext(nn.Module):
    """
    E4-v2: Channel Activity Context

    关键点：
        1. 使用 RevIN 前的 x_raw 提取通道统计；
        2. 对每个样本内部做相对归一化；
        3. alpha 初始约为 0.10；
        4. C == 1 时自动跳过，不影响 U；
        5. 可选导出模型 forward 中真实使用的 ch_std_rel / ch_diff_rel。
    """

    def __init__(self, embed_dim=64, context_hidden=64, init_alpha=0.10, eps=1e-6):
        super(ChannelActivityContext, self).__init__()

        self.embed_dim = embed_dim
        self.eps = eps

        # 输入统计特征维度固定为 8：
        # [std_mean, std_std, std_max, std_min,
        #  diff_mean, diff_std, diff_max, diff_min]
        self.context_mlp = nn.Sequential(
            nn.Linear(8, context_hidden),
            nn.LayerNorm(context_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(context_hidden, embed_dim)
        )

        # 残差注入强度，初始 alpha≈0.10
        init_alpha = min(max(init_alpha, 1e-4), 1.0 - 1e-4)
        init_logit = math.log(init_alpha / (1.0 - init_alpha))

        self.alpha_logit = nn.Parameter(
            torch.tensor(init_logit, dtype=torch.float32)
        )

        # ------------------------------------------------------------
        # 用于严格导出 CAC 内部通道活跃度。
        # 默认关闭，不影响正常训练、推理和实验结果。
        # 只有外部脚本显式设置 collect_activity=True 时才会记录。
        # ------------------------------------------------------------
        self.collect_activity = False
        self.activity_cache = []

    def clear_activity_cache(self):
        """
        清空 CAC 活跃度缓存。
        通常在测试某个文件之前调用。
        """
        self.activity_cache = []

    def get_activity_cache(self):
        """
        返回缓存的 CAC 活跃度。
        每个元素是一个 dict:
            {
                "ch_std_rel": Tensor, shape=(B, C),
                "ch_diff_rel": Tensor, shape=(B, C)
            }
        """
        return self.activity_cache

    def forward(self, h, x_raw):
        """
        h:
            CNN 得到的 patch embedding, shape = (B, D)

        x_raw:
            RevIN 前的输入 patch, shape = (B, C, L)
        """

        if x_raw.ndim != 3:
            return h

        B, C, L = x_raw.shape

        # 单变量不使用通道上下文，避免影响 U
        if C <= 1:
            return h

        # 1. 每个通道的波动强度 std
        ch_std = x_raw.std(dim=-1, unbiased=False)  # (B, C)

        # 2. 每个通道的一阶差分强度 diff
        if L > 1:
            ch_diff = torch.mean(
                torch.abs(x_raw[:, :, 1:] - x_raw[:, :, :-1]),
                dim=-1
            )
        else:
            ch_diff = torch.zeros_like(ch_std)

        ch_std = torch.nan_to_num(ch_std, nan=0.0, posinf=0.0, neginf=0.0)
        ch_diff = torch.nan_to_num(ch_diff, nan=0.0, posinf=0.0, neginf=0.0)

        # 3. 每个样本内部做相对归一化
        std_mean_per_sample = ch_std.mean(dim=1, keepdim=True)
        diff_mean_per_sample = ch_diff.mean(dim=1, keepdim=True)

        ch_std_rel = ch_std / (std_mean_per_sample + self.eps)
        ch_diff_rel = ch_diff / (diff_mean_per_sample + self.eps)

        ch_std_rel = torch.nan_to_num(
            ch_std_rel,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        ch_diff_rel = torch.nan_to_num(
            ch_diff_rel,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # ------------------------------------------------------------
        # 严格记录模型 forward 中实际使用的 CAC 通道活跃度。
        # 注意：
        #   1. 这里记录的是 RevIN 前 x_raw 计算得到的 ch_std_rel/ch_diff_rel；
        #   2. 只在 collect_activity=True 时记录；
        #   3. detach().cpu() 确保不影响计算图，也避免占用 GPU 显存。
        # ------------------------------------------------------------
        if self.collect_activity:
            self.activity_cache.append({
                "ch_std_rel": ch_std_rel.detach().cpu(),
                "ch_diff_rel": ch_diff_rel.detach().cpu()
            })

        # 4. 提取固定维度统计上下文
        std_mean = ch_std_rel.mean(dim=1)
        std_std = ch_std_rel.std(dim=1, unbiased=False)
        std_max = ch_std_rel.max(dim=1).values
        std_min = ch_std_rel.min(dim=1).values

        diff_mean = ch_diff_rel.mean(dim=1)
        diff_std = ch_diff_rel.std(dim=1, unbiased=False)
        diff_max = ch_diff_rel.max(dim=1).values
        diff_min = ch_diff_rel.min(dim=1).values

        stats = torch.stack(
            [
                std_mean,
                std_std,
                std_max,
                std_min,
                diff_mean,
                diff_std,
                diff_max,
                diff_min
            ],
            dim=1
        )  # (B, 8)

        stats = torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)

        # 5. 映射成 embedding context
        context = self.context_mlp(stats)
        context = torch.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)

        # 6. 残差注入
        alpha = torch.sigmoid(self.alpha_logit)

        h_new = h + alpha * context
        h_new = torch.nan_to_num(h_new, nan=0.0, posinf=0.0, neginf=0.0)

        return h_new


class PatchEncoder(nn.Module):
    """
    Adaptive PaAno PatchEncoder

    C == 1:
        使用 E12-UOnly Dual-Pooling Embedding Head

    C > 1:
        使用 E4-v2 Channel Activity Context

    注意：
        M 分支的 ChannelActivityContext 必须使用 RevIN 前的 x_raw。
    """

    def __init__(
        self,
        in_channels=1,
        projection_dim=256,
        layers=[128, 256, 128, 64],
        kss=[7, 5, 3, 3],
        use_revin: bool = True,
        revin_affine: bool = False,
        revin_eps: float = 1e-5,
        revin_min_sigma: float = 1e-5
    ):
        super(PatchEncoder, self).__init__()

        self.in_channels = in_channels
        self.layers = layers
        self.kss = kss
        self.projection_dim = projection_dim

        # 根据输入通道数判断数据类型
        self.is_univariate = in_channels == 1
        self.is_multivariate = in_channels > 1

        # RevIN
        self.revin = None
        if use_revin:
            self.revin = RevIN1d(
                num_channels=in_channels,
                eps=revin_eps,
                min_sigma=revin_min_sigma,
                affine=revin_affine
            )

        # 原始 PaAno Conv blocks
        self.convblocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    layers[i - 1] if i > 0 else in_channels,
                    self.layers[i],
                    kernel_size=self.kss[i],
                    stride=1,
                    padding=self.kss[i] // 2,
                    bias=False
                ),
                nn.BatchNorm1d(self.layers[i]),
                nn.ReLU(inplace=True)
            )
            for i in range(len(self.layers))
        ])

        # 原始 AvgPool
        self.fc_embedding = nn.AdaptiveAvgPool1d(output_size=1)
        self.gap = nn.AdaptiveAvgPool1d(output_size=1)

        # C == 1：启用 E12-UOnly
        self.dual_pooling_head = None
        if self.is_univariate:
            self.dual_pooling_head = DualPoolingEmbeddingHead(
                embed_dim=self.layers[-1],
                init_alpha=0.05
            )

        # C > 1：启用 E4-v2 Channel Activity Context
        self.channel_context = None
        if self.is_multivariate:
            self.channel_context = ChannelActivityContext(
                embed_dim=self.layers[-1],
                context_hidden=64,
                init_alpha=0.10
            )

        # 原始 projection head
        self.projection_head = nn.Sequential(
            nn.Linear(self.layers[-1], self.projection_dim),
            nn.ReLU(),
            nn.Linear(self.projection_dim, self.projection_dim)
        )

        # 原始 classification head
        self.classification_head = nn.Linear(self.layers[-1] * 2, 1)

    def forward(self, x, return_embedding=False, return_projection=False):

        # 关键：保存 RevIN 前输入，用于 E4-v2 通道上下文
        x_raw = x

        if self.revin is not None:
            x = self.revin.norm(x)

        for block in self.convblocks:
            x = block(x)

        # C == 1：使用 E12 Dual-Pooling
        if self.dual_pooling_head is not None:
            h = self.dual_pooling_head(x)

        # C > 1：使用原始 AvgPool
        else:
            h = self.fc_embedding(x).flatten(start_dim=1)

        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

        # C > 1：使用 E4-v2 Channel Activity Context
        if self.channel_context is not None:
            h = self.channel_context(h, x_raw)

        if return_embedding:
            return h

        if return_projection:
            return self.projection_head(h)

        raise ValueError("The forward method is not designed to handle classification directly.")

    def embedding(self, x):
        return self.forward(x, return_embedding=True)

    def projection(self, h):
        return self.projection_head(h)