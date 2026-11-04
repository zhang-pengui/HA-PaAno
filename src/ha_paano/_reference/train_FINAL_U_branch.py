import copy, math
from tqdm import tqdm
import torch
import torch.nn.functional as F
from .utils.utils import *


def _shift_1d_batch(x, max_shift=2):
    """
    只用于单变量增强。
    x: (B, C, L)
    """
    if max_shift <= 0:
        return x

    B, C, L = x.shape
    shifts = torch.randint(
        low=-max_shift,
        high=max_shift + 1,
        size=(B,),
        device=x.device
    )

    out = x.clone()

    for s in shifts.unique().tolist():
        s = int(s)
        idx = shifts == s

        if s == 0:
            continue

        if s > 0:
            out[idx, :, s:] = x[idx, :, :-s]
            out[idx, :, :s] = x[idx, :, :1].expand(-1, -1, s)
        else:
            ss = -s
            out[idx, :, :-ss] = x[idx, :, ss:]
            out[idx, :, -ss:] = x[idx, :, -1:].expand(-1, -1, ss)

    return out


def patch_augment_uonly(x):
    """
    E3-UOnly: Univariate-only Patch Augmentation

    核心逻辑：
    1. 如果是单变量 C=1：
       启用 E3-Full 增强。
       aug_prob  = 0.50
       noise_std = 0.01
       scale_std = 0.05
       shift_max = 2

    2. 如果是多变量 C>1：
       不做任何增强，直接返回原始 x。
       这样避免破坏多变量数据中的通道同步关系、离散状态值和相对幅值关系。
    """
    B, C, L = x.shape

    # 多变量：完全不增强，保持原始 PaAno 训练逻辑
    if C > 1:
        return x

    # 单变量：使用 E3-Full 增强
    device = x.device
    aug_prob = 0.50
    noise_std = 0.01
    scale_std = 0.05
    shift_max = 2

    mask = (torch.rand(B, 1, 1, device=device) < aug_prob).float()

    x_aug = x

    # 1. 幅值缩放
    if scale_std > 0:
        scale = 1.0 + scale_std * torch.randn(B, C, 1, device=device)
        x_scaled = x_aug * scale
        x_aug = mask * x_scaled + (1.0 - mask) * x_aug

    # 2. 小幅高斯噪声
    if noise_std > 0:
        local_std = x_aug.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-5)
        noise = noise_std * local_std * torch.randn_like(x_aug)
        x_noisy = x_aug + noise
        x_aug = mask * x_noisy + (1.0 - mask) * x_aug

    # 3. 轻微时间平移
    if shift_max > 0:
        x_shifted = _shift_1d_batch(x_aug, max_shift=shift_max)
        x_aug = mask * x_shifted + (1.0 - mask) * x_aug

    x_aug = torch.nan_to_num(x_aug, nan=0.0, posinf=0.0, neginf=0.0)
    return x_aug


def train_model(model, train_loader, train_patches, device, num_iter=200, pretext_step=64,
                lr=1e-4, see_loss=None, use_patch_aug=True):

    # fixed hyperparams in PaAno
    radius = 2
    lambda_weight = 1
    temperature = 1.0
    num_rand_patches = 5
    initial_lr = lr
    final_lr = lr / 10

    def cosine_annealed_lr(iteration):
        t = min(iteration, num_iter)
        cosine_factor = 0.5 * (1 + math.cos(math.pi * t / num_iter))
        return final_lr + (initial_lr - final_lr) * cosine_factor

    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    pos_weight = torch.tensor([1.0]).to(device)
    criterion_pretext = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    iteration_count = 0
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    print("    [Training Info]")
    pbar = tqdm(total=num_iter, desc="    >> Training", ncols=80)

    _offsets = torch.tensor([*range(-radius, 0), *range(1, radius + 1)], dtype=torch.long)

    while iteration_count < num_iter:
        for batch_data, batch_indexes in train_loader:
            if iteration_count >= num_iter:
                break

            iteration_count += 1

            # Update LR
            lr = cosine_annealed_lr(iteration_count)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            batch_data = batch_data.to(device, non_blocking=True)
            batch_indexes = batch_indexes.squeeze()

            anchors = batch_data
            M = batch_data.shape[0]
            mu = 10
            total_len = len(train_patches)

            # positives
            _cand = batch_indexes.view(-1, 1) + _offsets.view(1, -1)
            _valid = (_cand >= 0) & (_cand < total_len)
            _noise = torch.rand_like(_cand.float())
            _score = torch.where(_valid, _noise, torch.full_like(_noise, -1.0))
            _choice = _score.argmax(dim=1)
            _pos_idx = _cand.gather(1, _choice.view(-1, 1)).squeeze(1)

            _none_valid = _valid.sum(dim=1) == 0
            if _none_valid.any():
                _pos_idx[_none_valid] = batch_indexes[_none_valid]

            positives = torch.stack(
                [train_patches[i] for i in _pos_idx.tolist()],
                dim=0
            ).to(device, non_blocking=True)

            # E3-UOnly：单变量增强，多变量不增强
            if use_patch_aug:
                anchors = patch_augment_uonly(anchors)
                positives = patch_augment_uonly(positives)

            if iteration_count < (num_iter / 5):
                current_lambda_pretext = lambda_weight * (1 - (iteration_count / (num_iter / 5)))
            else:
                current_lambda_pretext = 0.0

            if current_lambda_pretext > 0.0:
                pretext_patches = []
                pretext_valid_mask = []

                _tgt = batch_indexes - pretext_step
                _pre_mask = (_tgt >= 0) & (_tgt < total_len)
                _tgt_clamped = _tgt.clamp(0, total_len - 1)

                for i in range(M):
                    if _pre_mask[i]:
                        pretext_patches.append(train_patches[_tgt_clamped[i].item()].unsqueeze(0))
                        pretext_valid_mask.append(True)
                    else:
                        pretext_patches.append(torch.zeros_like(train_patches[0].unsqueeze(0)))
                        pretext_valid_mask.append(False)

                pretext_patches = torch.cat(pretext_patches, dim=0).to(device, non_blocking=True)
                pretext_valid_mask = torch.tensor(pretext_valid_mask, dtype=torch.bool, device=device)

                if use_patch_aug:
                    pretext_patches = patch_augment_uonly(pretext_patches)

                all_patches = torch.cat([anchors, positives, pretext_patches], dim=0)
                all_embeddings = model.embedding(all_patches)

                h_anchors = all_embeddings[:M]
                h_pos = all_embeddings[M:2 * M]
                h_pretext = all_embeddings[2 * M:3 * M]

            else:
                pretext_patches = None
                pretext_valid_mask = None

                all_patches = torch.cat([anchors, positives], dim=0)
                all_embeddings = model.embedding(all_patches)

                h_anchors = all_embeddings[:M]
                h_pos = all_embeddings[M:2 * M]

            # Triplet Loss
            z_anchor = model.projection(h_anchors)
            z_pos = model.projection(h_pos)

            z_anchor = F.normalize(z_anchor, dim=1)
            z_pos = F.normalize(z_pos, dim=1)

            _sim_ap = (z_anchor @ z_pos.T) / temperature
            pos_sims = _sim_ap.diag()

            _sim_ap_f = _sim_ap.clone()
            _sim_ap_f.diagonal().fill_(+float('inf'))

            neg_dists = 1 - _sim_ap_f
            hard_neg_dists, _ = torch.max(neg_dists, dim=1)

            pos_dists = 1 - pos_sims
            triplet_loss = F.relu(pos_dists - hard_neg_dists + 0.1).mean() / mu

            # Pretext Task
            if current_lambda_pretext > 0.0:
                h_pre = h_pretext[pretext_valid_mask]
                h_anchor_pre = h_anchors[pretext_valid_mask]
                h_concat_pre = torch.cat([h_anchor_pre, h_pre], dim=1)

                all_indices = torch.arange(M, device=device)
                anchor_indices = all_indices.repeat_interleave(num_rand_patches)
                rand_offsets = torch.randint(1, M, (M * num_rand_patches,), device=device)
                unadj_indices = (anchor_indices + rand_offsets) % M

                h_unadj = h_anchors[unadj_indices]
                h_anchor_unadj = h_anchors.repeat_interleave(num_rand_patches, dim=0)
                h_concat_unadj = torch.cat([h_anchor_unadj, h_unadj], dim=1)

                all_pretext_features = torch.cat([h_concat_pre, h_concat_unadj], dim=0)
                all_pretext_labels = torch.cat([
                    torch.ones(h_concat_pre.size(0), device=device),
                    torch.zeros(h_concat_unadj.size(0), device=device)
                ])

                pretext_outputs = model.classification_head(all_pretext_features).squeeze(1)
                pretext_loss_all = criterion_pretext(pretext_outputs, all_pretext_labels)

                loss_pre = pretext_loss_all[:h_concat_pre.size(0)].mean()
                loss_unadj = pretext_loss_all[h_concat_pre.size(0):].mean()
                pretext_loss = loss_pre + loss_unadj
            else:
                pretext_loss = torch.tensor(0.0, device=device)

            final_loss = triplet_loss + current_lambda_pretext * pretext_loss

            optimizer.zero_grad(set_to_none=True)
            final_loss.backward()
            optimizer.step()

            pbar.update(1)

            if final_loss.item() < best_loss:
                best_loss = final_loss.item()
                best_model_wts = copy.deepcopy(model.state_dict())

    pbar.close()
    model.load_state_dict(best_model_wts)

    # 不保存模型，避免额外文件
    # torch.save(model.state_dict(), 'best_trained_encoder.pth')