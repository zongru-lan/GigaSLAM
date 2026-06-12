"""
================================================================================
Metric depth recovery module for CASS-DEPTH (v4)
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_METRIC_MIN_DEPTH = 1e-3
DEFAULT_METRIC_MAX_DEPTH = 120.0

def sanitize_depth_tensor(depth, min_depth=DEFAULT_METRIC_MIN_DEPTH, max_depth=DEFAULT_METRIC_MAX_DEPTH):
    depth = torch.nan_to_num(depth, nan=min_depth, posinf=max_depth, neginf=min_depth)
    return torch.clamp(depth, min=min_depth, max=max_depth)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad, bias=False),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.GELU(),
        )
    def forward(self, x):
        return self.block(x)

class MetricDepthHead(nn.Module):
    def __init__(self, hidden_dim=32, max_residual=1.0,
                 min_depth=DEFAULT_METRIC_MIN_DEPTH,
                 max_depth=DEFAULT_METRIC_MAX_DEPTH):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_residual = max_residual
        self.min_depth = min_depth
        self.max_depth = max_depth

        self.stem = nn.Sequential(
            ConvBlock(4, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.residual_head = nn.Sequential(
            ConvBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    @staticmethod
    def _bound_relative_depth(relative_depth, q_low=0.02, q_high=0.98):
        d = relative_depth.unsqueeze(1)
        orig_dtype = d.dtype
        d_float = torch.nan_to_num(d.float(), nan=0.0, posinf=1e4, neginf=0.0)
        flat = d_float.flatten(2)
        ql = torch.quantile(flat, q_low, dim=2, keepdim=True).unsqueeze(-1)
        qh = torch.quantile(flat, q_high, dim=2, keepdim=True).unsqueeze(-1)
        denom = (qh - ql).clamp(min=1e-6)
        bounded = (d_float - ql) / denom
        bounded = torch.clamp(bounded, 0.0, 1.0)
        bounded = torch.nan_to_num(bounded, nan=0.0, posinf=1.0, neginf=0.0)
        return bounded.to(orig_dtype)

    def forward(self, relative_depth, rgb_norm):
        d_bounded = self._bound_relative_depth(relative_depth)
        x = torch.cat([rgb_norm, d_bounded], dim=1)
        feat = self.stem(x)

        pooled = self.global_pool(feat).flatten(1)
        scale_raw, shift_raw = self.global_mlp(pooled).chunk(2, dim=1)

        scale = 20.0 * torch.sigmoid(scale_raw) + 0.1
        shift = torch.clamp(shift_raw, min=-5.0, max=5.0)

        residual = self.residual_head(feat).squeeze(1)
        residual = self.max_residual * torch.tanh(residual)

        raw_metric = scale.view(-1, 1, 1) * d_bounded.squeeze(1) + shift.view(-1, 1, 1) + residual
        metric_depth = self.min_depth + (self.max_depth - self.min_depth) * torch.sigmoid(raw_metric)
        metric_depth = sanitize_depth_tensor(metric_depth, self.min_depth, self.max_depth)

        aux = {
            "scale": scale.squeeze(1),
            "shift": shift.squeeze(1),
            "residual": residual,
            "bounded_depth": d_bounded.squeeze(1),
        }
        return metric_depth, aux

class SparseLogL1Loss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, pred, gt, mask, weight=None):
        valid = mask > 0
        if valid.sum() < 10:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        pred = sanitize_depth_tensor(pred)
        gt = sanitize_depth_tensor(gt)
        p = pred[valid].clamp(min=self.eps)
        g = gt[valid].clamp(min=self.eps)
        diff = torch.abs(torch.log(p) - torch.log(g))
        if weight is not None:
            w = torch.nan_to_num(weight[valid], nan=0.0, posinf=1.0, neginf=0.0).clamp(min=0.0)
            return (w * diff).sum() / (w.sum() + self.eps)
        return diff.mean()

class SparseL1Loss(nn.Module):
    def forward(self, pred, gt, mask, weight=None):
        valid = mask > 0
        if valid.sum() < 10:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        pred = sanitize_depth_tensor(pred)
        gt = sanitize_depth_tensor(gt)
        diff = torch.abs(pred[valid] - gt[valid])
        if weight is not None:
            w = torch.nan_to_num(weight[valid], nan=0.0, posinf=1.0, neginf=0.0).clamp(min=0.0)
            return (w * diff).sum() / (w.sum() + 1e-6)
        return diff.mean()

class MetricCombinedLoss(nn.Module):
    def __init__(self, l1_weight=1.0, log_weight=0.2):
        super().__init__()
        self.l1 = SparseL1Loss()
        self.log_l1 = SparseLogL1Loss()
        self.l1_weight = l1_weight
        self.log_weight = log_weight
    def forward(self, pred, gt, mask, weight=None):
        pred = sanitize_depth_tensor(pred)
        gt = sanitize_depth_tensor(gt)
        return (self.l1_weight * self.l1(pred, gt, mask, weight) + 
                self.log_weight * self.log_l1(pred, gt, mask, weight))

@torch.no_grad()
def generate_pseudo_dense_depth(depth_sparse, mask, num_iters=6, kernel_size=5):
    if num_iters <= 0:
        return depth_sparse.clone(), mask.clone()
    k = kernel_size
    pad = k // 2
    depth_sparse = sanitize_depth_tensor(depth_sparse, min_depth=0.0, max_depth=DEFAULT_METRIC_MAX_DEPTH)
    filled = depth_sparse.clone() * mask
    valid = mask.clone()

    for _ in range(num_iters):
        valid_4d = valid.unsqueeze(1)
        filled_4d = (filled * valid).unsqueeze(1)
        count = F.avg_pool2d(valid_4d, kernel_size=k, stride=1, padding=pad) * (k * k)
        local_sum = F.avg_pool2d(filled_4d, kernel_size=k, stride=1, padding=pad) * (k * k)
        proposal = local_sum / count.clamp(min=1.0)
        proposal = sanitize_depth_tensor(proposal.squeeze(1), min_depth=0.0, max_depth=DEFAULT_METRIC_MAX_DEPTH)

        can_fill = (valid < 0.5) & (count.squeeze(1) >= 1.0)
        filled = torch.where(can_fill, proposal, filled)
        filled = sanitize_depth_tensor(filled, min_depth=0.0, max_depth=DEFAULT_METRIC_MAX_DEPTH)
        valid = torch.where(can_fill, torch.ones_like(valid), valid)
    return filled, valid

@torch.no_grad()
def compute_distance_weight(depth_gt, start=60.0, end=100.0, min_w=0.05):
    weight = 1.0 - (depth_gt - start) / (end - start + 1e-6)
    return weight.clamp(min=min_w, max=1.0)

@torch.no_grad()
def build_metric_supervision_weight(depth_gt, mask, cass_module, cfg):
    if cass_module is not None:
        weight = cass_module(depth_gt, mask)
    else:
        weight = compute_distance_weight(
            depth_gt, start=cfg.depth_weight_decay_start,
            end=cfg.depth_weight_decay_end, min_w=cfg.depth_weight_min)
    valid = mask > 0
    if valid.any():
        floor = getattr(cfg, "metric_weight_floor", 0.2)
        weight = torch.where(valid, torch.clamp(weight, min=floor), torch.zeros_like(weight))
    return torch.nan_to_num(weight, nan=0.0, posinf=1.0, neginf=0.0).clamp(min=0.0, max=1.0)

# ==============================================================================
# 新增：导师建议的 3B 实验专属 Loss (Inverse Depth Smoothness & Grad Consistency)
# ==============================================================================

def edge_aware_smoothness_loss(pred_depth, rgb, valid_mask=None, eps=1e-3, alpha=10.0):
    """
    基于 Inverse Depth 的边缘感知平滑损失 (Advisor Version)
    为了最大化收益，这只在没有 LiDAR 监督的地方起效。
    """
    if pred_depth.dim() == 3:
        pred_depth = pred_depth.unsqueeze(1)
    if rgb.shape[-2:] != pred_depth.shape[-2:]:
        rgb = F.interpolate(rgb, size=pred_depth.shape[-2:], mode='bilinear', align_corners=False)
    
    # 强制将 RGB 映射到 [0, 1] 区间，以保证 alpha=10 的指数衰减表现稳定
    b, c, h, w = rgb.shape
    rgb_min = rgb.view(b, c, -1).min(dim=2, keepdim=True)[0].view(b, c, 1, 1)
    rgb_max = rgb.view(b, c, -1).max(dim=2, keepdim=True)[0].view(b, c, 1, 1)
    rgb_norm = (rgb - rgb_min) / (rgb_max - rgb_min + eps)

    # 转换到 Normalized Inverse Depth (如导师所言，对尺度更稳定)
    depth = torch.clamp(pred_depth, min=eps)
    inv_depth = 1.0 / depth
    inv_depth = inv_depth / (inv_depth.mean(dim=(2, 3), keepdim=True) + eps)

    grad_d_x = torch.abs(inv_depth[:, :, :, :-1] - inv_depth[:, :, :, 1:])
    grad_d_y = torch.abs(inv_depth[:, :, :-1, :] - inv_depth[:, :, 1:, :])

    grad_i_x = torch.mean(torch.abs(rgb_norm[:, :, :, :-1] - rgb_norm[:, :, :, 1:]), dim=1, keepdim=True)
    grad_i_y = torch.mean(torch.abs(rgb_norm[:, :, :-1, :] - rgb_norm[:, :, 1:, :]), dim=1, keepdim=True)

    weight_x = torch.exp(-alpha * grad_i_x)
    weight_y = torch.exp(-alpha * grad_i_y)

    loss_x = grad_d_x * weight_x
    loss_y = grad_d_y * weight_y

    # 如果传入了 valid_mask，只在无 LiDAR 监督的区域做平滑 (防干扰硬核真值)
    if valid_mask is not None:
        if valid_mask.dim() == 3:
            valid_mask = valid_mask.unsqueeze(1)
        smooth_mask = 1.0 - valid_mask
        loss_x = loss_x * smooth_mask[:, :, :, :-1]
        loss_y = loss_y * smooth_mask[:, :, :-1, :]

    return loss_x.mean() + loss_y.mean()


def gradient_consistency_loss(pred_abs, stageA_rel, edge_mask=None, eps=1e-3):
    """
    Stage A 边界保真约束 (Advisor Version)
    核心逻辑：Stage B 不要把 Stage A 已经学到的可靠几何边界弄坏。
    """
    if pred_abs.dim() == 3:
        pred_abs = pred_abs.unsqueeze(1)
    if stageA_rel.dim() == 3:
        stageA_rel = stageA_rel.unsqueeze(1)

    # 简单归一化，只比较结构梯度特征，无视绝对数值
    rel = stageA_rel / (stageA_rel.mean(dim=(2,3), keepdim=True) + eps)
    absn = pred_abs / (pred_abs.mean(dim=(2,3), keepdim=True) + eps)

    gx_abs = absn[:, :, :, :-1] - absn[:, :, :, 1:]
    gy_abs = absn[:, :, :-1, :] - absn[:, :, 1:, :]

    gx_rel = rel[:, :, :, :-1] - rel[:, :, :, 1:]
    gy_rel = rel[:, :, :-1, :] - rel[:, :, 1:, :]

    loss_x = torch.abs(gx_abs - gx_rel)
    loss_y = torch.abs(gy_abs - gy_rel)

    if edge_mask is not None:
        if edge_mask.dim() == 3:
            edge_mask = edge_mask.unsqueeze(1)
        loss_x = loss_x * edge_mask[:, :, :, :-1]
        loss_y = loss_y * edge_mask[:, :, :-1, :]

    return loss_x.mean() + loss_y.mean()