"""
================================================================================
Depth Anything V2 + LoRA + CASS + Metric Recovery Head
================================================================================

这是在你原始 train_depth_anything_lora.py 基础上的“最小侵入式”扩展版：

新增能力:
  1. 保留原始 relative depth adaptation（Stage A）
  2. 新增轻量 metric recovery head（Stage B）
  3. 使用 sparse LiDAR 作为主监督 + 在线 pseudo-dense 作为辅监督
  4. 新增 Advisor 3B 实验: Inverse Depth Smoothness & Gradient Consistency
================================================================================
"""

import os
import csv
import json
import time
import math
import random
import argparse
import warnings
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForDepthEstimation
from peft import PeftModel

# ---- 复用原始脚本的成熟组件，尽量减少重复代码 ----
import train_depth_anything_lora as base_impl
from metric_depth_module_v4 import (
    MetricDepthHead,
    MetricCombinedLoss,
    generate_pseudo_dense_depth,
    build_metric_supervision_weight,
    sanitize_depth_tensor,
    DEFAULT_METRIC_MAX_DEPTH,
    edge_aware_smoothness_loss,
    gradient_consistency_loss  # <--- 新引入的导师版保真 Loss
)

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  配置                                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def get_config():
    parser = argparse.ArgumentParser(description="Depth Anything V2 LoRA + Metric Recovery Head (v4)")

    parser.add_argument("--data_root", type=str, default="OSDaR23/train_data")
    parser.add_argument("--input_size", type=int, default=518)
    parser.add_argument("--model_name", type=str, default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--train_decoder", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--metric_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--loss_type", type=str, default="combined")
    parser.add_argument("--use_cass", action="store_true", default=True)
    parser.add_argument("--no_cass", dest="use_cass", action="store_false")
    parser.add_argument("--cass_density_kernel", type=int, default=7)
    parser.add_argument("--cass_smooth_kernel", type=int, default=5)
    parser.add_argument("--cass_smooth_sigma", type=float, default=0.15)
    parser.add_argument("--depth_weight_decay_start", type=float, default=60.0)
    parser.add_argument("--depth_weight_decay_end", type=float, default=100.0)
    parser.add_argument("--depth_weight_min", type=float, default=0.05)
    parser.add_argument("--cass_disable_density", action="store_true", default=False)
    parser.add_argument("--cass_disable_smooth", action="store_true", default=False)
    parser.add_argument("--cass_disable_distance", action="store_true", default=False)

    parser.add_argument("--enable_metric_head", action="store_true", default=True)
    parser.add_argument("--disable_metric_head", dest="enable_metric_head", action="store_false")
    parser.add_argument("--metric_stage_start_epoch", type=int, default=12)
    parser.add_argument("--metric_hidden_dim", type=int, default=32)
    parser.add_argument("--metric_max_residual", type=float, default=1.0)
    parser.add_argument("--metric_max_depth", type=float, default=120.0)
    parser.add_argument("--metric_freeze_backbone", action="store_true", default=True)
    parser.add_argument("--lambda_rel_stage1", type=float, default=1.0)
    parser.add_argument("--lambda_rel_stage2", type=float, default=0.0)
    parser.add_argument("--lambda_metric_sparse", type=float, default=0.5)
    parser.add_argument("--lambda_metric_pseudo", type=float, default=0.0)
    
    # ==== 导师建议的 3B 实验专属 Loss 权重 ====
    parser.add_argument("--lambda_metric_edge", type=float, default=0.0,
                        help="Inverse Depth 平滑损失权重 (推荐 0.001)")
    parser.add_argument("--lambda_metric_grad", type=float, default=0.0,
                        help="Stage A 梯度一致性/边界保真约束权重 (推荐 0.05)")
    # ==========================================

    # ==== 新增：对抗 LiDAR Distribution Leakage 的 Mask Dropout ====
    parser.add_argument("--mask_dropout", type=float, default=0.0,
                        help="训练时随机丢弃 LiDAR 点的比例，对抗传感器分布过拟合 (推荐 0.2~0.3)")
    
    parser.add_argument("--metric_ramp_epochs", type=int, default=3)
    parser.add_argument("--disable_amp_stage2", action="store_true", default=True)
    parser.add_argument("--metric_weight_floor", type=float, default=0.20)
    parser.add_argument("--pseudo_dense_iters", type=int, default=6)
    parser.add_argument("--pseudo_dense_kernel", type=int, default=5)
    parser.add_argument("--metric_loss_l1_weight", type=float, default=1.0)
    parser.add_argument("--metric_loss_log_weight", type=float, default=0.2)
    parser.add_argument("--output_dir", type=str, default="checkpoints_metric")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_ckpt", type=str, default="")
    parser.add_argument("--start_epoch", type=int, default=0)
    return parser.parse_args()

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  模型保存 / 加载                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def save_full_model(base_model, metric_head, save_dir, cfg):
    os.makedirs(save_dir, exist_ok=True)
    base_model.save_pretrained(save_dir)
    decoder_state = {
        n: p.detach().cpu().clone()
        for n, p in base_model.named_parameters()
        if ("neck" in n or "head" in n) and ("lora" not in n.lower())
    }
    torch.save(decoder_state, os.path.join(save_dir, "decoder_head.pt"))
    if metric_head is not None:
        torch.save(metric_head.state_dict(), os.path.join(save_dir, "metric_head.pt"))
        with open(os.path.join(save_dir, "metric_head_config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"metric_hidden_dim": cfg.metric_hidden_dim, "metric_max_residual": cfg.metric_max_residual, "metric_max_depth": cfg.metric_max_depth},
                f, indent=2, ensure_ascii=False
            )

def _patch_adapter_config(save_dir):
    """Remove fields unknown to the installed PEFT version before loading."""
    import inspect
    from peft import LoraConfig
    known = set(inspect.signature(LoraConfig.__init__).parameters.keys())
    cfg_path = os.path.join(save_dir, "adapter_config.json")
    if not os.path.exists(cfg_path):
        return
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    unknown = [k for k in cfg if k not in known and k != "peft_type"]
    if unknown:
        for k in unknown:
            cfg.pop(k)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"  [patch] removed unknown adapter_config keys: {unknown}")


def load_full_model(model_name, save_dir, device="cuda"):
    _patch_adapter_config(save_dir)
    base = AutoModelForDepthEstimation.from_pretrained(model_name)
    base_model = PeftModel.from_pretrained(base, save_dir)
    decoder_path = os.path.join(save_dir, "decoder_head.pt")
    if os.path.exists(decoder_path):
        extra = torch.load(decoder_path, map_location="cpu", weights_only=True)
        params = dict(base_model.named_parameters())
        loaded = 0
        for n, t in extra.items():
            if n in params:
                params[n].data.copy_(t)
                loaded += 1
        print(f"  decoder head: {loaded}/{len(extra)} 参数已加载")
    else:
        print("  [警告] decoder_head.pt 不存在，仅加载 LoRA")

    metric_head = None
    metric_cfg_path = os.path.join(save_dir, "metric_head_config.json")
    metric_path = os.path.join(save_dir, "metric_head.pt")
    if os.path.exists(metric_path):
        if os.path.exists(metric_cfg_path):
            with open(metric_cfg_path, "r", encoding="utf-8") as f:
                metric_cfg = json.load(f)
        else:
            metric_cfg = {"metric_hidden_dim": 32, "metric_max_residual": 1.0, "metric_max_depth": 120.0}
        metric_head = MetricDepthHead(
            hidden_dim=metric_cfg.get("metric_hidden_dim", 32),
            max_residual=metric_cfg.get("metric_max_residual", 1.0),
            max_depth=metric_cfg.get("metric_max_depth", 120.0),
        )
        metric_head.load_state_dict(torch.load(metric_path, map_location="cpu", weights_only=True))
        metric_head = metric_head.to(device)
        print("  metric head: 已加载")
    else:
        print("  metric head: 未检测到，按纯 relative 模型处理")
    return base_model.to(device), metric_head

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  构建模型 / 辅助函数                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def build_model(cfg):
    base_model, _processor = base_impl.build_model(cfg)
    metric_head = None
    if cfg.enable_metric_head:
        metric_head = MetricDepthHead(
            hidden_dim=cfg.metric_hidden_dim, max_residual=cfg.metric_max_residual, max_depth=cfg.metric_max_depth)
        print("  Metric head: 已启用")
    return base_model, metric_head

def get_relative_criterion(cfg):
    if cfg.loss_type == "silog": return base_impl.SparseSILogLoss(lambd=0.5)
    if cfg.loss_type == "l1": return base_impl.SparseL1Loss()
    return base_impl.CombinedLoss()

def get_metric_criterion(cfg):
    return MetricCombinedLoss(l1_weight=cfg.metric_loss_l1_weight, log_weight=cfg.metric_loss_log_weight)

def get_stage(epoch, cfg):
    if not cfg.enable_metric_head: return "relative"
    if epoch < cfg.metric_stage_start_epoch: return "relative"
    return "metric"

def configure_stage_trainability(base_model, metric_head, base_trainable_names, stage, cfg):
    for name, param in base_model.named_parameters():
        param.requires_grad = name in base_trainable_names
    if metric_head is not None:
        for param in metric_head.parameters():
            param.requires_grad = False
    if stage != "relative":
        if cfg.metric_freeze_backbone:
            for _, param in base_model.named_parameters():
                param.requires_grad = False
        if metric_head is not None:
            for param in metric_head.parameters():
                param.requires_grad = True

def infer_base_trainable_names(base_model, cfg):
    names = set()
    for name, _ in base_model.named_parameters():
        if "lora" in name.lower() or (cfg.train_decoder and ("neck" in name or "head" in name)):
            names.add(name)
    return names

def build_optimizer_and_scheduler(base_model, metric_head, train_loader_len, cfg, stage):
    param_groups = []
    if stage == "relative":
        base_params = [p for p in base_model.parameters() if p.requires_grad]
        param_groups.append({"params": base_params, "lr": cfg.lr})
        stage_epochs = max(cfg.metric_stage_start_epoch, 1) if cfg.enable_metric_head else cfg.epochs
        max_lr = cfg.lr
    else:
        metric_params = [] if metric_head is None else [p for p in metric_head.parameters() if p.requires_grad]
        if not cfg.metric_freeze_backbone:
            base_params = [p for p in base_model.parameters() if p.requires_grad]
            if base_params: param_groups.append({"params": base_params, "lr": cfg.lr * 0.25})
        if metric_params:
            param_groups.append({"params": metric_params, "lr": cfg.metric_lr})
        stage_epochs = max(cfg.epochs - cfg.metric_stage_start_epoch, 1)
        max_lr = max(g["lr"] for g in param_groups)

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(train_loader_len // max(cfg.grad_accum, 1), 1)
    total_steps = max(stage_epochs * steps_per_epoch, 1)
    warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*optimizer.step.*")
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=cfg.warmup_ratio, anneal_strategy="cos")
    scaler = base_impl.amp_grad_scaler(enabled=cfg.amp)
    return optimizer, scheduler, scaler, total_steps

def forward_branches(base_model, metric_head, pixel_values, target_hw, compute_metric=True):
    outputs = base_model(pixel_values=pixel_values)
    pred_rel = outputs.predicted_depth
    if pred_rel.shape[-2:] != target_hw:
        pred_rel = F.interpolate(pred_rel.unsqueeze(1), size=target_hw, mode="bilinear", align_corners=False).squeeze(1)

    pred_metric, metric_aux = None, None
    if metric_head is not None and compute_metric:
        rgb_for_metric = F.interpolate(pixel_values, size=pred_rel.shape[-2:], mode="bilinear", align_corners=False)
        pred_metric, metric_aux = metric_head(pred_rel, rgb_for_metric)
        pred_metric = sanitize_depth_tensor(pred_metric, min_depth=1e-3, max_depth=getattr(metric_head, "max_depth", DEFAULT_METRIC_MAX_DEPTH))

    pred_rel = torch.nan_to_num(pred_rel, nan=1e-3, posinf=1e4, neginf=1e-3).clamp(min=1e-3)
    return pred_rel, pred_metric, metric_aux

def build_relative_weight(depth_gt, mask, cfg, cass_module=None):
    if cfg.use_cass and cass_module is not None: return cass_module(depth_gt, mask)
    return base_impl.compute_distance_weight(depth_gt, start=cfg.depth_weight_decay_start, end=cfg.depth_weight_decay_end, min_w=cfg.depth_weight_min)

def get_metric_stage_scale(epoch, cfg):
    if not getattr(cfg, "enable_metric_head", False): return 1.0
    ramp_epochs = max(int(getattr(cfg, "metric_ramp_epochs", 1)), 1)
    stage_epoch = max(epoch - int(cfg.metric_stage_start_epoch), 0) + 1
    return float(min(stage_epoch / ramp_epochs, 1.0))

def _safe_metric_compute(pred, gt, mask):
    return base_impl.compute_metrics(sanitize_depth_tensor(pred), sanitize_depth_tensor(gt), mask)

def maybe_save_stage_best(base_model, metric_head, run_dir, cfg, stage, val_metrics, tracker, epoch):
    saved = {"relative": False, "metric": False}
    current = float(val_metrics["AbsRel"])
    if stage == "relative":
        if current < tracker["best_relative_absrel"]:
            tracker["best_relative_absrel"] = current
            tracker["best_relative_epoch"] = epoch + 1
            save_full_model(base_model, metric_head, os.path.join(run_dir, "best_relative_model"), cfg)
            saved["relative"] = True
    else:
        if current < tracker["best_metric_absrel"]:
            tracker["best_metric_absrel"] = current
            tracker["best_metric_epoch"] = epoch + 1
            save_full_model(base_model, metric_head, os.path.join(run_dir, "best_metric_model"), cfg)
            saved["metric"] = True
    return saved

def format_stage_best_summary(tracker):
    parts = []
    if tracker["best_relative_epoch"] is not None:
        parts.append(f"relative-best: epoch {tracker['best_relative_epoch']} / AbsRel={tracker['best_relative_absrel']:.4f}")
    if tracker["best_metric_epoch"] is not None:
        parts.append(f"metric-best: epoch {tracker['best_metric_epoch']} / AbsRel={tracker['best_metric_absrel']:.4f}")
    return " | ".join(parts) if parts else "暂无 stage-aware best checkpoint"


# ==============================================================================
# 核心改动点：融入导师建议的 3B 实验专属约束
# ==============================================================================
def compute_primary_loss(stage, pred_rel, pred_metric, depth_gt, mask, pixel_values,
                         rel_criterion, metric_criterion, cfg, cass_module, epoch):
    loss_terms = {}
    pred_rel = sanitize_depth_tensor(pred_rel, min_depth=1e-3, max_depth=1e4)
    depth_gt = sanitize_depth_tensor(depth_gt, min_depth=1e-3, max_depth=cfg.metric_max_depth)
    pred_aligned = sanitize_depth_tensor(base_impl.align_pred_to_gt(pred_rel, depth_gt, mask), max_depth=cfg.metric_max_depth)

    if stage == "relative" or pred_metric is None:
        rel_weight = build_relative_weight(depth_gt, mask, cfg, cass_module)
        loss_rel = rel_criterion(pred_aligned, depth_gt, mask, rel_weight)
        total = cfg.lambda_rel_stage1 * loss_rel
        loss_terms["loss_rel"] = float(loss_rel.detach().item())
        return sanitize_depth_tensor(total, min_depth=0.0, max_depth=1e6), pred_aligned, pred_aligned, loss_terms

    # ---- Stage B: absolute depth 主训练目标 ----
    pred_metric = sanitize_depth_tensor(pred_metric, min_depth=1e-3, max_depth=cfg.metric_max_depth)
    metric_scale = get_metric_stage_scale(epoch, cfg)

    # 1. Sparse LiDAR Loss
    metric_weight = build_metric_supervision_weight(depth_gt, mask, cass_module, cfg)
    loss_metric_sparse = metric_criterion(pred_metric, depth_gt, mask, metric_weight)

    # 2. Pseudo-Dense Loss
    loss_metric_pseudo = torch.tensor(0.0, device=depth_gt.device)
    if cfg.lambda_metric_pseudo > 0:
        pseudo_depth, pseudo_mask = generate_pseudo_dense_depth(
            depth_gt, mask, num_iters=cfg.pseudo_dense_iters, kernel_size=cfg.pseudo_dense_kernel)
        pseudo_weight = build_metric_supervision_weight(pseudo_depth, pseudo_mask, None, cfg)
        loss_metric_pseudo = metric_criterion(pred_metric, pseudo_depth, pseudo_mask, pseudo_weight)

    # 3. 导师版：Inverse Depth Edge-aware Smoothness
    loss_edge = torch.tensor(0.0, device=depth_gt.device)
    if cfg.lambda_metric_edge > 0:
        valid_mask = (mask > 0).float()
        loss_edge = edge_aware_smoothness_loss(pred_metric, pixel_values, valid_mask=valid_mask)

    # 4. 导师版：Gradient Consistency (Stage A 边界保真)
    loss_grad = torch.tensor(0.0, device=depth_gt.device)
    if cfg.lambda_metric_grad > 0:
        # 动态提取 Stage A 相对深度(pred_rel)的结构跃变，作为需要死死守住的边缘 Mask
        rel_norm = pred_rel.unsqueeze(1) / (pred_rel.unsqueeze(1).mean(dim=(2,3), keepdim=True) + 1e-3)
        grad_rel_x = torch.abs(rel_norm[:, :, :, :-1] - rel_norm[:, :, :, 1:])
        grad_rel_y = torch.abs(rel_norm[:, :, :-1, :] - rel_norm[:, :, 1:, :])
        grad_mag = F.pad(grad_rel_x, (0, 1, 0, 0)) + F.pad(grad_rel_y, (0, 0, 0, 1))
        
        # 提取软边界掩码 (变化越剧烈，权重越接近1，保护力度越强)
        edge_mask = torch.clamp(grad_mag * 10.0, 0.0, 1.0)
        
        loss_grad = gradient_consistency_loss(pred_metric, pred_rel, edge_mask=edge_mask)

    total = metric_scale * (
        cfg.lambda_metric_sparse * loss_metric_sparse
        + cfg.lambda_metric_pseudo * loss_metric_pseudo
        + cfg.lambda_metric_edge * loss_edge
        + cfg.lambda_metric_grad * loss_grad
    )

    if (not cfg.metric_freeze_backbone) and cfg.lambda_rel_stage2 > 0:
        rel_weight = build_relative_weight(depth_gt, mask, cfg, cass_module)
        loss_rel = rel_criterion(pred_aligned, depth_gt, mask, rel_weight)
        total = total + cfg.lambda_rel_stage2 * loss_rel
        loss_terms["loss_rel"] = float(loss_rel.detach().item())

    loss_terms["metric_scale"] = metric_scale
    loss_terms["loss_metric_sparse"] = float(loss_metric_sparse.detach().item())
    loss_terms["loss_metric_pseudo"] = float(loss_metric_pseudo.detach().item())
    loss_terms["loss_metric_edge"] = float(loss_edge.detach().item())
    loss_terms["loss_metric_grad"] = float(loss_grad.detach().item())
    return total, pred_metric, pred_aligned, loss_terms


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  训练 / 验证                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def train_one_epoch(base_model, metric_head, loader, optimizer, scheduler, scaler,
                    rel_criterion, metric_criterion, cfg, epoch, stage, device, cass_module=None):
    if stage == "metric" and cfg.metric_freeze_backbone:
        base_model.eval()
    else:
        base_model.train()
    if metric_head is not None:
        metric_head.train(stage == "metric")

    total_loss, n_batches = 0.0, 0
    step_losses, lr_history = [], []
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train:{stage}]", leave=False)
    for step, batch in enumerate(pbar):
        global_step = epoch * len(loader) + step
        pixel_values, depth_gt, mask = batch["pixel_values"].to(device), batch["depth_gt"].to(device), batch["mask"].to(device)

        # =====================================================================
        # 新增：Mask Dropout (Density-Pattern Randomization)
        # 仅在 Stage B (metric) 训练时启用，随机抹除部分 LiDAR 点
        # =====================================================================
        if stage == "metric" and cfg.mask_dropout > 0.0:
            # 生成一个与 mask 形状相同的随机张量
            rand_tensor = torch.rand_like(mask.float())
            # 只有大于 dropout 比例的像素才予以保留（即丢弃 cfg.mask_dropout 比例的点）
            keep_mask = rand_tensor > cfg.mask_dropout
            # 更新掩码
            mask = mask * keep_mask
        # =====================================================================

        amp_enabled = cfg.amp and not (stage == "metric" and cfg.disable_amp_stage2)
        with base_impl.amp_autocast(enabled=amp_enabled):
            pred_rel, pred_metric, _ = forward_branches(
                base_model, metric_head, pixel_values, depth_gt.shape[-2:], compute_metric=(stage == "metric" and metric_head is not None))
            loss, _, _, loss_terms = compute_primary_loss(
                stage, pred_rel, pred_metric, depth_gt, mask, pixel_values,
                rel_criterion, metric_criterion, cfg, cass_module, epoch)
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            loss = loss / cfg.grad_accum

        scaler.scale(loss).backward()
        if (step + 1) % cfg.grad_accum == 0:
            scaler.unscale_(optimizer)
            trainable_params = [p for g in optimizer.param_groups for p in g["params"] if p.grad is not None]
            if trainable_params: nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None: scheduler.step()

        loss_val = loss.item() * cfg.grad_accum
        total_loss += loss_val
        n_batches += 1
        step_losses.append((global_step, loss_val))
        lr_history.append((global_step, optimizer.param_groups[0]["lr"]))

        if (step + 1) % cfg.log_every == 0:
            avg = total_loss / max(n_batches, 1)
            postfix = {"loss": f"{avg:.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"}
            for k, v in loss_terms.items(): postfix[k] = f"{v:.3f}"
            pbar.set_postfix(postfix)

    return total_loss / max(n_batches, 1), step_losses, lr_history

@torch.no_grad()
def validate(base_model, metric_head, loader, rel_criterion, metric_criterion,
             cfg, epoch, stage, device, cass_module=None):
    base_model.eval()
    if metric_head is not None: metric_head.eval()

    total_loss, n = 0.0, 0
    primary_metrics = {"AbsRel": [], "RMSE": [], "delta1": [], "delta2": [], "delta3": []}
    aligned_metrics_ref = {"AbsRel": [], "RMSE": [], "delta1": [], "delta2": [], "delta3": []}

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Val:{stage}]", leave=False)
    for batch in pbar:
        pixel_values, depth_gt, mask = batch["pixel_values"].to(device), batch["depth_gt"].to(device), batch["mask"].to(device)

        amp_enabled = cfg.amp and not (stage == "metric" and cfg.disable_amp_stage2)
        with base_impl.amp_autocast(enabled=amp_enabled):
            pred_rel, pred_metric, _ = forward_branches(
                base_model, metric_head, pixel_values, depth_gt.shape[-2:], compute_metric=(stage == "metric" and metric_head is not None))
            loss, primary_pred, aligned_ref, _ = compute_primary_loss(
                stage, pred_rel, pred_metric, depth_gt, mask, pixel_values,
                rel_criterion, metric_criterion, cfg, cass_module, epoch)

        total_loss += loss.item()
        n += 1
        metrics_primary = _safe_metric_compute(primary_pred, depth_gt, mask)
        for k in primary_metrics: primary_metrics[k].append(metrics_primary[k])

        if stage == "metric":
            metrics_aligned = _safe_metric_compute(aligned_ref, depth_gt, mask)
            for k in aligned_metrics_ref: aligned_metrics_ref[k].append(metrics_aligned[k])

    avg = {k: float(np.mean(v)) for k, v in primary_metrics.items()}
    avg["loss"] = total_loss / max(n, 1)
    if stage == "metric" and aligned_metrics_ref["AbsRel"]:
        for k in aligned_metrics_ref: avg[f"aligned_{k}"] = float(np.mean(aligned_metrics_ref[k]))
    return avg

@torch.no_grad()
def visualize_val_predictions(base_model, metric_head, val_ds, cfg, epoch, stage, vis_dir, device, num_samples=4, vis_max=80.0):
    base_model.eval()
    if metric_head is not None: metric_head.eval()
    os.makedirs(vis_dir, exist_ok=True)
    indices = list(range(0, len(val_ds), max(1, len(val_ds) // num_samples)))[:num_samples]

    for i, idx in enumerate(indices):
        sample = val_ds[idx]
        pv, gt, mask = sample["pixel_values"].unsqueeze(0).to(device), sample["depth_gt"].numpy(), sample["mask"].numpy()
        amp_enabled = cfg.amp and not (stage == "metric" and cfg.disable_amp_stage2)
        with base_impl.amp_autocast(enabled=amp_enabled):
            pred_rel, pred_metric, _ = forward_branches(base_model, metric_head, pv, gt.shape, compute_metric=(stage == "metric" and metric_head is not None))

        pred_aligned = base_impl.align_pred_to_gt(
            pred_rel.squeeze(0).unsqueeze(0), torch.from_numpy(gt).unsqueeze(0).to(device), torch.from_numpy(mask).unsqueeze(0).to(device)
        ).squeeze(0).float().cpu().numpy()
        pred_aligned = np.clip(np.nan_to_num(pred_aligned, nan=0.0, posinf=cfg.metric_max_depth), 0.0, cfg.metric_max_depth)

        if stage == "metric" and pred_metric is not None:
            pred_np = np.clip(np.nan_to_num(pred_metric.squeeze(0).float().cpu().numpy(), nan=0.0, posinf=cfg.metric_max_depth), 0.0, cfg.metric_max_depth)
            pred_label = "Metric prediction"
        else:
            pred_np, pred_label = pred_aligned, "Aligned relative"

        def to_color(d, vmax):
            u8 = (np.clip(np.nan_to_num(d / max(vmax, 1e-6)), 0, 1) * 255).astype(np.uint8)
            c = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
            c[u8 == 0] = 0
            return c

        # 先将其转为 uint8
        rgb_u8 = ((sample["pixel_values"].numpy().transpose(1, 2, 0) * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1) * 255).astype(np.uint8)
        # 再传给 OpenCV 转换通道
        rgb_bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
        # rgb_bgr = cv2.cvtColor((sample["pixel_values"].numpy().transpose(1, 2, 0) * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1) * 255, cv2.COLOR_RGB2BGR).astype(np.uint8)
        pred_color = to_color(pred_np, vis_max)
        gt_u8 = (np.clip(np.nan_to_num(gt, nan=0.0) / max(vis_max, 1e-6), 0, 1) * 255).astype(np.uint8)
        gt_color = cv2.applyColorMap(cv2.dilate(gt_u8, np.ones((3, 3), np.uint8)), cv2.COLORMAP_TURBO)
        gt_color[cv2.dilate(gt_u8, np.ones((3, 3), np.uint8)) == 0] = 0

        err_color = cv2.applyColorMap((np.clip(np.nan_to_num(np.abs(pred_np - gt) * mask / 10.0), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        err_color[mask == 0] = [40, 40, 40]

        panel = np.concatenate([rgb_bgr, pred_color, gt_color, err_color], axis=1)
        w4 = panel.shape[1] // 4
        for j, label in enumerate([f"RGB ({sample['stem']})", pred_label, "GT (sparse)", "Error"]):
            cv2.putText(panel, label, (j * w4 + 5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(os.path.join(vis_dir, f"epoch{epoch+1:03d}_sample{i}.jpg"), panel, [cv2.IMWRITE_JPEG_QUALITY, 90])

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  主流程                                                                ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def main():
    cfg = get_config()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    train_ds = base_impl.OSDaR23DepthDataset(cfg.data_root, os.path.join(cfg.data_root, "splits", "train.txt"), cfg.input_size, augment=True)
    val_ds = base_impl.OSDaR23DepthDataset(cfg.data_root, os.path.join(cfg.data_root, "splits", "val.txt"), cfg.input_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    if cfg.resume_ckpt:
        print(f"从 checkpoint 恢复: {cfg.resume_ckpt}")
        base_model, metric_head = load_full_model(cfg.model_name, cfg.resume_ckpt, device=device)
        if metric_head is None and cfg.enable_metric_head:
            metric_head = MetricDepthHead(hidden_dim=cfg.metric_hidden_dim, max_residual=cfg.metric_max_residual, max_depth=cfg.metric_max_depth).to(device)
    else:
        base_model, metric_head = build_model(cfg)
        base_model = base_model.to(device)
        if metric_head is not None: metric_head = metric_head.to(device)

    base_trainable_names = infer_base_trainable_names(base_model, cfg)
    rel_criterion, metric_criterion = get_relative_criterion(cfg), get_metric_criterion(cfg)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    logger = base_impl.TrainingLogger(run_dir)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f: json.dump(vars(cfg), f, indent=2, ensure_ascii=False)

    cass_module = None
    if cfg.use_cass:
        from confidence_module import ConfidenceAwareSparseSupervision
        cass_module = ConfidenceAwareSparseSupervision(
            density_kernel=cfg.cass_density_kernel, smooth_kernel=cfg.cass_smooth_kernel, smooth_sigma=cfg.cass_smooth_sigma,
            dist_start=cfg.depth_weight_decay_start, dist_end=cfg.depth_weight_decay_end, dist_min=cfg.depth_weight_min,
            disable_density=cfg.cass_disable_density, disable_smooth=cfg.cass_disable_smooth, disable_distance=cfg.cass_disable_distance)

    tracker = {"best_relative_absrel": float("inf"), "best_metric_absrel": float("inf"), "best_relative_epoch": None, "best_metric_epoch": None}
    current_stage, train_start_time = None, time.time()

    print("\n" + "=" * 64)
    print("开始训练：Relative Adaptation  →  Metric Recovery (v4_Advisor_3B)")
    print("=" * 64)

    for epoch in range(cfg.start_epoch, cfg.epochs):
        stage = get_stage(epoch, cfg)
        if stage != current_stage:
            current_stage = stage
            configure_stage_trainability(base_model, metric_head, base_trainable_names, stage, cfg)
            optimizer, scheduler, scaler, _ = build_optimizer_and_scheduler(base_model, metric_head, len(train_loader), cfg, stage)

        t0 = time.time()
        train_loss, step_losses, lr_history = train_one_epoch(
            base_model, metric_head, train_loader, optimizer, scheduler, scaler,
            rel_criterion, metric_criterion, cfg, epoch, stage, device, cass_module)
        logger.log_steps(step_losses, lr_history)

        val_metrics, is_best = None, False
        if (epoch + 1) % cfg.eval_every == 0:
            val_metrics = validate(base_model, metric_head, val_loader, rel_criterion, metric_criterion, cfg, epoch, stage, device, cass_module)
            saved_flags = maybe_save_stage_best(base_model, metric_head, run_dir, cfg, stage, val_metrics, tracker, epoch)
            is_best = bool(saved_flags["relative"] or saved_flags["metric"])
            visualize_val_predictions(base_model, metric_head, val_ds, cfg, epoch, stage, logger.vis_dir, device, num_samples=4, vis_max=cfg.depth_weight_decay_end)

        elapsed = time.time() - t0
        logger.log_epoch(epoch, train_loss, val_metrics, optimizer.param_groups[0]["lr"], elapsed, is_best)

        print(f"\nEpoch {epoch+1}/{cfg.epochs} [{stage}] ({elapsed:.0f}s)")
        print(f"  Train loss: {train_loss:.4f}")
        if val_metrics is not None:
            print(f"  Val loss:   {val_metrics['loss']:.4f}")
            print(f"  AbsRel:     {val_metrics['AbsRel']:.4f}")
            print(f"  RMSE:       {val_metrics['RMSE']:.2f}m")
            if is_best:
                if stage == "relative" and tracker["best_relative_epoch"] == epoch + 1: print(f"  ★ 新 relative-best! AbsRel={tracker['best_relative_absrel']:.4f}")
                if stage == "metric" and tracker["best_metric_epoch"] == epoch + 1: print(f"  ★ 新 metric-best! AbsRel={tracker['best_metric_absrel']:.4f}")

        if (epoch + 1) % cfg.save_every == 0:
            ckpt_path = os.path.join(run_dir, f"epoch_{epoch+1}")
            save_full_model(base_model, metric_head, ckpt_path, cfg)
            print(f"  Checkpoint → {ckpt_path}")

        logger.plot_all()

    final_path = os.path.join(run_dir, "final_model")
    save_full_model(base_model, metric_head, final_path, cfg)
    total_time = time.time() - train_start_time
    logger.plot_all()

    extra_summary = os.path.join(run_dir, "metric_recovery_summary.txt")
    with open(extra_summary, "w", encoding="utf-8") as f:
        f.write("=" * 64 + "\nMetric recovery training summary (v4 Advisor 3B)\n" + "=" * 64 + "\n\n")
        f.write(f"lambda_metric_sparse: {cfg.lambda_metric_sparse}\n")
        f.write(f"lambda_metric_pseudo: {cfg.lambda_metric_pseudo}\n")
        f.write(f"lambda_metric_edge: {cfg.lambda_metric_edge}\n")
        f.write(f"lambda_metric_grad: {cfg.lambda_metric_grad}\n")
        if tracker["best_metric_epoch"] is not None:
            f.write(f"Best metric AbsRel: {tracker['best_metric_absrel']:.4f} (epoch {tracker['best_metric_epoch']})\n")

    print(f"\n训练完成! 最终模型: {final_path}")

if __name__ == "__main__":
    main()