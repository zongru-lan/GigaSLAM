"""
================================================================================
Depth Anything V2 — LoRA 微调脚本 (OSDaR23 铁路场景)
================================================================================

功能:
  在 OSDaR23 铁路数据集上对 Depth Anything V2 进行 LoRA 微调，
  使用稀疏 LiDAR 投影深度作为监督信号，实现铁路场景的域适配。

模型:
  Depth Anything V2 ViT-Small (适合 24GB 4090 的 LoRA 微调)
  - 预训练权重自动从 HuggingFace 下载
  - 仅对 ViT encoder 注入 LoRA adapter，decoder head 可选冻结/微调

硬件需求:
  - GPU: 24GB VRAM (RTX 4090 / A5000 等)
  - 训练分辨率: 518×518 (DA V2 原生输入尺寸)
  - batch_size=4, 混合精度 → 约 12-16GB 显存占用

用法:
  python train_depth_anything_lora.py

  训练前确保:
    1. 已运行 generate_depth_training_data.py 生成训练数据
    2. OSDaR23/train_data/ 目录下有 rgb/, depth/, splits/ 子目录

依赖:
  pip install torch torchvision transformers peft opencv-python tqdm tensorboard
================================================================================
"""

import os
import cv2
import csv
import time
import json
import random
import argparse
import warnings
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 兼容 PyTorch 1.x 和 2.x 的混合精度 API
try:
    # PyTorch >= 2.0 推荐写法
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler
    def amp_autocast(enabled=True):
        return _autocast(device_type="cuda", enabled=enabled)
    def amp_grad_scaler(enabled=True):
        return _GradScaler("cuda", enabled=enabled)
except ImportError:
    # PyTorch < 2.0 回退
    from torch.cuda.amp import autocast as _autocast, GradScaler as _GradScaler
    def amp_autocast(enabled=True):
        return _autocast(enabled=enabled)
    def amp_grad_scaler(enabled=True):
        return _GradScaler(enabled=enabled)

from transformers import AutoModelForDepthEstimation, AutoImageProcessor
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

from tqdm import tqdm


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  配置                                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def get_config():
    """
    获取训练配置。

    通过命令行参数或直接修改默认值来调整。
    所有路径、超参数集中管理，方便复现。
    """
    parser = argparse.ArgumentParser(description="Depth Anything V2 LoRA Fine-tuning")

    # ---------- 数据 ----------
    parser.add_argument("--data_root", type=str, default="OSDaR23/train_data",
                        help="训练数据根目录")
    parser.add_argument("--input_size", type=int, default=518,
                        help="模型输入尺寸 (DA V2 原生为 518)")

    # ---------- 模型 ----------
    parser.add_argument("--model_name", type=str,
                        default="depth-anything/Depth-Anything-V2-Small-hf",
                        help="HuggingFace 模型名称")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA 秩 (rank)，越大表达能力越强但参数更多")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA 缩放因子，通常设为 2×r")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--train_decoder", action="store_true", default=True,
                        help="是否微调 DPT decoder head（推荐开启）")

    # ---------- 训练 ----------
    parser.add_argument("--epochs", type=int, default=20,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="批大小，4090 24GB 下建议 4")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                        help="学习率 warmup 比例")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="混合精度训练 (节省显存)")
    parser.add_argument("--grad_accum", type=int, default=2,
                        help="梯度累积步数，等效 batch = batch_size × grad_accum")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader 工作进程数")

    # ---------- 损失函数 ----------
    parser.add_argument("--loss_type", type=str, default="silog",
                        choices=["l1", "silog", "combined"],
                        help="损失函数类型")
    parser.add_argument("--use_cass", action="store_true", default=True,
                        help="使用置信度感知稀疏监督 (CASS) 模块")
    parser.add_argument("--no_cass", dest="use_cass", action="store_false",
                        help="禁用 CASS (消融实验用)")
    parser.add_argument("--cass_density_kernel", type=int, default=7,
                        help="CASS 局部密度统计窗口")
    parser.add_argument("--cass_smooth_kernel", type=int, default=5,
                        help="CASS 深度一致性统计窗口")
    parser.add_argument("--cass_smooth_sigma", type=float, default=0.15,
                        help="CASS 深度变化容忍度")
    parser.add_argument("--depth_weight_decay_start", type=float, default=60.0,
                        help="距离衰减起点 (米)，此距离内权重为 1")
    parser.add_argument("--depth_weight_decay_end", type=float, default=100.0,
                        help="距离衰减终点 (米)，此距离外权重最低")
    parser.add_argument("--depth_weight_min", type=float, default=0.05,
                        help="远距离点的最低权重")
    # CASS 因子消融 (论文 Table: CASS ablation)
    parser.add_argument("--cass_disable_density", action="store_true", default=False,
                        help="禁用 CASS 密度因子 (消融实验)")
    parser.add_argument("--cass_disable_smooth", action="store_true", default=False,
                        help="禁用 CASS 平滑度因子 (消融实验)")
    parser.add_argument("--cass_disable_distance", action="store_true", default=False,
                        help="禁用 CASS 距离因子 (消融实验)")
    # 在 get_config() 的 CASS 参数区域后面加:
    parser.add_argument("--confidence_method", type=str, default="cass",
                        choices=["cass", "uniform", "image_gradient", "variance_filter"],
                        help="置信度方法")

    # ---------- 输出 ----------
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="模型保存目录")
    parser.add_argument("--save_every", type=int, default=5,
                        help="每 N 个 epoch 保存一次 checkpoint")
    parser.add_argument("--log_every", type=int, default=20,
                        help="每 N 步打印一次日志")
    parser.add_argument("--eval_every", type=int, default=1,
                        help="每 N 个 epoch 评估一次验证集")

    # ---------- 其他 ----------
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  数据集                                                                ║
# ╚═══════════════════════════════════════════════════════════════════════╝

class OSDaR23DepthDataset(Dataset):
    """
    OSDaR23 稀疏深度监督数据集。

    数据对:
      - RGB 图像:  train_data/rgb/{stem}.png    (4112×2504, 3通道)
      - 深度 GT:   train_data/depth/{stem}.png  (4112×2504, uint16, 单位 cm)

    预处理:
      1. RGB: resize → input_size × input_size → 归一化 (ImageNet mean/std)
      2. Depth: resize → model_output_size × model_output_size
         (DA V2 输出分辨率 ≠ 输入分辨率，需要在 loss 中对齐)
      3. 深度 GT 中 0 值表示无效像素，不参与 loss 计算

    注意:
      深度图 resize 使用 NEAREST 插值，避免在有效/无效像素边界产生虚假深度值。
    """

    def __init__(self, data_root, split_file, input_size=518, augment=False):
        """
        参数:
          data_root:  训练数据根目录 (包含 rgb/, depth/ 子目录)
          split_file: 划分文件路径 (每行一个 stem，无扩展名)
          input_size: 模型输入尺寸 (正方形)
          augment:    是否做数据增强 (训练集 True, 验证集 False)
        """
        self.rgb_dir = os.path.join(data_root, "rgb")
        self.depth_dir = os.path.join(data_root, "depth")
        self.input_size = input_size
        self.augment = augment

        # 读取文件列表
        with open(split_file, "r") as f:
            self.stems = [line.strip() for line in f if line.strip()]

        # 过滤不存在的文件
        valid = []
        for s in self.stems:
            if (os.path.exists(os.path.join(self.rgb_dir, s + ".png")) and
                os.path.exists(os.path.join(self.depth_dir, s + ".png"))):
                valid.append(s)
        self.stems = valid

        # ImageNet 归一化参数 (Depth Anything V2 使用 DINOv2 backbone)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        print(f"  数据集: {split_file} → {len(self.stems)} 帧")

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]

        # ---------- 加载 RGB ----------
        rgb_path = os.path.join(self.rgb_dir, stem + ".png")
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)      # BGR, uint8
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)         # → RGB

        # ---------- 加载深度 GT ----------
        depth_path = os.path.join(self.depth_dir, stem + ".png")
        depth_u16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # uint16, cm
        depth_m = depth_u16.astype(np.float32) / 100.0            # → 米

        # ---------- 数据增强 ----------
        if self.augment:
            rgb, depth_m = self._augment(rgb, depth_m)

        # ---------- Resize ----------
        # RGB → input_size × input_size (双线性插值)
        rgb_resized = cv2.resize(rgb, (self.input_size, self.input_size),
                                 interpolation=cv2.INTER_LINEAR)

        # 深度 GT → input_size × input_size (最近邻插值，保持稀疏性)
        depth_resized = cv2.resize(depth_m, (self.input_size, self.input_size),
                                   interpolation=cv2.INTER_NEAREST)

        # ---------- 归一化 ----------
        # RGB: [0,255] uint8 → [0,1] float → ImageNet normalize
        rgb_norm = rgb_resized.astype(np.float32) / 255.0
        rgb_norm = (rgb_norm - self.mean) / self.std

        # HWC → CHW
        rgb_tensor = torch.from_numpy(rgb_norm.transpose(2, 0, 1))    # (3, H, W)
        depth_tensor = torch.from_numpy(depth_resized)                 # (H, W)

        # 有效掩码: depth > 0 的位置参与 loss
        mask_tensor = (depth_tensor > 0).float()                       # (H, W)

        return {
            "pixel_values": rgb_tensor,
            "depth_gt": depth_tensor,
            "mask": mask_tensor,
            "stem": stem,
        }

    def _augment(self, rgb, depth):
        """
        轻量数据增强。

        铁路场景的增强需要保守:
          - 水平翻转: 轨道通常左右对称，翻转是安全的
          - 颜色抖动: 只做亮度和对比度，不改色相 (避免信号灯颜色语义错乱)
          - 不做旋转/透视变换: 会破坏深度的几何一致性
        """
        # 50% 概率水平翻转
        if random.random() < 0.5:
            rgb = np.fliplr(rgb).copy()
            depth = np.fliplr(depth).copy()

        # 亮度/对比度抖动
        if random.random() < 0.5:
            # 亮度: ±15%
            brightness = random.uniform(0.85, 1.15)
            rgb = np.clip(rgb.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

        if random.random() < 0.3:
            # 对比度: ±10%
            contrast = random.uniform(0.9, 1.1)
            mean_val = rgb.mean()
            rgb = np.clip(
                (rgb.astype(np.float32) - mean_val) * contrast + mean_val,
                0, 255
            ).astype(np.uint8)

        return rgb, depth


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  损失函数                                                              ║
# ╚═══════════════════════════════════════════════════════════════════════╝

class SparseSILogLoss(nn.Module):
    """
    Scale-Invariant Logarithmic Loss (SILog) 的稀疏版本。

    原始 SILog (Eigen et al., 2014):
      d_i = log(pred_i) - log(gt_i)
      loss = mean(d²) - λ·mean(d)²

    其中 λ 控制 scale-invariant 的程度:
      λ=1.0: 完全 scale-invariant (忽略全局尺度偏移)
      λ=0.5: 部分 scale-invariant (保留一些绝对尺度约束，推荐)
      λ=0.0: 退化为 MSE-in-log-space

    稀疏适配:
      只在 mask > 0 的像素上计算，并支持距离衰减加权。

    为什么选 SILog:
      1. 在对数空间操作，自然平衡近处/远处的贡献
         (10m 处 1m 误差 vs 100m 处 10m 误差，在 log 空间中权重相当)
      2. 深度估计领域的标准 loss，便于与其他工作对比
      3. 对稀疏监督友好：少量有效像素也能产生稳定梯度
    """

    def __init__(self, lambd=0.5, eps=1e-6):
        """
        参数:
          lambd: scale-invariance 系数，0.5 为平衡选择
          eps:   防止 log(0) 的小常数
        """
        super().__init__()
        self.lambd = lambd
        self.eps = eps

    def forward(self, pred, gt, mask, weight=None):
        """
        参数:
          pred:   (B, H, W) 预测深度 (米，正值)
          gt:     (B, H, W) GT 深度 (米，正值)
          mask:   (B, H, W) 有效像素掩码 (1=有效, 0=无效)
          weight: (B, H, W) 可选距离衰减权重

        返回:
          loss: 标量
        """
        # 只取有效像素
        valid = mask > 0
        if valid.sum() < 10:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        pred_valid = pred[valid].clamp(min=self.eps)
        gt_valid = gt[valid].clamp(min=self.eps)

        # log 空间的差异
        d = torch.log(pred_valid) - torch.log(gt_valid)

        if weight is not None:
            w = weight[valid]
            w = w / (w.sum() + self.eps)   # 归一化权重
            loss = (w * d ** 2).sum() - self.lambd * (w * d).sum() ** 2
        else:
            loss = (d ** 2).mean() - self.lambd * d.mean() ** 2

        return loss


class SparseL1Loss(nn.Module):
    """
    稀疏 L1 Loss，支持距离衰减加权。

    简单直接，对异常值鲁棒。
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, mask, weight=None):
        valid = mask > 0
        if valid.sum() < 10:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        diff = torch.abs(pred[valid] - gt[valid])

        if weight is not None:
            w = weight[valid]
            return (w * diff).sum() / (w.sum() + 1e-6)
        else:
            return diff.mean()


class CombinedLoss(nn.Module):
    """
    SILog + L1 组合损失。

    SILog 擅长学习相对深度排序和对数空间的尺度，
    L1 直接约束绝对深度值。两者互补。
    """

    def __init__(self, silog_weight=1.0, l1_weight=0.1):
        super().__init__()
        self.silog = SparseSILogLoss(lambd=0.5)
        self.l1 = SparseL1Loss()
        self.w_silog = silog_weight
        self.w_l1 = l1_weight

    def forward(self, pred, gt, mask, weight=None):
        return (self.w_silog * self.silog(pred, gt, mask, weight)
                + self.w_l1 * self.l1(pred, gt, mask, weight))


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  距离衰减权重                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def compute_distance_weight(depth_gt, start=60.0, end=100.0, min_w=0.05):
    """
    根据 GT 深度计算距离衰减权重。

    原理:
      近处 LiDAR 精度高 + 投影偏移小 → 权重 = 1.0
      远处 LiDAR 精度低 + 投影偏移大 → 权重逐渐衰减到 min_w

    权重函数:
      depth ≤ start:          w = 1.0
      start < depth < end:    w = 线性衰减
      depth ≥ end:            w = min_w

    参数:
      depth_gt: (B, H, W) GT 深度
      start:    衰减起点 (米)
      end:      衰减终点 (米)
      min_w:    最低权重

    返回:
      weight: (B, H, W) 权重图
    """
    weight = 1.0 - (depth_gt - start) / (end - start + 1e-6)
    weight = weight.clamp(min=min_w, max=1.0)
    return weight


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  评估指标                                                              ║
# ╚═══════════════════════════════════════════════════════════════════════╝

@torch.no_grad()
def compute_metrics(pred, gt, mask):
    """
    计算深度估计标准指标。

    参数:
      pred: (B, H, W) 预测深度
      gt:   (B, H, W) GT 深度
      mask: (B, H, W) 有效掩码

    返回:
      dict: {
        AbsRel:  平均相对误差
        RMSE:    均方根误差
        delta1:  阈值准确率 (max(pred/gt, gt/pred) < 1.25)
        delta2:  阈值准确率 (< 1.25²)
        delta3:  阈值准确率 (< 1.25³)
      }
    """
    valid = mask > 0
    n = valid.sum().item()
    if n < 10:
        return {"AbsRel": 0, "RMSE": 0, "delta1": 0, "delta2": 0, "delta3": 0}

    p = pred[valid].clamp(min=1e-3)
    g = gt[valid].clamp(min=1e-3)

    # AbsRel: mean(|pred - gt| / gt)
    abs_rel = (torch.abs(p - g) / g).mean().item()

    # RMSE
    rmse = torch.sqrt(((p - g) ** 2).mean()).item()

    # 阈值准确率
    ratio = torch.max(p / g, g / p)
    delta1 = (ratio < 1.25).float().mean().item()
    delta2 = (ratio < 1.25 ** 2).float().mean().item()
    delta3 = (ratio < 1.25 ** 3).float().mean().item()

    return {
        "AbsRel": abs_rel,
        "RMSE": rmse,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
    }


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  模型保存与加载 (解决 decoder head 丢失问题)                              ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def save_full_model(model, save_dir):
    """
    保存完整模型: LoRA adapter + 微调过的 decoder head 权重。

    PeftModel.save_pretrained() 只保存 LoRA adapter，但我们同时微调了
    DPT decoder head (~2.6M 参数)。如果不额外保存，加载时 decoder 恢复
    预训练状态，导致推理指标与训练时严重不一致。
    """
    os.makedirs(save_dir, exist_ok=True)
    # 保存 LoRA adapter
    model.save_pretrained(save_dir)
    # 额外保存非 LoRA 的可训练参数 (decoder head)
    extra = {n: p.data.cpu().clone() for n, p in model.named_parameters()
             if p.requires_grad and "lora" not in n.lower()}
    if extra:
        torch.save(extra, os.path.join(save_dir, "decoder_head.pt"))


def load_full_model(model_name, save_dir, device="cuda"):
    """加载完整模型: base + LoRA + decoder head。与 save_full_model 配对。"""
    base = AutoModelForDepthEstimation.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base, save_dir)
    dh_path = os.path.join(save_dir, "decoder_head.pt")
    if os.path.exists(dh_path):
        extra = torch.load(dh_path, map_location="cpu", weights_only=True)
        params = dict(model.named_parameters())
        loaded = sum(1 for n, t in extra.items() if n in params and params[n].data.copy_(t) is not None)
        print(f"  decoder head: {loaded}/{len(extra)} 参数已加载")
    else:
        print("  [警告] decoder_head.pt 不存在，仅加载 LoRA")
    return model.to(device)


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  模型构建                                                              ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def build_model(cfg):
    """
    加载 Depth Anything V2 并注入 LoRA。

    架构:
      Depth Anything V2 = DINOv2 ViT encoder + DPT decoder head

    LoRA 策略:
      - 对 ViT encoder 的 query/value 投影层注入 LoRA adapter
      - DPT decoder 可选择全量微调 (推荐) 或冻结
      - 这样总训练参数约为完整模型的 2-5%

    Depth Anything V2 的特殊处理:
      - 模型输出的是 inverse depth (1/d)，需要转换为 metric depth
      - HuggingFace 版本的输出: model.predicted_depth → (B, H, W)

    返回:
      model: 注入 LoRA 后的模型
      processor: 图像预处理器 (本脚本中未使用，数据集自行处理)
    """
    print(f"加载模型: {cfg.model_name}")
    model = AutoModelForDepthEstimation.from_pretrained(cfg.model_name)
    processor = AutoImageProcessor.from_pretrained(cfg.model_name)

    # ---- 注入 LoRA ----
    # 目标: ViT encoder 中的 attention projection 层
    # HF Depth Anything V2 的模块命名取决于 transformers 版本:
    #   - 可能是 "qkv" (fused) 或 "query"/"key"/"value" (分离)
    # 这里自动检测实际存在的模块名
    all_names = [n for n, _ in model.named_modules()]
    if any("qkv" in n for n in all_names):
        target = ["qkv"]           # fused QKV (DINOv2 风格)
    elif any("query" in n for n in all_names):
        target = ["query", "value"]  # 分离 Q/K/V，只对 Q 和 V 做 LoRA
    else:
        # 回退: 对所有 Linear 层尝试 LoRA
        target = ["query", "value", "qkv"]
        print("  [警告] 未自动检测到 attention 模块名，使用宽松匹配")

    print(f"  LoRA target_modules: {target}")

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=target,
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    # ---- Decoder 微调控制 ----
    if cfg.train_decoder:
        # 解冻 DPT decoder 的所有参数
        for name, param in model.named_parameters():
            # neck 和 head 是 DPT decoder 的组成部分
            if "neck" in name or "head" in name:
                param.requires_grad = True
    else:
        # 只训练 LoRA 参数
        pass

    # 统计可训练参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  总参数:   {total / 1e6:.1f}M")
    print(f"  可训练:   {trainable / 1e6:.1f}M ({trainable/total*100:.1f}%)")

    return model, processor


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  训练与验证循环                                                          ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion,
                    cfg, epoch, device, cass_module=None):
    """
    单个 epoch 的训练循环。

    关键实现细节:
      1. Depth Anything V2 输出 relative/inverse depth，
         需要通过 scale-and-shift 对齐到 metric depth
      2. 只在有效像素上计算 loss
      3. 支持梯度累积和混合精度

    返回:
      avg_loss:   float, 平均 epoch loss
      step_losses: list of (global_step, loss_value), 每步 loss 记录
      lr_history:  list of (global_step, lr), 学习率记录
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    step_losses = []    # 每步 loss
    lr_history = []     # 学习率变化

    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for step, batch in enumerate(pbar):
        global_step = epoch * len(loader) + step

        pixel_values = batch["pixel_values"].to(device)   # (B, 3, H, W)
        depth_gt = batch["depth_gt"].to(device)            # (B, H, W)
        mask = batch["mask"].to(device)                     # (B, H, W)

        with amp_autocast(enabled=cfg.amp):
            # 前向推理
            outputs = model(pixel_values=pixel_values)
            pred_depth = outputs.predicted_depth             # (B, H, W)

            # resize 到与 GT 相同的尺寸 (DA V2 输出可能与输入尺寸不同)
            if pred_depth.shape[-2:] != depth_gt.shape[-2:]:
                pred_depth = F.interpolate(
                    pred_depth.unsqueeze(1),
                    size=depth_gt.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                ).squeeze(1)

            # ---- Metric depth 对齐 ----
            pred_aligned = align_pred_to_gt(pred_depth, depth_gt, mask)

            # ---- 置信度权重 ----
            if hasattr(cfg, 'use_cass') and cfg.use_cass and cass_module is not None:
                # CASS: 综合密度 + 平滑度 + 距离的置信度
                weight = cass_module(depth_gt, mask, pixel_values=pixel_values)
            else:
                # 回退: 仅距离衰减 (消融实验用)
                weight = compute_distance_weight(
                    depth_gt,
                    start=cfg.depth_weight_decay_start,
                    end=cfg.depth_weight_decay_end,
                    min_w=cfg.depth_weight_min,
                )

            # 计算 loss
            loss = criterion(pred_aligned, depth_gt, mask, weight)
            loss = loss / cfg.grad_accum   # 梯度累积缩放

        # 反向传播
        scaler.scale(loss).backward()

        # 梯度累积: 每 grad_accum 步做一次参数更新
        if (step + 1) % cfg.grad_accum == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # scheduler 必须在 optimizer.step() 之后调用
        # 但对于 OneCycleLR，每个 optimizer step 调一次
        # 放在 grad_accum 块之外、用 flag 控制
        if (step + 1) % cfg.grad_accum == 0 and scheduler is not None:
            scheduler.step()

        loss_val = loss.item() * cfg.grad_accum
        total_loss += loss_val
        n_batches += 1

        # 记录 step-level 数据
        step_losses.append((global_step, loss_val))
        lr_now = optimizer.param_groups[0]["lr"]
        lr_history.append((global_step, lr_now))

        # 日志
        if (step + 1) % cfg.log_every == 0:
            avg = total_loss / n_batches
            pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr_now:.2e}")

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, step_losses, lr_history


def align_pred_to_gt(pred, gt, mask):
    """
    对 relative depth 做 least-squares scale-and-shift 对齐。

    目的:
      Depth Anything V2 输出的是 affine-invariant 的相对深度，
      不同图像之间的尺度和偏移不一致。
      需要对齐到 metric GT 后才能计算有意义的 loss。

    方法:
      对每个样本，在有效像素上求解:
        gt_i ≈ scale × pred_i + shift

      使用最小二乘法:
        [Σ(p²)  Σ(p)] [scale]   [Σ(p·g)]
        [Σ(p)   N   ] [shift] = [Σ(g)  ]

    参数:
      pred: (B, H, W) 模型原始输出
      gt:   (B, H, W) metric GT
      mask: (B, H, W) 有效掩码

    返回:
      aligned: (B, H, W) 对齐后的 metric depth
    """
    B = pred.shape[0]
    aligned = pred.clone()

    for i in range(B):
        m = mask[i] > 0
        n = m.sum().item()
        if n < 10:
            continue

        p = pred[i][m]
        g = gt[i][m]

        # 构建法方程
        p_sum = p.sum()
        g_sum = g.sum()
        pp_sum = (p * p).sum()
        pg_sum = (p * g).sum()

        # 求解 2×2 线性方程组
        det = pp_sum * n - p_sum * p_sum
        if det.abs() < 1e-8:
            # 退化情况: 所有预测值相同
            scale = g.median() / (p.median() + 1e-8)
            shift = torch.tensor(0.0, device=pred.device)
        else:
            scale = (pg_sum * n - p_sum * g_sum) / det
            shift = (pp_sum * g_sum - p_sum * pg_sum) / det

        # 确保 scale > 0 (深度应为正值)
        if scale <= 0:
            scale = g.median() / (p.median() + 1e-8)
            shift = torch.tensor(0.0, device=pred.device)

        aligned[i] = scale * pred[i] + shift

    return aligned.clamp(min=1e-3)


@torch.no_grad()
def validate(model, loader, criterion, cfg, epoch, device):
    """
    验证循环。

    除了 loss，还计算标准深度估计指标:
      AbsRel, RMSE, δ<1.25, δ<1.25², δ<1.25³
    """
    model.eval()
    total_loss = 0.0
    all_metrics = {"AbsRel": [], "RMSE": [], "delta1": [], "delta2": [], "delta3": []}
    n = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device)
        depth_gt = batch["depth_gt"].to(device)
        mask = batch["mask"].to(device)

        with amp_autocast(enabled=cfg.amp):
            outputs = model(pixel_values=pixel_values)
            pred_depth = outputs.predicted_depth

            if pred_depth.shape[-2:] != depth_gt.shape[-2:]:
                pred_depth = F.interpolate(
                    pred_depth.unsqueeze(1),
                    size=depth_gt.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                ).squeeze(1)

            pred_aligned = align_pred_to_gt(pred_depth, depth_gt, mask)

            weight = compute_distance_weight(
                depth_gt,
                start=cfg.depth_weight_decay_start,
                end=cfg.depth_weight_decay_end,
                min_w=cfg.depth_weight_min,
            )

            loss = criterion(pred_aligned, depth_gt, mask, weight)

        total_loss += loss.item()
        n += 1

        # 计算评估指标
        metrics = compute_metrics(pred_aligned, depth_gt, mask)
        for k in all_metrics:
            all_metrics[k].append(metrics[k])

    avg_loss = total_loss / max(n, 1)
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    avg_metrics["loss"] = avg_loss

    return avg_metrics


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  训练日志与可视化                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════╝

class TrainingLogger:
    """
    训练全过程记录器。

    功能:
      1. CSV 日志: 逐 epoch 记录 train_loss, val_loss, 所有指标, lr
      2. Step-level CSV: 逐步记录 loss 和 lr (用于绘制细粒度曲线)
      3. Loss 曲线图: train/val loss + 指标随 epoch 变化
      4. 学习率曲线图
      5. 训练结束后生成完整 summary

    输出结构:
      run_dir/
      ├── config.json           ← 完整超参数
      ├── epoch_log.csv         ← 逐 epoch 日志
      ├── step_log.csv          ← 逐步 loss + lr
      ├── plots/
      │   ├── loss_curve.png    ← train/val loss 曲线
      │   ├── metrics_curve.png ← AbsRel, RMSE, δ<1.25 曲线
      │   ├── lr_curve.png      ← 学习率曲线
      │   └── step_loss.png     ← 细粒度 step loss
      ├── val_vis/              ← 每个 epoch 的验证集可视化
      ├── best_model/           ← 最佳模型 (按 AbsRel)
      ├── final_model/          ← 最终模型
      └── training_summary.txt  ← 纯文本训练总结
    """

    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.plot_dir = os.path.join(run_dir, "plots")
        self.vis_dir = os.path.join(run_dir, "val_vis")
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)

        # epoch 级别记录
        self.epoch_records = []

        # step 级别记录
        self.all_step_losses = []
        self.all_lr_history = []

        # 写 epoch CSV 头
        self.epoch_csv = os.path.join(run_dir, "epoch_log.csv")
        with open(self.epoch_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "epoch", "train_loss", "val_loss",
                "AbsRel", "RMSE", "delta1", "delta2", "delta3",
                "lr", "time_sec", "best"
            ])

        # step CSV
        self.step_csv = os.path.join(run_dir, "step_log.csv")
        with open(self.step_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["global_step", "loss", "lr"])

    def log_epoch(self, epoch, train_loss, val_metrics, lr, elapsed, is_best):
        """记录一个 epoch 的数据。"""
        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics.get("loss", None) if val_metrics else None,
            "AbsRel": val_metrics.get("AbsRel", None) if val_metrics else None,
            "RMSE": val_metrics.get("RMSE", None) if val_metrics else None,
            "delta1": val_metrics.get("delta1", None) if val_metrics else None,
            "delta2": val_metrics.get("delta2", None) if val_metrics else None,
            "delta3": val_metrics.get("delta3", None) if val_metrics else None,
            "lr": lr,
            "time_sec": elapsed,
            "best": is_best,
        }
        self.epoch_records.append(record)

        # 追加写 CSV
        with open(self.epoch_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                record["epoch"],
                f"{record['train_loss']:.6f}",
                f"{record['val_loss']:.6f}" if record["val_loss"] is not None else "",
                f"{record['AbsRel']:.6f}" if record["AbsRel"] is not None else "",
                f"{record['RMSE']:.4f}" if record["RMSE"] is not None else "",
                f"{record['delta1']:.6f}" if record["delta1"] is not None else "",
                f"{record['delta2']:.6f}" if record["delta2"] is not None else "",
                f"{record['delta3']:.6f}" if record["delta3"] is not None else "",
                f"{record['lr']:.8f}",
                f"{record['time_sec']:.1f}",
                "*" if is_best else "",
            ])

    def log_steps(self, step_losses, lr_history):
        """记录 step-level 数据。"""
        self.all_step_losses.extend(step_losses)
        self.all_lr_history.extend(lr_history)

        # 追加写 step CSV
        with open(self.step_csv, "a", newline="") as f:
            w = csv.writer(f)
            for (gs, loss), (_, lr) in zip(step_losses, lr_history):
                w.writerow([gs, f"{loss:.6f}", f"{lr:.8f}"])

    def plot_all(self):
        """生成所有训练曲线图。"""
        try:
            import matplotlib
            matplotlib.use("Agg")  # 无头模式
            import matplotlib.pyplot as plt
        except ImportError:
            print("  [警告] matplotlib 未安装，跳过曲线绘制")
            print("  安装: pip install matplotlib")
            return

        records = self.epoch_records
        if not records:
            return

        epochs = [r["epoch"] for r in records]
        train_losses = [r["train_loss"] for r in records]
        val_losses = [r["val_loss"] for r in records if r["val_loss"] is not None]
        val_epochs = [r["epoch"] for r in records if r["val_loss"] is not None]

        # ---- 1. Loss 曲线 ----
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(epochs, train_losses, "b-o", markersize=4, label="Train loss", linewidth=1.5)
        if val_losses:
            ax.plot(val_epochs, val_losses, "r-s", markersize=4, label="Val loss", linewidth=1.5)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Training & Validation Loss", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.plot_dir, "loss_curve.png"), dpi=150)
        plt.close(fig)

        # ---- 2. 指标曲线 ----
        absrel = [r["AbsRel"] for r in records if r["AbsRel"] is not None]
        rmse = [r["RMSE"] for r in records if r["RMSE"] is not None]
        d1 = [r["delta1"] for r in records if r["delta1"] is not None]
        d2 = [r["delta2"] for r in records if r["delta2"] is not None]
        d3 = [r["delta3"] for r in records if r["delta3"] is not None]

        if absrel:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # AbsRel (越低越好)
            axes[0, 0].plot(val_epochs, absrel, "r-o", markersize=4, linewidth=1.5)
            axes[0, 0].set_title("AbsRel ↓", fontsize=12)
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].grid(True, alpha=0.3)

            # RMSE (越低越好)
            axes[0, 1].plot(val_epochs, rmse, "orange", marker="o", markersize=4, linewidth=1.5)
            axes[0, 1].set_title("RMSE (m) ↓", fontsize=12)
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].grid(True, alpha=0.3)

            # δ<1.25 (越高越好)
            axes[1, 0].plot(val_epochs, d1, "g-o", markersize=4, label="δ<1.25", linewidth=1.5)
            axes[1, 0].plot(val_epochs, d2, "b-s", markersize=4, label="δ<1.25²", linewidth=1.5)
            axes[1, 0].plot(val_epochs, d3, "m-^", markersize=4, label="δ<1.25³", linewidth=1.5)
            axes[1, 0].set_title("Threshold Accuracy ↑", fontsize=12)
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].legend(fontsize=10)
            axes[1, 0].grid(True, alpha=0.3)

            # 空白子图写 best 结果文本
            axes[1, 1].axis("off")
            best_idx = np.argmin(absrel)
            summary_text = (
                f"Best Results (Epoch {val_epochs[best_idx]})\n"
                f"{'─'*30}\n"
                f"AbsRel:   {absrel[best_idx]:.4f}\n"
                f"RMSE:     {rmse[best_idx]:.2f} m\n"
                f"δ<1.25:   {d1[best_idx]:.4f}\n"
                f"δ<1.25²:  {d2[best_idx]:.4f}\n"
                f"δ<1.25³:  {d3[best_idx]:.4f}"
            )
            axes[1, 1].text(0.1, 0.5, summary_text, fontsize=13,
                            fontfamily="monospace", verticalalignment="center",
                            transform=axes[1, 1].transAxes,
                            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))

            fig.suptitle("Validation Metrics over Training", fontsize=14, y=1.01)
            fig.tight_layout()
            fig.savefig(os.path.join(self.plot_dir, "metrics_curve.png"), dpi=150)
            plt.close(fig)

        # ---- 3. 学习率曲线 ----
        if self.all_lr_history:
            steps, lrs = zip(*self.all_lr_history)
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            ax.plot(steps, lrs, "purple", linewidth=1)
            ax.set_xlabel("Global Step", fontsize=12)
            ax.set_ylabel("Learning Rate", fontsize=12)
            ax.set_title("Learning Rate Schedule", fontsize=14)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(self.plot_dir, "lr_curve.png"), dpi=150)
            plt.close(fig)

        # ---- 4. Step-level loss (细粒度) ----
        if self.all_step_losses:
            steps, losses = zip(*self.all_step_losses)
            fig, ax = plt.subplots(1, 1, figsize=(12, 5))

            # 原始 step loss (半透明)
            ax.plot(steps, losses, color="steelblue", alpha=0.15, linewidth=0.5)

            # 滑动平均 (清晰趋势)
            window = min(50, len(losses) // 5 + 1)
            if window >= 3:
                kernel = np.ones(window) / window
                smooth = np.convolve(losses, kernel, mode="valid")
                smooth_steps = steps[window-1:]
                ax.plot(smooth_steps, smooth, color="darkblue", linewidth=1.5,
                        label=f"Moving avg (w={window})")

            ax.set_xlabel("Global Step", fontsize=12)
            ax.set_ylabel("Loss", fontsize=12)
            ax.set_title("Step-level Training Loss", fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(self.plot_dir, "step_loss.png"), dpi=150)
            plt.close(fig)

        print(f"  曲线图已保存到 {self.plot_dir}/")

    def save_summary(self, cfg, best_absrel, total_time):
        """保存纯文本训练总结。"""
        path = os.path.join(self.run_dir, "training_summary.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("训练总结\n")
            f.write("=" * 60 + "\n\n")

            f.write("【超参数】\n")
            f.write(f"  模型:          {cfg.model_name}\n")
            f.write(f"  LoRA rank:     {cfg.lora_r}\n")
            f.write(f"  LoRA alpha:    {cfg.lora_alpha}\n")
            f.write(f"  Decoder 微调:  {cfg.train_decoder}\n")
            f.write(f"  损失函数:      {cfg.loss_type}\n")
            f.write(f"  学习率:        {cfg.lr}\n")
            f.write(f"  Batch size:    {cfg.batch_size} × {cfg.grad_accum} (accum)\n")
            f.write(f"  Epochs:        {cfg.epochs}\n")
            f.write(f"  输入尺寸:      {cfg.input_size}\n")
            f.write(f"  距离衰减:      [{cfg.depth_weight_decay_start}, "
                    f"{cfg.depth_weight_decay_end}]m, min={cfg.depth_weight_min}\n")
            if hasattr(cfg, 'use_cass'):
                f.write(f"  CASS 模块:     {'启用' if cfg.use_cass else '禁用'}\n")
                if cfg.use_cass:
                    f.write(f"  CASS 密度核:   {cfg.cass_density_kernel}\n")
                    f.write(f"  CASS 平滑核:   {cfg.cass_smooth_kernel}\n")
                    f.write(f"  CASS 平滑σ:    {cfg.cass_smooth_sigma}\n")
            f.write(f"\n")

            f.write("【训练结果】\n")
            f.write(f"  最佳 AbsRel:   {best_absrel:.4f}\n")
            f.write(f"  总训练时间:    {total_time / 60:.1f} min\n")
            f.write(f"\n")

            # 逐 epoch 表格
            f.write("【Epoch 日志】\n")
            f.write(f"  {'Ep':>3} {'TrainLoss':>10} {'ValLoss':>10} "
                    f"{'AbsRel':>8} {'RMSE':>7} {'δ1':>7} {'δ2':>7} {'δ3':>7}  B\n")
            f.write("  " + "-" * 75 + "\n")
            for r in self.epoch_records:
                vl = f"{r['val_loss']:.4f}" if r['val_loss'] is not None else "   -   "
                ar = f"{r['AbsRel']:.4f}" if r['AbsRel'] is not None else "   -   "
                rm = f"{r['RMSE']:.2f}" if r['RMSE'] is not None else "  -   "
                d1 = f"{r['delta1']:.4f}" if r['delta1'] is not None else "   -   "
                d2 = f"{r['delta2']:.4f}" if r['delta2'] is not None else "   -   "
                d3 = f"{r['delta3']:.4f}" if r['delta3'] is not None else "   -   "
                best_mark = " *" if r['best'] else "  "
                f.write(f"  {r['epoch']:3d} {r['train_loss']:10.4f} {vl:>10} "
                        f"{ar:>8} {rm:>7} {d1:>7} {d2:>7} {d3:>7}{best_mark}\n")

            f.write("\n  (* = best model saved)\n")
            f.write("=" * 60 + "\n")


@torch.no_grad()
def visualize_val_predictions(model, val_ds, cfg, epoch, vis_dir, device,
                              num_samples=4, vis_max=80.0):
    """
    在验证集上采样几张图，保存 [RGB | 预测 | GT | 误差] 四联可视化。

    每个 epoch 保存一组，方便观察模型训练过程中的变化。
    """
    model.eval()
    indices = list(range(0, len(val_ds), max(1, len(val_ds) // num_samples)))[:num_samples]

    for i, idx in enumerate(indices):
        sample = val_ds[idx]
        pv = sample["pixel_values"].unsqueeze(0).to(device)
        gt = sample["depth_gt"].numpy()
        mask = sample["mask"].numpy()

        with amp_autocast(enabled=cfg.amp):
            out = model(pixel_values=pv)
            pred = out.predicted_depth.squeeze(0)

        # resize to GT size
        if pred.shape != gt.shape:
            pred = F.interpolate(
                pred.unsqueeze(0).unsqueeze(0),
                size=gt.shape, mode="bilinear", align_corners=False
            ).squeeze()

        pred_np = pred.float().cpu().numpy()   # .float() 防止 AMP float16 导致 lstsq 报错

        # scale-shift align
        valid = mask > 0
        if valid.sum() > 10:
            p, g = pred_np[valid], gt[valid]
            A = np.stack([p, np.ones_like(p)], axis=1)
            res = np.linalg.lstsq(A, g, rcond=None)
            s, sh = res[0]
            if s > 0:
                pred_np = s * pred_np + sh
            pred_np = np.clip(pred_np, 0.001, None)

        # 颜色映射
        def to_color(d, vmax):
            n = np.clip(d / vmax, 0, 1)
            u8 = (n * 255).astype(np.uint8)
            c = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
            c[u8 == 0] = 0
            return c

        # 反归一化 RGB
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        rgb = sample["pixel_values"].numpy().transpose(1, 2, 0)
        rgb = ((rgb * std + mean) * 255).clip(0, 255).astype(np.uint8)
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        pred_color = to_color(pred_np, vis_max)

        # GT with dilation
        gt_u8 = np.clip(gt / vis_max, 0, 1)
        gt_u8 = (gt_u8 * 255).astype(np.uint8)
        gt_u8 = cv2.dilate(gt_u8, np.ones((3, 3), np.uint8))
        gt_color = cv2.applyColorMap(gt_u8, cv2.COLORMAP_TURBO)
        gt_color[gt_u8 == 0] = 0

        # error map
        err = np.abs(pred_np - gt) * mask
        err_u8 = np.clip(err / 10.0, 0, 1)
        err_u8 = (err_u8 * 255).astype(np.uint8)
        err_color = cv2.applyColorMap(err_u8, cv2.COLORMAP_HOT)
        err_color[mask == 0] = [40, 40, 40]

        # 四联拼接
        panel = np.concatenate([rgb_bgr, pred_color, gt_color, err_color], axis=1)

        # 添加标签
        h = panel.shape[0]
        w4 = panel.shape[1] // 4
        font = cv2.FONT_HERSHEY_SIMPLEX
        labels = [f"RGB ({sample['stem']})", "Predicted", "GT (sparse)", "Error"]
        for j, label in enumerate(labels):
            cv2.putText(panel, label, (j * w4 + 5, 18), font, 0.5, (255, 255, 255), 1)

        out_path = os.path.join(vis_dir, f"epoch{epoch+1:03d}_sample{i}.jpg")
        cv2.imwrite(out_path, panel, [cv2.IMWRITE_JPEG_QUALITY, 90])


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  主训练流程                                                             ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def main():
    cfg = get_config()

    # ---- 随机种子 ----
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- 数据集 ----
    train_split = os.path.join(cfg.data_root, "splits", "train.txt")
    val_split = os.path.join(cfg.data_root, "splits", "val.txt")

    train_ds = OSDaR23DepthDataset(cfg.data_root, train_split, cfg.input_size, augment=True)
    val_ds = OSDaR23DepthDataset(cfg.data_root, val_split, cfg.input_size, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    # ---- 模型 ----
    model, processor = build_model(cfg)
    model = model.to(device)

    # ---- 损失函数 ----
    if cfg.loss_type == "silog":
        criterion = SparseSILogLoss(lambd=0.5)
    elif cfg.loss_type == "l1":
        criterion = SparseL1Loss()
    else:
        criterion = CombinedLoss()

    # ---- 优化器 ----
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # ---- 学习率调度 ----
    total_steps = len(train_loader) * cfg.epochs // cfg.grad_accum
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    # OneCycleLR 在 __init__ 时内部调用 step()，会触发 PyTorch 的
    # "lr_scheduler.step() before optimizer.step()" 警告。
    # 这是 OneCycleLR 的已知行为，实际不影响训练，在此抑制。
    warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*optimizer.step.*")

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        total_steps=total_steps,
        pct_start=cfg.warmup_ratio,
        anneal_strategy="cos",
    )

    # ---- 混合精度 ----
    scaler = amp_grad_scaler(enabled=cfg.amp)

    # ---- 输出目录与日志 ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    logger = TrainingLogger(run_dir)

    # 保存配置
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(cfg), f, indent=2, ensure_ascii=False)

    print(f"\n输出目录: {run_dir}")
    print(f"总训练步数: {total_steps}")
    print(f"Warmup 步数: {warmup_steps}")

    # ---- CASS 模块 ----
    cass_module = None
    if cfg.use_cass:
        from confidence_module import ConfidenceAwareSparseSupervision
        cass_module = ConfidenceAwareSparseSupervision(
            density_kernel=cfg.cass_density_kernel,
            smooth_kernel=cfg.cass_smooth_kernel,
            smooth_sigma=cfg.cass_smooth_sigma,
            dist_start=cfg.depth_weight_decay_start,
            dist_end=cfg.depth_weight_decay_end,
            dist_min=cfg.depth_weight_min,
            disable_density=cfg.cass_disable_density,
            disable_smooth=cfg.cass_disable_smooth,
            disable_distance=cfg.cass_disable_distance,
        )
        print(f"CASS 模块: 已启用")
        print(f"  density_kernel={cfg.cass_density_kernel}, "
              f"smooth_kernel={cfg.cass_smooth_kernel}, "
              f"smooth_sigma={cfg.cass_smooth_sigma}")
        if cfg.cass_disable_density or cfg.cass_disable_smooth or cfg.cass_disable_distance:
            disabled = []
            if cfg.cass_disable_density: disabled.append("density")
            if cfg.cass_disable_smooth: disabled.append("smooth")
            if cfg.cass_disable_distance: disabled.append("distance")
            print(f"  [消融] 已禁用因子: {', '.join(disabled)}")

        # 生成 CASS 置信度可视化（论文方法图用）
        try:
            from confidence_module import visualize_confidence
            sample = train_ds[0]
            vis_path = os.path.join(run_dir, "cass_confidence_vis.jpg")
            visualize_confidence(
                sample["depth_gt"].numpy(),
                sample["mask"].numpy(),
                save_path=vis_path,
            )
        except Exception as e:
            print(f"  CASS 可视化生成失败: {e}")
    else:
        print(f"CASS 模块: 已禁用 (仅距离衰减)")

    # ---- 训练主循环 ----
    best_absrel = float("inf")
    train_start_time = time.time()

    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)

    for epoch in range(cfg.epochs):
        t0 = time.time()

        # 训练
        train_loss, step_losses, lr_history = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            criterion, cfg, epoch, device, cass_module
        )

        # 记录 step-level 数据
        logger.log_steps(step_losses, lr_history)

        # 验证
        val_metrics = None
        is_best = False
        if (epoch + 1) % cfg.eval_every == 0:
            val_metrics = validate(model, val_loader, criterion, cfg, epoch, device)

            # 保存最佳模型
            if val_metrics["AbsRel"] < best_absrel:
                best_absrel = val_metrics["AbsRel"]
                best_path = os.path.join(run_dir, "best_model")
                save_full_model(model, best_path)
                is_best = True

            # 每个 epoch 保存验证集预测可视化
            visualize_val_predictions(
                model, val_ds, cfg, epoch, logger.vis_dir, device,
                num_samples=4, vis_max=cfg.depth_weight_decay_end
            )

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        # 记录 epoch 日志
        logger.log_epoch(epoch, train_loss, val_metrics, current_lr, elapsed, is_best)

        # 打印 epoch 摘要
        print(f"\nEpoch {epoch+1}/{cfg.epochs} ({elapsed:.0f}s)")
        print(f"  Train loss: {train_loss:.4f}")
        if val_metrics:
            print(f"  Val loss:   {val_metrics['loss']:.4f}")
            print(f"  AbsRel:     {val_metrics['AbsRel']:.4f}")
            print(f"  RMSE:       {val_metrics['RMSE']:.2f}m")
            print(f"  δ<1.25:     {val_metrics['delta1']:.4f}")
            print(f"  δ<1.25²:    {val_metrics['delta2']:.4f}")
            print(f"  δ<1.25³:    {val_metrics['delta3']:.4f}")
            if is_best:
                print(f"  ★ 新最佳! AbsRel={best_absrel:.4f}")

        # 定期保存 checkpoint
        if (epoch + 1) % cfg.save_every == 0:
            ckpt_path = os.path.join(run_dir, f"epoch_{epoch+1}")
            save_full_model(model, ckpt_path)
            print(f"  Checkpoint → {ckpt_path}")

        # 每个 epoch 更新曲线图 (方便中途查看进展)
        logger.plot_all()

    # ---- 保存最终模型 ----
    final_path = os.path.join(run_dir, "final_model")
    save_full_model(model, final_path)

    total_time = time.time() - train_start_time

    # ---- 生成最终报告和曲线 ----
    logger.plot_all()
    logger.save_summary(cfg, best_absrel, total_time)

    print(f"\n{'=' * 60}")
    print("训练完成!")
    print(f"{'=' * 60}")
    print(f"  总训练时间:    {total_time / 60:.1f} min")
    print(f"  最佳 AbsRel:   {best_absrel:.4f}")
    print(f"  最终模型:      {final_path}")
    print(f"  最佳模型:      {os.path.join(run_dir, 'best_model')}")
    print(f"  训练曲线:      {logger.plot_dir}/")
    print(f"  验证可视化:    {logger.vis_dir}/")
    print(f"  逐 epoch 日志: {logger.epoch_csv}")
    print(f"  逐步日志:      {logger.step_csv}")
    print(f"  训练总结:      {os.path.join(run_dir, 'training_summary.txt')}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
