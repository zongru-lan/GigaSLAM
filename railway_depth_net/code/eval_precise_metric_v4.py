"""
================================================================================
Precise evaluation for relative + metric depth (v3)
================================================================================

用途:
  1. 同时评估 aligned relative metrics 与 metric absolute metrics
  2. 生成 range-wise report（近/中/远距离分段）
  3. 输出逐帧 CSV，方便论文附录与错误分析

说明:
  - aligned protocol: 对 relative depth 做 per-sample alignment 后评估
  - metric protocol : 对 absolute depth 直接评估，不做 alignment
  - 当 checkpoint 不含 metric_head.pt 时，只报告 aligned 指标
  
python eval_precise_metric_v4.py     --lora_path runs_Stage_B_ablation/full/best_metric_model     --output_dir eval_results_with_edge/ours
python eval_precise_metric_v4.py     --lora_path runs_Stage_B_ablation/floor_0.20/best_metric_model     --output_dir eval_results_with_edge/Range-aware-weight
python eval_precise_metric_v4.py     --lora_path runs_Stage_B/best_metric_model     --output_dir eval_results_with_edge/Sparse-metric-loss-only
================================================================================
"""

import os
import sys
import csv
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForDepthEstimation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train_depth_anything_lora as base_impl
from metric_depth_module_v4 import sanitize_depth_tensor
from train_depth_anything_lora_metric_v4 import load_full_model, forward_branches
from edge_metrics import compute_edge_aware_metrics


@dataclass
class PixelMetricAccumulator:
    """按像素精确累计误差，避免先逐帧再平均带来的偏差。"""
    absrel_sum: float = 0.0
    sqerr_sum: float = 0.0
    delta1_sum: float = 0.0
    delta2_sum: float = 0.0
    delta3_sum: float = 0.0
    count: int = 0

    def update(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor):
        valid = mask > 0
        if valid.sum().item() < 1:
            return
        p = sanitize_depth_tensor(pred)[valid]
        g = sanitize_depth_tensor(gt)[valid]
        ratio = torch.max(p / g.clamp(min=1e-6), g / p.clamp(min=1e-6))
        self.absrel_sum += float((torch.abs(p - g) / g.clamp(min=1e-6)).sum().item())
        self.sqerr_sum += float(((p - g) ** 2).sum().item())
        self.delta1_sum += float((ratio < 1.25).float().sum().item())
        self.delta2_sum += float((ratio < 1.25 ** 2).float().sum().item())
        self.delta3_sum += float((ratio < 1.25 ** 3).float().sum().item())
        self.count += int(valid.sum().item())

    def summary(self) -> Optional[Dict[str, float]]:
        if self.count == 0:
            return None
        return {
            "AbsRel": self.absrel_sum / self.count,
            "RMSE": float(np.sqrt(self.sqerr_sum / self.count)),
            "delta1": self.delta1_sum / self.count,
            "delta2": self.delta2_sum / self.count,
            "delta3": self.delta3_sum / self.count,
            "count": self.count,
        }


def get_args():
    p = argparse.ArgumentParser(description="Precise metric evaluation v4")
    p.add_argument("--model_name", type=str,
                   default="depth-anything/Depth-Anything-V2-Small-hf")
    p.add_argument("--lora_path", type=str, required=True,
                   help="训练输出的 best_relative_model / best_metric_model / final_model 路径")
    p.add_argument("--data_root", type=str, default="OSDaR23/train_data")
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--input_size", type=int, default=518)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--compare_zeroshot", action="store_true")
    p.add_argument("--output_dir", type=str, default="eval_precise_metric_v3")
    p.add_argument("--amp", action="store_true", default=False,
                   help="默认关闭 AMP，优先保证带 metric head 时的评估稳定")
    p.add_argument("--range_bins", type=str, default="0,30,60,100",
                   help="近中远距离分段，格式如 0,30,60,100；会自动补一个 [last, +inf) 段")
    return p.parse_args()


def parse_range_bins(spec: str) -> List[Tuple[float, float]]:
    vals = [float(x) for x in spec.split(",") if x.strip()]
    vals = sorted(vals)
    if len(vals) < 2:
        vals = [0.0, 30.0, 60.0, 100.0]
    bins = []
    for a, b in zip(vals[:-1], vals[1:]):
        bins.append((a, b))
    bins.append((vals[-1], float("inf")))
    return bins


def range_name(lo: float, hi: float) -> str:
    if np.isinf(hi):
        return f"[{int(lo)},+inf)"
    return f"[{int(lo)},{int(hi)})"


@torch.no_grad()
def compute_frame_row(stem: str,
                      pred_aligned: torch.Tensor,
                      pred_metric: Optional[torch.Tensor],
                      gt: torch.Tensor,
                      mask: torch.Tensor) -> Dict[str, Optional[float]]:
    row = {
        "stem": stem,
        "aligned_AbsRel": None,
        "aligned_RMSE": None,
        "aligned_delta1": None,
        "aligned_delta2": None,
        "aligned_delta3": None,
        "metric_AbsRel": None,
        "metric_RMSE": None,
        "metric_delta1": None,
        "metric_delta2": None,
        "metric_delta3": None,
    }
    valid = mask > 0
    if valid.sum().item() < 10:
        return row

    pa = sanitize_depth_tensor(pred_aligned)[valid]
    g = sanitize_depth_tensor(gt)[valid]
    ratio_a = torch.max(pa / g.clamp(min=1e-6), g / pa.clamp(min=1e-6))
    row.update({
        "aligned_AbsRel": float((torch.abs(pa - g) / g.clamp(min=1e-6)).mean().item()),
        "aligned_RMSE": float(torch.sqrt(((pa - g) ** 2).mean()).item()),
        "aligned_delta1": float((ratio_a < 1.25).float().mean().item()),
        "aligned_delta2": float((ratio_a < 1.25 ** 2).float().mean().item()),
        "aligned_delta3": float((ratio_a < 1.25 ** 3).float().mean().item()),
    })

    if pred_metric is not None:
        pm = sanitize_depth_tensor(pred_metric)[valid]
        ratio_m = torch.max(pm / g.clamp(min=1e-6), g / pm.clamp(min=1e-6))
        row.update({
            "metric_AbsRel": float((torch.abs(pm - g) / g.clamp(min=1e-6)).mean().item()),
            "metric_RMSE": float(torch.sqrt(((pm - g) ** 2).mean()).item()),
            "metric_delta1": float((ratio_m < 1.25).float().mean().item()),
            "metric_delta2": float((ratio_m < 1.25 ** 2).float().mean().item()),
            "metric_delta3": float((ratio_m < 1.25 ** 3).float().mean().item()),
        })
    return row


@torch.no_grad()
def evaluate_model(base_model, metric_head, loader, device, amp: bool, bins: List[Tuple[float, float]], label="Model"):
    base_model.eval()
    if metric_head is not None:
        metric_head.eval()

    aligned_acc = PixelMetricAccumulator()
    metric_acc = PixelMetricAccumulator()
    aligned_range = {range_name(lo, hi): PixelMetricAccumulator() for lo, hi in bins}
    metric_range = {range_name(lo, hi): PixelMetricAccumulator() for lo, hi in bins}
    per_frame = []

    # 边界误差累计
    edge_metrics_list = []

    pbar = tqdm(loader, desc=f"Eval [{label}]", leave=True)
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device)
        depth_gt = batch["depth_gt"].to(device)
        mask = batch["mask"].to(device)
        stems = batch["stem"]

        use_amp = amp and (metric_head is None)
        with base_impl.amp_autocast(enabled=use_amp):
            pred_rel, pred_metric, _aux = forward_branches(
                base_model, metric_head, pixel_values, depth_gt.shape[-2:],
                compute_metric=(metric_head is not None)
            )

        pred_aligned = base_impl.align_pred_to_gt(pred_rel, depth_gt, mask)
        pred_aligned = sanitize_depth_tensor(pred_aligned)
        if pred_metric is not None:
            pred_metric = sanitize_depth_tensor(pred_metric)

        B = pixel_values.shape[0]
        for i in range(B):
            aligned_acc.update(pred_aligned[i], depth_gt[i], mask[i])
            if pred_metric is not None:
                metric_acc.update(pred_metric[i], depth_gt[i], mask[i])

                # 计算边界误差（仅对 metric depth）
                pred_np = pred_metric[i].squeeze().cpu().numpy()
                gt_np = depth_gt[i].squeeze().cpu().numpy()
                mask_np = mask[i].squeeze().cpu().numpy().astype(np.uint8)

                edge_m = compute_edge_aware_metrics(pred_np, gt_np, mask_np)
                if not np.isnan(edge_m['Edge_AbsRel']):
                    edge_metrics_list.append(edge_m)

            for lo, hi in bins:
                rmask = mask[i] * ((depth_gt[i] >= lo) & (depth_gt[i] < hi)).float()
                name = range_name(lo, hi)
                aligned_range[name].update(pred_aligned[i], depth_gt[i], rmask)
                if pred_metric is not None:
                    metric_range[name].update(pred_metric[i], depth_gt[i], rmask)

            per_frame.append(compute_frame_row(
                stems[i], pred_aligned[i], None if pred_metric is None else pred_metric[i], depth_gt[i], mask[i]
            ))

    aligned_summary = aligned_acc.summary()
    metric_summary = metric_acc.summary() if metric_acc.count > 0 else None
    aligned_range_summary = {k: v.summary() for k, v in aligned_range.items()}
    metric_range_summary = {k: v.summary() for k, v in metric_range.items()} if metric_summary is not None else None

    # 计算平均边界误差
    edge_summary = None
    if edge_metrics_list:
        edge_summary = {
            'Edge_AbsRel': np.mean([m['Edge_AbsRel'] for m in edge_metrics_list]),
            'NonEdge_AbsRel': np.mean([m['NonEdge_AbsRel'] for m in edge_metrics_list]),
            'Edge_Ratio': np.mean([m['Edge_Ratio'] for m in edge_metrics_list if not np.isnan(m['Edge_Ratio'])]),
        }

    return aligned_summary, metric_summary, aligned_range_summary, metric_range_summary, per_frame, edge_summary


@torch.no_grad()
def evaluate_zero_shot(model, loader, device, amp: bool, bins: List[Tuple[float, float]], label="Zero-shot"):
    model.eval()
    aligned_acc = PixelMetricAccumulator()
    aligned_range = {range_name(lo, hi): PixelMetricAccumulator() for lo, hi in bins}

    pbar = tqdm(loader, desc=f"Eval [{label}]", leave=True)
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device)
        depth_gt = batch["depth_gt"].to(device)
        mask = batch["mask"].to(device)

        with base_impl.amp_autocast(enabled=amp):
            pred = model(pixel_values=pixel_values).predicted_depth
            if pred.shape[-2:] != depth_gt.shape[-2:]:
                pred = F.interpolate(pred.unsqueeze(1), size=depth_gt.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)

        pred_aligned = sanitize_depth_tensor(base_impl.align_pred_to_gt(pred, depth_gt, mask))
        B = pixel_values.shape[0]
        for i in range(B):
            aligned_acc.update(pred_aligned[i], depth_gt[i], mask[i])
            for lo, hi in bins:
                rmask = mask[i] * ((depth_gt[i] >= lo) & (depth_gt[i] < hi)).float()
                aligned_range[range_name(lo, hi)].update(pred_aligned[i], depth_gt[i], rmask)

    return aligned_acc.summary(), {k: v.summary() for k, v in aligned_range.items()}


def print_metrics_block(title: str, metrics: Optional[Dict[str, float]]):
    print(f"\n[{title}]")
    if metrics is None:
        print("  N/A")
        return
    for k in ["AbsRel", "RMSE", "delta1", "delta2", "delta3"]:
        print(f"  {k:<10} {metrics[k]:.4f}")
    if "count" in metrics:
        print(f"  {'count':<10} {metrics['count']}")


def write_text_summary(path: str,
                       ft_aligned: Dict[str, float],
                       ft_metric: Optional[Dict[str, float]],
                       ft_aligned_range: Dict[str, Optional[Dict[str, float]]],
                       ft_metric_range: Optional[Dict[str, Optional[Dict[str, float]]]],
                       zs_aligned: Optional[Dict[str, float]] = None,
                       zs_aligned_range: Optional[Dict[str, Optional[Dict[str, float]]]] = None):
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write("Metric recovery evaluation summary (v4)\n")
        f.write("=" * 72 + "\n\n")

        def _write_block(name, metrics):
            f.write(f"[{name}]\n")
            if metrics is None:
                f.write("  N/A\n\n")
                return
            for k in ["AbsRel", "RMSE", "delta1", "delta2", "delta3", "count"]:
                if k in metrics:
                    f.write(f"  {k:<10} {metrics[k]:.4f}\n" if isinstance(metrics[k], float) else f"  {k:<10} {metrics[k]}\n")
            f.write("\n")

        _write_block("Fine-tuned | aligned protocol", ft_aligned)
        _write_block("Fine-tuned | metric protocol | no alignment", ft_metric)
        if zs_aligned is not None:
            _write_block("Zero-shot | aligned protocol", zs_aligned)

        f.write("[Range-wise | Fine-tuned aligned]\n")
        for name, metrics in ft_aligned_range.items():
            if metrics is None:
                f.write(f"  {name}: N/A\n")
            else:
                f.write(f"  {name}: AbsRel={metrics['AbsRel']:.4f}, RMSE={metrics['RMSE']:.4f}, delta1={metrics['delta1']:.4f}, count={metrics['count']}\n")
        f.write("\n")

        if ft_metric_range is not None:
            f.write("[Range-wise | Fine-tuned metric]\n")
            for name, metrics in ft_metric_range.items():
                if metrics is None:
                    f.write(f"  {name}: N/A\n")
                else:
                    f.write(f"  {name}: AbsRel={metrics['AbsRel']:.4f}, RMSE={metrics['RMSE']:.4f}, delta1={metrics['delta1']:.4f}, count={metrics['count']}\n")
            f.write("\n")

        if zs_aligned_range is not None:
            f.write("[Range-wise | Zero-shot aligned]\n")
            for name, metrics in zs_aligned_range.items():
                if metrics is None:
                    f.write(f"  {name}: N/A\n")
                else:
                    f.write(f"  {name}: AbsRel={metrics['AbsRel']:.4f}, RMSE={metrics['RMSE']:.4f}, delta1={metrics['delta1']:.4f}, count={metrics['count']}\n")


def write_range_csv(path: str,
                    aligned_range: Dict[str, Optional[Dict[str, float]]],
                    metric_range: Optional[Dict[str, Optional[Dict[str, float]]]],
                    zs_range: Optional[Dict[str, Optional[Dict[str, float]]]] = None):
    rows = []
    for name, metrics in aligned_range.items():
        row = {"range": name, "protocol": "aligned_finetuned"}
        if metrics is not None:
            row.update(metrics)
        rows.append(row)
    if metric_range is not None:
        for name, metrics in metric_range.items():
            row = {"range": name, "protocol": "metric_finetuned"}
            if metrics is not None:
                row.update(metrics)
            rows.append(row)
    if zs_range is not None:
        for name, metrics in zs_range.items():
            row = {"range": name, "protocol": "aligned_zeroshot"}
            if metrics is not None:
                row.update(metrics)
            rows.append(row)

    fieldnames = ["range", "protocol", "AbsRel", "RMSE", "delta1", "delta2", "delta3", "count"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_zero_shot(model_name, device):
    return AutoModelForDepthEstimation.from_pretrained(model_name).to(device).eval()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bins = parse_range_bins(args.range_bins)

    print("=" * 72)
    print("精确评估：aligned relative + metric absolute + range-wise report")
    print("=" * 72)

    split_file = os.path.join(args.data_root, "splits", f"{args.split}.txt")
    ds = base_impl.OSDaR23DepthDataset(args.data_root, split_file, args.input_size, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    print("\n加载 fine-tuned 模型...")
    base_model, metric_head = load_full_model(args.model_name, args.lora_path, device=device)

    print("\n评估 fine-tuned...")
    ft_aligned, ft_metric, ft_aligned_range, ft_metric_range, per_frame, ft_edge = evaluate_model(
        base_model, metric_head, loader, device, amp=args.amp, bins=bins, label="Fine-tuned"
    )

    zs_aligned = None
    zs_aligned_range = None
    if args.compare_zeroshot:
        print("\n加载 zero-shot baseline...")
        zs_model = load_zero_shot(args.model_name, device)
        print("评估 zero-shot...")
        zs_aligned, zs_aligned_range = evaluate_zero_shot(
            zs_model, loader, device, amp=args.amp, bins=bins, label="Zero-shot"
        )

    print_metrics_block("Fine-tuned | aligned protocol", ft_aligned)
    print_metrics_block("Fine-tuned | metric protocol | no alignment", ft_metric)
    if ft_edge is not None:
        print("\n" + "=" * 60)
        print("边界感知误差 (Edge-Aware Metrics):")
        print("=" * 60)
        print(f"  Edge AbsRel:    {ft_edge['Edge_AbsRel']:.4f}")
        print(f"  NonEdge AbsRel: {ft_edge['NonEdge_AbsRel']:.4f}")
        print(f"  Edge/NonEdge Ratio: {ft_edge['Edge_Ratio']:.4f}")
        print("=" * 60)
    if zs_aligned is not None:
        print_metrics_block("Zero-shot | aligned protocol", zs_aligned)

    print("\n[Range-wise | Fine-tuned metric]")
    if ft_metric_range is not None:
        for name, metrics in ft_metric_range.items():
            if metrics is None:
                print(f"  {name:<12} N/A")
            else:
                print(f"  {name:<12} AbsRel={metrics['AbsRel']:.4f}  RMSE={metrics['RMSE']:.2f}  delta1={metrics['delta1']:.4f}  count={metrics['count']}")
    else:
        print("  当前 checkpoint 不含 metric head，仅有 aligned 指标。")

    text_path = os.path.join(args.output_dir, "metrics_summary.txt")
    write_text_summary(text_path, ft_aligned, ft_metric, ft_aligned_range, ft_metric_range, zs_aligned, zs_aligned_range)

    csv_path = os.path.join(args.output_dir, "per_frame_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_frame[0].keys())
        writer.writeheader()
        writer.writerows(per_frame)

    range_csv = os.path.join(args.output_dir, "range_metrics.csv")
    write_range_csv(range_csv, ft_aligned_range, ft_metric_range, zs_aligned_range)

    # 保存边界误差
    if ft_edge is not None:
        edge_json = os.path.join(args.output_dir, "edge_metrics.json")
        with open(edge_json, 'w') as f:
            json.dump(ft_edge, f, indent=2)
        print(f"\n💾 边界误差已保存到: {edge_json}")

    json_path = os.path.join(args.output_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "fine_tuned_aligned": ft_aligned,
                "fine_tuned_metric": ft_metric,
                "fine_tuned_aligned_range": ft_aligned_range,
                "fine_tuned_metric_range": ft_metric_range,
                "zero_shot_aligned": zs_aligned,
                "zero_shot_aligned_range": zs_aligned_range,
            },
            f, indent=2, ensure_ascii=False,
        )

    print(f"\n汇总 → {text_path}")
    print(f"逐帧 CSV → {csv_path}")
    print(f"Range CSV → {range_csv}")
    print(f"JSON → {json_path}")
    print(f"\n全部输出 → {args.output_dir}/")


if __name__ == "__main__":
    main()
