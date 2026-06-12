#!/usr/bin/env python3
"""Evaluate a trained TEP-style EgoPath model on pseudo ego-path labels.

This script is for model-level inspection before wiring EgoPathGuidance into
GigaSLAM. It runs inference on original images, compares predictions against
the prepared pseudo ego-path labels, and saves overlay images for human review.
"""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate EgoPath model overlays and metrics.")
    parser.add_argument("--tep-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--runtime", default="pytorch", choices=["pytorch", "tensorrt"])
    parser.add_argument("--crop", default="none", help="'none', 'auto', or x0,y0,x1,y1")
    parser.add_argument("--sample-rows", type=int, default=32)
    parser.add_argument("--save-first", type=int, default=80)
    parser.add_argument("--save-worst", type=int, default=80)
    parser.add_argument("--save-random", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smooth",
        default="none",
        choices=["none", "moving_average", "savgol"],
        help="Optional smoothing applied to predicted rails before metrics/overlays.",
    )
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--smooth-polyorder", type=int, default=2)
    return parser.parse_args()


def parse_crop(value):
    if value in ("", "none", "None", "null"):
        return None
    if value == "auto":
        return "auto"
    parts = [int(v) for v in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--crop must be none, auto, or x0,y0,x1,y1")
    return tuple(parts)


def import_detector(tep_root):
    tep_root = str(Path(tep_root).resolve())
    if tep_root not in sys.path:
        sys.path.insert(0, tep_root)
    from src.utils.interface import Detector

    return Detector


def rails_to_polygon_mask(rails, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)
    if rails is None or len(rails) != 2:
        return mask.astype(bool)
    left = np.asarray(rails[0], dtype=np.float32)
    right = np.asarray(rails[1], dtype=np.float32)
    if len(left) < 2 or len(right) < 2:
        return mask.astype(bool)
    pts = np.concatenate([left, right[::-1]], axis=0)
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask.astype(bool)


def rail_x_at_y(rail, ys):
    pts = np.asarray(rail, dtype=np.float32)
    if len(pts) < 2:
        return np.full_like(ys, np.nan, dtype=np.float32)
    order = np.argsort(pts[:, 1])
    pts = pts[order]
    y = pts[:, 1]
    x = pts[:, 0]
    keep = np.concatenate([[True], np.diff(y) > 1e-3])
    y = y[keep]
    x = x[keep]
    if len(y) < 2:
        return np.full_like(ys, np.nan, dtype=np.float32)
    out = np.interp(ys, y, x)
    out[(ys < y.min()) | (ys > y.max())] = np.nan
    return out.astype(np.float32)


def line_metrics(pred, gt, width, height, sample_rows):
    result = {
        "valid_rows": 0,
        "center_mae_px": float("nan"),
        "width_mae_px": float("nan"),
        "bottom_center_error_px": float("nan"),
        "bottom_width_error_px": float("nan"),
        "pred_bottom_width_px": float("nan"),
        "gt_bottom_width_px": float("nan"),
        "pred_crossing": True,
    }
    if pred is None or gt is None or len(pred) != 2 or len(gt) != 2:
        return result

    pred_left = np.asarray(pred[0], dtype=np.float32)
    pred_right = np.asarray(pred[1], dtype=np.float32)
    gt_left = np.asarray(gt[0], dtype=np.float32)
    gt_right = np.asarray(gt[1], dtype=np.float32)
    if min(len(pred_left), len(pred_right), len(gt_left), len(gt_right)) < 2:
        return result

    y0 = max(pred_left[:, 1].min(), pred_right[:, 1].min(), gt_left[:, 1].min(), gt_right[:, 1].min())
    y1 = min(pred_left[:, 1].max(), pred_right[:, 1].max(), gt_left[:, 1].max(), gt_right[:, 1].max())
    if y1 <= y0:
        return result
    ys = np.linspace(y0, y1, int(sample_rows), dtype=np.float32)
    pl = rail_x_at_y(pred_left, ys)
    pr = rail_x_at_y(pred_right, ys)
    gl = rail_x_at_y(gt_left, ys)
    gr = rail_x_at_y(gt_right, ys)
    valid = np.isfinite(pl) & np.isfinite(pr) & np.isfinite(gl) & np.isfinite(gr)
    valid &= (pr > pl) & (gr > gl)
    result["valid_rows"] = int(valid.sum())
    result["pred_crossing"] = bool(np.any(np.isfinite(pl) & np.isfinite(pr) & (pr <= pl)))
    if valid.sum() == 0:
        return result

    pred_center = 0.5 * (pl[valid] + pr[valid])
    gt_center = 0.5 * (gl[valid] + gr[valid])
    pred_width = pr[valid] - pl[valid]
    gt_width = gr[valid] - gl[valid]
    result["center_mae_px"] = float(np.mean(np.abs(pred_center - gt_center)))
    result["width_mae_px"] = float(np.mean(np.abs(pred_width - gt_width)))
    result["bottom_center_error_px"] = float(abs(pred_center[-1] - gt_center[-1]))
    result["bottom_width_error_px"] = float(abs(pred_width[-1] - gt_width[-1]))
    result["pred_bottom_width_px"] = float(pred_width[-1])
    result["gt_bottom_width_px"] = float(gt_width[-1])
    return result


def compute_metrics(pred, gt, width, height, sample_rows):
    pred_mask = rails_to_polygon_mask(pred, width, height)
    gt_mask = rails_to_polygon_mask(gt, width, height)
    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = float(inter / union) if union > 0 else 0.0
    m = line_metrics(pred, gt, width, height, sample_rows)
    m["iou"] = iou
    return m


def smooth_rails(rails, method="savgol", window=9, polyorder=2):
    """Smooth predicted left/right rails while preserving a valid rail pair.

    The smoothing is applied to centerline and width instead of left/right
    independently. This suppresses high-frequency anchor jitter and avoids
    creating left/right crossings.
    """
    if method in (None, "none"):
        return rails
    if rails is None or len(rails) != 2:
        return rails
    left = np.asarray(rails[0], dtype=np.float32)
    right = np.asarray(rails[1], dtype=np.float32)
    n = min(len(left), len(right))
    if n < 3:
        return rails
    left = left[:n].copy()
    right = right[:n].copy()
    y = 0.5 * (left[:, 1] + right[:, 1])
    order = np.argsort(y)
    y_sorted = y[order]
    left_x = left[:, 0][order]
    right_x = right[:, 0][order]

    center = 0.5 * (left_x + right_x)
    width = np.maximum(right_x - left_x, 1e-3)
    center_s = _smooth_series(center, method, window, polyorder)
    width_s = _smooth_series(width, method, window, polyorder)
    width_s = np.maximum(width_s, max(float(np.percentile(width, 5)) * 0.5, 1e-3))

    left_s = np.column_stack([center_s - 0.5 * width_s, y_sorted])
    right_s = np.column_stack([center_s + 0.5 * width_s, y_sorted])
    return [left_s.tolist(), right_s.tolist()]


def _smooth_series(values, method, window, polyorder):
    values = np.asarray(values, dtype=np.float32)
    n = len(values)
    if n < 3:
        return values
    win = int(window)
    if win % 2 == 0:
        win += 1
    win = max(3, min(win, n if n % 2 == 1 else n - 1))
    if win < 3:
        return values
    if method == "savgol":
        try:
            from scipy.signal import savgol_filter

            order = max(1, min(int(polyorder), win - 1))
            return savgol_filter(values, win, order, mode="interp").astype(np.float32)
        except Exception:
            pass
    pad = win // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def draw_polyline(draw, rail, color, width):
    pts = [tuple(map(float, p)) for p in rail]
    if len(pts) >= 2:
        draw.line(pts, fill=color, width=width, joint="curve")


def save_overlay(image_path, gt, pred, metrics, out_path, label="Pred"):
    img = Image.open(image_path).convert("RGB")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(overlay)
    if gt and len(gt) == 2 and len(gt[0]) >= 2 and len(gt[1]) >= 2:
        gt_poly = [tuple(p) for p in gt[0]] + [tuple(p) for p in gt[1][::-1]]
        mask_draw.polygon(gt_poly, fill=(0, 180, 0, 45))
    if pred and len(pred) == 2 and len(pred[0]) >= 2 and len(pred[1]) >= 2:
        pred_poly = [tuple(p) for p in pred[0]] + [tuple(p) for p in pred[1][::-1]]
        mask_draw.polygon(pred_poly, fill=(255, 0, 0, 40))
    vis = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(vis)
    for rail in gt:
        draw_polyline(draw, rail, (0, 210, 0), 5)
    for rail in pred:
        draw_polyline(draw, rail, (255, 0, 0), 4)
    text = (
        f"GT=green  {label}=red  IoU={metrics['iou']:.3f}  "
        f"centerMAE={metrics['center_mae_px']:.1f}px  widthMAE={metrics['width_mae_px']:.1f}px"
    )
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 26)
    except OSError:
        font = None
    draw.rectangle((8, 8, 8 + 1040, 48), fill=(255, 255, 255))
    draw.text((18, 14), text, fill=(0, 0, 0), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis.save(out_path, quality=92)


def summarize(rows):
    ious = np.asarray([r["iou"] for r in rows], dtype=np.float32)
    centers = np.asarray([r["center_mae_px"] for r in rows if np.isfinite(r["center_mae_px"])], dtype=np.float32)
    widths = np.asarray([r["width_mae_px"] for r in rows if np.isfinite(r["width_mae_px"])], dtype=np.float32)
    return {
        "num_images": int(len(rows)),
        "mean_iou": float(np.mean(ious)) if len(ious) else 0.0,
        "median_iou": float(np.median(ious)) if len(ious) else 0.0,
        "p10_iou": float(np.percentile(ious, 10)) if len(ious) else 0.0,
        "p25_iou": float(np.percentile(ious, 25)) if len(ious) else 0.0,
        "num_iou_ge_0_90": int((ious >= 0.90).sum()),
        "num_iou_ge_0_80": int((ious >= 0.80).sum()),
        "num_iou_lt_0_70": int((ious < 0.70).sum()),
        "mean_center_mae_px": float(np.mean(centers)) if len(centers) else None,
        "median_center_mae_px": float(np.median(centers)) if len(centers) else None,
        "mean_width_mae_px": float(np.mean(widths)) if len(widths) else None,
        "median_width_mae_px": float(np.median(widths)) if len(widths) else None,
        "num_crossing": int(sum(bool(r["pred_crossing"]) for r in rows)),
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    Detector = import_detector(args.tep_root)
    detector = Detector(args.model_path, parse_crop(args.crop), args.runtime, args.device)
    annotations = json.load(open(args.annotations, encoding="utf-8"))
    names = sorted(annotations)
    images_dir = Path(args.images)

    rows = []
    predictions = {}
    for name in names:
        image_path = images_dir / name
        img = Image.open(image_path).convert("RGB")
        pred = detector.detect(img)
        pred = smooth_rails(pred, args.smooth, args.smooth_window, args.smooth_polyorder)
        gt = [annotations[name]["left_rail"], annotations[name]["right_rail"]]
        metrics = compute_metrics(pred, gt, img.width, img.height, args.sample_rows)
        row = {"image": name, **metrics}
        rows.append(row)
        predictions[name] = {"pred": pred, "gt": gt, "metrics": metrics}

    rows_sorted = sorted(rows, key=lambda r: r["image"])
    with open(out_dir / "per_image_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        writer.writeheader()
        writer.writerows(rows_sorted)

    summary = summarize(rows)
    summary["smooth"] = args.smooth
    summary["smooth_window"] = args.smooth_window
    summary["smooth_polyorder"] = args.smooth_polyorder
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    first = rows_sorted[: args.save_first]
    worst = sorted(rows, key=lambda r: r["iou"])[: args.save_worst]
    random_rows = random.sample(rows, min(args.save_random, len(rows)))
    groups = [("first", first), ("worst_iou", worst), ("random", random_rows)]
    for group, selected in groups:
        for row in selected:
            name = row["image"]
            pred_gt = predictions[name]
            stem = Path(name).stem
            save_overlay(
                images_dir / name,
                pred_gt["gt"],
                pred_gt["pred"],
                pred_gt["metrics"],
                out_dir / group / f"{stem}_iou{row['iou']:.3f}.jpg",
                label=("Pred" if args.smooth == "none" else f"Pred {args.smooth}"),
            )

    print(json.dumps(summary, indent=2))
    print(f"Wrote metrics and overlays to {out_dir}")


if __name__ == "__main__":
    main()
