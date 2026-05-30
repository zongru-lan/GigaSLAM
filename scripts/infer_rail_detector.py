"""Run the trained rail detector on an image sequence and write diagnostics.

This is an offline sanity-check step before wiring the detector into SLAM scale
correction. It does not modify SLAM outputs.

Example:
    python scripts/infer_rail_detector.py \
        --checkpoint /autodl-fs/data/GigaSLAM/railbench/rail_detector_runs/segformer_b0/best.pt \
        --config configs/railway/rowtrack350/scenes/scene_16_train.yaml \
        --out /autodl-fs/data/GigaSLAM/railbench/rail_detector_runs/segformer_b0/scene_16_diag \
        --device cuda \
        --save-overlays \
        --overlay-every 10
"""

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.rail_detector_model import build_segformer_rail_detector  # noqa: E402


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Infer rail detector and write sequence diagnostics.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt/last.pt from train_rail_detector.py.")
    parser.add_argument("--images", default=None, help="Image directory. If omitted, use Dataset.color_path from --config.")
    parser.add_argument("--config", default=None, help="GigaSLAM YAML config used to resolve Dataset.color_path.")
    parser.add_argument("--out", required=True, help="Output directory for diagnostics.")
    parser.add_argument("--model-name", default=None, help="Override model name; defaults to checkpoint args.")
    parser.add_argument("--image-size", nargs=2, type=int, default=None, metavar=("W", "H"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--roi-y-min", type=float, default=0.45)
    parser.add_argument("--roi-y-max", type=float, default=0.95)
    parser.add_argument("--sample-y-fracs", nargs="+", type=float, default=[0.58, 0.65, 0.72, 0.80, 0.88])
    parser.add_argument("--min-peak", type=float, default=0.25)
    parser.add_argument("--min-width-px", type=float, default=12.0)
    parser.add_argument("--center-x-frac", type=float, default=0.50)
    parser.add_argument("--peak-topk", type=int, default=8)
    parser.add_argument("--peak-nms-px", type=int, default=24)
    parser.add_argument("--max-width-px-frac", type=float, default=0.45)
    parser.add_argument("--max-width-jump-ratio", type=float, default=0.12)
    parser.add_argument("--max-center-jump-ratio", type=float, default=0.18)
    parser.add_argument("--max-images", type=int, default=0, help="Debug limit; 0 means all images.")
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--overlay-every", type=int, default=10)
    parser.add_argument("--save-heatmaps", action="store_true")
    return parser.parse_args()


def load_image_dir(args):
    if args.images:
        return args.images
    if not args.config:
        raise ValueError("Provide --images or --config.")
    try:
        import yaml
    except Exception as exc:
        raise ImportError("Reading --config requires pyyaml.") from exc
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    image_dir = cfg.get("Dataset", {}).get("color_path")
    if not image_dir:
        raise KeyError("Dataset.color_path not found in config.")
    return image_dir


def list_images(image_dir, max_images=0):
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(os.path.join(image_dir, pattern)))
    paths = sorted(paths)
    if max_images > 0:
        paths = paths[:max_images]
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return paths


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def preprocess_bgr(image_bgr, image_size):
    out_w, out_h = image_size
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor


def smooth_1d(values, kernel=9):
    if kernel <= 1:
        return values
    kernel = int(kernel)
    if kernel % 2 == 0:
        kernel += 1
    pad = kernel // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    filt = np.ones(kernel, dtype=np.float32) / kernel
    return np.convolve(padded, filt, mode="valid")


def rail_width_diagnostics(probs, args, prev_width=None, prev_center=None):
    """Return simple per-frame quality signals from two rail heatmaps."""
    left = probs[0]
    right = probs[1]
    h, w = left.shape
    roi_y0 = int(np.clip(args.roi_y_min, 0.0, 1.0) * h)
    roi_y1 = int(np.clip(args.roi_y_max, 0.0, 1.0) * h)
    roi_y1 = max(roi_y1, roi_y0 + 1)

    left_roi = left[roi_y0:roi_y1]
    right_roi = right[roi_y0:roi_y1]
    left_conf = float(left_roi.max(initial=0.0))
    right_conf = float(right_roi.max(initial=0.0))
    left_area = int((left_roi >= args.threshold).sum())
    right_area = int((right_roi >= args.threshold).sum())

    widths = []
    centers = []
    row_points = []
    target_center = prev_center if prev_center is not None else float(w) * args.center_x_frac
    width_prior = prev_width
    for frac in sorted(args.sample_y_fracs, reverse=True):
        y = int(np.clip(frac, 0.0, 1.0) * h)
        y0 = max(0, y - 2)
        y1 = min(h, y + 3)
        lp = smooth_1d(left[y0:y1].mean(axis=0))
        rp = smooth_1d(right[y0:y1].mean(axis=0))
        pair = select_pair_at_row(lp, rp, w, args, target_center, width_prior)
        if pair is None:
            row_points.append((y, 0, 0, 0.0, 0.0, False))
            continue
        lx, rx, lpeak, rpeak = pair
        width = float(rx - lx)
        valid = lpeak >= args.min_peak and rpeak >= args.min_peak and width >= args.min_width_px
        if valid:
            widths.append(width)
            centers.append(0.5 * (float(lx) + float(rx)))
            if prev_center is None and len(centers) == 1:
                target_center = centers[-1]
            if prev_width is None and len(widths) == 1:
                width_prior = widths[-1]
        row_points.append((y, lx, rx, lpeak, rpeak, valid))

    if widths:
        widths_arr = np.asarray(widths, dtype=np.float32)
        width_med = float(np.median(widths_arr))
        width_mad = float(np.median(np.abs(widths_arr - width_med)))
        centers_arr = np.asarray(centers, dtype=np.float32)
        center_med = float(np.median(centers_arr))
        center_mad = float(np.median(np.abs(centers_arr - center_med)))
    else:
        width_med = 0.0
        width_mad = 0.0
        center_med = 0.0
        center_mad = 0.0

    width_jump = False
    center_jump = False
    if prev_width is not None and width_med > 0:
        width_jump = abs(width_med - prev_width) / max(prev_width, 1.0) > args.max_width_jump_ratio
    if prev_center is not None and center_med > 0:
        center_jump = abs(center_med - prev_center) / max(float(w), 1.0) > args.max_center_jump_ratio

    usable = bool(
        left_conf >= args.min_peak
        and right_conf >= args.min_peak
        and len(widths) >= max(2, len(args.sample_y_fracs) // 2)
        and not width_jump
        and not center_jump
    )
    return {
        "left_conf": left_conf,
        "right_conf": right_conf,
        "left_area": left_area,
        "right_area": right_area,
        "width_med_px": width_med,
        "width_mad_px": width_mad,
        "center_med_px": center_med,
        "center_mad_px": center_mad,
        "valid_rows": len(widths),
        "width_jump": width_jump,
        "center_jump": center_jump,
        "usable": usable,
        "row_points": row_points,
    }


def select_pair_at_row(left_profile, right_profile, width, args, target_center, width_prior=None):
    left_peaks = top_profile_peaks(left_profile, args.min_peak, args.peak_topk, args.peak_nms_px)
    right_peaks = top_profile_peaks(right_profile, args.min_peak, args.peak_topk, args.peak_nms_px)
    if not left_peaks or not right_peaks:
        return None

    max_width = float(width) * args.max_width_px_frac
    best = None
    for lx, lpeak in left_peaks:
        for rx, rpeak in right_peaks:
            if rx <= lx:
                continue
            pair_width = float(rx - lx)
            if pair_width < args.min_width_px or pair_width > max_width:
                continue
            center = 0.5 * (float(lx) + float(rx))
            center_err = abs(center - target_center) / max(float(width), 1.0)
            straddle_bonus = 0.25 if lx <= target_center <= rx else 0.0
            width_penalty = 0.0
            if width_prior is not None:
                width_penalty = abs(pair_width - width_prior) / max(width_prior, 1.0)
            score = float(lpeak) + float(rpeak) + straddle_bonus - 1.8 * center_err - 0.7 * width_penalty
            if best is None or score > best[0]:
                best = (score, int(lx), int(rx), float(lpeak), float(rpeak))
    if best is None:
        return None
    return best[1], best[2], best[3], best[4]


def top_profile_peaks(profile, threshold=0.25, topk=8, nms_px=24):
    values = np.asarray(profile, dtype=np.float32)
    if values.size == 0:
        return []
    order = np.argsort(values)[::-1]
    peaks = []
    for idx in order:
        score = float(values[idx])
        if score < threshold:
            break
        if any(abs(int(idx) - int(prev_idx)) < nms_px for prev_idx, _ in peaks):
            continue
        peaks.append((int(idx), score))
        if len(peaks) >= topk:
            break
    return peaks


def make_overlay(image_bgr, probs, diag):
    h, w = probs.shape[-2:]
    image = cv2.resize(image_bgr, (w, h), interpolation=cv2.INTER_AREA)
    overlay = image.copy()
    left = np.clip(probs[0] * 255, 0, 255).astype(np.uint8)
    right = np.clip(probs[1] * 255, 0, 255).astype(np.uint8)
    heat = np.zeros_like(image)
    heat[..., 2] = left
    heat[..., 1] = right
    overlay = cv2.addWeighted(overlay, 0.65, heat, 0.85, 0)
    for y, lx, rx, _lpeak, _rpeak, valid in diag["row_points"]:
        color = (255, 255, 255) if valid else (128, 128, 128)
        cv2.circle(overlay, (lx, y), 4, color, -1)
        cv2.circle(overlay, (rx, y), 4, color, -1)
        cv2.line(overlay, (lx, y), (rx, y), color, 1)
    return overlay


def save_heatmaps(path, probs):
    left = np.clip(probs[0] * 255, 0, 255).astype(np.uint8)
    right = np.clip(probs[1] * 255, 0, 255).astype(np.uint8)
    color = np.zeros((left.shape[0], left.shape[1], 3), dtype=np.uint8)
    color[..., 2] = left
    color[..., 1] = right
    cv2.imwrite(str(path), color)


@torch.no_grad()
def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_dir / "overlays"
    heatmap_dir = out_dir / "heatmaps"
    if args.save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    if args.save_heatmaps:
        heatmap_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, device)
    ckpt_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    model_name = args.model_name or ckpt_args.get("model_name") or "nvidia/segformer-b0-finetuned-ade-512-512"
    image_size = tuple(args.image_size or ckpt_args.get("image_size") or [768, 432])

    model = build_segformer_rail_detector(model_name=model_name, num_channels=2, pretrained=False).to(device)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()

    image_dir = load_image_dir(args)
    image_paths = list_images(image_dir, args.max_images)
    diag_path = out_dir / "rail_detector_diag.csv"
    fields = [
        "frame_id",
        "file_name",
        "left_conf",
        "right_conf",
        "left_area",
        "right_area",
        "width_med_px",
        "width_mad_px",
        "center_med_px",
        "center_mad_px",
        "valid_rows",
        "width_jump",
        "center_jump",
        "usable",
    ]

    usable_count = 0
    prev_width = None
    prev_center = None
    with open(diag_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for start in range(0, len(image_paths), args.batch_size):
            batch_paths = image_paths[start:start + args.batch_size]
            images = []
            tensors = []
            for path in batch_paths:
                image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
                if image_bgr is None:
                    raise FileNotFoundError(f"Failed to read image: {path}")
                images.append(image_bgr)
                tensors.append(preprocess_bgr(image_bgr, image_size))

            pixel_values = torch.stack(tensors).to(device)
            logits = model(pixel_values=pixel_values).logits
            logits = torch.nn.functional.interpolate(
                logits,
                size=(image_size[1], image_size[0]),
                mode="bilinear",
                align_corners=False,
            )
            probs_batch = torch.sigmoid(logits).detach().cpu().numpy()

            for local_i, (path, image_bgr, probs) in enumerate(zip(batch_paths, images, probs_batch)):
                frame_id = start + local_i
                diag = rail_width_diagnostics(probs, args, prev_width=prev_width, prev_center=prev_center)
                if diag["usable"]:
                    prev_width = diag["width_med_px"]
                    prev_center = diag["center_med_px"]
                usable_count += int(diag["usable"])
                row = {k: diag[k] for k in fields if k not in ("frame_id", "file_name")}
                row["frame_id"] = frame_id
                row["file_name"] = os.path.basename(path)
                writer.writerow(row)

                should_save_overlay = args.save_overlays and (
                    args.overlay_every <= 1 or frame_id % args.overlay_every == 0
                )
                if should_save_overlay:
                    overlay = make_overlay(image_bgr, probs, diag)
                    cv2.imwrite(str(overlay_dir / os.path.basename(path)), overlay)
                if args.save_heatmaps:
                    save_heatmaps(heatmap_dir / os.path.basename(path), probs)

            print(
                f"[rail-infer] {min(start + len(batch_paths), len(image_paths))}/{len(image_paths)} "
                f"usable={usable_count}"
            )

    print(f"[rail-infer] wrote {diag_path}")
    print(f"[rail-infer] usable_frames={usable_count}/{len(image_paths)}")


if __name__ == "__main__":
    main()
