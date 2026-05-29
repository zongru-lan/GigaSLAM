"""
Single-image inference with the railway depth network.

Usage:
    python infer_single.py --image /path/to/image.png --ckpt ../ckpt --output depth_vis.jpg
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_depth_anything_lora_metric_v4 import load_full_model, forward_branches

MODEL_NAME = "./Depth-Anything-V2-Small-hf"
INPUT_SIZE = 518


def preprocess(image_path, device):
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    return pixel_values, orig_w, orig_h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--ckpt", default="../ckpt", help="Checkpoint directory")
    parser.add_argument("--output", default="depth_vis.jpg", help="Output visualization path")
    parser.add_argument("--max_depth", type=float, default=60.0, help="Max depth for colormap (meters)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model...")
    model, metric_head = load_full_model(MODEL_NAME, args.ckpt, device=device)
    model.eval()
    if metric_head is not None:
        metric_head.eval()

    print("Preprocessing image...")
    pixel_values, orig_w, orig_h = preprocess(args.image, device)
    target_hw = (INPUT_SIZE, INPUT_SIZE)

    print("Running inference...")
    with torch.no_grad():
        with torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu"):
            pred_rel, pred_metric, _ = forward_branches(
                model, metric_head, pixel_values, target_hw, compute_metric=(metric_head is not None)
            )

    # Use metric depth if available, else relative depth
    if pred_metric is not None:
        depth_np = pred_metric.squeeze().float().cpu().numpy()
        label = "metric"
    else:
        depth_np = pred_rel.squeeze().float().cpu().numpy()
        label = "relative"

    print(f"Depth type: {label}, range: [{depth_np.min():.2f}, {depth_np.max():.2f}] m")

    # Upsample to original resolution
    depth_full = cv2.resize(depth_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # Region statistics: top half (sky/far) vs bottom half (ground/rail)
    h = depth_full.shape[0]
    top = depth_full[:h//2, :]
    bot = depth_full[h//2:, :]
    print(f"  全图:     min={depth_full.min():.2f}  max={depth_full.max():.2f}  mean={depth_full.mean():.2f} m")
    print(f"  上半部分: min={top.min():.2f}  max={top.max():.2f}  mean={top.mean():.2f} m")
    print(f"  下半部分: min={bot.min():.2f}  max={bot.max():.2f}  mean={bot.mean():.2f} m  (铁轨/地面)")

    # Colormap visualization (jet, near=red, far=blue)
    depth_u8 = (np.clip(depth_full, 0, args.max_depth) / args.max_depth * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)

    # Side-by-side: original RGB | depth colormap
    rgb_bgr = cv2.imread(args.image)
    rgb_bgr = cv2.resize(rgb_bgr, (orig_w, orig_h))
    combined = np.hstack([rgb_bgr, depth_color])
    cv2.imwrite(args.output, combined)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
