"""Prepare RAIL-BENCH rail polyline annotations as heatmap training data.

The RAIL-BENCH rail annotations are COCO-like JSON files with polyline entries.
This script rasterizes left/right rail polylines into separate heatmaps and
ignore areas into a mask. It keeps the output deliberately simple so any later
detector can consume it without depending on the SLAM runtime.

Example:
    python scripts/prepare_railbench_rail.py \
        --root /root/GigaSLAM/railbench/object_and_rail \
        --out /root/GigaSLAM/railbench/object_and_rail/prepared_rail \
        --width 960 \
        --line-thickness 5 \
        --sigma 3
"""

import argparse
import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare RAIL-BENCH rail heatmaps.")
    parser.add_argument("--root", required=True, help="RAIL-BENCH object_and_rail root directory.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits to prepare.")
    parser.add_argument("--width", type=int, default=960, help="Output image width. Height keeps aspect ratio.")
    parser.add_argument("--line-thickness", type=int, default=5, help="Polyline thickness after resize.")
    parser.add_argument("--ignore-thickness", type=int, default=21, help="Fallback thickness for ignore polylines.")
    parser.add_argument("--sigma", type=float, default=3.0, help="Gaussian blur sigma for rail heatmaps; <=0 disables blur.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing split output.")
    parser.add_argument("--max-images", type=int, default=0, help="Debug limit per split; 0 means all.")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def group_annotations(data):
    grouped = defaultdict(list)
    for ann in data.get("annotations", []):
        grouped[int(ann["image_id"])].append(ann)
    return grouped


def image_records(data):
    return {int(img["id"]): img for img in data.get("images", [])}


def ensure_clean_split(out_split, overwrite):
    if out_split.exists():
        if not overwrite:
            raise FileExistsError(f"{out_split} exists; pass --overwrite to replace it.")
        shutil.rmtree(out_split)
    for sub in ["images", "left_heatmap", "right_heatmap", "ignore_mask"]:
        (out_split / sub).mkdir(parents=True, exist_ok=True)


def resize_image(image, out_width):
    h, w = image.shape[:2]
    scale = out_width / float(w)
    out_height = int(round(h * scale))
    resized = cv2.resize(image, (out_width, out_height), interpolation=cv2.INTER_AREA)
    return resized, scale, out_height


def scaled_polyline(polyline, scale):
    pts = np.asarray(polyline, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
        return None
    pts = np.round(pts * float(scale)).astype(np.int32)
    return pts.reshape((-1, 1, 2))


def draw_polyline(mask, polyline, scale, value, thickness):
    pts = scaled_polyline(polyline, scale)
    if pts is None:
        return False
    cv2.polylines(mask, [pts], isClosed=False, color=value, thickness=thickness, lineType=cv2.LINE_AA)
    return True


def draw_ignore(mask, ann, scale, value, thickness):
    if "segmentation" in ann and ann["segmentation"]:
        segs = ann["segmentation"]
        if isinstance(segs, list):
            for seg in segs:
                pts = np.asarray(seg, dtype=np.float32).reshape(-1, 2)
                if pts.shape[0] >= 3:
                    cv2.fillPoly(mask, [np.round(pts * scale).astype(np.int32)], value)
                    return True
    if "polyline" in ann:
        return draw_polyline(mask, ann["polyline"], scale, value, thickness)
    return False


def heatmap_postprocess(mask, sigma):
    if sigma <= 0:
        return mask
    blurred = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
    max_val = int(blurred.max())
    if max_val > 0:
        blurred = (blurred.astype(np.float32) * (255.0 / max_val)).clip(0, 255).astype(np.uint8)
    return blurred


def split_image_path(root, split, file_name):
    return root / "images" / split / file_name


def prepare_split(root, out_root, split, args):
    ann_path = root / "annotations" / "rails" / f"annotations_{split}.json"
    if not ann_path.exists():
        print(f"[prepare] skip {split}: annotation not found: {ann_path}")
        return

    data = load_json(ann_path)
    images = image_records(data)
    annotations = group_annotations(data)
    out_split = out_root / split
    ensure_clean_split(out_split, args.overwrite)

    manifest_path = out_split / "manifest.csv"
    stats = {
        "images": 0,
        "missing_images": 0,
        "left_polylines": 0,
        "right_polylines": 0,
        "ignore_items": 0,
    }

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id", "file_name", "split", "image",
                "left_heatmap", "right_heatmap", "ignore_mask",
                "orig_width", "orig_height", "out_width", "out_height",
                "left_count", "right_count", "ignore_count",
            ],
        )
        writer.writeheader()

        for i, image_id in enumerate(sorted(images)):
            if args.max_images and i >= args.max_images:
                break
            img = images[image_id]
            file_name = img["file_name"]
            src = split_image_path(root, split, file_name)
            if not src.exists():
                stats["missing_images"] += 1
                print(f"[prepare] missing image: {src}")
                continue

            image = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if image is None:
                stats["missing_images"] += 1
                print(f"[prepare] failed to read image: {src}")
                continue
            resized, scale, out_height = resize_image(image, args.width)

            left = np.zeros((out_height, args.width), dtype=np.uint8)
            right = np.zeros_like(left)
            ignore = np.zeros_like(left)
            left_count = right_count = ignore_count = 0

            for ann in annotations.get(image_id, []):
                category_id = int(ann.get("category_id", 0))
                if category_id == 1 and "polyline" in ann:
                    right_rail = int(ann.get("rightRail", 0)) == 1
                    target = right if right_rail else left
                    if draw_polyline(target, ann["polyline"], scale, 255, args.line_thickness):
                        if right_rail:
                            right_count += 1
                        else:
                            left_count += 1
                elif category_id == 2:
                    if draw_ignore(ignore, ann, scale, 255, args.ignore_thickness):
                        ignore_count += 1

            left = heatmap_postprocess(left, args.sigma)
            right = heatmap_postprocess(right, args.sigma)

            out_img = out_split / "images" / file_name
            left_path = out_split / "left_heatmap" / file_name
            right_path = out_split / "right_heatmap" / file_name
            ignore_path = out_split / "ignore_mask" / file_name

            cv2.imwrite(str(out_img), resized)
            cv2.imwrite(str(left_path), left)
            cv2.imwrite(str(right_path), right)
            cv2.imwrite(str(ignore_path), ignore)

            stats["images"] += 1
            stats["left_polylines"] += left_count
            stats["right_polylines"] += right_count
            stats["ignore_items"] += ignore_count
            writer.writerow({
                "image_id": image_id,
                "file_name": file_name,
                "split": split,
                "image": str(out_img.relative_to(out_split)),
                "left_heatmap": str(left_path.relative_to(out_split)),
                "right_heatmap": str(right_path.relative_to(out_split)),
                "ignore_mask": str(ignore_path.relative_to(out_split)),
                "orig_width": img.get("width", ""),
                "orig_height": img.get("height", ""),
                "out_width": args.width,
                "out_height": out_height,
                "left_count": left_count,
                "right_count": right_count,
                "ignore_count": ignore_count,
            })

            if stats["images"] % 100 == 0:
                print(f"[prepare] {split}: {stats['images']} images")

    print(
        f"[prepare] {split}: images={stats['images']} missing={stats['missing_images']} "
        f"left={stats['left_polylines']} right={stats['right_polylines']} "
        f"ignore={stats['ignore_items']} manifest={manifest_path}"
    )


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        prepare_split(root, out_root, split, args)


if __name__ == "__main__":
    main()
