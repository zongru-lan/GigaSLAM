"""Prepare pseudo ego-path labels from RAIL-BENCH rail polylines.

The object_and_rail annotations are COCO-like and contain many left/right rail
edge polylines. They do not directly label the train ego-path. This script
selects one plausible left/right rail pair per image and exports the JSON
format expected by train-ego-path-detection/src/utils/dataset.py:

    {
      "img_001_train.png": {
        "left_rail": [[x, y], ...],
        "right_rail": [[x, y], ...]
      }
    }

Low-confidence or ambiguous images are skipped so the first EgoPath model is
trained on conservative pseudo labels.
"""

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare pseudo ego-path labels.")
    parser.add_argument("--root", required=True, help="railbench/object_and_rail root.")
    parser.add_argument("--out", required=True, help="Output prepared_ego_path directory.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--anchors", type=int, default=64)
    parser.add_argument("--min-valid-anchors", type=int, default=14)
    parser.add_argument("--min-bottom-width-frac", type=float, default=0.025)
    parser.add_argument("--max-bottom-width-frac", type=float, default=0.30)
    parser.add_argument("--max-width-frac", type=float, default=0.80)
    parser.add_argument("--bottom-extrapolate-frac", type=float, default=0.08)
    parser.add_argument("--max-center-error", type=float, default=0.30)
    parser.add_argument("--max-width-cv", type=float, default=0.60)
    parser.add_argument("--min-score", type=float, default=2.5)
    parser.add_argument("--min-score-margin", type=float, default=0.10)
    parser.add_argument("--preview-count", type=int, default=0)
    parser.add_argument("--preview-width", type=int, default=1280)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-images", type=int, default=0, help="Debug limit per split; 0 means all.")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_out(out_root, overwrite):
    if out_root.exists():
        if not overwrite:
            raise FileExistsError(f"{out_root} exists; pass --overwrite to replace it.")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)


def grouped_annotations(data):
    grouped = defaultdict(list)
    for ann in data.get("annotations", []):
        grouped[int(ann["image_id"])].append(ann)
    return grouped


def finite_polyline(polyline):
    pts = np.asarray(polyline, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
        return None
    mask = np.isfinite(pts).all(axis=1)
    pts = pts[mask]
    if pts.shape[0] < 2:
        return None
    return pts


def x_at_y(polyline, y, bottom_extrapolate_px):
    pts = finite_polyline(polyline)
    if pts is None:
        return None
    y = float(y)
    candidates = []
    for p0, p1 in zip(pts[:-1], pts[1:]):
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        dy = y1 - y0
        if abs(dy) < 1e-6:
            if abs(y - y0) <= 0.75:
                candidates.append(0.5 * (x0 + x1))
            continue
        ymin = min(y0, y1) - 1e-6
        ymax = max(y0, y1) + 1e-6
        if ymin <= y <= ymax:
            t = (y - y0) / dy
            candidates.append(x0 + t * (x1 - x0))
    if candidates:
        return float(np.median(np.asarray(candidates, dtype=np.float32)))

    y_max = float(np.max(pts[:, 1]))
    if y <= y_max or y - y_max > bottom_extrapolate_px:
        return None

    best_segment = None
    best_y = -float("inf")
    for p0, p1 in zip(pts[:-1], pts[1:]):
        seg_ymax = max(float(p0[1]), float(p1[1]))
        if seg_ymax > best_y and abs(float(p1[1]) - float(p0[1])) > 1e-6:
            best_y = seg_ymax
            best_segment = (p0, p1)
    if best_segment is None:
        return None
    p0, p1 = best_segment
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    return float(x0 + (y - y0) * (x1 - x0) / (y1 - y0))


def sample_pair(left_ann, right_ann, image_width, image_height, args):
    y_values = np.linspace(image_height - 1, 0, args.anchors, dtype=np.float32)
    bottom_extrapolate_px = float(args.bottom_extrapolate_frac) * float(image_height)
    left_x = []
    right_x = []
    valid = []

    for y in y_values:
        lx = x_at_y(left_ann["polyline"], y, bottom_extrapolate_px)
        rx = x_at_y(right_ann["polyline"], y, bottom_extrapolate_px)
        ok = (
            lx is not None
            and rx is not None
            and np.isfinite(lx)
            and np.isfinite(rx)
            and -0.05 * image_width <= lx <= 1.05 * image_width
            and -0.05 * image_width <= rx <= 1.05 * image_width
            and lx < rx
        )
        left_x.append(float(lx) if lx is not None else float("nan"))
        right_x.append(float(rx) if rx is not None else float("nan"))
        valid.append(bool(ok))

    valid_count = 0
    for ok in valid:
        if not ok:
            break
        valid_count += 1
    if valid_count < args.min_valid_anchors:
        return None

    lx = np.asarray(left_x[:valid_count], dtype=np.float32)
    rx = np.asarray(right_x[:valid_count], dtype=np.float32)
    ys = np.asarray(y_values[:valid_count], dtype=np.float32)
    widths = rx - lx
    bottom_width = float(widths[0])
    max_width = float(np.max(widths))
    if bottom_width < args.min_bottom_width_frac * image_width:
        return None
    if bottom_width > args.max_bottom_width_frac * image_width:
        return None
    if max_width > args.max_width_frac * image_width:
        return None

    bottom_center = 0.5 * float(lx[0] + rx[0])
    center_err = abs(bottom_center - 0.5 * image_width) / max(float(image_width), 1.0)
    center_score = max(0.0, 1.0 - 2.0 * center_err)
    coverage_score = valid_count / max(float(args.anchors), 1.0)
    top_y = float(ys[-1])
    span_score = (float(image_height - 1) - top_y) / max(float(image_height - 1), 1.0)
    mean_width = float(np.mean(widths))
    width_cv = float(np.std(widths) / max(mean_width, 1.0))
    if center_err > args.max_center_error:
        return None
    if width_cv > args.max_width_cv:
        return None
    convergence = (bottom_width - float(widths[-1])) / max(bottom_width, 1.0)
    convergence_score = float(np.clip(convergence, 0.0, 1.0))
    score = (
        2.5 * center_score
        + 1.7 * coverage_score
        + 1.0 * span_score
        + 0.6 * convergence_score
        - 0.5 * width_cv
    )

    left_pts = np.column_stack([lx, ys])
    right_pts = np.column_stack([rx, ys])
    left_pts = clip_points(left_pts, image_width, image_height)[::-1]
    right_pts = clip_points(right_pts, image_width, image_height)[::-1]

    return {
        "score": float(score),
        "valid_anchors": int(valid_count),
        "coverage": float(coverage_score),
        "bottom_width": bottom_width,
        "bottom_center": bottom_center,
        "center_error": float(center_err),
        "width_cv": width_cv,
        "left_points": left_pts,
        "right_points": right_pts,
        "left_ann_id": left_ann.get("id"),
        "right_ann_id": right_ann.get("id"),
    }


def clip_points(points, image_width, image_height):
    points = np.asarray(points, dtype=np.float32).copy()
    points[:, 0] = np.clip(points[:, 0], 0, image_width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, image_height - 1)
    return points


def select_ego_pair(annotations, image_width, image_height, args):
    left_anns = [
        a for a in annotations
        if int(a.get("category_id", 0)) == 1 and int(a.get("rightRail", 0)) == 0 and "polyline" in a
    ]
    right_anns = [
        a for a in annotations
        if int(a.get("category_id", 0)) == 1 and int(a.get("rightRail", 0)) == 1 and "polyline" in a
    ]
    candidates = []
    for left_ann in left_anns:
        for right_ann in right_anns:
            sampled = sample_pair(left_ann, right_ann, image_width, image_height, args)
            if sampled is not None:
                candidates.append(sampled)
    candidates.sort(key=lambda item: item["score"], reverse=True)
    if not candidates:
        return None, "no_valid_pair", 0, None
    best = candidates[0]
    second_score = candidates[1]["score"] if len(candidates) > 1 else -float("inf")
    margin = best["score"] - second_score if math.isfinite(second_score) else float("inf")
    best["second_score"] = second_score if math.isfinite(second_score) else ""
    best["score_margin"] = margin if math.isfinite(margin) else ""
    if best["score"] < args.min_score:
        return None, "low_score", len(candidates), best
    if len(candidates) > 1 and margin < args.min_score_margin:
        return None, "ambiguous_pair", len(candidates), best
    return best, "ok", len(candidates), best


def points_to_json(points):
    return [[int(round(float(x))), int(round(float(y)))] for x, y in points]


def draw_polyline(vis, polyline, color, thickness):
    pts = finite_polyline(polyline)
    if pts is None:
        return
    pts = np.round(pts).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(vis, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def save_preview(root, split, image_record, annotations, selected, reject_reason, out_path, preview_width):
    image_path = root / "images" / split / image_record["file_name"]
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return
    for ann in annotations:
        if int(ann.get("category_id", 0)) != 1 or "polyline" not in ann:
            continue
        color = (96, 96, 96)
        if int(ann.get("rightRail", 0)) == 0:
            color = (120, 80, 80)
        elif int(ann.get("rightRail", 0)) == 1:
            color = (80, 120, 80)
        draw_polyline(image, ann["polyline"], color, 2)

    if selected is not None:
        left = np.round(selected["left_points"]).astype(np.int32).reshape((-1, 1, 2))
        right = np.round(selected["right_points"]).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [left], False, (0, 0, 255), 8, lineType=cv2.LINE_AA)
        cv2.polylines(image, [right], False, (0, 255, 255), 8, lineType=cv2.LINE_AA)
        text = f"selected score={selected['score']:.2f} anchors={selected['valid_anchors']}"
    else:
        text = f"rejected: {reject_reason}"
    cv2.putText(image, text, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 6, cv2.LINE_AA)
    cv2.putText(image, text, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 2, cv2.LINE_AA)

    if preview_width > 0 and image.shape[1] > preview_width:
        scale = preview_width / float(image.shape[1])
        image = cv2.resize(image, (preview_width, int(round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)


def process_split(root, out_root, split, args):
    ann_path = root / "annotations" / "rails" / f"annotations_{split}.json"
    if not ann_path.exists():
        print(f"[ego-path] skip {split}: missing {ann_path}")
        return

    data = load_json(ann_path)
    images = {int(img["id"]): img for img in data.get("images", [])}
    annotations = grouped_annotations(data)
    out_annotations = {}
    quality_rows = []
    reject_counts = defaultdict(int)
    preview_saved = 0

    for i, image_id in enumerate(sorted(images)):
        if args.max_images and i >= args.max_images:
            break
        image_record = images[image_id]
        file_name = image_record["file_name"]
        image_width = int(image_record["width"])
        image_height = int(image_record["height"])
        anns = annotations.get(image_id, [])
        selected, status, candidate_count, best_seen = select_ego_pair(
            anns, image_width, image_height, args
        )
        if selected is not None:
            out_annotations[file_name] = {
                "left_rail": points_to_json(selected["left_points"]),
                "right_rail": points_to_json(selected["right_points"]),
            }
        else:
            reject_counts[status] += 1

        row = {
            "image_id": image_id,
            "file_name": file_name,
            "split": split,
            "status": status,
            "accepted": selected is not None,
            "candidate_count": candidate_count,
            "score": "",
            "second_score": "",
            "score_margin": "",
            "valid_anchors": "",
            "coverage": "",
            "bottom_width": "",
            "bottom_center": "",
            "center_error": "",
            "width_cv": "",
            "left_ann_id": "",
            "right_ann_id": "",
        }
        if best_seen is not None:
            for key in [
                "score", "second_score", "score_margin", "valid_anchors", "coverage",
                "bottom_width", "bottom_center", "center_error", "width_cv",
                "left_ann_id", "right_ann_id",
            ]:
                row[key] = best_seen.get(key, "")
        quality_rows.append(row)

        if args.preview_count and preview_saved < args.preview_count:
            preview_path = out_root / "preview" / split / f"{Path(file_name).stem}_egopath.jpg"
            save_preview(root, split, image_record, anns, selected, status, preview_path, args.preview_width)
            preview_saved += 1

    out_json = out_root / f"annotations_{split}_egopath.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_annotations, f, indent=2)

    quality_path = out_root / f"label_quality_{split}.csv"
    with open(quality_path, "w", newline="", encoding="utf-8") as f:
        fields = list(quality_rows[0].keys()) if quality_rows else ["image_id", "file_name", "status"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(quality_rows)

    print(
        f"[ego-path] {split}: images={len(quality_rows)} accepted={len(out_annotations)} "
        f"rejected={len(quality_rows) - len(out_annotations)} output={out_json}"
    )
    if reject_counts:
        print("[ego-path] reject counts:", dict(sorted(reject_counts.items())))


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    out_root = Path(args.out).resolve()
    ensure_out(out_root, args.overwrite)
    with open(out_root / "prepare_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    for split in args.splits:
        process_split(root, out_root, split, args)


if __name__ == "__main__":
    main()
