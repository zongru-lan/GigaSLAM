#!/usr/bin/env python3
"""Correlate RailScale / depth diagnostics with trajectory errors.

This is a post-run diagnostic tool. It does not change SLAM outputs or
re-evaluate poses. It reads the current result folders and writes derived CSV
and plot artifacts for debugging depth-scale drift.
"""

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from evo.core import metrics, trajectory
from evo.core.trajectory import PosePath3D


PER_FRAME_FIELDS = [
    "scene",
    "run",
    "frame_id",
    "trajectory_error_m",
    "lateral_error_m",
    "along_error_m",
    "rail_status",
    "rail_width_m",
    "rail_confidence",
    "rail_samples",
    "rail_raw_scale",
    "rail_applied_scale",
    "rail_scale_step",
    "rail_pixel_width",
    "rail_pixel_width_mad",
    "rail_pixel_center",
    "rail_pixel_center_mad",
    "row_total",
    "row_selected",
    "row_selected_ratio",
    "row_reject_reasons",
    "row_depth_median",
    "row_depth_lr_rel_diff_median",
    "row_width_3d_median",
    "row_width_3d_mad",
    "row_scale_raw_median",
    "row_scale_raw_mad",
    "row_center_shift_max_px",
    "vo_method",
    "vo_matches",
    "vo_pnp_inlier_ratio",
    "vo_step_norm",
    "vo_depth_median",
    "vo_depth_valid_ratio",
    "rendered_depth_available",
    "rendered_depth_valid_ratio",
    "rendered_depth_median",
    "input_depth_median",
    "depth_gs_residual_median",
    "depth_gs_residual_p95",
    "depth_gs_rel_residual_median",
    "depth_gs_rel_residual_p95",
    "risk_flags",
    "risk_score",
]


SUMMARY_FIELDS = [
    "scene",
    "run",
    "frames",
    "ate_rmse",
    "error_p95",
    "max_error",
    "max_error_frame_id",
    "corrected_ratio",
    "hold_ratio",
    "rail_scale_median",
    "rail_scale_p05",
    "rail_scale_p95",
    "rail_scale_step_p95",
    "rail_width_median",
    "rail_width_p05",
    "rail_width_p95",
    "row_selected_ratio",
    "row_scale_raw_mad_median",
    "row_depth_lr_rel_diff_p95",
    "vo_depth_median_p50",
    "vo_depth_median_p95",
    "vo_step_norm_p95",
    "corr_error_vs_scale",
    "corr_error_vs_vo_depth",
    "corr_error_vs_row_depth",
    "depth_gs_residual_median_p50",
    "depth_gs_residual_p95",
    "corr_error_vs_depth_gs_residual",
    "rendered_depth_available",
]


SHAPE_SUMMARY_FIELDS = [
    "scene",
    "run",
    "ate_rmse",
    "ate_legacy_ordered_rmse",
    "length_ratio",
    "end_error",
    "max_error",
    "max_error_frame_id",
    "lat_p95",
    "along_p95",
    "first25_rmse",
    "mid50_rmse",
    "last25_rmse",
]


def as_float(value, default=float("nan")):
    if value is None:
        return default
    if isinstance(value, (float, int)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def as_int(value, default=None):
    f = as_float(value)
    if math.isnan(f):
        return default
    return int(round(f))


def as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def finite(values):
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    return arr


def percentile(values, q):
    arr = finite(values)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def median(values):
    return percentile(values, 50)


def mad(values):
    arr = finite(values)
    if arr.size == 0:
        return float("nan")
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


def corrcoef(x_values, y_values):
    pairs = [
        (x, y)
        for x, y in zip(x_values, y_values)
        if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)
    ]
    if len(pairs) < 3:
        return float("nan")
    x = np.asarray([p[0] for p in pairs], dtype=float)
    y = np.asarray([p[1] for p in pairs], dtype=float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def load_csv_by_frame(path):
    if not path.exists():
        return {}
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            frame_id = as_int(row.get("frame_id"))
            if frame_id is not None:
                rows[frame_id] = row
    return rows


def load_rows_aggregate(path):
    if not path.exists():
        return {}
    grouped = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            frame_id = as_int(row.get("frame_id"))
            if frame_id is not None:
                grouped[frame_id].append(row)

    aggregated = {}
    for frame_id, rows in grouped.items():
        selected_rows = [row for row in rows if as_bool(row.get("selected"))]
        reason_counts = Counter(
            row.get("reject_reason", "").strip()
            for row in rows
            if row.get("reject_reason", "").strip()
        )
        depth_values = []
        depth_lr_rel_diff = []
        widths = []
        row_scales = []
        center_shifts = []
        for row in selected_rows:
            dl = as_float(row.get("depth_left"))
            dr = as_float(row.get("depth_right"))
            if np.isfinite(dl) and dl > 0:
                depth_values.append(dl)
            if np.isfinite(dr) and dr > 0:
                depth_values.append(dr)
            if np.isfinite(dl) and np.isfinite(dr) and dl > 0 and dr > 0:
                denom = max(0.5 * (dl + dr), 1e-6)
                depth_lr_rel_diff.append(abs(dl - dr) / denom)
            widths.append(as_float(row.get("width_3d")))
            row_scales.append(as_float(row.get("row_scale_raw")))
            center_shifts.append(as_float(row.get("row_center_shift")))

        aggregated[frame_id] = {
            "row_total": len(rows),
            "row_selected": len(selected_rows),
            "row_selected_ratio": len(selected_rows) / len(rows) if rows else float("nan"),
            "row_reject_reasons": ";".join(
                f"{name}:{count}" for name, count in sorted(reason_counts.items())
            ),
            "row_depth_median": median(depth_values),
            "row_depth_lr_rel_diff_median": median(depth_lr_rel_diff),
            "row_width_3d_median": median(widths),
            "row_width_3d_mad": mad(widths),
            "row_scale_raw_median": median(row_scales),
            "row_scale_raw_mad": mad(row_scales),
            "row_center_shift_max_px": percentile([abs(v) for v in center_shifts], 100),
        }
    return aggregated


def load_pose_list(values):
    return [np.asarray(pose, dtype=float) for pose in values]


def trajectory_diagnostics(trj_path):
    data = json.load(open(trj_path, encoding="utf-8"))
    frame_ids = [int(v) for v in data.get("trj_id", [])]
    poses_gt = load_pose_list(data["trj_gt"])
    poses_est = load_pose_list(data["trj_est"])
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    traj_est_aligned = trajectory.align_trajectory(
        traj_est,
        traj_ref,
        correct_scale=False,
    )

    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data((traj_ref, traj_est_aligned))
    ape_stats = ape_metric.get_all_statistics()

    gt_xyz = np.asarray([pose[:3, 3] for pose in traj_ref.poses_se3], dtype=float)
    est_xyz = np.asarray([pose[:3, 3] for pose in traj_est_aligned.poses_se3], dtype=float)
    gt_xy = gt_xyz[:, :2]
    est_xy = est_xyz[:, :2]

    lateral = []
    along = []
    for idx in range(len(gt_xy)):
        if len(gt_xy) == 1:
            tangent = np.asarray([1.0, 0.0])
        elif idx == 0:
            tangent = gt_xy[1] - gt_xy[0]
        elif idx == len(gt_xy) - 1:
            tangent = gt_xy[-1] - gt_xy[-2]
        else:
            tangent = gt_xy[idx + 1] - gt_xy[idx - 1]
        norm = float(np.linalg.norm(tangent))
        if norm < 1e-9:
            tangent = np.asarray([1.0, 0.0])
        else:
            tangent = tangent / norm
        normal = np.asarray([-tangent[1], tangent[0]])
        err_xy = est_xy[idx] - gt_xy[idx]
        along.append(float(np.dot(err_xy, tangent)))
        lateral.append(float(np.dot(err_xy, normal)))

    by_frame = {}
    for idx, frame_id in enumerate(frame_ids):
        by_frame[frame_id] = {
            "trajectory_error_m": float(ape_metric.error[idx]),
            "lateral_error_m": float(lateral[idx]),
            "along_error_m": float(along[idx]),
        }
    return by_frame, ape_stats


def discover_runs(results_dir, label, all_runs=False):
    pattern = f"GigaSLAM_railway_data_scene_*_train/*-No-LC/plot/trj_{label}.json"
    trj_paths = sorted(results_dir.glob(pattern))
    if all_runs:
        return [path.parents[1] for path in trj_paths]

    latest_by_scene = {}
    for trj_path in trj_paths:
        run_dir = trj_path.parents[1]
        scene = run_dir.parent.name.replace("GigaSLAM_railway_data_", "")
        if scene not in latest_by_scene or run_dir.name > latest_by_scene[scene].name:
            latest_by_scene[scene] = run_dir
    return [latest_by_scene[scene] for scene in sorted(latest_by_scene)]


def scene_from_run(run_dir):
    return run_dir.parent.name.replace("GigaSLAM_railway_data_", "")


def make_per_frame_rows(run_dir, label):
    scene = scene_from_run(run_dir)
    run = run_dir.name
    trj_rows, ape_stats = trajectory_diagnostics(run_dir / "plot" / f"trj_{label}.json")
    rail_rows = load_csv_by_frame(run_dir / "rail_scale_diag.csv")
    row_agg = load_rows_aggregate(run_dir / "rail_scale_rows.csv")
    vo_rows = load_csv_by_frame(run_dir / "vo_diag.csv")
    render_rows = load_csv_by_frame(run_dir / "rendered_depth_diag" / "rendered_depth_diag.csv")

    frame_ids = sorted(set(trj_rows) | set(rail_rows) | set(row_agg) | set(vo_rows) | set(render_rows))
    scale_values = {
        frame_id: as_float(rail_rows.get(frame_id, {}).get("applied_scale"))
        for frame_id in frame_ids
    }
    scale_steps = {}
    prev_scale = float("nan")
    for frame_id in frame_ids:
        scale = scale_values[frame_id]
        scale_steps[frame_id] = abs(scale - prev_scale) if np.isfinite(scale) and np.isfinite(prev_scale) else float("nan")
        if np.isfinite(scale):
            prev_scale = scale

    rows = []
    for frame_id in frame_ids:
        rail = rail_rows.get(frame_id, {})
        agg = row_agg.get(frame_id, {})
        vo = vo_rows.get(frame_id, {})
        rendered = render_rows.get(frame_id, {})
        trj = trj_rows.get(frame_id, {})

        row = {
            "scene": scene,
            "run": run,
            "frame_id": frame_id,
            "trajectory_error_m": trj.get("trajectory_error_m", float("nan")),
            "lateral_error_m": trj.get("lateral_error_m", float("nan")),
            "along_error_m": trj.get("along_error_m", float("nan")),
            "rail_status": rail.get("status", ""),
            "rail_width_m": as_float(rail.get("width")),
            "rail_confidence": as_float(rail.get("confidence")),
            "rail_samples": as_float(rail.get("samples")),
            "rail_raw_scale": as_float(rail.get("raw_scale")),
            "rail_applied_scale": as_float(rail.get("applied_scale")),
            "rail_scale_step": scale_steps[frame_id],
            "rail_pixel_width": as_float(rail.get("pixel_width")),
            "rail_pixel_width_mad": as_float(rail.get("pixel_width_mad")),
            "rail_pixel_center": as_float(rail.get("pixel_center")),
            "rail_pixel_center_mad": as_float(rail.get("pixel_center_mad")),
            "vo_method": vo.get("method", ""),
            "vo_matches": as_float(vo.get("matches")),
            "vo_pnp_inlier_ratio": as_float(vo.get("pnp_inlier_ratio")),
            "vo_step_norm": as_float(vo.get("step_norm")),
            "vo_depth_median": as_float(vo.get("depth_median")),
            "vo_depth_valid_ratio": as_float(vo.get("depth_valid_ratio")),
            "rendered_depth_available": bool(rendered),
            "rendered_depth_valid_ratio": as_float(rendered.get("valid_ratio")),
            "rendered_depth_median": as_float(rendered.get("render_depth_median")),
            "input_depth_median": as_float(rendered.get("input_depth_median")),
            "depth_gs_residual_median": as_float(rendered.get("median_abs_residual")),
            "depth_gs_residual_p95": as_float(rendered.get("p95_abs_residual")),
            "depth_gs_rel_residual_median": as_float(rendered.get("median_rel_residual")),
            "depth_gs_rel_residual_p95": as_float(rendered.get("p95_rel_residual")),
        }
        row.update({
            "row_total": agg.get("row_total", 0),
            "row_selected": agg.get("row_selected", 0),
            "row_selected_ratio": agg.get("row_selected_ratio", float("nan")),
            "row_reject_reasons": agg.get("row_reject_reasons", ""),
            "row_depth_median": agg.get("row_depth_median", float("nan")),
            "row_depth_lr_rel_diff_median": agg.get("row_depth_lr_rel_diff_median", float("nan")),
            "row_width_3d_median": agg.get("row_width_3d_median", float("nan")),
            "row_width_3d_mad": agg.get("row_width_3d_mad", float("nan")),
            "row_scale_raw_median": agg.get("row_scale_raw_median", float("nan")),
            "row_scale_raw_mad": agg.get("row_scale_raw_mad", float("nan")),
            "row_center_shift_max_px": agg.get("row_center_shift_max_px", float("nan")),
        })
        rows.append(row)

    add_risk_flags(rows)
    return rows, ape_stats


def add_risk_flags(rows):
    error_p90 = percentile([row["trajectory_error_m"] for row in rows], 90)
    scale_step_p95 = percentile([row["rail_scale_step"] for row in rows], 95)
    row_scale_mad_p90 = percentile([row["row_scale_raw_mad"] for row in rows], 90)
    depth_lr_p95 = percentile([row["row_depth_lr_rel_diff_median"] for row in rows], 95)
    vo_step_p95 = percentile([row["vo_step_norm"] for row in rows], 95)
    depth_gs_p90 = percentile([row["depth_gs_residual_median"] for row in rows], 90)

    for row in rows:
        flags = []
        status = row.get("rail_status", "")
        if status and status != "corrected":
            flags.append(f"rail_status:{status}")
        if np.isfinite(row["trajectory_error_m"]) and row["trajectory_error_m"] >= error_p90:
            flags.append("high_traj_error_p90")
        if np.isfinite(row["rail_scale_step"]) and row["rail_scale_step"] >= max(scale_step_p95, 0.05):
            flags.append("scale_step_high")
        if np.isfinite(row["row_selected_ratio"]) and row["row_selected_ratio"] < 0.6:
            flags.append("few_selected_rows")
        if np.isfinite(row["row_scale_raw_mad"]) and row["row_scale_raw_mad"] >= row_scale_mad_p90 and row["row_scale_raw_mad"] > 0.02:
            flags.append("row_scale_dispersion_high")
        if np.isfinite(row["row_depth_lr_rel_diff_median"]) and row["row_depth_lr_rel_diff_median"] >= max(depth_lr_p95, 0.05):
            flags.append("left_right_depth_mismatch")
        if np.isfinite(row["row_center_shift_max_px"]) and row["row_center_shift_max_px"] >= 300.0:
            flags.append("row_center_shift_high")
        if np.isfinite(row["vo_pnp_inlier_ratio"]) and 0.0 < row["vo_pnp_inlier_ratio"] < 0.25:
            flags.append("low_pnp_inlier_ratio")
        if np.isfinite(row["vo_depth_valid_ratio"]) and row["vo_depth_valid_ratio"] < 0.2:
            flags.append("low_depth_valid_ratio")
        if np.isfinite(row["vo_step_norm"]) and row["vo_step_norm"] >= max(vo_step_p95, 2.0):
            flags.append("vo_step_high")
        if np.isfinite(row["depth_gs_residual_median"]) and row["depth_gs_residual_median"] >= max(depth_gs_p90, 1.0):
            flags.append("depth_gs_residual_high")
        row["risk_flags"] = ";".join(flags)
        row["risk_score"] = len(flags)


def summarize_scene(rows, ape_stats):
    scene = rows[0]["scene"] if rows else ""
    run = rows[0]["run"] if rows else ""
    errors = [row["trajectory_error_m"] for row in rows]
    max_idx = int(np.nanargmax(np.asarray(errors, dtype=float))) if finite(errors).size else -1
    statuses = [row["rail_status"] for row in rows if row["rail_status"]]
    corrected = sum(1 for status in statuses if status == "corrected")
    hold = sum(1 for status in statuses if status.endswith("_hold"))

    selected = finite([row["row_selected"] for row in rows])
    totals = finite([row["row_total"] for row in rows])
    row_selected_ratio = float(selected.sum() / totals.sum()) if totals.size and totals.sum() > 0 else float("nan")
    rendered_available = any(bool(row.get("rendered_depth_available")) for row in rows)

    return {
        "scene": scene,
        "run": run,
        "frames": len(rows),
        "ate_rmse": float(ape_stats.get("rmse", float("nan"))),
        "error_p95": percentile(errors, 95),
        "max_error": errors[max_idx] if max_idx >= 0 else float("nan"),
        "max_error_frame_id": rows[max_idx]["frame_id"] if max_idx >= 0 else "",
        "corrected_ratio": corrected / len(statuses) if statuses else float("nan"),
        "hold_ratio": hold / len(statuses) if statuses else float("nan"),
        "rail_scale_median": median([row["rail_applied_scale"] for row in rows]),
        "rail_scale_p05": percentile([row["rail_applied_scale"] for row in rows], 5),
        "rail_scale_p95": percentile([row["rail_applied_scale"] for row in rows], 95),
        "rail_scale_step_p95": percentile([row["rail_scale_step"] for row in rows], 95),
        "rail_width_median": median([row["rail_width_m"] for row in rows]),
        "rail_width_p05": percentile([row["rail_width_m"] for row in rows], 5),
        "rail_width_p95": percentile([row["rail_width_m"] for row in rows], 95),
        "row_selected_ratio": row_selected_ratio,
        "row_scale_raw_mad_median": median([row["row_scale_raw_mad"] for row in rows]),
        "row_depth_lr_rel_diff_p95": percentile([row["row_depth_lr_rel_diff_median"] for row in rows], 95),
        "vo_depth_median_p50": median([row["vo_depth_median"] for row in rows]),
        "vo_depth_median_p95": percentile([row["vo_depth_median"] for row in rows], 95),
        "vo_step_norm_p95": percentile([row["vo_step_norm"] for row in rows], 95),
        "corr_error_vs_scale": corrcoef(errors, [row["rail_applied_scale"] for row in rows]),
        "corr_error_vs_vo_depth": corrcoef(errors, [row["vo_depth_median"] for row in rows]),
        "corr_error_vs_row_depth": corrcoef(errors, [row["row_depth_median"] for row in rows]),
        "depth_gs_residual_median_p50": median([row["depth_gs_residual_median"] for row in rows]),
        "depth_gs_residual_p95": percentile([row["depth_gs_residual_p95"] for row in rows], 95),
        "corr_error_vs_depth_gs_residual": corrcoef(errors, [row["depth_gs_residual_median"] for row in rows]),
        "rendered_depth_available": rendered_available,
    }


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key)) for key in fields})


def format_value(value):
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        return f"{float(value):.6g}"
    return value


def maybe_plot(ax, frames, values, label, color, *, linestyle="-", alpha=1.0):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if not np.any(valid):
        return False
    ax.plot(
        frames[valid],
        values[valid],
        label=label,
        color=color,
        linestyle=linestyle,
        alpha=alpha,
    )
    return True


def add_high_risk_guides(axes, rows, max_guides=60):
    high_risk = [
        row["frame_id"]
        for row in rows
        if as_int(row.get("risk_score"), 0) >= 3
    ]
    if len(high_risk) > max_guides:
        step = max(1, len(high_risk) // max_guides)
        high_risk = high_risk[::step]
    for ax in axes:
        for frame_id in high_risk:
            ax.axvline(frame_id, color="#d62728", alpha=0.08, linewidth=0.8)


def plot_scene(rows, output_path):
    frames = np.asarray([row["frame_id"] for row in rows], dtype=float)
    errors = np.asarray([row["trajectory_error_m"] for row in rows], dtype=float)
    along = np.asarray([row["along_error_m"] for row in rows], dtype=float)
    lateral = np.asarray([row["lateral_error_m"] for row in rows], dtype=float)
    scale = np.asarray([row["rail_applied_scale"] for row in rows], dtype=float)
    rail_width = np.asarray([row["rail_width_m"] for row in rows], dtype=float)
    vo_step = np.asarray([row["vo_step_norm"] for row in rows], dtype=float)
    pnp_ratio = np.asarray([row["vo_pnp_inlier_ratio"] for row in rows], dtype=float)
    rendered_depth = np.asarray([row["rendered_depth_median"] for row in rows], dtype=float)
    input_depth = np.asarray([row["input_depth_median"] for row in rows], dtype=float)
    depth_gs_median = np.asarray([row["depth_gs_residual_median"] for row in rows], dtype=float)
    depth_gs_p95 = np.asarray([row["depth_gs_residual_p95"] for row in rows], dtype=float)
    depth_gs_rel_p95 = np.asarray([row["depth_gs_rel_residual_p95"] for row in rows], dtype=float)

    fig, axes = plt.subplots(5, 1, figsize=(13, 12), dpi=150, sharex=True)
    fig.suptitle(
        f"{rows[0]['scene']} | trajectory / RailScale / VO / rendered-depth diagnostics",
        fontweight="semibold",
    )

    axes[0].plot(frames, errors, label="trajectory APE", color="#1f77b4")
    axes[0].plot(frames, along, label="along error", color="#ff7f0e", alpha=0.85)
    axes[0].plot(frames, lateral, label="lateral error", color="#2ca02c", alpha=0.85)
    axes[0].axhline(0.0, color="#9098a3", linewidth=0.8)
    axes[0].set_ylabel("error [m]")
    axes[0].legend(loc="upper left", ncol=3)

    axes[1].plot(frames, scale, label="RailScale applied scale", color="#9467bd")
    axes[1].set_ylabel("scale")
    ax1b = axes[1].twinx()
    maybe_plot(ax1b, frames, rail_width, "rail width", "#8c564b", alpha=0.85)
    ax1b.set_ylabel("rail width [m]")
    lines = axes[1].get_lines() + ax1b.get_lines()
    axes[1].legend(lines, [line.get_label() for line in lines], loc="upper left", ncol=2)

    maybe_plot(axes[2], frames, vo_step, "VO step norm", "#d62728")
    axes[2].set_ylabel("VO step [m]")
    ax2b = axes[2].twinx()
    maybe_plot(ax2b, frames, pnp_ratio, "PnP inlier ratio", "#2ca02c", alpha=0.85)
    ax2b.set_ylabel("PnP inlier ratio")
    ax2b.set_ylim(0.0, 1.05)
    lines = axes[2].get_lines() + ax2b.get_lines()
    axes[2].legend(lines, [line.get_label() for line in lines], loc="upper left", ncol=2)

    maybe_plot(axes[3], frames, input_depth, "input depth median", "#17becf")
    maybe_plot(axes[3], frames, rendered_depth, "rendered depth median", "#8c564b", alpha=0.85)
    axes[3].set_ylabel("depth [m]")
    axes[3].legend(loc="upper left", ncol=2)

    maybe_plot(axes[4], frames, depth_gs_median, "GS-depth residual median", "#1f77b4")
    maybe_plot(axes[4], frames, depth_gs_p95, "GS-depth residual p95", "#ff7f0e", alpha=0.8)
    ax4b = axes[4].twinx()
    maybe_plot(ax4b, frames, depth_gs_rel_p95, "relative residual p95", "#2ca02c", alpha=0.75)
    axes[4].set_ylabel("abs residual [m]")
    ax4b.set_ylabel("relative residual")
    axes[4].set_xlabel("frame id")
    lines = axes[4].get_lines() + ax4b.get_lines()
    if lines:
        axes[4].legend(lines, [line.get_label() for line in lines], loc="upper left", ncol=3)

    for ax in axes:
        ax.grid(True, color="#d9dde3", linewidth=0.8)
        ax.set_xlim(float(np.nanmin(frames)), float(np.nanmax(frames)))
    add_high_risk_guides(axes, rows)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, facecolor="white")
    plt.close(fig)


def load_json(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def trajectory_shape_summary_row(run_dir, label):
    plot_dir = run_dir / "plot"
    stats = load_json(plot_dir / f"stats_{label}.json")
    legacy_stats = load_json(plot_dir / "stats_legacy_ordered.json")
    shape = load_json(plot_dir / f"shape_{label}.json")
    return {
        "scene": scene_from_run(run_dir),
        "run": run_dir.name,
        "ate_rmse": as_float(stats.get("rmse")),
        "ate_legacy_ordered_rmse": as_float(legacy_stats.get("rmse", stats.get("rmse"))),
        "length_ratio": as_float(shape.get("length_ratio")),
        "end_error": as_float(shape.get("end_error")),
        "max_error": as_float(shape.get("max_error")),
        "max_error_frame_id": shape.get("max_error_frame_id", ""),
        "lat_p95": as_float(shape.get("lat_p95")),
        "along_p95": as_float(shape.get("along_p95")),
        "first25_rmse": as_float(shape.get("first25_rmse")),
        "mid50_rmse": as_float(shape.get("mid50_rmse")),
        "last25_rmse": as_float(shape.get("last25_rmse")),
    }


def top_anomalies_by_scene(rows, top_k):
    grouped = defaultdict(list)
    for row in rows:
        if row["risk_score"] > 0:
            grouped[row["scene"]].append(row)
    selected = []
    for scene in sorted(grouped):
        selected.extend(
            sorted(
                grouped[scene],
                key=lambda row: (
                    row["risk_score"],
                    row["trajectory_error_m"] if np.isfinite(row["trajectory_error_m"]) else -1.0,
                ),
                reverse=True,
            )[:top_k]
        )
    return selected


def dominant_risk_flags(rows, limit=3):
    counts = Counter()
    for row in rows:
        for flag in str(row.get("risk_flags", "")).split(";"):
            flag = flag.strip()
            if flag:
                counts[flag] += 1
    return ", ".join(f"{flag} ({count})" for flag, count in counts.most_common(limit))


def write_markdown_report(output_path, summaries, shape_rows, all_frame_rows):
    shape_by_scene = {row["scene"]: row for row in shape_rows}
    rows_by_scene = defaultdict(list)
    for row in all_frame_rows:
        rows_by_scene[row["scene"]].append(row)

    priority_order = [
        "scene_19_train",
        "scene_17_train",
        "scene_14_train",
        "scene_16_train",
        "scene_13_train",
        "scene_11_train",
    ]
    summary_by_scene = {row["scene"]: row for row in summaries}
    ordered_scenes = [scene for scene in priority_order if scene in summary_by_scene]
    ordered_scenes.extend(scene for scene in sorted(summary_by_scene) if scene not in ordered_scenes)

    lines = [
        "# Depth / Trajectory 诊断报告",
        "",
        "本报告由每个 scene 的最新结果目录生成，只做后处理诊断，不改变 SLAM 输出。",
        "",
        "## Scene 汇总",
        "",
        "| scene | run | ATE RMSE | end error | along p95 | lat p95 | corr(error,row depth) | corr(error,GS-depth residual) | 主要风险 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for scene in ordered_scenes:
        summary = summary_by_scene[scene]
        shape = shape_by_scene.get(scene, {})
        lines.append(
            "| {scene} | {run} | {ate:.3f} | {end:.3f} | {along:.3f} | {lat:.3f} | {corr_row:.3f} | {corr_gs:.3f} | {risks} |".format(
                scene=scene,
                run=summary["run"],
                ate=as_float(summary.get("ate_rmse")),
                end=as_float(shape.get("end_error")),
                along=as_float(shape.get("along_p95")),
                lat=as_float(shape.get("lat_p95")),
                corr_row=as_float(summary.get("corr_error_vs_row_depth")),
                corr_gs=as_float(summary.get("corr_error_vs_depth_gs_residual")),
                risks=dominant_risk_flags(rows_by_scene[scene]) or "-",
            )
        )

    lines.extend([
        "",
        "## 当前判断",
        "",
        "- 当前剩余主要问题是沿轨累计漂移和末端误差，不是明显横向偏轨。",
        "- 6 个序列都有 rendered depth residual，但它和轨迹误差的相关性随 scene 变化，现阶段应先作为诊断信号。",
        "- 优先检查 scene19 和 scene17，因为它们分别暴露末端漂移和长序列退化问题。",
        "- scene13 当前表现较好，后续新策略必须把它作为道岔/弯道回归保护样本。",
        "",
        "## 受控实验分组",
        "",
        "- E1：只做诊断可视化，不改算法，确认每个误差峰值更像 RailScale、VO 还是 GS-depth residual 问题。",
        "- E2：VO 退化门控，测试 low PnP inlier ratio 和 high VO step 是否适合作为降权触发。",
        "- E3：RailScale 保守门控，重点处理 row-scale dispersion、width-jump hold、too-few-samples hold，不修改 metric_width_m。",
        "- E4：只有当 E1 证明 GS-depth residual 有清晰关系后，再把它作为异常检测或权重调节信号，不直接加入优化损失。",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze depth/RailScale consistency against trajectory errors."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/depth_scale_diagnostics"))
    parser.add_argument("--label", default="original")
    parser.add_argument("--all-runs", action="store_true", help="Analyze all matching runs instead of only latest per scene.")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--top-k-per-scene",
        type=int,
        default=10,
        help="Number of anomaly rows to keep per scene in depth_scale_anomaly_frames_by_scene.csv.",
    )
    parser.add_argument(
        "--shape-summary-path",
        type=Path,
        default=None,
        help="Where to write trajectory shape summary. Defaults to results/trajectory_shape_summary.csv for latest-only analysis.",
    )
    parser.add_argument(
        "--skip-shape-summary",
        action="store_true",
        help="Do not rebuild trajectory_shape_summary.csv.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = discover_runs(results_dir, args.label, all_runs=args.all_runs)
    if not run_dirs:
        raise FileNotFoundError(f"No result runs with plot/trj_{args.label}.json under {results_dir}")

    all_frame_rows = []
    summaries = []
    shape_summaries = []
    scenes = []
    for run_dir in run_dirs:
        rows, ape_stats = make_per_frame_rows(run_dir, args.label)
        scene = scene_from_run(run_dir)
        scenes.append(scene)
        scene_csv = output_dir / f"{scene}_depth_scale_consistency.csv"
        write_csv(scene_csv, rows, PER_FRAME_FIELDS)
        if not args.no_plots:
            plot_scene(rows, output_dir / f"{scene}_depth_scale_consistency.png")
        all_frame_rows.extend(rows)
        summaries.append(summarize_scene(rows, ape_stats))
        shape_summaries.append(trajectory_shape_summary_row(run_dir, args.label))

    write_csv(output_dir / "depth_scale_consistency_summary.csv", summaries, SUMMARY_FIELDS)
    write_csv(output_dir / "depth_scale_consistency_all_frames.csv", all_frame_rows, PER_FRAME_FIELDS)

    anomalies = sorted(
        [row for row in all_frame_rows if row["risk_score"] > 0],
        key=lambda row: (
            row["risk_score"],
            row["trajectory_error_m"] if np.isfinite(row["trajectory_error_m"]) else -1.0,
        ),
        reverse=True,
    )
    write_csv(output_dir / "depth_scale_anomaly_frames.csv", anomalies[: args.top_k], PER_FRAME_FIELDS)
    write_csv(
        output_dir / "depth_scale_anomaly_frames_by_scene.csv",
        top_anomalies_by_scene(all_frame_rows, args.top_k_per_scene),
        PER_FRAME_FIELDS,
    )

    shape_summary_path = None
    if not args.skip_shape_summary:
        if args.shape_summary_path is not None:
            shape_summary_path = args.shape_summary_path.resolve()
        elif not args.all_runs:
            shape_summary_path = results_dir / "trajectory_shape_summary.csv"
    if shape_summary_path is not None:
        write_csv(shape_summary_path, shape_summaries, SHAPE_SUMMARY_FIELDS)

    report_path = output_dir / "diagnostic_report.md"
    write_markdown_report(report_path, summaries, shape_summaries, all_frame_rows)

    rendered_depth_available = any(bool(summary.get("rendered_depth_available")) for summary in summaries)
    rendered_depth_note = (
        "Numeric Gaussian-rendered depth diagnostics were found and merged from rendered_depth_diag/rendered_depth_diag.csv."
        if rendered_depth_available
        else "Current result folders do not save numeric rendered depth maps. This diagnostic correlates RailScale row depth, VO depth and trajectory error; Gaussian-rendered depth residual columns are reserved for runs with rendered_depth_diag enabled."
    )

    manifest = {
        "diagnostic": "depth_scale_consistency",
        "alignment_mode": "SE3 no scale",
        "label": args.label,
        "results_dir": str(results_dir),
        "output_dir": str(output_dir),
        "latest_only": not args.all_runs,
        "rendered_depth_available": rendered_depth_available,
        "rendered_depth_note": rendered_depth_note,
        "scenes": scenes,
        "files": {
            "summary_csv": str(output_dir / "depth_scale_consistency_summary.csv"),
            "all_frames_csv": str(output_dir / "depth_scale_consistency_all_frames.csv"),
            "anomaly_csv": str(output_dir / "depth_scale_anomaly_frames.csv"),
            "anomaly_by_scene_csv": str(output_dir / "depth_scale_anomaly_frames_by_scene.csv"),
            "trajectory_shape_summary_csv": str(shape_summary_path) if shape_summary_path else None,
            "diagnostic_report_md": str(report_path),
        },
    }
    with open(output_dir / "depth_scale_consistency_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True)

    print(f"Analyzed {len(run_dirs)} run(s): {', '.join(scenes)}")
    print(f"Wrote summary: {output_dir / 'depth_scale_consistency_summary.csv'}")
    print(f"Wrote all-frame diagnostics: {output_dir / 'depth_scale_consistency_all_frames.csv'}")
    print(f"Wrote top anomalies: {output_dir / 'depth_scale_anomaly_frames.csv'}")
    print(f"Wrote per-scene anomalies: {output_dir / 'depth_scale_anomaly_frames_by_scene.csv'}")
    if shape_summary_path is not None:
        print(f"Wrote trajectory shape summary: {shape_summary_path}")
    print(f"Wrote diagnostic report: {report_path}")
    for summary in summaries:
        print(
            f"{summary['scene']}: ATE={summary['ate_rmse']:.4f} m, "
            f"corrected={summary['corrected_ratio']:.2%}, "
            f"scale_p05-p95={summary['rail_scale_p05']:.3f}-{summary['rail_scale_p95']:.3f}, "
            f"corr(error,row_depth)={summary['corr_error_vs_row_depth']:.3f}, "
            f"corr(error,gs_depth_res)={summary['corr_error_vs_depth_gs_residual']:.3f}"
        )


if __name__ == "__main__":
    main()
