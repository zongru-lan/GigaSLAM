#!/usr/bin/env python3
"""Generate presentation-friendly trajectory plots from GigaSLAM eval outputs.

The report plots are display artifacts only. They do not replace the official
stats_original.json / shape_original.json metrics or evo_2dplot_original.png.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path

if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator
import numpy as np
from evo.core import metrics, trajectory
from evo.core.trajectory import PosePath3D


AXIS_BOX = (0.10, 0.15, 0.70, 0.70)
CBAR_BOX = (0.84, 0.15, 0.025, 0.70)

ALIGNMENT_MODES = {
    "se3_no_scale": {
        "title": "SE3 no scale",
        "manifest": "SE3 no scale",
        "correct_scale": False,
        "validate_against_original": True,
    },
    "sim3_scale_corrected": {
        "title": "Sim3 scale corrected",
        "manifest": "Sim3 scale corrected",
        "correct_scale": True,
        "validate_against_original": False,
    },
}


def load_summary(results_dir):
    path = results_dir / "trajectory_shape_summary.csv"
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {row["scene"]: row for row in csv.DictReader(f)}


def load_pose_list(values):
    return [np.asarray(pose, dtype=float) for pose in values]


def positions_xy(poses):
    return np.asarray([pose[:3, 3] for pose in poses], dtype=float)[:, :2]


def positions_xyz(poses):
    return np.asarray([pose[:3, 3] for pose in poses], dtype=float)


def infer_scene_name(trj_path):
    scene_dir = trj_path.parents[2].name
    scene_pos = scene_dir.find("scene_")
    if scene_pos >= 0:
        return scene_dir[scene_pos:]
    return scene_dir


def grid_shape(count):
    cols = max(1, math.ceil(math.sqrt(count)))
    rows = math.ceil(count / cols)
    return rows, cols


def discover_trj_paths(args):
    pattern = args.trj_glob or f"*scene_*_train/*-No-LC/plot/trj_{args.label}.json"
    paths = sorted(args.results_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No trajectory files matched '{pattern}' under {args.results_dir}")
    return paths


def evaluate_trj(trj_path, alignment):
    data = json.load(open(trj_path, encoding="utf-8"))
    poses_gt = load_pose_list(data["trj_gt"])
    poses_est = load_pose_list(data["trj_est"])
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    traj_est_aligned = trajectory.align_trajectory(
        traj_est,
        traj_ref,
        correct_scale=ALIGNMENT_MODES[alignment]["correct_scale"],
    )

    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data((traj_ref, traj_est_aligned))
    ape_stats = ape_metric.get_all_statistics()

    scene = infer_scene_name(trj_path)
    run = trj_path.parents[1].name
    return {
        "scene": scene,
        "run": run,
        "trj_path": trj_path,
        "frame_ids": data.get("trj_id", []),
        "gt_xyz": positions_xyz(traj_ref.poses_se3),
        "est_xyz": positions_xyz(traj_est_aligned.poses_se3),
        "gt_xy": positions_xy(traj_ref.poses_se3),
        "est_xy": positions_xy(traj_est_aligned.poses_se3),
        "errors": np.asarray(ape_metric.error, dtype=float),
        "ate_rmse": float(ape_stats["rmse"]),
        "stats": ape_stats,
    }


def read_stats_rmse(trj_path):
    stats_path = trj_path.with_name("stats_original.json")
    if not stats_path.exists():
        return None
    stats = json.load(open(stats_path, encoding="utf-8"))
    return float(stats["rmse"])


def validate_metrics(items, summary, alignment, tolerance=1e-4):
    if not ALIGNMENT_MODES[alignment]["validate_against_original"]:
        return
    for item in items:
        stats_rmse = read_stats_rmse(item["trj_path"])
        if stats_rmse is not None and abs(stats_rmse - item["ate_rmse"]) > tolerance:
            raise ValueError(
                f"ATE mismatch for {item['scene']}: computed={item['ate_rmse']:.6f}, "
                f"stats_original={stats_rmse:.6f}"
            )
        row = summary.get(item["scene"])
        if row:
            summary_rmse = float(row["ate_rmse"])
            if abs(summary_rmse - item["ate_rmse"]) > tolerance:
                raise ValueError(
                    f"ATE mismatch for {item['scene']}: computed={item['ate_rmse']:.6f}, "
                    f"summary={summary_rmse:.6f}"
                )


def rounded_vmax(items, requested=None):
    if requested is not None:
        return float(requested)
    max_error = max(float(item["errors"].max()) for item in items)
    return float(max(1, math.ceil(max_error)))


def scene_limits(item, target_ratio, major_grid, minor_grid, padding_frac):
    xy = np.vstack([item["gt_xy"], item["est_xy"]])
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    span_x = max(float(xmax - xmin), minor_grid)
    span_y = max(float(ymax - ymin), minor_grid)
    pad = max(minor_grid, max(span_x, span_y) * padding_frac)
    span_x += 2.0 * pad
    span_y += 2.0 * pad

    if span_x / span_y < target_ratio:
        span_x = span_y * target_ratio
    else:
        span_y = span_x / target_ratio

    step = float(minor_grid)
    xlim = [math.floor((cx - span_x / 2.0) / step) * step,
            math.ceil((cx + span_x / 2.0) / step) * step]
    ylim = [math.floor((cy - span_y / 2.0) / step) * step,
            math.ceil((cy + span_y / 2.0) / step) * step]

    # Re-expand after rounding so the final limits keep the same panel aspect.
    span_x = xlim[1] - xlim[0]
    span_y = ylim[1] - ylim[0]
    if span_x / span_y < target_ratio:
        wanted = math.ceil((span_y * target_ratio) / step) * step
        extra = wanted - span_x
        xlim[0] -= extra / 2.0
        xlim[1] += extra / 2.0
    else:
        wanted = math.ceil((span_x / target_ratio) / step) * step
        extra = wanted - span_y
        ylim[0] -= extra / 2.0
        ylim[1] += extra / 2.0
    return tuple(xlim), tuple(ylim)


def colored_line(ax, xy, values, cmap, norm):
    if len(xy) < 2:
        ax.scatter(xy[:, 0], xy[:, 1], c=values, cmap=cmap, norm=norm, s=20, zorder=3)
        return None
    mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = [mapper.to_rgba(value) for value in values]
    segments = [list(zip(x_pair, y_pair)) for x_pair, y_pair in zip(
        zip(xy[:-1, 0], xy[1:, 0]),
        zip(xy[:-1, 1], xy[1:, 1]),
    )]
    collection = LineCollection(segments, colors=colors, alpha=1.0, linestyle="solid")
    collection.set_zorder(3)
    ax.add_collection(collection)
    return collection


def style_axis(ax, item, xlim, ylim, major_grid, minor_grid, title=None):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(major_grid))
    ax.yaxis.set_major_locator(MultipleLocator(major_grid))
    ax.xaxis.set_minor_locator(MultipleLocator(minor_grid))
    ax.yaxis.set_minor_locator(MultipleLocator(minor_grid))
    ax.grid(which="major", color="#d9dde3", linewidth=0.9)
    ax.grid(which="minor", color="#edf0f4", linewidth=0.55)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    if title:
        ax.set_title(title, pad=12, fontsize=13, fontweight="semibold")
    for spine in ax.spines.values():
        spine.set_color("#b8bec8")


def draw_scene(
    ax,
    item,
    xlim,
    ylim,
    cmap,
    norm,
    major_grid,
    minor_grid,
    *,
    title=True,
    legend=True,
    show_alignment=True,
):
    ax.plot(
        item["gt_xy"][:, 0],
        item["gt_xy"][:, 1],
        linestyle="--",
        color="#6e6e6e",
        linewidth=2.0,
        label="gt",
        zorder=2,
    )
    colored_line(ax, item["est_xy"], item["errors"], cmap, norm)
    plot_title = None
    if title:
        plot_title = f"{item['scene']} | ATE RMSE: {item['ate_rmse']:.2f} m"
        if show_alignment:
            plot_title += f" | {ALIGNMENT_MODES[item['alignment']]['title']}"
    style_axis(ax, item, xlim, ylim, major_grid, minor_grid, title=plot_title)
    if legend:
        ax.legend(loc="upper right", frameon=True, framealpha=0.92)


def save_single_plot(item, output_dir, cmap, norm, limits, args, manifest_entry):
    fig = plt.figure(figsize=(args.single_width / args.dpi, args.single_height / args.dpi), dpi=args.dpi)
    ax = fig.add_axes(AXIS_BOX)
    cax = fig.add_axes(CBAR_BOX)
    xlim, ylim = limits[item["scene"]]
    draw_scene(ax, item, xlim, ylim, cmap, norm, args.major_grid, args.minor_grid)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("translation error [m]")

    png = output_dir / f"{item['scene']}_trajectory_report.png"
    pdf = output_dir / f"{item['scene']}_trajectory_report.pdf"
    fig.savefig(png, dpi=args.dpi, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    manifest_entry["single_png"] = str(png)
    manifest_entry["single_pdf"] = str(pdf)


def save_grid_plot(items, output_dir, cmap, norm, limits, args):
    rows, cols = grid_shape(len(items))
    fig, axes = plt.subplots(rows, cols, figsize=(6.0 * cols, 4.9 * rows), dpi=args.dpi, squeeze=False)
    axes = axes.ravel()
    for ax, item in zip(axes, items):
        xlim, ylim = limits[item["scene"]]
        draw_scene(
            ax,
            item,
            xlim,
            ylim,
            cmap,
            norm,
            args.major_grid,
            args.minor_grid,
            title=True,
            legend=True,
            show_alignment=False,
        )
    for ax in axes[len(items):]:
        ax.axis("off")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.subplots_adjust(left=0.055, right=0.90, top=0.92, bottom=0.08, wspace=0.22, hspace=0.28)
    cbar = fig.colorbar(sm, ax=axes.tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("translation error [m]")
    alignment_title = ALIGNMENT_MODES[items[0]["alignment"]]["title"]
    fig.suptitle(f"Railway Trajectories | {alignment_title}", fontsize=16, fontweight="semibold")

    png = output_dir / "trajectory_report_grid.png"
    pdf = output_dir / "trajectory_report_grid.pdf"
    fig.savefig(png, dpi=args.dpi, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return png, pdf, {"rows": rows, "cols": cols}


def write_manifest(output_dir, items, limits, args, colorbar_vmax, grid_png, grid_pdf, grid_layout):
    manifest = {
        "alignment_mode": ALIGNMENT_MODES[args.alignment]["manifest"],
        "plot_style": "evo_like_report_v2",
        "colormap": args.cmap,
        "line_style": "evo_default_gt_under_est",
        "results_dir": str(args.results_dir),
        "major_grid_m": args.major_grid,
        "minor_grid_m": args.minor_grid,
        "colorbar_vmin": 0.0,
        "colorbar_vmax": colorbar_vmax,
        "single_plot_px": [args.single_width, args.single_height],
        "grid_shape": grid_layout,
        "grid_png": str(grid_png),
        "grid_pdf": str(grid_pdf),
        "scenes": [],
    }
    for item in items:
        xlim, ylim = limits[item["scene"]]
        entry = {
            "scene": item["scene"],
            "run": item["run"],
            "ate_rmse": item["ate_rmse"],
            "max_error": float(item["errors"].max()),
            "axis_limits": {
                "xlim": [float(xlim[0]), float(xlim[1])],
                "ylim": [float(ylim[0]), float(ylim[1])],
            },
        }
        save_single_plot(item, output_dir, plt.get_cmap(args.cmap), Normalize(0.0, colorbar_vmax), limits, args, entry)
        manifest["scenes"].append(entry)

    manifest_path = output_dir / "report_plot_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate report-ready GigaSLAM trajectory plots.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/report_plots"))
    parser.add_argument("--label", default="original")
    parser.add_argument(
        "--trj-glob",
        default=None,
        help="Glob under --results-dir for trajectory json files. Defaults to "
        "'*scene_*_train/*-No-LC/plot/trj_<label>.json'.",
    )
    parser.add_argument("--major-grid", type=float, default=50.0)
    parser.add_argument("--minor-grid", type=float, default=25.0)
    parser.add_argument("--padding-frac", type=float, default=0.08)
    parser.add_argument("--single-width", type=int, default=1600)
    parser.add_argument("--single-height", type=int, default=1000)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--cmap", default="jet")
    parser.add_argument("--colorbar-vmax", type=float, default=None)
    parser.add_argument(
        "--alignment",
        choices=sorted(ALIGNMENT_MODES),
        default="se3_no_scale",
        help="Trajectory alignment for report plots. se3_no_scale preserves metric scale; "
        "sim3_scale_corrected applies global scale correction for diagnosis.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.results_dir = args.results_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trj_paths = discover_trj_paths(args)

    items = [evaluate_trj(path, args.alignment) for path in trj_paths]
    for item in items:
        item["alignment"] = args.alignment
    summary = load_summary(args.results_dir)
    validate_metrics(items, summary, args.alignment)

    cmap = plt.get_cmap(args.cmap)
    vmax = rounded_vmax(items, args.colorbar_vmax)
    norm = Normalize(vmin=0.0, vmax=vmax)

    target_ratio = AXIS_BOX[2] * args.single_width / (AXIS_BOX[3] * args.single_height)
    limits = {
        item["scene"]: scene_limits(
            item,
            target_ratio=target_ratio,
            major_grid=args.major_grid,
            minor_grid=args.minor_grid,
            padding_frac=args.padding_frac,
        )
        for item in items
    }

    # Create grid first so the manifest can refer to it; single plots are saved while writing manifest.
    grid_png, grid_pdf, grid_layout = save_grid_plot(items, args.output_dir, cmap, norm, limits, args)
    manifest_path = write_manifest(args.output_dir, items, limits, args, vmax, grid_png, grid_pdf, grid_layout)

    print(f"Wrote {len(items)} single-scene report plots to {args.output_dir}")
    print(f"Wrote grid plot: {grid_png}")
    print(f"Wrote manifest: {manifest_path}")
    for item in items:
        xlim, ylim = limits[item["scene"]]
        print(
            f"{item['scene']}: ATE={item['ate_rmse']:.4f} m, "
            f"max_err={item['errors'].max():.4f} m, xlim={xlim}, ylim={ylim}"
        )


if __name__ == "__main__":
    main()
