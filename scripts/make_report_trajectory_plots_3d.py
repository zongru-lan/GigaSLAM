#!/usr/bin/env python3
"""Generate report-ready 3D trajectory plots from GigaSLAM eval outputs.

These plots are visual supplements for pose-estimation reports. They reuse the
same trajectory alignment and APE coloring as make_report_trajectory_plots.py,
but draw the camera-center trajectories in XYZ instead of top-view XY.
"""

import argparse
import json
import math
import os
from pathlib import Path

if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import numpy as np

from make_report_trajectory_plots import (
    ALIGNMENT_MODES,
    discover_trj_paths,
    evaluate_trj,
    grid_shape,
    load_summary,
    rounded_vmax,
    validate_metrics,
)


def rounded_limits(values, step, padding_frac):
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    span = max(vmax - vmin, step)
    pad = max(step * 0.5, span * padding_frac)
    return (
        math.floor((vmin - pad) / step) * step,
        math.ceil((vmax + pad) / step) * step,
    )


def scene_limits_3d(item, xy_grid, z_grid, padding_frac):
    xyz = np.vstack([item["gt_xyz"], item["est_xyz"]])
    xlim = rounded_limits(xyz[:, 0], xy_grid, padding_frac)
    ylim = rounded_limits(xyz[:, 1], xy_grid, padding_frac)
    zlim = rounded_limits(xyz[:, 2], z_grid, padding_frac)
    return xlim, ylim, zlim


def colored_line_3d(ax, xyz, values, cmap, norm):
    if len(xyz) < 2:
        ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            c=values,
            cmap=cmap,
            norm=norm,
            s=20,
            depthshade=False,
        )
        return

    segments = np.stack([xyz[:-1], xyz[1:]], axis=1)
    mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    segment_values = 0.5 * (values[:-1] + values[1:])
    colors = [mapper.to_rgba(value) for value in segment_values]
    collection = Line3DCollection(segments, colors=colors, alpha=1.0, linestyle="solid")
    collection.set_zorder(3)
    ax.add_collection3d(collection)


def set_box_aspect_3d(ax, aspect, zoom):
    try:
        ax.set_box_aspect(aspect, zoom=zoom)
    except TypeError:
        ax.set_box_aspect(aspect)


def style_axis_3d(
    ax,
    item,
    limits,
    args,
    *,
    title=True,
    show_alignment=True,
    compact=False,
):
    xlim, ylim, zlim = limits
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    aspect = (xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0])
    set_box_aspect_3d(ax, aspect, args.grid_zoom if compact else args.single_zoom)
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_proj_type(args.projection)
    ax.grid(True, color="#d9dde3", linewidth=0.75)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.zaxis.set_major_locator(MaxNLocator(3))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor("#c5c9cf")

    if compact:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.tick_params(axis="both", which="major", length=0, pad=0)
        title_size = 11
        title_pad = 5
    else:
        ax.set_xlabel("x [m]", labelpad=3, fontsize=9)
        ax.set_ylabel("y [m]", labelpad=3, fontsize=9)
        ax.set_zlabel("z [m]", labelpad=3, fontsize=9)
        ax.tick_params(axis="both", which="major", labelsize=7, pad=-1)
        ax.zaxis.set_tick_params(labelsize=7, pad=-1)
        title_size = 13
        title_pad = 8

    if title:
        plot_title = f"{item['scene']} | ATE RMSE: {item['ate_rmse']:.2f} m"
        if show_alignment:
            plot_title += f" | {ALIGNMENT_MODES[item['alignment']]['title']}"
        ax.set_title(plot_title, pad=title_pad, fontsize=title_size, fontweight="semibold")


def draw_scene_3d(
    ax,
    item,
    limits,
    cmap,
    norm,
    args,
    *,
    title=True,
    legend=True,
    show_alignment=True,
    compact=False,
):
    ax.plot(
        item["gt_xyz"][:, 0],
        item["gt_xyz"][:, 1],
        item["gt_xyz"][:, 2],
        linestyle="--",
        color="#6e6e6e",
        linewidth=2.0,
        label="gt",
        zorder=2,
    )
    colored_line_3d(ax, item["est_xyz"], item["errors"], cmap, norm)
    style_axis_3d(
        ax,
        item,
        limits,
        args,
        title=title,
        show_alignment=show_alignment,
        compact=compact,
    )
    if legend:
        ax.legend(
            loc="upper right",
            frameon=True,
            framealpha=0.92,
            fontsize=8,
            handlelength=1.5,
            borderpad=0.22,
            labelspacing=0.18,
        )


def save_single_plot(item, output_dir, cmap, norm, limits, args, manifest_entry):
    fig = plt.figure(figsize=(args.single_width / args.dpi, args.single_height / args.dpi), dpi=args.dpi)
    ax = fig.add_axes([0.035, 0.055, 0.80, 0.84], projection="3d")
    cax = fig.add_axes([0.895, 0.20, 0.022, 0.60])
    draw_scene_3d(ax, item, limits[item["scene"]], cmap, norm, args)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("translation error [m]", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    png = output_dir / f"{item['scene']}_trajectory_report_3d.png"
    pdf = output_dir / f"{item['scene']}_trajectory_report_3d.pdf"
    fig.savefig(png, dpi=args.dpi, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    manifest_entry["single_png"] = str(png)
    manifest_entry["single_pdf"] = str(pdf)


def save_grid_plot(items, output_dir, cmap, norm, limits, args):
    rows, cols = grid_shape(len(items))
    fig = plt.figure(figsize=(6.0 * cols, 4.9 * rows), dpi=args.dpi)
    axes = []
    for index in range(rows * cols):
        ax = fig.add_subplot(rows, cols, index + 1, projection="3d")
        axes.append(ax)
        if index >= len(items):
            ax.set_axis_off()
            continue
        item = items[index]
        draw_scene_3d(
            ax,
            item,
            limits[item["scene"]],
            cmap,
            norm,
            args,
            title=True,
            legend=True,
            show_alignment=False,
            compact=True,
        )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.subplots_adjust(left=0.025, right=0.895, top=0.91, bottom=0.045, wspace=0.02, hspace=0.16)
    cax = fig.add_axes([0.925, 0.17, 0.018, 0.66])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("translation error [m]", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    alignment_title = ALIGNMENT_MODES[items[0]["alignment"]]["title"]
    fig.suptitle(f"Railway 3D Trajectories | {alignment_title}", fontsize=16, fontweight="semibold")

    png = output_dir / "trajectory_report_grid_3d.png"
    pdf = output_dir / "trajectory_report_grid_3d.pdf"
    fig.savefig(png, dpi=args.dpi, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return png, pdf, {"rows": rows, "cols": cols}


def write_manifest(output_dir, items, limits, args, colorbar_vmax, grid_png, grid_pdf, grid_layout):
    manifest = {
        "alignment_mode": ALIGNMENT_MODES[args.alignment]["manifest"],
        "plot_style": "trajectory_3d_line_report_v3_single_clean_grid_compact",
        "colormap": args.cmap,
        "line_style": "gt_dashed_est_error_colored",
        "results_dir": str(args.results_dir),
        "view": {
            "elev": args.elev,
            "azim": args.azim,
            "projection": args.projection,
            "single_zoom": args.single_zoom,
            "grid_zoom": args.grid_zoom,
        },
        "xy_grid_m": args.xy_grid,
        "z_grid_m": args.z_grid,
        "colorbar_vmin": 0.0,
        "colorbar_vmax": colorbar_vmax,
        "single_plot_px": [args.single_width, args.single_height],
        "grid_shape": grid_layout,
        "grid_png": str(grid_png),
        "grid_pdf": str(grid_pdf),
        "scenes": [],
    }

    for item in items:
        xlim, ylim, zlim = limits[item["scene"]]
        entry = {
            "scene": item["scene"],
            "run": item["run"],
            "ate_rmse": item["ate_rmse"],
            "max_error": float(item["errors"].max()),
            "axis_limits": {
                "xlim": [float(xlim[0]), float(xlim[1])],
                "ylim": [float(ylim[0]), float(ylim[1])],
                "zlim": [float(zlim[0]), float(zlim[1])],
            },
        }
        save_single_plot(item, output_dir, plt.get_cmap(args.cmap), Normalize(0.0, colorbar_vmax), limits, args, entry)
        manifest["scenes"].append(entry)

    manifest_path = output_dir / "report_plot_manifest_3d.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate report-ready GigaSLAM 3D trajectory plots.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/report_plots/se3_no_scale_3d"))
    parser.add_argument("--label", default="original")
    parser.add_argument(
        "--trj-glob",
        default=None,
        help="Glob under --results-dir for trajectory json files. Defaults to "
        "'*scene_*_train/*-No-LC/plot/trj_<label>.json'.",
    )
    parser.add_argument("--xy-grid", type=float, default=50.0)
    parser.add_argument("--z-grid", type=float, default=5.0)
    parser.add_argument("--padding-frac", type=float, default=0.08)
    parser.add_argument("--single-width", type=int, default=1600)
    parser.add_argument("--single-height", type=int, default=1000)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--cmap", default="jet")
    parser.add_argument("--colorbar-vmax", type=float, default=None)
    parser.add_argument("--elev", type=float, default=25.0)
    parser.add_argument("--azim", type=float, default=-60.0)
    parser.add_argument("--projection", choices=["ortho", "persp"], default="ortho")
    parser.add_argument("--single-zoom", type=float, default=1.55)
    parser.add_argument("--grid-zoom", type=float, default=1.35)
    parser.add_argument(
        "--alignment",
        choices=sorted(ALIGNMENT_MODES),
        default="se3_no_scale",
        help="Trajectory alignment. se3_no_scale preserves metric scale; "
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

    limits = {
        item["scene"]: scene_limits_3d(
            item,
            xy_grid=args.xy_grid,
            z_grid=args.z_grid,
            padding_frac=args.padding_frac,
        )
        for item in items
    }

    grid_png, grid_pdf, grid_layout = save_grid_plot(items, args.output_dir, cmap, norm, limits, args)
    manifest_path = write_manifest(args.output_dir, items, limits, args, vmax, grid_png, grid_pdf, grid_layout)

    print(f"Wrote {len(items)} single-scene 3D report plots to {args.output_dir}")
    print(f"Wrote 3D grid plot: {grid_png}")
    print(f"Wrote 3D manifest: {manifest_path}")
    for item in items:
        xlim, ylim, zlim = limits[item["scene"]]
        print(
            f"{item['scene']}: ATE={item['ate_rmse']:.4f} m, "
            f"max_err={item['errors'].max():.4f} m, xlim={xlim}, ylim={ylim}, zlim={zlim}"
        )


if __name__ == "__main__":
    main()
