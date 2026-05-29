#!/usr/bin/env python3
"""Compare trajectory metrics from a candidate experiment against a frozen baseline.

This is a post-run regression guard. It does not run SLAM and does not modify
result folders. By default it compares each baseline scene against the latest
complete result folder for that scene.
"""

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path


LOWER_IS_BETTER = [
    "ate_rmse",
    "end_error",
    "max_error",
    "lat_p95",
    "along_p95",
    "first25_rmse",
    "mid50_rmse",
    "last25_rmse",
]
TARGET_ONE = ["length_ratio"]
METRICS = LOWER_IS_BETTER + TARGET_ONE


def as_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def format_value(value):
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.6g}"
    return value


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key, "")) for key in fields})


def scene_from_run_dir(run_dir):
    return run_dir.parent.name.replace("GigaSLAM_railway_data_", "")


def run_dir_for_scene(results_dir, scene, run_name):
    return results_dir / f"GigaSLAM_railway_data_{scene}" / run_name


def latest_complete_run(results_dir, scene):
    scene_dir = results_dir / f"GigaSLAM_railway_data_{scene}"
    if not scene_dir.exists():
        return None
    candidates = []
    for run_dir in scene_dir.glob("*-No-LC"):
        if (run_dir / "plot/stats_original.json").exists() and (run_dir / "plot/shape_original.json").exists():
            candidates.append(run_dir)
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def load_metrics_from_run(run_dir):
    stats_path = run_dir / "plot/stats_original.json"
    shape_path = run_dir / "plot/shape_original.json"
    stats = json.load(open(stats_path, encoding="utf-8"))
    shape = json.load(open(shape_path, encoding="utf-8"))
    row = {
        "scene": scene_from_run_dir(run_dir),
        "run": run_dir.name,
        "ate_rmse": as_float(stats.get("rmse")),
    }
    for key in [
        "length_ratio",
        "end_error",
        "max_error",
        "max_error_frame_id",
        "lat_p95",
        "along_p95",
        "first25_rmse",
        "mid50_rmse",
        "last25_rmse",
    ]:
        row[key] = shape.get(key, "")
    return row


def parse_candidate_runs(items):
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected --candidate-run SCENE=RUN, got {item}")
        scene, run = item.split("=", 1)
        out[scene.strip()] = run.strip()
    return out


def metric_verdict(metric, baseline, candidate, args):
    if not math.isfinite(baseline) or not math.isfinite(candidate):
        return "missing", float("nan"), float("nan")
    delta = candidate - baseline
    rel = delta / abs(baseline) if abs(baseline) > 1e-12 else 0.0

    if metric in TARGET_ONE:
        base_dev = abs(baseline - 1.0)
        cand_dev = abs(candidate - 1.0)
        dev_delta = cand_dev - base_dev
        if dev_delta > args.length_ratio_abs_regress:
            return "regress", delta, rel
        if dev_delta < -args.length_ratio_abs_improve:
            return "improve", delta, rel
        return "neutral", delta, rel

    if metric == "ate_rmse":
        if delta > args.ate_abs_regress and rel > args.ate_rel_regress:
            return "regress", delta, rel
        if delta < -args.ate_abs_improve and rel < -args.ate_rel_improve:
            return "improve", delta, rel
        return "neutral", delta, rel

    abs_regress = args.lat_abs_regress if metric == "lat_p95" else args.error_abs_regress
    rel_regress = args.lat_rel_regress if metric == "lat_p95" else args.error_rel_regress
    abs_improve = args.lat_abs_improve if metric == "lat_p95" else args.error_abs_improve
    rel_improve = args.lat_rel_improve if metric == "lat_p95" else args.error_rel_improve
    if delta > abs_regress and rel > rel_regress:
        return "regress", delta, rel
    if delta < -abs_improve and rel < -rel_improve:
        return "improve", delta, rel
    return "neutral", delta, rel


def compare_scene(baseline_row, candidate_row, args):
    rows = []
    regressions = 0
    improvements = 0
    for metric in METRICS:
        b = as_float(baseline_row.get(metric))
        c = as_float(candidate_row.get(metric))
        verdict, delta, rel = metric_verdict(metric, b, c, args)
        regressions += int(verdict == "regress")
        improvements += int(verdict == "improve")
        rows.append({
            "scene": baseline_row["scene"],
            "baseline_run": baseline_row["run"],
            "candidate_run": candidate_row["run"],
            "metric": metric,
            "baseline": b,
            "candidate": c,
            "delta": delta,
            "delta_pct": rel * 100.0 if math.isfinite(rel) else float("nan"),
            "verdict": verdict,
        })
    if candidate_row["run"] == baseline_row["run"]:
        status = "baseline_only"
    elif regressions:
        status = "regress"
    elif improvements:
        status = "pass_with_improvement"
    else:
        status = "pass_neutral"
    summary = {
        "scene": baseline_row["scene"],
        "baseline_run": baseline_row["run"],
        "candidate_run": candidate_row["run"],
        "scene_status": status,
        "regressions": regressions,
        "improvements": improvements,
        "ate_delta": as_float(candidate_row.get("ate_rmse")) - as_float(baseline_row.get("ate_rmse")),
        "end_error_delta": as_float(candidate_row.get("end_error")) - as_float(baseline_row.get("end_error")),
        "along_p95_delta": as_float(candidate_row.get("along_p95")) - as_float(baseline_row.get("along_p95")),
        "lat_p95_delta": as_float(candidate_row.get("lat_p95")) - as_float(baseline_row.get("lat_p95")),
        "length_ratio_abs_error_delta": abs(as_float(candidate_row.get("length_ratio")) - 1.0) - abs(as_float(baseline_row.get("length_ratio")) - 1.0),
    }
    return summary, rows


def overall_status(scene_summaries, allow_baseline_only=False):
    if any(row["scene_status"] == "regress" for row in scene_summaries):
        return "FAIL_REGRESSION"
    if not allow_baseline_only and any(row["scene_status"] == "baseline_only" for row in scene_summaries):
        return "INCOMPLETE"
    if any(row["scene_status"] == "pass_with_improvement" for row in scene_summaries):
        return "PASS_WITH_IMPROVEMENT"
    return "PASS_NEUTRAL"


def write_markdown(path, baseline_path, output_dir, scene_summaries, metric_rows, args):
    status_counts = {}
    for row in scene_summaries:
        status_counts[row["scene_status"]] = status_counts.get(row["scene_status"], 0) + 1
    overall = overall_status(scene_summaries, allow_baseline_only=args.allow_baseline_only)
    lines = [
        "# Trajectory Regression Report",
        "",
        f"Overall: **{overall}**",
        "",
        f"Baseline: `{baseline_path}`",
        f"Output: `{output_dir}`",
        "",
        "Default policy requires a candidate result for every baseline scene; unchanged baseline scenes make the report INCOMPLETE.",
        "",
        "## Scene Summary",
        "",
        "| scene | status | baseline run | candidate run | ATE delta | end delta | along p95 delta | lat p95 delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in scene_summaries:
        lines.append(
            "| {scene} | {status} | {base} | {cand} | {ate:.4f} | {end:.4f} | {along:.4f} | {lat:.4f} |".format(
                scene=row["scene"],
                status=row["scene_status"],
                base=row["baseline_run"],
                cand=row["candidate_run"],
                ate=row["ate_delta"],
                end=row["end_error_delta"],
                along=row["along_p95_delta"],
                lat=row["lat_p95_delta"],
            )
        )
    lines.extend([
        "",
        "## Regressions",
        "",
    ])
    regressions = [row for row in metric_rows if row["verdict"] == "regress"]
    if not regressions:
        lines.append("No metric crossed the configured regression threshold.")
    else:
        lines.extend([
            "| scene | metric | baseline | candidate | delta | delta % |",
            "|---|---|---:|---:|---:|---:|",
        ])
        for row in regressions:
            lines.append(
                "| {scene} | {metric} | {baseline:.4f} | {candidate:.4f} | {delta:.4f} | {delta_pct:.2f}% |".format(**row)
            )
    lines.extend([
        "",
        "## Thresholds",
        "",
        f"- ATE regression: delta > {args.ate_abs_regress} m and relative > {args.ate_rel_regress:.1%}",
        f"- Shape-error regression: delta > {args.error_abs_regress} m and relative > {args.error_rel_regress:.1%}",
        f"- Lateral p95 regression: delta > {args.lat_abs_regress} m and relative > {args.lat_rel_regress:.1%}",
        f"- Length-ratio regression: abs(length_ratio - 1) worsens by > {args.length_ratio_abs_regress}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--candidate-run",
        action="append",
        default=[],
        help="Explicit candidate run as SCENE=RUN. May be repeated. Defaults to latest complete run per scene.",
    )
    parser.add_argument("--ate-abs-regress", type=float, default=0.05)
    parser.add_argument("--ate-rel-regress", type=float, default=0.02)
    parser.add_argument("--ate-abs-improve", type=float, default=0.05)
    parser.add_argument("--ate-rel-improve", type=float, default=0.02)
    parser.add_argument("--error-abs-regress", type=float, default=0.10)
    parser.add_argument("--error-rel-regress", type=float, default=0.03)
    parser.add_argument("--error-abs-improve", type=float, default=0.10)
    parser.add_argument("--error-rel-improve", type=float, default=0.03)
    parser.add_argument("--lat-abs-regress", type=float, default=0.10)
    parser.add_argument("--lat-rel-regress", type=float, default=0.10)
    parser.add_argument("--lat-abs-improve", type=float, default=0.10)
    parser.add_argument("--lat-rel-improve", type=float, default=0.10)
    parser.add_argument("--length-ratio-abs-regress", type=float, default=0.005)
    parser.add_argument("--length-ratio-abs-improve", type=float, default=0.005)
    parser.add_argument(
        "--allow-baseline-only",
        action="store_true",
        help="Allow scenes whose latest candidate run is the frozen baseline run. By default this makes the report INCOMPLETE.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    baseline_path = args.baseline_csv.resolve()
    results_dir = args.results_dir.resolve()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir.resolve() if args.output_dir else results_dir / "regression_reports" / stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = read_csv(baseline_path)
    if not baseline_rows:
        raise ValueError(f"No rows in baseline CSV: {baseline_path}")
    explicit_runs = parse_candidate_runs(args.candidate_run)

    scene_summaries = []
    metric_rows = []
    candidate_rows = []
    for baseline_row in baseline_rows:
        scene = baseline_row["scene"]
        if scene in explicit_runs:
            run_dir = run_dir_for_scene(results_dir, scene, explicit_runs[scene])
            if not run_dir.exists():
                raise FileNotFoundError(f"Missing explicit candidate run for {scene}: {run_dir}")
        else:
            run_dir = latest_complete_run(results_dir, scene)
            if run_dir is None:
                raise FileNotFoundError(f"No complete candidate run found for {scene}")
        candidate_row = load_metrics_from_run(run_dir)
        candidate_rows.append(candidate_row)
        summary, rows = compare_scene(baseline_row, candidate_row, args)
        scene_summaries.append(summary)
        metric_rows.extend(rows)

    write_csv(output_dir / "candidate_shape_summary.csv", candidate_rows, list(baseline_rows[0].keys()))
    write_csv(
        output_dir / "regression_scene_summary.csv",
        scene_summaries,
        [
            "scene",
            "baseline_run",
            "candidate_run",
            "scene_status",
            "regressions",
            "improvements",
            "ate_delta",
            "end_error_delta",
            "along_p95_delta",
            "lat_p95_delta",
            "length_ratio_abs_error_delta",
        ],
    )
    write_csv(
        output_dir / "regression_metric_details.csv",
        metric_rows,
        [
            "scene",
            "baseline_run",
            "candidate_run",
            "metric",
            "baseline",
            "candidate",
            "delta",
            "delta_pct",
            "verdict",
        ],
    )
    write_markdown(output_dir / "regression_report.md", baseline_path, output_dir, scene_summaries, metric_rows, args)

    status = overall_status(scene_summaries, allow_baseline_only=args.allow_baseline_only)
    manifest = {
        "baseline_csv": str(baseline_path),
        "results_dir": str(results_dir),
        "output_dir": str(output_dir),
        "scenes": [row["scene"] for row in scene_summaries],
        "overall_status": status,
        "files": {
            "candidate_shape_summary": str(output_dir / "candidate_shape_summary.csv"),
            "scene_summary": str(output_dir / "regression_scene_summary.csv"),
            "metric_details": str(output_dir / "regression_metric_details.csv"),
            "report": str(output_dir / "regression_report.md"),
        },
    }
    with open(output_dir / "regression_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True)

    print(f"Overall: {manifest['overall_status']}")
    print(f"Wrote report: {output_dir / 'regression_report.md'}")
    for row in scene_summaries:
        print(
            f"{row['scene']}: {row['scene_status']} | "
            f"ATE {row['ate_delta']:+.4f} m, "
            f"end {row['end_error_delta']:+.4f} m, "
            f"along_p95 {row['along_p95_delta']:+.4f} m"
        )
    return 1 if manifest["overall_status"] in {"FAIL_REGRESSION", "INCOMPLETE"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
