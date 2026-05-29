#!/usr/bin/env python3
"""Run GigaSLAM metric-width RailScale experiments over multiple railway scenes.

The script reads a base YAML config, writes one generated config per scene, and
runs slam.py sequentially while teeing each run to logs/. It does not modify the
base config.
"""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml


DEFAULT_SCENES = ["scene_13_train", "scene_17_train", "scene_11_train", "scene_14_train"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        default="configs/rgb_12mp_middle.yaml",
        help="Template config to read. It is not modified.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=DEFAULT_SCENES,
        help="Scenes to run, in order. Example: scene_13_train scene_17_train",
    )
    parser.add_argument(
        "--metric-width-m",
        type=float,
        default=1.600,
        help="RailScale.metric_width_m to set in generated configs.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run slam.py. Defaults to this interpreter.",
    )
    parser.add_argument(
        "--logs-dir",
        default="logs",
        help="Directory for per-scene run logs.",
    )
    parser.add_argument(
        "--generated-config-dir",
        default="configs/generated_metric_width",
        help="Directory for generated per-scene configs.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional experiment tag inserted into generated config and log names.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later scenes if one run fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate inputs and write generated configs; do not run slam.py.",
    )
    return parser.parse_args()


def scene_id(scene_name):
    # scene_13_train -> scene13, used in log names.
    parts = scene_name.split("_")
    if len(parts) >= 2 and parts[0] == "scene":
        return f"scene{parts[1]}"
    return scene_name.replace("_", "")


def require_path(path, what):
    if not path.exists():
        raise FileNotFoundError(f"Missing {what}: {path}")


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path, cfg):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)


def validate_scene(project_root, scene):
    image_dir = project_root / "railway_data" / scene
    gt_path = project_root / "railway_data" / "gt_poses" / "npy" / f"{scene}.npy"
    require_path(image_dir, "image directory")
    require_path(gt_path, "GT pose file")
    image_count = sum(1 for p in image_dir.iterdir() if p.is_file())
    gt = np.load(gt_path, mmap_mode="r")
    if gt.ndim != 3 or gt.shape[1:] != (4, 4):
        raise ValueError(f"Unexpected GT shape for {scene}: {gt.shape}")
    if image_count != gt.shape[0]:
        raise ValueError(
            f"Frame/GT count mismatch for {scene}: images={image_count}, gt={gt.shape[0]}"
        )
    return image_dir, gt_path, image_count


def build_scene_config(base_cfg, scene, image_dir, gt_path, metric_width_m):
    cfg = json.loads(json.dumps(base_cfg))
    cfg.setdefault("Dataset", {})
    cfg["Dataset"]["color_path"] = str(image_dir)
    cfg["Dataset"]["pose_path"] = str(gt_path)

    cfg.setdefault("Results", {})
    cfg["Results"]["auto_eval_ate"] = True
    cfg["Results"]["auto_eval_use_pose_idx"] = True
    cfg["Results"]["auto_eval_compare_pose_idx"] = True
    cfg["Results"]["auto_eval_est_convention"] = "c2w"
    cfg["Results"]["auto_eval_monocular"] = False
    cfg["Results"]["auto_postprocess_trajectory"] = False

    cfg.setdefault("SLAM", {})
    cfg["SLAM"]["motion_thresh"] = 0.0
    cfg["SLAM"]["log_vo_diag"] = True

    cfg.setdefault("RailScale", {})
    rail = cfg["RailScale"]
    rail["enabled"] = True
    rail["apply_correction"] = True
    rail["correction_mode"] = "metric_width"
    rail["metric_width_m"] = float(metric_width_m)
    candidates = rail.get("metric_width_candidates_m") or []
    if all(abs(float(v) - metric_width_m) > 1e-6 for v in candidates):
        candidates.append(float(metric_width_m))
    rail["metric_width_candidates_m"] = candidates
    return cfg


def run_and_tee(cmd, cwd, log_path):
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        return proc.wait()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    base_config = (project_root / args.base_config).resolve()
    require_path(base_config, "base config")

    logs_dir = (project_root / args.logs_dir).resolve()
    generated_dir = (project_root / args.generated_config_dir).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_yaml(base_config)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    width_tag = f"{int(round(args.metric_width_m * 1000)):04d}"
    clean_tag = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in args.tag.strip())
    tag_prefix = f"{clean_tag}_" if clean_tag else ""
    tag_infix = f"_{clean_tag}" if clean_tag else ""

    print(f"Project: {project_root}")
    print(f"Base config: {base_config}")
    print(f"Scenes: {', '.join(args.scenes)}")
    print(f"Metric width: {args.metric_width_m:.3f} m")
    print(f"Logs: {logs_dir}")
    print(f"Generated configs: {generated_dir}")

    failures = []
    for scene in args.scenes:
        image_dir, gt_path, image_count = validate_scene(project_root, scene)
        cfg = build_scene_config(base_cfg, scene, image_dir, gt_path, args.metric_width_m)
        config_path = generated_dir / f"{scene}_{tag_prefix}metric_width_{width_tag}_{stamp}.yaml"
        write_yaml(config_path, cfg)

        log_path = logs_dir / f"{scene_id(scene)}{tag_infix}_metric_width_{width_tag}_{stamp}.log"
        print("\n" + "=" * 80)
        print(f"Scene: {scene} ({image_count} frames)")
        print(f"Config: {config_path}")
        print(f"Log: {log_path}")

        if args.dry_run:
            continue

        cmd = [args.python, "slam.py", "--config", str(config_path)]
        code = run_and_tee(cmd, project_root, log_path)
        if code != 0:
            failures.append((scene, code, log_path))
            print(f"[batch] {scene} failed with exit code {code}; log={log_path}")
            if not args.continue_on_error:
                break
        else:
            print(f"[batch] {scene} finished successfully; log={log_path}")

    if failures:
        print("\nFailures:")
        for scene, code, log_path in failures:
            print(f"  {scene}: exit={code}, log={log_path}")
        return 1

    print("\nAll requested scenes completed." if not args.dry_run else "\nDry run complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
