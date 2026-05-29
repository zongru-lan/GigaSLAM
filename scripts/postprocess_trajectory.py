"""
Offline pure-vision trajectory drift post-processing for GigaSLAM.

Input poses are flattened C2W matrices from poses_est.txt. The script only
updates translation, keeps rotations unchanged, and writes a new pose file.
It uses weak railway motion priors: monotonic forward motion, smooth step
length, and strongly smoothed lateral drift in the selected ground plane.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


PLANE_AXES = {
    "xy": (0, 1),
    "xz": (0, 2),
    "yz": (1, 2),
}


def load_poses(path):
    poses = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            vals = line.strip().split()
            if not vals:
                continue
            if len(vals) != 16:
                raise ValueError(f"Expected 16 floats per pose line, got {len(vals)} in {path}")
            poses.append(np.array([float(v) for v in vals], dtype=np.float64).reshape(4, 4))
    if not poses:
        raise ValueError(f"No poses found in {path}")
    return np.stack(poses, axis=0)


def save_poses(path, poses):
    with open(path, "w", encoding="utf-8") as f:
        for pose in poses:
            f.write(" ".join(f"{v:.10g}" for v in pose.reshape(-1)) + "\n")


def load_pose_indices(est_path, num_poses):
    idx_path = est_path.parent / "poses_idx.txt"
    if not idx_path.exists():
        print(f"[postprocess] poses_idx.txt not found next to {est_path}; using sequential GT indices.")
        return list(range(num_poses)), None

    indices = []
    with open(idx_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                indices.append(int(line))

    if len(indices) != num_poses:
        print(
            f"[postprocess] poses_idx.txt has {len(indices)} entries but poses_est has "
            f"{num_poses}; truncating to the common length."
        )
    n = min(len(indices), num_poses)
    return indices[:n], str(idx_path)


def _odd_window(window, n):
    window = int(max(1, window))
    if window % 2 == 0:
        window += 1
    return min(window, n if n % 2 == 1 else max(1, n - 1))


def median_filter(values, window):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values.copy()
    window = _odd_window(window, values.size)
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.array([np.median(padded[i:i + window]) for i in range(values.size)])


def moving_average(values, window):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values.copy()
    window = _odd_window(window, values.size)
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def hampel_filter(values, window=7, n_sigmas=3.0):
    values = np.asarray(values, dtype=np.float64).copy()
    if values.size < 3:
        return values
    window = _odd_window(window, values.size)
    radius = window // 2
    filtered = values.copy()
    for i in range(values.size):
        lo = max(0, i - radius)
        hi = min(values.size, i + radius + 1)
        local = values[lo:hi]
        med = np.median(local)
        mad = np.median(np.abs(local - med))
        sigma = 1.4826 * mad
        if sigma > 1e-12 and abs(values[i] - med) > n_sigmas * sigma:
            filtered[i] = med
    return filtered


def estimate_main_direction(points_2d, stable_frames):
    n = points_2d.shape[0]
    end_idx = min(max(1, stable_frames), n - 1)
    direction = points_2d[end_idx] - points_2d[0]
    if np.linalg.norm(direction) < 1e-8:
        centered = points_2d - points_2d.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return np.array([1.0, 0.0], dtype=np.float64)
    direction = direction / norm
    if np.dot(points_2d[-1] - points_2d[0], direction) < 0:
        direction = -direction
    return direction


def postprocess_translations(
    translations,
    plane="xz",
    stable_frames=20,
    step_hampel_window=7,
    step_smooth_window=9,
    lateral_smooth_window=31,
    lateral_keep=0.25,
):
    if translations.shape[0] < 3:
        return translations.copy(), {"changed": False, "reason": "fewer than 3 poses"}

    axes = PLANE_AXES[plane]
    points = translations[:, axes].astype(np.float64)
    origin = points[0].copy()
    rel = points - origin

    forward = estimate_main_direction(points, stable_frames)
    lateral = np.array([-forward[1], forward[0]], dtype=np.float64)

    long_coord = rel @ forward
    lat_coord = rel @ lateral

    raw_steps = np.diff(long_coord)
    forward_steps = hampel_filter(raw_steps, step_hampel_window)
    forward_steps = median_filter(forward_steps, step_hampel_window)
    forward_steps = moving_average(forward_steps, step_smooth_window)
    forward_steps = np.maximum(forward_steps, 0.0)

    target_forward = max(long_coord[-1] - long_coord[0], 0.0)
    smooth_total = float(forward_steps.sum())
    if smooth_total > 1e-12 and target_forward > 1e-12:
        forward_steps *= target_forward / smooth_total
    elif target_forward > 1e-12:
        forward_steps[:] = target_forward / forward_steps.size

    long_new = np.concatenate([[long_coord[0]], long_coord[0] + np.cumsum(forward_steps)])

    lat_smooth = moving_average(lat_coord, lateral_smooth_window)
    lat_new = lat_smooth[0] + float(lateral_keep) * (lat_smooth - lat_smooth[0])

    points_new = origin + long_new[:, None] * forward[None, :] + lat_new[:, None] * lateral[None, :]
    out = translations.copy().astype(np.float64)
    out[:, axes[0]] = points_new[:, 0]
    out[:, axes[1]] = points_new[:, 1]

    monotonic_violations = int(np.sum(np.diff(long_new) < -1e-9))
    stats = {
        "changed": True,
        "plane": plane,
        "num_poses": int(translations.shape[0]),
        "main_direction": forward.tolist(),
        "target_forward": float(target_forward),
        "raw_step_median": float(np.median(raw_steps)),
        "post_step_median": float(np.median(np.diff(long_new))),
        "raw_lateral_range": float(lat_coord.max() - lat_coord.min()),
        "post_lateral_range": float(lat_new.max() - lat_new.min()),
        "monotonic_violations": monotonic_violations,
    }
    return out, stats


def maybe_evaluate(original_poses, post_poses, est_path, gt_path, save_dir, monocular, est_convention="c2w"):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from gaussian_splatting.utils.system_utils import mkdir_p
    from utils.eval_utils import evaluate_evo

    gt_w2c = np.load(gt_path)
    pose_indices, idx_path = load_pose_indices(est_path, len(original_poses))
    n = min(len(original_poses), len(post_poses), len(pose_indices))
    valid_pairs = [(pose_i, pose_indices[pose_i]) for pose_i in range(n) if 0 <= pose_indices[pose_i] < len(gt_w2c)]
    if not valid_pairs:
        raise ValueError("No valid estimated pose / GT frame pairs for ATE evaluation.")

    if est_convention not in {"w2c", "c2w"}:
        raise ValueError(f"Unknown est_convention: {est_convention}")

    poses_gt = [np.linalg.inv(gt_w2c[gt_i]) for _, gt_i in valid_pairs]
    if est_convention == "w2c":
        poses_original = [np.linalg.inv(original_poses[pose_i]) for pose_i, _ in valid_pairs]
        poses_post = [np.linalg.inv(post_poses[pose_i]) for pose_i, _ in valid_pairs]
    else:
        poses_original = [original_poses[pose_i] for pose_i, _ in valid_pairs]
        poses_post = [post_poses[pose_i] for pose_i, _ in valid_pairs]
    frame_ids = [gt_i for _, gt_i in valid_pairs]

    mkdir_p(save_dir)
    with open(os.path.join(save_dir, "trj_original_post.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "trj_id": frame_ids,
                "poses_idx_path": idx_path,
                "est_convention": est_convention,
                "trj_gt": [p.tolist() for p in poses_gt],
                "trj_original": [p.tolist() for p in poses_original],
                "trj_post": [p.tolist() for p in poses_post],
            },
            f,
            indent=4,
        )

    ate_original = evaluate_evo(
        poses_gt,
        poses_original,
        save_dir,
        "original",
        monocular=monocular,
        frame_ids=frame_ids,
    )
    ate_post = evaluate_evo(
        poses_gt,
        poses_post,
        save_dir,
        "post",
        monocular=monocular,
        frame_ids=frame_ids,
    )
    return {"ate_original": float(ate_original), "ate_post": float(ate_post)}


def run_postprocess(
    est_path,
    out_path=None,
    plane="xz",
    stable_frames=20,
    step_hampel_window=7,
    step_smooth_window=9,
    lateral_smooth_window=31,
    lateral_keep=0.25,
    gt_path=None,
    save_dir=None,
    monocular=False,
    est_convention="c2w",
):
    """Run trajectory post-processing and optional original/post ATE eval."""
    est_path = Path(est_path)
    out_path = Path(out_path) if out_path else est_path.with_name("poses_est_post.txt")
    poses = load_poses(est_path)
    post = poses.copy()

    translations = poses[:, :3, 3]
    translations_post, stats = postprocess_translations(
        translations,
        plane=plane,
        stable_frames=stable_frames,
        step_hampel_window=step_hampel_window,
        step_smooth_window=step_smooth_window,
        lateral_smooth_window=lateral_smooth_window,
        lateral_keep=lateral_keep,
    )
    post[:, :3, 3] = translations_post
    save_poses(out_path, post)

    eval_stats = None
    if gt_path:
        eval_dir = save_dir or str(out_path.parent / "plot")
        eval_stats = maybe_evaluate(poses, post, est_path, gt_path, eval_dir, monocular, est_convention)

    return {
        "out_path": str(out_path),
        "postprocess": stats,
        "eval": eval_stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Post-process GigaSLAM poses_est.txt with weak railway motion priors."
    )
    parser.add_argument("--est", required=True, help="Input poses_est.txt (flattened C2W matrices)")
    parser.add_argument("--out", default=None, help="Output path; default: poses_est_post.txt next to input")
    parser.add_argument("--plane", choices=sorted(PLANE_AXES), default="xz", help="Ground plane axes")
    parser.add_argument("--stable_frames", type=int, default=20, help="Initial segment used to estimate forward direction")
    parser.add_argument("--step_hampel_window", type=int, default=7, help="Window for step outlier filtering")
    parser.add_argument("--step_smooth_window", type=int, default=9, help="Window for forward step smoothing")
    parser.add_argument("--lateral_smooth_window", type=int, default=31, help="Window for lateral low-pass smoothing")
    parser.add_argument("--lateral_keep", type=float, default=0.25, help="Fraction of smoothed lateral motion to preserve")
    parser.add_argument("--gt", default=None, help="Optional GT .npy W2C file for evaluation only")
    parser.add_argument("--save_dir", default=None, help="Optional eval output dir; default: <out_dir>/plot")
    parser.add_argument(
        "--est_convention",
        choices=["w2c", "c2w"],
        default="c2w",
        help="pose convention stored in --est; default matches poses_est.txt saved by GigaSLAM",
    )
    parser.add_argument("--monocular", action="store_true", help="Use evo scale correction during optional eval")
    args = parser.parse_args()

    result = run_postprocess(
        args.est,
        out_path=args.out,
        plane=args.plane,
        stable_frames=args.stable_frames,
        step_hampel_window=args.step_hampel_window,
        step_smooth_window=args.step_smooth_window,
        lateral_smooth_window=args.lateral_smooth_window,
        lateral_keep=args.lateral_keep,
        gt_path=args.gt,
        save_dir=args.save_dir,
        monocular=args.monocular,
        est_convention=args.est_convention,
    )
    print(f"Wrote post-processed poses: {result['out_path']}")
    print(json.dumps(result["postprocess"], indent=2))
    if result["eval"] is not None:
        print(json.dumps(result["eval"], indent=2))


if __name__ == "__main__":
    main()
