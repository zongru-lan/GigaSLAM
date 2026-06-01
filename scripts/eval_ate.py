"""
Compute ATE from saved poses_est.txt and GT poses_gt.npy.

GigaSLAM saves poses_est.txt as flattened C2W matrices. The railway GT
poses_gt.npy files are W2C, so GT poses are inverted to C2W before evo eval.
Use --est_convention w2c only for legacy pose files that were saved as W2C.

Usage:
    python scripts/eval_ate.py \
        --est results/.../poses_est.txt \
        --gt railway_data/poses_gt.npy \
        --save_dir results/.../plot \
        --rpe_deltas 1 5 10
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.eval_utils import evaluate_evo
from gaussian_splatting.utils.system_utils import mkdir_p


def load_est(path):
    poses = []
    with open(path) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 16:
                poses.append(np.array(vals).reshape(4, 4))
    return poses


def load_pose_indices(est_path, num_poses):
    idx_path = os.path.join(os.path.dirname(os.path.abspath(est_path)), "poses_idx.txt")
    if not os.path.exists(idx_path):
        print(f"[eval_ate] poses_idx.txt not found next to {est_path}; using sequential GT indices.")
        return list(range(num_poses)), None

    indices = []
    with open(idx_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                indices.append(int(line))

    if len(indices) != num_poses:
        print(
            f"[eval_ate] poses_idx.txt has {len(indices)} entries but poses_est has "
            f"{num_poses}; truncating to the common length."
        )
    n = min(len(indices), num_poses)
    return indices[:n], idx_path


def evaluate_saved_ate(
    est_path,
    gt_path,
    save_dir,
    label="final",
    monocular=False,
    est_convention="c2w",
    use_pose_idx=True,
    rpe_deltas=(1, 5, 10),
):
    """Evaluate saved poses against W2C GT, using poses_idx.txt if present.

    est_convention:
        c2w: default GigaSLAM convention; saved poses are already C2W.
        w2c: legacy convention; invert saved poses to C2W before eval.
    """
    est_poses = load_est(est_path)
    gt_w2c = np.load(gt_path)
    if use_pose_idx:
        pose_indices, idx_path = load_pose_indices(est_path, len(est_poses))
    else:
        pose_indices, idx_path = list(range(len(est_poses))), None
    valid_pairs = [(pose_i, gt_i) for pose_i, gt_i in enumerate(pose_indices) if 0 <= gt_i < len(gt_w2c)]
    if len(valid_pairs) < len(pose_indices):
        dropped = len(pose_indices) - len(valid_pairs)
        print(f"[eval_ate] dropped {dropped} poses whose frame index is outside GT length {len(gt_w2c)}.")
    if not valid_pairs:
        raise ValueError("No valid estimated pose / GT frame pairs for ATE evaluation.")

    if est_convention not in {"w2c", "c2w"}:
        raise ValueError(f"Unknown est_convention: {est_convention}")

    # evaluate_evo expects C2W
    if est_convention == "w2c":
        poses_est = [np.linalg.inv(est_poses[pose_i]) for pose_i, _ in valid_pairs]
    else:
        poses_est = [est_poses[pose_i] for pose_i, _ in valid_pairs]
    poses_gt = [np.linalg.inv(gt_w2c[gt_i]) for _, gt_i in valid_pairs]
    frame_ids = [gt_i for _, gt_i in valid_pairs]

    mkdir_p(save_dir)

    trj_data = {
        "trj_id": frame_ids,
        "poses_idx_path": idx_path,
        "est_convention": est_convention,
        "use_pose_idx": use_pose_idx,
        "trj_est": [p.tolist() for p in poses_est],
        "trj_gt": [p.tolist() for p in poses_gt],
    }
    with open(os.path.join(save_dir, f"trj_{label}.json"), "w", encoding="utf-8") as f:
        json.dump(trj_data, f, indent=4)

    ate = evaluate_evo(
        poses_gt,
        poses_est,
        save_dir,
        label,
        monocular=monocular,
        frame_ids=frame_ids,
        rpe_deltas=rpe_deltas,
    )
    return float(ate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--est", required=True, help="poses_est.txt")
    parser.add_argument("--gt", required=True, help="poses_gt.npy (N,4,4) W2C")
    parser.add_argument("--save_dir", default="plot")
    parser.add_argument("--label", default="final", help="suffix for stats/plot output files")
    parser.add_argument(
        "--est_convention",
        choices=["w2c", "c2w"],
        default="c2w",
        help="pose convention stored in --est; default matches poses_est.txt saved by GigaSLAM",
    )
    parser.add_argument(
        "--ignore_pose_idx",
        action="store_true",
        help="ignore poses_idx.txt and match estimated poses to GT sequentially, matching the legacy manual script",
    )
    parser.add_argument("--monocular", action="store_true")
    parser.add_argument(
        "--rpe_deltas",
        nargs="*",
        type=int,
        default=[1, 5, 10],
        help="frame-index deltas for RPE; pass no values to skip RPE output",
    )
    args = parser.parse_args()

    ate = evaluate_saved_ate(
        args.est,
        args.gt,
        args.save_dir,
        label=args.label,
        monocular=args.monocular,
        est_convention=args.est_convention,
        use_pose_idx=not args.ignore_pose_idx,
        rpe_deltas=args.rpe_deltas,
    )
    print(f"ATE RMSE: {ate:.4f} m  |  saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
