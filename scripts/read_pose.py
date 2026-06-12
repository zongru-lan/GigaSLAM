"""
Convert scene_14_train.parquet GT poses to per-frame 4x4 W2C numpy arrays.

Output: poses_gt.npy  — shape (N, 4, 4), W2C matrices aligned to image order.

Usage:
    python read_pose.py \
        --parquet scene_14_train.parquet \
        --img_dir /home/leizongru/GigaSLAM/railway_data/scene_14_train \
        --output 14_poses_gt.npy
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def quat_trans_to_w2c(row):
    """Build 4x4 W2C matrix from quaternion (r_x,r_y,r_z,r_w) + translation."""
    R = Rotation.from_quat([row.r_x, row.r_y, row.r_z, row.r_w]).as_matrix()
    T = np.array([row.t_x, row.t_y, row.t_z])
    w2c = np.eye(4)
    w2c[:3, :3] = R.T
    w2c[:3, 3] = -R.T @ T
    return w2c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default="scene_14_train.parquet")
    parser.add_argument("--img_dir", default="railway_data/scene_14_train")
    parser.add_argument("--output", default="poses_gt.npy")
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    gt_timestamps = df["timestamp"].values  # (M,)

    # image filenames: {index}_{timestamp}.png, sorted by index
    img_paths = sorted(glob.glob(os.path.join(args.img_dir, "*.png")))
    assert img_paths, f"No images found in {args.img_dir}"

    poses_w2c = []
    for path in img_paths:
        fname = os.path.splitext(os.path.basename(path))[0]  # e.g. "000_1638358313.099999905"
        parts = fname.split("_", 1)
        assert len(parts) == 2, f"Unexpected filename format: {fname}"
        img_ts = float(parts[1])

        # nearest-neighbour match to parquet timestamp
        diffs = np.abs(gt_timestamps - img_ts)
        idx = np.argmin(diffs)
        if diffs[idx] > 0.1:
            print(f"WARNING: {fname} timestamp gap {diffs[idx]:.3f}s")
        poses_w2c.append(quat_trans_to_w2c(df.iloc[idx]))

    poses_w2c = np.stack(poses_w2c, axis=0)  # (N, 4, 4)
    np.save(args.output, poses_w2c)
    print(f"Images: {len(img_paths)}, Parquet rows: {len(df)}, Saved: {len(poses_w2c)} W2C poses -> {args.output}")


if __name__ == "__main__":
    main()
