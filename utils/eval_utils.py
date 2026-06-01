import csv
import json
import os
from pathlib import Path

import cv2
import evo
import evo.tools.plot
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
from evo.core import metrics, trajectory
from evo.core.trajectory import PosePath3D
from evo.tools.settings import SETTINGS
from matplotlib import pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import wandb
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.pose_utils import get_pose
from utils.logging_utils import Log

from utils.anchor_utils import anchor_in_frustum

import torchvision
import torch.nn.functional as F


def save_estimated_poses(cameras, save_dir):
    """Dump per-frame estimated poses to {save_dir}/poses_est.txt + poses_idx.txt.

    Decoupled from eval_rendering() so the trajectory is preserved even when
    rendering evaluation is skipped (e.g. fast diagnostic runs). Each line is a
    flattened C2W 4x4 matrix obtained by inverting the camera's stored W2C
    (R, T), so scripts/eval_ate.py should read it with est_convention=c2w.
    """
    if save_dir is None:
        return
    pose_txt_path = os.path.join(save_dir, 'poses_est.txt')
    pose_idx_txt_path = os.path.join(save_dir, 'poses_idx.txt')

    with open(pose_idx_txt_path, 'w') as f:
        for idx in cameras.keys():
            f.write(f'{idx}\n')

    with open(pose_txt_path, 'w') as f:
        for idx in cameras.keys():
            frame = cameras[idx]
            R = frame.R.cpu().numpy() if hasattr(frame.R, 'cpu') else frame.R
            T = frame.T.cpu().numpy() if hasattr(frame.T, 'cpu') else frame.T
            w2c = np.eye(4)
            w2c[:3, :3] = R
            w2c[:3, 3] = T
            pose_est = np.linalg.inv(w2c)
            line = ' '.join(map(str, pose_est.flatten()))
            f.write(line + '\n')


def _pose_positions(poses):
    if len(poses) == 0:
        return np.empty((0, 3), dtype=float)
    return np.asarray([np.asarray(p)[:3, 3] for p in poses], dtype=float)


def _path_length(points):
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _rmse(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None
    return float(np.sqrt(np.mean(values * values)))


def _stat_summary(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "rmse": None,
            "std": None,
            "min": None,
            "max": None,
            "p95": None,
        }
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "rmse": float(np.sqrt(np.mean(finite * finite))),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "p95": float(np.percentile(finite, 95)),
    }


def _percentile_abs(values, percentile):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None
    return float(np.percentile(np.abs(values), percentile))


def _shape_stats(poses_gt, poses_est_aligned, errors, frame_ids, label, alignment_mode):
    gt_xyz = _pose_positions(poses_gt)
    est_xyz = _pose_positions(poses_est_aligned)
    errors = np.asarray(errors, dtype=float)
    n = len(errors)

    length_gt = _path_length(gt_xyz)
    length_est = _path_length(est_xyz)
    max_idx = int(np.argmax(errors)) if n else None
    if frame_ids is not None and max_idx is not None and len(frame_ids) == n:
        max_frame_id = int(frame_ids[max_idx])
    else:
        max_frame_id = max_idx

    gt_xy = gt_xyz[:, :2]
    est_xy = est_xyz[:, :2]
    residual_xy = est_xy - gt_xy
    if n >= 2:
        tangent = np.gradient(gt_xy, axis=0)
        tangent_norm = np.linalg.norm(tangent, axis=1)
        valid = tangent_norm > 1e-9
        tangent_unit = np.zeros_like(tangent)
        tangent_unit[valid] = tangent[valid] / tangent_norm[valid, None]
        normal_unit = np.stack([-tangent_unit[:, 1], tangent_unit[:, 0]], axis=1)
        along_error = np.sum(residual_xy * tangent_unit, axis=1)
        lateral_error = np.sum(residual_xy * normal_unit, axis=1)
    else:
        along_error = np.zeros(n, dtype=float)
        lateral_error = np.zeros(n, dtype=float)

    first_end = max(1, n // 4) if n else 0
    last_start = 3 * n // 4 if n else 0
    segments = {
        "first25_rmse": errors[:first_end],
        "mid50_rmse": errors[first_end:last_start],
        "last25_rmse": errors[last_start:],
    }

    return {
        "label": str(label),
        "alignment_mode": alignment_mode,
        "num_poses": int(n),
        "length_gt": float(length_gt),
        "length_est_aligned": float(length_est),
        "length_ratio": float(length_est / length_gt) if length_gt > 0 else None,
        "end_error": float(errors[-1]) if n else None,
        "max_error": float(errors[max_idx]) if max_idx is not None else None,
        "max_error_frame_id": max_frame_id,
        "lat_p95": _percentile_abs(lateral_error, 95),
        "along_p95": _percentile_abs(along_error, 95),
        "first25_rmse": _rmse(segments["first25_rmse"]),
        "mid50_rmse": _rmse(segments["mid50_rmse"]),
        "last25_rmse": _rmse(segments["last25_rmse"]),
    }


def _rotation_angle_deg(R):
    cos_theta = (float(np.trace(R)) - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def _compute_rpe_stats(poses_gt, poses_est, frame_ids=None, deltas=(1, 5, 10), label="final", alignment_mode="SE3 no scale"):
    """Compute frame-index RPE on paired C2W trajectories.

    Delta is measured in pose sequence indices after poses_idx.txt alignment.
    frame_ids are saved for traceability and do not change the delta pairing.
    """
    poses_gt = [np.asarray(p, dtype=float) for p in poses_gt]
    poses_est = [np.asarray(p, dtype=float) for p in poses_est]
    n = min(len(poses_gt), len(poses_est))
    if frame_ids is None:
        frame_ids = list(range(n))
    frame_ids = list(frame_ids)[:n]

    gt_xyz = _pose_positions(poses_gt[:n])
    est_xyz = _pose_positions(poses_est[:n])

    out = {
        "label": str(label),
        "alignment_mode": alignment_mode,
        "pose_convention": "c2w",
        "delta_unit": "frames",
        "num_poses": int(n),
        "translation_m_note": "evo-style SE(3) relative-pose translation error",
        "world_translation_m_note": "world-frame displacement-vector error, useful when camera-axis conventions differ",
        "deltas": {},
    }

    for delta in deltas or []:
        delta = int(delta)
        if delta <= 0:
            continue

        trans_errors = []
        world_trans_errors = []
        rot_errors = []
        top_pairs = []
        for i in range(0, n - delta):
            j = i + delta
            delta_gt = np.linalg.inv(poses_gt[i]) @ poses_gt[j]
            delta_est = np.linalg.inv(poses_est[i]) @ poses_est[j]
            err = np.linalg.inv(delta_gt) @ delta_est
            trans_err = float(np.linalg.norm(err[:3, 3]))
            world_trans_err = float(
                np.linalg.norm((est_xyz[j] - est_xyz[i]) - (gt_xyz[j] - gt_xyz[i]))
            )
            rot_err = _rotation_angle_deg(err[:3, :3])
            trans_errors.append(trans_err)
            world_trans_errors.append(world_trans_err)
            rot_errors.append(rot_err)
            top_pairs.append(
                {
                    "i": int(i),
                    "j": int(j),
                    "frame_id_i": int(frame_ids[i]) if i < len(frame_ids) else int(i),
                    "frame_id_j": int(frame_ids[j]) if j < len(frame_ids) else int(j),
                    "translation_error_m": trans_err,
                    "world_translation_error_m": world_trans_err,
                    "rotation_error_deg": rot_err,
                }
            )

        top_pairs.sort(key=lambda row: row["translation_error_m"], reverse=True)
        out["deltas"][str(delta)] = {
            "delta": int(delta),
            "num_pairs": int(len(trans_errors)),
            "translation_m": _stat_summary(trans_errors),
            "world_translation_m": _stat_summary(world_trans_errors),
            "rotation_deg": _stat_summary(rot_errors),
            "top_translation_error_pairs": top_pairs[:10],
        }

    return out


def evaluate_evo(poses_gt, poses_est, plot_dir, label, monocular=False, frame_ids=None, rpe_deltas=(1, 5, 10)):
    ## Plot
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    traj_est_aligned = trajectory.align_trajectory(
        traj_est, traj_ref, correct_scale=monocular
    )
    alignment_mode = "Sim3 scale corrected" if monocular else "SE3 no scale"

    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    Log("RMSE ATE \[m]", ape_stat, tag="Eval")

    with open(
        os.path.join(plot_dir, "stats_{}.json".format(str(label))),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(ape_stats, f, indent=4)

    shape_stats = _shape_stats(
        poses_gt=traj_ref.poses_se3,
        poses_est_aligned=traj_est_aligned.poses_se3,
        errors=ape_metric.error,
        frame_ids=frame_ids,
        label=label,
        alignment_mode=alignment_mode,
    )
    with open(
        os.path.join(plot_dir, "shape_{}.json".format(str(label))),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(shape_stats, f, indent=4)

    rpe_stats = _compute_rpe_stats(
        poses_gt=traj_ref.poses_se3,
        poses_est=traj_est_aligned.poses_se3,
        frame_ids=frame_ids,
        deltas=rpe_deltas,
        label=label,
        alignment_mode=alignment_mode,
    )
    with open(
        os.path.join(plot_dir, "rpe_{}.json".format(str(label))),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(rpe_stats, f, indent=4)

    trajectory_metrics = {
        "label": str(label),
        "alignment_mode": alignment_mode,
        "ate": ape_stats,
        "shape": shape_stats,
        "rpe": rpe_stats,
    }
    with open(
        os.path.join(plot_dir, "trajectory_metrics_{}.json".format(str(label))),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(trajectory_metrics, f, indent=4)

    plot_mode = evo.tools.plot.PlotMode.xy
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(f"ATE RMSE: {ape_stat:.4f} m ({alignment_mode})")
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")

    # patch matplotlib colorbar to inject ax= (evo uses plt.gcf().colorbar() without ax)
    import matplotlib.figure as _mpl_fig
    _orig_colorbar = _mpl_fig.Figure.colorbar
    def _patched_colorbar(self, mappable, **kwargs):
        kwargs.setdefault("ax", ax)
        return _orig_colorbar(self, mappable, **kwargs)
    _mpl_fig.Figure.colorbar = _patched_colorbar
    try:
        evo.tools.plot.traj_colormap(
            ax, traj_est_aligned, ape_metric.error, plot_mode,
            min_map=ape_stats["min"], max_map=ape_stats["max"],
        )
    finally:
        _mpl_fig.Figure.colorbar = _orig_colorbar
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "evo_2dplot_{}.png".format(str(label))), dpi=90)
    plt.close(fig)

    return ape_stat


def eval_ate(frames, kf_ids, save_dir, iterations, final=False, monocular=False):
    trj_data = dict()
    latest_frame_idx = kf_ids[-1] + 2 if final else kf_ids[-1] + 1
    trj_id, trj_est, trj_gt = [], [], []
    trj_est_np, trj_gt_np = [], []

    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        return pose

    for kf_id in kf_ids:
        kf = frames[kf_id]
        pose_est = np.linalg.inv(gen_pose_matrix(kf.R, kf.T))
        pose_gt = np.linalg.inv(gen_pose_matrix(kf.R_gt, kf.T_gt))

        trj_id.append(frames[kf_id].uid)
        trj_est.append(pose_est.tolist())
        trj_gt.append(pose_gt.tolist())

        trj_est_np.append(pose_est)
        trj_gt_np.append(pose_gt)

    trj_data["trj_id"] = trj_id
    trj_data["trj_est"] = trj_est
    trj_data["trj_gt"] = trj_gt

    plot_dir = os.path.join(save_dir, "plot")
    mkdir_p(plot_dir)

    label_evo = "final" if final else "{:04}".format(iterations)
    with open(
        os.path.join(plot_dir, f"trj_{label_evo}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(trj_data, f, indent=4)

    ate = evaluate_evo(
        poses_gt=trj_gt_np,
        poses_est=trj_est_np,
        plot_dir=plot_dir,
        label=label_evo,
        monocular=monocular,
        frame_ids=trj_id,
    )
    wandb.log({"frame_idx": latest_frame_idx, "ate": ate})
    return ate


# 定义一个函数，将张量转换为 JSON 可序列化的格式
def tensor_to_serializable(obj):
    if isinstance(obj, torch.Tensor):  # 如果是张量
        return obj.cpu().tolist()  # 移动到 CPU 并转换为列表
    elif isinstance(obj, (list, tuple)):  # 如果是列表或元组
        return [tensor_to_serializable(item) for item in obj]  # 递归处理每个元素
    elif isinstance(obj, dict):  # 如果是字典
        return {key: tensor_to_serializable(value) for key, value in obj.items()}  # 递归处理每个键值对
    else:
        return obj  # 其他类型直接返回



def _as_2d_numpy(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        arr = value.detach().float().squeeze().cpu().numpy()
    else:
        arr = np.asarray(value).squeeze()
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        return None
    return arr.astype(np.float32, copy=False)


def _resize_like(arr, target_shape, interpolation=cv2.INTER_LINEAR):
    if arr is None:
        return None
    if arr.shape == target_shape:
        return arr
    return cv2.resize(arr, (target_shape[1], target_shape[0]), interpolation=interpolation).astype(np.float32)


def _stat_percentile(values, q):
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def _save_depth_preview(path, rendered_depth, input_depth, residual, valid_mask, max_depth):
    masked_residual = np.where(valid_mask, np.abs(residual), np.nan)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=120)
    panels = [
        ("rendered depth", rendered_depth, 0.0, max_depth),
        ("input depth", input_depth, 0.0, max_depth),
        ("abs residual", masked_residual, 0.0, _stat_percentile(masked_residual[np.isfinite(masked_residual)], 95)),
    ]
    for ax, (title, image, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(image, cmap="magma", vmin=vmin, vmax=vmax if np.isfinite(vmax) and vmax > vmin else None)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def _record_rendered_depth_diag(
    diag_rows,
    diag_cfg,
    save_dir,
    frame_idx,
    render_pkg,
    frame,
    is_keyframe,
    saved_npz_count,
    saved_preview_count,
):
    rendered_depth = _as_2d_numpy(render_pkg.get("depth"))
    input_depth = _as_2d_numpy(frame.depth)
    if rendered_depth is None or input_depth is None:
        return saved_npz_count, saved_preview_count

    rendered_depth = _resize_like(rendered_depth, input_depth.shape)
    opacity = _resize_like(_as_2d_numpy(render_pkg.get("opacity")), input_depth.shape)
    opacity_threshold = float(diag_cfg.get("opacity_threshold", 0.95))

    valid_mask = (
        np.isfinite(input_depth)
        & np.isfinite(rendered_depth)
        & (input_depth > 0.0)
        & (rendered_depth > 0.0)
    )
    if opacity is not None:
        valid_mask &= np.isfinite(opacity) & (opacity > opacity_threshold)

    total_pixels = int(valid_mask.size)
    valid_pixels = int(valid_mask.sum())
    residual = rendered_depth - input_depth
    abs_residual = np.abs(residual[valid_mask])
    rel_residual = abs_residual / np.maximum(np.abs(input_depth[valid_mask]), 1e-6)
    rendered_valid = rendered_depth[valid_mask]
    input_valid = input_depth[valid_mask]

    row = {
        "frame_id": frame_idx,
        "is_keyframe": bool(is_keyframe),
        "height": int(input_depth.shape[0]),
        "width": int(input_depth.shape[1]),
        "valid_pixels": valid_pixels,
        "total_pixels": total_pixels,
        "valid_ratio": float(valid_pixels / total_pixels) if total_pixels else float("nan"),
        "opacity_threshold": opacity_threshold,
        "render_depth_median": _stat_percentile(rendered_valid, 50),
        "input_depth_median": _stat_percentile(input_valid, 50),
        "median_abs_residual": _stat_percentile(abs_residual, 50),
        "p95_abs_residual": _stat_percentile(abs_residual, 95),
        "median_rel_residual": _stat_percentile(rel_residual, 50),
        "p95_rel_residual": _stat_percentile(rel_residual, 95),
    }
    diag_rows.append(row)

    diag_root = os.path.join(save_dir, "rendered_depth_diag")
    if diag_cfg.get("save_npz", False) and saved_npz_count < int(diag_cfg.get("save_npz_limit", 20)):
        npz_dir = os.path.join(diag_root, "depth_npz")
        mkdir_p(npz_dir)
        np.savez_compressed(
            os.path.join(npz_dir, f"frame_{frame_idx:05d}.npz"),
            rendered_depth=rendered_depth.astype(np.float32),
            input_depth=input_depth.astype(np.float32),
            valid_mask=valid_mask.astype(np.uint8),
            residual=residual.astype(np.float32),
        )
        saved_npz_count += 1

    if diag_cfg.get("save_preview", True) and saved_preview_count < int(diag_cfg.get("save_preview_limit", 20)):
        preview_dir = os.path.join(diag_root, "depth_preview")
        mkdir_p(preview_dir)
        _save_depth_preview(
            os.path.join(preview_dir, f"frame_{frame_idx:05d}.png"),
            rendered_depth,
            input_depth,
            residual,
            valid_mask,
            float(diag_cfg.get("preview_max_depth", 80.0)),
        )
        saved_preview_count += 1

    return saved_npz_count, saved_preview_count


def _write_rendered_depth_diag(save_dir, diag_rows):
    if not diag_rows:
        return
    diag_root = os.path.join(save_dir, "rendered_depth_diag")
    mkdir_p(diag_root)
    fieldnames = [
        "frame_id",
        "is_keyframe",
        "height",
        "width",
        "valid_pixels",
        "total_pixels",
        "valid_ratio",
        "opacity_threshold",
        "render_depth_median",
        "input_depth_median",
        "median_abs_residual",
        "p95_abs_residual",
        "median_rel_residual",
        "p95_rel_residual",
    ]
    with open(os.path.join(diag_root, "rendered_depth_diag.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in diag_rows:
            writer.writerow(row)

def eval_rendering(
    frames,
    gaussians,
    dataset,
    save_dir,
    pipe,
    background,
    kf_indices,
    intrinsics,
    height,
    width,
    iteration="final",
    upsampling_method = 'bicubic',
    rendered_depth_diag=None,
    rendering_eval=None,
):

    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        return pose
    interval = 5
    img_pred, img_gt, saved_frame_idx = [], [], []
    end_idx = len(frames) - 1 if iteration == "final" or "before_opt" else iteration
    psnr_array, ssim_array, lpips_array = [], [], []

    pose_txt_path = os.path.join(save_dir, 'poses_est.txt')
    pose_idx_txt_path = os.path.join(save_dir, 'poses_idx.txt')
    diag_cfg = rendered_depth_diag or {}
    render_cfg = rendering_eval or {}
    diag_enabled = bool(diag_cfg.get("enabled", False))
    save_rgb = bool(render_cfg.get("save_rgb", diag_cfg.get("save_rgb", True)))
    eval_rgb_metrics = bool(render_cfg.get("eval_rgb_metrics", diag_cfg.get("eval_rgb_metrics", True)))
    save_rgb_keyframes_only = bool(render_cfg.get("save_rgb_keyframes_only", False))
    save_downsample_rgb = bool(render_cfg.get("save_downsample_rgb", True))
    filename_from_input = bool(render_cfg.get("filename_from_input", False))
    cal_lpips = None
    if eval_rgb_metrics:
        cal_lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to("cuda")
    img_save_path = os.path.join(save_dir, "img")
    if save_rgb:
        os.makedirs(img_save_path, exist_ok=True)
    kf_set = set(kf_indices or [])
    saved_rgb_rows = []
    if diag_enabled and diag_cfg.get("keyframes_only", False):
        render_indices = [idx for idx in frames.keys() if idx in kf_set]
        if not render_indices:
            Log("rendered_depth_diag keyframes_only found no matching keyframes; falling back to all frames", tag="Eval")
            render_indices = list(frames.keys())
    else:
        render_indices = list(frames.keys())
    diag_rows = []
    saved_npz_count = 0
    saved_preview_count = 0

    with open(pose_idx_txt_path, 'w') as f:
        for idx in render_indices:
            f.write(f'{idx}' + '\n')

    with open(pose_txt_path, 'w') as f:
        for idx in render_indices:
            saved_frame_idx.append(idx)
            frame = frames[idx]
            gt_image_ori = None
            if save_rgb or eval_rgb_metrics:
                _, _, gt_image_ori, _, _ = dataset[idx]
                gt_image_ori = gt_image_ori / 255
                h_ori, w_ori = gt_image_ori.shape[1], gt_image_ori.shape[2]

            pose_est = np.linalg.inv(gen_pose_matrix(frame.R, frame.T)) # C2W
            # pose_est = gen_pose_matrix(frame.R, frame.T) # W2C
            
            flat_matrix = pose_est.flatten()
            line = ' '.join(map(str, flat_matrix))
            f.write(line + '\n')


            opt_mask = torch.zeros(gaussians.get_anchor.shape[0], dtype=torch.bool, device='cuda')
            m = anchor_in_frustum(anchors=gaussians.get_anchor, 
                                        intrinsics=intrinsics, 
                                        pose=get_pose(frame), 
                                        cam_center=frame.camera_center,
                                        distance_lis = gaussians.distance_lis,
                                        levels=gaussians.get_level, 
                                        h=height, 
                                        w=width)
            opt_mask.bitwise_or_(m)

            render_pkg = render(
                frame, 
                gaussians, 
                pipe, 
                background, 
                visible_mask=opt_mask
            )
            rendering = render_pkg['render']
            if diag_enabled:
                saved_npz_count, saved_preview_count = _record_rendered_depth_diag(
                    diag_rows,
                    diag_cfg,
                    save_dir,
                    idx,
                    render_pkg,
                    frame,
                    idx in kf_set,
                    saved_npz_count,
                    saved_preview_count,
                )
            should_save_rgb = save_rgb and (not save_rgb_keyframes_only or idx in kf_set)
            if should_save_rgb and save_downsample_rgb:
                downsample_name = f'downsample_img_{idx}.png'
                torchvision.utils.save_image(rendering, os.path.join(img_save_path, downsample_name))

            if save_rgb or eval_rgb_metrics:
                rendering = F.interpolate(rendering.unsqueeze(0),
                                          size = (h_ori, w_ori),
                                          mode = upsampling_method,
                                          align_corners=False)
                rendering = rendering.squeeze(0)

                image = torch.clamp(rendering, 0.0, 1.0)
                if should_save_rgb:
                    input_image = Path(dataset.color_paths[idx]).name if hasattr(dataset, "color_paths") else ""
                    if filename_from_input and input_image:
                        frame_token = Path(input_image).stem.split("_", 1)[0]
                        img_name = f"img_{frame_token}.png"
                    else:
                        img_name = f"img_{idx}.png"
                    input_suffix = f", input={input_image}" if input_image else ""
                    Log(f"Saving rendered frame {idx}: {img_name}{input_suffix}", tag="Eval")
                    torchvision.utils.save_image(rendering, os.path.join(img_save_path, img_name))
                    saved_rgb_rows.append({
                        "frame_idx": idx,
                        "input_image": input_image,
                        "rendered_image": img_name,
                        "is_keyframe": idx in kf_set,
                    })

                if eval_rgb_metrics:
                    image = image.permute(1, 2, 0)
                    gt_image_t = torch.tensor(gt_image_ori, dtype=torch.float32).cuda().permute(1, 2, 0)

                    gt = (gt_image_t.detach().cpu().numpy()).astype(np.uint8)
                    pred = (image.detach().cpu().numpy() * 255).astype(np.uint8)
                    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
                    pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
                    img_pred.append(pred)
                    img_gt.append(gt)

                    mask = gt_image_t > 0

                    psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image_t[mask]).unsqueeze(0))
                    ssim_score = ssim((image).unsqueeze(0), (gt_image_t).unsqueeze(0))
                    lpips_score = cal_lpips((image.permute(2, 0, 1)).unsqueeze(0), (gt_image_t.permute(2, 0, 1)).unsqueeze(0))

                    psnr_array.append(psnr_score.item())
                    ssim_array.append(ssim_score.item())
                    lpips_array.append(lpips_score.item())

    if diag_enabled:
        _write_rendered_depth_diag(save_dir, diag_rows)
    if save_rgb and saved_rgb_rows:
        csv_path = os.path.join(img_save_path, "rendered_keyframes.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["frame_idx", "input_image", "rendered_image", "is_keyframe"],
            )
            writer.writeheader()
            writer.writerows(saved_rgb_rows)
        Log(
            f"Saved {len(saved_rgb_rows)} rendered keyframe images to img/",
            tag="Eval",
        )

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array)) if psnr_array else None
    output["mean_ssim"] = float(np.mean(ssim_array)) if ssim_array else None
    output["mean_lpips"] = float(np.mean(lpips_array)) if lpips_array else None
    output["psnr_array"] = psnr_array
    output["ssim_array"] = ssim_array
    output["eval_indices"] = saved_frame_idx

    if eval_rgb_metrics:
        Log(
            f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}',
            tag="Eval",
        )

        psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
        mkdir_p(psnr_save_dir)

        json.dump(
            output,
            open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
            indent=4,
        )
    else:
        Log("RGB rendering metrics disabled; saved depth diagnostics only", tag="Eval")
    return output


def save_gaussians(gaussians, name, iteration, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, "point_cloud/iteration_{}".format(str(iteration))
        )
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))


def plot_trajectory(cameras, kf_indices, save_dir):
    """关键帧轨迹俯视图（XZ 平面）"""
    xs, zs = [], []
    for idx in kf_indices:
        if idx not in cameras:
            continue
        cam = cameras[idx]
        # W2C → C2W: t_world = -R^T @ T
        R = cam.R.cpu().numpy()
        T = cam.T.cpu().numpy()
        pos = -R.T @ T
        xs.append(pos[0])
        zs.append(pos[2])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xs, zs, 'b-', linewidth=1.0, label='estimated')
    ax.scatter(xs[0], zs[0], c='green', s=60, zorder=5, label='start')
    ax.scatter(xs[-1], zs[-1], c='red', s=60, zorder=5, label='end')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('Keyframe Trajectory (Top View)')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True)

    mkdir_p(os.path.join(save_dir, "plot"))
    fig.savefig(os.path.join(save_dir, "plot", "trajectory_topview.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    Log("Saved trajectory_topview.png", tag="Eval")


def plot_metrics_curve(psnr_array, ssim_array, kf_indices, save_dir):
    """PSNR / SSIM 随关键帧变化曲线"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(kf_indices, psnr_array, 'b-o', markersize=3)
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('Rendering Quality per Keyframe')
    ax1.grid(True)

    ax2.plot(kf_indices, ssim_array, 'g-o', markersize=3)
    ax2.set_ylabel('SSIM')
    ax2.set_xlabel('Frame Index')
    ax2.grid(True)

    mkdir_p(os.path.join(save_dir, "plot"))
    fig.savefig(os.path.join(save_dir, "plot", "metrics_curve.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    Log("Saved metrics_curve.png", tag="Eval")


def plot_anchor_growth(anchor_log, save_dir):
    """锚点数量随关键帧增长曲线。anchor_log: list of (frame_idx, n_anchors)"""
    if not anchor_log:
        return
    indices, counts = zip(*anchor_log)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(indices, counts, 'r-o', markersize=3)
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Number of Anchors')
    ax.set_title('Anchor Growth over Keyframes')
    ax.grid(True)

    mkdir_p(os.path.join(save_dir, "plot"))
    fig.savefig(os.path.join(save_dir, "plot", "anchor_growth.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    Log("Saved anchor_growth.png", tag="Eval")


def save_render_video(save_dir, fps=10):
    """将 img/ 目录下的渲染图合成 mp4 视频"""
    img_dir = os.path.join(save_dir, "img")
    if not os.path.exists(img_dir):
        Log("img/ not found, skip video generation", tag="Eval")
        return

    frames = sorted([f for f in os.listdir(img_dir) if f.startswith("img_") and f.endswith(".png")])
    if not frames:
        return

    first = cv2.imread(os.path.join(img_dir, frames[0]))
    h, w = first.shape[:2]
    out_path = os.path.join(save_dir, "render.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        img = cv2.imread(os.path.join(img_dir, f))
        writer.write(img)
    writer.release()
    Log(f"Saved render.mp4 ({len(frames)} frames, {fps}fps)", tag="Eval")


def save_flythrough_video(save_dir, fps=10):
    """将 flythrough_frames/ 目录下的高斯渲染帧合成飞行视频"""
    frames_dir = os.path.join(save_dir, "flythrough_frames")
    if not os.path.exists(frames_dir):
        Log("flythrough_frames/ not found, skip flythrough video", tag="Eval")
        return

    frames = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_") and f.endswith(".png")])
    if not frames:
        return

    first = cv2.imread(os.path.join(frames_dir, frames[0]))
    h, w = first.shape[:2]
    out_path = os.path.join(save_dir, "flythrough.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        img = cv2.imread(os.path.join(frames_dir, f))
        writer.write(img)
    writer.release()
    Log(f"Saved flythrough.mp4 ({len(frames)} frames, {fps}fps)", tag="Eval")
