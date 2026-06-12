# Baseline Export Standard

本文档定义第三方 baseline 的最小结果导出格式。目标是让不同方法的原始输出可以很乱，但进入论文对比和统一评估时都变成同一套清爽结构。所有 baseline 对比结果统一放在 `/root/GigaSLAM/results/paper_compare` 下。

## 目录结构

推荐导出到：

```text
results/paper_compare/<BaselineName>/<scene_name>/
  rendered_keyframes/
    img_038.png
    img_043.png
    img_048.png
    ...
  trajectory/
    poses_est_c2w.txt 或 full_traj_est_c2w.txt
    poses_frame_ids.txt 或 full_traj_frame_ids.txt
  manifest.json
  README.md
```

只保留两类核心产物：

- 关键帧渲染结果图：用于论文视觉对比。
- 估计位姿文件：用于复用 GigaSLAM 的轨迹评估和绘图代码。

baseline 原始运行目录可以保留或归档，但不作为论文对比目录。

## 渲染图规范

`rendered_keyframes/` 中只保存 baseline 的关键帧渲染结果。不要保存 GT 图像副本；统一评估脚本会从原始 `railway_data/<scene>/` 临时读取 GT RGB。

命名必须使用原始输入帧号：

```text
img_048.png -> railway_data/<scene>/048_*.png
```

图像分辨率必须统一到 GigaSLAM 主线输出的论文展示分辨率。对于当前 railway_data，scene16 的主线输出为：

```text
4112 x 2504
```

这些图用于论文视觉展示。不要用 resize 后的展示图冒充 baseline 原生渲染分辨率；如果论文中提到指标口径，应说明指标由统一脚本重新计算，并在 `manifest.json` 中记录 native render size。

GT 图像不进入 paper export 目录，避免重复占用磁盘。

## 轨迹文件规范

估计位姿文件采用 C2W：

```text
trajectory/poses_est_c2w.txt
```

或：

```text
trajectory/full_traj_est_c2w.txt
```

每行一个 flattened `4x4` 矩阵，共 16 个浮点数。

帧号文件：

```text
trajectory/poses_frame_ids.txt
```

或：

```text
trajectory/full_traj_frame_ids.txt
```

每行一个原始输入帧号。第 `i` 行帧号对应第 `i` 行位姿。

如果 baseline 只能输出关键帧位姿，也可以保存关键帧位姿，但必须在 `manifest.json` 里说明 `trajectory_subset: keyframes`。如果能输出全帧位姿，优先保存全帧。

### 位姿约定必须显式记录

不同 baseline 的位姿输出可能不在同一约定下，不能默认直接和 GigaSLAM 的 GT 比较。至少要明确：

- `trajectory_est_convention`: `c2w` 或 `w2c`。统一评估脚本会把 `w2c` 先取逆成 `c2w`。
- `coordinate_system`: 例如 `opencv_camera`、`opengl_camera`、`world_xyz`、`unknown`。
- `coordinate_transform_note`: 如果 baseline exporter 做过坐标轴变换、尺度恢复、单位换算，需要写清楚。

对于 OpenGL/OpenCV、左手/右手坐标系、轴交换、单位不是米、轨迹被 normalize 等情况，不能靠评估脚本自动猜。应在对应 baseline exporter 里先转换成项目统一的 C2W 米制世界坐标，并在 manifest 中记录转换方式。

GigaSLAM 的铁路 GT `.npy` 是 W2C，统一评估时会先取逆为 C2W。默认轨迹指标是 `SE3 no scale`，不会做尺度修正，因此 baseline 输出如果是任意尺度或归一化尺度，指标会反映真实尺度错误。

## manifest.json 建议字段

```json
{
  "method": "Splat-SLAM",
  "scene": "scene_16_train",
  "rendered_image_size": {"width": 4112, "height": 2504},
  "trajectory_file": "trajectory/full_traj_est_c2w.txt",
  "trajectory_frame_ids": "trajectory/full_traj_frame_ids.txt",
  "trajectory_est_convention": "c2w",
  "coordinate_system": "world_xyz_meter",
  "coordinate_transform_note": "exporter converts baseline poses to C2W metric world coordinates before saving",
  "gt_pose_for_eval": "/root/GigaSLAM/railway_data/gt_poses/npy/scene_16_train.npy",
  "keyframes": [
    {
      "frame_id": 48,
      "image_name": "img_048.png",
      "original_image_path": "/root/GigaSLAM/railway_data/scene_16_train/048_*.png",
      "rendered_export_path": "rendered_keyframes/img_048.png"
    }
  ]
}
```

`original_image_path` 很重要。统一评估脚本会用它找到 GT RGB 图像，重新计算 PSNR / SSIM / LPIPS。

## 统一评估口径

使用：

```bash
cd /root/GigaSLAM
/root/miniconda3/envs/gigaslam/bin/python scripts/evaluate_baseline_export.py \
  --export-dir results/paper_compare/Splat-SLAM/scene_16_train \
  --gt railway_data/gt_poses/npy/scene_16_train.npy \
  --image-dir railway_data/scene_16_train \
  --est-convention c2w
```

输出默认写到：

```text
results/paper_compare/<BaselineName>/<scene_name>/eval/
```

### 轨迹指标

统一脚本使用 GigaSLAM 轨迹口径：

- estimated poses: 由 `--est-convention` 或 manifest 的 `trajectory_est_convention` 指定，评估前统一转为 C2W
- railway GT npy: W2C，评估前取逆成 C2W
- ATE: SE3 no scale，不做尺度对齐
- RPE: 默认 delta 为 `1 / 5 / 10` 帧
- 额外形态指标：`length_ratio`、`end_error`、`lat_p95`、`along_p95`

### 渲染指标

统一脚本会从 `rendered_keyframes/img_XXX.png` 找到同帧原始 GT 图像，重新计算：

- PSNR
- SSIM
- LPIPS

输出：

```text
eval/render_metrics.csv
eval/render_metrics_summary.json
```

这些指标默认是 keyframe-only，因为 baseline export 只保存关键帧渲染图。论文表格中应标注 `keyframe rendering metrics` 或明确说明该列的评估子集。

如果需要 all-frame rendering metrics，baseline 必须额外导出所有帧渲染图，或者保留原始运行结果并写专门的 adapter。

## 轨迹绘图

统一评估会生成：

```text
eval/trajectory/plot/trj_original.json
```

可以继续复用 GigaSLAM 的 report plot 脚本：

```bash
/root/miniconda3/envs/gigaslam/bin/python scripts/make_report_trajectory_plots.py \
  --results-dir results/paper_compare/Splat-SLAM/scene_16_train/eval \
  --trj-glob 'trajectory/plot/trj_original.json' \
  --output-dir results/paper_compare/Splat-SLAM/scene_16_train/eval/report_plots/se3_no_scale \
  --alignment se3_no_scale \
  --break-step-m 25
```

`--break-step-m` 用于断开明显不连续的轨迹跳变，避免绘图时把异常瞬移硬连成误导性线段。

## 原则

- 原始 baseline 结果可以复杂，但 paper export 必须干净，并统一放在 `results/paper_compare`。
- 论文展示图统一尺寸，便于排版。
- 指标由统一脚本重算，避免不同 baseline 使用不同 PSNR/SSIM/LPIPS 实现。
- 轨迹评估复用 GigaSLAM 的 C2W / W2C / SE3 no scale / RPE 口径。

## 坐标系检查清单

接入新 baseline 前，必须逐项确认：

- 位姿矩阵是 C2W 还是 W2C。
- 平移单位是否是米。
- 是否经过 Sim3/尺度归一化。
- 世界坐标轴是否与保存的 GT 可通过刚体旋转对齐；如果涉及反射或手性变化，必须在 exporter 中修正。
- 帧号是否对应原始输入图像编号，而不是 baseline 内部连续 keyframe id。
- 是否输出全帧轨迹；如果只输出关键帧轨迹，论文表格要注明子集。

统一评估脚本只负责常规的 C2W/W2C 转换、GT 取逆、SE3 no scale 对齐和 RPE 计算。更复杂的坐标系转换必须由 baseline-specific exporter 完成。
