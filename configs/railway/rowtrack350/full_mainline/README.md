# Full Mainline Configs

这组配置用于生成完整主线结果，包括轨迹评估、Gaussian 建图、渲染评估和颜色 refinement。

公共模板：

- `configs/railway/rowtrack350/full_mainline/template.yaml`

关键覆盖项：

- `Results.save_dir: results/full_mainline`
- `Results.eval_rendering: true`
- `Results.rendering_eval.eval_rgb_metrics: true`
- `Results.rendering_eval.save_rgb_keyframes_only: true`
- `Results.rendering_eval.save_downsample_rgb: false`
- `Results.rendering_eval.filename_from_input: true`
- `Hierarchical.color_refinement_iter: 10`
- `SLAM.motion_thresh: 0.0`

`SLAM.motion_thresh: 0.0` 表示前端不按运动阈值跳帧。每个输入帧都会进入深度估计、RailScale、VO 位姿估计和 pose 保存；后端仍然只接收关键帧。

单序列运行示例：

```bash
cd /root/GigaSLAM
mkdir -p logs/full_mainline results/full_mainline

/root/miniconda3/envs/gigaslam/bin/python slam.py \
  --config configs/railway/rowtrack350/full_mainline/scenes/scene_11_train.yaml \
  2>&1 | tee logs/full_mainline/scene_11_train_full_mainline_$(date +%Y%m%d_%H%M%S).log
```


## 渲染图片保存策略

full-mainline 会对所有前端帧计算 PSNR/SSIM/LPIPS，但 `img/` 目录只保存关键帧渲染图。渲染图按输入图像编号命名，例如输入 `078_1638364468.000000000.png` 对应保存为 `img_078.png`。

同时会保存 `img/rendered_keyframes.csv`，记录 `frame_idx`、输入图像名、渲染图文件名和关键帧标记。

不会保存 `downsample_img_*`，以减少磁盘占用。
