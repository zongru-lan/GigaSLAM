# Full Mainline 全量运行说明

本文记录 full-mainline 运行配置、`motion_thresh` 的含义，以及前端逐帧位姿估计和后端关键帧建图之间的关系。

## 目标

本轮 full-mainline 的目标是生成完整结果，包括：

- 每帧前端位姿估计；
- ATE 与轨迹图；
- 后端 Gaussian 建图；
- 渲染评估；
- 颜色 refinement；
- 输出到单独结果目录，避免覆盖当前主线轨迹实验结果。

结果目录：

- `results/full_mainline`

日志目录：

- `logs/full_mainline`

## 配置位置

公共模板：

- `configs/railway/rowtrack350/full_mainline/template.yaml`

单序列配置：

- `configs/railway/rowtrack350/full_mainline/scenes/scene_11_train.yaml`
- `configs/railway/rowtrack350/full_mainline/scenes/scene_13_train.yaml`
- `configs/railway/rowtrack350/full_mainline/scenes/scene_14_train.yaml`
- `configs/railway/rowtrack350/full_mainline/scenes/scene_16_train.yaml`
- `configs/railway/rowtrack350/full_mainline/scenes/scene_17_train.yaml`
- `configs/railway/rowtrack350/full_mainline/scenes/scene_19_train.yaml`

## 关键配置

```yaml
Results:
  save_dir: results/full_mainline
  eval_rendering: true
  logging:
    quiet: true
  rendering_eval:
    eval_rgb_metrics: true
    save_rgb: true
    save_rgb_keyframes_only: true
    save_downsample_rgb: false
    filename_from_input: true

SLAM:
  motion_thresh: 0.0

Hierarchical:
  color_refinement_iter: 10
```

含义：

| 参数 | 当前值 | 含义 | 为什么这样设 |
|---|---:|---|---|
| `Results.save_dir` | `results/full_mainline` | 指定 full-mainline 结果输出根目录。 | 与当前轨迹主线、消融实验结果隔离。 |
| `Results.eval_rendering` | `true` | SLAM 结束后执行渲染评估，生成渲染指标和可选渲染图。 | 本轮目标包含建图/渲染质量。 |
| `Results.logging.quiet` | `true` | 启用安静日志模式，过滤逐帧刷屏信息。 | 保留 tqdm、阶段日志、评估指标和错误信息，让日志更适合长期保存。 |
| `Results.rendering_eval.eval_rgb_metrics` | `true` | 对所有评估帧计算 PSNR/SSIM/LPIPS。 | 保留全序列渲染质量指标口径。 |
| `Results.rendering_eval.save_rgb_keyframes_only` | `true` | 只把关键帧渲染图保存到 `img/`。 | 减少磁盘占用，同时保留人工检查需要的代表帧。 |
| `Results.rendering_eval.save_downsample_rgb` | `false` | 不保存 `downsample_img_*`。 | 最终查看主要看上采样后的 `img_*.png`。 |
| `Results.rendering_eval.filename_from_input` | `true` | 渲染图文件名使用输入图像编号，例如 `img_078.png`。 | 方便追踪渲染图与原始输入帧的对应关系。 |
| `Hierarchical.color_refinement_iter` | `10` | 后端颜色 refinement 迭代数。实际总迭代数约为 `color_refinement_iter * keyframe_count`。 | 当前模板用于较快验证；若做最终渲染质量结果，可按实验需求调回更高迭代数。 |
| `SLAM.motion_thresh` | `0.0` | 前端运动量跳帧阈值。`0.0` 表示关闭该跳帧机制。 | 保证每帧都做位姿估计，所有保存的 pose 可参与 ATE/轨迹评估。 |

## 安静日志模式

`Results.logging.quiet: true` 只影响终端/日志输出，不改变 SLAM、建图、渲染和评估逻辑。

当前会过滤的高频信息包括：

- `BACKEND: add_next_kf idx:*`
- `Loop Closure is disabled!`
- `GUI gaussian packet sent`
- `VIZ render:*`

保留的信息包括：运行结果目录、初始化/重置、`Selected keyframe: frame ...`、color refinement 的 tqdm 进度条、`Map refinement done`、`Saving rendered frame ...`、ATE、PSNR/SSIM/LPIPS、保存图表摘要、报错和 traceback。安静模式下，color refinement 的 tqdm 会优先直接写到当前终端，避免被 `tee` 记进日志；如果当前环境没有 tty，则退回默认输出。

关键帧和渲染图片对应关系不会丢失，仍会写入结果目录中的 `kf_indices`、`poses_idx.txt`、`img/rendered_keyframes.csv` 等文件。

## 渲染评估与图片保存策略

full-mainline 的渲染评估采用“全帧算指标、关键帧存图片”的口径。

具体行为：

- PSNR / SSIM / LPIPS：对所有前端保存的帧计算；
- `poses_est.txt` 和 `poses_idx.txt`：仍对应所有评估帧；
- `img/`：只保存关键帧渲染图；
- `downsample_img_*`：不保存；
- `img/rendered_keyframes.csv`：记录关键帧渲染图与输入图像的对应关系。

命名规则：

```text
输入图像: 078_1638364468.000000000.png
渲染图:   img_078.png
```

这样可以同时满足两个需求：指标仍然是全序列口径，人工查看时只需要检查关键帧渲染结果。

## 前端逐帧位姿估计

当 `SLAM.motion_thresh: 0.0` 时，前端不会因为图像运动量太小而跳过帧。

每个输入帧都会进入以下流程：

1. 读取当前图像；
2. 运行深度估计；
3. 运行 RailScale，对深度尺度进行米制约束；
4. 执行前端 VO / tracking；
5. 保存当前帧 pose；
6. 根据关键帧逻辑判断是否送入后端。

因此，最终 `poses_est.txt` 和 ATE 评估使用的是前端保存的所有非跳过帧 pose，而不是只使用关键帧。

## `motion_thresh` 的真实作用

`motion_thresh` 不是后端关键帧阈值。

代码中它位于前端处理早期：

```python
if cur_frame_idx > 2 and self.motion_thresh > 0:
    flow = self.classic_tracking.motion_flow_keyframe(...)
    if flow < self.motion_thresh:
        cur_frame_idx += 1
        continue
```

所以：

- `motion_thresh > 0`：可能整帧跳过；
- 被跳过的帧不会做深度估计；
- 被跳过的帧不会做 RailScale；
- 被跳过的帧不会做 VO；
- 被跳过的帧不会保存 pose；
- 被跳过的帧也不会进入后端。

本项目当前需要“每帧都有前端位姿”，因此 full-mainline 继续使用：

```yaml
SLAM:
  motion_thresh: 0.0
```

## 后端关键帧选择机制

后端并不是接收所有帧，而是只接收前端判定为 keyframe 的帧。

关键帧判断函数是 `utils/slam_frontend.py` 中的 `is_keyframe()`，主要使用：

```yaml
Training:
  kf_interval: 15
  kf_translation: 0.08
  kf_min_translation: 0.05
  kf_overlap: 0.9
```

当前代码中的主要判断是：

```python
dist_check = dist > kf_translation * median_depth
dist_check2 = dist > kf_min_translation * median_depth
interval_check = (cur_frame_idx - last_keyframe_idx) >= kf_interval

return interval_check or dist_check2 or dist_check
```

工程含义：

| 条件 | 含义 |
|---|---|
| `interval_check` | 距离上一个 keyframe 的帧间隔达到 `kf_interval`，即使运动不大也定期加入关键帧。 |
| `dist_check` | 当前帧相对上一个 keyframe 的相机平移超过 `kf_translation * median_depth`。 |
| `dist_check2` | 当前帧相对上一个 keyframe 的相机平移超过 `kf_min_translation * median_depth`。 |

注意：当前代码中 `kf_overlap` 被读取，但在实际 return 逻辑中没有参与最终 keyframe 判断。

## 前端和后端的关系

本轮 full-mainline 的期望行为是：

- 前端：每帧都进行深度估计、RailScale、VO 位姿估计；
- 评估：使用每帧保存的 pose 计算 ATE 和绘制轨迹；
- 后端：只接收关键帧，用关键帧更新 Gaussian map；
- 渲染评估：SLAM 结束后基于后端 Gaussian map 执行；
- 颜色 refinement：渲染评估前触发，迭代数由 `color_refinement_iter` 控制。

这正好对应当前配置：

```yaml
SLAM.motion_thresh: 0.0
Results.eval_rendering: true
Hierarchical.color_refinement_iter: 10
```

## 单序列运行命令

以 `scene_11_train` 为例：

```bash
cd /root/GigaSLAM
mkdir -p logs/full_mainline results/full_mainline

/root/miniconda3/envs/gigaslam/bin/python slam.py \
  --config configs/railway/rowtrack350/full_mainline/scenes/scene_11_train.yaml \
  2>&1 | tee logs/full_mainline/scene_11_train_full_mainline_$(date +%Y%m%d_%H%M%S).log
```

其他序列只需要替换配置文件和日志文件名前缀。

## 注意事项

- 当前模板为 `color_refinement_iter: 10`，适合先验证流程、日志和输出文件；如果要做最终渲染质量结果，可以按实验需求提高迭代数并在记录中注明。
- 如果出现 OOM，优先不要改 `motion_thresh`，因为它会破坏“每帧都有 pose”的需求。
- full-mainline 结果应与 `results/trajectory_shape_summary.csv` 当前主线轨迹结果分开记录，避免混淆。

