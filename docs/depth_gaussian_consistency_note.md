# Depth-Gaussian Consistency 实验说明

本文记录 DGC（Depth-Gaussian Consistency）大胆尝试方案。它是一个可开关的实验模块，不属于当前 rowtrack350 主线；默认关闭，失败后可以整分支放弃，不影响 RailScale 论文主线。

## 动机

当前主线中，RailScale 先用铁路物理宽度先验修正单目深度尺度，然后前端 VO 和后端 Gaussian mapping 使用这份深度。此前 rendered depth 只作为诊断信号保存出来，用来比较 Gaussian map 渲染深度和输入单目深度是否一致。

DGC 的问题意识是：如果 Gaussian map 已经能渲染出深度，那么 mapping 阶段是否可以让三件事更一致：

- RailScale-corrected monocular depth
- Gaussian rendered depth
- 后端 pose refinement 与 Gaussian map 参数

这不是训练 UniDepth，也不是替换 RailScale；它只是把 rendered depth residual 从“离线诊断”升级为“mapping 阶段的弱约束”。

## 实现口径

配置入口为：

```yaml
DepthGaussianConsistency:
  enabled: false
```

主线 `configs/railway/rowtrack350/template.yaml` 默认关闭。实验配置为：

```text
configs/railway/rowtrack350/experiments/dgc_bold.yaml
```

在 monocular mapping loss 中，原 RGB loss 保留：

```text
L_mapping = L_rgb + lambda_depth * L_dgc
```

DGC 使用 log-depth residual：

```text
L_dgc_data = mean(|log(D_render) - log(D_input * exp(delta_s))|)
```

其中：

- `D_render` 来自 Gaussian renderer 的 `render_pkg["depth"]`。
- `D_input` 是当前 keyframe 的 RailScale 修正后单目深度 `viewpoint.depth`。
- `delta_s` 是每个 keyframe 的轻量 depth scale adapter。
- `delta_s` 被 clamp 到 `[-0.15, 0.15]`，并带 L2 prior。

完整损失为：

```text
L_dgc = L_dgc_data + depth_scale_prior_weight * delta_s^2
```

有效 mask 同时满足：

- input depth 与 rendered depth 是有限值；
- depth 在 `[min_depth, max_depth]` 范围内；
- RGB 有效区域；
- opacity 大于 `opacity_threshold`。

为控制显存和速度，DGC 默认 `downsample: 4`，只在 mapping loss 中使用下采样后的深度残差。

## depth scale adapter 的含义

`depth_log_scale_delta` 是每个 keyframe 的局部深度尺度微调量。它不是 RailScale 的替代品，也不是全局尺度。它表达的是：在 Gaussian map 和当前 keyframe 深度之间，如果存在小尺度偏差，是否允许后端用一个很小的 adapter 吸收这部分误差。

风险也在这里：如果 adapter 长期贴住 clamp 边界，说明它可能在自我强化错误几何，而不是修正小偏差。因此实验必须检查 `learned_depth_scale` 是否饱和。

## 诊断输出

启用 DGC 后，每次运行会在结果目录保存：

```text
depth_gaussian_consistency_diag.csv
```

主要字段：

| 字段 | 含义 |
|---|---|
| `iteration` | 后端 mapping 迭代编号 |
| `frame_id` | keyframe 原始帧编号 |
| `dgc_loss` | DGC 总损失 |
| `dgc_data_loss` | log-depth residual 数据项 |
| `dgc_prior_loss` | depth scale adapter 先验项 |
| `valid_ratio` | DGC mask 有效像素比例 |
| `learned_depth_scale` | `exp(delta_s)` 后的 keyframe 局部尺度 |
| `median_abs_residual` | rendered depth 与 target depth 的米制中位绝对残差 |
| `median_log_residual` | log-depth 残差中位数 |

同时保留已有：

```text
rendered_depth_diag/rendered_depth_diag.csv
```

用于离线比较 Gaussian rendered depth residual 与轨迹误差、RailScale 状态之间的关系。当前 `dgc_bold.yaml` 与 `full_mainline` 保持同口径：全帧计算 PSNR/SSIM/LPIPS，只保存关键帧 RGB 渲染图；rendered depth diagnostic 也记录全帧 CSV。

## 运行方案

先跑 scene16 做崩溃级 smoke test：

```bash
cd /root/GigaSLAM
mkdir -p logs/dgc_bold results/dgc_bold

/root/miniconda3/envs/gigaslam/bin/python scripts/run_metric_width_scenes.py \
  --base-config configs/railway/rowtrack350/experiments/dgc_bold.yaml \
  --scenes scene_16_train \
  --generated-config-dir configs/generated_metric_width/dgc_bold_scene16 \
  --logs-dir logs/dgc_bold \
  --tag dgc_bold \
  --continue-on-error
```

scene16 不崩后，再跑 6 个序列：

```bash
/root/miniconda3/envs/gigaslam/bin/python scripts/run_metric_width_scenes.py \
  --base-config configs/railway/rowtrack350/experiments/dgc_bold.yaml \
  --scenes scene_11_train scene_13_train scene_14_train scene_16_train scene_17_train scene_19_train \
  --generated-config-dir configs/generated_metric_width/dgc_bold \
  --logs-dir logs/dgc_bold \
  --tag dgc_bold \
  --continue-on-error
```

## 成功标准

- 6 个序列都能跑完，无 CUDA illegal memory 或 OOM。
- scene13 不明显退化。
- scene14/16/17/19 至少一个序列的 `end_error`、`along_p95` 或 ATE 有可解释改善。
- `learned_depth_scale` 不长期贴住 clamp 边界。
- DGC `valid_ratio` 稳定大于 0.05。

## 放弃标准

- scene16 就崩溃，或显存明显不可控。
- 全 6 序列 ATE/shape 普遍退化。
- learned depth scale 大量饱和，说明 map-depth 闭环在自我强化错误。
- rendered depth residual 与轨迹/形态指标仍无稳定关系。

## 论文写作位置

如果 DGC 成功，它可以作为“深度-地图一致性约束”的实验模块，和 RailScale 的物理先验形成互补：RailScale 提供可解释尺度先验，DGC 检查并约束 Gaussian map 与 corrected depth 的几何一致性。

如果 DGC 失败，也有价值：它说明在当前铁路单目 SLAM 中，直接把 rendered depth residual 加入 mapping loss 可能会放大单目深度和 Gaussian map 的共同偏差，因此主线应保持 RailScale 这种外部物理先验，而不是让 map-depth 闭环自监督。
