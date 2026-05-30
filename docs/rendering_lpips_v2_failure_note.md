# Rendering-LPIPS-v2 失败实验记录

本文记录 `exp/rendering-lpips-v2` 中已经放弃的建图质量实验。该实验不再作为后续主线继续推进。

## 实验动机

铁路场景的 LPIPS 偏高，直观看起来与轨道、枕木、碎石、接触网等高频重复纹理有关。最初尝试通过增强外观优化来降低 LPIPS：

- 打开 spherical harmonics，让 Gaussian 颜色具备视角相关表达；
- 将 `color_refinement_iter` 提高到 `400`；
- 在 color refinement 后半段加入低权重 LPIPS loss；
- refinement 阶段采样非关键帧作为额外渲染监督；
- 轨迹、RailScale、关键帧机制保持不变。

## Scene16 结果

对比 baseline：

```text
baseline:
  results/full_mainline/GigaSLAM_railway_data_scene_16_train/2026-05-29-13-50-06-No-LC

rendering_lpips_v2:
  results/rendering_lpips_v2/GigaSLAM_railway_data_scene_16_train/2026-05-29-22-44-32-No-LC
```

| 指标 | full_mainline | rendering_lpips_v2 | 变化 |
|---|---:|---:|---:|
| PSNR | 21.5284 | 21.7941 | 提升 |
| SSIM | 0.8829 | 0.8951 | 提升 |
| LPIPS | 0.6571 | 0.7083 | 变差 |
| ATE RMSE | 2.7601 m | 2.7701 m | 基本不变 |

结论：该方法对 PSNR/SSIM 有帮助，但没有完成核心目标，LPIPS 反而变差。

## 失败原因判断

最可能的原因不是 loss 权重不合适，而是问题类型不对。

铁路枕木、碎石、轨道边缘和接触网属于高频重复纹理。普通 3DGS 在远近尺度变化、像素 footprint 改变、关键帧与非关键帧视角差异较大时，容易产生 aliasing、纹理不稳定和细节错位。LPIPS 对这种高频结构错位很敏感。

因此，直接加入 LPIPS loss 可能只是在固定渲染器和固定 Gaussian 表达下做外观拟合，不能解决：

- 远距离纹理采样 aliasing；
- 近远尺度变化导致的频率不一致；
- Gaussian densification 对细小高频区域覆盖不足；
- 多视角监督下的纹理平均或轻微模糊；
- 非关键帧 pose/geometry 轻微误差造成的高频错位。

## 放弃结论

该方向停止，不再继续调：

- LPIPS loss weight；
- LPIPS start fraction；
- LPIPS downsample；
- non-keyframe sample ratio；
- color refinement iteration。

后续改为源码级机制实验，重点借鉴 anti-aliasing、multi-scale filtering、pixel-aware densification 和 view-adaptive appearance。

## 清理状态

该实验的代码、配置、结果和日志已经从当前工作区移除。本文仅保留负实验结论，避免后续重复试错。
