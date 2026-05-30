# 渲染评估口径与分辨率拆解记录

## 背景

当前 full-mainline 的 `eval_rendering()` 口径是：Gaussian renderer 在 `Hierarchical.rendering_width=1280` 的工作分辨率下渲染，然后将渲染图上采样到原始输入图像大小 `4112x2504`，再与原始 GT 计算 PSNR / SSIM / LPIPS。

铁路场景包含大量高频重复纹理，例如枕木、碎石、钢轨边缘、接触网和远处轨道线。LPIPS 对这些高频结构的错位和插值伪影非常敏感。因此需要拆清楚：LPIPS 偏高到底来自建图质量，还是来自“低分辨率渲染结果与 4K GT 直接比较”的评估口径。

## 诊断方法

脚本：`scripts/analyze_render_resolution_effect.py`

输入：scene16 full-mainline 已保存的关键帧渲染图与 `img/rendered_keyframes.csv`。

输出：`results/rendering_resolution_diagnostics/scene16_full_mainline_keyframes/`

做法：不重跑 SLAM，也不重渲染。直接使用已保存的关键帧渲染 PNG，与对应原始输入图像在不同比较分辨率下重新计算指标：

- `saved_original`: 当前保存图像与原始 GT 的比较口径，约 `4112x2504`；
- `w1280`: 渲染图和 GT 同时 resize 到宽度 1280；
- `w960`: 同时 resize 到宽度 960；
- `w640`: 同时 resize 到宽度 640。

注意：这个实验只覆盖已保存的关键帧渲染图，因此是 keyframe-only 诊断，不替代正式 all-frame rendering 指标。

## scene16 结果

| 比较分辨率 | PSNR mean | SSIM mean | LPIPS mean | LPIPS p95 |
| --- | ---: | ---: | ---: | ---: |
| saved_original `4112x2504` | 23.3055 | 0.6247 | 0.6301 | 0.6599 |
| w1280 `1280x779` | 23.2924 | 0.6015 | 0.4113 | 0.4472 |
| w960 `960x585` | 23.2828 | 0.6262 | 0.3619 | 0.3999 |
| w640 `640x390` | 23.2812 | 0.6672 | 0.2874 | 0.3329 |

## 结论

LPIPS 高有很大一部分来自评估分辨率口径，而不完全是 Gaussian map 外观能力不足。

关键观察：PSNR 在不同分辨率下几乎不变，约 `23.28-23.31`；但 LPIPS 从 `0.6301` 降到 `0.4113`，再到 `0.2874`。这说明 LPIPS 对铁路高频纹理和上采样后的结构错位非常敏感。当前 `rendering_width=1280` 的渲染结果如果直接上采样到 4K 与 GT 比较，会显著放大 LPIPS 惩罚。

## 对后续实验的影响

1. 不应把当前 4K LPIPS 过高简单解释为颜色 refinement 或 LPIPS loss 没有优化好。
2. 正式论文中建议同时报告：
   - all-frame 4K-upsample 口径：保持与现有 full-mainline 一致；
   - keyframe/native-resolution 诊断口径：解释 LPIPS 对高频铁路纹理和分辨率的敏感性；
   - 如果后续补全 all-frame native-resolution 指标，可作为更公平的 rendering-width 对齐指标。
3. 后续提升 LPIPS 的优先方向不是继续调 LPIPS loss，而是：
   - 提高 renderer 原生输出分辨率或多尺度评估；
   - 使用 Mip/Analytic anti-aliasing 这类针对采样与高频纹理的机制；
   - 在论文中明确 LPIPS 在铁路高频重复纹理场景下的口径敏感性。

## 状态

本诊断实验已完成。它不是失败实验；它解释了为什么先前 LPIPS 优化方向收益有限：主要矛盾中包含明显的评估分辨率因素。
