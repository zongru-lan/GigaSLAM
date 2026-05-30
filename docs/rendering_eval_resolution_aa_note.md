# Rendering Eval / Resolution / AA 拆解实验记录

本文档记录 `exp/rendering-eval-resolution-aa` 分支的实验口径。当前阶段只针对 `scene_16_train`，目标是拆开 LPIPS 偏高背后的四个因素：评估口径、原生渲染分辨率、多尺度指标、Mip-style anti-aliasing。

## 评估输出口径

本实验曾临时实现多种渲染指标口径，用来拆解 LPIPS 偏高的来源；实验结束后，相关代码已经回退到 `main` 状态，当前主线不保留这些临时评估输出。

实验中比较过的口径包括：

- `full_4k`：保持严格口径，渲染图上采样到原始输入大小后计算 PSNR / SSIM / LPIPS。
- `native`：在当前 `Hierarchical.rendering_width` 对应的原生渲染分辨率下，与下采样 GT 比较。
- `w1280`、`w960`、`w640`：渲染图和 GT 同时 resize 到相同宽度后比较。

临时输出曾包括 `render_metrics.csv`、`render_metrics_summary.json` 和扩展版 `final_result.json`。这些只是诊断产物，不是当前主线评估代码的一部分。

## Scene16 实验配置

本实验已经完成，`results/rendering_eval_resolution_aa/` 和对应 generated configs 已清理，只保留本文的结果表和结论。

如需复现实验，请从当前维护配置重新派生临时 YAML，不要把这些临时配置重新放入主线目录。

## 判断规则

- 如果 `full_4k` LPIPS 高，但 `native/w1280/w960/w640` 明显低，说明 4K 上采样评估口径是重要因素。
- 如果 `w1920` 的 `full_4k` LPIPS 明显降低，说明提高原生渲染分辨率有效。
- 如果 `mip2d_w1280` 降低 LPIPS p95 且 PSNR/SSIM 不崩，说明 AA 比单纯调 loss 更对症。
- 如果 Mip-style 画面变糊或指标不稳，停止调 kernel，记录为失败实验。


## Scene16 已完成结果

主指标仍采用严格的 `full_4k all_frames` 口径，即渲染图上采样回原始输入图大小后计算指标。

| 实验 | ATE | full_4k PSNR | full_4k SSIM | full_4k LPIPS | full_4k LPIPS p95 | native LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| native_w0640 | 2.7787 | 21.3003 | 0.8724 | 0.7003 | 0.7499 | 0.3189 |
| native_w0960 | 2.7676 | 21.1974 | 0.8729 | 0.6706 | 0.7195 | 0.3994 |
| native_w1280 | 2.7754 | 21.3394 | 0.8753 | 0.6426 | 0.6887 | 0.4267 |
| native_w1920 | 2.7677 | 21.3180 | 0.8736 | 0.6275 | 0.6778 | 0.4932 |

结论：

- 低分辨率 `native` 指标更好，不代表建图质量更好；它主要反映下采样后高频纹理被弱化，指标变宽松。
- 严格的 `full_4k` 口径下，`rendering_width` 从 640 提升到 1920 会逐步降低 LPIPS，但收益不大且 `w1920` 显存接近 24GB 上限。
- 这说明铁路场景的 LPIPS 高值是真问题，且明显受高频重复纹理、上采样和细节错位影响；后续不应通过换评估口径美化结果，而应优先研究 anti-aliasing / Mip-style / 多尺度渲染机制。
- `mip2d_w1280` 仅启动后被中止，未形成有效结果。

## 注意

`rendering_width` 会影响输入缩放、前端跟踪、后端建图和最终渲染。因此每个实验都必须同时查看 ATE/RPE 或至少 ATE，避免把轨迹变化误判成纯渲染变化。
