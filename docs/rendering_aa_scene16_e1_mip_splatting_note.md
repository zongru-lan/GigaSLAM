# E1 Mip-Splatting 抗混叠移植记录

## 目标

scene16 的 LPIPS 偏高主要来自铁路枕木、碎石、轨道边缘等高频重复纹理在视角变化下的错位和 aliasing。E1 不再继续调 LPIPS loss，而是直接移植 Mip-Splatting 的抗混叠机制，测试它是否能提升 Gaussian map 的渲染质量。

## 实现口径

本项目没有直接替换外部 rasterizer。原因是当前 `submodules/diff-gaussian-rasterization` 已经扩展了 depth、opacity、`n_touched` 和后端 pose delta 梯度，直接替换会破坏 SLAM 后端。

本次实现采用源码级移植：

- 在现有 CUDA rasterizer 中加入 Mip-Splatting 风格的 2D footprint filter。
- 将原先固定 `0.3` 的低通项替换为 Mip 默认 `kernel_size=0.1`。
- 根据 2D covariance determinant ratio 调整 opacity，避免单纯扩大 footprint 造成过亮或过糊。
- 在 Python 渲染入口增加可选 3D smoothing：`pipeline_params.railway_mip_filter: true`。
- 3D smoothing 使用 `distance / focal * sqrt(0.2)` 估计 filter，并对 scale / opacity 做 determinant compensation。

## 实验配置

配置文件：`configs/railway/rowtrack350/experiments/mip_splatting_scene16.yaml`

关键设置：

- 继承 full mainline。
- `Hierarchical.color_refinement_iter: 400`。
- `pipeline_params.railway_mip_filter: true`。
- 只跑 `scene_16_train`。
- 输出到 `results/rendering_aa_scene16/e1_mip_splatting`。

## 对比基线

使用 full mainline scene16：

- 结果目录：`results/full_mainline/GigaSLAM_railway_data_scene_16_train/2026-05-29-13-50-06-No-LC`
- PSNR: `21.5284`
- SSIM: `0.8829`
- LPIPS: `0.6571`
- ATE: `2.7601 m`

## 判定标准

优先看 LPIPS mean / p95 是否下降，同时检查 PSNR、SSIM 和关键帧渲染图是否没有明显变糊。若编译失败、显存不可控、LPIPS 不降或画面明显糊化，则记录为失败并切换到下一类源码机制。

## 运行结果

运行命令由 `scripts/run_metric_width_scenes.py` 生成 scene16 配置并执行。

- 日志：`logs/rendering_aa_scene16/e1_mip_splatting/scene16_e1_mip_splatting_20260530_125348.log`
- 脚本内部日志：`/autodl-fs/data/GigaSLAM/logs/scene16_e1_mip_splatting_metric_width_1600_20260530_125451.log`
- 结果目录：`results/rendering_aa_scene16/e1_mip_splatting/GigaSLAM_railway_data_scene_16_train/2026-05-30-12-55-16-No-LC`
- 保存关键帧渲染图：`90` 张，命名为 `img_000.png` 到 `img_178.png` 的偶数帧编号。
- `color_refinement`: `36000` iterations，约 10 分钟。
- GPU 显存峰值观察值：约 `22.9 GB / 24.6 GB`，没有 OOM。

| 方法 | PSNR | SSIM | LPIPS | ATE | 结论 |
|---|---:|---:|---:|---:|---|
| full mainline | 21.5284 | 0.8829 | 0.6571 | 2.7601 m | baseline |
| E1 Mip-Splatting | 21.3395 | 0.8732 | 0.6445 | 2.7690 m | LPIPS 小幅改善，PSNR/SSIM 小幅下降 |
| delta | -0.1889 | -0.0097 | -0.0126 | +0.0089 m | partial positive |

## 判断

E1 不是强成功，但也不是失败。它证明了铁路高频纹理问题更适合从 anti-aliasing / multi-scale footprint 入手，而不是单纯优化 LPIPS loss。当前结果的主要不足是 PSNR/SSIM 同时下降，说明 footprint 过滤可能牺牲了一部分像素级锐度。下一轮更建议尝试 pixel-aware densification 或更细粒度的边缘/像素级 Gaussian 增密，而不是继续调 `kernel_size`。
