# E3 Pixel-GS 风格像素感知 Densification 实验记录

## 动机

E1 的 Mip-Splatting 风格滤波让 scene16 的 LPIPS 从 0.6571 降到 0.6445，但 PSNR/SSIM 略降，说明抗锯齿方向有一定信号但不够强。铁路场景里的枕木、碎石、轨道边缘属于高频重复结构，问题可能不只是采样滤波，也可能是局部 Gaussian 表达密度不足。

Pixel-GS 的核心启发是：densification 不应只看屏幕空间梯度，还应考虑每个 Gaussian 对像素的实际覆盖/贡献。覆盖大量像素且梯度较高的 Gaussian 更可能处在欠表达区域，应优先触发分裂或新增点。

## 本项目实现口径

本实验不直接替换 renderer，不改变 RailScale、前端 VO、关键帧选择和 pose 评估。我们利用当前 rasterizer 已经返回的 `n_touched`，把它接入 Scaffold-GS 原本存在但主线未启用的 anchor growing 统计路径：

- `training_statis()` 保留原始 2D mean gradient 统计；
- 当 `RenderingAADensification.pixel_aware=true` 时，用 `sqrt(n_touched)` 对 gradient 和 denominator 加权；
- 只在 mapping 阶段的关键帧窗口上统计，不采样非关键帧；
- 每隔固定 iteration 调用一次 `adjust_anchor()`，允许新增或剪枝 anchor；
- color refinement 仍为 400 iteration/frame，用于和 full_mainline 及 E1 对齐。

## 配置

配置文件：`configs/railway/rowtrack350/experiments/pixel_densification_scene16.yaml`

关键参数：

| 参数 | 当前值 | 含义 |
| --- | --- | --- |
| `RenderingAADensification.enabled` | `true` | 打开像素感知 densification 实验 |
| `pixel_aware` | `true` | 用 `n_touched` 加权梯度统计 |
| `start_iter` / `until_iter` | `10` / `1800` | 只在早中期建图阶段触发，避免后期结构剧烈变化 |
| `interval` / `check_interval` | `50` / `50` | 每 50 次 mapping iteration 检查一次 anchor growing |
| `grad_threshold` | `0.0002` | 使用 Scaffold-GS 原始默认梯度阈值 |
| `color_refinement_iter` | `400` | 与 scene16 full rendering baseline 对齐 |

## 运行命令

```bash
cd /root/GigaSLAM
mkdir -p logs/rendering_aa_scene16/e3_pixel_densification results/rendering_aa_scene16/e3_pixel_densification
/root/miniconda3/envs/gigaslam/bin/python scripts/run_metric_width_scenes.py   --base-config configs/railway/rowtrack350/experiments/pixel_densification_scene16.yaml   --scenes scene_16_train   --generated-config-dir configs/generated_metric_width/rendering_aa_scene16/e3_pixel_densification   --tag e3_pixel_densification   --continue-on-error   2>&1 | tee logs/rendering_aa_scene16/e3_pixel_densification/scene16_e3_pixel_densification_$(date +%Y%m%d_%H%M%S).log
```

## 判定标准

- 优先看 LPIPS mean / p95 是否低于 baseline 0.6571；
- PSNR/SSIM 不能灾难性下降；
- 轨迹 ATE/RPE 应基本不异常变化；
- 若 anchor 数量爆炸、OOM、渲染变糊或 LPIPS 不降，则快速放弃该机制。


## 接入修正：受控 anchor growing

第一次直接启用 Scaffold-GS 原始 `adjust_anchor()` 后，scene16 在第 2 帧附近出现严重卡顿：显存约 19GB，GPU 利用率接近 0，日志长期不前进。判断不是指标失败，而是原始 anchor growing 在铁路高分辨率帧上候选过多，duplicate removal 成本过高。

因此 E3 改为受控版本：每次 growing pass 只保留累计梯度最高的 `max_new_anchors=500` 个候选，并把检查间隔改为 200 iteration。这个改动保留 Pixel-GS 的像素覆盖度加权思想，同时避免无限制扩点导致工程上不可运行。


## 失败结论

E3 已停止并回退源码。两次 scene16 尝试都在第 2 帧附近长期停滞：

1. 直接启用像素感知 `adjust_anchor()`：显存约 19GB，GPU 利用率接近 0，日志不前进。
2. 加入 `max_new_anchors=500` 的受控候选上限后：仍在同一区域停滞，说明瓶颈不只是候选数量，而是当前 SLAM 后端与 Scaffold 原始 anchor growing / optimizer state / 多进程同步路径不兼容。

处理：删除 E3 运行结果和临时配置，回退 `utils/slam_backend.py` 与 `gaussian_splatting/scene/scaffold_model.py` 的 E3 修改。后续不再在当前分支继续 Pixel-GS-style online anchor growing。

下一步转向不改变在线建图拓扑、但直接改善高频纹理采样/外观表达的机制，例如 Analytic-Splatting 的像素面积积分或 Scaffold-GS 的 view-adaptive appearance。
