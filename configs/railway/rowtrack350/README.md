# rowtrack350 Railway Configs

这组配置是当前铁路 SLAM 主线。目录目标是：配置少、入口清楚、历史实验不混进主线。

## 目录结构

```text
configs/railway/rowtrack350/
├── template.yaml                 # color-refinement-0 轨迹主线模板
├── scenes/                       # 每个 scene 只覆盖 Dataset 路径
├── full_mainline/                # color-refinement-400 完整建图/渲染配置
└── ablations/no_railscale/       # RailScale 消融配置
```


## 应该跑哪个配置

轨迹主线，输出到 `results/color_refinement_0/`：

```bash
cd /home/leizongru/lzr_ws/GigaSLAM
/home/leizongru/miniconda3/envs/gigaslam/bin/python slam.py --config configs/railway/rowtrack350/scenes/scene_16_train.yaml
```

完整建图/渲染，输出到 `results/color_refinement_400/`：

```bash
cd /home/leizongru/lzr_ws/GigaSLAM
/home/leizongru/miniconda3/envs/gigaslam/bin/python slam.py --config configs/railway/rowtrack350/full_mainline/scenes/scene_16_train.yaml
```

RailScale 消融，输出到 `results/color_refinement_0/RailScale_ablation/`：

```bash
cd /home/leizongru/lzr_ws/GigaSLAM
/home/leizongru/miniconda3/envs/gigaslam/bin/python slam.py --config configs/railway/rowtrack350/ablations/no_railscale/scene_16_train.yaml
```

## 主线配置含义

`template.yaml` 只保留当前主线需要看的覆盖项，基础默认在 `configs/base_config.yaml` 中。

| 配置块 | 作用 |
|---|---|
| `Results` | 输出目录、ATE 自动评估、SE3 no scale 评估口径。 |
| `Training` | 只覆盖主线需要改的 mapping 迭代数。 |
| `Dataset.Calibration` | 12MP middle camera 标定。 |
| `SLAM` | 前端 VO、特征、跳帧、诊断输出设置。 |
| `DepthModel` | UniDepth 本地模型路径和固定版本信息。 |
| `RailScale` | 当前论文主线的米制尺度约束模块。 |
| `Hierarchical` | Gaussian map 层级、渲染宽度和 color refinement 设置。 |

## RailScale 核心参数

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `metric_width_m` | `1.6` | 当前 rail-pair 有效宽度先验，用于把单目深度拉回米制尺度。 |
| `detector_max_width_px_frac` | `0.2` | 防止跨轨/邻轨被误当成左右轨。 |
| `detector_row_consistency` | `true` | 启用行级一致性，降低道岔/多轨误选风险。 |
| `detector_row_target_tracking` | `true` | 从近处行向远处行跟踪同一对 ego rail。 |
| `detector_max_row_center_shift_px` | `350.0` | 行间中心最大允许偏移，rowtrack350 名称来自这里。 |
| `hold_last_scale_on_reject` | `true` | 检测异常时沿用上一帧有效 scale，避免坏检测污染深度。 |

RailScale 的方法解释见 `docs/railscale_module_note.md`。前端/后端帧流转见 `docs/frontend_backend_frame_flow_note.md`。

## 当前结果口径

- `results/color_refinement_0/`：6 个 train scene 的轨迹主线结果，`color_refinement_iter=0`。
- `results/color_refinement_0/trajectory_shape_summary.csv`：当前主线轨迹形态汇总。
- `results/color_refinement_0/RailScale_ablation/`：w/o RailScale 消融结果。
- `results/color_refinement_400/`：完整建图/渲染结果，目前至少包含 scene16。

## 维护规则

- 新主线参数只放在 `template.yaml`，不要复制到每个 scene。
- 每个 scene 配置只写 `inherit_from` 和 `Dataset.color_path/pose_path`。
- 批跑脚本生成的临时 YAML 默认写到 `logs/generated_configs/`，不要放进 `configs/` 主目录。
- 不针对单个 scene 调主线参数；任何主线改动都要和 `results/color_refinement_0/trajectory_shape_summary.csv` 做跨 scene 对比。
