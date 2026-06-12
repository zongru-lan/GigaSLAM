# GigaSLAM 项目指南

> 这是当前铁路 GigaSLAM 项目的交接文档。以当前代码和 YAML 为准，本文只保留已经验证过的结论、运行方式和下一步方向。前任留下的大量纯 VO 调参流水账已从主文档移除，因为那些记录没有解决系统性尺度漂移，继续沿着它们调参会误导判断。

## 当前主线

当前主线是纯视觉 GigaSLAM，canonical 配置入口在：

```text
configs/railway/rowtrack350/template.yaml
configs/railway/rowtrack350/scenes/*.yaml
```

复现单个序列时优先使用显式 scene 配置，例如：

```bash
python slam.py --config configs/railway/rowtrack350/scenes/scene_16_train.yaml
```

当前不再维护 `configs/railway/rowtrack350/scenes/scene_16_train.yaml` 兼容入口；复现实验请直接使用 `configs/railway/rowtrack350/scenes/*.yaml`。

当前主要实验环境是 AutoDL / SeetaCloud 远端 `205`：

```text
SSH Host: 205
Remote project: /root/GigaSLAM
Remote codex: /usr/local/bin/codex -> /root/miniconda3/envs/gigaslam/bin/codex
Python env: conda env gigaslam
```

截至 `2026-05-23`，本地 Codex App 已能通过 SSH Remote 连接 `205`。新开 Codex
会话时，应优先在远端工作区 `/root/GigaSLAM` 中读取本文档和当前 YAML，而不是依赖旧聊天记录。



ATE / evo 评估口径（2026-05-26 修正）：

- `poses_est.txt` 保存的是 flattened C2W 矩阵；铁路 GT `poses_gt.npy` 是 W2C。
- `scripts/eval_ate.py` 默认 `--est_convention c2w`，评估时只对 GT 取逆到 C2W。
- `auto_eval_monocular: false` 表示 evo 只做 SE(3) 刚体对齐，不做 Sim(3) 尺度对齐；真实尺度误差会保留下来。
- 自动评估会额外输出 `plot/shape_original.json`，用于量化轨迹形态。

正式 ATE 配置：

```yaml
Results:
  auto_eval_ate: true
  auto_eval_use_pose_idx: true
  auto_eval_compare_pose_idx: true
  auto_eval_est_convention: c2w

SLAM:
  motion_thresh: 0.0
  pnp_first: true

RailScale:
  enabled: true
  apply_correction: true
  correction_mode: metric_width
  metric_width_m: 1.600
  mode: detector
  metric_width_candidates_m: [1.435, 1.500, 1.550, 1.600]
```

`motion_thresh: 0.0` 表示前端不再跳帧。它不表示所有帧都进入后端建图。当前代码中，每一帧都会进入 VO 并保存位姿；后端 Gaussian mapping 仍然只接收稀疏关键帧。

## 当前输出

运行结束后，结果统一保存到：

```text
results/GigaSLAM_railway_data_scene_16_train/<timestamp>-No-LC/
```

主要输出包括：

- `config.yml`
- `poses_est.txt`
- `poses_idx.txt`
- `gaussian_save/`
- `plot/trajectory_topview.png`
- `plot/anchor_growth.png`
- `plot/stats_original.json`
- `plot/shape_original.json`，轨迹长度、终点误差、横向/沿轨误差等形态指标
- `plot/evo_2dplot_original.png`
- `plot/trj_original.json`
- `plot/stats_legacy_ordered.json`
- `plot/shape_legacy_ordered.json`
- `plot/evo_2dplot_legacy_ordered.png`
- `vo_diag.csv`，当 `SLAM.log_vo_diag: true`

`original` 是正式指标，读取 `poses_idx.txt` 按真实帧号对齐 GT。`legacy_ordered` 只用于历史对比，在不跳帧的情况下两者应基本一致。

批量重算或整理后，可查看 `results/trajectory_shape_summary.csv` 横向比较各 scene 的 ATE 与形态指标。

### results 目录约定

`results/` 根目录只保留当前冻结主线相关内容，避免历史实验混在一起误导判断：

- `results/GigaSLAM_railway_data_scene_*_train/`：每个 scene 只保留一个当前冻结 baseline run。
- `results/trajectory_shape_summary.csv`：当前 6 个 scene baseline 的形态指标汇总。
- `results/report_plots/`：汇报用轨迹图。
- `results/regression_reports/`：跨 scene 回归门禁输出。
- `results/archive/`：非当前主线的运行结果及其派生诊断输出，按用途归档保留。

目录说明见 `results/README.md` 和 `results/archive/README.md`。不要直接删除历史结果；若不再作为当前主线使用，移动到 `results/archive/` 下对应类别。

## 汇报图生成

汇报或 PPT 使用统一样式的 report 图，不覆盖原始 evo 评估图：

```bash
cd /root/GigaSLAM
/root/miniconda3/envs/gigaslam/bin/python scripts/make_report_trajectory_plots.py \
  --results-dir results \
  --output-dir results/report_plots
```

输出文件：

- `results/report_plots/<scene>_trajectory_report.png` 和 `.pdf`：单序列汇报图，固定 `1600x1000 px`。
- `results/report_plots/trajectory_report_grid.png` 和 `.pdf`：6 个序列的 2x3 总览图。
- `results/report_plots/report_plot_manifest.json`：记录 scene、run、ATE、坐标范围、网格间距和统一色条范围。

report 图使用 `SE3 no scale` 口径，和正式 ATE 一致；它只是展示美化版，正式指标仍以 `plot/stats_original.json`、`plot/shape_original.json`、`results/trajectory_shape_summary.csv` 为准。


## 主线冻结与回归门禁

当前论文主线固定为：

```text
configs/railway/rowtrack350/template.yaml
```

主线方法是 `rowtrack350 + metric_width_m=1.6 + widthcap020 + row tracking + hold-last-scale`。后续不要为了单个序列继续调参；任何新想法必须解释其机制，并通过跨 scene 回归比较。

RailScale 的论文写作参考见 `docs/railscale_module_note.md`。该文档按模块动机、核心公式、鲁棒性设计、论文命名和消融实验组织，适合写 Method/Discussion 时参考；具体参数含义仍以 `configs/railway/rowtrack350/README.md` 为准。

Rail detector 的训练目标和与 RailScale 的分工见 `docs/rail_detector_note.md`。该文档解释 detector 只预测 left/right rail-edge heatmaps，尺度由 RailScale 几何计算得到。

EgoPathGuidance 的模块设计和训练流程见 `docs/ego_path_guidance_note.md`。这是基于 TEP-Net 思路的新候选模块，用于预测当前列车 ego-path；当前只作为 shadow diagnostic/后续 RailScale-v2 方向，默认不改变冻结主线。

冻结 baseline：

```text
results/color_refinement_0/trajectory_shape_summary.csv
```

当前 6 个 train scene 的主线轨迹结果保存在 `results/color_refinement_0/`。后续实验至少应与该汇总表中的 ATE、end_error、along_p95、lat_p95 和 length_ratio 对比；如果只改善单个 scene、没有覆盖全部 6 个 baseline scene，或使 scene13 道岔/弯道回归样本退化，则不能作为主线改进。

判断原则：不能只看 ATE，也不能只看单个 scene。若新方法让 scene19 改善但破坏 scene13 道岔/弯道，不能进入主线；若某个改动没有跨 scene 稳定收益，只保留为探针实验。

## Depth / RailScale 一致性诊断

DepthSplat 的直接框架不适合作为当前在线 SLAM 主线，但它给出的方向很有用：把 depth、尺度修正和 Gaussian/轨迹几何放在一起诊断。当前先采用不改变 SLAM 行为的后处理工具，关联 RailScale、VO depth 和轨迹误差：

```bash
cd /root/GigaSLAM
/root/miniconda3/envs/gigaslam/bin/python scripts/analyze_depth_scale_consistency.py \
  --results-dir results/archive/depthdiag_runs_20260526 \
  --output-dir results/archive/depthdiag_runs_20260526/depth_scale_diagnostics \
  --skip-shape-summary
```

输出文件：

- `results/archive/depthdiag_runs_20260526/depth_scale_diagnostics/depth_scale_consistency_summary.csv`：每个 scene 的 ATE、scale 分布、row-level depth 稳定性、VO depth 分布以及误差相关性。
- `results/archive/depthdiag_runs_20260526/depth_scale_diagnostics/depth_scale_consistency_all_frames.csv`：逐帧合并表，包含轨迹 APE、横向/沿轨误差、RailScale 状态、row depth、VO depth、PnP inlier ratio、rendered/input depth residual 和风险标记。
- `results/archive/depthdiag_runs_20260526/depth_scale_diagnostics/depth_scale_anomaly_frames.csv`：全局 top 异常帧，按风险分数和轨迹误差排序。
- `results/archive/depthdiag_runs_20260526/depth_scale_diagnostics/depth_scale_anomaly_frames_by_scene.csv`：每个 scene 单独保留 top 异常帧，避免长序列把短序列的风险淹没。
- `results/archive/depthdiag_runs_20260526/depth_scale_diagnostics/<scene>_depth_scale_consistency.png`：每个 scene 的帧级诊断图，包含 trajectory/lateral/along error、RailScale scale/rail width、VO step/PnP inlier、rendered/input depth 和 residual。
- `results/archive/depthdiag_runs_20260526/depth_scale_diagnostics/diagnostic_report.md`：自动生成的当前诊断结论和 E1-E4 后续实验路线。
- `results/archive/depthdiag_runs_20260526/depth_scale_diagnostics/depth_scale_consistency_manifest.json`：记录本次诊断范围和输出路径。

注意：普通运行不会保存数值版 Gaussian rendered depth map，所以 clean mainline 结果中 `rendered_depth_available=false`，`depth_gs_residual_*` 列为空。当前 rendered-depth feasibility/depthdiag 配置已清理；分析历史归档 depthdiag 结果时建议加 `--skip-shape-summary`，避免覆盖 root 下 clean mainline 的 `results/trajectory_shape_summary.csv`。

Rendered depth feasibility/depthdiag 配置已经清理，不再作为当前维护入口。历史 depthdiag 结果保留在 `results/archive/depthdiag_runs_20260526/`，可继续用上面的 `scripts/analyze_depth_scale_consistency.py` 做离线分析。若后续确实要恢复 rendered-depth 诊断，建议从已归档结果目录中的 `config.yml` 或 Git 历史恢复为新的、有明确命名的实验配置。

当前下一步按受控实验推进：

- `E1`：只做诊断可视化，不改算法，确认误差峰值对应 RailScale、VO 还是 GS-depth residual。
- `E2`：针对 VO 退化做门控实验，重点看 `low_pnp_inlier_ratio` 和 `vo_step_high`。
- `E3`：只保留 RailScale 异常诊断，不再采用 hard-hold 保守门控；如果继续改进 RailScale，必须是机制明确且跨 scene 有收益的通用设计，不盲目改 `metric_width_m`。
- `E4`：只有当 E1 证明 rendered depth residual 有清晰关系后，再尝试把它作为异常检测或权重调节信号，不直接作为优化损失。

优先分析顺序为 `scene_19_train`、`scene_17_train`、`scene_14_train`/`scene_16_train`、`scene_13_train`、`scene_11_train`。当前主要问题是沿轨累计漂移和末端误差，不是明显横向偏轨.

## 当前诊断

旧结果里的 `8.58m` ATE 不是可靠正式指标。它来自旧评估路径：估计轨迹被按顺序和 GT 对齐，跳帧时会把估计位姿配到错误的 GT 帧。

修正评估逻辑并关闭前端跳帧后，纯 VO / shadow 诊断基线正式 ATE 约为：

```text
ATE RMSE original ~= 22.86m
```

引入 `RailScale.correction_mode=metric_width`、`metric_width_m=1.600` 后，
`2026-05-24-13-56-10-No-LC` 的正式 ATE 降到：

```text
ATE RMSE original = 2.8285m
```

这次结果已用 `scripts/eval_ate.py` 默认读取 `poses_idx.txt` 手动复核，
`poseidx_check` 同样为 `2.8285m`。

`vo_diag.csv` 的结论：

- `pnp_first=True` 时，大部分帧走 `pnp_xval`，ATE 约 `22.86m`。
- `pnp_first=False` 时，大部分帧走 `e_depth`，ATE 约 `23.04m`。
- 两条路线都表现出类似尺度漂移。
- 后段 `depth_median` 上升。
- `step_norm` 随 `depth_median` 一起上升。

结论：核心问题不是某个 VO 参数，也不是后端关键帧密度。主要失败模式是单目系统的尺度漂移，根源是铁路场景中 UniDepth 伪度量深度尺度不稳定。

## 已删除或停用的路线

| 路线 | 结果 | 当前状态 |
|---|---|---|
| grid NMS | 无稳定收益 | 已删除 |
| F 矩阵预过滤 | 误删铁路场景中的有效匹配，ATE 变差 | 已删除 |
| Scale Corrector / MLP | OOD 泛化失败，复杂度增加 | 已删除 |
| 常数尺度补偿 | 只对局部直线段有拟合效果，弯道和后段失败 | 已删除 |
| fused 深度融合分支 | 增加复杂度，未改善 ATE | 已删除 |
| NeuralAssist / CUT3R pose override | ATE 对齐后看似降低，但轨迹形态严重错误 | 已删除 |
| 纯轨迹后处理 | 回避真实尺度问题，不能反向改善建图 | 默认关闭 |
| Hough RailScale | rail pair 检测不稳定，ATE 约 `23.27m` | 默认关闭 |

这些路线不再作为主线继续调参。主文档只保留它们的失败结论，避免后续接手者重复踩坑。

## 配置说明

当前配置文件：

```text
configs/railway/rowtrack350/scenes/scene_16_train.yaml
```

关键项：

```yaml
Results:
  auto_eval_ate: true
  auto_postprocess_trajectory: false
  auto_eval_monocular: false
  auto_eval_est_convention: c2w
  auto_eval_use_pose_idx: true
  auto_eval_compare_pose_idx: true

SLAM:
  motion_thresh: 0.0
  pnp_first: true
  log_vo_diag: true

DepthModel:
  backend: unidepth

RailScale:
  enabled: true
  apply_correction: true
  correction_mode: metric_width
  metric_width_m: 1.600
  mode: detector
  metric_width_candidates_m: [1.435, 1.500, 1.550, 1.600]
  detector_checkpoint: /autodl-fs/data/GigaSLAM/railbench/rail_detector_runs/segformer_b0/best.pt
  min_scale: 0.50
  max_scale: 2.20
  max_scale_step: 0.20
  max_width_jump_ratio: 0.12
  log_row_diag: true
```

云服务器路径写在 YAML 中，属于实验运行环境，不要因为本地路径不同而改掉。

## 评估逻辑

`scripts/eval_ate.py` 当前支持：

- `--est_convention c2w`：当前 GigaSLAM `poses_est.txt` 的正式口径，也是默认值
- `--est_convention w2c`：仅用于兼容历史上保存为 W2C 的旧轨迹文件
- 默认读取估计文件同目录下的 `poses_idx.txt`
- `--ignore_pose_idx` 可关闭按帧号对齐

正式评估必须使用：

```bash
python scripts/eval_ate.py \
  --est results/.../poses_est.txt \
  --gt /root/GigaSLAM/railway_data/gt_poses/npy/scene_16_train.npy \
  --save_dir results/.../plot \
  --est_convention c2w
```

不要再用顺序对齐作为正式指标。顺序对齐只可作为排查历史差异的参考。

## 前端与后端关键帧

当前前端逻辑已经解耦：

- 每一帧都进入 VO。
- 每一帧都保存到 `self.cameras`，用于输出 `poses_est.txt`。
- `poses_idx.txt` 记录每个估计位姿对应的原始帧号。
- 后端只接收初始化帧、未初始化阶段帧，以及 `is_keyframe()` 选中的关键帧。

因此：

- `motion_thresh: 0.0` 表示不跳过 VO 帧。
- 它不等于所有帧都成为后端关键帧。
- ATE 应以 `poses_idx.txt` 对齐后的 `original` 为准。

## 铁路几何方向

项目下一阶段不应继续盲调纯 VO 参数。合理方向是：使用铁路几何为单目系统提供尺度约束。

重要限制：

- 不应硬编码轨距 `1.435m`。
- RAIL-BENCH 的 rail annotation 是图像中的钢轨线标注，不保证直接等价于标准轨距边界。
- 目前数据可能来自不同线路和标注定义，固定物理轨距会带来错误先验。

更稳妥的思路：

1. 用 RAIL-BENCH object_and_rail 标注训练 rail detector。
2. 在目标序列中稳定检测左右轨。
3. 用轨道宽度的时间一致性约束深度尺度，而不是使用固定轨距。
4. 将尺度约束以开关形式接入，效果不好可以立即回退。

## RAIL-BENCH Rail 数据

用户已经在云端准备了 RAIL-BENCH `object_and_rail` 数据：

```text
railbench/object_and_rail/
  annotations/
    rails/
      annotations_train.json
      annotations_val.json
    objects/
      annotations_train.json
      annotations_val.json
  images/
    train/
    val/
    test/
  data_overview.csv
```

标注格式：

- 顶层字段：`images`、`categories`、`annotations`
- rail 类别：`id=1, name=rail`
- ignore 类别：`id=2, name=ignore_area`
- rail annotation 中包含 `polyline`
- `rightRail: 1` 表示右轨，`rightRail: 0` 表示左轨

数据量：

- train: 1500 images
- val: 500 images
- test: 500 images

## 准备 Rail Detector 数据

脚本：

```text
scripts/prepare_railbench_rail.py
```

作用：

- 读取 RAIL-BENCH rail polyline 标注。
- 将原图缩放到统一宽度。
- 生成左右轨两个 heatmap。
- 生成 ignore mask。
- 写出 `manifest.csv`。

云端示例命令：

```bash
python scripts/prepare_railbench_rail.py \
  --root /root/GigaSLAM/railbench/object_and_rail \
  --out /autodl-fs/data/GigaSLAM/railbench/object_and_rail/prepared_rail \
  --width 960 \
  --sigma 3
```

已验证输出：

```text
[prepare] train: images=1500 missing=0 left=7504 right=7848 ignore=0
[prepare] val: images=500 missing=0 left=2478 right=2609 ignore=0
```

输出结构：

```text
prepared_rail/
  train/
    images/
    left_heatmap/
    right_heatmap/
    ignore_mask/
    manifest.csv
  val/
    images/
    left_heatmap/
    right_heatmap/
    ignore_mask/
    manifest.csv
```

## 训练 Rail Detector

相关文件：

```text
utils/rail_detector_model.py
scripts/train_rail_detector.py
```

模型选择：

- 使用 SegFormer-B0 作为第一版 rail detector。
- 预训练权重默认：`nvidia/segformer-b0-finetuned-ade-512-512`。
- 输出 2 个通道：left rail heatmap、right rail heatmap。
- 损失：BCEWithLogits + soft Dice。
- 指标：left/right/mean threshold IoU。

云端训练命令：

```bash
python scripts/train_rail_detector.py \
  --data /autodl-fs/data/GigaSLAM/railbench/object_and_rail/prepared_rail \
  --out /autodl-fs/data/GigaSLAM/railbench/rail_detector_runs/segformer_b0 \
  --epochs 30 \
  --batch-size 4 \
  --image-size 768 432 \
  --lr 1e-4 \
  --device cuda \
  --amp
```

训练输出：

```text
rail_detector_runs/segformer_b0/
  best.pt
  last.pt
  metrics.json
  val_preview/
```

等待训练完成后，先看：

```bash
cat /autodl-fs/data/GigaSLAM/railbench/rail_detector_runs/segformer_b0/metrics.json
find /autodl-fs/data/GigaSLAM/railbench/rail_detector_runs/segformer_b0/val_preview -maxdepth 1 -type f | head
```

如果左右轨 heatmap 稳定，再接入 GigaSLAM 的尺度约束。不要在 detector 尚未可靠前继续调 RailScale 参数。

## Rail Detector 离线诊断

训练完成后，先不要直接接进 SLAM。先用 detector 跑目标序列，确认它在 `scene_16_train` 上是否稳定。

脚本：

```text
scripts/infer_rail_detector.py
```

云端示例命令：

```bash
python scripts/infer_rail_detector.py \
  --checkpoint /autodl-fs/data/GigaSLAM/railbench/rail_detector_runs/segformer_b0/best.pt \
  --config configs/railway/rowtrack350/scenes/scene_16_train.yaml \
  --out /autodl-fs/data/GigaSLAM/railbench/rail_detector_runs/segformer_b0/scene_16_diag \
  --device cuda \
  --save-overlays \
  --overlay-every 10
```

输出：

```text
scene_16_diag/
  rail_detector_diag.csv
  overlays/
```

先检查：

```bash
head -5 /autodl-fs/data/GigaSLAM/railbench/rail_detector_runs/segformer_b0/scene_16_diag/rail_detector_diag.csv
find /autodl-fs/data/GigaSLAM/railbench/rail_detector_runs/segformer_b0/scene_16_diag/overlays -maxdepth 1 -type f | head
```

判断标准：

- `usable` 帧比例要高。
- `width_med_px` 应随时间平滑变化，不应频繁跳到另一组轨道。
- overlay 中左右轨应落在目标行驶轨道上，而不是远处岔线或邻线。
- 只有通过这个诊断后，才进入 SLAM 内部的 RailScale 接入。

当前 `scene_16_train` 离线诊断结论：

- `rail_detector_diag.csv` 共 179 帧。
- `usable`: 179 / 179，比例 `100%`。
- `width_med_px` 中位数约 `127.5 px`。
- `width_med_px` 5% 到 95% 分位约 `106.0 px` 到 `136.0 px`。
- 相邻帧宽度变化中位数约 `0.5 px`，95% 分位约 `8.0 px`。
- 最大跳变出现在 `147 -> 148`，从 `136.0 px` 跳到 `106.0 px`。
- `157 -> 160` 附近 `valid_rows=2`，宽度约 `99-100 px`，应视为异常或低可信区间。

结论：SegFormer rail detector 可作为下一阶段输入，但不能直接把所有帧的 rail pair 当作尺度约束。接入时应采用近端轨道宽度、时间一致性滤波、异常跳变剔除，并在低可信帧保持 scale=1 或沿用前一稳定 scale。

## 下一步

当前 **RailScale metric rail-head prior correction** 已经完成物理宽度对照。

已验证结果：

```text
shadow baseline: ATE RMSE ~= 22.88m
metric_width_m=1.550 clean: ATE RMSE = 4.3752m
metric_width_m=1.600 clean: ATE RMSE = 2.8259m
scene_19 metric_width_m=1.600: ATE RMSE = 3.0667m
```

结论：`1.600m` 明显优于 `1.550m`，而且干净配置复跑结果与早先
`1.600m` 结果 `2.8285m` 基本一致，说明改善不是配置乱码或线程数差异造成的。

当前推荐固定：

```yaml
RailScale:
  apply_correction: true
  correction_mode: metric_width
  metric_width_m: 1.600
```

当前最可信的 `1.600m` 干净配置结果目录：

```text
results/GigaSLAM_railway_data_scene_16_train/2026-05-24-17-27-50-No-LC/
```

关键结果：

- `plot/stats_original.json`: RMSE `2.8259m`。
- `trj_original.json`: `use_pose_idx=true`，`trj_id=0..178`。
- `rail_scale_diag.csv`: 179/179 帧 `status=corrected`。
- `rail_scale_rows.csv`: 895/895 row selected，reject 为 0。
- `applied_scale` 中位约 `1.644`，5%-95% 约 `1.545` 到 `1.735`。

scene_19 跨 scene 验证也支持 `1.600m`。下一步不应继续在 `1.55~1.60` 内做密集搜索，
更合理的是扩展到更多 scene，并重点检查失败是否集中在序列末段或特殊轨道几何。

不推荐：

- 继续搜索 `pnp_first`、`motion_thresh`、depth xval 阈值等纯 VO 参数。
- 用 GT/GPS/IMU 参与后处理。
- 用轨迹后处理掩盖建图和位姿估计问题。
- 把 `1.435m` 标准轨距当作默认强先验；当前 detector 更像 rail-head outer edges。

### 已完成：Metric-width shadow diagnostic

`2026-05-24-13-18-36-No-LC` 完成了 `apply_correction=false` 的 shadow 诊断。结论：

- 正式 ATE RMSE 约 `22.88m`，符合 shadow 不改变轨迹的预期。
- `rail_scale_diag.csv` 共 179 帧，`rail_scale_rows.csv` 共 895 行，每帧 5 个 row。
- detector 信号稳定：row 全部 selected，`pixel_width` 中位约 `640px`，5%-95% 约 `631px` 到 `644px`。
- gauge-free 相对 scale 只能带来有限离线改善，主要问题仍是全局物理尺度。
- 离线 sanity check 显示 `1.55m ~ 1.60m` 的 rail-head prior 比 `1.435m` 更符合当前检测边界。

### 已完成：Metric rail-head prior correction

`2026-05-24-13-56-10-No-LC` 使用 `metric_width_m: 1.600` 和
`apply_correction=true` 完成正式 correction 实验。结论：

- 自动 ATE original RMSE `2.8285m`。
- 手动 `scripts/eval_ate.py` pose_idx 复核 RMSE `2.8285m`。
- `status=corrected` 覆盖 179/179 帧。
- `applied_scale` 全程非 1，中位约 `1.644`。
- `2026-05-24-17-27-50-No-LC` 使用干净配置复跑 `metric_width_m: 1.600`，ATE RMSE `2.8259m`。
- `2026-05-24-17-09-46-No-LC` 使用干净配置 `metric_width_m: 1.550`，ATE RMSE `4.3752m`。
- scene_19 跨 scene 验证 `2026-05-24-18-39-58-No-LC` 使用 `metric_width_m: 1.600`，ATE RMSE `3.0667m`。
  RailScale 289/289 帧 corrected，1445/1445 row selected。误差主要集中在末段 `251..288`。

注意：这次运行前 YAML 的 `Results` 段有乱码注释导致 `auto_eval_use_pose_idx` 未被解析，
但本序列没有跳帧，`poses_idx.txt` 为连续 `0..178`，且手动 pose_idx 复核数值一致。
当前 YAML 已修复，后续运行会正确保存 `auto_eval_use_pose_idx: true`。

由于 RAIL-BENCH rail annotation 标的是图像中的 rail-head outer edges，`1.435m` 不能被当作默认真值。下一轮正式实验应验证 `1.550m` 与 `1.600m` 的物理假设差异，而不是大范围搜索。

### 已完成：全部 train scene 跨 scene 验证

`metric_width_m: 1.600` 已在当前 6 个 train scene 上完成验证。最新正式结果：

| scene | frames | ATE RMSE | median | max | RailScale 状态 | 备注 |
|---|---:|---:|---:|---:|---|---|
| scene_11_train | 288 | 1.6295m | 1.4988m | 3.0765m | corrected 266/288 | 11 帧 width_jump，11 帧 too_few_samples，但轨迹结果最好 |
| scene_13_train | 149 | 22.5911m | 21.0419m | 41.0538m | corrected 54/149 | 大量 width_jump，需优先诊断 |
| scene_14_train | 298 | 5.7540m | 4.7914m | 15.2434m | corrected 298/298 | scale 中位约 1.86，末段/几何需检查 |
| scene_16_train | 179 | 2.8259m | 2.7142m | 5.3467m | corrected 179/179 | 干净配置基准 |
| scene_17_train | 235 | 3.8101m | 3.4429m | 6.2336m | corrected 234/235 | 首次运行 CUDA illegal memory access，重跑成功 |
| scene_19_train | 289 | 3.0667m | 2.2601m | 7.8201m | corrected 289/289 | 误差主要集中末段 251..288 |

结果目录可在 `results/GigaSLAM_railway_data_<scene>/latest-No-LC` 对应时间戳下查看。
结论：`1.600m` 在 5/6 个 scene 上明显可用；`scene_13_train` 是当前最重要的失败案例，
应先诊断 rail pair 跳变、邻线/岔线误检或该序列物理宽度假设是否不同。

### 进行中：scene_13 rail pair 漂移修复

`scene_13_train_metric_width_1600_widthcap020.yaml` 将 `detector_max_width_px_frac` 从 `0.45` 收紧到 `0.20`，
ATE RMSE 从 `22.5911m` 降到 `6.7931m`。它解决了早期超宽 rail pair 误检：
`pixel_width` p95 从约 `1559px` 降到 `714px`，`width_jump` 从 95 帧降到 7 帧。

剩余问题是道岔/弯道区域的 rail pair 拓扑歧义：约 `96..117` 帧开始，近处行仍能落在主轨，
但远处行会被右侧岔线/邻轨吸走。例如 `f108` 底部行中心约 `1896.5px`，其余远处行可到
`3061..3247px`；`f117` 底部两行约 `2100..2125px`，远处行可到 `3530..3769px`。
因此不应把整帧中心慢漂移简单当成异常，也不应优先使用固定初始中心锚点。

当前主线 scene 配置：
`configs/railway/rowtrack350/scenes/scene_13_train.yaml`。
历史运行用的 generated 配置已归档到
历史 generated 配置已清理；当前复现请使用 `configs/railway/rowtrack350/scenes/scene_13_train.yaml`。
该配置保留 `detector_max_width_px_frac: 0.20`，新增 near-field row-consistency：从近处行向远处行跟踪同一 rail pair，
后续行相对近处参考中心偏移超过 `350px` 时拒绝为 `row_center_inconsistent`，样本不足时沿用上一帧有效 scale。
这个门限来自已跑正常 scene 的行内中心跨度统计：正常最大约 `178.5px`，scene13 错误道岔帧可超过 `1800px`。

运行结果 `2026-05-24-21-29-22-No-LC`：ATE RMSE `3.2071m`，mean `2.9006m`，median `2.6096m`，max `5.6023m`。
对比 `widthcap020` 的 `6.7931m` 明显改善，尤其 `118..148` 段 RMSE 从 `11.45m` 降到 `3.51m`。
RailScale 状态为 `corrected 123/149`、`width_jump_hold 26/149`；`row_center_inconsistent` 只在行级诊断中出现 3 次，
主要作用不是大量 hold，而是 row target tracking 让 `f108/f117` 等道岔帧重新选择回主轨 rail pair。
剩余最大误差位于 `f79` 附近，约 `5.60m`。下一步应做跨 scene 回归验证，而不是继续只在 scene13 上收紧门限。


### 配置目录说明（2026-05-26）

- `configs/railway/rowtrack350/template.yaml`：当前 color-refinement-0 轨迹主线配置，默认输出到 `results/color_refinement_0/`。
- `configs/railway/rowtrack350/scenes/*.yaml`：每个 train scene 的薄配置，只覆盖 Dataset 路径。
- `configs/railway/rowtrack350/full_mainline/*.yaml`：color-refinement-400 完整建图/渲染配置，默认输出到 `results/color_refinement_400/`。
- `configs/railway/rowtrack350/ablations/no_railscale/*.yaml`：RailScale 消融配置，默认输出到 `results/color_refinement_0/RailScale_ablation/`。
- `configs/railway/rowtrack350/README.md`：`template.yaml` 完整中文参数表和调整建议。

### 交接：rowtrack350 跨 scene 回归配置（2026-05-26）

目标：验证 scene13 成功的 `widthcap020 + rowtrack350 + hold` 方案是否会让其他 train scene 退化。
不要继续优先调 `centeranchor250_hold`；它是早期诊断配置，已归档，不是当前主线。

已生成稳定配置：

- 基准配置：`configs/railway/rowtrack350/template.yaml`
- scene 配置：
  - `configs/railway/rowtrack350/scenes/scene_11_train.yaml`
  - `configs/railway/rowtrack350/scenes/scene_13_train.yaml`
  - `configs/railway/rowtrack350/scenes/scene_14_train.yaml`
  - `configs/railway/rowtrack350/scenes/scene_16_train.yaml`
  - `configs/railway/rowtrack350/scenes/scene_17_train.yaml`
  - `configs/railway/rowtrack350/scenes/scene_19_train.yaml`

基准配置关键 RailScale 参数速览，完整解释见 `configs/railway/rowtrack350/README.md`：

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `metric_width_m` | `1.600` | 使用 rail-head 物理宽度先验修正单目尺度。 |
| `detector_max_width_px_frac` | `0.20` | widthcap020，防止跨轨/邻轨超宽误配。 |
| `detector_row_consistency` | `true` | 启用行内一致性，拦截同帧不同采样行跳到不同轨道分支。 |
| `detector_row_target_tracking` | `true` | 从近处行向远处行跟踪同一 rail pair，是 scene13 道岔改善核心。 |
| `detector_max_row_center_shift_px` | `350.0` | 后续行相对近处参考中心最大允许偏移。 |
| `hold_last_scale_on_reject` | `true` | 检测异常时沿用上一帧稳定 scale，避免错误检测污染深度。 |

已完成 dry-run 校验，5 个待回归 scene 的图像数和 GT pose 数匹配：

| scene | frames |
|---|---:|
| scene_11_train | 288 |
| scene_14_train | 298 |
| scene_16_train | 179 |
| scene_17_train | 235 |
| scene_19_train | 289 |

建议同事正式运行命令：

```bash
cd /root/GigaSLAM
/root/miniconda3/envs/gigaslam/bin/python scripts/run_metric_width_scenes.py \
  --base-config configs/railway/rowtrack350/template.yaml \
  --scenes scene_11_train scene_14_train scene_16_train scene_17_train scene_19_train \
  --generated-config-dir logs/generated_configs/rowtrack350_regression \
  --continue-on-error
```

运行完成后，记录每个 scene 的：

- 日志：`logs/scene*_metric_width_1600_<timestamp>.log`
- 结果目录：`results/GigaSLAM_railway_data_<scene>/<timestamp>-No-LC`
- 指标：`plot/stats_original.json`
- RailScale 诊断：`rail_scale_diag.csv`、`rail_scale_rows.csv`

对比旧基线：

| scene | 旧基线 ATE RMSE | 备注 |
|---|---:|---|
| scene_11_train | 1.6295m | 原先最好，重点防退化 |
| scene_14_train | 5.7540m | 原先偏高，可能有改善空间 |
| scene_16_train | 2.8259m | 干净配置基准 |
| scene_17_train | 3.8101m | 首次曾 CUDA error，重跑成功 |
| scene_19_train | 3.0667m | 末段误差较集中 |
| scene_13_train | 3.2071m | rowtrack350 已完成，作为当前成功案例 |

判定原则：如果 5 个待回归 scene 无明显退化，`rowtrack350 + widthcap020 + hold` 可以作为下一版主线 RailScale 配置；
如果某个 scene 退化超过约 `0.5..1.0m`，优先检查该 scene 的 `rail_scale_rows.csv` 中 `row_center_inconsistent`、`width_jump_hold` 分布，不要先盲目改 metric width。


### 回归结果：rowtrack350 批次（2026-05-26）

用户执行了旧 generated 基准配置完成跨 scene 回归批次，日志时间戳为 `20260526_102107`。
该旧 generated 基准配置已不再保留；当前主线配置以 `configs/railway/rowtrack350/template.yaml` 为准。
注意：结果目录是 `2026-05-26`，不是 2024 年。

| scene | 最新结果目录 | 最新 ATE RMSE | 旧基线 ATE RMSE | 变化 | 状态 |
|---|---|---:|---:|---:|---|
| scene_11_train | `2026-05-26-10-21-31-No-LC` | 1.8992m | 1.6295m | +0.2697m | 完成，小幅退化 |
| scene_14_train | `2026-05-26-10-42-07-No-LC` | 5.7137m | 5.7540m | -0.0403m | 完成，基本持平略好 |
| scene_16_train | `2026-05-26-11-04-50-No-LC` | 2.8000m | 2.8259m | -0.0259m | 完成，基本持平略好 |
| scene_17_train | `2026-05-26-11-19-38-No-LC` | 3.7392m | 3.8101m | -0.0709m | 完成，略好 |
| scene_19_train | `2026-05-26-12-16-56-No-LC` | 3.0652m | 3.0667m | -0.0015m | 重跑完成，持平略好 |

RailScale 状态摘要：

| scene | RailScale 状态 |
|---|---|
| scene_11_train | corrected 266/288, width_jump_hold 11, too_few_samples_hold 11 |
| scene_14_train | corrected 298/298 |
| scene_16_train | corrected 179/179 |
| scene_17_train | corrected 230/235, width_jump_hold 5 |
| scene_19_train | corrected 289/289 |

scene19 首次运行日志 `logs/scene19_metric_width_1600_20260526_102107.log` 在后端初始化时报
`RuntimeError: CUDA error: an illegal memory access was encountered`，只留下部分输出。
已按原参数单独重跑成功：`logs/scene19_metric_width_1600_20260526_121634.log`，
结果目录 `results/GigaSLAM_railway_data_scene_19_train/2026-05-26-12-16-56-No-LC`。

当前判断：rowtrack350 对 14/16/17/19 没有回归风险，scene13 明显改善；scene11 小幅退化 `+0.27m`。
如果要严格保护所有 scene，可考虑按 scene 分支配置；如果允许 scene11 轻微回退，rowtrack350 可以作为下一版主线 RailScale 配置候选。

## 常用命令

远端新会话启动检查：

```bash
pwd
hostname
which python
which codex
python - <<'PY'
from utils.config_utils import load_config
cfg = load_config('configs/railway/rowtrack350/scenes/scene_16_train.yaml')
print('config ok')
print('RailScale.enabled=', cfg['RailScale']['enabled'])
print('RailScale.apply_correction=', cfg['RailScale']['apply_correction'])
print('metric_width_candidates_m=', cfg['RailScale'].get('metric_width_candidates_m'))
PY
```

检查当前配置：

```bash
python - <<'PY'
from utils.config_utils import load_config
cfg = load_config('configs/railway/rowtrack350/scenes/scene_16_train.yaml')
print('auto_eval_ate=', cfg['Results']['auto_eval_ate'])
print('auto_eval_use_pose_idx=', cfg['Results']['auto_eval_use_pose_idx'])
print('est_convention=', cfg['Results']['auto_eval_est_convention'])
print('motion_thresh=', cfg['SLAM']['motion_thresh'])
print('pnp_first=', cfg['SLAM']['pnp_first'])
print('RailScale.enabled=', cfg['RailScale']['enabled'])
PY
```

运行主实验：

```bash
python slam.py --config configs/railway/rowtrack350/scenes/scene_16_train.yaml
```

手动正式 ATE：

```bash
python scripts/eval_ate.py \
  --est results/GigaSLAM_railway_data_scene_16_train/<timestamp>-No-LC/poses_est.txt \
  --gt /root/GigaSLAM/railway_data/gt_poses/npy/scene_16_train.npy \
  --save_dir results/GigaSLAM_railway_data_scene_16_train/<timestamp>-No-LC/plot \
  --label original \
  --est_convention c2w
```

语法检查：

```bash
python -m py_compile \
  slam.py \
  scripts/eval_ate.py \
  scripts/prepare_railbench_rail.py \
  scripts/train_rail_detector.py \
  scripts/infer_rail_detector.py \
  utils/rail_detector_model.py \
  utils/rail_scale.py \
  utils/visual_odometry.py
```

## 文件索引

核心运行：

- `slam.py`
- `configs/railway/rowtrack350/scenes/scene_16_train.yaml`
- `utils/slam_frontend.py`
- `utils/visual_odometry.py`
- `utils/eval_utils.py`
- `scripts/eval_ate.py`

轨迹后处理：

- `scripts/postprocess_trajectory.py`

RailScale 旧启发式路线：

- `utils/rail_scale.py`

Rail detector 新路线：

- `scripts/prepare_railbench_rail.py`
- `scripts/train_rail_detector.py`
- `scripts/infer_rail_detector.py`
- `utils/rail_detector_model.py`

## 交接原则

后续实验记录应围绕“假设、代码开关、输入输出、正式指标、失败原因”写，不再堆砌参数表。

当前最重要的判断是：这个项目的核心瓶颈不是轨迹画图或 ATE 脚本，而是单目铁路场景中的尺度约束不足。下一步应围绕可靠 rail detector 和 gauge-free 几何尺度约束展开。
