# rowtrack350 railway configs

这是当前铁路 train scene 的 rowtrack350 主线配置说明。本文只解释 `template.yaml` 中显式写出的参数；`configs/base_config.yaml` 继承来的默认参数不在这里展开。

## 配置文件结构

- `template.yaml`：主线模板，包含共享的 SLAM、深度模型、相机标定、RailScale 和渲染层级参数。
- `scenes/*.yaml`：每个 scene 的薄配置，只覆盖 `Dataset.color_path` 和 `Dataset.pose_path`。
- `experiments/*.yaml`：受控实验配置，只覆盖少量实验参数。
- `configs/generated_metric_width/`：脚本生成配置的输出目录，不是维护中的主线配置。
- `configs/archive/generated_metric_width_legacy/`：历史实验、dry-run、retry 和旧 generated 配置归档。

推荐批跑入口：

```bash
cd /root/GigaSLAM
/root/miniconda3/envs/gigaslam/bin/python scripts/run_metric_width_scenes.py \
  --base-config configs/railway/rowtrack350/template.yaml \
  --scenes scene_11_train scene_13_train scene_14_train scene_16_train scene_17_train scene_19_train \
  --generated-config-dir configs/generated_metric_width/rowtrack350_regression \
  --continue-on-error
```

## 核心思路

rowtrack350 的目标是让 RailScale 在道岔、弯道、多轨场景中更稳：先用 rail detector 找左右轨，再用 `metric_width_m: 1.6` 的 rail-pair 有效宽度先验估计深度尺度；同时用 `detector_max_width_px_frac: 0.2` 防止跨轨超宽误配，并用 near-field row tracking 从近处行向远处行约束同一 rail pair。


## Baseline 与回归门禁

当前主线 baseline 固定在：

```text
configs/railway/rowtrack350/baselines/rowtrack350_mainline_20260526.csv
```

它记录 6 个 train scene 的 baseline run 和形态指标。未来任何实验都应使用：

```bash
cd /root/GigaSLAM
/root/miniconda3/envs/gigaslam/bin/python scripts/compare_trajectory_regression.py \
  --baseline-csv configs/railway/rowtrack350/baselines/rowtrack350_mainline_20260526.csv \
  --results-dir results
```

如果实验只改善单个 scene、没有覆盖全部 6 个 baseline scene，或使 scene13 道岔/弯道回归样本退化，则不能作为主线改进。

## 参数说明表

### 顶层

| 参数 | 当前值 | 含义 | 为什么这样设 | 调整建议 |
|---|---:|---|---|---|
| `inherit_from` | `configs/base_config.yaml` | 继承项目基础配置。 | 只在模板里覆盖铁路实验需要的参数，避免复制全量默认配置。 | 不建议改，除非基础配置文件移动。 |

### Results

| 参数 | 当前值 | 含义 | 为什么这样设 | 调整建议 |
|---|---:|---|---|---|
| `Results.use_gui` | `false` | 关闭交互 GUI。 | 批量实验更稳定，日志和结果文件足够分析。 | 调试可视化时再打开。 |
| `Results.eval_rendering` | `false` | 不跑渲染质量评估。 | 当前实验关注轨迹 ATE 和尺度，不让渲染评估拖慢流程。 | 做最终渲染质量对比时再打开。 |
| `Results.rendered_depth_diag` | `{enabled: false, keyframes_only: true, save_rgb: true, eval_rgb_metrics: true}` | 可选保存 Gaussian rendered depth 与输入 depth 的关键帧级残差诊断；`save_rgb/eval_rgb_metrics` 控制是否额外保存 RGB render 和 PSNR/SSIM/LPIPS。 | 默认关闭诊断且保留普通渲染评估行为；depthdiag 测试配置会把 RGB 保存和指标关闭以节省时间/空间。 | 做 depth-Gaussian consistency 测试时使用 `configs/railway/rowtrack350/experiments/depthdiag.yaml`。 |
| `Results.auto_eval_ate` | `true` | SLAM 结束后自动计算 ATE。 | 每次运行直接得到 `plot/stats_original.json`、`plot/shape_original.json` 和 `plot/evo_2dplot_original.png`。 | 建议保持开启。 |
| `Results.auto_postprocess_trajectory` | `false` | 不做轨迹后处理。 | 避免后处理掩盖真实尺度漂移。 | 研究后处理本身时再打开。 |
| `Results.auto_eval_monocular` | `false` | ATE 评估不做 Sim(3) 尺度对齐。 | 保留真实 metric scale 误差，验证 RailScale 是否真的修正尺度。 | 不建议改；改成 `true` 会让尺度问题被对齐隐藏。 |
| `Results.auto_eval_est_convention` | `c2w` | 指定保存轨迹的位姿约定。 | `poses_est.txt` 实际保存 flattened C2W，GT `poses_gt.npy` 是 W2C，评估时只对 GT 取逆。 | 建议保持 `c2w`；只有评估旧 W2C 轨迹文件时才改成 `w2c`。 |
| `Results.auto_eval_use_pose_idx` | `true` | ATE 按 `poses_idx.txt` 对齐 GT 帧号。 | 防止跳帧或关键帧索引导致 GT 错配。 | 建议保持开启。 |
| `Results.auto_eval_compare_pose_idx` | `true` | 额外输出 legacy ordered 对比。 | 方便和历史结果排查差异。 | 稳定后可关闭，但目前保留利于回溯。 |

### Training

| 参数 | 当前值 | 含义 | 为什么这样设 | 调整建议 |
|---|---:|---|---|---|
| `Training.mapping_itr_num` | `3` | 每次后端 mapping 的迭代数。 | 控制运行时间，当前轨迹实验不追求高渲染质量。 | 需要更好地图质量时可提高。 |
| `Training.init_itr_num` | `1050` | 初始化地图的迭代数。 | 沿用当前 GigaSLAM 稳定初始化设置。 | 初始化不稳时再调。 |
| `Training.tracking_itr_num` | `300` | 前端 tracking 优化迭代数。 | 保持铁路场景 tracking 稳定。 | 若运行太慢或 tracking 震荡，再做对照实验。 |

### Dataset.Calibration

| 参数 | 当前值 | 含义 | 为什么这样设 | 调整建议 |
|---|---:|---|---|---|
| `Dataset.Calibration.fx` | `7267.95450880415` | 相机 x 方向焦距。 | 来自 12MP middle camera 标定。 | 只在更换相机/数据集时改。 |
| `Dataset.Calibration.fy` | `7267.95450880415` | 相机 y 方向焦距。 | 与当前相机标定一致。 | 只在更换相机/数据集时改。 |
| `Dataset.Calibration.cx` | `2056.049238502414` | 主点 x 坐标。 | 对应 `width: 4112` 的相机中心附近。 | 只在标定更新时改。 |
| `Dataset.Calibration.cy` | `1232.862908875167` | 主点 y 坐标。 | 对应 `height: 2504` 的相机主点。 | 只在标定更新时改。 |
| `Dataset.Calibration.k1` | `-0.0790171` | 径向畸变参数。 | 保存原始标定参数，虽然当前 `distorted: false`。 | 若启用畸变处理，需重新核对。 |
| `Dataset.Calibration.k2` | `-0.17716` | 径向畸变参数。 | 同上。 | 同上。 |
| `Dataset.Calibration.p1` | `0.0` | 切向畸变参数。 | 当前标定为 0。 | 同上。 |
| `Dataset.Calibration.p2` | `0.0` | 切向畸变参数。 | 当前标定为 0。 | 同上。 |
| `Dataset.Calibration.k3` | `1.7629` | 高阶径向畸变参数。 | 保存原始标定参数。 | 同上。 |
| `Dataset.Calibration.width` | `4112` | 原始图像宽度。 | RailScale 像素比例阈值依赖此尺寸。 | 更换分辨率必须同步改。 |
| `Dataset.Calibration.height` | `2504` | 原始图像高度。 | ROI 和采样行按高度比例计算。 | 更换分辨率必须同步改。 |
| `Dataset.Calibration.distorted` | `false` | 输入是否仍带畸变。 | 当前流程按已去畸变图像处理。 | 若输入改为原始畸变图，需要整体确认前处理。 |

### SLAM

| 参数 | 当前值 | 含义 | 为什么这样设 | 调整建议 |
|---|---:|---|---|---|
| `SLAM.viz` | `true` | 保留可视化相关输出。 | 当前运行会保存 keyframe/match 可视化，便于排查。 | 只追求速度时可关闭。 |
| `SLAM.loop_closure` | `false` | 关闭回环。 | 当前 railway train scene 主要验证前端尺度和轨迹，不引入回环变量。 | 做长序列闭环实验时再开启。 |
| `SLAM.motion_thresh` | `0.0` | VO 帧跳过阈值。 | 0 表示不因运动小而跳帧，保证 `poses_idx` 连续可评估。 | 不建议改，除非专门研究跳帧策略。 |
| `SLAM.loop_detect_thresh` | `0.034` | 回环检测阈值。 | 回环关闭时基本不参与主实验。 | 开回环时再调。 |
| `SLAM.2d2d_thread` | `20` | 2D-2D 匹配相关线程数。 | 沿用当前高分辨率运行设置。 | 受 CPU 资源限制时可降低。 |
| `SLAM.n_max` | `5000` | 后端/地图中使用的上限规模参数。 | 控制地图规模和显存压力。 | 如显存不足可降低。 |
| `SLAM.disk_max_size` | `2048` | DISK 特征提取前的最大图像边长。 | 12MP 图像太大，需降采样控制显存。 | 特征不足可增大，OOM 则降低。 |
| `SLAM.num_features_disk` | `8192` | DISK 特征数量。 | 补偿 12MP 降采样后的铁路纹理稀疏。 | 匹配少可增加，速度慢可降低。 |
| `SLAM.scale_smooth_alpha` | `0.7` | VO 内部尺度平滑系数。 | 减少帧间尺度抖动。 | 若尺度响应太慢可提高或降低做对照。 |
| `SLAM.pnp_first` | `true` | 优先尝试 PnP 路径。 | 当前铁路实验 PnP-first 更稳定。 | 只在 VO 路线对比实验中改。 |
| `SLAM.depth_crop_top` | `0.2` | 深度用于 VO 尺度时裁掉顶部比例。 | 天空/远景深度不可靠，避免污染估计。 | 场景视野变化时再调。 |
| `SLAM.scale_crop_bottom` | `0.6` | 深度尺度使用区域的底部比例。 | 保留中下区域，减少近处极端点影响。 | 需要更多近处点时再调。 |
| `SLAM.depth_xval_thresh` | `0.3` | 深度交叉验证的相对阈值。 | 过滤深度不一致的匹配。 | 过严会丢匹配，过松会引入坏点。 |
| `SLAM.3d2d_thread` | `40` | 3D-2D 匹配/求解相关线程数。 | 高分辨率实验下提高并行度。 | CPU 紧张时可降低。 |
| `SLAM.log_vo_diag` | `true` | 保存 VO 诊断 CSV。 | 方便分析尺度和匹配失败原因。 | 正式大量跑可按需关闭。 |

### DepthModel

| 参数 | 当前值 | 含义 | 为什么这样设 | 调整建议 |
|---|---:|---|---|---|
| `DepthModel.backend` | `unidepth` | 使用的深度模型后端。 | 当前实验以 UniDepth 作为深度来源。 | 切到 railway 深度网时改为 `railway` 并验证。 |
| `DepthModel.railway_ckpt_dir` | `/root/GigaSLAM/railway_depth_net/ckpt` | railway 深度模型 checkpoint 目录。 | 备用后端路径，当前 `unidepth` 不作为主线使用。 | 只有使用 railway backend 时需要维护。 |
| `DepthModel.railway_model_name` | `/root/GigaSLAM/railway_depth_net/Depth-Anything-V2-Small-hf` | railway 深度模型名称或本地路径。 | 备用后端配置。 | 同上。 |
| `DepthModel.from_huggingface` | `false` | 是否从 Hugging Face 拉取 UniDepth。 | 离线/内网环境更稳定，使用本地 snapshot。 | 网络稳定且需更新模型时再打开。 |
| `DepthModel.huggingface.model_name` | `lpiccinelli/unidepth-v2-vitl14` | UniDepth Hugging Face 模型名。 | 记录模型来源。 | 只在切模型时改。 |
| `DepthModel.huggingface.commit_hash` | `1d0d3c52f60b5164629d279bb9a7546458e6dcc4` | 固定 Hugging Face 模型版本。 | 避免上游模型更新造成不可复现。 | 更新模型时必须同步记录新 hash。 |
| `DepthModel.local_snapshot_path` | `/root/GigaSLAM/unidepth_model` | 本地 UniDepth 模型路径。 | 避免运行时联网失败。 | 移动模型目录时改。 |

### RailScale

| 参数 | 当前值 | 含义 | 为什么这样设 | 调整建议 |
|---|---:|---|---|---|
| `RailScale.enabled` | `true` | 启用轨道尺度约束。 | 当前提升主要来自 RailScale。 | 只做 ablation 时关闭。 |
| `RailScale.apply_correction` | `true` | 将估计 scale 应用于深度。 | 不只是 shadow 诊断，而是真正修正单目尺度。 | 诊断模式可设为 `false`。 |
| `RailScale.correction_mode` | `metric_width` | 使用显式物理宽度修正尺度。 | 比早期相对宽度标定更可跨 scene。 | 不建议回到 `relative`，除非没有物理宽度先验。 |
| `RailScale.metric_width_m` | `1.6` | 当前 rail-pair 有效宽度先验，单位米。 | 实验验证 `1.600m` 优于 `1.550m`，scene16/13 等表现更好。 | 这是高影响参数，改动必须跑全 scene 对比。 |
| `RailScale.mode` | `detector` | 使用学习式 rail detector 选左右轨。 | Hough 方案不稳定，detector 是当前主线。 | 不建议改为 `hough`。 |
| `RailScale.metric_width_candidates_m` | `[1.435, 1.5, 1.55, 1.6]` | 行级诊断中额外输出候选 metric scale。 | 便于离线比较不同物理宽度假设。 | 可追加候选，但不影响当前修正值。 |
| `RailScale.detector_checkpoint` | `/autodl-fs/data/GigaSLAM/railbench/rail_detector_runs/segformer_b0/best.pt` | rail detector 权重路径。 | 当前最稳定的 SegFormer-B0 rail detector。 | 换 detector 后必须重跑诊断和回归。 |
| `RailScale.detector_device` | `cuda` | detector 推理设备。 | GPU 推理速度可接受。 | 无 GPU 时改 `cpu`，但会变慢。 |
| `RailScale.detector_image_size` | `[768, 432]` | detector 输入尺寸。 | 在速度和 rail heatmap 质量之间折中。 | 小尺寸可能丢细轨，大尺寸更耗显存。 |
| `RailScale.detector_threshold` | `0.3` | detector 概率阈值参数。 | 保留默认诊断阈值。 | 当前主流程主要用 peak 参数，谨慎调。 |
| `RailScale.detector_min_peak` | `0.25` | 单行左右轨 peak 的最低置信度。 | 低于此值容易选噪声。 | 检测漏行多时可略降，误检多时提高。 |
| `RailScale.detector_min_width_px` | `12.0` | rail pair 最小像素宽度。 | 排除几乎重合或无效的左右轨。 | 通常不需要改。 |
| `RailScale.detector_center_x_frac` | `0.5` | 无历史中心时的初始目标中心。 | 初始假设相机沿图像中心附近轨道行驶。 | 偏置相机或特殊安装才改。 |
| `RailScale.detector_peak_topk` | `8` | 每行候选 peak 数量。 | 多轨场景需要保留多个候选再评分。 | 候选不足可增大，误配多可降低。 |
| `RailScale.detector_peak_nms_px` | `24` | peak 非极大值抑制半径。 | 避免同一 rail 附近产生多个重复峰。 | 图像分辨率变化时再调。 |
| `RailScale.detector_max_width_px_frac` | `0.2` | rail pair 最大像素宽度占图宽比例。 | widthcap020，防止跨轨/邻轨超宽误配，是 scene13 改善第一步。 | 过小会拒绝真实近处轨距，过大会放进跨轨误配。 |
| `RailScale.detector_max_center_jump_ratio` | `0.18` | 相邻有效帧中心最大跳变比例。 | 拦截突然跳到另一轨的异常。 | 对慢漂移无效，已由 row tracking 补强。 |
| `RailScale.calibration_frames` | `30` | relative 模式下的参考宽度标定帧数。 | 当前 metric 模式基本不用，但保留兼容。 | metric 主线不需要调。 |
| `RailScale.min_confidence` | `0.65` | 帧级检测置信度最低门槛。 | 低置信检测不应用于尺度修正。 | 漏修正多时可降低，误修正多时提高。 |
| `RailScale.ema_alpha` | `0.2` | scale EMA 平滑系数。 | 避免尺度帧间抖动。 | 响应慢可提高，抖动大可降低。 |
| `RailScale.min_scale` | `0.5` | scale 下限。 | 防止单帧错误检测把深度压得过小。 | 大幅改动会影响尺度修正范围。 |
| `RailScale.max_scale` | `2.2` | scale 上限。 | 允许 UniDepth 在铁路场景中较大尺度偏差被修正。 | 若出现过大 scale 污染，可降低。 |
| `RailScale.max_scale_step` | `0.2` | 单帧 raw scale 最大变化步长。 | 防止尺度突变。 | 变化太慢可增大，尖峰多可降低。 |
| `RailScale.max_width_jump_ratio` | `0.12` | 相邻帧像素宽度允许跳变比例。 | 宽度突变通常意味着选错轨。 | scene11 的部分 hold 与此有关，若保护 scene11 可专项评估。 |
| `RailScale.min_samples` | `3` | 每帧至少需要多少采样行有效。 | 5 个采样行中至少 3 个一致才信任。 | 道岔复杂但近处可靠时可考虑 2，但风险更高。 |
| `RailScale.max_detect_width` | `1280` | Hough 检测最大图像宽度。 | detector 模式基本不依赖，保留兼容。 | detector 主线不需要调。 |
| `RailScale.roi_y_min` | `0.45` | rail detector 采样 ROI 上边界比例。 | 排除天空和远处不稳定区域。 | 需要更多远处轨道时可降低，但道岔风险增加。 |
| `RailScale.roi_y_max` | `0.95` | rail detector 采样 ROI 下边界比例。 | 使用图像下方近处轨道。 | 近处遮挡严重时可降低。 |
| `RailScale.sample_y_fracs` | `[0.58, 0.65, 0.72, 0.8, 0.88]` | 采样行位置比例。 | 覆盖中近距离轨道，并按 near-to-far 顺序用于 row tracking。 | 道岔策略依赖这些行，改动需看 `rail_scale_rows.csv`。 |
| `RailScale.log_diag` | `true` | 保存帧级 RailScale 诊断。 | 生成 `rail_scale_diag.csv`，便于复盘。 | 大规模跑可关闭，但研究阶段建议开。 |
| `RailScale.log_row_diag` | `true` | 保存行级 RailScale 诊断。 | 生成 `rail_scale_rows.csv`，是分析道岔误选的关键。 | 建议保持开启。 |
| `RailScale.save_debug` | `true` | 保存 RailScale debug 图。 | 方便肉眼核对 detector 选轨。 | 存储压力大时可关闭。 |
| `RailScale.debug_every` | `20` | 每隔多少帧保存一次 debug 图。 | 控制文件数量。 | 需要密集排查时调小。 |
| `RailScale.detector_row_consistency` | `true` | 启用行内一致性约束。 | rowtrack350 核心：防止同帧远处行被岔线/邻轨吸走。 | 不建议关闭，除非做 ablation。 |
| `RailScale.detector_row_target_tracking` | `true` | 从近处行向远处行更新目标中心。 | scene13 道岔改善的关键，让采样行跟随同一主轨 rail pair。 | 不建议关闭。 |
| `RailScale.detector_max_row_center_shift_px` | `350.0` | 后续行相对近处参考中心的最大偏移像素。 | 正常 scene 行内中心跨度最大约 178.5px，scene13 错误岔线可超过 1800px，350px 是安全间隔。 | 改动需同时看 scene13 和 scene11 回归。 |
| `RailScale.detector_max_row_center_shift_ratio` | `0.0` | 行中心偏移的比例阈值。 | 当前用固定 350px，比例阈值关闭。 | 分辨率变化时可考虑改用比例。 |
| `RailScale.detector_center_anchor_frames` | `0` | 初始中心锚点所需帧数。 | 早期诊断方向，当前主线关闭。 | 道岔/弯道场景不优先使用固定锚点。 |
| `RailScale.detector_max_anchor_center_shift_px` | `0.0` | 相对初始中心锚点的像素偏移限制。 | 当前关闭，避免真实弯道被误拒绝。 | 只在直轨场景专项实验中考虑。 |
| `RailScale.detector_max_anchor_center_shift_ratio` | `0.0` | 相对初始中心锚点的比例偏移限制。 | 当前关闭。 | 同上。 |
| `RailScale.hold_last_scale_on_reject` | `true` | 检测被拒绝时沿用上一帧有效 scale。 | 避免异常检测污染深度，尤其道岔或短暂漏检。 | 如果长期 hold 导致滞后，需要查看状态分布后再调。 |
| `RailScale.hold_last_scale_statuses` | `[width_jump, center_jump, row_inconsistent, too_few_samples]` | 哪些拒绝状态触发 hold。 | 覆盖宽度突变、中心跳变、行不一致和样本不足四类常见异常。 | 新增状态前要确认不会掩盖真实变化。 |
| `RailScale.ego_path_guidance.enabled` | `false` | 是否启用 TEP-style ego-path shadow 诊断。 | 当前冻结主线不使用 EgoPath，只预留 RailScale-v2 研究入口。 | 训练好 EgoPath 模型并做 shadow 验证后再打开。 |
| `RailScale.ego_path_guidance.model_path` | `/autodl-fs/data/GigaSLAM/railbench/ego_path_runs/resnet18_regression_v1` | EgoPath 模型目录，需包含 `best.pt` 和 `config.yaml`。 | 与新增训练脚本输出保持一致。 | 换模型时改成新的 run 目录。 |
| `RailScale.ego_path_guidance.tep_root` | `/root/GigaSLAM/train-ego-path-detection` | TEP 开源代码目录。 | 复用其 regression model 和 postprocess。 | 移动仓库时同步改。 |
| `RailScale.ego_path_guidance.crop` | `none` | EgoPath 推理裁剪方式。 | 第一阶段保持全图 shadow 诊断，避免引入额外裁剪状态。 | 若模型验证需要，可测试 `auto` 或固定 crop。 |
| `RailScale.ego_path_guidance.log_diag` | `true` | 保存 `ego_path_diag.csv`。 | 用于比较 EgoPath 预测和 RailScale 当前选轨差异。 | 研究阶段建议保持开启。 |
| `RailScale.ego_path_guidance.save_debug` | `false` | 保存 EgoPath overlay 图。 | 默认不增加磁盘压力。 | 抽查模型预测时打开。 |
| `RailScale.ego_path_guidance.device` | `cuda` | EgoPath 模型推理设备。 | shadow 诊断和 SLAM 同环境使用 GPU。 | 无 GPU 或只做小样本检查时可改 `cpu`。 |
| `RailScale.ego_path_guidance.runtime` | `pytorch` | 使用 TEP PyTorch 推理。 | 当前不依赖 TensorRT，便于训练后直接验证。 | TensorRT 只在模型稳定、需要加速时考虑。 |
| `RailScale.ego_path_guidance.debug_every` | `20` | 每隔多少帧保存 EgoPath debug 图。 | 和 RailScale debug 默认间隔一致。 | 需要密集核对时调小。 |
| `RailScale.ego_path_guidance.sample_y_fracs` | `[0.58, 0.65, 0.72, 0.8, 0.88]` | EgoPath 与 RailScale 对比的采样行。 | 使用同一组中近距离行，便于比较中心/宽度差异。 | 若 RailScale 采样行改变，应同步。 |

### Hierarchical

| 参数 | 当前值 | 含义 | 为什么这样设 | 调整建议 |
|---|---:|---|---|---|
| `Hierarchical.voxel_size_lis` | `[0.1, 0.3, 1.0, 4.0, 15.0]` | 多层级高斯/锚点体素尺度。 | 沿用当前铁路 12MP 实验设置。 | 主要影响地图表达和渲染，不是本轮尺度主因。 |
| `Hierarchical.distance_lis` | `[10.0, 30.0, 70.0, 150.0]` | 各层级距离划分。 | 适配铁路场景长距离视野。 | 场景尺度变化大时再调。 |
| `Hierarchical.n_offsets` | `32` | 每个锚点的 offset 数量。 | 控制表示能力。 | 显存不足可降低，质量不足可提高。 |
| `Hierarchical.point_ratio` | `16` | 点/锚点抽样比例参数。 | 控制点云密度。 | 与渲染质量和速度相关。 |
| `Hierarchical.appearance_dim` | `64` | 外观特征维度。 | 保持当前渲染分支设置。 | 主要影响渲染，不是轨迹尺度主线。 |
| `Hierarchical.color_refinement_iter` | `0` | 颜色 refinement 迭代数。 | 当前轨迹实验不做颜色精修。 | 做最终可视化时可增加。 |
| `Hierarchical.rendering_width` | `1280` | 渲染输出宽度。 | 控制渲染评估/可视化成本。 | 渲染质量需求高时可增大。 |
| `Hierarchical.upsampling_method` | `bicubic` | 上采样方法。 | 当前渲染输出使用 bicubic。 | 只在渲染质量对比时改。 |
