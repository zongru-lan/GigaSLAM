# 轨迹评估指标与口径说明

本文说明当前 GigaSLAM 铁路主线中轨迹评估的文件约定、对齐方式和指标含义。目标是让 ATE、RPE 和轨迹形态指标使用同一套清晰口径，避免再次出现位姿约定或帧号对齐误解。

## 核心结论

当前主线评估应坚持以下口径：

```text
估计轨迹 poses_est.txt：C2W
GT 轨迹 poses_gt.npy：W2C，评估前必须取逆为 C2W
帧号对齐：必须使用 poses_idx.txt
全局对齐：SE3 no scale，不做 Sim3 尺度校正
```

其中最重要的是：

```text
不要按顺序硬配估计 pose 和 GT pose；必须按 poses_idx.txt 取对应 GT 帧。
不要用 Sim3 scale correction 作为正式轨迹指标；否则会掩盖单目尺度误差。
```

## 评估输入文件

每个结果目录中与轨迹评估相关的文件主要有：

| 文件 | 含义 | 位姿约定 |
|---|---|---|
| `poses_est.txt` | SLAM 保存的估计相机位姿，每行一个 flattened 4x4 矩阵 | C2W |
| `poses_idx.txt` | 每个估计 pose 对应的原始输入帧编号 | frame id |
| `config.yml` | 本次运行使用的配置，包含 `Dataset.pose_path` | - |
| `Dataset.pose_path` 指向的 `.npy` | GT 位姿数组，shape 为 `(N, 4, 4)` | W2C |

铁路数据的 GT `.npy` 来自 `scripts/read_pose.py`，该脚本输出的是 W2C 矩阵。因此评估时必须先做：

```text
T_gt_c2w[i] = inv(T_gt_w2c[i])
```

而 `poses_est.txt` 当前已经是 C2W，不应再取逆。

## 帧号对齐

`poses_idx.txt` 是评估中必须使用的文件。它记录：

```text
第 k 个估计 pose 对应原始数据集中的第 frame_id 帧
```

评估时应构造配对：

```text
T_est[k]       <-> T_gt_c2w[poses_idx[k]]
```

而不是：

```text
T_est[k]       <-> T_gt_c2w[k]
```

后者在存在跳帧、丢帧或非连续处理时会把估计位姿配到错误 GT 帧，导致 ATE/RPE 口径错误。

当前 rowtrack350 主线中：

```yaml
SLAM:
  motion_thresh: 0.0
```

因此运动阈值跳帧机制关闭，通常每帧都会进入前端处理。但评估代码仍应始终使用 `poses_idx.txt`，因为这是更稳健、可复现的口径。

## SE3 no scale 对齐

当前正式轨迹评估使用：

```text
SE3 no scale
```

含义是：

```text
允许估计轨迹和 GT 之间做一个全局刚体对齐：旋转 R + 平移 t
不允许额外缩放尺度 s
```

即对估计轨迹应用：

```text
T_est_aligned = A_se3 * T_est
```

其中 `A_se3` 只包含旋转和平移。

这和 Sim3 scale corrected 不同。Sim3 会额外估计一个全局尺度：

```text
T_est_aligned = A_sim3 * T_est,  A_sim3 = scale + rotation + translation
```

对于单目 SLAM，Sim3 尺度校正会把全局尺度误差吸收掉，使轨迹看起来更好，但这会掩盖我们真正关心的米制尺度能力。由于 RailScale 的目标正是改善 metric scale，正式结果必须使用 `SE3 no scale`。

Sim3 scale corrected 可以作为诊断或可视化辅助，但不应作为主表正式指标。

## ATE：全局绝对轨迹误差

ATE 是 Absolute Trajectory Error。它衡量每一帧相机位置在全局对齐后的绝对偏差。

在完成 `poses_idx.txt` 配对、GT W2C 转 C2W、SE3 no scale 对齐后，ATE 平移误差为：

```text
e_i = || trans(T_gt_i) - trans(T_est_aligned_i) ||
```

常用汇总包括：

| 指标 | 含义 |
|---|---|
| `rmse` | 整条轨迹的均方根位置误差，主指标 |
| `mean` | 平均位置误差 |
| `median` | 中位位置误差 |
| `max` | 最大单帧位置误差 |
| `std` | 误差波动程度 |

ATE 更关注整条轨迹的全局一致性，适合回答：

```text
整条轨迹整体离 GT 有多远？
最终轨迹尺度和方向是否合理？
```

但 ATE 不足以解释局部 VO 是否抖动，也不容易区分短时局部退化和长期累计漂移。

## RPE：局部相对位姿误差

RPE 是 Relative Pose Error。它衡量一小段时间内的相对运动估计是否准确。

给定 C2W 位姿：

```text
T_gt_i,  T_gt_j
T_est_i, T_est_j
j = i + delta
```

先计算 GT 的相对运动：

```text
Delta_gt = inv(T_gt_i) @ T_gt_j
```

再计算估计轨迹的相对运动：

```text
Delta_est = inv(T_est_i) @ T_est_j
```

相对运动误差为：

```text
E_i = inv(Delta_gt) @ Delta_est
```

### RPE translation

平移 RPE 取误差矩阵的平移部分：

```text
rpe_trans_i = || E_i[0:3, 3] ||
```

单位是米。

它回答：

```text
这 delta 帧内，系统估计的局部位移和 GT 差多少？
```

### RPE rotation

旋转 RPE 取误差矩阵旋转部分的旋转角：

```text
R_err = E_i[0:3, 0:3]
rpe_rot_i = arccos((trace(R_err) - 1) / 2)
```

通常输出为 degree：

```text
rpe_rot_deg_i = rpe_rot_i * 180 / pi
```

它回答：

```text
这 delta 帧内，系统估计的局部转向和 GT 差多少？
```

## delta 的含义

建议默认计算：

```text
delta = 1, 5, 10 frames
```

| delta | 解释 | 主要用途 |
|---:|---|---|
| `1` | 相邻帧相对运动误差 | 检查前端 VO 是否逐帧抖动 |
| `5` | 短窗口相对运动误差 | 检查短期累计漂移 |
| `10` | 更长局部窗口误差 | 检查局部漂移趋势 |

这里的 delta 建议基于已经对齐后的估计 pose 序列下标计算，同时在输出中记录实际 `frame_id_i` 和 `frame_id_j`。当前主线不开启跳帧时，序列下标和原始帧号通常一致；如果未来开启跳帧，记录 frame id 可以避免误读。

## RPE 与 SE3 对齐的关系

RPE 比较的是相对运动。若估计轨迹只做全局 SE3 刚体左乘对齐：

```text
T'_est_i = A * T_est_i
T'_est_j = A * T_est_j
```

则：

```text
inv(T'_est_i) @ T'_est_j = inv(T_est_i) @ T_est_j
```

因此，在 `SE3 no scale` 口径下，RPE 对全局刚体对齐不敏感。实际实现中可以在 ATE 对齐后的同一组配对 pose 上计算 RPE，也可以在配对后的原始估计 pose 上计算；两者的相对运动误差等价。

但如果使用 Sim3 scale correction，RPE translation 会被全局尺度影响。因此正式 RPE 也应遵循 `SE3 no scale`，不使用 Sim3 尺度校正。

## 轨迹形态指标

除 ATE/RPE 外，当前项目还保存轨迹形态指标，用于解释“ATE 还行但轨迹形状不理想”的情况。

典型输出文件：

```text
plot/shape_original.json
```

核心字段包括：

| 字段 | 含义 |
|---|---|
| `length_gt` | GT 轨迹长度 |
| `length_est_aligned` | SE3 对齐后的估计轨迹长度 |
| `length_ratio` | 估计长度 / GT 长度，反映整体尺度是否偏长或偏短 |
| `end_error` | 终点误差，反映累计漂移的最终结果 |
| `lat_p95` | 横向误差 95 分位，反映偏轨风险 |
| `along_p95` | 沿轨误差 95 分位，反映前后方向累计漂移 |
| `first25_rmse` | 前 25% 轨迹 RMSE |
| `mid50_rmse` | 中间 50% 轨迹 RMSE |
| `last25_rmse` | 后 25% 轨迹 RMSE |

铁路场景里，横向误差和沿轨误差都很重要：

- `lat_p95` 高：可能存在偏到邻轨、道岔误选或轨迹横向形态错误。
- `along_p95` 高：可能存在沿轨方向累计漂移，ATE 可能不极端但终点/局部形态不理想。

## 推荐输出文件

当前已有输出：

```text
plot/stats_original.json      # ATE 统计
plot/shape_original.json      # 轨迹形态指标
plot/trj_original.json        # 对齐后的轨迹数据和帧号
plot/evo_2dplot_original.png  # 2D 轨迹图
```

建议新增 RPE 后输出：

```text
plot/rpe_original.json
plot/trajectory_metrics_original.json
```

建议 `rpe_original.json` 保存：

```text
alignment_mode
est_convention
gt_convention
deltas
每个 delta 的 translation / rotation mean, median, rmse, p95, max
每个 delta 的 world_translation 诊断指标
可选：top error frame pairs
```

其中 `translation_m` 是 evo-style SE(3) 相对位姿平移误差，严格遵循 RPE 定义；`world_translation_m` 是相隔 delta 帧的世界坐标位移向量误差，用于排查相机坐标轴约定差异时的局部位移一致性。正式汇报时应说明采用哪一个口径。

建议 `trajectory_metrics_original.json` 汇总：

```text
ATE rmse / mean / median / max
RPE delta=1/5/10 translation
RPE delta=1/5/10 world_translation
RPE delta=1/5/10 rotation
length_ratio
end_error
lat_p95
along_p95
num_poses
alignment_mode
```

这样后续汇报和论文表格可以同时呈现全局轨迹误差、局部运动误差和铁路场景下更有解释力的轨迹形态误差。

## 实现注意事项

RPE 接入代码时建议遵守以下规则：

1. 复用当前 ATE 评估的数据读取和帧号配对逻辑。
2. GT `.npy` 必须按 W2C 读取并取逆为 C2W。
3. `poses_est.txt` 默认按 C2W 解释。
4. 必须使用 `poses_idx.txt` 对齐 GT 帧号。
5. 正式指标使用 `SE3 no scale`，不做 Sim3 尺度校正。
6. RPE 的 delta 默认使用 `1, 5, 10`。
7. 输出 rotation RPE 时同时写明单位，推荐 degree。
8. 若某个 delta 超出轨迹长度，应跳过并记录有效 pair 数。
9. 输出中应记录 `num_pairs`，避免不同序列长度造成误读。

推荐实现位置：

```text
utils/eval_utils.py
```

原因是 ATE、shape、轨迹图已经在该文件中生成。RPE 应与它们共享同一套读取、配对和评估口径。

手动重算旧结果时，`scripts/eval_ate.py` 也应复用同一套 RPE 逻辑，确保自动评估和手动评估输出一致。


## Baseline 统一评估

与其他算法对比时，不能直接使用 baseline 原仓库默认输出的 ATE，因为不同仓库常见差异包括：

| 差异 | 风险 |
|---|---|
| 是否做 Sim3 尺度校正 | 可能掩盖单目尺度误差 |
| GT/估计位姿 C2W/W2C 约定 | 可能把轨迹反向或错误取逆 |
| 是否只评估关键帧 | 可能和 GigaSLAM 全帧评估不一致 |
| 是否按帧号对齐 | 跳帧或关键帧场景下可能错配 GT |

因此，baseline 对比应尽量转换成 GigaSLAM 的统一口径：

```text
估计轨迹：C2W
GT：铁路 .npy W2C，评估前取逆
帧号：显式 frame_id 对齐
对齐：SE3 no scale
输出：ATE + RPE + shape metrics
```

### Splat-SLAM 适配器

Splat-SLAM 的铁路适配结果可以用以下脚本重算：

```bash
cd /root/GigaSLAM
/root/miniconda3/envs/gigaslam/bin/python scripts/evaluate_splat_slam_trajectory.py \
  --result-dir results/baselines/Splat-SLAM/scene_16_train
```

脚本会自动读取：

```text
results/baselines/Splat-SLAM/scene_16_train/cfg.yaml
results/baselines/Splat-SLAM/scene_16_train/video.npz
```

并输出：

```text
results/baselines/Splat-SLAM/scene_16_train/gigaslam_eval/
```

其中：

| 子目录 | 含义 |
|---|---|
| `keyframes/` | 从 Splat-SLAM `video.npz` 中读取关键帧 C2W pose 后重算 |
| `full/` | 如果存在 `traj/full_traj_est_c2w.txt`，则读取全帧 C2W pose 后重算 |

当前 Splat-SLAM 原始代码默认只把 full trajectory 用于在线评估，不保存到磁盘。后续已补充输出钩子：完整运行结束后会保存：

```text
traj/full_traj_est_c2w.txt
traj/full_traj_frame_ids.txt
```

这样重跑 Splat-SLAM 后，适配器会自动额外生成 all-frame 的 `full/` 评估结果。若旧结果没有这两个文件，只能对 `video.npz` 中已有的关键帧轨迹做统一评估；这可以用于诊断，但不能替代全帧公平对比。

为方便和 GigaSLAM 的渲染结果图逐帧对比，Splat-SLAM 的渲染评估还会保存关键帧纯 RGB 单图：

```text
rendered_keyframes_after_refine/img_038.png   # Splat-SLAM 渲染图
gt_keyframes_after_refine/img_038.png         # 对应输入 GT 图
rendered_keyframes_after_refine_manifest.json # frame_id/video_idx/文件路径映射
```

命名中的 `038` 使用原始输入帧号，而不是 Splat-SLAM 内部 keyframe 编号。这样可以直接和 GigaSLAM 的 `img_038.png` 对齐查看。

### Splat-SLAM 轨迹汇报图

统一评估完成后，可以复用 GigaSLAM 的 report plot 脚本绘制同风格轨迹图：

```bash
cd /root/GigaSLAM
/root/miniconda3/envs/gigaslam/bin/python scripts/make_report_trajectory_plots.py \
  --results-dir results/baselines/Splat-SLAM \
  --trj-glob 'scene_16_train/gigaslam_eval/*/plot/trj_original.json' \
  --output-dir results/baselines/Splat-SLAM/scene_16_train/gigaslam_eval/report_plots/se3_no_scale \
  --alignment se3_no_scale \
  --colorbar-vmax 100
```

如果也需要 Sim3 诊断图，只改输出目录和 alignment：

```bash
/root/miniconda3/envs/gigaslam/bin/python scripts/make_report_trajectory_plots.py \
  --results-dir results/baselines/Splat-SLAM \
  --trj-glob 'scene_16_train/gigaslam_eval/*/plot/trj_original.json' \
  --output-dir results/baselines/Splat-SLAM/scene_16_train/gigaslam_eval/report_plots/sim3_scale_corrected \
  --alignment sim3_scale_corrected
```

旧结果只有 `keyframes/` 时只会画关键帧轨迹；重跑 Splat-SLAM 生成 `full/` 统一评估后，同一条 `--trj-glob` 会同时画 `keyframes` 和 `full`。

## 汇报时怎么讲

可以这样解释三类指标的互补关系：

```text
ATE measures global trajectory accuracy after rigid SE3 alignment without scale correction.
RPE measures local relative motion consistency over short frame intervals.
Shape metrics decompose railway-specific trajectory errors into length ratio, endpoint drift, lateral drift, and along-track drift.
```

中文表述：

```text
ATE 看整条轨迹整体准不准；RPE 看局部运动估计稳不稳；轨迹形态指标进一步把铁路场景中的尺度、终点、横向偏轨和沿轨漂移拆开解释。
```
