# DGC 借鉴矩阵

更新时间：2026-05-29

本文配合 [dgc_literature_survey.md](dgc_literature_survey.md) 使用，用于快速判断 DGC 可以从哪些高质量工作中借鉴机制，以及哪些方向最可能提升铁路场景效果。这里采用 **effect-first** 口径：已有工作覆盖程度不作为放弃理由，只用于判断机制成熟度、复现优先级和论文表述方式。

## 矩阵说明

符号含义：

- `Y`：明确支持或作为核心设计。
- `P`：部分相关，或可通过扩展支持。
- `N`：不是该工作的重点。
- `?`：需要读代码/全文细节进一步确认。

## 方法对比矩阵

| 方法 | online SLAM | metric scale prior | rail-specific prior | selective confidence | rendered depth loss | pose-depth joint optimization | trajectory metric | rendering metric | 对我们最有用的借鉴 |
|---|---|---|---|---|---|---|---|---|---|
| GigaSLAM | Y | P | N | N | P | P | Y | Y | 项目基础框架；RailScale 是我们的领域扩展 |
| Gaussian Splatting SLAM | Y | P | N | P | P | Y | Y | Y | 证明 GS 可用于 SLAM tracking/mapping |
| GS-SLAM | Y | Y(RGB-D) | N | P | Y | Y | Y | Y | RGB-D 深度约束成熟，但不解决单目铁路尺度 |
| SplaTAM | Y | Y(RGB-D) | N | P | Y | Y | Y | Y | RGB-D SLAM baseline，说明 mask/visible region 很重要 |
| Splat-SLAM | Y | P | N | P | Y | Y | Y | Y | DSPO：pose/depth/scale 联合优化；重点借鉴 selective prior |
| UDGS-SLAM | Y | P(UniDepth) | N | Y(local consistency) | Y | Y | Y | Y | UniDepth + local consistency filtering + pose/map 联合优化 |
| MonoGS++ | Y | P(VO sparse geometry) | N | P | P | Y | Y | Y | 更偏 VO + map regularization，不直接覆盖 RailScale |
| FSGS / DepthRegGS | N | P(COLMAP scale/offset) | N | N | Y | N | N | Y | 说明 monocular depth regularization 是常规做法 |
| DNGaussian | N | P(normalization) | N | P(local normalization) | Y | N | N | Y | 支持 global-local normalized DGC |
| CDGS | N | P(SfM/mono depth) | N | Y | Y | N | N | Y | 支持 confidence-aware DGC |
| In Depth We Trust | N | P(weak alignment) | N | Y(selective) | Y | N | N | Y | 支持 selective DGC，反对 naive 全图监督 |
| DET-GS | N | P(mono depth) | N | Y(edge-aware) | Y | N | N | Y | 支持 edge/structure-aware DGC |
| SAD-GS | N | Y(depth supervision) | N | P | Y | N | N | Y | 提醒只约束 depth center 不够，还要考虑 Gaussian shape |
| DepthSplat | N | P(depth model) | N | P | P | N | N | Y | 背景意义强，工程路线不同 |
| MonoSplat | N | P(depth foundation model) | N | P | P | N | N | Y | 启发使用 depth features/confidence，不是在线 SLAM |
| 我们的 RailScale | Y | Y(rail width) | Y | Y(row/width/status) | N | P | Y | P | 当前最强、最可解释主线创新 |
| 当前 naive DGC | Y | P(RailScale-corrected depth) | N | P(opacity only) | Y | P | Y | Y | 工程收益弱，适合作为 E-DGC0 基线 |
| 候选 Rail-aware selective DGC | Y | Y(RailScale) | Y | Y | Y | P | Y | Y | 最值得尝试的效果路线，必须全 6 序列验证 |

## 可借鉴强度与工程优先级

| 候选点 | 已有工作成熟度 | 工程优先级 | 备注 |
|---|---|---|---|
| rendered depth 与 monocular depth 做 loss | 高 | 中 | 可作为 E-DGC0 基线，但不值得只调 lambda |
| 每 keyframe 学一个 depth scale adapter | 中到高 | 中 | 可借鉴 DSPO 思路，但要防止 scale_delta 饱和 |
| UniDepth 辅助 Gaussian SLAM | 高 | 中 | 可借鉴 local consistency filtering，不必重新发明 |
| 全图 naive DGC | 高 | 低 | 缺少可靠性选择，scene16 收益弱 |
| opacity-gated DGC | 中 | 中 | 容易实现，但只用 opacity 可能不足 |
| RailScale confidence-gated DGC | 中低 | 高 | 关键是 RailScale 状态、row consistency、width stability |
| rail-neighborhood local normalized DGC | 中低 | 高 | 可把 DNGaussian 的 local normalization 改成铁路结构区域 |
| diagnostic-only rendered depth residual | 中 | 高 | 风险低，能解释误差机制，也能指导 gate |
| RailScale metric prior | 中 | 最高 | 当前主线，继续作为 baseline 和解释核心 |

## 对 DGC 下一步的决策树

```text
先跑 E-DGC0 全 6 序列
        |
        v
是否相对 rowtrack350 baseline 有稳定收益？
        |
   +----+----+
   |         |
  否        是
   |         |
停止 naive  继续检查是否牺牲渲染质量、
DGC 调参    scene13 是否退化、scale 是否饱和
   |
   v
只保留为 negative experiment
并转向 E-DGC3 diagnostic-only
        |
        v
residual 是否能稳定解释 along/end error？
        |
   +----+----+
   |         |
  否        是
   |         |
不继续 DGC  设计 E-DGC1/E-DGC2:
           rail-aware confidence-gated /
           rail-neighborhood normalized DGC
```

## 实验设计矩阵

| 实验 | 目的 | 是否进 loss | 关键 mask/权重 | 全 6 scene 必跑 | 成功标准 | 放弃条件 |
|---|---|---|---|---|---|---|
| E-DGC0 naive | 验证当前 DGC 是否有稳定价值 | Y | opacity + valid depth | Y | 至少多个 scene ATE/shape 改善且渲染不退化 | 只在单 scene 微小提升或普遍退化 |
| E-DGC1 confidence-gated | 验证可靠区域选择是否必要 | Y | opacity + RailScale status + row consistency + width stability + PnP quality | Y | scene14/17/19 至少一类 along/end error 可解释改善，scene13 不退化 | gate 只触发少数帧或指标无改善 |
| E-DGC2 normalized | 避免绝对深度冲突 | Y | rail-neighborhood global-local normalization | Y | 比 E-DGC0 更稳，scale_delta 不饱和 | 渲染下降或轨迹形态变差 |
| E-DGC3 diagnostic-only | 做误差机制分析 | N | residual curve + anomaly frames | Y | residual 与异常帧/along error 有稳定对应 | residual 与轨迹误差无关 |

## 写论文时的推荐表述

### 建议强调

- RailScale 是主线：利用铁路轨距与轨道行级一致性，将单目深度修正到 metric scale。
- DGC 调研结果显示，通用 depth-render consistency 已有很多成熟技巧，值得借鉴其中能提升效果的部分。
- 我们的工程重点在于：把这些成熟技巧和铁路物理结构结合，而不是盲目全图施加深度监督。
- 如果加入 DGC，应优先尝试 rail-aware selective consistency，而不是全图深度监督。

### 避免表述

- 不要说“首次提出 monocular depth 和 Gaussian rendered depth 一致性”。这会被 UDGS-SLAM、Splat-SLAM、FSGS、DNGaussian 等直接覆盖。
- 不要把 scene16 的 0.013 m ATE 改善夸大成稳定有效提升。
- 不要只报告 ATE。铁路场景必须同时报告 `along_p95`、`end_error`、`lat_p95` 和 `length_ratio`。

## 最推荐的论文模块包装

```text
RailScale: Railway-Structure-Guided Metric Depth Scaling
    |
    +-- rail pair detection and row-wise consistency
    +-- physical rail-width metric prior
    +-- conservative scale rejection / hold-last-scale
    +-- trajectory-shape diagnostics

Optional Analysis:
Depth-Gaussian Consistency as a diagnostic signal
    |
    +-- rendered depth residual
    +-- residual vs along-track drift
    +-- failure analysis and future rail-aware selective consistency
```

如果后续 E-DGC1/E-DGC2 有明显改善，再升级为：

```text
RailScale-v2: Rail-Aware Selective Depth-Gaussian Consistency
```

但在当前证据下，**不要把 DGC 放在 RailScale 前面**。

## 参考文献链接

- [GigaSLAM](https://arxiv.org/abs/2503.08071)
- [Gaussian Splatting SLAM](https://arxiv.org/abs/2312.06741)
- [GS-SLAM](https://openaccess.thecvf.com/content/CVPR2024/html/Yan_GS-SLAM_Dense_Visual_SLAM_with_3D_Gaussian_Splatting_CVPR_2024_paper.html)
- [SplaTAM](https://openaccess.thecvf.com/content/CVPR2024/papers/Keetha_SplaTAM_Splat_Track__Map_3D_Gaussians_for_Dense_RGB-D_CVPR_2024_paper.pdf)
- [Splat-SLAM](https://openaccess.thecvf.com/content/CVPR2025W/VOCVALC/papers/Sandstrom_Splat-SLAM_Globally_Optimized_RGB-only_SLAM_with_3D_Gaussians_CVPRW_2025_paper.pdf)
- [UDGS-SLAM](https://arxiv.org/abs/2409.00362)
- [MonoGS++](https://arxiv.org/abs/2504.02437)
- [Depth-Regularized Optimization for 3D Gaussian Splatting](https://arxiv.org/abs/2311.13398)
- [DNGaussian](https://openaccess.thecvf.com/content/CVPR2024/html/Li_DNGaussian_Optimizing_Sparse-View_3D_Gaussian_Radiance_Fields_with_Global-Local_Depth_CVPR_2024_paper.html)
- [CDGS](https://arxiv.org/abs/2502.14684)
- [In Depth We Trust](https://arxiv.org/abs/2604.05715)
- [DET-GS](https://arxiv.org/abs/2508.04099)
- [SAD-GS](https://openaccess.thecvf.com/content/CVPR2024W/NRI/html/Kung_SAD-GS_Shape-aligned_Depth-supervised_Gaussian_Splatting_CVPRW_2024_paper.html)
- [DepthSplat](https://openaccess.thecvf.com/content/CVPR2025/html/Xu_DepthSplat_Connecting_Gaussian_Splatting_and_Depth_CVPR_2025_paper.html)
- [MonoSplat](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_MonoSplat_Generalizable_3D_Gaussian_Splatting_from_Monocular_Depth_Foundation_Models_CVPR_2025_paper.html)
