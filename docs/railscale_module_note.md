# RailScale 模块写作参考

本文是论文写作参考，不是参数说明表。它用于解释当前主线中的 RailScale 模块为什么成立、如何工作、应该如何包装成论文方法，以及写作时需要避免的过度表述。具体 YAML 参数仍以 `configs/railway/rowtrack350/template.yaml` 和 `configs/railway/rowtrack350/README.md` 为准。

## 一句话定义

RailScale 是一个面向铁路场景的单目深度尺度修正模块。它利用铁路轨道在真实世界中具有稳定物理宽度这一先验，在每帧图像中检测左右轨结构，并用当前单目深度将左右轨点反投影到 3D；如果测得的 3D 轨道有效宽度偏离物理先验，则用二者比例估计深度尺度因子，在跟踪和 Gaussian 建图之前修正整帧深度。

英文写作可用：

> RailScale is a rail-guided metric depth scaling module that exploits the stable physical width of railway tracks to correct frame-wise monocular depth scale before visual tracking and Gaussian mapping.

更正式的模块名可考虑：

- RailScale: Rail-Guided Metric Depth Scaling
- Rail-Constrained Metric Scale Correction
- Rail-Guided Metric Scale Consistency

## 要解决的问题

当前系统依赖单目深度模型提供 metric depth。即使使用 UniDepth 这类 metric depth model，深度在铁路场景中仍可能存在帧级尺度偏差或缓慢漂移。这个问题会影响两部分：

- 跟踪：PnP/VO 需要用深度把 2D 匹配点提升到 3D，深度尺度偏差会直接转化为相机位移尺度偏差。
- 建图：Gaussian 初始化和空间分布依赖深度，深度尺度不稳会造成地图几何尺度不稳。

RailScale 的目标不是在评估后对轨迹做尺度对齐，而是在前端跟踪和后端建图之前，对输入深度做在线尺度修正。

## 核心假设

RailScale 只依赖一个核心物理假设：

> 图像中被检测到的左右轨结构，在 3D 中应该对应一个稳定的有效物理宽度。

当前主线中的 `metric_width_m: 1.6` 应理解为当前 detector 定义下的 rail-pair effective metric width prior。写论文时建议避免把它直接说成严格标准轨距或单根钢轨宽度。更稳妥的说法是：

> an effective metric width prior for the detected rail pair

这样既保留物理含义，也避免和标准轨距、钢轨头宽度等铁路工程术语发生不必要冲突。

## 系统位置

RailScale 位于单目深度模型之后、VO/PnP 跟踪和 Gaussian 建图之前：

```text
RGB frame
   |
   v
monocular metric depth model
   |
   v
RailScale depth correction
   |
   +--> VO / PnP tracking
   |
   +--> Gaussian mapping
```

因此它不是后处理模块。它改变的是进入跟踪与建图的深度输入，所以会同时影响轨迹尺度、位姿估计和地图几何。

## 核心流程

对每一帧，RailScale 执行以下步骤：

1. 输入 RGB 图像、单目预测深度 `D_t` 和相机内参 `K`。
2. 使用 rail detector 得到左右轨响应。
3. 在图像中下部选取若干采样行。
4. 在每条采样行上选择一对左右轨点。
5. 用当前深度将左右轨点反投影到 3D。
6. 计算当前深度下的 3D rail-pair width。
7. 对多条采样行的宽度取 median，得到鲁棒宽度估计。
8. 用物理宽度先验和测量宽度的比例得到尺度因子。
9. 将该尺度因子应用到整帧深度。

公式上，对第 `t` 帧第 `i` 条采样行：

```text
P^L_{t,i} = D_t(u^L_{t,i}) K^{-1} \tilde{u}^L_{t,i}
P^R_{t,i} = D_t(u^R_{t,i}) K^{-1} \tilde{u}^R_{t,i}
```

其中 `u^L` 和 `u^R` 是 detector 选出的左右轨像素位置，`\tilde{u}` 是齐次像素坐标。当前深度下测得的 3D 宽度为：

```text
w_{t,i} = ||P^R_{t,i} - P^L_{t,i}||
```

多行鲁棒聚合：

```text
\hat{w}_t = median_i(w_{t,i})
```

尺度修正因子：

```text
s_t = W_rail / \hat{w}_t
```

修正后的深度：

```text
D'_t = s_t D_t
```

其中 `W_rail` 是 rail-pair effective metric width prior。

## 为什么它可解释

RailScale 的每一次修正都有明确物理含义：

- 如果当前深度把左右轨测得过窄，说明深度尺度偏小，需要放大深度。
- 如果当前深度把左右轨测得过宽，说明深度尺度偏大，需要缩小深度。

例子：

```text
当前深度下测得 rail-pair width = 1.2 m
有效物理宽度先验 = 1.6 m
scale = 1.6 / 1.2 = 1.33
```

这表示当前帧深度整体偏小，RailScale 将深度乘以 `1.33`。

反过来：

```text
当前深度下测得 rail-pair width = 2.0 m
有效物理宽度先验 = 1.6 m
scale = 1.6 / 2.0 = 0.8
```

这表示当前帧深度整体偏大，RailScale 将深度乘以 `0.8`。

这种解释是论文中最有价值的部分：它不是黑盒学习出的尺度，而是由场景物理结构直接约束出的尺度。

## 鲁棒性设计

铁路场景不是理想直轨。多轨、道岔、弯道、遮挡和检测噪声都会让左右轨选择出错。RailScale 的鲁棒性设计可以概括为：

> only update the metric scale when the detected rail geometry is spatially plausible and temporally stable.

当前主线中的 rowtrack350、widthcap020 和 hold-last-scale 可以包装成三个子机制。

### 1. Geometric Plausibility Filtering

检测到的左右轨在图像中不能过宽。如果 rail pair 的像素宽度超过图像宽度的一定比例，通常意味着 detector 选到了邻轨、跨轨或远近不一致的错误组合。

论文中可写为：

> We reject rail pairs whose image-space width exceeds a conservative fraction of the image width, preventing cross-track or neighboring-track mismatches.

### 2. Row-wise Rail-Pair Consistency

RailScale 不只看单条采样行，而是从近处到远处在多条行上选择 rail pair。近处轨道更清楚、更不容易和远处岔线混淆，因此可以用近处行的 rail center 作为后续远处行的参考。

如果后续行的 rail center 相对参考中心跳变过大，则认为该行可能选到了岔线或邻轨，不参与尺度估计。

论文中可写为：

> We enforce row-wise rail-pair consistency by tracking the rail-pair center from near-field rows to farther rows, rejecting rows that deviate excessively from the current rail-pair hypothesis.

这个机制是解释道岔、弯道场景鲁棒性的关键。

### 3. Temporal Scale Stabilization

如果当前帧检测不可靠，RailScale 不会把错误检测转化为错误尺度。对于样本不足、中心跳变、宽度突变、行级不一致等情况，模块沿用上一帧可靠尺度。

论文中可写为：

> When the rail observation fails the reliability checks, we retain the last reliable scale instead of updating from a potentially corrupted rail measurement.

这个机制的意义是避免短时检测错误污染深度、跟踪和地图。

## 和 DepthSplat 的关系

DepthSplat 不适合作为当前系统的直接模块，因为它是 feed-forward multi-view 3DGS/NVS 框架，而当前项目是在线 monocular Gaussian SLAM。更合理的关系是：

- DepthSplat 强调 depth 与 Gaussian representation 之间的几何一致性。
- RailScale 在 depth 进入 Gaussian SLAM 前，先用铁路几何先验修正 depth scale。
- rendered depth diagnostic 则可以作为后续验证 depth-Gaussian consistency 的分析工具。

可以这样写：

> Inspired by the broader observation that depth and Gaussian geometry should be mutually consistent, our method first enforces a railway-specific metric consistency on monocular depth before it is used for tracking and mapping.

但不建议写成：

> We use DepthSplat.

或：

> Our method is based on DepthSplat.

## 模块图建议

论文中的模块图可以画成：

```text
RGB frame ----------------------+
                                |
                                v
                         rail detector
                                |
                                v
                    row-wise rail pair sampling
                                |
                                v
monocular depth -----> 3D rail width measurement
                                |
                                v
              effective metric width prior
                                |
                                v
                    robust scale estimation
                                |
                                v
                temporal reliability gate
                                |
                                v
                    corrected metric depth
                                |
                                v
             tracking + Gaussian mapping
```

图中要突出两条输入：

- 单目深度提供当前几何尺度。
- rail detector 提供可解释的铁路结构测量。

输出是 corrected metric depth，而不是直接输出 pose。

## 论文 Method 可用草稿

中文草稿：

> 为缓解单目深度在铁路场景中的尺度偏差，我们提出 RailScale，一个基于轨道几何先验的深度尺度修正模块。给定单目深度预测和相机内参，RailScale 首先在图像中检测左右轨结构，并在多条图像行上采样成对的左右轨点。随后，模块利用当前深度将这些点反投影至相机坐标系，计算当前深度尺度下的 3D rail-pair width。由于铁路轨道结构具有稳定的物理宽度，我们将多行宽度的中位数与有效物理宽度先验进行比较，得到帧级尺度因子，并用该因子修正整帧深度。修正后的深度被送入后续视觉跟踪和 Gaussian 建图模块，从而为单目 Gaussian SLAM 提供更稳定的 metric scale。

英文草稿：

> To mitigate scale bias in monocular metric depth for railway scenes, we introduce RailScale, a rail-guided depth scaling module. Given a monocular depth prediction and camera intrinsics, RailScale detects the left and right rail structures and samples rail pairs across multiple image rows. The sampled rail points are back-projected into 3D using the current depth, yielding a set of rail-pair width measurements under the predicted depth scale. We robustly aggregate these measurements with a median operator and compare the resulting width with an effective metric rail-width prior. The ratio provides a frame-wise depth scale factor, which is applied to the full depth map before visual tracking and Gaussian mapping.

鲁棒性段落：

> To avoid corrupting the depth scale with erroneous rail detections, RailScale updates the scale only when the rail-pair observation is spatially plausible and temporally stable. We reject overly wide rail pairs that are likely to correspond to neighboring tracks, enforce row-wise consistency of the rail-pair center from near-field to farther rows, and retain the last reliable scale when the current observation is ambiguous. These checks are particularly important in railway switches and curved tracks, where naive rail-pair selection can easily jump to a different track.

## 实验和消融建议

RailScale 适合做以下消融：

| 实验 | 目的 | 预期说明 |
|---|---|---|
| w/o RailScale | 验证铁路几何尺度先验是否必要 | 关闭整个模块，比较 ATE、end error、length ratio 和 along drift |
| metric width vs relative width | 验证显式物理先验是否优于自标定宽度 | metric prior 应更适合跨 scene |
| w/o row-wise consistency | 验证道岔/弯道中行级一致性的作用 | scene13 这类场景应更容易退化 |
| w/o hold-last-scale | 验证错误检测是否会污染尺度 | 关注 width_jump、too_few_samples 后的尺度尖峰 |
| different metric width priors | 验证 prior 选择敏感性 | 不能只看单 scene，必须跨 6 个 scene 比较 |

指标不要只用 ATE。建议同时报告：

- ATE RMSE
- end error
- length ratio
- along_p95
- lat_p95
- rail scale status distribution
- width/scale temporal stability

这样能支撑“尺度修正改善轨迹形态”而不是只追求单个数字。

## 写作时要避免的说法

不建议写：

- RailScale solves monocular SLAM scale drift completely.
- RailScale directly optimizes camera poses.
- `metric_width_m: 1.6` is the standard railway gauge.
- The method uses DepthSplat.
- The module is tuned for each sequence.

建议写：

- RailScale provides an interpretable metric prior for monocular depth scale.
- RailScale corrects depth before tracking and mapping.
- `metric_width_m` is an effective rail-pair width prior under the current detector definition.
- The method is evaluated with a fixed cross-scene configuration.
- The module is designed for railway scenes where rail structures provide stable geometric cues.

## 当前主线中的对应实现

代码位置：

- `utils/rail_scale.py`：RailScale 主体实现。
- `utils/slam_frontend.py`：单目深度预测后调用 RailScale，并将修正后的深度送入 VO/PnP 和 mapping。

配置入口：

- `configs/railway/rowtrack350/template.yaml`
- `configs/railway/rowtrack350/scenes/*.yaml`

诊断输出：

- `rail_scale_diag.csv`：帧级尺度、宽度、状态、置信度。
- `rail_scale_rows.csv`：行级采样、左右轨点、拒绝原因。
- `rail_scale_debug/`：可视化检查图。

当前论文主线应描述为：

```text
RailScale = metric rail-pair width prior
           + row-wise rail-pair consistency
           + temporal scale stabilization
```

这比列举 `rowtrack350`、`widthcap020`、`hold-last-scale` 更适合论文表达。
