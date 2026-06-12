# Rail Detector 与 RailScale 关系说明

本文解释当前 RailScale 模块中“训练了什么模型、模型预测什么、RailScale 如何使用模型输出”。它和 `docs/railscale_module_note.md` 互补：前者讲 detector，后者讲完整的尺度修正模块。

## 一句话结论

我们训练的是一个基于 SegFormer-B0 的 rail-edge detector。它输入铁路 RGB 图像，输出两个 dense heatmap channel：左轨边缘概率图和右轨边缘概率图。模型本身不预测深度、不预测相机位姿、不预测尺度，也不直接识别 ego track；RailScale 会从这些 heatmap 候选中选择一个几何一致的 rail pair，用于后续 3D 宽度测量和深度尺度修正。

英文写作可用：

> We train a lightweight SegFormer-based rail-edge detector to predict two dense heatmaps corresponding to the left and right rail edges. RailScale then selects a geometrically consistent rail pair from these heatmaps for metric scale estimation.

## 模型预测什么

模型输入：

```text
RGB railway image
```

模型输出：

```text
left rail-edge heatmap
right rail-edge heatmap
```

每个 heatmap 是一张像素级概率图。它表示：

- 某个像素像不像 left rail edge；
- 某个像素像不像 right rail edge。

因此 detector 是一个 dense rail-edge heatmap predictor，而不是完整的轨道实例分割器。

## 模型不预测什么

写论文或汇报时要特别避免把 detector 说过头。它不预测：

- monocular depth；
- camera pose；
- SLAM trajectory；
- ATE 或轨迹误差；
- depth scale；
- 3D Gaussian；
- 当前列车所在 ego track 的实例 ID。

更准确的说法是：detector 提供可见轨道结构的 2D 几何观测，RailScale 后续再根据几何规则选择目标 rail pair。

## 模型结构

当前实现使用 HuggingFace 的 `SegformerForSemanticSegmentation`，并把 segmentation head 当作 two-channel heatmap head 使用。

代码位置：

```text
utils/rail_detector_model.py
```

核心结构：

```text
SegFormer-B0 backbone + segmentation head
num_channels = 2
```

两个输出通道分别对应：

```text
channel 0: left rail edge
channel 1: right rail edge
```

当前主线配置中的 checkpoint：

```text
/autodl-fs/data/GigaSLAM/railbench/rail_detector_runs/segformer_b0/best.pt
```

对应配置项：

```yaml
RailScale:
  mode: detector
  detector_checkpoint: /autodl-fs/data/GigaSLAM/railbench/rail_detector_runs/segformer_b0/best.pt
  detector_image_size: [768, 432]
```

## 训练目标

模型以左右轨边缘 heatmap 作为监督信号，使用：

```text
BCEWithLogits + soft Dice
```

也就是说，它被训练成一个两通道热力图预测器，而不是普通的多类别语义分割器。

训练目标可以概括为：

> Given an RGB railway image, predict dense probability heatmaps for left and right rail edges.

## 为什么不是“找出所有轨道”

可以说 detector 会响应图像中可见的轨道结构，但不能简单说“模型找出所有轨道”。原因是：

- 输出是 heatmap，不是轨道实例列表；
- 多轨、邻轨、道岔都会在 heatmap 中产生响应；
- detector 不负责判断哪一对轨道是当前目标轨道；
- ego track selection 是 RailScale 后处理完成的。

因此更准确的描述是：

```text
detector: predicts possible left/right rail-edge pixels
RailScale: selects one geometrically consistent rail pair
```

论文中建议写：

> The detector predicts dense left/right rail-edge heatmaps, from which RailScale selects a geometrically consistent rail pair for metric scale estimation.

不建议写：

> The detector identifies the ego railway track.

也不建议写：

> The detector detects all railway tracks.

除非后续真的做了轨道实例级标注和实例选择。

## RailScale 如何使用 heatmap

RailScale 并不是直接把整张 heatmap 拿来算尺度，而是在若干图像行上进行采样和选择。

流程如下：

1. 对 RGB 图像运行 rail detector，得到 left/right heatmap。
2. 在图像中下部选取若干采样行。
3. 每一行分别在 left/right heatmap 上寻找 peak。
4. 将 left peak 和 right peak 组合成候选 rail pair。
5. 用几何规则筛选候选：
   - right peak 必须在 left peak 右侧；
   - rail pair 像素宽度不能过小或过大；
   - pair center 应接近目标中心或上一帧中心；
   - 多条采样行之间的中心应保持一致。
6. 得到当前帧用于尺度估计的目标 rail pair。
7. 若候选不可靠，则拒绝本帧新观测，必要时沿用上一帧稳定 scale。

这个过程说明：

```text
模型输出 = dense rail-edge candidates
RailScale 输出 = selected rail pair + depth scale
```

## detector 与 RailScale 的分工

| 组件 | 输入 | 输出 | 作用 |
|---|---|---|---|
| Rail detector | RGB image | left/right rail-edge heatmaps | 提供 2D 轨道边缘候选 |
| UniDepth | RGB image | monocular metric depth | 提供当前帧深度 |
| RailScale geometry | heatmaps + depth + intrinsics | frame-wise depth scale | 用 rail-pair 3D 宽度反推深度尺度 |

最重要的分工是：

> detector only provides 2D rail observations; metric scale is computed by geometry, not learned directly.

中文表述：

> detector 只提供二维轨道观测，尺度不是网络直接学出来的，而是由轨道物理宽度约束和当前深度反投影几何计算得到的。

## 和当前目标轨道选择的关系

在多轨或道岔场景中，heatmap 里可能同时出现多条轨道的响应。RailScale 通过 row-wise consistency 和 temporal stabilization 来选择稳定 rail pair：

- 近处行通常更清楚、更不容易和远处岔线混淆；
- 从近处行向远处行跟踪 rail-pair center；
- 行间中心跳变过大的候选会被拒绝；
- 当前帧不可靠时不更新尺度。

这就是为什么我们不能把 detector 简单理解为“直接找到了当前轨道”。当前轨道选择是 detector heatmap 和 RailScale 几何规则共同完成的。

## 论文 Method 可用草稿

中文草稿：

> 为获得稳定的铁路结构观测，我们训练了一个基于 SegFormer-B0 的 rail-edge detector。该网络以 RGB 图像为输入，输出两个密集热力图，分别表示左轨边缘和右轨边缘的像素级概率。需要强调的是，该网络不直接预测深度、位姿或尺度；它只提供二维轨道边缘候选。随后，RailScale 在多条图像采样行上从左右热力图中选择满足几何一致性的 rail pair，并结合单目深度和相机内参反投影到 3D，用于估计帧级深度尺度。

英文草稿：

> To obtain reliable railway-structure observations, we train a SegFormer-B0 based rail-edge detector. Given an RGB frame, the detector predicts two dense heatmaps corresponding to the left and right rail edges. The detector does not directly estimate depth, camera pose, or metric scale; instead, it provides 2D rail-edge candidates. RailScale then samples these heatmaps across multiple image rows and selects a geometrically consistent rail pair, which is back-projected with monocular depth and camera intrinsics for frame-wise depth scale estimation.

## 写作时推荐和避免的说法

推荐：

- rail-edge detector
- left/right rail-edge heatmaps
- dense rail-edge probability maps
- geometrically consistent rail pair selection
- 2D rail observation for metric scale estimation

避免：

- ego-track detector
- all-track instance detector
- scale prediction network
- depth prediction network
- pose prediction network

## 当前实现对应位置

代码：

```text
utils/rail_detector_model.py
utils/rail_scale.py
```

配置：

```text
configs/railway/rowtrack350/template.yaml
```

关键配置项：

```yaml
RailScale:
  mode: detector
  detector_checkpoint: /autodl-fs/data/GigaSLAM/railbench/rail_detector_runs/segformer_b0/best.pt
  detector_image_size: [768, 432]
  detector_peak_topk: 8
  detector_peak_nms_px: 24
  detector_max_width_px_frac: 0.2
  detector_row_consistency: true
  detector_row_target_tracking: true
```

一句话总结：

> 我们训练了一个左右轨边缘 heatmap detector；它负责提供 2D rail candidates，RailScale 再通过几何一致性选择目标 rail pair，并用物理宽度先验计算深度尺度修正。
