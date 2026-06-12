# 前端跟踪、关键帧建图与 RailScale 作用位置说明

本文记录当前 GigaSLAM 主线配置下，每一帧在前端、后端和评估中的作用。该文档用于后续汇报、消融实验解释和论文写作参考。

## 核心结论

当前 rowtrack350 主线中，可以简化理解为：

```text
所有帧参与前端 tracking / VO；
只有关键帧进入后端 Gaussian mapping；
RailScale 作用在每一帧进入 VO 之前的 depth 上；
ATE 和轨迹图基于保存下来的所有估计帧，而不是只基于关键帧。
```

## 每帧处理流程

对每一个未被跳过的输入帧，前端执行以下流程：

```text
RGB frame
  ↓
Depth model 预测单目 depth
  ↓
RailScale 修正 depth，若开启
  ↓
ClassicTracking / VO 使用修正后的 depth 估计当前 pose
  ↓
记录当前帧 pose
  ↓
判断当前帧是否为 keyframe
  ↓
如果是 keyframe：送入后端 Gaussian mapping
如果不是 keyframe：不进入后端建图，但 pose 仍保留用于轨迹评估
```

因此，RailScale 的作用位置在前端 tracking 之前：

```text
depth prediction → RailScale correction → VO / PnP tracking
```

它不是只作用于关键帧，也不是后处理阶段的轨迹尺度对齐。

## 前端 tracking 与后端 mapping 的区别

### 前端 tracking / VO

前端 tracking 对每个未跳过帧运行，用于估计当前帧相机位姿。

当前主线里，每帧会先进行深度估计。如果 RailScale 开启，预测 depth 会先经过 RailScale 的 rail-guided metric scale correction，再送入 VO / PnP tracking。

这意味着 RailScale 会影响所有前端估计帧的 pose，而不仅仅影响关键帧。

### 后端 Gaussian mapping

后端不是处理所有帧，而是只处理被前端选中的关键帧。

当前帧被判定为 keyframe 后，前端会调用 `request_keyframe(...)`，将该帧、当前窗口和 depth map 送入后端。后端随后执行：

```text
add_next_kf(...)
map(current_window, ...)
```

也就是说，只有关键帧会参与 Gaussian anchor 初始化、建图和局部窗口优化。

## 关键帧选择

当前帧是否成为 keyframe 主要由 `is_keyframe(...)` 判断。判断条件包括：

- 与上一关键帧的帧间隔是否达到 `Training.kf_interval`
- 与上一关键帧的相对位移是否超过阈值
- 位移阈值会结合当前关键帧 depth 的 median depth 做尺度化

因此，关键帧不是固定每一帧都取，而是由运动间隔和位移条件共同决定。

## 轨迹评估使用哪些帧

最终保存的 `poses_est.txt` 来自前端记录的相机 pose。只要某帧被正常处理并保存在前端 `self.cameras` 中，它就会进入最终估计轨迹。

因此：

```text
ATE / evo_2dplot_original.png 评估的是已处理帧的前端估计轨迹；
不是只评估后端 keyframe 轨迹。
```

这点对解释实验结果很重要。非关键帧虽然不参与后端 Gaussian mapping，但它们的 pose 仍然影响 ATE 和轨迹图。

## 什么是被跳过的帧

代码中存在一个运动阈值跳帧逻辑：

```python
if cur_frame_idx > 2 and self.motion_thresh > 0:
    flow = self.classic_tracking.motion_flow_keyframe(
        viewpoint.image_ori, prev_kf=False
    )
    if flow < self.motion_thresh:
        cur_frame_idx += 1
        continue
```

含义是：如果当前帧相对参考帧运动太小，并且 `SLAM.motion_thresh > 0`，则该帧会被跳过。

被跳过的帧不会继续执行：

- depth prediction
- RailScale correction
- VO / tracking
- keyframe 判断
- backend mapping

不过当前 rowtrack350 主线配置中：

```yaml
SLAM:
  motion_thresh: 0.0
```

所以该跳帧机制实际上关闭。当前实验中可以认为所有帧都会进入前端处理。

## 对 RailScale 消融实验的含义

`w/o RailScale` 消融不是只关闭后端建图中的尺度约束，而是关闭了每帧进入 VO 前的 depth scale correction。

因此该消融会同时影响：

1. 每一帧前端 VO / tracking 使用的 depth scale；
2. 被选为 keyframe 的帧进入后端 Gaussian mapping 时使用的 depth scale；
3. 最终所有已处理帧的估计 pose 和 ATE。

换句话说，RailScale 的作用链路可以理解为：

```text
单目 depth scale 更准
  ↓
前端 VO / PnP 使用的 3D 点尺度更合理
  ↓
位姿估计更稳定，轨迹尺度更接近真实米制
  ↓
关键帧进入后端时 depth 也更合理
  ↓
Gaussian anchor 初始化和局部 mapping 的几何结构更稳定
```

因此，RailScale 不只是一个用于改善 ATE 的轨迹后处理模块，也不是只服务于后端建图的深度预处理模块。它位于 depth prediction 和 tracking / mapping 之间，通过修正单目深度的米制尺度，同时影响前端位姿估计和后端关键帧建图。

汇报时可以说：

```text
RailScale 在单目深度进入前端跟踪和后端建图之前进行米制尺度修正，因此既能改善前端位姿估计，也能为后端 Gaussian 建图提供更可靠的几何初始化。
```

对应英文表述：

```text
RailScale improves both frontend tracking and backend mapping by correcting the metric scale of monocular depth before it is consumed by VO and Gaussian map initialization.
```

需要注意的是，建图质量还会受到纹理、运动、关键帧选择和 Gaussian 优化等因素影响。因此写作时建议使用“有助于提升”“provides more reliable metric geometry”等相对稳妥的表述，避免说 RailScale 一定提升所有场景或完全解决建图质量问题。

更准确的消融解释是：

```text
w/o RailScale removes the rail-guided metric depth correction before frontend tracking and backend mapping.
```

中文可以表述为：

```text
w/o RailScale 消融去除了跟踪和建图之前的轨道几何尺度修正，因此同时影响前端位姿估计和后端关键帧建图。
```

## 汇报时推荐说法

可以在汇报中这样描述当前系统的数据流：

> 在当前实验配置中，所有帧都会进入前端完成单目深度估计和 VO tracking。RailScale 位于深度预测和 VO tracking 之间，对每一帧的 depth 进行基于轨道几何的米制尺度修正。随后系统根据运动和时间间隔选择关键帧，只有关键帧会进入后端 Gaussian mapping。最终 ATE 评估使用前端保存的所有已处理帧位姿，而不是只使用关键帧。

英文写作可以使用：

> RailScale is applied to each processed frame before frontend visual tracking. The corrected depth is used by VO/PnP for pose estimation, while only selected keyframes are sent to the backend Gaussian mapping module. Therefore, disabling RailScale affects both per-frame frontend tracking and keyframe-based mapping, and the final ATE is computed over all saved estimated poses rather than keyframes only.

## 相关代码位置

- `utils/slam_frontend.py`
  - 每帧 depth prediction、RailScale correction、ClassicTracking update
  - `is_keyframe(...)`
  - `request_keyframe(...)`
- `utils/slam_backend.py`
  - 后端接收 `"init"` 和 `"keyframe"` 消息
  - `add_next_kf(...)`
  - `map(current_window, ...)`
- `utils/rail_scale.py`
  - RailScale 主体实现
- `utils/eval_utils.py`
  - `save_estimated_poses(...)`
  - 轨迹保存和评估相关逻辑
