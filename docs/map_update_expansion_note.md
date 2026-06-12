# GigaSLAM Map Update 和 Map Expansion 笔记

这份笔记结合 GigaSLAM 论文中的 **Map Update** 和 **Map Expansion** 两段，整理地图如何被优化和扩展，并对照当前项目代码中的主要实现位置。

## 1. 总体理解

GigaSLAM 的地图维护可以理解成两个动作：

```text
Map Expansion：看到新区域 -> 新增 voxel / anchor
Map Update：已有地图渲染当前视角 -> 用 loss 优化 anchor / Gaussian / MLP / pose
```

一句话概括：

```text
扩展解决“地图有没有”；
更新解决“地图准不准”。
```

## 2. Map Expansion：地图扩展

地图扩展发生在前端选出新的 keyframe 后。前端会把 keyframe 的相机位姿、RGB、深度图和当前窗口发送给后端。后端用这一帧的深度图生成新的 3D 点云，再把点云转成分层 voxel / anchor。

流程大致是：

```text
keyframe depth map
-> 反投影成 3D point cloud
-> 按 LoD 距离分层 voxelize
-> 每个非空 voxel 生成一个 anchor
-> spatial hash 去重
-> 只把新 anchor 加入地图
```

代码入口在：

```text
gaussian_splatting/scene/scaffold_model.py:extend_gaussian
```

其中：

```text
init=True  -> create_from_pcd，初始化地图
init=False -> add_pcd，扩展已有地图
```

## 3. 从深度图到点云

扩展地图时，系统先根据当前 keyframe 的深度图和相机位姿生成点云。

代码逻辑大致对应：

```python
point_cloud, color = self.pcd_from_depth(depthmap, T_w2c, rgb=rgb, mask=...)
```

其中 mask 会对深度点进行采样，避免所有像素都转成点导致地图过密。

## 4. 分层体素化

生成点云后，会按照 LoD 配置进行分层体素化。例如：

```yaml
voxel_size_lis: [0.1, 0.3, 1.0, 4.0, 15.0]
distance_lis: [10.0, 30.0, 70.0, 150.0]
```

含义是：

```text
0-10m       使用 0.1m 体素
10-30m      使用 0.3m 体素
30-70m      使用 1.0m 体素
70-150m     使用 4.0m 体素
150m以上    使用 15.0m 体素
```

代码里会对每一层分别做 voxelize：

```python
fused_point_cloud = self.voxelize_sample(points, voxel_size=self.voxel_size_lis[i])
```

然后根据点到当前相机中心的距离，把点分配到对应 LoD 层级。

## 5. Spatial Hash 去重

论文中提到 Map Expansion 的关键问题是：连续帧之间视野重叠很大，后续 keyframe 生成的新 voxel 可能已经存在于地图中。如果不去重，会反复加入重复 anchor，导致显存和计算增长，也可能产生重影。

因此论文使用 spatial hashing 做去重。

当前代码中的对应逻辑是：

```python
result = self.anchor_dict[level].hash_detect(fused_point_cloud)
fused_point_cloud = fused_point_cloud[~result]
```

含义是：

```text
如果该 voxel / anchor 已经存在 -> 丢弃
如果该 voxel / anchor 不存在 -> 加入地图
```

每个 LoD level 有自己的 hash table，因此去重也是按层级进行的。

## 6. 新 Anchor 的初始化

对于真正新增的 anchor，系统会初始化以下量：

```text
_anchor       anchor 位置
_offset       n_offsets 个局部偏移，初始为 0
_anchor_feat  anchor 特征，初始为 0
_scaling      基础尺度
_rotation     初始旋转
_opacity      初始透明度
_level        所属 LoD 层级
_anchor_index 来自哪个 keyframe
```

这些量之后会通过渲染 loss 反向传播不断优化。

## 7. Map Update：地图更新

Map Update 是对已有地图进行优化。每次后端收到 keyframe 后，会在当前窗口内做若干次 mapping iteration。

一次地图更新大致是：

```text
1. 取当前窗口中的 keyframes
2. 对每个 keyframe，选择视锥内且 LoD 距离匹配的 anchors
3. 用这些 anchors 解码出 Gaussians
4. differentiable rendering 得到渲染图像
5. 渲染图像与真实 RGB 计算 loss
6. loss 反向传播，更新地图参数和部分 pose / exposure
```

代码中对应位置主要在：

```text
utils/slam_backend.py:map
```

其中会先调用：

```python
anchor_in_frustum(...)
```

选出当前视角下需要参与渲染和优化的 anchors。

## 8. 渲染和损失

论文中的渲染损失写为：

```text
L_render = L1(I_GS, I_gt) + SSIM(I_GS, I_gt)
```

这里：

```text
I_GS 是 Gaussian Splatting 渲染图像
I_gt 是真实 RGB 图像
```

论文还加入：

```text
L_smooth：深度平滑正则，减少单目稀疏输入下的过拟合
L_iso：各向同性正则，惩罚长条形 / 高 aspect ratio 的 Gaussian
```

论文中的总损失是：

```text
L_total = L_render + λ_i L_iso + λ_s L_smooth
```

## 9. 当前代码中的损失实现

当前代码和论文描述整体方向一致，但具体实现略有差异。

在线 mapping 阶段主要使用：

```text
RGB L1 loss
+ isotropic regularization
```

其中 RGB loss 在：

```text
utils/slam_utils.py:get_loss_mapping_rgb
```

isotropic regularization 在：

```text
utils/slam_backend.py:map
```

大致形式是让 Gaussian 的不同尺度维度不要差异过大，避免出现特别细长或畸形的 Gaussian。

`color_refinement` 阶段会使用：

```text
L1 + SSIM + scaling regularization
```

对应代码在：

```text
utils/slam_backend.py:color_refinement
```

深度平滑相关函数在代码中存在，例如 `depth_reg`，但当前在线 mapping 主路径中没有显式加入论文公式里的 `L_smooth`。

## 10. 反向传播优化哪些参数

地图更新时，loss 会反向传播并优化：

```text
anchor position
n_offsets 个 offset
anchor_feat
anchor scaling
MLP 参数
部分 keyframe pose
exposure 参数
```

其中：

```text
anchor + offset * anchor_scale
```

决定每个 Gaussian 的中心位置。

MLP 根据：

```text
anchor_feat + view direction + distance + level (+ camera appearance)
```

预测该 anchor 下 `n_offsets` 个 Gaussian 的颜色、透明度、尺度和旋转。

## 11. Map Expansion 和 Map Update 的关系

二者是互补关系：

```text
Map Expansion：负责发现新区域，把新 voxel / anchor 加入地图。
Map Update：负责优化已有 anchor / Gaussian / MLP，使地图能正确渲染当前观测。
```

如果只有 Expansion，没有 Update，地图只是粗糙地把深度点塞进 voxel 中，外观和几何不会精细。

如果只有 Update，没有 Expansion，地图无法覆盖新区域。

因此在 SLAM 过程中，每来一个关键帧，系统通常会先尝试扩展地图，再基于当前窗口做地图更新。

## 12. 一句话总结

```text
GigaSLAM 的地图扩展把新 keyframe 的深度转成分层 voxel / anchor，并用 spatial hashing 去重；
地图更新则把当前地图渲染到 keyframe 视角，通过 RGB / SSIM / 正则损失反向传播，优化 anchor、offset、feature、MLP 和部分相机参数。
```
