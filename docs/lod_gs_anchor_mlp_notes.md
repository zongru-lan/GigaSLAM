# LoD GS 中 Voxel、Anchor、Offset 和 MLP 的理解

这份笔记整理 GigaSLAM / Scaffold-GS 风格的 LoD Gaussian Splatting 表示，帮助理解体素、anchor、`n_offsets` 个 Gaussian 以及 MLP 解码之间的关系。

## 1. LoD 分层和体素

LoD 的核心是按相机距离使用不同大小的体素。

例如：

```yaml
voxel_size_lis: [0.1, 0.3, 1.0, 4.0, 15.0]
distance_lis: [10.0, 30.0, 70.0, 150.0]
```

可以理解为：

```text
0-10m       voxel size = 0.1m
10-30m      voxel size = 0.3m
30-70m      voxel size = 1.0m
70-150m     voxel size = 4.0m
150m以上    voxel size = 15.0m
```

近处体素小，anchor 更密，细节更多；远处体素大，anchor 更稀疏，显存和计算更低。

## 2. Voxel 和 Anchor 的关系

一个 anchor 可以理解为一个非空体素的代表点。

流程大致是：

```text
3D 点云 -> 按 voxel size 体素化 -> 每个非空体素生成一个 anchor
```

因此：

```text
一个非空体素 ≈ 一个 anchor
```

anchor 不一定是严格几何中心，但可以直观理解为该体素的中心或代表坐标。

## 3. 一个 Anchor 如何展开成多个 Gaussian

一个 anchor 表征一个局部区域的基础状态。每个 anchor 内部有 `n_offsets` 个局部偏移，用来放置多个 Gaussian。

关系可以写成：

```text
一个 anchor -> n_offsets 个 Gaussian
```

Gaussian 的中心位置主要由 anchor 位置和 offset 决定：

```text
Gaussian center = anchor position + offset * anchor_scale
```

其中：

```text
anchor position   局部区域的基准位置
anchor_scale      局部区域的尺度
offset[j]         第 j 个 Gaussian 相对 anchor 的偏移
```

所以 `n_offsets` 的作用是让一个 anchor 能在局部区域内放置多个 Gaussian，从而细化该区域的几何和外观。

## 4. Anchor 中保存哪些基础量

在这个项目中，anchor 相关的基础量主要包括：

```text
_anchor        anchor 位置
_offset        每个 anchor 的 n_offsets 个局部偏移
_anchor_feat   anchor 的可学习特征向量
_scaling       anchor 的基础尺度
_rotation      anchor 的基础旋转
_opacity       anchor 的基础 opacity
_level         anchor 所属的 LoD 层级
```

其中 `anchor_feat` 是 feature 的缩写，可以理解为每个 anchor 自己携带的一段可学习 latent code。它不是人工指定的颜色或语义，而是一串会被优化的数字，用来描述该局部区域。

## 5. MLP 的输入和输出

MLP 的输入可以简单理解为：

```text
anchor_feat + 视角方向 + 距离 + level
```

其中：

```text
anchor_feat   该局部区域的可学习特征
视角方向       相机从哪个方向看这个 anchor
距离           相机到 anchor 的连续距离
level          anchor 所属的离散 LoD 层级
```

这里需要区分“保存在 anchor 中的量”和“渲染时动态计算的量”。

`level` 会保存在 anchor 中，对应代码里的 `_level`。它表示该 anchor 属于哪一个 LoD 层级。

但距离不保存在 anchor 中。距离是在每次渲染时，根据当前相机中心和 anchor 位置动态计算出来的：

```text
distance = norm(anchor_position - camera_center)
```

同理，视角方向也不保存在 anchor 中，而是在渲染时动态计算：

```text
view_direction = normalize(anchor_position - camera_center)
```

原因是相机每一帧都在运动，同一个 anchor 对不同相机的距离和视角方向都会变化。因此：

```text
_level 保存在 anchor 中
distance 不保存在 anchor 中
view_direction 不保存在 anchor 中
```

另外，color MLP 还可能额外输入 `camera appearance`。它不是相机位姿或内参，而是每个相机/帧对应的一段可学习外观编码，用来描述曝光、光照、白平衡、颜色偏差等图像风格差异。

可以理解为：

```text
camera appearance = 当前帧/当前相机的可学习外观 latent code
```

它主要用于颜色预测。例如同一个局部区域在不同帧中可能因为曝光或光照变化呈现不同颜色，color MLP 可以结合 camera appearance 预测更符合当前图像的 RGB。

因此 color MLP 的输入可以更完整地理解为：

```text
anchor_feat + 视角方向 + 距离 + level + camera appearance
```

其中 camera appearance 主要影响外观颜色，不直接表示几何。

注意：距离项是否输入取决于配置开关，例如 `add_opacity_dist`、`add_color_dist`、`add_cov_dist`。

MLP 不是针对单个 Gaussian 单独运行，而是对一个 anchor 运行一次，输出该 anchor 下所有 `n_offsets` 个 Gaussian 的参数。

该项目里主要有三个 MLP：

```text
mlp_opacity -> 输出 n_offsets 个 opacity
mlp_color   -> 输出 3 * n_offsets 个 RGB
mlp_cov     -> 输出 7 * n_offsets 个几何参数，通常对应 scale 和 rotation
```

因此，对于第 `j` 个 Gaussian：

```text
center  = anchor + offset[j] * anchor_scale
color   = mlp_color 输出中的第 j 组 RGB
opacity = mlp_opacity 输出中的第 j 个 opacity
scale   = mlp_cov 输出中第 j 组的 scale 部分
rot     = mlp_cov 输出中第 j 组的 rotation 部分
```

也就是说，Gaussian 的中心位置不是 MLP 直接输出的，而是由 anchor 和 offset 计算得到；MLP 主要负责预测颜色、透明度、尺度和旋转等属性。

## 6. MLP 为什么能预测外观和几何

MLP 本质上是一个简单的感知机，例如：

```text
Linear -> ReLU -> Linear
```

它之所以能预测每个 Gaussian 的外观和几何，是因为训练时会通过渲染损失反向传播优化：

```text
anchor_feat
n_offsets 个 offset
anchor_scale
MLP 参数
```

训练过程可以理解为：

```text
给定某个可见 anchor 的 anchor_feat、视角方向、距离和 level，
MLP 解码出该 anchor 下 n_offsets 个 Gaussian 的颜色、透明度、尺度和旋转；
这些 Gaussian 被渲染成图像；
渲染图像与真实图像计算 photometric loss；
loss 反向传播，优化 anchor_feat、offset、scale 和 MLP 参数；
最终使渲染图像逐渐接近真实图像。
```

所以 MLP 不是凭空计算出高斯参数，而是作为一个共享解码器，把每个 anchor 的局部特征和观察条件解码成一组能正确渲染该局部区域的 Gaussian 属性。

## 7. LoD GS 为什么能减少显存开销

LoD GS 能减少显存开销，核心原因是：远处使用更大的体素，从源头减少 anchor 数量；anchor 数量减少后，由 anchor 展开的 Gaussian 数量也会减少。

在这个项目里，可以近似理解为：

```text
体素数量 ≈ anchor 数量
Gaussian 数量 ≈ anchor 数量 × n_offsets
```

如果没有 LoD，例如：

```yaml
voxel_size_lis: [0.1]
distance_lis: []
```

这表示不管近处还是远处，全部使用 0.1m 的小体素。远处的大面积场景也会被切成大量小体素，于是会产生大量 anchor。每个 anchor 又会展开 `n_offsets` 个 Gaussian，渲染时 MLP 还要为这些可见 anchor 输出 color、opacity、scale 和 rotation 等中间结果，因此显存开销很高。

有 LoD 后，例如：

```yaml
voxel_size_lis: [0.1, 0.3, 1.0, 4.0, 15.0]
distance_lis: [10.0, 30.0, 70.0, 150.0]
```

远处会使用更大的体素：

```text
0-10m       0.1m 体素
10-30m      0.3m 体素
30-70m      1.0m 体素
70-150m     4.0m 体素
150m以上    15.0m 体素
```

这样远处很多点会被合并到更少的体素或 anchor 中：

```text
远处 anchor 数量下降
可见 anchor 数量下降
展开的 Gaussian 数量下降
MLP 中间输出张量下降
rasterization 显存下降
```

因此，LoD GS 的省显存机制不是单纯压缩单个 Gaussian 的参数，而是通过“远处粗、近处细”的分层体素表示，减少需要存储、解码和渲染的 anchor/Gaussian 数量。

一句话理解：

```text
LoD GS 通过远处粗体素减少 anchor 密度，从源头减少需要存储和渲染的 Gaussian 数量，因此降低显存开销。
```

## 7. LoD GS 中的 Collision Issue

论文中提到 LoD GS 可以缓解 potential “collision” issues。这里的 collision 不是物理碰撞，而是地图表示中的 Gaussian 重叠、错位、重复或鬼影问题。

在长序列 SLAM 中，前面某些视角会看到远处物体，例如远处轨道、建筑、杆子或天空附近的结构。由于远处视差小、深度不确定、位姿也可能有误差，如果远处仍然使用很细的体素，例如 0.1m，就会生成大量细粒度 Gaussian。

当相机继续前进，后续帧再次看到同一区域时，由于深度和位姿估计误差，后续生成的 Gaussian 可能和前面生成的 Gaussian 位置不完全一致。于是地图中可能出现：

```text
旧视角生成的 Gaussian
和
新视角生成的 Gaussian
在空间中重叠、错位或重复
```

这就是论文所说的 potential collision issue。

这种现象会带来两个影响：

```text
1. 地图中出现重复或错位的 Gaussian，渲染结果可能模糊、重影或遮挡错误。
2. tracking 依赖渲染图像与真实图像对齐，错误的渲染会反过来影响相机位姿估计。
```

LoD GS 的缓解方式是：

```text
近处使用小体素和高密度 anchor，保留细节；
远处使用大体素和低密度 anchor，降低不确定区域的细碎 Gaussian 数量。
```

这样，远处大量不可靠的细粒度点会被合并到更粗的体素或 anchor 中，减少长序列中重复生成、错位重叠的 Gaussian，从而降低鬼影和 tracking 干扰。

一句话理解：

```text
LoD GS 用“近处细、远处粗”的地图表示，避免远处不可靠的细粒度 Gaussian 在长序列中不断重复生成并互相重叠。
```

## 7. 一句话总结

```text
LoD GS = 用不同尺度的体素控制 anchor 密度，
再用每个 anchor 的 n_offsets 个 Gaussian 表达局部细节，
其中 Gaussian 的位置由 anchor 和 offset 决定，
外观和部分几何属性由 MLP 根据 anchor 特征和观察条件预测。
```
