# Railway Rendering Anti-Aliasing 文献调研

本文服务于 scene16-only 建图质量实验。目标不是继续堆 LPIPS loss，而是借鉴高质量 3DGS 工作中处理高频纹理、尺度变化和视角相关外观的方法。

## 问题定义

铁路场景的难点集中在：

- 枕木、碎石、轨道、接触网是密集高频重复纹理；
- 近处纹理巨大、远处纹理被压缩，像素 footprint 变化强；
- 轻微几何错位会让 LPIPS 明显升高；
- 关键帧建图后评估全帧，非关键帧视角泛化要求更高；
- 普通 3DGS 容易在远距离产生 aliasing、闪烁、纹理不稳定或过度锐化。

## Mip-Splatting

Repo: `https://github.com/autonomousvision/mip-splatting`

核心启发：

- 原始 3DGS 在尺度变化下缺少对频率的约束；
- 3D smoothing filter 约束 Gaussian 的最高频率；
- 2D Mip filter 根据像素 footprint 做屏幕空间过滤；
- 目标是减少放大/缩小时的 aliasing，而不是靠 loss 把图像拟合得更锐。

对本项目的价值：

- 与铁路远近尺度变化高度匹配；
- 优先级最高；
- 适合直接移植 renderer/rasterizer 侧 filtering 机制。

风险：

- 当前项目使用带 pose/depth Jacobian 的自定义 rasterizer，直接替换可能破坏 SLAM 后端梯度；
- 需要小心保持 rendered depth、opacity、pose refinement 输出接口。

## Analytic-Splatting

Repo: `https://github.com/lzhnb/Analytic-Splatting`

核心启发：

- 将像素视为面积窗口进行解析积分，而不是只在像素中心点采样；
- 从积分角度减少 aliasing；
- 对细线、重复纹理、远处压缩纹理理论上更合理。

对本项目的价值：

- 如果 Mip-Splatting 接入成本高或效果不稳定，可以作为第二优先级；
- 对轨道线、接触网、枕木这种细结构可能更有效。

风险：

- rasterizer 替换成本可能高；
- 需要验证是否支持当前需要的 depth/opacity/pose gradient。

## Pixel-GS

Repo: `https://github.com/zhengzhang01/Pixel-GS`

核心启发：

- 传统 densification 只看梯度容易漏掉像素覆盖不足的区域；
- Pixel-aware gradient / pixel coverage 可以让新增 Gaussian 更关注图像细节区域；
- 对高频区域表达能力不足的问题更直接。

对本项目的价值：

- 适合枕木、碎石、轨道边缘、接触网等细节不足的区域；
- 可以结合 RailScale 的轨道区域先验，优先增强 rail/ballast ROI。

风险：

- 可能增加 Gaussian 数量和显存；
- 如果 geometry/pose 本身有偏，增加细节可能放大错位。

## Scaffold-GS

Repo: `https://github.com/city-super/Scaffold-GS`

核心启发：

- 使用 anchor + learned offsets 组织 Gaussian；
- 引入 view-adaptive appearance；
- 对大场景和视角变化更友好。

对本项目的价值：

- 可作为 view-dependent appearance 的更系统方案；
- 适合处理反光、车窗、天气/曝光变化。

风险：

- 与当前后端结构差异大；
- 不是 anti-aliasing 的第一优先解法。

## 结论

下一步优先级：

1. Mip-Splatting：最匹配铁路高频纹理与尺度变化 aliasing。
2. Analytic-Splatting：作为更激进的 rasterizer 替换候选。
3. Pixel-GS：用于补足高频细节和 densification。
4. Scaffold-GS：后续针对 view-adaptive appearance，再考虑。

不再优先尝试单纯 LPIPS loss、反复调 color refinement 或小幅 scale modifier。
