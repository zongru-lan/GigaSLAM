# DGC 与铁路场景效果提升文献调研

更新时间：2026-05-29

本文用于判断当前 DGC（Depth-Gaussian Consistency）实验是否能帮助提升铁路单目 SLAM 的轨迹和建图效果，以及高质量论文里哪些机制可以被我们借鉴。本文采用 **effect-first** 口径：别人是否做过不是是否尝试的决定因素；已有工作覆盖程度只影响论文表述和模块包装，不影响工程验证。结论先放前面：**naive DGC，即把 monocular depth 和 Gaussian rendered depth 做全图一致性约束，当前 scene16 收益很弱；更值得优先尝试的是 rail-aware / confidence-aware / selective DGC，或者把 rendered depth residual 作为诊断与门控信号。**

## 使用原则

- 目标是提升效果，不是为了回避已有工作。
- 高质量论文里的成熟机制可以直接借鉴，先看是否能改善 ATE、end_error、along_p95、lat_p95、PSNR、SSIM、LPIPS。
- “已有工作覆盖程度”只用于判断：这个机制是否成熟、应该怎样复现、论文里如何避免夸大表述。
- RailScale 仍是当前最稳定、最可解释的主线；DGC 是从文献中借来的几何一致性增强/诊断工具。

## 当前项目观察

当前 DGC 实验口径是：在 mapping loss 中加入 `render_pkg["depth"]` 与 RailScale-corrected monocular depth 的 log-depth residual，并允许每个 keyframe 有一个很小的 `depth_log_scale_delta` adapter。

scene16 的 full-render 对比显示，DGC 不是强信号：

| 指标 | full_mainline scene16 | dgc_bold scene16 | 变化 |
|---|---:|---:|---:|
| ATE RMSE | 2.7772 m | 2.7637 m | -0.0135 m |
| end_error | 3.3340 m | 3.3211 m | -0.0128 m |
| along_p95 | 4.9334 m | 4.9136 m | -0.0198 m |
| lat_p95 | 0.7408 m | 0.7325 m | -0.0083 m |
| length_ratio | 0.9543 | 0.9545 | +0.0002 |

这个结果不能证明 DGC 完全无效，但足以说明：**如果没有可靠区域选择、尺度置信度和结构先验，DGC 对轨迹形态的改善很小，且可能损害渲染指标。**

## 论文脉络总览

相关工作大致分成四条线：

1. **Monocular 3DGS SLAM / joint pose-depth-map optimization**  
   这条线和我们最接近，核心问题是单目输入下如何同时优化 pose、depth 和 Gaussian map。这里的 novelty risk 最大。

2. **Depth-supervised / depth-regularized 3DGS**  
   这条线通常不是 SLAM，而是 sparse-view NVS 或 reconstruction，但它们大量讨论 monocular depth supervision 的尺度、噪声、置信度和归一化问题，对我们改 DGC 很有参考价值。这里的重点不是“新不新”，而是哪些技巧能稳定提升几何。

3. **Depth/edge/shape-aware GS geometry constraints**  
   这条线提示 depth loss 不能只看中心深度，还要考虑边缘、局部结构、Gaussian shape 和表面几何。

4. **DepthSplat / feed-forward depth-GS coupling**  
   这条线说明 depth 和 GS 可以互相促进，但多是 feed-forward NVS / reconstruction，不是我们的在线铁路 SLAM。它适合作为背景，不适合作为直接工程主线。

## 逐篇调研记录

### 1. GigaSLAM: Large-Scale Monocular SLAM with Hierarchical Gaussian Splats

- 来源：[arXiv:2503.08071](https://arxiv.org/abs/2503.08071)
- 任务：大规模、无界户外 monocular RGB SLAM。
- 输入：单目 RGB。
- 深度来源：metric depth model。
- scale 处理：依赖 metric depth 和前端几何估计，本项目进一步引入 RailScale 做铁路物理尺度修正。
- depth/render loss：原始框架强调 hierarchical Gaussian splats 的 scalable mapping 和 rendering。
- 是否优化 pose：是，前端使用 metric depth、epipolar geometry 和 PnP。
- 与本项目关系：这是项目基础框架。我们的 RailScale 和 DGC 都是在这个基础上扩展。
- 可借鉴：保持前端 tracking 与后端 mapping 解耦；大规模 outdoor 场景需要关注轨迹尺度，不只看渲染。
- novelty risk：RailScale 的铁路物理先验不在原 GigaSLAM 中，是我们更强的差异点；DGC 如果只做通用 depth-render loss，差异性不够。

### 2. Gaussian Splatting SLAM

- 来源：[arXiv:2312.06741](https://arxiv.org/abs/2312.06741)
- 任务：将 3D Gaussian Splatting 用于 monocular / RGB-D SLAM。
- 输入：monocular 或 RGB-D。
- 深度来源：monocular 设置下主要依赖直接优化和几何正则，RGB-D 设置可使用传感器深度。
- depth/render loss：通过对 3D Gaussians 的 differentiable rendering 做 tracking/mapping。
- 是否优化 pose：是，提出对 Gaussians 的直接相机 tracking。
- 与本项目关系：说明 3DGS 可作为 SLAM map representation，但单目下几何歧义明显。
- 可借鉴：rendered map 可以参与 tracking；但要防止 Gaussian map 与错误 pose 互相强化。
- novelty risk：render-to-track / render-to-map 本身不是新点。

### 3. GS-SLAM: Dense Visual SLAM with 3D Gaussian Splatting

- 来源：[CVF OpenAccess, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Yan_GS-SLAM_Dense_Visual_SLAM_with_3D_Gaussian_Splatting_CVPR_2024_paper.html)
- 任务：RGB-D dense visual SLAM with 3DGS。
- 输入：RGB-D。
- 深度来源：传感器深度。
- depth/render loss：实时 differentiable splatting rendering，用于 map optimization 和 RGB-D rendering。
- 是否优化 pose：是，tracking 中选择可靠 Gaussian 表示做 coarse-to-fine pose optimization。
- 与本项目关系：强调可靠 Gaussian selection 对 tracking 很重要。
- 可借鉴：不要全图/全 Gaussian 不加区分地进入约束；可靠性选择本身就是 SLAM 稳定性的关键。
- novelty risk：RGB-D 场景的 depth consistency 已经成熟；我们的差异必须来自单目铁路尺度与 rail-aware gating。

### 4. SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM

- 来源：[CVF OpenAccess, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Keetha_SplaTAM_Splat_Track__Map_3D_Gaussians_for_Dense_RGB-D_CVPR_2024_paper.pdf)
- 任务：RGB-D SLAM 中用 3DGS 同时 tracking 和 mapping。
- 输入：RGB-D。
- 深度来源：传感器深度。
- depth/render loss：RGB-D rendering 与 silhouette mask 辅助 map expansion。
- 是否优化 pose：是。
- 与本项目关系：RGB-D 给出的深度比单目深度可靠得多，不能直接照搬。
- 可借鉴：mask / silhouette / mapped-region 判断可以避免把未充分观测区域硬塞进 loss。
- novelty risk：如果 DGC 只是模仿 RGB-D depth loss，论文上很弱；铁路场景需要说明“单目深度为什么可信、哪里可信”。

### 5. Splat-SLAM: Globally Optimized RGB-only SLAM with 3D Gaussians

- 来源：[CVPR 2025 Workshop page](https://cvpr.thecvf.com/virtual/2025/35758)，[CVF PDF](https://openaccess.thecvf.com/content/CVPR2025W/VOCVALC/papers/Sandstrom_Splat-SLAM_Globally_Optimized_RGB-only_SLAM_with_3D_Gaussians_CVPRW_2025_paper.pdf)
- 任务：RGB-only dense SLAM with 3D Gaussians。
- 输入：RGB-only。
- 深度来源：dense optical flow、multi-view depth、monocular depth prior 组合成 proxy depth。
- scale 处理：DSPO（Disparity, Scale and Pose Optimization）联合优化 pose、depth、monocular depth scale。
- depth/render loss：mapping 使用 proxy depth 与 keyframe pose 驱动 Gaussian map。
- 是否优化 pose：是，而且强调全局一致性。
- 与本项目关系：这是 DGC 最大的 novelty 风险来源。它已经明确做了 pose-depth-scale 联合优化。
- 可借鉴：不要让 monocular prior 全像素强监督。论文中区分高误差/低误差深度区域，低误差区域固定以稳定 scale，高误差区域再用 monocular prior。
- 对我们的启发：如果继续 DGC，应从“全图约束”转为“只约束需要帮助的区域”，并将 RailScale confidence 纳入选择。
- novelty risk：高。当前 `depth_log_scale_delta + rendered/input depth residual` 很容易被认为是 DSPO/UDGS 类思想的简化版本。

### 6. UDGS-SLAM: UniDepth Assisted Gaussian Splatting for Monocular SLAM

- 来源：[arXiv:2409.00362](https://arxiv.org/abs/2409.00362)，[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S259000562500027X)
- 任务：UniDepth 辅助的 monocular Gaussian Splatting SLAM。
- 输入：monocular RGB。
- 深度来源：UniDepth。
- scale 处理：利用 UniDepth 的 metric depth，并用 statistical filtering 做 local consistency。
- depth/render loss：将 photometric error 和 geometric error 组合，联合优化 camera pose 与 Gaussian map。
- 是否优化 pose：是。
- 与本项目关系：和我们“UniDepth/RailScale depth + Gaussian rendered depth + pose/map optimization”的重合度很高。
- 可借鉴：local consistency filtering 比盲目全图 depth loss 更稳；filter 的价值在于减少单目深度局部异常。
- 对我们的启发：我们不应宣传“UniDepth + Gaussian depth loss”本身，而应强调 RailScale 提供了铁路物理尺度和轨道区域可靠性。
- novelty risk：很高。

### 7. MonoGS++: Fast and Accurate Monocular RGB Gaussian SLAM

- 来源：[arXiv:2504.02437](https://arxiv.org/abs/2504.02437)
- 任务：monocular RGB Gaussian SLAM。
- 输入：RGB-only。
- 深度来源：在线 VO 生成 sparse point clouds，不依赖 depth sensor。
- depth/render loss：重点在 Gaussian insertion、densification、planar regularization，而不是直接 monocular depth-to-rendered-depth loss。
- 是否优化 pose：是。
- 与本项目关系：说明 monocular GS-SLAM 也可以走“VO sparse geometry + map regularization”的路线。
- 可借鉴：减少冗余 Gaussian、平面/低纹理区域正则，可能比全图 DGC 更能改善建图。
- novelty risk：中。它不覆盖铁路尺度先验，但覆盖 monocular GS-SLAM 主体。

### 8. Depth-Regularized Optimization for 3D Gaussian Splatting in Few-Shot Images

- 来源：[arXiv:2311.13398](https://arxiv.org/abs/2311.13398)
- 任务：few-shot images 下的 3DGS 优化。
- 输入：少量 posed images。
- 深度来源：预训练 monocular depth。
- scale 处理：用 sparse COLMAP feature points 对 monocular depth 做 scale 和 offset alignment。
- depth/render loss：dense depth map 作为 geometry guide，缓解 overfitting 和 floating artifacts。
- 是否优化 pose：通常使用已知/估计相机位姿，不是在线 SLAM 重点。
- 与本项目关系：说明 monocular depth regularization 是常规做法，且必须先做尺度/偏移对齐。
- 可借鉴：DGC 不应默认相信单目绝对深度；RailScale 正是我们的 domain-specific scale alignment。
- novelty risk：中到高。深度正则本身不新，铁路物理尺度对齐才是可写点。

### 9. DNGaussian: Global-Local Depth Normalization

- 来源：[CVF OpenAccess, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Li_DNGaussian_Optimizing_Sparse-View_3D_Gaussian_Radiance_Fields_with_Global-Local_Depth_CVPR_2024_paper.html)
- 任务：sparse-view 3DGS NVS。
- 输入：sparse views。
- 深度来源：coarse monocular depth。
- scale 处理：Global-Local Depth Normalization，避免粗糙单目深度直接绝对监督带来的误差。
- depth/render loss：Hard and Soft Depth Regularization。
- 是否优化 pose：一般不是 SLAM pose optimization 主线。
- 与本项目关系：直接解释了为什么 naive absolute DGC 可能弱：单目深度的全局尺度和局部细节都可能不可靠。
- 可借鉴：E-DGC2 可以尝试 global-local normalized DGC，而不是直接米制 depth residual。
- novelty risk：中。global-local depth normalization 已有，铁路上可以把 local window 设计成 rail-row / rail-neighborhood。

### 10. CDGS: Confidence-Aware Depth Regularization for 3D Gaussian Splatting

- 来源：[arXiv:2502.14684](https://arxiv.org/abs/2502.14684)
- 任务：提高 3DGS 的几何重建精度。
- 输入：multi-view RGB + sparse SfM depth + monocular depth。
- 深度来源：monocular depth 与 SfM depth。
- confidence/gating：使用 monocular depth 多线索置信图和 sparse SfM depth，动态调节 depth supervision。
- depth/render loss：confidence-aware depth regularization。
- 是否优化 pose：不是 SLAM pose 主任务。
- 与本项目关系：这是 confidence-aware DGC 的直接参考。
- 可借鉴：RailScale 的 row consistency、width stability、scale reject/hold 状态可以变成 railway-specific confidence map。
- novelty risk：中。confidence-aware depth regularization 已有，但 railway-specific confidence 没有直接覆盖。

### 11. In Depth We Trust: Reliable Monocular Depth Supervision for Gaussian Splatting

- 来源：[arXiv:2604.05715](https://arxiv.org/abs/2604.05715)
- 任务：可靠地使用 noisy / scale-ambiguous monocular depth supervision。
- 输入：GS training images + monocular depth priors。
- 深度来源：foundation monocular depth models。
- confidence/gating：强调选择 ill-posed geometry 做 selective depth regularization，避免把 depth inaccuracies 传播到已经重建好的结构。
- depth/render loss：selective monocular depth regularization。
- 是否优化 pose：不是在线 SLAM 重点。
- 与本项目关系：直接支持我们放弃 naive 全图 DGC，转向 selective DGC。
- 可借鉴：E-DGC1 / E-DGC3 的设计依据。DGC 应该只在几何弱约束、低纹理或 RailScale 高置信区域发挥作用。
- novelty risk：中。selective depth supervision 已有，但 rail-aware selection 和 trajectory drift diagnosis 仍有空间。

### 12. DET-GS: Depth- and Edge-Aware Regularization

- 来源：[arXiv:2508.04099](https://arxiv.org/abs/2508.04099)
- 任务：sparse-view 3DGS 中提升结构保真和渲染质量。
- 输入：multi-view RGB + monocular depth prior。
- depth/render loss：hierarchical geometric depth supervision、edge-aware depth regularization、RGB-guided edge-preserving TV。
- confidence/gating：通过边缘/语义边界避免非局部平滑破坏结构。
- 是否优化 pose：不是 SLAM pose 主任务。
- 与本项目关系：铁路图像里轨道边缘、轨枕、道岔结构很强，边缘感知比全图平滑更合理。
- 可借鉴：DGC 的 mask 可以避开 rail-edge 附近的错误深度平滑，也可以只在轨道内部/轨面附近做一致性。
- novelty risk：中。edge-aware 已有，但 railway rail-edge aware 可以结合 RailScale。

### 13. SAD-GS: Shape-aligned Depth-supervised Gaussian Splatting

- 来源：[CVF OpenAccess, CVPRW 2024](https://openaccess.thecvf.com/content/CVPR2024W/NRI/html/Kung_SAD-GS_Shape-aligned_Depth-supervised_Gaussian_Splatting_CVPRW_2024_paper.html)
- 任务：depth-supervised GS 中提高 3D geometry 和 mesh accuracy。
- 输入：RGB + depth supervision。
- 深度来源：深度监督。
- depth/render loss：shape-aligned loss，不只约束 Gaussian center，还约束 Gaussian shape。
- 是否优化 pose：不是 SLAM pose 主任务。
- 与本项目关系：说明只让 rendered depth 对齐 input depth 不一定能保证 Gaussian shape 正确。
- 可借鉴：如果我们后续关心建图质量，可考虑轨面附近 Gaussian 的形状/法向/平面一致性，而不是只看 depth residual。
- novelty risk：中。shape-aligned GS 已有，铁路轨面/钢轨结构约束可作为差异化。

### 14. DepthSplat: Connecting Gaussian Splatting and Depth

- 来源：[CVF OpenAccess, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Xu_DepthSplat_Connecting_Gaussian_Splatting_and_Depth_CVPR_2025_paper.html)
- 任务：连接 Gaussian splatting 和 depth estimation，做 feed-forward multi-view 3DGS/NVS。
- 输入：多视角图像。
- 深度来源：利用预训练 monocular depth features 构建 robust multi-view depth model。
- depth/render coupling：展示 GS 目标可以作为无监督预训练信号，depth 和 GS 可以互相促进。
- 是否优化 pose：不是在线 SLAM 主任务。
- 与本项目关系：提供理论背景：depth 与 GS 互相促进是合理方向，但它不是我们的工程路线。
- 可借鉴：可以在论文动机里引用“depth-GS synergy”，但要明确我们是 online railway SLAM，不是 feed-forward NVS。
- novelty risk：低到中。它不直接覆盖 RailScale/DGC，但覆盖“depth 与 GS 互相促进”的宏观叙述。

### 15. MonoSplat: Generalizable 3D Gaussian Splatting from Monocular Depth Foundation Models

- 来源：[CVF OpenAccess, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_MonoSplat_Generalizable_3D_Gaussian_Splatting_from_Monocular_Depth_Foundation_Models_CVPR_2025_paper.html)
- 任务：从 monocular depth foundation model 中提取先验，用于 generalizable 3DGS。
- 输入：单目/多视角设置下的图像特征。
- 深度来源：depth foundation model features。
- depth/render coupling：Mono-Multi Feature Adapter 和 Integrated Gaussian Prediction。
- 是否优化 pose：不是在线 SLAM 主任务。
- 与本项目关系：说明 foundation depth 的“特征/置信度”可能比单张 depth map 数值更有用。
- 可借鉴：未来如果使用 UniDepth confidence 或 feature，不应只拿最终 depth 值；但这会扩大工程复杂度。
- novelty risk：低到中。它是 feed-forward/generalizable reconstruction，不是铁路在线 SLAM。

## Innovation Questions 回答

### Q1: naive DGC 是否值得继续作为效果路线？

**不建议继续盲目调 naive DGC；但可以把它作为 E-DGC0 基线跑全 6 序列，确认是否有稳定收益。**

证据：

- UDGS-SLAM 已经把 UniDepth、local consistency filtering、rendered RGB/depth loss、pose + Gaussian map 联合优化放在 monocular SLAM 框架里。
- Splat-SLAM 已经做了 DSPO，联合优化 pose、depth 和 monocular depth scale，并强调不是所有 pixel 都应被 monocular prior 一视同仁地监督。
- FSGS / DNGaussian / CDGS / In Depth We Trust 都说明 monocular depth regularization 已经是 3DGS 的常用工具，关键差别在 scale alignment、normalization、confidence 和 selective supervision。

因此，工程上不该继续只调 `lambda_depth`。更合理的路线是借鉴这些论文里更成熟的筛选、归一化和置信度机制。

### Q2: 哪些机制最可能提升我们项目效果？

更有希望的机制不是“全图 DGC”，而是把 DGC 做得更稳：

1. **Rail-aware metric scale prior**  
   普通 monocular GS/SLAM 工作没有铁路轨距这种强物理先验。RailScale 将轨道检测、行级一致性和已知 rail-head width 转成 metric depth scale，这是最可解释的创新。

2. **Rail-aware confidence / selective DGC**  
   如果继续 DGC，应只让高置信 railway geometry 参与，例如 ego rail pair 附近、row consistency 稳定、宽度合理、scale 没有被 reject/hold 的区域。

3. **Scale-drift diagnostic / gating**  
   对铁路前视序列，主要误差常是 along-track drift 和末端误差，不一定表现为横向偏轨。rendered depth residual 可作为异常帧解释信号，用于定位 RailScale、VO 或 mapping 的失配。

4. **Railway-specific evaluation**  
   不能只报告 ATE。应同时报告 `end_error`、`along_p95`、`lat_p95`、`length_ratio`、渲染指标和 depth consistency 指标。

### Q3: 是否应该保留 DGC 作为主线？

当前不建议。

建议定位：

- **主线**：RailScale / rowtrack350，强调铁路物理尺度先验、轨道候选筛选、row consistency 和 hold-last-scale 的稳定性。
- **DGC naive**：保留为 exploratory baseline，用来判断直接 depth-render loss 的上限。
- **DGC next**：如果继续，优先做 rail-aware 或 confidence-aware selective DGC，因为这些方向更可能提升效果。

## 可借鉴 / 不建议借鉴 / 需要实验验证

### 可借鉴

- 从 Splat-SLAM 借鉴：不要全像素相信 monocular prior；只在高误差或不稳定区域使用深度先验。
- 从 UDGS-SLAM 借鉴：local consistency filtering 是必要步骤；深度进入 SLAM 前需要过滤。
- 从 DNGaussian 借鉴：不要只做绝对米制 residual；考虑 global-local normalization。
- 从 CDGS 借鉴：confidence map 应调节 depth supervision 权重。
- 从 In Depth We Trust 借鉴：只在 ill-posed geometry 上做 selective regularization，避免污染已重建好的区域。
- 从 DET-GS 借鉴：边缘/结构区域不能被普通 depth smoothing 破坏。
- 从 SAD-GS 借鉴：建图质量不仅是 depth center，对 Gaussian shape/表面几何也要关注。

### 不建议借鉴

- 不建议直接引入 DepthSplat/MonoSplat 的 feed-forward 网络路线，工程复杂度高，且与当前在线 SLAM 主线不一致。
- 不建议把 DGC 当成新的主损失反复调 `lambda_depth`。这会变成无解释调参，很难写进论文。
- 不建议只用 rendered depth residual 判断轨迹好坏。scene16 已经显示它不是强稳定信号。

### 需要实验验证

- RailScale confidence map 是否能预测 DGC 有效区域。
- DGC residual 是否与 `along_p95`、`end_error` 的异常帧有稳定相关性。
- global-local normalized DGC 是否比 absolute log-depth DGC 更稳。
- rail-aware selective DGC 是否在不破坏 scene13 的前提下改善 scene14/17/19。

## 建议实验路线

### E-DGC0: naive DGC 全 6 序列复现实验

目的：把当前 DGC 作为完整 baseline/negative experiment。

要求：

- 全 6 scene 跑完。
- 同时报告轨迹、形态、渲染、DGC valid ratio、learned depth scale saturation。
- 不调单序列参数。

决策：

- 如果全 6 序列无稳定收益，停止 naive DGC。
- 如果只有某一两个 scene 微小提升，不作为主线。

### E-DGC1: confidence / opacity / RailScale gated DGC

目的：只在可靠区域做 DGC。

候选 mask：

- renderer opacity 高；
- input depth 有效；
- RailScale 状态为 `corrected`，而不是 reject/hold；
- row consistency 通过；
- rail width 在合理范围；
- PnP inlier ratio 不是极低。

论文价值：

- 从“通用 DGC”变成“rail-aware confidence-guided consistency”。
- 可解释：轨道物理先验决定哪里能信深度。

### E-DGC2: global-local normalized DGC

目的：减少绝对深度尺度冲突。

做法：

- 不直接比较 `D_render` 和 `D_input` 的米制值。
- 在局部 row/window 或 rail-neighborhood 内做 normalized depth residual。
- 可参考 DNGaussian 的 global-local normalization，但把 local region 改成铁路结构区域。

论文价值：

- 解释单目深度局部形态比绝对值更可靠。
- 避免 DGC 和 RailScale 的 metric prior 打架。

### E-DGC3: diagnostic-only DGC

目的：DGC 不进 loss，只输出 residual 与异常帧解释。

用法：

- 生成每帧 rendered/input depth residual 曲线。
- 和 RailScale 状态、VO step、PnP inlier ratio、along/lateral error 对齐。
- 用于论文中的 failure analysis / mechanism analysis。

论文价值：

- 即使 DGC 不能改善指标，也能作为可解释诊断工具。
- 风险最低，不会污染主线结果。

## 最终建议

短期建议：

1. 先完成 `E-DGC0` 全 6 序列，只为判断 naive DGC 是否有稳定价值。
2. 如果没有稳定收益，停止 naive DGC，不再调 `lambda_depth`。
3. 优先推进 `E-DGC3 diagnostic-only`，把 rendered depth residual 做成论文里的误差机制分析。
4. 只有当 residual 与 along/end error 有稳定关系时，再做 `E-DGC1 rail-aware gated DGC`。

论文写法建议：

- RailScale 是主创新：铁路物理结构提供 metric scale prior。
- DGC 是补充探索：通用 depth-render consistency 已有大量成熟做法，我们可以借鉴其有效机制，但不应把全图 naive loss 当作最终方案。
- 如果 rail-aware selective DGC 成功，可以作为 RailScale-v2 或附加模块；如果失败，也可以诚实写成 negative observation：在铁路单目前视 SLAM 中，闭环式 rendered-depth supervision 容易继承单目深度与 map 的共同偏差，外部物理尺度先验更可靠。

## 参考链接

- [GigaSLAM: Large-Scale Monocular SLAM with Hierarchical Gaussian Splats](https://arxiv.org/abs/2503.08071)
- [Gaussian Splatting SLAM](https://arxiv.org/abs/2312.06741)
- [GS-SLAM: Dense Visual SLAM with 3D Gaussian Splatting](https://openaccess.thecvf.com/content/CVPR2024/html/Yan_GS-SLAM_Dense_Visual_SLAM_with_3D_Gaussian_Splatting_CVPR_2024_paper.html)
- [SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM](https://openaccess.thecvf.com/content/CVPR2024/papers/Keetha_SplaTAM_Splat_Track__Map_3D_Gaussians_for_Dense_RGB-D_CVPR_2024_paper.pdf)
- [Splat-SLAM: Globally Optimized RGB-only SLAM with 3D Gaussians](https://openaccess.thecvf.com/content/CVPR2025W/VOCVALC/papers/Sandstrom_Splat-SLAM_Globally_Optimized_RGB-only_SLAM_with_3D_Gaussians_CVPRW_2025_paper.pdf)
- [UDGS-SLAM: UniDepth Assisted Gaussian Splatting for Monocular SLAM](https://arxiv.org/abs/2409.00362)
- [MonoGS++: Fast and Accurate Monocular RGB Gaussian SLAM](https://arxiv.org/abs/2504.02437)
- [Depth-Regularized Optimization for 3D Gaussian Splatting in Few-Shot Images](https://arxiv.org/abs/2311.13398)
- [DNGaussian: Optimizing Sparse-View 3D Gaussian Radiance Fields with Global-Local Depth Normalization](https://openaccess.thecvf.com/content/CVPR2024/html/Li_DNGaussian_Optimizing_Sparse-View_3D_Gaussian_Radiance_Fields_with_Global-Local_Depth_CVPR_2024_paper.html)
- [CDGS: Confidence-Aware Depth Regularization for 3D Gaussian Splatting](https://arxiv.org/abs/2502.14684)
- [In Depth We Trust: Reliable Monocular Depth Supervision for Gaussian Splatting](https://arxiv.org/abs/2604.05715)
- [DET-GS: Depth- and Edge-Aware Regularization for High-Fidelity 3D Gaussian Splatting](https://arxiv.org/abs/2508.04099)
- [SAD-GS: Shape-aligned Depth-supervised Gaussian Splatting](https://openaccess.thecvf.com/content/CVPR2024W/NRI/html/Kung_SAD-GS_Shape-aligned_Depth-supervised_Gaussian_Splatting_CVPRW_2024_paper.html)
- [DepthSplat: Connecting Gaussian Splatting and Depth](https://openaccess.thecvf.com/content/CVPR2025/html/Xu_DepthSplat_Connecting_Gaussian_Splatting_and_Depth_CVPR_2025_paper.html)
- [MonoSplat: Generalizable 3D Gaussian Splatting from Monocular Depth Foundation Models](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_MonoSplat_Generalizable_3D_Gaussian_Splatting_from_Monocular_Depth_Foundation_Models_CVPR_2025_paper.html)
