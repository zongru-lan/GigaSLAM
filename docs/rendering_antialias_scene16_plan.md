# Scene16-Only Anti-Aliasing 建图质量实验计划

本文记录新的大胆试错路线。目标是快速判断高质量 3DGS anti-aliasing / multi-scale 源码机制是否能改善铁路场景建图质量，尤其 LPIPS。

## 固定设置

- 只跑 `scene_16_train`；
- 不跑全 6 序列，除非 scene16 明显成功后再开新阶段；
- RailScale、前端 VO、关键帧机制保持固定；
- 不继续调 LPIPS loss；
- 不做小幅参数搜索；
- 每轮实验都可失败，失败后记录、删除结果、回退分支。

## Baseline

固定对比：

```text
results/full_mainline/GigaSLAM_railway_data_scene_16_train/2026-05-29-13-50-06-No-LC
```

baseline 指标：

| 指标 | 值 |
|---|---:|
| PSNR | 21.5284 |
| SSIM | 0.8829 |
| LPIPS | 0.6571 |
| ATE RMSE | 2.7601 m |

## 实验顺序

### E1: Mip-Splatting

优先移植 Mip-Splatting 的 3D smoothing + 2D Mip filter / anti-alias filtering。

目标：

- 降低远近尺度变化导致的 aliasing；
- 减少枕木、碎石、接触网的高频不稳定；
- 优先降低 LPIPS mean 和 LPIPS p95。

### E2: Analytic-Splatting

若 E1 编译/接口成本过高或效果不明显，尝试 Analytic-Splatting 的 pixel-area integration。

目标：

- 用像素面积积分替代点采样式 splatting；
- 改善细线和重复纹理的采样稳定性。

### E3: Pixel-GS

若 filtering 不足以改善 LPIPS，移植 Pixel-GS 的 pixel-aware densification。

目标：

- 增强轨道边缘、枕木、碎石等高频区域的 Gaussian 覆盖；
- 后续可结合 RailScale ROI 做 railway-aware densification。

### E4: Scaffold-GS Appearance

最后再尝试 Scaffold-GS 风格 view-adaptive appearance。

目标：

- 处理视角相关颜色、反光、曝光变化；
- 不作为第一优先级，因为当前主要问题更像 aliasing 和细节覆盖。

## 输出组织

每个实验使用独立目录：

```text
results/rendering_aa_scene16/<experiment>/
logs/rendering_aa_scene16/<experiment>/
configs/generated_metric_width/rendering_aa_scene16/<experiment>/
```

参考源码克隆到：

```text
research_repos/rendering/
```

该目录不提交到 git。

## 评价指标

主指标：

- LPIPS mean；
- LPIPS p95；
- 人工检查关键帧渲染图：是否更清晰、是否减少高频错位、是否只是变糊。

辅助指标：

- PSNR；
- SSIM；
- ATE；
- RPE；
- 运行时间；
- 显存占用。

## 成功与放弃

成功：

- scene16 LPIPS 明显低于 `0.6571`；
- PSNR/SSIM 不灾难性下降；
- 渲染图不是靠明显模糊换 LPIPS；
- ATE/RPE 基本不异常退化。

放弃：

- 编译/接入超过合理时间仍跑不通；
- LPIPS 不降；
- 画面更糊或几何错位更严重；
- 显存/时间不可接受。

失败处理：

- 记录到 `docs/rendering_aa_scene16_failure_log.md`；
- 删除该实验结果；
- 回退该实验分支改动；
- 进入下一篇源码方案。

## 当前进展：E1 已完成

E1 Mip-Splatting 移植已经在 `scene_16_train` 上跑通。结果见 `docs/rendering_aa_scene16_e1_mip_splatting_note.md` 和 `results/rendering_aa_scene16/README.md`。

初步结论：LPIPS 从 `0.6571` 降到 `0.6445`，但 PSNR/SSIM 分别从 `21.5284` / `0.8829` 降到 `21.3395` / `0.8732`。因此 E1 是 partial positive，不建议继续小幅调 Mip 参数；下一轮优先尝试 pixel-aware densification。


## 已执行进展

### E3 Pixel-GS-style densification

已尝试，失败并回退。当前 SLAM 后端与 Scaffold 原始 online anchor growing 不兼容，scene16 在第 2 帧附近停滞。结论是不继续把 Pixel-GS densification 作为快速试错方向。

### E4 Scaffold-GS view-adaptive appearance

已尝试，失败并回退。训练/refinement 能完成，但 eval rendering 在第 0 帧显存接近满载并长期无输出。结论是重型 view-adaptive appearance 对当前高分辨率铁路评估成本过高，暂不继续。

### 下一步调整

E1 Mip-style AA 是目前唯一跑通且 LPIPS 有正向变化的机制。后续优先做两个判断：

1. 指标口径拆解：all-frame / keyframe / non-keyframe 以及 native render resolution / 4K upsampled GT 的差异；
2. 若确认不是评估分辨率主导，再考虑完整移植 Analytic-Splatting 的 CUDA forward/backward。


### 分辨率口径拆解

已完成 keyframe-only 诊断。scene16 full-mainline 在 `4112x2504` 比较口径下 LPIPS mean 为 `0.6301`，将渲染图和 GT 同时降到 `1280x779` 后 LPIPS mean 降到 `0.4113`，`640x390` 后降到 `0.2874`。这说明 LPIPS 高很大一部分来自 `rendering_width=1280` 渲染后与 4K GT 比较的口径，而不完全是外观模型能力不足。后续应先建立 native-resolution/all-frame 的补充评估，再决定是否做完整 Analytic-Splatting CUDA 移植。
