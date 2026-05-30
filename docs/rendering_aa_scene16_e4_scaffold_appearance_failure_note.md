# E4 Scaffold-GS View-Adaptive Appearance 实验记录

## 动机

E1 的 Mip-Splatting 风格滤波只带来轻微 LPIPS 改善；E3 的 Pixel-GS 风格在线 densification 与当前 SLAM 后端不兼容，已回退。下一步尝试不改变在线建图拓扑的外观表达机制：启用 Scaffold-GS 里已有但当前主线硬编码关闭的 view-adaptive feature bank 和 distance-aware MLP 输入。

铁路场景中枕木、碎石、轨道边缘在远近尺度变化时非常容易出现高频错位和外观不稳定。相比单纯 LPIPS loss，view-adaptive appearance 允许颜色、协方差和透明度预测显式感知视角/距离，有机会改善不同帧视角下的渲染一致性。

## 实现口径

修改 `slam.py`，让以下参数可由 `Hierarchical` 配置控制，而不是固定为 `False`：

- `use_feat_bank`
- `add_color_dist`
- `add_cov_dist`
- `add_opacity_dist`

实验配置：`configs/railway/rowtrack350/experiments/scaffold_appearance_scene16.yaml`

关键设置：

| 参数 | 当前值 | 含义 |
| --- | --- | --- |
| `use_feat_bank` | `true` | 启用 Scaffold-GS 的多尺度特征 bank，用视角信息组合不同层次特征 |
| `add_color_dist` | `true` | 颜色 MLP 输入加入距离，提升远近尺度下的外观适配能力 |
| `add_cov_dist` | `true` | 协方差 MLP 输入加入距离，允许 footprint/形状随距离调整 |
| `add_opacity_dist` | `true` | 透明度 MLP 输入加入距离，增强跨尺度 alpha 表达 |
| `color_refinement_iter` | `400` | 与 scene16 full rendering baseline 对齐 |

## 运行命令

```bash
cd /root/GigaSLAM
mkdir -p logs/rendering_aa_scene16/e4_scaffold_appearance results/rendering_aa_scene16/e4_scaffold_appearance
/root/miniconda3/envs/gigaslam/bin/python scripts/run_metric_width_scenes.py   --base-config configs/railway/rowtrack350/experiments/scaffold_appearance_scene16.yaml   --scenes scene_16_train   --generated-config-dir configs/generated_metric_width/rendering_aa_scene16/e4_scaffold_appearance   --tag e4_scaffold_appearance   --continue-on-error   2>&1 | tee logs/rendering_aa_scene16/e4_scaffold_appearance/scene16_e4_scaffold_appearance_$(date +%Y%m%d_%H%M%S).log
```

## 判定标准

- LPIPS mean / p95 优先；
- PSNR/SSIM 不应灾难性下降；
- 若 LPIPS 下降但图像明显变糊，需要人工截图判断；
- ATE/RPE 不作为主目标，但不能出现明显异常。


## 失败结论

E4 已停止并回退源码。scene16 前端和 36000 次 color refinement 都能完成，但进入渲染评估后卡在第 0 帧：`poses_idx.txt` 已写入 179 行，`poses_est.txt` 只有 1 行，`img/` 为空，GPU 显存接近 24GB 且长时间无有效输出。

判断：启用 `use_feat_bank + add_color_dist + add_cov_dist + add_opacity_dist` 后，模型外观表达更重，当前高分辨率 railway scene16 的 eval rendering 成本不可接受。这个方向并非理论上无效，但作为“快速提升 LPIPS”的 scene16-only 大胆实验，工程代价过高，暂不继续。

处理：删除 E4 结果、临时配置和日志，回退 `slam.py` 的 E4 配置入口。后续继续保留 E1 Mip-style AA 作为唯一已跑通并有轻微 LPIPS 改善的方向。
