# Configs

当前维护中的配置入口尽量少，只保留可复现主线和必要消融。

## 当前入口

- `configs/base_config.yaml`：全局默认配置。一般不要直接运行它。
- `configs/railway/rowtrack350/template.yaml`：轨迹主线模板，`color_refinement_iter=0`，输出到 `results/color_refinement_0/`。
- `configs/railway/rowtrack350/scenes/*.yaml`：每个 scene 的薄配置，只覆盖数据路径。
- `configs/railway/rowtrack350/full_mainline/`：完整建图/渲染配置，`color_refinement_iter=400`，输出到 `results/color_refinement_400/`。
- `configs/railway/rowtrack350/ablations/no_railscale/`：RailScale 消融配置。

## 推荐命令

轨迹主线：

```bash
cd /home/leizongru/lzr_ws/GigaSLAM
/home/leizongru/miniconda3/envs/gigaslam/bin/python slam.py --config configs/railway/rowtrack350/scenes/scene_16_train.yaml
```

完整建图/渲染：

```bash
cd /home/leizongru/lzr_ws/GigaSLAM
/home/leizongru/miniconda3/envs/gigaslam/bin/python slam.py --config configs/railway/rowtrack350/full_mainline/scenes/scene_16_train.yaml
```

## 维护规则

- 不再维护 `configs/generated_metric_width/`、`configs/archive/`、旧 `rgb_12mp_middle.yaml` 入口。
- 批跑脚本生成的临时 YAML 默认写到 `logs/generated_configs/`，不要放回 `configs/`。
- 新实验如果只是临时验证，优先写入 `logs/generated_configs/` 或单独实验分支；确认有长期价值后再进入 `configs/railway/rowtrack350/`。
