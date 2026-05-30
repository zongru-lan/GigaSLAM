# Results

当前结果目录按实验口径分桶，避免主线、消融和完整渲染结果混在一起。

## 目录说明

- `results/color_refinement_0/`：当前轨迹主线结果。对应配置 `configs/railway/rowtrack350/scenes/*.yaml`，`color_refinement_iter=0`。
- `results/color_refinement_0/trajectory_shape_summary.csv`：当前主线轨迹形态汇总。
- `results/color_refinement_0/report_plots/`：当前主线轨迹汇报图。
- `results/color_refinement_0/RailScale_ablation/`：`w/o RailScale` 消融结果。
- `results/color_refinement_400/`：完整建图/渲染结果。对应配置 `configs/railway/rowtrack350/full_mainline/scenes/*.yaml`，`color_refinement_iter=400`。

## 维护规则

- 新主线结果放入 `color_refinement_0/` 或新的明确命名目录，不要直接散放在 `results/` 根目录。
- 完整渲染/建图质量结果放入 `color_refinement_400/`。
- 临时失败实验结果跑完分析后可以删除，只把结论写入 `docs/`。
- 每个正式 run 目录中的 `config.yml` 是复现实验的重要依据，不要删除。
