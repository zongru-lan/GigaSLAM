# RailScale 消融配置

这组配置用于 `w/o RailScale` 消融实验：关闭 RailScale，其余 rowtrack350 主线设置保持一致。

输出目录：

```text
results/color_refinement_0/RailScale_ablation/
```

单序列运行：

```bash
cd /root/GigaSLAM
/root/miniconda3/envs/gigaslam/bin/python slam.py \
  --config configs/railway/rowtrack350/ablations/no_railscale/scene_11_train.yaml
```

带日志运行：

```bash
cd /root/GigaSLAM
mkdir -p logs/RailScale_ablation results/color_refinement_0/RailScale_ablation
/root/miniconda3/envs/gigaslam/bin/python slam.py \
  --config configs/railway/rowtrack350/ablations/no_railscale/scene_11_train.yaml \
  2>&1 | tee logs/RailScale_ablation/scene_11_no_railscale_$(date +%Y%m%d_%H%M%S).log
```

已知情况：`scene_17_train` 的 no-RailScale 消融曾出现 OOM，因此当前消融汇总以其它完成序列为主。
