# no_railscale ablation configs

These configs disable RailScale while keeping the current rowtrack350 scene
settings unchanged. They are intended for the `w/o RailScale` ablation.

Outputs are written under:

```text
results/RailScale_ablation/
```

Run one scene manually:

```bash
cd /root/GigaSLAM
/root/miniconda3/envs/gigaslam/bin/python slam.py \
  --config configs/railway/rowtrack350/ablations/no_railscale/scene_11_train.yaml
```

Recommended logging pattern:

```bash
mkdir -p logs/RailScale_ablation results/RailScale_ablation
/root/miniconda3/envs/gigaslam/bin/python slam.py \
  --config configs/railway/rowtrack350/ablations/no_railscale/scene_11_train.yaml \
  2>&1 | tee logs/RailScale_ablation/scene_11_no_railscale_$(date +%Y%m%d_%H%M%S).log
```
