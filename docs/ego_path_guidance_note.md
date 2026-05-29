# EgoPathGuidance 模块设计与训练说明

本文记录 TEP-Net 思路如何进入 GigaSLAM。它是 RailScale 的候选增强模块，不是当前主线的替代实现。

## 一句话定义

EgoPathGuidance 是一个用于预测当前列车 ego-path 的视觉模块。它输入单帧铁路图像，输出当前行驶轨道的左右轨边曲线，用来帮助 RailScale 在多轨、道岔和弯道场景中判断哪一对 rail pair 更可能属于本车轨道。

英文写法可用：

> EgoPathGuidance predicts the train ego-path, i.e. the rail pair corresponding to the train's immediate path, and provides a track-level prior for RailScale candidate selection.

## 和当前 rail detector 的区别

当前 RailScale 使用的 rail detector 是 dense heatmap detector：

```text
RGB image -> left/right rail-edge heatmaps
```

它会响应图像中可见的 rail edges，但不直接判断哪一对轨道是 ego track。

EgoPathGuidance 的目标不同：

```text
RGB image -> ego left rail curve + ego right rail curve + y-limit
```

因此它更接近“目标轨道选择器”，而不是所有轨道边缘检测器。

## 为什么不能直接用 object_and_rail 训练

`railbench/object_and_rail` 的 rail 标注是 COCO-like polyline：

- `category_id=1` 表示 rail。
- `rightRail=0/1` 只区分左/右 rail edge。
- 一张图可能有多条左轨和右轨。
- 标注没有直接给出 ego-track ID。

所以训练 TEP-style 模型前，需要先生成保守的伪 ego-path 标签：从所有左右轨候选中选择一对最像当前行驶轨道的 rail pair，歧义样本直接跳过。

## 伪标签生成

脚本：

```text
scripts/prepare_railbench_ego_path.py
```

推荐命令：

```bash
cd /root/GigaSLAM
/root/miniconda3/envs/gigaslam/bin/python scripts/prepare_railbench_ego_path.py \
  --root railbench/object_and_rail \
  --out railbench/object_and_rail/prepared_ego_path \
  --splits train val \
  --anchors 64 \
  --preview-count 200 \
  --overwrite
```

输出：

```text
railbench/object_and_rail/prepared_ego_path/annotations_train_egopath.json
railbench/object_and_rail/prepared_ego_path/annotations_val_egopath.json
railbench/object_and_rail/prepared_ego_path/label_quality_train.csv
railbench/object_and_rail/prepared_ego_path/label_quality_val.csv
railbench/object_and_rail/prepared_ego_path/preview/
```

选择规则：

- 枚举所有 left/right rail polyline pair。
- 要求 bottom anchor 可见、`left_x < right_x`、有效 anchor 数足够。
- 使用保守门控过滤跨轨/邻轨伪标签：底部宽度默认不超过图像宽度的 `0.30`，底部中心偏差默认不超过 `0.30`，宽度变化系数默认不超过 `0.60`。
- 以底部中心接近图像中心、覆盖长度、宽度收敛性和稳定性打分。
- 最优候选分数低于 `2.5`，或多候选时和次优候选 margin 小于 `0.10`，样本被拒绝。

`label_quality_*.csv` 是训练前必须检查的文件。若大量样本因 `ambiguous_pair` 或 `low_score` 被拒绝，应先抽查 `preview/`，不要急着放宽阈值。

## 模型训练

脚本：

```text
scripts/train_ego_path_detector.py
```

它复用 `/root/GigaSLAM/train-ego-path-detection/src` 中的 `RegressionNet` 和 regression loss，但不使用原仓库的 W&B 登录和随机切分。

首轮推荐训练：

```bash
cd /root/GigaSLAM
mkdir -p EgoPathGuidance/logs
/root/miniconda3/envs/gigaslam/bin/python scripts/train_ego_path_detector.py \
  --tep-root train-ego-path-detection \
  --train-images railbench/object_and_rail/images/train \
  --train-annotations railbench/object_and_rail/prepared_ego_path/annotations_train_egopath.json \
  --val-images railbench/object_and_rail/images/val \
  --val-annotations railbench/object_and_rail/prepared_ego_path/annotations_val_egopath.json \
  --out /autodl-fs/data/GigaSLAM/railbench/ego_path_runs/resnet18_regression_v1 \
  --backbone resnet18 \
  --epochs 120 \
  --batch-size 8 \
  --device cuda \
  2>&1 | tee EgoPathGuidance/logs/egopath_resnet18_regression_v1_$(date +%Y%m%d_%H%M%S).log
```

输出：

```text
best.pt
last.pt
config.yaml
args.json
metrics.json
preview/
```

`best.pt` 和 `config.yaml` 保持 TEP repo `Detector` 可读取的格式。

## SLAM 中的第一阶段用法

当前只做 shadow diagnostic，不改变 RailScale 选择结果。配置入口在：

```yaml
RailScale:
  ego_path_guidance:
    enabled: false
    model_path: /autodl-fs/data/GigaSLAM/railbench/ego_path_runs/resnet18_regression_v1
    tep_root: /root/GigaSLAM/train-ego-path-detection
    crop: none
    log_diag: true
    save_debug: false
```

打开后，每个结果目录会额外保存：

```text
ego_path_diag.csv
ego_path_debug/   # 仅 save_debug: true 时生成
```

`ego_path_diag.csv` 会记录 EgoPath 预测的中心/宽度，以及它与当前 RailScale 所选 rail pair 的中心/宽度差异。

## 论文包装建议

当前主线论文仍应以 RailScale 为核心。EgoPathGuidance 可以作为后续扩展或 RailScale-v2：

> To reduce ambiguity in multi-track scenes, we further explore an ego-path guidance module that predicts the rail pair corresponding to the train's immediate path. In the current stage, this module is used diagnostically to assess whether track-level path prediction can improve RailScale candidate selection.

避免写成：

> We replace RailScale with TEP-Net.

或：

> The current baseline already uses ego-path prediction.

因为当前冻结 baseline 仍然是 rowtrack350 RailScale。
