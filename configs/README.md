# configs directory guide

Use this as the source-of-truth map for configuration files.

## Mainline

- `configs/railway/rowtrack350/template.yaml`: frozen rowtrack350 mainline template.
- `configs/railway/rowtrack350/scenes/*.yaml`: thin per-scene dataset path configs.
- `configs/railway/rowtrack350/baselines/`: frozen cross-scene regression baselines.

## Maintained Experiments

- `configs/railway/rowtrack350/experiments/depthdiag.yaml`: rendered-depth diagnostic experiment.
- `configs/railway/rowtrack350/experiments/depthdiag_scene16.yaml`: single-scene depthdiag smoke test.

## Compatibility / Generated Output

- `configs/rgb_12mp_middle.yaml`: compatibility shim for older single-run commands; it now inherits `configs/railway/rowtrack350/scenes/scene_16_train.yaml`.
- `configs/generated_metric_width/`: output location for generated configs. Do not treat it as mainline.
- `configs/archive/`: historical configs, legacy root configs, backups, and generated-run archives retained for reproducibility.
