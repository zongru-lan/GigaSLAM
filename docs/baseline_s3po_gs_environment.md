# S3PO-GS Baseline 环境配置记录

本文档记录 `/root/GigaSLAM/baselines/S3PO-GS` 的环境配置状态，便于后续复现和排查。

## 当前状态

环境已经配置并通过 import 验收：

```bash
conda activate /autodl-fs/data/conda_envs/S3PO-GS
cd /root/GigaSLAM/baselines/S3PO-GS

python -c "import torch; import simple_knn._C; import diff_gaussian_rasterization; from mast3r.model import AsymmetricMASt3R; from dust3r.inference import inference; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print('S3PO-GS env ok')"
```

预期输出：

```text
2.1.0+cu118 11.8 True
S3PO-GS env ok
```

注意：最终检查中已经没有 `cannot find cuda-compiled version of RoPE2D` 警告，说明 MASt3R/DUSt3R 使用的 `curope` CUDA 扩展已经被正确识别。

## 环境入口

```bash
conda activate /autodl-fs/data/conda_envs/S3PO-GS
cd /root/GigaSLAM/baselines/S3PO-GS
```

不需要每次手动 `export OMP_NUM_THREADS` 或 CUDA 编译变量。当前 conda 环境已持久化：

```text
PIP_CACHE_DIR=/autodl-fs/data/pip_cache
TMPDIR=/autodl-fs/data/tmp
TORCH_CUDA_ARCH_LIST=8.9
```

这些变量只在该 conda 环境激活时生效，退出环境后不会污染其他环境。

## 关键版本

```text
Python: 3.11
torch: 2.1.0+cu118
torchvision: 0.16.0+cu118
torchaudio: 2.1.0+cu118
triton: 2.1.0
numpy: 1.24.4
opencv-python: 4.8.1.78
scipy: 1.11.4
pandas: 2.0.3
open3d: 0.18.0
evo: 1.11.0
lpips: 0.1.4
simple_knn: 0.0.0
diff_gaussian_rasterization: 0.0.0
```

## 已编译扩展

以下 CUDA/C++ 扩展已编译并通过 import：

```text
submodules/simple-knn
submodules/diff-gaussian-rasterization
croco/models/curope
dust3r/croco/models/curope
```

其中 `dust3r/croco/models/curope` 很关键。MASt3R 通过 DUSt3R 子模块导入 RoPE2D，如果只编译顶层 `croco/models/curope`，仍会退回慢速 PyTorch 版本。

编译日志保存在：

```text
/root/GigaSLAM/logs/baseline_env/S3PO-GS/simple_knn_build.log
/root/GigaSLAM/logs/baseline_env/S3PO-GS/diff_gaussian_rasterization_build.log
/root/GigaSLAM/logs/baseline_env/S3PO-GS/curope_build.log
/root/GigaSLAM/logs/baseline_env/S3PO-GS/dust3r_curope_build.log
```

## 特殊处理

配置过程中遇到过 `cannot find -lcudart`，原因是 conda 环境中的 `libcudart.so` 链接不完整。当前已将：

```text
/autodl-fs/data/conda_envs/S3PO-GS/lib/libcudart.so
```

指向 PyTorch wheel 自带的 CUDA runtime：

```text
/autodl-fs/data/conda_envs/S3PO-GS/lib/python3.11/site-packages/torch/lib/libcudart-d0da41ae.so.11.0
```

如果后续重装 CUDA runtime 或 PyTorch，建议重新执行最终 import 检查。

## 权重状态

S3PO-GS 已改成优先加载本地 MASt3R checkpoint：

```text
/root/GigaSLAM/baselines/S3PO-GS/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
```

如果该文件存在，`slam.py` 会直接使用本地 `.pth`；如果不存在，才回退到 Hugging Face：

```python
naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric
```

当前已验证本地 checkpoint 能被解析到，文件格式检查结果为 `dict_keys(['model', 'args'])`。

## 最小验证命令

```bash
conda activate /autodl-fs/data/conda_envs/S3PO-GS
cd /root/GigaSLAM/baselines/S3PO-GS

python -c "import torch; import simple_knn._C; import diff_gaussian_rasterization; from mast3r.model import AsymmetricMASt3R; from dust3r.inference import inference; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print('S3PO-GS env ok')"
```

如果该命令通过，就说明环境主体可用。
