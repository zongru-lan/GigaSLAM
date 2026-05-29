#!/usr/bin/env bash
# 把 22 个 parquet GT 位姿文件批量转为 .npy 格式（W2C，shape N×4×4）。
# 运行前提：
#   cd /root/GigaSLAM
#   conda activate gigaslam（或对应环境）
#
# 用法：
#   bash scripts/convert_all_poses.sh
#
# 输出目录：railway_data/gt_poses/npy/
#   scene_01_train.npy
#   scene_02_train.npy
#   ...

set -e

RAILWAY_DATA="/root/GigaSLAM/railway_data"
PARQUET_DIR="${RAILWAY_DATA}/gt_poses"
NPY_DIR="${RAILWAY_DATA}/gt_poses/npy"

mkdir -p "${NPY_DIR}"

for scene in scene_{01..22}_train; do
    parquet="${PARQUET_DIR}/${scene}.parquet"
    img_dir="${RAILWAY_DATA}/${scene}"
    output="${NPY_DIR}/${scene}.npy"

    if [ ! -f "${parquet}" ]; then
        echo "[SKIP] ${scene}: parquet not found"
        continue
    fi
    if [ ! -d "${img_dir}" ]; then
        echo "[SKIP] ${scene}: image dir not found"
        continue
    fi

    echo "=== Converting ${scene} ==="
    python scripts/read_pose.py \
        --parquet "${parquet}" \
        --img_dir  "${img_dir}" \
        --output   "${output}"
done

echo ""
echo "Done. NPY files in ${NPY_DIR}:"
ls -1 "${NPY_DIR}"
