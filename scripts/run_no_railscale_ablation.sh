#!/usr/bin/env bash
set -euo pipefail

cd /root/GigaSLAM

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/gigaslam/bin/python}"
LOG_DIR="logs/RailScale_ablation"
RESULT_DIR="results/RailScale_ablation"
CONFIG_DIR="configs/railway/rowtrack350/ablations/no_railscale"

mkdir -p "${LOG_DIR}" "${RESULT_DIR}"

scenes=(
  scene_17_train
)

echo "[no_railscale] started at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[no_railscale] python: ${PYTHON_BIN}"
echo "[no_railscale] results: ${RESULT_DIR}"
echo "[no_railscale] logs: ${LOG_DIR}"

for scene in "${scenes[@]}"; do
  config="${CONFIG_DIR}/${scene}.yaml"
  timestamp="$(date +%Y%m%d_%H%M%S)"
  log_path="${LOG_DIR}/${scene}_no_railscale_${timestamp}.log"

  echo
  echo "[no_railscale] running ${scene}"
  echo "[no_railscale] config: ${config}"
  echo "[no_railscale] log: ${log_path}"

  "${PYTHON_BIN}" slam.py --config "${config}" 2>&1 | tee "${log_path}"

  echo "[no_railscale] finished ${scene} at $(date '+%Y-%m-%d %H:%M:%S')"
done

echo
echo "[no_railscale] all scenes finished at $(date '+%Y-%m-%d %H:%M:%S')"
