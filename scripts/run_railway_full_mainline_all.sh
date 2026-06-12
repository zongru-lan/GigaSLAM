#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leizongru/lzr_ws/GigaSLAM}"
DATA_ROOT="${DATA_ROOT:-/home/leizongru/lzr_ws/railway_data}"
PYTHON_BIN="${PYTHON_BIN:-/home/leizongru/miniconda3/envs/gigaslam/bin/python}"
CONFIG_TEMPLATE_REL="${CONFIG_TEMPLATE_REL:-configs/railway/rowtrack350/full_mainline/template.yaml}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results/color_refinement_400}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/full_mainline}"
GENERATED_CONFIG_ROOT="${GENERATED_CONFIG_ROOT:-${PROJECT_ROOT}/logs/generated_configs}"
RUN_TAG="${RUN_TAG:-railway_full_mainline_$(date +%Y%m%d_%H%M%S)}"
GPU_MAX_MEM_MB="${GPU_MAX_MEM_MB:-1024}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-10}"
GT_TOL_SEC="${GT_TOL_SEC:-0.05}"

DEFAULT_SCENES=(
  scene_05_train
  scene_11_train
  scene_13_train
  scene_14_train
  scene_16_train
  scene_17_train
  scene_19_train
)

if [[ -n "${SCENES:-}" ]]; then
  read -r -a SCENE_LIST <<< "${SCENES//,/ }"
else
  SCENE_LIST=("${DEFAULT_SCENES[@]}")
fi

if [[ ${#SCENE_LIST[@]} -eq 0 ]]; then
  echo "ERROR: no scenes selected." >&2
  exit 2
fi

mkdir -p "$LOG_ROOT" "$GENERATED_CONFIG_ROOT" "$RESULTS_DIR" "$DATA_ROOT/gt_poses/npy"

SUMMARY="${LOG_ROOT}/${RUN_TAG}_summary.tsv"
FAILED="${LOG_ROOT}/${RUN_TAG}_failed.txt"
QUEUE="${LOG_ROOT}/${RUN_TAG}_queue.txt"
LAUNCHER_LOG="${LOG_ROOT}/${RUN_TAG}_launcher.log"

: > "$FAILED"
printf "scene\tgpu\tstatus\texit_code\tstart\tend\tlog\tconfig\n" > "$SUMMARY"
printf "%s\n" "${SCENE_LIST[@]}" > "$QUEUE"

log_msg() {
  local msg="$1"
  printf "[%s] %s\n" "$(date '+%F %T')" "$msg" | tee -a "$LAUNCHER_LOG"
}

require_path() {
  local path="$1"
  local desc="$2"
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing ${desc}: ${path}" >&2
    exit 2
  fi
}

check_runtime_imports() {
  set +e
  "$PYTHON_BIN" -c "from gaussian_splatting.scene.scaffold_model import GaussianModel; from gaussian_splatting.gaussian_renderer import render" >/dev/null 2>&1
  local rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    cat >&2 <<EOF
ERROR: GigaSLAM runtime import check failed.
Missing or incomplete dependency: gaussian_splatting.scene.scaffold_model

Expected one of:
  - a project-local gaussian_splatting/ package containing scene/scaffold_model.py
  - an installed package in ${PYTHON_BIN}'s environment that provides it

Fix that dependency first, then rerun this batch script.
EOF
    exit 2
  fi
}

detect_gpus() {
  GPU_LIST=()

  if [[ -n "${GPUS:-}" ]]; then
    read -r -a GPU_LIST <<< "${GPUS//,/ }"
  else
    local lines line idx mem util
    mapfile -t lines < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
    for line in "${lines[@]}"; do
      IFS=',' read -r idx mem util <<< "$line"
      idx="${idx//[[:space:]]/}"
      mem="${mem//[[:space:]]/}"
      util="${util//[[:space:]]/}"
      if [[ "$mem" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]]; then
        if (( mem <= GPU_MAX_MEM_MB && util <= GPU_MAX_UTIL )); then
          GPU_LIST+=("$idx")
        fi
      fi
    done
  fi

  if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
    echo "ERROR: no usable GPU found. Set GPUS=0,1 to override." >&2
    exit 2
  fi
}

ensure_gt_npy() {
  local scene="$1"
  local color_dir="${DATA_ROOT}/${scene}"
  local parquet_path="${DATA_ROOT}/gt_poses/${scene}.parquet"
  local npy_path="${DATA_ROOT}/gt_poses/npy/${scene}.npy"

  require_path "$color_dir" "color directory for ${scene}"
  require_path "$parquet_path" "GT parquet for ${scene}"

  if [[ -f "$npy_path" && "$npy_path" -nt "$parquet_path" ]]; then
    return
  fi

  "$PYTHON_BIN" - "$scene" "$color_dir" "$parquet_path" "$npy_path" "$GT_TOL_SEC" <<'PY'
import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def quat_to_rot(qx, qy, qz, qw):
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm <= 0:
        raise ValueError("zero-norm quaternion")
    x, y, z, w = q / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def image_timestamp(path):
    stem = Path(path).stem
    if "_" not in stem:
        raise ValueError(f"cannot parse timestamp from image name: {Path(path).name}")
    return float(stem.split("_", 1)[1])


scene, color_dir, parquet_path, npy_path, tol_s = sys.argv[1:6]
tol_s = float(tol_s)

image_paths = []
for pattern in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
    image_paths.extend(glob.glob(os.path.join(color_dir, pattern)))
image_paths = sorted(image_paths)
if not image_paths:
    raise FileNotFoundError(f"no images found in {color_dir}")

df = pd.read_parquet(parquet_path)
required = ["timestamp", "t_x", "t_y", "t_z", "r_x", "r_y", "r_z", "r_w"]
missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"{parquet_path} is missing required columns: {missing}")

gt_timestamps = df["timestamp"].to_numpy(dtype=np.float64)
order = np.argsort(gt_timestamps)
sorted_timestamps = gt_timestamps[order]

poses = []
rows = []
first_inv = None
for frame_idx, image_path in enumerate(image_paths):
    ts = image_timestamp(image_path)
    pos = int(np.searchsorted(sorted_timestamps, ts))
    candidates = []
    if pos < len(sorted_timestamps):
        candidates.append(pos)
    if pos > 0:
        candidates.append(pos - 1)
    if not candidates:
        raise ValueError(f"no GT pose candidate for image timestamp {ts}")

    best_pos = min(candidates, key=lambda i: abs(sorted_timestamps[i] - ts))
    dt = float(abs(sorted_timestamps[best_pos] - ts))
    if dt > tol_s:
        raise ValueError(
            f"no GT pose within {tol_s}s for {Path(image_path).name}; nearest dt={dt}"
        )

    gt_idx = int(order[best_pos])
    row = df.iloc[gt_idx]
    c2w_abs = np.eye(4, dtype=np.float64)
    c2w_abs[:3, :3] = quat_to_rot(row["r_x"], row["r_y"], row["r_z"], row["r_w"])
    c2w_abs[:3, 3] = [row["t_x"], row["t_y"], row["t_z"]]
    if first_inv is None:
        first_inv = np.linalg.inv(c2w_abs)

    c2w_rel = first_inv @ c2w_abs
    w2c_rel = np.linalg.inv(c2w_rel)
    poses.append(w2c_rel.astype(np.float32))
    rows.append(
        {
            "frame_idx": frame_idx,
            "image_name": Path(image_path).name,
            "image_timestamp": f"{ts:.9f}",
            "gt_pose_index": gt_idx,
            "gt_timestamp": f"{float(sorted_timestamps[best_pos]):.9f}",
            "timestamp_error_sec": f"{dt:.9f}",
        }
    )

out = Path(npy_path)
out.parent.mkdir(parents=True, exist_ok=True)
np.save(out, np.stack(poses, axis=0))

csv_path = out.with_suffix(".pose_matches.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "frame_idx",
            "image_name",
            "image_timestamp",
            "gt_pose_index",
            "gt_timestamp",
            "timestamp_error_sec",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"{scene}: saved {len(poses)} W2C GT poses to {out}")
PY
}

write_generated_config() {
  local scene="$1"
  local cfg="${GENERATED_CONFIG_ROOT}/${RUN_TAG}_${scene}.yaml"
  local npy_path="${DATA_ROOT}/gt_poses/npy/${scene}.npy"

  cat > "$cfg" <<YAML
inherit_from: ${CONFIG_TEMPLATE_REL}

Dataset:
  color_path: ${DATA_ROOT}/${scene}
  pose_path: ${npy_path}

DepthModel:
  local_snapshot_path: ${PROJECT_ROOT}/unidepth_model

RailScale:
  detector_checkpoint: ${PROJECT_ROOT}/railbench/rail_detector_runs/segformer_b0/best.pt

Results:
  save_dir: ${RESULTS_DIR}
  use_gui: false
  logging:
    quiet: true
YAML

  printf "%s" "$cfg"
}

pop_scene() {
  local scene=""
  {
    flock -x 200
    if [[ -s "$QUEUE" ]]; then
      scene="$(head -n 1 "$QUEUE")"
      local tmp="${QUEUE}.${BASHPID}.tmp"
      tail -n +2 "$QUEUE" > "$tmp"
      mv "$tmp" "$QUEUE"
    fi
    printf "%s" "$scene"
  } 200>"${QUEUE}.lock"
}

append_summary() {
  local scene="$1"
  local gpu="$2"
  local status="$3"
  local exit_code="$4"
  local start_time="$5"
  local end_time="$6"
  local log_path="$7"
  local cfg_path="$8"

  {
    flock -x 201
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$scene" "$gpu" "$status" "$exit_code" "$start_time" "$end_time" "$log_path" "$cfg_path" >> "$SUMMARY"
    if [[ "$status" != "OK" && "$status" != "DRY_RUN" ]]; then
      printf "%s\tgpu=%s\texit_code=%s\tlog=%s\n" "$scene" "$gpu" "$exit_code" "$log_path" >> "$FAILED"
    fi
  } 201>"${SUMMARY}.lock"
}

run_scene() {
  local scene="$1"
  local gpu="$2"
  local cfg="${GENERATED_CONFIG_ROOT}/${RUN_TAG}_${scene}.yaml"
  local log_path="${LOG_ROOT}/${RUN_TAG}_${scene}_gpu${gpu}.log"
  local start_time
  local end_time
  local exit_code
  local status

  start_time="$(date '+%F %T')"
  {
    printf "run_tag: %s\n" "$RUN_TAG"
    printf "scene: %s\n" "$scene"
    printf "gpu: %s\n" "$gpu"
    printf "config: %s\n" "$cfg"
    printf "start: %s\n" "$start_time"
    printf "repo: %s\n" "$PROJECT_ROOT"
    printf "python: %s\n" "$PYTHON_BIN"
    printf "data_root: %s\n" "$DATA_ROOT"
    printf "\ncommand:\n"
    printf "CUDA_VISIBLE_DEVICES=%s %s slam.py --config %s\n\n" "$gpu" "$PYTHON_BIN" "$cfg"
  } > "$log_path"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    end_time="$(date '+%F %T')"
    append_summary "$scene" "$gpu" "DRY_RUN" "0" "$start_time" "$end_time" "$log_path" "$cfg"
    return 0
  fi

  set +e
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" slam.py --config "$cfg" >> "$log_path" 2>&1
  exit_code=$?
  set -e

  end_time="$(date '+%F %T')"
  if [[ "$exit_code" -eq 0 ]]; then
    status="OK"
  else
    status="FAIL"
  fi
  {
    printf "\nend: %s\n" "$end_time"
    printf "exit_code: %s\n" "$exit_code"
    printf "status: %s\n" "$status"
  } >> "$log_path"

  append_summary "$scene" "$gpu" "$status" "$exit_code" "$start_time" "$end_time" "$log_path" "$cfg"
}

run_worker() {
  local gpu="$1"
  local scene
  while true; do
    scene="$(pop_scene)"
    if [[ -z "$scene" ]]; then
      break
    fi
    log_msg "GPU ${gpu}: start ${scene}"
    run_scene "$scene" "$gpu"
    log_msg "GPU ${gpu}: finish ${scene}"
  done
}

trap 'log_msg "Interrupted; stopping workers."; jobs -pr | xargs -r kill; exit 130' INT TERM

require_path "$PROJECT_ROOT" "project root"
require_path "$PYTHON_BIN" "python binary"
require_path "${PROJECT_ROOT}/${CONFIG_TEMPLATE_REL}" "full-mainline template"
require_path "${PROJECT_ROOT}/unidepth_model" "UniDepth local snapshot"
require_path "${PROJECT_ROOT}/railbench/rail_detector_runs/segformer_b0/best.pt" "rail detector checkpoint"

cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}${EXTRA_PYTHONPATH:+:${EXTRA_PYTHONPATH}}"

check_runtime_imports
detect_gpus

log_msg "Run tag: ${RUN_TAG}"
log_msg "Scenes: ${SCENE_LIST[*]}"
log_msg "GPUs: ${GPU_LIST[*]}"
log_msg "Logs: ${LOG_ROOT}"
log_msg "Summary: ${SUMMARY}"
log_msg "Results: ${RESULTS_DIR}"

for scene in "${SCENE_LIST[@]}"; do
  ensure_gt_npy "$scene"
  cfg="$(write_generated_config "$scene")"
  log_msg "Prepared ${scene}: ${cfg}"
done

for gpu in "${GPU_LIST[@]}"; do
  run_worker "$gpu" &
done

wait

log_msg "Batch summary:"
cat "$SUMMARY" | tee -a "$LAUNCHER_LOG"

if [[ -s "$FAILED" ]]; then
  log_msg "Some scenes failed. Failed list: ${FAILED}"
  exit 1
fi

log_msg "All scenes finished successfully."
