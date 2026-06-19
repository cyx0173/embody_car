#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-$PWD/datasets/orange_to_bowl_marked_dual_camera_place_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/train/orange_to_bowl_marked_dual_act_8gpu_5000}"
JOB_NAME="${JOB_NAME:-orange_to_bowl_marked_dual_act_8gpu_5000}"

if [ ! -d "$DATASET_ROOT" ]; then
  echo "Dataset not found: $DATASET_ROOT" >&2
  exit 1
fi

if [ -e "$OUTPUT_DIR" ]; then
  echo "Output dir already exists: $OUTPUT_DIR" >&2
  echo "Choose a new name with: OUTPUT_DIR=/path/to/new/output JOB_NAME=name bash train_place_8gpu_5000.sh" >&2
  exit 1
fi

torchrun --standalone --nproc_per_node=8 ./train_lerobot_clean.py \
  --dataset.repo_id=embody_car/orange_to_bowl_marked_dual_camera_place_v1 \
  --dataset.root="$DATASET_ROOT" \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.pretrained_backbone_weights=null \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --batch_size=16 \
  --steps=5000 \
  --log_freq=20 \
  --save_freq=500 \
  --eval_freq=0 \
  --num_workers=2 \
  --wandb.enable=false
