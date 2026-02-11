#!/bin/bash
# Launch GR00T N1.6 fine-tuning for Franka MCX Card Block Insertion.
#
# Prerequisites:
#   1. Convert HDF5 to LeRobot v2:
#      python scripts/data_pipeline/convert_hdf5_to_lerobot_v2.py \
#        --hdf5 ~/IsaacLab/mcx_card_training_augmented.hdf5 \
#        --output ~/groot_data/mcx_card_lerobot_v2
#
#   2. Install Isaac-GR00T:
#      cd ~/Isaac-GR00T && pip install -e .
#
# Usage:
#   bash scripts/groot_finetune/launch_finetune_mcx.sh

set -e

# Ensure conda env binaries (ffmpeg, ffprobe) are on PATH
export PATH="/home/tshiamo/miniforge3/envs/isaaclab/bin:$PATH"

# Memory optimization for RTX 5090 (32GB)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GROOT_DIR="/home/tshiamo/Isaac-GR00T"
DATASET_PATH="/home/tshiamo/groot_data/mcx_card_lerobot_v2"
MODALITY_CONFIG="/home/tshiamo/SIMULATION_MANIPULATION/franka_factory/scripts/groot_finetune/franka_mcx_config.py"
BASE_MODEL="nvidia/GR00T-N1.6-3B"
OUTPUT_DIR="/home/tshiamo/groot_data/finetune_output_n16"

cd "${GROOT_DIR}"

python /home/tshiamo/SIMULATION_MANIPULATION/franka_factory/scripts/groot_finetune/launch_finetune.py \
    --base-model-path "${BASE_MODEL}" \
    --dataset-path "${DATASET_PATH}" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "${MODALITY_CONFIG}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-gpus 1 \
    --global-batch-size 4 \
    --gradient-accumulation-steps 8 \
    --learning-rate 1e-4 \
    --max-steps 10000 \
    --save-steps 1000 \
    --save-total-limit 5 \
    --dataloader-num-workers 4 \
    --num-shards-per-epoch 5000 \
    --shard-size 512 \
    --episode-sampling-rate 0.3 \
    --no-tune-llm \
    --no-tune-visual \
    --tune-projector \
    --tune-diffusion-model \
    --warmup-ratio 0.05 \
    --weight-decay 1e-5
