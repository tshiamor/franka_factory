#!/usr/bin/env bash
# =============================================================================
# Pi-Zero (LeRobot) Training Setup for MCX Card Demos (NVIDIA Brev)
# =============================================================================
#
# Run this script on a cloud instance with 4-8x A100/H100 GPUs.
# It will:
#   1. Install dependencies and conda
#   2. Clone LeRobot repository
#   3. Pull the dataset from HuggingFace (tshiamor/mcx-card-pizero)
#   4. Train Pi-Zero policy on the dataset
#   5. Upload trained model to HuggingFace
#
# Usage:
#   export HF_TOKEN="your_huggingface_token"
#   bash brev_train_pizero.sh
#
# =============================================================================

set -euo pipefail

# ---- Configuration ----
HF_DATASET="tshiamor/mcx-card-pizero"
HF_MODEL_REPO="tshiamor/pizero-mcx-card"
WORK_DIR="${HOME}/pizero-training"
DATASET_DIR="${WORK_DIR}/dataset"
OUTPUT_DIR="${WORK_DIR}/outputs"
NUM_EPOCHS=100
BATCH_SIZE=32
LEARNING_RATE="1e-4"
NUM_GPUS=4

# Pi-Zero specific settings
ACTION_HORIZON=16
OBSERVATION_HORIZON=2
CHUNK_SIZE=16

echo "============================================="
echo "Pi-Zero (LeRobot) Training for MCX Card Demos"
echo "============================================="
echo "Dataset: ${HF_DATASET}"
echo "Output model: ${HF_MODEL_REPO}"
echo "Action horizon: ${ACTION_HORIZON}"
echo "GPUs: ${NUM_GPUS}"
echo ""

# ---- Check prerequisites ----
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN environment variable not set."
    echo "  export HF_TOKEN='your_huggingface_token'"
    exit 1
fi

# Check GPU
if ! nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Are you on a GPU instance?"
    exit 1
fi

echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected ${GPU_COUNT} GPUs"
echo ""

# ---- Step 1: Install system dependencies + conda ----
echo "[Step 1/6] Installing system dependencies..."
sudo apt-get update -qq || echo "  Warning: apt-get update had errors"
sudo apt-get install -y -qq git git-lfs curl wget ffmpeg libgl1-mesa-glx cmake build-essential libegl1-mesa-dev > /dev/null 2>&1 || true
git lfs install

# Install Miniconda if not present
if [ ! -d "${HOME}/miniconda3" ]; then
    echo "  Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/Miniconda3.sh
    bash /tmp/Miniconda3.sh -b -p "${HOME}/miniconda3"
fi

# Activate conda (always do this to ensure it's in PATH)
echo "  Activating conda..."
eval "$(${HOME}/miniconda3/bin/conda shell.bash hook)"
echo "  Conda version: $(conda --version)"

# Accept Conda ToS (required for non-interactive use)
echo "  Accepting Conda Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# ---- Step 2: Create conda environment ----
echo "[Step 2/6] Setting up conda environment..."
ENV_NAME="pizero"

if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "  Creating ${ENV_NAME} environment..."
    conda create -n ${ENV_NAME} python=3.10 -y --override-channels -c conda-forge
fi

conda activate ${ENV_NAME}

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q

# Clone and install LeRobot
mkdir -p "${WORK_DIR}"
if [ ! -d "${WORK_DIR}/lerobot" ]; then
    echo "  Cloning LeRobot..."
    git clone https://github.com/huggingface/lerobot.git "${WORK_DIR}/lerobot"
fi

cd "${WORK_DIR}/lerobot"
pip install -e ".[all]" -q

# Additional dependencies
pip install huggingface_hub wandb imageio[ffmpeg] -q

echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"

# ---- Step 3: Download dataset from HuggingFace ----
echo "[Step 3/6] Downloading dataset from HuggingFace..."

export DATASET_DIR HF_DATASET
python - <<'PYEOF'
import os
from huggingface_hub import snapshot_download

dataset_dir = os.environ.get("DATASET_DIR", os.path.expanduser("~/pizero-training/dataset"))
hf_dataset = os.environ.get("HF_DATASET", "tshiamor/mcx-card-pizero")

print(f"Downloading {hf_dataset} to {dataset_dir}...")
snapshot_download(
    repo_id=hf_dataset,
    repo_type="dataset",
    local_dir=dataset_dir,
)
print(f"Dataset downloaded to {dataset_dir}")
PYEOF

echo "  Dataset contents:"
ls -la "${DATASET_DIR}/"

# ---- Step 4: Create training configuration ----
echo "[Step 4/6] Creating training configuration..."

mkdir -p "${WORK_DIR}/configs"

# Create training config for Pi-Zero style policy (using ACT as base)
cat > "${WORK_DIR}/configs/mcx_card_pizero.yaml" << 'CONFIGEOF'
# Pi-Zero style training config for MCX Card manipulation
# Based on ACT (Action Chunking Transformer) architecture

seed: 42
dataset_repo_id: tshiamor/mcx-card-pizero

training:
  offline_steps: 100000
  online_steps: 0
  eval_freq: 10000
  save_freq: 25000
  log_freq: 100
  save_checkpoint: true

  batch_size: 32
  lr: 1e-4
  lr_backbone: 1e-5
  weight_decay: 1e-4
  grad_clip_norm: 10

  # Data augmentation
  image_transforms:
    enable: true
    brightness:
      weight: 0.3
      min_max: [0.8, 1.2]
    contrast:
      weight: 0.3
      min_max: [0.8, 1.2]
    saturation:
      weight: 0.3
      min_max: [0.8, 1.2]
    hue:
      weight: 0.1
      min_max: [-0.05, 0.05]

policy:
  name: act

  # Vision encoder
  vision_backbone: resnet18
  pretrained_backbone: true

  # Transformer config
  n_obs_steps: 2
  chunk_size: 16
  n_action_steps: 16

  dim_model: 512
  n_heads: 8
  n_encoder_layers: 4
  n_decoder_layers: 1

  # Input normalization
  input_normalization_modes:
    observation.images.wrist_rgb: mean_std
    observation.images.table_rgb: mean_std
    observation.state: mean_std
    action: mean_std

eval:
  n_episodes: 10
  batch_size: 10
  use_async_envs: false
CONFIGEOF

# ---- Step 5: Run training ----
echo "[Step 5/6] Starting training..."

cd "${WORK_DIR}/lerobot"

# Set environment variables
export WANDB_PROJECT="pizero-mcx-card"
export DATA_DIR="${DATASET_DIR}"
export OUTPUT_DIR="${OUTPUT_DIR}"

# Run training with LeRobot
python lerobot/scripts/train.py \
    --config-path="${WORK_DIR}/configs" \
    --config-name=mcx_card_pizero \
    hydra.run.dir="${OUTPUT_DIR}" \
    training.offline_steps=100000 \
    training.batch_size=${BATCH_SIZE} \
    training.lr=${LEARNING_RATE} \
    policy.chunk_size=${CHUNK_SIZE} \
    policy.n_action_steps=${ACTION_HORIZON} \
    wandb.enable=true \
    wandb.project="pizero-mcx-card" \
    2>&1 | tee "${WORK_DIR}/training.log"

# ---- Step 6: Upload to HuggingFace ----
echo "[Step 6/6] Uploading trained model to HuggingFace..."

export OUTPUT_DIR HF_MODEL_REPO
python - <<'UPLOADEOF'
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

output_dir = Path(os.environ.get("OUTPUT_DIR", "~/pizero-training/outputs")).expanduser()
repo_id = os.environ.get("HF_MODEL_REPO", "tshiamor/pizero-mcx-card")

api = HfApi()

# Create model repo
try:
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    print(f"Repository created/verified: {repo_id}")
except Exception as e:
    print(f"Note: {e}")

# Find the latest checkpoint
checkpoints = sorted(output_dir.glob("checkpoint*"))
if checkpoints:
    latest_ckpt = checkpoints[-1]
    print(f"Uploading checkpoint: {latest_ckpt}")
else:
    latest_ckpt = output_dir
    print(f"Uploading output directory: {output_dir}")

# Upload model
api.upload_folder(
    folder_path=str(latest_ckpt),
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload Pi-Zero (ACT) model for MCX Card manipulation",
)
print(f"Model uploaded: https://huggingface.co/{repo_id}")
UPLOADEOF

echo ""
echo "============================================="
echo "TRAINING COMPLETE"
echo "============================================="
echo "Dataset: https://huggingface.co/datasets/${HF_DATASET}"
echo "Model: https://huggingface.co/${HF_MODEL_REPO}"
echo ""
echo "To use the model with LeRobot:"
echo "  python lerobot/scripts/eval.py \\"
echo "    --pretrained-policy-path=${HF_MODEL_REPO}"
echo "============================================="
