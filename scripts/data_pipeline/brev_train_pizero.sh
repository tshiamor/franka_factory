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
BATCH_SIZE=8  # Conservative for Pi-Zero (3.5B params) on A100 40GB
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

# Clone and install LeRobot (it will install compatible PyTorch automatically)
mkdir -p "${WORK_DIR}"
if [ ! -d "${WORK_DIR}/lerobot" ]; then
    echo "  Cloning LeRobot..."
    git clone https://github.com/huggingface/lerobot.git "${WORK_DIR}/lerobot"
fi

cd "${WORK_DIR}/lerobot"

# Install LeRobot with all dependencies (includes PyTorch with CUDA)
echo "  Installing LeRobot (this includes PyTorch)..."
pip install -e ".[all]" -q 2>&1 | grep -v "^ERROR:" || true

# Additional dependencies
pip install huggingface_hub wandb imageio[ffmpeg] -q

# Install compatible transformers version for Pi-Zero
pip install "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi" -q

echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"

# ---- Step 3: Verify dataset on HuggingFace ----
echo "[Step 3/6] Verifying dataset on HuggingFace..."

# LeRobot will download the dataset automatically during training
# Just verify it exists
python - <<'PYEOF'
import os
from huggingface_hub import HfApi

hf_dataset = os.environ.get("HF_DATASET", "tshiamor/mcx-card-pizero")
api = HfApi()

try:
    info = api.dataset_info(hf_dataset)
    print(f"Dataset found: {hf_dataset}")
    print(f"  Size: {info.cardData.get('size_categories', 'unknown') if info.cardData else 'unknown'}")
except Exception as e:
    print(f"Warning: Could not verify dataset {hf_dataset}: {e}")
    print("Training will attempt to download it anyway...")
PYEOF

# ---- Step 4: Prepare training ----
echo "[Step 4/6] Preparing training..."

# Remove existing output directory (LeRobot won't overwrite)
rm -rf "${OUTPUT_DIR}"

# ---- Step 5: Run training ----
echo "[Step 5/6] Starting training..."

cd "${WORK_DIR}/lerobot"

# Set environment variables
export WANDB_PROJECT="pizero-mcx-card"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONPATH="${WORK_DIR}/lerobot:${PYTHONPATH:-}"

# Verify lerobot is importable
echo "  Verifying LeRobot installation..."
python -c "import lerobot; print(f'LeRobot version: {lerobot.__version__}')" || {
    echo "  LeRobot not importable, reinstalling..."
    pip install -e . --no-deps -q
}

# Clear any cached old dataset versions (both lerobot and hub caches)
rm -rf ~/.cache/huggingface/lerobot/tshiamor 2>/dev/null || true
rm -rf ~/.cache/huggingface/hub/datasets--tshiamor--mcx-card-pizero 2>/dev/null || true

# Run training with LeRobot Pi-Zero (fine-tuning pre-trained VLA model)
python -m lerobot.scripts.lerobot_train \
    --policy.type pi0 \
    --policy.pretrained_path lerobot/pi0_base \
    --dataset.repo_id "${HF_DATASET}" \
    --dataset.revision main \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size ${BATCH_SIZE} \
    --steps 30000 \
    --save_freq 5000 \
    --log_freq 100 \
    --policy.chunk_size ${CHUNK_SIZE} \
    --policy.n_action_steps ${ACTION_HORIZON} \
    --wandb.enable false \
    2>&1 | tee "${WORK_DIR}/training.log"

# ---- Step 6: Upload to HuggingFace ----
echo "[Step 6/6] Uploading trained model to HuggingFace..."

export HF_MODEL_REPO
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
