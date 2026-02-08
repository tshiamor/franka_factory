#!/usr/bin/env bash
# =============================================================================
# NVIDIA GR00T N1.5 Fine-tuning via LeRobot (NVIDIA Brev)
# =============================================================================
#
# Fine-tunes the real NVIDIA GR00T N1.5-3B foundation model on your dataset.
# Uses LeRobot's GR00T integration with the pre-trained nvidia/GR00T-N1.5-3B.
#
# Requirements:
#   - GPU with 40GB+ VRAM (A100/H100 recommended)
#   - Flash Attention support
#
# Usage:
#   export HF_TOKEN="your_huggingface_token"
#   bash brev_train_groot.sh
#
# =============================================================================

set -euo pipefail

# ---- Configuration ----
HF_DATASET="tshiamor/mcx-card-pizero"  # LeRobot v3.0 format dataset
HF_MODEL_REPO="tshiamor/groot-n15-mcx-card"
BASE_MODEL="nvidia/GR00T-N1.5-3B"
WORK_DIR="${HOME}/groot-training"
OUTPUT_DIR="${WORK_DIR}/outputs"
BATCH_SIZE=4  # GR00T 3B is large, conservative batch size
STEPS=30000
SAVE_FREQ=5000
LOG_FREQ=100

echo "============================================="
echo "GR00T N1.5-3B Fine-tuning for MCX Card Demos"
echo "============================================="
echo "Base model: ${BASE_MODEL}"
echo "Dataset: ${HF_DATASET}"
echo "Output model: ${HF_MODEL_REPO}"
echo "Batch size: ${BATCH_SIZE}"
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
echo "[Step 1/5] Installing system dependencies..."
sudo apt-get update -qq || echo "  Warning: apt-get update had errors"
sudo apt-get install -y git git-lfs curl wget ffmpeg libgl1-mesa-glx cmake build-essential libegl1-mesa-dev 2>/dev/null || true

# Install git-lfs explicitly
echo "  Setting up git-lfs..."
if ! command -v git-lfs &>/dev/null; then
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash 2>/dev/null || true
    sudo apt-get install -y git-lfs 2>/dev/null || true
fi
git lfs install 2>/dev/null || true

# Set up CUDA environment (required for flash-attn)
echo "  Detecting CUDA..."
if [ -z "${CUDA_HOME:-}" ]; then
    # Common CUDA locations
    for cuda_path in /usr/local/cuda /usr/local/cuda-12.1 /usr/local/cuda-12.2 /usr/local/cuda-12.3 /usr/local/cuda-12.4 /usr/lib/cuda; do
        if [ -d "$cuda_path" ]; then
            export CUDA_HOME="$cuda_path"
            break
        fi
    done
fi

if [ -n "${CUDA_HOME:-}" ]; then
    echo "  CUDA_HOME: ${CUDA_HOME}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
else
    echo "  Warning: CUDA not found, trying to install CUDA toolkit..."
    # Install CUDA toolkit for compilation
    sudo apt-get install -y nvidia-cuda-toolkit 2>/dev/null || true
    export CUDA_HOME=/usr
fi

# Install Miniconda if not present
if [ ! -d "${HOME}/miniconda3" ]; then
    echo "  Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/Miniconda3.sh
    bash /tmp/Miniconda3.sh -b -p "${HOME}/miniconda3"
fi

# Activate conda
echo "  Activating conda..."
eval "$(${HOME}/miniconda3/bin/conda shell.bash hook)"
echo "  Conda version: $(conda --version)"

# Accept Conda ToS
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# ---- Step 2: Create conda environment ----
echo "[Step 2/5] Setting up conda environment..."
ENV_NAME="groot"

if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "  Creating ${ENV_NAME} environment..."
    conda create -n ${ENV_NAME} python=3.10 -y --override-channels -c conda-forge
fi

conda activate ${ENV_NAME}

# Install PyTorch with CUDA 12.1 (matches flash-attn wheels)
echo "  Installing PyTorch with CUDA 12.1..."
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 -q

# Clone and install LeRobot FIRST (without groot extras)
mkdir -p "${WORK_DIR}"
if [ ! -d "${WORK_DIR}/lerobot" ]; then
    echo "  Cloning LeRobot..."
    git clone https://github.com/huggingface/lerobot.git "${WORK_DIR}/lerobot"
fi

cd "${WORK_DIR}/lerobot"

# Install base LeRobot first
echo "  Installing LeRobot (base)..."
pip install -e . -q 2>&1 || pip install -e . --no-build-isolation -q

# Install Flash Attention (required for GR00T)
echo "  Installing Flash Attention (required for GR00T)..."
pip install ninja packaging -q

# Try pre-built wheel first (much faster), fall back to source build
echo "  Attempting to install pre-built flash-attn wheel..."
pip install flash-attn --no-build-isolation -q 2>&1 || {
    echo "  Pre-built wheel failed, trying source build..."
    # Ensure CUDA is available for source build
    if [ -n "${CUDA_HOME:-}" ] && [ -f "${CUDA_HOME}/bin/nvcc" ]; then
        MAX_JOBS=4 pip install flash-attn --no-build-isolation -q 2>&1 || {
            echo "  WARNING: Flash Attention installation failed!"
            echo "  GR00T requires Flash Attention. Trying one more method..."
            pip install "flash-attn>=2.5.9,<3.0.0" --no-build-isolation 2>&1 || true
        }
    else
        echo "  WARNING: nvcc not found at ${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
        echo "  Flash Attention requires CUDA toolkit with nvcc."
        echo "  Install with: sudo apt-get install nvidia-cuda-toolkit"
    fi
}

# Install GR00T-specific dependencies
echo "  Installing GR00T dependencies..."
pip install "transformers>=4.40.0" "accelerate>=0.26.0" einops timm -q

# Additional dependencies
pip install huggingface_hub wandb -q

echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Verify flash attention
python -c "import flash_attn; print(f'  Flash Attention {flash_attn.__version__} ready')" 2>/dev/null || {
    echo "  WARNING: Flash Attention not working!"
    echo "  GR00T training will likely fail without Flash Attention."
    echo ""
    echo "  To fix manually on Brev:"
    echo "    1. Ensure CUDA toolkit is installed: sudo apt-get install nvidia-cuda-toolkit"
    echo "    2. Set CUDA_HOME: export CUDA_HOME=/usr/local/cuda"
    echo "    3. Reinstall: pip install flash-attn --no-build-isolation"
    echo ""
    read -p "  Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
}

# ---- Step 3: Verify dataset ----
echo "[Step 3/5] Verifying dataset..."

python - <<'PYEOF'
import os
from huggingface_hub import HfApi

hf_dataset = os.environ.get("HF_DATASET", "tshiamor/mcx-card-pizero")
api = HfApi()

try:
    info = api.dataset_info(hf_dataset)
    print(f"Dataset found: {hf_dataset}")
    print(f"  This is the LeRobot v3.0 format dataset used for Pi-Zero")
    print(f"  GR00T will use the same dataset format")
except Exception as e:
    print(f"Warning: Could not verify dataset {hf_dataset}: {e}")
PYEOF

# ---- Step 4: Run GR00T fine-tuning ----
echo "[Step 4/5] Starting GR00T N1.5-3B fine-tuning..."

cd "${WORK_DIR}/lerobot"

# Clear cache to ensure fresh dataset
rm -rf ~/.cache/huggingface/lerobot/tshiamor 2>/dev/null || true

# Set environment
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HUB_ENABLE_HF_TRANSFER=1

# Run training with LeRobot
# GR00T uses similar interface to Pi-Zero
python -m lerobot.scripts.lerobot_train \
    --policy.type groot \
    --policy.base_model_path "${BASE_MODEL}" \
    --policy.tune_llm false \
    --policy.tune_visual false \
    --policy.tune_projector true \
    --policy.tune_diffusion_model true \
    --dataset.repo_id "${HF_DATASET}" \
    --dataset.revision main \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size ${BATCH_SIZE} \
    --steps ${STEPS} \
    --save_freq ${SAVE_FREQ} \
    --log_freq ${LOG_FREQ} \
    --policy.repo_id "${HF_MODEL_REPO}" \
    --wandb.enable false \
    2>&1 | tee "${WORK_DIR}/training.log"

# ---- Step 5: Upload to HuggingFace ----
echo "[Step 5/5] Uploading trained model to HuggingFace..."

export OUTPUT_DIR HF_MODEL_REPO
python - <<'UPLOADEOF'
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

output_dir = Path(os.environ.get("OUTPUT_DIR", "~/groot-training/outputs")).expanduser()
repo_id = os.environ.get("HF_MODEL_REPO", "tshiamor/groot-n15-mcx-card")

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
    commit_message="Upload fine-tuned GR00T N1.5-3B model for MCX Card manipulation",
)
print(f"Model uploaded: https://huggingface.co/{repo_id}")
UPLOADEOF

echo ""
echo "============================================="
echo "TRAINING COMPLETE"
echo "============================================="
echo "Base model: ${BASE_MODEL} (3B parameters)"
echo "Dataset: https://huggingface.co/datasets/${HF_DATASET}"
echo "Fine-tuned model: https://huggingface.co/${HF_MODEL_REPO}"
echo ""
echo "This is the REAL GR00T N1.5-3B foundation model fine-tuned on your data!"
echo "============================================="
