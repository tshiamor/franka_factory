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

set -eo pipefail

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
sudo apt-get install -y git git-lfs curl wget ffmpeg libgl1-mesa-glx cmake build-essential 2>/dev/null || true

# Install git-lfs explicitly
echo "  Setting up git-lfs..."
if ! command -v git-lfs &>/dev/null; then
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash 2>/dev/null || true
    sudo apt-get install -y git-lfs 2>/dev/null || true
fi
git lfs install 2>/dev/null || true

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

# ---- Step 2: Create fresh conda environment with CUDA ----
echo "[Step 2/5] Setting up conda environment with CUDA toolkit..."
ENV_NAME="groot"

# Remove old environment if it exists (for clean install)
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  Removing existing ${ENV_NAME} environment for clean install..."
    conda deactivate 2>/dev/null || true
    conda env remove -n ${ENV_NAME} -y
fi

echo "  Creating ${ENV_NAME} environment with Python 3.10 and CUDA toolkit..."
# Install CUDA toolkit via conda - this ensures nvcc is available and matches
conda create -n ${ENV_NAME} python=3.10 cuda-toolkit=12.1 -y -c nvidia -c conda-forge

# Re-initialize conda (needed after env creation on some systems)
eval "$(${HOME}/miniconda3/bin/conda shell.bash hook)"

# Set NVCC_PREPEND_FLAGS to avoid unbound variable error in CUDA activation script
export NVCC_PREPEND_FLAGS=""

conda activate ${ENV_NAME}

# Verify CUDA toolkit
echo "  Verifying CUDA toolkit..."
which nvcc && nvcc --version

# Set CUDA environment
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

echo "  CUDA_HOME: ${CUDA_HOME}"

# ---- Step 3: Install PyTorch and Flash Attention ----
echo "[Step 3/5] Installing PyTorch and Flash Attention..."

# Install PyTorch with CUDA 12.1 (matching conda cuda-toolkit)
echo "  Installing PyTorch 2.5.1 with CUDA 12.1..."
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch
python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# Install ninja for faster compilation
pip install ninja packaging wheel setuptools

# Install Flash Attention (now nvcc is available from conda)
echo "  Installing Flash Attention..."
MAX_JOBS=4 pip install flash-attn --no-build-isolation

# Verify Flash Attention
python -c "import flash_attn; print(f'  Flash Attention {flash_attn.__version__} installed successfully!')"

# ---- Step 4: Install LeRobot and dependencies ----
echo "[Step 4/5] Installing LeRobot..."

mkdir -p "${WORK_DIR}"
if [ ! -d "${WORK_DIR}/lerobot" ]; then
    echo "  Cloning LeRobot..."
    git clone https://github.com/huggingface/lerobot.git "${WORK_DIR}/lerobot"
fi

cd "${WORK_DIR}/lerobot"

# Install LeRobot WITHOUT dependencies (to avoid PyTorch override)
echo "  Installing LeRobot (without overriding dependencies)..."
pip install -e . --no-deps

# Install LeRobot dependencies manually (excluding torch which we already have)
echo "  Installing LeRobot dependencies..."
pip install \
    "transformers>=4.40.0,<4.50.0" \
    "accelerate>=0.26.0" \
    "huggingface_hub>=0.34.0,<1.0.0" \
    "safetensors>=0.4.0" \
    "einops>=0.8.0" \
    "timm>=1.0.0" \
    "pillow>=10.0.0" \
    "numpy>=1.24.0,<2.0.0" \
    "pyarrow>=15.0.0" \
    "datasets>=2.19.0" \
    "imageio>=2.34.0" \
    "imageio-ffmpeg>=0.4.9" \
    "pyyaml>=6.0" \
    "tqdm>=4.66.0" \
    "draccus>=0.10.0" \
    "omegaconf>=2.3.0" \
    "termcolor>=2.4.0" \
    "pyserial>=3.5" \
    "diffusers>=0.27.0" \
    "opencv-python-headless>=4.9.0" \
    "av>=15.0.0" \
    "gymnasium>=1.1.0" \
    "jsonlines>=4.0.0" \
    "deepdiff>=7.0.1" \
    wandb

# Final verification
echo ""
echo "  ===== Environment Verification ====="
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  Flash Attention: $(python -c 'import flash_attn; print(flash_attn.__version__)')"
echo "  LeRobot: $(python -c 'import lerobot; print(lerobot.__version__)')"
echo "  ====================================="
echo ""

# ---- Step 5: Verify dataset ----
echo "[Step 5/5] Verifying dataset and starting training..."

python - <<'PYEOF'
import os
from huggingface_hub import HfApi

hf_dataset = os.environ.get("HF_DATASET", "tshiamor/mcx-card-pizero")
api = HfApi()

try:
    info = api.dataset_info(hf_dataset)
    print(f"Dataset found: {hf_dataset}")
    print(f"  This is the LeRobot v3.0 format dataset")
except Exception as e:
    print(f"Warning: Could not verify dataset {hf_dataset}: {e}")
PYEOF

# Clear cache to ensure fresh dataset
rm -rf ~/.cache/huggingface/lerobot/tshiamor 2>/dev/null || true

# Set environment
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HUB_ENABLE_HF_TRANSFER=1

# ---- Run GR00T fine-tuning ----
echo ""
echo "Starting GR00T N1.5-3B fine-tuning..."
echo ""

cd "${WORK_DIR}/lerobot"

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

# ---- Upload to HuggingFace ----
echo ""
echo "Uploading trained model to HuggingFace..."

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
