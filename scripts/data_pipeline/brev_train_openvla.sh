#!/usr/bin/env bash
# =============================================================================
# OpenVLA Fine-tuning Setup for MCX Card Demos (NVIDIA Brev)
# =============================================================================
#
# Run this script on a cloud instance with 4-8x A100/H100 GPUs.
# It will:
#   1. Install dependencies and conda
#   2. Clone OpenVLA repository
#   3. Pull the dataset from HuggingFace (tshiamor/mcx-card-openvla)
#   4. Fine-tune OpenVLA on the dataset
#   5. Upload trained model to HuggingFace
#
# Usage:
#   export HF_TOKEN="your_huggingface_token"
#   bash brev_train_openvla.sh
#
# =============================================================================

set -euo pipefail

# ---- Configuration ----
HF_DATASET="tshiamor/mcx-card-openvla"
HF_MODEL_REPO="tshiamor/openvla-mcx-card"
BASE_MODEL="openvla/openvla-7b"
WORK_DIR="${HOME}/openvla-training"
DATASET_DIR="${WORK_DIR}/dataset"
OUTPUT_DIR="${WORK_DIR}/checkpoints"
NUM_EPOCHS=10
BATCH_SIZE=4
LEARNING_RATE="2e-5"
NUM_GPUS=4

echo "============================================="
echo "OpenVLA Fine-tuning for MCX Card Demos"
echo "============================================="
echo "Dataset: ${HF_DATASET}"
echo "Base model: ${BASE_MODEL}"
echo "Output model: ${HF_MODEL_REPO}"
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
if [ "${GPU_COUNT}" -lt "${NUM_GPUS}" ]; then
    echo "WARNING: Found ${GPU_COUNT} GPUs but configured for ${NUM_GPUS}. Using ${GPU_COUNT} GPUs."
    NUM_GPUS="${GPU_COUNT}"
fi
echo ""

# ---- Step 1: Install system dependencies + conda ----
echo "[Step 1/6] Installing system dependencies..."
sudo apt-get update -qq || echo "  Warning: apt-get update had errors (non-critical)"
sudo apt-get install -y -qq git git-lfs curl wget > /dev/null 2>&1 || true
git lfs install

# Install Miniconda if not present
if [ ! -d "${HOME}/miniconda3" ]; then
    echo "  Installing Miniconda..."
    CONDA_INSTALLER="/tmp/Miniconda3-latest-Linux-x86_64.sh"
    if [ ! -f "${CONDA_INSTALLER}" ]; then
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "${CONDA_INSTALLER}"
    fi
    bash "${CONDA_INSTALLER}" -b -p "${HOME}/miniconda3"
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
ENV_NAME="openvla"

if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "  Creating ${ENV_NAME} environment..."
    conda create -n ${ENV_NAME} python=3.10 -y --override-channels -c conda-forge
fi

conda activate ${ENV_NAME}

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q

# Install OpenVLA dependencies
pip install transformers accelerate datasets bitsandbytes scipy -q
pip install huggingface_hub wandb -q
pip install flash-attn --no-build-isolation -q 2>/dev/null || echo "  flash-attn install may have failed (optional)"

# Clone OpenVLA
mkdir -p "${WORK_DIR}"
if [ ! -d "${WORK_DIR}/openvla" ]; then
    echo "  Cloning OpenVLA..."
    git clone https://github.com/openvla/openvla.git "${WORK_DIR}/openvla"
fi

cd "${WORK_DIR}/openvla"
pip install -e . -q

echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"

# ---- Step 3: Download dataset from HuggingFace ----
echo "[Step 3/6] Downloading dataset from HuggingFace..."
pip install huggingface_hub -q

python - <<'PYEOF'
import os
from huggingface_hub import snapshot_download

dataset_dir = os.environ.get("DATASET_DIR", os.path.expanduser("~/openvla-training/dataset"))
hf_dataset = os.environ.get("HF_DATASET", "tshiamor/mcx-card-openvla")

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

# ---- Step 4: Create training script ----
echo "[Step 4/6] Preparing training configuration..."

cat > "${WORK_DIR}/train_openvla.py" << 'TRAINEOF'
#!/usr/bin/env python3
"""
Fine-tune OpenVLA on MCX Card manipulation dataset.
Uses RLDS format data with language instructions.
"""

import os
import json
import torch
from pathlib import Path
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from datasets import load_from_disk
import wandb

def main():
    # Configuration
    dataset_dir = Path(os.environ.get("DATASET_DIR", "~/openvla-training/dataset")).expanduser()
    output_dir = Path(os.environ.get("OUTPUT_DIR", "~/openvla-training/checkpoints")).expanduser()
    base_model = os.environ.get("BASE_MODEL", "openvla/openvla-7b")
    num_epochs = int(os.environ.get("NUM_EPOCHS", "10"))
    batch_size = int(os.environ.get("BATCH_SIZE", "4"))
    learning_rate = float(os.environ.get("LEARNING_RATE", "2e-5"))

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Base model: {base_model}")
    print(f"Dataset: {dataset_dir}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {num_epochs}, Batch size: {batch_size}, LR: {learning_rate}")

    # Initialize wandb
    wandb.init(project="openvla-mcx-card", name="finetune")

    # Load model and processor
    print("Loading model...")
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    # Load dataset
    print("Loading dataset...")
    train_dataset = load_from_disk(str(dataset_dir / "train"))
    val_dataset = load_from_disk(str(dataset_dir / "val")) if (dataset_dir / "val").exists() else None

    print(f"Train samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Val samples: {len(val_dataset)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=500,
        eval_steps=500 if val_dataset else None,
        evaluation_strategy="steps" if val_dataset else "no",
        save_total_limit=3,
        bf16=True,
        dataloader_num_workers=4,
        report_to="wandb",
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    print("Saving model...")
    trainer.save_model(str(output_dir / "final"))
    processor.save_pretrained(str(output_dir / "final"))

    print("Training complete!")
    wandb.finish()

if __name__ == "__main__":
    main()
TRAINEOF

# ---- Step 5: Run training ----
echo "[Step 5/6] Starting training..."
export DATASET_DIR OUTPUT_DIR BASE_MODEL NUM_EPOCHS BATCH_SIZE LEARNING_RATE

# Use accelerate for multi-GPU training
accelerate config default --mixed_precision bf16

accelerate launch --num_processes ${NUM_GPUS} "${WORK_DIR}/train_openvla.py" 2>&1 | tee "${WORK_DIR}/training.log"

# ---- Step 6: Upload to HuggingFace ----
echo "[Step 6/6] Uploading trained model to HuggingFace..."

python - <<'UPLOADEOF'
import os
from huggingface_hub import HfApi, create_repo

output_dir = os.path.expanduser(os.environ.get("OUTPUT_DIR", "~/openvla-training/checkpoints"))
repo_id = os.environ.get("HF_MODEL_REPO", "tshiamor/openvla-mcx-card")
final_model_dir = os.path.join(output_dir, "final")

api = HfApi()

# Create model repo
try:
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    print(f"Repository created/verified: {repo_id}")
except Exception as e:
    print(f"Note: {e}")

# Upload model
print(f"Uploading {final_model_dir} to {repo_id}...")
api.upload_folder(
    folder_path=final_model_dir,
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload fine-tuned OpenVLA model for MCX Card manipulation",
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
echo "To use the model:"
echo "  from transformers import AutoModelForVision2Seq, AutoProcessor"
echo "  model = AutoModelForVision2Seq.from_pretrained('${HF_MODEL_REPO}')"
echo "============================================="
