#!/usr/bin/env bash
# =============================================================================
# OpenVLA Fine-tuning Setup for MCX Card Demos (NVIDIA Brev)
# =============================================================================
#
# Fine-tunes OpenVLA 7B on your MCX card manipulation dataset.
#
# Requirements:
#   - GPU with 40GB+ VRAM (A100/H100 recommended)
#   - Flash Attention support
#
# Usage:
#   export HF_TOKEN="your_huggingface_token"
#   bash brev_train_openvla.sh
#
# =============================================================================

set -eo pipefail

# ---- Configuration ----
HF_DATASET="tshiamor/mcx-card-openvla"
HF_MODEL_REPO="tshiamor/openvla-mcx-card"
BASE_MODEL="openvla/openvla-7b"
WORK_DIR="${HOME}/openvla-training"
OUTPUT_DIR="${WORK_DIR}/checkpoints"
DATASET_DIR="${WORK_DIR}/dataset"
NUM_EPOCHS=10
BATCH_SIZE=8  # A100 80GB can handle batch size 8
LEARNING_RATE="2e-5"

echo "============================================="
echo "OpenVLA Fine-tuning for MCX Card Demos"
echo "============================================="
echo "Dataset: ${HF_DATASET}"
echo "Base model: ${BASE_MODEL}"
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
ENV_NAME="openvla"

# Remove old environment if it exists (for clean install)
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  Removing existing ${ENV_NAME} environment for clean install..."
    conda deactivate 2>/dev/null || true
    conda env remove -n ${ENV_NAME} -y
fi

echo "  Creating ${ENV_NAME} environment with Python 3.10 and CUDA toolkit..."
conda create -n ${ENV_NAME} python=3.10 cuda-toolkit=12.1 -y -c nvidia -c conda-forge

# Re-initialize conda
eval "$(${HOME}/miniconda3/bin/conda shell.bash hook)"

# Set NVCC_PREPEND_FLAGS to avoid unbound variable error
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

# Install PyTorch with CUDA 12.1
echo "  Installing PyTorch 2.5.1 with CUDA 12.1..."
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch
python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# Install ninja for faster compilation
pip install ninja packaging wheel setuptools

# Install Flash Attention
echo "  Installing Flash Attention..."
MAX_JOBS=4 pip install flash-attn --no-build-isolation

# Verify Flash Attention
python -c "import flash_attn; print(f'  Flash Attention {flash_attn.__version__} installed successfully!')"

# ---- Step 4: Install OpenVLA and dependencies ----
echo "[Step 4/5] Installing OpenVLA and dependencies..."

mkdir -p "${WORK_DIR}"

# Clone OpenVLA (for reference, but we won't use their training script)
if [ ! -d "${WORK_DIR}/openvla" ]; then
    echo "  Cloning OpenVLA..."
    git clone https://github.com/openvla/openvla.git "${WORK_DIR}/openvla"
fi

# Install dependencies (OpenVLA-compatible versions)
pip install \
    "transformers>=4.40.0,<4.50.0" \
    "accelerate>=0.26.0" \
    "huggingface_hub>=0.20.0" \
    "hf_transfer>=0.1.6" \
    "safetensors>=0.4.0" \
    "datasets>=2.19.0" \
    "bitsandbytes>=0.42.0" \
    "scipy>=1.11.0" \
    "pillow>=10.0.0" \
    "numpy>=1.24.0,<2.0.0" \
    "tqdm>=4.66.0" \
    "peft>=0.10.0" \
    "timm>=0.9.10,<1.0.0" \
    wandb

# Final verification
echo ""
echo "  ===== Environment Verification ====="
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  Flash Attention: $(python -c 'import flash_attn; print(flash_attn.__version__)')"
echo "  timm: $(python -c 'import timm; print(timm.__version__)')"
echo "  ====================================="
echo ""

# ---- Step 5: Download dataset and run training ----
echo "[Step 5/5] Downloading dataset and starting training..."

# Set environment
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HUB_ENABLE_HF_TRANSFER=1

# Create training script
cat > "${WORK_DIR}/train_openvla.py" << 'TRAINEOF'
#!/usr/bin/env python3
"""
Fine-tune OpenVLA on MCX Card manipulation dataset (numpy file format).
"""

import os
import torch
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm
import wandb


class OpenVLANumpyDataset(Dataset):
    """Dataset loader for OpenVLA numpy file format."""

    def __init__(self, data_dir, processor):
        self.data_dir = Path(data_dir)
        self.processor = processor

        # Find all episode directories
        self.episodes = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and d.name.startswith("episode_")
        ])
        print(f"Found {len(self.episodes)} episodes in {data_dir}")

        # Build sample index (episode_idx, step_idx)
        self.samples = []
        for ep_idx, ep_dir in enumerate(self.episodes):
            images_path = ep_dir / "images.npy"
            if images_path.exists():
                images = np.load(images_path, mmap_mode='r')
                num_steps = len(images)
                for t in range(num_steps):
                    self.samples.append((ep_idx, t))

        print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_idx, t = self.samples[idx]
        ep_dir = self.episodes[ep_idx]

        # Load image
        images = np.load(ep_dir / "images.npy", mmap_mode='r')
        image = images[t]  # (H, W, 3) uint8

        # Load action
        actions_path = ep_dir / "actions.npy"
        if actions_path.exists():
            actions = np.load(actions_path)
            action = actions[t] if t < len(actions) else np.zeros(7, dtype=np.float32)
        else:
            action = np.zeros(7, dtype=np.float32)

        # Load language instruction
        metadata_path = ep_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                meta = json.load(f)
            instruction = meta.get("language_instruction",
                "Pick up the blue block and place it on the target")
        else:
            instruction = "Pick up the blue block and place it on the target"

        # Convert to PIL
        pil_image = Image.fromarray(image)

        return {
            "image": pil_image,
            "instruction": instruction,
            "action": torch.tensor(action, dtype=torch.float32),
        }


def collate_fn(batch, processor):
    """Collate batch for OpenVLA."""
    images = [item["image"] for item in batch]
    instructions = [item["instruction"] for item in batch]
    actions = torch.stack([item["action"] for item in batch])

    inputs = processor(
        images=images,
        text=instructions,
        return_tensors="pt",
        padding=True,
    )

    inputs["labels"] = inputs["input_ids"].clone()
    inputs["actions"] = actions

    return inputs


def main():
    # Configuration
    hf_dataset = os.environ.get("HF_DATASET", "tshiamor/mcx-card-openvla")
    output_dir = Path(os.environ.get("OUTPUT_DIR", "~/openvla-training/checkpoints")).expanduser()
    dataset_dir = Path(os.environ.get("DATASET_DIR", "~/openvla-training/dataset")).expanduser()
    base_model = os.environ.get("BASE_MODEL", "openvla/openvla-7b")
    num_epochs = int(os.environ.get("NUM_EPOCHS", "10"))
    batch_size = int(os.environ.get("BATCH_SIZE", "8"))
    learning_rate = float(os.environ.get("LEARNING_RATE", "2e-5"))

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Base model: {base_model}")
    print(f"Dataset: {hf_dataset}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {num_epochs}, Batch size: {batch_size}, LR: {learning_rate}")

    # Initialize wandb
    if os.environ.get("WANDB_MODE") != "offline":
        wandb.init(project="openvla-mcx-card", name="finetune")

    # Download dataset
    print(f"Downloading dataset {hf_dataset}...")
    snapshot_download(
        repo_id=hf_dataset,
        repo_type="dataset",
        local_dir=dataset_dir,
    )
    print(f"Dataset downloaded to {dataset_dir}")

    # Load model and processor
    print("Loading OpenVLA model...")
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Load dataset
    print("Loading dataset from numpy files...")
    train_dataset = OpenVLANumpyDataset(dataset_dir / "train", processor)

    if len(train_dataset) == 0:
        raise ValueError(f"No samples found in {dataset_dir / 'train'}. Check dataset structure.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda b: collate_fn(b, processor),
        pin_memory=True,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(0.05 * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Training loop
    print(f"Starting training: {len(train_dataset)} samples, {len(train_loader)} batches/epoch")
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Convert pixel_values to bfloat16 to match model dtype
            pixel_values = batch.get("pixel_values").to(torch.bfloat16)

            outputs = model(
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
                pixel_values=pixel_values,
                labels=batch.get("labels"),
            )

            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

            if global_step % 100 == 0 and os.environ.get("WANDB_MODE") != "offline":
                wandb.log({
                    "train_loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "step": global_step,
                })

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}: avg_train_loss={avg_train_loss:.4f}")

        if (epoch + 1) % 2 == 0:
            ckpt_dir = output_dir / f"checkpoint-epoch-{epoch+1}"
            model.save_pretrained(ckpt_dir)
            processor.save_pretrained(ckpt_dir)
            print(f"  Saved checkpoint: {ckpt_dir}")

    # Save final model
    print("Saving final model...")
    final_dir = output_dir / "final"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)

    print("Training complete!")
    if os.environ.get("WANDB_MODE") != "offline":
        wandb.finish()


if __name__ == "__main__":
    main()
TRAINEOF

# Run training
echo ""
echo "Starting OpenVLA fine-tuning..."
echo ""

export HF_DATASET OUTPUT_DIR DATASET_DIR BASE_MODEL NUM_EPOCHS BATCH_SIZE LEARNING_RATE

python "${WORK_DIR}/train_openvla.py" 2>&1 | tee "${WORK_DIR}/training.log"

# ---- Upload to HuggingFace ----
echo ""
echo "Uploading trained model to HuggingFace..."

export HF_MODEL_REPO
python - <<'UPLOADEOF'
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

output_dir = Path(os.environ.get("OUTPUT_DIR", "~/openvla-training/checkpoints")).expanduser()
repo_id = os.environ.get("HF_MODEL_REPO", "tshiamor/openvla-mcx-card")
final_model_dir = output_dir / "final"

api = HfApi()

# Create model repo
try:
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    print(f"Repository created/verified: {repo_id}")
except Exception as e:
    print(f"Note: {e}")

# Upload model
if final_model_dir.exists():
    print(f"Uploading {final_model_dir} to {repo_id}...")
    api.upload_folder(
        folder_path=str(final_model_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload fine-tuned OpenVLA model for MCX Card manipulation",
    )
    print(f"Model uploaded: https://huggingface.co/{repo_id}")
else:
    print(f"Warning: {final_model_dir} not found")
UPLOADEOF

echo ""
echo "============================================="
echo "TRAINING COMPLETE"
echo "============================================="
echo "Base model: ${BASE_MODEL} (7B parameters)"
echo "Dataset: https://huggingface.co/datasets/${HF_DATASET}"
echo "Fine-tuned model: https://huggingface.co/${HF_MODEL_REPO}"
echo ""
echo "To use the model:"
echo "  from transformers import AutoModelForVision2Seq, AutoProcessor"
echo "  model = AutoModelForVision2Seq.from_pretrained('${HF_MODEL_REPO}')"
echo "============================================="
