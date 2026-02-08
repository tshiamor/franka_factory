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
sudo apt-get install -y -qq git git-lfs curl wget cmake build-essential > /dev/null 2>&1 || true
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

Dataset structure (from prepare_openvla.py):
    train/episode_XXXX/images.npy      - (T, 224, 224, 3) uint8
    train/episode_XXXX/actions.npy     - (T, 7) float32
    train/episode_XXXX/metadata.json   - {language_instruction, num_steps, ...}
    train/episode_XXXX/eef_pos.npy     - (T, 3) float32 (optional)
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
)
from tqdm import tqdm
import wandb


class OpenVLADataset(Dataset):
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
            metadata_path = ep_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    meta = json.load(f)
                num_steps = meta.get("num_steps", 0)
            else:
                # Fallback: check images.npy shape
                images_path = ep_dir / "images.npy"
                if images_path.exists():
                    images = np.load(images_path, mmap_mode='r')
                    num_steps = len(images)
                else:
                    continue

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
        image = images[t]  # (224, 224, 3) uint8

        # Load action
        actions = np.load(ep_dir / "actions.npy")
        action = actions[t] if t < len(actions) else np.zeros(7, dtype=np.float32)

        # Load language instruction
        metadata_path = ep_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                meta = json.load(f)
            instruction = meta.get("language_instruction",
                "Pick up the blue block and place it on the target")
        else:
            instruction = "Pick up the blue block and place it on the target"

        # Convert image to PIL for processor
        pil_image = Image.fromarray(image)

        # Process with OpenVLA processor
        inputs = self.processor(
            images=pil_image,
            text=instruction,
            return_tensors="pt",
            padding=True,
        )

        # Flatten batch dimension (processor adds it)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Add action as label
        inputs["labels"] = torch.tensor(action, dtype=torch.float32)

        return inputs


def collate_fn(batch):
    """Custom collate function for OpenVLA batches."""
    # Stack all tensors
    result = {}
    for key in batch[0].keys():
        if key == "labels":
            result[key] = torch.stack([item[key] for item in batch])
        elif isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([item[key] for item in batch])
        else:
            result[key] = [item[key] for item in batch]
    return result


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
    print("Loading OpenVLA model...")
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Enable gradient checkpointing to save memory
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Load datasets
    print("Loading dataset...")
    train_dataset = OpenVLADataset(dataset_dir / "train", processor)
    val_dataset = OpenVLADataset(dataset_dir / "val", processor) if (dataset_dir / "val").exists() else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    ) if val_dataset else None

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    num_training_steps = len(train_loader) * num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps)

    # Training loop
    print("Starting training...")
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Forward pass
            outputs = model(
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
                pixel_values=batch.get("pixel_values"),
                labels=batch.get("input_ids"),  # For language modeling
            )

            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            pbar.set_postfix({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})

            if global_step % 100 == 0:
                wandb.log({
                    "train_loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "step": global_step,
                })

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}: avg_train_loss={avg_train_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "avg_train_loss": avg_train_loss})

        # Validation
        if val_loader and (epoch + 1) % 2 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                    outputs = model(
                        input_ids=batch.get("input_ids"),
                        attention_mask=batch.get("attention_mask"),
                        pixel_values=batch.get("pixel_values"),
                        labels=batch.get("input_ids"),
                    )
                    val_loss += outputs.loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"  val_loss={avg_val_loss:.4f}")
            wandb.log({"val_loss": avg_val_loss})

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model.save_pretrained(output_dir / "best")
                processor.save_pretrained(output_dir / "best")
                print(f"  Saved best model")

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            model.save_pretrained(output_dir / f"checkpoint-{epoch+1}")
            processor.save_pretrained(output_dir / f"checkpoint-{epoch+1}")

    # Save final model
    print("Saving final model...")
    model.save_pretrained(output_dir / "final")
    processor.save_pretrained(output_dir / "final")

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
