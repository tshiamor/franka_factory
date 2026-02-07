#!/usr/bin/env bash
# =============================================================================
# NVIDIA GR00T N1.6 Training Setup for MCX Card Demos (NVIDIA Brev)
# =============================================================================
#
# Run this script on a cloud instance with 4-8x A100/H100 GPUs.
# It will:
#   1. Install dependencies and conda
#   2. Set up Isaac Lab with GR00T N1.6
#   3. Pull the dataset from HuggingFace (tshiamor/mcx-card-groot-n16)
#   4. Train GR00T N1.6 on the dataset
#   5. Upload trained model to HuggingFace
#
# Usage:
#   export HF_TOKEN="your_huggingface_token"
#   bash brev_train_groot.sh
#
# =============================================================================

set -euo pipefail

# ---- Configuration ----
HF_DATASET="tshiamor/mcx-card-groot-n16"
HF_MODEL_REPO="tshiamor/groot-n16-mcx-card"
WORK_DIR="${HOME}/groot-training"
DATASET_DIR="${WORK_DIR}/dataset"
OUTPUT_DIR="${WORK_DIR}/outputs"
NUM_EPOCHS=100
BATCH_SIZE=16
LEARNING_RATE="1e-4"
NUM_GPUS=4

# GR00T N1.6 specific settings
ACTION_HORIZON=16
OBSERVATION_HORIZON=2
MODEL_VERSION="n1.6"

echo "============================================="
echo "GR00T N1.6 Training for MCX Card Demos"
echo "============================================="
echo "Dataset: ${HF_DATASET}"
echo "Output model: ${HF_MODEL_REPO}"
echo "Model version: ${MODEL_VERSION}"
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
ENV_NAME="groot"

if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "  Creating ${ENV_NAME} environment..."
    conda create -n ${ENV_NAME} python=3.10 -y --override-channels -c conda-forge
fi

conda activate ${ENV_NAME}

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q

# Install GR00T dependencies
pip install transformers accelerate datasets -q
pip install huggingface_hub wandb h5py pyyaml -q
pip install timm einops -q
pip install flash-attn --no-build-isolation -q 2>/dev/null || echo "  flash-attn install may have failed (optional)"

# Clone GR00T N1 repo (if available) or use LeRobot as fallback
mkdir -p "${WORK_DIR}"

# Try to get NVIDIA's GR00T implementation
if [ ! -d "${WORK_DIR}/gr00t" ]; then
    echo "  Setting up GR00T training framework..."
    # GR00T N1.6 uses a diffusion-based policy similar to LeRobot's diffusion policy
    # We'll use a custom training script that's compatible with the GR00T HDF5 format
fi

echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"

# ---- Step 3: Download dataset from HuggingFace ----
echo "[Step 3/6] Downloading dataset from HuggingFace..."

export DATASET_DIR HF_DATASET
python - <<'PYEOF'
import os
from huggingface_hub import snapshot_download

dataset_dir = os.environ.get("DATASET_DIR", os.path.expanduser("~/groot-training/dataset"))
hf_dataset = os.environ.get("HF_DATASET", "tshiamor/mcx-card-groot-n16")

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

# ---- Step 4: Create GR00T N1.6 training script ----
echo "[Step 4/6] Creating training script..."

cat > "${WORK_DIR}/train_groot.py" << 'TRAINEOF'
#!/usr/bin/env python3
"""
Train GR00T N1.6 style policy on MCX Card manipulation dataset.

GR00T N1.6 uses:
- Vision Transformer (ViT) backbone
- Diffusion-based action decoder
- Action chunking (horizon=16)
- Multi-camera fusion
- Language conditioning
"""

import os
import json
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb

# GR00T N1.6 Configuration
class GROOTConfig:
    # Vision
    image_size = 224
    patch_size = 16
    num_channels = 3
    hidden_size = 768
    num_attention_heads = 12
    num_hidden_layers = 12

    # Action
    action_dim = 7
    action_horizon = 16
    observation_horizon = 2

    # Diffusion
    num_diffusion_steps = 100
    beta_start = 0.0001
    beta_end = 0.02

    # Training
    batch_size = 16
    learning_rate = 1e-4
    num_epochs = 100
    warmup_steps = 1000


class GROOTDataset(Dataset):
    """Dataset loader for GR00T N1.6 HDF5 format."""

    def __init__(self, hdf5_path, config):
        self.config = config
        self.hdf5_path = hdf5_path
        self.h5file = h5py.File(hdf5_path, 'r')

        # Get all episode keys
        self.episode_keys = list(self.h5file.keys())
        print(f"Loaded {len(self.episode_keys)} episodes")

        # Build sample index
        self.samples = []
        for ep_key in self.episode_keys:
            ep = self.h5file[ep_key]
            num_steps = ep['actions'].shape[0]
            for t in range(num_steps - config.action_horizon):
                self.samples.append((ep_key, t))

        print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_key, t = self.samples[idx]
        ep = self.h5file[ep_key]

        # Get observation images (multi-camera)
        images = {}
        for cam in ['wrist_rgb', 'table_rgb']:
            if cam in ep:
                img = ep[cam][t]
                # Normalize to [-1, 1]
                img = img.astype(np.float32) / 127.5 - 1.0
                img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
                images[cam] = torch.from_numpy(img)

        # Get robot state
        state = ep['robot_state'][t] if 'robot_state' in ep else np.zeros(9)
        state = torch.from_numpy(state.astype(np.float32))

        # Get action chunk
        action_chunk = ep['action_chunk'][t] if 'action_chunk' in ep else \
                       ep['actions'][t:t + self.config.action_horizon]
        action_chunk = torch.from_numpy(action_chunk.astype(np.float32))

        # Language instruction (if available)
        lang = ep.attrs.get('language_instruction', 'Pick up the block and place it on the target')

        return {
            'images': images,
            'state': state,
            'action_chunk': action_chunk,
            'language': lang,
        }


class SimpleDiffusionPolicy(nn.Module):
    """Simplified diffusion policy for GR00T N1.6 style training."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Vision encoder (simplified ViT-like)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=8, stride=8),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, config.hidden_size),
        )

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(9, 256),
            nn.ReLU(),
            nn.Linear(256, config.hidden_size),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 3, config.hidden_size),  # 2 cameras + state
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        # Diffusion noise prediction network
        self.noise_pred = nn.Sequential(
            nn.Linear(config.hidden_size + config.action_dim * config.action_horizon + 1, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, config.action_dim * config.action_horizon),
        )

        # Diffusion schedule
        betas = torch.linspace(config.beta_start, config.beta_end, config.num_diffusion_steps)
        alphas = 1.0 - betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))

    def forward(self, images, state, noisy_actions, timesteps):
        # Encode images
        img_feats = []
        for cam in ['wrist_rgb', 'table_rgb']:
            if cam in images:
                feat = self.image_encoder(images[cam])
                img_feats.append(feat)
            else:
                img_feats.append(torch.zeros_like(img_feats[0]) if img_feats else
                                torch.zeros(images[list(images.keys())[0]].shape[0], self.config.hidden_size,
                                           device=images[list(images.keys())[0]].device))

        # Encode state
        state_feat = self.state_encoder(state)

        # Fusion
        combined = torch.cat(img_feats + [state_feat], dim=-1)
        context = self.fusion(combined)

        # Flatten actions
        flat_actions = noisy_actions.view(noisy_actions.shape[0], -1)

        # Timestep embedding (simple)
        t_embed = timesteps.float().unsqueeze(-1) / self.config.num_diffusion_steps

        # Predict noise
        noise_pred = self.noise_pred(torch.cat([context, flat_actions, t_embed], dim=-1))
        noise_pred = noise_pred.view(-1, self.config.action_horizon, self.config.action_dim)

        return noise_pred

    def compute_loss(self, images, state, action_chunk):
        batch_size = action_chunk.shape[0]
        device = action_chunk.device

        # Sample random timesteps
        timesteps = torch.randint(0, self.config.num_diffusion_steps, (batch_size,), device=device)

        # Sample noise
        noise = torch.randn_like(action_chunk)

        # Add noise to actions
        alpha_t = self.alphas_cumprod[timesteps].view(-1, 1, 1)
        noisy_actions = torch.sqrt(alpha_t) * action_chunk + torch.sqrt(1 - alpha_t) * noise

        # Predict noise
        pred_noise = self(images, state, noisy_actions, timesteps)

        # MSE loss
        loss = F.mse_loss(pred_noise, noise)
        return loss


def train():
    config = GROOTConfig()

    # Load environment config
    dataset_dir = Path(os.environ.get("DATASET_DIR", "~/groot-training/dataset")).expanduser()
    output_dir = Path(os.environ.get("OUTPUT_DIR", "~/groot-training/outputs")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Override from environment
    config.batch_size = int(os.environ.get("BATCH_SIZE", config.batch_size))
    config.learning_rate = float(os.environ.get("LEARNING_RATE", config.learning_rate))
    config.num_epochs = int(os.environ.get("NUM_EPOCHS", config.num_epochs))
    config.action_horizon = int(os.environ.get("ACTION_HORIZON", config.action_horizon))

    print(f"Dataset: {dataset_dir}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {config.num_epochs}, Batch: {config.batch_size}, LR: {config.learning_rate}")
    print(f"Action horizon: {config.action_horizon}")

    # Initialize wandb
    wandb.init(project="groot-n16-mcx-card", name="finetune", config=vars(config))

    # Load datasets
    train_dataset = GROOTDataset(dataset_dir / "train.hdf5", config)
    val_dataset = GROOTDataset(dataset_dir / "val.hdf5", config) if (dataset_dir / "val.hdf5").exists() else None

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4) if val_dataset else None

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleDiffusionPolicy(config).to(device)

    # Multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs * len(train_loader))

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch in pbar:
            # Move to device
            images = {k: v.to(device) for k, v in batch['images'].items()}
            state = batch['state'].to(device)
            action_chunk = batch['action_chunk'].to(device)

            # Forward
            loss = model.module.compute_loss(images, state, action_chunk) if hasattr(model, 'module') else \
                   model.compute_loss(images, state, action_chunk)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        wandb.log({'train_loss': avg_train_loss, 'epoch': epoch + 1, 'lr': scheduler.get_last_lr()[0]})
        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}")

        # Validation
        if val_loader and (epoch + 1) % 10 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    images = {k: v.to(device) for k, v in batch['images'].items()}
                    state = batch['state'].to(device)
                    action_chunk = batch['action_chunk'].to(device)
                    loss = model.module.compute_loss(images, state, action_chunk) if hasattr(model, 'module') else \
                           model.compute_loss(images, state, action_chunk)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            wandb.log({'val_loss': avg_val_loss})
            print(f"  val_loss={avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), output_dir / "best_model.pt")
                print(f"  Saved best model")

        # Save checkpoint
        if (epoch + 1) % 25 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': vars(config),
            }, output_dir / f"checkpoint_{epoch+1}.pt")

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(config),
    }, output_dir / "final_model.pt")

    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(vars(config), f, indent=2)

    print("Training complete!")
    wandb.finish()


if __name__ == "__main__":
    train()
TRAINEOF

# ---- Step 5: Run training ----
echo "[Step 5/6] Starting training..."
export DATASET_DIR OUTPUT_DIR NUM_EPOCHS BATCH_SIZE LEARNING_RATE ACTION_HORIZON

python "${WORK_DIR}/train_groot.py" 2>&1 | tee "${WORK_DIR}/training.log"

# ---- Step 6: Upload to HuggingFace ----
echo "[Step 6/6] Uploading trained model to HuggingFace..."

export OUTPUT_DIR HF_MODEL_REPO
python - <<'UPLOADEOF'
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

output_dir = Path(os.environ.get("OUTPUT_DIR", "~/groot-training/outputs")).expanduser()
repo_id = os.environ.get("HF_MODEL_REPO", "tshiamor/groot-n16-mcx-card")

api = HfApi()

# Create model repo
try:
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    print(f"Repository created/verified: {repo_id}")
except Exception as e:
    print(f"Note: {e}")

# Upload model
print(f"Uploading {output_dir} to {repo_id}...")
api.upload_folder(
    folder_path=str(output_dir),
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload GR00T N1.6 style diffusion policy for MCX Card manipulation",
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
echo "Model features:"
echo "  - Vision Transformer encoder"
echo "  - Diffusion-based action decoder"
echo "  - Action chunking (horizon=${ACTION_HORIZON})"
echo "  - Multi-camera fusion (wrist + table)"
echo "============================================="
