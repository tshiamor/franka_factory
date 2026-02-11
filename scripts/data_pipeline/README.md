# Data Pipeline for Cosmos Augmentation & VLA Model Training

This directory contains scripts for:
1. **Cosmos Augmentation** - Scale 215 demos to 1000+ using NVIDIA Cosmos-Transfer2.5
2. **VLA Data Preparation** - Convert demos to OpenVLA, Pi-Zero, and GR00T N1.6 formats
3. **VLA Model Training** - Fine-tune VLA models on NVIDIA Brev cloud or locally
4. **GR00T N1.6 Local Pipeline** - End-to-end local fine-tuning using Isaac-GR00T

## Pipeline Overview

```
                                    ┌─────────────────────────────────────┐
                                    │         NVIDIA Brev (Multi-GPU)     │
                                    │                                     │
┌─────────────────┐                 │  ┌─────────────────┐                │
│  Isaac Lab      │                 │  │ Cosmos-Transfer │                │
│  (Local)        │                 │  │ 2.5 Augment     │──┐             │
│                 │   ┌──────────┐  │  │ 215 → 1000+     │  │             │
│  215 demos      │──▶│ HuggingFace│─▶│  └─────────────────┘  │             │
│  224×224 VLA    │   │ Hub      │  │                       ▼             │
└─────────────────┘   └──────────┘  │  ┌─────────────────┐                │
        │                           │  │ VLA Training    │                │
        │                           │  │ OpenVLA/PiZero  │──▶ Models      │
        ▼                           │  │ GR00T N1.6      │                │
┌─────────────────┐                 │  └─────────────────┘                │
│  Data Prep      │                 │                                     │
│  OpenVLA/PiZero │─────────────────┘                                     │
│  GR00T formats  │                 └─────────────────────────────────────┘
└─────────────────┘
```

## HuggingFace Datasets

| Dataset | URL | Format | Size |
|---------|-----|--------|------|
| Original Demos | [tshiamor/mcx-card-demos-vla](https://huggingface.co/datasets/tshiamor/mcx-card-demos-vla) | HDF5 + Videos | 215 episodes |
| Cosmos Augmented | [tshiamor/mcx-card-cosmos-augmented](https://huggingface.co/datasets/tshiamor/mcx-card-cosmos-augmented) | Videos | 100 augmented |
| OpenVLA Format | [tshiamor/mcx-card-openvla](https://huggingface.co/datasets/tshiamor/mcx-card-openvla) | RLDS/TFRecord | 215 episodes |
| Pi-Zero Format | [tshiamor/mcx-card-pizero](https://huggingface.co/datasets/tshiamor/mcx-card-pizero) | LeRobot | 215 episodes |
| GR00T N1.6 Format | [tshiamor/mcx-card-groot-n16](https://huggingface.co/datasets/tshiamor/mcx-card-groot-n16) | HDF5 + Actions | 215 episodes |

## Scripts Overview

| Script | Purpose | Output |
|--------|---------|--------|
| `upload_to_huggingface.py` | Upload HDF5 demos to HuggingFace | Videos + metadata |
| `prepare_openvla.py` | Convert to OpenVLA RLDS format | TFRecord dataset |
| `prepare_pizero.py` | Convert to Pi-Zero LeRobot format | LeRobot dataset |
| `prepare_groot.py` | Convert to GR00T N1.6 format | HDF5 with action chunks |
| `reconstruct_augmented_hdf5.py` | Merge Cosmos videos with states | Augmented HDF5 |
| `brev_cosmos_augment.sh` | Run Cosmos augmentation on Brev | Augmented videos |
| `brev_train_openvla.sh` | Train OpenVLA on Brev | Fine-tuned model |
| `brev_train_pizero.sh` | Train Pi-Zero on Brev | ACT policy model |
| `brev_train_groot.sh` | Train GR00T N1.5 on Brev (LeRobot) | Diffusion policy |
| `brev_train_groot_n16.sh` | Train GR00T N1.6 locally (Isaac-GR00T) | Fine-tuned N1.6 model |

---

## Step 1: Prepare Data for VLA Models (Local)

### Install Requirements
```bash
pip install -r requirements.txt
```

### Prepare for OpenVLA
```bash
python prepare_openvla.py \
    --source local \
    --hdf5_path /path/to/demos.hdf5 \
    --output_dir ./openvla_data \
    --cameras wrist_rgb table_rgb
```

### Prepare for Pi-Zero (LeRobot)
```bash
python prepare_pizero.py \
    --source local \
    --hdf5_path /path/to/demos.hdf5 \
    --output_dir ./pizero_data \
    --cameras wrist_rgb table_rgb \
    --fps 30
```

### Prepare for GR00T N1.6
```bash
python prepare_groot.py \
    --source local \
    --hdf5_path /path/to/demos.hdf5 \
    --output_dir ./groot_data \
    --cameras wrist_rgb table_rgb \
    --action_horizon 16
```

---

## Step 2: Upload to HuggingFace

### Upload Raw Demos with Videos
```bash
huggingface-cli login

python upload_to_huggingface.py \
    --input_file /path/to/demos.hdf5 \
    --repo_id YOUR_USERNAME/dataset-name \
    --extract_videos \
    --cameras wrist_rgb table_rgb
```

### Upload Prepared VLA Datasets
```python
from huggingface_hub import HfApi
api = HfApi()

# Upload OpenVLA dataset
api.upload_folder(
    folder_path="./openvla_data",
    repo_id="YOUR_USERNAME/mcx-card-openvla",
    repo_type="dataset"
)

# Upload Pi-Zero dataset
api.upload_folder(
    folder_path="./pizero_data",
    repo_id="YOUR_USERNAME/mcx-card-pizero",
    repo_type="dataset"
)

# Upload GR00T dataset
api.upload_folder(
    folder_path="./groot_data",
    repo_id="YOUR_USERNAME/mcx-card-groot-n16",
    repo_type="dataset"
)
```

---

## Step 3: Run Cosmos Augmentation (NVIDIA Brev)

Scale 215 demos to 1000+ using NVIDIA Cosmos-Transfer2.5 World Foundation Model.

### Setup
1. Create NVIDIA Brev instance with 4+ A100/H100 GPUs
2. Accept license at https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B

### Run Augmentation
```bash
export HF_TOKEN="your_huggingface_token"
bash brev_cosmos_augment.sh
```

### Configuration (edit script)
```bash
HF_DATASET="tshiamor/mcx-card-demos-vla"
HF_OUTPUT_REPO="tshiamor/mcx-card-cosmos-augmented"
NUM_AUGMENTATIONS=5    # Augmentations per video (215×5=1075)
NUM_GPUS=4
MAX_VIDEOS=100         # Set to 0 for all videos
MAX_FRAMES=448         # Max frames per video
```

### Expected Output

Starting with 215 demonstrations:

| Augmentation Factor | Total Demos | Estimated Time (4×A100) |
|---------------------|-------------|-------------------------|
| 1× (original only)  | 215         | - |
| 3×                  | 645         | ~2 hours |
| 5×                  | 1,075       | ~3.5 hours |
| 10×                 | 2,150       | ~7 hours |

### Cosmos Augmentation Types

Cosmos-Transfer2.5 applies diverse visual augmentations:

| Augmentation | Description | Use Case |
|--------------|-------------|----------|
| Lighting     | Vary lighting conditions | Robustness to illumination |
| Background   | Modify background textures | Domain randomization |
| Color        | Adjust color temperature | Sensor variation |
| Style        | Apply style transfer | Visual diversity |
| Environment  | Change scene context | Sim-to-real transfer |

### Output Structure
```
cosmos_augmented/
├── demo_0_wrist_rgb.mp4          # Original
├── demo_0_wrist_rgb_aug0.mp4     # Augmented variant 1
├── demo_0_wrist_rgb_aug1.mp4     # Augmented variant 2
├── demo_0_table_rgb.mp4          # Original (table cam)
├── demo_0_table_rgb_aug0.mp4     # Augmented
└── metadata.json
```

### Reconstruct Augmented Dataset
After Cosmos augmentation, merge augmented videos with original robot states/actions:
```bash
python reconstruct_augmented_hdf5.py \
    --original_hdf5 /path/to/original.hdf5 \
    --augmented_videos /path/to/augmented/videos \
    --output_hdf5 ./augmented_dataset.hdf5 \
    --push_to_hub \
    --hub_repo_id YOUR_USERNAME/augmented-dataset
```

This creates a new HDF5 with:
- Original robot states, actions, and trajectories preserved
- Augmented RGB observations from Cosmos
- Ready for VLA training with 5-10× more data

---

## Step 4: Train VLA Models (NVIDIA Brev)

### Train OpenVLA
```bash
export HF_TOKEN="your_huggingface_token"
bash brev_train_openvla.sh
```

**Features:**
- Fine-tunes OpenVLA-7B base model
- Multi-GPU training with accelerate
- Automatic upload to HuggingFace

**Output Model:** `tshiamor/openvla-mcx-card`

### Train Pi-Zero (LeRobot)
```bash
export HF_TOKEN="your_huggingface_token"
bash brev_train_pizero.sh
```

**Features:**
- ACT (Action Chunking Transformer) policy
- Action horizon: 16 steps
- Video-based training with LeRobot

**Output Model:** `tshiamor/pizero-mcx-card`

### Train GR00T N1.5 (Brev)
```bash
export HF_TOKEN="your_huggingface_token"
bash brev_train_groot.sh
```

**Features:**
- Uses LeRobot's GR00T integration
- Pre-trained nvidia/GR00T-N1.5-3B base model
- Diffusion-based action decoder

**Output Model:** `tshiamor/groot-n15-mcx-card`

### Train GR00T N1.6 (Local - Recommended)
```bash
bash brev_train_groot_n16.sh
```

**Features:**
- Uses Isaac-GR00T's native fine-tuning pipeline (NOT LeRobot)
- Pre-trained nvidia/GR00T-N1.6-3B base model (3B params)
- Eagle3 vision backbone + Qwen2 LLM + diffusion action head
- Memory optimizations for RTX 5090 / consumer GPUs (32GB VRAM)
- End-to-end: HDF5 conversion → training → model post-processing

**Output Model:** `~/groot_data/finetune_output_n16/`

**Configuration (environment variables):**
```bash
HDF5_PATH=/path/to/demos.hdf5       # Training data
BATCH_SIZE=4                          # Per-GPU batch size
GRAD_ACCUM=8                          # Gradient accumulation steps
MAX_STEPS=10000                       # Training steps (~3.5 hours on RTX 5090)
LEARNING_RATE=1e-4                    # Learning rate
```

**Run with evaluation:**
```bash
bash brev_train_groot_n16.sh --eval
```

---

## Step 5: Evaluate Fine-tuned Models in Isaac Lab

After training, evaluate models on the MCX Card Block Insertion task in Isaac Lab.

### GR00T N1.6 (local fine-tuned)
```bash
CONDA_PREFIX=~/miniforge3/envs/isaaclab \
  ~/IsaacLab/isaaclab.sh -p \
  scripts/eval/eval_vla_policy.py \
  --task Franka-Factory-MCXCardBlockInsert-Mimic-v0 \
  --policy groot_n16 \
  --model ~/groot_data/finetune_output_n16 \
  --enable_cameras --headless --episodes 10
```

### Pi-Zero
```bash
CONDA_PREFIX=~/miniforge3/envs/isaaclab \
  ~/IsaacLab/isaaclab.sh -p \
  scripts/eval/eval_vla_policy.py \
  --task Franka-Factory-MCXCardBlockInsert-Mimic-v0 \
  --policy pizero \
  --model tshiamor/pizero-mcx-card \
  --enable_cameras --headless --episodes 10
```

### GR00T N1.5
```bash
CONDA_PREFIX=~/miniforge3/envs/isaaclab \
  ~/IsaacLab/isaaclab.sh -p \
  scripts/eval/eval_vla_policy.py \
  --task Franka-Factory-MCXCardBlockInsert-Mimic-v0 \
  --policy groot \
  --model tshiamor/groot-n15-mcx-card \
  --enable_cameras --headless --episodes 10
```

### OpenVLA
```bash
CONDA_PREFIX=~/miniforge3/envs/isaaclab \
  ~/IsaacLab/isaaclab.sh -p \
  scripts/eval/eval_vla_policy.py \
  --task Franka-Factory-MCXCardBlockInsert-Mimic-v0 \
  --policy openvla \
  --model tshiamor/openvla-mcx-card \
  --enable_cameras --headless --episodes 10
```

### Supported policies

| Policy | Model | Type | Description |
|--------|-------|------|-------------|
| `pizero` | `tshiamor/pizero-mcx-card` | HuggingFace | Pi-Zero via LeRobot |
| `groot` | `tshiamor/groot-n15-mcx-card` | HuggingFace | GR00T N1.5 via LeRobot |
| `groot_n16` | Local path | Local | GR00T N1.6 via Isaac-GR00T |
| `openvla` | `tshiamor/openvla-mcx-card` | HuggingFace | OpenVLA-7B |

---

## Training Configuration

### OpenVLA (brev_train_openvla.sh)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `BASE_MODEL` | openvla/openvla-7b | Pre-trained model |
| `NUM_EPOCHS` | 10 | Training epochs |
| `BATCH_SIZE` | 4 | Per-GPU batch size |
| `LEARNING_RATE` | 2e-5 | Learning rate |
| `NUM_GPUS` | 4 | Number of GPUs |

### Pi-Zero (brev_train_pizero.sh)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `ACTION_HORIZON` | 16 | Action chunk size |
| `CHUNK_SIZE` | 16 | Prediction chunk |
| `NUM_EPOCHS` | 100 | Training epochs |
| `BATCH_SIZE` | 32 | Per-GPU batch size |
| `LEARNING_RATE` | 1e-4 | Learning rate |

### GR00T N1.5 (brev_train_groot.sh)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `BASE_MODEL` | nvidia/GR00T-N1.5-3B | Pre-trained model |
| `STEPS` | 30000 | Training steps |
| `BATCH_SIZE` | 16 | Per-GPU batch size |

### GR00T N1.6 Local (brev_train_groot_n16.sh)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `BASE_MODEL` | nvidia/GR00T-N1.6-3B | Pre-trained model (3B params) |
| `BATCH_SIZE` | 4 | Per-GPU batch size |
| `GRAD_ACCUM` | 8 | Gradient accumulation (effective=32) |
| `MAX_STEPS` | 10000 | Training steps (~3.5 hrs RTX 5090) |
| `LEARNING_RATE` | 1e-4 | Learning rate |
| Optimizer | adamw_bnb_8bit | 8-bit optimizer (saves ~75% VRAM) |
| Grad checkpoint | True | Saves ~50% activation VRAM |

---

## Data Format Reference

### OpenVLA (RLDS/TFRecord)
```
openvla_data/
├── train/
│   ├── demo_000/
│   │   ├── observation.image.wrist_rgb.npy
│   │   ├── observation.image.table_rgb.npy
│   │   ├── observation.state.npy
│   │   ├── action.npy
│   │   └── language_instruction.txt
│   └── ...
├── val/
└── dataset_info.json
```

### Pi-Zero (LeRobot)
```
pizero_data/
└── lerobot/
    ├── data/
    │   └── data.json (or .parquet)
    ├── videos/
    │   ├── episode_000000.mp4
    │   └── ...
    └── meta_data/
        └── info.json
```

### GR00T N1.6 (HDF5)
```
groot_data/
├── train.hdf5
│   └── episode_XXX/
│       ├── wrist_rgb (T, 224, 224, 3)
│       ├── table_rgb (T, 224, 224, 3)
│       ├── robot_state (T, 9)
│       ├── actions (T, 7)
│       └── action_chunk (T, 16, 7)
├── val.hdf5
├── metadata.json
├── groot_config.yaml
└── lerobot_config.json
```

---

## Troubleshooting

### "CUDA out of memory"
- Reduce batch size in training scripts
- Use gradient accumulation
- For GR00T N1.6: `BATCH_SIZE=2 GRAD_ACCUM=16 bash brev_train_groot_n16.sh`

### "Module not found"
```bash
pip install huggingface_hub h5py imageio[ffmpeg] pyyaml
```

### "Upload failed"
```bash
huggingface-cli whoami  # Verify login
huggingface-cli login   # Re-authenticate
```

### "Video extraction failed"
```bash
pip install imageio-ffmpeg imageio[ffmpeg]
```

### GR00T N1.6: "Eagle3_VLConfig" / "_attn_implementation_autoset" error
Fine-tuning requires transformers==4.51.3 (NOT 4.57.x):
```bash
pip install transformers==4.51.3
rm -rf ~/.cache/huggingface/modules/transformers_modules/Eagle_hyphen_Block2A_hyphen_2B_hyphen_v2/
```

### GR00T N1.6: "Unrecognized processing class" during inference
Processor files are in `processor/` subdirectory but AutoProcessor looks in model root:
```bash
cd ~/groot_data/finetune_output_n16
cp processor/processor_config.json processor/embodiment_id.json processor/statistics.json .
```
(The `brev_train_groot_n16.sh` script handles this automatically in Step 5.)

### GR00T N1.6: "No module named 'gr00t'"
Isaac-GR00T must be installed or on `sys.path`:
```bash
cd ~/Isaac-GR00T && pip install -e .
# Or set: export PYTHONPATH=~/Isaac-GR00T:$PYTHONPATH
```

---

## Tips for VLA Training

1. **Balance augmentation**: Don't over-augment - keep some original demos
2. **Preserve actions**: Cosmos modifies visuals, not trajectories
3. **Validate quality**: Check a sample of augmented videos before training
4. **Mix cameras**: Use both wrist and table camera augmentations
5. **Start small**: Test with 100 videos before processing all 215

---

## Complete Pipeline Examples

### Option A: Direct VLA Training (215 demos)

```bash
# 1. Prepare all VLA formats locally
python prepare_openvla.py --source local --hdf5_path ./demos.hdf5 --output_dir ./openvla_data
python prepare_pizero.py --source local --hdf5_path ./demos.hdf5 --output_dir ./pizero_data
python prepare_groot.py --source local --hdf5_path ./demos.hdf5 --output_dir ./groot_data

# 2. Upload to HuggingFace
huggingface-cli login
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(folder_path='./openvla_data', repo_id='USER/mcx-card-openvla', repo_type='dataset')
api.upload_folder(folder_path='./pizero_data', repo_id='USER/mcx-card-pizero', repo_type='dataset')
api.upload_folder(folder_path='./groot_data', repo_id='USER/mcx-card-groot-n16', repo_type='dataset')
"

# 3. On NVIDIA Brev - Train models
export HF_TOKEN="your_token"
bash brev_train_openvla.sh  # Fine-tune OpenVLA
bash brev_train_pizero.sh   # Train Pi-Zero
bash brev_train_groot.sh    # Train GR00T N1.6

# 4. Models automatically uploaded to HuggingFace
```

### Option B: Cosmos Augmentation + VLA Training (1000+ demos)

```bash
# 1. Upload original demos to HuggingFace (with videos for Cosmos)
python upload_to_huggingface.py \
    --input_file ./demos.hdf5 \
    --repo_id USER/mcx-card-demos-vla \
    --extract_videos

# 2. On NVIDIA Brev - Run Cosmos augmentation
export HF_TOKEN="your_token"
bash brev_cosmos_augment.sh
# Output: tshiamor/mcx-card-cosmos-augmented (1000+ videos)

# 3. Locally - Reconstruct augmented HDF5
python reconstruct_augmented_hdf5.py \
    --original_hdf5 ./demos.hdf5 \
    --augmented_source huggingface \
    --hf_repo tshiamor/mcx-card-cosmos-augmented \
    --output_hdf5 ./augmented_demos.hdf5

# 4. Prepare augmented data for VLA models
python prepare_openvla.py --source local --hdf5_path ./augmented_demos.hdf5 --output_dir ./openvla_augmented
python prepare_pizero.py --source local --hdf5_path ./augmented_demos.hdf5 --output_dir ./pizero_augmented
python prepare_groot.py --source local --hdf5_path ./augmented_demos.hdf5 --output_dir ./groot_augmented

# 5. Upload and train with 5-10× more data
```

---

## License

Apache 2.0
