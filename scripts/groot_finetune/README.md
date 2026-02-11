# GR00T N1.6 Fine-tuning for Franka MCX Card Block Insertion

## Overview

Fine-tunes NVIDIA's GR00T N1.6 (3B param VLA model) on the MCX Card Block Insertion
task using demonstration data collected in Isaac Lab. The pipeline converts HDF5 teleop
data to LeRobot v2 format, then runs Isaac-GR00T's fine-tuning with memory optimizations
for RTX 5090 (32GB VRAM).

---

## Quick Start (New HDF5 File)

### Step 1: Convert HDF5 to LeRobot v2

```bash
conda activate isaaclab

python /home/tshiamo/SIMULATION_MANIPULATION/franka_factory/scripts/data_pipeline/convert_hdf5_to_lerobot_v2.py \
    --hdf5 /path/to/your/new_training_data.hdf5 \
    --output /home/tshiamo/groot_data/mcx_card_lerobot_v2
```

This converts demonstrations into:
- Parquet files (state + action per timestep)
- MP4 videos (wrist_rgb + table_rgb per episode)
- Meta files (info.json, episodes.jsonl, tasks.jsonl, modality.json)

### Step 2: Generate dataset statistics

```bash
cd /home/tshiamo/Isaac-GR00T

python -c "
from gr00t.data.dataset.lerobot_dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag

# Register modality config
import importlib, sys
from pathlib import Path
config_path = Path('/home/tshiamo/SIMULATION_MANIPULATION/franka_factory/scripts/groot_finetune/franka_mcx_config.py')
sys.path.append(str(config_path.parent))
importlib.import_module(config_path.stem)

ds = LeRobotSingleDataset(
    dataset_path='/home/tshiamo/groot_data/mcx_card_lerobot_v2',
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
ds.compute_stats()
print('Stats saved to meta/stats.json')
"
```

### Step 3: Launch fine-tuning

```bash
bash /home/tshiamo/SIMULATION_MANIPULATION/franka_factory/scripts/groot_finetune/launch_finetune_mcx.sh
```

Or run manually with custom params:

```bash
cd /home/tshiamo/Isaac-GR00T
export PATH="/home/tshiamo/miniforge3/envs/isaaclab/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m gr00t.experiment.launch_finetune \
    --base-model-path "nvidia/GR00T-N1.6-3B" \
    --dataset-path "/home/tshiamo/groot_data/mcx_card_lerobot_v2" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "/home/tshiamo/SIMULATION_MANIPULATION/franka_factory/scripts/groot_finetune/franka_mcx_config.py" \
    --output-dir "/home/tshiamo/groot_data/finetune_output_n16" \
    --num-gpus 1 \
    --global-batch-size 2 \
    --gradient-accumulation-steps 16 \
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
```

### Step 4: Evaluate (after training completes)

The checkpoint will be at `/home/tshiamo/groot_data/finetune_output_n16/checkpoint-XXXX/`.

**Important**: Before inference, upgrade transformers back:
```bash
pip install transformers==4.57.1
```
(Fine-tuning requires 4.51.3, inference requires >=4.57.1)

---

## Files Created / Modified

### New files

| File | Purpose |
|------|---------|
| `scripts/groot_finetune/franka_mcx_config.py` | Modality config mapping Franka MCX data to GR00T input format |
| `scripts/groot_finetune/launch_finetune_mcx.sh` | Launch script with all CLI args and memory optimizations |
| `scripts/data_pipeline/convert_hdf5_to_lerobot_v2.py` | Converts HDF5 teleop data to LeRobot v2 format |

### Isaac-GR00T — NO modifications

No files in `/home/tshiamo/Isaac-GR00T/` are modified. Memory optimizations (8-bit optimizer,
gradient checkpointing) are applied in our own `launch_finetune.py` wrapper script.

### Generated data

| Path | Contents |
|------|----------|
| `/home/tshiamo/groot_data/mcx_card_lerobot_v2/` | LeRobot v2 dataset (265 demos, 497MB) |
| `/home/tshiamo/groot_data/finetune_output_n16/` | Fine-tuned model checkpoints |

### Package changes in isaaclab conda env

| Package | Before | After | Reason |
|---------|--------|-------|--------|
| transformers | 4.57.6 | 4.51.3 | 4.57.x breaks Eagle3_VLConfig in fine-tuning |
| scipy | not installed | installed | Required by Isaac-GR00T stats computation |
| lmdb | not installed | installed | Required by Eagle3 processor |
| ffmpeg | not installed | installed (conda) | Required for video decoding in dataset loading |

---

## Training Configuration

### Memory optimizations for RTX 5090 (32GB)

The GR00T N1.6 3B model OOMs with default settings. These optimizations were needed:

| Setting | Default | Our value | VRAM savings |
|---------|---------|-----------|-------------|
| per_device_batch_size | 64 | 2 | Major |
| gradient_accumulation_steps | 1 | 16 | (maintains effective batch=32) |
| optimizer | adamw_torch | adamw_bnb_8bit | ~75% on optimizer states |
| gradient_checkpointing | False | True | ~50% on activations |
| PYTORCH_CUDA_ALLOC_CONF | unset | expandable_segments:True | Reduces fragmentation |

### What gets trained

| Component | Trainable | Notes |
|-----------|-----------|-------|
| LLM backbone (layers 0-11) | Frozen | Too expensive to train |
| LLM top layers (12-15) | Yes (fp32) | Always trained, controlled by `tune_top_llm_layers=4` |
| Vision encoder | Frozen | |
| Multimodal projector | Yes | `--tune-projector` |
| Diffusion action head | Yes | `--tune-diffusion-model` |

### Training speed

- ~1.35-1.42 seconds/step on RTX 5090
- 10,000 steps takes ~3.8 hours
- Checkpoints saved every 1,000 steps

---

## Data Format

### HDF5 input format (from Isaac Lab teleop)

```
data/
  demo_0/
    obs/
      wrist_rgb: (T, 240, 320, 3) uint8
      table_rgb: (T, 240, 320, 3) uint8
      eef_pos: (T, 3) float32         # end-effector position
      eef_quat: (T, 4) float32        # end-effector quaternion
      gripper_pos: (T, 1) float32     # gripper opening
    actions: (T, 7) float32           # [dx,dy,dz,droll,dpitch,dyaw,gripper]
```

### LeRobot v2 output format (for GR00T)

```
mcx_card_lerobot_v2/
  data/chunk-000/episode_XXXXXX.parquet   # state + action columns
  videos/chunk-000/wrist/episode_XXXXXX.mp4
  videos/chunk-000/table/episode_XXXXXX.mp4
  meta/
    info.json          # dataset metadata
    episodes.jsonl     # episode boundaries
    tasks.jsonl        # task descriptions
    modality.json      # state/action/video column mapping
    stats.json         # normalization statistics
```

### Modality config mapping

```
video:  wrist (camera 1), table (camera 2)
state:  eef_pos[0:3], eef_quat[3:7], gripper[7:8]
action: arm[0:6] (delta pose), gripper[6:7]
language: "Pick up the block and insert it into the MCX card slot"
```

---

## Troubleshooting

### CUDA OOM
Reduce `--global-batch-size` to 1. If still OOM, also set `--no-tune-projector`.

### `Eagle3_VLConfig` / `_attn_implementation_autoset` error
```bash
pip install transformers==4.51.3
rm -rf ~/.cache/huggingface/modules/transformers_modules/Eagle_hyphen_Block2A_hyphen_2B_hyphen_v2/
```

### `model_type gr00t_n1_5 not recognized`
The fine-tuning pipeline only supports N1.6. Use `--base-model-path nvidia/GR00T-N1.6-3B`.

### `invalid choice 'new_embodiment'`
Enum must be uppercase: `--embodiment-tag NEW_EMBODIMENT`

### Boolean flags ignored
tyro uses toggle syntax: `--no-tune-llm` / `--tune-llm` (not `--tune-llm False`).

### Missing lmdb/scipy
```bash
pip install lmdb scipy
```

### ffprobe not found
```bash
conda install -c conda-forge ffmpeg
export PATH="/home/tshiamo/miniforge3/envs/isaaclab/bin:$PATH"
```
