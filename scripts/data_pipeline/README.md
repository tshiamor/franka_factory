# Data Pipeline for Cosmos Augmentation

This directory contains scripts for uploading demonstration data to Hugging Face
and running NVIDIA Cosmos augmentation to scale 215 demos to 1000+.

## Pipeline Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Isaac Lab      │     │  Hugging Face   │     │  NVIDIA Brev    │
│  (Local)        │────▶│  Hub            │────▶│  (Multi-GPU)    │
│                 │     │                 │     │                 │
│  215 demos      │     │  HDF5 + Videos  │     │  Cosmos 2.5     │
│  224×224 VLA    │     │                 │     │  1000+ demos    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Step 1: Upload to Hugging Face (Local)

```bash
# Install requirements
pip install -r requirements.txt

# Login to Hugging Face
huggingface-cli login

# Upload with video extraction
python upload_to_huggingface.py \
    --input_file /home/tshiamo/IsaacLab/demos/mcx_card_demos_vla_224.hdf5 \
    --repo_id YOUR_USERNAME/mcx-card-demos \
    --extract_videos \
    --video_format mp4 \
    --cameras wrist_rgb table_rgb \
    --fps 30
```

### Upload Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input_file` | Required | Path to HDF5 dataset |
| `--repo_id` | Required | Hugging Face repo (username/repo-name) |
| `--extract_videos` | False | Extract MP4 videos for Cosmos |
| `--video_format` | mp4 | Video format (mp4, webm, gif) |
| `--cameras` | wrist_rgb table_rgb | Cameras to extract |
| `--upload_hdf5` | True | Upload original HDF5 file |
| `--private` | False | Make repo private |
| `--skip_upload` | False | Only prepare, don't upload |

## Step 2: Run Cosmos on NVIDIA Brev

### Setup Brev Instance

1. Create NVIDIA Brev instance with 4+ A100/H100 GPUs
2. Install Cosmos Transfer 2.5:
   ```bash
   # Follow NVIDIA Cosmos installation guide
   pip install cosmos-transfer
   ```

### Run Augmentation

```bash
# Download and run Cosmos augmentation
python cosmos_augmentation_brev.py \
    --repo_id YOUR_USERNAME/mcx-card-demos \
    --output_dir ./cosmos_augmented \
    --num_augmentations 5 \
    --gpus 0,1,2,3
```

### Augmentation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--repo_id` | Required | Hugging Face repo to download |
| `--num_augmentations` | 5 | Augmentations per video (215×5=1075) |
| `--gpus` | 0 | Comma-separated GPU IDs |
| `--cosmos_model` | cosmos-transfer-2.5 | Cosmos model name |
| `--batch_size` | 1 | Batch size per GPU |
| `--dry_run` | False | Print commands without running |

## Expected Output

Starting with 215 demonstrations:

| Augmentation Factor | Total Demos | Estimated Time (4×A100) |
|---------------------|-------------|-------------------------|
| 1× (original) | 215 | - |
| 3× | 645 | ~2 hours |
| 5× | 1,075 | ~3.5 hours |
| 10× | 2,150 | ~7 hours |

## Output Structure

```
cosmos_augmented/
├── demo_0_wrist_rgb_aug0.mp4
├── demo_0_wrist_rgb_aug1.mp4
├── demo_0_wrist_rgb_aug2.mp4
├── demo_0_table_rgb_aug0.mp4
├── ...
└── augmented_metadata.json
```

## Step 3: Reconstruct HDF5 (Optional)

After Cosmos augmentation, you may want to reconstruct an HDF5 dataset
with the augmented videos for VLA training:

```python
# Example reconstruction script
import h5py
import cv2
import numpy as np
from pathlib import Path

def reconstruct_hdf5(augmented_dir, output_file):
    augmented_videos = list(Path(augmented_dir).glob("*.mp4"))

    with h5py.File(output_file, "w") as f:
        data_group = f.create_group("data")

        for video_path in augmented_videos:
            # Parse episode name from filename
            ep_name = video_path.stem  # e.g., "demo_0_wrist_rgb_aug0"

            # Read video frames
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()

            # Save to HDF5
            ep_group = data_group.create_group(ep_name)
            obs_group = ep_group.create_group("obs")
            obs_group.create_dataset("rgb", data=np.array(frames), compression="gzip")
```

## Cosmos Augmentation Types

Cosmos Transfer 2.5 can apply various augmentations:

| Augmentation | Description | Use Case |
|--------------|-------------|----------|
| Lighting | Vary lighting conditions | Robustness to illumination |
| Background | Modify background textures | Domain randomization |
| Color | Adjust color temperature | Sensor variation |
| Noise | Add realistic sensor noise | Real-world transfer |
| Style | Apply style transfer | Visual diversity |

## Tips for VLA Training

1. **Balance augmentation**: Don't over-augment - keep some original demos
2. **Preserve actions**: Cosmos modifies visuals, not trajectories
3. **Validate quality**: Check a sample of augmented videos before training
4. **Mix cameras**: Use both wrist and table camera augmentations

## Troubleshooting

### "CUDA out of memory"
- Reduce batch size: `--batch_size 1`
- Use fewer GPUs with more memory each

### "Video extraction failed"
- Check if `imageio[ffmpeg]` is installed
- Ensure HDF5 has RGB data in expected format

### "Upload failed"
- Check Hugging Face token: `huggingface-cli whoami`
- Verify repo name format: `username/repo-name`
