#!/usr/bin/env python
# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Upload rendered demonstrations to Hugging Face for Cosmos augmentation.

This script:
1. Extracts video frames from HDF5 dataset
2. Creates video files for Cosmos input
3. Uploads to Hugging Face Hub
4. Generates metadata for Cosmos World Foundation Model

Usage:
    python upload_to_huggingface.py \
        --input_file ./demos/mcx_card_demos_vla_224.hdf5 \
        --repo_id your-username/mcx-card-demos \
        --extract_videos \
        --video_format mp4

Requirements:
    pip install huggingface_hub opencv-python imageio[ffmpeg]
"""

import argparse
import h5py
import json
import numpy as np
import os
import shutil
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Upload demos to Hugging Face for Cosmos")
    parser.add_argument("--input_file", type=str, required=True, help="Input HDF5 file")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repo ID (username/repo-name)")
    parser.add_argument("--output_dir", type=str, default="./hf_upload", help="Temporary output directory")
    parser.add_argument("--extract_videos", action="store_true", help="Extract videos from HDF5")
    parser.add_argument("--video_format", type=str, default="mp4", choices=["mp4", "webm", "gif"], help="Video format")
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate")
    parser.add_argument("--cameras", type=str, nargs="+", default=["wrist_rgb", "table_rgb"], help="Cameras to extract")
    parser.add_argument("--upload_hdf5", action="store_true", default=True, help="Upload original HDF5 file")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    parser.add_argument("--skip_upload", action="store_true", help="Only prepare, don't upload")
    parser.add_argument("--max_episodes", type=int, default=None, help="Max episodes to process")
    return parser.parse_args()


def extract_video_from_episode(
    ep_data: dict,
    camera_key: str,
    output_path: str,
    fps: int = 30,
    video_format: str = "mp4"
):
    """Extract video from episode observations."""
    try:
        import imageio
    except ImportError:
        print("Please install imageio: pip install imageio[ffmpeg]")
        return False

    if camera_key not in ep_data.get("obs", {}):
        return False

    frames = ep_data["obs"][camera_key]
    if len(frames) == 0:
        return False

    # Ensure frames are uint8
    if frames.dtype != np.uint8:
        if frames.max() <= 1.0:
            frames = (frames * 255).astype(np.uint8)
        else:
            frames = frames.astype(np.uint8)

    # Write video
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    return True


def create_cosmos_metadata(
    ep_name: str,
    ep_data: dict,
    camera_key: str,
    video_path: str,
) -> dict:
    """Create metadata for Cosmos World Foundation Model."""
    metadata = {
        "episode_id": ep_name,
        "camera": camera_key,
        "video_path": video_path,
        "num_frames": len(ep_data.get("obs", {}).get(camera_key, [])),
        "task": "mcx_card_block_insert",
        "robot": "franka_panda",
        "modality": "rgb",
    }

    # Add action info if available
    if "actions" in ep_data:
        metadata["num_actions"] = len(ep_data["actions"])
        metadata["action_dim"] = ep_data["actions"].shape[-1] if len(ep_data["actions"]) > 0 else 0

    # Add robot state info
    if "obs" in ep_data:
        obs = ep_data["obs"]
        if "eef_pos" in obs:
            metadata["has_eef_pos"] = True
        if "eef_quat" in obs:
            metadata["has_eef_quat"] = True
        if "joint_pos" in obs:
            metadata["has_joint_pos"] = True

    return metadata


def prepare_dataset_for_cosmos(
    input_file: str,
    output_dir: str,
    cameras: list,
    fps: int = 30,
    video_format: str = "mp4",
    max_episodes: int = None,
) -> dict:
    """Prepare dataset for Cosmos augmentation."""

    output_path = Path(output_dir)
    videos_dir = output_path / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    metadata_list = []

    with h5py.File(input_file, "r") as f:
        episode_names = list(f["data"].keys())
        if max_episodes:
            episode_names = episode_names[:max_episodes]

        env_name = f["data"].attrs.get("env_name", "unknown")

        for ep_name in tqdm(episode_names, desc="Extracting videos"):
            ep_group = f["data"][ep_name]

            # Load episode data
            ep_data = {"obs": {}}
            if "actions" in ep_group:
                ep_data["actions"] = np.array(ep_group["actions"])

            if "obs" in ep_group:
                for key in ep_group["obs"].keys():
                    ep_data["obs"][key] = np.array(ep_group["obs"][key])

            # Extract videos for each camera
            for camera in cameras:
                if camera in ep_data.get("obs", {}):
                    video_filename = f"{ep_name}_{camera}.{video_format}"
                    video_path = videos_dir / video_filename

                    success = extract_video_from_episode(
                        ep_data, camera, str(video_path), fps, video_format
                    )

                    if success:
                        metadata = create_cosmos_metadata(
                            ep_name, ep_data, camera, f"videos/{video_filename}"
                        )
                        metadata_list.append(metadata)

    # Save metadata
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump({
            "env_name": env_name,
            "num_episodes": len(episode_names),
            "cameras": cameras,
            "fps": fps,
            "video_format": video_format,
            "episodes": metadata_list,
        }, f, indent=2)

    # Create dataset card
    create_dataset_card(output_path, env_name, len(episode_names), cameras)

    return {
        "videos_dir": str(videos_dir),
        "metadata_file": str(metadata_file),
        "num_videos": len(metadata_list),
    }


def create_dataset_card(output_dir: Path, env_name: str, num_episodes: int, cameras: list):
    """Create a dataset card (README.md) for Hugging Face."""
    readme_content = f"""---
license: apache-2.0
task_categories:
  - robotics
tags:
  - robotics
  - manipulation
  - franka
  - isaac-sim
  - cosmos
  - visuomotor
size_categories:
  - n<1K
---

# MCX Card Block Insert - Franka Manipulation Dataset

## Overview

This dataset contains {num_episodes} demonstration episodes for the MCX Card Block Insert task
using a Franka Panda robot in Isaac Sim.

## Task Description

The robot must:
1. Grasp a blue block (10cm × 2cm × 1cm)
2. Move and place it on a target platform near MCX network cards

## Dataset Structure

```
├── mcx_card_demos_vla_224.hdf5    # Full dataset with all modalities
├── videos/                         # Extracted video files for Cosmos
│   ├── demo_0_wrist_rgb.mp4
│   ├── demo_0_table_rgb.mp4
│   └── ...
└── metadata.json                   # Episode metadata
```

## Modalities (224×224 resolution)

| Modality | Cameras | Shape | Description |
|----------|---------|-------|-------------|
| RGB | wrist, table | (T, 224, 224, 3) | Color images |
| Depth | wrist, table | (T, 224, 224, 1) | Depth maps |
| Normals | wrist, table | (T, 224, 224, 3) | Surface normals |
| Semantic | wrist, table | (T, 224, 224, 1) | Semantic segmentation |
| Instance | wrist, table | (T, 224, 224, 1) | Instance segmentation |
| Motion Vectors | wrist, table | (T, 224, 224, 2) | Optical flow |

## Robot State

| Observation | Shape | Description |
|-------------|-------|-------------|
| eef_pos | (T, 3) | End-effector position |
| eef_quat | (T, 4) | End-effector orientation |
| gripper_pos | (T, 2) | Gripper finger positions |
| joint_pos | (T, 9) | Joint positions |
| joint_vel | (T, 9) | Joint velocities |
| actions | (T, 7) | 6 DoF pose + 1 gripper |

## Usage for Cosmos Augmentation

### Download dataset
```python
from huggingface_hub import hf_hub_download

# Download HDF5 file
hdf5_path = hf_hub_download(
    repo_id="YOUR_REPO_ID",
    filename="mcx_card_demos_vla_224.hdf5",
    repo_type="dataset"
)

# Download videos for Cosmos
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="YOUR_REPO_ID",
    repo_type="dataset",
    local_dir="./mcx_demos",
    allow_patterns=["videos/*", "metadata.json"]
)
```

### Use with Cosmos World Foundation Model
```python
# See NVIDIA Cosmos documentation for video-to-video generation
# Videos are in videos/ directory, metadata in metadata.json
```

## Environment

- **Environment**: {env_name}
- **Robot**: Franka Panda (7 DoF + gripper)
- **Simulator**: NVIDIA Isaac Sim / Isaac Lab
- **Cameras**: {', '.join(cameras)}

## Citation

If you use this dataset, please cite Isaac Lab and Franka Factory.

## License

Apache 2.0
"""
    readme_path = output_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)


def upload_to_huggingface(
    output_dir: str,
    input_file: str,
    repo_id: str,
    upload_hdf5: bool = True,
    private: bool = False,
):
    """Upload prepared dataset to Hugging Face Hub."""
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Please install huggingface_hub: pip install huggingface_hub")
        return False

    api = HfApi()

    # Create repo if doesn't exist
    try:
        create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
        print(f"Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"Error creating repo: {e}")
        return False

    # Upload files
    output_path = Path(output_dir)

    # Upload README
    readme_path = output_path / "README.md"
    if readme_path.exists():
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("Uploaded README.md")

    # Upload metadata
    metadata_path = output_path / "metadata.json"
    if metadata_path.exists():
        api.upload_file(
            path_or_fileobj=str(metadata_path),
            path_in_repo="metadata.json",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("Uploaded metadata.json")

    # Upload videos folder
    videos_dir = output_path / "videos"
    if videos_dir.exists():
        api.upload_folder(
            folder_path=str(videos_dir),
            path_in_repo="videos",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"Uploaded videos folder ({len(list(videos_dir.glob('*')))} files)")

    # Upload HDF5 file
    if upload_hdf5 and os.path.exists(input_file):
        print(f"Uploading HDF5 file ({os.path.getsize(input_file) / 1e9:.2f} GB)...")
        api.upload_file(
            path_or_fileobj=input_file,
            path_in_repo=os.path.basename(input_file),
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"Uploaded {os.path.basename(input_file)}")

    print(f"\nDataset uploaded to: https://huggingface.co/datasets/{repo_id}")
    return True


def main():
    args = parse_args()

    # Check input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return 1

    print(f"Input file: {args.input_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Repository: {args.repo_id}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Extract videos if requested
    if args.extract_videos:
        print("\nExtracting videos for Cosmos...")
        result = prepare_dataset_for_cosmos(
            input_file=args.input_file,
            output_dir=args.output_dir,
            cameras=args.cameras,
            fps=args.fps,
            video_format=args.video_format,
            max_episodes=args.max_episodes,
        )
        print(f"Extracted {result['num_videos']} videos")
    else:
        # Just create dataset card
        with h5py.File(args.input_file, "r") as f:
            env_name = f["data"].attrs.get("env_name", "unknown")
            num_episodes = len(f["data"].keys())
        create_dataset_card(Path(args.output_dir), env_name, num_episodes, args.cameras)

    # Copy HDF5 to output dir for reference
    hdf5_dest = Path(args.output_dir) / os.path.basename(args.input_file)
    if not hdf5_dest.exists() and args.upload_hdf5:
        print(f"\nCopying HDF5 file to output directory...")
        shutil.copy2(args.input_file, hdf5_dest)

    # Upload to Hugging Face
    if not args.skip_upload:
        print("\nUploading to Hugging Face...")
        success = upload_to_huggingface(
            output_dir=args.output_dir,
            input_file=args.input_file,
            repo_id=args.repo_id,
            upload_hdf5=args.upload_hdf5,
            private=args.private,
        )
        if not success:
            return 1

    print("\nDone!")
    print(f"\nNext steps for Cosmos augmentation on NVIDIA Brev:")
    print(f"1. Download dataset: huggingface-cli download {args.repo_id}")
    print(f"2. Run Cosmos World Foundation Model on the videos")
    print(f"3. Generate 1000+ augmented demonstrations")

    return 0


if __name__ == "__main__":
    exit(main())
