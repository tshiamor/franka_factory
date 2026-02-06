#!/usr/bin/env python3
# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""
Prepare MCX Card demos for π₀ (Pi-Zero) training.

π₀ uses LeRobot format (HuggingFace datasets) with:
- RGB images (224×224)
- Actions (continuous)
- Robot state observations
- Language instructions (optional)

Usage:
    # From HuggingFace augmented dataset
    python prepare_pizero.py \
        --source huggingface \
        --repo_id tshiamor/mcx-card-cosmos-augmented \
        --output_dir ./pizero_data

    # From local HDF5
    python prepare_pizero.py \
        --source local \
        --hdf5_path /path/to/mcx_card_demos_vla_224.hdf5 \
        --output_dir ./pizero_data

    # Upload to HuggingFace for LeRobot
    python prepare_pizero.py \
        --source local \
        --hdf5_path /path/to/demos.hdf5 \
        --output_dir ./pizero_data \
        --push_to_hub \
        --hub_repo_id your-username/mcx-card-pizero
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


TASK_INSTRUCTION = "Pick up the blue block and place it on the target platform."


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data for Pi-Zero (LeRobot format)")
    parser.add_argument("--source", type=str, choices=["huggingface", "local"], default="local")
    parser.add_argument("--repo_id", type=str, default="tshiamor/mcx-card-demos-vla")
    parser.add_argument("--hdf5_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./pizero_data")
    parser.add_argument("--camera", type=str, default="wrist_rgb")
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_repo_id", type=str, default=None)
    return parser.parse_args()


def download_from_huggingface(repo_id: str, local_dir: str):
    """Download dataset from HuggingFace."""
    from huggingface_hub import snapshot_download

    print(f"Downloading {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
    )

    hdf5_files = list(Path(local_dir).glob("*.hdf5"))
    if hdf5_files:
        return str(hdf5_files[0])
    return None


def create_lerobot_dataset(episodes: list, output_dir: str, fps: int = 30):
    """
    Create LeRobot-compatible dataset structure.

    LeRobot expects:
    - data/episode_XXXXXX/
      - observation.images.camera/*.png or as parquet
      - action, state as parquet files
    - meta_data/
      - info.json, episodes.json, tasks.json
    """
    output_path = Path(output_dir)
    data_dir = output_path / "data"
    meta_dir = output_path / "meta_data"
    videos_dir = output_path / "videos"

    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    episode_info = []
    all_frames = []
    frame_idx = 0

    for ep_idx, episode in enumerate(tqdm(episodes, desc="Processing episodes")):
        images = episode["images"]
        actions = episode["actions"]
        states = episode.get("states", {})

        num_steps = len(images)
        ep_length = num_steps

        # Save video
        try:
            import imageio
            video_path = videos_dir / f"episode_{ep_idx:06d}.mp4"
            writer = imageio.get_writer(str(video_path), fps=fps, codec='libx264', quality=8)
            for img in images:
                writer.append_data(img)
            writer.close()
        except ImportError:
            # Fallback: save as individual images
            img_dir = data_dir / f"episode_{ep_idx:06d}" / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            for t, img in enumerate(images):
                from PIL import Image
                Image.fromarray(img).save(img_dir / f"{t:06d}.png")

        # Build frame data
        for t in range(num_steps):
            frame_data = {
                "episode_index": ep_idx,
                "frame_index": frame_idx,
                "timestamp": t / fps,
                "index_in_episode": t,
            }

            # Actions
            if t < len(actions):
                action = actions[t]
                for i, val in enumerate(action):
                    frame_data[f"action_{i}"] = float(val)

            # States
            for key, values in states.items():
                if t < len(values):
                    if isinstance(values[t], np.ndarray):
                        for i, val in enumerate(values[t]):
                            frame_data[f"state.{key}_{i}"] = float(val)
                    else:
                        frame_data[f"state.{key}"] = float(values[t])

            all_frames.append(frame_data)
            frame_idx += 1

        episode_info.append({
            "episode_index": ep_idx,
            "length": ep_length,
            "task": TASK_INSTRUCTION,
            "video_path": f"videos/episode_{ep_idx:06d}.mp4",
        })

    # Save as parquet
    try:
        import pandas as pd
        df = pd.DataFrame(all_frames)
        df.to_parquet(data_dir / "data.parquet", index=False)
        print(f"Saved {len(all_frames)} frames to parquet")
    except ImportError:
        # Fallback: save as JSON
        with open(data_dir / "data.json", "w") as f:
            json.dump(all_frames, f)
        print(f"Saved {len(all_frames)} frames to JSON (install pandas for parquet)")

    # Save metadata
    info = {
        "fps": fps,
        "video_backend": "pyav",
        "robot_type": "franka_panda",
        "features": {
            "observation.images.wrist": {
                "dtype": "video",
                "shape": [224, 224, 3],
                "names": ["height", "width", "channels"],
            },
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
            },
        },
    }

    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    with open(meta_dir / "episodes.json", "w") as f:
        json.dump(episode_info, f, indent=2)

    tasks = [{"task_index": 0, "task": TASK_INSTRUCTION}]
    with open(meta_dir / "tasks.json", "w") as f:
        json.dump(tasks, f, indent=2)

    print(f"LeRobot dataset saved to {output_dir}")
    return output_path


def push_to_huggingface(dataset_dir: str, repo_id: str):
    """Push LeRobot dataset to HuggingFace Hub."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    api.upload_folder(
        folder_path=dataset_dir,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload Pi-Zero / LeRobot dataset",
    )

    print(f"Pushed to https://huggingface.co/datasets/{repo_id}")


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get HDF5 path
    if args.source == "huggingface":
        download_dir = output_dir / "download"
        hdf5_path = download_from_huggingface(args.repo_id, str(download_dir))
        if not hdf5_path:
            print("No HDF5 file found. Video-based loading not yet implemented.")
            return 1
    else:
        hdf5_path = args.hdf5_path
        if not hdf5_path or not os.path.exists(hdf5_path):
            print(f"HDF5 file not found: {hdf5_path}")
            return 1

    print(f"Loading from {hdf5_path}")

    # Load episodes
    episodes = []

    with h5py.File(hdf5_path, "r") as f:
        episode_names = list(f["data"].keys())
        if args.max_episodes:
            episode_names = episode_names[:args.max_episodes]

        for ep_name in tqdm(episode_names, desc="Loading episodes"):
            ep_group = f["data"][ep_name]

            ep_data = {"images": None, "actions": None, "states": {}}

            # Get images
            if "obs" in ep_group and args.camera in ep_group["obs"]:
                images = np.array(ep_group["obs"][args.camera])
                if images.dtype != np.uint8:
                    if images.max() <= 1.0:
                        images = (images * 255).astype(np.uint8)
                    else:
                        images = images.astype(np.uint8)
                ep_data["images"] = images

            # Get actions
            if "actions" in ep_group:
                ep_data["actions"] = np.array(ep_group["actions"])

            # Get states
            if "obs" in ep_group:
                for key in ["eef_pos", "eef_quat", "gripper_pos", "joint_pos"]:
                    if key in ep_group["obs"]:
                        ep_data["states"][key] = np.array(ep_group["obs"][key])

            if ep_data["images"] is not None:
                episodes.append(ep_data)

    print(f"Loaded {len(episodes)} episodes")

    # Create LeRobot dataset
    lerobot_dir = output_dir / "lerobot"
    create_lerobot_dataset(episodes, str(lerobot_dir), fps=args.fps)

    # Push to hub if requested
    if args.push_to_hub and args.hub_repo_id:
        push_to_huggingface(str(lerobot_dir), args.hub_repo_id)

    # Save dataset info
    info = {
        "format": "lerobot",
        "num_episodes": len(episodes),
        "fps": args.fps,
        "camera": args.camera,
        "image_size": [224, 224, 3],
        "action_dim": 7,
        "task": TASK_INSTRUCTION,
    }

    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nPi-Zero dataset prepared at: {output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
