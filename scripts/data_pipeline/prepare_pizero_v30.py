#!/usr/bin/env python3
"""
Prepare MCX Card demos for LeRobot v3.0 format.

LeRobot v3.0 requires:
- data/chunk-XXX/file-XXX.parquet (frame data)
- videos/<feature>/chunk-XXX/episode_XXXXXX.mp4
- meta/info.json with features schema
- meta/episodes/chunk-XXX/file-XXX.parquet
- meta/tasks.parquet
- meta/stats.json

Usage:
    python prepare_pizero_v30.py \
        --hdf5_path /path/to/mcx_card_demos_vla_224.hdf5 \
        --output_dir ./pizero_v30_data \
        --push_to_hub \
        --hub_repo_id your-username/mcx-card-pizero
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


TASK_INSTRUCTION = "Pick up the card and place it on the target."
CODEBASE_VERSION = "3.0"
FPS = 30
IMAGE_SIZE = 224


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data for LeRobot v3.0")
    parser.add_argument("--hdf5_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./pizero_v30_data")
    parser.add_argument("--camera", type=str, default="wrist_rgb")
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_repo_id", type=str, default=None)
    return parser.parse_args()


def create_lerobot_v30_dataset(hdf5_path: str, output_dir: str, camera: str = "wrist_rgb",
                                max_episodes: int = None, fps: int = 30):
    """Create LeRobot v3.0 compatible dataset."""
    output_path = Path(output_dir)

    # Create directories
    data_dir = output_path / "data" / "chunk-000"
    meta_dir = output_path / "meta"
    episodes_dir = meta_dir / "episodes" / "chunk-000"
    videos_dir = output_path / "videos" / f"observation.images.{camera}" / "chunk-000"

    for d in [data_dir, meta_dir, episodes_dir, videos_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load HDF5 and process
    all_frames = []
    episode_data = []
    total_frames = 0

    with h5py.File(hdf5_path, "r") as f:
        episode_names = sorted(list(f["data"].keys()))
        if max_episodes:
            episode_names = episode_names[:max_episodes]

        num_episodes = len(episode_names)
        action_dim = None
        state_dim = None

        for ep_idx, ep_name in enumerate(tqdm(episode_names, desc="Processing episodes")):
            ep_group = f["data"][ep_name]

            # Get images
            if "obs" not in ep_group or camera not in ep_group["obs"]:
                print(f"Warning: {ep_name} missing {camera}, skipping")
                continue

            images = np.array(ep_group["obs"][camera])
            if images.dtype != np.uint8:
                if images.max() <= 1.0:
                    images = (images * 255).astype(np.uint8)
                else:
                    images = images.astype(np.uint8)

            # Get actions
            actions = np.array(ep_group["actions"]) if "actions" in ep_group else None
            if actions is not None:
                action_dim = actions.shape[-1]

            # Get states
            states = {}
            if "obs" in ep_group:
                for key in ["joint_pos", "eef_pos", "eef_quat", "gripper_pos"]:
                    if key in ep_group["obs"]:
                        states[key] = np.array(ep_group["obs"][key])

            # Compute state dimension (concatenate all state components)
            if states:
                state_arrays = [states[k] for k in sorted(states.keys()) if len(states[k]) > 0]
                if state_arrays:
                    combined_state = np.concatenate([s[0] for s in state_arrays])
                    state_dim = combined_state.shape[0]

            num_steps = len(images)
            ep_start_frame = total_frames

            # Save video
            try:
                import imageio
                video_path = videos_dir / f"episode_{ep_idx:06d}.mp4"
                writer = imageio.get_writer(str(video_path), fps=fps, codec='libx264', quality=8)
                for img in images:
                    writer.append_data(img)
                writer.close()
            except Exception as e:
                print(f"Warning: Could not save video for {ep_name}: {e}")

            # Build frame data
            for t in range(num_steps):
                frame = {
                    "episode_index": ep_idx,
                    "frame_index": total_frames,
                    "timestamp": t / fps,
                }

                # Add action
                if actions is not None and t < len(actions):
                    for i in range(actions.shape[-1]):
                        frame[f"action_{i}"] = float(actions[t, i])

                # Add state (concatenated)
                if states:
                    state_idx = 0
                    for key in sorted(states.keys()):
                        if t < len(states[key]):
                            state_val = states[key][t]
                            if isinstance(state_val, np.ndarray):
                                for i, v in enumerate(state_val):
                                    frame[f"observation.state_{state_idx}"] = float(v)
                                    state_idx += 1
                            else:
                                frame[f"observation.state_{state_idx}"] = float(state_val)
                                state_idx += 1

                all_frames.append(frame)
                total_frames += 1

            # Episode metadata
            episode_data.append({
                "episode_index": ep_idx,
                "tasks": json.dumps([TASK_INSTRUCTION]),
                "length": num_steps,
            })

    # Save frame data as parquet
    print(f"Saving {len(all_frames)} frames...")
    df = pd.DataFrame(all_frames)
    df.to_parquet(data_dir / "file-000.parquet", index=False)

    # Save episode metadata
    ep_df = pd.DataFrame(episode_data)
    ep_df.to_parquet(episodes_dir / "file-000.parquet", index=False)

    # Save tasks
    tasks_df = pd.DataFrame([{"task_index": 0, "task": TASK_INSTRUCTION}])
    tasks_df.to_parquet(meta_dir / "tasks.parquet", index=False)

    # Build features schema
    features = {
        f"observation.images.{camera}": {
            "dtype": "video",
            "shape": [IMAGE_SIZE, IMAGE_SIZE, 3],
            "names": ["height", "width", "channels"],
            "video_info": {
                "video.fps": fps,
                "video.codec": "libx264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            }
        },
    }

    # Add action features
    if action_dim:
        features["action"] = {
            "dtype": "float32",
            "shape": [action_dim],
            "names": [f"action_{i}" for i in range(action_dim)],
        }

    # Add state features
    if state_dim:
        features["observation.state"] = {
            "dtype": "float32",
            "shape": [state_dim],
            "names": [f"state_{i}" for i in range(state_dim)],
        }

    # Save info.json
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": "franka_panda",
        "fps": fps,
        "total_episodes": num_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": num_episodes,
        "total_chunks": 1,
        "chunks_size": 1000,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/episode_{episode_index:06d}.mp4",
        "features": features,
        "splits": {"train": f"0:{num_episodes}"},
    }

    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # Compute and save stats
    stats = {}
    for col in df.columns:
        if col.startswith("action_") or col.startswith("observation.state_"):
            values = df[col].dropna().values
            if len(values) > 0:
                stats[col] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }

    # Add image stats (ImageNet defaults)
    stats[f"observation.images.{camera}"] = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }

    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nLeRobot v3.0 dataset created at: {output_dir}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Frames: {total_frames}")
    print(f"  Action dim: {action_dim}")
    print(f"  State dim: {state_dim}")

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
        commit_message="Upload LeRobot v3.0 dataset for Pi-Zero training",
    )

    # Tag with version
    try:
        api.create_tag(repo_id=repo_id, tag=f"v{CODEBASE_VERSION}", repo_type="dataset")
    except Exception as e:
        print(f"Note: Could not create tag: {e}")

    print(f"Pushed to https://huggingface.co/datasets/{repo_id}")


def main():
    args = parse_args()

    if not os.path.exists(args.hdf5_path):
        print(f"HDF5 file not found: {args.hdf5_path}")
        return 1

    output_path = create_lerobot_v30_dataset(
        args.hdf5_path,
        args.output_dir,
        args.camera,
        args.max_episodes,
        args.fps,
    )

    if args.push_to_hub and args.hub_repo_id:
        push_to_huggingface(str(output_path), args.hub_repo_id)

    return 0


if __name__ == "__main__":
    exit(main())
