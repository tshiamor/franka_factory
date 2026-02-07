#!/usr/bin/env python3
"""
Prepare MCX Card demos for LeRobot v3.0 format.

LeRobot v3.0 requires:
- data/chunk-XXX/file-XXX.parquet (frame data with nested list columns)
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
import pyarrow as pa
import pyarrow.parquet as pq
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
    """Create LeRobot v3.0 compatible dataset with nested list columns."""
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

    # For stats computation
    all_actions = []
    all_states = []

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

            # Get states and concatenate them
            state_arrays = []
            if "obs" in ep_group:
                for key in ["eef_pos", "eef_quat", "gripper_pos", "joint_pos"]:
                    if key in ep_group["obs"]:
                        state_arrays.append(np.array(ep_group["obs"][key]))

            # Concatenate state components along last axis
            if state_arrays:
                # All arrays should have shape (T, dim_i)
                combined_states = np.concatenate(state_arrays, axis=-1)
                state_dim = combined_states.shape[-1]
            else:
                combined_states = None

            num_steps = len(images)

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

            # Build frame data with nested lists
            for t in range(num_steps):
                frame = {
                    "episode_index": np.int64(ep_idx),
                    "frame_index": np.int64(total_frames),
                    "timestamp": float(t / fps),
                    "task_index": np.int64(0),
                    "index": np.int64(total_frames),
                }

                # Add action as a list (not individual columns)
                if actions is not None and t < len(actions):
                    action_list = [float(x) for x in actions[t]]
                    frame["action"] = action_list
                    all_actions.append(actions[t])

                # Add state as a list (not individual columns)
                if combined_states is not None and t < len(combined_states):
                    state_list = [float(x) for x in combined_states[t]]
                    frame["observation.state"] = state_list
                    all_states.append(combined_states[t])

                all_frames.append(frame)
                total_frames += 1

            # Episode metadata
            episode_data.append({
                "episode_index": np.int64(ep_idx),
                "tasks": json.dumps([TASK_INSTRUCTION]),
                "length": np.int64(num_steps),
            })

    # Create PyArrow schema for nested list columns
    print(f"Saving {len(all_frames)} frames with nested list columns...")

    # Build the schema (use variable-length lists like LeRobot reference datasets)
    schema_fields = [
        pa.field("observation.state", pa.list_(pa.float32())),
        pa.field("action", pa.list_(pa.float32())),
        pa.field("episode_index", pa.int64()),
        pa.field("frame_index", pa.int64()),
        pa.field("timestamp", pa.float32()),
        pa.field("next.done", pa.bool_()),
        pa.field("index", pa.int64()),
        pa.field("task_index", pa.int64()),
    ]

    schema = pa.schema(schema_fields)

    # Convert to PyArrow table (use variable-length lists)
    action_data = [f.get("action", [0.0] * action_dim) for f in all_frames]
    state_data = [f.get("observation.state", [0.0] * state_dim) for f in all_frames]

    arrays = {
        "observation.state": pa.array(state_data, type=pa.list_(pa.float32())),
        "action": pa.array(action_data, type=pa.list_(pa.float32())),
        "episode_index": pa.array([f["episode_index"] for f in all_frames], type=pa.int64()),
        "frame_index": pa.array([f["frame_index"] for f in all_frames], type=pa.int64()),
        "timestamp": pa.array([f["timestamp"] for f in all_frames], type=pa.float32()),
        "next.done": pa.array([False] * len(all_frames), type=pa.bool_()),
        "index": pa.array([f["index"] for f in all_frames], type=pa.int64()),
        "task_index": pa.array([f["task_index"] for f in all_frames], type=pa.int64()),
    }

    # Create table and write to parquet
    table = pa.table(arrays, schema=schema)
    pq.write_table(table, data_dir / "file-000.parquet")

    # Save episode metadata
    ep_df = pd.DataFrame(episode_data)
    ep_df.to_parquet(episodes_dir / "file-000.parquet", index=False)

    # Save tasks
    tasks_df = pd.DataFrame([{"task_index": 0, "task": TASK_INSTRUCTION}])
    tasks_df.to_parquet(meta_dir / "tasks.parquet", index=False)

    # Build features schema (must include ALL columns in parquet)
    features = {
        f"observation.images.{camera}": {
            "dtype": "video",
            "shape": [IMAGE_SIZE, IMAGE_SIZE, 3],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": float(fps),
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            }
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [state_dim] if state_dim else [1],
            "names": {"motors": [f"state_{i}" for i in range(state_dim or 1)]},
            "fps": float(fps),
        },
        "action": {
            "dtype": "float32",
            "shape": [action_dim] if action_dim else [1],
            "names": {"motors": [f"action_{i}" for i in range(action_dim or 1)]},
            "fps": float(fps),
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [1],
            "names": None,
            "fps": float(fps),
        },
        "frame_index": {
            "dtype": "int64",
            "shape": [1],
            "names": None,
            "fps": float(fps),
        },
        "timestamp": {
            "dtype": "float32",
            "shape": [1],
            "names": None,
            "fps": float(fps),
        },
        "next.done": {
            "dtype": "bool",
            "shape": [1],
            "names": None,
            "fps": float(fps),
        },
        "index": {
            "dtype": "int64",
            "shape": [1],
            "names": None,
            "fps": float(fps),
        },
        "task_index": {
            "dtype": "int64",
            "shape": [1],
            "names": None,
            "fps": float(fps),
        },
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

    # Compute and save stats (for the nested list columns)
    stats = {}

    if all_actions:
        all_actions_arr = np.array(all_actions)
        stats["action"] = {
            "mean": [float(x) for x in np.mean(all_actions_arr, axis=0)],
            "std": [float(x) for x in np.std(all_actions_arr, axis=0)],
            "min": [float(x) for x in np.min(all_actions_arr, axis=0)],
            "max": [float(x) for x in np.max(all_actions_arr, axis=0)],
        }

    if all_states:
        all_states_arr = np.array(all_states)
        stats["observation.state"] = {
            "mean": [float(x) for x in np.mean(all_states_arr, axis=0)],
            "std": [float(x) for x in np.std(all_states_arr, axis=0)],
            "min": [float(x) for x in np.min(all_states_arr, axis=0)],
            "max": [float(x) for x in np.max(all_states_arr, axis=0)],
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
