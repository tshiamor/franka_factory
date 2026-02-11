#!/usr/bin/env python3
"""Convert MCX Card HDF5 dataset to GR00T LeRobot v2 format.

Converts the Isaac Lab teleop HDF5 demos into the directory structure
expected by Isaac-GR00T's launch_finetune.py:

    output_dir/
    ├── meta/
    │   ├── info.json
    │   ├── episodes.jsonl
    │   ├── tasks.jsonl
    │   └── modality.json
    ├── data/
    │   └── chunk-000/
    │       ├── episode_000000.parquet
    │       └── ...
    └── videos/
        └── chunk-000/
            ├── observation.images.wrist/
            │   ├── episode_000000.mp4
            │   └── ...
            └── observation.images.table/
                ├── episode_000000.mp4
                └── ...

State vector (8D): eef_pos(3) + eef_quat(4) + gripper(1)
Action vector (7D): arm_delta(6) + gripper(1)

Usage:
    python convert_hdf5_to_lerobot_v2.py \
        --hdf5 /home/tshiamo/IsaacLab/mcx_card_training_augmented.hdf5 \
        --output /home/tshiamo/groot_data/mcx_card_lerobot_v2 \
        --fps 30
"""

import argparse
import json
import os

import h5py
import imageio
import numpy as np
import pandas as pd
from tqdm import tqdm


TASK_DESCRIPTION = "pick up the blue block and insert it into the MCX card slot"

# Cameras to extract
CAMERA_KEYS = {
    "observation.images.wrist": "wrist_rgb",
    "observation.images.table": "table_rgb",
}


def encode_video(frames: np.ndarray, output_path: str, fps: int = 30):
    """Encode numpy frames (T, H, W, 3) uint8 to MP4 using imageio + ffmpeg."""
    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        output_params=["-preset", "fast"],
    )
    for i in range(frames.shape[0]):
        writer.append_data(frames[i])
    writer.close()


def convert(hdf5_path: str, output_dir: str, fps: int = 30, chunks_size: int = 1000):
    """Convert HDF5 demos to LeRobot v2 format."""
    # Create directory structure
    os.makedirs(f"{output_dir}/meta", exist_ok=True)

    with h5py.File(hdf5_path, "r") as f:
        demos = sorted(f["data"].keys())
        print(f"Found {len(demos)} demos in {hdf5_path}")

        global_index = 0
        total_frames = 0
        episodes_meta = []

        for ep_idx, demo_name in enumerate(tqdm(demos, desc="Converting")):
            demo = f["data"][demo_name]
            T = demo["actions"].shape[0]

            chunk_id = ep_idx // chunks_size
            chunk_dir = f"chunk-{chunk_id:03d}"

            # Ensure directories exist
            os.makedirs(f"{output_dir}/data/{chunk_dir}", exist_ok=True)
            for cam_key in CAMERA_KEYS:
                os.makedirs(f"{output_dir}/videos/{chunk_dir}/{cam_key}", exist_ok=True)

            # Build state: eef_pos(3) + eef_quat(4) + gripper(1) = 8D
            eef_pos = demo["obs"]["eef_pos"][:].astype(np.float32)      # (T, 3)
            eef_quat = demo["obs"]["eef_quat"][:].astype(np.float32)    # (T, 4)
            gripper = demo["obs"]["gripper_pos"][:, :1].astype(np.float32)  # (T, 1)
            state = np.concatenate([eef_pos, eef_quat, gripper], axis=1)  # (T, 8)

            # Actions as-is: arm_delta(6) + gripper(1) = 7D
            actions = demo["actions"][:].astype(np.float32)  # (T, 7)

            # Build parquet rows
            records = []
            for t in range(T):
                records.append({
                    "observation.state": state[t],
                    "action": actions[t],
                    "timestamp": float(t) / fps,
                    "frame_index": t,
                    "episode_index": ep_idx,
                    "index": global_index + t,
                    "task_index": 0,
                })

            df = pd.DataFrame(records)
            df.to_parquet(f"{output_dir}/data/{chunk_dir}/episode_{ep_idx:06d}.parquet")

            # Encode videos
            for cam_key, hdf5_key in CAMERA_KEYS.items():
                frames = demo["obs"][hdf5_key][:]  # (T, 224, 224, 3) uint8
                vid_path = f"{output_dir}/videos/{chunk_dir}/{cam_key}/episode_{ep_idx:06d}.mp4"
                encode_video(frames, vid_path, fps)

            episodes_meta.append({
                "episode_index": ep_idx,
                "tasks": [TASK_DESCRIPTION],
                "length": T,
            })

            global_index += T
            total_frames += T

    num_episodes = len(episodes_meta)
    num_chunks = (num_episodes - 1) // chunks_size + 1

    # -- meta/info.json --
    info = {
        "codebase_version": "v2.1",
        "robot_type": "franka_panda",
        "total_episodes": num_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": chunks_size,
        "fps": fps,
        "splits": {"train": f"0:{num_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {
                "dtype": "float32",
                "names": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
                "shape": [7],
            },
            "observation.state": {
                "dtype": "float32",
                "names": ["eef_x", "eef_y", "eef_z", "qx", "qy", "qz", "qw", "gripper"],
                "shape": [8],
            },
            "observation.images.wrist": {
                "dtype": "video",
                "shape": [224, 224, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": 224,
                    "video.width": 224,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": fps,
                    "video.channels": 3,
                    "has_audio": False,
                },
            },
            "observation.images.table": {
                "dtype": "video",
                "shape": [224, 224, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": 224,
                    "video.width": 224,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": fps,
                    "video.channels": 3,
                    "has_audio": False,
                },
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
        "total_chunks": num_chunks,
        "total_videos": num_episodes * len(CAMERA_KEYS),
    }
    with open(f"{output_dir}/meta/info.json", "w") as fp:
        json.dump(info, fp, indent=4)

    # -- meta/episodes.jsonl --
    with open(f"{output_dir}/meta/episodes.jsonl", "w") as fp:
        for ep in episodes_meta:
            fp.write(json.dumps(ep) + "\n")

    # -- meta/tasks.jsonl --
    with open(f"{output_dir}/meta/tasks.jsonl", "w") as fp:
        fp.write(json.dumps({"task_index": 0, "task": TASK_DESCRIPTION}) + "\n")

    # -- meta/modality.json --
    modality = {
        "state": {
            "eef_pos": {"start": 0, "end": 3},
            "eef_quat": {"start": 3, "end": 7},
            "gripper": {"start": 7, "end": 8},
        },
        "action": {
            "arm": {"start": 0, "end": 6},
            "gripper": {"start": 6, "end": 7},
        },
        "video": {
            "wrist": {"original_key": "observation.images.wrist"},
            "table": {"original_key": "observation.images.table"},
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"},
        },
    }
    with open(f"{output_dir}/meta/modality.json", "w") as fp:
        json.dump(modality, fp, indent=4)

    print(f"\nConversion complete!")
    print(f"  Episodes: {num_episodes}")
    print(f"  Total frames: {total_frames}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HDF5 to LeRobot v2 format")
    parser.add_argument("--hdf5", type=str, required=True, help="Input HDF5 path")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    args = parser.parse_args()
    convert(args.hdf5, args.output, args.fps)
