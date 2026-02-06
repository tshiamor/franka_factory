#!/usr/bin/env python3
# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""
Prepare MCX Card demos for NVIDIA GR00T N1 training.

GR00T expects:
- RGB images (224×224)
- Actions (7D for Franka: 6 DoF + gripper)
- Robot proprioception (joint positions, velocities, EEF pose)
- Language annotations

Usage:
    # From HuggingFace augmented dataset
    python prepare_groot.py \
        --source huggingface \
        --repo_id tshiamor/mcx-card-cosmos-augmented \
        --output_dir ./groot_data

    # From local HDF5
    python prepare_groot.py \
        --source local \
        --hdf5_path /path/to/mcx_card_demos_vla_224.hdf5 \
        --output_dir ./groot_data
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


TASK_DESCRIPTION = "Pick up the blue block and place it on the target platform near the MCX network cards."

# GR00T robot configuration for Franka Panda
ROBOT_CONFIG = {
    "robot_type": "franka_panda",
    "num_joints": 7,
    "num_gripper_joints": 2,
    "action_space": {
        "type": "continuous",
        "dim": 7,  # 6 DoF pose delta + 1 gripper
        "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"],
    },
    "observation_space": {
        "image": {"shape": [224, 224, 3], "dtype": "uint8"},
        "joint_pos": {"shape": [9], "dtype": "float32"},  # 7 arm + 2 gripper
        "joint_vel": {"shape": [9], "dtype": "float32"},
        "eef_pos": {"shape": [3], "dtype": "float32"},
        "eef_quat": {"shape": [4], "dtype": "float32"},
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data for NVIDIA GR00T")
    parser.add_argument("--source", type=str, choices=["huggingface", "local"], default="local")
    parser.add_argument("--repo_id", type=str, default="tshiamor/mcx-card-demos-vla")
    parser.add_argument("--hdf5_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./groot_data")
    parser.add_argument("--camera", type=str, default="wrist_rgb")
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--chunk_size", type=int, default=100, help="Episodes per HDF5 chunk")
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


def create_groot_episode(ep_data: dict, camera: str) -> dict:
    """Convert episode to GR00T format."""

    obs = ep_data.get("obs", {})
    actions = ep_data.get("actions", np.array([]))

    if camera not in obs:
        return None

    images = obs[camera]
    num_steps = len(images)

    if num_steps == 0:
        return None

    # Ensure images are uint8
    if images.dtype != np.uint8:
        if images.max() <= 1.0:
            images = (images * 255).astype(np.uint8)
        else:
            images = images.astype(np.uint8)

    episode = {
        "observation": {
            "image": images,
        },
        "action": actions[:num_steps] if len(actions) >= num_steps else np.zeros((num_steps, 7)),
        "language": TASK_DESCRIPTION,
        "done": np.zeros(num_steps, dtype=bool),
        "reward": np.zeros(num_steps, dtype=np.float32),
    }

    episode["done"][-1] = True
    episode["reward"][-1] = 1.0  # Success reward at end

    # Add proprioception
    for key in ["eef_pos", "eef_quat", "joint_pos", "joint_vel", "gripper_pos"]:
        if key in obs:
            data = obs[key][:num_steps]
            episode["observation"][key] = data

    return episode


def save_groot_hdf5(episodes: list, output_path: str, split: str = "train"):
    """Save episodes in GR00T HDF5 format."""

    with h5py.File(output_path, "w") as f:
        # Metadata
        f.attrs["format"] = "groot_v1"
        f.attrs["robot_type"] = ROBOT_CONFIG["robot_type"]
        f.attrs["num_episodes"] = len(episodes)
        f.attrs["split"] = split
        f.attrs["task_description"] = TASK_DESCRIPTION

        # Create groups
        data_group = f.create_group("data")

        for ep_idx, episode in enumerate(tqdm(episodes, desc=f"Saving {split}")):
            ep_name = f"episode_{ep_idx:06d}"
            ep_group = data_group.create_group(ep_name)

            # Observations
            obs_group = ep_group.create_group("observation")

            # Images - store compressed
            images = episode["observation"]["image"]
            obs_group.create_dataset(
                "image",
                data=images,
                compression="gzip",
                compression_opts=4,
                chunks=(1, 224, 224, 3),
            )

            # Proprioception
            for key in ["eef_pos", "eef_quat", "joint_pos", "joint_vel", "gripper_pos"]:
                if key in episode["observation"]:
                    obs_group.create_dataset(key, data=episode["observation"][key])

            # Actions
            ep_group.create_dataset("action", data=episode["action"])

            # Done and reward
            ep_group.create_dataset("done", data=episode["done"])
            ep_group.create_dataset("reward", data=episode["reward"])

            # Language
            ep_group.attrs["language"] = episode["language"]
            ep_group.attrs["num_steps"] = len(images)

    print(f"Saved {len(episodes)} episodes to {output_path}")


def save_groot_tfrecord(episodes: list, output_dir: str, split: str = "train"):
    """Save episodes in TFRecord format (alternative format for GR00T)."""
    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow not installed. Using HDF5 format.")
        return

    output_path = Path(output_dir) / f"{split}.tfrecord"

    def serialize_example(episode, step_idx):
        image = episode["observation"]["image"][step_idx]
        action = episode["action"][step_idx] if step_idx < len(episode["action"]) else np.zeros(7)

        feature = {
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
            "action": tf.train.Feature(float_list=tf.train.FloatList(value=action.flatten())),
            "language": tf.train.Feature(bytes_list=tf.train.BytesList(value=[episode["language"].encode()])),
            "done": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(episode["done"][step_idx])])),
        }

        # Add proprioception
        for key in ["eef_pos", "eef_quat", "joint_pos"]:
            if key in episode["observation"]:
                data = episode["observation"][key][step_idx]
                feature[key] = tf.train.Feature(float_list=tf.train.FloatList(value=data.flatten()))

        return tf.train.Example(features=tf.train.Features(feature=feature))

    with tf.io.TFRecordWriter(str(output_path)) as writer:
        for episode in tqdm(episodes, desc=f"Writing {split} TFRecords"):
            for t in range(len(episode["observation"]["image"])):
                example = serialize_example(episode, t)
                writer.write(example.SerializeToString())

    print(f"Saved TFRecord to {output_path}")


def create_groot_metadata(output_dir: str, train_episodes: int, val_episodes: int):
    """Create metadata files for GR00T training."""

    metadata = {
        "dataset_name": "mcx_card_block_insert",
        "task_description": TASK_DESCRIPTION,
        "robot_config": ROBOT_CONFIG,
        "splits": {
            "train": {"num_episodes": train_episodes},
            "val": {"num_episodes": val_episodes},
        },
        "total_episodes": train_episodes + val_episodes,
        "modalities": ["rgb", "proprioception", "action", "language"],
    }

    with open(Path(output_dir) / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # GR00T config file
    groot_config = {
        "model": {
            "type": "groot_n1",
            "image_size": 224,
            "patch_size": 14,
            "action_dim": 7,
            "proprio_dim": 22,  # 9 joint_pos + 9 joint_vel + 3 eef_pos + 4 eef_quat - 3 overlap
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 1e-4,
            "epochs": 100,
            "action_horizon": 16,
            "observation_horizon": 2,
        },
        "data": {
            "train_path": "train.hdf5",
            "val_path": "val.hdf5",
            "image_key": "observation/image",
            "action_key": "action",
        },
    }

    with open(Path(output_dir) / "groot_config.yaml", "w") as f:
        import yaml
        yaml.dump(groot_config, f, default_flow_style=False)

    print(f"Created metadata and config files in {output_dir}")


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get HDF5 path
    if args.source == "huggingface":
        download_dir = output_dir / "download"
        hdf5_path = download_from_huggingface(args.repo_id, str(download_dir))
        if not hdf5_path:
            print("No HDF5 file found in dataset.")
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

            ep_data = {"obs": {}}

            if "actions" in ep_group:
                ep_data["actions"] = np.array(ep_group["actions"])

            if "obs" in ep_group:
                for key in ep_group["obs"].keys():
                    ep_data["obs"][key] = np.array(ep_group["obs"][key])

            episode = create_groot_episode(ep_data, args.camera)
            if episode:
                episodes.append(episode)

    print(f"Loaded {len(episodes)} episodes")

    # Split train/val
    split_idx = int(len(episodes) * args.train_split)
    train_episodes = episodes[:split_idx]
    val_episodes = episodes[split_idx:]

    print(f"Train: {len(train_episodes)}, Val: {len(val_episodes)}")

    # Save in GR00T HDF5 format
    save_groot_hdf5(train_episodes, str(output_dir / "train.hdf5"), split="train")
    save_groot_hdf5(val_episodes, str(output_dir / "val.hdf5"), split="val")

    # Create metadata
    create_groot_metadata(str(output_dir), len(train_episodes), len(val_episodes))

    print(f"\nGR00T dataset prepared at: {output_dir}")
    print(f"  - train.hdf5: {len(train_episodes)} episodes")
    print(f"  - val.hdf5: {len(val_episodes)} episodes")
    print(f"  - metadata.json: dataset configuration")
    print(f"  - groot_config.yaml: training configuration")

    return 0


if __name__ == "__main__":
    exit(main())
