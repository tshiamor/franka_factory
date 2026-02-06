#!/usr/bin/env python3
# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""
Prepare MCX Card demos for OpenVLA training.

OpenVLA expects RLDS format with:
- RGB images (224×224)
- Language instructions
- Actions (7D: 6 DoF pose + gripper)
- Robot state observations

Usage:
    # From HuggingFace augmented dataset
    python prepare_openvla.py \
        --source huggingface \
        --repo_id tshiamor/mcx-card-cosmos-augmented \
        --output_dir ./openvla_data

    # From local HDF5
    python prepare_openvla.py \
        --source local \
        --hdf5_path /path/to/mcx_card_demos_vla_224.hdf5 \
        --output_dir ./openvla_data
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


# Task instruction for language conditioning
TASK_INSTRUCTION = "Pick up the blue block and place it on the target platform near the MCX network cards."

# Alternative instructions for data augmentation
INSTRUCTION_VARIANTS = [
    "Pick up the blue block and place it on the target platform near the MCX network cards.",
    "Grasp the blue rectangular block and move it to the designated area beside the network cards.",
    "Take the blue component and position it on the platform next to the MCX cards.",
    "Grab the blue block from the table and place it on the target location.",
    "Move the blue block to the platform near the server components.",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data for OpenVLA")
    parser.add_argument("--source", type=str, choices=["huggingface", "local"], default="local")
    parser.add_argument("--repo_id", type=str, default="tshiamor/mcx-card-demos-vla")
    parser.add_argument("--hdf5_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./openvla_data")
    parser.add_argument("--camera", type=str, default="wrist_rgb", help="Camera to use (wrist_rgb or table_rgb)")
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--train_split", type=float, default=0.9)
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

    # Find HDF5 file
    hdf5_files = list(Path(local_dir).glob("*.hdf5"))
    if hdf5_files:
        return str(hdf5_files[0])
    return None


def create_openvla_episode(ep_data: dict, camera: str, instruction: str) -> dict:
    """Convert episode to OpenVLA format."""

    obs = ep_data.get("obs", {})
    actions = ep_data.get("actions", np.array([]))

    # Get RGB images
    if camera not in obs:
        return None

    images = obs[camera]
    num_steps = len(images)

    if num_steps == 0:
        return None

    # Ensure images are uint8, 224×224
    if images.dtype != np.uint8:
        if images.max() <= 1.0:
            images = (images * 255).astype(np.uint8)
        else:
            images = images.astype(np.uint8)

    # Build episode data
    episode = {
        "observation": {
            "image": images,  # (T, 224, 224, 3)
        },
        "action": actions[:num_steps] if len(actions) >= num_steps else actions,
        "language_instruction": instruction,
        "is_terminal": np.zeros(num_steps, dtype=bool),
        "is_first": np.zeros(num_steps, dtype=bool),
    }

    episode["is_first"][0] = True
    episode["is_terminal"][-1] = True

    # Add robot state if available
    if "eef_pos" in obs:
        episode["observation"]["eef_pos"] = obs["eef_pos"][:num_steps]
    if "eef_quat" in obs:
        episode["observation"]["eef_quat"] = obs["eef_quat"][:num_steps]
    if "gripper_pos" in obs:
        episode["observation"]["gripper_pos"] = obs["gripper_pos"][:num_steps]
    if "joint_pos" in obs:
        episode["observation"]["joint_pos"] = obs["joint_pos"][:num_steps]

    return episode


def save_openvla_tfrecord(episodes: list, output_path: str):
    """Save episodes in TFRecord format for OpenVLA."""
    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow not installed. Saving as numpy instead.")
        save_openvla_numpy(episodes, output_path)
        return

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    writer = tf.io.TFRecordWriter(output_path)

    for ep_idx, episode in enumerate(tqdm(episodes, desc="Writing TFRecords")):
        images = episode["observation"]["image"]
        actions = episode["action"]
        instruction = episode["language_instruction"]

        for t in range(len(images)):
            feature = {
                "image": _bytes_feature(images[t].tobytes()),
                "image_shape": tf.train.Feature(int64_list=tf.train.Int64List(value=images[t].shape)),
                "action": _float_feature(actions[t] if t < len(actions) else np.zeros(7)),
                "language_instruction": _bytes_feature(instruction.encode()),
                "episode_id": _int64_feature(ep_idx),
                "step": _int64_feature(t),
                "is_first": _int64_feature(int(episode["is_first"][t])),
                "is_terminal": _int64_feature(int(episode["is_terminal"][t])),
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    writer.close()
    print(f"Saved TFRecord to {output_path}")


def save_openvla_numpy(episodes: list, output_dir: str):
    """Save episodes as numpy files (fallback if TF not installed)."""
    os.makedirs(output_dir, exist_ok=True)

    for ep_idx, episode in enumerate(tqdm(episodes, desc="Saving episodes")):
        ep_dir = Path(output_dir) / f"episode_{ep_idx:04d}"
        ep_dir.mkdir(exist_ok=True)

        # Save images
        np.save(ep_dir / "images.npy", episode["observation"]["image"])

        # Save actions
        np.save(ep_dir / "actions.npy", episode["action"])

        # Save metadata
        metadata = {
            "language_instruction": episode["language_instruction"],
            "num_steps": len(episode["observation"]["image"]),
            "is_first": episode["is_first"].tolist(),
            "is_terminal": episode["is_terminal"].tolist(),
        }
        with open(ep_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save robot state if available
        for key in ["eef_pos", "eef_quat", "gripper_pos", "joint_pos"]:
            if key in episode["observation"]:
                np.save(ep_dir / f"{key}.npy", episode["observation"][key])

    print(f"Saved {len(episodes)} episodes to {output_dir}")


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get HDF5 path
    if args.source == "huggingface":
        download_dir = output_dir / "download"
        hdf5_path = download_from_huggingface(args.repo_id, str(download_dir))
        if not hdf5_path:
            print("No HDF5 file found in dataset. Using video-based loading...")
            print("Note: Video-based loading not yet implemented for OpenVLA")
            return 1
    else:
        hdf5_path = args.hdf5_path
        if not hdf5_path or not os.path.exists(hdf5_path):
            print(f"HDF5 file not found: {hdf5_path}")
            return 1

    print(f"Loading data from {hdf5_path}")

    # Load and convert episodes
    episodes = []

    with h5py.File(hdf5_path, "r") as f:
        episode_names = list(f["data"].keys())
        if args.max_episodes:
            episode_names = episode_names[:args.max_episodes]

        for ep_name in tqdm(episode_names, desc="Loading episodes"):
            ep_group = f["data"][ep_name]

            # Load episode data
            ep_data = {"obs": {}}

            if "actions" in ep_group:
                ep_data["actions"] = np.array(ep_group["actions"])

            if "obs" in ep_group:
                for key in ep_group["obs"].keys():
                    ep_data["obs"][key] = np.array(ep_group["obs"][key])

            # Use variant instructions for diversity
            instruction = INSTRUCTION_VARIANTS[len(episodes) % len(INSTRUCTION_VARIANTS)]

            # Convert to OpenVLA format
            episode = create_openvla_episode(ep_data, args.camera, instruction)
            if episode:
                episodes.append(episode)

    print(f"Loaded {len(episodes)} episodes")

    # Split train/val
    split_idx = int(len(episodes) * args.train_split)
    train_episodes = episodes[:split_idx]
    val_episodes = episodes[split_idx:]

    print(f"Train: {len(train_episodes)}, Val: {len(val_episodes)}")

    # Save in OpenVLA format
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"

    save_openvla_numpy(train_episodes, str(train_dir))
    save_openvla_numpy(val_episodes, str(val_dir))

    # Save dataset info
    info = {
        "dataset_name": "mcx_card_block_insert",
        "task_instruction": TASK_INSTRUCTION,
        "num_episodes": len(episodes),
        "num_train": len(train_episodes),
        "num_val": len(val_episodes),
        "camera": args.camera,
        "image_size": [224, 224, 3],
        "action_dim": 7,
        "source": args.repo_id if args.source == "huggingface" else hdf5_path,
    }

    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nOpenVLA dataset prepared at: {output_dir}")
    print(f"Dataset info saved to: {output_dir / 'dataset_info.json'}")

    return 0


if __name__ == "__main__":
    exit(main())
