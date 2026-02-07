#!/usr/bin/env python3
# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""
Reconstruct HDF5 dataset combining augmented videos with original states/actions.

Cosmos augmentation only generates videos - this script pairs them with the
original robot states and actions to create a complete training dataset.

The key insight: Cosmos preserves motion/structure while changing visual appearance,
so original actions/states remain valid for augmented videos.

Usage:
    # Pull BOTH datasets from HuggingFace and merge:
    python reconstruct_augmented_hdf5.py \
        --original_repo tshiamor/mcx-card-demos-vla \
        --augmented_repo tshiamor/mcx-card-cosmos-augmented \
        --output_hdf5 ./mcx_card_augmented_full.hdf5

    # Or use local original HDF5 + HuggingFace augmented:
    python reconstruct_augmented_hdf5.py \
        --original_hdf5 /path/to/mcx_card_demos_vla_224.hdf5 \
        --augmented_repo tshiamor/mcx-card-cosmos-augmented \
        --output_hdf5 ./mcx_card_augmented_full.hdf5

    # Or fully local:
    python reconstruct_augmented_hdf5.py \
        --original_hdf5 /path/to/mcx_card_demos_vla_224.hdf5 \
        --augmented_dir ./cosmos_augmented/videos \
        --output_hdf5 ./mcx_card_augmented_full.hdf5
"""

import argparse
import json
import os
import re
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Reconstruct HDF5 with augmented videos + original states")
    parser.add_argument("--original_hdf5", type=str, default=None, help="Local path to original HDF5 with states/actions")
    parser.add_argument("--original_repo", type=str, default=None, help="HuggingFace repo with original HDF5 (e.g., tshiamor/mcx-card-demos-vla)")
    parser.add_argument("--augmented_repo", type=str, default=None, help="HuggingFace repo with augmented videos")
    parser.add_argument("--augmented_dir", type=str, default=None, help="Local dir with augmented videos")
    parser.add_argument("--output_hdf5", type=str, required=True, help="Output HDF5 path")
    parser.add_argument("--cameras", type=str, nargs="+", default=["wrist_rgb", "table_rgb"])
    parser.add_argument("--include_originals", action="store_true", default=True, help="Include original demos too")
    parser.add_argument("--max_episodes", type=int, default=None)
    return parser.parse_args()


def download_original_hdf5(repo_id: str, local_dir: str) -> str:
    """Download original HDF5 from HuggingFace."""
    from huggingface_hub import snapshot_download

    print(f"Downloading original dataset from {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=["*.hdf5"],
    )

    # Find HDF5 file
    hdf5_files = list(Path(local_dir).glob("*.hdf5"))
    if hdf5_files:
        print(f"Found HDF5: {hdf5_files[0]}")
        return str(hdf5_files[0])

    return None


def download_augmented_videos(repo_id: str, local_dir: str) -> str:
    """Download augmented videos from HuggingFace."""
    from huggingface_hub import snapshot_download

    print(f"Downloading augmented videos from {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=["videos/*", "*.mp4"],
    )

    # Find videos directory
    videos_dir = Path(local_dir) / "videos"
    if videos_dir.exists():
        return str(videos_dir)

    # Check if videos are in root
    mp4_files = list(Path(local_dir).glob("*.mp4"))
    if mp4_files:
        return str(local_dir)

    return None


def load_video_frames(video_path: str) -> np.ndarray:
    """Load video frames as numpy array."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return None

    return np.array(frames, dtype=np.uint8)


def parse_augmented_filename(filename: str) -> tuple:
    """
    Parse augmented video filename to get original demo info.

    Examples:
        demo_0_wrist_rgb_aug0.mp4 -> (demo_0, wrist_rgb, 0)
        demo_123_table_rgb_aug2.mp4 -> (demo_123, table_rgb, 2)
    """
    # Pattern: demo_X_camera_augY.mp4
    match = re.match(r"(demo_\d+)_(wrist_rgb|table_rgb)_aug(\d+)", filename)
    if match:
        demo_name = match.group(1)
        camera = match.group(2)
        aug_idx = int(match.group(3))
        return demo_name, camera, aug_idx
    return None, None, None


def get_augmented_videos_map(videos_dir: str) -> dict:
    """
    Build mapping of original demos to their augmented videos.

    Returns:
        {
            "demo_0": {
                "wrist_rgb": ["path/to/demo_0_wrist_rgb_aug0.mp4", ...],
                "table_rgb": ["path/to/demo_0_table_rgb_aug0.mp4", ...],
            },
            ...
        }
    """
    videos_dir = Path(videos_dir)
    augmented_map = {}

    for video_path in videos_dir.glob("*_aug*.mp4"):
        demo_name, camera, aug_idx = parse_augmented_filename(video_path.stem)
        if demo_name is None:
            continue

        if demo_name not in augmented_map:
            augmented_map[demo_name] = {}
        if camera not in augmented_map[demo_name]:
            augmented_map[demo_name][camera] = []

        augmented_map[demo_name][camera].append(str(video_path))

    # Sort augmented videos by aug index
    for demo_name in augmented_map:
        for camera in augmented_map[demo_name]:
            augmented_map[demo_name][camera].sort()

    return augmented_map


def main():
    args = parse_args()

    # Validate inputs
    if args.original_hdf5 is None and args.original_repo is None:
        print("Error: Must specify --original_hdf5 (local path) or --original_repo (HuggingFace)")
        return 1

    if args.augmented_repo is None and args.augmented_dir is None:
        print("Error: Must specify --augmented_repo or --augmented_dir")
        return 1

    # Get original HDF5
    if args.original_repo:
        download_dir = Path(args.output_hdf5).parent / "original_download"
        original_hdf5 = download_original_hdf5(args.original_repo, str(download_dir))
        if not original_hdf5:
            print(f"Error: Could not find HDF5 in {args.original_repo}")
            return 1
    else:
        original_hdf5 = args.original_hdf5
        if not os.path.exists(original_hdf5):
            print(f"Error: Original HDF5 not found: {original_hdf5}")
            return 1

    # Get augmented videos
    if args.augmented_repo:
        download_dir = Path(args.output_hdf5).parent / "augmented_download"
        videos_dir = download_augmented_videos(args.augmented_repo, str(download_dir))
    else:
        videos_dir = args.augmented_dir

    if not videos_dir or not os.path.exists(videos_dir):
        print(f"Error: Augmented videos directory not found: {videos_dir}")
        return 1

    print(f"Augmented videos directory: {videos_dir}")

    # Build augmented videos map
    augmented_map = get_augmented_videos_map(videos_dir)
    print(f"Found augmented videos for {len(augmented_map)} demos")

    # Load original HDF5 and create output
    print(f"Loading original HDF5: {original_hdf5}")

    output_dir = Path(args.output_hdf5).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(original_hdf5, "r") as f_in:
        with h5py.File(args.output_hdf5, "w") as f_out:
            # Copy attributes
            for attr_name, attr_value in f_in.attrs.items():
                f_out.attrs[attr_name] = attr_value

            # Create data group
            data_out = f_out.create_group("data")

            # Get episode names
            episode_names = list(f_in["data"].keys())
            if args.max_episodes:
                episode_names = episode_names[:args.max_episodes]

            total_episodes = 0

            for ep_name in tqdm(episode_names, desc="Processing episodes"):
                ep_in = f_in["data"][ep_name]

                # Include original demo if requested
                if args.include_originals:
                    ep_out = data_out.create_group(ep_name)

                    # Copy all data from original
                    for key in ep_in.keys():
                        if isinstance(ep_in[key], h5py.Group):
                            ep_in.copy(key, ep_out)
                        else:
                            ep_out.create_dataset(key, data=ep_in[key][:])

                    # Copy attributes
                    for attr_name, attr_value in ep_in.attrs.items():
                        ep_out.attrs[attr_name] = attr_value

                    total_episodes += 1

                # Create augmented episodes
                if ep_name in augmented_map:
                    for camera in args.cameras:
                        if camera not in augmented_map[ep_name]:
                            continue

                        for aug_video_path in augmented_map[ep_name][camera]:
                            # Parse aug index from filename
                            _, _, aug_idx = parse_augmented_filename(Path(aug_video_path).stem)
                            aug_ep_name = f"{ep_name}_aug{aug_idx}"

                            # Skip if already exists
                            if aug_ep_name in data_out:
                                continue

                            # Load augmented video frames
                            aug_frames = load_video_frames(aug_video_path)
                            if aug_frames is None:
                                print(f"Warning: Could not load {aug_video_path}")
                                continue

                            # Create augmented episode
                            ep_out = data_out.create_group(aug_ep_name)

                            # Copy non-image data from original
                            if "actions" in ep_in:
                                orig_actions = ep_in["actions"][:]
                                # Trim or pad actions to match video length
                                if len(orig_actions) > len(aug_frames):
                                    orig_actions = orig_actions[:len(aug_frames)]
                                ep_out.create_dataset("actions", data=orig_actions)

                            # Create obs group
                            if "obs" in ep_in:
                                obs_out = ep_out.create_group("obs")

                                for obs_key in ep_in["obs"].keys():
                                    orig_data = ep_in["obs"][obs_key][:]

                                    # Replace the augmented camera with new frames
                                    if obs_key == camera:
                                        # Use augmented frames
                                        obs_out.create_dataset(
                                            obs_key,
                                            data=aug_frames,
                                            compression="gzip",
                                            compression_opts=4,
                                        )
                                    else:
                                        # Keep original data (trim if needed)
                                        if len(orig_data) > len(aug_frames):
                                            orig_data = orig_data[:len(aug_frames)]
                                        obs_out.create_dataset(obs_key, data=orig_data)

                            # Copy attributes
                            for attr_name, attr_value in ep_in.attrs.items():
                                ep_out.attrs[attr_name] = attr_value
                            ep_out.attrs["augmented"] = True
                            ep_out.attrs["augmentation_source"] = aug_video_path

                            total_episodes += 1

            # Update total episodes attribute
            f_out.attrs["num_episodes"] = total_episodes
            data_out.attrs["num_episodes"] = total_episodes

    print(f"\nReconstruction complete!")
    print(f"Total episodes: {total_episodes}")
    print(f"Output: {args.output_hdf5}")

    # Print summary
    with h5py.File(args.output_hdf5, "r") as f:
        ep_names = list(f["data"].keys())
        original_count = len([n for n in ep_names if "_aug" not in n])
        augmented_count = len([n for n in ep_names if "_aug" in n])
        print(f"  Original episodes: {original_count}")
        print(f"  Augmented episodes: {augmented_count}")

    return 0


if __name__ == "__main__":
    exit(main())
