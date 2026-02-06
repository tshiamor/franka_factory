#!/usr/bin/env python
# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Cosmos World Foundation Model augmentation script for NVIDIA Brev.

This script downloads demonstration videos from Hugging Face and uses
NVIDIA Cosmos Transfer 2.5 to generate augmented demonstrations.

Run this on NVIDIA Brev with multiple GPUs for best performance.

Usage:
    # On NVIDIA Brev instance
    python cosmos_augmentation_brev.py \
        --repo_id your-username/mcx-card-demos \
        --output_dir ./cosmos_augmented \
        --num_augmentations 5 \
        --gpus 0,1,2,3

Requirements:
    - NVIDIA Cosmos Transfer 2.5
    - Multiple A100/H100 GPUs (recommended)
    - pip install huggingface_hub torch
"""

import argparse
import json
import os
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Cosmos augmentation on Brev")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repo ID")
    parser.add_argument("--local_dir", type=str, default="./cosmos_input", help="Local download directory")
    parser.add_argument("--output_dir", type=str, default="./cosmos_output", help="Output directory")
    parser.add_argument("--num_augmentations", type=int, default=5, help="Augmentations per video")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU IDs")
    parser.add_argument("--cosmos_model", type=str, default="cosmos-transfer-2.5", help="Cosmos model name")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--skip_download", action="store_true", help="Skip download step")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    return parser.parse_args()


def download_dataset(repo_id: str, local_dir: str):
    """Download dataset from Hugging Face."""
    from huggingface_hub import snapshot_download

    print(f"Downloading dataset from {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=["videos/*", "metadata.json"],
    )
    print(f"Downloaded to {local_dir}")


def load_metadata(local_dir: str) -> dict:
    """Load metadata from downloaded dataset."""
    metadata_path = Path(local_dir) / "metadata.json"
    with open(metadata_path, "r") as f:
        return json.load(f)


def run_cosmos_augmentation(
    input_video: str,
    output_dir: str,
    num_augmentations: int,
    gpu_id: int,
    cosmos_model: str,
    dry_run: bool = False,
) -> list:
    """Run Cosmos augmentation on a single video."""

    output_videos = []
    video_name = Path(input_video).stem

    for aug_idx in range(num_augmentations):
        output_path = Path(output_dir) / f"{video_name}_aug{aug_idx}.mp4"

        # Cosmos Transfer command (adjust based on actual Cosmos CLI)
        cmd = [
            "python", "-m", "cosmos.transfer",
            "--model", cosmos_model,
            "--input", input_video,
            "--output", str(output_path),
            "--device", f"cuda:{gpu_id}",
            "--seed", str(aug_idx * 1000),  # Different seed for each augmentation
            # Add Cosmos-specific parameters here
            "--style_transfer", "lighting_variation",
            "--strength", "0.3",
        ]

        if dry_run:
            print(f"[GPU {gpu_id}] Would run: {' '.join(cmd)}")
        else:
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                output_videos.append(str(output_path))
            except subprocess.CalledProcessError as e:
                print(f"Error processing {input_video}: {e}")

    return output_videos


def parallel_augmentation(
    video_paths: list,
    output_dir: str,
    num_augmentations: int,
    gpu_ids: list,
    cosmos_model: str,
    dry_run: bool = False,
):
    """Run augmentation in parallel across GPUs."""

    os.makedirs(output_dir, exist_ok=True)

    # Distribute videos across GPUs
    results = []

    with ProcessPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = []

        for i, video_path in enumerate(video_paths):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            future = executor.submit(
                run_cosmos_augmentation,
                video_path,
                output_dir,
                num_augmentations,
                gpu_id,
                cosmos_model,
                dry_run,
            )
            futures.append((video_path, future))

        for video_path, future in tqdm(futures, desc="Augmenting videos"):
            try:
                output_videos = future.result()
                results.extend(output_videos)
            except Exception as e:
                print(f"Error with {video_path}: {e}")

    return results


def create_augmented_metadata(
    original_metadata: dict,
    augmented_videos: list,
    output_dir: str,
):
    """Create metadata for augmented dataset."""

    augmented_metadata = {
        "source_dataset": original_metadata.get("env_name", "unknown"),
        "num_original_episodes": original_metadata.get("num_episodes", 0),
        "num_augmented_videos": len(augmented_videos),
        "augmentation_method": "cosmos_transfer_2.5",
        "videos": augmented_videos,
    }

    metadata_path = Path(output_dir) / "augmented_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(augmented_metadata, f, indent=2)

    print(f"Saved augmented metadata to {metadata_path}")


def main():
    args = parse_args()

    # Parse GPU IDs
    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    print(f"Using GPUs: {gpu_ids}")

    # Download dataset
    if not args.skip_download:
        download_dataset(args.repo_id, args.local_dir)

    # Load metadata
    metadata = load_metadata(args.local_dir)
    print(f"Loaded metadata: {metadata.get('num_episodes', 0)} episodes")

    # Get video paths
    videos_dir = Path(args.local_dir) / "videos"
    video_paths = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.webm"))
    print(f"Found {len(video_paths)} videos to augment")

    if len(video_paths) == 0:
        print("No videos found. Exiting.")
        return 1

    # Calculate expected output
    expected_augmented = len(video_paths) * args.num_augmentations
    print(f"Will generate {expected_augmented} augmented videos")

    # Run augmentation
    print(f"\nStarting Cosmos augmentation...")
    augmented_videos = parallel_augmentation(
        video_paths=[str(p) for p in video_paths],
        output_dir=args.output_dir,
        num_augmentations=args.num_augmentations,
        gpu_ids=gpu_ids,
        cosmos_model=args.cosmos_model,
        dry_run=args.dry_run,
    )

    # Create metadata
    if not args.dry_run:
        create_augmented_metadata(metadata, augmented_videos, args.output_dir)

    print(f"\nAugmentation complete!")
    print(f"Generated {len(augmented_videos)} augmented videos")
    print(f"Output directory: {args.output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
