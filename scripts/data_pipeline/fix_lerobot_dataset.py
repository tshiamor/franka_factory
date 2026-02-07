#!/usr/bin/env python3
"""
Fix LeRobot dataset format for compatibility with LeRobot v2.0.

1. Renames meta_data/ to meta/
2. Adds codebase_version to info.json
3. Tags the HuggingFace repo with the version
"""

import argparse
import json
import os
import shutil
from pathlib import Path


LEROBOT_VERSION = "2.1"  # Current LeRobot codebase version


def fix_local_dataset(dataset_dir: str):
    """Fix local dataset structure."""
    dataset_path = Path(dataset_dir)

    # Rename meta_data to meta
    meta_data_dir = dataset_path / "meta_data"
    meta_dir = dataset_path / "meta"

    if meta_data_dir.exists() and not meta_dir.exists():
        print(f"Renaming {meta_data_dir} to {meta_dir}")
        shutil.move(str(meta_data_dir), str(meta_dir))
    elif not meta_dir.exists():
        print(f"Creating {meta_dir}")
        meta_dir.mkdir(parents=True, exist_ok=True)

    # Update info.json
    info_path = meta_dir / "info.json"

    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
    else:
        info = {}

    # Add required fields
    info["codebase_version"] = LEROBOT_VERSION
    info.setdefault("fps", 30)
    info.setdefault("robot_type", "franka_panda")
    info.setdefault("total_episodes", 0)
    info.setdefault("total_frames", 0)
    info.setdefault("total_tasks", 1)

    # Count episodes if possible
    data_dir = dataset_path / "data"
    videos_dir = dataset_path / "videos"

    if videos_dir.exists():
        video_files = list(videos_dir.glob("*.mp4"))
        info["total_episodes"] = len(video_files)

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"Updated {info_path}")
    print(f"  codebase_version: {info['codebase_version']}")

    return info


def download_fix_upload(repo_id: str, output_dir: str = None):
    """Download dataset, fix it, and re-upload."""
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()

    # Download
    local_dir = output_dir or f"/tmp/fix_lerobot_{repo_id.replace('/', '_')}"
    print(f"Downloading {repo_id} to {local_dir}...")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
    )

    # Fix
    info = fix_local_dataset(local_dir)
    version = info["codebase_version"]

    # Upload fixed files
    print(f"Uploading fixed dataset to {repo_id}...")

    # Upload the meta directory
    meta_dir = Path(local_dir) / "meta"
    if meta_dir.exists():
        api.upload_folder(
            folder_path=str(meta_dir),
            path_in_repo="meta",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Fix LeRobot format: add meta/info.json with codebase_version={version}",
        )

    # Delete old meta_data if it exists on hub
    try:
        api.delete_folder(
            path_in_repo="meta_data",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Remove old meta_data directory",
        )
        print("Deleted old meta_data/ directory from hub")
    except Exception as e:
        print(f"Note: Could not delete meta_data/ (may not exist): {e}")

    # Tag the repo
    print(f"Tagging repo with version {version}...")
    try:
        api.create_tag(
            repo_id=repo_id,
            tag=f"v{version}",
            repo_type="dataset",
        )
        print(f"Created tag: v{version}")
    except Exception as e:
        print(f"Note: Could not create tag (may already exist): {e}")

    print(f"\nDataset fixed and uploaded: https://huggingface.co/datasets/{repo_id}")
    return local_dir


def main():
    parser = argparse.ArgumentParser(description="Fix LeRobot dataset format")
    parser.add_argument("--local", type=str, help="Path to local dataset directory")
    parser.add_argument("--repo_id", type=str, help="HuggingFace dataset repo ID")
    parser.add_argument("--output_dir", type=str, help="Output directory for downloaded dataset")
    args = parser.parse_args()

    if args.local:
        fix_local_dataset(args.local)
    elif args.repo_id:
        download_fix_upload(args.repo_id, args.output_dir)
    else:
        print("Please specify --local or --repo_id")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
