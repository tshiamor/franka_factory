#!/usr/bin/env bash
# =============================================================================
# Cosmos-Transfer2.5 Augmentation Setup for MCX Card Demos (Multi-GPU Parallel)
# =============================================================================
#
# Run this script on a cloud instance with 4-8x H100/A100 GPUs.
# It will:
#   1. Install dependencies and conda
#   2. Clone cosmos-transfer2.5
#   3. Pull the dataset from HuggingFace (tshiamor/mcx-card-demos-vla)
#   4. Build the Docker container
#   5. Run augmentation in parallel across GPUs (430 videos -> ~2150 videos)
#   6. Upload results to HuggingFace
#
# Usage:
#   export HF_TOKEN="your_huggingface_token"
#   bash brev_cosmos_augment.sh
#
# =============================================================================

set -euo pipefail

# ---- Configuration ----
HF_DATASET="tshiamor/mcx-card-demos-vla"
HF_OUTPUT_REPO="tshiamor/mcx-card-cosmos-augmented"
NUM_AUGMENTATIONS=1
NUM_GPUS=4
MAX_FRAMES=448
MAX_VIDEOS=100  # Set to 0 or empty to process all videos
WORK_DIR="${HOME}/cosmos-augmentation-mcx"
COSMOS_REPO="${WORK_DIR}/cosmos-transfer2.5"
DATASET_DIR="${WORK_DIR}/dataset"
OUTPUT_DIR="${WORK_DIR}/augmented_output"

echo "============================================="
echo "Cosmos-Transfer2.5 MCX Card Augmentation"
echo "============================================="
echo "Dataset: ${HF_DATASET}"
echo "Output:  ${HF_OUTPUT_REPO}"
echo "Augmentations per video: ${NUM_AUGMENTATIONS}"
echo "Max frames per video: ${MAX_FRAMES}"
echo "Max videos to process: ${MAX_VIDEOS:-all}"
echo "GPUs: ${NUM_GPUS}"
echo ""

# ---- Check prerequisites ----
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN environment variable not set."
    echo "  export HF_TOKEN='your_huggingface_token'"
    exit 1
fi

# Check GPU
if ! nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Are you on a GPU instance?"
    exit 1
fi

echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected ${GPU_COUNT} GPUs"
if [ "${GPU_COUNT}" -lt "${NUM_GPUS}" ]; then
    echo "WARNING: Found ${GPU_COUNT} GPUs but configured for ${NUM_GPUS}. Using ${GPU_COUNT} GPUs."
    NUM_GPUS="${GPU_COUNT}"
fi
echo ""

# ---- Step 1: Install system dependencies + conda ----
echo "[Step 1/7] Installing system dependencies..."
sudo apt-get update -qq || echo "  Warning: apt-get update had errors (non-critical, continuing)"
sudo apt-get install -y -qq git git-lfs curl ffmpeg wget > /dev/null 2>&1 || true
git lfs install

# Install Docker if not present
if ! command -v docker &>/dev/null; then
    echo "  Installing Docker..."
    sudo apt-get install -y -qq docker.io > /dev/null 2>&1 || true
fi

# Install Miniconda if no usable python/pip is available
if ! command -v conda &>/dev/null; then
    if [ -d "${HOME}/miniconda3" ]; then
        echo "  Miniconda directory exists, activating..."
    else
        echo "  Installing Miniconda..."
        CONDA_INSTALLER="/tmp/Miniconda3-latest-Linux-x86_64.sh"
        if [ ! -f "${CONDA_INSTALLER}" ]; then
            wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "${CONDA_INSTALLER}"
        fi
        bash "${CONDA_INSTALLER}" -b -p "${HOME}/miniconda3"
        echo "  Miniconda installed to ${HOME}/miniconda3"
    fi
fi

# Activate conda in current shell
if [ -f "${HOME}/miniconda3/bin/conda" ]; then
    eval "$(${HOME}/miniconda3/bin/conda shell.bash hook)"
    echo "  Conda version: $(conda --version)"
fi

# Ensure pip is available (use conda's pip or system pip)
if ! command -v pip &>/dev/null && ! command -v pip3 &>/dev/null; then
    echo "  Installing pip via conda..."
    conda install -y pip > /dev/null 2>&1
fi

# Use pip3 if pip is not available
if ! command -v pip &>/dev/null; then
    PIP_CMD="pip3"
else
    PIP_CMD="pip"
fi

# Use python3 if python is not available
if ! command -v python3 &>/dev/null; then
    PYTHON_CMD="python"
else
    PYTHON_CMD="python3"
fi

echo "  Python: $(${PYTHON_CMD} --version)"
echo "  Pip: $(${PIP_CMD} --version)"

# Ensure NVIDIA Container Toolkit is available
if ! sudo docker info 2>/dev/null | grep -q "nvidia"; then
    echo "  Installing NVIDIA Container Toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null
    curl -s -L "https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list" | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
    sudo apt-get update -qq || true
    sudo apt-get install -y -qq nvidia-container-toolkit > /dev/null 2>&1 || true
    sudo nvidia-ctk runtime configure --runtime=docker || true
    sudo systemctl restart docker || true
fi

# ---- Step 2: Clone Cosmos-Transfer2.5 ----
echo "[Step 2/7] Cloning Cosmos-Transfer2.5..."
mkdir -p "${WORK_DIR}"

if [ ! -d "${COSMOS_REPO}" ]; then
    git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git "${COSMOS_REPO}"
    cd "${COSMOS_REPO}"
    git lfs pull
else
    echo "  Already cloned."
fi

# ---- Step 3: Download dataset from HuggingFace ----
echo "[Step 3/7] Downloading dataset from HuggingFace..."
${PIP_CMD} install -q huggingface_hub

export DATASET_DIR HF_DATASET
${PYTHON_CMD} - <<'PYEOF'
import os
from huggingface_hub import snapshot_download

dataset_dir = os.environ.get("DATASET_DIR", os.path.expanduser("~/cosmos-augmentation-mcx/dataset"))
hf_dataset = os.environ.get("HF_DATASET", "tshiamor/mcx-card-demos-vla")

print(f"Downloading {hf_dataset} to {dataset_dir}...")
snapshot_download(
    repo_id=hf_dataset,
    repo_type="dataset",
    local_dir=dataset_dir,
    allow_patterns=["videos/*", "metadata.json"],
)
print(f"Dataset downloaded to {dataset_dir}")
PYEOF

echo "  Dataset contents:"
ls -la "${DATASET_DIR}/"
VIDEO_COUNT=$(ls "${DATASET_DIR}/videos/"*.mp4 2>/dev/null | wc -l || echo "0")
echo "  Videos: ${VIDEO_COUNT}"

# ---- Step 4: Build Docker container ----
echo "[Step 4/7] Building Docker container..."
cd "${COSMOS_REPO}"

# Detect GPU architecture
GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
echo "  GPU Compute Capability: ${GPU_ARCH}"

if [ "${GPU_ARCH}" -ge "120" ]; then
    echo "  Blackwell GPU detected, using nightly Dockerfile"
    DOCKERFILE="docker/nightly.Dockerfile"
else
    echo "  Ampere/Hopper GPU detected, using standard Dockerfile"
    DOCKERFILE="Dockerfile"
fi

sudo docker build -f "${DOCKERFILE}" -t cosmos-transfer2.5 . 2>&1 | tail -5

echo "  Docker image built."

# ---- Step 4b: Verify HuggingFace access to gated model ----
echo "  Verifying access to nvidia/Cosmos-Transfer2.5-2B..."
${PYTHON_CMD} - <<'CHECKEOF'
import os, sys
from huggingface_hub import HfApi
api = HfApi()
try:
    info = api.model_info("nvidia/Cosmos-Transfer2.5-2B", token=os.environ.get("HF_TOKEN"))
    print(f"  OK: Access granted to {info.id}")
except Exception as e:
    print(f"\n  ERROR: Cannot access nvidia/Cosmos-Transfer2.5-2B")
    print(f"  {e}")
    print(f"\n  You need to accept the license at:")
    print(f"    https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B")
    print(f"  Then re-run this script.")
    sys.exit(1)
CHECKEOF

if [ $? -ne 0 ]; then
    echo ""
    echo "ABORTING: HuggingFace model access denied."
    echo "Go to https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B and accept the license."
    exit 1
fi

# ---- Step 5: Create augmentation script inside workspace ----
echo "[Step 5/7] Preparing augmentation script..."

cat > "${WORK_DIR}/run_augmentation.py" << 'AUGEOF'
#!/usr/bin/env python3
"""
Augment MCX card insertion demos using Cosmos-Transfer2.5.
Runs inside the cosmos-transfer2.5 Docker container.

Supports GPU sharding: set SHARD_INDEX and NUM_SHARDS env vars
to split work across multiple containers.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Prompts tailored for MCX card / block manipulation task
AUGMENTATION_PROMPTS = [
    "A Franka Panda robot arm in a modern data center, precisely inserting a blue network card into a server slot. Cool blue ambient lighting with server racks visible in the background.",
    "A robotic manipulator in a clean electronics manufacturing facility, handling a small circuit board component. Bright overhead LED lighting, ESD-safe workstation visible.",
    "A precision robot arm in an industrial automation setting, placing electronic modules onto a platform. Warm tungsten lighting, machinery and conveyor belts around.",
    "A Franka robot gripper in a research laboratory, carefully positioning a blue rectangular component near network equipment. Natural daylight mixed with fluorescent lighting.",
    "A collaborative robot arm in a server room environment, manipulating hardware components. Green and blue LED indicators glowing, cable management systems visible.",
    "A robotic system in a futuristic tech workshop, handling precision electronics. Purple and cyan accent lighting, holographic displays in the background.",
    "A mechanical arm in a quality control station, inspecting and placing circuit boards. Bright white task lighting, measurement equipment nearby.",
    "A robot manipulator in an aerospace assembly facility, handling sensitive electronic modules. Clean room environment with filtered lighting.",
]

def create_spec_file(video_path, prompt, output_name, specs_dir, max_frames=150):
    spec = {
        "name": output_name,
        "prompt": prompt,
        "video_path": str(video_path),
        "guidance": 3,
        "seed": 2025,
        "num_input_frames": max_frames,
        "edge": {"control_weight": 0.9},
    }
    spec_path = Path(specs_dir) / f"{output_name}_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    return spec_path

def run_inference(spec_file, output_dir):
    cmd = [
        sys.executable,
        "/workspace/examples/inference.py",
        "-i", str(spec_file),
        "-o", str(output_dir),
        "--disable-guardrails",
    ]
    print(f"  CMD: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, timeout=1800)
    return result.returncode == 0

def main():
    input_dir = Path(os.environ.get("INPUT_DIR", "/data/input"))
    output_dir = Path(os.environ.get("OUTPUT_DIR", "/data/output"))
    num_aug = int(os.environ.get("NUM_AUGMENTATIONS", "1"))
    max_frames = int(os.environ.get("MAX_FRAMES", "448"))
    max_videos = int(os.environ.get("MAX_VIDEOS", "0"))  # 0 = all videos
    shard_index = int(os.environ.get("SHARD_INDEX", "0"))
    num_shards = int(os.environ.get("NUM_SHARDS", "1"))

    videos_dir = input_dir / "videos"

    # Output structure
    videos_out = output_dir / "videos"
    specs_dir = output_dir / "specs"
    for d in [videos_out, specs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Copy metadata (only shard 0)
    if shard_index == 0:
        metadata_src = input_dir / "metadata.json"
        if metadata_src.exists():
            shutil.copy(metadata_src, output_dir / "metadata.json")

    # Get all videos, limit if MAX_VIDEOS is set, then select this shard's subset
    all_videos = sorted(videos_dir.glob("*.mp4"))
    if max_videos > 0:
        all_videos = all_videos[:max_videos]
        print(f"Limiting to first {max_videos} videos")
    shard_videos = [v for i, v in enumerate(all_videos) if i % num_shards == shard_index]

    print(f"Shard {shard_index}/{num_shards}: processing {len(shard_videos)} of {len(all_videos)} videos")
    print(f"Augmentations per video: {num_aug}, max frames: {max_frames}")

    total = 0
    failed = 0

    for i, video_path in enumerate(shard_videos):
        video_name = video_path.stem  # e.g., demo_0_wrist_rgb

        print(f"\n[{i+1}/{len(shard_videos)}] {video_name}", flush=True)

        # Copy original video
        shutil.copy(video_path, videos_out / f"{video_name}.mp4")
        total += 1

        # Generate augmentations
        for aug_idx in range(num_aug):
            prompt = AUGMENTATION_PROMPTS[aug_idx % len(AUGMENTATION_PROMPTS)]
            aug_name = f"{video_name}_aug{aug_idx}"
            print(f"  [{aug_idx+1}/{num_aug}] {aug_name}", flush=True)

            spec_file = create_spec_file(video_path, prompt, aug_name, specs_dir, max_frames)
            temp_out = output_dir / f"temp_inference_shard{shard_index}"
            temp_out.mkdir(exist_ok=True)

            try:
                if run_inference(spec_file, temp_out):
                    # Search for generated video in multiple possible locations
                    generated = None
                    candidates = [
                        temp_out / f"{aug_name}.mp4",
                        temp_out / aug_name / f"{aug_name}.mp4",
                        temp_out / f"{aug_name}" / "output.mp4",
                    ]
                    # Also search recursively for any .mp4
                    for mp4 in temp_out.rglob("*.mp4"):
                        candidates.append(mp4)

                    for candidate in candidates:
                        if candidate.exists():
                            generated = candidate
                            break

                    if generated:
                        shutil.move(str(generated), str(videos_out / f"{aug_name}.mp4"))
                        total += 1
                        print(f"    OK: {aug_name}.mp4", flush=True)
                    else:
                        print(f"    WARN: output video not found in {temp_out}", flush=True)
                        for f in sorted(temp_out.rglob("*")):
                            print(f"      found: {f}", flush=True)
                        failed += 1
                else:
                    print(f"    FAIL: inference error", flush=True)
                    failed += 1
            except Exception as e:
                print(f"    ERROR: {e}", flush=True)
                failed += 1

            shutil.rmtree(temp_out, ignore_errors=True)

    print(f"\n{'='*50}")
    print(f"Shard {shard_index} done! Total: {total} videos ({len(shard_videos)} original + {total - len(shard_videos)} augmented)")
    print(f"Failed: {failed}")
    print(f"Output: {output_dir}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
AUGEOF

# ---- Step 6: Run augmentation (parallel across GPUs) ----
echo "[Step 6/7] Running augmentation across ${NUM_GPUS} GPUs in parallel..."
VIDEO_COUNT=$(ls "${DATASET_DIR}/videos/"*.mp4 2>/dev/null | wc -l || echo "430")
echo "  Each GPU processes ~$((VIDEO_COUNT / NUM_GPUS)) videos with ${NUM_AUGMENTATIONS} augmentation(s) each."
echo "  Videos trimmed to ${MAX_FRAMES} frames."
echo ""

# Pre-download model weights with a single container first
echo "  Pre-downloading model weights (shared cache)..."
sudo docker run --rm \
    --runtime=nvidia \
    --gpus '"device=0"' \
    --ipc=host \
    -v "${COSMOS_REPO}":/workspace \
    -v "${HOME}/.cache":/root/.cache \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" \
    -e HF_HOME="/root/.cache/huggingface" \
    cosmos-transfer2.5 \
    bash -c "huggingface-cli login --token \${HF_TOKEN} 2>/dev/null || true; python3 -c \"
from huggingface_hub import snapshot_download
snapshot_download('nvidia/Cosmos-Transfer2.5-2B', local_dir='/root/.cache/huggingface/hub/models--nvidia--Cosmos-Transfer2.5-2B')
print('Model weights downloaded.')
\"" || echo "  Note: Pre-download may have completed already (cached)."

echo "  Launching ${NUM_GPUS} parallel containers..."

PIDS=()
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    SHARD_OUTPUT="${OUTPUT_DIR}/shard_${GPU_ID}"
    mkdir -p "${SHARD_OUTPUT}"

    echo "  Starting shard ${GPU_ID} on GPU ${GPU_ID}..."

    sudo docker run --rm \
        --runtime=nvidia \
        --gpus "\"device=${GPU_ID}\"" \
        --ipc=host \
        -v "${COSMOS_REPO}":/workspace \
        -v "${DATASET_DIR}":/data/input:ro \
        -v "${SHARD_OUTPUT}":/data/output \
        -v "${WORK_DIR}/run_augmentation.py":/data/run_augmentation.py:ro \
        -v "${HOME}/.cache":/root/.cache \
        -e HF_TOKEN="${HF_TOKEN}" \
        -e HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" \
        -e HF_HOME="/root/.cache/huggingface" \
        -e INPUT_DIR=/data/input \
        -e OUTPUT_DIR=/data/output \
        -e NUM_AUGMENTATIONS="${NUM_AUGMENTATIONS}" \
        -e MAX_FRAMES="${MAX_FRAMES}" \
        -e MAX_VIDEOS="${MAX_VIDEOS}" \
        -e SHARD_INDEX="${GPU_ID}" \
        -e NUM_SHARDS="${NUM_GPUS}" \
        cosmos-transfer2.5 \
        bash -c "huggingface-cli login --token \${HF_TOKEN} 2>/dev/null || true; python3 /data/run_augmentation.py" \
        > "${WORK_DIR}/shard_${GPU_ID}.log" 2>&1 &

    PIDS+=($!)
done

echo "  All ${NUM_GPUS} shards launched. PIDs: ${PIDS[*]}"
echo "  Monitor progress:"
echo "    tail -f ${WORK_DIR}/shard_*.log"
echo "    nvidia-smi"
echo ""

# Wait for all shards to complete
echo "  Waiting for all shards to complete..."
FAILED_SHARDS=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "  Shard ${i} completed successfully."
    else
        echo "  WARNING: Shard ${i} failed (exit code $?)."
        FAILED_SHARDS=$((FAILED_SHARDS + 1))
    fi
done

if [ "${FAILED_SHARDS}" -gt 0 ]; then
    echo "  WARNING: ${FAILED_SHARDS} shard(s) failed. Check logs in ${WORK_DIR}/shard_*.log"
fi

# Merge shard outputs into final output directory
echo "  Merging shard outputs..."
FINAL_OUTPUT="${OUTPUT_DIR}/merged"
mkdir -p "${FINAL_OUTPUT}/videos"

for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    SHARD_DIR="${OUTPUT_DIR}/shard_${GPU_ID}"
    if [ -d "${SHARD_DIR}/videos" ]; then
        cp -n "${SHARD_DIR}/videos/"*.mp4 "${FINAL_OUTPUT}/videos/" 2>/dev/null || true
    fi
    # Copy metadata if present
    if [ -f "${SHARD_DIR}/metadata.json" ]; then
        cp -n "${SHARD_DIR}/metadata.json" "${FINAL_OUTPUT}/" 2>/dev/null || true
    fi
done

TOTAL_VIDEOS=$(ls "${FINAL_OUTPUT}/videos/"*.mp4 2>/dev/null | wc -l)
echo "  Augmentation complete."
echo "  Total output videos: ${TOTAL_VIDEOS}"
echo "  Expected: $((VIDEO_COUNT * (NUM_AUGMENTATIONS + 1))) (originals + augmented)"

# ---- Step 7: Upload to HuggingFace ----
echo "[Step 7/7] Uploading results to HuggingFace..."

export OUTPUT_DIR="${FINAL_OUTPUT}" HF_OUTPUT_REPO
${PYTHON_CMD} - <<'UPLOADEOF'
import os
from huggingface_hub import HfApi

output_dir = os.environ.get("OUTPUT_DIR", os.path.expanduser("~/cosmos-augmentation-mcx/augmented_output/merged"))
repo_id = os.environ.get("HF_OUTPUT_REPO", "tshiamor/mcx-card-cosmos-augmented")

api = HfApi()

# Create dataset repo if it doesn't exist
try:
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
except Exception as e:
    print(f"Note: {e}")

print(f"Uploading {output_dir} to {repo_id}...")
api.upload_folder(
    folder_path=output_dir,
    repo_id=repo_id,
    repo_type="dataset",
    commit_message="Upload cosmos-augmented MCX card demos (430 original + augmented)",
)
print(f"Upload complete: https://huggingface.co/datasets/{repo_id}")
UPLOADEOF

echo ""
echo "============================================="
echo "PIPELINE COMPLETE"
echo "============================================="
echo "Input:  https://huggingface.co/datasets/${HF_DATASET}"
echo "Output: https://huggingface.co/datasets/${HF_OUTPUT_REPO}"
echo ""
echo "Summary:"
echo "  - Max videos limit: ${MAX_VIDEOS:-all}"
echo "  - Videos processed: ${VIDEO_COUNT}"
echo "  - Augmentations per video: ${NUM_AUGMENTATIONS}"
echo "  - Max frames: ${MAX_FRAMES}"
echo "  - Total output: ${TOTAL_VIDEOS} videos"
echo "============================================="
