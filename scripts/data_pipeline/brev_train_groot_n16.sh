#!/usr/bin/env bash
# =============================================================================
# GR00T N1.6 Local Fine-tuning Pipeline (Isaac-GR00T)
# =============================================================================
#
# End-to-end pipeline: data conversion → fine-tuning → inference evaluation
# for NVIDIA GR00T N1.6-3B on the Franka MCX Card Block Insertion task.
#
# Uses Isaac-GR00T's native fine-tuning (NOT LeRobot) with memory
# optimizations for consumer GPUs (RTX 5090 32GB).
#
# Prerequisites:
#   - Isaac Lab installed with isaaclab conda env
#   - Isaac-GR00T cloned at ~/Isaac-GR00T (pip install -e . done)
#   - HDF5 training data from Isaac Lab teleop
#   - transformers==4.51.3 (required for fine-tuning)
#
# Usage:
#   bash brev_train_groot_n16.sh
#
# Steps:
#   1. Verify environment and dependencies
#   2. Convert HDF5 demos to LeRobot v2 format
#   3. Generate dataset statistics
#   4. Fine-tune GR00T N1.6 with memory optimizations
#   5. Post-process model for inference
#   6. (Optional) Run inference evaluation in Isaac Lab
#
# =============================================================================

set -eo pipefail

# ---- Configuration ----
ISAAC_GROOT_DIR="${ISAAC_GROOT_DIR:-${HOME}/Isaac-GR00T}"
CONDA_ENV="${CONDA_ENV:-isaaclab}"
CONDA_DIR="${CONDA_DIR:-${HOME}/miniforge3}"
ISAACLAB_DIR="${ISAACLAB_DIR:-${HOME}/IsaacLab}"

# Paths - edit these for your setup
HDF5_PATH="${HDF5_PATH:-${HOME}/IsaacLab/mcx_card_training_augmented.hdf5}"
GROOT_DATA_DIR="${GROOT_DATA_DIR:-${HOME}/groot_data}"
DATASET_PATH="${GROOT_DATA_DIR}/mcx_card_lerobot_v2"
OUTPUT_DIR="${GROOT_DATA_DIR}/finetune_output_n16"

# Model
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.6-3B}"

# Script paths (relative to franka_factory repo)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRANKA_FACTORY_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONVERT_SCRIPT="${SCRIPT_DIR}/convert_hdf5_to_lerobot_v2.py"
MODALITY_CONFIG="${FRANKA_FACTORY_DIR}/scripts/groot_finetune/franka_mcx_config.py"
LAUNCH_FINETUNE="${FRANKA_FACTORY_DIR}/scripts/groot_finetune/launch_finetune.py"
EVAL_SCRIPT="${FRANKA_FACTORY_DIR}/scripts/eval/eval_vla_policy.py"

# Training hyperparameters
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# Inference evaluation
EVAL_EPISODES="${EVAL_EPISODES:-10}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-2400}"

echo "============================================="
echo "GR00T N1.6-3B Local Fine-tuning Pipeline"
echo "============================================="
echo "Isaac-GR00T:    ${ISAAC_GROOT_DIR}"
echo "HDF5 data:      ${HDF5_PATH}"
echo "Dataset output: ${DATASET_PATH}"
echo "Model output:   ${OUTPUT_DIR}"
echo "Base model:     ${BASE_MODEL}"
echo "Batch size:     ${BATCH_SIZE} (grad_accum=${GRAD_ACCUM}, effective=${BATCH_SIZE}*${GRAD_ACCUM}=$((BATCH_SIZE * GRAD_ACCUM)))"
echo "Max steps:      ${MAX_STEPS}"
echo ""

# ---- Step 1: Verify environment ----
echo "[Step 1/6] Verifying environment and dependencies..."

# Check GPU
if ! nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Are you on a GPU machine?"
    exit 1
fi

echo "  GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Activate conda
eval "$(${CONDA_DIR}/bin/conda shell.bash hook)"
conda activate ${CONDA_ENV}
export PATH="${CONDA_DIR}/envs/${CONDA_ENV}/bin:${PATH}"
export CONDA_PREFIX="${CONDA_DIR}/envs/${CONDA_ENV}"

# Check Isaac-GR00T
if [ ! -d "${ISAAC_GROOT_DIR}" ]; then
    echo "ERROR: Isaac-GR00T not found at ${ISAAC_GROOT_DIR}"
    echo "  Clone it: git clone https://github.com/NVIDIA/Isaac-GR00T.git ${ISAAC_GROOT_DIR}"
    echo "  Install:  cd ${ISAAC_GROOT_DIR} && pip install -e ."
    exit 1
fi

# Check HDF5 data
if [ ! -f "${HDF5_PATH}" ]; then
    echo "ERROR: HDF5 training data not found: ${HDF5_PATH}"
    echo "  Set HDF5_PATH=/path/to/your/demos.hdf5"
    exit 1
fi

# Check transformers version
TRANSFORMERS_VER=$(python -c "import transformers; print(transformers.__version__)" 2>/dev/null)
if [[ "${TRANSFORMERS_VER}" != 4.51.* ]]; then
    echo "  WARNING: transformers==${TRANSFORMERS_VER} detected."
    echo "  Fine-tuning requires transformers==4.51.3 (Eagle3_VLConfig compatibility)."
    echo "  Installing correct version..."
    pip install transformers==4.51.3 -q
fi

# Install other deps if missing
python -c "import lmdb" 2>/dev/null || pip install lmdb -q
python -c "import scipy" 2>/dev/null || pip install scipy -q
python -c "import bitsandbytes" 2>/dev/null || pip install bitsandbytes -q

echo "  Python:       $(python --version 2>&1)"
echo "  PyTorch:      $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA:         $(python -c 'import torch; print(torch.version.cuda)')"
echo "  transformers: $(python -c 'import transformers; print(transformers.__version__)')"
echo "  Isaac-GR00T:  ${ISAAC_GROOT_DIR}"
echo ""

# ---- Step 2: Convert HDF5 to LeRobot v2 ----
echo "[Step 2/6] Converting HDF5 demos to LeRobot v2 format..."

if [ -d "${DATASET_PATH}/data" ] && [ -d "${DATASET_PATH}/videos" ]; then
    echo "  Dataset already exists at ${DATASET_PATH}, skipping conversion."
    echo "  (Delete the directory to re-convert)"
else
    python "${CONVERT_SCRIPT}" \
        --hdf5 "${HDF5_PATH}" \
        --output "${DATASET_PATH}" \
        --fps 30
fi
echo ""

# ---- Step 3: Generate dataset statistics ----
echo "[Step 3/6] Generating dataset statistics..."

if [ -f "${DATASET_PATH}/meta/stats.json" ]; then
    echo "  Statistics already exist, skipping."
else
    cd "${ISAAC_GROOT_DIR}"
    python - <<'STATSEOF'
import sys, importlib
from pathlib import Path

# Register our modality config
config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("franka_mcx_config.py")
sys.path.append(str(config_path.parent))
importlib.import_module(config_path.stem)

from gr00t.data.dataset.lerobot_dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag

ds = LeRobotSingleDataset(
    dataset_path=sys.argv[2] if len(sys.argv) > 2 else "",
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
ds.compute_stats()
print("  Dataset statistics saved to meta/stats.json")
STATSEOF
fi
echo ""

# ---- Step 4: Fine-tune GR00T N1.6 ----
echo "[Step 4/6] Fine-tuning GR00T N1.6..."
echo "  This will take ~3-4 hours on RTX 5090 for ${MAX_STEPS} steps."
echo ""

# Memory optimization for consumer GPUs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${ISAAC_GROOT_DIR}"

python "${LAUNCH_FINETUNE}" \
    --base-model-path "${BASE_MODEL}" \
    --dataset-path "${DATASET_PATH}" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "${MODALITY_CONFIG}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-gpus 1 \
    --global-batch-size ${BATCH_SIZE} \
    --gradient-accumulation-steps ${GRAD_ACCUM} \
    --learning-rate ${LEARNING_RATE} \
    --max-steps ${MAX_STEPS} \
    --save-steps ${SAVE_STEPS} \
    --save-total-limit ${SAVE_TOTAL_LIMIT} \
    --dataloader-num-workers ${NUM_WORKERS} \
    --num-shards-per-epoch 5000 \
    --shard-size 512 \
    --episode-sampling-rate 0.3 \
    --no-tune-llm \
    --no-tune-visual \
    --tune-projector \
    --tune-diffusion-model \
    --warmup-ratio 0.05 \
    --weight-decay 1e-5 \
    2>&1 | tee "${GROOT_DATA_DIR}/training_n16.log"

echo ""
echo "  Training complete! Logs: ${GROOT_DATA_DIR}/training_n16.log"
echo ""

# ---- Step 5: Post-process model for inference ----
echo "[Step 5/6] Post-processing model for inference..."

# Copy processor files from processor/ subdirectory to model root
# (AutoProcessor.from_pretrained looks in the root, but fine-tuning saves them in processor/)
if [ -d "${OUTPUT_DIR}/processor" ]; then
    for f in processor_config.json embodiment_id.json statistics.json; do
        if [ -f "${OUTPUT_DIR}/processor/${f}" ] && [ ! -f "${OUTPUT_DIR}/${f}" ]; then
            cp "${OUTPUT_DIR}/processor/${f}" "${OUTPUT_DIR}/${f}"
            echo "  Copied processor/${f} to model root"
        fi
    done
fi

echo "  Model ready at: ${OUTPUT_DIR}"
echo ""

# ---- Step 6: (Optional) Run inference evaluation ----
echo "[Step 6/6] Inference evaluation"
echo ""
echo "To evaluate the fine-tuned model in Isaac Lab, run:"
echo ""
echo "  CONDA_PREFIX=${CONDA_DIR}/envs/${CONDA_ENV} \\"
echo "    ${ISAACLAB_DIR}/isaaclab.sh -p \\"
echo "    ${EVAL_SCRIPT} \\"
echo "    --task Franka-Factory-MCXCardBlockInsert-Mimic-v0 \\"
echo "    --policy groot_n16 \\"
echo "    --model ${OUTPUT_DIR} \\"
echo "    --enable_cameras \\"
echo "    --headless \\"
echo "    --num_envs 1 \\"
echo "    --episodes ${EVAL_EPISODES} \\"
echo "    --max_steps ${EVAL_MAX_STEPS}"
echo ""

# Auto-run eval if --eval flag is detected
if [[ " $* " == *" --eval "* ]]; then
    echo "Running evaluation..."
    # Upgrade transformers for inference (N1.6 inference works with 4.51.3 too)
    CONDA_PREFIX="${CONDA_DIR}/envs/${CONDA_ENV}" \
        "${ISAACLAB_DIR}/isaaclab.sh" -p \
        "${EVAL_SCRIPT}" \
        --task Franka-Factory-MCXCardBlockInsert-Mimic-v0 \
        --policy groot_n16 \
        --model "${OUTPUT_DIR}" \
        --enable_cameras \
        --headless \
        --num_envs 1 \
        --episodes ${EVAL_EPISODES} \
        --max_steps ${EVAL_MAX_STEPS}
fi

echo "============================================="
echo "PIPELINE COMPLETE"
echo "============================================="
echo "Base model:      ${BASE_MODEL} (3B parameters)"
echo "Training data:   ${HDF5_PATH}"
echo "LeRobot dataset: ${DATASET_PATH}"
echo "Fine-tuned model: ${OUTPUT_DIR}"
echo ""
echo "Memory optimizations applied:"
echo "  - 8-bit AdamW optimizer (bitsandbytes)"
echo "  - Gradient checkpointing"
echo "  - Batch=${BATCH_SIZE}, GradAccum=${GRAD_ACCUM}"
echo "  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "============================================="
