#!/usr/bin/env bash
# =============================================================================
# Run all VLA model evaluations on MCX Card Block Insertion task
# =============================================================================
#
# Runs Pi-Zero, GR00T N1.5, GR00T N1.6, and OpenVLA in sequence,
# handling transformers version switches and GPU cleanup between runs.
#
# Usage:
#   bash scripts/eval/run_all_evals.sh
#   bash scripts/eval/run_all_evals.sh --episodes 5 --max_steps 1200
#   bash scripts/eval/run_all_evals.sh --gui
#
# =============================================================================

set -eo pipefail

# ---- Configuration ----
CONDA_DIR="${CONDA_DIR:-${HOME}/miniforge3}"
CONDA_ENV="${CONDA_ENV:-isaaclab}"
ISAACLAB_DIR="${ISAACLAB_DIR:-${HOME}/IsaacLab}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_vla_policy.py"

TASK="Franka-Factory-MCXCardBlockInsert-Mimic-v0"
EPISODES="${EPISODES:-10}"
MAX_STEPS="${MAX_STEPS:-2400}"
NUM_ENVS=1
HEADLESS="--headless"

# Models
PIZERO_MODEL="tshiamor/pizero-mcx-card"
GROOT_N15_MODEL="tshiamor/groot-n15-mcx-card"
GROOT_N16_MODEL="${GROOT_N16_MODEL:-${HOME}/groot_data/finetune_output_n16}"
OPENVLA_MODEL="tshiamor/openvla-mcx-card"

# Transformers versions required per model
TRANSFORMERS_PIZERO="4.51.3"
TRANSFORMERS_GROOT_N15="4.57.1"
TRANSFORMERS_GROOT_N16="4.51.3"
TRANSFORMERS_OPENVLA="4.45.0"

# Parse CLI args
for arg in "$@"; do
    case $arg in
        --episodes=*) EPISODES="${arg#*=}" ;;
        --episodes) shift_next=episodes ;;
        --max_steps=*) MAX_STEPS="${arg#*=}" ;;
        --max_steps) shift_next=max_steps ;;
        --gui) HEADLESS="" ;;
        *)
            if [ "$shift_next" = "episodes" ]; then EPISODES="$arg"; shift_next=""; fi
            if [ "$shift_next" = "max_steps" ]; then MAX_STEPS="$arg"; shift_next=""; fi
            ;;
    esac
done

# Output log directory
LOG_DIR="${SCRIPT_DIR}/eval_logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="${LOG_DIR}/summary_${TIMESTAMP}.txt"

PIP="${CONDA_DIR}/envs/${CONDA_ENV}/bin/pip"
PYTHON="${CONDA_DIR}/envs/${CONDA_ENV}/bin/python"

echo "============================================="
echo "VLA Model Evaluation Batch Runner"
echo "============================================="
echo "Task:       ${TASK}"
echo "Episodes:   ${EPISODES}"
echo "Max steps:  ${MAX_STEPS}"
echo "GUI:        $([ -z "$HEADLESS" ] && echo 'yes' || echo 'no')"
echo "Log dir:    ${LOG_DIR}"
echo "============================================="
echo ""

# ---- Helper functions ----

cleanup_gpu() {
    echo "  Cleaning up GPU processes..."
    nvidia-smi --query-compute-apps=pid,name --format=csv,noheader 2>/dev/null | \
        grep python | awk -F',' '{print $1}' | xargs -r kill -9 2>/dev/null || true
    sleep 3
}

set_transformers() {
    local version="$1"
    local current
    current=$("${PYTHON}" -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "unknown")
    if [ "$current" != "$version" ]; then
        echo "  Switching transformers: ${current} -> ${version}"
        "${PIP}" install "transformers==${version}" -q 2>&1 | tail -1
    else
        echo "  transformers==${version} already installed"
    fi
}

run_eval() {
    local policy="$1"
    local model="$2"
    local log_file="$3"

    CONDA_PREFIX="${CONDA_DIR}/envs/${CONDA_ENV}" \
        "${ISAACLAB_DIR}/isaaclab.sh" -p \
        "${EVAL_SCRIPT}" \
        --task "${TASK}" \
        --policy "${policy}" \
        --model "${model}" \
        --enable_cameras \
        ${HEADLESS} \
        --num_envs ${NUM_ENVS} \
        --episodes ${EPISODES} \
        --max_steps ${MAX_STEPS} \
        2>&1 | tee "${log_file}"
}

extract_results() {
    local log_file="$1"
    local name="$2"

    local completed
    completed=$(grep -c "^Episode " "$log_file" 2>/dev/null || echo "0")
    local successes
    successes=$(grep "success=True" "$log_file" 2>/dev/null | wc -l || echo "0")

    if [ "$completed" -gt 0 ]; then
        local rate
        rate=$(python3 -c "print(f'{${successes}/${completed}*100:.1f}%')")
        echo "${name}|${completed}|${successes}|${rate}"
    else
        echo "${name}|0|0|FAILED"
    fi
}

# ---- Results storage ----
declare -a RESULTS

# ============================================================
# 1. Pi-Zero
# ============================================================
echo ""
echo "============================================="
echo "[1/4] Pi-Zero: ${PIZERO_MODEL}"
echo "============================================="
cleanup_gpu
set_transformers "${TRANSFORMERS_PIZERO}"

LOG="${LOG_DIR}/pizero_${TIMESTAMP}.log"
echo "  Log: ${LOG}"
echo ""

if run_eval "pizero" "${PIZERO_MODEL}" "${LOG}"; then
    RESULTS+=("$(extract_results "$LOG" "Pi-Zero")")
else
    RESULTS+=("Pi-Zero|0|0|ERROR")
fi

# ============================================================
# 2. GR00T N1.5
# ============================================================
echo ""
echo "============================================="
echo "[2/4] GR00T N1.5: ${GROOT_N15_MODEL}"
echo "============================================="
cleanup_gpu
set_transformers "${TRANSFORMERS_GROOT_N15}"

LOG="${LOG_DIR}/groot_n15_${TIMESTAMP}.log"
echo "  Log: ${LOG}"
echo ""

if run_eval "groot" "${GROOT_N15_MODEL}" "${LOG}"; then
    RESULTS+=("$(extract_results "$LOG" "GR00T N1.5")")
else
    RESULTS+=("GR00T N1.5|0|0|ERROR")
fi

# ============================================================
# 3. GR00T N1.6 (local fine-tuned)
# ============================================================
echo ""
echo "============================================="
echo "[3/4] GR00T N1.6: ${GROOT_N16_MODEL}"
echo "============================================="
cleanup_gpu
set_transformers "${TRANSFORMERS_GROOT_N16}"

if [ ! -d "${GROOT_N16_MODEL}" ]; then
    echo "  WARNING: GR00T N1.6 model not found at ${GROOT_N16_MODEL}"
    echo "  Skipping. Train with: bash scripts/data_pipeline/brev_train_groot_n16.sh"
    RESULTS+=("GR00T N1.6|0|0|SKIPPED")
else
    LOG="${LOG_DIR}/groot_n16_${TIMESTAMP}.log"
    echo "  Log: ${LOG}"
    echo ""

    if run_eval "groot_n16" "${GROOT_N16_MODEL}" "${LOG}"; then
        RESULTS+=("$(extract_results "$LOG" "GR00T N1.6")")
    else
        RESULTS+=("GR00T N1.6|0|0|ERROR")
    fi
fi

# ============================================================
# 4. OpenVLA
# ============================================================
echo ""
echo "============================================="
echo "[4/4] OpenVLA: ${OPENVLA_MODEL}"
echo "============================================="
cleanup_gpu
set_transformers "${TRANSFORMERS_OPENVLA}"

LOG="${LOG_DIR}/openvla_${TIMESTAMP}.log"
echo "  Log: ${LOG}"
echo ""

if run_eval "openvla" "${OPENVLA_MODEL}" "${LOG}"; then
    RESULTS+=("$(extract_results "$LOG" "OpenVLA")")
else
    RESULTS+=("OpenVLA|0|0|ERROR")
fi

# ============================================================
# Restore transformers to default (4.51.3 for GR00T compat)
# ============================================================
cleanup_gpu
set_transformers "4.51.3"

# ============================================================
# Print summary
# ============================================================
echo ""
echo ""
echo "============================================="
echo "EVALUATION SUMMARY"
echo "============================================="
echo "Task: ${TASK}"
echo "Episodes per model: ${EPISODES}"
echo "Max steps per episode: ${MAX_STEPS}"
echo "Date: $(date)"
echo ""

# Print table header
printf "%-20s | %10s | %10s | %12s\n" "Model" "Episodes" "Successes" "Success Rate"
printf "%-20s-+-%10s-+-%10s-+-%12s\n" "--------------------" "----------" "----------" "------------"

for result in "${RESULTS[@]}"; do
    IFS='|' read -r name episodes successes rate <<< "$result"
    printf "%-20s | %10s | %10s | %12s\n" "$name" "$episodes" "$successes" "$rate"
done

echo ""
echo "Logs: ${LOG_DIR}/*_${TIMESTAMP}.log"
echo "============================================="

# Save summary to file
{
    echo "VLA Evaluation Summary"
    echo "Task: ${TASK}"
    echo "Date: $(date)"
    echo "Episodes: ${EPISODES}, Max steps: ${MAX_STEPS}"
    echo ""
    printf "%-20s | %10s | %10s | %12s\n" "Model" "Episodes" "Successes" "Success Rate"
    printf "%-20s-+-%10s-+-%10s-+-%12s\n" "--------------------" "----------" "----------" "------------"
    for result in "${RESULTS[@]}"; do
        IFS='|' read -r name episodes successes rate <<< "$result"
        printf "%-20s | %10s | %10s | %12s\n" "$name" "$episodes" "$successes" "$rate"
    done
} > "${SUMMARY_FILE}"

echo "Summary saved to: ${SUMMARY_FILE}"
