#!/usr/bin/env bash
set -euo pipefail

# V4 Teacher -> Student -> Force Corrector pipeline.
# This file is intentionally only a command pipeline; it is not executed by
# the repository tooling automatically.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
FEATURE_DIR="${FEATURE_DIR:-Feature_Selection/DataSet}"
MERGED_CSV="${MERGED_CSV:-Feature_Selection/DataSet/merged_error_dataset.csv}"
V4_SAVE_DIR="${V4_SAVE_DIR:-results_v4}"
V4_BATCH_SIZE="${V4_BATCH_SIZE:-64}"
CORRECTOR_EPOCHS="${CORRECTOR_EPOCHS:-40}"
CORRECTOR_BATCH_SIZE="${CORRECTOR_BATCH_SIZE:-256}"
CORRECTOR_LR="${CORRECTOR_LR:-1e-3}"
V4_FINETUNE_LR="${V4_FINETUNE_LR:-1e-5}"
RUN_JOINT_FINETUNE="${RUN_JOINT_FINETUNE:-0}"

latest_checkpoint() {
    local stage_dir="$1"
    local filename="$2"
    find "${stage_dir}" -type f -name "${filename}" -printf '%T@ %p\n' \
        | sort -nr \
        | head -n 1 \
        | cut -d' ' -f2-
}

echo "[Pipeline] Stage A: V4 Teacher"
"${PYTHON_BIN}" train/train_graph_model_v4.py \
    --train_stage teacher \
    --teacher_future_len 3 \
    --batch_size "${V4_BATCH_SIZE}" \
    --feature_dir "${FEATURE_DIR}" \
    --merged_csv "${MERGED_CSV}" \
    --save_dir "${V4_SAVE_DIR}" \
    --device "${DEVICE}"

TEACHER_CHECKPOINT="$(latest_checkpoint "${V4_SAVE_DIR}/teacher" best_safe.pt)"
if [[ -z "${TEACHER_CHECKPOINT}" ]]; then
    TEACHER_CHECKPOINT="$(latest_checkpoint "${V4_SAVE_DIR}/teacher" best_total.pt)"
fi
if [[ -z "${TEACHER_CHECKPOINT}" ]]; then
    echo "[Pipeline] ERROR: no V4 Teacher checkpoint found" >&2
    exit 1
fi
echo "[Pipeline] Teacher checkpoint: ${TEACHER_CHECKPOINT}"

echo "[Pipeline] Stage B: V4 Student"
"${PYTHON_BIN}" train/train_graph_model_v4.py \
    --train_stage student \
    --init_ckpt "${TEACHER_CHECKPOINT}" \
    --teacher_future_len 3 \
    --batch_size "${V4_BATCH_SIZE}" \
    --feature_dir "${FEATURE_DIR}" \
    --merged_csv "${MERGED_CSV}" \
    --save_dir "${V4_SAVE_DIR}" \
    --device "${DEVICE}"

STUDENT_CHECKPOINT="$(latest_checkpoint "${V4_SAVE_DIR}/student" best_safe.pt)"
if [[ -z "${STUDENT_CHECKPOINT}" ]]; then
    STUDENT_CHECKPOINT="$(latest_checkpoint "${V4_SAVE_DIR}/student" best_total.pt)"
fi
if [[ -z "${STUDENT_CHECKPOINT}" ]]; then
    echo "[Pipeline] ERROR: no V4 Student checkpoint found" >&2
    exit 1
fi
echo "[Pipeline] Student checkpoint: ${STUDENT_CHECKPOINT}"

CORRECTOR_OUTPUT="${CORRECTOR_OUTPUT:-${V4_SAVE_DIR}/force_corrector_stage1}"
echo "[Pipeline] Stage C: frozen V4 Student + Force Corrector"
"${PYTHON_BIN}" tools/residual_corrector/train_v4_force_corrector.py \
    --feature_dir "${FEATURE_DIR}" \
    --merged_csv "${MERGED_CSV}" \
    --student_checkpoint "${STUDENT_CHECKPOINT}" \
    --output_dir "${CORRECTOR_OUTPUT}" \
    --device "${DEVICE}" \
    --batch_size "${CORRECTOR_BATCH_SIZE}" \
    --epochs "${CORRECTOR_EPOCHS}" \
    --corrector_lr "${CORRECTOR_LR}"

if [[ "${RUN_JOINT_FINETUNE}" == "1" ]]; then
    JOINT_OUTPUT="${JOINT_OUTPUT:-${V4_SAVE_DIR}/force_corrector_joint}"
    echo "[Pipeline] Optional Stage D: joint fine-tune"
    "${PYTHON_BIN}" tools/residual_corrector/train_v4_force_corrector.py \
        --feature_dir "${FEATURE_DIR}" \
        --merged_csv "${MERGED_CSV}" \
        --student_checkpoint "${STUDENT_CHECKPOINT}" \
        --output_dir "${JOINT_OUTPUT}" \
        --device "${DEVICE}" \
        --batch_size "${CORRECTOR_BATCH_SIZE}" \
        --epochs "${CORRECTOR_EPOCHS}" \
        --corrector_lr "${CORRECTOR_LR}" \
        --v4_finetune_lr "${V4_FINETUNE_LR}" \
        --joint_finetune
else
    echo "[Pipeline] Joint fine-tune disabled (set RUN_JOINT_FINETUNE=1 to enable)"
fi

echo "[Pipeline] Completed"
