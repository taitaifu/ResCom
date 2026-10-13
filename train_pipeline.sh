#!/usr/bin/env bash
set -euo pipefail

# V5 state -> force cascade pipeline.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
FEATURE_DIR="${FEATURE_DIR:-Feature_Selection/DataSet}"
MERGED_CSV="${MERGED_CSV:-Feature_Selection/DataSet/merged_error_dataset.csv}"
V5_RUN_JOINT="${V5_RUN_JOINT:-0}"
V5_RUN_DIAGNOSTIC="${V5_RUN_DIAGNOSTIC:-1}"
V5_RUN_STATE_INFER="${V5_RUN_STATE_INFER:-1}"
V5_RUN_FORCE_INFER="${V5_RUN_FORCE_INFER:-1}"
V5_INFER_SPLIT="${V5_INFER_SPLIT:-val}"

latest_checkpoint() {
    local stage_dir="$1"
    local filename="$2"
    [[ -d "${stage_dir}" ]] || return 0
    find "${stage_dir}" -type f -name "${filename}" -printf '%T@ %p\n' \
        | sort -nr \
        | head -n 1 \
        | cut -d' ' -f2-
}

run_v5_pipeline() {
    local save_dir="${V5_SAVE_DIR:-results_v5}"
    local batch_size="${V5_BATCH_SIZE:-128}"
    local grad_accum_steps="${V5_GRAD_ACCUM_STEPS:-1}"
    local epochs="${V5_EPOCHS:-100}"
    local history_len="${V5_HISTORY_LEN:-9}"
    local teacher_future_len="${V5_TEACHER_FUTURE_LEN:-3}"
    local body_rollout_len="${V5_BODY_ROLLOUT_LEN:-5}"
    local num_workers="${V5_NUM_WORKERS:-4}"
    local lr="${V5_LR:-3e-4}"
    local seed="${V5_SEED:-42}"
    local run_stamp="$(date +%Y%m%d_%H%M%S)"
    local common_args=(
        --feature_dir "${FEATURE_DIR}"
        --merged_csv "${MERGED_CSV}"
        --save_dir "${save_dir}"
        --history_len "${history_len}"
        --teacher_future_len "${teacher_future_len}"
        --body_rollout_len "${body_rollout_len}"
        --batch_size "${batch_size}"
        --grad_accum_steps "${grad_accum_steps}"
        --epochs "${epochs}"
        --lr "${lr}"
        --num_workers "${num_workers}"
        --seed "${seed}"
        --device "${DEVICE}"
    )

    echo "[Pipeline] Stage A: V5 state_teacher"
    # "${PYTHON_BIN}" train/train_graph_model_v5.py \
    #     "${common_args[@]}" \
    #     --train_stage state_teacher

    local teacher_checkpoint
    teacher_checkpoint="$(latest_checkpoint "${save_dir}/state_teacher" best_state_total.pt)"
    if [[ -z "${teacher_checkpoint}" ]]; then
        teacher_checkpoint="$(latest_checkpoint "${save_dir}/state_teacher" best_state_output.pt)"
    fi
    if [[ -z "${teacher_checkpoint}" ]]; then
        echo "[Pipeline] ERROR: no V5 state_teacher checkpoint found" >&2
        exit 1
    fi
    echo "[Pipeline] State Teacher checkpoint: ${teacher_checkpoint}"

    echo "[Pipeline] Stage B: V5 state_student"
    # "${PYTHON_BIN}" train/train_graph_model_v5.py \
    #     "${common_args[@]}" \
    #     --train_stage state_student \
    #     --init_ckpt "${teacher_checkpoint}"

    local student_checkpoint
    student_checkpoint="$(latest_checkpoint "${save_dir}/state_student" best_rollout.pt)"
    if [[ -z "${student_checkpoint}" ]]; then
        student_checkpoint="$(latest_checkpoint "${save_dir}/state_student" best_state_output.pt)"
    fi
    if [[ -z "${student_checkpoint}" ]]; then
        student_checkpoint="$(latest_checkpoint "${save_dir}/state_student" best_state_total.pt)"
    fi
    if [[ -z "${student_checkpoint}" ]]; then
        echo "[Pipeline] ERROR: no V5 state_student checkpoint found" >&2
        exit 1
    fi
    echo "[Pipeline] State Student checkpoint: ${student_checkpoint}"

    if [[ "${V5_RUN_STATE_INFER}" == "1" ]]; then
        local state_infer_output="${V5_STATE_INFER_OUTPUT:-${save_dir}/pipeline_${run_stamp}/state_student}"
        echo "[Pipeline] Stage B2: V5 State Independent Inference (${V5_INFER_SPLIT})"
        "${PYTHON_BIN}" infer/infer_graph_model_v5.py \
            --checkpoint "${student_checkpoint}" \
            --mode state_student \
            --infer_mode merged_csv \
            --split "${V5_INFER_SPLIT}" \
            --feature_dir "${FEATURE_DIR}" \
            --merged_csv "${MERGED_CSV}" \
            --history_len "${history_len}" \
            --teacher_future_len "${teacher_future_len}" \
            --seed "${seed}" \
            --device "${DEVICE}" \
            --output_dir "${state_infer_output}"
    else
        echo "[Pipeline] Stage B2: V5 State Independent Inference disabled"
    fi

    if [[ "${V5_RUN_DIAGNOSTIC}" == "1" ]]; then
        local diagnostic_output="${V5_DIAGNOSTIC_OUTPUT:-tools/v5_teacher_student_diagnostic/results/pipeline_${run_stamp}}"
        echo "[Pipeline] Stage B3: V5 Teacher-Student Diagnostic (${V5_INFER_SPLIT})"
        "${PYTHON_BIN}" tools/v5_teacher_student_diagnostic/compare_teacher_student.py \
            --teacher_checkpoint "${teacher_checkpoint}" \
            --student_checkpoint "${student_checkpoint}" \
            --feature_dir "${FEATURE_DIR}" \
            --merged_csv "${MERGED_CSV}" \
            --split "${V5_INFER_SPLIT}" \
            --history_len "${history_len}" \
            --teacher_future_len "${teacher_future_len}" \
            --seed "${seed}" \
            --device "${DEVICE}" \
            --output_dir "${diagnostic_output}"
    else
        echo "[Pipeline] Stage B3: V5 Teacher-Student Diagnostic disabled"
    fi

    echo "[Pipeline] Stage C: V5 force (frozen state network)"
    "${PYTHON_BIN}" train/train_graph_model_v5.py \
        "${common_args[@]}" \
        --train_stage force \
        --state_checkpoint "${student_checkpoint}"

    local force_checkpoint
    force_checkpoint="$(latest_checkpoint "${save_dir}/force" best_force.pt)"
    if [[ -z "${force_checkpoint}" ]]; then
        echo "[Pipeline] ERROR: no V5 Force checkpoint found" >&2
        exit 1
    fi
    echo "[Pipeline] Force checkpoint: ${force_checkpoint}"

    if [[ "${V5_RUN_FORCE_INFER}" == "1" ]]; then
        local force_infer_output="${V5_FORCE_INFER_OUTPUT:-${save_dir}/pipeline_${run_stamp}/force}"
        echo "[Pipeline] Stage C2: V5 Force Independent Inference (${V5_INFER_SPLIT})"
        "${PYTHON_BIN}" infer/infer_graph_model_v5.py \
            --checkpoint "${force_checkpoint}" \
            --mode force \
            --infer_mode merged_csv \
            --split "${V5_INFER_SPLIT}" \
            --feature_dir "${FEATURE_DIR}" \
            --merged_csv "${MERGED_CSV}" \
            --history_len "${history_len}" \
            --teacher_future_len "${teacher_future_len}" \
            --seed "${seed}" \
            --device "${DEVICE}" \
            --output_dir "${force_infer_output}"
    else
        echo "[Pipeline] Stage C2: V5 Force Independent Inference disabled"
    fi

    if [[ "${V5_RUN_JOINT}" == "1" ]]; then
        echo "[Pipeline] Optional Stage D: V5 joint"
        "${PYTHON_BIN}" train/train_graph_model_v5.py \
            "${common_args[@]}" \
            --train_stage joint \
            --force_checkpoint "${force_checkpoint}"
    else
        echo "[Pipeline] Joint stage disabled (set V5_RUN_JOINT=1 to enable)"
    fi
}

echo "[Pipeline] Selected V5 cascade"
run_v5_pipeline
echo "[Pipeline] Completed (v5)"
