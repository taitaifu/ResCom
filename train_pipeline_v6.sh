#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ABLATION="${ABLATION:-v6_a}"
DEVICE="${DEVICE:-cuda:0}"
FEATURE_DIR="${FEATURE_DIR:-Feature_Selection/DataSet}"
MERGED_CSV="${MERGED_CSV:-Feature_Selection/DataSet/merged_error_dataset.csv}"
RESULT_ROOT="results_v6/${ABLATION}"
RUN_DIR="${RESULT_ROOT}/pipeline_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "${RUN_DIR}"
trap 'printf "Pipeline stopped at stage %s. See %s\n" "${CURRENT_STAGE:-preflight}" "${RUN_DIR}" >&2' ERR
common=(--ablation "${ABLATION}" --feature_dir "${FEATURE_DIR}" --merged_csv "${MERGED_CSV}"
    --history_len "${V6_HISTORY_LEN:-9}" --teacher_future_len "${V6_TEACHER_FUTURE_LEN:-3}"
    --body_rollout_len "${V6_BODY_ROLLOUT_LEN:-5}" --batch_size "${V6_BATCH_SIZE:-64}"
    --grad_accum_steps "${V6_GRAD_ACCUM_STEPS:-1}" --epochs "${V6_EPOCHS:-100}"
    --patience "${V6_PATIENCE:-15}"
    --num_workers "${V6_NUM_WORKERS:-0}" --gradient_log_interval "${V6_GRADIENT_LOG_INTERVAL:-100}"
    --seed "${V6_SEED:-42}" --device "${DEVICE}")
for flag in "$@"; do
    if [[ "${flag}" == --ablation || "${flag}" == --ablation=* ]]; then
        printf 'Select preset using ABLATION environment variable.\n' >&2
        exit 2
    fi
done
common+=("$@")

run_stage() {
    local stage="$1"; shift
    CURRENT_STAGE="${stage}"
    local skip_variable="V6_SKIP_${stage^^}_CKPT"
    local skip_checkpoint="${!skip_variable:-}"
    if [[ -n "${skip_checkpoint}" ]]; then
        "${PYTHON_BIN}" tools/v6_report.py --register_checkpoint "${skip_checkpoint}" --stage "${stage}" \
            --stage_dir "${RUN_DIR}/${stage}" --expected_ablation "${ABLATION}" --config_path "${RUN_DIR}/config_v6.json" 2>&1 | tee "${RUN_DIR}/${stage}.log"
        return
    fi
    "${PYTHON_BIN}" -u train/train_graph_model_v6.py "${common[@]}" \
        --train_stage "${stage}" --run_dir "${RUN_DIR}/${stage}" "$@" 2>&1 | tee "${RUN_DIR}/${stage}.log"
    [[ -s "${RUN_DIR}/${stage}/best.pt" && -s "${RUN_DIR}/${stage}/stage_status.json" ]]
}

CURRENT_STAGE=environment
"${PYTHON_BIN}" tools/check_v6.py --syntax --environment --feature_dir "${FEATURE_DIR}" --merged_csv "${MERGED_CSV}" | tee "${RUN_DIR}/environment.log"
CURRENT_STAGE=config
"${PYTHON_BIN}" train/train_graph_model_v6.py "${common[@]}" --check_config --report_path "${RUN_DIR}/config_v6.json" | tee "${RUN_DIR}/config.log"
CURRENT_STAGE=smoke
"${PYTHON_BIN}" v6_tests/test_v6.py --smoke --ablation "${ABLATION}" --config_json "${RUN_DIR}/config_v6.json" --device "${DEVICE}" | tee "${RUN_DIR}/smoke.log"
CURRENT_STAGE=data
"${PYTHON_BIN}" train/train_graph_model_v6.py "${common[@]}" --prepare_only --run_dir "${RUN_DIR}/data" | tee "${RUN_DIR}/data.log"

# Stage A: Teacher. Optional V5 Teacher migration uses strict feature/scaler checks.
teacher_init=()
if [[ -n "${V6_TEACHER_INIT:-}" ]]; then teacher_init=(--init_ckpt "${V6_TEACHER_INIT}"); fi
run_stage state_teacher --lr "${V6_LR:-3e-4}" "${teacher_init[@]}"
run_stage state_student --lr "${V6_LR:-3e-4}" --init_ckpt "${RUN_DIR}/state_teacher/best.pt"
run_stage force --lr "${V6_FORCE_LR:-3e-4}" --state_checkpoint "${RUN_DIR}/state_student/best.pt"
JOINT_INIT="${RUN_DIR}/force/best.pt"
# Stage D: only applicable to force/state coupled presets. V6-A bypasses this command.
NEEDS_PHYSICS="$("${PYTHON_BIN}" tools/v6_report.py --config_path "${RUN_DIR}/config_v6.json" --needs_physics_stage)"
if [[ "${NEEDS_PHYSICS}" == yes ]]; then
    run_stage physics --lr "${V6_PHYSICS_LR:-1e-4}" --force_checkpoint "${JOINT_INIT}"
    JOINT_INIT="${RUN_DIR}/physics/best.pt"
fi
run_stage joint --lr "${V6_JOINT_LR:-3e-5}" --force_checkpoint "${JOINT_INIT}"
FINAL_CKPT="${RUN_DIR}/joint/best.pt"
for split in val test; do
    CURRENT_STAGE="evaluate_${split}"
    report=validation_report.json
    if [[ "${split}" == test ]]; then report=test_report.json; fi
    "${PYTHON_BIN}" -u train/train_graph_model_v6.py "${common[@]}" --eval_checkpoint "${FINAL_CKPT}" \
        --eval_split "${split}" --report_path "${RUN_DIR}/${report}" 2>&1 | tee "${RUN_DIR}/evaluate_${split}.log"
done
# Evaluate observed LF compensation and state recurrence as distinct modes.
for mode in lf_conditioned autonomous; do
    CURRENT_STAGE="continuous_${mode}"
    "${PYTHON_BIN}" -u infer/infer_graph_model_v6.py --checkpoint "${FINAL_CKPT}" \
        --infer_mode merged_csv --split test --feature_dir "${FEATURE_DIR}" --merged_csv "${MERGED_CSV}" \
        --rollout_mode "${mode}" --device "${DEVICE}" --output_dir "${RUN_DIR}/${mode}" 2>&1 | tee "${RUN_DIR}/${mode}.log"
done
CURRENT_STAGE=report
v5_report=()
if [[ -n "${V5_COMPARE_REPORT:-}" ]]; then v5_report=(--v5_report "${V5_COMPARE_REPORT}"); fi
"${PYTHON_BIN}" tools/v6_report.py --run_dir "${RUN_DIR}" --publish_best "${RESULT_ROOT}/best_final.pt" \
    "${v5_report[@]}" | tee "${RUN_DIR}/final_summary.log"
printf 'All stages complete. Final checkpoint: %s\nFinal report: %s\n' "${RESULT_ROOT}/best_final.pt" "${RUN_DIR}/final_report.json"
