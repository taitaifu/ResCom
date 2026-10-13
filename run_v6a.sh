#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
# Only the corrected V6-A baseline executes from this entry point.
# Preset disables all new physics NETWORK modules; original V5 history remains.
export ABLATION=v6_a
bash train_pipeline_v6.sh

# Later Stage B: Physics-aware Graph, edge decoder, enhanced causal history/gates.
# Enable by running this command manually after V6-A acceptance:
# ABLATION=v6_b bash train_pipeline_v6.sh

# Later Stage C: Force Aggregation, Dynamics Head and State Refinement.
# ABLATION=v6_c bash train_pipeline_v6.sh

# Later Stage D: Physics Refinement, Pushforward and Joint Fine-tuning.
# ABLATION=v6_d bash train_pipeline_v6.sh

# Separate later Physics refinement stage (existing V6-C Force checkpoint required):
# python train/train_graph_model_v6.py --ablation v6_c --train_stage physics --force_checkpoint results_v6/v6_c/best_final.pt
# Physics refinement + 2 same-time correction iterations (V6-D checkpoint required):
# python train/train_graph_model_v6.py --ablation v6_d --train_stage joint --physics_iterations 2 --force_checkpoint results_v6/v6_d/best_final.pt
# Pushforward is enabled only in v6_d; override on a compatible v6_c Joint run:
# python train/train_graph_model_v6.py --ablation v6_c --enable_pushforward --train_stage joint --force_checkpoint results_v6/v6_c/best_final.pt
# Later independent B/C/D inference (choose the checkpoint of that ablation):
# python infer/infer_graph_model_v6.py --checkpoint results_v6/v6_b/best_final.pt --split val --output_dir results_v6/v6_b/infer_val
# python infer/infer_graph_model_v6.py --checkpoint results_v6/v6_c/best_final.pt --split val --output_dir results_v6/v6_c/infer_val
# python infer/infer_graph_model_v6.py --checkpoint results_v6/v6_d/best_final.pt --split val --output_dir results_v6/v6_d/infer_val
# Later B/C/D independent validation:
# python train/train_graph_model_v6.py --ablation v6_b --eval_checkpoint results_v6/v6_b/best_final.pt --eval_split val --report_path results_v6/v6_b/validation.json
# python train/train_graph_model_v6.py --ablation v6_c --eval_checkpoint results_v6/v6_c/best_final.pt --eval_split val --report_path results_v6/v6_c/validation.json
# python train/train_graph_model_v6.py --ablation v6_d --eval_checkpoint results_v6/v6_d/best_final.pt --eval_split val --report_path results_v6/v6_d/validation.json
