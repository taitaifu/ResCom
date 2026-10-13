#!/usr/bin/env python3
"""Compare V5 State Teacher and causal Student on identical cases/timestamps."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INFER = ROOT / "infer" / "infer_graph_model_v5.py"
AXES = ("x", "y", "z")


def run_state_inference(args, mode: str, checkpoint: str, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(INFER), "--checkpoint", checkpoint, "--mode", mode,
        "--infer_mode", "merged_csv", "--split", args.split,
        "--feature_dir", args.feature_dir, "--merged_csv", args.merged_csv,
        "--history_len", str(args.history_len), "--teacher_future_len", str(args.teacher_future_len),
        "--seed", str(args.seed), "--device", args.device, "--output_dir", str(output_dir),
    ]
    subprocess.run(cmd, check=True)
    path = output_dir / "predictions_v5.csv"
    if not path.exists():
        raise FileNotFoundError(f"V5 inference did not produce {path}")
    summary_path = output_dir / "metrics_v5.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"V5 inference did not produce {summary_path}")
    return pd.read_csv(path)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def _angle_error_deg(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    q0 = q0 / np.maximum(np.linalg.norm(q0, axis=1, keepdims=True), 1e-12)
    q1 = q1 / np.maximum(np.linalg.norm(q1, axis=1, keepdims=True), 1e-12)
    cosine = np.abs(np.sum(q0 * q1, axis=1)).clip(0.0, 1.0)
    return np.degrees(2.0 * np.arccos(cosine))


def _quat_to_euler_deg(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q.T
    roll = np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
    pitch = np.arcsin(np.clip(2 * (w*y - z*x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    return np.degrees(np.column_stack([roll, pitch, yaw]))


def _quat_columns(frame: pd.DataFrame, source: str, group: str) -> List[str]:
    return [f"{source}::{group}::q{i}" for i in range(4)]


def _vector_columns(source: str, group: str, kind: str, axes: Tuple[str, ...] = AXES) -> List[str]:
    return [f"{source}::{group}::{kind}::{axis}" for axis in axes]


def _score_vector(
    merged: pd.DataFrame,
    rows: List[Dict[str, object]],
    metric: str,
    pred_cols: List[str],
    lf_cols: List[str],
    true_cols: List[str],
) -> None:
    if not all(c in merged for c in pred_cols + lf_cols + true_cols):
        return
    pred = merged[pred_cols].to_numpy(dtype=np.float64)
    lf = merged[lf_cols].to_numpy(dtype=np.float64)
    true = merged[true_cols].to_numpy(dtype=np.float64)
    pred_error = np.mean((pred - true) ** 2, axis=1)
    lf_error = np.mean((lf - true) ** 2, axis=1)
    pred_rmse, lf_rmse = _rmse(pred, true), _rmse(lf, true)
    rows.append({"metric": f"{metric}_teacher_rmse", "value": pred_rmse})
    rows.append({"metric": f"{metric}_lf_rmse", "value": lf_rmse})
    rows.append({"metric": f"{metric}_teacher_improvement_rate", "value": (lf_rmse - pred_rmse) / max(lf_rmse, 1e-12)})
    rows.append({"metric": f"{metric}_teacher_harmful_ratio", "value": float(np.mean(pred_error > lf_error))})


def _score_attitude(frame: pd.DataFrame, group: str, model: str, rows: List[Dict[str, object]]) -> None:
    pred_cols, lf_cols, true_cols = (_quat_columns(frame, source, group) for source in ("pred", "lf", "true"))
    if not all(c in frame for c in pred_cols + lf_cols + true_cols):
        return
    pred = frame[pred_cols].to_numpy(dtype=np.float64)
    lf = frame[lf_cols].to_numpy(dtype=np.float64)
    true = frame[true_cols].to_numpy(dtype=np.float64)
    pred_error, lf_error = _angle_error_deg(pred, true), _angle_error_deg(lf, true)
    pred_rmse = float(np.sqrt(np.mean(pred_error ** 2)))
    lf_rmse = float(np.sqrt(np.mean(lf_error ** 2)))
    rows.extend([
        {"metric": f"{model}_{group}_attitude_rmse_deg", "value": pred_rmse},
        {"metric": f"{model}_{group}_attitude_improvement_rate", "value": (lf_rmse - pred_rmse) / max(lf_rmse, 1e-12)},
        {"metric": f"{model}_{group}_attitude_harmful_ratio", "value": float(np.mean(pred_error > lf_error))},
    ])


def _add_rollout_metrics(frame: pd.DataFrame, model: str, rows: List[Dict[str, object]]) -> None:
    pred_cols = _vector_columns("pred", "body", "position")
    prop_cols = _vector_columns("pred_prop", "body", "position")
    true_cols = _vector_columns("true", "body", "position")
    if not all(c in frame for c in pred_cols + prop_cols + true_cols):
        return
    final = frame[pred_cols].to_numpy(dtype=np.float64)
    prop = frame[prop_cols].to_numpy(dtype=np.float64)
    true = frame[true_cols].to_numpy(dtype=np.float64)
    rows.append({"metric": f"{model}_body_continuous_prop_rmse", "value": _rmse(prop, true)})
    rows.append({"metric": f"{model}_body_continuous_final_rmse", "value": _rmse(final, true)})
    drift, endpoint_error, accumulated_step_error = [], [], []
    for _, case in frame.groupby("case_name", sort=False):
        case = case.sort_values("time")
        p = case[pred_cols].to_numpy(dtype=np.float64)
        h = case[true_cols].to_numpy(dtype=np.float64)
        valid = np.isfinite(p).all(axis=1) & np.isfinite(h).all(axis=1)
        p, h = p[valid], h[valid]
        if len(p) < 2:
            continue
        drift.append(float(np.linalg.norm((p[-1] - p[0]) - (h[-1] - h[0]))))
        endpoint_error.append(float(np.linalg.norm(p[-1] - h[-1])))
        accumulated_step_error.append(float(np.sum(np.linalg.norm(np.diff(p, axis=0) - np.diff(h, axis=0), axis=1))))
    rows.extend([
        {"metric": f"{model}_body_case_endpoint_error_mean", "value": float(np.mean(endpoint_error)) if endpoint_error else float("nan")},
        {"metric": f"{model}_body_case_net_drift_error_mean", "value": float(np.mean(drift)) if drift else float("nan")},
        {"metric": f"{model}_body_case_accumulated_step_error_mean", "value": float(np.mean(accumulated_step_error)) if accumulated_step_error else float("nan")},
    ])


def compare_latents(output_dir: Path, rows: List[Dict[str, object]]) -> pd.DataFrame:
    t_path, s_path = output_dir / "teacher" / "latents_v5.csv", output_dir / "student" / "latents_v5.csv"
    if not t_path.exists() or not s_path.exists():
        return pd.DataFrame()
    teacher, student = pd.read_csv(t_path), pd.read_csv(s_path)
    keys = ["case_name", "time"]
    zt = sorted(c for c in teacher.columns if c.startswith("z_"))
    zs = sorted(c for c in student.columns if c.startswith("z_"))
    if zt != zs:
        raise ValueError(f"Teacher/Student latent dimensions differ: {len(zt)} vs {len(zs)}")
    paired = teacher.merge(student, on=keys, suffixes=("_teacher", "_student"), validate="one_to_one")
    a = paired[[f"{c}_teacher" for c in zt]].to_numpy(dtype=np.float64)
    b = paired[[f"{c}_student" for c in zs]].to_numpy(dtype=np.float64)
    rows.append({"metric": "latent_distillation_mse", "value": float(np.mean((a - b) ** 2))})
    rows.append({"metric": "latent_distillation_rmse", "value": float(np.sqrt(np.mean((a - b) ** 2)))})
    cosine = np.sum(a * b, axis=1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
    rows.append({"metric": "latent_distillation_cosine_mean", "value": float(np.mean(cosine))})
    return paired


def make_comparison_plots(merged: pd.DataFrame, output_dir: Path) -> None:
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    groups = ["body", *[f"rocker_{n}" for n in ("lf", "lm", "lb", "rf", "rm", "rb")], *[f"wheel{i}" for i in range(6)]]
    for case_name, case in merged.groupby("case_name", sort=False):
        case = case.sort_values("time")
        case_dir = plots / str(case_name)
        case_dir.mkdir(parents=True, exist_ok=True)
        time = case.time.to_numpy(dtype=np.float64)
        for group in groups:
            for axis in AXES:
                pred_key = f"pred::{group}::position::{axis}"
                lf_key = f"lf::{group}::position::{axis}"
                true_key = f"true::{group}::position::{axis}"
                if not all(f"{key}_{suffix}" in case for key in (pred_key, lf_key, true_key) for suffix in ("teacher", "student")):
                    continue
                fig, ax = plt.subplots(figsize=(10, 4))
                for label, key in (("LF", lf_key), ("Teacher", pred_key), ("Student", pred_key), ("HF", true_key)):
                    suffix = "teacher" if label == "Teacher" else "student" if label == "Student" else "teacher"
                    ax.plot(time, case[f"{key}_{suffix}"].to_numpy(dtype=np.float64), label=label)
                ax.set(title=f"{case_name} | {group} position {axis}", xlabel="time / s", ylabel=f"position {axis}")
                ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
                fig.savefig(case_dir / f"{group}_position_{axis}.png", dpi=130); plt.close(fig)
            qsets = {label: [f"{col}_{suffix}" for col in _quat_columns(case, source, group)] for label, source, suffix in (("LF", "lf", "teacher"), ("Teacher", "pred", "teacher"), ("HF", "true", "teacher"))}
            student_cols = [f"{col}_student" for col in _quat_columns(case, "pred", group)]
            if all(all(col in case for col in cols) for cols in qsets.values()):
                eulers = {label: _quat_to_euler_deg(case[cols].to_numpy(dtype=np.float64)) for label, cols in qsets.items()}
                student_euler = _quat_to_euler_deg(case[student_cols].to_numpy(dtype=np.float64)) if all(c in case for c in student_cols) else None
                for j, axis in enumerate(("roll", "pitch", "yaw")):
                    fig, ax = plt.subplots(figsize=(10, 4))
                    for label in ("LF", "Teacher"):
                        ax.plot(time, eulers[label][:, j], label=label)
                    if student_euler is not None:
                        ax.plot(time, student_euler[:, j], label="Student")
                    ax.plot(time, eulers["HF"][:, j], label="HF")
                    ax.set(title=f"{case_name} | {group} attitude {axis}", xlabel="time / s", ylabel=f"{axis} / deg")
                    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
                    fig.savefig(case_dir / f"{group}_attitude_{axis}.png", dpi=130); plt.close(fig)


def run(args: argparse.Namespace) -> Dict[str, object]:
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    teacher = run_state_inference(args, "state_teacher", args.teacher_checkpoint, root / "teacher")
    student = run_state_inference(args, "state_student", args.student_checkpoint, root / "student")
    if teacher.duplicated(["case_name", "time"]).any() or student.duplicated(["case_name", "time"]).any():
        raise ValueError("inference output contains duplicate case/time rows")
    keys = ["case_name", "time"]
    t_keys, s_keys = set(map(tuple, teacher[keys].to_numpy())), set(map(tuple, student[keys].to_numpy()))
    if t_keys != s_keys:
        raise ValueError(f"Teacher and Student timestamp sets differ: teacher={len(t_keys)}, student={len(s_keys)}")
    merged = teacher.merge(student, on=keys, suffixes=("_teacher", "_student"), validate="one_to_one")
    # The reference/input values must be identical after timestamp alignment.
    for col in teacher.columns:
        if col.startswith(("lf::", "true::")) and col in student:
            a, b = teacher[col].to_numpy(), student[col].to_numpy()
            if not np.allclose(a, b, equal_nan=True, atol=1e-6, rtol=1e-6):
                raise ValueError(f"LF/HF reference mismatch between Teacher and Student at column {col}")

    report: List[Dict[str, object]] = []
    metric_groups = ["body", *[f"rocker_{n}" for n in ("lf", "lm", "lb", "rf", "rm", "rb")], *[f"wheel{i}" for i in range(6)]]
    for model_name in ("teacher", "student"):
        model_frame = teacher if model_name == "teacher" else student
        for group in metric_groups:
            _score_vector(
                model_frame, report, f"{model_name}_{group}_position",
                _vector_columns("pred", group, "position"),
                _vector_columns("lf", group, "position"),
                _vector_columns("true", group, "position"),
            )
            _score_attitude(model_frame, group, model_name, report)
        for axis in AXES:
            _score_vector(
                model_frame, report, f"{model_name}_body_velocity_{axis}",
                [f"pred::body::velocity::{axis}"], [f"lf::body::velocity::{axis}"], [f"true::body::velocity::{axis}"],
            )
        _add_rollout_metrics(model_frame, model_name, report)

    # Per-row teacher/student state difference and harmful ratio, paired on exact keys.
    for group in metric_groups:
        pred_cols = _vector_columns("pred", group, "position")
        truth_cols = _vector_columns("true", group, "position")
        if all(c in teacher and c in student for c in pred_cols):
            t = teacher[pred_cols].to_numpy(dtype=np.float64)
            s = student[pred_cols].to_numpy(dtype=np.float64)
            report.append({"metric": f"{group}_student_teacher_position_rmse", "value": _rmse(s, t)})
            if all(c in teacher and c in student for c in truth_cols):
                hf = teacher[truth_cols].to_numpy(dtype=np.float64)
                report.append({"metric": f"{group}_student_harmful_vs_teacher_ratio", "value": float(np.mean(np.mean((s-hf)**2,axis=1) > np.mean((t-hf)**2,axis=1)))})
        tq, sq = _quat_columns(teacher, "pred", group), _quat_columns(student, "pred", group)
        if all(c in teacher and c in student for c in tq + sq):
            err = _angle_error_deg(teacher[tq].to_numpy(dtype=np.float64), student[sq].to_numpy(dtype=np.float64))
            report.append({"metric": f"{group}_student_teacher_attitude_difference_mean_deg", "value": float(np.mean(err))})
    velocity_cols = [f"pred::body::velocity::{axis}" for axis in AXES]
    if all(c in teacher and c in student for c in velocity_cols):
        report.append({
            "metric": "body_student_teacher_velocity_rmse",
            "value": _rmse(student[velocity_cols].to_numpy(dtype=np.float64), teacher[velocity_cols].to_numpy(dtype=np.float64)),
        })
    latent_pairs = compare_latents(root, report)
    make_comparison_plots(merged, root)
    report_df = pd.DataFrame(report)
    report_df.to_csv(root / "summary.csv", index=False)
    merged.to_csv(root / "paired_predictions.csv", index=False)
    result = {
        "split": args.split,
        "teacher_checkpoint": str(Path(args.teacher_checkpoint).resolve()),
        "student_checkpoint": str(Path(args.student_checkpoint).resolve()),
        "num_aligned_timestamps": int(len(merged)),
        "num_cases": int(merged.case_name.nunique()),
        "latent_rows": int(len(latent_pairs)),
        "summary": {str(row.metric): (None if not np.isfinite(row.value) else float(row.value)) for row in report_df.itertuples()},
        "outputs": {
            "teacher_predictions": str(root / "teacher" / "predictions_v5.csv"),
            "student_predictions": str(root / "student" / "predictions_v5.csv"),
            "paired_predictions": str(root / "paired_predictions.csv"),
            "summary_csv": str(root / "summary.csv"),
            "plots": str(root / "plots"),
        },
    }
    (root / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher_checkpoint", required=True)
    parser.add_argument("--student_checkpoint", required=True)
    parser.add_argument("--feature_dir", default=str(ROOT / "Feature_Selection" / "DataSet"))
    parser.add_argument("--merged_csv", default=str(ROOT / "Feature_Selection" / "DataSet" / "merged_error_dataset.csv"))
    parser.add_argument("--output_dir", default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--history_len", type=int, default=9)
    parser.add_argument("--teacher_future_len", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
