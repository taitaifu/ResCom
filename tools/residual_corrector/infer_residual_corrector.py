from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from tools.residual_corrector.base.residual_dataset import (  # noqa: E402
    ResidualWindowDataset,
    base_feature_dim,
    feature_dim,
    load_residual_arrays,
)
from tools.residual_corrector.base.residual_state import StateResidualCorrector  # noqa: E402
from tools.residual_corrector.base.teacher_student_state import TeacherStudentResidualCorrector  # noqa: E402
from tools.residual_corrector.base.residual_tcn import ResidualTCN  # noqa: E402
from tools.residual_corrector.base.residual_utils import ensure_dir, log, save_npz


def load_config(corrector_dir: Path) -> Dict:
    with open(corrector_dir / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_alpha(corrector_dir: Path) -> np.ndarray:
    with open(corrector_dir / "alpha.json", "r", encoding="utf-8") as f:
        obj = json.load(f)
    return np.asarray([obj["alpha_x"], obj["alpha_y"], obj["alpha_z"]], dtype=np.float32)


def run_online(args: argparse.Namespace) -> None:
    proof_dir = Path(args.proof_dir)
    corrector_dir = Path(args.corrector_dir)
    out_dir = ensure_dir(args.output_dir or (corrector_dir / f"online_{args.split}"))
    config = load_config(corrector_dir)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    arrays, meta = load_residual_arrays(proof_dir, args.split)
    input_scaler = joblib.load(corrector_dir / "input_scaler.joblib")
    target_scaler = joblib.load(corrector_dir / "target_scaler.joblib")
    history_len = int(config.get("history_len", args.history_len or 10))
    residual_history_len = int(config.get("residual_history_len", args.residual_history_len or 10))
    predict_delta_residual = bool(config.get("predict_delta_residual", False))
    state_estimator = str(config.get("mode", "")) == "latent_state_estimator" or bool(config.get("state_estimator", False))
    ema_beta = float(config.get("ema_beta", 0.8))
    state_ema_beta = float(config.get("state_ema_beta", 0.8))
    ds = ResidualWindowDataset(
        arrays,
        history_len=history_len,
        residual_history_len=residual_history_len,
        input_scaler=input_scaler,
        target_scaler=target_scaler,
        oracle_residual_history=False,
        predict_delta_residual=predict_delta_residual,
        state_estimator=state_estimator,
    )
    if config.get("mode") == "teacher_student_six_wheel":
        # This branch deliberately builds no target/residual history.  F_HF is
        # present in an offline proof archive only as an evaluation label and
        # is never passed through the Student forward call.
        names = list(config["feature_names"])
        if any("hf" in name.lower() or "r_true" in name.lower() for name in names):
            raise RuntimeError("student-only input schema contains an HF/residual feature")
        model = TeacherStudentResidualCorrector(
            input_dim=int(config["input_dim"]),
            body_indices=list(config["body_indices"]),
            wheel_state_dim=int(config.get("wheel_state_dim", 32)),
            wheel_embedding_dim=int(config.get("wheel_embedding_dim", 8)),
        ).to(device)
        ckpt = torch.load(corrector_dir / "best_student.pt", map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        grouped: Dict[str, Dict[float, Dict[int, tuple[int, int]]]] = {}
        for start, end in ds.index:
            case, time, wheel = str(ds.arrays["case_name"][end]), float(ds.arrays["time"][end]), int(ds.arrays["wheel_id"][end])
            grouped.setdefault(case, {}).setdefault(time, {})[wheel] = (start, end)
        parts: Dict[str, List[np.ndarray]] = {k: [] for k in ["case_name", "time", "wheel_id", "F_base", "R_hat", "F_corrected"]}
        with torch.no_grad():
            for _, times in grouped.items():
                state = torch.zeros((1, 6, model.wheel_state_dim), device=device)
                for _, wheels in sorted(times.items()):
                    if set(wheels) != set(range(6)):
                        continue
                    row = [wheels[i] for i in range(6)]
                    x = np.stack([ds.make_model_input(start, end, np.zeros((residual_history_len, 3), dtype=np.float32))[-1] for start, end in row])
                    # Student-only API: x + its own [1,6,H] recurrent state.
                    state, r_hat = model.student_step_all(torch.from_numpy(x).unsqueeze(0).to(device), state)
                    for wheel, (_, end) in enumerate(row):
                        f_base = ds.arrays["F_base"][end].astype(np.float32, copy=False)
                        residual = r_hat[0, wheel].cpu().numpy().astype(np.float32)
                        parts["case_name"].append(np.asarray([str(ds.arrays["case_name"][end])], dtype=str))
                        parts["time"].append(np.asarray([ds.arrays["time"][end]], dtype=np.float64))
                        parts["wheel_id"].append(np.asarray([wheel], dtype=np.int64))
                        parts["F_base"].append(f_base[None, :])
                        parts["R_hat"].append(residual[None, :])
                        parts["F_corrected"].append((f_base + residual)[None, :])
        out_arrays = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
        save_npz(out_dir / f"residual_corrected_{args.split}.npz", out_arrays, {
            "proof_dir": str(proof_dir), "corrector_dir": str(corrector_dir), "split": args.split,
            "mode": "online_teacher_student_student_only", "uses_hf": False,
            "uses_true_residual": False, "uses_teacher_hidden": False,
            "residual_convention": "R_true=F_HF-F_base; F_corr=F_base+R_pred",
            "state_initialization": "zeros_per_case",
        })
        log("INFER", f"saved student-only corrected forces: {out_dir / f'residual_corrected_{args.split}.npz'}")
        return
    if state_estimator:
        model = StateResidualCorrector(
            input_dim=base_feature_dim(),
            state_dim=int(config.get("state_dim", 64)),
            wheel_embedding_dim=int(config.get("wheel_embedding_dim", 8)),
            enable_target_encoder=False,
        ).to(device)
        ckpt_name = "best_state_supervised.pt" if (corrector_dir / "best_state_supervised.pt").exists() else "best_state_corrector.pt"
    else:
        model = ResidualTCN(
            input_dim=feature_dim(residual_history_len),
            wheel_embedding_dim=int(config.get("wheel_embedding_dim", 8)),
        ).to(device)
        ckpt_name = "best_delta_ar_corrector.pt" if predict_delta_residual else "best_autoregressive_corrector.pt"
    ckpt = torch.load(corrector_dir / ckpt_name, map_location=device)
    if state_estimator:
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        unexpected = [k for k in unexpected if not k.startswith("target_encoder.")]
        if missing or unexpected:
            raise RuntimeError(f"State corrector checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    else:
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    alpha = load_alpha(corrector_dir)

    if state_estimator:
        z = np.zeros((model.state_dim,), dtype=np.float32)
        parts: Dict[str, List[np.ndarray]] = {k: [] for k in ["case_name", "time", "wheel_id", "F_base", "R_hat", "F_corrected"]}
        last_key = None
        with torch.no_grad():
            for x_start, end in ds.index:
                key = (str(ds.arrays["case_name"][end]), int(ds.arrays["wheel_id"][end]))
                if key != last_key:
                    z[:] = 0.0
                    last_key = key
                x = torch.from_numpy(ds.make_model_input(x_start, end, np.zeros((residual_history_len, 3), dtype=np.float32))).unsqueeze(0).to(device)
                z_prev = torch.from_numpy(z[None, :]).to(device)
                wid = torch.tensor([int(ds.arrays["wheel_id"][end])], dtype=torch.long, device=device)
                z_hat, pred_scaled = model(x, z_prev, wid)
                r_hat = target_scaler.inverse_transform(pred_scaled.detach().cpu().numpy()).astype(np.float32)[0]
                z_new = z_hat.detach().cpu().numpy().astype(np.float32)[0]
                z = state_ema_beta * z + (1.0 - state_ema_beta) * z_new
                f_base = ds.arrays["F_base"][end].astype(np.float32, copy=False)
                f_corrected = f_base - alpha * r_hat
                parts["case_name"].append(np.asarray([str(ds.arrays["case_name"][end])], dtype=str))
                parts["time"].append(np.asarray([ds.arrays["time"][end]], dtype=np.float64))
                parts["wheel_id"].append(np.asarray([ds.arrays["wheel_id"][end]], dtype=np.int64))
                parts["F_base"].append(f_base[None, :])
                parts["R_hat"].append(r_hat[None, :])
                parts["F_corrected"].append(f_corrected[None, :])
        out_arrays = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
        save_npz(out_dir / f"residual_corrected_{args.split}.npz", out_arrays, {
            "proof_dir": str(proof_dir),
            "corrector_dir": str(corrector_dir),
            "split": args.split,
            "mode": "online_latent_state",
            "state_ema_beta": state_ema_beta,
            "uses_hf": False,
            "state_initialization": "zeros",
        })
        log("INFER", f"saved online corrected forces: {out_dir / f'residual_corrected_{args.split}.npz'}")
        return

    memory_rows = np.zeros_like(ds.arrays["R"], dtype=np.float32)
    parts: Dict[str, List[np.ndarray]] = {k: [] for k in ["case_name", "time", "wheel_id", "F_base", "R_hat", "Delta_R_hat", "F_corrected"]}
    with torch.no_grad():
        for x_start, end in ds.index:
            hist = memory_rows[end - residual_history_len:end]
            x = torch.from_numpy(ds.make_model_input(x_start, end, hist)).unsqueeze(0).to(device)
            wid = torch.tensor([int(ds.arrays["wheel_id"][end])], dtype=torch.long, device=device)
            pred_scaled = model(x, wid)
            if predict_delta_residual:
                delta_hat = target_scaler.inverse_transform(torch.clamp(pred_scaled, -5.0, 5.0).detach().cpu().numpy()).astype(np.float32)[0]
                prev_memory = memory_rows[end - 1]
                r_hat = ds.clamp_residual_physical(prev_memory + delta_hat)
                memory_rows[end] = ds.clamp_residual_physical(ema_beta * prev_memory + (1.0 - ema_beta) * r_hat)
            else:
                delta_hat = np.zeros((3,), dtype=np.float32)
                r_hat = target_scaler.inverse_transform(pred_scaled.detach().cpu().numpy()).astype(np.float32)[0]
                memory_rows[end] = r_hat
            f_base = ds.arrays["F_base"][end].astype(np.float32, copy=False)
            f_corrected = f_base - alpha * r_hat
            parts["case_name"].append(np.asarray([str(ds.arrays["case_name"][end])], dtype=str))
            parts["time"].append(np.asarray([ds.arrays["time"][end]], dtype=np.float64))
            parts["wheel_id"].append(np.asarray([ds.arrays["wheel_id"][end]], dtype=np.int64))
            parts["F_base"].append(f_base[None, :])
            parts["R_hat"].append(r_hat[None, :])
            parts["Delta_R_hat"].append(delta_hat[None, :])
            parts["F_corrected"].append(f_corrected[None, :])

    out_arrays = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
    save_npz(out_dir / f"residual_corrected_{args.split}.npz", out_arrays, {
        "proof_dir": str(proof_dir),
        "corrector_dir": str(corrector_dir),
        "split": args.split,
        "mode": "online_autoregressive",
        "predict_delta_residual": predict_delta_residual,
        "ema_beta": ema_beta,
        "uses_hf": False,
        "residual_buffer_initialization": "zeros",
    })
    log("INFER", f"saved online corrected forces: {out_dir / f'residual_corrected_{args.split}.npz'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proof_dir", required=True, type=str)
    parser.add_argument("--corrector_dir", required=True, type=str)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--history_len", type=int, default=None)
    parser.add_argument("--residual_history_len", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    run_online(parse_args())


if __name__ == "__main__":
    main()
