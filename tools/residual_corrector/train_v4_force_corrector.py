"""Train a feed-forward Force Corrector directly on a frozen V4 Student.

This module intentionally does not create latent archives and does not use the
legacy residual Teacher/Student/GRU rollout path.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from models.data_utils_v4 import WHEEL_IDS, get_group_dims, graph_temporal_collate_fn, prepare_datasets_and_scaler
from models.graph_temporal_hgt_compensation_v4 import GraphTemporalHGTCompensationModelV4
from tools.residual_corrector.v4_force_corrector import V4ForceCorrector, corrected_force_loss, freeze_v4_student
from train.train_graph_model_v4 import (
    compute_losses_v4,
    inverse_transform_tensor,
    load_compatible_state_dict,
    move_batch_to_device,
    suffix_indices,
)
from infer.infer_graph_model_v4 import build_model_from_checkpoint_v4


def _force_hf(batch, spec, scaler, device):
    values = []
    scales = []
    for wheel in WHEEL_IDS:
        raw = inverse_transform_tensor(batch[f"hf_wheel{wheel}_contact"], scaler, f"hf_wheel{wheel}_contact")
        # Contact columns are prefixed (e.g. ``wheel0_Fx``); use the same
        # suffix-based lookup as the V4 trainer rather than exact names.
        force_idx = suffix_indices(spec.target_groups.wheel_contact_cols[wheel], ["Fx", "Fy", "Fz"])
        if len(force_idx) != 3:
            raise ValueError(
                f"wheel{wheel} contact group has no complete Fx/Fy/Fz suffixes: "
                f"{spec.target_groups.wheel_contact_cols[wheel]}"
            )
        values.append(raw[:, 0, force_idx])
        scales.append([scaler.scalers[f"hf_wheel{wheel}_contact"].std_[j] for j in force_idx])
    return torch.stack(values, dim=1).to(device), torch.as_tensor(scales, dtype=torch.float32, device=device)


def _load_student(checkpoint: Path, spec, device: torch.device):
    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=True)
    student, _ = build_model_from_checkpoint_v4(checkpoint_data, spec, "student", device)
    return student, checkpoint_data


def _run_epoch(student, corrector, loader, spec, scaler, device, optimizer, args, train: bool, v4_loss_args):
    corrector.train(train)
    student.train(args.joint_finetune and train)
    total = 0.0
    batches = 0
    raw_parts, corrected_parts, hf_parts = [], [], []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        if args.joint_finetune:
            v4_out = student(batch, input_prefix="student", current_index=args.history_len)
        else:
            with torch.no_grad():
                v4_out = student(batch, input_prefix="student", current_index=args.history_len)
        z_force, force_raw = v4_out["z_force"], v4_out["force_raw"]
        force_hf, force_scale = _force_hf(batch, spec, scaler, device)
        residual_pred = corrector(z_force, force_raw)
        corrected_force = force_raw + residual_pred
        corr_loss = corrected_force_loss(force_raw, residual_pred, force_hf, force_scale)
        loss = corr_loss
        if args.joint_finetune:
            # Reuse the original V4 multi-task loss namespace from the
            # Student checkpoint arguments. Corrector loss is added only via
            # lambda_corr; no old corrector state/rollout losses are used.
            v4_losses = compute_losses_v4(batch, v4_out, spec, scaler, v4_loss_args)
            loss = v4_losses["total"] + args.lambda_corr * corr_loss
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(corrector.parameters(), args.grad_clip)
            if args.joint_finetune:
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.v4_grad_clip)
            optimizer.step()
        total += float(loss.detach()); batches += 1
        if not train:
            raw_parts.append(force_raw.detach().cpu().numpy())
            corrected_parts.append(corrected_force.detach().cpu().numpy())
            hf_parts.append(force_hf.detach().cpu().numpy())
    result = {"loss": total / max(batches, 1)}
    if not train and hf_parts:
        raw, corrected, target = np.concatenate(raw_parts), np.concatenate(corrected_parts), np.concatenate(hf_parts)
        raw_rmse = float(np.sqrt(np.mean((raw - target) ** 2)))
        corrected_rmse = float(np.sqrt(np.mean((corrected - target) ** 2)))
        result.update({"v4_raw_rmse": raw_rmse, "corrected_rmse": corrected_rmse, "improvement_pct": 100.0 * (raw_rmse - corrected_rmse) / max(raw_rmse, 1e-12)})
        for axis, index in zip(("Fx", "Fy", "Fz"), range(3)):
            true_residual = target[:, :, index] - raw[:, :, index]
            pred_residual = corrected[:, :, index] - raw[:, :, index]
            result[f"{axis}_v4_raw_rmse"] = float(np.sqrt(np.mean((raw[:, :, index] - target[:, :, index]) ** 2)))
            result[f"{axis}_corrected_rmse"] = float(np.sqrt(np.mean((corrected[:, :, index] - target[:, :, index]) ** 2)))
            result[f"{axis}_residual_corr"] = float(np.corrcoef(pred_residual.reshape(-1), true_residual.reshape(-1))[0, 1]) if np.std(pred_residual) > 1e-12 and np.std(true_residual) > 1e-12 else float("nan")
            result[f"{axis}_R2"] = float(1.0 - np.sum((pred_residual - true_residual) ** 2) / max(np.sum((true_residual - true_residual.mean()) ** 2), 1e-12))
            result[f"{axis}_std_pred"] = float(np.std(pred_residual))
            result[f"{axis}_std_true"] = float(np.std(true_residual))
            result[f"{axis}_std_ratio"] = float(np.std(pred_residual) / max(np.std(true_residual), 1e-12))
            sign_mask = np.abs(true_residual) > 1e-6
            result[f"{axis}_sign_accuracy"] = float(np.mean(np.sign(pred_residual[sign_mask]) == np.sign(true_residual[sign_mask]))) if np.any(sign_mask) else float("nan")
    return result


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.student_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    ckpt_args = checkpoint.get("args") or {}
    spec, scaler, _, _, _, train_ds, val_ds, test_ds = prepare_datasets_and_scaler(
        args.feature_dir, args.merged_csv, seq_len=args.history_len + 1,
        pred_horizon=0, pred_seq_len=1, seed=int(ckpt_args.get("seed", 42)),
        history_len=args.history_len, teacher_future_len=0,
        force_lpf_alpha=float(ckpt_args.get("force_lpf_alpha", .15)),
        sinkage_max=float(ckpt_args.get("sinkage_max", .08)),
    )
    student, _ = _load_student(checkpoint_path, spec, device)
    if not args.joint_finetune:
        freeze_v4_student(student)
    z_dim = int(student.force_fusion_dim if student.use_force_tcn else student.z_dim + student.node_hidden_dim + student.readout_dim)
    corrector = V4ForceCorrector(z_dim, args.corrector_hidden_dim, args.corrector_dropout).to(device)
    loader_kwargs = {"num_workers": args.num_workers, "collate_fn": graph_temporal_collate_fn, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    corrector_parameters = list(corrector.parameters())
    optimizer_groups = [{"params": corrector_parameters, "lr": args.corrector_lr, "group_name": "corrector"}]
    if args.joint_finetune:
        optimizer_groups.append({"params": [p for p in student.parameters() if p.requires_grad], "lr": args.v4_finetune_lr, "group_name": "v4_finetune"})
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=args.weight_decay)
    v4_loss_args = SimpleNamespace(**ckpt_args)
    history, best = [], math.inf
    for epoch in range(1, args.epochs + 1):
        train_stats = _run_epoch(student, corrector, train_loader, spec, scaler, device, optimizer, args, True, v4_loss_args)
        with torch.no_grad():
            val_stats = _run_epoch(student, corrector, val_loader, spec, scaler, device, optimizer, args, False, v4_loss_args)
        history.append({"epoch": epoch, "train": train_stats, "val": val_stats})
        val_score = val_stats.get("corrected_rmse", val_stats["loss"])
        if val_score < best:
            best = val_score
            torch.save({"model_state_dict": corrector.state_dict(), "student_state_dict": student.state_dict() if args.joint_finetune else None, "epoch": epoch, "best_val_loss": best, "student_checkpoint": str(checkpoint_path), "z_force_dim": z_dim, "residual_convention": "R_true=F_HF-F_v4_raw; F_corr=F_v4_raw+R_pred", "joint_finetune": bool(args.joint_finetune)}, Path(args.output_dir) / "best_force_corrector.pt")
        torch.save({"model_state_dict": corrector.state_dict(), "student_state_dict": student.state_dict() if args.joint_finetune else None, "epoch": epoch, "student_checkpoint": str(checkpoint_path), "z_force_dim": z_dim}, Path(args.output_dir) / "last_force_corrector.pt")
    with open(Path(args.output_dir) / "history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    best_payload = torch.load(Path(args.output_dir) / "best_force_corrector.pt", map_location=device, weights_only=True)
    corrector.load_state_dict(best_payload["model_state_dict"])
    with torch.no_grad():
        test_stats = _run_epoch(student, corrector, test_loader, spec, scaler, device, optimizer, args, False, v4_loss_args)
    with open(Path(args.output_dir) / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump({"best_val_score": best, "test": test_stats, "student_checkpoint": str(checkpoint_path), "residual_convention": "R_true=F_HF-F_v4_raw; F_corr=F_v4_raw+R_pred"}, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", required=True)
    parser.add_argument("--merged_csv", required=True)
    parser.add_argument("--student_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--history_len", type=int, default=9)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--corrector_hidden_dim", type=int, default=128)
    parser.add_argument("--corrector_dropout", type=float, default=.1)
    parser.add_argument("--corrector_lr", type=float, default=1e-3)
    parser.add_argument("--v4_finetune_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_corr", type=float, default=1.0)
    parser.add_argument("--joint_finetune", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--v4_grad_clip", type=float, default=5.0)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
