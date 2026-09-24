from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn

from .base.residual_dataset import base_feature_names, build_datasets
from .base.teacher_student_state import TeacherStudentResidualCorrector, hidden_distillation_loss
from .base.residual_utils import ensure_dir, finite_rmse, set_seed, write_json


def _case_sequences(dataset):
    grouped: Dict[str, Dict[float, Dict[int, tuple[int, int]]]] = {}
    for start, end in dataset.index:
        case, time, wheel = str(dataset.arrays["case_name"][end]), float(dataset.arrays["time"][end]), int(dataset.arrays["wheel_id"][end])
        grouped.setdefault(case, {}).setdefault(time, {})[wheel] = (start, end)
    return [
        [[wheels[i] for i in range(6)] for _, wheels in sorted(times.items()) if set(wheels) == set(range(6))]
        for _, times in grouped.items()
    ]


def _case_batches(sequences, batch_size: int):
    """Yield independent cases in bounded parallel batches."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(sequences), batch_size):
        yield sequences[start:start + batch_size]


def _archive_to_true(residual_archive: np.ndarray) -> np.ndarray:
    """Canonicalize legacy archive R_archive=F_base-F_HF to physical R_true."""
    return -np.asarray(residual_archive, dtype=np.float32)


def _batch_step(dataset, batch_sequences, step, device):
    active_indices = [i for i, sequence in enumerate(batch_sequences) if step < len(sequence)]
    active = [batch_sequences[i][step] for i in active_indices]
    if not active:
        return None
    xs = np.stack([[dataset.make_model_input(start, end, np.zeros((dataset.residual_history_len, 3), dtype=np.float32)) for start, end in row] for row in active])
    ends = np.asarray([[end for _, end in row] for row in active], dtype=np.int64)
    r_true = _archive_to_true(dataset.arrays["R"][ends])
    return torch.from_numpy(xs).to(device), torch.from_numpy(r_true.astype(np.float32)).to(device), ends, active_indices


def _previous_true_residual(dataset, ends: np.ndarray) -> np.ndarray:
    """Return R_true(t-1), or zero only for the actual first frame of a case."""
    previous = np.zeros((*ends.shape, 3), dtype=np.float32)
    for batch, row in np.ndindex(ends.shape):
        prev = int(ends[batch, row]) - 1
        now = int(ends[batch, row])
        if prev >= 0 and dataset.arrays["case_name"][prev] == dataset.arrays["case_name"][now] and dataset.arrays["wheel_id"][prev] == dataset.arrays["wheel_id"][now]:
            previous[batch, row] = _archive_to_true(dataset.arrays["R"][prev])
    return previous


def _force_loss(f_base, r_pred, f_hf, scale):
    return nn.functional.huber_loss((f_base + r_pred - f_hf) / scale.reshape(1, 1, 3), torch.zeros_like(f_hf), reduction="mean")


@torch.no_grad()
def rollout(model, dataset, device, scaler, teacher: bool = False, case_batch_size: int = 1):
    model.eval(); rows = []
    for case_batch in _case_batches(_case_sequences(dataset), case_batch_size):
        state = torch.zeros((len(case_batch), 6, model.wheel_state_dim), device=device)
        prev = torch.zeros((len(case_batch), 6, 3), device=device)
        for step in range(max(map(len, case_batch))):
            item = _batch_step(dataset, case_batch, step, device)
            if item is None:
                continue
            x, r_true, ends, active_indices = item
            active = torch.as_tensor(active_indices, dtype=torch.long, device=device)
            state_active = state.index_select(0, active)
            if teacher:
                prev_active = prev.index_select(0, active)
                if step == 0:
                    prev_active = torch.from_numpy(_previous_true_residual(dataset, ends)).to(device)
                next_active, r_pred = model.teacher_step_all(x[:, :, -1], state_active, prev_active)
                prev = prev.index_copy(0, active, r_true)
            else:
                next_active, r_pred = model.student_step_all(x[:, :, -1], state_active)
            state = state.index_copy(0, active, next_active)
            for local_batch, case_index in enumerate(active_indices):
                for wheel in range(6):
                    end = ends[local_batch, wheel]
                    rows.append((r_pred[local_batch, wheel].cpu().numpy(), r_true[local_batch, wheel].cpu().numpy(), dataset.arrays["F_base"][end], dataset.arrays["F_HF"][end], int(dataset.arrays["wheel_id"][end]), float(dataset.arrays["time"][end]), str(dataset.arrays["case_name"][end]), end))
    keys = ("R_pred", "R_true", "F_base", "F_HF", "wheel_id", "time", "case_name", "row_index")
    out = {k: [] for k in keys}
    for rp, rt, fb, fh, wid, time, case, end in rows:
        out["R_pred"].append(rp); out["R_true"].append(rt); out["F_base"].append(fb); out["F_HF"].append(fh); out["wheel_id"].append(wid); out["time"].append(time); out["case_name"].append(case); out["row_index"].append(end)
    for key in ("R_pred", "R_true", "F_base", "F_HF"): out[key] = np.asarray(out[key], dtype=np.float32)
    for key in ("wheel_id", "row_index"): out[key] = np.asarray(out[key], dtype=np.int64)
    out["time"] = np.asarray(out["time"], dtype=np.float32); out["case_name"] = np.asarray(out["case_name"], dtype=str)
    return out


def _metrics(pred):
    corr = pred["F_base"] + pred["R_pred"]
    base, corrected = finite_rmse(pred["F_base"], pred["F_HF"]), finite_rmse(corr, pred["F_HF"])
    out = {"base_overall_rmse": base, "corrected_overall_rmse": corrected, "overall_improvement_pct": 100 * (base - corrected) / max(base, 1e-12)}
    for j, axis in enumerate(("Fx", "Fy", "Fz")):
        p, t = pred["R_pred"][:, j], pred["R_true"][:, j]
        out[f"{axis}_corr"] = float(np.corrcoef(p, t)[0, 1]) if np.std(p) > 1e-12 and np.std(t) > 1e-12 else float("nan")
        out[f"{axis}_std_ratio"] = float(np.std(p) / max(np.std(t), 1e-12))
        out[f"base_{axis}_rmse"] = finite_rmse(pred["F_base"][:, j], pred["F_HF"][:, j]); out[f"corrected_{axis}_rmse"] = finite_rmse(corr[:, j], pred["F_HF"][:, j])
        out[f"{axis}_improvement_pct"] = 100 * (out[f"base_{axis}_rmse"] - out[f"corrected_{axis}_rmse"]) / max(out[f"base_{axis}_rmse"], 1e-12)
    return out


def run(args):
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")); out = ensure_dir(Path(args.output_dir) if args.output_dir else Path(args.proof_dir) / "corrector_teacher_student")
    train, val, test, scalers, meta = build_datasets(args.proof_dir, history_len=args.history_len, residual_history_len=args.residual_history_len, state_estimator=True)
    names = base_feature_names(); body_idx = [i for i, name in enumerate(names) if name.startswith("body_")]
    model = TeacherStudentResidualCorrector(len(names), body_idx, args.wheel_state_dim, args.wheel_embedding_dim).to(device)
    scale = torch.as_tensor(scalers.target_scaler.scale_, device=device, dtype=torch.float32).clamp_min(1e-6)
    wheel_names = [name for i, name in enumerate(names) if i not in body_idx]
    missing_optional = [key for key in ("body_vel", "body_yaw_rate", "wheel_omega", "wheel_vx_local", "wheel_vy_local", "slip_angle") if key not in train.arrays]
    config = {
        **vars(args), "mode": "teacher_student_six_wheel", "input_dim": len(names),
        "feature_names": names, "body_indices": body_idx,
        "student_input": {"body_context_features": [names[i] for i in body_idx], "wheel_features": wheel_names},
        "teacher_input": {"student_input": "same as student", "additional_causal_feature": "R_true(t-1) = -(R_archive(t-1)); zero only at case start"},
        "missing_optional_archive_fields": missing_optional,
        "residual_convention": "R_true=F_HF-F_base; F_corr=F_base+R_pred",
        "student_evaluation": "student_step_all(x, state) only; F_HF/R_true/teacher_state are not model inputs",
    }
    write_json(out / "config.json", config); scalers.save(out)
    teacher_path = Path(args.teacher_checkpoint) if args.teacher_checkpoint else out / "best_teacher.pt"
    stages = ["teacher", "student"] if args.train_stage == "all" else [args.train_stage]
    histories = {}
    for stage in stages:
        if stage == "student":
            ckpt = torch.load(teacher_path, map_location=device); model.load_state_dict(ckpt["model_state_dict"]); model.freeze_teacher()
            # Student decoder starts from Teacher decoder, then diverges.
            model.student_decoder.load_state_dict(model.teacher_decoder.state_dict())
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
        else: optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best, stale, history = float("inf"), 0, []
        for epoch in range(1, args.epochs + 1):
            rollout_len = 10 if epoch <= 3 else 20 if epoch <= 6 else 50; model.train()
            if stage == "student": model.freeze_teacher()
            losses=[]; distills=[]; cosines=[]; teacher_norms=[]; student_norms=[]
            for case_batch in _case_batches(_case_sequences(train), args.batch_size):
                batch_cases = len(case_batch)
                state_s = torch.zeros((batch_cases, 6, args.wheel_state_dim), device=device)
                state_t = torch.zeros_like(state_s)
                prev = torch.zeros((batch_cases, 6, 3), device=device)
                for offset in range(0, max(map(len, case_batch)), rollout_len):
                    segments = [sequence[offset:offset + rollout_len] for sequence in case_batch]
                    optimizer.zero_grad(); step_losses=[]; step_distill=[]
                    for step in range(rollout_len):
                        item = _batch_step(train, segments, step, device)
                        if item is None:
                            continue
                        x, r_true, ends, active_indices = item
                        active = torch.as_tensor(active_indices, dtype=torch.long, device=device)
                        x = x[:, :, -1]
                        base = torch.from_numpy(train.arrays["F_base"][ends]).to(device)
                        hf = torch.from_numpy(train.arrays["F_HF"][ends]).to(device)
                        state_t_active = state_t.index_select(0, active)
                        prev_active = prev.index_select(0, active)
                        if offset == 0 and step == 0:
                            prev_active = torch.from_numpy(_previous_true_residual(train, ends)).to(device)
                        if stage == "teacher":
                            next_t, rp = model.teacher_step_all(x, state_t_active, prev_active)
                            loss = _force_loss(base, rp, hf, scale)
                        else:
                            with torch.no_grad():
                                next_t, _ = model.teacher_step_all(x, state_t_active, prev_active)
                            state_s_active, rp = model.student_step_all(x, state_s.index_select(0, active))
                            d = hidden_distillation_loss(state_s_active, next_t)
                            loss = _force_loss(base, rp, hf, scale) + args.lambda_distill * d
                            step_distill.append(d.detach())
                            cosines.append(float(torch.nn.functional.cosine_similarity(torch.nn.functional.normalize(state_s_active, dim=-1), torch.nn.functional.normalize(next_t, dim=-1), dim=-1).mean().detach()))
                            state_s = state_s.index_copy(0, active, state_s_active)
                        state_t = state_t.index_copy(0, active, next_t)
                        prev = prev.index_copy(0, active, r_true)
                        if not torch.isfinite(loss) or not torch.isfinite(state_t).all() or (stage == "student" and not torch.isfinite(state_s).all()):
                            raise FloatingPointError(f"non-finite {stage} hidden state or loss at epoch={epoch}, case-batch segment={offset}")
                        step_losses.append(loss)
                    if not step_losses:
                        continue
                    loss = torch.stack(step_losses).mean(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                    losses.append(float(loss.detach())); distills.extend([float(value) for value in step_distill])
                    # Truncated BPTT: retain each case's causal state but cut
                    # the graph only at the rollout curriculum boundary.
                    state_s, state_t = state_s.detach(), state_t.detach()
                    teacher_norms.append(state_t.norm(dim=-1).mean(dim=0).cpu().numpy())
                    if stage == "student": student_norms.append(state_s.norm(dim=-1).mean(dim=0).cpu().numpy())
            pred = rollout(model, val, device, scalers.target_scaler, teacher=stage == "teacher", case_batch_size=args.batch_size); metrics = _metrics(pred)
            mean_loss, mean_distill = float(np.mean(losses)), float(np.mean(distills) if distills else 0.0)
            mean_teacher_norm = np.mean(teacher_norms, axis=0).tolist() if teacher_norms else [float("nan")] * 6
            mean_student_norm = np.mean(student_norms, axis=0).tolist() if student_norms else []
            history.append({"epoch": epoch, "rollout_len": rollout_len, "loss": mean_loss, "distill_loss": mean_distill, "hidden_cosine_similarity": float(np.mean(cosines) if cosines else float("nan")), "teacher_hidden_norm_by_wheel": mean_teacher_norm, "student_hidden_norm_by_wheel": mean_student_norm, **metrics})
            print(f"[{stage}] epoch={epoch} rollout={rollout_len} loss={mean_loss:.5f} distill={mean_distill:.5f} val={metrics['corrected_overall_rmse']:.3f} Fx/Fy/Fz_corr={metrics['Fx_corr']:.3f}/{metrics['Fy_corr']:.3f}/{metrics['Fz_corr']:.3f}", flush=True)
            print(f"[{stage}] teacher_hidden_norm={np.round(mean_teacher_norm, 4).tolist()} student_hidden_norm={np.round(mean_student_norm, 4).tolist() if mean_student_norm else 'n/a'} cosine={history[-1]['hidden_cosine_similarity']:.4f}", flush=True)
            path_best, path_last = out / f"best_{stage}.pt", out / f"last_{stage}.pt"; payload={"model_state_dict": model.state_dict(), "config": config, "epoch":epoch, "metrics":metrics}
            torch.save(payload, path_last)
            if metrics["corrected_overall_rmse"] < best - .05: best, stale = metrics["corrected_overall_rmse"], 0; torch.save(payload, path_best)
            else: stale += 1
            if stale >= args.patience: break
        histories[stage] = history
        write_json(out / f"train_history_{stage}.json", history)

    teacher_ckpt = torch.load(teacher_path if args.train_stage == "student" else out / "best_teacher.pt", map_location=device)
    teacher = TeacherStudentResidualCorrector(len(names), body_idx, args.wheel_state_dim, args.wheel_embedding_dim).to(device)
    teacher.load_state_dict(teacher_ckpt["model_state_dict"])
    teacher_val = rollout(teacher, val, device, scalers.target_scaler, True, args.batch_size)
    teacher_test = rollout(teacher, test, device, scalers.target_scaler, True, args.batch_size)
    results = {"teacher_val": _metrics(teacher_val), "teacher_test": _metrics(teacher_test)}
    if args.train_stage != "teacher":
        student_ckpt = torch.load(out / "best_student.pt", map_location=device)
        model.load_state_dict(student_ckpt["model_state_dict"])
        student_val = rollout(model, val, device, scalers.target_scaler, False, args.batch_size)
        student_test = rollout(model, test, device, scalers.target_scaler, False, args.batch_size)
        results.update({"student_val": _metrics(student_val), "student_test": _metrics(student_test)})
        # The existing diagnostic files remain the deployment (student-only) diagnostics.
        from .train_residual_corrector import write_residual_diagnostics
        alpha = {"alpha_x": 1.0, "alpha_y": 1.0, "alpha_z": 1.0}
        write_json(out / "alpha.json", {**alpha, "usage": "identity; retained for compatibility and alpha diagnostics"})
        write_residual_diagnostics(out, student_val, student_test, val, test, meta, alpha, pred_is_physical=True)
        write_residual_diagnostics(ensure_dir(out / "teacher_diagnostics"), teacher_val, teacher_test, val, test, meta, alpha, pred_is_physical=True)
        last_student_history = histories.get("student", [])
        results["distillation"] = {
            "lambda_distill": args.lambda_distill,
            "hidden_cosine_similarity": last_student_history[-1].get("hidden_cosine_similarity") if last_student_history else None,
            "hidden_smooth_l1": last_student_history[-1].get("distill_loss") if last_student_history else None,
        }
    write_json(out / "teacher_student_metrics.json", results)
    print("[Teacher]", json.dumps(results["teacher_test"], ensure_ascii=False), flush=True)
    if "student_test" in results:
        print("[Student]", json.dumps(results["student_test"], ensure_ascii=False), flush=True)
    print(json.dumps({"output_dir": str(out), "metrics": results}, indent=2), flush=True)
