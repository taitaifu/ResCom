from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from tools.residual_corrector.base.residual_dataset import (  # noqa: E402
    base_feature_dim,
    build_datasets,
    collate_residual,
    feature_dim,
)
from tools.residual_corrector.base.residual_state import StateResidualCorrector  # noqa: E402
from tools.residual_corrector.base.residual_tcn import ResidualTCN  # noqa: E402
from tools.residual_corrector.base.residual_utils import (  # noqa: E402
    DIR_LABELS,
    WHEEL_IDS,
    check_no_hf_leakage,
    ensure_dir,
    finite_rmse,
    force_metrics,
    log,
    plot_scatter,
    plot_series,
    set_seed,
    split_overlap_report,
    write_json,
)


def move_batch(batch: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def stage_for_epoch(
    epoch: int,
    total_epochs: int,
    teacher_epochs: int,
    scheduled_start_epoch: int,
    autoregressive_epochs: int,
) -> Tuple[str, float]:
    ar_start = max(scheduled_start_epoch, total_epochs - autoregressive_epochs + 1)
    if epoch <= teacher_epochs:
        return "teacher_forcing", 1.0
    if epoch >= ar_start:
        return "autoregressive", 0.0
    scheduled_start = max(scheduled_start_epoch, teacher_epochs + 1)
    scheduled_end = ar_start - 1
    progress = (epoch - scheduled_start) / max(1, scheduled_end - scheduled_start)
    ratio = 0.95 + (0.2 - 0.95) * float(np.clip(progress, 0.0, 1.0))
    return "scheduled_sampling", float(ratio)


def save_nonfinite_checkpoint(out_dir: Path, model, optimizer, epoch: int, batch_idx: int, teacher_ratio: float, loss, pred, target, residual_history, label: str = "pred") -> None:
    path = out_dir / "nonfinite_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "batch_idx": batch_idx,
            "teacher_ratio": teacher_ratio,
        },
        path,
    )
    print(
        "Non-finite loss detected: "
        f"epoch={epoch} batch={batch_idx} teacher_ratio={teacher_ratio:.4f} loss={float(loss.detach().cpu()) if torch.isfinite(loss.detach()).item() else loss.item()} "
        f"{label}_min={float(torch.nan_to_num(pred.detach()).min().cpu()):.6g} {label}_max={float(torch.nan_to_num(pred.detach()).max().cpu()):.6g} "
        f"target_min={float(torch.nan_to_num(target.detach()).min().cpu()):.6g} target_max={float(torch.nan_to_num(target.detach()).max().cpu()):.6g} "
        f"R_memory_min={float(torch.nan_to_num(residual_history.detach()).min().cpu()):.6g} "
        f"R_memory_max={float(torch.nan_to_num(residual_history.detach()).max().cpu()):.6g} "
        f"checkpoint={path}",
        flush=True,
    )
    raise SystemExit(1)


def save_training_checkpoint(path: Path, model, optimizer, config: Dict, epoch: int, metrics: Dict, history: List[Dict], best_ar: float, no_improve_count: int) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "epoch": epoch,
            "metrics": metrics,
            "history": history,
            "best_ar": best_ar,
            "no_improve_count": no_improve_count,
        },
        path,
    )


def residual_rmse(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    return {f"R{axis.lower()}_rmse": finite_rmse(pred[:, j], true[:, j]) for j, axis in enumerate(["x", "y", "z"])}


def delta_rmse(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    return {f"Delta_R{axis.lower()}_rmse": finite_rmse(pred[:, j], true[:, j]) for j, axis in enumerate(["x", "y", "z"])}


def apply_alpha(f_base: np.ndarray, r_pred: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return f_base - r_pred * alpha.reshape(1, 3)


def apply_alpha_add(f_base: np.ndarray, r_pred: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return f_base + r_pred * alpha.reshape(1, 3)


def calibrate_alpha(pred: Dict[str, np.ndarray]) -> Dict[str, float]:
    alphas = np.arange(0.0, 1.0001, 0.05)
    out = {}
    for j, axis in enumerate(["x", "y", "z"]):
        best_alpha = 0.0
        best_rmse = float("inf")
        for alpha in alphas:
            rmse = finite_rmse(pred["F_base"][:, j] - alpha * pred["R_pred"][:, j], pred["F_HF"][:, j])
            if rmse < best_rmse:
                best_alpha = float(alpha)
                best_rmse = rmse
        out[f"alpha_{axis}"] = best_alpha
    return out


def calibrate_alpha_add(pred: Dict[str, np.ndarray]) -> Dict[str, float]:
    alphas = np.arange(0.0, 1.0001, 0.05)
    out = {}
    for j, axis in enumerate(["x", "y", "z"]):
        best_alpha = 0.0
        best_rmse = float("inf")
        for alpha in alphas:
            rmse = finite_rmse(pred["F_base"][:, j] + alpha * pred["R_pred"][:, j], pred["F_HF"][:, j])
            if rmse < best_rmse:
                best_alpha = float(alpha)
                best_rmse = rmse
        out[f"alpha_{axis}"] = best_alpha
    return out


def eval_final(pred: Dict[str, np.ndarray], alpha_dict: Dict[str, float]) -> Dict[str, float]:
    alpha = np.asarray([alpha_dict["alpha_x"], alpha_dict["alpha_y"], alpha_dict["alpha_z"]], dtype=np.float32)
    corrected = apply_alpha(pred["F_base"], pred["R_pred"], alpha)
    out = force_metrics(pred["F_base"], pred["F_HF"], corrected)
    out.update(residual_rmse(pred["R_pred"], pred["R_true"]))
    if "Delta_R_pred" in pred and "Delta_R_true" in pred:
        out.update(delta_rmse(pred["Delta_R_pred"], pred["Delta_R_true"]))
    return out


def eval_final_add(pred: Dict[str, np.ndarray], alpha_dict: Dict[str, float]) -> Dict[str, float]:
    alpha = np.asarray([alpha_dict["alpha_x"], alpha_dict["alpha_y"], alpha_dict["alpha_z"]], dtype=np.float32)
    corrected = apply_alpha_add(pred["F_base"], pred["R_pred"], alpha)
    out = force_metrics(pred["F_base"], pred["F_HF"], corrected)
    out.update(residual_rmse(pred["R_pred"], pred["R_true"]))
    return out


def rollout_len_for_epoch(epoch: int) -> int:
    if epoch <= 3:
        return 10
    if epoch <= 6:
        return 20
    return 50


def target_inverse_torch(y_scaled: torch.Tensor, target_scaler) -> torch.Tensor:
    mean = torch.as_tensor(target_scaler.mean_, dtype=y_scaled.dtype, device=y_scaled.device)
    scale = torch.as_tensor(target_scaler.scale_, dtype=y_scaled.dtype, device=y_scaled.device)
    return y_scaled * scale + mean


def force_loss_components(f_corr: torch.Tensor, f_hf: torch.Tensor, force_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    err = (f_corr - f_hf) / force_scale.reshape(1, 3)
    per_dim = torch.nn.functional.huber_loss(err, torch.zeros_like(err), reduction="none").mean(dim=0)
    return per_dim.sum(), per_dim


# These diagnostics deliberately operate on the completed AR rollout only.  They
# are not fed back into optimisation, calibration, or the model inputs.
DIAG_EPS = 1e-12
SIGN_EPS = 1e-6
AXES = ("Fx", "Fy", "Fz")


def _finite_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) <= DIAG_EPS or np.std(y) <= DIAG_EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _residual_metric_row(r_pred: np.ndarray, r_true: np.ndarray, f_base: np.ndarray, f_corr: np.ndarray, f_hf: np.ndarray) -> Dict[str, float]:
    valid = np.isfinite(r_pred) & np.isfinite(r_true) & np.isfinite(f_base) & np.isfinite(f_corr) & np.isfinite(f_hf)
    p, t, base, corr, hf = r_pred[valid].astype(np.float64), r_true[valid].astype(np.float64), f_base[valid].astype(np.float64), f_corr[valid].astype(np.float64), f_hf[valid].astype(np.float64)
    out: Dict[str, float] = {"sample_count": int(len(t)), "corr": _finite_corr(p, t)}
    if not len(t):
        return {**out, **{k: float("nan") for k in ["R2", "std_pred", "std_true", "std_ratio", "slope", "bias", "sign_acc", "base_rmse", "corrected_rmse", "improvement_pct"]}}
    ss_tot = float(np.sum((t - np.mean(t)) ** 2))
    out["R2"] = float(1.0 - np.sum((t - p) ** 2) / ss_tot) if ss_tot > DIAG_EPS else float("nan")
    out["std_pred"] = float(np.std(p))
    out["std_true"] = float(np.std(t))
    out["std_ratio"] = float(out["std_pred"] / (out["std_true"] + DIAG_EPS))
    if len(t) >= 2 and np.std(p) > DIAG_EPS:
        slope, bias = np.polyfit(p, t, 1)
        out["slope"], out["bias"] = float(slope), float(bias)
    else:
        out["slope"], out["bias"] = float("nan"), float("nan")
    sign_mask = np.abs(t) > SIGN_EPS
    out["sign_acc"] = float(np.mean(np.sign(p[sign_mask]) == np.sign(t[sign_mask]))) if np.any(sign_mask) else float("nan")
    out["base_rmse"] = finite_rmse(base, hf)
    out["corrected_rmse"] = finite_rmse(corr, hf)
    out["improvement_pct"] = float(100.0 * (out["base_rmse"] - out["corrected_rmse"]) / out["base_rmse"]) if out["base_rmse"] > DIAG_EPS else float("nan")
    return out


def _terrain_for_cases(case_names: np.ndarray, arrays: Mapping[str, np.ndarray], meta: Mapping[str, Any]) -> np.ndarray:
    """Prefer explicit terrain metadata; fall back to terrain words in case names."""
    names = np.asarray(case_names).astype(str)
    if "terrain" in arrays and len(arrays["terrain"]) == len(arrays["case_name"]):
        by_case = {str(c): str(t) for c, t in zip(arrays["case_name"], arrays["terrain"])}
        return np.asarray([by_case.get(c, "unknown") for c in names], dtype=str)
    for key in ("terrain_by_case", "case_terrain"):
        mapping = meta.get(key)
        if isinstance(mapping, Mapping):
            return np.asarray([str(mapping.get(c, "unknown")) for c in names], dtype=str)
    # The residual archives currently retain case_name but not a terrain column.
    # Only emit labels which actually occur; unmatched cases remain "unknown".
    known = ("flat", "rough", "slope")
    return np.asarray([next((label for label in known if label in c.lower()), "unknown") for c in names], dtype=str)


def _sample_diagnostic_frame(split: str, pred: Dict[str, np.ndarray], alpha: np.ndarray, dataset, meta: Mapping[str, Any], pred_is_physical: bool = False) -> pd.DataFrame:
    # Archives/model use R_internal = F_base - F_HF and subtract it at the
    # force output.  Export the requested physical convention instead:
    # R_true/R_pred = F_HF - F_base, while retaining the actual F_corr.
    f_corr = apply_alpha_add(pred["F_base"], pred["R_pred"], alpha) if pred_is_physical else apply_alpha(pred["F_base"], pred["R_pred"], alpha)
    terrain = _terrain_for_cases(pred["case_name"], dataset.arrays, meta)
    data: Dict[str, Any] = {"split": split, "case_name": pred["case_name"], "terrain": terrain, "wheel_id": pred["wheel_id"], "time": pred["time"], "row_index": pred["row_index"]}
    for j, axis in enumerate(("x", "y", "z")):
        r_pred = pred["R_pred"][:, j] if pred_is_physical else -pred["R_pred"][:, j]
        data.update({f"R_true_{axis}": pred["F_HF"][:, j] - pred["F_base"][:, j], f"R_pred_{axis}": r_pred, f"F_base_{axis}": pred["F_base"][:, j], f"F_corr_{axis}": f_corr[:, j], f"F_HF_{axis}": pred["F_HF"][:, j]})
    return pd.DataFrame(data)


def _metrics_from_frame(frame: pd.DataFrame, mask: np.ndarray, axis_idx: int) -> Dict[str, float]:
    axis = ("x", "y", "z")[axis_idx]
    s = frame.loc[mask]
    return _residual_metric_row(s[f"R_pred_{axis}"].to_numpy(), s[f"R_true_{axis}"].to_numpy(), s[f"F_base_{axis}"].to_numpy(), s[f"F_corr_{axis}"].to_numpy(), s[f"F_HF_{axis}"].to_numpy())


def _fy_lag_diagnostics(split: str, pred: Dict[str, np.ndarray], dataset) -> tuple[List[Dict[str, Any]], List[str]]:
    # Alias lists make this future-compatible while reporting every unavailable
    # requested feature explicitly rather than silently substituting a proxy.
    candidates = {
        "body_vy": [("body_vel", 1), ("body_velocity", 1), ("vy", None)],
        "yaw_rate": [("body_yaw_rate", None), ("yaw_rate", None)],
        "steering_angle": [("steering_angle", None), ("steer", None)],
        "wheel_vx_local": [("wheel_vx_local", None)], "wheel_vy_local": [("wheel_vy_local", None)],
        "slip_angle": [("slip_angle", None)], "longitudinal_slip": [("slip_long", None)],
        "wheel_omega": [("wheel_omega", None), ("wheel_angular_velocity", None)],
        "contact": [("in_contact", None)], "sinkage": [("sinkage", None)],
    }
    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    row_indices = pred["row_index"].astype(np.int64)
    groups = pd.DataFrame({"case_name": pred["case_name"].astype(str), "wheel_id": pred["wheel_id"], "time": pred["time"], "row_index": row_indices, "r_true": pred["F_HF"][:, 1] - pred["F_base"][:, 1]}).sort_values(["case_name", "wheel_id", "time", "row_index"]).groupby(["case_name", "wheel_id"], sort=False)
    for feature, aliases in candidates.items():
        source = next(((key, component) for key, component in aliases if key in dataset.arrays), None)
        if source is None:
            skipped.append(feature)
            continue
        key, component = source
        raw = np.asarray(dataset.arrays[key])
        feature_values = raw if raw.ndim == 1 or component is None else raw[:, component]
        corrs = []
        for lag in range(21):
            xs, ys = [], []
            for _, group in groups:
                idx, target = group["row_index"].to_numpy(dtype=np.int64), group["r_true"].to_numpy(dtype=np.float64)
                if len(idx) > lag:
                    xs.append(feature_values[idx[:-lag or None]] if lag else feature_values[idx])
                    ys.append(target[lag:] if lag else target)
            corr = _finite_corr(np.concatenate(xs), np.concatenate(ys)) if xs else float("nan")
            corrs.append(corr)
        finite = np.isfinite(corrs)
        best = int(np.nanargmax(np.abs(corrs))) if np.any(finite) else 0
        for lag, corr in enumerate(corrs):
            rows.append({"split": split, "feature": feature, "source": key, "lag": lag, "corr": corr, "corr_at_lag0": corrs[0], "best_lag": best, "best_corr": corrs[best], "best_abs_corr": abs(corrs[best]) if np.isfinite(corrs[best]) else float("nan")})
    return rows, skipped


def write_residual_diagnostics(out_dir: Path, val_pred: Dict[str, np.ndarray], test_pred: Dict[str, np.ndarray], val_ds, test_ds, meta: Mapping[str, Any], alpha_dict: Dict[str, float], pred_is_physical: bool = False) -> Dict[str, Any]:
    alpha = np.asarray([alpha_dict["alpha_x"], alpha_dict["alpha_y"], alpha_dict["alpha_z"]], dtype=np.float32)
    frames = {"val": _sample_diagnostic_frame("val", val_pred, alpha, val_ds, meta, pred_is_physical), "test": _sample_diagnostic_frame("test", test_pred, alpha, test_ds, meta, pred_is_physical)}
    for split, frame in frames.items():
        frame.to_csv(out_dir / f"residual_diagnostics_{split}.csv", index=False)
    terrain_rows, quantile_rows = [], []
    summary: Dict[str, Any] = {"eps_sign": SIGN_EPS, "splits": {}, "labels": {}}
    for split, frame in frames.items():
        overall = {}
        for j, label in enumerate(AXES):
            metric = _metrics_from_frame(frame, np.ones(len(frame), dtype=bool), j)
            overall[label] = metric
            for terrain in sorted(frame["terrain"].dropna().unique()):
                terrain_rows.append({"split": split, "terrain": terrain, "axis": label, **_metrics_from_frame(frame, (frame["terrain"] == terrain).to_numpy(), j)})
        summary["splits"][split] = overall
        fy_abs = np.abs(frame["R_true_y"].to_numpy(dtype=float))
        q = np.quantile(fy_abs[np.isfinite(fy_abs)], [0.0, .5, .7, .9, 1.0]) if np.any(np.isfinite(fy_abs)) else np.full(5, np.nan)
        for i, name in enumerate(("0_50", "50_70", "70_90", "90_100")):
            mask = (fy_abs >= q[i]) & ((fy_abs <= q[i + 1]) if i == 3 else (fy_abs < q[i + 1]))
            quantile_rows.append({"split": split, "quantile_bin": name, "abs_R_true_min": q[i], "abs_R_true_max": q[i + 1], **_metrics_from_frame(frame, mask, 1)})
    pd.DataFrame(terrain_rows).to_csv(out_dir / "residual_diagnostics_by_terrain.csv", index=False)
    pd.DataFrame(quantile_rows).to_csv(out_dir / "fy_quantile_diagnostics.csv", index=False)
    lag_rows, skipped = [], {}
    for split, pred, ds in (("val", val_pred, val_ds), ("test", test_pred, test_ds)):
        rows, absent = _fy_lag_diagnostics(split, pred, ds)
        lag_rows.extend(rows); skipped[split] = absent
        for name in absent:
            print(f"[Fy lag correlation] {split}: {name} skipped (feature unavailable)", flush=True)
    pd.DataFrame(lag_rows).to_csv(out_dir / "fy_lag_correlation.csv", index=False)
    summary["lag_skipped_features"] = skipped
    for split, metrics in summary["splits"].items():
        for axis, metric in metrics.items():
            labels = []
            if abs(metric["corr"]) > .5 and metric["std_ratio"] < .5: labels.append("trend_learned_amplitude_shrunk")
            if abs(metric["corr"]) < .2: labels.append("weak_residual_learnability")
            summary["labels"].setdefault(split, {})[axis] = {"labels": labels, "metrics": metric}
        fy_q = [r for r in quantile_rows if r["split"] == split]
        high = next((r for r in fy_q if r["quantile_bin"] == "90_100"), {})
        if np.isfinite(high.get("corr", np.nan)) and high["corr"] - metrics["Fy"]["corr"] >= .15:
            summary["labels"][split]["Fy"]["labels"].append("high_residual_more_learnable")
            summary["labels"][split]["Fy"]["high_residual_metrics"] = high
    for row in lag_rows:
        if row["lag"] == 0 and row["best_abs_corr"] - abs(row["corr_at_lag0"]) >= .15:
            summary.setdefault("history_dependency", {}).setdefault(row["split"], []).append({"feature": row["feature"], "corr_at_lag0": row["corr_at_lag0"], "best_lag": row["best_lag"], "best_corr": row["best_corr"], "best_abs_corr": row["best_abs_corr"], "label": "history_dependency"})
    write_json(out_dir / "residual_diagnostics_summary.json", summary)
    return {"summary": summary, "quantiles": quantile_rows, "lags": lag_rows}


def teacher_forcing_predict(model, loader, target_scaler, device, desc: str = "Validation TF", predict_delta_residual: bool = False) -> Dict[str, np.ndarray]:
    model.eval()
    parts: Dict[str, List] = {k: [] for k in ["R_pred", "R_true", "F_base", "F_HF", "wheel_id", "time", "case_name", "row_index", "Delta_R_pred", "Delta_R_true"]}
    losses = []
    loss_fn = nn.HuberLoss(reduction="mean")
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=True, dynamic_ncols=True, file=sys.stdout):
            batch = move_batch(batch, device)
            pred_scaled = model(batch["x"], batch["wheel_id"])
            losses.append(float(loss_fn(pred_scaled, batch["y"]).detach().cpu()))
            pred_raw = target_scaler.inverse_transform(pred_scaled.detach().cpu().numpy()).astype(np.float32)
            target_raw = batch["y_raw"].detach().cpu().numpy().astype(np.float32)
            if predict_delta_residual:
                prev_memory = batch["residual_history_raw"][:, -1, :].detach().cpu().numpy().astype(np.float32)
                parts["Delta_R_pred"].append(pred_raw)
                parts["Delta_R_true"].append(target_raw)
                parts["R_pred"].append(prev_memory + pred_raw)
                parts["R_true"].append(batch["R_true"].detach().cpu().numpy().astype(np.float32))
            else:
                parts["R_pred"].append(pred_raw)
                parts["R_true"].append(target_raw)
            parts["F_base"].append(batch["F_base"].detach().cpu().numpy().astype(np.float32))
            parts["F_HF"].append(batch["F_HF"].detach().cpu().numpy().astype(np.float32))
            parts["wheel_id"].append(batch["wheel_id"].detach().cpu().numpy())
            parts["time"].append(batch["time"].detach().cpu().numpy())
            parts["row_index"].append(batch["row_index"].detach().cpu().numpy())
            parts["case_name"].extend(batch["case_name"])
    return pack_prediction(parts, losses)


def pack_prediction(parts: Dict[str, List], losses: List[float] | None = None) -> Dict[str, np.ndarray]:
    out = {}
    for key in ["R_pred", "R_true", "F_base", "F_HF"]:
        out[key] = np.concatenate(parts[key], axis=0) if parts[key] else np.empty((0, 3), dtype=np.float32)
    for key in ["Delta_R_pred", "Delta_R_true"]:
        if key in parts:
            out[key] = np.concatenate(parts[key], axis=0) if parts[key] else np.empty((0, 3), dtype=np.float32)
    for key in ["wheel_id", "time", "row_index"]:
        out[key] = np.concatenate(parts[key], axis=0) if parts[key] else np.empty((0,))
    out["case_name"] = np.asarray(parts["case_name"], dtype=str)
    out["loss"] = np.asarray(losses or [], dtype=np.float64)
    return out


def _rollout_streams(dataset) -> List[List[Tuple[int, int]]]:
    streams: List[List[Tuple[int, int]]] = []
    current: List[Tuple[int, int]] = []
    current_key = None
    cases = dataset.arrays["case_name"].astype(str)
    wheels = dataset.arrays["wheel_id"].astype(np.int64)
    for x_start, end in dataset.index:
        key = (str(cases[end]), int(wheels[end]))
        if current_key is None:
            current_key = key
        if key != current_key:
            streams.append(current)
            current = []
            current_key = key
        current.append((x_start, end))
    if current:
        streams.append(current)
    return streams


def autoregressive_rollout(
    model,
    dataset,
    device,
    target_scaler,
    loss: bool = False,
    desc: str = "Validation AR",
    rollout_batch_size: int = 4096,
    predict_delta_residual: bool = False,
    ema_beta: float = 0.8,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    model.eval()
    memory_rows = np.zeros_like(dataset.arrays["R"], dtype=np.float32)
    parts: Dict[str, List] = {k: [] for k in ["R_pred", "R_true", "F_base", "F_HF", "wheel_id", "time", "case_name", "row_index", "Delta_R_pred", "Delta_R_true"]}
    losses = []
    loss_fn = nn.HuberLoss(reduction="mean")
    streams = _rollout_streams(dataset)
    max_len = max((len(s) for s in streams), default=0)
    total = sum(len(s) for s in streams)
    pbar = tqdm(total=total, desc=desc, leave=True, dynamic_ncols=True, file=sys.stdout)
    with torch.no_grad():
        try:
            for pos in range(max_len):
                active = [stream[pos] for stream in streams if pos < len(stream)]
                for chunk_start in range(0, len(active), rollout_batch_size):
                    chunk = active[chunk_start:chunk_start + rollout_batch_size]
                    ends = np.asarray([end for _, end in chunk], dtype=np.int64)
                    xs = np.stack([
                        dataset.make_model_input(x_start, end, memory_rows[end - dataset.residual_history_len:end])
                        for x_start, end in chunk
                    ], axis=0)
                    wids = torch.from_numpy(dataset.arrays["wheel_id"][ends].astype(np.int64)).to(device)
                    x_tensor = torch.from_numpy(xs).to(device)
                    pred_scaled = model(x_tensor, wids)
                    if predict_delta_residual:
                        feedback_scaled = torch.clamp(pred_scaled, -5.0, 5.0)
                        delta_feedback = target_scaler.inverse_transform(feedback_scaled.detach().cpu().numpy()).astype(np.float32)
                        delta_pred = target_scaler.inverse_transform(pred_scaled.detach().cpu().numpy()).astype(np.float32)
                        prev_memory = memory_rows[ends - 1]
                        r_pred = dataset.clamp_residual_physical(prev_memory + delta_feedback)
                        memory = dataset.clamp_residual_physical(float(ema_beta) * prev_memory + (1.0 - float(ema_beta)) * r_pred)
                        memory_rows[ends] = memory
                    else:
                        delta_pred = np.empty((len(ends), 3), dtype=np.float32)
                        r_pred = target_scaler.inverse_transform(pred_scaled.detach().cpu().numpy()).astype(np.float32)
                        memory_rows[ends] = r_pred
                    if loss:
                        y_np = np.stack([dataset.target_scaled(int(end)) for end in ends], axis=0)
                        y = torch.from_numpy(y_np).to(device)
                        losses.append(float(loss_fn(pred_scaled, y).detach().cpu()))
                    parts["R_pred"].append(r_pred.astype(np.float32, copy=False))
                    parts["R_true"].append(dataset.arrays["R"][ends].astype(np.float32, copy=False))
                    if predict_delta_residual:
                        parts["Delta_R_pred"].append(delta_pred.astype(np.float32, copy=False))
                        parts["Delta_R_true"].append(np.stack([dataset.target_raw(int(end)) for end in ends], axis=0).astype(np.float32, copy=False))
                    parts["F_base"].append(dataset.arrays["F_base"][ends].astype(np.float32, copy=False))
                    parts["F_HF"].append(dataset.arrays["F_HF"][ends].astype(np.float32, copy=False))
                    parts["wheel_id"].append(dataset.arrays["wheel_id"][ends].astype(np.int64, copy=False))
                    parts["time"].append(dataset.arrays["time"][ends].astype(np.float32, copy=False))
                    parts["row_index"].append(ends)
                    parts["case_name"].extend(dataset.arrays["case_name"][ends].astype(str).tolist())
                    pbar.update(len(chunk))
        finally:
            pbar.close()
    return pack_prediction(parts, losses), memory_rows


def per_wheel_metrics(split: str, pred: Dict[str, np.ndarray], alpha_dict: Dict[str, float]) -> List[Dict]:
    alpha = np.asarray([alpha_dict["alpha_x"], alpha_dict["alpha_y"], alpha_dict["alpha_z"]], dtype=np.float32)
    corrected = apply_alpha(pred["F_base"], pred["R_pred"], alpha)
    rows = []
    for wid in WHEEL_IDS:
        mask = pred["wheel_id"] == wid
        if np.any(mask):
            rows.append({"split": split, "wheel_id": wid, **force_metrics(pred["F_base"][mask], pred["F_HF"][mask], corrected[mask])})
    return rows


def per_wheel_metrics_add(split: str, pred: Dict[str, np.ndarray], alpha_dict: Dict[str, float]) -> List[Dict]:
    alpha = np.asarray([alpha_dict["alpha_x"], alpha_dict["alpha_y"], alpha_dict["alpha_z"]], dtype=np.float32)
    corrected = apply_alpha_add(pred["F_base"], pred["R_pred"], alpha)
    rows = []
    for wid in WHEEL_IDS:
        mask = pred["wheel_id"] == wid
        if np.any(mask):
            rows.append({"split": split, "wheel_id": wid, **force_metrics(pred["F_base"][mask], pred["F_HF"][mask], corrected[mask])})
    return rows


def make_plots(out_dir: Path, pred: Dict[str, np.ndarray], alpha_dict: Dict[str, float]) -> None:
    fig_dir = ensure_dir(out_dir / "plots")
    alpha = np.asarray([alpha_dict["alpha_x"], alpha_dict["alpha_y"], alpha_dict["alpha_z"]], dtype=np.float32)
    corrected = apply_alpha(pred["F_base"], pred["R_pred"], alpha)
    for j, label in enumerate(DIR_LABELS):
        plot_scatter(fig_dir / f"{label}_scatter.png", pred["R_true"][:, j], pred["R_pred"][:, j], f"{label} residual rollout", "R true / N", "R pred / N")
    idx = np.arange(min(1200, len(pred["time"])))
    if idx.size:
        plot_series(fig_dir / "residual_rollout.png", pred["time"][idx], {"R_true_Fz": pred["R_true"][idx, 2], "R_pred_Fz": pred["R_pred"][idx, 2]}, "Fz residual rollout", "residual / N")
        plot_series(fig_dir / "force_compare.png", pred["time"][idx], {"F_base": pred["F_base"][idx, 2], "F_corrected": corrected[idx, 2], "F_HF": pred["F_HF"][idx, 2]}, "Force compare Fz", "force / N")
        plot_series(fig_dir / "fz_compare.png", pred["time"][idx], {"Base Fz": pred["F_base"][idx, 2], "Corrected Fz": corrected[idx, 2], "HF Fz": pred["F_HF"][idx, 2]}, "Fz compare", "force / N")


def make_plots_add(out_dir: Path, pred: Dict[str, np.ndarray], alpha_dict: Dict[str, float]) -> None:
    fig_dir = ensure_dir(out_dir / "plots")
    alpha = np.asarray([alpha_dict["alpha_x"], alpha_dict["alpha_y"], alpha_dict["alpha_z"]], dtype=np.float32)
    corrected = apply_alpha_add(pred["F_base"], pred["R_pred"], alpha)
    for j, label in enumerate(DIR_LABELS):
        plot_scatter(fig_dir / f"{label}_scatter.png", pred["R_true"][:, j], pred["R_pred"][:, j], f"{label} residual rollout", "R true / N", "R pred / N")
    idx = np.arange(min(1200, len(pred["time"])))
    if idx.size:
        plot_series(fig_dir / "residual_rollout.png", pred["time"][idx], {"R_true_Fz": pred["R_true"][idx, 2], "R_pred_Fz": pred["R_pred"][idx, 2]}, "Fz residual rollout", "residual / N")
        plot_series(fig_dir / "force_compare.png", pred["time"][idx], {"F_base": pred["F_base"][idx, 2], "F_corrected": corrected[idx, 2], "F_HF": pred["F_HF"][idx, 2]}, "Force compare Fz", "force / N")
        plot_series(fig_dir / "fz_compare.png", pred["time"][idx], {"Base Fz": pred["F_base"][idx, 2], "Corrected Fz": corrected[idx, 2], "HF Fz": pred["F_HF"][idx, 2]}, "Fz compare", "force / N")


def state_rollout(
    model: StateResidualCorrector,
    dataset,
    device,
    target_scaler,
    loss: bool = False,
    desc: str = "State AR",
    rollout_batch_size: int = 4096,
    state_ema_beta: float = 0.8,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    model.eval()
    streams = _rollout_streams(dataset)
    max_len = max((len(s) for s in streams), default=0)
    total = sum(len(s) for s in streams)
    stream_state = np.zeros((len(streams), model.state_dim), dtype=np.float32)
    state_rows = np.zeros((len(dataset.arrays["R"]), model.state_dim), dtype=np.float32)
    parts: Dict[str, List] = {k: [] for k in ["R_pred", "R_true", "F_base", "F_HF", "wheel_id", "time", "case_name", "row_index"]}
    losses = []
    loss_fn = nn.HuberLoss(reduction="mean")
    pbar = tqdm(total=total, desc=desc, leave=True, dynamic_ncols=True, file=sys.stdout)
    with torch.no_grad():
        try:
            for pos in range(max_len):
                active = [(si, stream[pos]) for si, stream in enumerate(streams) if pos < len(stream)]
                for chunk_start in range(0, len(active), rollout_batch_size):
                    chunk = active[chunk_start:chunk_start + rollout_batch_size]
                    stream_ids = np.asarray([si for si, _ in chunk], dtype=np.int64)
                    pairs = [pair for _, pair in chunk]
                    ends = np.asarray([end for _, end in pairs], dtype=np.int64)
                    xs = np.stack([dataset.make_model_input(x_start, end, np.zeros((dataset.residual_history_len, 3), dtype=np.float32)) for x_start, end in pairs], axis=0)
                    x_tensor = torch.from_numpy(xs).to(device)
                    z_prev = torch.from_numpy(stream_state[stream_ids]).to(device)
                    wids = torch.from_numpy(dataset.arrays["wheel_id"][ends].astype(np.int64)).to(device)
                    z_hat, residual_scaled = model(x_tensor, z_prev, wids)
                    residual = -target_scaler.inverse_transform(residual_scaled.detach().cpu().numpy()).astype(np.float32)
                    z_np = z_hat.detach().cpu().numpy().astype(np.float32)
                    z_mem = float(state_ema_beta) * stream_state[stream_ids] + (1.0 - float(state_ema_beta)) * z_np
                    stream_state[stream_ids] = z_mem
                    state_rows[ends] = z_mem
                    if loss:
                        y_np = np.stack([dataset.target_scaled(int(end)) for end in ends], axis=0)
                        y = torch.from_numpy(y_np).to(device)
                        losses.append(float(loss_fn(residual_scaled, y).detach().cpu()))
                    parts["R_pred"].append(residual)
                    parts["R_true"].append((-dataset.arrays["R"][ends]).astype(np.float32, copy=False))
                    parts["F_base"].append(dataset.arrays["F_base"][ends].astype(np.float32, copy=False))
                    parts["F_HF"].append(dataset.arrays["F_HF"][ends].astype(np.float32, copy=False))
                    parts["wheel_id"].append(dataset.arrays["wheel_id"][ends].astype(np.int64, copy=False))
                    parts["time"].append(dataset.arrays["time"][ends].astype(np.float32, copy=False))
                    parts["row_index"].append(ends)
                    parts["case_name"].extend(dataset.arrays["case_name"][ends].astype(str).tolist())
                    pbar.update(len(chunk))
        finally:
            pbar.close()
    return pack_prediction(parts, losses), state_rows


def train_state(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    proof_dir = Path(args.proof_dir)
    out_dir = ensure_dir(Path(args.output_dir) if args.output_dir else proof_dir / "corrector_state_ar_force")
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    train_ds, val_ds, test_ds, scalers, meta = build_datasets(
        proof_dir,
        history_len=args.history_len,
        residual_history_len=args.residual_history_len,
        oracle_residual_history=False,
        predict_delta_residual=False,
        state_estimator=True,
    )
    leaks = check_no_hf_leakage(scalers.feature_names, allow_oracle=False)
    if leaks:
        raise RuntimeError("HF leakage detected in state estimator inputs: " + ", ".join(leaks))
    overlap = split_overlap_report(proof_dir)
    if any(overlap.values()):
        raise RuntimeError(f"Split leakage detected: {overlap}")

    model = StateResidualCorrector(
        input_dim=base_feature_dim(),
        state_dim=args.state_dim,
        wheel_embedding_dim=args.wheel_embedding_dim,
        enable_target_encoder=False,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.HuberLoss(reduction="mean")
    force_scale = torch.as_tensor(scalers.target_scaler.scale_, dtype=torch.float32, device=device)
    force_scale = torch.clamp(force_scale, min=1e-6)
    config = vars(args).copy()
    config.update({
        "mode": "latent_state_ar_force",
        "input_dim": base_feature_dim(),
        "feature_names": scalers.feature_names,
        "residual_convention": "R_pred = F_HF - F_base; F_corrected = F_base + alpha * R_pred",
        "loss": "Huber(F_base + R_pred, F_HF), normalized by train residual std, L = L_Fx + L_Fy + L_Fz",
        "lambda_state": 0.0,
        "V4_F_base_source": meta.get("F_base_source"),
    })
    write_json(out_dir / "config.json", config)
    scalers.save(out_dir)
    history = []
    best_ar = float("inf")
    no_improve_count = 0
    min_delta = 0.05
    config["early_stopping_metric"] = "val AR corrected overall RMSE"
    config["early_stopping_min_delta"] = min_delta
    config["early_stopping_patience"] = int(args.patience)
    write_json(out_dir / "config.json", config)
    streams = _rollout_streams(train_ds)

    for epoch in range(1, args.epochs + 1):
        stage = "autoregressive"
        teacher_ratio = 0.0
        rollout_len = rollout_len_for_epoch(epoch)
        assert stage == "autoregressive"
        assert teacher_ratio == 0.0
        assert rollout_len in {10, 20, 50}
        segments = []
        for stream in streams:
            for start in range(0, len(stream), rollout_len):
                segment = stream[start:start + rollout_len]
                if segment:
                    segments.append(segment)
        rng = np.random.default_rng(args.seed + epoch)
        rng.shuffle(segments)
        model.train()
        train_losses = []
        train_dir_losses = []
        residual_losses = []
        latent_norms = []
        pred_sum = np.zeros(3, dtype=np.float64)
        pred_sq_sum = np.zeros(3, dtype=np.float64)
        true_sum = np.zeros(3, dtype=np.float64)
        true_sq_sum = np.zeros(3, dtype=np.float64)
        residual_count = 0
        progress = tqdm(range(0, len(segments), args.batch_size), desc=f"Epoch {epoch}/{args.epochs}", leave=True, dynamic_ncols=True, file=sys.stdout)
        for batch_idx, seg_start in enumerate(progress, start=1):
            seg_batch = segments[seg_start:seg_start + args.batch_size]
            if not seg_batch:
                continue
            optimizer.zero_grad(set_to_none=True)
            z_prev = torch.zeros((len(seg_batch), args.state_dim), dtype=torch.float32, device=device)
            step_losses = []
            step_dir_losses = []
            step_residual_losses = []
            step_norms = []
            for step in range(rollout_len):
                active = [(i, segment[step]) for i, segment in enumerate(seg_batch) if step < len(segment)]
                if not active:
                    continue
                rows = np.asarray([end for _, (_, end) in active], dtype=np.int64)
                xs = np.stack([
                    train_ds.make_model_input(x_start, end, np.zeros((train_ds.residual_history_len, 3), dtype=np.float32))
                    for _, (x_start, end) in active
                ], axis=0)
                active_idx = torch.as_tensor([i for i, _ in active], dtype=torch.long, device=device)
                x_tensor = torch.from_numpy(xs).to(device)
                wids = torch.from_numpy(train_ds.arrays["wheel_id"][rows].astype(np.int64)).to(device)
                z_in = z_prev.index_select(0, active_idx)
                z_hat, pred_scaled = model(x_tensor, z_in, wids)
                if not torch.isfinite(z_hat).all():
                    bad = int(rows[0])
                    raise RuntimeError(f"Non-finite latent state: epoch={epoch} batch={batch_idx} step={step} case={train_ds.arrays['case_name'][bad]} time={train_ds.arrays['time'][bad]}")
                residual_old = target_inverse_torch(pred_scaled, scalers.target_scaler)
                residual_correction = -residual_old
                if not torch.isfinite(residual_correction).all():
                    mask = ~torch.isfinite(residual_correction)
                    bad_i = int(torch.nonzero(mask, as_tuple=False)[0, 0].detach().cpu())
                    bad_d = int(torch.nonzero(mask, as_tuple=False)[0, 1].detach().cpu())
                    bad = int(rows[bad_i])
                    raise RuntimeError(
                        f"Non-finite R_pred: epoch={epoch} batch={batch_idx} step={step} direction={DIR_LABELS[bad_d]} "
                        f"case={train_ds.arrays['case_name'][bad]} time={train_ds.arrays['time'][bad]}"
                    )
                f_base = torch.from_numpy(train_ds.arrays["F_base"][rows].astype(np.float32, copy=False)).to(device)
                f_hf = torch.from_numpy(train_ds.arrays["F_HF"][rows].astype(np.float32, copy=False)).to(device)
                f_corr = f_base + residual_correction
                force_loss, dir_loss = force_loss_components(f_corr, f_hf, force_scale)
                if not torch.isfinite(dir_loss).all():
                    bad_d = int(torch.nonzero(~torch.isfinite(dir_loss), as_tuple=False)[0, 0].detach().cpu())
                    bad = int(rows[0])
                    raise RuntimeError(
                        f"Non-finite force loss: epoch={epoch} batch={batch_idx} step={step} direction={DIR_LABELS[bad_d]} "
                        f"case={train_ds.arrays['case_name'][bad]} time={train_ds.arrays['time'][bad]}"
                    )
                y_np = np.stack([train_ds.target_scaled(int(end)) for end in rows], axis=0)
                y = torch.from_numpy(y_np).to(device)
                residual_diag_loss = loss_fn(pred_scaled, y)
                z_mem = float(args.state_ema_beta) * z_in + (1.0 - float(args.state_ema_beta)) * z_hat
                z_prev = z_prev.index_copy(0, active_idx, z_mem)
                step_losses.append(force_loss)
                step_dir_losses.append(dir_loss.detach())
                step_residual_losses.append(residual_diag_loss.detach())
                step_norms.append(torch.linalg.norm(z_hat.detach(), dim=-1).mean())
                pred_np = residual_correction.detach().cpu().numpy().astype(np.float64)
                true_np = (f_hf - f_base).detach().cpu().numpy().astype(np.float64)
                pred_sum += pred_np.sum(axis=0)
                pred_sq_sum += np.square(pred_np).sum(axis=0)
                true_sum += true_np.sum(axis=0)
                true_sq_sum += np.square(true_np).sum(axis=0)
                residual_count += pred_np.shape[0]
            if not step_losses:
                continue
            force_loss = torch.stack(step_losses).mean()
            dir_loss = torch.stack(step_dir_losses).mean(dim=0)
            loss = force_loss
            assert loss is force_loss
            if not torch.isfinite(loss):
                first = seg_batch[0][0][1]
                print(
                    f"Non-finite force loss: epoch={epoch} batch={batch_idx} case={train_ds.arrays['case_name'][first]} "
                    f"time={train_ds.arrays['time'][first]} rollout_len={rollout_len}",
                    flush=True,
                )
                save_nonfinite_checkpoint(out_dir, model, optimizer, epoch, batch_idx, teacher_ratio, loss, force_loss.reshape(1), torch.zeros_like(force_loss).reshape(1), z_prev, label="force_loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            train_losses.append(loss_value)
            train_dir_losses.append(dir_loss.cpu().numpy().astype(np.float64))
            residual_losses.append(float(torch.stack(step_residual_losses).mean().cpu()))
            latent_norms.append(float(torch.stack(step_norms).mean().cpu()))
            progress.set_postfix(loss=f"{loss_value:.4f}", Lx=f"{train_dir_losses[-1][0]:.4f}", Ly=f"{train_dir_losses[-1][1]:.4f}", Lz=f"{train_dir_losses[-1][2]:.4f}", z=f"{latent_norms[-1]:.3f}", rollout_len=rollout_len)

        ar_pred, _ = state_rollout(model, val_ds, device, scalers.target_scaler, loss=True, desc=f"Epoch {epoch}/{args.epochs} val state AR", rollout_batch_size=args.rollout_batch_size, state_ema_beta=args.state_ema_beta)
        alpha_ar = calibrate_alpha_add(ar_pred)
        ar_metrics = eval_final_add(ar_pred, alpha_ar)
        dir_losses = np.mean(np.stack(train_dir_losses, axis=0), axis=0) if train_dir_losses else np.full(3, np.nan)
        pred_std = np.sqrt(np.maximum(pred_sq_sum / max(1, residual_count) - np.square(pred_sum / max(1, residual_count)), 0.0))
        true_std = np.sqrt(np.maximum(true_sq_sum / max(1, residual_count) - np.square(true_sum / max(1, residual_count)), 0.0))
        row = {
            "epoch": epoch,
            "stage": stage,
            "teacher_ratio": teacher_ratio,
            "rollout_len": rollout_len,
            "train_loss": float(np.mean(train_losses)),
            "force_loss": float(np.mean(train_losses)),
            "L_Fx": float(dir_losses[0]),
            "L_Fy": float(dir_losses[1]),
            "L_Fz": float(dir_losses[2]),
            "residual_loss": float(np.mean(residual_losses)),
            "latent_state_norm": float(np.mean(latent_norms)),
            "std_R_pred_x": float(pred_std[0]),
            "std_R_pred_y": float(pred_std[1]),
            "std_R_pred_z": float(pred_std[2]),
            "std_R_true_x": float(true_std[0]),
            "std_R_true_y": float(true_std[1]),
            "std_R_true_z": float(true_std[2]),
            "AR_rmse": ar_metrics["corrected_overall_rmse"],
            "AR_Fz_rmse": ar_metrics["corrected_Fz_rmse"],
            "ar_base_overall_rmse": ar_metrics["base_overall_rmse"],
            "ar_corrected_overall_rmse": ar_metrics["corrected_overall_rmse"],
            "ar_base_Fx_rmse": ar_metrics["base_Fx_rmse"],
            "ar_corrected_Fx_rmse": ar_metrics["corrected_Fx_rmse"],
            "ar_base_Fy_rmse": ar_metrics["base_Fy_rmse"],
            "ar_corrected_Fy_rmse": ar_metrics["corrected_Fy_rmse"],
            "ar_base_Fz_rmse": ar_metrics["base_Fz_rmse"],
            "ar_corrected_Fz_rmse": ar_metrics["corrected_Fz_rmse"],
            "overall_improvement_pct": ar_metrics["overall_improvement_pct"],
            "Fx_improvement_pct": ar_metrics["Fx_improvement_pct"],
            "Fy_improvement_pct": ar_metrics["Fy_improvement_pct"],
            "Fz_improvement_pct": ar_metrics["Fz_improvement_pct"],
            "early_stop_count": no_improve_count,
        }
        history.append(row)
        improved = ar_metrics["corrected_overall_rmse"] < (best_ar - min_delta)
        if improved:
            best_ar = ar_metrics["corrected_overall_rmse"]
            no_improve_count = 0
            save_training_checkpoint(out_dir / "best_ar.pt", model, optimizer, config, epoch, ar_metrics, history, best_ar, no_improve_count)
        else:
            no_improve_count += 1
        row["early_stop_count"] = no_improve_count
        save_training_checkpoint(out_dir / "last.pt", model, optimizer, config, epoch, ar_metrics, history, best_ar, no_improve_count)
        if epoch % 5 == 0:
            save_training_checkpoint(out_dir / f"epoch_{epoch:03d}.pt", model, optimizer, config, epoch, ar_metrics, history, best_ar, no_improve_count)
        print(
            f"epoch={epoch:03d} stage={stage} teacher_ratio={teacher_ratio:.3f} rollout_len={rollout_len} "
            f"train_total_loss={row['train_loss']:.6f} L_Fx/L_Fy/L_Fz={row['L_Fx']:.6f}/{row['L_Fy']:.6f}/{row['L_Fz']:.6f} "
            f"R_diag_loss={row['residual_loss']:.6f} ||z||={row['latent_state_norm']:.3f} "
            f"std_R_pred={row['std_R_pred_x']:.3f}/{row['std_R_pred_y']:.3f}/{row['std_R_pred_z']:.3f} "
            f"std_R_true={row['std_R_true_x']:.3f}/{row['std_R_true_y']:.3f}/{row['std_R_true_z']:.3f} "
            f"AR base overall={ar_metrics['base_overall_rmse']:.3f} AR corrected overall={ar_metrics['corrected_overall_rmse']:.3f} "
            f"AR base/corrected Fx={ar_metrics['base_Fx_rmse']:.3f}/{ar_metrics['corrected_Fx_rmse']:.3f} "
            f"Fy={ar_metrics['base_Fy_rmse']:.3f}/{ar_metrics['corrected_Fy_rmse']:.3f} "
            f"Fz={ar_metrics['base_Fz_rmse']:.3f}/{ar_metrics['corrected_Fz_rmse']:.3f} "
            f"improve overall/Fx/Fy/Fz={ar_metrics['overall_improvement_pct']:.2f}/{ar_metrics['Fx_improvement_pct']:.2f}/{ar_metrics['Fy_improvement_pct']:.2f}/{ar_metrics['Fz_improvement_pct']:.2f}% "
            f"early_stop_count={no_improve_count}/{args.patience}",
            flush=True,
        )
        if no_improve_count >= args.patience:
            break

    pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)
    ckpt = torch.load(out_dir / "best_ar.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    val_pred, _ = state_rollout(model, val_ds, device, scalers.target_scaler, loss=True, desc="Final val state AR", rollout_batch_size=args.rollout_batch_size, state_ema_beta=args.state_ema_beta)
    alpha = calibrate_alpha_add(val_pred)
    write_json(out_dir / "alpha.json", alpha)
    test_pred, _ = state_rollout(model, test_ds, device, scalers.target_scaler, loss=True, desc="Final test state AR", rollout_batch_size=args.rollout_batch_size, state_ema_beta=args.state_ema_beta)
    metrics = {"val": eval_final_add(val_pred, alpha), "test": eval_final_add(test_pred, alpha)}
    write_json(out_dir / "metrics.json", metrics)
    pd.DataFrame(per_wheel_metrics_add("test_state_ar", test_pred, alpha)).to_csv(out_dir / "per_wheel_metrics.csv", index=False)
    make_plots_add(out_dir, test_pred, alpha)
    diagnostics = write_residual_diagnostics(out_dir, val_pred, test_pred, val_ds, test_ds, meta, alpha, pred_is_physical=True)
    print("[Residual Diagnostic]", flush=True)
    for axis, m in diagnostics["summary"]["splits"]["test"].items():
        print(f"{axis}: corr={m['corr']:.4f} R2={m['R2']:.4f} std_pred={m['std_pred']:.4f} std_true={m['std_true']:.4f} std_ratio={m['std_ratio']:.4f} slope={m['slope']:.4f} sign_acc={m['sign_acc']:.4f} improvement={m['improvement_pct']:.2f}%", flush=True)
    state_q = {r["quantile_bin"]: r for r in diagnostics["quantiles"] if r["split"] == "test"}
    state_lags = [r for r in diagnostics["lags"] if r["split"] == "test" and r["lag"] == 0]
    strongest = max(state_lags, key=lambda r: r["best_abs_corr"] if np.isfinite(r["best_abs_corr"]) else -np.inf, default=None)
    print("[Fy Diagnostic]", flush=True)
    print(f"low residual corr={state_q.get('0_50', {}).get('corr', float('nan')):.4f} high residual corr={state_q.get('90_100', {}).get('corr', float('nan')):.4f} high residual std_ratio={state_q.get('90_100', {}).get('std_ratio', float('nan')):.4f}", flush=True)
    print("strongest history feature=none (all requested candidates unavailable)" if strongest is None else f"strongest history feature={strongest['feature']} lag0 corr={strongest['corr_at_lag0']:.4f} best lag={strongest['best_lag']} best lag corr={strongest['best_corr']:.4f}", flush=True)
    print(json.dumps({"output_dir": str(out_dir), "alpha": alpha, "metrics": metrics}, ensure_ascii=False, indent=2), flush=True)


def train(args: argparse.Namespace) -> None:
    # The wheel-wise Teacher→Student implementation is the default.  Keep the
    # previous state model available as an explicit baseline, rather than
    # changing its checkpoint format or behaviour.
    if args.corrector_arch == "teacher_student":
        from tools.residual_corrector.train_teacher_student import run as train_teacher_student
        train_teacher_student(args)
        return
    if args.corrector_arch == "legacy_tcn":
        args.state_estimator = False
    if args.state_estimator:
        train_state(args)
        return
    set_seed(args.seed)
    proof_dir = Path(args.proof_dir)
    oracle = bool(args.oracle_residual_history)
    if args.output_dir:
        out_dir = Path(args.output_dir)
    elif oracle:
        out_dir = proof_dir / "corrector" / "oracle"
    elif args.predict_delta_residual:
        out_dir = proof_dir / "corrector_delta_ar"
    else:
        out_dir = proof_dir / "corrector_ar"
    out_dir = ensure_dir(out_dir)
    if oracle:
        log("TRAIN", "ORACLE ONLY - NOT DEPLOYABLE")
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    train_ds, val_ds, test_ds, scalers, meta = build_datasets(
        proof_dir,
        history_len=args.history_len,
        residual_history_len=args.residual_history_len,
        oracle_residual_history=oracle,
        predict_delta_residual=args.predict_delta_residual,
    )
    leaks = check_no_hf_leakage(scalers.feature_names, allow_oracle=oracle)
    if leaks:
        raise RuntimeError("HF leakage detected in formal inputs: " + ", ".join(leaks))
    overlap = split_overlap_report(proof_dir)
    if any(overlap.values()):
        raise RuntimeError(f"Split leakage detected: {overlap}")

    train_ds.set_scheduled_sampling_mode(args.scheduled_sampling_mode)
    val_ds.set_scheduled_sampling_mode(args.scheduled_sampling_mode)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_residual)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_residual)
    model = ResidualTCN(input_dim=feature_dim(args.residual_history_len), wheel_embedding_dim=args.wheel_embedding_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.HuberLoss(reduction="mean")
    history = []
    best_tf = float("inf")
    best_ar = float("inf")
    no_improve_count = 0
    early_stop_start_epoch = int(args.scheduled_start_epoch) + 5

    config = vars(args).copy()
    config.update({
        "input_dim": feature_dim(args.residual_history_len),
        "feature_names": scalers.feature_names,
        "history_len": train_ds.history_len,
        "residual_history_len": train_ds.residual_history_len,
        "predict_delta_residual": bool(args.predict_delta_residual),
        "ema_beta": float(args.ema_beta),
        "oracle_warning": "ORACLE ONLY - NOT DEPLOYABLE" if oracle else "",
        "V4_F_base_source": meta.get("F_base_source"),
        "R_true_formula": meta.get("R_true_formula", "R = F_base - F_HF"),
    })
    write_json(out_dir / "config.json", config)
    scalers.save(out_dir)

    for epoch in range(1, args.epochs + 1):
        stage, teacher_ratio = stage_for_epoch(
            epoch,
            total_epochs=args.epochs,
            teacher_epochs=args.teacher_epochs,
            scheduled_start_epoch=args.scheduled_start_epoch,
            autoregressive_epochs=args.autoregressive_epochs,
        )
        if oracle:
            stage, teacher_ratio = "oracle_teacher_forcing", 1.0
        if stage in {"scheduled_sampling", "autoregressive"} and not oracle:
            _, train_pred_rows = autoregressive_rollout(
                model,
                train_ds,
                device,
                scalers.target_scaler,
                desc=f"Epoch {epoch}/{args.epochs} train AR history",
                rollout_batch_size=args.rollout_batch_size,
                predict_delta_residual=args.predict_delta_residual,
                ema_beta=args.ema_beta,
            )
            train_ds.set_predicted_residuals(train_pred_rows, teacher_ratio=teacher_ratio)
            train_ds.set_sample_mode("scheduled" if teacher_ratio > 0.0 else "autoregressive", teacher_ratio)
        else:
            train_ds.set_predicted_residuals(None, teacher_ratio=1.0)
            train_ds.set_sample_mode("teacher", 1.0)

        model.train()
        train_losses = []
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs}",
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        for batch_idx, batch in enumerate(progress, start=1):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch["x"], batch["wheel_id"])
            loss = loss_fn(pred, batch["y"])
            if not torch.isfinite(loss):
                save_nonfinite_checkpoint(out_dir, model, optimizer, epoch, batch_idx, teacher_ratio, loss, pred, batch["y"], batch["residual_history_raw"], label="delta_pred" if args.predict_delta_residual else "pred")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            train_losses.append(loss_value)
            progress.set_postfix(loss=f"{loss_value:.4f}", ratio=f"{teacher_ratio:.2f}", beta=f"{args.ema_beta:.2f}")

        val_ds.set_sample_mode("teacher", 1.0)
        tf_pred = teacher_forcing_predict(model, val_loader, scalers.target_scaler, device, desc=f"Epoch {epoch}/{args.epochs} val TF", predict_delta_residual=args.predict_delta_residual)
        ar_pred, _ = autoregressive_rollout(
            model,
            val_ds,
            device,
            scalers.target_scaler,
            loss=True,
            desc=f"Epoch {epoch}/{args.epochs} val AR",
            rollout_batch_size=args.rollout_batch_size,
            predict_delta_residual=args.predict_delta_residual,
            ema_beta=args.ema_beta,
        )
        alpha_tf = calibrate_alpha(tf_pred)
        alpha_ar = calibrate_alpha(ar_pred)
        tf_metrics = eval_final(tf_pred, alpha_tf)
        ar_metrics = eval_final(ar_pred, alpha_ar)
        if epoch == args.scheduled_start_epoch:
            best_ar = float("inf")
            no_improve_count = 0
        tf_loss = float(np.mean(tf_pred["loss"])) if tf_pred["loss"].size else float("nan")
        ar_loss = float(np.mean(ar_pred["loss"])) if ar_pred["loss"].size else float("nan")
        row = {
            "epoch": epoch,
            "stage": stage,
            "teacher_ratio": teacher_ratio,
            "train_loss": float(np.mean(train_losses)),
            "TF_rmse": tf_metrics["corrected_overall_rmse"],
            "AR_rmse": ar_metrics["corrected_overall_rmse"],
            "AR_Fz_rmse": ar_metrics["corrected_Fz_rmse"],
            "Delta_R_rmse": float(np.sqrt(np.nanmean(np.square(ar_pred.get("Delta_R_pred", np.zeros((0, 3))) - ar_pred.get("Delta_R_true", np.zeros((0, 3))))))) if args.predict_delta_residual and len(ar_pred.get("Delta_R_pred", [])) else float("nan"),
            "Recovered_R_rmse": ar_metrics.get("Rz_rmse", float("nan")),
            "teacher_forcing_val_loss": tf_loss,
            "autoregressive_val_loss": ar_loss,
            "tf_corrected_overall_rmse": tf_metrics["corrected_overall_rmse"],
            "ar_corrected_overall_rmse": ar_metrics["corrected_overall_rmse"],
            "ar_base_overall_rmse": ar_metrics["base_overall_rmse"],
            "ar_base_Fz_rmse": ar_metrics["base_Fz_rmse"],
            "ar_corrected_Fz_rmse": ar_metrics["corrected_Fz_rmse"],
            "early_stop_count": no_improve_count,
        }
        history.append(row)
        ckpt_name = "best_delta_ar_corrector.pt" if args.predict_delta_residual else "best_autoregressive_corrector.pt"
        improved = ar_metrics["corrected_overall_rmse"] < best_ar
        if improved:
            best_ar = ar_metrics["corrected_overall_rmse"]
            no_improve_count = 0
            save_training_checkpoint(out_dir / ckpt_name, model, optimizer, config, epoch, ar_metrics, history, best_ar, no_improve_count)
        elif epoch >= early_stop_start_epoch:
            no_improve_count += 1
        row["early_stop_count"] = no_improve_count
        if epoch == args.teacher_epochs:
            save_training_checkpoint(out_dir / f"epoch_{epoch:03d}.pt", model, optimizer, config, epoch, ar_metrics, history, best_ar, no_improve_count)
        print(
            f"epoch={epoch:03d} stage={stage} teacher_ratio={teacher_ratio:.3f} train_loss={row['train_loss']:.6f} "
            f"TF corrected overall/Fz={tf_metrics['corrected_overall_rmse']:.3f}/{tf_metrics['corrected_Fz_rmse']:.3f} "
            f"AR base overall={ar_metrics['base_overall_rmse']:.3f} AR corrected overall={ar_metrics['corrected_overall_rmse']:.3f} "
            f"AR base Fx/Fy/Fz={ar_metrics['base_Fx_rmse']:.3f}/{ar_metrics['base_Fy_rmse']:.3f}/{ar_metrics['base_Fz_rmse']:.3f} "
            f"AR corrected Fx/Fy/Fz={ar_metrics['corrected_Fx_rmse']:.3f}/{ar_metrics['corrected_Fy_rmse']:.3f}/{ar_metrics['corrected_Fz_rmse']:.3f} "
            f"Delta_R_rmse={row['Delta_R_rmse']:.3f} early_stop_count={no_improve_count}/{args.patience}",
            flush=True,
        )
        if tf_metrics["corrected_overall_rmse"] < best_tf:
            best_tf = tf_metrics["corrected_overall_rmse"]
            torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": epoch, "metrics": tf_metrics}, out_dir / "best_teacher_forcing_corrector.pt")
        if epoch >= early_stop_start_epoch and no_improve_count >= args.patience:
            break

    pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)
    ckpt_name = "best_delta_ar_corrector.pt" if args.predict_delta_residual else "best_autoregressive_corrector.pt"
    ckpt = torch.load(out_dir / ckpt_name, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    val_ar_pred, _ = autoregressive_rollout(model, val_ds, device, scalers.target_scaler, loss=True, desc="Final val AR", rollout_batch_size=args.rollout_batch_size, predict_delta_residual=args.predict_delta_residual, ema_beta=args.ema_beta)
    alpha = calibrate_alpha(val_ar_pred)
    write_json(out_dir / "alpha.json", alpha)
    base_metrics = force_metrics(val_ar_pred["F_base"], val_ar_pred["F_HF"])
    tf_pred = teacher_forcing_predict(model, val_loader, scalers.target_scaler, device, desc="Final val TF", predict_delta_residual=args.predict_delta_residual)
    tf_metrics = eval_final(tf_pred, alpha)
    ar_metrics = eval_final(val_ar_pred, alpha)
    test_pred, _ = autoregressive_rollout(model, test_ds, device, scalers.target_scaler, loss=True, desc="Final test AR", rollout_batch_size=args.rollout_batch_size, predict_delta_residual=args.predict_delta_residual, ema_beta=args.ema_beta)
    test_metrics = eval_final(test_pred, alpha)
    write_json(out_dir / "base_metrics.json", base_metrics)
    write_json(out_dir / "teacher_forcing_metrics.json", tf_metrics)
    write_json(out_dir / "autoregressive_metrics.json", {"val": ar_metrics, "test": test_metrics})
    write_json(out_dir / "metrics.json", {"base": base_metrics, "teacher_forcing": tf_metrics, "autoregressive": {"val": ar_metrics, "test": test_metrics}})
    pd.DataFrame(per_wheel_metrics("val_ar", val_ar_pred, alpha) + per_wheel_metrics("test_ar", test_pred, alpha)).to_csv(out_dir / "per_wheel_metrics.csv", index=False)
    make_plots(out_dir, test_pred, alpha)
    diagnostics = write_residual_diagnostics(out_dir, val_ar_pred, test_pred, val_ds, test_ds, meta, alpha)
    test_diag = diagnostics["summary"]["splits"]["test"]
    print("[Residual Diagnostic]", flush=True)
    for axis in AXES:
        m = test_diag[axis]
        print(
            f"{axis}: corr={m['corr']:.4f} R2={m['R2']:.4f} std_pred={m['std_pred']:.4f} "
            f"std_true={m['std_true']:.4f} std_ratio={m['std_ratio']:.4f} slope={m['slope']:.4f} "
            f"sign_acc={m['sign_acc']:.4f} improvement={m['improvement_pct']:.2f}%",
            flush=True,
        )
    test_q = {r["quantile_bin"]: r for r in diagnostics["quantiles"] if r["split"] == "test"}
    test_lag = [r for r in diagnostics["lags"] if r["split"] == "test" and r["lag"] == 0]
    strongest = max(test_lag, key=lambda r: r["best_abs_corr"] if np.isfinite(r["best_abs_corr"]) else -np.inf, default=None)
    low, high = test_q.get("0_50", {}), test_q.get("90_100", {})
    print("[Fy Diagnostic]", flush=True)
    print(
        f"low residual corr={low.get('corr', float('nan')):.4f} high residual corr={high.get('corr', float('nan')):.4f} "
        f"high residual std_ratio={high.get('std_ratio', float('nan')):.4f}",
        flush=True,
    )
    if strongest is None:
        print("strongest history feature=none (all requested candidates unavailable)", flush=True)
    else:
        print(
            f"strongest history feature={strongest['feature']} lag0 corr={strongest['corr_at_lag0']:.4f} "
            f"best lag={strongest['best_lag']} best lag corr={strongest['best_corr']:.4f}",
            flush=True,
        )
    print(json.dumps({"output_dir": str(out_dir), "alpha": alpha, "test_autoregressive": test_metrics}, ensure_ascii=False, indent=2), flush=True)
    if test_metrics["overall_improvement_pct"] < 3.0:
        print("Residual corrector does not provide a meaningful improvement.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proof_dir", required=True, type=str)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--oracle_residual_history", action="store_true")
    parser.add_argument("--history_len", type=int, default=10)
    parser.add_argument("--residual_history_len", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--teacher_epochs", type=int, default=20)
    parser.add_argument("--scheduled_start_epoch", type=int, default=21)
    parser.add_argument("--autoregressive_epochs", type=int, default=10)
    parser.add_argument("--scheduled_sampling_mode", choices=["progressive", "random"], default="progressive")
    parser.add_argument("--progressive_sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--rollout_batch_size", type=int, default=4096)
    parser.add_argument("--state_estimator", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--corrector_arch", choices=["teacher_student", "legacy_state", "legacy_tcn"], default="teacher_student")
    parser.add_argument("--train_stage", choices=["teacher", "student", "all"], default="all")
    parser.add_argument("--teacher_checkpoint", type=str, default=None)
    parser.add_argument("--wheel_state_dim", type=int, default=32)
    parser.add_argument("--lambda_distill", type=float, default=0.1)
    parser.add_argument("--state_dim", type=int, default=64)
    parser.add_argument("--state_ema_beta", type=float, default=0.8)
    parser.add_argument("--lambda_state", type=float, default=0.1)
    parser.add_argument("--predict_delta_residual", action="store_true")
    parser.add_argument("--ema_beta", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--wheel_embedding_dim", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.progressive_sampling:
        args.scheduled_sampling_mode = "random"
    return args


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
