"""Offline Fy residual-history feature experiment; does not alter the corrector."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from tools.residual_corrector.base.residual_dataset import StandardScaler, load_residual_arrays
from tools.residual_corrector.base.residual_utils import ensure_dir, set_seed, write_json


HISTORY_ORDER = ("j_x", "j_y", "contact_duration", "sinkage_ema", "sinkage_max_since_contact")


def build_history_features(arrays: Dict[str, np.ndarray], beta: float = .9) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Causally integrate deployable wheel features inside each case/wheel only."""
    n = len(arrays["time"])
    available, unavailable = {}, []
    has_contact, has_sinkage = "in_contact" in arrays, "sinkage" in arrays
    has_jy = "wheel_vy_local" in arrays
    has_jx = "wheel_vx_slip" in arrays or ("wheel_vx_local" in arrays and "slip_long" in arrays)
    for name, ok in (("j_y", has_jy), ("j_x", has_jx), ("contact_duration", has_contact), ("sinkage_ema", has_contact and has_sinkage), ("sinkage_max_since_contact", has_contact and has_sinkage)):
        if ok:
            available[name] = np.zeros(n, dtype=np.float32)
        else:
            unavailable.append(name)
    if not available:
        return available, unavailable
    order = np.lexsort((np.asarray(arrays["time"]), np.asarray(arrays["wheel_id"]), np.asarray(arrays["case_name"]).astype(str)))
    cases, wheels, times = arrays["case_name"].astype(str), arrays["wheel_id"], arrays["time"].astype(np.float64)
    contact = arrays.get("in_contact", np.zeros(n, dtype=np.float32)) > .5
    sinkage = arrays.get("sinkage", np.zeros(n, dtype=np.float32)).astype(np.float32)
    vy = arrays.get("wheel_vy_local")
    if "wheel_vx_slip" in arrays:
        vx_slip = arrays["wheel_vx_slip"]
    elif has_jx:
        # No direct relative longitudinal-slip speed is archived.  This is
        # the available causal proxy: local speed multiplied by slip ratio.
        vx_slip = arrays["wheel_vx_local"] * arrays["slip_long"]
    else:
        vx_slip = None
    group_start = 0
    while group_start < n:
        group_end = group_start + 1
        first = order[group_start]
        while group_end < n:
            current = order[group_end]
            if cases[current] != cases[first] or wheels[current] != wheels[first]:
                break
            group_end += 1
        idx = order[group_start:group_end]
        jy = jx = duration = ema = maximum = 0.0
        was_contact = False
        previous_time = float(times[idx[0]])
        for pos in idx:
            dt = max(0.0, float(times[pos]) - previous_time)
            previous_time = float(times[pos])
            if not contact[pos]:
                jy = jx = duration = ema = maximum = 0.0
            else:
                if was_contact:
                    duration += dt
                    if has_jy:
                        jy += float(vy[pos]) * dt
                    if has_jx:
                        jx += float(vx_slip[pos]) * dt
                    ema = beta * ema + (1.0 - beta) * float(sinkage[pos])
                    maximum = max(maximum, float(sinkage[pos]))
                else:
                    ema = maximum = float(sinkage[pos])
            if "j_y" in available: available["j_y"][pos] = jy
            if "j_x" in available: available["j_x"][pos] = jx
            if "contact_duration" in available: available["contact_duration"][pos] = duration
            if "sinkage_ema" in available: available["sinkage_ema"][pos] = ema
            if "sinkage_max_since_contact" in available: available["sinkage_max_since_contact"][pos] = maximum
            was_contact = bool(contact[pos])
        group_start = group_end
    return available, unavailable


def baseline_matrix(arrays: Dict[str, np.ndarray], indices: np.ndarray) -> Tuple[np.ndarray, List[str], List[str]]:
    parts, names, unavailable = [], [], []
    vector_keys = ("F_base", "F_LF", "Fphy", "wheel_vel", "wheel_acc", "wheel_rel_vel", "body_acc", "body_vel")
    scalar_keys = ("in_contact", "sinkage", "slip_long", "slip_lat", "body_yaw_rate", "wheel_omega", "wheel_vx_local", "wheel_vy_local", "slip_angle")
    for key in vector_keys:
        if key not in arrays:
            unavailable.append(key); continue
        parts.append(arrays[key][indices].astype(np.float32, copy=False)); names.extend([f"{key}_{axis}" for axis in "xyz"])
    for key in scalar_keys:
        if key not in arrays:
            unavailable.append(key); continue
        parts.append(arrays[key][indices, None].astype(np.float32, copy=False)); names.append(key)
    one_hot = np.eye(6, dtype=np.float32)[arrays["wheel_id"][indices].astype(np.int64)]
    parts.append(one_hot); names.extend([f"wheel_id_{i}" for i in range(6)])
    return np.concatenate(parts, axis=1), names, unavailable


def _sample_indices(n: int, maximum: int, seed: int) -> np.ndarray:
    if maximum <= 0 or n <= maximum:
        return np.arange(n, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(n, size=maximum, replace=False))


class SmallMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def fit_predict(x_train, y_train, x_eval, args) -> np.ndarray:
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = SmallMLP(x_train.shape[1]).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)), batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    for _ in range(args.epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = nn.functional.smooth_l1_loss(model(xb), yb); loss.backward(); opt.step()
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x_eval), args.batch_size):
            predictions.append(model(torch.from_numpy(x_eval[start:start + args.batch_size]).to(device)).cpu().numpy())
    return np.concatenate(predictions).astype(np.float32)


def metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    pred, true = np.asarray(pred, dtype=np.float64), np.asarray(true, dtype=np.float64)
    residual = pred - true
    corr = float(np.corrcoef(pred, true)[0, 1]) if np.std(pred) > 1e-12 and np.std(true) > 1e-12 else float("nan")
    sign = np.abs(true) > 1e-6
    return {"corr": corr, "R2": float(1.0 - np.sum(residual ** 2) / max(np.sum((true - true.mean()) ** 2), 1e-12)), "RMSE": float(np.sqrt(np.mean(residual ** 2))), "MAE": float(np.mean(np.abs(residual))), "std_pred": float(np.std(pred)), "std_true": float(np.std(true)), "std_ratio": float(np.std(pred) / max(np.std(true), 1e-12)), "sign_accuracy": float(np.mean(np.sign(pred[sign]) == np.sign(true[sign]))) if np.any(sign) else float("nan")}


def ridge_importance(x: np.ndarray, y: np.ndarray, names: List[str]) -> pd.DataFrame:
    xtx = x.T @ x + 1e-3 * np.eye(x.shape[1], dtype=np.float32)
    coef = np.linalg.solve(xtx, x.T @ y)
    return pd.DataFrame({"feature": names, "linear_ridge_coefficient": coef, "importance": np.abs(coef)}).sort_values("importance", ascending=False)


def main(args: argparse.Namespace) -> None:
    out = ensure_dir(Path(args.output_dir) if args.output_dir else Path(args.proof_dir) / "fy_history_feature_test")
    train, _ = load_residual_arrays(args.proof_dir, "train")
    val, _ = load_residual_arrays(args.proof_dir, "val")
    test, _ = load_residual_arrays(args.proof_dir, "test")
    train_history, history_missing = build_history_features(train, args.sinkage_ema_beta)
    val_history, val_missing = build_history_features(val, args.sinkage_ema_beta)
    test_history, test_missing = build_history_features(test, args.sinkage_ema_beta)
    history_names = [name for name in HISTORY_ORDER if name in train_history and name in val_history and name in test_history]
    unavailable = sorted(set(history_missing + val_missing + test_missing))
    for name in unavailable: print(f"[Fy History] {name} skipped (required LF/base field unavailable)", flush=True)
    train_idx = _sample_indices(len(train["time"]), args.max_train_samples, args.seed)
    val_idx = _sample_indices(len(val["time"]), args.max_eval_samples, args.seed + 1)
    test_idx = _sample_indices(len(test["time"]), args.max_eval_samples, args.seed + 2)
    x_train_base, base_names, base_missing = baseline_matrix(train, train_idx)
    x_val_base, _, val_base_missing = baseline_matrix(val, val_idx)
    x_test_base, _, test_base_missing = baseline_matrix(test, test_idx)
    y_train = (train["F_HF"][train_idx, 1] - train["F_base"][train_idx, 1]).astype(np.float32)
    y_val = (val["F_HF"][val_idx, 1] - val["F_base"][val_idx, 1]).astype(np.float32)
    y_test = (test["F_HF"][test_idx, 1] - test["F_base"][test_idx, 1]).astype(np.float32)
    configurations = [("baseline", [])]
    for name, features in (("baseline_plus_j_y", ["j_y"]), ("baseline_plus_jx_jy", ["j_x", "j_y"]), ("baseline_plus_jx_jy_contact", ["j_x", "j_y", "contact_duration"]), ("history", list(HISTORY_ORDER))):
        configurations.append((name, [feature for feature in features if feature in history_names]))
    seen, results, wheel_rows, predictions = set(), {}, [], {}
    for name, requested in configurations:
        key = tuple(requested)
        if key in seen: continue
        seen.add(key)
        hx_train = np.column_stack([train_history[feature][train_idx] for feature in requested]).astype(np.float32) if requested else np.empty((len(train_idx), 0), np.float32)
        hx_val = np.column_stack([val_history[feature][val_idx] for feature in requested]).astype(np.float32) if requested else np.empty((len(val_idx), 0), np.float32)
        hx_test = np.column_stack([test_history[feature][test_idx] for feature in requested]).astype(np.float32) if requested else np.empty((len(test_idx), 0), np.float32)
        raw_train, raw_val, raw_test = np.concatenate([x_train_base, hx_train], 1), np.concatenate([x_val_base, hx_val], 1), np.concatenate([x_test_base, hx_test], 1)
        feature_scaler = StandardScaler().fit(raw_train)
        target_scaler = StandardScaler().fit(y_train[:, None])
        x_train_norm = feature_scaler.transform(raw_train).astype(np.float32)
        y_train_norm = target_scaler.transform(y_train[:, None]).astype(np.float32)[:, 0]
        eval_raw = np.concatenate([raw_val, raw_test], axis=0)
        predict_all = target_scaler.inverse_transform(fit_predict(x_train_norm, y_train_norm, feature_scaler.transform(eval_raw).astype(np.float32), args)[:, None])[:, 0]
        predict_val, predict_test = predict_all[:len(raw_val)], predict_all[len(raw_val):]
        predictions[name] = {"val": predict_val, "test": predict_test}
        results[name] = {"features": requested, "val": metrics(predict_val, y_val), "test": metrics(predict_test, y_test)}
        for split, pred, arrays, indices, target in (("val", predict_val, val, val_idx, y_val), ("test", predict_test, test, test_idx, y_test)):
            for wheel in range(6):
                mask = arrays["wheel_id"][indices] == wheel
                wheel_rows.append({"experiment": name, "split": split, "wheel_id": wheel, "sample_count": int(mask.sum()), **metrics(pred[mask], target[mask])})
        if name == "history":
            importance = ridge_importance(x_train_norm, y_train_norm, base_names + requested)
            importance.to_csv(out / "feature_importance.csv", index=False)
    baseline_key = "baseline"; history_key = "history" if "history" in results else list(results)[-1]
    delta = {"delta_corr": results[history_key]["test"]["corr"] - results[baseline_key]["test"]["corr"], "delta_R2": results[history_key]["test"]["R2"] - results[baseline_key]["test"]["R2"], "delta_RMSE": results[history_key]["test"]["RMSE"] - results[baseline_key]["test"]["RMSE"]}
    label = "history_features_useful" if delta["delta_corr"] >= .15 else "history_features_weakly_useful" if delta["delta_corr"] >= .05 else "history_features_not_useful"
    metrics_rows = [{"experiment": name, "split": split, "features": ",".join(value["features"]), **split_metrics} for name, value in results.items() for split, split_metrics in (("val", value["val"]), ("test", value["test"]))]
    pd.DataFrame(metrics_rows).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(wheel_rows).to_csv(out / "wheel_metrics.csv", index=False)
    pd.DataFrame([row for row in metrics_rows if row["experiment"] != "baseline"]).to_csv(out / "ablation_metrics.csv", index=False)
    importance_frame = pd.read_csv(out / "feature_importance.csv") if (out / "feature_importance.csv").exists() else pd.DataFrame()
    history_importance = importance_frame[importance_frame["feature"].isin(history_names)]
    best = history_importance.iloc[0].to_dict() if len(history_importance) else {}
    jx_source = "wheel_vx_slip" if "wheel_vx_slip" in train else "wheel_vx_local * slip_long (wheel_vx_slip unavailable)"
    summary = {"model": "SmallMLP (LightGBM unavailable)", "residual_convention": "R_true_y=F_HF_y-F_base_y", "baseline_features": base_names, "history_features": history_names, "history_velocity_sources": {"j_y": "wheel_vy_local", "j_x": jx_source}, "unavailable_history_features": unavailable, "unavailable_baseline_fields": sorted(set(base_missing + val_base_missing + test_base_missing)), "sample_counts": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))}, "results": results, "improvement": delta, "label": label, "best_history_feature": best}
    write_json(out / "metrics.json", summary)
    print(f"[Baseline]\ncorr={results[baseline_key]['test']['corr']:.4f}\nR2={results[baseline_key]['test']['R2']:.4f}\nRMSE={results[baseline_key]['test']['RMSE']:.4f}", flush=True)
    print(f"[History]\ncorr={results[history_key]['test']['corr']:.4f}\nR2={results[history_key]['test']['R2']:.4f}\nRMSE={results[history_key]['test']['RMSE']:.4f}", flush=True)
    print(f"[Improvement]\ndelta_corr={delta['delta_corr']:.4f}\ndelta_R2={delta['delta_R2']:.4f}\ndelta_RMSE={delta['delta_RMSE']:.4f}\nlabel={label}", flush=True)
    print(f"[Best history feature]\n{best.get('feature', 'unavailable')} importance={best.get('importance', float('nan'))}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proof_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_train_samples", type=int, default=500000, help="0 uses all training samples")
    parser.add_argument("--max_eval_samples", type=int, default=0, help="0 evaluates every val/test sample")
    parser.add_argument("--sinkage_ema_beta", type=float, default=.9)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
