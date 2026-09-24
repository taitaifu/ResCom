from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


DIRECTIONS = ("x", "y", "z")
DIR_LABELS = ("Fx", "Fy", "Fz")
WHEEL_IDS = tuple(range(6))
FORCE_KEYS = ("F_base", "F_HF", "F_LF", "Fphy", "R")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log(prefix: str, message: str) -> None:
    print(f"[{prefix}] {message}", flush=True)


def write_json(path: str | Path, obj: Mapping) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def finite_rmse(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(true)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean(np.square(pred[mask] - true[mask]))))


def finite_mae(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(true)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(pred[mask] - true[mask])))


def residual_stats(values: np.ndarray) -> Dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {k: float("nan") for k in ["rmse", "mae", "mean", "std", "q50", "q90", "q95", "q99"]}
    return {
        "rmse": float(np.sqrt(np.mean(np.square(x)))),
        "mae": float(np.mean(np.abs(x))),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "q50": float(np.quantile(np.abs(x), 0.50)),
        "q90": float(np.quantile(np.abs(x), 0.90)),
        "q95": float(np.quantile(np.abs(x), 0.95)),
        "q99": float(np.quantile(np.abs(x), 0.99)),
    }


def force_metrics(f_base: np.ndarray, f_hf: np.ndarray, f_corrected: Optional[np.ndarray] = None) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out["base_overall_rmse"] = finite_rmse(f_base, f_hf)
    if f_corrected is not None:
        out["corrected_overall_rmse"] = finite_rmse(f_corrected, f_hf)
        out["overall_improvement_pct"] = improvement_pct(out["base_overall_rmse"], out["corrected_overall_rmse"])
    for j, label in enumerate(DIR_LABELS):
        out[f"base_{label}_rmse"] = finite_rmse(f_base[..., j], f_hf[..., j])
        if f_corrected is not None:
            out[f"corrected_{label}_rmse"] = finite_rmse(f_corrected[..., j], f_hf[..., j])
            out[f"{label}_improvement_pct"] = improvement_pct(out[f"base_{label}_rmse"], out[f"corrected_{label}_rmse"])
    return out


def improvement_pct(base: float, new: float) -> float:
    if not np.isfinite(base) or abs(base) <= 1e-12 or not np.isfinite(new):
        return float("nan")
    return float(100.0 * (base - new) / base)


def contact_transition_mask(contact: np.ndarray) -> np.ndarray:
    c = np.asarray(contact).astype(bool)
    out = np.zeros_like(c, dtype=bool)
    if c.size == 0:
        return out
    out[1:] |= c[1:] != c[:-1]
    out[:-1] |= c[1:] != c[:-1]
    return out


def save_npz(path: str | Path, arrays: Mapping[str, np.ndarray], meta: Mapping) -> None:
    payload = {k: np.asarray(v) for k, v in arrays.items()}
    payload["metadata_json"] = np.asarray(json.dumps(meta, ensure_ascii=False))
    np.savez_compressed(path, **payload)


def load_npz(path: str | Path) -> Tuple[Dict[str, np.ndarray], Dict]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files if k != "metadata_json"}
        meta = json.loads(str(data["metadata_json"])) if "metadata_json" in data.files else {}
    return arrays, meta


def arrays_to_frame(arrays: Mapping[str, np.ndarray]) -> pd.DataFrame:
    out: Dict[str, np.ndarray] = {}
    for key, arr in arrays.items():
        arr = np.asarray(arr)
        if arr.ndim == 1:
            out[key] = arr
        elif arr.ndim == 2 and arr.shape[1] == 3:
            if key in FORCE_KEYS or key in {"wheel_vel", "wheel_acc", "wheel_rel_vel", "body_acc"}:
                suffixes = DIR_LABELS if key in FORCE_KEYS else DIRECTIONS
                for j, suffix in enumerate(suffixes):
                    out[f"{key}_{suffix}"] = arr[:, j]
        else:
            out[key] = arr.reshape(arr.shape[0], -1)[:, 0]
    return pd.DataFrame(out)


def write_summary_csv(path: str | Path, rows: List[Mapping]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def plot_series(path: str | Path, time: np.ndarray, series: Mapping[str, np.ndarray], title: str, ylabel: str) -> None:
    ensure_dir(Path(path).parent)
    fig = plt.figure(figsize=(10, 4))
    for label, y in series.items():
        plt.plot(time, y, label=label, linewidth=1.5)
    plt.xlabel("time / s")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def plot_scatter(path: str | Path, x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str) -> None:
    ensure_dir(Path(path).parent)
    fig = plt.figure(figsize=(5, 5))
    plt.scatter(x, y, s=4, alpha=0.35)
    lo = float(np.nanmin([np.nanmin(x), np.nanmin(y)]))
    hi = float(np.nanmax([np.nanmax(x), np.nanmax(y)]))
    if np.isfinite(lo) and np.isfinite(hi):
        plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def check_no_hf_leakage(feature_names: Sequence[str], allow_oracle: bool = False) -> List[str]:
    banned = []
    for name in feature_names:
        low = name.lower()
        if "f_hf" in low or low.startswith("hf_") or "future" in low:
            banned.append(name)
        if not allow_oracle and ("r_true" in low or "true_residual" in low):
            banned.append(name)
    return sorted(set(banned))


def split_case_sets(proof_dir: str | Path) -> Dict[str, set[str]]:
    out: Dict[str, set[str]] = {}
    for split in ["train", "val", "test"]:
        arrays, _ = load_npz(Path(proof_dir) / f"residual_{split}.npz")
        out[split] = set(np.asarray(arrays["case_name"]).astype(str).tolist())
    return out


def split_overlap_report(proof_dir: str | Path) -> Dict[str, List[str]]:
    sets = split_case_sets(proof_dir)
    return {
        "train_val": sorted(sets["train"] & sets["val"]),
        "train_test": sorted(sets["train"] & sets["test"]),
        "val_test": sorted(sets["val"] & sets["test"]),
    }
