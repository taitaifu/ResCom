from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import chi2

from .residual_utils import DIR_LABELS, WHEEL_IDS, contact_transition_mask, ensure_dir, finite_rmse, residual_stats

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
except Exception:  # pragma: no cover
    acorr_ljungbox = None


ACF_LAGS = [1, 2, 3, 5, 10, 20]
LB_LAGS = [5, 10, 20]


class RidgeAR:
    def __init__(self, alpha: float = 1e-3) -> None:
        self.alpha = float(alpha)
        self.coef_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeAR":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        xb = np.column_stack([np.ones(x.shape[0]), x])
        reg = self.alpha * np.eye(xb.shape[1])
        reg[0, 0] = 0.0
        self.coef_ = np.linalg.solve(xb.T @ xb + reg, xb.T @ y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("RidgeAR is not fitted")
        xb = np.column_stack([np.ones(x.shape[0]), np.asarray(x, dtype=np.float64)])
        return xb @ self.coef_


def grouped_series(df: pd.DataFrame):
    for (case_name, wheel_id), sub in df.sort_values(["case_name", "wheel_id", "time"]).groupby(["case_name", "wheel_id"], sort=False):
        yield str(case_name), int(wheel_id), sub.reset_index(drop=True)


def _acf_one(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size <= lag + 1:
        return float("nan")
    a = x[:-lag]
    b = x[lag:]
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def run_acf(df: pd.DataFrame, out_dir: str | Path) -> pd.DataFrame:
    rows: List[Dict] = []
    per_plot: Dict[Tuple[int, str], Dict[int, List[float]]] = {}
    masks = {
        "all": lambda s: np.ones(len(s), dtype=bool),
        "in_contact": lambda s: s["in_contact"].to_numpy(dtype=np.float64) >= 0.5,
        "contact_transition": lambda s: contact_transition_mask(s["in_contact"].to_numpy(dtype=np.float64) >= 0.5),
    }
    for wheel in WHEEL_IDS:
        for direction in DIR_LABELS:
            for mask_name, mask_fn in masks.items():
                lag_values = {lag: [] for lag in ACF_LAGS}
                for _, wid, sub in grouped_series(df):
                    if wid != wheel:
                        continue
                    mask = mask_fn(sub)
                    values = sub.loc[mask, f"R_{direction}"].to_numpy(dtype=np.float64)
                    for lag in ACF_LAGS:
                        v = _acf_one(values, lag)
                        if np.isfinite(v):
                            lag_values[lag].append(v)
                for lag, vals in lag_values.items():
                    arr = np.asarray(vals, dtype=np.float64)
                    rows.append({
                        "wheel_id": wheel,
                        "direction": direction,
                        "subset": mask_name,
                        "lag": lag,
                        "acf_mean": float(np.mean(arr)) if arr.size else float("nan"),
                        "acf_median": float(np.median(arr)) if arr.size else float("nan"),
                        "acf_std": float(np.std(arr)) if arr.size else float("nan"),
                        "valid_case_count": int(arr.size),
                    })
                    if mask_name == "all":
                        per_plot.setdefault((wheel, direction), {})[lag] = vals

    fig_dir = ensure_dir(Path(out_dir) / "figures")
    for (wheel, direction), lag_map in per_plot.items():
        fig = plt.figure(figsize=(7, 4))
        means = [np.nanmean(lag_map.get(lag, [np.nan])) for lag in ACF_LAGS]
        plt.plot(ACF_LAGS, means, marker="o")
        plt.xlabel("lag")
        plt.ylabel("ACF")
        plt.title(f"wheel{wheel} {direction} residual ACF")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / f"acf_wheel{wheel}_{direction}.png", dpi=150)
        plt.close(fig)
    out = pd.DataFrame(rows)
    out.to_csv(Path(out_dir) / "acf_summary.csv", index=False)
    return out


def run_ljung_box(df: pd.DataFrame, out_dir: str | Path) -> pd.DataFrame:
    rows: List[Dict] = []
    backend = "statsmodels" if acorr_ljungbox is not None else "scipy_fallback"
    for wheel in WHEEL_IDS:
        for direction in DIR_LABELS:
            pvals = {lag: [] for lag in LB_LAGS}
            for _, wid, sub in grouped_series(df):
                if wid != wheel:
                    continue
                x = sub[f"R_{direction}"].to_numpy(dtype=np.float64)
                x = x[np.isfinite(x)]
                if x.size <= max(LB_LAGS) + 1 or np.std(x) <= 1e-12:
                    continue
                if acorr_ljungbox is not None:
                    lb = acorr_ljungbox(x, lags=LB_LAGS, return_df=True)
                    for lag in LB_LAGS:
                        p = float(lb.loc[lag, "lb_pvalue"])
                        if np.isfinite(p):
                            pvals[lag].append(p)
                else:
                    centered = x - np.mean(x)
                    denom = float(np.dot(centered, centered))
                    if denom <= 1e-12:
                        continue
                    acfs = np.asarray([
                        float(np.dot(centered[:-lag], centered[lag:]) / denom)
                        for lag in range(1, max(LB_LAGS) + 1)
                    ])
                    n = float(x.size)
                    for lag in LB_LAGS:
                        used = acfs[:lag]
                        q = n * (n + 2.0) * float(np.sum((used * used) / np.maximum(n - np.arange(1, lag + 1), 1.0)))
                        p = float(chi2.sf(q, df=lag))
                        if np.isfinite(p):
                            pvals[lag].append(p)
            for lag, vals in pvals.items():
                arr = np.asarray(vals, dtype=np.float64)
                rows.append({
                    "wheel_id": wheel,
                    "direction": direction,
                    "lag": lag,
                    "backend": backend,
                    "valid_case_count": int(arr.size),
                    "reject_ratio_p005": float(np.mean(arr < 0.05)) if arr.size else float("nan"),
                    "mean_pvalue": float(np.mean(arr)) if arr.size else float("nan"),
                    "median_pvalue": float(np.median(arr)) if arr.size else float("nan"),
                })
    out = pd.DataFrame(rows)
    out.to_csv(Path(out_dir) / "ljung_box_summary.csv", index=False)
    return out


def _frequency_bands(freq: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    positive = freq[freq > 0]
    if positive.size == 0:
        return np.zeros_like(freq, dtype=bool), np.zeros_like(freq, dtype=bool), np.zeros_like(freq, dtype=bool)
    q1, q2 = np.quantile(positive, [1 / 3, 2 / 3])
    return freq <= q1, (freq > q1) & (freq <= q2), freq > q2


def run_psd(df: pd.DataFrame, out_dir: str | Path) -> pd.DataFrame:
    rows: List[Dict] = []
    fig_dir = ensure_dir(Path(out_dir) / "figures")
    for wheel in WHEEL_IDS:
        for direction in DIR_LABELS:
            psds = []
            freqs_ref = None
            ratios = []
            for _, wid, sub in grouped_series(df):
                if wid != wheel:
                    continue
                x = sub[f"R_{direction}"].to_numpy(dtype=np.float64)
                t = sub["time"].to_numpy(dtype=np.float64)
                mask = np.isfinite(x) & np.isfinite(t)
                x = x[mask]
                t = t[mask]
                if x.size < 8:
                    continue
                dt = np.diff(t)
                dt = dt[np.isfinite(dt) & (dt > 1e-12)]
                fs = 1.0 / float(np.median(dt)) if dt.size else 1.0
                nperseg = min(256, x.size)
                freq, pxx = signal.welch(x - np.mean(x), fs=fs, nperseg=nperseg)
                low, mid, high = _frequency_bands(freq)
                total = float(np.trapz(pxx, freq)) + 1e-12
                ratios.append((
                    float(np.trapz(pxx[low], freq[low]) / total) if np.any(low) else float("nan"),
                    float(np.trapz(pxx[mid], freq[mid]) / total) if np.any(mid) else float("nan"),
                    float(np.trapz(pxx[high], freq[high]) / total) if np.any(high) else float("nan"),
                ))
                if freqs_ref is None:
                    freqs_ref = freq
                    psds.append(pxx)
                else:
                    psds.append(np.interp(freqs_ref, freq, pxx))
            ratio_arr = np.asarray(ratios, dtype=np.float64)
            rows.append({
                "wheel_id": wheel,
                "direction": direction,
                "valid_case_count": int(ratio_arr.shape[0]),
                "low_frequency_energy_ratio": float(np.nanmean(ratio_arr[:, 0])) if ratio_arr.size else float("nan"),
                "mid_frequency_energy_ratio": float(np.nanmean(ratio_arr[:, 1])) if ratio_arr.size else float("nan"),
                "high_frequency_energy_ratio": float(np.nanmean(ratio_arr[:, 2])) if ratio_arr.size else float("nan"),
            })
            if freqs_ref is not None and psds:
                fig = plt.figure(figsize=(7, 4))
                plt.semilogy(freqs_ref, np.nanmean(np.asarray(psds), axis=0))
                plt.xlabel("frequency / Hz")
                plt.ylabel("PSD")
                plt.title(f"wheel{wheel} {direction} residual PSD")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(fig_dir / f"psd_wheel{wheel}_{direction}.png", dpi=150)
                plt.close(fig)
    out = pd.DataFrame(rows)
    out.to_csv(Path(out_dir) / "psd_summary.csv", index=False)
    return out


def run_residual_statistics(df: pd.DataFrame, out_dir: str | Path) -> pd.DataFrame:
    rows: List[Dict] = []
    subset_masks = {
        "all": np.ones(len(df), dtype=bool),
        "contact": df["in_contact"].to_numpy(dtype=np.float64) >= 0.5,
        "non_contact": df["in_contact"].to_numpy(dtype=np.float64) < 0.5,
        "contact_transition": np.zeros(len(df), dtype=bool),
    }
    transition = np.zeros(len(df), dtype=bool)
    for _, sub in df.sort_values(["case_name", "wheel_id", "time"]).groupby(["case_name", "wheel_id"], sort=False):
        transition[sub.index.to_numpy()] = contact_transition_mask(sub["in_contact"].to_numpy(dtype=np.float64) >= 0.5)
    subset_masks["contact_transition"] = transition
    for wheel in list(WHEEL_IDS) + ["overall"]:
        wheel_mask = np.ones(len(df), dtype=bool) if wheel == "overall" else df["wheel_id"].to_numpy() == int(wheel)
        for direction in DIR_LABELS:
            values = df[f"R_{direction}"].to_numpy(dtype=np.float64)
            for subset, subset_mask in subset_masks.items():
                stats = residual_stats(values[wheel_mask & subset_mask])
                rows.append({"wheel_id": wheel, "direction": direction, "subset": subset, **stats})
    out = pd.DataFrame(rows)
    out.to_csv(Path(out_dir) / "residual_statistics.csv", index=False)
    return out


def _ar_xy(df: pd.DataFrame, history_len: int, direction: str) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for _, _, sub in grouped_series(df):
        r = sub[f"R_{direction}"].to_numpy(dtype=np.float64)
        for end in range(history_len, len(r)):
            window = r[end - history_len:end + 1]
            if np.all(np.isfinite(window)):
                xs.append(window[:-1])
                ys.append(window[-1])
    if not xs:
        return np.empty((0, history_len)), np.empty((0,))
    return np.asarray(xs), np.asarray(ys)


def run_ar_predictability(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, history_len: int, out_dir: str | Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for split_name, eval_df in [("val", val_df), ("test", test_df)]:
        for wheel in list(WHEEL_IDS) + ["overall"]:
            tr = train_df if wheel == "overall" else train_df[train_df["wheel_id"] == int(wheel)]
            ev = eval_df if wheel == "overall" else eval_df[eval_df["wheel_id"] == int(wheel)]
            for direction in DIR_LABELS:
                x_train, y_train = _ar_xy(tr, history_len, direction)
                x_eval, y_eval = _ar_xy(ev, history_len, direction)
                if len(y_train) < 2 or len(y_eval) < 1:
                    zero_rmse = ar_rmse = rel = float("nan")
                else:
                    model = RidgeAR(alpha=1e-3).fit(x_train, y_train)
                    pred = model.predict(x_eval)
                    zero_rmse = finite_rmse(np.zeros_like(y_eval), y_eval)
                    ar_rmse = finite_rmse(pred, y_eval)
                    rel = float((zero_rmse - ar_rmse) / zero_rmse) if zero_rmse > 1e-12 else float("nan")
                rows.append({
                    "split": split_name,
                    "wheel_id": wheel,
                    "direction": direction,
                    "zero_rmse": zero_rmse,
                    "ar_rmse": ar_rmse,
                    "relative_improvement": rel,
                })
    out = pd.DataFrame(rows)
    out.to_csv(Path(out_dir) / "ar_predictability.csv", index=False)
    return out


def run_counterfactual(train_df: pd.DataFrame, val_df: pd.DataFrame, history_len: int, out_dir: str | Path, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []
    for direction in DIR_LABELS:
        x_val, y_val = _ar_xy(val_df, history_len, direction)
        x_seq, y_seq = _ar_xy(train_df, history_len, direction)
        variants = {"sequential": (x_seq, y_seq)}
        shuffled = train_df.copy()
        shuffled[f"R_{direction}"] = rng.permutation(shuffled[f"R_{direction}"].to_numpy())
        variants["time_shuffled"] = _ar_xy(shuffled, history_len, direction)
        noisy = train_df.copy()
        mu = float(np.nanmean(noisy[f"R_{direction}"].to_numpy(dtype=np.float64)))
        sd = float(np.nanstd(noisy[f"R_{direction}"].to_numpy(dtype=np.float64)))
        noisy[f"R_{direction}"] = rng.normal(mu, sd, size=len(noisy))
        variants["gaussian_noise"] = _ar_xy(noisy, history_len, direction)
        for name, (x_train, y_train) in variants.items():
            if len(y_train) < 2 or len(y_val) < 1:
                rmse = float("nan")
            else:
                model = RidgeAR(alpha=1e-3).fit(x_train, y_train)
                rmse = finite_rmse(model.predict(x_val), y_val)
            rows.append({"direction": direction, "train_variant": name, "val_rmse": rmse})
    out = pd.DataFrame(rows)
    out.to_csv(Path(out_dir) / "counterfactual_summary.csv", index=False)
    return out
