from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
WHEEL_IDS = range(6)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def finite_array(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)


def build_mask(df: pd.DataFrame, wheel_id: int, contact_filter: str) -> np.ndarray:
    required = [
        f"lf_wheel{wheel_id}_pos_z",
        f"hf_wheel{wheel_id}_pos_z",
        f"lf_wheel{wheel_id}_sinkage",
        f"hf_wheel{wheel_id}_sinkage",
    ]
    mask = np.ones(len(df), dtype=bool)
    for col in required:
        mask &= np.isfinite(finite_array(df[col]))

    lf_contact_col = f"lf_wheel{wheel_id}_in_contact"
    if contact_filter == "lf" and lf_contact_col in df.columns:
        mask &= finite_array(df[lf_contact_col]) > 0.5
    return mask


def collect_wheel_points(df: pd.DataFrame, wheel_id: int, contact_filter: str) -> pd.DataFrame:
    mask = build_mask(df, wheel_id, contact_filter)
    lf_z = finite_array(df[f"lf_wheel{wheel_id}_pos_z"])
    hf_z = finite_array(df[f"hf_wheel{wheel_id}_pos_z"])
    lf_sinkage = finite_array(df[f"lf_wheel{wheel_id}_sinkage"])
    hf_sinkage = finite_array(df[f"hf_wheel{wheel_id}_sinkage"])

    out = pd.DataFrame(
        {
            "wheel_id": wheel_id,
            "x_neg_delta_z": -(hf_z[mask] - lf_z[mask]),
            "y_delta_sinkage": hf_sinkage[mask] - lf_sinkage[mask],
        }
    )
    if "case_name" in df.columns:
        out["case_name"] = df.loc[mask, "case_name"].astype(str).to_numpy()
    if "time" in df.columns:
        out["time"] = finite_array(df.loc[mask, "time"])
    return out


def compute_stats(points: pd.DataFrame) -> Dict[str, float]:
    x = points["x_neg_delta_z"].to_numpy(dtype=np.float64)
    y = points["y_delta_sinkage"].to_numpy(dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return {"count": 0}

    err = y - x
    denom = float(np.dot(x, x))
    slope_zero_intercept = float(np.dot(x, y) / denom) if denom > 1e-12 else float("nan")
    if x.size >= 2:
        slope, intercept = np.polyfit(x, y, deg=1)
        corr = np.corrcoef(x, y)[0, 1] if np.std(x) > 1e-12 and np.std(y) > 1e-12 else float("nan")
    else:
        slope, intercept, corr = float("nan"), float("nan"), float("nan")

    return {
        "count": int(x.size),
        "x_mean": float(np.mean(x)),
        "y_mean": float(np.mean(y)),
        "x_std": float(np.std(x)),
        "y_std": float(np.std(y)),
        "corr": float(corr),
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_zero_intercept": slope_zero_intercept,
        "mae_y_minus_x": float(np.mean(np.abs(err))),
        "rmse_y_minus_x": float(np.sqrt(np.mean(err * err))),
        "p95_abs_y_minus_x": float(np.percentile(np.abs(err), 95)),
    }


def downsample(points: pd.DataFrame, max_points: int, seed: int) -> pd.DataFrame:
    if max_points <= 0 or len(points) <= max_points:
        return points
    return points.sample(n=max_points, random_state=seed)


def plot_scatter(points: pd.DataFrame, title: str, output_path: Path, max_points: int, seed: int) -> None:
    sample = downsample(points, max_points=max_points, seed=seed)
    x = sample["x_neg_delta_z"].to_numpy(dtype=np.float64)
    y = sample["y_delta_sinkage"].to_numpy(dtype=np.float64)
    if x.size == 0:
        return

    lim_min = float(np.nanmin([x.min(), y.min()]))
    lim_max = float(np.nanmax([x.max(), y.max()]))
    pad = max((lim_max - lim_min) * 0.05, 1e-4)
    lim_min -= pad
    lim_max += pad

    fig = plt.figure(figsize=(6.8, 6.2))
    plt.scatter(x, y, s=4, alpha=0.25, edgecolors="none")
    plt.plot([lim_min, lim_max], [lim_min, lim_max], color="black", linewidth=1.5, label="y = x")
    plt.axhline(0.0, color="gray", linewidth=0.8, alpha=0.5)
    plt.axvline(0.0, color="gray", linewidth=0.8, alpha=0.5)
    plt.xlim(lim_min, lim_max)
    plt.ylim(lim_min, lim_max)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x = -(wheel_z_HF - wheel_z_LF) / m")
    plt.ylabel("y = sinkage_HF - sinkage_LF / m")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def validate_columns(df: pd.DataFrame) -> None:
    missing: List[str] = []
    for i in WHEEL_IDS:
        missing.extend(
            col
            for col in [
                f"lf_wheel{i}_pos_z",
                f"hf_wheel{i}_pos_z",
                f"lf_wheel{i}_sinkage",
                f"hf_wheel{i}_sinkage",
            ]
            if col not in df.columns
        )
    if missing:
        preview = ", ".join(missing[:20])
        suffix = " ..." if len(missing) > 20 else ""
        raise KeyError(f"输入 CSV 缺少必要列: {preview}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate wheel-z delta and sinkage delta geometric consistency.")
    parser.add_argument("--merged_csv", type=str, default=str(ROOT / "Feature_Selection" / "DataSet" / "merged_error_dataset.csv"))
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "results_v2" / "diagnostics" / "wheel_z_sinkage_consistency"))
    parser.add_argument("--contact_filter", choices=["none", "lf"], default="lf", help="默认只按 LF in_contact 筛选；HF in_contact 始终忽略")
    parser.add_argument("--max_plot_points", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    df = pd.read_csv(args.merged_csv)
    validate_columns(df)

    wheel_points = [collect_wheel_points(df, i, args.contact_filter) for i in WHEEL_IDS]
    all_points = pd.concat(wheel_points, ignore_index=True)
    all_points.to_csv(output_dir / "wheel_z_sinkage_points.csv", index=False)

    stats: Dict[str, Dict[str, float]] = {
        "all_wheels": compute_stats(all_points),
        "by_wheel": {f"wheel{i}": compute_stats(points) for i, points in enumerate(wheel_points)},
    }
    stats["config"] = {
        "merged_csv": str(args.merged_csv),
        "contact_filter": str(args.contact_filter),
        "x": "-(wheel_z_HF - wheel_z_LF)",
        "y": "sinkage_HF - sinkage_LF",
    }
    with open(output_dir / "wheel_z_sinkage_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    plot_scatter(
        all_points,
        title=f"All wheels | contact_filter={args.contact_filter}",
        output_path=output_dir / "scatter_all_wheels.png",
        max_points=args.max_plot_points,
        seed=args.seed,
    )
    for i, points in enumerate(wheel_points):
        plot_scatter(
            points,
            title=f"Wheel {i} | contact_filter={args.contact_filter}",
            output_path=output_dir / f"scatter_wheel{i}.png",
            max_points=args.max_plot_points,
            seed=args.seed,
        )

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"points saved to: {output_dir / 'wheel_z_sinkage_points.csv'}")
    print(f"stats saved to: {output_dir / 'wheel_z_sinkage_stats.json'}")
    print(f"plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
