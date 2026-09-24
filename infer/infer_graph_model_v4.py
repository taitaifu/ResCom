from __future__ import annotations

import argparse
import json
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from Feature_Selection.datasetProcess import (  # noqa: E402
    add_prefix,
    add_rotvec_residuals,
    read_case_data,
)
from models.data_utils_v4 import (  # noqa: E402
    GraphTemporalSequenceDatasetV4,
    GroupStandardizer,
    NumpyStandardScaler,
    WHEEL_IDS,
    check_columns_exist,
    ensure_time_feature_columns,
    get_group_dims,
    graph_temporal_collate_fn,
    load_column_spec,
    load_merged_dataset,
    split_train_val_test_by_case,
)
from models.load_transfer_force_v4 import load_transfer_force_torch  # noqa: E402
from models.graph_temporal_hgt_compensation_v4 import GraphTemporalHGTCompensationModelV4  # noqa: E402
from train.train_graph_model_v4 import (  # noqa: E402
    apply_axis_gate,
    inverse_transform_tensor,
    load_compatible_state_dict,
    omega_index,
    reconstruct_body_kinematic,
    reconstruct_wheel_kin_with_body_delta,
    residual_raw_like_target,
    suffix_indices,
)


CONTROL_PARAM_ORDER = [
    "idle_time",
    "accel_time",
    "accel_speed",
    "const_time",
    "sinusoid_time",
    "sinusoid_speed",
    "sinusoid_freq",
    "step_time",
    "dec_time",
    "idle_end",
    "step_low_speed",
    "step_high_speed",
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def log_status(message: str) -> None:
    print(f"[infer_v4] {message}", flush=True)


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out: Dict = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def get_arg_with_default(ckpt_args: Dict, key: str, cli_value, default):
    if cli_value is not None:
        return cli_value
    if ckpt_args and key in ckpt_args:
        return ckpt_args[key]
    return default


def select_dataframe(
    df: pd.DataFrame,
    split: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    case_name: Optional[str],
) -> pd.DataFrame:
    if split == "all":
        selected = df.copy()
    else:
        df_train, df_val, df_test = split_train_val_test_by_case(
            df,
            case_col="case_name",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
        selected = {"train": df_train, "val": df_val, "test": df_test}[split]
    if case_name is not None:
        selected = selected[selected["case_name"].astype(str) == str(case_name)].copy()
    selected = selected.sort_values(["case_name", "time"]).reset_index(drop=True)
    if len(selected) == 0:
        raise ValueError("筛选后的推理数据为空，请检查 split / case_name 参数")
    return selected


def split_case_ids(
    case_ids: List[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, set[str]]:
    shuffled = list(case_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio))) if n >= 3 else max(0, n - n_train)
    n_train = min(n_train, n)
    n_val = min(n_val, max(0, n - n_train))
    return {
        "train": set(shuffled[:n_train]),
        "val": set(shuffled[n_train:n_train + n_val]),
        "test": set(shuffled[n_train + n_val:]),
    }


def read_case_ids_from_csv(path: str, case_col: str, chunksize: int) -> List[str]:
    seen: set[str] = set()
    case_ids: List[str] = []
    rows = 0
    for chunk in pd.read_csv(path, usecols=[case_col], chunksize=chunksize):
        rows += len(chunk)
        for case_id in chunk[case_col].astype(str).unique().tolist():
            if case_id not in seen:
                seen.add(case_id)
                case_ids.append(case_id)
        log_status(f"scanned case ids: rows={rows}, cases={len(case_ids)}")
    return case_ids


def inference_required_columns(spec, available_cols: set[str], case_col: str, time_col: str) -> List[str]:
    required = [case_col, time_col]
    required.extend(spec.base_feature_cols)
    required.extend(spec.proxy_feature_cols)
    required.extend(spec.res_cols)
    required.extend(spec.target_cols)

    out: List[str] = []
    seen: set[str] = set()
    for col in required:
        if col in available_cols and col not in seen:
            seen.add(col)
            out.append(col)
    return out


def load_selected_merged_dataset(
    merged_csv_path: str,
    spec,
    split: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    case_name: Optional[str],
    chunksize: int,
) -> pd.DataFrame:
    if split == "all" and case_name is None:
        return load_merged_dataset(merged_csv_path, spec, case_col="case_name", time_col="time")

    header = pd.read_csv(merged_csv_path, nrows=0)
    available_cols = set(header.columns.astype(str))
    if "case_name" not in available_cols or "time" not in available_cols:
        raise KeyError("merged_error_dataset.csv 必须包含 case_name 和 time")

    if split == "all":
        selected_cases = {str(case_name)}
    else:
        case_ids = read_case_ids_from_csv(merged_csv_path, "case_name", chunksize)
        split_ids = split_case_ids(case_ids, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
        selected_cases = split_ids[split]
        if case_name is not None:
            selected_cases = selected_cases & {str(case_name)}
    if not selected_cases:
        raise ValueError("筛选后的推理 case 为空，请检查 split / case_name 参数")

    usecols = inference_required_columns(spec, available_cols, "case_name", "time")
    frames: List[pd.DataFrame] = []
    rows = 0
    selected_rows = 0
    for chunk in pd.read_csv(merged_csv_path, usecols=usecols, chunksize=chunksize):
        rows += len(chunk)
        sub = chunk[chunk["case_name"].astype(str).isin(selected_cases)]
        if len(sub):
            frames.append(sub.copy())
            selected_rows += len(sub)
        log_status(f"loaded selected split: rows={rows}, selected_rows={selected_rows}")

    if not frames:
        raise ValueError("筛选后的推理数据为空，请检查 split / case_name 参数")
    df = pd.concat(frames, ignore_index=True)
    df = ensure_time_feature_columns(df, case_col="case_name", time_col="time")
    check_columns_exist(df, spec.base_feature_cols, "base_feature_cols")
    check_columns_exist(df, spec.proxy_feature_cols, "proxy_feature_cols")
    check_columns_exist(df, spec.res_cols, "res_cols")
    check_columns_exist(df, spec.target_cols, "target_cols")
    return df.sort_values(["case_name", "time"]).reset_index(drop=True)


def safe_load_checkpoint(path: str, device: torch.device) -> Dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def flatten_group_rows(
    rows: List[Dict[str, float]],
    sample_idx_offset: int,
    case_names: List[str],
    times: torch.Tensor,
    pred_raw: torch.Tensor,
    true_raw: torch.Tensor,
    columns: List[str],
    group_name: str,
) -> None:
    pred_np = pred_raw.detach().cpu().numpy()
    true_np = true_raw.detach().cpu().numpy()
    time_np = times.detach().cpu().numpy()
    batch_size, horizon, _ = pred_np.shape

    for b in range(batch_size):
        for h in range(horizon):
            row_idx = sample_idx_offset + b * horizon + h
            if len(rows) <= row_idx:
                rows.append(
                    {
                        "sample_index": sample_idx_offset // max(1, horizon) + b,
                        "case_name": case_names[b],
                        "time": float(time_np[b]),
                        "horizon_index": h,
                    }
                )
            row = rows[row_idx]
            for j, col in enumerate(columns):
                row[f"pred::{group_name}::{col}"] = float(pred_np[b, h, j])
                row[f"true::{group_name}::{col}"] = float(true_np[b, h, j])


def flatten_reference_rows(
    rows: List[Dict[str, float]],
    sample_idx_offset: int,
    times: torch.Tensor,
    ref_raw: torch.Tensor,
    columns: List[str],
    group_name: str,
    prefix: str,
) -> None:
    ref_np = ref_raw.detach().cpu().numpy()
    time_np = times.detach().cpu().numpy()
    batch_size, horizon, _ = ref_np.shape
    for b in range(batch_size):
        for h in range(horizon):
            row_idx = sample_idx_offset + b * horizon + h
            row = rows[row_idx]
            row["time"] = float(time_np[b])
            for j, col in enumerate(columns):
                row[f"{prefix}::{group_name}::{col}"] = float(ref_np[b, h, j])


def map_case_basename(case_dir: Path) -> str:
    name = case_dir.name
    if "_case" in name:
        return "case" + name.split("_case", 1)[1]
    return name


def infer_output_case_label(args: argparse.Namespace) -> str:
    if args.infer_case_name:
        return str(args.infer_case_name)
    if args.case_name:
        return str(args.case_name)
    if args.infer_mode == "custom_case" and args.custom_case_dir:
        return map_case_basename(Path(args.custom_case_dir))
    if args.infer_mode == "custom_cases":
        return "cases"
    return "all"


def safe_path_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))
    return safe or "unknown"


def build_control_params(args: argparse.Namespace) -> Optional[Dict[str, float]]:
    values = getattr(args, "control_params", None)
    if not values:
        return None
    if len(values) != len(CONTROL_PARAM_ORDER):
        raise ValueError(f"--control_params 需要 {len(CONTROL_PARAM_ORDER)} 个数值")
    return {name: float(value) for name, value in zip(CONTROL_PARAM_ORDER, values)}


def build_custom_inference_dataframe(args: argparse.Namespace, spec) -> pd.DataFrame:
    if not args.custom_case_dir:
        raise ValueError("纯推理模式需要提供 --custom_case_dir")
    control_params = build_control_params(args)
    case_name = args.infer_case_name or map_case_basename(Path(args.custom_case_dir))
    case_params = control_params or {}

    lf_df = read_case_data(args.custom_case_dir, case_params, compute_features=True)
    if lf_df is None or len(lf_df) == 0:
        raise ValueError("custom_case 数据读取失败或为空")
    lf_df = lf_df.copy()
    lf_df["case_name"] = case_name
    lf_df = add_prefix(lf_df, "lf_")

    if args.sph_case_dir:
        hf_df = read_case_data(args.sph_case_dir, case_params, compute_features=False)
        if hf_df is None or len(hf_df) == 0:
            raise ValueError("sph_case 数据读取失败或为空")
        hf_df = hf_df.copy()
        hf_df["case_name"] = case_name

        overlap_start = max(float(lf_df["time"].min()), float(hf_df["time"].min()))
        overlap_end = min(float(lf_df["time"].max()), float(hf_df["time"].max()))
        if overlap_end < overlap_start:
            raise ValueError("custom/sph 时间范围没有重叠，无法做对齐推理")
        lf_df = lf_df[(lf_df["time"] >= overlap_start) & (lf_df["time"] <= overlap_end)].copy()
        hf_df = hf_df[(hf_df["time"] >= overlap_start) & (hf_df["time"] <= overlap_end)].copy()
        if lf_df.empty or hf_df.empty:
            raise ValueError("custom/sph 裁剪到重叠时间段后为空")

        hf_df = add_prefix(hf_df, "hf_")
        df = pd.merge_asof(
            lf_df.sort_values(["case_name", "time"]),
            hf_df.sort_values(["case_name", "time"]),
            on="time",
            by="case_name",
            direction="nearest",
        )
        if df.empty:
            raise ValueError("custom/sph 数据对齐后为空")
        res_cols: Dict[str, pd.Series] = {}
        for hf_col in spec.target_cols:
            if not hf_col.startswith("hf_"):
                continue
            suffix = hf_col[len("hf_"):]
            lf_col = "lf_" + suffix
            if lf_col in df.columns and hf_col in df.columns:
                res_cols["res_" + suffix] = pd.to_numeric(df[hf_col], errors="coerce").fillna(0.0) - pd.to_numeric(df[lf_col], errors="coerce").fillna(0.0)
        if res_cols:
            df = pd.concat([df, pd.DataFrame(res_cols, index=df.index)], axis=1)
        df = add_rotvec_residuals(df, body=True, wheels=range(6))
    else:
        df = lf_df.copy()
        df["case_name"] = case_name

    for col in spec.target_cols + spec.res_cols:
        if col not in df.columns:
            df[col] = 0.0
    for col in spec.base_feature_cols + spec.proxy_feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    ordered_cols = ["time"] + spec.base_feature_cols + spec.proxy_feature_cols + ["case_name"] + spec.target_cols + spec.res_cols
    return df[ordered_cols].sort_values(["case_name", "time"]).reset_index(drop=True)


def discover_case_dirs(parent_dir: str, name_token: str) -> Dict[str, Path]:
    parent = Path(parent_dir)
    if not parent.exists():
        raise FileNotFoundError(f"case 父目录不存在: {parent}")
    if not parent.is_dir():
        raise NotADirectoryError(f"case 父路径不是目录: {parent}")

    cases: Dict[str, Path] = {}
    for child in sorted(parent.iterdir()):
        if not child.is_dir() or name_token not in child.name or "_case" not in child.name:
            continue
        case_name = map_case_basename(child)
        if case_name in cases:
            raise ValueError(f"发现重复 case 名称 {case_name}: {cases[case_name]} 和 {child}")
        cases[case_name] = child
    if not cases:
        raise ValueError(f"未在 {parent} 下找到包含 {name_token} 和 _case 的 case 目录")
    return cases


def build_custom_cases_inference_dataframe(args: argparse.Namespace, spec) -> pd.DataFrame:
    custom_cases = discover_case_dirs(args.custom_cases_dir, "Custom")
    sph_cases = discover_case_dirs(args.sph_cases_dir, "SPH") if args.sph_cases_dir else {}

    missing_sph = sorted(set(custom_cases) - set(sph_cases)) if sph_cases else []
    if missing_sph:
        preview = ", ".join(missing_sph[:10])
        suffix = " ..." if len(missing_sph) > 10 else ""
        raise ValueError(f"以下 Custom case 缺少对应 SPH case: {preview}{suffix}")

    frames = []
    for case_name, custom_dir in tqdm(custom_cases.items(), desc="Build custom cases", dynamic_ncols=True):
        case_args = argparse.Namespace(**vars(args))
        case_args.custom_case_dir = str(custom_dir)
        case_args.sph_case_dir = str(sph_cases[case_name]) if sph_cases else None
        case_args.infer_case_name = case_name
        frames.append(build_custom_inference_dataframe(case_args, spec))
    return pd.concat(frames, ignore_index=True).sort_values(["case_name", "time"]).reset_index(drop=True)


def write_run_info(args: argparse.Namespace, output_dir: str, model_role: str, meta: Dict) -> None:
    lines = [
        f"infer_mode: {args.infer_mode}",
        f"checkpoint: {args.checkpoint}",
        f"model_role: {model_role}",
        f"feature_dir: {args.feature_dir}",
    ]
    if args.infer_mode in ("custom_case", "custom_cases"):
        lines.extend(
            [
                f"custom_case_dir: {args.custom_case_dir}",
                f"sph_case_dir: {args.sph_case_dir or ''}",
                f"custom_cases_dir: {args.custom_cases_dir or ''}",
                f"sph_cases_dir: {args.sph_cases_dir or ''}",
                f"infer_case_name: {args.infer_case_name or ''}",
            ]
        )
    else:
        lines.extend(
            [
                f"merged_csv: {args.merged_csv}",
                f"split: {args.split}",
                f"case_name: {args.case_name or ''}",
            ]
        )
    lines.extend(
        [
            f"batch_size: {args.batch_size}",
            f"num_workers: {args.num_workers}",
            f"device: {args.device or ''}",
            f"history_len: {meta.get('history_len', '')}",
            f"teacher_future_len: {meta.get('teacher_future_len', '')}",
        ]
    )
    with open(os.path.join(output_dir, "run_info.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def compute_group_rmse(pred: np.ndarray, true: np.ndarray) -> float:
    if pred.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def compute_metrics_from_prediction_dataframe(pred_df: pd.DataFrame, spec) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    overall_sqerr_sum = 0.0
    overall_count = 0

    metric_groups = [("hf_body", spec.target_groups.body_cols)]
    for i in WHEEL_IDS:
        metric_groups.append((f"hf_wheel{i}_kin", spec.target_groups.wheel_kin_cols[i]))
        metric_groups.append((f"hf_wheel{i}_contact", spec.target_groups.wheel_contact_cols[i]))

    for group_name, cols in metric_groups:
        pred_series = []
        true_series = []
        for col in cols:
            pred_col = f"pred::{group_name}::{col}"
            true_col = f"true::{group_name}::{col}"
            if pred_col not in pred_df.columns or true_col not in pred_df.columns:
                continue
            pred_series.append(pd.to_numeric(pred_df[pred_col], errors="coerce").to_numpy(dtype=np.float64))
            true_series.append(pd.to_numeric(pred_df[true_col], errors="coerce").to_numpy(dtype=np.float64))
        if not pred_series or not true_series:
            continue

        pred_arr = np.column_stack(pred_series)
        true_arr = np.column_stack(true_series)
        valid_mask = np.isfinite(pred_arr).all(axis=1) & np.isfinite(true_arr).all(axis=1)
        if not np.any(valid_mask):
            continue
        pred_arr = pred_arr[valid_mask]
        true_arr = true_arr[valid_mask]
        metrics[f"{group_name}_rmse"] = compute_group_rmse(pred_arr, true_arr)
        overall_sqerr_sum += float(np.square(pred_arr - true_arr).sum())
        overall_count += int(pred_arr.size)

    if overall_count > 0:
        metrics["overall_rmse"] = float(np.sqrt(overall_sqerr_sum / overall_count))
    return metrics


def sanitize_plot_name(value: str) -> str:
    return safe_path_name(value)


def integrate_position_from_velocity(time: np.ndarray, velocity: np.ndarray, initial_position: float) -> np.ndarray:
    integrated = np.full_like(velocity, np.nan, dtype=np.float64)
    valid = np.isfinite(time) & np.isfinite(velocity)
    if not np.isfinite(initial_position) or not valid.any():
        return integrated

    integrated[0] = float(initial_position)
    for i in range(1, len(integrated)):
        if not (np.isfinite(time[i]) and np.isfinite(time[i - 1]) and np.isfinite(velocity[i]) and np.isfinite(velocity[i - 1])):
            integrated[i] = integrated[i - 1]
            continue
        dt = max(0.0, float(time[i] - time[i - 1]))
        integrated[i] = integrated[i - 1] + 0.5 * (velocity[i - 1] + velocity[i]) * dt
    return integrated


def maybe_add_velocity_integrated_body_position(
    series_map: Dict[str, np.ndarray],
    case_df: pd.DataFrame,
    time: np.ndarray,
    group_name: str,
    suffix: str,
) -> None:
    if group_name != "hf_body" or suffix not in {"hf_pos_x", "hf_pos_y", "hf_pos_z"}:
        return

    axis = suffix.rsplit("_", 1)[-1]
    integrated_col = f"pred_vel_integrated::hf_body::{suffix}"
    if integrated_col in case_df.columns:
        series_map["Velocity integrated"] = pd.to_numeric(case_df[integrated_col], errors="coerce").to_numpy(dtype=np.float64)
        return
    vel_col = f"pred::hf_body::hf_vel_{axis}"
    init_col = f"custom::hf_body::{suffix}"
    fallback_col = f"pred::hf_body::{suffix}"
    if vel_col not in case_df.columns:
        return

    velocity = pd.to_numeric(case_df[vel_col], errors="coerce").to_numpy(dtype=np.float64)
    if init_col in case_df.columns:
        initial_position = pd.to_numeric(case_df[init_col], errors="coerce").to_numpy(dtype=np.float64)[0]
    elif fallback_col in case_df.columns:
        initial_position = pd.to_numeric(case_df[fallback_col], errors="coerce").to_numpy(dtype=np.float64)[0]
    else:
        return
    series_map["Velocity integrated"] = integrate_position_from_velocity(time, velocity, initial_position)


def add_body_velocity_integrated_columns(pred_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty or "case_name" not in pred_df.columns or "time" not in pred_df.columns:
        return pred_df
    out = pred_df.copy()
    for _, idx in out.groupby("case_name", sort=False).groups.items():
        case_idx = list(idx)
        case_df = out.loc[case_idx].sort_values("time")
        ordered_idx = case_df.index.to_list()
        time = pd.to_numeric(case_df["time"], errors="coerce").to_numpy(dtype=np.float64)
        for axis in ["x", "y", "z"]:
            suffix = f"hf_pos_{axis}"
            vel_col = f"pred::hf_body::hf_vel_{axis}"
            init_col = f"custom::hf_body::{suffix}"
            out_col = f"pred_vel_integrated::hf_body::{suffix}"
            if vel_col not in case_df.columns or init_col not in case_df.columns:
                continue
            velocity = pd.to_numeric(case_df[vel_col], errors="coerce").to_numpy(dtype=np.float64)
            initial_position = pd.to_numeric(case_df[init_col], errors="coerce").to_numpy(dtype=np.float64)[0]
            out.loc[ordered_idx, out_col] = integrate_position_from_velocity(time, velocity, initial_position)
    return out


def corrcoef_finite(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 2:
        return float("nan")
    aa = a[mask]
    bb = b[mask]
    if np.std(aa) <= 1e-12 or np.std(bb) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def compute_body_position_diagnostics(pred_df: pd.DataFrame) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if pred_df.empty or "case_name" not in pred_df.columns or "time" not in pred_df.columns:
        return metrics
    for axis in ["x", "y", "z"]:
        pos_suffix = f"hf_pos_{axis}"
        pred_col = f"pred::hf_body::{pos_suffix}"
        integrated_col = f"pred_vel_integrated::hf_body::{pos_suffix}"
        true_col = f"sph_ref::hf_body::{pos_suffix}" if f"sph_ref::hf_body::{pos_suffix}" in pred_df.columns else f"true::hf_body::{pos_suffix}"
        custom_col = f"custom::hf_body::{pos_suffix}"
        if true_col in pred_df.columns:
            true = pd.to_numeric(pred_df[true_col], errors="coerce").to_numpy(dtype=np.float64)
            for label, col in [("body_pos", pred_col), ("body_pos_vel_integrated", integrated_col), ("body_pos_lf", custom_col)]:
                if col not in pred_df.columns:
                    continue
                values = pd.to_numeric(pred_df[col], errors="coerce").to_numpy(dtype=np.float64)
                mask = np.isfinite(values) & np.isfinite(true)
                if mask.any():
                    metrics[f"{label}_{axis}_rmse"] = float(np.sqrt(np.mean((values[mask] - true[mask]) ** 2)))
        vel_col = f"pred::hf_body::hf_vel_{axis}"
        if pred_col in pred_df.columns and vel_col in pred_df.columns:
            corrs = []
            for _, case_df in pred_df.groupby("case_name", sort=False):
                case_df = case_df.sort_values("time")
                time = pd.to_numeric(case_df["time"], errors="coerce").to_numpy(dtype=np.float64)
                pos = pd.to_numeric(case_df[pred_col], errors="coerce").to_numpy(dtype=np.float64)
                vel = pd.to_numeric(case_df[vel_col], errors="coerce").to_numpy(dtype=np.float64)
                dt = np.diff(time)
                valid_dt = np.isfinite(dt) & (dt > 1e-12)
                diff_vel = np.full_like(dt, np.nan, dtype=np.float64)
                diff_vel[valid_dt] = np.diff(pos)[valid_dt] / dt[valid_dt]
                corrs.append(corrcoef_finite(diff_vel, vel[:-1]))
            finite_corrs = [c for c in corrs if np.isfinite(c)]
            if finite_corrs:
                metrics[f"body_pos_vel_consistency_corr_{axis}"] = float(np.mean(finite_corrs))
    pred_x = "pred::hf_body::hf_pos_x"
    if pred_x in pred_df.columns:
        back_counts = []
        max_backs = []
        for _, case_df in pred_df.groupby("case_name", sort=False):
            x = pd.to_numeric(case_df.sort_values("time")[pred_x], errors="coerce").to_numpy(dtype=np.float64)
            dx = np.diff(x)
            back = dx[np.isfinite(dx) & (dx < 0.0)]
            back_counts.append(int(back.size))
            max_backs.append(float((-back).max()) if back.size else 0.0)
        metrics["body_pos_x_backward_count"] = int(sum(back_counts))
        metrics["body_pos_x_max_backward"] = float(max(max_backs) if max_backs else 0.0)
    return metrics


def generate_inference_plots(pred_df: pd.DataFrame, output_dir: str) -> None:
    if pred_df.empty or "case_name" not in pred_df.columns or "time" not in pred_df.columns:
        return

    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    group_suffixes = {
        "hf_body": ["hf_pos_x", "hf_pos_y", "hf_pos_z", "hf_vel_x", "hf_vel_y", "hf_vel_z"],
        **{f"hf_wheel{i}_kin": [f"hf_wheel{i}_pos_x", f"hf_wheel{i}_pos_y", f"hf_wheel{i}_pos_z", f"hf_wheel{i}_ang_vel_z"] for i in WHEEL_IDS},
        **{f"hf_wheel{i}_contact": [f"hf_wheel{i}_Fx", f"hf_wheel{i}_Fy", f"hf_wheel{i}_Fz"] for i in WHEEL_IDS},
    }
    reference_prefixes = ["true", "custom", "sph_ref"]
    label_map = {"true": "HF true", "custom": "LF current", "sph_ref": "SPH ref"}
    plot_styles = {
        "HF true": {"linewidth": 2.0, "alpha": 0.95, "linestyle": "-"},
        "Compensated": {"linewidth": 1.8, "alpha": 0.95, "linestyle": "-"},
        "Velocity integrated": {"linewidth": 1.8, "alpha": 0.9, "linestyle": "-."},
        "LF current": {"linewidth": 1.5, "alpha": 0.5, "linestyle": "--"},
        "SPH ref": {"linewidth": 1.5, "alpha": 0.45, "linestyle": ":"},
    }

    for case_name, case_df in pred_df.groupby("case_name", sort=False):
        case_df = case_df.sort_values("time").reset_index(drop=True)
        x = pd.to_numeric(case_df["time"], errors="coerce").to_numpy(dtype=np.float64)
        case_dir = plots_dir / sanitize_plot_name(case_name)
        case_dir.mkdir(parents=True, exist_ok=True)

        for group_name, suffixes in group_suffixes.items():
            for suffix in suffixes:
                series_map = {}
                pred_col = f"pred::{group_name}::{suffix}"
                if pred_col in case_df.columns:
                    series_map["Compensated"] = pd.to_numeric(case_df[pred_col], errors="coerce").to_numpy(dtype=np.float64)
                for prefix in reference_prefixes:
                    col = f"{prefix}::{group_name}::{suffix}"
                    if col in case_df.columns:
                        series_map[label_map[prefix]] = pd.to_numeric(case_df[col], errors="coerce").to_numpy(dtype=np.float64)
                maybe_add_velocity_integrated_body_position(series_map, case_df, x, group_name, suffix)
                if "Compensated" not in series_map or "HF true" not in series_map:
                    continue

                fig = plt.figure(figsize=(10, 4))
                for label in ["HF true", "Compensated", "Velocity integrated", "LF current", "SPH ref"]:
                    values = series_map.get(label)
                    if values is None:
                        continue
                    plt.plot(x, values, label=label, **plot_styles[label])
                plt.xlabel("time / s")
                plt.ylabel(suffix)
                plt.title(f"{case_name} | {group_name} | {suffix}")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                file_name = f"{sanitize_plot_name(group_name)}__{sanitize_plot_name(suffix)}.png"
                plt.savefig(case_dir / file_name, dpi=150)
                plt.close(fig)


def restore_scaler_v4(checkpoint_path: str, ckpt: Dict) -> GroupStandardizer:
    scaler_state = ckpt.get("scaler_state")
    if scaler_state:
        scaler = GroupStandardizer()
        scaler.scalers = {name: NumpyStandardScaler.from_dict(state) for name, state in scaler_state.items()}
        return scaler

    ckpt_dir = Path(checkpoint_path).resolve().parent
    path = ckpt_dir / "group_scaler_v4.joblib"
    if path.exists():
        return GroupStandardizer.load(str(path))
    raise FileNotFoundError(f"未在 checkpoint 或 {ckpt_dir} 下找到 group_scaler_v4.joblib")


def resolve_model_role_v4(model_role: str, ckpt: Dict) -> str:
    if model_role != "auto":
        return model_role
    train_stage = str((ckpt.get("args") or {}).get("train_stage", ckpt.get("train_stage", "")))
    return "teacher" if train_stage == "teacher" else "student"


def build_model_from_checkpoint_v4(
    ckpt: Dict,
    spec,
    model_role: str,
    device: torch.device,
) -> Tuple[GraphTemporalHGTCompensationModelV4, Dict]:
    ckpt_args = ckpt.get("args") or {}
    group_dims = ckpt.get("group_dims") or get_group_dims(spec)

    hidden_dim = int(get_arg_with_default(ckpt_args, "hidden_dim", None, 128))
    graph_layers = int(get_arg_with_default(ckpt_args, "graph_layers", None, 3))
    tcn_dim = int(get_arg_with_default(ckpt_args, "tcn_dim", None, 192))
    lstm_dim = int(get_arg_with_default(ckpt_args, "lstm_dim", None, 128))
    lstm_layers = int(get_arg_with_default(ckpt_args, "lstm_layers", None, 2))
    dropout = float(get_arg_with_default(ckpt_args, "dropout", None, 0.1))
    use_force_tcn = bool(get_arg_with_default(ckpt_args, "use_force_tcn", None, True))
    force_tcn_hidden = int(get_arg_with_default(ckpt_args, "force_tcn_hidden", None, 64))
    force_tcn_kernel_size = int(get_arg_with_default(ckpt_args, "force_tcn_kernel_size", None, 3))
    force_tcn_dilations = list(get_arg_with_default(ckpt_args, "force_tcn_dilations", None, [1, 2, 4]))
    force_tcn_dropout = float(get_arg_with_default(ckpt_args, "force_tcn_dropout", None, 0.1))

    model = GraphTemporalHGTCompensationModelV4(
        group_dims=group_dims,
        node_hidden_dim=hidden_dim,
        graph_layers=graph_layers,
        tcn_hidden_dim=tcn_dim,
        lstm_hidden_dim=lstm_dim,
        lstm_layers=lstm_layers,
        dropout=dropout,
        role=model_role,
        use_force_tcn=use_force_tcn,
        force_tcn_hidden=force_tcn_hidden,
        force_tcn_kernel_size=force_tcn_kernel_size,
        force_tcn_dilations=force_tcn_dilations,
        force_tcn_dropout=force_tcn_dropout,
    ).to(device)
    model_state = ckpt["model_state_dict"]
    uses_safety_gates = any("gate_head" in key for key in model_state)
    if not uses_safety_gates:
        raise RuntimeError("V4 inference requires checkpoint with trained safety gate heads; old no-gate checkpoints are not supported")
    load_compatible_state_dict(model, model_state, "model inference checkpoint")
    model.eval()

    meta = {
        "group_dims": group_dims,
        "pred_seq_len": 1,
        "model_role": model_role,
        "hidden_dim": hidden_dim,
        "tcn_dim": tcn_dim,
        "lstm_dim": lstm_dim,
        "history_len": int(get_arg_with_default(ckpt_args, "history_len", None, 9)),
        "teacher_future_len": int(get_arg_with_default(ckpt_args, "teacher_future_len", None, 3)),
        "uses_safety_gates": uses_safety_gates,
        "use_force_tcn": use_force_tcn,
        "force_tcn_hidden": force_tcn_hidden,
        "force_tcn_kernel_size": force_tcn_kernel_size,
        "force_tcn_dilations": force_tcn_dilations,
        "force_tcn_dropout": force_tcn_dropout,
    }
    return model, meta


def build_v4_contact_prediction_raw(
    batch: Dict[str, torch.Tensor],
    output: Dict[str, torch.Tensor],
    spec,
    scaler,
    pred_body_raw: torch.Tensor,
    pred_wheel_raw: Dict[int, torch.Tensor],
    wheel_id: int,
    sinkage_max: float,
) -> torch.Tensor:
    wheel_cols = spec.target_groups.wheel_kin_cols[wheel_id]
    wpos = suffix_indices(wheel_cols, ["pos_x", "pos_y", "pos_z"])
    input_contact_cols = spec.input_groups.wheel_contact_cols[wheel_id]
    input_contact_raw = inverse_transform_tensor(batch[f"wheel{wheel_id}_contact"], scaler, f"wheel{wheel_id}_contact")
    sink_idx = suffix_indices(input_contact_cols, ["sinkage"])
    sinkage_lf = input_contact_raw[:, -1:, sink_idx[0]] if sink_idx else pred_body_raw.new_zeros(*pred_body_raw.shape[:2])
    wheel_z_lf = batch[f"lf_wheel{wheel_id}_kin_current"][..., wpos[2]] if len(wpos) == 3 else sinkage_lf.new_zeros(sinkage_lf.shape)
    wheel_z_pred = pred_wheel_raw[wheel_id][..., wpos[2]] if len(wpos) == 3 else wheel_z_lf

    f_delta = output[f"pred_force_delta_wheel{wheel_id}"]
    contact_cols = spec.target_groups.wheel_contact_cols[wheel_id]
    force_idx = suffix_indices(contact_cols, ["Fx", "Fy", "Fz"])
    if len(force_idx) != 3:
        return f_delta
    lf_force = batch[f"lf_wheel{wheel_id}_contact_current"][..., force_idx].float()
    gate = output[f"gate_force_wheel{wheel_id}"]
    return lf_force + gate * f_delta


def run_inference_v4(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    log_status(f"start: checkpoint={args.checkpoint}, mode={args.infer_mode}, split={args.split}, device={device}")
    ckpt = safe_load_checkpoint(args.checkpoint, device)
    log_status("checkpoint loaded")
    spec = load_column_spec(args.feature_dir)
    log_status("column spec loaded")
    scaler = restore_scaler_v4(args.checkpoint, ckpt)
    log_status("scaler loaded")
    ensure_dir(args.output_dir)
    log_status(f"output_dir={args.output_dir}")

    ckpt_args = ckpt.get("args") or {}
    seed = int(get_arg_with_default(ckpt_args, "seed", args.seed, 42))
    history_len = int(get_arg_with_default(ckpt_args, "history_len", args.history_len, 9))
    teacher_future_len_ckpt = int(get_arg_with_default(ckpt_args, "teacher_future_len", args.teacher_future_len, 3))
    pred_horizon = int(get_arg_with_default(ckpt_args, "pred_horizon", args.pred_horizon, 0))
    train_ratio = float(get_arg_with_default(ckpt_args, "train_ratio", args.train_ratio, 0.7))
    val_ratio = float(get_arg_with_default(ckpt_args, "val_ratio", args.val_ratio, 0.15))
    sinkage_max = float(get_arg_with_default(ckpt_args, "sinkage_max", args.sinkage_max, 0.08))
    kinematic_pos_gate_scale = float(get_arg_with_default(ckpt_args, "kinematic_pos_gate_scale", args.kinematic_pos_gate_scale, 1.0))

    if args.infer_mode == "custom_case":
        args.split = "all"
        if args.case_name is None and args.infer_case_name is not None:
            args.case_name = args.infer_case_name
        df = build_custom_inference_dataframe(args, spec)
        infer_csv_path = os.path.join(args.output_dir, "generated_infer_input.csv")
        df.to_csv(infer_csv_path, index=False)
    elif args.infer_mode == "custom_cases":
        args.split = "all"
        df = build_custom_cases_inference_dataframe(args, spec)
        infer_csv_path = os.path.join(args.output_dir, "generated_infer_input.csv")
        df.to_csv(infer_csv_path, index=False)
    else:
        df = load_selected_merged_dataset(
            args.merged_csv,
            spec,
            args.split,
            seed,
            train_ratio,
            val_ratio,
            args.case_name,
            args.csv_chunksize,
        )
        infer_csv_path = None
    log_status(f"source rows={len(df)}")

    selected_df = select_dataframe(df, "all" if args.infer_mode == "merged_csv" else args.split, seed, train_ratio, val_ratio, args.case_name)
    log_status(f"selected rows={len(selected_df)}, cases={selected_df['case_name'].astype(str).nunique()}")
    model_role = resolve_model_role_v4(args.model_role, ckpt)
    infer_teacher_future_len = teacher_future_len_ckpt if model_role == "teacher" else 0
    dataset = GraphTemporalSequenceDatasetV4(
        selected_df,
        spec,
        scaler,
        seq_len=history_len + 1,
        pred_horizon=pred_horizon,
        pred_seq_len=1,
        case_col="case_name",
        time_col="time",
        history_len=history_len,
        teacher_future_len=infer_teacher_future_len,
    )
    if args.max_samples is not None and args.max_samples > 0:
        dataset.index_map = dataset.index_map[: args.max_samples]
    log_status(f"dataset samples={len(dataset)}, model_role={model_role}, history_len={history_len}")

    loader_kwargs = {
        "num_workers": args.num_workers,
        "collate_fn": graph_temporal_collate_fn,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    model, meta = build_model_from_checkpoint_v4(ckpt, spec, model_role, device)
    log_status("model loaded; starting batches")
    meta["history_len"] = history_len
    meta["teacher_future_len"] = infer_teacher_future_len

    all_rows: List[Dict[str, float]] = []
    metrics_store: Dict[str, List[np.ndarray]] = {}
    reference_prefix = "true"
    if args.infer_mode in {"custom_case", "custom_cases"} and (args.sph_case_dir or args.sph_cases_dir):
        reference_prefix = "sph_ref"

    for batch in tqdm(loader, desc="Infer V4", dynamic_ncols=True):
        batch = move_batch_to_device(batch, device)
        with torch.no_grad():
            prefix = "teacher" if model_role == "teacher" else "student"
            output = model(batch, input_prefix=prefix, current_index=history_len)

        dt_batch = batch.get("dt", output["pred_res_body"].new_full((output["pred_res_body"].shape[0],), args.dt))
        prev_body_raw = batch.get("lf_body_prev_raw", batch["lf_body_current"])
        pred_body_scaled, pred_body_raw, body_delta_raw = reconstruct_body_kinematic(
            output["pred_res_body"],
            batch["lf_body_current"],
            prev_body_raw,
            dt_batch,
            spec.res_groups.body_cols,
            spec.target_groups.body_cols,
            scaler,
        )
        body_pos_idx = suffix_indices(spec.target_groups.body_cols, ["pos_x", "pos_y", "pos_z"])
        if len(body_pos_idx) == 3:
            body_gate = output["gate_body_pos"] * max(0.0, min(1.0, kinematic_pos_gate_scale))
            pred_body_raw = apply_axis_gate(pred_body_raw, batch["lf_body_current"], body_pos_idx, body_gate)
        lf_body_pos = batch["lf_body_current"][..., body_pos_idx].float() if len(body_pos_idx) == 3 else pred_body_raw.new_zeros(*pred_body_raw.shape[:2], 3)
        body_global_delta = pred_body_raw[..., body_pos_idx] - lf_body_pos if len(body_pos_idx) == 3 else pred_body_raw.new_zeros(*pred_body_raw.shape[:2], 3)
        pred_wheel_raw: Dict[int, torch.Tensor] = {}
        pred_wheel_scaled: Dict[int, torch.Tensor] = {}
        for i in WHEEL_IDS:
            scaled, raw, _ = reconstruct_wheel_kin_with_body_delta(
                output[f"pred_res_wheel{i}_kin"],
                batch[f"lf_wheel{i}_kin_current"],
                body_global_delta,
                spec.res_groups.wheel_kin_cols[i],
                spec.target_groups.wheel_kin_cols[i],
                f"hf_wheel{i}_kin",
                f"res_wheel{i}_kin",
                scaler,
            )
            wheel_cols = spec.target_groups.wheel_kin_cols[i]
            wpos = suffix_indices(wheel_cols, ["pos_x", "pos_y", "pos_z"])
            omega_i = omega_index(wheel_cols)
            if len(wpos) == 3:
                wheel_pos_gate = output[f"gate_wheel{i}_pos"] * max(0.0, min(1.0, kinematic_pos_gate_scale))
                raw = apply_axis_gate(raw, batch[f"lf_wheel{i}_kin_current"], wpos, wheel_pos_gate)
            if omega_i is not None:
                raw = apply_axis_gate(raw, batch[f"lf_wheel{i}_kin_current"], [omega_i], output[f"gate_wheel{i}_omega"])
            pred_wheel_scaled[i] = scaled
            pred_wheel_raw[i] = raw

        batch_row_offset = len(all_rows)
        group_raw_defs = [("hf_body", pred_body_raw, batch["lf_body_current"], batch["hf_body"], spec.target_groups.body_cols)]
        for i in WHEEL_IDS:
            group_raw_defs.append((f"hf_wheel{i}_kin", pred_wheel_raw[i], batch[f"lf_wheel{i}_kin_current"], batch[f"hf_wheel{i}_kin"], spec.target_groups.wheel_kin_cols[i]))
            pred_contact_raw = build_v4_contact_prediction_raw(batch, output, spec, scaler, pred_body_raw, pred_wheel_raw, i, sinkage_max)
            group_raw_defs.append((f"hf_wheel{i}_contact", pred_contact_raw, batch[f"lf_wheel{i}_contact_current"], batch[f"hf_wheel{i}_contact"], spec.target_groups.wheel_contact_cols[i]))

        for group_name, pred_raw, lf_current_raw, true_scaled, hf_cols in group_raw_defs:
            true_raw = inverse_transform_tensor(true_scaled, scaler, group_name)
            flatten_group_rows(all_rows, batch_row_offset, batch["case_name"], batch["time"], pred_raw, true_raw, hf_cols, group_name)
            flatten_reference_rows(all_rows, batch_row_offset, batch["time"], lf_current_raw, hf_cols, group_name, prefix="custom")
            if reference_prefix != "true":
                flatten_reference_rows(all_rows, batch_row_offset, batch["time"], true_raw, hf_cols, group_name, prefix=reference_prefix)
            metrics_store.setdefault(f"{group_name}::pred", []).append(pred_raw.detach().cpu().numpy())
            metrics_store.setdefault(f"{group_name}::true", []).append(true_raw.detach().cpu().numpy())

    pred_df = add_body_velocity_integrated_columns(pd.DataFrame(all_rows))
    write_run_info(args, args.output_dir, model_role, meta)
    pred_path = os.path.join(args.output_dir, "predictions_v4.csv")
    pred_df.to_csv(pred_path, index=False)
    log_status(f"predictions saved: {pred_path}")
    generate_inference_plots(pred_df, args.output_dir)
    log_status("plots generated")
    metrics = compute_metrics_from_prediction_dataframe(pred_df, spec)
    metrics.update(compute_body_position_diagnostics(pred_df))

    summary = {
        "checkpoint": args.checkpoint,
        "infer_mode": args.infer_mode,
        "model_role": model_role,
        "split": args.split,
        "case_name": args.case_name,
        "device": str(device),
        "num_samples": int(len(dataset)),
        "pred_seq_len": 1,
        "metrics": metrics,
        "body_position_mode": "kinematic_velocity_integration_plus_body_delta",
        "uses_future_lf": bool(model_role == "teacher"),
        "teacher_t_index": int(history_len),
        "kinematic_pos_gate_scale": float(kinematic_pos_gate_scale),
    }
    metrics_path = os.path.join(args.output_dir, "metrics_v4.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if infer_csv_path is not None:
        print(f"generated infer csv saved to: {infer_csv_path}")
    print(f"predictions saved to: {pred_path}")
    print(f"metrics saved to: {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="V4 训练得到的 best_model.pt")
    parser.add_argument("--feature_dir", type=str, default=str(ROOT / "Feature_Selection" / "DataSet"))
    parser.add_argument("--merged_csv", type=str, default=str(ROOT / "Feature_Selection" / "DataSet" / "merged_error_dataset.csv"))
    parser.add_argument("--infer_mode", type=str, choices=["merged_csv", "custom_case", "custom_cases"], default="custom_cases")
    parser.add_argument("--custom_case_dir", type=str, default=None)
    parser.add_argument("--sph_case_dir", type=str, default=None)
    parser.add_argument("--custom_cases_dir", type=str, default="/media/user/新加卷/RoverSimData/Multi_Custom_test_output")
    parser.add_argument("--sph_cases_dir", type=str, default="/media/user/新加卷/RoverSimData/Multi_SPH_test_output")
    parser.add_argument("--infer_case_name", type=str, default=None)
    parser.add_argument("--control_params", type=float, nargs=12, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--split", type=str, choices=["all", "train", "val", "test"], default="test")
    parser.add_argument("--case_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=1, help="num_workers>0 时每个 worker 预取 batch 数；降低可减少共享内存占用")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model_role", type=str, choices=["auto", "student", "teacher"], default="auto")
    parser.add_argument("--history_len", type=int, default=None)
    parser.add_argument("--teacher_future_len", type=int, default=None)
    parser.add_argument("--pred_horizon", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train_ratio", type=float, default=None)
    parser.add_argument("--val_ratio", type=float, default=None)
    parser.add_argument("--csv_chunksize", type=int, default=200000, help="merged_csv 推理按块读取的行数")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--sinkage_max", type=float, default=0.08)
    parser.add_argument("--dt", type=float, default=0.015)
    parser.add_argument("--kinematic_pos_gate_scale", type=float, default=None, help="推理时 body/wheel 位置 gate 的全局上限缩放；默认使用 checkpoint，缺省为 1.0")
    args = parser.parse_args()

    if args.output_dir is None:
        ckpt_dir = Path(args.checkpoint).resolve().parent
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        case_label = safe_path_name(infer_output_case_label(args))
        args.output_dir = str(ckpt_dir / f"infer_{case_label}_{ts}")
    return args


def main() -> None:
    run_inference_v4(parse_args())


if __name__ == "__main__":
    main()
