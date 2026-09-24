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
    TIME_TOL,
    add_prefix,
    add_rotvec_residuals,
    build_cmd_speed,
    read_case_data,
)
from models.data_utils import (  # noqa: E402
    GroupStandardizer,
    GraphTemporalSequenceDataset,
    NumpyStandardScaler,
    WHEEL_IDS,
    get_group_dims,
    graph_temporal_collate_fn,
    load_column_spec,
    load_merged_dataset,
    split_train_val_test_by_case,
)
from models.graph_temporal_hgt_compensation import (  # noqa: E402
    GraphTemporalHGTCompensationModel,
    TeacherGateNet,
)
from train.train_graph_model import (  # noqa: E402
    TEACHER_GATE_SUMMARY_DIM,
    build_model_group_columns,
    build_teacher_relation_gate,
    inverse_transform_tensor,
    reconstruct_hf_scaled,
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out: Dict = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def restore_scaler_from_checkpoint(ckpt: Dict) -> GroupStandardizer:
    scaler_state = ckpt.get("scaler_state")
    if not scaler_state:
        raise KeyError("checkpoint 中缺少 scaler_state，无法保证推理标准化与训练一致")
    scaler = GroupStandardizer()
    scaler.scalers = {
        name: NumpyStandardScaler.from_dict(state)
        for name, state in scaler_state.items()
    }
    return scaler


def infer_teacher_hidden_dim(ckpt: Dict, cli_value: Optional[int]) -> int:
    if cli_value is not None:
        return int(cli_value)
    teacher_state = ckpt.get("teacher_gate_state_dict") or {}
    weight = teacher_state.get("net.0.weight")
    if isinstance(weight, torch.Tensor):
        return int(weight.shape[0])
    args = ckpt.get("args") or {}
    if "teacher_hidden_dim" in args:
        return int(args["teacher_hidden_dim"])
    return 128


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
        split_map = {
            "train": df_train,
            "val": df_val,
            "test": df_test,
        }
        selected = split_map[split]
    if case_name is not None:
        selected = selected[selected["case_name"].astype(str) == str(case_name)].copy()
    selected = selected.sort_values(["case_name", "time"]).reset_index(drop=True)
    if len(selected) == 0:
        raise ValueError("筛选后的推理数据为空，请检查 split / case_name 参数")
    return selected


def resolve_gate_mode(gate_mode: str, ckpt: Dict) -> str:
    if gate_mode != "auto":
        return gate_mode
    train_stage = ckpt.get("train_stage")
    if train_stage == "teacher_full":
        return "teacher"
    return "student"


def safe_load_checkpoint(path: str, device: torch.device) -> Dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model_from_checkpoint(
    ckpt: Dict,
    spec,
    gate_mode: str,
    teacher_hidden_dim: Optional[int],
    device: torch.device,
) -> Tuple[GraphTemporalHGTCompensationModel, Optional[TeacherGateNet], Dict]:
    ckpt_args = ckpt.get("args") or {}
    group_dims = ckpt.get("group_dims") or get_group_dims(spec)
    group_columns = build_model_group_columns(spec)

    hidden_dim = int(get_arg_with_default(ckpt_args, "hidden_dim", None, 96))
    graph_layers = int(get_arg_with_default(ckpt_args, "graph_layers", None, 3))
    tcn_dim = int(get_arg_with_default(ckpt_args, "tcn_dim", None, 192))
    lstm_dim = int(get_arg_with_default(ckpt_args, "lstm_dim", None, 192))
    lstm_layers = int(get_arg_with_default(ckpt_args, "lstm_layers", None, 2))
    dropout = float(get_arg_with_default(ckpt_args, "dropout", None, 0.1))
    pred_seq_len = int(get_arg_with_default(ckpt_args, "pred_seq_len", None, 1))
    teacher_hidden_dim_resolved = None
    if gate_mode == "teacher":
        teacher_hidden_dim_resolved = infer_teacher_hidden_dim(ckpt, teacher_hidden_dim)

    model = GraphTemporalHGTCompensationModel(
        group_dims=group_dims,
        node_hidden_dim=hidden_dim,
        graph_layers=graph_layers,
        tcn_hidden_dim=tcn_dim,
        lstm_hidden_dim=lstm_dim,
        lstm_layers=lstm_layers,
        dropout=dropout,
        enable_relation_gate=True,
        enable_edge_gate=True,
        pred_seq_len=pred_seq_len,
        group_columns=group_columns,
        gate_summary_dim=TEACHER_GATE_SUMMARY_DIM,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    teacher_gate_net: Optional[TeacherGateNet] = None
    if gate_mode == "teacher":
        teacher_gate_net = TeacherGateNet(
            in_dim=TEACHER_GATE_SUMMARY_DIM,
            num_relations=model.num_relations,
            hidden_dim=int(teacher_hidden_dim_resolved),
            dropout=dropout,
        ).to(device)
        teacher_state = ckpt.get("teacher_gate_state_dict")
        if teacher_state is None:
            raise KeyError("teacher 模式推理需要 checkpoint 中包含 teacher_gate_state_dict")
        teacher_gate_net.load_state_dict(teacher_state, strict=False)
        teacher_gate_net.eval()

    meta = {
        "group_dims": group_dims,
        "pred_seq_len": pred_seq_len,
        "teacher_hidden_dim": teacher_hidden_dim_resolved,
    }
    return model, teacher_gate_net, meta


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


def compute_group_rmse(pred: np.ndarray, true: np.ndarray) -> float:
    if pred.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((pred - true) ** 2)))


BODY_STATE_COLUMNS = [
    "pos_x", "pos_y", "pos_z",
    "q0", "q1", "q2", "q3",
    "vel_x", "vel_y", "vel_z",
    "acc_x", "acc_y", "acc_z",
]
BODY_PROXY_COLUMNS = [
    "ang_vel_x", "ang_vel_y", "ang_vel_z",
    "ang_acc_x", "ang_acc_y", "ang_acc_z",
]
WHEEL_STATE_COLUMNS = [
    "pos_x", "pos_y", "pos_z",
    "q0", "q1", "q2", "q3",
    "vel_x", "vel_y", "vel_z",
    "acc_x", "acc_y", "acc_z",
    "ang_vel_x", "ang_vel_y", "ang_vel_z",
    "ang_acc_x", "ang_acc_y", "ang_acc_z",
    "Fx", "Fy", "Fz", "Mx", "My", "Mz",
    "slip_long", "slip_lat", "sinkage", "in_contact",
]
ROCKER_FILE_TO_CODE = {
    "rocker_LB_body.csv": "lb",
    "rocker_LF_body.csv": "lf",
    "rocker_LM_body.csv": "lm",
    "rocker_RB_body.csv": "rb",
    "rocker_RF_body.csv": "rf",
    "rocker_RM_body.csv": "rm",
}
EPS = 1e-8
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


def normalize_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "time" not in out.columns:
        raise KeyError("原始 case 数据缺少 time 列")
    out["time"] = pd.to_numeric(out["time"], errors="coerce")
    out = out.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    if "dt" in out.columns:
        out["dt"] = pd.to_numeric(out["dt"], errors="coerce")
    return out


def infer_dt_series(df: pd.DataFrame) -> pd.Series:
    if "dt" in df.columns:
        dt = pd.to_numeric(df["dt"], errors="coerce")
    else:
        dt = pd.Series(np.nan, index=df.index, dtype=float)
    time_diff = df["time"].diff()
    dt = dt.fillna(time_diff)
    if len(dt) > 1 and pd.isna(dt.iloc[0]):
        dt.iloc[0] = dt.iloc[1]
    dt = dt.fillna(0.0).astype(float)
    return dt


def subtract_initial_position(df: pd.DataFrame, position_cols: List[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    # x/y use relative displacement for cross-case alignment; z stays in world height
    # so LF/HF initial height differences remain comparable to the terrain plane.
    cols = position_cols or ["pos_x", "pos_y"]
    for col in cols:
        if col not in out.columns or out.empty:
            continue
        series = pd.to_numeric(out[col], errors="coerce")
        first_valid = series.dropna()
        base = float(first_valid.iloc[0]) if not first_valid.empty else 0.0
        out[col] = series.fillna(base) - base
    return out


def safe_gradient(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.zeros_like(values, dtype=np.float64)
    if len(values) == 1:
        return np.zeros_like(values, dtype=np.float64)
    try:
        return np.gradient(values.astype(np.float64), times.astype(np.float64), edge_order=1)
    except Exception:
        return np.zeros_like(values, dtype=np.float64)


def normalize_quaternion_array(quat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quat, axis=1, keepdims=True)
    norms = np.where(norms < EPS, 1.0, norms)
    quat = quat / norms
    if len(quat):
        ref = quat[0].copy()
        for i in range(1, len(quat)):
            if np.dot(ref, quat[i]) < 0.0:
                quat[i] = -quat[i]
            ref = quat[i]
    return quat


def quaternion_to_euler_xyz(quat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    quat = normalize_quaternion_array(quat.copy())
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.unwrap(np.arctan2(siny_cosp, cosy_cosp))
    return roll, pitch, yaw


def align_to_times(df: pd.DataFrame, target_times: np.ndarray, columns: List[str]) -> pd.DataFrame:
    work = normalize_time_columns(df)
    aligned = pd.DataFrame({"time": target_times})
    source = work[["time"] + [c for c in columns if c in work.columns]].copy()
    merged = pd.merge_asof(
        aligned.sort_values("time"),
        source.sort_values("time"),
        on="time",
        direction="nearest",
    )
    return merged


def build_body_frame(csv_path: Path, prefix: str) -> pd.DataFrame:
    df = normalize_time_columns(pd.read_csv(csv_path))
    df = subtract_initial_position(df)
    data: Dict[str, np.ndarray] = {"time": df["time"].to_numpy(dtype=np.float64)}
    for col in BODY_STATE_COLUMNS:
        src = col if col in df.columns else None
        if src:
            data[f"{prefix}{col}"] = pd.to_numeric(df[src], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        else:
            data[f"{prefix}{col}"] = np.zeros(len(df), dtype=np.float64)
    for col in BODY_PROXY_COLUMNS:
        src = col if col in df.columns else None
        if src:
            data[f"{prefix}{col}"] = pd.to_numeric(df[src], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        else:
            data[f"{prefix}{col}"] = np.zeros(len(df), dtype=np.float64)

    quat = np.column_stack([
        data[f"{prefix}q0"],
        data[f"{prefix}q1"],
        data[f"{prefix}q2"],
        data[f"{prefix}q3"],
    ])
    roll, pitch, yaw = quaternion_to_euler_xyz(quat)
    times = data["time"]
    d_roll = safe_gradient(roll, times)
    d_pitch = safe_gradient(pitch, times)
    d_yaw = safe_gradient(yaw, times)
    data[f"{prefix}roll"] = roll
    data[f"{prefix}pitch"] = pitch
    data[f"{prefix}yaw"] = yaw
    data[f"{prefix}sin_yaw"] = np.sin(yaw)
    data[f"{prefix}cos_yaw"] = np.cos(yaw)
    data[f"{prefix}d_roll"] = d_roll
    data[f"{prefix}dd_roll"] = safe_gradient(d_roll, times)
    data[f"{prefix}d_pitch"] = d_pitch
    data[f"{prefix}dd_pitch"] = safe_gradient(d_pitch, times)
    data[f"{prefix}d_yaw"] = d_yaw
    data[f"{prefix}dd_yaw"] = safe_gradient(d_yaw, times)
    if f"{prefix}dt" not in data:
        data[f"{prefix}dt"] = infer_dt_series(df).to_numpy(dtype=np.float64)
    return pd.DataFrame(data)


def build_wheel_frame(csv_path: Path, prefix: str, target_times: Optional[np.ndarray] = None) -> pd.DataFrame:
    df = normalize_time_columns(pd.read_csv(csv_path))
    df["wheel"] = df["wheel"].astype(str)
    if target_times is None:
        target_times = np.sort(df["time"].unique().astype(float))
    data: Dict[str, np.ndarray] = {"time": target_times.astype(np.float64)}

    for wheel_id in WHEEL_IDS:
        wheel_name = f"wheel{wheel_id}_body"
        sub = df[df["wheel"] == wheel_name].copy().reset_index(drop=True)
        if sub.empty:
            raise ValueError(f"{csv_path} 中缺少 {wheel_name} 数据")
        sub = subtract_initial_position(sub)
        sub["dt"] = infer_dt_series(sub)
        cols = [c for c in WHEEL_STATE_COLUMNS if c in sub.columns] + ["dt"]
        aligned = align_to_times(sub, target_times, cols)
        base = f"{prefix}wheel{wheel_id}_"
        for col in WHEEL_STATE_COLUMNS:
            if col in aligned.columns:
                data[f"{base}{col}"] = pd.to_numeric(aligned[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            else:
                data[f"{base}{col}"] = np.zeros(len(target_times), dtype=np.float64)
        if "dt" in aligned.columns:
            data[f"{base}dt"] = pd.to_numeric(aligned["dt"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        else:
            data[f"{base}dt"] = np.zeros(len(target_times), dtype=np.float64)

        quat = np.column_stack([
            data[f"{base}q0"],
            data[f"{base}q1"],
            data[f"{base}q2"],
            data[f"{base}q3"],
        ])
        roll, pitch, yaw = quaternion_to_euler_xyz(quat)
        times = data["time"]
        d_roll = safe_gradient(roll, times)
        d_pitch = safe_gradient(pitch, times)
        d_yaw = safe_gradient(yaw, times)
        data[f"{base}roll"] = roll
        data[f"{base}pitch"] = pitch
        data[f"{base}yaw"] = yaw
        data[f"{base}sin_yaw"] = np.sin(yaw)
        data[f"{base}cos_yaw"] = np.cos(yaw)
        data[f"{base}d_roll"] = d_roll
        data[f"{base}dd_roll"] = safe_gradient(d_roll, times)
        data[f"{base}d_pitch"] = d_pitch
        data[f"{base}dd_pitch"] = safe_gradient(d_pitch, times)
        data[f"{base}d_yaw"] = d_yaw
        data[f"{base}dd_yaw"] = safe_gradient(d_yaw, times)

    return pd.DataFrame(data)


def build_aux_body_frame(csv_path: Path, prefix: str, target_times: np.ndarray) -> pd.DataFrame:
    src = build_body_frame(csv_path, prefix="")
    cols = [c for c in src.columns if c != "time"]
    aligned = align_to_times(src, target_times, cols)
    aligned = aligned.rename(columns={col: f"{prefix}{col}" for col in cols})
    return aligned


def rolling_std(values: pd.Series, window: int = 5) -> pd.Series:
    return values.rolling(window=window, min_periods=1).std().fillna(0.0)


def enrich_proxy_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = out["lf_sim_dt"].replace(0.0, np.nan).bfill().ffill().fillna(1.0)

    out["lf_acc_x_std"] = rolling_std(out["lf_acc_x"])
    out["lf_acc_y_std"] = rolling_std(out["lf_acc_y"])
    out["lf_acc_z_std"] = rolling_std(out["lf_acc_z"])
    out["lf_ang_acc_x_std"] = rolling_std(out["lf_ang_acc_x"])
    out["lf_ang_acc_y_std"] = rolling_std(out["lf_ang_acc_y"])
    out["lf_ang_acc_z_std"] = rolling_std(out["lf_ang_acc_z"])

    for wheel_id in WHEEL_IDS:
        base = f"lf_wheel{wheel_id}_"
        out[f"{base}slip_long_rate"] = out[base + "slip_long"].diff().fillna(0.0) / dt
        out[f"{base}slip_lat_rate"] = out[base + "slip_lat"].diff().fillna(0.0) / dt
        out[f"{base}sinkage_rate"] = out[base + "sinkage"].diff().fillna(0.0) / dt
        out[f"{base}contact_switch"] = out[base + "in_contact"].diff().abs().fillna(0.0)
        for key in ["Fx", "Fy", "Fz", "Mx", "My", "Mz", "acc_x", "acc_y", "acc_z"]:
            out[f"{base}{key}_std"] = rolling_std(out[f"{base}{key}"])
        out[f"{base}rel_pos_x"] = out[f"{base}pos_x"] - out["lf_pos_x"]
        out[f"{base}rel_pos_y"] = out[f"{base}pos_y"] - out["lf_pos_y"]
        out[f"{base}rel_pos_z"] = out[f"{base}pos_z"] - out["lf_pos_z"]
        out[f"{base}rel_vel_x"] = out[f"{base}vel_x"] - out["lf_vel_x"]
        out[f"{base}rel_vel_y"] = out[f"{base}vel_y"] - out["lf_vel_y"]
        out[f"{base}rel_vel_z"] = out[f"{base}vel_z"] - out["lf_vel_z"]

    left_fz = out["lf_wheel0_Fz"] + out["lf_wheel2_Fz"] + out["lf_wheel4_Fz"]
    right_fz = out["lf_wheel1_Fz"] + out["lf_wheel3_Fz"] + out["lf_wheel5_Fz"]
    front_fz = out["lf_wheel0_Fz"] + out["lf_wheel1_Fz"]
    mid_fz = out["lf_wheel2_Fz"] + out["lf_wheel3_Fz"]
    rear_fz = out["lf_wheel4_Fz"] + out["lf_wheel5_Fz"]
    total_fz = left_fz + right_fz
    out["lf_load_diff_left_right"] = left_fz - right_fz
    out["lf_load_transfer_lat"] = (left_fz - right_fz) / total_fz.replace(0.0, np.nan)
    out["lf_load_diff_front_mid"] = front_fz - mid_fz
    out["lf_load_diff_mid_rear"] = mid_fz - rear_fz
    out["lf_load_transfer_lon"] = (front_fz - rear_fz) / (front_fz + mid_fz + rear_fz).replace(0.0, np.nan)
    out["lf_load_transfer_lat"] = out["lf_load_transfer_lat"].fillna(0.0)
    out["lf_load_transfer_lon"] = out["lf_load_transfer_lon"].fillna(0.0)
    return out


def map_case_basename(case_dir: Path) -> str:
    name = case_dir.name
    if "_case" in name:
        return "case" + name.split("_case", 1)[1]
    return name


def build_control_params(args: argparse.Namespace) -> Optional[Dict[str, float]]:
    values = getattr(args, "control_params", None)
    if not values:
        return None
    if len(values) != len(CONTROL_PARAM_ORDER):
        raise ValueError(f"--control_params 需要 {len(CONTROL_PARAM_ORDER)} 个数值")
    return {name: float(value) for name, value in zip(CONTROL_PARAM_ORDER, values)}


def build_target_names() -> List[str]:
    target_names = [
        "pos_x", "pos_y", "pos_z",
        "vel_x", "vel_y", "vel_z",
        "acc_x", "acc_y", "acc_z",
        "q0", "q1", "q2", "q3",
    ]
    for i in WHEEL_IDS:
        target_names.extend([
            f"wheel{i}_pos_x", f"wheel{i}_pos_y", f"wheel{i}_pos_z",
            f"wheel{i}_q0", f"wheel{i}_q1", f"wheel{i}_q2", f"wheel{i}_q3",
            f"wheel{i}_vel_x", f"wheel{i}_vel_y", f"wheel{i}_vel_z",
            f"wheel{i}_acc_x", f"wheel{i}_acc_y", f"wheel{i}_acc_z",
            f"wheel{i}_Fx", f"wheel{i}_Fy", f"wheel{i}_Fz",
            f"wheel{i}_Mx", f"wheel{i}_My", f"wheel{i}_Mz",
            f"wheel{i}_slip_long", f"wheel{i}_sinkage",
        ])
    return target_names


def build_raw_case_feature_dataframe(
    case_dir: str,
    case_prefix: str,
    control_params: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    case_path = Path(case_dir)
    body_dir = case_path / "BODY"
    wheel_path = case_path / "WHEEL" / "wheel_state.csv"
    body_df = build_body_frame(body_dir / "chassis_body.csv", prefix=case_prefix)
    target_times = body_df["time"].to_numpy(dtype=np.float64)
    wheel_df = build_wheel_frame(wheel_path, prefix=case_prefix, target_times=target_times)
    merged = body_df.merge(wheel_df, on="time", how="inner")

    for file_name, rocker_code in ROCKER_FILE_TO_CODE.items():
        aux = build_aux_body_frame(body_dir / file_name, prefix=f"{case_prefix}susp_rocker_{rocker_code}_", target_times=target_times)
        merged = merged.merge(aux, on="time", how="left")
    for wheel_id in WHEEL_IDS:
        aux = build_aux_body_frame(body_dir / f"upright{wheel_id}_body.csv", prefix=f"{case_prefix}susp_upright{wheel_id}_", target_times=target_times)
        merged = merged.merge(aux, on="time", how="left")

    if case_prefix == "lf_":
        if control_params is not None:
            merged["lf_cmd_speed"] = merged["time"].apply(lambda x: build_cmd_speed(float(x), control_params))
        else:
            merged["lf_cmd_speed"] = np.sqrt(
                merged["lf_vel_x"] ** 2 + merged["lf_vel_y"] ** 2 + merged["lf_vel_z"] ** 2
            )

        for name in CONTROL_PARAM_ORDER:
            merged[f"lf_{name}"] = float(control_params.get(name, 0.0)) if control_params is not None else 0.0
        for col in ["lf_acc_residual_x", "lf_acc_residual_y", "lf_acc_residual_z"]:
            merged[col] = 0.0
        merged["lf_sim_dt"] = merged["time"].diff().fillna(0.0)
        if len(merged) > 1:
            merged.loc[0, "lf_sim_dt"] = merged.loc[1, "lf_sim_dt"]
        merged = enrich_proxy_features(merged)
    return merged


def build_reference_target_frame(case_dir: str, target_times: np.ndarray) -> pd.DataFrame:
    ref_df = build_raw_case_feature_dataframe(case_dir, case_prefix="hf_")
    cols = [c for c in ref_df.columns if c != "time"]
    aligned = align_to_times(ref_df, target_times, cols)
    for col in cols:
        if col not in aligned.columns:
            aligned[col] = 0.0
    return aligned


def derive_residual_frame(df: pd.DataFrame, spec) -> pd.DataFrame:
    out = df.copy()
    for hf_col in spec.target_cols:
        if not hf_col.startswith("hf_"):
            continue
        suffix = hf_col[len("hf_"):]
        lf_col = "lf_" + suffix
        res_col = "res_" + suffix
        if lf_col in out.columns and hf_col in out.columns:
            out[res_col] = pd.to_numeric(out[hf_col], errors="coerce").fillna(0.0) - pd.to_numeric(out[lf_col], errors="coerce").fillna(0.0)
        elif res_col not in out.columns:
            out[res_col] = 0.0
    for col in spec.res_cols:
        if col not in out.columns:
            out[col] = 0.0
    return out


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
        for name in build_target_names():
            lf_col = f"lf_{name}"
            hf_col = f"hf_{name}"
            if lf_col in df.columns and hf_col in df.columns:
                res_cols[f"res_{name}"] = df[hf_col] - df[lf_col]
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
    df = df[ordered_cols].sort_values("time").reset_index(drop=True)
    return df


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

    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["case_name", "time"]).reset_index(drop=True)


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


def write_run_info(args: argparse.Namespace, output_dir: str, gate_mode: str, meta: Dict) -> None:
    lines = [
        f"infer_mode: {args.infer_mode}",
        f"checkpoint: {args.checkpoint}",
        f"gate_mode: {gate_mode}",
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
            f"teacher_hidden_dim: {meta.get('teacher_hidden_dim') if meta.get('teacher_hidden_dim') is not None else ''}",
        ]
    )
    info_path = os.path.join(output_dir, "run_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def sanitize_plot_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))
    return safe or "unknown"


def add_body_position_blend_columns(pred_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    if pred_df.empty or "case_name" not in pred_df.columns or "time" not in pred_df.columns:
        return pred_df

    alpha = float(np.clip(alpha, 0.0, 1.0))
    out = pred_df.copy()
    pos_axes = ("x", "y", "z")
    required_pred_cols = [f"pred::hf_body::hf_pos_{axis}" for axis in pos_axes]
    required_vel_cols = [f"pred::hf_body::hf_vel_{axis}" for axis in pos_axes]
    if not all(col in out.columns for col in required_pred_cols + required_vel_cols):
        return out

    for _, case_idx in out.groupby("case_name", sort=False).groups.items():
        case_indices = list(case_idx)
        case_df = out.loc[case_indices].copy()
        case_df["time"] = pd.to_numeric(case_df["time"], errors="coerce")
        case_df = case_df.sort_values("time")
        sorted_idx = case_df.index.to_list()
        times = case_df["time"].to_numpy(dtype=np.float64)
        if len(times) == 0:
            continue

        dt = np.zeros(len(times), dtype=np.float64)
        if len(times) > 1:
            dt[1:] = np.diff(times)

        for axis in pos_axes:
            pred_pos_col = f"pred::hf_body::hf_pos_{axis}"
            pred_vel_col = f"pred::hf_body::hf_vel_{axis}"
            custom_pos_col = f"custom::hf_body::hf_pos_{axis}"
            pos_from_vel_col = f"pred_pos_from_vel::hf_body::hf_pos_{axis}"
            final_pos_col = f"pred_final::hf_body::hf_pos_{axis}"

            pred_pos = pd.to_numeric(case_df[pred_pos_col], errors="coerce").to_numpy(dtype=np.float64)
            pred_vel = pd.to_numeric(case_df[pred_vel_col], errors="coerce").to_numpy(dtype=np.float64)
            if custom_pos_col in case_df.columns:
                init_pos = float(pd.to_numeric(case_df.iloc[0][custom_pos_col], errors="coerce"))
            else:
                init_pos = float(pred_pos[0])

            pos_from_vel = np.empty(len(times), dtype=np.float64)
            pos_from_vel[0] = init_pos
            for i in range(1, len(times)):
                pos_from_vel[i] = pos_from_vel[i - 1] + pred_vel[i - 1] * dt[i]

            final_pos = alpha * pred_pos + (1.0 - alpha) * pos_from_vel
            out.loc[sorted_idx, pos_from_vel_col] = pos_from_vel
            out.loc[sorted_idx, final_pos_col] = final_pos

    return out


def compute_metrics_from_prediction_dataframe(pred_df: pd.DataFrame, spec) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    overall_sqerr_sum = 0.0
    overall_count = 0

    group_columns = [("hf_body", spec.target_groups.body_cols)]
    for i in WHEEL_IDS:
        group_columns.append((f"hf_wheel{i}_kin", spec.target_groups.wheel_kin_cols[i]))
        group_columns.append((f"hf_wheel{i}_contact", spec.target_groups.wheel_contact_cols[i]))

    for group_name, cols in group_columns:
        pred_series = []
        true_series = []
        for col in cols:
            pred_col = f"pred::{group_name}::{col}"
            if group_name == "hf_body" and col.startswith("hf_pos_"):
                pred_col = f"pred_final::{group_name}::{col}" if f"pred_final::{group_name}::{col}" in pred_df.columns else pred_col
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


def generate_inference_plots(pred_df: pd.DataFrame, output_dir: str) -> None:
    if pred_df.empty or "case_name" not in pred_df.columns or "time" not in pred_df.columns:
        return

    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    group_suffixes = {
        "hf_body": ["hf_pos_x", "hf_pos_y", "hf_pos_z", "hf_vel_x", "hf_vel_y", "hf_vel_z"],
        **{
            f"hf_wheel{i}_kin": [
                f"hf_wheel{i}_pos_x", f"hf_wheel{i}_pos_y", f"hf_wheel{i}_pos_z",
                f"hf_wheel{i}_vel_x", f"hf_wheel{i}_vel_y", f"hf_wheel{i}_vel_z",
            ]
            for i in WHEEL_IDS
        },
        **{
            f"hf_wheel{i}_contact": [
                f"hf_wheel{i}_Fx", f"hf_wheel{i}_Fy", f"hf_wheel{i}_Fz",
                f"hf_wheel{i}_Mx", f"hf_wheel{i}_My", f"hf_wheel{i}_Mz",
            ]
            for i in WHEEL_IDS
        },
    }

    reference_prefixes = ["true", "custom", "sph_ref"]
    for case_name, case_df in pred_df.groupby("case_name", sort=False):
        case_df = case_df.sort_values("time").reset_index(drop=True)
        x = pd.to_numeric(case_df["time"], errors="coerce").to_numpy(dtype=np.float64)
        case_dir = plots_dir / sanitize_plot_name(case_name)
        case_dir.mkdir(parents=True, exist_ok=True)

        def save_plot(
            plot_group_name: str,
            suffix: str,
            series_map: Dict[str, np.ndarray],
            title_group_name: str,
        ) -> None:
            plot_styles = {
                "HF true": {"linewidth": 2.0, "alpha": 0.95, "linestyle": "-"},
                "Compensated": {"linewidth": 1.8, "alpha": 0.95, "linestyle": "-"},
                "LF current": {"linewidth": 1.5, "alpha": 0.5, "linestyle": "--"},
                "SPH ref": {"linewidth": 1.5, "alpha": 0.45, "linestyle": ":"},
            }
            fig = plt.figure(figsize=(10, 4))
            for label in ["HF true", "Compensated", "LF current", "SPH ref"]:
                values = series_map.get(label)
                if values is None:
                    continue
                plt.plot(x, values, label=label, **plot_styles[label])

            plt.xlabel("time / s")
            plt.ylabel(suffix)
            plt.title(f"{case_name} | {title_group_name} | {suffix}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            file_name = f"{sanitize_plot_name(plot_group_name)}__{sanitize_plot_name(suffix)}.png"
            plt.savefig(case_dir / file_name, dpi=150)
            plt.close(fig)

        def numeric_series(col_name: str) -> Optional[np.ndarray]:
            if col_name not in case_df.columns:
                return None
            return pd.to_numeric(case_df[col_name], errors="coerce").to_numpy(dtype=np.float64)

        def derive_velocity_component(prefix: str, source_group_name: str, axis: str) -> Optional[np.ndarray]:
            pos_cols = [f"{prefix}::{source_group_name}::{base}" for base in ("hf_pos_x", "hf_pos_y", "hf_pos_z")]
            if source_group_name.startswith("hf_wheel"):
                wheel_prefix = source_group_name.replace("_kin", "")
                pos_cols = [f"{prefix}::{source_group_name}::{wheel_prefix}_pos_{a}" for a in ("x", "y", "z")]
            if not all(col in case_df.columns for col in pos_cols):
                return None
            if len(x) < 2:
                return np.zeros_like(x)
            pos_xyz = np.column_stack([numeric_series(col) for col in pos_cols])
            axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
            return np.gradient(pos_xyz[:, axis_idx], x)

        def get_series(label: str, prefix: str, group_name: str, suffix: str) -> Optional[np.ndarray]:
            if label == "Compensated" and prefix == "pred" and group_name == "hf_body" and suffix.startswith("hf_pos_"):
                final_col = f"pred_final::{group_name}::{suffix}"
                values = numeric_series(final_col)
                if values is not None:
                    return values
            direct_col = f"{prefix}::{group_name}::{suffix}"
            values = numeric_series(direct_col)
            if values is not None:
                return values
            if suffix.endswith(("hf_vel_x", "hf_vel_y", "hf_vel_z")):
                return derive_velocity_component(prefix, group_name, suffix[-1])
            return None

        for group_name, suffixes in group_suffixes.items():
            for suffix in suffixes:
                series_map = {}
                for prefix in reference_prefixes:
                    label = {
                        "true": "HF true",
                        "custom": "LF current",
                        "sph_ref": "SPH ref",
                    }[prefix]
                    values = get_series(label, prefix, group_name, suffix)
                    if values is not None:
                        series_map[label] = values
                pred_values = get_series("Compensated", "pred", group_name, suffix)
                if pred_values is not None:
                    series_map["Compensated"] = pred_values
                if "Compensated" not in series_map or "HF true" not in series_map:
                    continue
                save_plot(
                    plot_group_name=group_name,
                    suffix=suffix,
                    series_map=series_map,
                    title_group_name=group_name,
                )

        speed_triplets = {
            "hf_body_speed": ("hf_body", ["hf_vel_x", "hf_vel_y", "hf_vel_z"]),
            **{f"hf_wheel{i}_speed": (f"hf_wheel{i}_kin", [f"hf_wheel{i}_vel_x", f"hf_wheel{i}_vel_y", f"hf_wheel{i}_vel_z"]) for i in WHEEL_IDS},
        }
        for plot_group_name, (source_group_name, speed_cols) in speed_triplets.items():
            series_map = {}
            for label, prefix in [
                ("Compensated", "pred"),
                ("HF true", "true"),
                ("LF current", "custom"),
                ("SPH ref", "sph_ref"),
            ]:
                vel_components = []
                for col in speed_cols:
                    values = get_series(label, prefix, source_group_name, col)
                    if values is None:
                        vel_components = []
                        break
                    vel_components.append(values)
                if not vel_components:
                    continue
                vel_components = np.column_stack(vel_components)
                series_map[label] = np.linalg.norm(vel_components, axis=1)

            if "Compensated" not in series_map or "HF true" not in series_map:
                continue

            save_plot(
                plot_group_name=plot_group_name,
                suffix="speed",
                series_map=series_map,
                title_group_name=source_group_name,
            )

        body_pos_axes = ("x", "y", "z")
        fusion_plot_styles = {
            "HF true": {"linewidth": 2.0, "alpha": 0.95, "linestyle": "-"},
            "LF current": {"linewidth": 1.5, "alpha": 0.45, "linestyle": "--"},
            "Direct pos": {"linewidth": 1.4, "alpha": 0.8, "linestyle": "-."},
            "Pos from vel": {"linewidth": 1.6, "alpha": 0.85, "linestyle": ":"},
            "Fused pos": {"linewidth": 2.0, "alpha": 0.95, "linestyle": "-"},
        }
        for axis in body_pos_axes:
            suffix = f"hf_pos_{axis}"
            direct_col = f"pred::hf_body::{suffix}"
            from_vel_col = f"pred_pos_from_vel::hf_body::{suffix}"
            fused_col = f"pred_final::hf_body::{suffix}"
            true_col = f"true::hf_body::{suffix}"
            custom_col = f"custom::hf_body::{suffix}"
            if not all(col in case_df.columns for col in [direct_col, from_vel_col, fused_col, true_col]):
                continue

            series_map = {
                "HF true": numeric_series(true_col),
                "Direct pos": numeric_series(direct_col),
                "Pos from vel": numeric_series(from_vel_col),
                "Fused pos": numeric_series(fused_col),
            }
            lf_values = numeric_series(custom_col)
            if lf_values is not None:
                series_map["LF current"] = lf_values

            fig = plt.figure(figsize=(10, 4))
            for label in ["HF true", "LF current", "Direct pos", "Pos from vel", "Fused pos"]:
                values = series_map.get(label)
                if values is None:
                    continue
                plt.plot(x, values, label=label, **fusion_plot_styles[label])

            plt.xlabel("time / s")
            plt.ylabel(suffix)
            plt.title(f"{case_name} | hf_body_fusion | {suffix}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            file_name = f"hf_body_fusion__{sanitize_plot_name(suffix)}.png"
            plt.savefig(case_dir / file_name, dpi=150)
            plt.close(fig)


def run_inference(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = safe_load_checkpoint(args.checkpoint, device)
    spec = load_column_spec(args.feature_dir)
    scaler = restore_scaler_from_checkpoint(ckpt)
    ensure_dir(args.output_dir)

    ckpt_args = ckpt.get("args") or {}
    seed = int(get_arg_with_default(ckpt_args, "seed", args.seed, 42))
    seq_len = int(get_arg_with_default(ckpt_args, "seq_len", args.seq_len, 30))
    pred_horizon = int(get_arg_with_default(ckpt_args, "pred_horizon", args.pred_horizon, 0))
    pred_seq_len = int(get_arg_with_default(ckpt_args, "pred_seq_len", args.pred_seq_len, 1))
    train_ratio = float(get_arg_with_default(ckpt_args, "train_ratio", args.train_ratio, 0.7))
    val_ratio = float(get_arg_with_default(ckpt_args, "val_ratio", args.val_ratio, 0.15))

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
        df = load_merged_dataset(args.merged_csv, spec, case_col="case_name", time_col="time")
        infer_csv_path = None

    selected_df = select_dataframe(
        df=df,
        split=args.split,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        case_name=args.case_name,
    )

    dataset = GraphTemporalSequenceDataset(
        selected_df,
        spec,
        scaler,
        seq_len=seq_len,
        pred_horizon=pred_horizon,
        pred_seq_len=pred_seq_len,
        case_col="case_name",
        time_col="time",
    )
    if args.max_samples is not None and args.max_samples > 0:
        dataset.index_map = dataset.index_map[: args.max_samples]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=graph_temporal_collate_fn,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    gate_mode = resolve_gate_mode(args.gate_mode, ckpt)
    model, teacher_gate_net, meta = build_model_from_checkpoint(
        ckpt=ckpt,
        spec=spec,
        gate_mode=gate_mode,
        teacher_hidden_dim=args.teacher_hidden_dim,
        device=device,
    )

    all_rows: List[Dict[str, float]] = []
    metrics_store: Dict[str, List[np.ndarray]] = {}
    reference_prefix = "true"
    if args.infer_mode == "custom_case" and args.sph_case_dir:
        reference_prefix = "sph_ref"
    elif args.infer_mode == "custom_cases" and args.sph_cases_dir:
        reference_prefix = "sph_ref"

    for batch_idx, batch in enumerate(tqdm(loader, desc="Infer", dynamic_ncols=True)):
        batch = move_batch_to_device(batch, device)
        with torch.no_grad():
            if gate_mode == "teacher":
                if teacher_gate_net is None:
                    raise RuntimeError("teacher 模式下 teacher_gate_net 不应为空")
                teacher_gate = build_teacher_relation_gate(teacher_gate_net, batch, batch["body"].shape[1])
                output = model(batch, relation_gate_override=teacher_gate)
            else:
                output = model(batch)

        group_defs = [
            (
                "hf_body",
                output["pred_res_body"],
                batch["lf_body_current"],
                spec.res_groups.body_cols,
                spec.target_groups.body_cols,
                "res_body",
            )
        ]
        for i in WHEEL_IDS:
            group_defs.append(
                (
                    f"hf_wheel{i}_kin",
                    output[f"pred_res_wheel{i}_kin"],
                    batch[f"lf_wheel{i}_kin_current"],
                    spec.res_groups.wheel_kin_cols[i],
                    spec.target_groups.wheel_kin_cols[i],
                    f"res_wheel{i}_kin",
                )
            )
            group_defs.append(
                (
                    f"hf_wheel{i}_contact",
                    output[f"pred_res_wheel{i}_contact"],
                    batch[f"lf_wheel{i}_contact_current"],
                    spec.res_groups.wheel_contact_cols[i],
                    spec.target_groups.wheel_contact_cols[i],
                    f"res_wheel{i}_contact",
                )
            )

        batch_row_offset = len(all_rows)
        for group_name, pred_res_scaled, lf_current_raw, res_cols, hf_cols, res_group_name in group_defs:
            pred_hf_scaled = reconstruct_hf_scaled(
                pred_res_scaled=pred_res_scaled,
                lf_current_raw=lf_current_raw,
                res_cols=res_cols,
                hf_cols=hf_cols,
                res_group_name=res_group_name,
                hf_group_name=group_name,
                scaler=scaler,
            )
            pred_raw = inverse_transform_tensor(pred_hf_scaled, scaler, group_name)
            true_raw = inverse_transform_tensor(batch[group_name], scaler, group_name)

            flatten_group_rows(
                rows=all_rows,
                sample_idx_offset=batch_row_offset,
                case_names=batch["case_name"],
                times=batch["time"],
                pred_raw=pred_raw,
                true_raw=true_raw,
                columns=hf_cols,
                group_name=group_name,
            )
            flatten_reference_rows(
                rows=all_rows,
                sample_idx_offset=batch_row_offset,
                times=batch["time"],
                ref_raw=lf_current_raw,
                columns=hf_cols,
                group_name=group_name,
                prefix="custom",
            )
            if reference_prefix != "true":
                flatten_reference_rows(
                    rows=all_rows,
                    sample_idx_offset=batch_row_offset,
                    times=batch["time"],
                    ref_raw=true_raw,
                    columns=hf_cols,
                    group_name=group_name,
                    prefix=reference_prefix,
                )

            metrics_store.setdefault(f"{group_name}::pred", []).append(pred_raw.detach().cpu().numpy())
            metrics_store.setdefault(f"{group_name}::true", []).append(true_raw.detach().cpu().numpy())

        if "student_relation_gate" in output:
            gate_np = output["student_relation_gate"].detach().cpu().numpy()
            batch_rows = all_rows[batch_row_offset:]
            flat_gate = gate_np.reshape(-1, gate_np.shape[-1])
            valid_count = min(len(batch_rows), len(flat_gate))
            for idx in range(valid_count):
                batch_rows[idx]["student_gate_mean"] = float(flat_gate[idx].mean())
                batch_rows[idx]["student_gate_std"] = float(flat_gate[idx].std())

    pred_df = pd.DataFrame(all_rows)
    pred_df = add_body_position_blend_columns(pred_df, alpha=args.body_pos_blend_alpha)
    write_run_info(args, args.output_dir, gate_mode, meta)
    pred_path = os.path.join(args.output_dir, "predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    generate_inference_plots(pred_df, args.output_dir)
    metrics = compute_metrics_from_prediction_dataframe(pred_df, spec)

    summary = {
        "checkpoint": args.checkpoint,
        "infer_mode": args.infer_mode,
        "gate_mode": gate_mode,
        "split": args.split,
        "case_name": args.case_name,
        "device": str(device),
        "num_samples": int(len(dataset)),
        "pred_seq_len": int(meta["pred_seq_len"]),
        "metrics": metrics,
    }
    if meta["teacher_hidden_dim"] is not None:
        summary["teacher_hidden_dim"] = int(meta["teacher_hidden_dim"])
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if infer_csv_path is not None:
        print(f"generated infer csv saved to: {infer_csv_path}")
    print(f"predictions saved to: {pred_path}")
    print(f"metrics saved to: {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="训练得到的 best_model.pt")
    parser.add_argument("--feature_dir", type=str, default=str(ROOT / "Feature_Selection" / "DataSet"))
    parser.add_argument("--merged_csv", type=str, default=str(ROOT / "Feature_Selection" / "DataSet" / "merged_error_dataset.csv"))
    parser.add_argument("--infer_mode", type=str, choices=["merged_csv", "custom_case", "custom_cases"], default="merged_csv")
    parser.add_argument("--custom_case_dir", type=str, default=None)
    parser.add_argument("--sph_case_dir", type=str, default=None)
    parser.add_argument(
        "--custom_cases_dir",
        type=str,
        default="/media/user/新加卷/RoverSimData/Multi_Custom_test_output",
        help="批量 custom_cases 推理的 Custom case 父目录",
    )
    parser.add_argument(
        "--sph_cases_dir",
        type=str,
        default="/media/user/新加卷/RoverSimData/Multi_SPH_test_output",
        help="批量 custom_cases 推理的 SPH case 父目录；设为空字符串则只使用 Custom",
    )
    parser.add_argument("--infer_case_name", type=str, default=None)
    parser.add_argument("--control_params", type=float, nargs=12, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--split", type=str, choices=["all", "train", "val", "test"], default="all")
    parser.add_argument("--case_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gate_mode", type=str, choices=["auto", "student", "teacher"], default="student")
    parser.add_argument("--teacher_hidden_dim", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--pred_horizon", type=int, default=None)
    parser.add_argument("--pred_seq_len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train_ratio", type=float, default=None)
    parser.add_argument("--val_ratio", type=float, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--body_pos_blend_alpha", type=float, default=0.5)
    args = parser.parse_args()

    if args.output_dir is None:
        ckpt_dir = Path(args.checkpoint).resolve().parent
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(ckpt_dir / f"infer_{ts}")
    return args


def main() -> None:
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
