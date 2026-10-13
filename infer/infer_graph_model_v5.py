from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Feature_Selection.datasetProcess import add_prefix, add_rotvec_residuals, read_case_data  # noqa: E402
from models.data_utils import GroupStandardizer, NumpyStandardScaler  # noqa: E402
from models.data_utils_v5 import (  # noqa: E402
    GraphTemporalSequenceDatasetV5,
    ROCKER_NAMES,
    WHEEL_IDS,
    ensure_time_feature_columns,
    get_group_dims,
    graph_temporal_collate_fn,
    load_column_spec,
    load_merged_dataset,
    split_train_val_test_by_case,
)
from models.graph_temporal_hgt_compensation_v5 import HGTGraphTemporalCompensationModelV5  # noqa: E402
from models.load_transfer_force_v5 import load_transfer_fz_torch  # noqa: E402

AXES = ("x", "y", "z")
FORCE_AXES = ("Fx", "Fy", "Fz")
CONTROL_PARAM_ORDER = [
    "idle_time", "accel_time", "accel_speed", "const_time", "sinusoid_time",
    "sinusoid_speed", "sinusoid_freq", "step_time", "dec_time", "idle_end",
    "step_low_speed", "step_high_speed",
]


def log(message: str) -> None:
    print(f"[infer_v5] {message}", flush=True)


def indices(cols: List[str], suffixes: List[str]) -> List[int]:
    out = []
    for suffix in suffixes:
        idx = next((i for i, col in enumerate(cols) if col.endswith(suffix)), None)
        if idx is not None:
            out.append(idx)
    return out


def omega_index(cols: List[str]) -> Optional[int]:
    return next((i for i, c in enumerate(cols) if c.endswith("omega") or "ang_vel" in c), None)


def inverse_group(value: torch.Tensor, scaler: GroupStandardizer, key: str) -> torch.Tensor:
    scaler_obj = scaler.scalers[key]
    mean = torch.as_tensor(scaler_obj.mean_, device=value.device, dtype=value.dtype)
    std = torch.as_tensor(scaler_obj.std_, device=value.device, dtype=value.dtype)
    mask = torch.as_tensor(scaler_obj.apply_mask_, device=value.device, dtype=torch.bool)
    out = value.float().clone()
    out[..., mask] = out[..., mask] * std[mask] + mean[mask]
    return out


def inverse_residual(
    value: torch.Tensor,
    residual_cols: List[str],
    target_cols: List[str],
    key: str,
    scaler: GroupStandardizer,
) -> torch.Tensor:
    raw = inverse_group(value, scaler, key)
    lookup = {name: i for i, name in enumerate(residual_cols)}
    out = raw.new_zeros(*raw.shape[:-1], len(target_cols))
    for j, col in enumerate(target_cols):
        name = "res_" + col[3:] if col.startswith("hf_") else ""
        if name in lookup:
            out[..., j] = raw[..., lookup[name]]
    return out


def quat_matrix(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y*y + z*z), 2 * (x*y-z*w), 2 * (x*z+y*w),
        2 * (x*y+z*w), 1 - 2 * (x*x+z*z), 2 * (y*z-x*w),
        2 * (x*z-y*w), 2 * (y*z+x*w), 1 - 2 * (x*x+y*y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


def rotvec_matrix(v: torch.Tensor) -> torch.Tensor:
    theta = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    axis = v / theta
    x, y, z = axis.unbind(-1)
    c, s, one = torch.cos(theta[..., 0]), torch.sin(theta[..., 0]), 1 - torch.cos(theta[..., 0])
    return torch.stack([
        c+x*x*one, x*y*one-z*s, x*z*one+y*s,
        y*x*one+z*s, c+y*y*one, y*z*one-x*s,
        z*x*one-y*s, z*y*one+x*s, c+z*z*one,
    ], dim=-1).reshape(*v.shape[:-1], 3, 3)


def matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    """Convert one 3x3 rotation matrix to scalar-first quaternion."""
    m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.asarray([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = np.asarray([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = np.asarray([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = np.asarray([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return q / max(np.linalg.norm(q), 1e-12)


def restore_scaler(checkpoint_path: str, ckpt: Dict[str, Any]) -> GroupStandardizer:
    scaler = GroupStandardizer()
    scaler_state = ckpt.get("scaler_state")
    if scaler_state:
        scaler.scalers = {name: NumpyStandardScaler.from_dict(state) for name, state in scaler_state.items()}
        return scaler
    path = Path(checkpoint_path).resolve().parent / "group_scaler_v5.joblib"
    if path.exists():
        return GroupStandardizer.load(str(path))
    raise FileNotFoundError(f"V5 checkpoint has no scaler_state and {path} is missing")


def load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model(ckpt: Dict[str, Any], spec, mode: str, device: torch.device):
    ckpt_args = ckpt.get("args") or {}
    state_args = ckpt.get("state_architecture_args") or {}
    val = lambda name, default: ckpt_args.get(name, state_args.get(name, default))
    body_cols = spec.res_groups.body_cols
    wheel_cols = spec.res_groups.wheel_kin_cols[0]
    pos_idx = indices(body_cols, ["pos_x", "pos_y", "pos_z"])
    vel_idx = indices(body_cols, ["vel_x", "vel_y", "vel_z"])
    wheel_pos = indices(wheel_cols, ["pos_x", "pos_y", "pos_z"])
    omega_idx = omega_index(wheel_cols)
    role = "teacher" if mode == "state_teacher" else "student"
    model = HGTGraphTemporalCompensationModelV5(
        get_group_dims(spec),
        node_hidden_dim=int(val("hidden_dim", 192)),
        graph_layers=int(val("graph_layers", 3)),
        tcn_hidden_dim=int(val("tcn_dim", 256)),
        lstm_hidden_dim=int(val("lstm_dim", 192)),
        lstm_layers=int(val("lstm_layers", 2)),
        dropout=float(val("dropout", 0.1)),
        role=role,
        body_pos_indices=pos_idx,
        body_vel_indices=vel_idx,
        wheel_pos_indices=wheel_pos,
        wheel_omega_index=omega_idx,
        force_hidden_dim=int(val("force_hidden_dim", 48)),
    ).to(device)
    state = ckpt.get("model_state_dict")
    if not isinstance(state, dict):
        raise KeyError("V5 checkpoint is missing model_state_dict")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, {"role": role, "body_pos": pos_idx, "body_vel": vel_idx, "wheel_pos": wheel_pos, "wheel_omega": omega_idx}


def checkpoint_default(ckpt: Dict[str, Any], key: str, value):
    return (ckpt.get("args") or {}).get(key, value)


def select_split(df: pd.DataFrame, split: str, seed: int) -> pd.DataFrame:
    if split == "all":
        selected = df.copy()
    else:
        train, val, test = split_train_val_test_by_case(df, "case_name", 0.7, 0.15, seed)
        selected = {"train": train, "val": val, "test": test}[split]
    if selected.empty:
        raise ValueError(f"split={split!r} selects no cases")
    return selected.sort_values(["case_name", "time"]).reset_index(drop=True)


def map_case_basename(case_dir: Path) -> str:
    name = case_dir.name
    return "case" + name.split("_case", 1)[1] if "_case" in name else name


def build_control_params(values) -> Dict[str, float]:
    if not values:
        return {}
    if len(values) != len(CONTROL_PARAM_ORDER):
        raise ValueError(f"--control_params requires {len(CONTROL_PARAM_ORDER)} numbers")
    return {k: float(v) for k, v in zip(CONTROL_PARAM_ORDER, values)}


def build_custom_case(args, spec) -> pd.DataFrame:
    if not args.custom_case_dir:
        raise ValueError("--custom_case_dir is required for custom_case mode")
    case_name = args.infer_case_name or map_case_basename(Path(args.custom_case_dir))
    controls = build_control_params(args.control_params)
    lf = read_case_data(args.custom_case_dir, controls, compute_features=True)
    if lf is None or lf.empty:
        raise ValueError(f"failed to load Custom case: {args.custom_case_dir}")
    lf = add_prefix(lf.copy(), "lf_")
    lf["case_name"] = case_name
    if args.sph_case_dir:
        hf = read_case_data(args.sph_case_dir, controls, compute_features=False)
        if hf is None or hf.empty:
            raise ValueError(f"failed to load HF reference: {args.sph_case_dir}")
        hf = add_prefix(hf.copy(), "hf_")
        hf["case_name"] = case_name
        start = max(float(lf.time.min()), float(hf.time.min()))
        end = min(float(lf.time.max()), float(hf.time.max()))
        if end < start:
            raise ValueError("Custom and HF reference time ranges do not overlap")
        lf = lf[(lf.time >= start) & (lf.time <= end)].sort_values("time")
        hf = hf[(hf.time >= start) & (hf.time <= end)].sort_values("time")
        df = pd.merge_asof(lf, hf, on="time", by="case_name", direction="nearest")
        residuals = {}
        for hf_col in spec.target_cols:
            if hf_col.startswith("hf_"):
                lf_col = "lf_" + hf_col[3:]
                if lf_col in df and hf_col in df:
                    residuals["res_" + hf_col[3:]] = pd.to_numeric(df[hf_col], errors="coerce").fillna(0) - pd.to_numeric(df[lf_col], errors="coerce").fillna(0)
        if residuals:
            df = pd.concat([df, pd.DataFrame(residuals, index=df.index)], axis=1)
        df = add_rotvec_residuals(df, body=True, wheels=range(6))
    else:
        df = lf
    for col in spec.target_cols + spec.res_cols + spec.base_feature_cols + spec.proxy_feature_cols:
        if col not in df:
            df[col] = 0.0
    return ensure_time_feature_columns(df, "case_name", "time").sort_values(["case_name", "time"]).reset_index(drop=True)


def discover_case_dirs(parent: str, token: str) -> Dict[str, Path]:
    root = Path(parent)
    if not root.is_dir():
        raise FileNotFoundError(f"case directory not found: {root}")
    result = {}
    for child in sorted(root.iterdir()):
        if child.is_dir() and token in child.name and "_case" in child.name:
            name = map_case_basename(child)
            if name in result:
                raise ValueError(f"duplicate case label {name}: {result[name]} and {child}")
            result[name] = child
    if not result:
        raise ValueError(f"no {token} case directories in {root}")
    return result


def build_custom_cases(args, spec) -> pd.DataFrame:
    custom = discover_case_dirs(args.custom_cases_dir, "Custom")
    sph = discover_case_dirs(args.sph_cases_dir, "SPH") if args.sph_cases_dir else {}
    missing = sorted(set(custom) - set(sph)) if sph else []
    if missing:
        raise ValueError(f"Custom cases missing matching HF references: {missing[:10]}")
    frames = []
    for case_name, case_dir in tqdm(custom.items(), desc="Build V5 custom cases", dynamic_ncols=True):
        case_args = argparse.Namespace(**vars(args))
        case_args.custom_case_dir = str(case_dir)
        case_args.sph_case_dir = str(sph[case_name]) if sph else None
        case_args.infer_case_name = case_name
        frames.append(build_custom_case(case_args, spec))
    return pd.concat(frames, ignore_index=True).sort_values(["case_name", "time"]).reset_index(drop=True)


def full_case_dataset(df: pd.DataFrame, spec, scaler, role: str, history_len: int, future_len: int):
    dataset = GraphTemporalSequenceDatasetV5(
        df, spec, scaler, history_len=history_len,
        teacher_future_len=future_len if role == "teacher" else 0,
        body_rollout_len=1,
    )
    # Inference traverses every target timestamp after warm-up, including the case tail.
    index_map = []
    start = 0
    for _, case in dataset.df.groupby("case_name", sort=False):
        n = len(case)
        times = case.time.to_numpy(dtype=np.float64)
        if len(times) > 1 and np.any(np.diff(times) <= 0):
            raise ValueError("case timestamps must be strictly increasing for continuous state propagation")
        for local_t in range(history_len, n):
            index_map.append((start, start + local_t))
        start += n
    dataset.index_map = index_map
    if not index_map:
        raise ValueError("selected cases are too short for the requested history_len")
    return dataset


def append_quat(row: Dict[str, Any], prefix: str, quat: Optional[np.ndarray]) -> None:
    if quat is None:
        return
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    q = q / max(np.linalg.norm(q), 1e-12)
    for i, val in enumerate(q):
        row[f"{prefix}::q{i}"] = float(val)


def add_pose_refs(row: Dict[str, Any], name: str, lf_pose: Optional[torch.Tensor], hf_pose: Optional[torch.Tensor], pred_rot: torch.Tensor) -> None:
    pred_q = matrix_to_quat(pred_rot.detach().cpu().numpy()[0, 0])
    append_quat(row, f"pred::{name}", pred_q)
    for label, pose in (("lf", lf_pose), ("true", hf_pose)):
        if pose is not None:
            append_quat(row, f"{label}::{name}", pose.detach().cpu().numpy()[0, 0, 3:7])


def write_vector(row: Dict[str, Any], prefix: str, values, names: Tuple[str, ...] = AXES) -> None:
    a = values.detach().cpu().numpy().reshape(-1) if isinstance(values, torch.Tensor) else np.asarray(values).reshape(-1)
    for key, val in zip(names, a):
        row[f"{prefix}::{key}"] = float(val)


def build_model_state(
    model, batch, sample_index: int, sample, dataset, spec, scaler, outputs,
    previous: Dict[str, Any], role: str, history_len: int, force_params=None,
):
    device = outputs["pred_res_body"].device
    body_cols = spec.target_groups.body_cols
    body_pos = indices(body_cols, [f"pos_{x}" for x in AXES])
    body_vel = indices(body_cols, [f"vel_{x}" for x in AXES])
    lf_body = batch["lf_body_current"].float()
    global_row = dataset.index_map[sample_index][1]
    body_delta = inverse_residual(outputs["pred_res_body"], spec.res_groups.body_cols, body_cols, "res_body", scaler)
    velocity = lf_body[..., body_vel] + body_delta[..., body_vel]
    dt = float(batch["dt"].reshape(-1)[0].item())
    if dt <= 0:
        raise ValueError(f"non-positive dt at {sample['case_name']} time={sample['time']}: {dt}")
    if previous.get("case") != sample["case_name"]:
        lf_prev = batch["lf_body_prev_raw"].float()
        prev_pos = lf_prev[..., body_pos]
        prev_vel = lf_prev[..., body_vel]
        previous.clear()
        previous.update({"case": sample["case_name"], "pos": prev_pos, "vel": prev_vel, "wheels": {}, "rockers": {}})
    else:
        prev_pos = previous["pos"]
        prev_vel = previous["vel"]
    dt_t = velocity.new_tensor(dt).view(1, 1, 1)
    prop = prev_pos + 0.5 * (prev_vel + velocity) * dt_t
    body_gate = outputs["gate_body_pos"]
    body_final = prop + body_gate * body_delta[..., body_pos]
    body_global_delta = body_final - lf_body[..., body_pos]

    body_pose_lf = batch.get("lf_body_pose_current")
    body_pose_hf = batch.get("hf_body_pose")
    if body_pose_lf is not None:
        r_lf = quat_matrix(body_pose_lf[..., 3:7])
    else:
        r_lf = torch.eye(3, device=device).view(1, 1, 3, 3)
    r_body = torch.matmul(r_lf, rotvec_matrix(outputs["gate_body_att"] * outputs["pred_delta_rot_body"]))

    state = {
        "z": outputs["z"], "body_node": outputs["body_node"],
        "wheel_kin_nodes": outputs["wheel_kin_nodes"], "wheel_contact_nodes": outputs["wheel_contact_nodes"],
        "body_velocity": velocity, "body_rotation": r_body,
        "wheel_position": {}, "wheel_omega": {}, "force_base": {}, "fphy_z": {},
    }
    previous_vel = prev_vel
    acceleration = (velocity - previous_vel) / dt_t
    state["body_acceleration"] = acceleration
    roll = torch.atan2(r_body[..., 2, 1], r_body[..., 2, 2]).unsqueeze(-1)
    pitch = torch.asin((-r_body[..., 2, 0]).clamp(-1, 1)).unsqueeze(-1)
    state["body_roll"], state["body_pitch"] = roll, pitch

    row: Dict[str, Any] = {"case_name": str(sample["case_name"]), "time": float(sample["time"]), "dt": dt}
    write_vector(row, "lf::body::position", lf_body[..., body_pos])
    write_vector(row, "lf::body::velocity", lf_body[..., body_vel])
    write_vector(row, "pred_prop::body::position", prop)
    write_vector(row, "pred::body::position", body_final)
    write_vector(row, "pred::body::velocity", velocity)
    write_vector(row, "correction::body::delta_p", body_delta[..., body_pos])
    write_vector(row, "correction::body::delta_v", body_delta[..., body_vel])
    write_vector(row, "correction::body::delta_rot", outputs["pred_delta_rot_body"])
    write_vector(row, "gate::body::position", body_gate)
    write_vector(row, "gate::body::attitude", outputs["gate_body_att"])
    true_body = inverse_group(batch["hf_body"], scaler, "hf_body")
    write_vector(row, "true::body::position", true_body[..., body_pos])
    write_vector(row, "true::body::velocity", true_body[..., body_vel])
    add_pose_refs(row, "body", body_pose_lf, body_pose_hf, r_body)

    rocker_states = {}
    for name in ROCKER_NAMES:
        lf_key, hf_key = f"lf_rocker_{name}_pose_current", f"hf_rocker_{name}_pose"
        if lf_key not in batch:
            continue
        lf_pose, hf_pose = batch[lf_key].float(), batch.get(hf_key)
        if hf_pose is not None:
            hf_pose = hf_pose.float()
        local = outputs[f"pred_rocker_{name}_local_delta_pos"]
        gate = outputs[f"gate_rocker_{name}_pos"]
        local_world = torch.matmul(r_body, (gate * local).unsqueeze(-1)).squeeze(-1)
        base = lf_pose[..., :3] + body_global_delta
        pred = base + local_world
        prev_rocker = previous["rockers"].get(name)
        if prev_rocker is None:
            lf_prev_cols = [f"lf_susp_rocker_{name}_pos_{axis}" for axis in AXES]
            if all(c in dataset.df.columns for c in lf_prev_cols) and global_row > 0:
                prev_values = dataset.df.iloc[global_row - 1][lf_prev_cols].to_numpy(dtype=np.float32)
                prev_rocker = torch.as_tensor(prev_values, device=device).view(1, 1, 3)
            else:
                prev_rocker = lf_pose[..., :3]
        rocker_velocity = (pred - prev_rocker) / dt_t
        lf_rot = quat_matrix(lf_pose[..., 3:7])
        rot = torch.matmul(torch.matmul(torch.matmul(r_body, r_lf.transpose(-1, -2)), lf_rot), rotvec_matrix(outputs[f"gate_rocker_{name}_att"] * outputs[f"pred_rocker_{name}_delta_rot"]))
        hf_pos = hf_pose[..., :3] if hf_pose is not None else None
        prefix = f"rocker_{name}"
        write_vector(row, f"lf::{prefix}::position", lf_pose[..., :3])
        write_vector(row, f"pred::{prefix}::velocity", rocker_velocity)
        write_vector(row, f"body_base::{prefix}::position", base)
        write_vector(row, f"pred::{prefix}::position", pred)
        write_vector(row, f"correction::{prefix}::local_delta", local)
        write_vector(row, f"correction::{prefix}::delta_rot", outputs[f"pred_rocker_{name}_delta_rot"])
        write_vector(row, f"gate::{prefix}::position", gate)
        write_vector(row, f"gate::{prefix}::attitude", outputs[f"gate_rocker_{name}_att"])
        if hf_pos is not None:
            write_vector(row, f"true::{prefix}::position", hf_pos)
        for label, source_prefix in (("lf", "lf"), ("true", "hf")):
            velocity_cols = [f"{source_prefix}_susp_rocker_{name}_vel_{axis}" for axis in AXES]
            if all(c in dataset.df.columns for c in velocity_cols):
                write_vector(row, f"{label}::{prefix}::velocity", dataset.df.iloc[global_row][velocity_cols].to_numpy(dtype=np.float32))
        add_pose_refs(row, prefix, lf_pose, hf_pose, rot)
        rocker_states[name] = {"pos": pred, "rot": rot, "vel": rocker_velocity}

    wheel_states = {}
    wheel_pos_idx = indices(spec.target_groups.wheel_kin_cols[0], [f"pos_{x}" for x in AXES])
    for wheel in WHEEL_IDS:
        target_cols = spec.target_groups.wheel_kin_cols[wheel]
        residual_cols = spec.res_groups.wheel_kin_cols[wheel]
        lf_w = batch[f"lf_wheel{wheel}_kin_current"].float()
        res = inverse_residual(outputs[f"pred_res_wheel{wheel}_kin"], residual_cols, target_cols, f"res_wheel{wheel}_kin", scaler)
        local = res[..., wheel_pos_idx]
        gate = outputs[f"gate_wheel{wheel}_pos"]
        local_world = torch.matmul(r_body, (gate * local).unsqueeze(-1)).squeeze(-1)
        base = lf_w[..., wheel_pos_idx] + body_global_delta
        pred = base + local_world
        wheel_states[wheel] = pred
        state["wheel_position"][wheel] = pred
        omega_idx = omega_index(target_cols)
        lf_omega = lf_w[..., omega_idx:omega_idx + 1] if omega_idx is not None else lf_w.new_zeros(*lf_w.shape[:-1], 1)
        if omega_idx is not None:
            omega_pred = lf_omega + outputs[f"gate_wheel{wheel}_omega"] * res[..., omega_idx:omega_idx + 1]
        else:
            omega_pred = lf_omega
        state["wheel_omega"][wheel] = omega_pred
        previous_wheel = previous["wheels"].get(wheel)
        if previous_wheel is None:
            prev_lf = dataset.df.iloc[global_row - 1][dataset.lf_wheel_kin_cols[wheel]].to_numpy(dtype=np.float32)
            prev_pos_np = prev_lf[wheel_pos_idx]
            previous_wheel = torch.as_tensor(prev_pos_np, device=device, dtype=torch.float32).view(1, 1, 3)
        wheel_vz = (pred[..., 2:3] - previous_wheel[..., 2:3]) / dt_t
        wheel_velocity = (pred - previous_wheel) / dt_t
        if force_params is not None:
            fphy = load_transfer_fz_torch(roll, pitch, acceleration, wheel_vz, wheel, force_params)
            lf_force = batch[f"wheel{wheel}_lf_force_current"].float()
            base_force = torch.cat([lf_force[..., :2], fphy], dim=-1)
            state["force_base"][wheel] = base_force
            state["fphy_z"][wheel] = fphy
            row.update({f"lf_force::wheel{wheel}::{axis}": float(lf_force[0, 0, j].item()) for j, axis in enumerate(FORCE_AXES)})
            row.update({f"physics_base::wheel{wheel}::{axis}": float(base_force[0, 0, j].item()) for j, axis in enumerate(FORCE_AXES)})
            row[f"Fphy_z::wheel{wheel}"] = float(fphy[0, 0, 0].item())
            hf_force = inverse_group(batch[f"hf_wheel{wheel}_contact"], scaler, f"hf_wheel{wheel}_contact")
            for j, axis in enumerate(FORCE_AXES):
                row[f"true_force::wheel{wheel}::{axis}"] = float(hf_force[0, 0, j].item())
        lf_pose_key, hf_pose_key = f"lf_wheel{wheel}_pose_current", f"hf_wheel{wheel}_pose"
        lf_pose = batch.get(lf_pose_key)
        hf_pose = batch.get(hf_pose_key)
        if lf_pose is not None:
            lf_pose = lf_pose.float()
            hf_pose = hf_pose.float() if hf_pose is not None else None
            rot = torch.matmul(torch.matmul(torch.matmul(r_body, r_lf.transpose(-1, -2)), quat_matrix(lf_pose[..., 3:7])), rotvec_matrix(outputs[f"gate_wheel{wheel}_att"] * outputs[f"pred_wheel{wheel}_delta_rot"]))
            prefix = f"wheel{wheel}"
            add_pose_refs(row, prefix, lf_pose, hf_pose, rot)
        write_vector(row, f"lf::wheel{wheel}::position", lf_w[..., wheel_pos_idx])
        write_vector(row, f"pred::wheel{wheel}::velocity", wheel_velocity)
        write_vector(row, f"body_base::wheel{wheel}::position", base)
        write_vector(row, f"pred::wheel{wheel}::position", pred)
        write_vector(row, f"correction::wheel{wheel}::local_delta", local)
        write_vector(row, f"correction::wheel{wheel}::delta_rot", outputs[f"pred_wheel{wheel}_delta_rot"])
        write_vector(row, f"gate::wheel{wheel}::position", gate)
        write_vector(row, f"gate::wheel{wheel}::attitude", outputs[f"gate_wheel{wheel}_att"])
        if omega_idx is not None:
            row[f"lf::wheel{wheel}::omega"] = float(lf_omega[0, 0, 0].item())
            row[f"pred::wheel{wheel}::omega"] = float(omega_pred[0, 0, 0].item())
            row[f"correction::wheel{wheel}::delta_omega"] = float(res[0, 0, omega_idx].item())
        true_wheel = inverse_group(batch[f"hf_wheel{wheel}_kin"], scaler, f"hf_wheel{wheel}_kin")
        write_vector(row, f"true::wheel{wheel}::position", true_wheel[..., wheel_pos_idx])
        for label, source_prefix in (("lf", "lf"), ("true", "hf")):
            velocity_cols = [f"{source_prefix}_wheel{wheel}_vel_{axis}" for axis in AXES]
            if all(c in dataset.df.columns for c in velocity_cols):
                write_vector(row, f"{label}::wheel{wheel}::velocity", dataset.df.iloc[global_row][velocity_cols].to_numpy(dtype=np.float32))

    previous.update({"case": sample["case_name"], "pos": body_final, "vel": velocity, "wheels": wheel_states, "rockers": {name: state["pos"] for name, state in rocker_states.items()}})
    return row, state


def finite_rmse(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2))) if mask.any() else float("nan")


def summarize(rows: pd.DataFrame, has_hf: bool, mode: str) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {"num_rows": int(len(rows)), "num_cases": int(rows.case_name.nunique()), "mode": mode}
    if not has_hf or rows.empty:
        return metrics
    for group in ("body", *[f"rocker_{n}" for n in ROCKER_NAMES], *[f"wheel{i}" for i in WHEEL_IDS]):
        axes = AXES
        for kind in ("position",):
            for source in ("lf", "body_base", "pred_prop", "pred"):
                cols = [f"{source}::{group}::{kind}::{axis}" for axis in axes]
                if source == "body_base" and group == "body":
                    continue
                if source == "pred_prop" and group != "body":
                    continue
                truth_cols = [f"true::{group}::{kind}::{axis}" for axis in axes]
                if all(c in rows for c in cols + truth_cols):
                    pred = rows[cols].to_numpy(dtype=np.float64)
                    true = rows[truth_cols].to_numpy(dtype=np.float64)
                    rmse = finite_rmse(pred, true)
                    metrics[f"{group}_{source}_position_rmse"] = rmse
                    if source in {"lf", "pred"}:
                        lf_cols = [f"lf::{group}::position::{axis}" for axis in axes]
                        if source == "pred" and all(c in rows for c in lf_cols):
                            lf_arr = rows[lf_cols].to_numpy(dtype=np.float64)
                            pred_err = np.mean((pred - true) ** 2, axis=1)
                            lf_err = np.mean((lf_arr - true) ** 2, axis=1)
                            metrics[f"{group}_pred_improvement_rate"] = float((finite_rmse(lf_arr, true) - rmse) / max(finite_rmse(lf_arr, true), 1e-12))
                            metrics[f"{group}_pred_harmful_ratio"] = float(np.mean(pred_err > lf_err))
        if group == "body":
            for axis in AXES:
                for source in ("lf", "pred"):
                    col = f"{source}::body::velocity::{axis}" if source == "pred" else f"lf::body::velocity::{axis}"
                    true_col = f"true::body::velocity::{axis}"
                    if col in rows and true_col in rows:
                        metrics[f"body_{source}_velocity_{axis}_rmse"] = finite_rmse(rows[col].to_numpy(), rows[true_col].to_numpy())
    for wheel in WHEEL_IDS:
        for axis in FORCE_AXES:
            pred, truth = f"pred_force::wheel{wheel}::{axis}", f"true_force::wheel{wheel}::{axis}"
            if pred in rows and truth in rows:
                metrics[f"wheel{wheel}_{axis}_rmse"] = finite_rmse(rows[pred].to_numpy(), rows[truth].to_numpy())
            baseline = f"physics_base::wheel{wheel}::{axis}"
            if baseline in rows and truth in rows:
                metrics[f"wheel{wheel}_{axis}_base_rmse"] = finite_rmse(rows[baseline].to_numpy(), rows[truth].to_numpy())
    return metrics


def create_plots(rows: pd.DataFrame, output_dir: Path) -> None:
    if rows.empty:
        return
    plot_root = output_dir / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    groups = ["body", *[f"rocker_{name}" for name in ROCKER_NAMES], *[f"wheel{i}" for i in WHEEL_IDS]]
    for case_name, frame in rows.groupby("case_name", sort=False):
        frame = frame.sort_values("time")
        out = plot_root / str(case_name)
        out.mkdir(parents=True, exist_ok=True)
        time = frame.time.to_numpy(dtype=np.float64)
        for group in groups:
            for axis in AXES:
                if group == "body":
                    keys = {"LF": f"lf::body::position::{axis}", "Propagated": f"pred_prop::body::position::{axis}", "Prediction": f"pred::body::position::{axis}", "HF": f"true::body::position::{axis}"}
                else:
                    keys = {"LF": f"lf::{group}::position::{axis}", "Body base": f"body_base::{group}::position::{axis}", "Prediction": f"pred::{group}::position::{axis}", "HF": f"true::{group}::position::{axis}"}
                present = [(label, key) for label, key in keys.items() if key in frame]
                if len(present) < 2:
                    continue
                fig, ax = plt.subplots(figsize=(10, 4))
                for label, key in present:
                    ax.plot(time, frame[key].to_numpy(dtype=np.float64), label=label)
                ax.set(title=f"{case_name} | {group} | position {axis}", xlabel="time / s", ylabel=f"position {axis}")
                ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
                fig.savefig(out / f"{group}_position_{axis}.png", dpi=130); plt.close(fig)


def run_inference(args: argparse.Namespace) -> Dict[str, Any]:
    if args.mode == "auto":
        probe = load_checkpoint(args.checkpoint, torch.device("cpu"))
        stage = str(probe.get("train_stage", probe.get("v5_role", (probe.get("args") or {}).get("train_stage", ""))))
        args.mode = "state_teacher" if stage == "state_teacher" else "force" if stage in {"force", "joint"} else "state_student"
        del probe
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = load_checkpoint(args.checkpoint, device)
    expected_stage = str(ckpt.get("train_stage", ckpt.get("v5_role", (ckpt.get("args") or {}).get("train_stage", ""))))
    if args.mode == "state_teacher" and expected_stage not in {"state_teacher", ""}:
        raise ValueError(f"state_teacher mode expects a Teacher checkpoint, got train_stage={expected_stage}")
    if args.mode == "state_student" and expected_stage not in {"state_student", ""}:
        raise ValueError(f"state_student mode expects a Student checkpoint, got train_stage={expected_stage}")
    if args.mode == "force" and expected_stage not in {"force", "joint", ""}:
        raise ValueError(f"force mode expects a force/joint checkpoint, got train_stage={expected_stage}")

    spec = load_column_spec(args.feature_dir)
    scaler = restore_scaler(args.checkpoint, ckpt)
    if args.infer_mode == "merged_csv":
        df = load_merged_dataset(args.merged_csv, spec, "case_name", "time")
        seed = int(args.seed if args.seed is not None else checkpoint_default(ckpt, "seed", 42))
        df = select_split(df, args.split, seed)
        if args.case_name:
            df = df[df.case_name.astype(str) == str(args.case_name)].copy()
            if df.empty:
                raise ValueError(f"case_name={args.case_name!r} not found in selected split")
    elif args.infer_mode == "custom_case":
        df = build_custom_case(args, spec)
    else:
        df = build_custom_cases(args, spec)
    df = df.sort_values(["case_name", "time"]).reset_index(drop=True)
    has_hf = all(col in df.columns for col in spec.target_cols) and (args.infer_mode == "merged_csv" or bool(args.sph_case_dir or args.sph_cases_dir))

    ckpt_args = ckpt.get("args") or {}
    history_len = int(args.history_len if args.history_len is not None else checkpoint_default(ckpt, "history_len", 9))
    future_len = int(args.teacher_future_len if args.teacher_future_len is not None else checkpoint_default(ckpt, "teacher_future_len", 3))
    role = "teacher" if args.mode == "state_teacher" else "student"
    dataset = full_case_dataset(df, spec, scaler, role, history_len, future_len)
    if args.max_samples is not None:
        dataset.index_map = dataset.index_map[:max(0, args.max_samples)]
    model, model_info = build_model(ckpt, spec, args.mode, device)
    force_params = ckpt.get("load_transfer_params") if args.mode == "force" else None
    if args.mode == "force" and force_params is None:
        raise KeyError("force checkpoint is missing train-only load_transfer_params")
    if args.mode == "force":
        force_params = torch.as_tensor(force_params, dtype=torch.float32, device=device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.infer_mode != "merged_csv":
        df.to_csv(output_dir / "generated_infer_input_v5.csv", index=False)
    prefix = role
    previous: Dict[str, Any] = {}
    records = []
    latent_records = []
    for sample_index in tqdm(range(len(dataset)), desc=f"V5 {args.mode}", dynamic_ncols=True):
        sample = dataset[sample_index]
        batch = graph_temporal_collate_fn([sample])
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.no_grad():
            state_out = model.forward_state(batch, input_prefix=prefix, current_index=history_len)
        row, corrected = build_model_state(
            model, batch, sample_index, sample, dataset, spec, scaler, state_out,
            previous, role, history_len, force_params,
        )
        if args.mode == "force":
            with torch.no_grad():
                force_out = model.forward_force(corrected, batch, input_prefix="student", current_index=history_len)
            for wheel in WHEEL_IDS:
                delta = force_out[f"delta_force_wheel{wheel}"][0, 0].detach().cpu().numpy()
                pred = force_out[f"force_pred_wheel{wheel}"][0, 0].detach().cpu().numpy()
                for j, axis in enumerate(FORCE_AXES):
                    row[f"delta_force::wheel{wheel}::{axis}"] = float(delta[j])
                    row[f"pred_force::wheel{wheel}::{axis}"] = float(pred[j])
        records.append(row)
        z = state_out["z"].detach().cpu().numpy().reshape(-1)
        latent_records.append({"case_name": str(sample["case_name"]), "time": float(sample["time"]), **{f"z_{i:03d}": float(v) for i, v in enumerate(z)}})

    result = pd.DataFrame(records)
    if result.empty:
        raise ValueError("no inferable rows after history warm-up")
    result = result.sort_values(["case_name", "time"]).reset_index(drop=True)
    prediction_path = output_dir / "predictions_v5.csv"
    result.to_csv(prediction_path, index=False)
    pd.DataFrame(latent_records).to_csv(output_dir / "latents_v5.csv", index=False)
    create_plots(result, output_dir)
    metrics = summarize(result, has_hf, args.mode)
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()), "mode": args.mode,
        "infer_mode": args.infer_mode, "split": args.split,
        "history_len": history_len, "teacher_future_len": future_len if role == "teacher" else 0,
        "body_rollout_len": int(checkpoint_default(ckpt, "body_rollout_len", 5)),
        "uses_future_lf": role == "teacher", "num_rows": len(result),
        "num_cases": int(result.case_name.nunique()), "has_hf_reference": bool(has_hf),
        "device": str(device), "metrics": metrics,
    }
    (output_dir / "metrics_v5.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    info = {"model": model_info, "checkpoint_stage": expected_stage, "state_architecture_args": ckpt.get("state_architecture_args"), "force_architecture_args": ckpt.get("force_architecture_args")}
    (output_dir / "run_info_v5.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"predictions: {prediction_path}")
    log(f"metrics: {output_dir / 'metrics_v5.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent V5 state and force inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=["auto", "state_teacher", "state_student", "force"], default="auto")
    parser.add_argument("--feature_dir", default=str(ROOT / "Feature_Selection" / "DataSet"))
    parser.add_argument("--merged_csv", default=str(ROOT / "Feature_Selection" / "DataSet" / "merged_error_dataset.csv"))
    parser.add_argument("--infer_mode", choices=["merged_csv", "custom_case", "custom_cases"], default="merged_csv")
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="test")
    parser.add_argument("--case_name")
    parser.add_argument("--custom_case_dir")
    parser.add_argument("--sph_case_dir")
    parser.add_argument("--custom_cases_dir", default=None)
    parser.add_argument("--sph_cases_dir", default=None)
    parser.add_argument("--infer_case_name")
    parser.add_argument("--control_params", type=float, nargs=12)
    parser.add_argument("--output_dir")
    parser.add_argument("--history_len", type=int)
    parser.add_argument("--teacher_future_len", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    if args.output_dir is None:
        ckpt = Path(args.checkpoint).resolve().parent
        args.output_dir = str(ckpt / f"infer_v5_{args.mode}_{args.split}_{datetime.now():%Y%m%d_%H%M%S}")
    if args.infer_mode == "custom_cases" and not args.custom_cases_dir:
        parser.error("--custom_cases_dir is required for custom_cases mode")
    return args


if __name__ == "__main__":
    run_inference(parse_args())
