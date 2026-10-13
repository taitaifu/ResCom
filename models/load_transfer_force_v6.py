from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

WHEEL_IDS = tuple(range(6))
LOAD_TRANSFER_FEATURES = ("bias", "roll", "pitch", "acc_x", "acc_y", "acc_z", "wheel_down_v")


def _first_existing(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    available = set(columns)
    return next((name for name in candidates if name in available), None)


def _quaternion_roll_pitch(df: pd.DataFrame, prefix: str) -> tuple[np.ndarray, np.ndarray] | None:
    cols = [f"{prefix}_q0", f"{prefix}_q1", f"{prefix}_q2", f"{prefix}_q3"]
    if not all(col in df.columns for col in cols):
        return None
    q = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q.T
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return roll.astype(np.float32), pitch.astype(np.float32)


def _state_roll_pitch(df: pd.DataFrame, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    roll_col = _first_existing(df.columns, (f"{prefix}_roll",))
    pitch_col = _first_existing(df.columns, (f"{prefix}_pitch",))
    quat = _quaternion_roll_pitch(df, prefix)
    if roll_col is not None and pitch_col is not None:
        return (
            pd.to_numeric(df[roll_col], errors="coerce").fillna(0).to_numpy(dtype=np.float32),
            pd.to_numeric(df[pitch_col], errors="coerce").fillna(0).to_numpy(dtype=np.float32),
        )
    if quat is None:
        raise ValueError(f"V6 train-only load-transfer fit needs {prefix} roll/pitch or quaternion columns")
    return quat


def _finite_difference_by_case(df: pd.DataFrame, values: np.ndarray, case_col: str, time_col: str) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    for _, group in df.groupby(case_col, sort=False).indices.items():
        idx = np.asarray(group, dtype=np.int64)
        order = np.argsort(pd.to_numeric(df.iloc[idx][time_col], errors="coerce").to_numpy(dtype=np.float64), kind="stable")
        idx = idx[order]
        times = pd.to_numeric(df.iloc[idx][time_col], errors="coerce").to_numpy(dtype=np.float64)
        dt = np.diff(times)
        valid = dt > 0
        delta = np.diff(values[idx], axis=0)
        derivative = np.zeros_like(delta, dtype=np.float32)
        derivative[valid] = (delta[valid] / dt[valid].reshape((-1,) + (1,) * (delta.ndim - 1))).astype(np.float32)
        out[idx[1:]] = derivative
    return out


def fit_load_transfer_params(
    df_train: pd.DataFrame,
    case_col: str = "case_name",
    time_col: str = "time",
    ridge: float = 1e-3,
    config=None,
) -> np.ndarray:
    """Fit six wheel-specific Fz regressions from train split only; rows are [bias, roll, pitch, ax, ay, az, wheel_down_v]."""
    required = [f"hf_vel_{axis}" for axis in "xyz"]
    if not all(col in df_train.columns for col in required):
        raise ValueError(f"V6 load-transfer calibration needs HF body velocity columns: {required}")
    roll, pitch = _state_roll_pitch(df_train, "hf")
    velocity = df_train[required].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    acceleration = _finite_difference_by_case(df_train, velocity, case_col, time_col)
    x_parts = [np.ones((len(df_train), 1), dtype=np.float32), roll[:, None], pitch[:, None], acceleration]
    params = np.zeros((len(WHEEL_IDS), len(LOAD_TRANSFER_FEATURES)), dtype=np.float32)
    for wheel_id in WHEEL_IDS:
        z_col = f"hf_wheel{wheel_id}_pos_z"
        fz_col = f"hf_wheel{wheel_id}_Fz"
        if z_col not in df_train.columns or fz_col not in df_train.columns:
            raise ValueError(f"V6 load-transfer calibration requires {z_col!r} and {fz_col!r} in df_train")
        wheel_z = pd.to_numeric(df_train[z_col], errors="coerce").to_numpy(dtype=np.float32)
        wheel_vz = _finite_difference_by_case(df_train, wheel_z[:, None], case_col, time_col)[:, 0]
        x = np.concatenate(x_parts + [np.maximum(-wheel_vz, 0.0)[:, None]], axis=1).astype(np.float64)
        y = pd.to_numeric(df_train[fz_col], errors="coerce").to_numpy(dtype=np.float64)
        if config is not None and config.input_force_frame=='world':
            from .data_utils_v6 import TerrainParameterLookup
            force_cols=[f'hf_wheel{wheel_id}_{axis}' for axis in ('Fx','Fy','Fz')]
            if not all(c in df_train for c in force_cols): raise ValueError('world force calibration requires HF xyz force')
            from pathlib import Path
            _,normal,_=TerrainParameterLookup.get(Path(config.terrain_file)).contact_plane(
                df_train[f'hf_wheel{wheel_id}_pos_x'].to_numpy(),df_train[f'hf_wheel{wheel_id}_pos_y'].to_numpy())
            y=(df_train[force_cols].to_numpy(np.float64)*normal).sum(-1)
        valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
        if valid.sum() < x.shape[1]:
            raise ValueError(f"Not enough valid train rows to fit wheel {wheel_id} load-transfer parameters")
        gram = x[valid].T @ x[valid]
        penalty = np.eye(gram.shape[0], dtype=np.float64) * float(ridge)
        penalty[0, 0] = 0.0
        params[wheel_id] = np.linalg.solve(gram + penalty, x[valid].T @ y[valid]).astype(np.float32)
    return params


def load_transfer_fz_torch(
    roll: torch.Tensor,
    pitch: torch.Tensor,
    acc_xyz: torch.Tensor,
    wheel_vz: torch.Tensor,
    wheel_id: int,
    params: torch.Tensor | np.ndarray,
) -> torch.Tensor:
    if not 0 <= int(wheel_id) < len(WHEEL_IDS):
        raise IndexError(f"wheel_id must be 0..5, got {wheel_id}")
    reference = acc_xyz
    feature = torch.cat([
        torch.ones_like(roll), roll, pitch, acc_xyz,
        torch.relu(-wheel_vz),
    ], dim=-1)
    coefficients = torch.as_tensor(params, device=reference.device, dtype=reference.dtype)[int(wheel_id)]
    if feature.shape[-1] != coefficients.shape[-1]:
        raise ValueError(f"load-transfer feature dim {feature.shape[-1]} != coefficient dim {coefficients.shape[-1]}")
    return (feature * coefficients).sum(dim=-1, keepdim=True)
