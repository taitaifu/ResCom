from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

WHEEL_IDS = tuple(range(6))
LOAD_TRANSFER_KEYS = ["roll", "pitch", "acc_x", "acc_y", "acc_z"]

# Identified from results_v3/diagnostics/hf_fz_load_transfer_case_val.
# Shape: [wheel, Fx/Fy/Fz].
DEFAULT_FORCE_BIAS = np.asarray(
    [
        [-3.3391566276550293, -0.9938580989837646, 359.0706787109375],
        [-0.03626157343387604, 0.7251942753791809, 289.4533996582031],
        [3.223991632461548, -2.808744192123413, 310.7593688964844],
        [0.9743970036506653, 2.115260124206543, 375.44940185546875],
        [1.2003172636032104, 1.166197657585144, 349.1532287597656],
        [-0.4608350992202759, -0.2934573292732239, 407.69000244140625],
    ],
    dtype=np.float32,
)

# Shape: [wheel, roll/pitch/acc_x/acc_y/acc_z].
DEFAULT_FZ_LOAD = np.asarray(
    [
        [-35.62766647338867, 19.603445053100586, -28.38101577758789, -11.31646728515625, 31.516796112060547],
        [34.67719650268555, -10.007822036743164, -27.092493057250977, 15.316393852233887, 29.20887565612793],
        [34.63836669921875, 34.454471588134766, 4.645228862762451, -9.493951797485352, 30.52886199951172],
        [26.09284782409668, 33.889400482177734, -1.3467657566070557, 15.00442123413086, 32.08620071411133],
        [33.623451232910156, -35.67652130126953, 26.384475708007812, -12.280450820922852, 29.860218048095703],
        [-34.7008056640625, -34.72236251831055, 27.835205078125, 14.090580940246582, 31.101062774658203],
    ],
    dtype=np.float32,
)

DEFAULT_FZ_DOWN_V = np.asarray(
    [-24.41008758544922, 26.930721282958984, -24.814929962158203, 0.8698863983154297, 24.979097366333008, 26.07672691345215],
    dtype=np.float32,
)


def _first_existing(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    col_set = set(columns)
    for name in candidates:
        if name in col_set:
            return name
    return None


def load_transfer_aux_columns(columns: Iterable[str]) -> Dict[str, str]:
    candidates = {
        "roll": ["lf_roll"],
        "pitch": ["lf_pitch"],
        "acc_x": ["lf_acc_x"],
        "acc_y": ["lf_acc_y"],
        "acc_z": ["lf_acc_z"],
    }
    out: Dict[str, str] = {}
    for key, names in candidates.items():
        col = _first_existing(columns, names)
        if col is not None:
            out[key] = col
    return out


def _quat_roll_pitch(df: pd.DataFrame, prefix: str) -> Tuple[np.ndarray, np.ndarray] | None:
    cols = [f"{prefix}_q0", f"{prefix}_q1", f"{prefix}_q2", f"{prefix}_q3"]
    if not all(c in df.columns for c in cols):
        return None
    q = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.maximum(norm, 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    return roll.astype(np.float32, copy=False), pitch.astype(np.float32, copy=False)


def load_transfer_aux_from_df(df: pd.DataFrame, aux_cols: Dict[str, str] | None = None) -> np.ndarray:
    aux_cols = aux_cols or load_transfer_aux_columns(df.columns.astype(str))
    out = np.zeros((len(df), len(LOAD_TRANSFER_KEYS)), dtype=np.float32)
    missing = []
    for idx, key in enumerate(LOAD_TRANSFER_KEYS):
        col = aux_cols.get(key)
        if col is not None and col in df.columns:
            out[:, idx] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32)
        elif key not in {"roll", "pitch"}:
            missing.append(f"lf_{key}")
    if "roll" not in aux_cols or "pitch" not in aux_cols:
        quat_rp = _quat_roll_pitch(df, "lf")
        if quat_rp is not None:
            roll, pitch = quat_rp
            if "roll" not in aux_cols:
                out[:, LOAD_TRANSFER_KEYS.index("roll")] = roll
            if "pitch" not in aux_cols:
                out[:, LOAD_TRANSFER_KEYS.index("pitch")] = pitch
        else:
            if "roll" not in aux_cols:
                missing.append("lf_roll or lf_q0/lf_q1/lf_q2/lf_q3")
            if "pitch" not in aux_cols:
                missing.append("lf_pitch or lf_q0/lf_q1/lf_q2/lf_q3")
    if missing:
        raise ValueError("V4 load-transfer Fphy requires LF-only inputs; missing: " + ", ".join(missing))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def wheel_vz_from_df(df: pd.DataFrame, wheel_id: int) -> np.ndarray:
    candidates = [
        f"lf_wheel{wheel_id}_vel_z",
        f"lf_wheel{wheel_id}_lin_vel_z",
    ]
    col = _first_existing(df.columns.astype(str), candidates)
    if col is None:
        raise ValueError(
            f"V4 load-transfer Fphy requires LF-only wheel vertical velocity for wheel {wheel_id}; "
            f"missing one of: {', '.join(candidates)}"
        )
    values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def load_transfer_force_numpy(aux: np.ndarray, wheel_vz: np.ndarray, wheel_id: int) -> np.ndarray:
    aux = np.asarray(aux, dtype=np.float32)
    wheel_vz = np.asarray(wheel_vz, dtype=np.float32)
    force = np.broadcast_to(DEFAULT_FORCE_BIAS[wheel_id], aux.shape[:-1] + (3,)).copy()
    fz_delta = (aux * DEFAULT_FZ_LOAD[wheel_id]).sum(axis=-1) + DEFAULT_FZ_DOWN_V[wheel_id] * np.maximum(-wheel_vz, 0.0)
    force[..., 2] += fz_delta
    return force.astype(np.float32, copy=False)


def load_transfer_force_torch(aux: torch.Tensor, wheel_vz: torch.Tensor, wheel_id: int) -> torch.Tensor:
    dtype = aux.dtype
    device = aux.device
    force_bias = torch.as_tensor(DEFAULT_FORCE_BIAS[wheel_id], device=device, dtype=dtype)
    fz_load = torch.as_tensor(DEFAULT_FZ_LOAD[wheel_id], device=device, dtype=dtype)
    fz_down_v = torch.as_tensor(DEFAULT_FZ_DOWN_V[wheel_id], device=device, dtype=dtype)
    force = force_bias.view(*([1] * (aux.ndim - 1)), 3).expand(*aux.shape[:-1], 3).clone()
    force[..., 2] = force[..., 2] + (aux * fz_load).sum(dim=-1) + fz_down_v * torch.relu(-wheel_vz)
    return force


def zero_sinkage_like(wheel_vz: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(wheel_vz)
