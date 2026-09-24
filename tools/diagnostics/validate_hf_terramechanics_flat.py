from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models.data_utils_v3 import DEFAULT_TERRAIN_PARAMS, TERRAIN_PARAM_KEYS, WHEEL_IDS  # noqa: E402
from models.differentiable_terramechanics import TerramechanicsParams, wheel_terrain_force  # noqa: E402


DEFAULT_MERGED_CSV = ROOT / "Feature_Selection" / "DataSet" / "merged_error_dataset.csv"
DEFAULT_OUTPUT_DIR = ROOT / "results_v3" / "diagnostics" / "hf_terramechanics_flat"
AXES = ("Fx", "Fy", "Fz")
BEKKER_PARAM_KEYS = ["Kc", "Kphi", "n", "c", "phi", "K"]
LOAD_TRANSFER_KEYS = ["roll", "pitch", "acc_x", "acc_y", "acc_z"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    print(f"[validate_hf_phy_flat] {message}", flush=True)


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    col_set = set(columns)
    for name in candidates:
        if name in col_set:
            return name
    return None


def required_columns(
    columns: Iterable[str],
    contact_filter: str,
    velocity_source: str,
    angular_source: str,
    force_frame: str,
) -> Dict[str, Dict[str, str]]:
    col_set = set(columns)
    mapping: Dict[str, Dict[str, str]] = {}
    body_vel = {}
    for axis in ("x", "y", "z"):
        col = first_existing(col_set, [f"hf_vel_{axis}", f"hf_body_vel_{axis}"])
        if col is None:
            raise KeyError(f"缺少 HF body velocity 列: hf_vel_{axis} 或 hf_body_vel_{axis}")
        body_vel[axis] = col
    mapping["body_vel"] = body_vel

    for wheel_id in WHEEL_IDS:
        wheel: Dict[str, str] = {}
        for axis in ("x", "y", "z"):
            col = first_existing(col_set, [f"hf_wheel{wheel_id}_pos_{axis}"])
            if col is None:
                raise KeyError(f"缺少列: hf_wheel{wheel_id}_pos_{axis}")
            wheel[f"pos_{axis}"] = col
        need_quat = velocity_source == "wheel_local" or angular_source == "local_y" or force_frame == "world_from_wheel"
        if need_quat:
            for comp in ("q0", "q1", "q2", "q3"):
                col = first_existing(col_set, [f"hf_wheel{wheel_id}_{comp}"])
                if col is None:
                    raise KeyError(f"需要列: hf_wheel{wheel_id}_{comp}")
                wheel[comp] = col
        if angular_source == "column":
            omega_col = first_existing(
                col_set,
                [
                    f"hf_wheel{wheel_id}_omega",
                    f"hf_wheel{wheel_id}_ang_vel_y",
                    f"hf_wheel{wheel_id}_ang_vel_z",
                    f"hf_wheel{wheel_id}_ang_vel_x",
                    f"lf_wheel{wheel_id}_omega",
                    f"lf_wheel{wheel_id}_ang_vel_y",
                    f"lf_wheel{wheel_id}_ang_vel_z",
                ],
            )
            if omega_col is None:
                raise KeyError(f"缺少 wheel{wheel_id} omega/ang_vel 列")
            wheel["omega"] = omega_col
        else:
            for axis in ("x", "y", "z"):
                col = first_existing(col_set, [f"hf_wheel{wheel_id}_ang_vel_{axis}"])
                if col is None:
                    raise KeyError(f"--angular_source local_y 需要列: hf_wheel{wheel_id}_ang_vel_{axis}")
                wheel[f"ang_vel_{axis}"] = col
        if velocity_source in {"wheel", "wheel_local"}:
            for axis in ("x", "y", "z"):
                col = first_existing(col_set, [f"hf_wheel{wheel_id}_vel_{axis}"])
                if col is None:
                    raise KeyError(f"--velocity_source {velocity_source} 需要列: hf_wheel{wheel_id}_vel_{axis}")
                wheel[f"vel_{axis}"] = col
        for axis in AXES:
            col = first_existing(col_set, [f"hf_wheel{wheel_id}_{axis}"])
            if col is None:
                raise KeyError(f"缺少列: hf_wheel{wheel_id}_{axis}")
            wheel[axis] = col
        if contact_filter == "hf":
            contact_col = first_existing(col_set, [f"hf_wheel{wheel_id}_in_contact"])
            if contact_col is None:
                raise KeyError(f"--contact_filter hf 需要列: hf_wheel{wheel_id}_in_contact")
            wheel["in_contact"] = contact_col
        mapping[f"wheel{wheel_id}"] = wheel
    return mapping


def optional_aux_columns(columns: Iterable[str]) -> Dict[str, str]:
    col_set = set(columns)
    candidates = {
        "roll": ["hf_roll", "lf_roll"],
        "pitch": ["hf_pitch", "lf_pitch"],
        "acc_x": ["hf_acc_x", "lf_acc_x"],
        "acc_y": ["hf_acc_y", "lf_acc_y"],
        "acc_z": ["hf_acc_z", "lf_acc_z"],
    }
    out: Dict[str, str] = {}
    for key, names in candidates.items():
        col = first_existing(col_set, names)
        if col is not None:
            out[key] = col
    return out


def numeric_aux_frame(df: pd.DataFrame, aux_cols: Dict[str, str]) -> np.ndarray:
    out = np.zeros((len(df), len(LOAD_TRANSFER_KEYS)), dtype=np.float32)
    for idx, key in enumerate(LOAD_TRANSFER_KEYS):
        col = aux_cols.get(key)
        if col is not None and col in df.columns:
            out[:, idx] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32)
    return out


def build_case_split(
    csv_path: Path,
    chunksize: int,
    train_ratio: float,
    seed: int,
) -> Dict[str, object]:
    if train_ratio >= 1.0:
        return {"enabled": False, "train": None, "val": None, "train_count": 0, "val_count": 0}
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("--case_split_ratio 必须在 (0,1] 范围内")
    cases: set[str] = set()
    for chunk in pd.read_csv(csv_path, usecols=["case_name"], chunksize=chunksize):
        cases.update(chunk["case_name"].dropna().astype(str).unique().tolist())
    case_list = np.asarray(sorted(cases), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(case_list)
    n_train = max(1, min(len(case_list) - 1, int(round(len(case_list) * train_ratio))))
    train = set(str(v) for v in case_list[:n_train])
    val = set(str(v) for v in case_list[n_train:])
    return {
        "enabled": True,
        "train": train,
        "val": val,
        "train_count": len(train),
        "val_count": len(val),
        "train_ratio": train_ratio,
        "seed": seed,
    }


def cases_for_split(case_split: Dict[str, object], split: str) -> Optional[set[str]]:
    if not case_split.get("enabled"):
        return None
    if split == "all":
        return None
    cases = case_split.get(split)
    if cases is None:
        raise ValueError(f"未知 case split: {split}")
    return cases  # type: ignore[return-value]


def filter_chunk_cases(chunk: pd.DataFrame, case_filter: Optional[set[str]]) -> pd.DataFrame:
    if case_filter is None:
        return chunk
    if "case_name" not in chunk.columns:
        raise KeyError("case-level split 需要 case_name 列")
    mask = chunk["case_name"].astype(str).isin(case_filter)
    return chunk.loc[mask].copy()


def numeric_frame(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    return df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)


def velocity_cols_for_wheel(mapping: Dict[str, Dict[str, str]], wheel_id: int, velocity_source: str) -> List[str]:
    if velocity_source == "body":
        return [mapping["body_vel"][axis] for axis in ("x", "y", "z")]
    wheel_map = mapping[f"wheel{wheel_id}"]
    return [wheel_map[f"vel_{axis}"] for axis in ("x", "y", "z")]


def quat_cols_for_wheel(mapping: Dict[str, Dict[str, str]], wheel_id: int) -> List[str]:
    wheel_map = mapping[f"wheel{wheel_id}"]
    return [wheel_map[comp] for comp in ("q0", "q1", "q2", "q3")]


def ang_vel_cols_for_wheel(mapping: Dict[str, Dict[str, str]], wheel_id: int) -> List[str]:
    wheel_map = mapping[f"wheel{wheel_id}"]
    return [wheel_map[f"ang_vel_{axis}"] for axis in ("x", "y", "z")]


def quat_rotate_back_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = q.astype(np.float64, copy=False)
    v = v.astype(np.float64, copy=False)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.maximum(norm, 1e-12)
    w = q[:, 0:1]
    xyz = q[:, 1:4]
    t = 2.0 * np.cross(xyz, v)
    return (v - w * t + np.cross(xyz, t)).astype(np.float32)


def quat_rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = q.astype(np.float64, copy=False)
    v = v.astype(np.float64, copy=False)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.maximum(norm, 1e-12)
    w = q[:, 0:1]
    xyz = q[:, 1:4]
    t = 2.0 * np.cross(xyz, v)
    return (v + w * t + np.cross(xyz, t)).astype(np.float32)


def quat_rotate_torch(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w = q[..., 0:1]
    xyz = q[..., 1:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def build_velocity_and_omega(
    chunk: pd.DataFrame,
    mapping: Dict[str, Dict[str, str]],
    wheel_id: int,
    velocity_source: str,
    angular_source: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    velocity = numeric_frame(chunk, velocity_cols_for_wheel(mapping, wheel_id, velocity_source))
    quat = None
    if velocity_source == "wheel_local" or angular_source == "local_y":
        quat = numeric_frame(chunk, quat_cols_for_wheel(mapping, wheel_id))
    if velocity_source == "wheel_local":
        velocity = quat_rotate_back_np(quat, velocity)
    if angular_source == "local_y":
        ang_vel_world = numeric_frame(chunk, ang_vel_cols_for_wheel(mapping, wheel_id))
        ang_vel_local = quat_rotate_back_np(quat, ang_vel_world)
        omega = ang_vel_local[:, 1].astype(np.float32, copy=False)
    else:
        omega = pd.to_numeric(chunk[mapping[f"wheel{wheel_id}"]["omega"]], errors="coerce").to_numpy(dtype=np.float32)
    return velocity, omega, quat


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.sum_true = np.zeros(3, dtype=np.float64)
        self.sum_pred = np.zeros(3, dtype=np.float64)
        self.sum_zero_abs = np.zeros(3, dtype=np.float64)
        self.sum_abs = np.zeros(3, dtype=np.float64)
        self.sum_sq = np.zeros(3, dtype=np.float64)
        self.sum_true_sq = np.zeros(3, dtype=np.float64)
        self.sum_pred_sq = np.zeros(3, dtype=np.float64)
        self.sum_true_pred = np.zeros(3, dtype=np.float64)
        self.sum_vec_dot = 0.0
        self.sum_vec_true_norm = 0.0
        self.sum_vec_pred_norm = 0.0
        self.sum_vec_err_sq = 0.0
        self.sum_zero_vec_err_sq = 0.0
        self.contact_count = 0
        self.sum_sinkage = 0.0
        self.max_sinkage = 0.0

    def update(self, pred: np.ndarray, true: np.ndarray, sinkage: np.ndarray, contact: np.ndarray) -> None:
        finite = np.isfinite(pred).all(axis=1) & np.isfinite(true).all(axis=1) & np.isfinite(sinkage)
        if not np.any(finite):
            return
        pred = pred[finite].astype(np.float64, copy=False)
        true = true[finite].astype(np.float64, copy=False)
        sinkage = sinkage[finite].astype(np.float64, copy=False)
        contact = contact[finite]
        err = pred - true
        n = int(pred.shape[0])
        self.count += n
        self.sum_true += true.sum(axis=0)
        self.sum_pred += pred.sum(axis=0)
        self.sum_abs += np.abs(err).sum(axis=0)
        self.sum_sq += (err * err).sum(axis=0)
        self.sum_zero_abs += np.abs(true).sum(axis=0)
        self.sum_true_sq += (true * true).sum(axis=0)
        self.sum_pred_sq += (pred * pred).sum(axis=0)
        self.sum_true_pred += (true * pred).sum(axis=0)
        self.sum_vec_dot += float((true * pred).sum())
        self.sum_vec_true_norm += float(np.linalg.norm(true, axis=1).sum())
        self.sum_vec_pred_norm += float(np.linalg.norm(pred, axis=1).sum())
        self.sum_vec_err_sq += float((err * err).sum())
        self.sum_zero_vec_err_sq += float((true * true).sum())
        self.contact_count += int(contact.sum())
        self.sum_sinkage += float(sinkage.sum())
        self.max_sinkage = max(self.max_sinkage, float(sinkage.max(initial=0.0)))

    def as_dict(self) -> Dict[str, object]:
        if self.count == 0:
            return {"count": 0}
        count = float(self.count)
        mean_true = self.sum_true / count
        mean_pred = self.sum_pred / count
        var_true = np.maximum(self.sum_true_sq / count - mean_true * mean_true, 0.0)
        var_pred = np.maximum(self.sum_pred_sq / count - mean_pred * mean_pred, 0.0)
        cov = self.sum_true_pred / count - mean_true * mean_pred
        denom = np.sqrt(np.maximum(var_true * var_pred, 0.0))
        corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 1e-12)
        rmse = np.sqrt(self.sum_sq / count)
        mae = self.sum_abs / count
        zero_rmse = np.sqrt(self.sum_true_sq / count)
        out = {
            "count": self.count,
            "contact_ratio": self.contact_count / count,
            "sinkage_mean": self.sum_sinkage / count,
            "sinkage_max": self.max_sinkage,
            "force_vector_rmse": float(np.sqrt(self.sum_vec_err_sq / count)),
            "zero_force_vector_rmse": float(np.sqrt(self.sum_zero_vec_err_sq / count)),
            "force_vector_improve_ratio_vs_zero": float(
                np.sqrt(self.sum_vec_err_sq / max(self.sum_zero_vec_err_sq, 1e-24))
            ),
            "mean_true_norm": self.sum_vec_true_norm / count,
            "mean_pred_norm": self.sum_vec_pred_norm / count,
            "global_cosine": float(
                self.sum_vec_dot / max(float(np.sqrt(self.sum_true_sq.sum() * self.sum_pred_sq.sum())), 1e-24)
            ),
            "axis": {},
        }
        for idx, axis in enumerate(AXES):
            out["axis"][axis] = {
                "rmse": float(rmse[idx]),
                "mae": float(mae[idx]),
                "bias_pred_minus_hf": float(mean_pred[idx] - mean_true[idx]),
                "corr": float(corr[idx]),
                "pred_mean": float(mean_pred[idx]),
                "hf_mean": float(mean_true[idx]),
                "zero_force_rmse": float(zero_rmse[idx]),
                "improve_ratio_vs_zero": float(rmse[idx] / max(zero_rmse[idx], 1e-12)),
            }
        return out


class LinearForceMapStats:
    def __init__(self, scopes: Sequence[str]):
        self.scopes = list(scopes)
        self.data: Dict[str, Dict[Tuple[int, int], Dict[str, float]]] = {scope: {} for scope in self.scopes}

    def _keys_for(self, wheel_id: int, axis_idx: int) -> List[Tuple[str, Tuple[int, int]]]:
        keys: List[Tuple[str, Tuple[int, int]]] = []
        if "overall" in self.data:
            keys.append(("overall", (-1, -1)))
        if "axis" in self.data:
            keys.append(("axis", (-1, axis_idx)))
        if "wheel" in self.data:
            keys.append(("wheel", (wheel_id, -1)))
        if "wheel_axis" in self.data:
            keys.append(("wheel_axis", (wheel_id, axis_idx)))
        return keys

    @staticmethod
    def _new_record() -> Dict[str, float]:
        return {
            "n": 0.0,
            "sum_x": 0.0,
            "sum_y": 0.0,
            "sum_x2": 0.0,
            "sum_y2": 0.0,
            "sum_xy": 0.0,
        }

    def update(self, wheel_id: int, pred: np.ndarray, true: np.ndarray) -> None:
        finite = np.isfinite(pred) & np.isfinite(true)
        for axis_idx in range(3):
            mask = finite[:, axis_idx]
            if not np.any(mask):
                continue
            x = pred[mask, axis_idx].astype(np.float64, copy=False)
            y = true[mask, axis_idx].astype(np.float64, copy=False)
            vals = {
                "n": float(x.size),
                "sum_x": float(x.sum()),
                "sum_y": float(y.sum()),
                "sum_x2": float((x * x).sum()),
                "sum_y2": float((y * y).sum()),
                "sum_xy": float((x * y).sum()),
            }
            for scope, key in self._keys_for(wheel_id, axis_idx):
                rec = self.data[scope].setdefault(key, self._new_record())
                for name, value in vals.items():
                    rec[name] += value

    @staticmethod
    def _fit_record(rec: Dict[str, float]) -> Dict[str, float]:
        n = rec["n"]
        if n <= 0:
            return {"count": 0}
        sx, sy, sx2, sy2, sxy = rec["sum_x"], rec["sum_y"], rec["sum_x2"], rec["sum_y2"], rec["sum_xy"]
        scale = sxy / max(sx2, 1e-24)
        sse_scale = sy2 - 2.0 * scale * sxy + scale * scale * sx2
        x_mean = sx / n
        y_mean = sy / n
        sxx = sx2 - sx * sx / n
        syy = sy2 - sy * sy / n
        sxy_centered = sxy - sx * sy / n
        affine_scale = sxy_centered / max(sxx, 1e-24)
        affine_bias = y_mean - affine_scale * x_mean
        sse_affine = (
            sy2
            - 2.0 * affine_scale * sxy
            - 2.0 * affine_bias * sy
            + affine_scale * affine_scale * sx2
            + 2.0 * affine_scale * affine_bias * sx
            + affine_bias * affine_bias * n
        )
        zero_rmse = np.sqrt(max(sy2 / n, 0.0))
        raw_rmse = np.sqrt(max((sx2 - 2.0 * sxy + sy2) / n, 0.0))
        scale_rmse = np.sqrt(max(sse_scale / n, 0.0))
        affine_rmse = np.sqrt(max(sse_affine / n, 0.0))
        corr = sxy_centered / np.sqrt(max(sxx * syy, 1e-24))
        r2_scale = 1.0 - sse_scale / max(syy, 1e-24)
        r2_affine = 1.0 - sse_affine / max(syy, 1e-24)
        return {
            "count": int(n),
            "raw_rmse": float(raw_rmse),
            "zero_rmse": float(zero_rmse),
            "scale": float(scale),
            "scale_rmse": float(scale_rmse),
            "scale_improve_ratio_vs_zero": float(scale_rmse / max(zero_rmse, 1e-12)),
            "affine_scale": float(affine_scale),
            "affine_bias": float(affine_bias),
            "affine_rmse": float(affine_rmse),
            "affine_improve_ratio_vs_zero": float(affine_rmse / max(zero_rmse, 1e-12)),
            "corr": float(corr),
            "r2_scale": float(r2_scale),
            "r2_affine": float(r2_affine),
            "pred_mean": float(x_mean),
            "hf_mean": float(y_mean),
        }

    def as_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = {}
        axis_names = {-1: "all", 0: "Fx", 1: "Fy", 2: "Fz"}
        for scope, records in self.data.items():
            items = []
            for (wheel_id, axis_idx), rec in sorted(records.items()):
                fitted = self._fit_record(rec)
                fitted["wheel_id"] = None if wheel_id < 0 else wheel_id
                fitted["axis"] = axis_names[axis_idx]
                items.append(fitted)
            out[scope] = items
        return out


TANGENT_TRANSFORMS = {
    "identity": ((0, 1), (1.0, 1.0)),
    "neg_fx": ((0, 1), (-1.0, 1.0)),
    "neg_fy": ((0, 1), (1.0, -1.0)),
    "neg_fx_fy": ((0, 1), (-1.0, -1.0)),
    "swap_xy": ((1, 0), (1.0, 1.0)),
    "swap_neg_x": ((1, 0), (-1.0, 1.0)),
    "swap_neg_y": ((1, 0), (1.0, -1.0)),
    "swap_neg_xy": ((1, 0), (-1.0, -1.0)),
}


def apply_tangent_transform(force: np.ndarray, name: str) -> np.ndarray:
    order, signs = TANGENT_TRANSFORMS[name]
    out = force.copy()
    out[:, 0] = signs[0] * force[:, order[0]]
    out[:, 1] = signs[1] * force[:, order[1]]
    return out


def summarize_transform_stats(transform_stats: Dict[str, RunningStats]) -> Dict[str, object]:
    rows = []
    for name, stats in transform_stats.items():
        data = stats.as_dict()
        if data.get("count", 0) == 0:
            continue
        axis = data["axis"]
        tangent_rmse = float(np.sqrt((axis["Fx"]["rmse"] ** 2 + axis["Fy"]["rmse"] ** 2) / 2.0))
        tangent_zero = float(np.sqrt((axis["Fx"]["zero_force_rmse"] ** 2 + axis["Fy"]["zero_force_rmse"] ** 2) / 2.0))
        tangent_corr = float((axis["Fx"]["corr"] + axis["Fy"]["corr"]) / 2.0)
        rows.append(
            {
                "name": name,
                "tangent_rmse": tangent_rmse,
                "tangent_zero_rmse": tangent_zero,
                "tangent_improve_ratio_vs_zero": tangent_rmse / max(tangent_zero, 1e-12),
                "tangent_corr_mean": tangent_corr,
                "Fx_rmse": axis["Fx"]["rmse"],
                "Fy_rmse": axis["Fy"]["rmse"],
                "Fx_corr": axis["Fx"]["corr"],
                "Fy_corr": axis["Fy"]["corr"],
            }
        )
    rows.sort(key=lambda x: x["tangent_rmse"])
    return {"ranking": rows, "best": rows[0] if rows else None}


def flat_terrain_params(
    shape: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
    values: Optional[Sequence[float] | torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    n, t = shape
    point = torch.zeros(n, t, 3, device=device, dtype=dtype)
    normal = torch.zeros(n, t, 3, device=device, dtype=dtype)
    normal[..., 2] = 1.0
    out: Dict[str, torch.Tensor] = {
        "contact_point": point,
        "contact_normal": normal,
    }
    if values is None:
        values_t = torch.as_tensor(DEFAULT_TERRAIN_PARAMS, device=device, dtype=dtype)
    else:
        values_t = torch.as_tensor(values, device=device, dtype=dtype)
    for idx, key in enumerate(TERRAIN_PARAM_KEYS):
        out[key] = values_t[idx].expand(n, t)
    return out


def compute_phy_force(
    body_velocity: np.ndarray,
    wheel_pos: np.ndarray,
    omega: np.ndarray,
    in_contact: Optional[np.ndarray],
    sinkage_max: float,
    device: torch.device,
    terrain_values: Optional[Sequence[float]] = None,
    wheel_radius: float = 0.135,
    wheel_width: float = 0.16,
    force_frame: str = "formula",
    wheel_quat: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    body_t = torch.from_numpy(body_velocity).to(device=device, dtype=torch.float32).unsqueeze(1)
    pos_t = torch.from_numpy(wheel_pos).to(device=device, dtype=torch.float32).unsqueeze(1)
    omega_t = torch.from_numpy(omega).to(device=device, dtype=torch.float32).unsqueeze(1)
    zero = torch.zeros(pos_t.shape[:2], device=device, dtype=torch.float32)
    contact_t = None
    if in_contact is not None:
        contact_t = torch.from_numpy(in_contact).to(device=device, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        out = wheel_terrain_force(
            omega=omega_t,
            body_velocity=body_t,
            wheel_z_pred=pos_t[..., 2],
            wheel_z_lf=pos_t[..., 2],
            sinkage_lf=zero,
            wheel_pos_pred=pos_t,
            in_contact=contact_t,
            terrain_params=flat_terrain_params(pos_t.shape[:2], device=device, dtype=torch.float32, values=terrain_values),
            params=TerramechanicsParams(r=wheel_radius, b=wheel_width, sinkage_max=sinkage_max),
        )
    force = out["force"].squeeze(1).cpu().numpy()
    if force_frame == "world_from_wheel":
        if wheel_quat is None:
            raise ValueError("force_frame=world_from_wheel 需要 wheel_quat")
        force = quat_rotate_np(wheel_quat, force)
    sinkage = out["sinkage"].squeeze(1).cpu().numpy()
    contact = (sinkage > TerramechanicsParams().contact_threshold)
    if in_contact is not None:
        contact = contact & (in_contact > 0.5)
    return force, sinkage, contact


def bekker_classic_force_torch(
    body_velocity: torch.Tensor,
    wheel_pos: torch.Tensor,
    omega: torch.Tensor,
    wheel_id: torch.Tensor,
    params: torch.Tensor,
    wheel_radius: float,
    wheel_width: float,
    sinkage_max: float,
    z_offset: Optional[torch.Tensor] = None,
    static_fz: Optional[torch.Tensor] = None,
    empirical_xy: Optional[Dict[str, torch.Tensor]] = None,
    n_theta: int = 32,
) -> Dict[str, torch.Tensor]:
    vx = body_velocity[..., 0]
    vy = body_velocity[..., 1] if body_velocity.shape[-1] > 1 else torch.zeros_like(vx)
    vz = body_velocity[..., 2] if body_velocity.shape[-1] > 2 else torch.zeros_like(vx)
    wheel_z = wheel_pos[..., 2]
    r = torch.as_tensor(wheel_radius, dtype=wheel_z.dtype, device=wheel_z.device)
    b = torch.as_tensor(wheel_width, dtype=wheel_z.dtype, device=wheel_z.device)
    z0 = torch.zeros((), dtype=wheel_z.dtype, device=wheel_z.device) if z_offset is None else z_offset.to(wheel_z)
    sinkage = (r - wheel_z + z0).clamp(0.0, sinkage_max)
    effective_sinkage = sinkage.clamp_min(0.0)
    contact = effective_sinkage > TerramechanicsParams().contact_threshold

    kc, kphi, n, cohesion, phi, shear_k = [p.to(wheel_z) for p in params]
    kc = kc.clamp_min(0.0)
    kphi = kphi.clamp_min(0.0)
    n = n.clamp(0.2, 3.0)
    cohesion = cohesion.clamp_min(0.0)
    phi = phi.clamp(0.0, TerramechanicsParams().max_tan_angle)
    shear_k = shear_k.clamp_min(1e-6)

    theta0 = torch.acos(((r - effective_sinkage) / r).clamp(-1.0 + 1e-4, 1.0 - 1e-4))
    unit = torch.linspace(-1.0, 1.0, n_theta, dtype=wheel_z.dtype, device=wheel_z.device)
    theta = theta0.unsqueeze(-1) * unit
    dtheta = (2.0 * theta0 / max(n_theta - 1, 1)).unsqueeze(-1)
    depth = (r * (torch.cos(theta) - torch.cos(theta0).unsqueeze(-1))).clamp_min(0.0)
    pressure = (kc / b + kphi) * depth.clamp_min(1e-8).pow(n)
    pressure = torch.where(contact.unsqueeze(-1), pressure, torch.zeros_like(pressure))
    d_area = b * r * dtheta

    fz_bekker = (pressure * torch.cos(theta).clamp_min(0.0) * d_area).sum(dim=-1)

    v_circ = r * omega
    denom = torch.maximum(torch.maximum(torch.abs(vx), torch.abs(v_circ)), torch.full_like(vx, 1e-3))
    slip_long = ((v_circ - vx) / denom).clamp(-1.0, 1.0)
    slip_lat = torch.atan2(vy, torch.abs(vx) + 1e-3).clamp(-1.45, 1.45)

    progress = (theta0.unsqueeze(-1) - theta).clamp_min(0.0)
    jx = (torch.abs(slip_long).unsqueeze(-1) * r * progress).clamp_min(0.0)
    jy = (torch.abs(torch.tan(slip_lat)).unsqueeze(-1) * r * progress).clamp_min(0.0)
    shear_limit = cohesion + pressure * torch.tan(phi)
    tau_x = shear_limit * (1.0 - torch.exp((-jx / shear_k).clamp(-50.0, 50.0)))
    tau_y = shear_limit * (1.0 - torch.exp((-jy / shear_k).clamp(-50.0, 50.0)))
    fx = torch.sign(slip_long) * (tau_x * d_area).sum(dim=-1)
    fy = -torch.sign(vy) * (tau_y * d_area).sum(dim=-1)

    fz = fz_bekker
    if static_fz is not None:
        static = static_fz.to(wheel_z)[wheel_id.long()].view_as(fz)
        fz = fz + static
    if empirical_xy is not None:
        wheel = wheel_id.long()
        fx = (
            empirical_xy["fx_scale"].to(wheel_z) * fx
            + empirical_xy["fx_v"].to(wheel_z) * vx
            + empirical_xy["fx_slip"].to(wheel_z) * slip_long
            + empirical_xy["fx_bias"].to(wheel_z)[wheel].view_as(fx)
        )
        fy = (
            empirical_xy["fy_scale"].to(wheel_z) * fy
            + empirical_xy["fy_v"].to(wheel_z) * vy
            + empirical_xy["fy_slip"].to(wheel_z) * slip_lat
            + empirical_xy["fy_bias"].to(wheel_z)[wheel].view_as(fy)
        )
        fz = (
            empirical_xy["fz_scale"].to(wheel_z) * fz_bekker
            + fz
            + empirical_xy["fz_v_down"].to(wheel_z) * torch.relu(-vz)
        )

    force = torch.nan_to_num(torch.stack([fx, fy, fz.clamp_min(0.0)], dim=-1), nan=0.0, posinf=1e6, neginf=-1e6)
    return {"force": force, "sinkage": sinkage, "contact": contact}


def compute_bekker_force(
    body_velocity: np.ndarray,
    wheel_pos: np.ndarray,
    omega: np.ndarray,
    wheel_id: int,
    sinkage_max: float,
    device: torch.device,
    bekker_values: Sequence[float],
    wheel_radius: float,
    wheel_width: float,
    z_offset: float = 0.0,
    static_fz_values: Optional[Sequence[float]] = None,
    empirical_xy_values: Optional[Dict[str, object]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    body_t = torch.from_numpy(body_velocity).to(device=device, dtype=torch.float32)
    pos_t = torch.from_numpy(wheel_pos).to(device=device, dtype=torch.float32)
    omega_t = torch.from_numpy(omega).to(device=device, dtype=torch.float32)
    wheel_t = torch.full_like(omega_t, int(wheel_id), dtype=torch.long)
    static_t = None
    if static_fz_values is not None:
        static_t = torch.as_tensor(static_fz_values, device=device, dtype=torch.float32)
    empirical_t = None
    if empirical_xy_values is not None:
        empirical_t = {}
        for key, value in empirical_xy_values.items():
            empirical_t[key] = torch.as_tensor(value, device=device, dtype=torch.float32)
    with torch.no_grad():
        out = bekker_classic_force_torch(
            body_velocity=body_t,
            wheel_pos=pos_t,
            omega=omega_t,
            wheel_id=wheel_t,
            params=torch.as_tensor(bekker_values, device=device, dtype=torch.float32),
            wheel_radius=wheel_radius,
            wheel_width=wheel_width,
            sinkage_max=sinkage_max,
            z_offset=torch.tensor(float(z_offset), device=device, dtype=torch.float32),
            static_fz=static_t,
            empirical_xy=empirical_t,
        )
    return (
        out["force"].cpu().numpy(),
        out["sinkage"].cpu().numpy(),
        out["contact"].cpu().numpy(),
    )


class TerrainParamFitter(torch.nn.Module):
    def __init__(self, init_values: Sequence[float]):
        super().__init__()
        vals = np.asarray(init_values, dtype=np.float64)
        self.raw_kc = torch.nn.Parameter(torch.tensor(float(vals[0]) / 1.0e5, dtype=torch.float32))
        self.raw_kphi = torch.nn.Parameter(torch.tensor(np.log(max(vals[1], 1e-6)), dtype=torch.float32))
        self.raw_n0 = torch.nn.Parameter(torch.tensor(_logit((np.clip(vals[2], 0.2001, 2.9999) - 0.2) / 2.8), dtype=torch.float32))
        self.raw_n1 = torch.nn.Parameter(torch.tensor(_logit((np.clip(vals[3], 0.2001, 2.9999) - 0.2) / 2.8), dtype=torch.float32))
        self.raw_c = torch.nn.Parameter(torch.tensor(np.log(max(vals[4], 1e-6)), dtype=torch.float32))
        self.raw_phi = torch.nn.Parameter(torch.tensor(np.arctanh(np.clip(vals[5] / 1.45, -0.9999, 0.9999)), dtype=torch.float32))
        self.raw_k = torch.nn.Parameter(torch.tensor(np.log(max(vals[6], 1e-8)), dtype=torch.float32))

    def values(self) -> torch.Tensor:
        return torch.stack(
            [
                (1.0e5 * self.raw_kc).clamp(-1.0e7, 1.0e7),
                torch.exp(self.raw_kphi).clamp_max(1.0e7),
                0.2 + 2.8 * torch.sigmoid(self.raw_n0),
                0.2 + 2.8 * torch.sigmoid(self.raw_n1),
                torch.exp(self.raw_c).clamp_max(1.0e6),
                1.45 * torch.tanh(self.raw_phi),
                torch.exp(self.raw_k).clamp_min(1e-6).clamp_max(10.0),
            ]
        )


class BekkerClassicFitter(torch.nn.Module):
    def __init__(self, init_values: Optional[Sequence[float]] = None, fit_z_offset: bool = False,
                 fit_static_fz: bool = False, fit_empirical_xy: bool = False):
        super().__init__()
        vals = np.asarray(init_values or [2.0e4, 6.0e5, 1.0, 50.0, 0.5, 0.02], dtype=np.float64)
        self.raw_kc = torch.nn.Parameter(torch.tensor(np.log(max(vals[0], 1e-6)), dtype=torch.float32))
        self.raw_kphi = torch.nn.Parameter(torch.tensor(np.log(max(vals[1], 1e-6)), dtype=torch.float32))
        self.raw_n = torch.nn.Parameter(torch.tensor(_logit((np.clip(vals[2], 0.2001, 2.9999) - 0.2) / 2.8), dtype=torch.float32))
        self.raw_c = torch.nn.Parameter(torch.tensor(np.log(max(vals[3], 1e-6)), dtype=torch.float32))
        self.raw_phi = torch.nn.Parameter(torch.tensor(_logit(np.clip(vals[4] / 1.45, 1e-6, 1.0 - 1e-6)), dtype=torch.float32))
        self.raw_k = torch.nn.Parameter(torch.tensor(np.log(max(vals[5], 1e-6)), dtype=torch.float32))
        self.fit_z_offset = fit_z_offset
        self.fit_static_fz = fit_static_fz
        self.fit_empirical_xy = fit_empirical_xy
        self.raw_z_offset = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=fit_z_offset)
        self.static_fz = torch.nn.Parameter(torch.full((len(WHEEL_IDS),), 300.0, dtype=torch.float32), requires_grad=fit_static_fz)
        self.fx_scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=fit_empirical_xy)
        self.fy_scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=fit_empirical_xy)
        self.fz_scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=fit_empirical_xy)
        self.fx_v = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=fit_empirical_xy)
        self.fy_v = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=fit_empirical_xy)
        self.fz_v_down = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=fit_empirical_xy)
        self.fx_slip = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=fit_empirical_xy)
        self.fy_slip = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=fit_empirical_xy)
        self.fx_bias = torch.nn.Parameter(torch.zeros(len(WHEEL_IDS), dtype=torch.float32), requires_grad=fit_empirical_xy)
        self.fy_bias = torch.nn.Parameter(torch.zeros(len(WHEEL_IDS), dtype=torch.float32), requires_grad=fit_empirical_xy)

    def values(self) -> torch.Tensor:
        return torch.stack(
            [
                torch.exp(self.raw_kc).clamp_max(1.0e7),
                torch.exp(self.raw_kphi).clamp_max(1.0e7),
                0.2 + 2.8 * torch.sigmoid(self.raw_n),
                torch.exp(self.raw_c).clamp_max(1.0e6),
                1.45 * torch.sigmoid(self.raw_phi),
                torch.exp(self.raw_k).clamp_min(1e-6).clamp_max(10.0),
            ]
        )

    def z_offset(self) -> torch.Tensor:
        return 0.1 * torch.tanh(self.raw_z_offset)

    def static_values(self) -> Optional[torch.Tensor]:
        if not self.fit_static_fz:
            return None
        return self.static_fz.clamp(0.0, 2000.0)

    def empirical_values(self) -> Optional[Dict[str, torch.Tensor]]:
        if not self.fit_empirical_xy:
            return None
        return {
            "fx_scale": self.fx_scale,
            "fy_scale": self.fy_scale,
            "fz_scale": self.fz_scale,
            "fx_v": self.fx_v,
            "fy_v": self.fy_v,
            "fz_v_down": self.fz_v_down,
            "fx_slip": self.fx_slip,
            "fy_slip": self.fy_slip,
            "fx_bias": self.fx_bias,
            "fy_bias": self.fy_bias,
        }

    def export(self) -> Dict[str, object]:
        values = self.values().detach().cpu().numpy().astype(float).tolist()
        out: Dict[str, object] = {
            "param_order": BEKKER_PARAM_KEYS,
            "bekker": dict(zip(BEKKER_PARAM_KEYS, values)),
            "z_offset": float(self.z_offset().detach().cpu()),
            "static_fz": None,
            "empirical_xy": None,
        }
        static = self.static_values()
        if static is not None:
            out["static_fz"] = [float(v) for v in static.detach().cpu().tolist()]
        empirical = self.empirical_values()
        if empirical is not None:
            out["empirical_xy"] = {
                key: ([float(v) for v in val.detach().cpu().flatten().tolist()] if val.ndim > 0 else float(val.detach().cpu()))
                for key, val in empirical.items()
            }
        return out


class GatedBekkerLoadTransferFitter(torch.nn.Module):
    def __init__(
        self,
        init_mean: np.ndarray,
        use_bekker: bool = True,
        use_load_transfer: bool = True,
        use_xy_empirical: bool = True,
    ):
        super().__init__()
        self.use_bekker = bool(use_bekker)
        self.use_load_transfer = bool(use_load_transfer)
        self.use_xy_empirical = bool(use_xy_empirical)
        init = torch.as_tensor(init_mean, dtype=torch.float32)
        self.force_bias = torch.nn.Parameter(init.clone())
        self.raw_alpha = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32), requires_grad=self.use_bekker)
        self.fz_load = torch.nn.Parameter(
            torch.zeros(len(WHEEL_IDS), len(LOAD_TRANSFER_KEYS), dtype=torch.float32),
            requires_grad=self.use_load_transfer,
        )
        self.fz_down_v = torch.nn.Parameter(torch.zeros(len(WHEEL_IDS), dtype=torch.float32), requires_grad=self.use_load_transfer)
        self.fx_vx = torch.nn.Parameter(torch.zeros(len(WHEEL_IDS), dtype=torch.float32), requires_grad=self.use_xy_empirical)
        self.fx_omega = torch.nn.Parameter(torch.zeros(len(WHEEL_IDS), dtype=torch.float32), requires_grad=self.use_xy_empirical)
        self.fy_vy = torch.nn.Parameter(torch.zeros(len(WHEEL_IDS), dtype=torch.float32), requires_grad=self.use_xy_empirical)
        self.fy_omega = torch.nn.Parameter(torch.zeros(len(WHEEL_IDS), dtype=torch.float32), requires_grad=self.use_xy_empirical)

    def alpha(self) -> torch.Tensor:
        return 0.2 * torch.tanh(self.raw_alpha)

    def forward(self, bekker_force: torch.Tensor, velocity: torch.Tensor, omega: torch.Tensor,
                wheel_id: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        wheel = wheel_id.long()
        base = self.force_bias[wheel]
        pred = base
        if self.use_bekker:
            pred = pred + self.alpha().view(1, 3) * bekker_force
        vx = velocity[:, 0]
        vy = velocity[:, 1] if velocity.shape[-1] > 1 else torch.zeros_like(vx)
        vz = velocity[:, 2] if velocity.shape[-1] > 2 else torch.zeros_like(vx)
        if self.use_xy_empirical:
            pred[:, 0] = pred[:, 0] + self.fx_vx[wheel] * vx + self.fx_omega[wheel] * omega
            pred[:, 1] = pred[:, 1] + self.fy_vy[wheel] * vy + self.fy_omega[wheel] * omega
        if self.use_load_transfer:
            pred[:, 2] = (
                pred[:, 2]
                + (self.fz_load[wheel] * aux).sum(dim=-1)
                + self.fz_down_v[wheel] * torch.relu(-vz)
            )
        return pred

    def export(self) -> Dict[str, object]:
        return {
            "use_bekker": self.use_bekker,
            "use_load_transfer": self.use_load_transfer,
            "use_xy_empirical": self.use_xy_empirical,
            "force_bias": self.force_bias.detach().cpu().numpy().astype(float).tolist(),
            "alpha_bekker": self.alpha().detach().cpu().numpy().astype(float).tolist(),
            "load_transfer_keys": LOAD_TRANSFER_KEYS,
            "fz_load": self.fz_load.detach().cpu().numpy().astype(float).tolist(),
            "fz_down_v": self.fz_down_v.detach().cpu().numpy().astype(float).tolist(),
            "fx_vx": self.fx_vx.detach().cpu().numpy().astype(float).tolist(),
            "fx_omega": self.fx_omega.detach().cpu().numpy().astype(float).tolist(),
            "fy_vy": self.fy_vy.detach().cpu().numpy().astype(float).tolist(),
            "fy_omega": self.fy_omega.detach().cpu().numpy().astype(float).tolist(),
        }


def fit_wheel_mean_force(arrays: Dict[str, np.ndarray]) -> Dict[str, object]:
    means = np.zeros((len(WHEEL_IDS), 3), dtype=np.float64)
    counts = np.zeros(len(WHEEL_IDS), dtype=np.int64)
    for idx, wheel_id in enumerate(WHEEL_IDS):
        mask = arrays["wheel_id"] == wheel_id
        counts[idx] = int(mask.sum())
        if counts[idx] > 0:
            means[idx] = arrays["true_force"][mask].mean(axis=0)
    return {"force_bias": means.astype(float).tolist(), "counts": counts.astype(int).tolist()}


def wheel_mean_predict(wheel_id: int, n: int, export: Dict[str, object]) -> np.ndarray:
    bias = np.asarray(export["force_bias"], dtype=np.float32)
    return np.broadcast_to(bias[wheel_id], (n, 3)).copy()


def gated_export_to_tensors(export: Dict[str, object], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(value, device=device, dtype=torch.float32)
        for key, value in export.items()
        if key
        in {
            "force_bias",
            "alpha_bekker",
            "fz_load",
            "fz_down_v",
            "fx_vx",
            "fx_omega",
            "fy_vy",
            "fy_omega",
        }
    }


def gated_predict_torch(
    bekker_force: torch.Tensor,
    velocity: torch.Tensor,
    omega: torch.Tensor,
    wheel_id: torch.Tensor,
    aux: torch.Tensor,
    params: Dict[str, torch.Tensor],
    use_bekker: bool = True,
    use_load_transfer: bool = True,
    use_xy_empirical: bool = True,
) -> torch.Tensor:
    wheel = wheel_id.long()
    pred = params["force_bias"][wheel]
    if use_bekker:
        pred = pred + params["alpha_bekker"].view(1, 3) * bekker_force
    vx = velocity[:, 0]
    vy = velocity[:, 1] if velocity.shape[-1] > 1 else torch.zeros_like(vx)
    vz = velocity[:, 2] if velocity.shape[-1] > 2 else torch.zeros_like(vx)
    if use_xy_empirical:
        pred[:, 0] = pred[:, 0] + params["fx_vx"][wheel] * vx + params["fx_omega"][wheel] * omega
        pred[:, 1] = pred[:, 1] + params["fy_vy"][wheel] * vy + params["fy_omega"][wheel] * omega
    if use_load_transfer:
        pred[:, 2] = pred[:, 2] + (params["fz_load"][wheel] * aux).sum(dim=-1) + params["fz_down_v"][wheel] * torch.relu(-vz)
    return pred


def _logit(x: float) -> float:
    x = float(np.clip(x, 1e-6, 1.0 - 1e-6))
    return float(np.log(x / (1.0 - x)))


def parse_axis_weights(text: str) -> torch.Tensor:
    parts = [float(v.strip()) for v in text.split(",") if v.strip()]
    if len(parts) != 3:
        raise ValueError("--fit_axis_weights 必须是 3 个逗号分隔数字，例如 1,1,1")
    return torch.tensor(parts, dtype=torch.float32)


def collect_fit_arrays(
    csv_path: Path,
    usecols: List[str],
    mapping: Dict[str, Dict[str, str]],
    max_rows: int,
    chunksize: int,
    contact_filter: str,
    velocity_source: str,
    angular_source: str,
    force_frame: str,
    seed: int,
    aux_cols: Optional[Dict[str, str]] = None,
    case_filter: Optional[set[str]] = None,
) -> Dict[str, np.ndarray]:
    rows: List[pd.DataFrame] = []
    seen = 0
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        seen += len(chunk)
        chunk = filter_chunk_cases(chunk, case_filter)
        if chunk.empty:
            log(f"fit collection scanned rows={seen}")
            continue
        rows.append(chunk)
        if max_rows > 0 and sum(len(x) for x in rows) >= max_rows:
            break
        log(f"fit collection scanned rows={seen}")
    if not rows:
        raise ValueError("没有可用于参数辨识的数据")
    df = pd.concat(rows, ignore_index=True)
    if max_rows > 0 and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    velocity_all: List[np.ndarray] = []
    pos_all: List[np.ndarray] = []
    omega_all: List[np.ndarray] = []
    true_all: List[np.ndarray] = []
    contact_all: List[np.ndarray] = []
    wheel_all: List[np.ndarray] = []
    quat_all: List[np.ndarray] = []
    aux_all: List[np.ndarray] = []
    aux_frame = numeric_aux_frame(df, aux_cols or {})
    for wheel_id in WHEEL_IDS:
        wheel_map = mapping[f"wheel{wheel_id}"]
        pos_cols = [wheel_map[f"pos_{axis}"] for axis in ("x", "y", "z")]
        force_cols = [wheel_map[axis] for axis in AXES]
        velocity, omega, quat = build_velocity_and_omega(df, mapping, wheel_id, velocity_source, angular_source)
        wheel_pos = numeric_frame(df, pos_cols)
        true_force = numeric_frame(df, force_cols)
        if contact_filter == "hf":
            in_contact = pd.to_numeric(df[wheel_map["in_contact"]], errors="coerce").to_numpy(dtype=np.float32)
        else:
            in_contact = np.ones(len(df), dtype=np.float32)
        finite = (
            np.isfinite(velocity).all(axis=1)
            & np.isfinite(wheel_pos).all(axis=1)
            & np.isfinite(omega)
            & np.isfinite(true_force).all(axis=1)
            & np.isfinite(in_contact)
        )
        if force_frame == "world_from_wheel":
            if quat is None:
                quat = numeric_frame(df, quat_cols_for_wheel(mapping, wheel_id))
            finite &= np.isfinite(quat).all(axis=1)
        velocity_all.append(velocity[finite])
        pos_all.append(wheel_pos[finite])
        omega_all.append(omega[finite])
        true_all.append(true_force[finite])
        contact_all.append(in_contact[finite])
        wheel_all.append(np.full(int(finite.sum()), wheel_id, dtype=np.int64))
        aux_all.append(aux_frame[finite])
        if force_frame == "world_from_wheel":
            quat_all.append(quat[finite])
    return {
        "body_velocity": np.concatenate(velocity_all, axis=0),
        "wheel_pos": np.concatenate(pos_all, axis=0),
        "omega": np.concatenate(omega_all, axis=0),
        "true_force": np.concatenate(true_all, axis=0),
        "in_contact": np.concatenate(contact_all, axis=0),
        "wheel_id": np.concatenate(wheel_all, axis=0),
        "aux": np.concatenate(aux_all, axis=0),
        "wheel_quat": np.concatenate(quat_all, axis=0) if quat_all else np.zeros((0, 4), dtype=np.float32),
    }


def force_with_params(
    body_velocity: torch.Tensor,
    wheel_pos: torch.Tensor,
    omega: torch.Tensor,
    in_contact: Optional[torch.Tensor],
    terrain_values: torch.Tensor,
    sinkage_max: float,
    wheel_radius: float,
    wheel_width: float,
    force_frame: str,
    wheel_quat: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    zero = torch.zeros(wheel_pos.shape[:2], device=wheel_pos.device, dtype=wheel_pos.dtype)
    out = wheel_terrain_force(
        omega=omega,
        body_velocity=body_velocity,
        wheel_z_pred=wheel_pos[..., 2],
        wheel_z_lf=wheel_pos[..., 2],
        sinkage_lf=zero,
        wheel_pos_pred=wheel_pos,
        in_contact=in_contact,
        terrain_params=flat_terrain_params(wheel_pos.shape[:2], wheel_pos.device, wheel_pos.dtype, terrain_values),
        params=TerramechanicsParams(r=wheel_radius, b=wheel_width, sinkage_max=sinkage_max),
    )
    if force_frame == "world_from_wheel":
        if wheel_quat is None:
            raise ValueError("force_frame=world_from_wheel 需要 wheel_quat")
        out = dict(out)
        out["force"] = quat_rotate_torch(wheel_quat, out["force"])
    return out


def fit_terrain_params(
    arrays: Dict[str, np.ndarray],
    output_dir: Path,
    steps: int,
    batch_size: int,
    lr: float,
    sinkage_max: float,
    device: torch.device,
    axis_weights: torch.Tensor,
    loss_name: str,
    seed: int,
    wheel_radius: float,
    wheel_width: float,
    force_frame: str,
) -> List[float]:
    n = arrays["true_force"].shape[0]
    if n == 0:
        raise ValueError("没有有限的 force 样本可用于参数辨识")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    fitter = TerrainParamFitter(DEFAULT_TERRAIN_PARAMS).to(device)
    opt = torch.optim.Adam(fitter.parameters(), lr=lr)
    body = torch.from_numpy(arrays["body_velocity"]).to(device=device, dtype=torch.float32).unsqueeze(1)
    pos = torch.from_numpy(arrays["wheel_pos"]).to(device=device, dtype=torch.float32).unsqueeze(1)
    omega = torch.from_numpy(arrays["omega"]).to(device=device, dtype=torch.float32).unsqueeze(1)
    true = torch.from_numpy(arrays["true_force"]).to(device=device, dtype=torch.float32).unsqueeze(1)
    contact = torch.from_numpy(arrays["in_contact"]).to(device=device, dtype=torch.float32).unsqueeze(1)
    wheel_quat = None
    if force_frame == "world_from_wheel":
        wheel_quat = torch.from_numpy(arrays["wheel_quat"]).to(device=device, dtype=torch.float32).unsqueeze(1)
    axis_weights = axis_weights.to(device=device, dtype=torch.float32).view(1, 1, 3)
    force_scale = true.flatten(0, 1).std(dim=0).clamp_min(1.0).view(1, 1, 3)
    history: List[Dict[str, float]] = []

    for step in range(1, steps + 1):
        if batch_size > 0 and batch_size < n:
            idx = torch.randperm(n, generator=generator)[:batch_size].to(device)
        else:
            idx = torch.arange(n, device=device)
        out = force_with_params(
            body_velocity=body[idx],
            wheel_pos=pos[idx],
            omega=omega[idx],
            in_contact=contact[idx],
            terrain_values=fitter.values(),
            sinkage_max=sinkage_max,
            wheel_radius=wheel_radius,
            wheel_width=wheel_width,
            force_frame=force_frame,
            wheel_quat=wheel_quat[idx] if wheel_quat is not None else None,
        )
        err = (out["force"] - true[idx]) / force_scale
        if loss_name == "mse":
            loss = ((err * err) * axis_weights).mean()
        else:
            loss = (torch.nn.functional.smooth_l1_loss(err, torch.zeros_like(err), reduction="none") * axis_weights).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 1 or step == steps or step % max(1, steps // 20) == 0:
            values = fitter.values().detach().cpu().numpy()
            record = {"step": step, "loss": float(loss.detach().cpu())}
            record.update({key: float(values[i]) for i, key in enumerate(TERRAIN_PARAM_KEYS)})
            history.append(record)
            log("fit " + " ".join([f"{k}={v:.6g}" for k, v in record.items() if isinstance(v, float)]))

    values = fitter.values().detach().cpu().numpy().astype(float).tolist()
    with open(output_dir / "fit_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with open(output_dir / "fitted_terrain_params.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "terrain": "flat plane",
                "param_order": TERRAIN_PARAM_KEYS,
                "initial": dict(zip(TERRAIN_PARAM_KEYS, [float(v) for v in DEFAULT_TERRAIN_PARAMS])),
                "fitted": dict(zip(TERRAIN_PARAM_KEYS, [float(v) for v in values])),
                "fit_config": {
                    "steps": steps,
                    "batch_size": batch_size,
                    "lr": lr,
                    "loss": loss_name,
                    "axis_weights": [float(v) for v in axis_weights.detach().cpu().flatten().tolist()],
                    "sample_count": int(n),
                    "wheel_radius": wheel_radius,
                    "wheel_width": wheel_width,
                    "force_frame": force_frame,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return values


def fit_bekker_classic(
    arrays: Dict[str, np.ndarray],
    output_dir: Path,
    steps: int,
    batch_size: int,
    lr: float,
    sinkage_max: float,
    device: torch.device,
    axis_weights: torch.Tensor,
    loss_name: str,
    seed: int,
    wheel_radius: float,
    wheel_width: float,
    fit_z_offset: bool,
    fit_static_fz: bool,
    fit_empirical_xy: bool,
) -> Dict[str, object]:
    n = arrays["true_force"].shape[0]
    if n == 0:
        raise ValueError("没有有限的 force 样本可用于 Bekker 参数辨识")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    fitter = BekkerClassicFitter(
        fit_z_offset=fit_z_offset,
        fit_static_fz=fit_static_fz,
        fit_empirical_xy=fit_empirical_xy,
    ).to(device)
    opt = torch.optim.Adam([p for p in fitter.parameters() if p.requires_grad], lr=lr)
    body = torch.from_numpy(arrays["body_velocity"]).to(device=device, dtype=torch.float32)
    pos = torch.from_numpy(arrays["wheel_pos"]).to(device=device, dtype=torch.float32)
    omega = torch.from_numpy(arrays["omega"]).to(device=device, dtype=torch.float32)
    wheel_id = torch.from_numpy(arrays["wheel_id"]).to(device=device, dtype=torch.long)
    true = torch.from_numpy(arrays["true_force"]).to(device=device, dtype=torch.float32)
    axis_weights = axis_weights.to(device=device, dtype=torch.float32).view(1, 3)
    force_scale = true.std(dim=0).clamp_min(1.0).view(1, 3)
    history: List[Dict[str, object]] = []

    for step in range(1, steps + 1):
        if batch_size > 0 and batch_size < n:
            idx = torch.randperm(n, generator=generator)[:batch_size].to(device)
        else:
            idx = torch.arange(n, device=device)
        out = bekker_classic_force_torch(
            body_velocity=body[idx],
            wheel_pos=pos[idx],
            omega=omega[idx],
            wheel_id=wheel_id[idx],
            params=fitter.values(),
            wheel_radius=wheel_radius,
            wheel_width=wheel_width,
            sinkage_max=sinkage_max,
            z_offset=fitter.z_offset(),
            static_fz=fitter.static_values(),
            empirical_xy=fitter.empirical_values(),
        )
        err = (out["force"] - true[idx]) / force_scale
        if loss_name == "mse":
            loss = ((err * err) * axis_weights).mean()
        else:
            loss = (torch.nn.functional.smooth_l1_loss(err, torch.zeros_like(err), reduction="none") * axis_weights).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 1 or step == steps or step % max(1, steps // 20) == 0:
            export = fitter.export()
            record: Dict[str, object] = {"step": step, "loss": float(loss.detach().cpu())}
            record.update(export["bekker"])
            record["z_offset"] = export["z_offset"]
            if export["static_fz"] is not None:
                record["static_fz_mean"] = float(np.mean(export["static_fz"]))
            history.append(record)
            log("bekker_fit " + " ".join([f"{k}={v:.6g}" for k, v in record.items() if isinstance(v, float)]))

    export = fitter.export()
    export["fit_config"] = {
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "loss": loss_name,
        "axis_weights": [float(v) for v in axis_weights.detach().cpu().flatten().tolist()],
        "sample_count": int(n),
        "wheel_radius": wheel_radius,
        "wheel_width": wheel_width,
        "fit_z_offset": bool(fit_z_offset),
        "fit_static_fz": bool(fit_static_fz),
        "fit_empirical_xy": bool(fit_empirical_xy),
    }
    with open(output_dir / "bekker_fit_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with open(output_dir / "fitted_bekker_params.json", "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    return export


def bekker_export_to_eval_values(export: Dict[str, object]) -> Tuple[List[float], float, Optional[List[float]], Optional[Dict[str, object]]]:
    bekker = export["bekker"]
    values = [float(bekker[key]) for key in BEKKER_PARAM_KEYS]
    z_offset = float(export.get("z_offset", 0.0))
    static = export.get("static_fz")
    static_values = [float(v) for v in static] if static is not None else None
    empirical = export.get("empirical_xy")
    return values, z_offset, static_values, empirical if isinstance(empirical, dict) else None


def fit_gated_bekker_load_transfer(
    arrays: Dict[str, np.ndarray],
    output_dir: Path,
    steps: int,
    batch_size: int,
    lr: float,
    sinkage_max: float,
    device: torch.device,
    axis_weights: torch.Tensor,
    loss_name: str,
    seed: int,
    wheel_radius: float,
    wheel_width: float,
    use_bekker: bool,
    use_load_transfer: bool,
    use_xy_empirical: bool,
) -> Dict[str, object]:
    n = arrays["true_force"].shape[0]
    if n == 0:
        raise ValueError("没有有限的 force 样本可用于 gated Bekker/load-transfer 拟合")
    mean_export = fit_wheel_mean_force(arrays)
    fitter = GatedBekkerLoadTransferFitter(
        np.asarray(mean_export["force_bias"], dtype=np.float32),
        use_bekker=use_bekker,
        use_load_transfer=use_load_transfer,
        use_xy_empirical=use_xy_empirical,
    ).to(device)
    opt = torch.optim.Adam([p for p in fitter.parameters() if p.requires_grad], lr=lr)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    body = torch.from_numpy(arrays["body_velocity"]).to(device=device, dtype=torch.float32)
    pos = torch.from_numpy(arrays["wheel_pos"]).to(device=device, dtype=torch.float32)
    omega = torch.from_numpy(arrays["omega"]).to(device=device, dtype=torch.float32)
    wheel_id = torch.from_numpy(arrays["wheel_id"]).to(device=device, dtype=torch.long)
    aux = torch.from_numpy(arrays["aux"]).to(device=device, dtype=torch.float32)
    true = torch.from_numpy(arrays["true_force"]).to(device=device, dtype=torch.float32)
    axis_weights = axis_weights.to(device=device, dtype=torch.float32).view(1, 3)
    force_scale = true.std(dim=0).clamp_min(1.0).view(1, 3)
    bekker_params = torch.as_tensor([2.0e4, 6.0e5, 1.0, 50.0, 0.5, 0.02], device=device, dtype=torch.float32)
    history: List[Dict[str, object]] = []

    for step in range(1, steps + 1):
        if batch_size > 0 and batch_size < n:
            idx = torch.randperm(n, generator=generator)[:batch_size].to(device)
        else:
            idx = torch.arange(n, device=device)
        bekker = bekker_classic_force_torch(
            body_velocity=body[idx],
            wheel_pos=pos[idx],
            omega=omega[idx],
            wheel_id=wheel_id[idx],
            params=bekker_params,
            wheel_radius=wheel_radius,
            wheel_width=wheel_width,
            sinkage_max=sinkage_max,
        )
        pred = fitter(bekker["force"], body[idx], omega[idx], wheel_id[idx], aux[idx])
        err = (pred - true[idx]) / force_scale
        if loss_name == "mse":
            loss = ((err * err) * axis_weights).mean()
        else:
            loss = (torch.nn.functional.smooth_l1_loss(err, torch.zeros_like(err), reduction="none") * axis_weights).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 1 or step == steps or step % max(1, steps // 20) == 0:
            export = fitter.export()
            record: Dict[str, object] = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "alpha_bekker": export["alpha_bekker"],
                "force_bias_fz_mean": float(np.asarray(export["force_bias"])[:, 2].mean()),
            }
            history.append(record)
            log(
                "gated_fit "
                + f"step={step} loss={float(loss.detach().cpu()):.6g} "
                + "alpha="
                + ",".join(f"{v:.5g}" for v in export["alpha_bekker"])
            )

    export = fitter.export()
    export["mean_baseline"] = mean_export
    export["bekker_param_order"] = BEKKER_PARAM_KEYS
    export["bekker_params"] = dict(zip(BEKKER_PARAM_KEYS, [2.0e4, 6.0e5, 1.0, 50.0, 0.5, 0.02]))
    export["fit_config"] = {
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "loss": loss_name,
        "axis_weights": [float(v) for v in axis_weights.detach().cpu().flatten().tolist()],
        "sample_count": int(n),
        "wheel_radius": wheel_radius,
        "wheel_width": wheel_width,
        "use_bekker": bool(use_bekker),
        "use_load_transfer": bool(use_load_transfer),
        "use_xy_empirical": bool(use_xy_empirical),
    }
    with open(output_dir / "gated_fit_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with open(output_dir / "fitted_gated_bekker_load_transfer.json", "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    return export


def sample_rows(
    samples: List[pd.DataFrame],
    max_samples: int,
    seed: int,
) -> pd.DataFrame:
    if not samples:
        return pd.DataFrame()
    df = pd.concat(samples, ignore_index=True)
    if max_samples > 0 and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
    return df


def append_sample(
    samples: List[pd.DataFrame],
    chunk: pd.DataFrame,
    wheel_id: int,
    pred: np.ndarray,
    true: np.ndarray,
    sinkage: np.ndarray,
    contact: np.ndarray,
    max_samples: int,
    seed: int,
) -> None:
    if max_samples <= 0:
        return
    keep = min(len(chunk), max(1000, max_samples // 6))
    if keep < len(chunk):
        local = np.sort(np.random.default_rng(seed + wheel_id).choice(len(chunk), size=keep, replace=False))
        sub = chunk.iloc[local]
    else:
        sub = chunk
        local = np.arange(len(chunk))
    out = pd.DataFrame(
        {
            "case_name": sub["case_name"].astype(str).to_numpy() if "case_name" in sub.columns else "",
            "time": pd.to_numeric(sub["time"], errors="coerce").to_numpy() if "time" in sub.columns else np.nan,
            "wheel_id": wheel_id,
            "phy_Fx": pred[local, 0],
            "phy_Fy": pred[local, 1],
            "phy_Fz": pred[local, 2],
            "hf_Fx": true[local, 0],
            "hf_Fy": true[local, 1],
            "hf_Fz": true[local, 2],
            "sinkage_flat": sinkage[local],
            "contact_flat": contact[local].astype(np.float32),
        }
    )
    samples.append(out)


def plot_samples(samples: pd.DataFrame, output_dir: Path) -> None:
    if samples.empty:
        return
    for axis in AXES:
        fig = plt.figure(figsize=(6.4, 6.0))
        x = samples[f"hf_{axis}"].to_numpy(dtype=np.float64)
        y = samples[f"phy_{axis}"].to_numpy(dtype=np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if x.size == 0:
            plt.close(fig)
            continue
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        pad = max((hi - lo) * 0.05, 1.0)
        lo -= pad
        hi += pad
        plt.scatter(x, y, s=5, alpha=0.22, edgecolors="none")
        plt.plot([lo, hi], [lo, hi], color="black", linewidth=1.2)
        plt.axhline(0.0, color="gray", linewidth=0.8, alpha=0.5)
        plt.axvline(0.0, color="gray", linewidth=0.8, alpha=0.5)
        plt.xlabel(f"HF {axis}")
        plt.ylabel(f"Flat terramechanics {axis}")
        plt.title(f"{axis}: flat terramechanics vs HF")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / f"scatter_{axis}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate HF-state terramechanics force calculation on a flat terrain plane."
    )
    parser.add_argument("--csv", type=str, default=str(DEFAULT_MERGED_CSV))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--chunksize", type=int, default=200000)
    parser.add_argument("--max_rows", type=int, default=0, help="0 表示读取全部行")
    parser.add_argument("--max_sample_rows", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sinkage_max", type=float, default=0.08)
    parser.add_argument("--wheel_radius", type=float, default=0.135)
    parser.add_argument("--wheel_width", type=float, default=0.16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--velocity_source",
        choices=["body", "wheel", "wheel_local"],
        default="body",
        help="body 使用 hf_vel_x/y/z；wheel 使用 world hf_wheel*_vel_x/y/z；wheel_local 使用轮局部速度",
    )
    parser.add_argument(
        "--angular_source",
        choices=["column", "local_y"],
        default="column",
        help="column 从列中取 omega/ang_vel；local_y 将 hf_wheel*_ang_vel 世界系转到轮局部后取 Y 轴",
    )
    parser.add_argument(
        "--force_frame",
        choices=["formula", "world_from_wheel"],
        default="formula",
        help="formula 直接比较公式输出；world_from_wheel 将公式输出按轮四元数旋回世界系后比较 HF force",
    )
    parser.add_argument(
        "--sph_aligned",
        action="store_true",
        help="按 SPH HF 定义使用 r=0.12,b=0.18,wheel_local velocity,local_y omega,world force",
    )
    parser.add_argument(
        "--tangent_transform_search",
        action="store_true",
        help="额外评估 Fx/Fy 取反、交换等坐标/符号候选，不改变主指标",
    )
    parser.add_argument(
        "--fit_linear_force_map",
        action="store_true",
        help="辨识 HF_force ~= scale * phy_force 以及 scale * phy_force + bias 的线性映射",
    )
    parser.add_argument(
        "--force_model",
        choices=["current", "bekker_classic", "bekker_static_empirical", "wheel_mean", "gated_bekker_load_transfer"],
        default="current",
        help="current 使用现有 terramechanics；wheel_mean 为每轮均值 baseline；gated_bekker_load_transfer 为可门控 Bekker + 载荷转移",
    )
    parser.add_argument(
        "--linear_map_scopes",
        type=str,
        default="overall,axis,wheel,wheel_axis",
        help="逗号分隔: overall,axis,wheel,wheel_axis",
    )
    parser.add_argument(
        "--contact_filter",
        choices=["none", "hf"],
        default="none",
        help="none 只由平地 sinkage 判定接触；hf 会额外使用 hf_wheel*_in_contact 门控",
    )
    parser.add_argument("--fit_params", action="store_true", help="在验证脚本内部辨识一组平地 terramechanics 全局参数")
    parser.add_argument("--fit_mean_force", action="store_true", help="拟合每轮 HF 均值 force baseline")
    parser.add_argument("--fit_bekker_classic", action="store_true", help="在验证脚本内部辨识传统 Bekker/Janosi 参数")
    parser.add_argument("--fit_gated_bekker", action="store_true", help="拟合 gated Bekker + load-transfer 验证模型")
    parser.add_argument("--gated_use_bekker", type=int, choices=[0, 1], default=1, help="gated 模型是否启用 Bekker force 小门控项")
    parser.add_argument("--gated_use_load_transfer", type=int, choices=[0, 1], default=1, help="gated 模型是否启用 Fz roll/pitch/acc 载荷转移项")
    parser.add_argument("--gated_use_xy_empirical", type=int, choices=[0, 1], default=1, help="gated 模型是否启用 Fx/Fy velocity/omega 经验项")
    parser.add_argument("--fit_z_offset", action="store_true", help="Bekker 辨识时额外拟合地表 z 偏置，仅用于验证脚本")
    parser.add_argument("--fit_static_fz", action="store_true", help="Bekker 辨识时额外拟合每轮静态 Fz 基线")
    parser.add_argument("--fit_empirical_xy", action="store_true", help="Bekker 辨识时额外拟合 Fx/Fy 速度和滑移经验项")
    parser.add_argument("--fit_steps", type=int, default=800)
    parser.add_argument("--fit_lr", type=float, default=0.03)
    parser.add_argument("--fit_batch_size", type=int, default=8192)
    parser.add_argument("--fit_max_rows", type=int, default=50000, help="用于参数辨识的原始 CSV 行数采样上限")
    parser.add_argument("--fit_loss", choices=["huber", "mse"], default="huber")
    parser.add_argument("--fit_axis_weights", type=str, default="1,1,1", help="Fx,Fy,Fz 的拟合权重")
    parser.add_argument("--case_split_ratio", type=float, default=1.0, help="按 case_name 划分训练集比例；1.0 表示不划分")
    parser.add_argument("--fit_case_split", choices=["all", "train", "val"], default="train", help="参数辨识使用哪个 case split")
    parser.add_argument("--eval_case_split", choices=["all", "train", "val"], default="all", help="最终指标评估使用哪个 case split")
    args = parser.parse_args()
    if args.sph_aligned:
        args.wheel_radius = 0.12
        args.wheel_width = 0.18
        args.velocity_source = "wheel_local"
        args.angular_source = "local_y"
        args.force_frame = "formula"
    if args.force_model == "bekker_static_empirical":
        args.fit_static_fz = True
        args.fit_empirical_xy = True
    if args.fit_bekker_classic and args.force_model == "current":
        args.force_model = "bekker_static_empirical" if (args.fit_static_fz or args.fit_empirical_xy) else "bekker_classic"
    if args.fit_mean_force and args.force_model == "current":
        args.force_model = "wheel_mean"
    if args.fit_gated_bekker and args.force_model == "current":
        args.force_model = "gated_bekker_load_transfer"

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    header = pd.read_csv(csv_path, nrows=0)
    case_split = build_case_split(csv_path, args.chunksize, args.case_split_ratio, args.seed)
    fit_case_filter = cases_for_split(case_split, args.fit_case_split)
    eval_case_filter = cases_for_split(case_split, args.eval_case_split)
    if case_split.get("enabled"):
        with open(output_dir / "case_split.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "train_count": case_split["train_count"],
                    "val_count": case_split["val_count"],
                    "train_ratio": case_split["train_ratio"],
                    "seed": case_split["seed"],
                    "fit_case_split": args.fit_case_split,
                    "eval_case_split": args.eval_case_split,
                    "train_cases": sorted(case_split["train"]),  # type: ignore[index]
                    "val_cases": sorted(case_split["val"]),  # type: ignore[index]
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    aux_cols = optional_aux_columns(header.columns.astype(str))
    mapping = required_columns(
        header.columns.astype(str),
        args.contact_filter,
        args.velocity_source,
        args.angular_source,
        args.force_frame,
    )
    usecols = sorted(
        {
            "case_name",
            "time",
            *[col for group in mapping.values() for col in group.values()],
            *aux_cols.values(),
        }
        & set(header.columns.astype(str))
    )

    device = torch.device(args.device)
    terrain_values: Optional[List[float]] = None
    fit_result: Optional[Dict[str, object]] = None
    bekker_result: Optional[Dict[str, object]] = None
    bekker_eval_values: Optional[List[float]] = None
    bekker_z_offset = 0.0
    bekker_static_fz: Optional[List[float]] = None
    bekker_empirical_xy: Optional[Dict[str, object]] = None
    mean_force_result: Optional[Dict[str, object]] = None
    gated_result: Optional[Dict[str, object]] = None
    gated_eval_tensors: Optional[Dict[str, torch.Tensor]] = None
    if args.fit_params:
        log("collecting samples for terrain parameter fitting")
        fit_arrays = collect_fit_arrays(
            csv_path=csv_path,
            usecols=usecols,
            mapping=mapping,
            max_rows=args.fit_max_rows,
            chunksize=args.chunksize,
            contact_filter=args.contact_filter,
            velocity_source=args.velocity_source,
            angular_source=args.angular_source,
            force_frame=args.force_frame,
            seed=args.seed,
            aux_cols=aux_cols,
            case_filter=fit_case_filter,
        )
        terrain_values = fit_terrain_params(
            arrays=fit_arrays,
            output_dir=output_dir,
            steps=args.fit_steps,
            batch_size=args.fit_batch_size,
            lr=args.fit_lr,
            sinkage_max=args.sinkage_max,
            device=device,
            axis_weights=parse_axis_weights(args.fit_axis_weights),
            loss_name=args.fit_loss,
            seed=args.seed,
            wheel_radius=args.wheel_radius,
            wheel_width=args.wheel_width,
            force_frame=args.force_frame,
        )
        fit_result = {
            "param_order": TERRAIN_PARAM_KEYS,
            "initial": dict(zip(TERRAIN_PARAM_KEYS, [float(v) for v in DEFAULT_TERRAIN_PARAMS])),
            "fitted": dict(zip(TERRAIN_PARAM_KEYS, [float(v) for v in terrain_values])),
        }
    if args.fit_bekker_classic:
        log("collecting samples for Bekker parameter fitting")
        fit_arrays = collect_fit_arrays(
            csv_path=csv_path,
            usecols=usecols,
            mapping=mapping,
            max_rows=args.fit_max_rows,
            chunksize=args.chunksize,
            contact_filter=args.contact_filter,
            velocity_source=args.velocity_source,
            angular_source=args.angular_source,
            force_frame=args.force_frame,
            seed=args.seed,
            aux_cols=aux_cols,
            case_filter=fit_case_filter,
        )
        bekker_result = fit_bekker_classic(
            arrays=fit_arrays,
            output_dir=output_dir,
            steps=args.fit_steps,
            batch_size=args.fit_batch_size,
            lr=args.fit_lr,
            sinkage_max=args.sinkage_max,
            device=device,
            axis_weights=parse_axis_weights(args.fit_axis_weights),
            loss_name=args.fit_loss,
            seed=args.seed,
            wheel_radius=args.wheel_radius,
            wheel_width=args.wheel_width,
            fit_z_offset=args.fit_z_offset,
            fit_static_fz=args.fit_static_fz,
            fit_empirical_xy=args.fit_empirical_xy,
        )
        bekker_eval_values, bekker_z_offset, bekker_static_fz, bekker_empirical_xy = bekker_export_to_eval_values(bekker_result)
    elif args.force_model in {"bekker_classic", "bekker_static_empirical"}:
        default_bekker = BekkerClassicFitter(
            fit_static_fz=args.force_model == "bekker_static_empirical",
            fit_empirical_xy=args.force_model == "bekker_static_empirical",
        ).export()
        bekker_eval_values, bekker_z_offset, bekker_static_fz, bekker_empirical_xy = bekker_export_to_eval_values(default_bekker)
    if args.fit_mean_force:
        log("collecting samples for wheel mean force fitting")
        fit_arrays = collect_fit_arrays(
            csv_path=csv_path,
            usecols=usecols,
            mapping=mapping,
            max_rows=args.fit_max_rows,
            chunksize=args.chunksize,
            contact_filter=args.contact_filter,
            velocity_source=args.velocity_source,
            angular_source=args.angular_source,
            force_frame=args.force_frame,
            seed=args.seed,
            aux_cols=aux_cols,
            case_filter=fit_case_filter,
        )
        mean_force_result = fit_wheel_mean_force(fit_arrays)
        with open(output_dir / "fitted_wheel_mean_force.json", "w", encoding="utf-8") as f:
            json.dump(mean_force_result, f, ensure_ascii=False, indent=2)
    if args.fit_gated_bekker:
        log("collecting samples for gated Bekker/load-transfer fitting")
        fit_arrays = collect_fit_arrays(
            csv_path=csv_path,
            usecols=usecols,
            mapping=mapping,
            max_rows=args.fit_max_rows,
            chunksize=args.chunksize,
            contact_filter=args.contact_filter,
            velocity_source=args.velocity_source,
            angular_source=args.angular_source,
            force_frame=args.force_frame,
            seed=args.seed,
            aux_cols=aux_cols,
            case_filter=fit_case_filter,
        )
        gated_result = fit_gated_bekker_load_transfer(
            arrays=fit_arrays,
            output_dir=output_dir,
            steps=args.fit_steps,
            batch_size=args.fit_batch_size,
            lr=args.fit_lr,
            sinkage_max=args.sinkage_max,
            device=device,
            axis_weights=parse_axis_weights(args.fit_axis_weights),
            loss_name=args.fit_loss,
            seed=args.seed,
            wheel_radius=args.wheel_radius,
            wheel_width=args.wheel_width,
            use_bekker=bool(args.gated_use_bekker),
            use_load_transfer=bool(args.gated_use_load_transfer),
            use_xy_empirical=bool(args.gated_use_xy_empirical),
        )
        gated_eval_tensors = gated_export_to_tensors(gated_result, device)
    elif args.force_model == "gated_bekker_load_transfer":
        raise ValueError("--force_model gated_bekker_load_transfer 需要同时使用 --fit_gated_bekker")
    if args.force_model == "wheel_mean" and mean_force_result is None:
        log("collecting samples for wheel mean force fitting")
        fit_arrays = collect_fit_arrays(
            csv_path=csv_path,
            usecols=usecols,
            mapping=mapping,
            max_rows=args.fit_max_rows,
            chunksize=args.chunksize,
            contact_filter=args.contact_filter,
            velocity_source=args.velocity_source,
            angular_source=args.angular_source,
            force_frame=args.force_frame,
            seed=args.seed,
            aux_cols=aux_cols,
            case_filter=fit_case_filter,
        )
        mean_force_result = fit_wheel_mean_force(fit_arrays)
        with open(output_dir / "fitted_wheel_mean_force.json", "w", encoding="utf-8") as f:
            json.dump(mean_force_result, f, ensure_ascii=False, indent=2)

    stats = {f"wheel{i}": RunningStats() for i in WHEEL_IDS}
    all_stats = RunningStats()
    transform_stats = {name: RunningStats() for name in TANGENT_TRANSFORMS} if args.tangent_transform_search else {}
    linear_map_stats = None
    if args.fit_linear_force_map:
        scopes = [scope.strip() for scope in args.linear_map_scopes.split(",") if scope.strip()]
        allowed = {"overall", "axis", "wheel", "wheel_axis"}
        bad = sorted(set(scopes) - allowed)
        if bad:
            raise ValueError(f"--linear_map_scopes 包含未知 scope: {bad}")
        linear_map_stats = LinearForceMapStats(scopes)
    samples: List[pd.DataFrame] = []
    rows_seen = 0

    for chunk_idx, chunk in enumerate(pd.read_csv(csv_path, usecols=usecols, chunksize=args.chunksize), start=1):
        chunk = filter_chunk_cases(chunk, eval_case_filter)
        if chunk.empty:
            log(f"processed chunk={chunk_idx}, rows={rows_seen}")
            continue
        if args.max_rows > 0:
            remaining = args.max_rows - rows_seen
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining].copy()
        rows_seen += len(chunk)
        for wheel_id in WHEEL_IDS:
            wheel_map = mapping[f"wheel{wheel_id}"]
            pos_cols = [wheel_map[f"pos_{axis}"] for axis in ("x", "y", "z")]
            force_cols = [wheel_map[axis] for axis in AXES]
            body_velocity, omega, wheel_quat = build_velocity_and_omega(
                chunk,
                mapping,
                wheel_id,
                args.velocity_source,
                args.angular_source,
            )
            if args.force_frame == "world_from_wheel" and wheel_quat is None:
                wheel_quat = numeric_frame(chunk, quat_cols_for_wheel(mapping, wheel_id))
            wheel_pos = numeric_frame(chunk, pos_cols)
            true_force = numeric_frame(chunk, force_cols)
            in_contact = None
            if args.contact_filter == "hf":
                in_contact = pd.to_numeric(chunk[wheel_map["in_contact"]], errors="coerce").to_numpy(dtype=np.float32)

            if args.force_model == "current":
                pred_force, sinkage, contact = compute_phy_force(
                    body_velocity=body_velocity,
                    wheel_pos=wheel_pos,
                    omega=omega,
                    in_contact=in_contact,
                    sinkage_max=args.sinkage_max,
                    device=device,
                    terrain_values=terrain_values,
                    wheel_radius=args.wheel_radius,
                    wheel_width=args.wheel_width,
                    force_frame=args.force_frame,
                    wheel_quat=wheel_quat,
                )
            elif args.force_model in {"bekker_classic", "bekker_static_empirical"}:
                if bekker_eval_values is None:
                    raise RuntimeError("Bekker force_model 缺少 bekker_eval_values")
                pred_force, sinkage, contact = compute_bekker_force(
                    body_velocity=body_velocity,
                    wheel_pos=wheel_pos,
                    omega=omega,
                    wheel_id=wheel_id,
                    sinkage_max=args.sinkage_max,
                    device=device,
                    bekker_values=bekker_eval_values,
                    wheel_radius=args.wheel_radius,
                    wheel_width=args.wheel_width,
                    z_offset=bekker_z_offset,
                    static_fz_values=bekker_static_fz,
                    empirical_xy_values=bekker_empirical_xy if args.force_model == "bekker_static_empirical" else None,
                )
            elif args.force_model == "wheel_mean":
                if mean_force_result is None:
                    raise RuntimeError("wheel_mean force_model 缺少 mean_force_result")
                pred_force = wheel_mean_predict(wheel_id, len(chunk), mean_force_result)
                sinkage = np.clip(args.wheel_radius - wheel_pos[:, 2], 0.0, args.sinkage_max)
                contact = sinkage > TerramechanicsParams().contact_threshold
            elif args.force_model == "gated_bekker_load_transfer":
                if gated_eval_tensors is None:
                    raise RuntimeError("gated_bekker_load_transfer 缺少 fitted 参数")
                aux_np = numeric_aux_frame(chunk, aux_cols)
                with torch.no_grad():
                    body_t = torch.from_numpy(body_velocity).to(device=device, dtype=torch.float32)
                    pos_t = torch.from_numpy(wheel_pos).to(device=device, dtype=torch.float32)
                    omega_t = torch.from_numpy(omega).to(device=device, dtype=torch.float32)
                    wheel_t = torch.full_like(omega_t, int(wheel_id), dtype=torch.long)
                    aux_t = torch.from_numpy(aux_np).to(device=device, dtype=torch.float32)
                    bekker = bekker_classic_force_torch(
                        body_velocity=body_t,
                        wheel_pos=pos_t,
                        omega=omega_t,
                        wheel_id=wheel_t,
                        params=torch.as_tensor([2.0e4, 6.0e5, 1.0, 50.0, 0.5, 0.02], device=device, dtype=torch.float32),
                        wheel_radius=args.wheel_radius,
                        wheel_width=args.wheel_width,
                        sinkage_max=args.sinkage_max,
                    )
                    pred_force_t = gated_predict_torch(
                        bekker["force"],
                        body_t,
                        omega_t,
                        wheel_t,
                        aux_t,
                        gated_eval_tensors,
                        use_bekker=bool(args.gated_use_bekker),
                        use_load_transfer=bool(args.gated_use_load_transfer),
                        use_xy_empirical=bool(args.gated_use_xy_empirical),
                    )
                pred_force = pred_force_t.cpu().numpy()
                sinkage = bekker["sinkage"].cpu().numpy()
                contact = bekker["contact"].cpu().numpy()
            else:
                raise ValueError(f"未知 force_model: {args.force_model}")
            stats[f"wheel{wheel_id}"].update(pred_force, true_force, sinkage, contact)
            all_stats.update(pred_force, true_force, sinkage, contact)
            if linear_map_stats is not None:
                linear_map_stats.update(wheel_id, pred_force, true_force)
            for transform_name, transform_stat in transform_stats.items():
                transform_stat.update(apply_tangent_transform(pred_force, transform_name), true_force, sinkage, contact)
            append_sample(
                samples,
                chunk,
                wheel_id,
                pred_force,
                true_force,
                sinkage,
                contact,
                args.max_sample_rows,
                args.seed + chunk_idx * 100,
            )
        log(f"processed chunk={chunk_idx}, rows={rows_seen}")
        if args.max_rows > 0 and rows_seen >= args.max_rows:
            break

    sample_df = sample_rows(samples, args.max_sample_rows, args.seed)
    if not sample_df.empty:
        sample_df.to_csv(output_dir / "flat_phy_vs_hf_samples.csv", index=False)
        plot_samples(sample_df, output_dir)

    result = {
        "config": {
            "csv": str(csv_path),
            "terrain": "flat plane: contact_point=(0,0,0), contact_normal=(0,0,1)",
            "terrain_params": dict(zip(TERRAIN_PARAM_KEYS, [float(v) for v in DEFAULT_TERRAIN_PARAMS])),
            "contact_filter": args.contact_filter,
            "sinkage_max": args.sinkage_max,
            "rows_seen": rows_seen,
            "chunksize": args.chunksize,
            "max_rows": args.max_rows,
            "case_split_enabled": bool(case_split.get("enabled")),
            "case_split_ratio": args.case_split_ratio,
            "case_split_train_count": case_split.get("train_count", 0),
            "case_split_val_count": case_split.get("val_count", 0),
            "fit_case_split": args.fit_case_split,
            "eval_case_split": args.eval_case_split,
            "fit_params": bool(args.fit_params),
            "fit_mean_force": bool(args.fit_mean_force),
            "fit_bekker_classic": bool(args.fit_bekker_classic),
            "fit_gated_bekker": bool(args.fit_gated_bekker),
            "gated_use_bekker": bool(args.gated_use_bekker),
            "gated_use_load_transfer": bool(args.gated_use_load_transfer),
            "gated_use_xy_empirical": bool(args.gated_use_xy_empirical),
            "fit_z_offset": bool(args.fit_z_offset),
            "fit_static_fz": bool(args.fit_static_fz),
            "fit_empirical_xy": bool(args.fit_empirical_xy),
            "force_model": args.force_model,
            "velocity_source": args.velocity_source,
            "angular_source": args.angular_source,
            "force_frame": args.force_frame,
            "wheel_radius": args.wheel_radius,
            "wheel_width": args.wheel_width,
            "sph_aligned": bool(args.sph_aligned),
            "tangent_transform_search": bool(args.tangent_transform_search),
            "fit_linear_force_map": bool(args.fit_linear_force_map),
            "linear_map_scopes": args.linear_map_scopes,
            "aux_columns": aux_cols,
        },
        "overall": all_stats.as_dict(),
        "by_wheel": {name: value.as_dict() for name, value in stats.items()},
    }
    if linear_map_stats is not None:
        linear_summary = linear_map_stats.as_dict()
        result["force_linear_map"] = linear_summary
        with open(output_dir / "force_linear_map.json", "w", encoding="utf-8") as f:
            json.dump(linear_summary, f, ensure_ascii=False, indent=2)
    if transform_stats:
        tangent_summary = summarize_transform_stats(transform_stats)
        result["tangent_transform_search"] = tangent_summary
        with open(output_dir / "tangent_transform_search.json", "w", encoding="utf-8") as f:
            json.dump(tangent_summary, f, ensure_ascii=False, indent=2)
    if fit_result is not None:
        result["fitted_params"] = fit_result
    if bekker_result is not None:
        result["fitted_bekker"] = bekker_result
    if mean_force_result is not None:
        result["fitted_wheel_mean_force"] = mean_force_result
    if gated_result is not None:
        result["fitted_gated_bekker_load_transfer"] = gated_result
    with open(output_dir / "flat_phy_vs_hf_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    log(f"metrics saved to: {output_dir / 'flat_phy_vs_hf_metrics.json'}")
    if not sample_df.empty:
        log(f"samples saved to: {output_dir / 'flat_phy_vs_hf_samples.csv'}")
        log(f"plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
