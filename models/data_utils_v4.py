from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from .load_transfer_force_v4 import (
    load_transfer_aux_columns,
    load_transfer_aux_from_df,
    load_transfer_force_numpy,
    wheel_vz_from_df,
)

from .data_utils import (  # noqa: F401
    BODY_BASE_KEYWORDS,
    BODY_PROXY_KEYWORDS,
    ROCKER_NAMES,
    ROCKER_PROXY_KEYWORDS,
    ROCKER_BASE_KEYWORDS,
    SYSTEM_GLOBAL_KEYWORDS,
    WHEEL_BASE_CONTACT_KEYWORDS,
    WHEEL_BASE_KIN_KEYWORDS,
    WHEEL_CONTACT_PROXY_KEYWORDS,
    WHEEL_IDS,
    WHEEL_KIN_PROXY_KEYWORDS,
    ColumnSpec,
    GraphTemporalSequenceDataset as _BaseDataset,
    GroupStandardizer,
    InputGroupSpec,
    NumpyStandardScaler,
    build_input_groups,
    check_columns_exist,
    ensure_time_feature_columns,
    extract_group_arrays_from_df,
    flatten_group_columns,
    get_group_dims,
    get_wheel_id,
    graph_temporal_collate_fn,
    has_any_keyword,
    is_body_col,
    is_no_standardize_col,
    is_rocker_col,
    is_rocker_state_col,
    is_system_col,
    is_wheel_col,
    is_wheel_contact_col,
    is_wheel_kin_col,
    load_column_list,
    load_merged_dataset,
    map_hf_cols_to_lf,
    preprocess_input_group_array,
    print_group_summary,
    save_column_spec_json,
    sort_columns,
    split_train_val_test_by_case,
)

BODY_OUTPUT_KEYWORDS = ["_pos_", "_vel_"]
WHEEL_OUTPUT_KIN_KEYWORDS = ["_pos_", "_ang_vel_"]
WHEEL_OUTPUT_CONTACT_KEYWORDS = ["_Fx", "_Fy", "_Fz"]
TERRAIN_PARAM_KEYS = ["Kc", "Kphi", "n0", "n1", "c", "phi", "K"]
DEFAULT_TERRAIN_PARAMS = np.asarray(
    [-20700.0, 1594800.0, 0.79, 0.70, 460.0, 0.61, 0.0133],
    dtype=np.float32,
)
DEFAULT_TERRAIN_FILE = Path(__file__).resolve().parent / "TerrainPlugins" / "zero_terrain_reader.txt"
DEFAULT_VEHICLE_CONFIG = Path(__file__).resolve().parents[1] / "vehicle_config" / "Zhurong_Config.json"
ASSEMBLY_SIDES = {
    "left": {"front": "lf", "rear": "lm", "sub": "lb", "front_wheel": 0, "mid_wheel": 2, "rear_wheel": 4, "front_motor": "front_l", "rear_motor": "rear_l", "sub_motor": "sub_l"},
    "right": {"front": "rf", "rear": "rm", "sub": "rb", "front_wheel": 1, "mid_wheel": 3, "rear_wheel": 5, "front_motor": "front_r", "rear_motor": "rear_r", "sub_motor": "sub_r"},
}

V4_EXCLUDED_EXACT_INPUT_COLS = {
    "lf_time",
    "lf_time_norm",
    "lf_sim_dt",
    "lf_idle_time",
    "lf_accel_time",
    "lf_const_time",
    "lf_sinusoid_time",
    "lf_step_time",
    "lf_dec_time",
    "lf_acc_residual_x",
    "lf_acc_residual_y",
    "lf_acc_residual_z",
    "lf_acc_x_std",
    "lf_acc_y_std",
    "lf_acc_z_std",
    "lf_ang_acc_x_std",
    "lf_ang_acc_y_std",
    "lf_ang_acc_z_std",
}
V4_REDUNDANT_ATTITUDE_SUFFIXES = (
    "_roll",
    "_pitch",
    "_sin_yaw",
    "_cos_yaw",
    "_d_roll",
    "_dd_roll",
    "_d_pitch",
    "_dd_pitch",
)
V4_UPRIGHT_RE = re.compile(r"^lf_susp_upright[0-5]_")
V4_WHEEL_ACC_STD_RE = re.compile(r"^lf_wheel\d+_acc_[xyz]_std$")
V4_WHEEL_CONTACT_RATE_RE = re.compile(r"^lf_wheel\d+_(slip_long|slip_lat|sinkage)_rate$")
FORCE_DIRECT_BODY_COLS = ["lf_acc_x", "lf_acc_y", "lf_acc_z"]
FORCE_DIRECT_WHEEL_SUFFIXES = [
    "Fx",
    "Fy",
    "Fz",
    "in_contact",
    "sinkage",
    "slip_long",
    "slip_lat",
    "vel_x",
    "vel_y",
    "vel_z",
    "acc_x",
    "acc_y",
    "acc_z",
    "rel_vel_x",
    "rel_vel_y",
    "rel_vel_z",
]


@dataclass
class OutputGroupSpec:
    body_cols: List[str]
    wheel_kin_cols: Dict[int, List[str]]
    wheel_contact_cols: Dict[int, List[str]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _is_omega_col(col: str) -> bool:
    return col.endswith("omega") or col.endswith("_omega") or col.endswith("ang_vel_y") or col.endswith("ang_vel_z") or col.endswith("ang_vel_x")


def is_body_output_col(col: str, prefix: str) -> bool:
    if not col.startswith(prefix) or is_wheel_col(col) or is_rocker_col(col):
        return False
    return has_any_keyword(col, BODY_OUTPUT_KEYWORDS)


def is_wheel_output_kin_col(col: str, prefix: str) -> bool:
    if not col.startswith(prefix) or not is_wheel_col(col):
        return False
    return has_any_keyword(col, ["_pos_"]) or _is_omega_col(col)


def is_wheel_output_contact_col(col: str, prefix: str) -> bool:
    if not col.startswith(prefix) or not is_wheel_col(col):
        return False
    return col.endswith("Fx") or col.endswith("Fy") or col.endswith("Fz")


def build_output_groups(cols: List[str], prefix: str) -> OutputGroupSpec:
    body_cols = sort_columns([c for c in cols if is_body_output_col(c, prefix)])
    wheel_kin_cols = {i: [] for i in WHEEL_IDS}
    wheel_contact_cols = {i: [] for i in WHEEL_IDS}
    for c in cols:
        wid = get_wheel_id(c)
        if wid is None:
            continue
        if is_wheel_output_kin_col(c, prefix):
            wheel_kin_cols[wid].append(c)
        elif is_wheel_output_contact_col(c, prefix):
            wheel_contact_cols[wid].append(c)
    for i in WHEEL_IDS:
        wheel_kin_cols[i] = sort_columns(wheel_kin_cols[i])
        wheel_contact_cols[i] = sort_columns(wheel_contact_cols[i])
    return OutputGroupSpec(body_cols, wheel_kin_cols, wheel_contact_cols)


def is_v4_excluded_input_col(col: str) -> bool:
    if col in V4_EXCLUDED_EXACT_INPUT_COLS:
        return True
    if V4_UPRIGHT_RE.match(col):
        return True
    if col.endswith(V4_REDUNDANT_ATTITUDE_SUFFIXES):
        return True
    if V4_WHEEL_ACC_STD_RE.match(col):
        return True
    if V4_WHEEL_CONTACT_RATE_RE.match(col):
        return True
    return False


def filter_v4_input_feature_cols(cols: List[str]) -> List[str]:
    return [col for col in cols if not is_v4_excluded_input_col(col)]


def load_column_spec(feature_dir: str) -> ColumnSpec:
    base_feature_cols = filter_v4_input_feature_cols(load_column_list(f"{feature_dir}/base_feature_columns.csv"))
    proxy_feature_cols = filter_v4_input_feature_cols(load_column_list(f"{feature_dir}/proxy_feature_columns.csv"))
    res_cols = load_column_list(f"{feature_dir}/res_columns.csv")
    target_cols = load_column_list(f"{feature_dir}/target_columns.csv")
    return ColumnSpec(
        base_feature_cols=base_feature_cols,
        proxy_feature_cols=proxy_feature_cols,
        res_cols=res_cols,
        target_cols=target_cols,
        input_groups=build_input_groups(base_feature_cols, proxy_feature_cols),
        res_groups=build_output_groups(res_cols, "res_"),
        target_groups=build_output_groups(target_cols, "hf_"),
    )


def _columns_present(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def _suffix_index(cols: List[str], suffix: str) -> Optional[int]:
    for idx, col in enumerate(cols):
        if col.endswith(suffix):
            return idx
    return None


def _suffix_indices(cols: List[str], suffixes: List[str]) -> List[int]:
    out: List[int] = []
    for suffix in suffixes:
        idx = _suffix_index(cols, suffix)
        if idx is not None:
            out.append(idx)
    return out


def _omega_index(cols: List[str]) -> Optional[int]:
    for idx, col in enumerate(cols):
        if col.endswith("omega") or col.endswith("_omega") or "ang_vel" in col:
            return idx
    return None


def rocker_pose_cols(name: str, prefix: str = "lf") -> List[str]:
    return [f"{prefix}_susp_rocker_{name}_{suffix}" for suffix in ["pos_x", "pos_y", "pos_z", "q0", "q1", "q2", "q3"]]


def body_pose_cols(prefix: str = "lf") -> List[str]:
    return [f"{prefix}_{suffix}" for suffix in ["pos_x", "pos_y", "pos_z", "q0", "q1", "q2", "q3"]]


def wheel_pose_cols(wheel_id: int, prefix: str = "lf") -> List[str]:
    return [f"{prefix}_wheel{wheel_id}_{suffix}" for suffix in ["pos_x", "pos_y", "pos_z", "q0", "q1", "q2", "q3"]]


def _mirror_y(point):
    x, y, z = [float(v) for v in point]
    return [x, -y, z]


def _enforce_config_symmetry(config: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(config)
    for key in ["wheel_rel_pos", "steering_upright_pos", "steer_motor_loc", "rocker_pos", "rocker_motor_loc"]:
        out[key] = dict(config[key])
    for left, right in [("lf", "rf"), ("lm", "rm"), ("lb", "rb")]:
        for key in ["wheel_rel_pos", "steering_upright_pos", "steer_motor_loc", "rocker_pos"]:
            out[key][right] = _mirror_y(out[key][left])
    for left, right in [("front_l", "front_r"), ("rear_l", "rear_r"), ("sub_l", "sub_r")]:
        out["rocker_motor_loc"][right] = _mirror_y(out["rocker_motor_loc"][left])
    return out


def _quat_to_matrix_np(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    w, x, y, z = q
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _pose_from_row_np(row: pd.Series, cols: List[str]) -> tuple[np.ndarray, np.ndarray]:
    p = row[cols[:3]].to_numpy(dtype=np.float64)
    r = _quat_to_matrix_np(row[cols[3:7]].to_numpy(dtype=np.float64))
    return p, r


def _lpf_np(x: np.ndarray, alpha: float) -> np.ndarray:
    if x.shape[0] <= 1:
        return x.astype(np.float32, copy=True)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    y = np.empty_like(x, dtype=np.float32)
    y[0] = x[0]
    for t in range(1, x.shape[0]):
        y[t] = y[t - 1] + alpha * (x[t] - y[t - 1])
    return y


class TerrainParameterLookup:
    _cache: Dict[str, "TerrainParameterLookup"] = {}

    def __init__(self, path: Path):
        self.path = path
        self.param_keys = list(TERRAIN_PARAM_KEYS)
        self.default_params = DEFAULT_TERRAIN_PARAMS.copy()
        self.is_constant = True
        self.constant_params = self.default_params.copy()
        self.grid_x: Optional[np.ndarray] = None
        self.grid_y: Optional[np.ndarray] = None
        self.grid_z: Optional[np.ndarray] = None
        self.grid_params: Optional[np.ndarray] = None
        self.points_xy: Optional[np.ndarray] = None
        self.points_z: Optional[np.ndarray] = None
        self.points_params: Optional[np.ndarray] = None
        self.source = "default"
        self._load()

    @classmethod
    def get(cls, path: Path = DEFAULT_TERRAIN_FILE) -> "TerrainParameterLookup":
        key = str(path.resolve())
        if key not in cls._cache:
            cls._cache[key] = cls(path)
        return cls._cache[key]

    def _load(self) -> None:
        if not self.path.exists():
            return
        data = np.loadtxt(str(self.path), dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 10:
            raise ValueError(f"地形文件至少需要 10 列 x y z Kc Kphi n0 n1 c phi K，实际 shape={data.shape}")
        terrain_z = data[:, 2].astype(np.float32, copy=False)
        params = data[:, 3:10].astype(np.float32, copy=False)
        self.default_params = params[0].copy()
        self.constant_params = params[0].copy()
        self.is_constant = bool(np.allclose(params, params[0:1], rtol=1e-6, atol=1e-8) and np.allclose(terrain_z, terrain_z[0:1], rtol=1e-6, atol=1e-8))
        self.source = str(self.path)

        xy = data[:, :2].astype(np.float32, copy=False)
        xs = np.unique(xy[:, 0])
        ys = np.unique(xy[:, 1])
        if xs.size * ys.size == xy.shape[0]:
            order = np.lexsort((xy[:, 1], xy[:, 0]))
            xy_sorted = xy[order]
            z_sorted = terrain_z[order]
            params_sorted = params[order]
            expected_x = np.repeat(xs, ys.size)
            expected_y = np.tile(ys, xs.size)
            if np.allclose(xy_sorted[:, 0], expected_x, rtol=0.0, atol=1e-6) and np.allclose(xy_sorted[:, 1], expected_y, rtol=0.0, atol=1e-6):
                self.grid_x = xs
                self.grid_y = ys
                self.grid_z = z_sorted.reshape(xs.size, ys.size)
                self.grid_params = params_sorted.reshape(xs.size, ys.size, len(TERRAIN_PARAM_KEYS))
                return

        self.points_xy = xy.copy()
        self.points_z = terrain_z.copy()
        self.points_params = params.copy()

    def sample_point(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        out_shape = x_arr.shape + (len(TERRAIN_PARAM_KEYS),)
        z_shape = x_arr.shape
        if self.grid_x is not None and self.grid_y is not None and self.grid_params is not None:
            flat_x = x_arr.reshape(-1)
            flat_y = y_arr.reshape(-1)
            ix = np.searchsorted(self.grid_x, flat_x) - 1
            iy = np.searchsorted(self.grid_y, flat_y) - 1
            ix = np.clip(ix, 0, self.grid_x.size - 2)
            iy = np.clip(iy, 0, self.grid_y.size - 2)
            x0 = self.grid_x[ix]
            y0 = self.grid_y[iy]
            x1 = self.grid_x[ix + 1]
            y1 = self.grid_y[iy + 1]
            tx = (flat_x - x0) / np.maximum(x1 - x0, 1e-8)
            ty = (flat_y - y0) / np.maximum(y1 - y0, 1e-8)
            u0 = (1.0 - tx) * (1.0 - ty)
            u1 = (1.0 - tx) * ty
            u2 = tx * (1.0 - ty)
            u3 = tx * ty
            z_grid = self.grid_z if self.grid_z is not None else np.zeros(self.grid_params.shape[:2], dtype=np.float32)
            z = (
                u0 * z_grid[ix, iy]
                + u1 * z_grid[ix, iy + 1]
                + u2 * z_grid[ix + 1, iy]
                + u3 * z_grid[ix + 1, iy + 1]
            ).reshape(z_shape).astype(np.float32, copy=False)
            p = (
                u0[:, None] * self.grid_params[ix, iy]
                + u1[:, None] * self.grid_params[ix, iy + 1]
                + u2[:, None] * self.grid_params[ix + 1, iy]
                + u3[:, None] * self.grid_params[ix + 1, iy + 1]
            ).reshape(out_shape).astype(np.float32, copy=False)
            return z, p

        if self.points_xy is None or self.points_params is None:
            return (
                np.zeros(z_shape, dtype=np.float32),
                np.broadcast_to(self.default_params, out_shape).astype(np.float32, copy=True),
            )
        flat_x = x_arr.reshape(-1)
        flat_y = y_arr.reshape(-1)
        result = np.empty((flat_x.size, len(TERRAIN_PARAM_KEYS)), dtype=np.float32)
        result_z = np.empty(flat_x.size, dtype=np.float32)
        query = np.stack([flat_x, flat_y], axis=1)
        chunk = 256
        for start in range(0, query.shape[0], chunk):
            q = query[start:start + chunk]
            dist2 = ((q[:, None, :] - self.points_xy[None, :, :]) ** 2).sum(axis=-1)
            nearest = np.argmin(dist2, axis=1)
            result[start:start + chunk] = self.points_params[nearest]
            result_z[start:start + chunk] = self.points_z[nearest] if self.points_z is not None else 0.0
        return result_z.reshape(z_shape), result.reshape(out_shape)

    def contact_plane(
        self,
        wheel_x: np.ndarray,
        wheel_y: np.ndarray,
        r: float = 0.135,
        b: float = 0.16,
        theta1_in: float = 0.01,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        wheel_x = np.asarray(wheel_x, dtype=np.float32)
        wheel_y = np.asarray(wheel_y, dtype=np.float32)
        dx = np.float32(r * np.sin(theta1_in))
        dy = np.float32(0.5 * b)
        z1, p1 = self.sample_point(wheel_x + dx, wheel_y + dy)
        z2, p2 = self.sample_point(wheel_x + dx, wheel_y - dy)
        z3, p3 = self.sample_point(wheel_x - dx, wheel_y)
        pa = np.stack([wheel_x + dx, wheel_y + dy, z1], axis=-1)
        pb = np.stack([wheel_x + dx, wheel_y - dy, z2], axis=-1)
        pc = np.stack([wheel_x - dx, wheel_y, z3], axis=-1)
        normal = np.cross(pb - pc, pa - pc)
        flip = normal[..., 2:3] < 0.0
        normal = np.where(flip, -normal, normal)
        norm = np.linalg.norm(normal, axis=-1, keepdims=True)
        normal = np.where(norm > 1e-8, normal / norm, np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
        params = ((p1 + p2 + p3) / 3.0).astype(np.float32, copy=False)
        return pa.astype(np.float32, copy=False), normal.astype(np.float32, copy=False), params

    def lookup(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        out_shape = x_arr.shape + (len(TERRAIN_PARAM_KEYS),)
        z_shape = x_arr.shape
        if self.is_constant:
            return (
                np.zeros(z_shape, dtype=np.float32),
                np.broadcast_to(self.constant_params, out_shape).astype(np.float32, copy=True),
            )

        flat_x = x_arr.reshape(-1)
        flat_y = y_arr.reshape(-1)
        if self.grid_x is not None and self.grid_y is not None and self.grid_params is not None:
            ix = np.searchsorted(self.grid_x, flat_x)
            iy = np.searchsorted(self.grid_y, flat_y)
            ix = np.clip(ix, 0, self.grid_x.size - 1)
            iy = np.clip(iy, 0, self.grid_y.size - 1)
            ix_prev = np.clip(ix - 1, 0, self.grid_x.size - 1)
            iy_prev = np.clip(iy - 1, 0, self.grid_y.size - 1)
            ix = np.where(np.abs(self.grid_x[ix_prev] - flat_x) <= np.abs(self.grid_x[ix] - flat_x), ix_prev, ix)
            iy = np.where(np.abs(self.grid_y[iy_prev] - flat_y) <= np.abs(self.grid_y[iy] - flat_y), iy_prev, iy)
            z = self.grid_z[ix, iy].reshape(z_shape).astype(np.float32, copy=False) if self.grid_z is not None else np.zeros(z_shape, dtype=np.float32)
            return z, self.grid_params[ix, iy].reshape(out_shape).astype(np.float32, copy=False)

        if self.points_xy is None or self.points_params is None:
            return (
                np.zeros(z_shape, dtype=np.float32),
                np.broadcast_to(self.default_params, out_shape).astype(np.float32, copy=True),
            )
        # Fallback for non-grid terrain files. This path is intentionally chunked to avoid large temporary matrices.
        result = np.empty((flat_x.size, len(TERRAIN_PARAM_KEYS)), dtype=np.float32)
        result_z = np.empty(flat_x.size, dtype=np.float32)
        query = np.stack([flat_x, flat_y], axis=1)
        chunk = 256
        for start in range(0, query.shape[0], chunk):
            q = query[start:start + chunk]
            dist2 = ((q[:, None, :] - self.points_xy[None, :, :]) ** 2).sum(axis=-1)
            nearest = np.argmin(dist2, axis=1)
            result[start:start + chunk] = self.points_params[nearest]
            result_z[start:start + chunk] = self.points_z[nearest] if self.points_z is not None else 0.0
        return result_z.reshape(z_shape), result.reshape(out_shape)


class GraphTemporalSequenceDatasetV4(_BaseDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        spec: ColumnSpec,
        scaler: Optional[GroupStandardizer],
        seq_len: Optional[int] = None,
        pred_horizon: int = 0,
        pred_seq_len: int = 1,
        case_col: str = "case_name",
        time_col: str = "time",
        *,
        history_len: Optional[int] = None,
        teacher_future_len: int = 3,
        body_rollout_len: int = 1,
        force_lpf_alpha: float = 0.15,
        sinkage_max: float = 0.08,
    ):
        if pred_horizon != 0:
            raise ValueError("V4 target 始终对应当前时刻 t，pred_horizon 必须为 0")
        if pred_seq_len != 1:
            raise ValueError("V4 暂只支持单目标时刻，pred_seq_len 必须为 1")
        if history_len is None:
            history_len = int(seq_len) - 1 if seq_len is not None else 9
        self.history_len = int(history_len)
        self.teacher_future_len = int(teacher_future_len)
        self.body_rollout_len = max(1, int(body_rollout_len))
        if self.history_len < 1:
            raise ValueError(f"history_len 必须 >= 1，实际为 {self.history_len}")
        if self.teacher_future_len < 0:
            raise ValueError(f"teacher_future_len 必须 >= 0，实际为 {self.teacher_future_len}")
        super().__init__(
            df,
            spec,
            scaler,
            seq_len=self.history_len + 1,
            pred_horizon=0,
            pred_seq_len=1,
            case_col=case_col,
            time_col=time_col,
        )
        self.lf_rocker_pose_cols = {name: rocker_pose_cols(name, "lf") for name in ROCKER_NAMES}
        self.hf_rocker_pose_cols = {name: rocker_pose_cols(name, "hf") for name in ROCKER_NAMES}
        self.lf_body_pose_cols = body_pose_cols("lf")
        self.hf_body_pose_cols = body_pose_cols("hf")
        self.lf_wheel_pose_cols = {i: wheel_pose_cols(i, "lf") for i in WHEEL_IDS}
        self.hf_wheel_pose_cols = {i: wheel_pose_cols(i, "hf") for i in WHEEL_IDS}
        df_columns = set(self.df.columns)
        self.available_rocker_pose = {
            name: all(c in df_columns for c in self.lf_rocker_pose_cols[name] + self.hf_rocker_pose_cols[name])
            for name in ROCKER_NAMES
        }
        self.available_body_pose = all(c in df_columns for c in self.lf_body_pose_cols + self.hf_body_pose_cols)
        self.available_wheel_pose = {
            i: all(c in df_columns for c in self.lf_wheel_pose_cols[i] + self.hf_wheel_pose_cols[i])
            for i in WHEEL_IDS
        }
        self.assembly_meta = self._build_assembly_metadata()
        self.force_phy_ref: Dict[int, np.ndarray] = {}
        self.force_phy_input: Dict[int, np.ndarray] = {}
        self.force_delta_target: Dict[int, np.ndarray] = {}
        self.load_transfer_aux_cols = load_transfer_aux_columns(self.df.columns.astype(str))
        self.load_transfer_aux = load_transfer_aux_from_df(self.df, self.load_transfer_aux_cols)
        self.load_transfer_wheel_vz = {i: wheel_vz_from_df(self.df, i) for i in WHEEL_IDS}
        self.force_phy_input = {
            i: load_transfer_force_numpy(self.load_transfer_aux, self.load_transfer_wheel_vz[i], i).astype(np.float32, copy=False)
            for i in WHEEL_IDS
        }
        self.force_direct_cols: Dict[int, List[str]] = {}
        self.force_direct_arrays: Dict[int, np.ndarray] = {}
        for i in WHEEL_IDS:
            cols = [f"lf_wheel{i}_{suffix}" for suffix in FORCE_DIRECT_WHEEL_SUFFIXES] + FORCE_DIRECT_BODY_COLS
            missing = [c for c in cols if c not in self.df.columns]
            if missing:
                raise ValueError(f"V4 force direct input for wheel {i} missing LF columns: {missing}")
            if any(c.startswith("hf_") for c in cols):
                raise ValueError(f"V4 force direct input for wheel {i} must be LF-only, got: {cols}")
            self.force_direct_cols[i] = cols
            sources = [
                (self.spec.input_groups.wheel_contact_cols[i], self.group_arrays[f"wheel{i}_contact"]),
                (self.spec.input_groups.wheel_kin_cols[i], self.group_arrays[f"wheel{i}_kin"]),
                (self.spec.input_groups.body_cols, self.group_arrays["body"]),
                (self.spec.input_groups.system_cols, self.group_arrays["system"]),
            ]
            direct_parts: List[np.ndarray] = []
            for col in cols:
                for src_cols, src_arr in sources:
                    if col in src_cols:
                        direct_parts.append(src_arr[:, src_cols.index(col):src_cols.index(col) + 1])
                        break
                else:
                    raise ValueError(f"V4 force direct input {col!r} is not present in input_groups")
            self.force_direct_arrays[i] = np.concatenate(direct_parts, axis=1).astype(np.float32, copy=False)
        self._build_force_reference_targets(float(force_lpf_alpha), float(sinkage_max))

    def _build_assembly_metadata(self) -> Dict[str, Dict[str, np.ndarray]]:
        if not DEFAULT_VEHICLE_CONFIG.exists():
            return {}
        config = _enforce_config_symmetry(json.loads(DEFAULT_VEHICLE_CONFIG.read_text(encoding="utf-8")))
        required = self.lf_body_pose_cols
        if not all(c in self.df.columns for c in required):
            return {}
        meta: Dict[str, Dict[str, np.ndarray]] = {}
        for case_name, sub in self.df.groupby(self.case_col, sort=False):
            if sub.empty:
                continue
            first = sub.sort_values(self.time_col).iloc[0]
            body_p, body_r = _pose_from_row_np(first, required)
            case_meta: Dict[str, np.ndarray] = {}

            def body_world(point):
                return body_p + body_r @ np.asarray(point, dtype=np.float64)

            def local(point_world, cols):
                p, r = _pose_from_row_np(first, cols)
                return r.T @ (point_world - p)

            for side, spec_side in ASSEMBLY_SIDES.items():
                front = spec_side["front"]
                rear = spec_side["rear"]
                sub_name = spec_side["sub"]
                needed = self.lf_rocker_pose_cols[front] + self.lf_rocker_pose_cols[rear] + self.lf_rocker_pose_cols[sub_name]
                if not all(c in self.df.columns for c in needed):
                    continue
                a_front = body_world(config["rocker_motor_loc"][spec_side["front_motor"]])
                a_rear = body_world(config["rocker_motor_loc"][spec_side["rear_motor"]])
                b = body_world(config["steer_motor_loc"][front])
                d = body_world(config["rocker_motor_loc"][spec_side["sub_motor"]])
                mid_key = "lm" if side == "left" else "rm"
                rear_key = "lb" if side == "left" else "rb"
                e = body_world(config["steer_motor_loc"][mid_key])
                f = body_world(config["steer_motor_loc"][rear_key])
                for key, value in {
                    "A_front_body": np.asarray(config["rocker_motor_loc"][spec_side["front_motor"]], dtype=np.float32),
                    "A_rear_body": np.asarray(config["rocker_motor_loc"][spec_side["rear_motor"]], dtype=np.float32),
                    "B_front_local": local(b, self.lf_rocker_pose_cols[front]).astype(np.float32),
                    "D_main_local": local(d, self.lf_rocker_pose_cols[rear]).astype(np.float32),
                    "D_sub_local": local(d, self.lf_rocker_pose_cols[sub_name]).astype(np.float32),
                    "E_local": local(e, self.lf_rocker_pose_cols[sub_name]).astype(np.float32),
                    "F_local": local(f, self.lf_rocker_pose_cols[sub_name]).astype(np.float32),
                }.items():
                    case_meta[f"{side}_{key}"] = value

                wheel_cols = {
                    "C_front": self.lf_wheel_kin_cols[int(spec_side["front_wheel"])],
                    "C_mid": self.lf_wheel_kin_cols[int(spec_side["mid_wheel"])],
                    "C_rear": self.lf_wheel_kin_cols[int(spec_side["rear_wheel"])],
                }
                points = {"A_front": a_front, "A_rear": a_rear, "B_front": b, "D_main": d, "D_sub": d, "E": e, "F": f}
                for point_name, cols in wheel_cols.items():
                    pos_idx = _suffix_indices(cols, ["pos_x", "pos_y", "pos_z"])
                    pos_cols = [cols[j] for j in pos_idx]
                    if all(c in self.df.columns for c in pos_cols):
                        points[point_name] = first[pos_cols].to_numpy(dtype=np.float64)
                refs = {
                    "front_AB": np.linalg.norm(points["A_front"] - points["B_front"]),
                    "front_BC": np.linalg.norm(points["B_front"] - points.get("C_front", points["B_front"])),
                    "front_AC": np.linalg.norm(points["A_front"] - points.get("C_front", points["A_front"])),
                    "rear_main_AD": np.linalg.norm(points["A_rear"] - points["D_main"]),
                    "bogie_DE": np.linalg.norm(points["D_main"] - points["E"]),
                    "bogie_DF": np.linalg.norm(points["D_main"] - points["F"]),
                    "middle_upright": np.linalg.norm(points["E"] - points.get("C_mid", points["E"])),
                    "rear_upright": np.linalg.norm(points["F"] - points.get("C_rear", points["F"])),
                    "middle_rear_wheel": np.linalg.norm(points.get("C_mid", points["E"]) - points.get("C_rear", points["F"])),
                    "sub_joint_coincidence": 0.0,
                }
                for key, value in refs.items():
                    case_meta[f"{side}_ref_{key}"] = np.asarray([value], dtype=np.float32)
            if case_meta:
                meta[str(case_name)] = case_meta
        return meta

    def _build_index(self) -> None:
        start = 0
        for _, sub in self.df.groupby(self.case_col, sort=False):
            n = len(sub)
            first_t = self.history_len
            # A sample predicts the next K rows and each Teacher window may
            # extend teacher_future_len rows beyond its target. Keep all of
            # those rows inside this case before adding the sample index.
            stop_t = n - self.body_rollout_len - self.teacher_future_len
            if stop_t <= first_t:
                start += n
                continue

            candidate_t = np.arange(first_t, stop_t, dtype=np.int64)
            if self.body_rollout_len > 0 and len(candidate_t):
                times = sub[self.time_col].to_numpy(dtype=np.float64, copy=False)
                non_increasing = (np.diff(times) <= 0).astype(np.int64, copy=False)
                bad_prefix = np.concatenate(([0], np.cumsum(non_increasing, dtype=np.int64)))
                invalid_count = bad_prefix[candidate_t + self.body_rollout_len] - bad_prefix[candidate_t]
                candidate_t = candidate_t[invalid_count == 0]

            for local_t in candidate_t:
                self.index_map.append((start, start + local_t))
            start += n

    def _slice_student(self, arr: np.ndarray, t_global: int) -> np.ndarray:
        return arr[t_global - self.history_len:t_global + 1]

    def _slice_teacher(self, arr: np.ndarray, t_global: int) -> np.ndarray:
        return arr[t_global - self.history_len:t_global + self.teacher_future_len + 1]

    def _raw_columns(self, cols: List[str]) -> List[str]:
        return [c for c in cols if c in self.df.columns]

    def _build_force_reference_targets(self, alpha: float, sinkage_max: float) -> None:
        for i in WHEEL_IDS:
            contact_cols = self.spec.target_groups.wheel_contact_cols[i]
            force_idx = _suffix_indices(contact_cols, ["Fx", "Fy", "Fz"])
            if len(force_idx) != 3:
                continue

            force_cols = [contact_cols[j] for j in force_idx]
            if not all(c in self.df.columns for c in force_cols):
                continue

            lf_contact_cols = self.spec.input_groups.wheel_contact_cols[i]
            lf_force_idx = _suffix_indices(lf_contact_cols, ["Fx", "Fy", "Fz"])
            if len(lf_force_idx) != 3:
                continue
            lf_force_cols = [lf_contact_cols[j] for j in lf_force_idx]
            if not all(c in self.df.columns for c in lf_force_cols):
                continue
            f_phy_ref = self.force_phy_input[i]
            f_hf = self.df[force_cols].to_numpy(dtype=np.float32)
            f_lf = self.df[lf_force_cols].to_numpy(dtype=np.float32)
            f_res = f_hf - f_lf
            self.force_phy_ref[i] = f_phy_ref
            self.force_delta_target[i] = f_res.astype(np.float32, copy=False)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        _, t_global = self.index_map[idx]
        target_start_idx = t_global
        target_end_idx = target_start_idx + self.pred_seq_len
        sample: Dict[str, Any] = {
            "case_name": str(self.df.iloc[target_start_idx][self.case_col]),
            "time": np.float32(self.df.iloc[target_start_idx][self.time_col]),
            "teacher_t_index": np.int64(self.history_len),
        }
        case_meta = self.assembly_meta.get(sample["case_name"], {})
        for key, value in case_meta.items():
            sample[f"assembly_{key}"] = torch.tensor(value, dtype=torch.float32)

        def add_input_group(key: str) -> None:
            student = torch.from_numpy(self._slice_student(self.group_arrays[key], t_global))
            teacher = torch.from_numpy(self._slice_teacher(self.group_arrays[key], t_global))
            sample[key] = student
            sample[f"student_{key}"] = student
            sample[f"teacher_{key}"] = teacher

        def add_force_direct_group(wheel_id: int) -> None:
            arr = self.force_direct_arrays[wheel_id]
            student = torch.from_numpy(self._slice_student(arr, t_global).copy())
            teacher = torch.from_numpy(self._slice_teacher(arr, t_global).copy())
            key = f"force_direct_wheel{wheel_id}"
            sample[key] = student
            sample[f"student_{key}"] = student
            sample[f"teacher_{key}"] = teacher

        for key in ["system", "body"]:
            add_input_group(key)
        for name in ROCKER_NAMES:
            add_input_group(name)
        for i in WHEEL_IDS:
            add_input_group(f"wheel{i}_kin")
            add_input_group(f"wheel{i}_contact")
            add_force_direct_group(i)

        body_lf = self.df.iloc[target_start_idx:target_end_idx][self.lf_body_cols].to_numpy(dtype=np.float32)
        sample["lf_body_current"] = torch.tensor(body_lf, dtype=torch.float32)
        if self.available_body_pose:
            sample["lf_body_pose_current"] = torch.tensor(
                self.df.iloc[target_start_idx:target_end_idx][self.lf_body_pose_cols].to_numpy(dtype=np.float32),
                dtype=torch.float32,
            )
            sample["hf_body_pose"] = torch.tensor(
                self.df.iloc[target_start_idx:target_end_idx][self.hf_body_pose_cols].to_numpy(dtype=np.float32),
                dtype=torch.float32,
            )
        for i in WHEEL_IDS:
            sample[f"lf_wheel{i}_kin_current"] = torch.tensor(
                self.df.iloc[target_start_idx:target_end_idx][self.lf_wheel_kin_cols[i]].to_numpy(dtype=np.float32),
                dtype=torch.float32,
            )
            sample[f"lf_wheel{i}_contact_current"] = torch.tensor(
                self.df.iloc[target_start_idx:target_end_idx][self.lf_wheel_contact_cols[i]].to_numpy(dtype=np.float32),
                dtype=torch.float32,
            )
            force_idx_current = _suffix_indices(self.lf_wheel_contact_cols[i], ["Fx", "Fy", "Fz"])
            if len(force_idx_current) == 3:
                sample[f"wheel{i}_lf_force_current"] = sample[f"lf_wheel{i}_contact_current"][..., force_idx_current]
            if self.available_wheel_pose.get(i, False):
                sample[f"lf_wheel{i}_pose_current"] = torch.tensor(
                    self.df.iloc[target_start_idx:target_end_idx][self.lf_wheel_pose_cols[i]].to_numpy(dtype=np.float32),
                    dtype=torch.float32,
                )
                sample[f"hf_wheel{i}_pose"] = torch.tensor(
                    self.df.iloc[target_start_idx:target_end_idx][self.hf_wheel_pose_cols[i]].to_numpy(dtype=np.float32),
                    dtype=torch.float32,
                )
        for name in ROCKER_NAMES:
            if self.available_rocker_pose.get(name, False):
                sample[f"lf_rocker_{name}_pose_current"] = torch.tensor(
                    self.df.iloc[target_start_idx:target_end_idx][self.lf_rocker_pose_cols[name]].to_numpy(dtype=np.float32),
                    dtype=torch.float32,
                )
                sample[f"hf_rocker_{name}_pose"] = torch.tensor(
                    self.df.iloc[target_start_idx:target_end_idx][self.hf_rocker_pose_cols[name]].to_numpy(dtype=np.float32),
                    dtype=torch.float32,
                )

        for key in ["res_body", "hf_body"]:
            sample[key] = torch.tensor(
                self.group_arrays[key][target_start_idx:target_end_idx],
                dtype=torch.float32,
            )
        for i in WHEEL_IDS:
            for key in [f"res_wheel{i}_kin", f"res_wheel{i}_contact", f"hf_wheel{i}_kin", f"hf_wheel{i}_contact"]:
                sample[key] = torch.tensor(
                    self.group_arrays[key][target_start_idx:target_end_idx],
                    dtype=torch.float32,
                )

        hist_start = target_start_idx - self.history_len
        hist_end = target_start_idx
        sample["hf_body_hist"] = torch.tensor(
            self.group_arrays["hf_body"][hist_start:hist_end],
            dtype=torch.float32,
        )
        for i in WHEEL_IDS:
            sample[f"hf_wheel{i}_kin_hist"] = torch.tensor(
                self.group_arrays[f"hf_wheel{i}_kin"][hist_start:hist_end],
                dtype=torch.float32,
            )

        prev_idx = max(target_start_idx - 1, 0)
        if self.df.iloc[prev_idx][self.case_col] != self.df.iloc[target_start_idx][self.case_col]:
            prev_idx = target_start_idx
        sample["dt"] = torch.tensor(np.float32(self.df.iloc[target_start_idx][self.time_col] - self.df.iloc[prev_idx][self.time_col]))
        sample["lf_body_prev_raw"] = torch.tensor(
            self.df.iloc[prev_idx:prev_idx + 1][self.lf_body_cols].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        sample["load_transfer_aux_current"] = torch.tensor(
            self.load_transfer_aux[target_start_idx:target_end_idx],
            dtype=torch.float32,
        )
        for i in WHEEL_IDS:
            sample[f"wheel{i}_load_transfer_vz_current"] = torch.tensor(
                self.load_transfer_wheel_vz[i][target_start_idx:target_end_idx],
                dtype=torch.float32,
            )
            sample[f"wheel{i}_fphy_current"] = torch.tensor(
                self.force_phy_input[i][target_start_idx:target_end_idx],
                dtype=torch.float32,
            )

        def add_force_targets(wheel_id: int) -> None:
            if wheel_id not in self.force_phy_ref:
                return
            sample[f"wheel{wheel_id}_force_phy_ref"] = torch.tensor(
                self.force_phy_ref[wheel_id][target_start_idx:target_end_idx],
                dtype=torch.float32,
            )
            sample[f"wheel{wheel_id}_force_delta_target"] = torch.tensor(
                self.force_delta_target[wheel_id][target_start_idx:target_end_idx],
                dtype=torch.float32,
            )

        for i in WHEEL_IDS:
            add_force_targets(i)

        # Each rollout input is independently causal and belongs to this sample's case.
        rollout_student: Dict[str, List[torch.Tensor]] = {}
        rollout_teacher: Dict[str, List[torch.Tensor]] = {}
        input_keys = ["system", "body", *ROCKER_NAMES, *[f"wheel{i}_kin" for i in WHEEL_IDS],
                      *[f"wheel{i}_contact" for i in WHEEL_IDS]]
        for step in range(1, self.body_rollout_len + 1):
            t_step = target_start_idx + step
            for key in input_keys:
                rollout_student.setdefault(key, []).append(torch.from_numpy(self._slice_student(self.group_arrays[key], t_step).copy()))
                rollout_teacher.setdefault(key, []).append(torch.from_numpy(self._slice_teacher(self.group_arrays[key], t_step).copy()))
            for i in WHEEL_IDS:
                rollout_student.setdefault(f"force_direct_wheel{i}", []).append(torch.from_numpy(self._slice_student(self.force_direct_arrays[i], t_step).copy()))
                rollout_teacher.setdefault(f"force_direct_wheel{i}", []).append(torch.from_numpy(self._slice_teacher(self.force_direct_arrays[i], t_step).copy()))
        for key, values in rollout_student.items():
            sample[f"rollout_student_{key}"] = torch.stack(values)
            sample[f"rollout_teacher_{key}"] = torch.stack(rollout_teacher[key])
        rollout_times = self.df.iloc[target_start_idx:target_start_idx + self.body_rollout_len + 1][self.time_col].to_numpy(dtype=np.float32)
        sample["rollout_dt"] = torch.from_numpy(np.diff(rollout_times).copy())
        for group_key, cols in (("body", self.lf_body_cols), ("hf_body", self.spec.target_groups.body_cols)):
            sample[f"rollout_{group_key}"] = torch.tensor(
                self.df.iloc[target_start_idx + 1:target_start_idx + self.body_rollout_len + 1][cols].to_numpy(dtype=np.float32),
                dtype=torch.float32,
            )
        sample["rollout_hf_body_scaled"] = torch.from_numpy(
            self.group_arrays["hf_body"][target_start_idx + 1:target_start_idx + self.body_rollout_len + 1].copy())
        if self.available_body_pose:
            sample["rollout_lf_body_pose"] = torch.tensor(self.df.iloc[target_start_idx + 1:target_start_idx + self.body_rollout_len + 1][self.lf_body_pose_cols].to_numpy(dtype=np.float32))
            sample["rollout_hf_body_pose"] = torch.tensor(self.df.iloc[target_start_idx + 1:target_start_idx + self.body_rollout_len + 1][self.hf_body_pose_cols].to_numpy(dtype=np.float32))
        for i in WHEEL_IDS:
            sample[f"rollout_lf_wheel{i}_kin"] = torch.tensor(
                self.df.iloc[target_start_idx + 1:target_start_idx + self.body_rollout_len + 1][self.lf_wheel_kin_cols[i]].to_numpy(dtype=np.float32))
            sample[f"rollout_hf_wheel{i}_kin"] = torch.tensor(
                self.df.iloc[target_start_idx + 1:target_start_idx + self.body_rollout_len + 1][self.spec.target_groups.wheel_kin_cols[i]].to_numpy(dtype=np.float32))
            if self.available_wheel_pose.get(i, False):
                sample[f"rollout_lf_wheel{i}_pose"] = torch.tensor(self.df.iloc[target_start_idx + 1:target_start_idx + self.body_rollout_len + 1][self.lf_wheel_pose_cols[i]].to_numpy(dtype=np.float32))
                sample[f"rollout_hf_wheel{i}_pose"] = torch.tensor(self.df.iloc[target_start_idx + 1:target_start_idx + self.body_rollout_len + 1][self.hf_wheel_pose_cols[i]].to_numpy(dtype=np.float32))
        for name in ROCKER_NAMES:
            if self.available_rocker_pose.get(name, False):
                sample[f"rollout_lf_rocker_{name}_pose"] = torch.tensor(self.df.iloc[target_start_idx + 1:target_start_idx + self.body_rollout_len + 1][self.lf_rocker_pose_cols[name]].to_numpy(dtype=np.float32))
                sample[f"rollout_hf_rocker_{name}_pose"] = torch.tensor(self.df.iloc[target_start_idx + 1:target_start_idx + self.body_rollout_len + 1][self.hf_rocker_pose_cols[name]].to_numpy(dtype=np.float32))
        return sample


def prepare_datasets_and_scaler(
    feature_dir: str,
    merged_csv_path: str,
    seq_len: Optional[int] = None,
    pred_horizon: int = 0,
    pred_seq_len: int = 1,
    case_col: str = "case_name",
    time_col: str = "time",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    history_len: Optional[int] = None,
    teacher_future_len: int = 3,
    body_rollout_len: int = 1,
    force_lpf_alpha: float = 0.15,
    sinkage_max: float = 0.08,
):
    spec = load_column_spec(feature_dir)
    df = load_merged_dataset(merged_csv_path, spec, case_col=case_col, time_col=time_col)
    df_train, df_val, df_test = split_train_val_test_by_case(df, case_col=case_col, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    scaler = GroupStandardizer().fit(df_train, spec)
    train_ds = GraphTemporalSequenceDatasetV4(
        df_train,
        spec,
        scaler,
        seq_len=seq_len,
        pred_horizon=pred_horizon,
        pred_seq_len=pred_seq_len,
        case_col=case_col,
        time_col=time_col,
        history_len=history_len,
        teacher_future_len=teacher_future_len,
        body_rollout_len=body_rollout_len,
        force_lpf_alpha=force_lpf_alpha,
        sinkage_max=sinkage_max,
    )
    val_ds = GraphTemporalSequenceDatasetV4(
        df_val,
        spec,
        scaler,
        seq_len=seq_len,
        pred_horizon=pred_horizon,
        pred_seq_len=pred_seq_len,
        case_col=case_col,
        time_col=time_col,
        history_len=history_len,
        teacher_future_len=teacher_future_len,
        body_rollout_len=body_rollout_len,
        force_lpf_alpha=force_lpf_alpha,
        sinkage_max=sinkage_max,
    ) if len(df_val) else None
    test_ds = GraphTemporalSequenceDatasetV4(
        df_test,
        spec,
        scaler,
        seq_len=seq_len,
        pred_horizon=pred_horizon,
        pred_seq_len=pred_seq_len,
        case_col=case_col,
        time_col=time_col,
        history_len=history_len,
        teacher_future_len=teacher_future_len,
        body_rollout_len=body_rollout_len,
        force_lpf_alpha=force_lpf_alpha,
        sinkage_max=sinkage_max,
    ) if len(df_test) else None
    return spec, scaler, df_train, df_val, df_test, train_ds, val_ds, test_ds
