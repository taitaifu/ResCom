from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

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
    GroupStandardizer as _BaseStandardizer,
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

V6_EXCLUDED_EXACT_INPUT_COLS = {
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
V6_REDUNDANT_ATTITUDE_SUFFIXES = (
    "_roll",
    "_pitch",
    "_sin_yaw",
    "_cos_yaw",
    "_d_roll",
    "_dd_roll",
    "_d_pitch",
    "_dd_pitch",
)
V6_UPRIGHT_RE = re.compile(r"^lf_susp_upright[0-5]_")
V6_WHEEL_ACC_STD_RE = re.compile(r"^lf_wheel\d+_acc_[xyz]_std$")
V6_WHEEL_CONTACT_RATE_RE = re.compile(r"^lf_wheel\d+_(slip_long|slip_lat|sinkage)_rate$")
CONTACT_HISTORY_SUFFIXES = ("in_contact", "sinkage", "slip_long", "slip_lat", "rel_vel_x", "rel_vel_y", "rel_vel_z")


def causal_preprocess(df, group, cols, arr):
    """The V5 force low-pass used a whole-case median dt; V6 uses observed dt."""
    if not (group.startswith('wheel') and group.endswith('_contact')):
        return arr
    ix=[j for j,c in enumerate(cols) if c.endswith(('Fx','Fy','Fz'))]
    if not ix or 'time' not in df or 'case_name' not in df: return arr
    result=arr.copy();rc=1/(2*np.pi*5.)
    for _,rows in df.groupby('case_name',sort=False).indices.items():
        rows=np.asarray(rows)
        times=df.iloc[rows].time.to_numpy(np.float64)
        for k in range(1,len(rows)):
            dt=max(times[k]-times[k-1],0.)
            alpha=dt/(rc+dt)
            result[rows[k],ix]=result[rows[k-1],ix]+alpha*(arr[rows[k],ix]-result[rows[k-1],ix])
    return result


class GroupStandardizer(_BaseStandardizer):
    def fit(self,df,spec):
        for group,cols in flatten_group_columns(spec).items():
            arr=df[cols].to_numpy(np.float32) if cols else np.zeros((len(df),0),np.float32)
            arr=causal_preprocess(df,group,cols,arr)
            scaler=NumpyStandardScaler()
            scaler.fit(arr,apply_mask=self._build_apply_mask(cols))
            self.scalers[group]=scaler
        return self

    def transform_group(self,df,group,cols):
        if not cols: return np.zeros((len(df),0),np.float32)
        arr=causal_preprocess(df,group,cols,df[cols].to_numpy(np.float32))
        return self.scalers[group].transform(arr).astype(np.float32)


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


def is_v6_excluded_input_col(col: str) -> bool:
    if col.startswith(("hf_", "res_")):
        return True
    if col in V6_EXCLUDED_EXACT_INPUT_COLS:
        return True
    if V6_UPRIGHT_RE.match(col):
        return True
    if col.endswith(V6_REDUNDANT_ATTITUDE_SUFFIXES):
        return True
    if V6_WHEEL_ACC_STD_RE.match(col):
        return True
    if V6_WHEEL_CONTACT_RATE_RE.match(col):
        return True
    return False


def filter_v6_input_feature_cols(cols: List[str]) -> List[str]:
    return [col for col in cols if not is_v6_excluded_input_col(col)]


def load_column_spec(feature_dir: str) -> ColumnSpec:
    base_feature_cols = filter_v6_input_feature_cols(load_column_list(f"{feature_dir}/base_feature_columns.csv"))
    proxy_feature_cols = filter_v6_input_feature_cols(load_column_list(f"{feature_dir}/proxy_feature_columns.csv"))
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


def _quat_batch_to_matrix_np(q: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of _quat_to_matrix_np for [...,4] wxyz inputs."""
    q=np.asarray(q,dtype=np.float64)
    q=q/np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-12)
    w,x,y,z=np.moveaxis(q,-1,0)
    rows=(1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
          2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
          2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y))
    return np.stack(rows,axis=-1).reshape(*q.shape[:-1],3,3)


def _pose_from_row_np(row: pd.Series, cols: List[str]) -> tuple[np.ndarray, np.ndarray]:
    p = row[cols[:3]].to_numpy(dtype=np.float64)
    r = _quat_to_matrix_np(row[cols[3:7]].to_numpy(dtype=np.float64))
    return p, r


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
        self.constant_height = float(terrain_z[0])
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
                np.full(z_shape, getattr(self,"constant_height",0.), dtype=np.float32),
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


class GraphTemporalSequenceDatasetV6(_BaseDataset):
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
    ):
        if pred_horizon != 0:
            raise ValueError("V6 target 始终对应当前时刻 t，pred_horizon 必须为 0")
        if pred_seq_len != 1:
            raise ValueError("V6 暂只支持单目标时刻，pred_seq_len 必须为 1")
        if history_len is None:
            history_len = int(seq_len) - 1 if seq_len is not None else 9
        self.history_len = int(history_len)
        self.teacher_future_len = int(teacher_future_len)
        self.body_rollout_len = max(1, int(body_rollout_len))
        if self.history_len < 1:
            raise ValueError(f"history_len 必须 >= 1，实际为 {self.history_len}")
        if self.teacher_future_len < 0:
            raise ValueError(f"teacher_future_len 必须 >= 0，实际为 {self.teacher_future_len}")
        if 'v6_lf_wheel0_omega_valid' not in df:
            df = normalize_optional_columns(df, spec, require_labels=False)
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

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        _, t_global = self.index_map[idx]
        if getattr(self,'compact_training_samples',False):
            # V6 training uses the v6_* windows and assembly metadata, not the
            # duplicated V5 pose/label/rollout fields below. Keep the default
            # sample API for standalone users and acceptance tests.
            if not hasattr(self,'v6_raw'): self._prepare_v6_cache()
            sample={'body':torch.from_numpy(self._slice_student(self.group_arrays['body'],t_global).copy())}
            case_name=self.v6_case_names[t_global]
            for key,value in self.assembly_meta.get(str(case_name),{}).items():
                sample[f'assembly_{key}']=torch.tensor(value,dtype=torch.float32)
            self._add_v6_fields(sample,t_global)
            return sample
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

        for key in ["system", "body"]:
            add_input_group(key)
        for name in ROCKER_NAMES:
            add_input_group(name)
        for i in WHEEL_IDS:
            add_input_group(f"wheel{i}_kin")
            add_input_group(f"wheel{i}_contact")
            contact_cols = self.spec.input_groups.wheel_contact_cols[i]
            contact_array = self.group_arrays[f"wheel{i}_contact"]
            selected = []
            for suffix in CONTACT_HISTORY_SUFFIXES:
                idx_col = next((j for j, col in enumerate(contact_cols) if col.endswith("_" + suffix)), None)
                if idx_col is None:
                    selected.append(np.zeros((self.history_len + 1,), dtype=np.float32))
                else:
                    selected.append(self._slice_student(contact_array[:, idx_col:idx_col + 1], t_global)[:, 0])
            sample[f"force_history_wheel{i}"] = torch.from_numpy(np.stack(selected, axis=-1).astype(np.float32, copy=False))

        body_lf = self.df.iloc[target_start_idx:target_end_idx][self.lf_body_cols].to_numpy(dtype=np.float32)
        sample["lf_body_current"] = torch.tensor(body_lf, dtype=torch.float32)
        if self.available_body_pose:
            sample["lf_body_pose_sequence"] = torch.tensor(
                self.df.iloc[target_start_idx - self.history_len:target_start_idx + 1][self.lf_body_pose_cols].to_numpy(dtype=np.float32),
                dtype=torch.float32,
            )
            sample["lf_body_pose_current"] = torch.tensor(
                self.df.iloc[target_start_idx:target_end_idx][self.lf_body_pose_cols].to_numpy(dtype=np.float32),
                dtype=torch.float32,
            )
            sample["hf_body_pose"] = torch.tensor(
                self.df.iloc[target_start_idx:target_end_idx][self.hf_body_pose_cols].to_numpy(dtype=np.float32),
                dtype=torch.float32,
            )
        for i in WHEEL_IDS:
            sample[f"lf_wheel{i}_kin_raw_sequence"] = torch.tensor(
                self.df.iloc[target_start_idx - self.history_len:target_start_idx + 1][self.lf_wheel_kin_cols[i]].to_numpy(dtype=np.float32),
                dtype=torch.float32,
            )
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

        prev_idx = max(target_start_idx - 1, 0)
        if self.df.iloc[prev_idx][self.case_col] != self.df.iloc[target_start_idx][self.case_col]:
            prev_idx = target_start_idx
        sample["dt"] = torch.tensor(np.float32(self.df.iloc[target_start_idx][self.time_col] - self.df.iloc[prev_idx][self.time_col]))
        sample["lf_body_prev_raw"] = torch.tensor(
            self.df.iloc[prev_idx:prev_idx + 1][self.lf_body_cols].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        prev_prev_idx = max(prev_idx - 1, 0)
        if self.df.iloc[prev_prev_idx][self.case_col] != self.df.iloc[target_start_idx][self.case_col]:
            prev_prev_idx = prev_idx
        sample["state_dt_prev"] = torch.tensor(np.float32(self.df.iloc[prev_idx][self.time_col] - self.df.iloc[prev_prev_idx][self.time_col]))
        sample["lf_body_raw_sequence"] = torch.tensor(
            self.df.iloc[target_start_idx - self.history_len:target_start_idx + 1][self.lf_body_cols].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
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
        self._add_v6_fields(sample, target_start_idx)
        return sample


    def _add_v6_fields(self, sample, t):
        if not hasattr(self,'v6_raw'):
            self._prepare_v6_cache()
        end=t+self.body_rollout_len+1
        for name,arr in self.v6_raw.items():
            sample[f'v6_{name}'] = torch.from_numpy(arr[t:end].copy())
            sample[f'v6_previous_{name}'] = torch.from_numpy(arr[t-1:t].copy())
        keys=['system','body',*ROCKER_NAMES,*[f'wheel{i}_{s}' for i in WHEEL_IDS for s in ('kin','contact')]]
        for role in ('student','teacher'):
            for key in keys:
                slicer=self._slice_teacher if role=='teacher' else self._slice_student
                sample[f'v6_inputs_{role}_{key}']=torch.stack([torch.from_numpy(slicer(self.group_arrays[key],t+k).copy()) for k in range(self.body_rollout_len+1)])
            slicer=self._slice_teacher if role=='teacher' else self._slice_student
            sample[f'{role}_physics_nodes']=torch.from_numpy(slicer(self.v6_physical,t).copy())
            sample[f'v6_inputs_{role}_physics_nodes']=torch.stack([torch.from_numpy(slicer(self.v6_physical,t+k).copy()) for k in range(self.body_rollout_len+1)])
        sample['physics_nodes']=sample['student_physics_nodes']
        times=self.v6_times
        sample['v6_dt']=torch.tensor(np.diff(times[t-1:end]).astype(np.float32))
        for i in WHEEL_IDS:
            for name,arr in ((f'contact_raw_wheel{i}',self.v6_contact[i]),(f'shear_history_wheel{i}',self.v6_shear[i])):
                sample[name]=torch.from_numpy(arr[t:t+1].copy()) if name.startswith('contact_raw') else torch.from_numpy(arr[t-self.history_len:t+1].copy())
                sample[f'v6_inputs_{name}']=torch.stack([torch.from_numpy((arr[t+k:t+k+1] if name.startswith('contact_raw') else arr[t+k-self.history_len:t+k+1]).copy()) for k in range(self.body_rollout_len+1)])
            sample[f'v6_inputs_force_history_wheel{i}']=torch.stack([torch.from_numpy(self._contact_history_scaled(i,t+k)) for k in range(self.body_rollout_len+1)])

    def _contact_history_scaled(self,i,t):
        cols=self.spec.input_groups.wheel_contact_cols[i]
        arr=self.group_arrays[f'wheel{i}_contact']
        out=[]
        for suffix in CONTACT_HISTORY_SUFFIXES:
            j=next((j for j,c in enumerate(cols) if c.endswith('_'+suffix)),None)
            out.append(np.zeros(self.history_len+1,np.float32) if j is None else arr[t-self.history_len:t+1,j])
        return np.stack(out,-1).astype(np.float32)

    def prepare_worker_cache(self):
        """Build once before fork; workers only read the inherited arrays.

        Building this lazily in each worker multiplies the full-dataset cache
        and its temporary arrays by num_workers. Sample slices are copied by
        _add_v6_fields, so read-only caches preserve the fork sharing.
        """
        if not hasattr(self,'v6_raw'): self._prepare_v6_cache()
        arrays=[self.v6_times,self.v6_physical,*self.v6_raw.values(),
                *self.v6_contact.values(),*self.v6_shear.values()]
        for arr in arrays: arr.setflags(write=False)
        # Legacy input windows use from_numpy views before collate makes copies.
        # Leave their flags writable to avoid PyTorch's read-only-array warning;
        # no dataset path mutates them.
        total=sum(arr.nbytes for arr in arrays+list(self.group_arrays.values()) if arr.flags.owndata)
        print(f'V6 parent cache ready: {len(self.df)} rows, owned arrays {total/2**20:.1f} MiB; read-only before fork',flush=True)

    def _prepare_v6_cache(self):
        n=len(self.df); df=self.df
        self.v6_times=df.time.to_numpy(np.float64)
        self.v6_case_names=df[self.case_col].to_numpy(copy=False)
        case_indices=list(df.groupby('case_name',sort=False).indices.values())
        self.v6_raw={}; physical=[]
        cfg=getattr(self,'config',None)
        need_graph=cfg is None or cfg.enable_physics_edge or cfg.enable_physics_attention
        radius=cfg.wheel_radius if cfg else .135
        rho=cfg.shear_rho if cfg else .95
        for part,prefix in [('body',''),*[(f'rocker_{s}',f'susp_rocker_{s}_') for s in ROCKER_NAMES],*[(f'wheel{i}',f'wheel{i}_') for i in WHEEL_IDS]]:
            for source in ('lf','hf'):
                pre=source+'_'+prefix
                cols=[pre+s for s in ('pos_x','pos_y','pos_z','vel_x','vel_y','vel_z','q0','q1','q2','q3','ang_vel_x','ang_vel_y','ang_vel_z','omega')]
                arr=np.stack([_numeric(df,c,1. if c.endswith('_q0') else 0.) for c in cols],-1)
                self.v6_raw[f'{source}_{part}']=np.nan_to_num(arr).astype(np.float32)
                mask=np.stack([np.isfinite(_numeric(df,c,np.nan)) for c in cols],-1)
                for j,c in enumerate(cols):
                    if 'v6_valid_'+c in df:
                        mask[:,j] &= df['v6_valid_'+c].to_numpy(bool)
                angular_marker=f'v6_{pre[:-1]}_angular_valid'
                if angular_marker in df: mask[:,10:13] &= df[angular_marker].to_numpy(bool)[:,None]
                pose_marker=f'v6_{pre[:-1]}_pose_valid'
                if pose_marker in df:
                    mask[:,6:10] &= df[pose_marker].to_numpy(bool)[:,None]
                    if source=='hf': mask[:,:3] &= df[pose_marker].to_numpy(bool)[:,None]
                if part.startswith('wheel'):
                    marker=f'v6_{source}_{part}_omega_valid'
                    mask[:,13]=df[marker].to_numpy(bool)
                else: mask[:,13]=False
                self.v6_raw[f'{source}_{part}_mask']=mask.astype(np.float32)
        if need_graph:
            control=np.zeros((n,32),np.float32); control[:,6:15]=np.eye(3).reshape(1,9)
            physical.append(control)
            for part in ('body',*[f'rocker_{s}' for s in ROCKER_NAMES],*[f'wheel{i}' for i in WHEEL_IDS]):
                raw=self.v6_raw[f'lf_{part}']; row=np.zeros((n,32),np.float32)
                row[:,:6]=raw[:,:6]
                row[:,6:15]=_quat_batch_to_matrix_np(raw[:,6:10]).reshape(n,9)
                row[:,15:18]=raw[:,10:13]
                if part.startswith('rocker_'):
                    body=self.v6_raw['lf_body']
                    body_r=physical[1][:,6:15].reshape(n,3,3)
                    rocker_r=row[:,6:15].reshape(n,3,3)
                    relative=np.swapaxes(body_r,-1,-2)@rocker_r
                    row[:,18]=np.arctan2(relative[:,0,2],relative[:,0,0])
                    row[:,19]=((raw[:,10:13]-body[:,10:13])*body_r[:,:,1]).sum(-1)
                    distance=np.linalg.norm(raw[:,:3]-body[:,:3],axis=-1)
                    row[:,31]=distance
                    for ix in case_indices: row[ix,30]=distance[ix[0]]
                physical.append(row)
        self.v6_contact={};self.v6_shear={}
        terrain=TerrainParameterLookup.get(Path(cfg.terrain_file)) if cfg else TerrainParameterLookup.get()
        for i in WHEEL_IDS:
            raw=self.v6_raw[f'lf_wheel{i}']
            plane,normal,_=terrain.contact_plane(raw[:,0],raw[:,1])
            explicit=[f'lf_wheel{i}_normal_{a}' for a in 'xyz']
            if all(c in df for c in explicit): normal=df[explicit].to_numpy(np.float32)
            normal=normal/np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-8)
            normal=np.where(normal[:,2:3]<0,-normal,normal)
            sink=_numeric(df,f'lf_wheel{i}_sinkage')
            if f'lf_wheel{i}_sinkage' not in df: sink=np.maximum(radius-(normal*(raw[:,:3]-plane)).sum(-1),0)
            contact=_numeric(df,f'lf_wheel{i}_in_contact') if f'lf_wheel{i}_in_contact' in df else ((sink>1e-6)|(_numeric(df,f'lf_wheel{i}_Fz')>1e-3)).astype(np.float32)
            slip=np.stack([_numeric(df,f'lf_wheel{i}_slip_long'),_numeric(df,f'lf_wheel{i}_slip_lat')],-1)
            rel=np.stack([_numeric(df,f'lf_wheel{i}_rel_vel_{a}') for a in 'xyz'],-1)
            steering=_numeric(df,f'lf_wheel{i}_steer_angle')
            c=np.concatenate([contact[:,None],normal,sink[:,None],slip,rel,raw[:,:3],raw[:,13:14],steering[:,None]],-1).astype(np.float32)
            self.v6_contact[i]=c
            forcecols=[f'lf_wheel{i}_{a}' for a in ('Fx','Fy','Fz')]
            self.v6_raw[f'lf_force_wheel{i}']=df[forcecols].to_numpy(np.float32)
            self.v6_raw[f'hf_force_wheel{i}']=np.stack([_numeric(df,f'hf_wheel{i}_{a}') for a in ('Fx','Fy','Fz')],-1)
            self.v6_raw[f'hf_force_wheel{i}_mask']=np.stack([df['v6_valid_'+f'hf_wheel{i}_{a}'].to_numpy(np.float32) for a in ('Fx','Fy','Fz')],-1)
            # No enhanced shear-history work when its module is disabled.
            history_indices=case_indices if cfg is None or cfg.enable_contact_history else []
            history=np.zeros((n,8),np.float32) if history_indices else np.broadcast_to(np.zeros(8,np.float32),(n,8))
            wheel_r=_quat_batch_to_matrix_np(raw[:,6:10]) if history_indices else None
            axle=np.asarray(cfg.spin_axis if cfg else (0,1,0))
            for indices in history_indices:
                j=np.zeros(2,np.float32);duration=0.;previous_slip=np.zeros(2,np.float32)
                previous_time=None
                for index in indices:
                    dt=0. if previous_time is None else max(float(self.v6_times[index])-previous_time,0.)
                    # Relative velocity is world-oriented in feature export; project
                    # onto terrain tangents using the wheel axle, not spinning x.
                    heading=np.cross(wheel_r[index]@axle,normal[index]);heading/=max(np.linalg.norm(heading),1e-8)
                    lateral=np.cross(normal[index],heading)
                    speed=np.array([radius*raw[index,13]-raw[index,3:6]@heading,-raw[index,3:6]@lateral])
                    if contact[index]>.5:
                        j=rho*j+speed*dt;duration+=dt
                    else: j[:]=0;duration=0.
                    history[index]=[*j,*speed,float(np.linalg.norm(speed-previous_slip)),duration,sink[index],raw[index,13]/(cfg.omega_scale if cfg else 10.)]
                    previous_slip=speed;previous_time=float(self.v6_times[index])
            self.v6_shear[i]=history
            if need_graph:
                motion=physical[2+len(ROCKER_NAMES)+i]
                motion[:,18:21]=normal;motion[:,21]=sink;motion[:,22:24]=slip;motion[:,24:27]=rel
                motion[:,27]=raw[:,13];motion[:,28]=steering;motion[:,29]=contact
                motion[:,30:32]=history[:,:2]
                physical.append(motion.copy())
        if need_graph:
            self.v6_physical=np.stack(physical,axis=1)
        else:
            # The disabled graph path never reads these features. Preserve its
            # batch schema without allocating n*20*32 floats or computing edges.
            template=np.zeros((2+len(ROCKER_NAMES)+2*len(WHEEL_IDS),32),np.float32)
            template[:,6:15]=np.eye(3,dtype=np.float32).reshape(1,9)
            self.v6_physical=np.broadcast_to(template,(n,*template.shape))


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
):
    spec = load_column_spec(feature_dir)
    df = load_merged_dataset(merged_csv_path, spec, case_col=case_col, time_col=time_col)
    df_train, df_val, df_test = split_train_val_test_by_case(df, case_col=case_col, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    scaler = GroupStandardizer().fit(df_train, spec)
    train_ds = GraphTemporalSequenceDatasetV6(
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
    )
    val_ds = GraphTemporalSequenceDatasetV6(
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
    ) if len(df_val) else None
    test_ds = GraphTemporalSequenceDatasetV6(
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
    ) if len(df_test) else None
    return spec, scaler, df_train, df_val, df_test, train_ds, val_ds, test_ds


def spec_from_dict(payload):
    """Restore exact checkpoint feature ordering, including integer wheel keys."""
    def out(d):
        return OutputGroupSpec(d['body_cols'],{int(k):v for k,v in d['wheel_kin_cols'].items()},
                               {int(k):v for k,v in d['wheel_contact_cols'].items()})
    ig = dict(payload['input_groups'])
    for key in ('wheel_kin_cols','wheel_contact_cols'): ig[key] = {int(k):v for k,v in ig[key].items()}
    return ColumnSpec(payload['base_feature_cols'],payload['proxy_feature_cols'],payload['res_cols'],
                      payload['target_cols'],InputGroupSpec(**ig),out(payload['res_groups']),out(payload['target_groups']))


def _numeric(df, col, default=0.):
    if col not in df: return np.full(len(df),default,dtype=np.float32)
    return pd.to_numeric(df[col],errors='coerce').to_numpy(dtype=np.float32)


def _merge_columns(df, additions):
    """Replace/add columns in one allocation, without duplicates or insert fragmentation."""
    if not additions: return df
    block=pd.DataFrame(additions,index=df.index)
    return pd.concat([df.drop(columns=[c for c in additions if c in df]),block],axis=1)


def _causal_angular_velocity(df, prefix):
    cols = [f'{prefix}_ang_vel_{a}' for a in 'xyz']
    if all(c in df for c in cols):
        return np.nan_to_num(df[cols].to_numpy(dtype=np.float32)), np.isfinite(df[cols].to_numpy(dtype=np.float32)).all(1)
    qcols = [f'{prefix}_q{i}' for i in range(4)]
    result = np.zeros((len(df),3),np.float32); valid = np.zeros(len(df),bool)
    if not all(c in df for c in qcols): return result,valid
    for _, group in df.groupby('case_name',sort=False).indices.items():
        ix = np.asarray(group); q = df.iloc[ix][qcols].to_numpy(dtype=np.float64)
        t = df.iloc[ix].time.to_numpy(dtype=np.float64)
        for k in range(1,len(ix)):
            if t[k]<=t[k-1] or not np.isfinite(q[k-1:k+1]).all(): continue
            r0,r1 = _quat_to_matrix_np(q[k-1]),_quat_to_matrix_np(q[k])
            relative = r1@r0.T
            skew = np.array([relative[2,1]-relative[1,2],relative[0,2]-relative[2,0],relative[1,0]-relative[0,1]])*.5
            sin = np.linalg.norm(skew); cos = np.clip((np.trace(relative)-1)*.5,-1,1)
            result[ix[k]] = skew*(np.arctan2(sin,cos)/max(sin,1e-7))/(t[k]-t[k-1])
            valid[ix[k]] = True
    return result,valid


def normalize_optional_columns(df, spec, require_labels=True, spin_axis=(0.,1.,0.), wheel_radius=.135):
    """Only LF missing wheel spin may use rolling estimate; never erase valid LF.

    HF angular labels may be derived causally from HF quaternions, as supervision.
    Missing labels are filled solely for array layout and masked from every loss.
    Mandatory LF feature columns still fail later in load_merged_dataset.
    """
    df = df.sort_values(['case_name','time']).reset_index(drop=True).copy()
    observed_columns = set(df.columns)
    # Per-row masks survive dataset slicing and optional inference label fills.
    prefixes = ['lf','hf'] + [f'{p}_susp_rocker_{n}' for p in ('lf','hf') for n in ROCKER_NAMES]
    prefixes += [f'{p}_wheel{i}' for p in ('lf','hf') for i in WHEEL_IDS]
    axis = np.asarray(spin_axis,np.float64); axis /= max(np.linalg.norm(axis),1e-8)
    additions = {}
    for prefix in prefixes:
        values,valid = _causal_angular_velocity(df,prefix)
        for a,v in zip('xyz',values.T):
            col=f'{prefix}_ang_vel_{a}'
            # Preserve valid values and fill invalid readings only after recording
            # validity. Missing spin will use its explicit rolling fallback.
            additions[col]=v
        marker=f'v6_{prefix}_angular_valid'
        if marker not in df: additions[marker]=valid
    df=_merge_columns(df,additions)
    additions = {}
    for i in WHEEL_IDS:
        for p in ('lf','hf'):
            prefix=f'{p}_wheel{i}'; omega=f'{prefix}_omega'
            if omega in df:
                val=_numeric(df,omega); valid=np.isfinite(val)
            else:
                av=df[[f'{prefix}_ang_vel_{a}' for a in 'xyz']].to_numpy(np.float32)
                valid=df[f'v6_{prefix}_angular_valid'].to_numpy(bool)
                qcols=[f'{prefix}_q{j}' for j in range(4)]
                if all(c in df for c in qcols):
                    axle=_quat_batch_to_matrix_np(df[qcols].to_numpy(np.float32))@axis
                    val=(av*axle).sum(-1)
                else: val=av@axis
            marker=f'v6_{p}_wheel{i}_omega_valid'
            if marker in df: valid=df[marker].to_numpy(bool)
            if p=='hf' and marker not in df and omega not in observed_columns:
                # HF quaternion derivatives supervise angular state, but do not
                # turn a missing exported HF spin label into omega supervision.
                exported_angular=all(f'{prefix}_ang_vel_{a}' in observed_columns for a in 'xyz')
                valid &= exported_angular
            if p=='lf':
                # Explicit fallback assumes pure rolling using available LF longitudinal
                # wheel/body velocity; missing mask remains false to identify estimates.
                speed=_numeric(df,f'lf_wheel{i}_vel_x') if f'lf_wheel{i}_vel_x' in df else _numeric(df,'lf_vel_x')
                val=np.where(valid,val,speed/wheel_radius)
            else: val=np.where(valid,val,0.)
            additions[omega]=np.nan_to_num(val)
            additions[marker]=valid
    df=_merge_columns(df,additions)
    additions={}
    for col in spec.target_cols+spec.res_cols:
        marker='v6_valid_'+col
        if marker not in df:
            additions[marker]=np.isfinite(_numeric(df,col,np.nan))
        if col not in df:
            if require_labels and not ('ang_vel' in col or col.endswith('omega')):
                raise KeyError(f'Missing mandatory training label {col}')
            additions[col]=0.
        else: additions[col]=np.nan_to_num(_numeric(df,col))
    df=_merge_columns(df,additions)
    additions={}
    for p in prefixes:
        pose=[f'{p}_{s}' for s in ('pos_x','pos_y','pos_z','q0','q1','q2','q3')]
        marker=f'v6_{p}_pose_valid'
        if marker not in df:
            present=all(c in observed_columns for c in pose)
            additions[marker]=present
        if not require_labels and p.startswith('hf'):
            for c in pose:
                if c not in df: additions[c]=1. if c.endswith('_q0') else 0.
    return _merge_columns(df,additions)


def load_merged_dataset(merged_csv_path, spec, case_col='case_name',time_col='time',require_labels=True,config=None):
    df=pd.read_csv(merged_csv_path)
    if case_col not in df or time_col not in df: raise KeyError('case_name/time required')
    kwargs={} if config is None else dict(spin_axis=config.spin_axis,wheel_radius=config.wheel_radius)
    df=normalize_optional_columns(df,spec,require_labels=require_labels,**kwargs)
    df=ensure_time_feature_columns(df,case_col,time_col)
    check_columns_exist(df,spec.base_feature_cols,'base_feature_cols')
    check_columns_exist(df,spec.proxy_feature_cols,'proxy_feature_cols')
    for col in spec.base_feature_cols+spec.proxy_feature_cols:
        if not np.isfinite(_numeric(df,col,np.nan)).all():
            raise ValueError(f'nonfinite LF input {col}; repair source data rather than fit a leaked/invalid scaler')
    return df


def input_only(batch, role='student'):
    """Enforced allowlist at network boundary: labels/future arrays never forwarded."""
    keys=['system','body',*ROCKER_NAMES,*[f'wheel{i}_{s}' for i in WHEEL_IDS for s in ('kin','contact')], 'physics_nodes']
    result={}
    for key in keys:
        src=f'{role}_{key}' if f'{role}_{key}' in batch else key
        if src in batch: result[key]=batch[src]
    for i in WHEEL_IDS:
        for key in (f'force_history_wheel{i}',f'shear_history_wheel{i}',f'contact_raw_wheel{i}'):
            if key in batch: result[key]=batch[key]
    return result
