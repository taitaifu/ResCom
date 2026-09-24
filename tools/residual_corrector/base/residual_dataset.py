from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import torch
from torch.utils.data import Dataset

from .residual_utils import check_no_hf_leakage, load_npz


FORMAL_FEATURE_BLOCKS = [
    "F_base",
    "F_LF",
    "Fphy",
    "in_contact",
    "sinkage",
    "slip_long",
    "slip_lat",
    "wheel_vel",
    "wheel_acc",
    "wheel_rel_vel",
    "body_acc",
    "body_vel",
    "body_yaw_rate",
    "wheel_omega",
    "wheel_vx_local",
    "wheel_vy_local",
    "slip_angle",
]


class StandardScaler:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        x = np.asarray(x, dtype=np.float64)
        self.mean_ = np.nanmean(x, axis=0)
        self.scale_ = np.nanstd(x, axis=0)
        self.scale_ = np.where(np.isfinite(self.scale_) & (self.scale_ > 1e-12), self.scale_, 1.0)
        self.mean_ = np.where(np.isfinite(self.mean_), self.mean_, 0.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler is not fitted")
        return (np.asarray(x, dtype=np.float64) - self.mean_) / self.scale_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler is not fitted")
        return np.asarray(x, dtype=np.float64) * self.scale_ + self.mean_


def base_feature_names() -> List[str]:
    names: List[str] = []
    for block in FORMAL_FEATURE_BLOCKS:
        if block in {"F_base", "F_LF", "Fphy"}:
            names.extend([f"{block}_{axis}" for axis in ["x", "y", "z"]])
        elif block in {"wheel_vel", "wheel_acc", "wheel_rel_vel", "body_acc", "body_vel"}:
            names.extend([f"{block}_{axis}" for axis in ["x", "y", "z"]])
        else:
            names.append(block)
    return names


def residual_history_feature_names(residual_history_len: int = 10, oracle: bool = False) -> List[str]:
    prefix = "ORACLE_R_true_memory" if oracle else "R_memory"
    return [f"{prefix}_{i:02d}_{axis}" for i in range(residual_history_len, 0, -1) for axis in ["x", "y", "z"]]


def default_feature_names(residual_history_len: int = 10, oracle: bool = False) -> List[str]:
    return base_feature_names() + residual_history_feature_names(residual_history_len, oracle=oracle)


def base_feature_dim() -> int:
    return len(base_feature_names())


def feature_dim(residual_history_len: int = 10) -> int:
    return base_feature_dim() + int(residual_history_len) * 3


@dataclass
class ResidualScalers:
    input_scaler: StandardScaler
    target_scaler: StandardScaler
    feature_names: List[str]

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        joblib.dump(self.input_scaler, out / "input_scaler.joblib")
        joblib.dump(self.target_scaler, out / "target_scaler.joblib")

    @classmethod
    def load(cls, out_dir: str | Path, feature_names: Sequence[str]) -> "ResidualScalers":
        out = Path(out_dir)
        return cls(
            input_scaler=joblib.load(out / "input_scaler.joblib"),
            target_scaler=joblib.load(out / "target_scaler.joblib"),
            feature_names=list(feature_names),
        )


class ResidualWindowDataset(Dataset):
    def __init__(
        self,
        arrays: Mapping[str, np.ndarray],
        history_len: int = 10,
        residual_history_len: int = 10,
        input_scaler: Optional[StandardScaler] = None,
        target_scaler: Optional[StandardScaler] = None,
        oracle_residual_history: bool = False,
        predict_delta_residual: bool = False,
        state_estimator: bool = False,
        fit_scalers: bool = False,
    ) -> None:
        self.arrays = {k: np.asarray(v) for k, v in arrays.items()}
        self.history_len = int(history_len)
        self.residual_history_len = int(residual_history_len)
        self.oracle_residual_history = bool(oracle_residual_history)
        self.predict_delta_residual = bool(predict_delta_residual)
        self.state_estimator = bool(state_estimator)
        self.feature_names = base_feature_names() if self.state_estimator else default_feature_names(self.residual_history_len, oracle=self.oracle_residual_history)
        leaks = check_no_hf_leakage(self.feature_names, allow_oracle=self.oracle_residual_history)
        if leaks:
            raise ValueError("Formal residual corrector input contains forbidden features: " + ", ".join(leaks))
        self.base_raw = self._build_base_raw()
        self.index: List[Tuple[int, int]] = self._build_index()
        if not self.index:
            raise ValueError("No valid residual windows; check case lengths, history_len, and residual_history_len")
        self.predicted_residuals: Optional[np.ndarray] = None
        self.residual_teacher_ratio = 1.0
        self.sample_mode = "teacher"
        self.scheduled_sampling_mode = "progressive"
        self.residual_mean = np.nanmean(self.arrays["R"].astype(np.float64), axis=0).astype(np.float32)
        self.residual_std = np.nanstd(self.arrays["R"].astype(np.float64), axis=0).astype(np.float32)
        self.residual_std = np.where(self.residual_std > 1e-6, self.residual_std, 1.0).astype(np.float32)

        raw_x, raw_y = self._scaler_fit_arrays()
        if fit_scalers:
            self.input_scaler = StandardScaler().fit(raw_x)
            self.target_scaler = StandardScaler().fit(raw_y)
        else:
            if input_scaler is None or target_scaler is None:
                raise ValueError("input_scaler and target_scaler are required when fit_scalers=False")
            self.input_scaler = input_scaler
            self.target_scaler = target_scaler

    def _build_base_raw(self) -> np.ndarray:
        n = len(self.arrays["case_name"])
        def optional(name: str, dim: int) -> np.ndarray:
            value = self.arrays.get(name)
            if value is None:
                print(f"[Residual Dataset] warning: {name} unavailable in residual archive; using zero placeholder for legacy compatibility", flush=True)
                return np.zeros((n, dim), dtype=np.float32)
            value = np.asarray(value, dtype=np.float32)
            value = value[:, None] if dim == 1 and value.ndim == 1 else value
            if value.shape != (n, dim):
                raise ValueError(f"{name} must have shape {(n, dim)}, got {value.shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
            return value
        parts = [
            self.arrays["F_base"],
            self.arrays["F_LF"],
            self.arrays["Fphy"],
            self.arrays["in_contact"][:, None],
            self.arrays["sinkage"][:, None],
            self.arrays["slip_long"][:, None],
            self.arrays["slip_lat"][:, None],
            self.arrays["wheel_vel"],
            self.arrays["wheel_acc"],
            self.arrays["wheel_rel_vel"],
            self.arrays["body_acc"],
            optional("body_vel", 3),
            optional("body_yaw_rate", 1),
            optional("wheel_omega", 1),
            optional("wheel_vx_local", 1),
            optional("wheel_vy_local", 1),
            optional("slip_angle", 1),
        ]
        return np.concatenate(parts, axis=1).astype(np.float32, copy=False)

    def _build_index(self) -> List[Tuple[int, int]]:
        cases = self.arrays["case_name"].astype(str)
        wheels = self.arrays["wheel_id"].astype(np.int64)
        index: List[Tuple[int, int]] = []
        min_end = self.history_len - 1 if self.state_estimator else max(self.history_len - 1, self.residual_history_len)
        for end in range(min_end, len(cases)):
            x_start = end - self.history_len + 1
            r_start = end if self.state_estimator else end - self.residual_history_len
            start = min(x_start, r_start)
            if np.all(cases[start:end + 1] == cases[end]) and np.all(wheels[start:end + 1] == wheels[end]):
                index.append((x_start, end))
        return index

    def set_predicted_residuals(self, predicted: Optional[np.ndarray], teacher_ratio: float = 1.0) -> None:
        self.predicted_residuals = None if predicted is None else np.asarray(predicted, dtype=np.float32)
        self.residual_teacher_ratio = float(np.clip(teacher_ratio, 0.0, 1.0))

    def set_sample_mode(self, mode: str, teacher_ratio: Optional[float] = None) -> None:
        if mode not in {"teacher", "scheduled", "autoregressive"}:
            raise ValueError(f"unknown sample mode: {mode}")
        self.sample_mode = mode
        if teacher_ratio is not None:
            self.residual_teacher_ratio = float(np.clip(teacher_ratio, 0.0, 1.0))

    def set_scheduled_sampling_mode(self, mode: str) -> None:
        if mode not in {"progressive", "random"}:
            raise ValueError(f"unknown scheduled sampling mode: {mode}")
        self.scheduled_sampling_mode = mode

    def residual_history_raw(self, end: int, mode: str = "teacher", rng: Optional[np.random.Generator] = None) -> np.ndarray:
        start = end - self.residual_history_len
        true_hist = self.arrays["R"][start:end].astype(np.float32, copy=True)
        if mode == "teacher" or self.oracle_residual_history:
            return true_hist
        pred = np.zeros_like(true_hist)
        if self.predicted_residuals is not None:
            pred = self.predicted_residuals[start:end].astype(np.float32, copy=True)
        if mode == "autoregressive" or self.residual_teacher_ratio <= 0.0:
            return pred
        if mode == "scheduled":
            if self.scheduled_sampling_mode == "progressive":
                replace_count = int(np.ceil((1.0 - self.residual_teacher_ratio) * self.residual_history_len))
                replace_count = int(np.clip(replace_count, 0, self.residual_history_len))
                out = true_hist.copy()
                if replace_count > 0:
                    out[-replace_count:] = pred[-replace_count:]
                return out.astype(np.float32, copy=False)
            rng = rng or np.random.default_rng()
            use_true = rng.random((self.residual_history_len, 1)) < self.residual_teacher_ratio
            return np.where(use_true, true_hist, pred).astype(np.float32, copy=False)
        raise ValueError(f"unknown residual history mode: {mode}")

    def _combine_raw(self, x_base: np.ndarray, residual_history: np.ndarray) -> np.ndarray:
        if self.state_estimator:
            return x_base.astype(np.float32, copy=False)
        flat_hist = residual_history.reshape(1, -1).repeat(x_base.shape[0], axis=0)
        return np.concatenate([x_base, flat_hist], axis=1).astype(np.float32, copy=False)

    def _scaler_fit_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        xs = []
        ys = []
        for _, end in self.index:
            if self.state_estimator:
                xs.append(self.base_raw[end])
                ys.append(self.arrays["R"][end].astype(np.float32, copy=False))
                continue
            hist = self.residual_history_raw(end, mode="teacher").reshape(-1)
            xs.append(np.concatenate([self.base_raw[end], hist], axis=0))
            ys.append(self.target_raw(end))
        return np.stack(xs, axis=0).astype(np.float32, copy=False), np.stack(ys, axis=0).astype(np.float32, copy=False)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        x_start, end = self.index[idx]
        residual_history = np.zeros((self.residual_history_len, 3), dtype=np.float32) if self.state_estimator else self.residual_history_raw(end, mode=self.sample_mode)
        x = self.input_scaler.transform(self._combine_raw(self.base_raw[x_start:end + 1], residual_history)).astype(np.float32, copy=False)
        y_raw = (self.arrays["R"][end] if self.state_estimator else self.target_raw(end))[None, :]
        y = self.target_scaler.transform(y_raw)[0].astype(np.float32, copy=False)
        prev_r = self.arrays["R"][end - 1] if end > 0 and self.arrays["case_name"][end - 1] == self.arrays["case_name"][end] and self.arrays["wheel_id"][end - 1] == self.arrays["wheel_id"][end] else np.zeros(3, dtype=np.float32)
        prev_y = self.target_scaler.transform(prev_r[None, :])[0].astype(np.float32, copy=False)
        return {
            "x": torch.from_numpy(x),
            "x_base": torch.from_numpy(x[:, :base_feature_dim()].copy()),
            "residual_history": torch.from_numpy((np.zeros((self.residual_history_len, 3), dtype=np.float32) if self.state_estimator else x[0, base_feature_dim():].reshape(self.residual_history_len, 3).copy())),
            "residual_history_raw": torch.from_numpy(residual_history.astype(np.float32, copy=False)),
            "y": torch.from_numpy(y),
            "prev_y": torch.from_numpy(prev_y),
            "y_raw": torch.from_numpy(y_raw[0].astype(np.float32, copy=False)),
            "R_true": torch.from_numpy(self.arrays["R"][end].astype(np.float32, copy=False)),
            "F_base": torch.from_numpy(self.arrays["F_base"][end].astype(np.float32, copy=False)),
            "F_HF": torch.from_numpy(self.arrays["F_HF"][end].astype(np.float32, copy=False)),
            "wheel_id": torch.tensor(int(self.arrays["wheel_id"][end]), dtype=torch.long),
            "time": torch.tensor(float(self.arrays["time"][end]), dtype=torch.float32),
            "case_name": str(self.arrays["case_name"][end]),
            "row_index": torch.tensor(int(end), dtype=torch.long),
        }

    def make_model_input(self, x_start: int, end: int, residual_history: np.ndarray) -> np.ndarray:
        combined = self._combine_raw(self.base_raw[x_start:end + 1], residual_history.astype(np.float32, copy=False))
        return self.input_scaler.transform(combined).astype(np.float32, copy=False)

    def target_scaled(self, end: int) -> np.ndarray:
        return self.target_scaler.transform(self.target_raw(end)[None, :])[0].astype(np.float32, copy=False)

    def target_raw(self, end: int) -> np.ndarray:
        if self.predict_delta_residual:
            return (self.arrays["R"][end] - self.arrays["R"][end - 1]).astype(np.float32, copy=False)
        return self.arrays["R"][end].astype(np.float32, copy=False)

    def clamp_residual_physical(self, residual: np.ndarray) -> np.ndarray:
        lo = self.residual_mean - 5.0 * self.residual_std
        hi = self.residual_mean + 5.0 * self.residual_std
        return np.clip(residual, lo, hi).astype(np.float32, copy=False)

    def inverse_target(self, y_scaled: np.ndarray) -> np.ndarray:
        return self.target_scaler.inverse_transform(y_scaled)


def load_residual_arrays(proof_dir: str | Path, split: str) -> Tuple[Dict[str, np.ndarray], Dict]:
    return load_npz(Path(proof_dir) / f"residual_{split}.npz")


def build_datasets(
    proof_dir: str | Path,
    history_len: Optional[int] = None,
    residual_history_len: int = 10,
    oracle_residual_history: bool = False,
    predict_delta_residual: bool = False,
    state_estimator: bool = False,
) -> Tuple[ResidualWindowDataset, ResidualWindowDataset, ResidualWindowDataset, ResidualScalers, Dict]:
    train_arrays, meta = load_residual_arrays(proof_dir, "train")
    val_arrays, _ = load_residual_arrays(proof_dir, "val")
    test_arrays, _ = load_residual_arrays(proof_dir, "test")
    history_len = 10 if history_len is None else int(history_len)
    train_ds = ResidualWindowDataset(train_arrays, history_len, residual_history_len, oracle_residual_history=oracle_residual_history, predict_delta_residual=predict_delta_residual, state_estimator=state_estimator, fit_scalers=True)
    scalers = ResidualScalers(train_ds.input_scaler, train_ds.target_scaler, train_ds.feature_names)
    val_ds = ResidualWindowDataset(val_arrays, history_len, residual_history_len, scalers.input_scaler, scalers.target_scaler, oracle_residual_history, predict_delta_residual, state_estimator)
    test_ds = ResidualWindowDataset(test_arrays, history_len, residual_history_len, scalers.input_scaler, scalers.target_scaler, oracle_residual_history, predict_delta_residual, state_estimator)
    return train_ds, val_ds, test_ds, scalers, meta


def collate_residual(batch: List[Dict]) -> Dict:
    out: Dict = {}
    for key in ["x", "x_base", "residual_history", "residual_history_raw", "y", "prev_y", "y_raw", "R_true", "F_base", "F_HF", "wheel_id", "time", "row_index"]:
        out[key] = torch.stack([b[key] for b in batch], dim=0)
    out["case_name"] = [b["case_name"] for b in batch]
    return out
