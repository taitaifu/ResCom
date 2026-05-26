from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

WHEEL_IDS = list(range(6))

def load_column_list(csv_path: str) -> List[str]: # 加载列名.CSV文件
    df = pd.read_csv(csv_path)
    if df.shape[1] != 1:
        raise ValueError(f"{csv_path} 应该只有一列，实际为 {df.shape[1]} 列")
    col = df.columns[0]
    vals = [str(x).strip() for x in df[col].tolist() if str(x).strip()]
    return vals


def get_wheel_id(col: str) -> Optional[int]: # 判断是不是车轮列，是则返回车轮ID
    m = re.search(r"wheel([0-5])", col)
    return int(m.group(1)) if m else None


def is_wheel_col(col: str) -> bool: # 判断是不是车轮列，返回的是布尔值
    return get_wheel_id(col) is not None


def has_any_keyword(col: str, keywords: List[str]) -> bool: # 判断列名是否包含任意关键字
    return any(k in col for k in keywords)


def sort_columns(columns: List[str]) -> List[str]: # 对列名进行排序，排序规则是字母顺序
    return sorted(columns)


def check_columns_exist(df: pd.DataFrame, columns: List[str], name: str) -> None: # 检查列是否存在于数据.csv中
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{name} 中有 {len(missing)} 个列在 merged_error_dataset.csv 中不存在，例如: {missing[:10]}")

# 定义分组规则
# 基础输入输出状态
# 车身基础状态
BODY_BASE_KEYWORDS = [
    "_pos_", "_vel_", "_acc_", "_roll", "_sin_yaw", "_pitch", "_cos_yaw",
]
# 车轮基础运动学特征
WHEEL_BASE_KIN_KEYWORDS = [
    "_pos_", "_vel_", "_acc_", "_roll", "_sin_yaw", "_pitch", "_cos_yaw",
]
# 车轮基础接触特征
WHEEL_BASE_CONTACT_KEYWORDS = [
    "_Fx", "_Fy", "_Fz", "_Mx", "_My", "_Mz", "_slip_long", "_sinkage",
]

# 代理特征
# 车身代理特征
BODY_PROXY_KEYWORDS = [
    "_ang_vel_", "_ang_acc_",
    # "_d_", "_dd_" # 其实就是角速度和角加速度
]
# 系统全局特征
SYSTEM_GLOBAL_KEYWORDS = [
    "_cmd_speed", "_sim_dt",
    "_residual_", "_std", "_contact_switch",
]
# 车轮运动学代理特征
WHEEL_KIN_PROXY_KEYWORDS = [
    "_ang_vel_", "_ang_acc_", "_rel_",
    # "_d_", "_dd_" # 其实就是角速度和角加速度
]
# 车轮接触代理特征
WHEEL_CONTACT_PROXY_KEYWORDS = [
    "_slip_lat", "_sinkage", "_in_contact", "_rate",    
]

# 姿态增量输出，表示从 LF 四元数旋转到 HF 四元数的旋转向量
ATTITUDE_RES_KEYWORDS = [
    "_att_x", "_att_y", "_att_z",
]
# 四元数输出
QUAT_KEYWORDS = [
    "_q0", "_q1", "_q2", "_q3",
]

@dataclass
class InputGroupSpec: # 输入特征分组
    system_cols: List[str]
    body_cols: List[str]
    wheel_kin_cols: Dict[int, List[str]]
    wheel_contact_cols: Dict[int, List[str]]

    def to_dict(self) -> Dict[str, Any]: # 把 dataclass 转成普通字典，方便保存为 JSON。
        return asdict(self)

@dataclass
class OutputGroupSpec: # 输出特征分组
    body_cols: List[str]
    wheel_kin_cols: Dict[int, List[str]]
    wheel_contact_cols: Dict[int, List[str]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ColumnSpec: # 总的列配置结构，保存四个原始列名.csv的内容，以及分组后的输入、残差输出和目标高保真输出
    base_feature_cols: List[str]
    proxy_feature_cols: List[str]
    res_cols: List[str]
    target_cols: List[str]
    input_groups: InputGroupSpec
    res_groups: OutputGroupSpec
    target_groups: OutputGroupSpec

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_feature_cols": self.base_feature_cols,
            "proxy_feature_cols": self.proxy_feature_cols,
            "res_cols": self.res_cols,
            "target_cols": self.target_cols,
            "input_groups": self.input_groups.to_dict(),
            "res_groups": self.res_groups.to_dict(),
            "target_groups": self.target_groups.to_dict(),
        }


def is_system_col(col: str) -> bool:
    """
    判断某列是否为系统全局特征。
    系统特征要求：
    1. 不属于 wheel0-wheel5 的逐轮特征
    2. 包含 SYSTEM_GLOBAL_KEYWORDS 中的关键词
    """
    return (not is_wheel_col(col)) and has_any_keyword(col, SYSTEM_GLOBAL_KEYWORDS)

def is_body_col(col: str) -> bool:
    """
    判断某列是否为车身输入特征。
    车身输入特征包含两类：
    1. 车身基础状态特征，例如位置、速度、加速度、四元数
    2. 车身代理特征，例如角速度、角加速度、姿态变化速度、姿态变化加速度
    要求：
    1. 不属于 wheel0-wheel5
    2. 不属于系统全局特征
    3. 包含 BODY_BASE_KEYWORDS 或 BODY_PROXY_KEYWORDS 中的关键词
    """
    if is_wheel_col(col):
        return False
    if is_system_col(col):
        return False
    return (
        has_any_keyword(col, BODY_BASE_KEYWORDS)
        or has_any_keyword(col, BODY_PROXY_KEYWORDS)
    )

def is_wheel_kin_col(col: str) -> bool:
    """
    判断某列是否为车轮运动学输入特征。
    车轮运动学输入特征包含两类：
    1. 车轮基础运动学特征，例如位置、速度、加速度、四元数
    2. 车轮运动学代理特征，例如角速度、角加速度、姿态变化速度、相对车身位置/速度
    要求：
    1. 属于 wheel0-wheel5
    2. 包含 WHEEL_BASE_KIN_KEYWORDS 或 WHEEL_KIN_PROXY_KEYWORDS 中的关键词
    """
    if not is_wheel_col(col):
        return False
    return (
        has_any_keyword(col, WHEEL_BASE_KIN_KEYWORDS)
        or has_any_keyword(col, WHEEL_KIN_PROXY_KEYWORDS)
    )

def is_wheel_contact_col(col: str) -> bool:
    """
    判断某列是否为车轮接触力学输入特征。
    车轮接触力学输入特征包含两类：
    1. 车轮基础接触特征，例如六维力、滑转率、沉陷量
    2. 车轮接触代理特征，例如侧偏滑移、变化率、接触状态、接触切换、高频标准差
    要求：
    1. 属于 wheel0-wheel5
    2. 包含 WHEEL_BASE_CONTACT_KEYWORDS 或 WHEEL_CONTACT_PROXY_KEYWORDS 中的关键词
    """
    if not is_wheel_col(col):
        return False
    return (
        has_any_keyword(col, WHEEL_BASE_CONTACT_KEYWORDS)
        or has_any_keyword(col, WHEEL_CONTACT_PROXY_KEYWORDS)
    )

def is_body_output_col(col: str, prefix: str) -> bool:
    """
    判断某列是否为车身输出目标。
    res_ 输出：
        普通残差 pos/vel/acc
        姿态增量 res_att_x/res_att_y/res_att_z
    hf_ 输出：
        高保真普通状态 pos/vel/acc
        高保真四元数 hf_q0/hf_q1/hf_q2/hf_q3
    """
    if not col.startswith(prefix):
        return False
    if is_wheel_col(col):
        return False

    if has_any_keyword(col, BODY_BASE_KEYWORDS):
        return True

    # 残差输出中加入姿态增量
    if prefix == "res_" and has_any_keyword(col, ATTITUDE_RES_KEYWORDS):
        return True

    # 高保真目标中加入四元数，用于后续从 hf_q* 映射到 lf_q*
    if prefix == "hf_" and has_any_keyword(col, QUAT_KEYWORDS):
        return True

    return False

def is_wheel_output_kin_col(col: str, prefix: str) -> bool:
    """
    判断某列是否为车轮运动学输出目标。
    res_ 输出：
        车轮普通运动学残差 pos/vel/acc
        车轮姿态增量 res_wheel{i}_att_x/y/z
    hf_ 输出：
        车轮高保真普通运动学状态 pos/vel/acc
        车轮高保真四元数 hf_wheel{i}_q0/q1/q2/q3
    """
    if not col.startswith(prefix):
        return False
    if not is_wheel_col(col):
        return False

    if has_any_keyword(col, WHEEL_BASE_KIN_KEYWORDS):
        return True

    # 残差输出中加入车轮姿态增量
    if prefix == "res_" and has_any_keyword(col, ATTITUDE_RES_KEYWORDS):
        return True

    # 高保真目标中加入车轮四元数
    if prefix == "hf_" and has_any_keyword(col, QUAT_KEYWORDS):
        return True

    return False

def is_wheel_output_contact_col(col: str, prefix: str) -> bool:
    """
    判断某列是否为车轮接触力学输出目标。
    输出只使用车轮基础接触关键词。
    例如：
    res_wheel0_Fx, res_wheel0_Fy, res_wheel0_Mz,
    res_wheel0_slip_long, res_wheel0_slip_lat, res_wheel0_sinkage
    """
    if not col.startswith(prefix):
        return False
    if not is_wheel_col(col):
        return False
    return has_any_keyword(col, WHEEL_BASE_CONTACT_KEYWORDS)


def build_input_groups(base_feature_cols: List[str], proxy_feature_cols: List[str]) -> InputGroupSpec:
    all_input_cols = list(base_feature_cols) + list(proxy_feature_cols)
    system_cols = sort_columns([c for c in all_input_cols if is_system_col(c)])
    body_cols = sort_columns([c for c in all_input_cols if is_body_col(c)])

    wheel_kin_cols = {i: [] for i in WHEEL_IDS}
    wheel_contact_cols = {i: [] for i in WHEEL_IDS}

    for c in all_input_cols:
        wid = get_wheel_id(c)
        if wid is None:
            continue
        if is_wheel_kin_col(c):
            wheel_kin_cols[wid].append(c)
        elif is_wheel_contact_col(c):
            wheel_contact_cols[wid].append(c)

    for i in WHEEL_IDS:
        wheel_kin_cols[i] = sort_columns(wheel_kin_cols[i])
        wheel_contact_cols[i] = sort_columns(wheel_contact_cols[i])

    return InputGroupSpec(system_cols, body_cols, wheel_kin_cols, wheel_contact_cols)


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


def load_column_spec(feature_dir: str) -> ColumnSpec: # 读取csv并生成列名配置ColumnSpec
    base_feature_cols = load_column_list(os.path.join(feature_dir, "base_feature_columns.csv"))
    proxy_feature_cols = load_column_list(os.path.join(feature_dir, "proxy_feature_columns.csv"))
    res_cols = load_column_list(os.path.join(feature_dir, "res_columns.csv"))
    target_cols = load_column_list(os.path.join(feature_dir, "target_columns.csv"))

    return ColumnSpec(
        base_feature_cols=base_feature_cols,
        proxy_feature_cols=proxy_feature_cols,
        res_cols=res_cols,
        target_cols=target_cols,
        input_groups=build_input_groups(base_feature_cols, proxy_feature_cols),
        res_groups=build_output_groups(res_cols, "res_"),
        target_groups=build_output_groups(target_cols, "hf_"),
    )


def save_column_spec_json(spec: ColumnSpec, out_path: str) -> None: # 保存列配置和打印分组统计
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(spec.to_dict(), f, ensure_ascii=False, indent=2)


def print_group_summary(spec: ColumnSpec) -> None: # 打印分组统计信息
    ig, rg, tg = spec.input_groups, spec.res_groups, spec.target_groups
    print("\n========== Input Groups ==========")
    print(f"system_cols      : {len(ig.system_cols)}")
    print(f"body_cols        : {len(ig.body_cols)}")
    for i in WHEEL_IDS:
        print(f"wheel{i}_kin     : {len(ig.wheel_kin_cols[i])}")
        print(f"wheel{i}_contact : {len(ig.wheel_contact_cols[i])}")

    print("\n========== Residual Output Groups ==========")
    print(f"body_cols        : {len(rg.body_cols)}")
    for i in WHEEL_IDS:
        print(f"res_wheel{i}_kin     : {len(rg.wheel_kin_cols[i])}")
        print(f"res_wheel{i}_contact : {len(rg.wheel_contact_cols[i])}")

    print("\n========== HF Output Groups ==========")
    print(f"body_cols        : {len(tg.body_cols)}")
    for i in WHEEL_IDS:
        print(f"hf_wheel{i}_kin     : {len(tg.wheel_kin_cols[i])}")
        print(f"hf_wheel{i}_contact : {len(tg.wheel_contact_cols[i])}")

def is_no_standardize_col(col: str) -> bool:
    """
    不进行标准化的列：
    1. 四元数 q0-q3
    2. 接触状态 in_contact
    3. 接触切换 contact_switch
    """
    no_std_keywords = [
        "_q0", "_q1", "_q2", "_q3",
        "q0", "q1", "q2", "q3",
        "_in_contact",
        "_contact_switch",
        "in_contact",
        "contact_switch",
    ]
    return has_any_keyword(col, no_std_keywords)


class NumpyStandardScaler:
    """
    支持部分列不标准化的 StandardScaler。
    对需要标准化的列：
        x_norm = (x - mean) / std
    对不需要标准化的列：
        x_norm = x
    """
    def __init__(self, eps: float = 1e-8):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.apply_mask_: Optional[np.ndarray] = None
        self.eps = eps
    def fit(
        self,
        x: np.ndarray,
        apply_mask: Optional[np.ndarray] = None,
    ) -> "NumpyStandardScaler":
        if x.ndim != 2:
            raise ValueError(f"x 应为二维数组，实际 shape={x.shape}")
        n_dim = x.shape[1]
        if apply_mask is None:
            apply_mask = np.ones((n_dim,), dtype=bool)
        else:
            apply_mask = np.asarray(apply_mask, dtype=bool)
            if apply_mask.shape[0] != n_dim:
                raise ValueError(
                    f"apply_mask 长度应为 {n_dim}，实际为 {apply_mask.shape[0]}"
                )
        self.apply_mask_ = apply_mask
        self.mean_ = np.zeros((n_dim,), dtype=np.float32)
        self.std_ = np.ones((n_dim,), dtype=np.float32)
        if n_dim > 0 and np.any(apply_mask):
            mean = x[:, apply_mask].mean(axis=0)
            std = x[:, apply_mask].std(axis=0)
            std[std < self.eps] = 1.0
            self.mean_[apply_mask] = mean.astype(np.float32)
            self.std_[apply_mask] = std.astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None or self.apply_mask_ is None:
            raise RuntimeError("Scaler 尚未 fit")
        y = x.astype(np.float32).copy()
        if y.shape[1] != self.mean_.shape[0]:
            raise ValueError(
                f"输入维度与 scaler 不一致，输入为 {y.shape[1]}，"
                f"scaler 为 {self.mean_.shape[0]}"
            )
        if np.any(self.apply_mask_):
            y[:, self.apply_mask_] = (
                y[:, self.apply_mask_] - self.mean_[self.apply_mask_]
            ) / self.std_[self.apply_mask_]
        return y

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None or self.apply_mask_ is None:
            raise RuntimeError("Scaler 尚未 fit")
        y = x.astype(np.float32).copy()
        if y.shape[1] != self.mean_.shape[0]:
            raise ValueError(
                f"输入维度与 scaler 不一致，输入为 {y.shape[1]}，"
                f"scaler 为 {self.mean_.shape[0]}"
            )
        if np.any(self.apply_mask_):
            y[:, self.apply_mask_] = (
                y[:, self.apply_mask_] * self.std_[self.apply_mask_]
                + self.mean_[self.apply_mask_]
            )
        return y

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "std": self.std_.tolist() if self.std_ is not None else None,
            "apply_mask": self.apply_mask_.tolist() if self.apply_mask_ is not None else None,
            "eps": self.eps,
        }

    @classmethod
    def from_dict(cls, state: Dict[str, Any]) -> "NumpyStandardScaler":
        obj = cls(state.get("eps", 1e-8))
        if state.get("mean") is not None:
            obj.mean_ = np.asarray(state["mean"], dtype=np.float32)
        if state.get("std") is not None:
            obj.std_ = np.asarray(state["std"], dtype=np.float32)
        if state.get("apply_mask") is not None:
            obj.apply_mask_ = np.asarray(state["apply_mask"], dtype=bool)
        return obj


class GroupStandardizer:
    """
    分组标准化器。
    每个组一个 scaler；
    每个 scaler 内部按列标准化；
    四元数、接触状态、接触切换列不标准化。
    """
    def __init__(self):
        self.scalers: Dict[str, NumpyStandardScaler] = {}
    def _build_apply_mask(self, cols: List[str]) -> np.ndarray:
        """
        True  表示该列需要标准化
        False 表示该列保持原值
        """
        return np.asarray(
            [not is_no_standardize_col(c) for c in cols],
            dtype=bool,
        )
    def fit(self, df: pd.DataFrame, spec: ColumnSpec) -> "GroupStandardizer":
        for name, cols in flatten_group_columns(spec).items():
            if len(cols):
                arr = df[cols].to_numpy(dtype=np.float32)
                apply_mask = self._build_apply_mask(cols)
            else:
                arr = np.zeros((len(df), 0), dtype=np.float32)
                apply_mask = np.zeros((0,), dtype=bool)
            scaler = NumpyStandardScaler()
            scaler.fit(arr, apply_mask=apply_mask)
            self.scalers[name] = scaler
        return self
    def transform_group(
        self,
        df: pd.DataFrame,
        group_name: str,
        cols: List[str],
    ) -> np.ndarray:
        if group_name not in self.scalers:
            raise KeyError(f"未找到 scaler: {group_name}")
        if len(cols) == 0:
            return np.zeros((len(df), 0), dtype=np.float32)

        arr = df[cols].to_numpy(dtype=np.float32)
        return self.scalers[group_name].transform(arr).astype(np.float32)
    def inverse_transform_group(
        self,
        arr: np.ndarray,
        group_name: str,
    ) -> np.ndarray:
        if group_name not in self.scalers:
            raise KeyError(f"未找到 scaler: {group_name}")
        return self.scalers[group_name].inverse_transform(arr)
    def save(self, path: str) -> None:
        joblib.dump(
            {k: v.to_dict() for k, v in self.scalers.items()},
            path,
        )

    @classmethod
    def load(cls, path: str) -> "GroupStandardizer":
        state = joblib.load(path)
        obj = cls()
        obj.scalers = {
            k: NumpyStandardScaler.from_dict(v)
            for k, v in state.items()
        }
        return obj


def flatten_group_columns(spec: ColumnSpec) -> Dict[str, List[str]]: # 把嵌套的列分组结构展开成一个普通字典。
    out: Dict[str, List[str]] = {}
    ig = spec.input_groups
    out["system"] = ig.system_cols
    out["body"] = ig.body_cols
    for i in WHEEL_IDS:
        out[f"wheel{i}_kin"] = ig.wheel_kin_cols[i]
        out[f"wheel{i}_contact"] = ig.wheel_contact_cols[i]

    rg = spec.res_groups
    out["res_body"] = rg.body_cols
    for i in WHEEL_IDS:
        out[f"res_wheel{i}_kin"] = rg.wheel_kin_cols[i]
        out[f"res_wheel{i}_contact"] = rg.wheel_contact_cols[i]

    tg = spec.target_groups
    out["hf_body"] = tg.body_cols
    for i in WHEEL_IDS:
        out[f"hf_wheel{i}_kin"] = tg.wheel_kin_cols[i]
        out[f"hf_wheel{i}_contact"] = tg.wheel_contact_cols[i]
    return out

# 读取数据集.csv
def load_merged_dataset(merged_csv_path: str, spec: ColumnSpec, case_col: str = "case_name", time_col: str = "time") -> pd.DataFrame:
    df = pd.read_csv(merged_csv_path)
    if case_col not in df.columns or time_col not in df.columns: # 检查是否包含工况列和时间列
        raise KeyError(f"merged_error_dataset.csv 必须包含 {case_col} 和 {time_col}")
    # 检查其他必要列
    check_columns_exist(df, spec.base_feature_cols, "base_feature_cols")
    check_columns_exist(df, spec.proxy_feature_cols, "proxy_feature_cols")
    check_columns_exist(df, spec.res_cols, "res_cols")
    check_columns_exist(df, spec.target_cols, "target_cols")
    df = df.sort_values([case_col, time_col]).reset_index(drop=True) # 按工况和时间排序
    return df

def split_train_val_test_by_case( # 按工况划分训练、验证、测试集
    df: pd.DataFrame,
    case_col: str = "case_name",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    case_ids = df[case_col].astype(str).unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(case_ids)
    n = len(case_ids)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio))) if n >= 3 else max(0, n - n_train) # 如果工况数量大于等于 3，至少保留 1 条验证工况。
    n_train = min(n_train, n)
    n_val = min(n_val, max(0, n - n_train))
    train_ids = set(case_ids[:n_train])
    val_ids = set(case_ids[n_train:n_train + n_val])
    test_ids = set(case_ids[n_train + n_val:])
    return (
        df[df[case_col].astype(str).isin(train_ids)].copy(),
        df[df[case_col].astype(str).isin(val_ids)].copy(),
        df[df[case_col].astype(str).isin(test_ids)].copy(),
    )

# 把 DataFrame 中各组列转成 numpy 数组
def extract_group_arrays_from_df(df: pd.DataFrame, spec: ColumnSpec, scaler: Optional[GroupStandardizer]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    groups = flatten_group_columns(spec)
    for name, cols in groups.items():
        if scaler is None: # 如果没有 标准化器 scaler，则直接使用原始数据
            out[name] = df[cols].to_numpy(dtype=np.float32) if len(cols) else np.zeros((len(df), 0), dtype=np.float32)
        else: # 如果有 scaler，则使用 scaler 进行标准化
            out[name] = scaler.transform_group(df, name, cols)
    return out

def map_hf_cols_to_lf(hf_cols: List[str]) -> List[str]:
    """
    将高保真目标列映射为对应的低保真当前状态列。
    用途：
        在 Dataset 中构造 lf_body_current、lf_wheel*_kin_current、
        lf_wheel*_contact_current，用于后续状态重构。
    映射规则：
        hf_pos_x          -> lf_pos_x
        hf_vel_x          -> lf_vel_x
        hf_q0             -> lf_q0
        hf_wheel0_q0      -> lf_wheel0_q0
        hf_wheel0_Fx      -> lf_wheel0_Fx
    注意：
        这个函数只接受 hf_ 开头的列。
        姿态增量 res_att_x/y/z 不在这里处理。
        姿态重构所需的 lf_q0-lf_q3，应通过 hf_q0-hf_q3 映射得到。
    """
    mapped: List[str] = []

    for col in hf_cols:
        if not col.startswith("hf_"):
            raise ValueError(
                f"map_hf_cols_to_lf 只接受 hf_ 开头的列，但收到: {col}"
            )

        lf_col = "lf_" + col[len("hf_"):]
        mapped.append(lf_col)

    return list(dict.fromkeys(mapped))

class GraphTemporalSequenceDataset(Dataset):
    '''
    PyTorch 数据集类。继承 Dataset。
    它负责把一条完整工况切成很多训练样本。
    每个样本包含：
    过去 seq_len 步输入序列
    当前时刻或未来时刻的残差标签
    当前时刻或未来时刻的高保真标签
    当前时刻低保真状态，用于重建监督
    '''
    def __init__(
        self,
        df: pd.DataFrame,
        spec: ColumnSpec,
        scaler: Optional[GroupStandardizer],
        seq_len: int = 20,
        pred_horizon: int = 0, # 预测时间步长，0 表示当前时刻，5 表示用过去窗口预测未来第5步
        case_col: str = "case_name",
        time_col: str = "time",
    ):
        self.df = df.copy()
        self.spec = spec
        self.scaler = scaler
        self.seq_len = int(seq_len)
        self.pred_horizon = int(pred_horizon)
        self.case_col = case_col
        self.time_col = time_col
        self.group_arrays = extract_group_arrays_from_df(self.df, self.spec, self.scaler) # 把 DataFrame 转成各组 numpy 数组，并完成标准化
        self.index_map: List[Tuple[int, int]] = [] # 样本索引表。它保存每个样本对应哪个时间点。
        self._build_index() # 会根据每个工况长度自动生成可用样本

        self.lf_body_cols = map_hf_cols_to_lf(self.spec.target_groups.body_cols)
        self.lf_wheel_kin_cols = {i: map_hf_cols_to_lf(self.spec.target_groups.wheel_kin_cols[i]) for i in WHEEL_IDS}
        self.lf_wheel_contact_cols = {i: map_hf_cols_to_lf(self.spec.target_groups.wheel_contact_cols[i]) for i in WHEEL_IDS}

    def _build_index(self) -> None: # 为每个工况生成滑动窗口样本索引
        start = 0
        for _, sub in self.df.groupby(self.case_col, sort=False):
            n = len(sub)
            for local_t in range(self.seq_len - 1, n - self.pred_horizon):
                self.index_map.append((start, start + local_t)) # start + local_t是每个样本窗口结束点的全局行号
            start += n # 更新起始位置，每个工况在df中的全局位置
            # 每个工况的不同样本存储在index_map中的起始时间步都是start

    def __len__(self) -> int: # 返回当前数据集的样本数量
        return len(self.index_map)

    def _slice_seq(self, arr: np.ndarray, t_global: int) -> np.ndarray: # 根据全局时间戳切片，返回过去 seq_len 步的输入序列
        return arr[t_global - self.seq_len + 1:t_global + 1]

    def __getitem__(self, idx: int) -> Dict[str, Any]: # Dataset 取单个样本
        _, t_global = self.index_map[idx] # 根据样本编号idx找到输入窗口结束位置t_global
        target_idx = t_global + self.pred_horizon # 计算目标时间戳位置
        sample: Dict[str, Any] = {
            "case_name": str(self.df.iloc[target_idx][self.case_col]),
            "time": np.float32(self.df.iloc[target_idx][self.time_col]),
        }

        for key in ["system", "body"]:
            sample[key] = torch.from_numpy(self._slice_seq(self.group_arrays[key], t_global))
        for i in WHEEL_IDS:
            sample[f"wheel{i}_kin"] = torch.from_numpy(self._slice_seq(self.group_arrays[f"wheel{i}_kin"], t_global))
            sample[f"wheel{i}_contact"] = torch.from_numpy(self._slice_seq(self.group_arrays[f"wheel{i}_contact"], t_global))

        # 当前时刻 LF，用于状态重建监督
        body_lf = self.df.iloc[target_idx][self.lf_body_cols].to_numpy(dtype=np.float32)
        sample["lf_body_current"] = torch.tensor(body_lf, dtype=torch.float32)
        for i in WHEEL_IDS:
            sample[f"lf_wheel{i}_kin_current"] = torch.tensor(
                self.df.iloc[target_idx][self.lf_wheel_kin_cols[i]].to_numpy(dtype=np.float32), dtype=torch.float32
            )
            sample[f"lf_wheel{i}_contact_current"] = torch.tensor(
                self.df.iloc[target_idx][self.lf_wheel_contact_cols[i]].to_numpy(dtype=np.float32), dtype=torch.float32
            )

        # 标签
        for key in ["res_body", "hf_body"]:
            sample[key] = torch.tensor(self.group_arrays[key][target_idx], dtype=torch.float32)
        for i in WHEEL_IDS:
            for key in [f"res_wheel{i}_kin", f"res_wheel{i}_contact", f"hf_wheel{i}_kin", f"hf_wheel{i}_contact"]:
                sample[key] = torch.tensor(self.group_arrays[key][target_idx], dtype=torch.float32)

        # =========================================================
        # 前 seq_len - 1 个高保真历史点
        # 用于和当前预测点拼接，计算运动学一致性和平滑性损失
        # =========================================================
        hist_len = self.seq_len - 1
        hist_start = target_idx - hist_len
        hist_end = target_idx

        if hist_start < 0:
            raise RuntimeError(
                f"历史序列长度不足: hist_start={hist_start}, target_idx={target_idx}"
            )

        case_now = self.df.iloc[target_idx][self.case_col]
        case_hist_start = self.df.iloc[hist_start][self.case_col]

        if case_hist_start != case_now:
            raise RuntimeError(
                f"历史高保真序列跨轨迹: hist_start={hist_start}, target_idx={target_idx}, "
                f"case_hist_start={case_hist_start}, case_now={case_now}"
            )

        sample["hf_body_hist"] = torch.tensor(
            self.group_arrays["hf_body"][hist_start:hist_end],
            dtype=torch.float32,
        )

        for i in WHEEL_IDS:
            sample[f"hf_wheel{i}_kin_hist"] = torch.tensor(
                self.group_arrays[f"hf_wheel{i}_kin"][hist_start:hist_end],
                dtype=torch.float32,
            )

        return sample

# Dataset.__getitem__() 返回的多个单样本字典，合并成一个 batch 字典，供模型一次性训练
def graph_temporal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in batch[0].keys():
        v0 = batch[0][k]
        if isinstance(v0, torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        elif isinstance(v0, (float, np.floating)):
            out[k] = torch.tensor([b[k] for b in batch], dtype=torch.float32)
        elif isinstance(v0, (int, np.integer)):
            out[k] = torch.tensor([b[k] for b in batch], dtype=torch.long)
        else:
            out[k] = [b[k] for b in batch]
    return out

# 构建车辆图拓扑
def build_vehicle_graph_edges(include_self_loops: bool = False) -> Tuple[List[Tuple[int, int]], Dict[str, int]]:
    node_map = {"system": 0, "body": 1, **{f"wheel{i}": i + 2 for i in WHEEL_IDS}}  # 定义节点编号，车轮是2-7
    edges: List[Tuple[int, int]] = []
    def add_ud(a: int, b: int): # 添加无向边，图网络一般用有向边表示消息传递，所以无向边会存成两个方向
        edges.append((a, b)); edges.append((b, a))

    S, B = node_map["system"], node_map["body"] # 取出系统节点和车身节点编号
    W = [node_map[f"wheel{i}"] for i in WHEEL_IDS] # 取出所有车轮节点编号
    add_ud(S, B) # 添加系统节点和车身节点之间的边
    for wi in W: # 添加车身节点和车轮节点之间的边
        add_ud(B, wi)
    add_ud(W[0], W[1]); add_ud(W[2], W[3]); add_ud(W[4], W[5]) # 左右相连
    add_ud(W[0], W[2]); add_ud(W[2], W[4]); add_ud(W[1], W[3]); add_ud(W[3], W[5]) # 前中，中后相连
    if include_self_loops: # 如果包含自环，添加每个节点到自己的边
        for i in range(len(node_map)):
            edges.append((i, i))
    return edges, node_map # 返回图的边和节点映射

# 获取每个组的维度
def get_group_dims(spec: ColumnSpec) -> Dict[str, int]:
    dims = {
        "system": len(spec.input_groups.system_cols),
        "body": len(spec.input_groups.body_cols),
        "res_body": len(spec.res_groups.body_cols),
        "hf_body": len(spec.target_groups.body_cols),
    }
    for i in WHEEL_IDS:
        dims[f"wheel{i}_kin"] = len(spec.input_groups.wheel_kin_cols[i])
        dims[f"wheel{i}_contact"] = len(spec.input_groups.wheel_contact_cols[i])
        dims[f"res_wheel{i}_kin"] = len(spec.res_groups.wheel_kin_cols[i])
        dims[f"res_wheel{i}_contact"] = len(spec.res_groups.wheel_contact_cols[i])
        dims[f"hf_wheel{i}_kin"] = len(spec.target_groups.wheel_kin_cols[i])
        dims[f"hf_wheel{i}_contact"] = len(spec.target_groups.wheel_contact_cols[i])
    return dims

# 一站式准备训练数据
def prepare_datasets_and_scaler(
    feature_dir: str,            # 特征目录
    merged_csv_path: str,        # 数据CSV文件路径
    seq_len: int = 20,           # 历史序列长度
    pred_horizon: int = 0,       # 预测步长
    case_col: str = "case_name", # 工况列名
    time_col: str = "time",      # 时间列名
    train_ratio: float = 0.7,    # 训练集比例
    val_ratio: float = 0.15,     # 验证集比例
    seed: int = 42,              # 随机种子
):
    # 读取列名并自动分组
    spec = load_column_spec(feature_dir)
    # 读取数据集并检查列是否存在
    df = load_merged_dataset(merged_csv_path, spec, case_col=case_col, time_col=time_col)
    # 按工况划分训练集、验证集和测试集
    df_train, df_val, df_test = split_train_val_test_by_case(df, case_col=case_col, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    # 只用训练集拟合标准化器
    scaler = GroupStandardizer().fit(df_train, spec)
    # 构建训练集
    train_ds = GraphTemporalSequenceDataset(df_train, spec, scaler, seq_len, pred_horizon, case_col, time_col)
    # 构建验证集
    val_ds = GraphTemporalSequenceDataset(df_val, spec, scaler, seq_len, pred_horizon, case_col, time_col) if len(df_val) else None
    # 构建测试集
    test_ds = GraphTemporalSequenceDataset(df_test, spec, scaler, seq_len, pred_horizon, case_col, time_col) if len(df_test) else None
    return spec, scaler, df_train, df_val, df_test, train_ds, val_ds, test_ds
