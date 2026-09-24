from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "vehicle_config" / "Zhurong_Config.json"
CSV_PATH = ROOT / "Feature_Selection" / "DataSet" / "merged_error_dataset.csv"
OUT_DIR = ROOT / "results_v3" / "assembly_validation"
SAMPLES_PATH = OUT_DIR / "assembly_constraint_samples.csv"
SUMMARY_PATH = OUT_DIR / "assembly_constraint_summary.csv"
RANDOM_SEED = 42
N_CASES = 5
FRAME_STRIDE = 100

SIDES = {
    "left": {
        "front": "lf",
        "rear_main": "lm",
        "sub": "lb",
        "front_wheel": 0,
        "mid_wheel": 2,
        "rear_wheel": 4,
        "front_motor": "front_l",
        "rear_motor": "rear_l",
        "sub_motor": "sub_l",
    },
    "right": {
        "front": "rf",
        "rear_main": "rm",
        "sub": "rb",
        "front_wheel": 1,
        "mid_wheel": 3,
        "rear_wheel": 5,
        "front_motor": "front_r",
        "rear_motor": "rear_r",
        "sub_motor": "sub_r",
    },
}


@dataclass
class PoseMapping:
    name: str
    pos_cols: Optional[List[str]]
    quat_cols: Optional[List[str]]
    quat_order: Optional[str]


@dataclass
class Pose:
    p: np.ndarray
    r: np.ndarray


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def status_from_error(relative_error_percent: float) -> str:
    if relative_error_percent < 0.1:
        return "PASS"
    if relative_error_percent < 1.0:
        return "WARNING"
    return "FAIL"


def status_from_zero_reference(abs_error: float) -> str:
    if abs_error < 0.001:
        return "PASS"
    if abs_error < 0.005:
        return "WARNING"
    return "FAIL"


def quat_to_matrix(q_values: Iterable[float], order: str) -> np.ndarray:
    q = np.asarray(list(q_values), dtype=np.float64)
    q_norm = np.linalg.norm(q)
    if not np.isfinite(q_norm) or q_norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    q = q / q_norm
    if order == "wxyz":
        w, x, y, z = q
    elif order == "xyzw":
        x, y, z, w = q
    else:
        raise ValueError(f"unknown quaternion order: {order}")
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def find_case_col(cols: List[str]) -> str:
    for name in ("case_name", "case", "case_id", "caseid"):
        if name in cols:
            return name
    raise KeyError("未找到 case 列，候选为 case_name/case/case_id/caseid")


def find_time_col(cols: List[str]) -> Optional[str]:
    for name in ("time", "lf_time", "timestamp", "t"):
        if name in cols:
            return name
    return None


def find_frame_col(cols: List[str]) -> Optional[str]:
    for name in ("frame", "frame_id", "step", "index"):
        if name in cols:
            return name
    return None


def find_pose_mapping(name: str, prefix: str, cols: List[str], quat_samples: Optional[pd.DataFrame] = None) -> PoseMapping:
    pos_cols = [f"{prefix}_pos_x", f"{prefix}_pos_y", f"{prefix}_pos_z"]
    if not all(c in cols for c in pos_cols):
        pos_cols = None
    quat_cols = [f"{prefix}_q0", f"{prefix}_q1", f"{prefix}_q2", f"{prefix}_q3"]
    quat_order = None
    if all(c in cols for c in quat_cols):
        quat_order = infer_quat_order(quat_cols, quat_samples)
    else:
        quat_cols = None
    return PoseMapping(name=name, pos_cols=pos_cols, quat_cols=quat_cols, quat_order=quat_order)


def infer_quat_order(quat_cols: List[str], samples: Optional[pd.DataFrame]) -> str:
    names = [c.rsplit("_", 1)[-1] for c in quat_cols]
    if names == ["q0", "q1", "q2", "q3"]:
        return "wxyz"
    return "wxyz"


def mirror_y(point: Iterable[float]) -> List[float]:
    x, y, z = [float(v) for v in point]
    return [x, -y, z]


def enforce_config_symmetry(config: Dict[str, object]) -> Dict[str, object]:
    mirrored = dict(config)
    wheel_rel_pos = dict(config["wheel_rel_pos"])
    steering_upright_pos = dict(config["steering_upright_pos"])
    steer_motor_loc = dict(config["steer_motor_loc"])
    rocker_pos = dict(config["rocker_pos"])
    rocker_motor_loc = dict(config["rocker_motor_loc"])

    for left, right in (("lf", "rf"), ("lm", "rm"), ("lb", "rb")):
        wheel_rel_pos[right] = mirror_y(wheel_rel_pos[left])
        steering_upright_pos[right] = mirror_y(steering_upright_pos[left])
        steer_motor_loc[right] = mirror_y(steer_motor_loc[left])
        rocker_pos[right] = mirror_y(rocker_pos[left])
    for left, right in (("front_l", "front_r"), ("rear_l", "rear_r"), ("sub_l", "sub_r")):
        rocker_motor_loc[right] = mirror_y(rocker_motor_loc[left])

    mirrored["wheel_rel_pos"] = wheel_rel_pos
    mirrored["steering_upright_pos"] = steering_upright_pos
    mirrored["steer_motor_loc"] = steer_motor_loc
    mirrored["rocker_pos"] = rocker_pos
    mirrored["rocker_motor_loc"] = rocker_motor_loc
    return mirrored


def pose_from_row(row: pd.Series, mapping: PoseMapping) -> Optional[Pose]:
    if mapping.pos_cols is None or mapping.quat_cols is None or mapping.quat_order is None:
        return None
    p = row[mapping.pos_cols].to_numpy(dtype=np.float64)
    r = quat_to_matrix(row[mapping.quat_cols].to_numpy(dtype=np.float64), mapping.quat_order)
    return Pose(p=p, r=r)


def transform_body_to_world(body: Pose, point_body: Iterable[float]) -> np.ndarray:
    return body.p + body.r @ np.asarray(point_body, dtype=np.float64)


def world_to_component_local(component: Pose, point_world: np.ndarray) -> np.ndarray:
    return component.r.T @ (point_world - component.p)


def component_local_to_world(component: Pose, point_local: np.ndarray) -> np.ndarray:
    return component.p + component.r @ point_local


def append_length_row(
    rows: List[Dict[str, object]],
    case_name: str,
    frame_value: object,
    time_value: object,
    side: str,
    constraint: str,
    reference_length: float,
    current_length: float,
) -> None:
    abs_error = abs(current_length - reference_length)
    if abs(reference_length) < 1e-12:
        rel = math.nan
        status = status_from_zero_reference(abs_error)
        current_distance_m = current_length
    else:
        rel = abs_error / abs(reference_length) * 100.0
        status = status_from_error(rel)
        current_distance_m = math.nan
    rows.append(
        {
            "case": case_name,
            "frame": frame_value,
            "time": time_value,
            "side": side,
            "constraint": constraint,
            "reference_length": reference_length,
            "current_length": current_length,
            "current_distance_m": current_distance_m,
            "abs_error": abs_error,
            "relative_error_percent": rel,
            "status": status,
        }
    )


def append_distance_constraint(
    rows: List[Dict[str, object]],
    refs: Dict[str, float],
    case_name: str,
    frame_value: object,
    time_value: object,
    side: str,
    name: str,
    p0: np.ndarray,
    p1: np.ndarray,
) -> None:
    key = f"{side}:{name}"
    append_length_row(rows, case_name, frame_value, time_value, side, name, refs[key], norm(p0 - p1))


def fmt_vec(v: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(x):.6f}" for x in v) + "]"


def theory_length(config: Dict[str, object], key0: str, key1: str) -> float:
    return norm(np.asarray(config[key0[0]][key0[1]], dtype=np.float64) - np.asarray(config[key1[0]][key1[1]], dtype=np.float64))


def print_left_frame0_debug(case_name: str, points: Dict[str, np.ndarray], refs: Dict[str, float], config: Dict[str, object]) -> None:
    side = "left"
    keys = [
        ("A_front_world_0", f"{side}:A_front"),
        ("B_front_world_0", f"{side}:B_front"),
        ("C_front_world_0", f"{side}:C_front"),
        ("A_rear_world_0", f"{side}:A_rear"),
        ("D_main_world_0", f"{side}:D_main"),
        ("D_sub_world_0", f"{side}:D_sub"),
        ("E_world_0", f"{side}:E"),
        ("F_world_0", f"{side}:F"),
        ("C_mid_world_0", f"{side}:C_mid"),
        ("C_rear_world_0", f"{side}:C_rear"),
    ]
    print(f"\n[Frame0 Debug] case={case_name} side=left")
    for label, key in keys:
        if key in points:
            print(f"{label}: {fmt_vec(points[key])}")
        else:
            print(f"{label}: MISSING")
    for name in [
        "front_AB",
        "front_BC",
        "front_AC",
        "rear_main_AD",
        "bogie_DE",
        "bogie_DF",
        "middle_upright",
        "rear_upright",
        "middle_rear_wheel",
        "sub_joint_coincidence",
    ]:
        ref_key = f"{side}:{name}"
        if ref_key in refs:
            label = f"{name}_frame0" if name == "sub_joint_coincidence" else f"{name}_ref"
            print(f"{label}: {refs[ref_key]:.9f}")
    theory = {
        "front_AB": theory_length(config, ("rocker_motor_loc", "front_l"), ("steer_motor_loc", "lf")),
        "front_BC": theory_length(config, ("steer_motor_loc", "lf"), ("wheel_rel_pos", "lf")),
        "front_AC": theory_length(config, ("rocker_motor_loc", "front_l"), ("wheel_rel_pos", "lf")),
        "rear_main_AD": theory_length(config, ("rocker_motor_loc", "rear_l"), ("rocker_motor_loc", "sub_l")),
        "bogie_DE": theory_length(config, ("rocker_motor_loc", "sub_l"), ("steer_motor_loc", "lm")),
        "bogie_DF": theory_length(config, ("rocker_motor_loc", "sub_l"), ("steer_motor_loc", "lb")),
        "middle_upright": theory_length(config, ("steer_motor_loc", "lm"), ("wheel_rel_pos", "lm")),
        "rear_upright": theory_length(config, ("steer_motor_loc", "lb"), ("wheel_rel_pos", "lb")),
        "middle_rear_wheel": theory_length(config, ("wheel_rel_pos", "lm"), ("wheel_rel_pos", "lb")),
    }
    print("[Frame0 Debug] JSON theoretical lengths for direct check only")
    for name, value in theory.items():
        print(f"{name}_json_theory: {value:.9f}")


def print_mapping(title: str, mappings: Dict[str, PoseMapping]) -> None:
    print(f"\n[{title}]")
    for key, mapping in mappings.items():
        print(
            f"{key}: pos={mapping.pos_cols or 'MISSING'} "
            f"quat={mapping.quat_cols or 'MISSING'} order={mapping.quat_order or 'MISSING'}"
        )


def build_mappings(cols: List[str], quat_samples: pd.DataFrame) -> Tuple[PoseMapping, Dict[str, PoseMapping], Dict[int, PoseMapping]]:
    body = find_pose_mapping("body", "lf", cols, quat_samples)
    rockers = {
        name: find_pose_mapping(f"rocker_{name}", f"lf_susp_rocker_{name}", cols, quat_samples)
        for name in ("lf", "rf", "lm", "rm", "lb", "rb")
    }
    wheels = {
        i: find_pose_mapping(f"wheel{i}", f"lf_wheel{i}", cols, quat_samples)
        for i in range(6)
    }
    return body, rockers, wheels


def missing_pose_reasons(label: str, mapping: PoseMapping) -> List[str]:
    out = []
    if mapping.pos_cols is None:
        out.append(f"{label} position")
    if mapping.quat_cols is None:
        out.append(f"{label} quaternion")
    return out


def validate_case(
    case_df: pd.DataFrame,
    case_name: str,
    frame_col: Optional[str],
    time_col: Optional[str],
    config: Dict[str, object],
    body_mapping: PoseMapping,
    rocker_mappings: Dict[str, PoseMapping],
    wheel_mappings: Dict[int, PoseMapping],
    calibration_log: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    case_df = case_df.sort_values(time_col if time_col else case_df.index.name or case_df.index).reset_index(drop=True)
    rows: List[Dict[str, object]] = []
    if len(case_df) == 0:
        return rows

    first = case_df.iloc[0]
    body0 = pose_from_row(first, body_mapping)
    if body0 is None:
        print(f"[SKIP] {case_name}: 缺少 LF body position/quaternion，无法进行 chassis-frame 标定")
        return rows

    refs: Dict[str, float] = {}
    local_points: Dict[str, np.ndarray] = {}
    ref_points: Dict[str, np.ndarray] = {}

    def calibrate_component_point(point_key: str, point_body: Iterable[float], component_label: str, component_mapping: PoseMapping) -> bool:
        component0 = pose_from_row(first, component_mapping)
        point_world0 = transform_body_to_world(body0, point_body)
        ref_points[point_key] = point_world0
        if component0 is None:
            calibration_log.append(
                {
                    "case": case_name,
                    "point": point_key,
                    "component": component_label,
                    "status": "SKIPPED",
                    "reason": "; ".join(missing_pose_reasons(component_label, component_mapping)),
                }
            )
            return False
        local_points[point_key] = world_to_component_local(component0, point_world0)
        calibration_log.append(
            {
                "case": case_name,
                "point": point_key,
                "component": component_label,
                "status": "OK",
                "reason": "",
            }
        )
        return True

    rocker_motor_loc = config["rocker_motor_loc"]
    steer_motor_loc = config["steer_motor_loc"]

    for side, spec in SIDES.items():
        front = spec["front"]
        rear_main = spec["rear_main"]
        sub = spec["sub"]
        front_wheel = int(spec["front_wheel"])
        mid_wheel = int(spec["mid_wheel"])
        rear_wheel = int(spec["rear_wheel"])

        a_front0 = transform_body_to_world(body0, rocker_motor_loc[spec["front_motor"]])
        a_rear0 = transform_body_to_world(body0, rocker_motor_loc[spec["rear_motor"]])
        ref_points[f"{side}:A_front"] = a_front0
        ref_points[f"{side}:A_rear"] = a_rear0
        calibration_log.append({"case": case_name, "point": f"{side}:A_front", "component": "body", "status": "OK", "reason": ""})
        calibration_log.append({"case": case_name, "point": f"{side}:A_rear", "component": "body", "status": "OK", "reason": ""})

        calibrate_component_point(f"{side}:B_front", steer_motor_loc[front], f"rocker_{front}", rocker_mappings[front])
        calibrate_component_point(f"{side}:D_main", rocker_motor_loc[spec["sub_motor"]], f"rocker_{rear_main}", rocker_mappings[rear_main])
        calibrate_component_point(f"{side}:D_sub", rocker_motor_loc[spec["sub_motor"]], f"rocker_{sub}", rocker_mappings[sub])
        calibrate_component_point(f"{side}:E", steer_motor_loc[rear_main], f"rocker_{sub}", rocker_mappings[sub])
        calibrate_component_point(f"{side}:F", steer_motor_loc[sub], f"rocker_{sub}", rocker_mappings[sub])

        wheel_front0 = pose_from_row(first, wheel_mappings[front_wheel])
        wheel_mid0 = pose_from_row(first, wheel_mappings[mid_wheel])
        wheel_rear0 = pose_from_row(first, wheel_mappings[rear_wheel])
        if wheel_front0 is not None:
            ref_points[f"{side}:C_front"] = wheel_front0.p
        if wheel_mid0 is not None:
            ref_points[f"{side}:C_mid"] = wheel_mid0.p
        if wheel_rear0 is not None:
            ref_points[f"{side}:C_rear"] = wheel_rear0.p

        pairs = [
            ("front_AB", f"{side}:A_front", f"{side}:B_front"),
            ("front_BC", f"{side}:B_front", f"{side}:C_front"),
            ("front_AC", f"{side}:A_front", f"{side}:C_front"),
            ("rear_main_AD", f"{side}:A_rear", f"{side}:D_main"),
            ("bogie_DE", f"{side}:D_main", f"{side}:E"),
            ("bogie_DF", f"{side}:D_main", f"{side}:F"),
            ("middle_upright", f"{side}:E", f"{side}:C_mid"),
            ("rear_upright", f"{side}:F", f"{side}:C_rear"),
            ("middle_rear_wheel", f"{side}:C_mid", f"{side}:C_rear"),
            ("sub_joint_coincidence", f"{side}:D_main", f"{side}:D_sub"),
        ]
        for name, k0, k1 in pairs:
            if k0 in ref_points and k1 in ref_points:
                refs[f"{side}:{name}"] = norm(ref_points[k0] - ref_points[k1])

    print_left_frame0_debug(case_name, ref_points, refs, config)

    sample_indices = list(range(0, len(case_df), FRAME_STRIDE))
    if 0 not in sample_indices:
        sample_indices.insert(0, 0)

    for idx in sample_indices:
        row = case_df.iloc[idx]
        body = pose_from_row(row, body_mapping)
        if body is None:
            continue
        frame_value = row[frame_col] if frame_col else idx
        time_value = row[time_col] if time_col else ""

        for side, spec in SIDES.items():
            front = spec["front"]
            rear_main = spec["rear_main"]
            sub = spec["sub"]
            front_wheel = int(spec["front_wheel"])
            mid_wheel = int(spec["mid_wheel"])
            rear_wheel = int(spec["rear_wheel"])

            points: Dict[str, np.ndarray] = {
                f"{side}:A_front": transform_body_to_world(body, rocker_motor_loc[spec["front_motor"]]),
                f"{side}:A_rear": transform_body_to_world(body, rocker_motor_loc[spec["rear_motor"]]),
            }
            front_rocker = pose_from_row(row, rocker_mappings[front])
            rear_rocker = pose_from_row(row, rocker_mappings[rear_main])
            sub_rocker = pose_from_row(row, rocker_mappings[sub])
            front_wheel_pose = pose_from_row(row, wheel_mappings[front_wheel])
            mid_wheel_pose = pose_from_row(row, wheel_mappings[mid_wheel])
            rear_wheel_pose = pose_from_row(row, wheel_mappings[rear_wheel])

            if front_rocker is not None and f"{side}:B_front" in local_points:
                points[f"{side}:B_front"] = component_local_to_world(front_rocker, local_points[f"{side}:B_front"])
            if rear_rocker is not None and f"{side}:D_main" in local_points:
                points[f"{side}:D_main"] = component_local_to_world(rear_rocker, local_points[f"{side}:D_main"])
            if sub_rocker is not None:
                for label in ("D_sub", "E", "F"):
                    key = f"{side}:{label}"
                    if key in local_points:
                        points[key] = component_local_to_world(sub_rocker, local_points[key])
            if front_wheel_pose is not None:
                points[f"{side}:C_front"] = front_wheel_pose.p
            if mid_wheel_pose is not None:
                points[f"{side}:C_mid"] = mid_wheel_pose.p
            if rear_wheel_pose is not None:
                points[f"{side}:C_rear"] = rear_wheel_pose.p

            pairs = [
                ("front_AB", f"{side}:A_front", f"{side}:B_front"),
                ("front_BC", f"{side}:B_front", f"{side}:C_front"),
                ("front_AC", f"{side}:A_front", f"{side}:C_front"),
                ("rear_main_AD", f"{side}:A_rear", f"{side}:D_main"),
                ("bogie_DE", f"{side}:D_main", f"{side}:E"),
                ("bogie_DF", f"{side}:D_main", f"{side}:F"),
                ("middle_upright", f"{side}:E", f"{side}:C_mid"),
                ("rear_upright", f"{side}:F", f"{side}:C_rear"),
                ("middle_rear_wheel", f"{side}:C_mid", f"{side}:C_rear"),
                ("sub_joint_coincidence", f"{side}:D_main", f"{side}:D_sub"),
            ]
            for name, k0, k1 in pairs:
                if f"{side}:{name}" in refs and k0 in points and k1 in points:
                    append_distance_constraint(rows, refs, case_name, frame_value, time_value, side, name, points[k0], points[k1])
    return rows


def summarize(samples: pd.DataFrame) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame()
    grouped = samples.groupby(["case", "side", "constraint"], dropna=False)
    summary = grouped.agg(
        reference_length=("reference_length", "first"),
        mean_length=("current_length", "mean"),
        std_length=("current_length", "std"),
        max_abs_error=("abs_error", "max"),
        mean_relative_error_percent=("relative_error_percent", "mean"),
        max_relative_error_percent=("relative_error_percent", "max"),
        mean_distance=("current_distance_m", "mean"),
        std_distance=("current_distance_m", "std"),
        max_distance=("current_distance_m", "max"),
        PASS_count=("status", lambda s: int((s == "PASS").sum())),
        WARNING_count=("status", lambda s: int((s == "WARNING").sum())),
        FAIL_count=("status", lambda s: int((s == "FAIL").sum())),
    ).reset_index()
    return summary


def main() -> None:
    raw_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    required_json = ["wheel_rel_pos", "steering_upright_pos", "steer_motor_loc", "rocker_pos", "rocker_motor_loc"]
    print("[JSON static geometry fields]")
    for key in required_json:
        print(f"{key}: {'OK' if key in raw_config else 'MISSING'}")
    missing_json = [key for key in required_json if key not in raw_config]
    if missing_json:
        raise KeyError(f"JSON 缺少字段: {missing_json}")
    config = enforce_config_symmetry(raw_config)
    print("已按 EnforceConfigSymmetry 复现右侧 MirrorY: rf/rm/rb/front_r/rear_r/sub_r 均由左侧镜像生成。")
    print("本脚本实际用于铰点几何的 JSON 字段: rocker_motor_loc, steer_motor_loc")
    print("wheel_rel_pos / steering_upright_pos / rocker_pos 已读取并确认存在，但不作为第一帧世界坐标。")

    header = pd.read_csv(CSV_PATH, nrows=0)
    cols = list(header.columns)
    case_col = find_case_col(cols)
    time_col = find_time_col(cols)
    frame_col = find_frame_col(cols)
    quat_sample_cols = [c for c in cols if c.startswith("lf") and any(c.endswith(f"_q{i}") for i in range(4))]
    quat_samples = pd.read_csv(CSV_PATH, usecols=[case_col, *(time_col and [time_col] or []), *quat_sample_cols], nrows=500)

    body_mapping, rocker_mappings, wheel_mappings = build_mappings(cols, quat_samples)
    print(f"\n[Case/Time Mapping]\ncase_col={case_col} time_col={time_col or 'MISSING'} frame_col={frame_col or 'MISSING'}")
    print_mapping("LF body mapping", {"body": body_mapping})
    print_mapping("LF rocker mapping", rocker_mappings)
    print_mapping("LF wheel mapping", {str(k): v for k, v in wheel_mappings.items()})

    needed_cols = {case_col}
    if time_col:
        needed_cols.add(time_col)
    if frame_col:
        needed_cols.add(frame_col)
    for mapping in [body_mapping, *rocker_mappings.values(), *wheel_mappings.values()]:
        if mapping.pos_cols:
            needed_cols.update(mapping.pos_cols)
        if mapping.quat_cols:
            needed_cols.update(mapping.quat_cols)
    df = pd.read_csv(CSV_PATH, usecols=sorted(needed_cols))
    sort_cols = [case_col] + ([time_col] if time_col else [])
    df = df.sort_values(sort_cols).reset_index(drop=True)

    cases = sorted(df[case_col].astype(str).unique())
    if len(cases) < N_CASES:
        selected = cases
    else:
        rng = random.Random(RANDOM_SEED)
        selected = sorted(rng.sample(cases, N_CASES))
    print(f"\n[Random cases] seed={RANDOM_SEED} selected={selected}")

    calibration_log: List[Dict[str, object]] = []
    all_rows: List[Dict[str, object]] = []
    for case_name in selected:
        case_df = df[df[case_col].astype(str) == case_name].copy()
        rows = validate_case(
            case_df,
            case_name,
            frame_col,
            time_col,
            config,
            body_mapping,
            rocker_mappings,
            wheel_mappings,
            calibration_log,
        )
        all_rows.extend(rows)

    samples = pd.DataFrame(all_rows)
    summary = summarize(samples)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples.to_csv(SAMPLES_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\n[Calibration]")
    calib = pd.DataFrame(calibration_log)
    if not calib.empty:
        print(calib.groupby(["point", "component", "status"], dropna=False).size().reset_index(name="count").to_string(index=False))
    else:
        print("No calibration points were completed.")

    print("\n[Computed Constraints]")
    if not samples.empty:
        print(samples[["case", "side", "constraint"]].drop_duplicates().to_string(index=False))
    else:
        print("No constraints computed.")

    print("\n[Per-case max relative error percent]")
    if not summary.empty:
        max_rel = summary.pivot_table(index=["case", "side"], columns="constraint", values="max_relative_error_percent", aggfunc="max")
        print(max_rel.to_string())
    else:
        print("No summary.")

    print(f"\nSaved samples: {SAMPLES_PATH}")
    print(f"Saved summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
