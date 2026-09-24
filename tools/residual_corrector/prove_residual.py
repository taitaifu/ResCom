from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from infer.infer_graph_model_v4 import (  # noqa: E402
    build_model_from_checkpoint_v4,
    build_v4_contact_prediction_raw,
    get_arg_with_default,
    resolve_model_role_v4,
    restore_scaler_v4,
    safe_load_checkpoint,
)
from models.data_utils_v4 import (  # noqa: E402
    GraphTemporalSequenceDatasetV4,
    TerrainParameterLookup,
    WHEEL_IDS,
    graph_temporal_collate_fn,
    load_column_spec,
    load_merged_dataset,
    split_train_val_test_by_case,
)
from train.train_graph_model_v4 import (  # noqa: E402
    apply_axis_gate,
    inverse_transform_tensor,
    omega_index,
    reconstruct_body_kinematic,
    reconstruct_wheel_kin_with_body_delta,
    suffix_indices,
)

from tools.residual_corrector.base.residual_analysis import (  # noqa: E402
    run_acf,
    run_ar_predictability,
    run_counterfactual,
    run_ljung_box,
    run_psd,
    run_residual_statistics,
)
from tools.residual_corrector.base.residual_utils import (  # noqa: E402
    DIR_LABELS,
    arrays_to_frame,
    ensure_dir,
    finite_rmse,
    log,
    save_npz,
    timestamp,
    write_json,
)


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out: Dict = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    return out


def _idx(cols: List[str], suffixes: List[str]) -> List[int]:
    return suffix_indices(cols, suffixes)


def _current_np(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().numpy()
    if arr.ndim == 3:
        return arr[:, 0, :]
    return arr


def _exact_indices(cols: List[str], names: List[str]) -> List[int]:
    out: List[int] = []
    for name in names:
        if name in cols:
            out.append(cols.index(name))
    return out


def _zeros(batch_size: int, dim: int = 3) -> np.ndarray:
    return np.zeros((batch_size, dim), dtype=np.float32)


def _tensor_cols_to_np(x: torch.Tensor, indices: List[int], batch_size: int, dim: int = 3) -> np.ndarray:
    if len(indices) != dim:
        return _zeros(batch_size, dim)
    return x[:, indices].detach().cpu().numpy().astype(np.float32, copy=False)


def _tensor_col_to_np(x: torch.Tensor, indices: List[int], batch_size: int) -> np.ndarray:
    if len(indices) != 1:
        return np.zeros((batch_size,), dtype=np.float32)
    return x[:, indices[0]].detach().cpu().numpy().astype(np.float32, copy=False)


def _sort_archive_arrays(parts: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    """Concatenate and time-sort every archive field, including optional dynamics."""
    arrays = {key: np.concatenate(values, axis=0) if values else np.empty((0,), dtype=np.float32) for key, values in parts.items()}
    order = np.lexsort((arrays["time"], arrays["wheel_id"], arrays["case_name"].astype(str)))
    out = {key: value[order] for key, value in arrays.items()}
    out["case_name"] = out["case_name"].astype(str)
    out["time"] = out["time"].astype(np.float64, copy=False)
    out["wheel_id"] = out["wheel_id"].astype(np.int64, copy=False)
    for key, value in out.items():
        if key not in {"case_name", "time", "wheel_id"}:
            out[key] = value.astype(np.float32, copy=False)
    return out


def _normalize_rows(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(norm > 1e-12, v / np.maximum(norm, 1e-12), fallback.reshape(1, 3)).astype(np.float32)


def _quat_rotate_world(q: np.ndarray, v_local: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    v = np.asarray(v_local, dtype=np.float32)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * np.cross(xyz, v)
    return (v + w * t + np.cross(xyz, t)).astype(np.float32)


def _contact_force_basis(body_quat: np.ndarray, contact_normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Python equivalent of TerramechanicsRigid::BuildContactFrameFromNormalAndHeading."""
    ez = _normalize_rows(np.asarray(contact_normal, dtype=np.float32), np.asarray([0., 0., 1.], dtype=np.float32))
    heading = _quat_rotate_world(body_quat, np.tile(np.asarray([[1., 0., 0.]], dtype=np.float32), (len(ez), 1)))
    ex = heading - np.sum(heading * ez, axis=1, keepdims=True) * ez
    fallback = np.tile(np.asarray([[1., 0., 0.]], dtype=np.float32), (len(ez), 1))
    near_vertical = np.abs(np.sum(fallback * ez, axis=1)) > .9
    fallback[near_vertical] = np.asarray([0., 1., 0.], dtype=np.float32)
    fallback = fallback - np.sum(fallback * ez, axis=1, keepdims=True) * ez
    ex = np.where(np.linalg.norm(ex, axis=1, keepdims=True) > 1e-12, ex, fallback)
    ex = _normalize_rows(ex, np.asarray([1., 0., 0.], dtype=np.float32))
    ey = _normalize_rows(np.cross(ez, ex), np.asarray([0., 1., 0.], dtype=np.float32))
    ex = _normalize_rows(np.cross(ey, ez), np.asarray([1., 0., 0.], dtype=np.float32))
    return ex, ey, ez


def _contact_normal_from_terrain(wheel_pos: np.ndarray, body_quat: np.ndarray, terrain: TerrainParameterLookup, radius: float = .135, width: float = .16) -> np.ndarray:
    """Replicates TerramechanicsRigid::CalculateTouchArea's three-point normal."""
    ex0 = _quat_rotate_world(body_quat, np.tile(np.asarray([[1., 0., 0.]], dtype=np.float32), (len(wheel_pos), 1)))
    ey0 = _quat_rotate_world(body_quat, np.tile(np.asarray([[0., 1., 0.]], dtype=np.float32), (len(wheel_pos), 1)))
    ez0 = _quat_rotate_world(body_quat, np.tile(np.asarray([[0., 0., 1.]], dtype=np.float32), (len(wheel_pos), 1)))
    theta = .01
    pa = wheel_pos + ex0 * (radius * np.sin(theta)) + ey0 * (width * .5) - ez0 * (radius * np.cos(theta))
    pb = wheel_pos + ex0 * (radius * np.sin(theta)) - ey0 * (width * .5) - ez0 * (radius * np.cos(theta))
    pc = wheel_pos - ex0 * (radius * np.sin(theta)) - ez0 * (radius * np.cos(theta))
    za, _ = terrain.sample_point(pa[:, 0], pa[:, 1]); zb, _ = terrain.sample_point(pb[:, 0], pb[:, 1]); zc, _ = terrain.sample_point(pc[:, 0], pc[:, 1])
    pa[:, 2], pb[:, 2], pc[:, 2] = za, zb, zc
    normal = np.cross(pb - pc, pa - pc)
    normal[normal[:, 2] < 0.] *= -1.
    return _normalize_rows(normal, np.asarray([0., 0., 1.], dtype=np.float32))


def _project_contact_velocity(v_world: np.ndarray, ex: np.ndarray, ey: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (np.sum(v_world * ex, axis=1).astype(np.float32), np.sum(v_world * ey, axis=1).astype(np.float32))


def infer_split(
    split_name: str,
    df: pd.DataFrame,
    spec,
    scaler,
    model,
    model_role: str,
    history_len: int,
    teacher_future_len: int,
    pred_horizon: int,
    sinkage_max: float,
    kinematic_pos_gate_scale: float,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    max_batches: int | None = None,
) -> Tuple[Dict[str, np.ndarray], Dict]:
    dataset = GraphTemporalSequenceDatasetV4(
        df,
        spec,
        scaler,
        seq_len=history_len + 1,
        pred_horizon=pred_horizon,
        pred_seq_len=1,
        case_col="case_name",
        time_col="time",
        history_len=history_len,
        teacher_future_len=teacher_future_len,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=graph_temporal_collate_fn)
    parts: Dict[str, List[np.ndarray]] = {
        "case_name": [],
        "time": [],
        "wheel_id": [],
        "F_base": [],
        "F_HF": [],
        "F_LF": [],
        "Fphy": [],
        "R": [],
        "in_contact": [],
        "sinkage": [],
        "slip_long": [],
        "slip_lat": [],
        "wheel_vel": [],
        "wheel_acc": [],
        "wheel_rel_vel": [],
        "body_acc": [],
    }
    prefix = "teacher" if model_role == "teacher" else "student"
    body_cols = spec.input_groups.body_cols
    body_acc_idx_raw = _exact_indices(body_cols, ["lf_acc_x", "lf_acc_y", "lf_acc_z"])
    body_vel_idx_raw = _exact_indices(body_cols, ["lf_vel_x", "lf_vel_y", "lf_vel_z"])
    body_yaw_rate_idx_raw = _exact_indices(body_cols, ["lf_ang_vel_z"])
    wheel_omega_idx = {wid: omega_index(spec.input_groups.wheel_kin_cols[wid]) for wid in WHEEL_IDS}
    wheel_pos_idx = {wid: _exact_indices(spec.input_groups.wheel_kin_cols[wid], [f"lf_wheel{wid}_pos_x", f"lf_wheel{wid}_pos_y", f"lf_wheel{wid}_pos_z"]) for wid in WHEEL_IDS}
    local_kinematics_available = bool(dataset.available_body_pose and all(len(indices) == 3 for indices in wheel_pos_idx.values()))
    terrain_lookup = TerrainParameterLookup.get()
    unavailable_dynamics = []
    if len(body_vel_idx_raw) == 3:
        parts["body_vel"] = []
    else:
        unavailable_dynamics.append("body_vel")
    if len(body_yaw_rate_idx_raw) == 1:
        parts["body_yaw_rate"] = []
    else:
        unavailable_dynamics.append("body_yaw_rate")
    if all(index is not None for index in wheel_omega_idx.values()):
        parts["wheel_omega"] = []
    else:
        unavailable_dynamics.append("wheel_omega")
    if local_kinematics_available:
        parts["wheel_vx_local"] = []
        parts["wheel_vy_local"] = []
        parts["slip_angle"] = []
    else:
        unavailable_dynamics.extend(["wheel_vx_local", "wheel_vy_local", "slip_angle"])
    for name in unavailable_dynamics:
        log("PROVE", f"warning: {name} is not available from the current LF input schema; it will not be written to the residual archive")
    for batch_i, batch in enumerate(tqdm(loader, desc=f"PROVE {split_name}", dynamic_ncols=True)):
        batch = move_batch_to_device(batch, device)
        with torch.no_grad():
            output = model(batch, input_prefix=prefix, current_index=history_len)
            dt_batch = batch.get("dt", output["pred_res_body"].new_full((output["pred_res_body"].shape[0],), 0.015))
            prev_body_raw = batch.get("lf_body_prev_raw", batch["lf_body_current"])
            _, pred_body_raw, _ = reconstruct_body_kinematic(
                output["pred_res_body"],
                batch["lf_body_current"],
                prev_body_raw,
                dt_batch,
                spec.res_groups.body_cols,
                spec.target_groups.body_cols,
                scaler,
            )
            body_pos_idx = _idx(spec.target_groups.body_cols, ["pos_x", "pos_y", "pos_z"])
            if len(body_pos_idx) == 3:
                body_gate = output["gate_body_pos"] * max(0.0, min(1.0, kinematic_pos_gate_scale))
                pred_body_raw = apply_axis_gate(pred_body_raw, batch["lf_body_current"], body_pos_idx, body_gate)
            lf_body_pos = batch["lf_body_current"][..., body_pos_idx].float() if len(body_pos_idx) == 3 else pred_body_raw.new_zeros(*pred_body_raw.shape[:2], 3)
            body_global_delta = pred_body_raw[..., body_pos_idx] - lf_body_pos if len(body_pos_idx) == 3 else pred_body_raw.new_zeros(*pred_body_raw.shape[:2], 3)
            pred_wheel_raw: Dict[int, torch.Tensor] = {}
            for wid in WHEEL_IDS:
                _, raw, _ = reconstruct_wheel_kin_with_body_delta(
                    output[f"pred_res_wheel{wid}_kin"],
                    batch[f"lf_wheel{wid}_kin_current"],
                    body_global_delta,
                    spec.res_groups.wheel_kin_cols[wid],
                    spec.target_groups.wheel_kin_cols[wid],
                    f"hf_wheel{wid}_kin",
                    f"res_wheel{wid}_kin",
                    scaler,
                )
                wheel_cols = spec.target_groups.wheel_kin_cols[wid]
                wpos = _idx(wheel_cols, ["pos_x", "pos_y", "pos_z"])
                omega_i = omega_index(wheel_cols)
                if len(wpos) == 3:
                    raw = apply_axis_gate(raw, batch[f"lf_wheel{wid}_kin_current"], wpos, output[f"gate_wheel{wid}_pos"] * max(0.0, min(1.0, kinematic_pos_gate_scale)))
                if omega_i is not None:
                    raw = apply_axis_gate(raw, batch[f"lf_wheel{wid}_kin_current"], [omega_i], output[f"gate_wheel{wid}_omega"])
                pred_wheel_raw[wid] = raw

            batch_n = len(batch["case_name"])
            case_names = list(batch["case_name"])
            times = batch["time"].detach().cpu().numpy().astype(np.float64)
            body_seq_raw = inverse_transform_tensor(batch["body"], scaler, "body")
            body_current = body_seq_raw[:, -1, :]
            body_acc_current = _tensor_cols_to_np(body_current, body_acc_idx_raw, batch_n, dim=3)
            body_vel_current = _tensor_cols_to_np(body_current, body_vel_idx_raw, batch_n, dim=3) if "body_vel" in parts else None
            body_yaw_rate_current = _tensor_col_to_np(body_current, body_yaw_rate_idx_raw, batch_n) if "body_yaw_rate" in parts else None
            body_quat_current = batch["lf_body_pose_current"][:, 0, 3:7].detach().cpu().numpy().astype(np.float32, copy=False) if local_kinematics_available else None

            for wid in WHEEL_IDS:
                contact_cols = spec.target_groups.wheel_contact_cols[wid]
                force_idx = _idx(contact_cols, ["Fx", "Fy", "Fz"])
                hf_raw = inverse_transform_tensor(batch[f"hf_wheel{wid}_contact"], scaler, f"hf_wheel{wid}_contact")
                f_hf = _current_np(hf_raw[..., force_idx])
                f_base = _current_np(build_v4_contact_prediction_raw(batch, output, spec, scaler, pred_body_raw, pred_wheel_raw, wid, sinkage_max))
                f_lf = _current_np(batch[f"wheel{wid}_lf_force_current"])
                fphy = _current_np(batch[f"wheel{wid}_fphy_current"])

                contact_seq_raw = inverse_transform_tensor(batch[f"wheel{wid}_contact"], scaler, f"wheel{wid}_contact")
                kin_seq_raw = inverse_transform_tensor(batch[f"wheel{wid}_kin"], scaler, f"wheel{wid}_kin")
                contact_current = contact_seq_raw[:, -1, :]
                kin_current = kin_seq_raw[:, -1, :]
                lf_contact_cols = spec.input_groups.wheel_contact_cols[wid]
                cidx = {name: _exact_indices(lf_contact_cols, [f"lf_wheel{wid}_{name}"]) for name in ["in_contact", "sinkage", "slip_long", "slip_lat"]}
                kin_cols = spec.input_groups.wheel_kin_cols[wid]
                vel_idx = _exact_indices(kin_cols, [f"lf_wheel{wid}_vel_x", f"lf_wheel{wid}_vel_y", f"lf_wheel{wid}_vel_z"])
                acc_idx = _exact_indices(kin_cols, [f"lf_wheel{wid}_acc_x", f"lf_wheel{wid}_acc_y", f"lf_wheel{wid}_acc_z"])
                rel_idx = _exact_indices(kin_cols, [f"lf_wheel{wid}_rel_vel_x", f"lf_wheel{wid}_rel_vel_y", f"lf_wheel{wid}_rel_vel_z"])
                wheel_vel = _tensor_cols_to_np(kin_current, vel_idx, batch_n, dim=3)
                wheel_acc = _tensor_cols_to_np(kin_current, acc_idx, batch_n, dim=3)
                wheel_rel = _tensor_cols_to_np(kin_current, rel_idx, batch_n, dim=3)
                wheel_omega = kin_current[:, wheel_omega_idx[wid]].detach().cpu().numpy().astype(np.float32, copy=False) if "wheel_omega" in parts else None
                if local_kinematics_available:
                    wheel_pos = _tensor_cols_to_np(kin_current, wheel_pos_idx[wid], batch_n, dim=3)
                    normal = _contact_normal_from_terrain(wheel_pos.copy(), body_quat_current, terrain_lookup)
                    basis_ex, basis_ey, basis_ez = _contact_force_basis(body_quat_current, normal)
                    orthogonality = np.maximum.reduce([np.abs(np.sum(basis_ex * basis_ey, axis=1)), np.abs(np.sum(basis_ex * basis_ez, axis=1)), np.abs(np.sum(basis_ey * basis_ez, axis=1))])
                    if not np.isfinite(orthogonality).all() or float(np.max(orthogonality)) > 1e-4:
                        raise RuntimeError(f"wheel {wid} force basis is not orthonormal; max_error={float(np.nanmax(orthogonality)):.3g}")
                    wheel_vx_local, wheel_vy_local = _project_contact_velocity(wheel_rel, basis_ex, basis_ey)
                    slip_angle = np.arctan2(wheel_vy_local, np.abs(wheel_vx_local) + 1e-6).astype(np.float32, copy=False)
                else:
                    wheel_vx_local = wheel_vy_local = slip_angle = None
                r = f_base - f_hf
                parts["case_name"].append(np.asarray(case_names, dtype=str))
                parts["time"].append(times)
                parts["wheel_id"].append(np.full((batch_n,), wid, dtype=np.int64))
                parts["F_base"].append(f_base.astype(np.float32, copy=False))
                parts["F_HF"].append(f_hf.astype(np.float32, copy=False))
                parts["F_LF"].append(f_lf.astype(np.float32, copy=False))
                parts["Fphy"].append(fphy.astype(np.float32, copy=False))
                parts["R"].append(r.astype(np.float32, copy=False))
                parts["in_contact"].append(_tensor_col_to_np(contact_current, cidx["in_contact"], batch_n))
                parts["sinkage"].append(_tensor_col_to_np(contact_current, cidx["sinkage"], batch_n))
                parts["slip_long"].append(_tensor_col_to_np(contact_current, cidx["slip_long"], batch_n))
                parts["slip_lat"].append(_tensor_col_to_np(contact_current, cidx["slip_lat"], batch_n))
                parts["wheel_vel"].append(wheel_vel)
                parts["wheel_acc"].append(wheel_acc)
                parts["wheel_rel_vel"].append(wheel_rel)
                parts["body_acc"].append(body_acc_current)
                if body_vel_current is not None:
                    parts["body_vel"].append(body_vel_current)
                if body_yaw_rate_current is not None:
                    parts["body_yaw_rate"].append(body_yaw_rate_current)
                if wheel_omega is not None:
                    parts["wheel_omega"].append(wheel_omega)
                if wheel_vx_local is not None:
                    parts["wheel_vx_local"].append(wheel_vx_local)
                    parts["wheel_vy_local"].append(wheel_vy_local)
                    parts["slip_angle"].append(slip_angle)
        if max_batches is not None and batch_i + 1 >= max_batches:
            break

    arrays = _sort_archive_arrays(parts)
    if arrays["F_base"].shape != arrays["F_HF"].shape or arrays["F_base"].shape[1:] != (3,):
        raise RuntimeError("F_base/F_HF must be time-aligned 3D force vectors in the same archive frame")
    if not np.isfinite(arrays["F_base"]).all() or not np.isfinite(arrays["F_HF"]).all():
        raise RuntimeError("F_base/F_HF contain non-finite force values")
    rng = np.random.default_rng(42)
    chosen = rng.choice(len(arrays["time"]), size=min(6, len(arrays["time"])), replace=False) if len(arrays["time"]) else []
    sample_rows = [{"wheel_id": int(arrays["wheel_id"][row]), "case_name": str(arrays["case_name"][row]), "time": float(arrays["time"][row]), "steering_angle": "unavailable", "local_basis": "unavailable: contact/chassis basis is not emitted by V4 archive", "F_base_xyz": arrays["F_base"][row].tolist(), "F_HF_xyz": arrays["F_HF"][row].tolist(), "R_true_xyz": (arrays["F_HF"][row] - arrays["F_base"][row]).tolist()} for row in chosen]
    for sample in sample_rows:
        log("FRAME", f"split={split_name} wheel={sample['wheel_id']} steering={sample['steering_angle']} basis={sample['local_basis']} F_base={sample['F_base_xyz']} F_HF={sample['F_HF_xyz']} R_true={sample['R_true_xyz']}")
    meta = {"split": split_name, "num_rows": int(len(arrays["time"])), "num_samples": int(len(dataset)), "available_dynamics": [name for name in ("body_vel", "body_yaw_rate", "wheel_omega", "wheel_vx_local", "wheel_vy_local", "slip_angle") if name in arrays], "unavailable_dynamics": unavailable_dynamics, "frame_audit": {"rel_vel_frame": "world (wheel world velocity - body world velocity; no rotation in its generator)", "force_frame": "TerramechanicsRigid contact frame", "force_basis_definition": "ez=contact normal; ex=projection of chassis rolling axis onto contact plane; ey=ez cross ex; ex=ey cross ez", "wheel_quaternion_transform_applied": False, "contact_basis_available": bool(local_kinematics_available), "contact_false_count": int(np.sum(arrays["in_contact"] < 0.5)), "sample_rows": sample_rows}}
    return arrays, meta


def residual_frame(arrays: Dict[str, np.ndarray]) -> pd.DataFrame:
    return arrays_to_frame(arrays)


def summarize_force(arrays: Dict[str, np.ndarray]) -> Dict[str, float]:
    out = {"overall_rmse": finite_rmse(arrays["F_base"], arrays["F_HF"])}
    for j, label in enumerate(DIR_LABELS):
        out[f"{label}_rmse"] = finite_rmse(arrays["F_base"][:, j], arrays["F_HF"][:, j])
    for wid in WHEEL_IDS:
        mask = arrays["wheel_id"] == wid
        out[f"wheel{wid}_rmse"] = finite_rmse(arrays["F_base"][mask], arrays["F_HF"][mask])
    return out


def classify_structure(acf_df: pd.DataFrame, lb_df: pd.DataFrame, ar_df: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
    fz_acf = acf_df[(acf_df["direction"] == "Fz") & (acf_df["subset"] == "all") & (acf_df["lag"].isin([1, 2, 3]))]["acf_mean"].abs().mean()
    fz_lb = lb_df[(lb_df["direction"] == "Fz") & (lb_df["lag"] == 10)]["reject_ratio_p005"].mean()
    fz_ar = ar_df[(ar_df["direction"] == "Fz") & (ar_df["wheel_id"].astype(str) == "overall") & (ar_df["split"] == "val")]["relative_improvement"].mean()
    score = 0
    score += 1 if np.isfinite(fz_acf) and fz_acf > 0.20 else 0
    score += 1 if np.isfinite(fz_lb) and fz_lb > 0.50 else 0
    score += 1 if np.isfinite(fz_ar) and fz_ar > 0.05 else 0
    if score >= 3 or (np.isfinite(fz_ar) and fz_ar > 0.15 and np.isfinite(fz_acf) and fz_acf > 0.15):
        label = "strong"
    elif score >= 2 or (np.isfinite(fz_ar) and fz_ar > 0.02):
        label = "moderate"
    else:
        label = "weak"
    return label, {"fz_acf_lag123_abs_mean": float(fz_acf), "fz_ljung_reject_ratio_lag10": float(fz_lb), "fz_ar_val_relative_improvement": float(fz_ar)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--feature_dir", type=str, default=str(ROOT / "Feature_Selection" / "DataSet"))
    parser.add_argument("--merged_csv", type=str, default=str(ROOT / "Feature_Selection" / "DataSet" / "merged_error_dataset.csv"))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model_role", choices=["auto", "student", "teacher"], default="auto")
    parser.add_argument("--max_batches_per_split", type=int, default=None, help="debug only; limits prove inference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir or (Path(__file__).resolve().parent / "results" / f"prove_{timestamp()}"))
    fig_dir = ensure_dir(out_dir / "figures")
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    log("PROVE", "Loading V4 checkpoint")
    ckpt = safe_load_checkpoint(args.checkpoint, device)
    spec = load_column_spec(args.feature_dir)
    scaler = restore_scaler_v4(args.checkpoint, ckpt)
    ckpt_args = ckpt.get("args") or {}
    seed = int(get_arg_with_default(ckpt_args, "seed", None, 42))
    history_len = int(get_arg_with_default(ckpt_args, "history_len", None, 9))
    teacher_future_len_ckpt = int(get_arg_with_default(ckpt_args, "teacher_future_len", None, 3))
    pred_horizon = int(get_arg_with_default(ckpt_args, "pred_horizon", None, 0))
    train_ratio = float(get_arg_with_default(ckpt_args, "train_ratio", None, 0.7))
    val_ratio = float(get_arg_with_default(ckpt_args, "val_ratio", None, 0.15))
    sinkage_max = float(get_arg_with_default(ckpt_args, "sinkage_max", None, 0.08))
    kinematic_pos_gate_scale = float(get_arg_with_default(ckpt_args, "kinematic_pos_gate_scale", None, 1.0))
    model_role = resolve_model_role_v4(args.model_role, ckpt)
    teacher_future_len = teacher_future_len_ckpt if model_role == "teacher" else 0
    model, meta = build_model_from_checkpoint_v4(ckpt, spec, model_role, device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    df = load_merged_dataset(args.merged_csv, spec, case_col="case_name", time_col="time")
    df_train, df_val, df_test = split_train_val_test_by_case(df, case_col="case_name", train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    split_dfs = {"train": df_train, "val": df_val, "test": df_test}
    all_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    split_metas: Dict[str, Dict] = {}
    summary_rows = []
    base_meta = {
        "checkpoint": str(args.checkpoint),
        "history_len": history_len,
        "model_role": model_role,
        "F_base_source": "build_v4_contact_prediction_raw(): lf_force + gate_force * pred_force_delta",
        "R_true_formula": "R = F_base - F_HF",
        "train_cases": sorted(df_train["case_name"].astype(str).unique().tolist()),
        "val_cases": sorted(df_val["case_name"].astype(str).unique().tolist()),
        "test_cases": sorted(df_test["case_name"].astype(str).unique().tolist()),
    }
    for split in ["train", "val", "test"]:
        log("PROVE", f"Running {split} inference")
        arrays, split_meta = infer_split(
            split,
            split_dfs[split],
            spec,
            scaler,
            model,
            model_role,
            history_len,
            teacher_future_len,
            pred_horizon,
            sinkage_max,
            kinematic_pos_gate_scale,
            args.batch_size,
            args.num_workers,
            device,
            args.max_batches_per_split,
        )
        all_arrays[split] = arrays
        split_metas[split] = split_meta
        metrics = summarize_force(arrays)
        summary_rows.append({"split": split, **split_meta, **metrics})

    log("PROVE", "Saving residual datasets")
    for split, arrays in all_arrays.items():
        save_npz(out_dir / f"residual_{split}.npz", arrays, {**base_meta, **split_metas[split], "split": split})
        write_json(out_dir / f"frame_audit_{split}.json", split_metas[split]["frame_audit"])
    pd.DataFrame(summary_rows).to_csv(out_dir / "residual_summary.csv", index=False)

    train_df = residual_frame(all_arrays["train"])
    val_df = residual_frame(all_arrays["val"])
    test_df = residual_frame(all_arrays["test"])
    analysis_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    log("PROVE", "Running ACF")
    acf_df = run_acf(analysis_df, out_dir)
    log("PROVE", "Running Ljung-Box")
    lb_df = run_ljung_box(analysis_df, out_dir)
    log("PROVE", "Running PSD")
    psd_df = run_psd(analysis_df, out_dir)
    run_residual_statistics(analysis_df, out_dir)
    log("PROVE", "Running AR predictability")
    ar_df = run_ar_predictability(train_df, val_df, test_df, history_len, out_dir)
    log("PROVE", "Running counterfactual test")
    cf_df = run_counterfactual(train_df, val_df, history_len, out_dir, seed=seed)

    structure, structure_evidence = classify_structure(acf_df, lb_df, ar_df)
    proof = {
        **base_meta,
        "output_dir": str(out_dir),
        "V4_base_force_rmse": summary_rows,
        "acf_fz_lag123_abs_mean": structure_evidence["fz_acf_lag123_abs_mean"],
        "ljung_box_fz_reject_ratio_lag10": structure_evidence["fz_ljung_reject_ratio_lag10"],
        "ar_fz_val_relative_improvement": structure_evidence["fz_ar_val_relative_improvement"],
        "psd_fz_energy_distribution": psd_df[psd_df["direction"] == "Fz"].to_dict(orient="records"),
        "counterfactual": cf_df.to_dict(orient="records"),
        "residual_structure": structure,
        "residual_structure_rule": "strong/moderate/weak uses Fz Ljung-Box reject ratio, lag-1/2/3 ACF magnitude, and AR relative improvement; p value alone is insufficient.",
        "frame_audit": {split: split_metas[split]["frame_audit"] for split in split_metas},
    }
    write_json(out_dir / "proof_summary.json", proof)
    with open(out_dir / "proof_report.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(proof, ensure_ascii=False, indent=2))
        f.write("\n")
    log("PROVE", "Finished")
    print(f"proof_dir: {out_dir}")


if __name__ == "__main__":
    main()
