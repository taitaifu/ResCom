from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as torch_mp
import torch.nn.functional as F
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

# Avoid DataLoader worker failures while passing tensor storage file descriptors.
torch_mp.set_sharing_strategy("file_system")

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models.data_utils_v3 import (  # noqa: E402
    ROCKER_NAMES,
    WHEEL_IDS,
    get_group_dims,
    graph_temporal_collate_fn,
    prepare_datasets_and_scaler,
    save_column_spec_json,
)
from models.differentiable_terramechanics import TerramechanicsParams, wheel_terrain_force  # noqa: E402
from models.graph_temporal_hgt_compensation_v3 import GraphTemporalHGTCompensationModelV3  # noqa: E402

ASSEMBLY_SIDES = {
    "left": {"front": "lf", "rear": "lm", "sub": "lb", "front_wheel": 0, "mid_wheel": 2, "rear_wheel": 4},
    "right": {"front": "rf", "rear": "rm", "sub": "rb", "front_wheel": 1, "mid_wheel": 3, "rear_wheel": 5},
}
ASSEMBLY_DISTANCE_CONSTRAINTS = (
    "front_AB",
    "front_BC",
    "front_AC",
    "rear_main_AD",
    "bogie_DE",
    "bogie_DF",
    "middle_upright",
    "rear_upright",
    "middle_rear_wheel",
)

LOSS_COMPONENT_KEYS = (
    "total",
    "val_total",
    "state",
    "attitude",
    "assembly",
    "assembly_distance",
    "assembly_joint",
    "force",
    "phy_state",
    "force_delta",
    "corr_loss",
    "gate_loss",
    "harm_loss",
    "output_score",
    "improve_score",
    "harmful_ratio",
    "dyn",
    "residual_reg",
    "weighted_force",
    "weighted_attitude",
    "weighted_assembly",
    "weighted_phy_state",
    "weighted_force_delta",
    "weighted_corr",
    "weighted_gate",
    "weighted_dyn",
    "weighted_reg",
    "F_final_rmse",
    "F_phy_pred_ref_rmse",
    "F_phy_pred_hf_rmse",
    "F_delta_rmse",
    "sinkage_lf_mean",
    "sinkage_pred_mean",
    "sinkage_delta_mean",
    "sinkage_lf_max",
    "sinkage_pred_max",
    "distill_z",
    "weighted_distill_z",
    "z_teacher_norm",
    "z_student_norm",
    "z_cosine_similarity",
)


def write_tensorboard_scalars(writer, prefix: str, metrics: Dict[str, float], epoch: int) -> None:
    if writer is None:
        return
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            writer.add_scalar(f"{prefix}/{key}", float(value), epoch)


def write_optimizer_lrs(writer, optimizer: torch.optim.Optimizer, epoch: int) -> None:
    if writer is None:
        return
    for idx, group in enumerate(optimizer.param_groups):
        name = str(group.get("group_name", f"group_{idx}"))
        writer.add_scalar(f"lr/{name}", float(group["lr"]), epoch)


def _finite_metric(metrics: Dict[str, float], key: str) -> Optional[float]:
    value = metrics.get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def write_v3_grouped_tensorboard(writer, phase: str, metrics: Dict[str, float], epoch: int) -> None:
    if writer is None:
        return
    scalar_groups = {
        "loss": ("attitude", "assembly", "assembly_distance", "assembly_joint"),
        "position": ("rocker_rmse",),
        "attitude": ("body_deg", "rocker_deg", "wheel_deg"),
        "assembly": (
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
        ),
    }
    for group, names in scalar_groups.items():
        for name in names:
            metric_key = f"{group}/{name}" if group in {"position", "attitude", "assembly"} else name
            value = _finite_metric(metrics, metric_key)
            if value is not None:
                writer.add_scalar(f"{group}/{name}/{phase}", value, epoch)


def log_step(message: str, start_time: Optional[float] = None) -> float:
    now = time.perf_counter()
    if start_time is None:
        print(f"[V3] {message}", flush=True)
    else:
        print(f"[V3] {message}: {now - start_time:.2f}s", flush=True)
    return now


def compact_metrics(metrics: Dict[str, float], keys: Tuple[str, ...] = LOSS_COMPONENT_KEYS) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            out[key] = float(value)
    return out


def format_metrics(metrics: Dict[str, float], keys: Tuple[str, ...] = LOSS_COMPONENT_KEYS) -> str:
    parts = []
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            parts.append(f"{key}={float(value):.6g}")
    return " ".join(parts)


def print_epoch_metrics(epoch: int, phase: str, metrics: Dict[str, float]) -> None:
    sections = [
        ("loss", ("total", "val_total", "state", "attitude", "assembly", "assembly_distance", "assembly_joint", "force", "phy_state", "force_delta", "corr_loss", "gate_loss", "harm_loss", "no_harm", "dyn", "residual_reg", "body_delta_reg", "body_delta_smooth")),
        ("select", ("output_score", "improve_score", "harmful_ratio", "body_improve_ratio", "wheel_kin_improve_ratio", "wheel_force_improve_ratio", "body_harmful_ratio", "wheel_kin_harmful_ratio", "wheel_force_harmful_ratio")),
        ("weighted", ("weighted_attitude", "weighted_assembly", "weighted_force", "weighted_phy_state", "weighted_force_delta", "weighted_corr", "weighted_gate", "weighted_harm", "weighted_dyn", "weighted_reg", "weighted_body_delta_reg", "weighted_body_delta_smooth")),
        ("pose", ("body_pos_xyz_rmse", "body_vel_xyz_rmse", "position/rocker_rmse", "wheel_local_xyz_rmse", "attitude/body_deg", "attitude/rocker_deg", "attitude/wheel_deg")),
        ("assembly", ("assembly/front_AB", "assembly/front_BC", "assembly/front_AC", "assembly/rear_main_AD", "assembly/bogie_DE", "assembly/bogie_DF", "assembly/middle_upright", "assembly/rear_upright", "assembly/middle_rear_wheel", "assembly/sub_joint_coincidence")),
        ("force", ("F_final_rmse", "F_phy_pred_ref_rmse", "F_phy_pred_hf_rmse", "F_delta_rmse", "wheel0_Fx_rmse", "wheel0_Fy_rmse", "wheel0_Fz_rmse")),
        ("sinkage", ("sinkage_lf_mean", "sinkage_pred_mean", "sinkage_delta_mean", "sinkage_lf_max", "sinkage_pred_max")),
        ("distill", ("distill_z", "weighted_distill_z", "z_teacher_norm", "z_student_norm", "z_cosine_similarity")),
    ]
    print(f"\nepoch {epoch:04d} {phase}", flush=True)
    for title, keys in sections:
        text = format_metrics(metrics, keys)
        if text:
            print(f"  {title:<8} {text}", flush=True)


def _finite_float(metrics: Dict[str, float], key: str) -> Optional[float]:
    value = metrics.get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _mean_present(values: List[Optional[float]]) -> Optional[float]:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def add_validation_selection_metrics(metrics: Dict[str, float], eps: float = 1e-8) -> Dict[str, float]:
    out = dict(metrics)
    total = _finite_float(out, "total")
    if total is not None:
        out["val_total"] = total
    body_pred = _finite_float(out, "body_pred_error")
    body_lf = _finite_float(out, "body_LF_error")
    wheel_kin_pred = _mean_present([_finite_float(out, f"wheel{i}_kin_pred_error") for i in WHEEL_IDS])
    wheel_kin_lf = _mean_present([_finite_float(out, f"wheel{i}_kin_LF_error") for i in WHEEL_IDS])
    wheel_force_pred = _mean_present([_finite_float(out, f"wheel{i}_force_pred_error") for i in WHEEL_IDS])
    wheel_force_lf = _mean_present([_finite_float(out, f"wheel{i}_force_LF_error") for i in WHEEL_IDS])

    body_improve = body_pred / (body_lf + eps) if body_pred is not None and body_lf is not None else None
    wheel_kin_improve = wheel_kin_pred / (wheel_kin_lf + eps) if wheel_kin_pred is not None and wheel_kin_lf is not None else None
    wheel_force_improve = wheel_force_pred / (wheel_force_lf + eps) if wheel_force_pred is not None and wheel_force_lf is not None else None

    body_harmful = _finite_float(out, "no_harm_body_violation_rate")
    wheel_kin_harmful = _mean_present([_finite_float(out, f"no_harm_wheel{i}_kin_violation_rate") for i in WHEEL_IDS])
    wheel_force_harmful = _mean_present([_finite_float(out, f"no_harm_wheel{i}_force_violation_rate") for i in WHEEL_IDS])

    derived = {
        "body_pred_error": body_pred,
        "wheel_kin_pred_error": wheel_kin_pred,
        "wheel_force_pred_error": wheel_force_pred,
        "body_LF_error": body_lf,
        "wheel_kin_LF_error": wheel_kin_lf,
        "wheel_force_LF_error": wheel_force_lf,
        "body_improve_ratio": body_improve,
        "wheel_kin_improve_ratio": wheel_kin_improve,
        "wheel_force_improve_ratio": wheel_force_improve,
        "body_harmful_ratio": body_harmful,
        "wheel_kin_harmful_ratio": wheel_kin_harmful,
        "wheel_force_harmful_ratio": wheel_force_harmful,
    }
    for key, value in derived.items():
        if value is not None and math.isfinite(float(value)):
            out[key] = float(value)

    output_score = _mean_present([body_pred, wheel_kin_pred, wheel_force_pred])
    improve_score = _mean_present([body_improve, wheel_kin_improve, wheel_force_improve])
    harmful_ratio = _mean_present([body_harmful, wheel_kin_harmful, wheel_force_harmful])
    if output_score is not None:
        out["output_score"] = output_score
    if improve_score is not None:
        out["improve_score"] = improve_score
    if harmful_ratio is not None:
        out["harmful_ratio"] = harmful_ratio
    return out


def checkpoint_payload(model, z_projector, group_dims, scaler, args, epoch: int, metric_name: str, metric_value: float) -> Dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "z_projector_state_dict": z_projector.state_dict(),
        "group_dims": group_dims,
        "scaler_state": {k: v.to_dict() for k, v in scaler.scalers.items()},
        "args": vars(args),
        "best": metric_value,
        "best_metric": metric_name,
        "best_epoch": epoch,
        "v3_role": args.train_stage,
        "teacher_frozen": args.train_stage == "student",
    }


def save_checkpoint(path: str, model, z_projector, group_dims, scaler, args, epoch: int, metric_name: str, metric_value: float) -> None:
    torch.save(
        checkpoint_payload(model, z_projector, group_dims, scaler, args, epoch, metric_name, metric_value),
        path,
    )


def tensor_finite_summary(x: torch.Tensor) -> str:
    if not isinstance(x, torch.Tensor):
        return ""
    if not x.dtype.is_floating_point:
        return f"shape={tuple(x.shape)} dtype={x.dtype}"
    detached = x.detach()
    finite = torch.isfinite(detached)
    nan_count = int(torch.isnan(detached).sum().item())
    inf_count = int(torch.isinf(detached).sum().item())
    if bool(finite.any()):
        vals = detached[finite].float()
        return (
            f"shape={tuple(detached.shape)} dtype={detached.dtype} "
            f"nan={nan_count} inf={inf_count} "
            f"finite_min={float(vals.min().cpu()):.6g} "
            f"finite_max={float(vals.max().cpu()):.6g} "
            f"finite_mean={float(vals.mean().cpu()):.6g}"
        )
    return f"shape={tuple(detached.shape)} dtype={detached.dtype} nan={nan_count} inf={inf_count} finite=0"


def collect_nonfinite_tensors(prefix: str, values: Dict[str, Any]) -> List[str]:
    bad: List[str] = []
    for key, value in values.items():
        if not isinstance(value, torch.Tensor) or not value.dtype.is_floating_point or value.numel() == 0:
            continue
        if not bool(torch.isfinite(value.detach()).all()):
            bad.append(f"{prefix}.{key}: {tensor_finite_summary(value)}")
    return bad


def collect_nonfinite_grads(model: torch.nn.Module, z_projector: Optional[torch.nn.Module] = None) -> List[str]:
    bad: List[str] = []
    modules = [("model", model)]
    if z_projector is not None:
        modules.append(("z_projector", z_projector))
    for module_prefix, module in modules:
        for name, param in module.named_parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            if grad.dtype.is_floating_point and grad.numel() > 0 and not bool(torch.isfinite(grad).all()):
                bad.append(f"grad.{module_prefix}.{name}: {tensor_finite_summary(grad)}")
    return bad


def batch_context_summary(batch: Dict[str, Any]) -> str:
    parts: List[str] = []
    case_names = batch.get("case_name")
    if isinstance(case_names, list):
        parts.append(f"case_name[:5]={case_names[:5]}")
    time_value = batch.get("time")
    if isinstance(time_value, torch.Tensor) and time_value.numel() > 0:
        flat = time_value.detach().float().reshape(-1)
        parts.append(f"time_first={float(flat[0].cpu()):.6g}")
        parts.append(f"time_last={float(flat[-1].cpu()):.6g}")
    return " ".join(parts)


def cpu_debug_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return {k: cpu_debug_copy(v) for k, v in value.items()}
    return value


def handle_nonfinite_debug(
    *,
    args,
    epoch: int,
    phase_name: str,
    batch_idx: int,
    batch: Dict[str, Any],
    output: Dict[str, Any],
    losses: Dict[str, Any],
    problems: List[str],
    stage: str,
) -> None:
    debug_dir = Path(args.debug_dump_dir) if args.debug_dump_dir else Path(args.save_dir) / "debug_finite"
    debug_dir.mkdir(parents=True, exist_ok=True)
    dump_path = debug_dir / f"nonfinite_epoch{epoch:04d}_{phase_name}_batch{batch_idx:06d}_{stage}.pt"
    torch.save(
        {
            "epoch": epoch,
            "phase": phase_name,
            "batch_idx": batch_idx,
            "stage": stage,
            "context": batch_context_summary(batch),
            "problems": problems,
            "losses": cpu_debug_copy(losses),
            "output": cpu_debug_copy(output),
            "batch": cpu_debug_copy(batch),
        },
        dump_path,
    )
    print("\n[V3][finite-check] 检测到 NaN/Inf，训练已停止。", flush=True)
    print(f"[V3][finite-check] epoch={epoch} phase={phase_name} batch={batch_idx} stage={stage}", flush=True)
    context = batch_context_summary(batch)
    if context:
        print(f"[V3][finite-check] {context}", flush=True)
    for item in problems[:40]:
        print(f"[V3][finite-check] {item}", flush=True)
    if len(problems) > 40:
        print(f"[V3][finite-check] ... 还有 {len(problems) - 40} 项未显示", flush=True)
    print(f"[V3][finite-check] debug batch 已保存: {dump_path}", flush=True)
    raise RuntimeError(f"Non-finite tensor detected at epoch={epoch}, phase={phase_name}, batch={batch_idx}, stage={stage}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def _scaler_tensors(scaler, group_name: str, device: torch.device):
    s = scaler.scalers[group_name]
    return (
        torch.as_tensor(s.mean_, dtype=torch.float32, device=device),
        torch.as_tensor(s.std_, dtype=torch.float32, device=device),
        torch.as_tensor(s.apply_mask_, dtype=torch.bool, device=device),
    )


def inverse_transform_tensor(x: torch.Tensor, scaler, group_name: str) -> torch.Tensor:
    mean, std, mask = _scaler_tensors(scaler, group_name, x.device)
    y = x.float().clone()
    if y.shape[-1] and bool(mask.any()):
        y[..., mask] = y[..., mask] * std[mask] + mean[mask]
    return y


def transform_tensor(x: torch.Tensor, scaler, group_name: str) -> torch.Tensor:
    mean, std, mask = _scaler_tensors(scaler, group_name, x.device)
    y = x.float().clone()
    if y.shape[-1] and bool(mask.any()):
        y[..., mask] = (y[..., mask] - mean[mask]) / std[mask].clamp_min(1e-8)
    return y


def suffix_indices(cols: List[str], suffixes: List[str]) -> List[int]:
    out: List[int] = []
    for suffix in suffixes:
        for i, c in enumerate(cols):
            if c.endswith(suffix):
                out.append(i)
                break
    return out


def omega_index(cols: List[str]) -> Optional[int]:
    for suffix in ["omega", "_omega", "ang_vel_y", "ang_vel_z", "ang_vel_x"]:
        idx = suffix_indices(cols, [suffix])
        if idx:
            return idx[0]
    return None


def contact_mask(batch: Dict[str, torch.Tensor], spec, wheel_id: int, like: torch.Tensor) -> Optional[torch.Tensor]:
    cols = spec.input_groups.wheel_contact_cols[wheel_id]
    idx = suffix_indices(cols, ["in_contact"])
    if not idx:
        return None
    mask = batch[f"wheel{wheel_id}_contact"][:, -1, idx[0]]
    if like.ndim == 3:
        mask = mask.unsqueeze(1).expand(-1, like.shape[1])
    return mask


def masked_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if pred.numel() == 0:
        return pred.new_tensor(0.0)
    if mask is None:
        return F.huber_loss(pred, target)
    if mask.ndim == pred.ndim - 1:
        mask = mask.unsqueeze(-1)
    mask = mask.to(device=pred.device, dtype=torch.bool).expand_as(pred)
    if not bool(mask.any()):
        return pred.new_tensor(0.0)
    return F.huber_loss(pred[mask], target[mask])


def axis_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    names = ["x", "y", "z"]
    losses = []
    metrics = {}
    for i, name in enumerate(names[: pred.shape[-1]]):
        li = masked_loss(pred[..., i:i + 1], target[..., i:i + 1], mask)
        losses.append(li)
        metrics[name] = li
    return torch.stack(losses).mean() if losses else pred.new_tensor(0.0), metrics


def reconstruct_scaled(pred_res_scaled, lf_raw, res_cols, hf_cols, res_group, hf_group, scaler):
    pred_res_raw = inverse_transform_tensor(pred_res_scaled, scaler, res_group)
    pred_raw = lf_raw.clone().float()
    res_idx = {c: i for i, c in enumerate(res_cols)}
    for j, hf_col in enumerate(hf_cols):
        if not hf_col.startswith("hf_"):
            continue
        res_col = "res_" + hf_col[len("hf_"):]
        if res_col in res_idx:
            pred_raw[..., j] = lf_raw[..., j] + pred_res_raw[..., res_idx[res_col]]
    return transform_tensor(pred_raw, scaler, hf_group), pred_raw


def residual_raw_like_target(pred_res_scaled, res_cols, hf_cols, res_group, scaler) -> torch.Tensor:
    pred_res_raw = inverse_transform_tensor(pred_res_scaled, scaler, res_group)
    out = pred_res_raw.new_zeros(*pred_res_raw.shape[:-1], len(hf_cols))
    res_idx = {c: i for i, c in enumerate(res_cols)}
    for j, hf_col in enumerate(hf_cols):
        if not hf_col.startswith("hf_"):
            continue
        res_col = "res_" + hf_col[len("hf_"):]
        if res_col in res_idx:
            out[..., j] = pred_res_raw[..., res_idx[res_col]]
    return out


def reconstruct_body_kinematic(
    pred_res_scaled: torch.Tensor,
    lf_body_raw: torch.Tensor,
    prev_body_raw: torch.Tensor,
    dt: torch.Tensor,
    res_cols: List[str],
    hf_cols: List[str],
    scaler,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_delta_raw = residual_raw_like_target(pred_res_scaled, res_cols, hf_cols, "res_body", scaler)
    pred_raw = lf_body_raw.clone().float()
    bpos = suffix_indices(hf_cols, ["pos_x", "pos_y", "pos_z"])
    bvel = suffix_indices(hf_cols, ["vel_x", "vel_y", "vel_z"])
    if len(bvel) == 3:
        pred_raw[..., bvel] = lf_body_raw[..., bvel].float() + pred_delta_raw[..., bvel]
    if len(bpos) == 3 and len(bvel) == 3:
        prev = prev_body_raw.float()
        if prev.ndim == 2:
            prev = prev.unsqueeze(1)
        prev_pos = prev[..., bpos]
        prev_vel = prev[..., bvel]
        body_delta_pos = pred_delta_raw[..., bpos]
        dt_view = dt.to(device=pred_raw.device, dtype=pred_raw.dtype).view(-1, 1, 1).clamp_min(0.0)
        integrated_pos = prev_pos + 0.5 * (prev_vel + pred_raw[..., bvel]) * dt_view
        use_lf_init = dt_view <= 1e-9
        init_pos = lf_body_raw[..., bpos].float()
        pred_raw[..., bpos] = torch.where(use_lf_init.expand_as(integrated_pos), init_pos, integrated_pos) + body_delta_pos
    return transform_tensor(pred_raw, scaler, "hf_body"), pred_raw, pred_delta_raw


def local_xyz_delta(pred_delta_raw: torch.Tensor, pos_idx: List[int]) -> torch.Tensor:
    local = pred_delta_raw.new_zeros(*pred_delta_raw.shape[:-1], 3)
    if len(pos_idx) == 3:
        local[..., 0] = pred_delta_raw[..., pos_idx[0]]
        local[..., 1] = pred_delta_raw[..., pos_idx[1]]
        local[..., 2] = pred_delta_raw[..., pos_idx[2]]
    return local


def reconstruct_component_position(
    component_lf_raw: torch.Tensor,
    body_global_delta_raw: torch.Tensor,
    pred_delta_raw: torch.Tensor,
    pos_idx: List[int],
    body_rot_pred: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pred_raw = component_lf_raw.clone().float()
    local_delta = local_xyz_delta(pred_delta_raw, pos_idx)
    if len(pos_idx) == 3:
        if body_rot_pred is not None:
            local_world = torch.matmul(body_rot_pred, local_delta.unsqueeze(-1)).squeeze(-1)
        else:
            local_world = local_delta
        pred_raw[..., pos_idx] = component_lf_raw[..., pos_idx].float() + body_global_delta_raw + local_world
    return pred_raw, local_delta


def reconstruct_wheel_kin_with_body_delta(
    pred_res_scaled: torch.Tensor,
    lf_wheel_raw: torch.Tensor,
    body_global_delta_raw: torch.Tensor,
    res_cols: List[str],
    hf_cols: List[str],
    hf_group: str,
    res_group: str,
    scaler,
    body_rot_pred: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_delta_raw = residual_raw_like_target(pred_res_scaled, res_cols, hf_cols, res_group, scaler)
    pred_raw = lf_wheel_raw.clone().float()
    wpos = suffix_indices(hf_cols, ["pos_x", "pos_y", "pos_z"])
    omega_i = omega_index(hf_cols)
    if len(wpos) == 3:
        pred_raw, _ = reconstruct_component_position(lf_wheel_raw, body_global_delta_raw, pred_delta_raw, wpos, body_rot_pred)
    if omega_i is not None:
        pred_raw[..., omega_i] = lf_wheel_raw[..., omega_i].float() + pred_delta_raw[..., omega_i]
    return transform_tensor(pred_raw, scaler, hf_group), pred_raw, pred_delta_raw


def reconstruct_rocker_position_with_body_delta(
    lf_pose_raw: torch.Tensor,
    body_global_delta_raw: torch.Tensor,
    local_delta_xyz: torch.Tensor,
    body_rot_pred: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pos_idx = [0, 1, 2]
    pred_raw, local_delta = reconstruct_component_position(
        lf_pose_raw,
        body_global_delta_raw,
        local_delta_xyz,
        pos_idx,
        body_rot_pred,
    )
    return pred_raw[..., pos_idx], local_delta


def quat_to_matrix_torch(q: torch.Tensor) -> torch.Tensor:
    q = q.float()
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(dim=-1)
    row0 = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], dim=-1)
    row1 = torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], dim=-1)
    row2 = torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    theta = rotvec.norm(dim=-1, keepdim=True)
    axis = rotvec / theta.clamp_min(1e-12)
    x, y, z = axis.unbind(dim=-1)
    zero = torch.zeros_like(x)
    k = torch.stack([
        torch.stack([zero, -z, y], dim=-1),
        torch.stack([z, zero, -x], dim=-1),
        torch.stack([-y, x, zero], dim=-1),
    ], dim=-2)
    eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype).view(*([1] * (rotvec.ndim - 1)), 3, 3)
    sin = torch.sin(theta).unsqueeze(-1)
    cos = torch.cos(theta).unsqueeze(-1)
    return eye + sin * k + (1 - cos) * torch.matmul(k, k)


def matrix_to_rotvec(rot: torch.Tensor) -> torch.Tensor:
    trace = rot.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)
    vee = torch.stack([
        rot[..., 2, 1] - rot[..., 1, 2],
        rot[..., 0, 2] - rot[..., 2, 0],
        rot[..., 1, 0] - rot[..., 0, 1],
    ], dim=-1)
    scale = theta / (2.0 * torch.sin(theta).clamp_min(1e-7))
    return vee * scale.unsqueeze(-1)


def compose_part_rotation(body_rot_pred: torch.Tensor, body_rot_lf: torch.Tensor, part_rot_lf: torch.Tensor, delta_rot: torch.Tensor) -> torch.Tensor:
    rel_lf = torch.matmul(body_rot_lf.transpose(-1, -2), part_rot_lf)
    return torch.matmul(torch.matmul(body_rot_pred, rel_lf), rotvec_to_matrix(delta_rot))


def so3_geodesic_angle(r_pred: torch.Tensor, r_true: torch.Tensor) -> torch.Tensor:
    rel = torch.matmul(r_pred.transpose(-1, -2), r_true)
    trace = rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos_theta)


def pose_indices(cols: List[str]) -> Tuple[List[int], List[int]]:
    return suffix_indices(cols, ["pos_x", "pos_y", "pos_z"]), suffix_indices(cols, ["q0", "q1", "q2", "q3"])


def transform_local_point(pos: torch.Tensor, rot: torch.Tensor, local_point: torch.Tensor) -> torch.Tensor:
    while local_point.ndim < pos.ndim:
        local_point = local_point.unsqueeze(1)
    return pos + torch.matmul(rot, local_point.unsqueeze(-1)).squeeze(-1)


def assembly_tensor(batch: Dict[str, torch.Tensor], key: str, like: torch.Tensor) -> Optional[torch.Tensor]:
    value = batch.get(key)
    if not isinstance(value, torch.Tensor):
        return None
    return value.to(device=like.device, dtype=like.dtype)


def smooth_l1_zero(x: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(x, torch.zeros_like(x))


def scaler_std_tensor(scaler, group: str, ref: torch.Tensor, indices: Optional[List[int]] = None) -> torch.Tensor:
    std = torch.as_tensor(scaler.scalers[group].std_, device=ref.device, dtype=ref.dtype).clamp_min(1e-6)
    if indices is not None:
        std = std[indices]
    return std


def add_no_harm_raw(
    terms: List[torch.Tensor],
    metrics: Dict[str, torch.Tensor],
    name: str,
    pred_raw: torch.Tensor,
    lf_raw: torch.Tensor,
    true_raw: torch.Tensor,
    std: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> None:
    e_pred = torch.abs(pred_raw - true_raw) / std
    e_lf = torch.abs(lf_raw.float() - true_raw) / std
    harm = torch.relu(e_pred - e_lf)
    violate = e_pred > e_lf
    if mask is not None:
        mask_bool = mask.to(dtype=torch.bool)
        while mask_bool.ndim < harm.ndim:
            mask_bool = mask_bool.unsqueeze(-1)
        mask_bool = mask_bool.expand_as(harm)
        harm = harm[mask_bool]
        violate = violate[mask_bool]
    terms.append(harm.mean() if harm.numel() else pred_raw.new_tensor(0.0))
    metrics[f"no_harm_{name}_violation_rate"] = violate.float().mean() if violate.numel() else pred_raw.new_tensor(0.0)


def add_no_harm_angle(
    terms: List[torch.Tensor],
    metrics: Dict[str, torch.Tensor],
    name: str,
    pred_angle: torch.Tensor,
    lf_angle: torch.Tensor,
) -> None:
    harm = torch.relu(pred_angle - lf_angle)
    terms.append(harm.mean())
    metrics[f"no_harm_{name}_violation_rate"] = (pred_angle > lf_angle).float().mean()


def gate_target_from_delta(delta_pred_raw: torch.Tensor, delta_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return ((delta_pred_raw * delta_true) / (delta_pred_raw.pow(2) + eps)).clamp(0.0, 1.0)


def apply_axis_gate(raw_pred: torch.Tensor, lf_raw: torch.Tensor, indices: List[int], gate: torch.Tensor) -> torch.Tensor:
    out = raw_pred.clone()
    if len(indices) == 0:
        return out
    out[..., indices] = lf_raw[..., indices].float() + gate * (raw_pred[..., indices] - lf_raw[..., indices].float())
    return out


def add_corr_and_gate_raw(
    corr_terms: List[torch.Tensor],
    gate_terms: List[torch.Tensor],
    metrics: Dict[str, torch.Tensor],
    name: str,
    pred_final: torch.Tensor,
    pred_ungated: torch.Tensor,
    lf_raw: torch.Tensor,
    true_raw: torch.Tensor,
    std: torch.Tensor,
    gate: Optional[torch.Tensor] = None,
    indices: Optional[List[int]] = None,
    mask: Optional[torch.Tensor] = None,
) -> None:
    if indices is not None:
        pred_final = pred_final[..., indices]
        pred_ungated = pred_ungated[..., indices]
        lf_raw = lf_raw[..., indices]
        true_raw = true_raw[..., indices]
        std = std[..., indices] if std.ndim > 1 else std[indices]
    delta_final = pred_final - lf_raw.float()
    delta_ungated = pred_ungated - lf_raw.float()
    delta_true = true_raw - lf_raw.float()
    corr = F.smooth_l1_loss(delta_final / std, delta_true / std, reduction="none")
    if mask is not None:
        mask_bool = mask.to(dtype=torch.bool)
        while mask_bool.ndim < corr.ndim:
            mask_bool = mask_bool.unsqueeze(-1)
        corr = corr[mask_bool.expand_as(corr)]
    corr_terms.append(corr.mean() if corr.numel() else pred_final.new_tensor(0.0))
    pred_norm_sq = ((pred_final - true_raw) / std).pow(2)
    lf_norm_sq = ((lf_raw.float() - true_raw) / std).pow(2)
    if mask is not None:
        mask_bool = mask.to(dtype=torch.bool)
        while mask_bool.ndim < pred_norm_sq.ndim:
            mask_bool = mask_bool.unsqueeze(-1)
        mask_bool = mask_bool.expand_as(pred_norm_sq)
        pred_norm_sq_masked = pred_norm_sq[mask_bool]
        lf_norm_sq_masked = lf_norm_sq[mask_bool]
    else:
        pred_norm_sq_masked = pred_norm_sq.reshape(-1)
        lf_norm_sq_masked = lf_norm_sq.reshape(-1)
    metrics[f"{name}_pred_error"] = torch.sqrt(pred_norm_sq_masked.mean()) if pred_norm_sq_masked.numel() else pred_final.new_tensor(0.0)
    metrics[f"{name}_LF_error"] = torch.sqrt(lf_norm_sq_masked.mean()) if lf_norm_sq_masked.numel() else pred_final.new_tensor(0.0)
    metrics[f"{name}_pred_vs_hf"] = torch.sqrt(((pred_final - true_raw) ** 2).mean())
    metrics[f"{name}_LF_vs_hf"] = torch.sqrt(((lf_raw.float() - true_raw) ** 2).mean())
    metrics[f"{name}_pred_delta_magnitude"] = torch.sqrt((delta_final ** 2).mean())
    metrics[f"{name}_true_delta_magnitude"] = torch.sqrt((delta_true ** 2).mean())
    if gate is not None:
        target = gate_target_from_delta(delta_ungated.detach(), delta_true)
        gate_loss = F.smooth_l1_loss(gate, target.detach(), reduction="none")
        if mask is not None:
            mask_bool = mask.to(dtype=torch.bool)
            while mask_bool.ndim < gate_loss.ndim:
                mask_bool = mask_bool.unsqueeze(-1)
            gate_loss = gate_loss[mask_bool.expand_as(gate_loss)]
        gate_terms.append(gate_loss.mean() if gate_loss.numel() else pred_final.new_tensor(0.0))
        metrics[f"{name}_gate_mean"] = gate.mean()


def add_corr_and_gate_angle(
    corr_terms: List[torch.Tensor],
    gate_terms: List[torch.Tensor],
    metrics: Dict[str, torch.Tensor],
    name: str,
    gated_delta: torch.Tensor,
    raw_delta: torch.Tensor,
    true_delta: torch.Tensor,
    gate: torch.Tensor,
) -> None:
    corr_terms.append(F.smooth_l1_loss(gated_delta, true_delta))
    gate_terms.append(F.smooth_l1_loss(gate, gate_target_from_delta(raw_delta.detach(), true_delta).detach()))
    metrics[f"{name}_gate_mean"] = gate.mean()
    metrics[f"{name}_pred_delta_magnitude"] = torch.sqrt((gated_delta ** 2).mean())
    metrics[f"{name}_true_delta_magnitude"] = torch.sqrt((true_delta ** 2).mean())


def add_gate_only_raw(
    gate_terms: List[torch.Tensor],
    metrics: Dict[str, torch.Tensor],
    name: str,
    gate: torch.Tensor,
    pred_ungated: torch.Tensor,
    lf_raw: torch.Tensor,
    true_raw: torch.Tensor,
    indices: List[int],
) -> None:
    if len(indices) == 0:
        return
    delta_ungated = pred_ungated[..., indices] - lf_raw[..., indices].float()
    delta_true = true_raw[..., indices] - lf_raw[..., indices].float()
    target = gate_target_from_delta(delta_ungated.detach(), delta_true)
    gate_terms.append(F.smooth_l1_loss(gate, target.detach()))
    metrics[f"{name}_gate_mean"] = gate.mean()


def compute_assembly_losses(
    batch: Dict[str, torch.Tensor],
    pred_body_pos: torch.Tensor,
    pred_body_rot: torch.Tensor,
    rocker_pred: Dict[str, Dict[str, torch.Tensor]],
    wheel_pred_pos: Dict[int, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    device_ref = pred_body_pos
    distance_losses: List[torch.Tensor] = []
    joint_losses: List[torch.Tensor] = []
    metrics: Dict[str, List[torch.Tensor]] = {f"assembly/{name}": [] for name in ASSEMBLY_DISTANCE_CONSTRAINTS}
    metrics["assembly/sub_joint_coincidence"] = []
    eps = 1e-6

    def add_distance(name: str, p0: torch.Tensor, p1: torch.Tensor, ref: torch.Tensor) -> None:
        current = torch.linalg.norm(p0 - p1, dim=-1)
        ref_view = ref.view(ref.shape[0], *([1] * (current.ndim - 1))).clamp_min(eps)
        rel_error = (current - ref_view) / ref_view
        distance_losses.append(smooth_l1_zero(rel_error))
        metrics[f"assembly/{name}"].append(rel_error.abs().mean())

    for side, side_spec in ASSEMBLY_SIDES.items():
        front = side_spec["front"]
        rear = side_spec["rear"]
        sub = side_spec["sub"]
        required_rockers = [front, rear, sub]
        required_wheels = [int(side_spec["front_wheel"]), int(side_spec["mid_wheel"]), int(side_spec["rear_wheel"])]
        if not all(name in rocker_pred for name in required_rockers):
            continue
        if not all(i in wheel_pred_pos for i in required_wheels):
            continue
        point_keys = [
            "A_front_body",
            "A_rear_body",
            "B_front_local",
            "D_main_local",
            "D_sub_local",
            "E_local",
            "F_local",
        ]
        points_meta = {name: assembly_tensor(batch, f"assembly_{side}_{name}", device_ref) for name in point_keys}
        refs_meta = {name: assembly_tensor(batch, f"assembly_{side}_ref_{name}", device_ref) for name in ASSEMBLY_DISTANCE_CONSTRAINTS}
        if any(v is None for v in points_meta.values()) or any(v is None for v in refs_meta.values()):
            continue

        a_front = transform_local_point(pred_body_pos, pred_body_rot, points_meta["A_front_body"])
        a_rear = transform_local_point(pred_body_pos, pred_body_rot, points_meta["A_rear_body"])
        b_front = transform_local_point(rocker_pred[front]["pos"], rocker_pred[front]["rot"], points_meta["B_front_local"])
        d_main = transform_local_point(rocker_pred[rear]["pos"], rocker_pred[rear]["rot"], points_meta["D_main_local"])
        d_sub = transform_local_point(rocker_pred[sub]["pos"], rocker_pred[sub]["rot"], points_meta["D_sub_local"])
        e_mid = transform_local_point(rocker_pred[sub]["pos"], rocker_pred[sub]["rot"], points_meta["E_local"])
        f_rear = transform_local_point(rocker_pred[sub]["pos"], rocker_pred[sub]["rot"], points_meta["F_local"])
        c_front = wheel_pred_pos[int(side_spec["front_wheel"])]
        c_mid = wheel_pred_pos[int(side_spec["mid_wheel"])]
        c_rear = wheel_pred_pos[int(side_spec["rear_wheel"])]

        add_distance("front_AB", a_front, b_front, refs_meta["front_AB"])
        add_distance("front_BC", b_front, c_front, refs_meta["front_BC"])
        add_distance("front_AC", a_front, c_front, refs_meta["front_AC"])
        add_distance("rear_main_AD", a_rear, d_main, refs_meta["rear_main_AD"])
        add_distance("bogie_DE", d_main, e_mid, refs_meta["bogie_DE"])
        add_distance("bogie_DF", d_main, f_rear, refs_meta["bogie_DF"])
        add_distance("middle_upright", e_mid, c_mid, refs_meta["middle_upright"])
        add_distance("rear_upright", f_rear, c_rear, refs_meta["rear_upright"])
        add_distance("middle_rear_wheel", c_mid, c_rear, refs_meta["middle_rear_wheel"])

        joint_vec = d_main - d_sub
        joint_losses.append(smooth_l1_zero(joint_vec))
        metrics["assembly/sub_joint_coincidence"].append(torch.linalg.norm(joint_vec, dim=-1).mean())

    l_distance = torch.stack(distance_losses).mean() if distance_losses else pred_body_pos.new_tensor(0.0)
    l_joint = torch.stack(joint_losses).mean() if joint_losses else pred_body_pos.new_tensor(0.0)
    metric_out = {
        key: torch.stack(values).mean() if values else pred_body_pos.new_tensor(0.0)
        for key, values in metrics.items()
    }
    return l_distance, l_joint, metric_out


def build_terrain_params(batch: Dict[str, torch.Tensor], wheel_id: int) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    terrain_z_name = f"wheel{wheel_id}_terrain_z"
    if terrain_z_name in batch:
        out["terrain_z"] = batch[terrain_z_name]
    point_name = f"wheel{wheel_id}_contact_point"
    normal_name = f"wheel{wheel_id}_contact_normal"
    if point_name in batch:
        out["contact_point"] = batch[point_name]
    if normal_name in batch:
        out["contact_normal"] = batch[normal_name]
    for key in ["Kc", "Kphi", "n0", "n1", "c", "phi", "K"]:
        name = f"wheel{wheel_id}_terrain_{key}"
        if name in batch:
            out[key] = batch[name]
    return out


def load_compatible_state_dict(module: torch.nn.Module, state_dict: Dict[str, torch.Tensor], label: str) -> None:
    current = module.state_dict()
    if "force_delta_head.net.0.weight" in current and "force_delta_head.net.0.weight" not in state_dict:
        merged = dict(state_dict)
        for key in current:
            if not key.startswith("force_delta_head."):
                continue
            suffix = key[len("force_delta_head."):]
            if suffix.startswith("net.4."):
                dyn_key = f"force_dynamic_head.{suffix}"
                b_key = f"force_bias_head.{suffix}"
                if dyn_key in state_dict and b_key in state_dict and tuple(state_dict[dyn_key].shape) == tuple(current[key].shape):
                    merged[key] = state_dict[dyn_key] + state_dict[b_key]
                    continue
            dynamic_key = f"force_dynamic_head.{suffix}"
            bias_key = f"force_bias_head.{suffix}"
            if dynamic_key in state_dict and tuple(state_dict[dynamic_key].shape) == tuple(current[key].shape):
                merged[key] = state_dict[dynamic_key]
            elif suffix.startswith("net.4."):
                if bias_key in state_dict and tuple(state_dict[bias_key].shape) == tuple(current[key].shape):
                    merged[key] = state_dict[bias_key]
        state_dict = merged
    compatible = {}
    skipped = []
    for key, value in state_dict.items():
        if key in current and tuple(current[key].shape) == tuple(value.shape):
            compatible[key] = value
        else:
            skipped.append(key)
    missing, unexpected = module.load_state_dict(compatible, strict=False)
    if skipped:
        preview = ", ".join(skipped[:8])
        suffix = " ..." if len(skipped) > 8 else ""
        print(f"[V3] {label} 跳过 {len(skipped)} 个 shape 不匹配参数: {preview}{suffix}", flush=True)
    if unexpected:
        print(f"[V3] {label} unexpected 参数: {list(unexpected)[:8]}", flush=True)
    if missing:
        print(f"[V3] {label} 未加载参数数: {len(missing)}", flush=True)


def compute_losses_v3(batch, output, spec, scaler, args) -> Dict[str, torch.Tensor]:
    losses: Dict[str, torch.Tensor] = {}
    true_body_raw = inverse_transform_tensor(batch["hf_body"], scaler, "hf_body")
    body_pos = suffix_indices(spec.target_groups.body_cols, ["pos_x", "pos_y", "pos_z"])
    body_vel = suffix_indices(spec.target_groups.body_cols, ["vel_x", "vel_y", "vel_z"])
    dt = batch.get("dt", output["pred_res_body"].new_full((output["pred_res_body"].shape[0],), args.dt))
    prev_body_raw = batch.get("lf_body_prev_raw", batch.get("hf_body_prev_raw", batch["lf_body_current"]))
    pred_body_scaled_ungated, pred_body_raw_ungated, body_delta_raw = reconstruct_body_kinematic(
        output["pred_res_body"],
        batch["lf_body_current"],
        prev_body_raw,
        dt,
        spec.res_groups.body_cols,
        spec.target_groups.body_cols,
        scaler,
    )
    body_gate = output.get("gate_body_pos", pred_body_raw_ungated.new_ones(*pred_body_raw_ungated.shape[:2], 3))
    pred_body_raw = apply_axis_gate(pred_body_raw_ungated, batch["lf_body_current"], body_pos, body_gate) if len(body_pos) == 3 else pred_body_raw_ungated
    pred_body_scaled = transform_tensor(pred_body_raw, scaler, "hf_body")
    body_delta_pos = body_delta_raw[..., body_pos] if len(body_pos) == 3 else pred_body_raw.new_zeros(*pred_body_raw.shape[:2], 3)
    lf_body_pos = batch["lf_body_current"][..., body_pos].float() if len(body_pos) == 3 else pred_body_raw.new_zeros(*pred_body_raw.shape[:2], 3)
    body_global_delta = pred_body_raw[..., body_pos] - lf_body_pos if len(body_pos) == 3 else body_delta_pos
    l_body_p, _ = axis_loss(pred_body_scaled[..., body_pos], batch["hf_body"][..., body_pos]) if len(body_pos) == 3 else (pred_body_scaled.new_tensor(0.0), {})
    l_body_v, _ = axis_loss(pred_body_scaled[..., body_vel], batch["hf_body"][..., body_vel]) if len(body_vel) == 3 else (pred_body_scaled.new_tensor(0.0), {})
    l_body_delta_reg = body_delta_pos.pow(2).mean() if len(body_pos) == 3 else pred_body_scaled.new_tensor(0.0)
    l_body_delta_smooth = pred_body_scaled.new_tensor(0.0)
    if "lf_body_pose_current" in batch and "hf_body_pose" in batch:
        lf_body_pose = batch["lf_body_pose_current"].float()
        hf_body_pose = batch["hf_body_pose"].float()
        body_rot_lf = quat_to_matrix_torch(lf_body_pose[..., 3:7])
        body_rot_hf = quat_to_matrix_torch(hf_body_pose[..., 3:7])
    else:
        eye = torch.eye(3, device=pred_body_raw.device, dtype=pred_body_raw.dtype).view(1, 1, 3, 3)
        body_rot_lf = eye.expand(*pred_body_raw.shape[:2], 3, 3)
        body_rot_hf = body_rot_lf
    body_att_gate = output.get("gate_body_att", output["pred_delta_rot_body"].new_ones(output["pred_delta_rot_body"].shape))
    body_delta_rot_gated = body_att_gate * output["pred_delta_rot_body"]
    body_rot_pred = torch.matmul(body_rot_lf, rotvec_to_matrix(body_delta_rot_gated))
    body_att_angle = so3_geodesic_angle(body_rot_pred, body_rot_hf)
    l_body_att = smooth_l1_zero(body_att_angle)

    wheel_losses = []
    wheel_local_losses = []
    wheel_local_regs = []
    wheel_local_rmse_terms = []
    wheel_local_abs_terms = []
    rocker_losses = []
    rocker_rmse_terms = []
    rocker_att_losses = []
    rocker_att_deg_terms = []
    wheel_att_losses = []
    wheel_att_deg_terms = []
    force_losses = []
    phy_state_losses = []
    force_delta_losses = []
    residual_regs = []
    no_harm_terms = []
    corr_terms = []
    gate_terms = []
    f_pred_all = []
    f_phy_pred_all = []
    f_phy_ref_all = []
    f_true_all = []
    f_delta_all = []
    f_delta_target_all = []
    sinkage_lf_all = []
    sinkage_pred_all = []
    wheel_gate_xyz_all = []
    wheel_gate_omega_all = []
    force_gate_all = []
    metrics: Dict[str, torch.Tensor] = {}

    add_no_harm_raw(
        no_harm_terms,
        metrics,
        "body",
        pred_body_raw,
        batch["lf_body_current"],
        true_body_raw,
        scaler_std_tensor(scaler, "hf_body", pred_body_raw),
    )
    body_lf_att_angle = so3_geodesic_angle(body_rot_lf, body_rot_hf)
    add_no_harm_angle(no_harm_terms, metrics, "body_attitude", body_att_angle, body_lf_att_angle)
    add_corr_and_gate_raw(
        corr_terms,
        gate_terms,
        metrics,
        "body",
        pred_body_raw,
        pred_body_raw_ungated,
        batch["lf_body_current"],
        true_body_raw,
        scaler_std_tensor(scaler, "hf_body", pred_body_raw),
    )
    add_gate_only_raw(
        gate_terms,
        metrics,
        "body_pos",
        body_gate,
        pred_body_raw_ungated,
        batch["lf_body_current"],
        true_body_raw,
        body_pos,
    )
    if "lf_body_pose_current" in batch and "hf_body_pose" in batch:
        body_true_delta_rot = matrix_to_rotvec(torch.matmul(body_rot_lf.transpose(-1, -2), body_rot_hf))
        add_corr_and_gate_angle(
            corr_terms,
            gate_terms,
            metrics,
            "body_attitude",
            body_delta_rot_gated,
            output["pred_delta_rot_body"],
            body_true_delta_rot,
            body_att_gate,
        )
    if len(body_pos) == 3:
        metrics["body_gate_x"] = body_gate[..., 0].mean()
        metrics["body_gate_y"] = body_gate[..., 1].mean()
        metrics["body_gate_z"] = body_gate[..., 2].mean()

    body_vel_pred = pred_body_raw[..., body_vel] if len(body_vel) == 3 else pred_body_raw.new_zeros(*pred_body_raw.shape[:2], 3)
    true_body_pos = true_body_raw[..., body_pos] if len(body_pos) == 3 else true_body_raw.new_zeros(*true_body_raw.shape[:2], 3)
    body_delta_target = true_body_pos - lf_body_pos
    rocker_pred: Dict[str, Dict[str, torch.Tensor]] = {}
    wheel_pred_pos: Dict[int, torch.Tensor] = {}

    for name in ROCKER_NAMES:
        lf_key = f"lf_rocker_{name}_pose_current"
        hf_key = f"hf_rocker_{name}_pose"
        if lf_key not in batch or hf_key not in batch:
            continue
        lf_pose = batch[lf_key].float()
        hf_pose = batch[hf_key].float()
        delta_local = output[f"pred_rocker_{name}_local_delta_pos"]
        pred_pos, delta_local = reconstruct_rocker_position_with_body_delta(
            lf_pose,
            body_global_delta,
            delta_local,
            body_rot_pred,
        )
        pred_pos_ungated = pred_pos
        rocker_pos_gate = output.get(f"gate_rocker_{name}_pos", pred_pos.new_ones(pred_pos.shape))
        pred_pos = lf_pose[..., 0:3].float() + rocker_pos_gate * (pred_pos_ungated - lf_pose[..., 0:3].float())
        rocker_losses.append(F.smooth_l1_loss(pred_pos, hf_pose[..., 0:3]))
        rocker_rmse_terms.append(torch.sqrt(((pred_pos - hf_pose[..., 0:3]) ** 2).mean()))
        rocker_local_target = (hf_pose[..., 0:3] - lf_pose[..., 0:3]) - body_delta_target
        metrics[f"rocker_{name}_local_xyz_abs"] = delta_local.abs().mean()
        metrics[f"rocker_{name}_local_xyz_rmse"] = torch.sqrt(((delta_local - rocker_local_target) ** 2).mean())

        lf_rot = quat_to_matrix_torch(lf_pose[..., 3:7])
        hf_rot = quat_to_matrix_torch(hf_pose[..., 3:7])
        rocker_att_gate = output.get(f"gate_rocker_{name}_att", output[f"pred_rocker_{name}_delta_rot"].new_ones(output[f"pred_rocker_{name}_delta_rot"].shape))
        rocker_delta_rot_gated = rocker_att_gate * output[f"pred_rocker_{name}_delta_rot"]
        pred_rot = compose_part_rotation(body_rot_pred, body_rot_lf, lf_rot, rocker_delta_rot_gated)
        att_angle = so3_geodesic_angle(pred_rot, hf_rot)
        rocker_att_losses.append(smooth_l1_zero(att_angle))
        rocker_att_deg = att_angle.mean() * (180.0 / math.pi)
        rocker_att_deg_terms.append(rocker_att_deg)
        metrics[f"attitude/rocker_{name}_deg"] = rocker_att_deg
        add_no_harm_raw(
            no_harm_terms,
            metrics,
            f"rocker_{name}_pos",
            pred_pos,
            lf_pose[..., 0:3],
            hf_pose[..., 0:3],
            pred_pos.new_ones(3),
        )
        add_corr_and_gate_raw(
            corr_terms,
            gate_terms,
            metrics,
            f"rocker_{name}_pos",
            pred_pos,
            pred_pos_ungated,
            lf_pose[..., 0:3],
            hf_pose[..., 0:3],
            pred_pos.new_ones(3),
            rocker_pos_gate,
        )
        add_no_harm_angle(
            no_harm_terms,
            metrics,
            f"rocker_{name}_attitude",
            att_angle,
            so3_geodesic_angle(lf_rot, hf_rot),
        )
        rocker_true_delta_rot = matrix_to_rotvec(torch.matmul(lf_rot.transpose(-1, -2), hf_rot))
        add_corr_and_gate_angle(
            corr_terms,
            gate_terms,
            metrics,
            f"rocker_{name}_attitude",
            rocker_delta_rot_gated,
            output[f"pred_rocker_{name}_delta_rot"],
            rocker_true_delta_rot,
            rocker_att_gate,
        )
        rocker_pred[name] = {"pos": pred_pos, "rot": pred_rot}

    for i in WHEEL_IDS:
        pred_wheel_scaled_ungated, pred_wheel_raw_ungated, pred_wheel_delta_raw = reconstruct_wheel_kin_with_body_delta(
            output[f"pred_res_wheel{i}_kin"],
            batch[f"lf_wheel{i}_kin_current"],
            body_global_delta,
            spec.res_groups.wheel_kin_cols[i],
            spec.target_groups.wheel_kin_cols[i],
            f"hf_wheel{i}_kin",
            f"res_wheel{i}_kin",
            scaler,
            body_rot_pred=body_rot_pred,
        )
        true_wheel_raw = inverse_transform_tensor(batch[f"hf_wheel{i}_kin"], scaler, f"hf_wheel{i}_kin")
        wheel_cols = spec.target_groups.wheel_kin_cols[i]
        wpos = suffix_indices(wheel_cols, ["pos_x", "pos_y", "pos_z"])
        omega_i = omega_index(wheel_cols)
        pred_wheel_raw = pred_wheel_raw_ungated
        wheel_pos_gate = output.get(f"gate_wheel{i}_pos", pred_wheel_raw.new_ones(*pred_wheel_raw.shape[:2], 3))
        wheel_omega_gate = output.get(f"gate_wheel{i}_omega", pred_wheel_raw.new_ones(*pred_wheel_raw.shape[:2], 1))
        if len(wpos) == 3:
            pred_wheel_raw = apply_axis_gate(pred_wheel_raw, batch[f"lf_wheel{i}_kin_current"], wpos, wheel_pos_gate)
            wheel_gate_xyz_all.append(wheel_pos_gate)
        if omega_i is not None:
            pred_wheel_raw = apply_axis_gate(pred_wheel_raw, batch[f"lf_wheel{i}_kin_current"], [omega_i], wheel_omega_gate)
            wheel_gate_omega_all.append(wheel_omega_gate)
        pred_wheel_scaled = transform_tensor(pred_wheel_raw, scaler, f"hf_wheel{i}_kin")
        add_no_harm_raw(
            no_harm_terms,
            metrics,
            f"wheel{i}_kin",
            pred_wheel_raw,
            batch[f"lf_wheel{i}_kin_current"],
            true_wheel_raw,
            scaler_std_tensor(scaler, f"hf_wheel{i}_kin", pred_wheel_raw),
        )
        add_corr_and_gate_raw(
            corr_terms,
            gate_terms,
            metrics,
            f"wheel{i}_kin",
            pred_wheel_raw,
            pred_wheel_raw_ungated,
            batch[f"lf_wheel{i}_kin_current"],
            true_wheel_raw,
            scaler_std_tensor(scaler, f"hf_wheel{i}_kin", pred_wheel_raw),
        )
        add_gate_only_raw(
            gate_terms,
            metrics,
            f"wheel{i}_pos",
            wheel_pos_gate,
            pred_wheel_raw_ungated,
            batch[f"lf_wheel{i}_kin_current"],
            true_wheel_raw,
            wpos,
        )
        add_gate_only_raw(
            gate_terms,
            metrics,
            f"wheel{i}_omega",
            wheel_omega_gate,
            pred_wheel_raw_ungated,
            batch[f"lf_wheel{i}_kin_current"],
            true_wheel_raw,
            [omega_i] if omega_i is not None else [],
        )
        if len(wpos) == 3:
            lp, _ = axis_loss(pred_wheel_scaled[..., wpos], batch[f"hf_wheel{i}_kin"][..., wpos])
            wheel_losses.append(lp)
            lf_wp = batch[f"lf_wheel{i}_kin_current"][..., wpos].float()
            true_wp = true_wheel_raw[..., wpos]
            wheel_local_target = (true_wp - lf_wp) - body_delta_target
            std_wp = torch.as_tensor([scaler.scalers[f"hf_wheel{i}_kin"].std_[j] for j in wpos], device=pred_wheel_raw.device).clamp_min(1e-6)
            pred_local = local_xyz_delta(pred_wheel_delta_raw, wpos)
            wheel_local_losses.append(axis_loss(pred_local / std_wp, wheel_local_target / std_wp)[0])
            wheel_local_regs.append((pred_local / std_wp).pow(2).mean())
            wheel_local_rmse_terms.append(torch.sqrt(((pred_local - wheel_local_target) ** 2).mean()))
            wheel_local_abs_terms.append(pred_local.abs().mean())
            wheel_pred_pos[i] = pred_wheel_raw[..., wpos]
            pose_lf_key = f"lf_wheel{i}_pose_current"
            pose_hf_key = f"hf_wheel{i}_pose"
            if pose_lf_key in batch and pose_hf_key in batch:
                lf_wheel_pose = batch[pose_lf_key].float()
                hf_wheel_pose = batch[pose_hf_key].float()
                lf_wheel_rot = quat_to_matrix_torch(lf_wheel_pose[..., 3:7])
                hf_wheel_rot = quat_to_matrix_torch(hf_wheel_pose[..., 3:7])
                wheel_att_gate = output.get(f"gate_wheel{i}_att", output[f"pred_wheel{i}_delta_rot"].new_ones(output[f"pred_wheel{i}_delta_rot"].shape))
                wheel_delta_rot_gated = wheel_att_gate * output[f"pred_wheel{i}_delta_rot"]
                pred_wheel_rot = compose_part_rotation(body_rot_pred, body_rot_lf, lf_wheel_rot, wheel_delta_rot_gated)
                wheel_att_angle = so3_geodesic_angle(pred_wheel_rot, hf_wheel_rot)
                wheel_att_losses.append(smooth_l1_zero(wheel_att_angle))
                wheel_att_deg_terms.append(wheel_att_angle.mean() * (180.0 / math.pi))
                metrics[f"attitude/wheel{i}_deg"] = wheel_att_angle.mean() * (180.0 / math.pi)
                add_no_harm_angle(
                    no_harm_terms,
                    metrics,
                    f"wheel{i}_attitude",
                    wheel_att_angle,
                    so3_geodesic_angle(lf_wheel_rot, hf_wheel_rot),
                )
                wheel_true_delta_rot = matrix_to_rotvec(torch.matmul(lf_wheel_rot.transpose(-1, -2), hf_wheel_rot))
                add_corr_and_gate_angle(
                    corr_terms,
                    gate_terms,
                    metrics,
                    f"wheel{i}_attitude",
                    wheel_delta_rot_gated,
                    output[f"pred_wheel{i}_delta_rot"],
                    wheel_true_delta_rot,
                    wheel_att_gate,
                )
        if omega_i is not None:
            wheel_losses.append(F.huber_loss(pred_wheel_scaled[..., omega_i:omega_i + 1], batch[f"hf_wheel{i}_kin"][..., omega_i:omega_i + 1]))

        contact_cols = spec.target_groups.wheel_contact_cols[i]
        force_idx = suffix_indices(contact_cols, ["Fx", "Fy", "Fz"])
        true_contact_raw = inverse_transform_tensor(batch[f"hf_wheel{i}_contact"], scaler, f"hf_wheel{i}_contact")
        lf_contact_raw = batch[f"lf_wheel{i}_contact_current"].float()
        input_contact_cols = spec.input_groups.wheel_contact_cols[i]
        input_contact_raw = inverse_transform_tensor(batch[f"wheel{i}_contact"], scaler, f"wheel{i}_contact")
        sink_i = suffix_indices(input_contact_cols, ["sinkage"])
        in_contact_i = suffix_indices(input_contact_cols, ["in_contact"])
        sinkage_lf = input_contact_raw[:, -1:, sink_i[0]] if sink_i else lf_contact_raw.new_zeros(*lf_contact_raw.shape[:2])
        in_contact = input_contact_raw[:, -1:, in_contact_i[0]] if in_contact_i else None
        wheel_z_lf = batch[f"lf_wheel{i}_kin_current"][..., wpos[2]] if len(wpos) == 3 else sinkage_lf.new_zeros(sinkage_lf.shape)
        wheel_z_pred = pred_wheel_raw[..., wpos[2]] if len(wpos) == 3 else wheel_z_lf
        wheel_pos_pred = pred_wheel_raw[..., wpos] if len(wpos) == 3 else None
        omega_pred = pred_wheel_raw[..., omega_i] if omega_i is not None else pred_wheel_raw.new_zeros(pred_wheel_raw.shape[:2])

        phy_loss = wheel_terrain_force(
            omega=omega_pred.detach(),
            body_velocity=body_vel_pred.detach(),
            wheel_z_pred=wheel_z_pred.detach(),
            wheel_z_lf=wheel_z_lf,
            sinkage_lf=sinkage_lf,
            wheel_pos_pred=wheel_pos_pred.detach() if wheel_pos_pred is not None else None,
            in_contact=in_contact,
            terrain_params=build_terrain_params(batch, i),
            params=TerramechanicsParams(sinkage_max=args.sinkage_max),
        )
        phy_dyn = wheel_terrain_force(
            omega=omega_pred,
            body_velocity=body_vel_pred,
            wheel_z_pred=wheel_z_pred,
            wheel_z_lf=wheel_z_lf,
            sinkage_lf=sinkage_lf,
            wheel_pos_pred=wheel_pos_pred,
            in_contact=in_contact,
            terrain_params=build_terrain_params(batch, i),
            params=TerramechanicsParams(sinkage_max=args.sinkage_max),
        )
        f_phy = phy_loss["force"]
        f_phy_dyn = phy_dyn["force"]
        sinkage_pred = phy_loss.get("sinkage")
        if isinstance(sinkage_pred, torch.Tensor):
            sinkage_lf_all.append(sinkage_lf)
            sinkage_pred_all.append(sinkage_pred)
            metrics[f"wheel{i}_sinkage_lf_mean"] = sinkage_lf.mean()
            metrics[f"wheel{i}_sinkage_pred_mean"] = sinkage_pred.mean()
            metrics[f"wheel{i}_sinkage_delta_mean"] = (sinkage_pred - sinkage_lf).mean()
        f_true = true_contact_raw[..., force_idx] if len(force_idx) == 3 else torch.zeros_like(f_phy)
        f_phy_ref = batch.get(f"wheel{i}_force_phy_ref", f_phy.detach())
        f_delta_target = batch.get(f"wheel{i}_force_delta_target", f_true - f_phy_ref)
        f_delta = output[f"pred_force_delta_wheel{i}"]
        f_raw = f_phy + f_delta
        f_raw_dyn = f_phy_dyn + f_delta
        force_gate = output.get(f"gate_force_wheel{i}", f_raw.new_ones(f_raw.shape))
        lf_force = lf_contact_raw[..., force_idx] if len(force_idx) == 3 else torch.zeros_like(f_raw)
        f_pred = lf_force.float() + force_gate * (f_raw - lf_force.float())
        f_pred_dyn = lf_force.float() + force_gate * (f_raw_dyn - lf_force.float())
        force_gate_all.append(force_gate)
        mask = contact_mask(batch, spec, i, f_pred)
        sigma_f = torch.as_tensor(
            [scaler.scalers[f"hf_wheel{i}_contact"].std_[j] for j in force_idx],
            device=f_pred.device,
            dtype=f_pred.dtype,
        ).clamp_min(1e-6) if len(force_idx) == 3 else f_pred.new_ones(3)
        force_losses.append(axis_loss(f_pred / sigma_f, f_true / sigma_f, mask)[0])
        phy_state_losses.append(axis_loss(f_phy_dyn / sigma_f, f_phy_ref / sigma_f, mask)[0])
        force_delta_losses.append(axis_loss(f_delta / sigma_f, f_delta_target / sigma_f, mask)[0])
        residual_regs.append(((f_delta / sigma_f) ** 2).mean())
        f_pred_all.append(f_pred_dyn)
        f_phy_pred_all.append(f_phy_dyn)
        f_phy_ref_all.append(f_phy_ref)
        f_true_all.append(f_true)
        f_delta_all.append(f_delta)
        f_delta_target_all.append(f_delta_target)

        if len(force_idx) == 3:
            add_corr_and_gate_raw(
                corr_terms,
                gate_terms,
                metrics,
                f"wheel{i}_force",
                f_pred,
                f_raw,
                lf_force,
                f_true,
                sigma_f,
                force_gate,
                mask=mask,
            )
            add_no_harm_raw(
                no_harm_terms,
                metrics,
                f"wheel{i}_force",
                f_pred,
                lf_contact_raw[..., force_idx],
                f_true,
                sigma_f,
                mask,
            )
        metrics[f"wheel{i}_Fx_rmse"] = torch.sqrt(((f_pred[..., 0] - f_true[..., 0]) ** 2).mean())
        metrics[f"wheel{i}_Fy_rmse"] = torch.sqrt(((f_pred[..., 1] - f_true[..., 1]) ** 2).mean())
        metrics[f"wheel{i}_Fz_rmse"] = torch.sqrt(((f_pred[..., 2] - f_true[..., 2]) ** 2).mean())

    l_wheel_state = torch.stack(wheel_losses).mean() if wheel_losses else pred_body_scaled.new_tensor(0.0)
    l_rocker_pos = torch.stack(rocker_losses).mean() if rocker_losses else pred_body_scaled.new_tensor(0.0)
    l_state = torch.stack([l_body_p, l_body_v, l_wheel_state, l_rocker_pos]).mean()
    l_wheel_local = torch.stack(wheel_local_losses).mean() if wheel_local_losses else pred_body_scaled.new_tensor(0.0)
    l_assembly_distance, l_assembly_joint, assembly_metrics = compute_assembly_losses(
        batch,
        pred_body_raw[..., body_pos] if len(body_pos) == 3 else pred_body_raw.new_zeros(*pred_body_raw.shape[:2], 3),
        body_rot_pred,
        rocker_pred,
        wheel_pred_pos,
    )
    metrics.update(assembly_metrics)
    l_assembly = l_assembly_distance + l_assembly_joint
    l_rocker_att = torch.stack(rocker_att_losses).mean() if rocker_att_losses else pred_body_scaled.new_tensor(0.0)
    l_wheel_att = torch.stack(wheel_att_losses).mean() if wheel_att_losses else pred_body_scaled.new_tensor(0.0)
    l_attitude = torch.stack([l_body_att, l_rocker_att, l_wheel_att]).mean()
    l_force = torch.stack(force_losses).mean() if force_losses else pred_body_scaled.new_tensor(0.0)
    l_phy_state = torch.stack(phy_state_losses).mean() if phy_state_losses else pred_body_scaled.new_tensor(0.0)
    l_force_delta = torch.stack(force_delta_losses).mean() if force_delta_losses else pred_body_scaled.new_tensor(0.0)
    l_reg = torch.stack(residual_regs).mean() if residual_regs else pred_body_scaled.new_tensor(0.0)
    l_wheel_local_reg = torch.stack(wheel_local_regs).mean() if wheel_local_regs else pred_body_scaled.new_tensor(0.0)

    l_kin = pred_body_scaled.new_tensor(0.0)
    l_dyn_cons = pred_body_scaled.new_tensor(0.0)
    if len(body_pos) == 3 and len(body_vel) == 3:
        prev = prev_body_raw.float()
        if prev.ndim == 2:
            prev = prev.unsqueeze(1)
        prev_p = prev[..., body_pos]
        prev_v = prev[..., body_vel]
        dt_view = dt.view(-1, 1, 1).to(device=pred_body_raw.device, dtype=pred_body_raw.dtype).clamp_min(1e-6)
        dp_pred = pred_body_raw[..., body_pos] - prev_p
        dp_kin = 0.5 * (prev_v + body_vel_pred) * dt_view
        l_kin = axis_loss(dp_pred, dp_kin)[0]
        a_kin = (body_vel_pred - prev_v) / dt_view
        sum_force = torch.stack(f_pred_all, dim=0).sum(dim=0) if f_pred_all else torch.zeros_like(a_kin)
        gravity = sum_force.new_tensor([0.0, 0.0, -9.81]).view(1, 1, 3)
        a_force = sum_force / max(args.mass, 1e-6) + gravity
        l_dyn_cons = axis_loss(a_kin, a_force)[0]

    l_no_harm = torch.stack(no_harm_terms).mean() if no_harm_terms else pred_body_scaled.new_tensor(0.0)
    l_corr = torch.stack(corr_terms).mean() if corr_terms else pred_body_scaled.new_tensor(0.0)
    l_gate = torch.stack(gate_terms).mean() if gate_terms else pred_body_scaled.new_tensor(0.0)
    if wheel_gate_xyz_all:
        wheel_gate_xyz = torch.cat(wheel_gate_xyz_all, dim=-2)
        metrics["wheel_gate_x"] = wheel_gate_xyz[..., 0].mean()
        metrics["wheel_gate_y"] = wheel_gate_xyz[..., 1].mean()
        metrics["wheel_gate_z"] = wheel_gate_xyz[..., 2].mean()
        metrics["wheel_gate_mean"] = wheel_gate_xyz.mean()
    if wheel_gate_omega_all:
        metrics["wheel_omega_gate_mean"] = torch.cat(wheel_gate_omega_all, dim=-2).mean()
    if force_gate_all:
        force_gate = torch.cat(force_gate_all, dim=-2)
        metrics["force_gate_Fx"] = force_gate[..., 0].mean()
        metrics["force_gate_Fy"] = force_gate[..., 1].mean()
        metrics["force_gate_Fz"] = force_gate[..., 2].mean()
        metrics["force_gate_mean"] = force_gate.mean()

    losses.update(metrics)
    losses["state"] = l_state
    losses["wheel_state"] = l_wheel_state
    losses["rocker_pos"] = l_rocker_pos
    losses["wheel_local"] = l_wheel_local
    losses["attitude"] = l_attitude
    losses["attitude_body"] = l_body_att
    losses["attitude_rocker"] = l_rocker_att
    losses["attitude_wheel"] = l_wheel_att
    losses["assembly"] = l_assembly
    losses["assembly_distance"] = l_assembly_distance
    losses["assembly_joint"] = l_assembly_joint
    losses["force"] = l_force
    losses["phy_state"] = l_phy_state
    losses["force_delta"] = l_force_delta
    losses["kin"] = l_kin
    losses["dyn"] = l_dyn_cons
    losses["residual_reg"] = l_reg
    losses["wheel_local_reg"] = l_wheel_local_reg
    losses["body_delta_reg"] = l_body_delta_reg
    losses["body_delta_smooth"] = l_body_delta_smooth
    losses["no_harm"] = l_no_harm
    losses["harm_loss"] = l_no_harm
    losses["corr_loss"] = l_corr
    losses["gate_loss"] = l_gate
    losses["weighted_wheel_local"] = args.lambda_wheel_local * l_wheel_local
    losses["weighted_attitude"] = getattr(args, "lambda_attitude", 0.2) * l_attitude
    losses["weighted_assembly"] = args.lambda_assembly * l_assembly
    losses["weighted_wheel_local_reg"] = args.lambda_wheel_local_reg * l_wheel_local_reg
    losses["weighted_body_delta_reg"] = getattr(args, "lambda_body_delta_reg", 1e-3) * l_body_delta_reg
    losses["weighted_body_delta_smooth"] = getattr(args, "lambda_body_delta_smooth", 0.0) * l_body_delta_smooth
    losses["weighted_force"] = args.lambda_F * l_force
    losses["weighted_phy_state"] = args.lambda_phy_state * l_phy_state
    losses["weighted_force_delta"] = args.lambda_force_delta * l_force_delta
    losses["weighted_corr"] = args.lambda_corr * l_corr
    losses["weighted_gate"] = args.lambda_gate * l_gate
    losses["weighted_kin"] = args.lambda_kin * l_kin
    losses["weighted_dyn"] = args.lambda_dyn * l_dyn_cons
    losses["weighted_reg"] = args.lambda_reg * l_reg
    losses["weighted_harm"] = args.lambda_harm * l_no_harm
    losses["body_pos_xyz_rmse"] = torch.sqrt(((pred_body_raw[..., body_pos] - true_body_raw[..., body_pos]) ** 2).mean()) if len(body_pos) == 3 else l_state.detach()
    losses["body_vel_xyz_rmse"] = torch.sqrt(((body_vel_pred - true_body_raw[..., body_vel]) ** 2).mean()) if len(body_vel) == 3 else l_state.detach()
    losses["position/rocker_rmse"] = torch.stack(rocker_rmse_terms).mean() if rocker_rmse_terms else l_rocker_pos.detach()
    losses["attitude/body_deg"] = body_att_angle.mean() * (180.0 / math.pi)
    losses["attitude/rocker_deg"] = torch.stack(rocker_att_deg_terms).mean() if rocker_att_deg_terms else l_rocker_att.detach()
    losses["attitude/wheel_deg"] = torch.stack(wheel_att_deg_terms).mean() if wheel_att_deg_terms else l_wheel_att.detach()
    losses["F_phy_pred_ref_rmse"] = torch.sqrt(((torch.cat(f_phy_pred_all, dim=-2) - torch.cat(f_phy_ref_all, dim=-2)) ** 2).mean()) if f_phy_pred_all else l_force.detach()
    losses["F_phy_pred_hf_rmse"] = torch.sqrt(((torch.cat(f_phy_pred_all, dim=-2) - torch.cat(f_true_all, dim=-2)) ** 2).mean()) if f_phy_pred_all else l_force.detach()
    losses["F_delta_rmse"] = torch.sqrt(((torch.cat(f_delta_all, dim=-2) - torch.cat(f_delta_target_all, dim=-2)) ** 2).mean()) if f_delta_all else l_force.detach()
    losses["F_phy_rmse"] = losses["F_phy_pred_hf_rmse"]
    losses["F_final_rmse"] = torch.sqrt(((torch.cat(f_pred_all, dim=-2) - torch.cat(f_true_all, dim=-2)) ** 2).mean()) if f_pred_all else l_force.detach()
    if sinkage_lf_all and sinkage_pred_all:
        sinkage_lf_cat = torch.cat(sinkage_lf_all, dim=-1)
        sinkage_pred_cat = torch.cat(sinkage_pred_all, dim=-1)
        losses["sinkage_lf_mean"] = sinkage_lf_cat.mean()
        losses["sinkage_pred_mean"] = sinkage_pred_cat.mean()
        losses["sinkage_delta_mean"] = (sinkage_pred_cat - sinkage_lf_cat).mean()
        losses["sinkage_lf_max"] = sinkage_lf_cat.max()
        losses["sinkage_pred_max"] = sinkage_pred_cat.max()
    losses["wheel_local_xyz_rmse"] = torch.stack(wheel_local_rmse_terms).mean() if wheel_local_rmse_terms else l_wheel_local.detach()
    losses["body_correction_xyz_abs"] = body_delta_pos.abs().mean() if len(body_pos) == 3 else l_state.detach()
    losses["wheel_local_xyz_abs"] = torch.stack(wheel_local_abs_terms).mean() if wheel_local_abs_terms else l_wheel_local.detach()
    losses["total"] = (
        l_state
        + losses["weighted_wheel_local"]
        + losses["weighted_attitude"]
        + losses["weighted_assembly"]
        + losses["weighted_wheel_local_reg"]
        + losses["weighted_body_delta_reg"]
        + losses["weighted_body_delta_smooth"]
        + losses["weighted_force"]
        + losses["weighted_phy_state"]
        + losses["weighted_force_delta"]
        + losses["weighted_corr"]
        + losses["weighted_gate"]
        + losses["weighted_kin"]
        + losses["weighted_dyn"]
        + losses["weighted_reg"]
        + losses["weighted_harm"]
    )
    return losses


def freeze_module(module: torch.nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


def build_z_projector(student: GraphTemporalHGTCompensationModelV3, teacher: Optional[GraphTemporalHGTCompensationModelV3], device: torch.device) -> torch.nn.Module:
    teacher_dim = int(teacher.z_dim) if teacher is not None else int(student.z_dim)
    student_dim = int(student.z_dim)
    if student_dim == teacher_dim:
        return torch.nn.Identity().to(device)
    return torch.nn.Linear(student_dim, teacher_dim).to(device)


def forward_stage(
    model: GraphTemporalHGTCompensationModelV3,
    teacher_model: Optional[GraphTemporalHGTCompensationModelV3],
    z_projector: torch.nn.Module,
    batch,
    args,
):
    teacher_t_index = int(args.history_len)
    if args.train_stage == "teacher":
        out = model(batch, input_prefix="teacher", current_index=teacher_t_index)
        out["z_teacher"] = out["z"]
        return out
    if teacher_model is None:
        raise RuntimeError("student 阶段必须提供已冻结的 teacher_model")
    with torch.no_grad():
        teacher_out = teacher_model(batch, input_prefix="teacher", current_index=teacher_t_index)
        z_teacher = teacher_out["z"].detach()
    out = model(batch, input_prefix="student", current_index=teacher_t_index)
    z_student = out["z"]
    z_student_projected = z_projector(z_student)
    distill = F.smooth_l1_loss(z_student_projected, z_teacher)
    cosine = F.cosine_similarity(
        z_student_projected.reshape(z_student_projected.shape[0], -1),
        z_teacher.reshape(z_teacher.shape[0], -1),
        dim=-1,
    ).mean()
    out["z_teacher"] = z_teacher
    out["z_student"] = z_student
    out["z_student_projected"] = z_student_projected
    out["distill_z"] = distill
    out["z_teacher_norm"] = z_teacher.norm(dim=-1).mean()
    out["z_student_norm"] = z_student.norm(dim=-1).mean()
    out["z_cosine_similarity"] = cosine
    return out


def run_epoch(model, teacher_model, z_projector, loader, optimizer, device, spec, scaler, args, train: bool, epoch: int):
    model.train(train)
    if teacher_model is not None:
        teacher_model.eval()
    z_projector.train(train and args.train_stage == "student")
    totals: Dict[str, float] = {}
    n_batches = 0
    phase_name = "train" if train else "val"
    total_batches = len(loader) if loader is not None else 0
    iterator = loader
    progress = None
    if tqdm is not None and bool(args.progress_bar):
        progress = tqdm(
            loader,
            total=total_batches,
            desc=phase_name,
            dynamic_ncols=True,
            leave=False,
            file=sys.stdout,
        )
        iterator = progress
    for batch in iterator:
        batch = move_batch_to_device(batch, device)
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            out = forward_stage(model, teacher_model, z_projector, batch, args)
            losses = compute_losses_v3(batch, out, spec, scaler, args)
            if args.train_stage == "student":
                losses["distill_z"] = out["distill_z"]
                losses["weighted_distill_z"] = args.lambda_distill_z * out["distill_z"]
                losses["z_teacher_norm"] = out["z_teacher_norm"]
                losses["z_student_norm"] = out["z_student_norm"]
                losses["z_cosine_similarity"] = out["z_cosine_similarity"]
                losses["total"] = losses["total"] + losses["weighted_distill_z"]
            n_batches += 1
            if args.debug_finite:
                problems = collect_nonfinite_tensors("loss", losses)
                problems.extend(collect_nonfinite_tensors("output", out))
                if problems:
                    if progress is not None:
                        progress.close()
                    handle_nonfinite_debug(
                        args=args,
                        epoch=epoch,
                        phase_name=phase_name,
                        batch_idx=n_batches,
                        batch=batch,
                        output=out,
                        losses=losses,
                        problems=problems,
                        stage="forward",
                    )
            if train:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                if args.debug_finite:
                    problems = collect_nonfinite_grads(model, z_projector if args.train_stage == "student" else None)
                    if problems:
                        if progress is not None:
                            progress.close()
                        handle_nonfinite_debug(
                            args=args,
                            epoch=epoch,
                            phase_name=phase_name,
                            batch_idx=n_batches,
                            batch=batch,
                            output=out,
                            losses=losses,
                            problems=problems,
                            stage="backward",
                        )
                grad_params = [p for p in list(model.parameters()) + list(z_projector.parameters()) if p.requires_grad]
                grad_norm = torch.nn.utils.clip_grad_norm_(grad_params, args.grad_clip)
                if args.debug_finite and isinstance(grad_norm, torch.Tensor) and not bool(torch.isfinite(grad_norm.detach())):
                    if progress is not None:
                        progress.close()
                    handle_nonfinite_debug(
                        args=args,
                        epoch=epoch,
                        phase_name=phase_name,
                        batch_idx=n_batches,
                        batch=batch,
                        output=out,
                        losses={**losses, "grad_norm": grad_norm},
                        problems=[f"grad_norm: {tensor_finite_summary(grad_norm)}"],
                        stage="grad_clip",
                    )
                optimizer.step()
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                totals[k] = totals.get(k, 0.0) + float(v.detach().cpu())
        if args.progress_interval > 0 and (n_batches == 1 or n_batches % args.progress_interval == 0 or n_batches == total_batches):
            current = {
                k: float(v.detach().cpu())
                for k, v in losses.items()
                if isinstance(v, torch.Tensor) and k in LOSS_COMPONENT_KEYS
            }
            if progress is not None:
                progress.set_postfix(compact_metrics(current), refresh=True)
            else:
                print(
                    f"[V3] {phase_name} batch {n_batches}/{total_batches} {format_metrics(current)}",
                    flush=True,
                )
    if progress is not None:
        progress.close()
    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def assert_all_finite(label: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value.detach()).all():
        raise RuntimeError(f"{label} contains NaN/Inf: {tensor_finite_summary(value)}")


def run_single_batch_smoke_test(model, teacher_model, z_projector, loader, device, spec, scaler, args) -> None:
    model.train()
    if teacher_model is not None:
        teacher_model.eval()
    z_projector.train(args.train_stage == "student")
    batch = move_batch_to_device(next(iter(loader)), device)
    out = forward_stage(model, teacher_model, z_projector, batch, args)

    for name in ROCKER_NAMES:
        delta = out[f"pred_rocker_{name}_local_delta_pos"]
        if delta.shape[-1] != 3:
            raise RuntimeError(f"rocker {name} local delta must be xyz, got shape={tuple(delta.shape)}")
    for i in WHEEL_IDS:
        pred_res = out[f"pred_res_wheel{i}_kin"]
        pos_idx = suffix_indices(spec.target_groups.wheel_kin_cols[i], ["pos_x", "pos_y", "pos_z"])
        pred_delta_raw = residual_raw_like_target(
            pred_res,
            spec.res_groups.wheel_kin_cols[i],
            spec.target_groups.wheel_kin_cols[i],
            f"res_wheel{i}_kin",
            scaler,
        )
        local_delta = local_xyz_delta(pred_delta_raw, pos_idx)
        if local_delta.shape[-1] != 3:
            raise RuntimeError(f"wheel {i} local delta must be xyz, got shape={tuple(local_delta.shape)}")

    losses = compute_losses_v3(batch, out, spec, scaler, args)
    if args.train_stage == "student":
        losses["distill_z"] = out["distill_z"]
        losses["weighted_distill_z"] = args.lambda_distill_z * out["distill_z"]
        losses["total"] = losses["total"] + losses["weighted_distill_z"]
    for key in [
        "total",
        "state",
        "attitude",
        "assembly",
        "assembly_distance",
        "assembly_joint",
        "force",
        "phy_state",
        "force_delta",
        "dyn",
    ]:
        if key in losses:
            assert_all_finite(f"smoke loss {key}", losses[key])

    if "lf_body_pose_current" in batch:
        body_rot_lf = quat_to_matrix_torch(batch["lf_body_pose_current"][..., 3:7])
        body_rot_pred = torch.matmul(body_rot_lf, rotvec_to_matrix(out["pred_delta_rot_body"]))
        eye = torch.eye(3, device=device, dtype=body_rot_pred.dtype).view(1, 1, 3, 3)
        ortho_err = torch.matmul(body_rot_pred.transpose(-1, -2), body_rot_pred) - eye
        det = torch.linalg.det(body_rot_pred)
        assert_all_finite("smoke body rotation matrix", body_rot_pred)
        if float(ortho_err.detach().abs().max().cpu()) > 5e-4:
            raise RuntimeError(f"body rotation matrix orthogonality check failed: max_err={float(ortho_err.detach().abs().max().cpu()):.6g}")
        if float((det.detach() - 1.0).abs().max().cpu()) > 5e-4:
            raise RuntimeError(f"body rotation matrix determinant check failed: max_err={float((det.detach() - 1.0).abs().max().cpu()):.6g}")

    model.zero_grad(set_to_none=True)
    z_projector.zero_grad(set_to_none=True)
    losses["total"].backward()
    problems = collect_nonfinite_grads(model, z_projector if args.train_stage == "student" else None)
    if problems:
        raise RuntimeError("smoke backward found non-finite gradients:\n" + "\n".join(problems[:20]))
    model.zero_grad(set_to_none=True)
    z_projector.zero_grad(set_to_none=True)
    print("[V3] forward + loss + backward smoke test passed", flush=True)


@torch.no_grad()
def run_dummy_forward_check(model, group_dims, history_len, teacher_future_len, device):
    batch = {}
    seq_len = history_len + 1
    teacher_seq_len = history_len + teacher_future_len + 1
    for key in ["system", "body", *ROCKER_NAMES, *[f"wheel{i}_kin" for i in WHEEL_IDS], *[f"wheel{i}_contact" for i in WHEEL_IDS]]:
        batch[key] = torch.zeros(2, seq_len, int(group_dims.get(key, 0)), device=device)
        batch[f"student_{key}"] = batch[key]
        batch[f"teacher_{key}"] = torch.zeros(2, teacher_seq_len, int(group_dims.get(key, 0)), device=device)
    prefix = "teacher" if model.role == "teacher" else "student"
    out = model(batch, input_prefix=prefix, current_index=history_len)
    assert out["pred_res_body"].shape == (2, 1, int(group_dims["res_body"]))
    assert out["pred_delta_rot_body"].shape == (2, 1, 3)
    for name in ROCKER_NAMES:
        assert out[f"pred_rocker_{name}_local_delta_pos"].shape == (2, 1, 3)
        assert out[f"pred_rocker_{name}_delta_rot"].shape == (2, 1, 3)
    for i in WHEEL_IDS:
        assert out[f"pred_wheel{i}_delta_rot"].shape == (2, 1, 3)
    assert out["pred_force_delta_wheel0"].shape == (2, 1, 3)
    assert out["z"].shape == (2, 1, int(model.z_dim))
    print(f"V3 {model.role} dummy forward check passed")


def resolve_init_checkpoint(stage: str, init_ckpt: Optional[str], save_dir: str) -> Optional[str]:
    if init_ckpt:
        return init_ckpt
    if stage != "student":
        return None
    candidates = list((Path(save_dir) / "teacher").glob("*/best_output.pt"))
    if not candidates:
        candidates = list((Path(save_dir) / "teacher").glob("*/best_safe.pt"))
    if not candidates:
        candidates = list((Path(save_dir) / "teacher").glob("*/best_total.pt"))
    if not candidates:
        candidates = list((Path(save_dir) / "teacher").glob("*/best_model.pt"))
    if not candidates:
        raise FileNotFoundError("student 阶段需要 --init_ckpt，或先在 save_dir/teacher 下训练 teacher")
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", type=str, default=str(ROOT / "Feature_Selection" / "DataSet"))
    parser.add_argument("--merged_csv", type=str, default=str(ROOT / "Feature_Selection" / "DataSet" / "merged_error_dataset.csv"))
    parser.add_argument("--save_dir", type=str, default=str(ROOT / "results_v3"))
    parser.add_argument("--log_dir", type=str, default=None, help="TensorBoard 日志目录；默认保存到当前 run 的 tensorboard 子目录")
    parser.add_argument("--seq_len", type=int, default=None, help="兼容旧参数；未显式设置 history_len 时使用 seq_len-1")
    parser.add_argument("--history_len", type=int, default=None)
    parser.add_argument("--teacher_future_len", type=int, default=3)
    parser.add_argument("--pred_horizon", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=192)
    parser.add_argument("--graph_layers", type=int, default=3)
    parser.add_argument("--tcn_dim", type=int, default=256)
    parser.add_argument("--lstm_dim", type=int, default=192)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--prefetch_factor", type=int, default=2, help="num_workers>0 时每个 worker 预取 batch 数；降低可减少共享内存占用")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_stage", choices=["teacher", "student"], default="teacher")
    parser.add_argument("--init_ckpt", type=str, default=None)
    parser.add_argument("--lambda_distill_z", type=float, default=1.0)
    parser.add_argument("--lambda_F", type=float, default=1.0)
    parser.add_argument("--lambda_wheel_local", type=float, default=0.75)
    parser.add_argument("--lambda_attitude", type=float, default=0.2)
    parser.add_argument("--lambda_assembly", type=float, default=0.2)
    parser.add_argument("--lambda_wheel_local_reg", type=float, default=1e-4)
    parser.add_argument("--lambda_body_delta_reg", type=float, default=1e-3)
    parser.add_argument("--lambda_body_delta_smooth", type=float, default=0.0)
    parser.add_argument("--lambda_phy_state", type=float, default=0.001)
    parser.add_argument("--lambda_force_delta", type=float, default=0.2)
    parser.add_argument("--lambda_corr", type=float, default=0.5)
    parser.add_argument("--lambda_gate", type=float, default=0.2)
    parser.add_argument("--lambda_kin", type=float, default=0.02)
    parser.add_argument("--lambda_dyn", type=float, default=0.005)
    parser.add_argument("--lambda_reg", type=float, default=5e-5)
    parser.add_argument("--lambda_harm", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.015)
    parser.add_argument("--mass", type=float, default=240.0)
    parser.add_argument("--sinkage_max", type=float, default=0.08)
    parser.add_argument("--force_lpf_alpha", type=float, default=0.15)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--progress_bar", action=argparse.BooleanOptionalAction, default=True, help="是否用 tqdm 进度条显示 train/val batch 进度")
    parser.add_argument("--progress_interval", type=int, default=10, help="每隔多少个 batch 刷新一次进度条 loss 组成；<=0 表示关闭")
    parser.add_argument("--debug_finite", action=argparse.BooleanOptionalAction, default=False, help="逐 batch 检查 loss/output/gradient 是否包含 NaN 或 Inf")
    parser.add_argument("--smoke_test", action=argparse.BooleanOptionalAction, default=True, help="正式训练前用一个真实 batch 执行 forward/loss/backward 检查")
    parser.add_argument("--debug_dump_dir", type=str, default=None, help="debug_finite 触发时保存异常 batch 的目录；默认保存到当前 run/debug_finite")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.history_len is None:
        args.history_len = int(args.seq_len) - 1 if args.seq_len is not None else 9

    if args.pred_horizon != 0:
        print("V3 target 始终对应当前时刻 t；pred_horizon 非 0 将由 dataset 拒绝。")
    set_seed(args.seed)
    device = torch.device(args.device)
    base_save_dir = args.save_dir
    args.init_ckpt = resolve_init_checkpoint(args.train_stage, args.init_ckpt, base_save_dir)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.save_dir = os.path.join(base_save_dir, args.train_stage, run_name)
    ensure_dir(args.save_dir)
    tb_dir = args.log_dir or os.path.join(args.save_dir, "tensorboard")
    writer = None
    if SummaryWriter is None:
        print("TensorBoard 未安装，跳过 TensorBoard 日志。可安装 tensorboard 后重新运行。")
    else:
        ensure_dir(tb_dir)
        writer = SummaryWriter(tb_dir)
        writer.add_text("run/save_dir", args.save_dir, 0)
        writer.add_text("run/train_stage", args.train_stage, 0)
        print(f"TensorBoard 日志目录: {tb_dir}")

    t0 = log_step("开始准备数据集")
    spec, scaler, _, _, _, train_ds, val_ds, _ = prepare_datasets_and_scaler(
        args.feature_dir,
        args.merged_csv,
        args.seq_len,
        args.pred_horizon,
        1,
        seed=args.seed,
        history_len=args.history_len,
        teacher_future_len=args.teacher_future_len,
        force_lpf_alpha=args.force_lpf_alpha,
        sinkage_max=args.sinkage_max,
    )
    log_step(f"数据集准备完成，train={len(train_ds)} val={len(val_ds) if val_ds is not None else 0}", t0)
    t0 = log_step("保存列配置和 scaler")
    save_column_spec_json(spec, os.path.join(args.save_dir, "column_spec_v3.json"))
    scaler.save(os.path.join(args.save_dir, "group_scaler_v3.joblib"))
    log_step("列配置和 scaler 保存完成", t0)
    t0 = log_step("初始化 V3 模型")
    group_dims = get_group_dims(spec)
    model = GraphTemporalHGTCompensationModelV3(
        group_dims=group_dims,
        node_hidden_dim=args.hidden_dim,
        graph_layers=args.graph_layers,
        tcn_hidden_dim=args.tcn_dim,
        lstm_hidden_dim=args.lstm_dim,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        role=args.train_stage,
    ).to(device)
    teacher_model = None
    z_projector = torch.nn.Identity().to(device)
    log_step("V3 模型初始化完成", t0)

    t0 = log_step("执行 dummy forward 检查")
    run_dummy_forward_check(model, group_dims, args.history_len, args.teacher_future_len, device)
    log_step("dummy forward 检查完成", t0)
    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location=device, weights_only=True)
        if args.train_stage == "teacher":
            load_compatible_state_dict(model, ckpt["model_state_dict"], "model init checkpoint")
        else:
            teacher_args = ckpt.get("args") or {}
            teacher_model = GraphTemporalHGTCompensationModelV3(
                group_dims=group_dims,
                node_hidden_dim=int(teacher_args.get("hidden_dim", args.hidden_dim)),
                graph_layers=int(teacher_args.get("graph_layers", args.graph_layers)),
                tcn_hidden_dim=int(teacher_args.get("tcn_dim", args.tcn_dim)),
                lstm_hidden_dim=int(teacher_args.get("lstm_dim", args.lstm_dim)),
                lstm_layers=int(teacher_args.get("lstm_layers", args.lstm_layers)),
                dropout=float(teacher_args.get("dropout", args.dropout)),
                role="teacher",
            ).to(device)
            load_compatible_state_dict(teacher_model, ckpt["model_state_dict"], "teacher init checkpoint")
            freeze_module(teacher_model)
            z_projector = build_z_projector(model, teacher_model, device)
            if "z_projector_state_dict" in ckpt and not isinstance(z_projector, torch.nn.Identity):
                load_compatible_state_dict(z_projector, ckpt["z_projector_state_dict"], "z projector init checkpoint")
        print(f"Loaded init checkpoint: {args.init_ckpt}")
    elif args.train_stage == "student":
        raise RuntimeError("student 阶段需要 teacher checkpoint")

    if args.train_stage == "teacher":
        z_projector = build_z_projector(model, None, device)
    params = [p for p in list(model.parameters()) + list(z_projector.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr if args.train_stage == "teacher" else min(args.lr, 5e-5), weight_decay=args.weight_decay)
    loader_kwargs = {
        "num_workers": args.num_workers,
        "collate_fn": graph_temporal_collate_fn,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs) if val_ds is not None else None
    if args.smoke_test:
        t0 = log_step("执行真实 batch forward/loss/backward smoke test")
        run_single_batch_smoke_test(model, teacher_model, z_projector, train_loader, device, spec, scaler, args)
        log_step("真实 batch smoke test 完成", t0)

    best_total = math.inf
    best_output = math.inf
    best_safe = math.inf
    best_safe_harmful = math.inf
    safe_tie_eps = 1e-4
    history = []
    for epoch in range(1, args.epochs + 1):
        epoch_t0 = log_step(f"开始 epoch {epoch}/{args.epochs}")
        tr = run_epoch(model, teacher_model, z_projector, train_loader, optimizer, device, spec, scaler, args, train=True, epoch=epoch)
        print_epoch_metrics(epoch, "train", tr)
        va = run_epoch(model, teacher_model, z_projector, val_loader, optimizer, device, spec, scaler, args, train=False, epoch=epoch) if val_loader else tr
        va = add_validation_selection_metrics(va)
        print_epoch_metrics(epoch, "val", va)
        history.append({"epoch": epoch, "train": tr, "val": va})
        write_tensorboard_scalars(writer, "train", tr, epoch)
        write_tensorboard_scalars(writer, "val", va, epoch)
        write_v3_grouped_tensorboard(writer, "train", tr, epoch)
        write_v3_grouped_tensorboard(writer, "val", va, epoch)
        write_optimizer_lrs(writer, optimizer, epoch)
        log_step(f"epoch {epoch} 完成", epoch_t0)
        total_score = va.get("total", math.inf)
        output_score = va.get("output_score", math.inf)
        improve_score = va.get("improve_score", math.inf)
        harmful_ratio = va.get("harmful_ratio", math.inf)
        if math.isfinite(float(total_score)) and float(total_score) < best_total:
            best_total = float(total_score)
            save_checkpoint(os.path.join(args.save_dir, "best_total.pt"), model, z_projector, group_dims, scaler, args, epoch, "val_total", best_total)
        if math.isfinite(float(output_score)) and float(output_score) < best_output:
            best_output = float(output_score)
            save_checkpoint(os.path.join(args.save_dir, "best_output.pt"), model, z_projector, group_dims, scaler, args, epoch, "output_score", best_output)
        if math.isfinite(float(improve_score)):
            safe_better = float(improve_score) < best_safe - safe_tie_eps
            safe_tie_better = abs(float(improve_score) - best_safe) <= safe_tie_eps and float(harmful_ratio) < best_safe_harmful
            if safe_better or safe_tie_better:
                best_safe = float(improve_score)
                best_safe_harmful = float(harmful_ratio)
                save_checkpoint(os.path.join(args.save_dir, "best_safe.pt"), model, z_projector, group_dims, scaler, args, epoch, "improve_score", best_safe)
    with open(os.path.join(args.save_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    if writer is not None:
        writer.flush()
        writer.close()
    print(
        f"V3 training finished. "
        f"Best_total={best_total:.6f} Best_output={best_output:.6f} "
        f"Best_safe={best_safe:.6f} Best_safe_harmful={best_safe_harmful:.6f}. "
        f"Saved to {args.save_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
