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

from models.data_utils_v2 import (  # noqa: E402
    ROCKER_NAMES,
    WHEEL_IDS,
    get_group_dims,
    graph_temporal_collate_fn,
    prepare_datasets_and_scaler,
    save_column_spec_json,
)
from models.differentiable_terramechanics import TerramechanicsParams, wheel_terrain_force  # noqa: E402
from models.graph_temporal_hgt_compensation_v2 import GraphTemporalHGTCompensationModelV2  # noqa: E402
from models.graph_temporal_hgt_compensation import TeacherGateNet  # noqa: E402

TEACHER_GATE_SUMMARY_DIM_V2 = 22
LOSS_COMPONENT_KEYS = (
    "total",
    "force",
    "phy_state",
    "bias",
    "dynamic",
    "dyn",
    "residual_reg",
    "weighted_force",
    "weighted_phy_state",
    "weighted_bias",
    "weighted_dynamic",
    "weighted_dyn",
    "weighted_reg",
    "F_final_rmse",
    "F_phy_pred_ref_rmse",
    "F_phy_pred_hf_rmse",
    "F_bias_rmse",
    "F_dynamic_rmse",
    "sinkage_lf_mean",
    "sinkage_pred_mean",
    "sinkage_delta_mean",
    "sinkage_lf_max",
    "sinkage_pred_max",
    "student_latent",
    "student_gate",
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


def log_step(message: str, start_time: Optional[float] = None) -> float:
    now = time.perf_counter()
    if start_time is None:
        print(f"[V2] {message}", flush=True)
    else:
        print(f"[V2] {message}: {now - start_time:.2f}s", flush=True)
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
        ("loss", ("total", "force", "phy_state", "bias", "dynamic", "dyn", "residual_reg", "body_delta_reg", "body_delta_smooth")),
        ("weighted", ("weighted_force", "weighted_phy_state", "weighted_bias", "weighted_dynamic", "weighted_dyn", "weighted_reg", "weighted_body_delta_reg", "weighted_body_delta_smooth")),
        ("force", ("F_final_rmse", "F_phy_pred_ref_rmse", "F_phy_pred_hf_rmse", "F_bias_rmse", "F_dynamic_rmse", "wheel0_Fx_rmse", "wheel0_Fy_rmse", "wheel0_Fz_rmse")),
        ("sinkage", ("sinkage_lf_mean", "sinkage_pred_mean", "sinkage_delta_mean", "sinkage_lf_max", "sinkage_pred_max")),
        ("student", ("student_latent", "student_gate")),
    ]
    print(f"\nepoch {epoch:04d} {phase}", flush=True)
    for title, keys in sections:
        text = format_metrics(metrics, keys)
        if text:
            print(f"  {title:<8} {text}", flush=True)


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


def collect_nonfinite_grads(model: torch.nn.Module, teacher_gate_net: torch.nn.Module) -> List[str]:
    bad: List[str] = []
    for module_prefix, module in [("model", model), ("teacher_gate_net", teacher_gate_net)]:
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
    print("\n[V2][finite-check] 检测到 NaN/Inf，训练已停止。", flush=True)
    print(f"[V2][finite-check] epoch={epoch} phase={phase_name} batch={batch_idx} stage={stage}", flush=True)
    context = batch_context_summary(batch)
    if context:
        print(f"[V2][finite-check] {context}", flush=True)
    for item in problems[:40]:
        print(f"[V2][finite-check] {item}", flush=True)
    if len(problems) > 40:
        print(f"[V2][finite-check] ... 还有 {len(problems) - 40} 项未显示", flush=True)
    print(f"[V2][finite-check] debug batch 已保存: {dump_path}", flush=True)
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


def _frame_mean_abs(x: torch.Tensor) -> torch.Tensor:
    return x.abs().mean(dim=-1) if x.numel() else x.new_zeros(x.shape[:2])


def _frame_std(x: torch.Tensor) -> torch.Tensor:
    return x.std(dim=-1) if x.shape[-1] > 1 else x.new_zeros(x.shape[:2])


def _frame_diff(x: torch.Tensor) -> torch.Tensor:
    out = x.new_zeros(x.shape[:2])
    if x.shape[1] > 1:
        out[:, 1:] = (x[:, 1:] - x[:, :-1]).abs().mean(dim=-1)
    return out


def _seq_summary(x: torch.Tensor) -> torch.Tensor:
    return torch.stack([_frame_mean_abs(x), _frame_std(x), _frame_diff(x)], dim=-1)


def _control_summary(x: torch.Tensor) -> torch.Tensor:
    max_abs = x.abs().amax(dim=-1) if x.numel() else x.new_zeros(x.shape[:2])
    return torch.stack([_frame_mean_abs(x), _frame_std(x), _frame_diff(x), max_abs], dim=-1)


def _target_summary(x: torch.Tensor, seq_len: int) -> torch.Tensor:
    if x.ndim == 3:
        x = x.reshape(x.shape[0], -1)
    mean_abs = x.abs().mean(dim=-1)
    std = x.std(dim=-1) if x.shape[-1] > 1 else x.new_zeros(x.shape[0])
    max_abs = x.abs().amax(dim=-1)
    return torch.stack([mean_abs, std, max_abs], dim=-1).unsqueeze(1).expand(-1, seq_len, -1)


def build_teacher_gate_summary_v2(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    seq_len = batch["body"].shape[1]
    wheel_kin_lf = torch.cat([batch[f"wheel{i}_kin"] for i in WHEEL_IDS], dim=-1)
    wheel_contact_lf = torch.cat([batch[f"wheel{i}_contact"] for i in WHEEL_IDS], dim=-1)
    res_wheel_kin = torch.cat([batch[f"res_wheel{i}_kin"] for i in WHEEL_IDS], dim=-1)
    res_wheel_contact = torch.cat([batch[f"res_wheel{i}_contact"] for i in WHEEL_IDS], dim=-1)
    parts = [
        _control_summary(batch["system"]),
        _seq_summary(batch["body"]),
        _seq_summary(wheel_kin_lf),
        _seq_summary(wheel_contact_lf),
        _target_summary(batch["res_body"], seq_len),
        _target_summary(res_wheel_kin, seq_len),
        _target_summary(res_wheel_contact, seq_len),
    ]
    return torch.cat(parts, dim=-1)


def build_group_columns(spec) -> Dict[str, List[str]]:
    cols = {"system": spec.input_groups.system_cols, "body": spec.input_groups.body_cols}
    for name in ROCKER_NAMES:
        cols[name] = spec.input_groups.rocker_cols[name]
    for i in WHEEL_IDS:
        cols[f"wheel{i}_kin"] = spec.input_groups.wheel_kin_cols[i]
        cols[f"wheel{i}_contact"] = spec.input_groups.wheel_contact_cols[i]
    return cols


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


def local_xz_delta(pred_delta_raw: torch.Tensor, pos_idx: List[int]) -> torch.Tensor:
    local = pred_delta_raw.new_zeros(*pred_delta_raw.shape[:-1], 3)
    if len(pos_idx) == 3:
        local[..., 0] = pred_delta_raw[..., pos_idx[0]]
        local[..., 2] = pred_delta_raw[..., pos_idx[2]]
    return local


def reconstruct_component_position(
    component_lf_raw: torch.Tensor,
    body_global_delta_raw: torch.Tensor,
    pred_delta_raw: torch.Tensor,
    pos_idx: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    pred_raw = component_lf_raw.clone().float()
    local_delta = local_xz_delta(pred_delta_raw, pos_idx)
    if len(pos_idx) == 3:
        # TODO:
        # 后续加入 body attitude 后，
        # local delta 必须定义在 body frame，
        # 再通过 body rotation 转换到 world frame。
        pred_raw[..., pos_idx] = component_lf_raw[..., pos_idx].float() + body_global_delta_raw + local_delta
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_delta_raw = residual_raw_like_target(pred_res_scaled, res_cols, hf_cols, res_group, scaler)
    pred_raw = lf_wheel_raw.clone().float()
    wpos = suffix_indices(hf_cols, ["pos_x", "pos_y", "pos_z"])
    omega_i = omega_index(hf_cols)
    if len(wpos) == 3:
        pred_raw, _ = reconstruct_component_position(lf_wheel_raw, body_global_delta_raw, pred_delta_raw, wpos)
    if omega_i is not None:
        pred_raw[..., omega_i] = lf_wheel_raw[..., omega_i].float() + pred_delta_raw[..., omega_i]
    return transform_tensor(pred_raw, scaler, hf_group), pred_raw, pred_delta_raw


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
        print(f"[V2] {label} 跳过 {len(skipped)} 个 shape 不匹配参数: {preview}{suffix}", flush=True)
    if unexpected:
        print(f"[V2] {label} unexpected 参数: {list(unexpected)[:8]}", flush=True)
    if missing:
        print(f"[V2] {label} 未加载参数数: {len(missing)}", flush=True)


def compute_losses_v2(batch, output, spec, scaler, args) -> Dict[str, torch.Tensor]:
    losses: Dict[str, torch.Tensor] = {}
    true_body_raw = inverse_transform_tensor(batch["hf_body"], scaler, "hf_body")
    body_pos = suffix_indices(spec.target_groups.body_cols, ["pos_x", "pos_y", "pos_z"])
    body_vel = suffix_indices(spec.target_groups.body_cols, ["vel_x", "vel_y", "vel_z"])
    dt = batch.get("dt", output["pred_res_body"].new_full((output["pred_res_body"].shape[0],), args.dt))
    prev_body_raw = batch.get("lf_body_prev_raw", batch.get("hf_body_prev_raw", batch["lf_body_current"]))
    pred_body_scaled, pred_body_raw, body_delta_raw = reconstruct_body_kinematic(
        output["pred_res_body"],
        batch["lf_body_current"],
        prev_body_raw,
        dt,
        spec.res_groups.body_cols,
        spec.target_groups.body_cols,
        scaler,
    )
    body_delta_pos = body_delta_raw[..., body_pos] if len(body_pos) == 3 else pred_body_raw.new_zeros(*pred_body_raw.shape[:2], 3)
    lf_body_pos = batch["lf_body_current"][..., body_pos].float() if len(body_pos) == 3 else pred_body_raw.new_zeros(*pred_body_raw.shape[:2], 3)
    body_global_delta = (pred_body_raw[..., body_pos] - lf_body_pos).detach() if len(body_pos) == 3 else body_delta_pos.detach()
    l_body_p, _ = axis_loss(pred_body_scaled[..., body_pos], batch["hf_body"][..., body_pos]) if len(body_pos) == 3 else (pred_body_scaled.new_tensor(0.0), {})
    l_body_v, _ = axis_loss(pred_body_scaled[..., body_vel], batch["hf_body"][..., body_vel]) if len(body_vel) == 3 else (pred_body_scaled.new_tensor(0.0), {})
    l_body_delta_reg = body_delta_pos.pow(2).mean() if len(body_pos) == 3 else pred_body_scaled.new_tensor(0.0)
    l_body_delta_smooth = pred_body_scaled.new_tensor(0.0)

    wheel_losses = []
    wheel_local_losses = []
    assembly_losses = []
    wheel_local_regs = []
    wheel_local_rmse_terms = []
    wheel_local_abs_terms = []
    force_losses = []
    phy_state_losses = []
    bias_losses = []
    dynamic_losses = []
    residual_regs = []
    no_harm_terms = []
    f_pred_all = []
    f_phy_pred_all = []
    f_phy_ref_all = []
    f_true_all = []
    f_bias_all = []
    f_bias_target_all = []
    f_dynamic_all = []
    f_dynamic_target_all = []
    sinkage_lf_all = []
    sinkage_pred_all = []
    metrics: Dict[str, torch.Tensor] = {}

    body_vel_pred = pred_body_raw[..., body_vel] if len(body_vel) == 3 else pred_body_raw.new_zeros(*pred_body_raw.shape[:2], 3)
    for i in WHEEL_IDS:
        pred_wheel_scaled, pred_wheel_raw, pred_wheel_delta_raw = reconstruct_wheel_kin_with_body_delta(
            output[f"pred_res_wheel{i}_kin"],
            batch[f"lf_wheel{i}_kin_current"],
            body_global_delta,
            spec.res_groups.wheel_kin_cols[i],
            spec.target_groups.wheel_kin_cols[i],
            f"hf_wheel{i}_kin",
            f"res_wheel{i}_kin",
            scaler,
        )
        true_wheel_raw = inverse_transform_tensor(batch[f"hf_wheel{i}_kin"], scaler, f"hf_wheel{i}_kin")
        wheel_cols = spec.target_groups.wheel_kin_cols[i]
        wpos = suffix_indices(wheel_cols, ["pos_x", "pos_y", "pos_z"])
        omega_i = omega_index(wheel_cols)
        if len(wpos) == 3:
            lp, _ = axis_loss(pred_wheel_scaled[..., wpos], batch[f"hf_wheel{i}_kin"][..., wpos])
            wheel_losses.append(lp)
            lf_wp = batch[f"lf_wheel{i}_kin_current"][..., wpos].float()
            true_wp = true_wheel_raw[..., wpos]
            true_body_pos = true_body_raw[..., body_pos] if len(body_pos) == 3 else true_body_raw.new_zeros(*true_body_raw.shape[:2], 3)
            body_delta_target = true_body_pos - lf_body_pos
            wheel_local_target = (true_wp - lf_wp) - body_delta_target
            std_wp = torch.as_tensor([scaler.scalers[f"hf_wheel{i}_kin"].std_[j] for j in wpos], device=pred_wheel_raw.device).clamp_min(1e-6)
            xz = [0, 2]
            pred_local = local_xz_delta(pred_wheel_delta_raw, wpos)
            wheel_local_losses.append(axis_loss(pred_local[..., xz] / std_wp[xz], wheel_local_target[..., xz] / std_wp[xz])[0])
            wheel_local_regs.append((pred_local[..., xz] / std_wp[xz]).pow(2).mean())
            assembly_pred = pred_wheel_raw[..., wpos] - pred_body_raw[..., body_pos].detach()
            assembly_true = true_wp - true_body_pos
            assembly_losses.append(axis_loss(assembly_pred / std_wp, assembly_true / std_wp)[0])
            wheel_local_rmse_terms.append(torch.sqrt(((pred_local[..., xz] - wheel_local_target[..., xz]) ** 2).mean()))
            wheel_local_abs_terms.append(pred_local[..., xz].abs().mean())
            wheel_pos_pred_e = torch.abs(pred_wheel_raw[..., wpos] - true_wp) / std_wp
            wheel_pos_lf_e = torch.abs(lf_wp - true_wp) / std_wp
            no_harm_terms.append(torch.relu(wheel_pos_pred_e - wheel_pos_lf_e).mean())
            wheel_pos_violate = wheel_pos_pred_e > wheel_pos_lf_e
            metrics[f"no_harm_wheel{i}_pos_violation_rate"] = wheel_pos_violate.float().mean()
        if omega_i is not None:
            wheel_losses.append(F.huber_loss(pred_wheel_scaled[..., omega_i:omega_i + 1], batch[f"hf_wheel{i}_kin"][..., omega_i:omega_i + 1]))
            std_om = torch.as_tensor(scaler.scalers[f"hf_wheel{i}_kin"].std_[omega_i], device=pred_wheel_raw.device).clamp_min(1e-6)
            lf_om = batch[f"lf_wheel{i}_kin_current"][..., omega_i:omega_i + 1].float()
            true_om = true_wheel_raw[..., omega_i:omega_i + 1]
            omega_pred_e = torch.abs(pred_wheel_raw[..., omega_i:omega_i + 1] - true_om) / std_om
            omega_lf_e = torch.abs(lf_om - true_om) / std_om
            no_harm_terms.append(torch.relu(omega_pred_e - omega_lf_e).mean())
            metrics[f"no_harm_wheel{i}_omega_violation_rate"] = (omega_pred_e > omega_lf_e).float().mean()

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
        f_bias_target = batch.get(f"wheel{i}_force_bias_target", f_true - f_phy_ref)
        f_dynamic_target = batch.get(f"wheel{i}_force_dynamic_target", torch.zeros_like(f_bias_target))
        f_bias = output[f"pred_force_bias_wheel{i}"]
        f_dynamic = output[f"pred_force_dynamic_wheel{i}"]
        f_pred = f_phy + f_bias + f_dynamic
        f_pred_dyn = f_phy_dyn + f_bias + f_dynamic
        mask = contact_mask(batch, spec, i, f_pred)
        sigma_f = torch.as_tensor(
            [scaler.scalers[f"hf_wheel{i}_contact"].std_[j] for j in force_idx],
            device=f_pred.device,
            dtype=f_pred.dtype,
        ).clamp_min(1e-6) if len(force_idx) == 3 else f_pred.new_ones(3)
        force_losses.append(axis_loss(f_pred / sigma_f, f_true / sigma_f, mask)[0])
        phy_state_losses.append(axis_loss(f_phy_dyn / sigma_f, f_phy_ref / sigma_f, mask)[0])
        bias_losses.append(axis_loss(f_bias / sigma_f, f_bias_target / sigma_f, mask)[0])
        dynamic_losses.append(axis_loss(f_dynamic / sigma_f, f_dynamic_target / sigma_f, mask)[0])
        residual_regs.append(((f_bias / sigma_f) ** 2).mean() + ((f_dynamic / sigma_f) ** 2).mean())
        f_pred_all.append(f_pred_dyn)
        f_phy_pred_all.append(f_phy_dyn)
        f_phy_ref_all.append(f_phy_ref)
        f_true_all.append(f_true)
        f_bias_all.append(f_bias)
        f_bias_target_all.append(f_bias_target)
        f_dynamic_all.append(f_dynamic)
        f_dynamic_target_all.append(f_dynamic_target)

        e_pred = torch.abs(f_pred - f_true) / sigma_f
        e_lf = torch.abs(lf_contact_raw[..., force_idx] - f_true) / sigma_f if len(force_idx) == 3 else e_pred.detach()
        nh = torch.relu(e_pred - e_lf)
        if mask is not None:
            nh = nh[mask.to(dtype=torch.bool)]
        no_harm_terms.append(nh.mean() if nh.numel() else f_pred.new_tensor(0.0))
        force_violate = e_pred > e_lf
        if mask is not None:
            force_violate = force_violate[mask.to(dtype=torch.bool)]
        metrics[f"no_harm_wheel{i}_force_violation_rate"] = force_violate.float().mean() if force_violate.numel() else f_pred.new_tensor(0.0)
        metrics[f"wheel{i}_Fx_rmse"] = torch.sqrt(((f_pred[..., 0] - f_true[..., 0]) ** 2).mean())
        metrics[f"wheel{i}_Fy_rmse"] = torch.sqrt(((f_pred[..., 1] - f_true[..., 1]) ** 2).mean())
        metrics[f"wheel{i}_Fz_rmse"] = torch.sqrt(((f_pred[..., 2] - f_true[..., 2]) ** 2).mean())

    l_wheel_state = torch.stack(wheel_losses).mean() if wheel_losses else pred_body_scaled.new_tensor(0.0)
    l_state = torch.stack([l_body_p, l_body_v, l_wheel_state]).mean()
    l_wheel_local = torch.stack(wheel_local_losses).mean() if wheel_local_losses else pred_body_scaled.new_tensor(0.0)
    l_assembly = torch.stack(assembly_losses).mean() if assembly_losses else pred_body_scaled.new_tensor(0.0)
    l_force = torch.stack(force_losses).mean() if force_losses else pred_body_scaled.new_tensor(0.0)
    l_phy_state = torch.stack(phy_state_losses).mean() if phy_state_losses else pred_body_scaled.new_tensor(0.0)
    l_bias = torch.stack(bias_losses).mean() if bias_losses else pred_body_scaled.new_tensor(0.0)
    l_dynamic = torch.stack(dynamic_losses).mean() if dynamic_losses else pred_body_scaled.new_tensor(0.0)
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

    lf_body_raw = batch["lf_body_current"].float()
    if len(body_pos) == 3:
        std = torch.as_tensor([scaler.scalers["hf_body"].std_[j] for j in body_pos], device=pred_body_raw.device).clamp_min(1e-6)
        pred_e = torch.abs(pred_body_raw[..., body_pos] - true_body_raw[..., body_pos]) / std
        lf_e = torch.abs(lf_body_raw[..., body_pos] - true_body_raw[..., body_pos]) / std
        no_harm_terms.append(torch.relu(pred_e - lf_e).mean())
        metrics["no_harm_body_pos_violation_rate"] = (pred_e > lf_e).float().mean()
    if len(body_vel) == 3:
        std = torch.as_tensor([scaler.scalers["hf_body"].std_[j] for j in body_vel], device=pred_body_raw.device).clamp_min(1e-6)
        pred_e = torch.abs(body_vel_pred - true_body_raw[..., body_vel]) / std
        lf_e = torch.abs(lf_body_raw[..., body_vel] - true_body_raw[..., body_vel]) / std
        no_harm_terms.append(torch.relu(pred_e - lf_e).mean())
        metrics["no_harm_body_vel_violation_rate"] = (pred_e > lf_e).float().mean()
    l_no_harm = torch.stack(no_harm_terms).mean() if no_harm_terms else pred_body_scaled.new_tensor(0.0)

    losses.update(metrics)
    losses["state"] = l_state
    losses["wheel_state"] = l_wheel_state
    losses["wheel_local"] = l_wheel_local
    losses["assembly"] = l_assembly
    losses["force"] = l_force
    losses["phy_state"] = l_phy_state
    losses["bias"] = l_bias
    losses["dynamic"] = l_dynamic
    losses["kin"] = l_kin
    losses["dyn"] = l_dyn_cons
    losses["residual_reg"] = l_reg
    losses["wheel_local_reg"] = l_wheel_local_reg
    losses["body_delta_reg"] = l_body_delta_reg
    losses["body_delta_smooth"] = l_body_delta_smooth
    losses["no_harm"] = l_no_harm
    losses["weighted_wheel_local"] = args.lambda_wheel_local * l_wheel_local
    losses["weighted_assembly"] = args.lambda_assembly * l_assembly
    losses["weighted_wheel_local_reg"] = args.lambda_wheel_local_reg * l_wheel_local_reg
    losses["weighted_body_delta_reg"] = getattr(args, "lambda_body_delta_reg", 1e-3) * l_body_delta_reg
    losses["weighted_body_delta_smooth"] = getattr(args, "lambda_body_delta_smooth", 0.0) * l_body_delta_smooth
    losses["weighted_force"] = args.lambda_F * l_force
    losses["weighted_phy_state"] = args.lambda_phy_state * l_phy_state
    losses["weighted_bias"] = args.lambda_bias * l_bias
    losses["weighted_dynamic"] = args.lambda_dynamic * l_dynamic
    losses["weighted_kin"] = args.lambda_kin * l_kin
    losses["weighted_dyn"] = args.lambda_dyn * l_dyn_cons
    losses["weighted_reg"] = args.lambda_reg * l_reg
    losses["weighted_harm"] = args.lambda_harm * l_no_harm
    losses["body_pos_xyz_rmse"] = torch.sqrt(((pred_body_raw[..., body_pos] - true_body_raw[..., body_pos]) ** 2).mean()) if len(body_pos) == 3 else l_state.detach()
    losses["body_vel_xyz_rmse"] = torch.sqrt(((body_vel_pred - true_body_raw[..., body_vel]) ** 2).mean()) if len(body_vel) == 3 else l_state.detach()
    losses["F_phy_pred_ref_rmse"] = torch.sqrt(((torch.cat(f_phy_pred_all, dim=-2) - torch.cat(f_phy_ref_all, dim=-2)) ** 2).mean()) if f_phy_pred_all else l_force.detach()
    losses["F_phy_pred_hf_rmse"] = torch.sqrt(((torch.cat(f_phy_pred_all, dim=-2) - torch.cat(f_true_all, dim=-2)) ** 2).mean()) if f_phy_pred_all else l_force.detach()
    losses["F_bias_rmse"] = torch.sqrt(((torch.cat(f_bias_all, dim=-2) - torch.cat(f_bias_target_all, dim=-2)) ** 2).mean()) if f_bias_all else l_force.detach()
    losses["F_dynamic_rmse"] = torch.sqrt(((torch.cat(f_dynamic_all, dim=-2) - torch.cat(f_dynamic_target_all, dim=-2)) ** 2).mean()) if f_dynamic_all else l_force.detach()
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
        + losses["weighted_assembly"]
        + losses["weighted_wheel_local_reg"]
        + losses["weighted_body_delta_reg"]
        + losses["weighted_body_delta_smooth"]
        + losses["weighted_force"]
        + losses["weighted_phy_state"]
        + losses["weighted_bias"]
        + losses["weighted_dynamic"]
        + losses["weighted_kin"]
        + losses["weighted_dyn"]
        + losses["weighted_reg"]
        + losses["weighted_harm"]
    )
    return losses


def set_stage_trainability(model, teacher_gate_net, stage: str) -> None:
    for p in model.parameters():
        p.requires_grad = stage == "teacher"
    for p in teacher_gate_net.parameters():
        p.requires_grad = stage == "teacher"
    if stage == "student":
        for p in model.student_summary_predictor.parameters():
            p.requires_grad = True


def forward_stage(model, teacher_gate_net, batch, stage: str):
    if stage == "teacher":
        summary = build_teacher_gate_summary_v2(batch)
        gate = teacher_gate_net(summary)
        out = model(batch, relation_gate_override=gate)
        out["teacher_latent"] = summary
        out["teacher_relation_gate"] = gate
        return out
    with torch.no_grad():
        teacher_summary = build_teacher_gate_summary_v2(batch)
        teacher_gate = teacher_gate_net(teacher_summary)
    out = model(batch, teacher_gate_net=teacher_gate_net)
    out["teacher_latent"] = teacher_summary.detach()
    out["teacher_relation_gate"] = teacher_gate.detach()
    return out


def run_epoch(model, teacher_gate_net, loader, optimizer, device, spec, scaler, args, train: bool, epoch: int):
    model.train(train)
    teacher_gate_net.train(train and args.train_stage == "teacher")
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
            out = forward_stage(model, teacher_gate_net, batch, args.train_stage)
            losses = compute_losses_v2(batch, out, spec, scaler, args)
            if args.train_stage == "student":
                latent_loss = F.mse_loss(out["student_gate_summary"], out["teacher_latent"])
                gate_loss = F.mse_loss(out["student_relation_gate"], out["teacher_relation_gate"])
                losses["student_latent"] = latent_loss
                losses["student_gate"] = gate_loss
                losses["total"] = args.lambda_latent * latent_loss + args.lambda_gate * gate_loss
                if args.train_main_in_stage2:
                    losses["total"] = losses["total"] + compute_losses_v2(batch, out, spec, scaler, args)["total"]
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
                    problems = collect_nonfinite_grads(model, teacher_gate_net)
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
                grad_norm = torch.nn.utils.clip_grad_norm_([p for p in list(model.parameters()) + list(teacher_gate_net.parameters()) if p.requires_grad], args.grad_clip)
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
                    f"[V2] {phase_name} batch {n_batches}/{total_batches} {format_metrics(current)}",
                    flush=True,
                )
    if progress is not None:
        progress.close()
    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def run_dummy_forward_check(model, teacher_gate_net, group_dims, seq_len, device):
    batch = {}
    for key in ["system", "body", *ROCKER_NAMES, *[f"wheel{i}_kin" for i in WHEEL_IDS], *[f"wheel{i}_contact" for i in WHEEL_IDS]]:
        batch[key] = torch.zeros(2, seq_len, int(group_dims.get(key, 0)), device=device)
    summary = torch.zeros(2, seq_len, TEACHER_GATE_SUMMARY_DIM_V2, device=device)
    out = model(batch, relation_gate_override=teacher_gate_net(summary))
    assert out["pred_res_body"].shape == (2, 1, int(group_dims["res_body"]))
    assert out["pred_force_bias_wheel0"].shape == (2, 1, 3)
    assert out["pred_force_dynamic_wheel0"].shape == (2, 1, 3)
    print("V2 dummy forward check passed")


def resolve_init_checkpoint(stage: str, init_ckpt: Optional[str], save_dir: str) -> Optional[str]:
    if init_ckpt:
        return init_ckpt
    if stage != "student":
        return None
    candidates = list((Path(save_dir) / "teacher").glob("*/best_model.pt"))
    if not candidates:
        raise FileNotFoundError("student 阶段需要 --init_ckpt，或先在 save_dir/teacher 下训练 teacher")
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", type=str, default=str(ROOT / "Feature_Selection" / "DataSet"))
    parser.add_argument("--merged_csv", type=str, default=str(ROOT / "Feature_Selection" / "DataSet" / "merged_error_dataset.csv"))
    parser.add_argument("--save_dir", type=str, default=str(ROOT / "results_v2" ))
    parser.add_argument("--log_dir", type=str, default=None, help="TensorBoard 日志目录；默认保存到当前 run 的 tensorboard 子目录")
    parser.add_argument("--seq_len", type=int, default=10)
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
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_stage", choices=["teacher", "student"], default="teacher")
    parser.add_argument("--init_ckpt", type=str, default=None)
    parser.add_argument("--train_main_in_stage2", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lambda_F", type=float, default=1.0)
    parser.add_argument("--lambda_wheel_local", type=float, default=0.75)
    parser.add_argument("--lambda_assembly", type=float, default=0.2)
    parser.add_argument("--lambda_wheel_local_reg", type=float, default=1e-4)
    parser.add_argument("--lambda_body_delta_reg", type=float, default=1e-3)
    parser.add_argument("--lambda_body_delta_smooth", type=float, default=0.0)
    parser.add_argument("--lambda_phy_state", type=float, default=0.005)
    parser.add_argument("--lambda_bias", type=float, default=0.6)
    parser.add_argument("--lambda_dynamic", type=float, default=0.35)
    parser.add_argument("--lambda_kin", type=float, default=0.02)
    parser.add_argument("--lambda_dyn", type=float, default=0.005)
    parser.add_argument("--lambda_reg", type=float, default=5e-5)
    parser.add_argument("--lambda_harm", type=float, default=0.1)
    parser.add_argument("--lambda_latent", type=float, default=1.0)
    parser.add_argument("--lambda_gate", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.015)
    parser.add_argument("--mass", type=float, default=240.0)
    parser.add_argument("--sinkage_max", type=float, default=0.08)
    parser.add_argument("--force_lpf_alpha", type=float, default=0.15)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--progress_bar", action=argparse.BooleanOptionalAction, default=True, help="是否用 tqdm 进度条显示 train/val batch 进度")
    parser.add_argument("--progress_interval", type=int, default=10, help="每隔多少个 batch 刷新一次进度条 loss 组成；<=0 表示关闭")
    parser.add_argument("--debug_finite", action=argparse.BooleanOptionalAction, default=False, help="逐 batch 检查 loss/output/gradient 是否包含 NaN 或 Inf")
    parser.add_argument("--debug_dump_dir", type=str, default=None, help="debug_finite 触发时保存异常 batch 的目录；默认保存到当前 run/debug_finite")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.pred_horizon != 0:
        print("V2 保持单步当前预测；pred_horizon 非 0 时仍只输出一个目标步。")
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
        force_lpf_alpha=args.force_lpf_alpha,
        sinkage_max=args.sinkage_max,
    )
    log_step(f"数据集准备完成，train={len(train_ds)} val={len(val_ds) if val_ds is not None else 0}", t0)
    t0 = log_step("保存列配置和 scaler")
    save_column_spec_json(spec, os.path.join(args.save_dir, "column_spec_v2.json"))
    scaler.save(os.path.join(args.save_dir, "group_scaler_v2.joblib"))
    log_step("列配置和 scaler 保存完成", t0)
    t0 = log_step("初始化 V2 模型")
    group_dims = get_group_dims(spec)
    model = GraphTemporalHGTCompensationModelV2(
        group_dims=group_dims,
        node_hidden_dim=args.hidden_dim,
        graph_layers=args.graph_layers,
        tcn_hidden_dim=args.tcn_dim,
        lstm_hidden_dim=args.lstm_dim,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        group_columns=build_group_columns(spec),
        gate_summary_dim=TEACHER_GATE_SUMMARY_DIM_V2,
    ).to(device)
    teacher_gate_net = TeacherGateNet(TEACHER_GATE_SUMMARY_DIM_V2, model.num_relations, max(128, args.hidden_dim * 2), args.dropout).to(device)
    log_step("V2 模型初始化完成", t0)

    t0 = log_step("执行 dummy forward 检查")
    run_dummy_forward_check(model, teacher_gate_net, group_dims, args.seq_len, device)
    log_step("dummy forward 检查完成", t0)
    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location=device, weights_only=True)
        load_compatible_state_dict(model, ckpt["model_state_dict"], "model init checkpoint")
        if "teacher_gate_state_dict" in ckpt:
            load_compatible_state_dict(teacher_gate_net, ckpt["teacher_gate_state_dict"], "teacher gate init checkpoint")
        print(f"Loaded init checkpoint: {args.init_ckpt}")

    set_stage_trainability(model, teacher_gate_net, args.train_stage)
    params = [p for p in list(model.parameters()) + list(teacher_gate_net.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr if args.train_stage == "teacher" else min(args.lr, 5e-5), weight_decay=args.weight_decay)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=graph_temporal_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=graph_temporal_collate_fn) if val_ds is not None else None

    best = math.inf
    history = []
    for epoch in range(1, args.epochs + 1):
        epoch_t0 = log_step(f"开始 epoch {epoch}/{args.epochs}")
        tr = run_epoch(model, teacher_gate_net, train_loader, optimizer, device, spec, scaler, args, train=True, epoch=epoch)
        print_epoch_metrics(epoch, "train", tr)
        va = run_epoch(model, teacher_gate_net, val_loader, optimizer, device, spec, scaler, args, train=False, epoch=epoch) if val_loader else tr
        print_epoch_metrics(epoch, "val", va)
        score = va.get("total", math.inf)
        history.append({"epoch": epoch, "train": tr, "val": va})
        write_tensorboard_scalars(writer, "train", tr, epoch)
        write_tensorboard_scalars(writer, "val", va, epoch)
        write_optimizer_lrs(writer, optimizer, epoch)
        log_step(f"epoch {epoch} 完成", epoch_t0)
        if score < best:
            best = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "teacher_gate_state_dict": teacher_gate_net.state_dict(),
                    "args": vars(args),
                    "best": best,
                },
                os.path.join(args.save_dir, "best_model.pt"),
            )
    with open(os.path.join(args.save_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    if writer is not None:
        writer.flush()
        writer.close()
    print(f"V2 training finished. Best={best:.6f}. Saved to {args.save_dir}")


if __name__ == "__main__":
    main()
