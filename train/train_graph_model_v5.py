from __future__ import annotations

import argparse
from itertools import islice
import json
import math
import os
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.data_utils_v5 import (  # noqa: E402
    ROCKER_NAMES, WHEEL_IDS, get_group_dims, graph_temporal_collate_fn,
    prepare_datasets_and_scaler, save_column_spec_json,
)
from models.graph_temporal_hgt_compensation_v5 import HGTGraphTemporalCompensationModelV5  # noqa: E402
from models.load_transfer_force_v5 import fit_load_transfer_params, load_transfer_fz_torch  # noqa: E402

STAGES = ("state_teacher", "state_student", "force", "joint")
ASSEMBLY_SIDES = {
    "left": {"front":"lf","rear":"lm","sub":"lb","front_wheel":0,"mid_wheel":2,"rear_wheel":4},
    "right": {"front":"rf","rear":"rm","sub":"rb","front_wheel":1,"mid_wheel":3,"rear_wheel":5},
}
ASSEMBLY_DISTANCE_CONSTRAINTS = ("front_AB","front_BC","front_AC","rear_main_AD","bogie_DE","bogie_DF","middle_upright","rear_upright","middle_rear_wheel")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def move_batch(batch: Dict, device: torch.device) -> Dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def accumulation_sample_weight(
    batch_idx: int, batch_size: int, loader_batch_size: int, dataset_size: int,
    grad_accum_steps: int, num_batches: int,
) -> float:
    """Scale each microbatch so an accumulation group averages by samples."""
    group_start = (batch_idx // max(1, grad_accum_steps)) * max(1, grad_accum_steps)
    group_end = min(group_start + max(1, grad_accum_steps), num_batches)
    sample_start = min(group_start * loader_batch_size, dataset_size)
    sample_end = min(group_end * loader_batch_size, dataset_size)
    group_sample_count = max(sample_end - sample_start, batch_size)
    return batch_size / group_sample_count


def restore_training_checkpoint(model, optimizer, checkpoint: Dict) -> Tuple[int, Dict[str, float]]:
    """Restore a complete-epoch checkpoint and return next epoch and best scores."""
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if "optimizer_state_dict" not in checkpoint:
        raise KeyError("resume checkpoint has no optimizer_state_dict")
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    rng_state = checkpoint.get("rng_state")
    if rng_state:
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch_cpu"])
        if torch.cuda.is_available() and rng_state.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
    best = {
        "total": float(checkpoint.get("best_metric", math.inf)),
        "state_output": float(checkpoint.get("best_state_output", math.inf)),
        "rollout": float(checkpoint.get("best_rollout", math.inf)),
    }
    return int(checkpoint.get("epoch", 0)) + 1, best


def _cuda_debug_enabled(args) -> bool:
    return bool(getattr(args, "debug_cuda", False) and args.device_obj.type == "cuda")


def _debug_sync(args) -> None:
    if _cuda_debug_enabled(args):
        torch.cuda.synchronize(args.device_obj)


def _debug_memory(args) -> Dict[str, Optional[float]]:
    if not _cuda_debug_enabled(args):
        return {"allocated_mib": None, "reserved_mib": None, "peak_allocated_mib": None, "peak_reserved_mib": None}
    device = args.device_obj
    return {
        "allocated_mib": round(torch.cuda.memory_allocated(device) / 2**20, 1),
        "reserved_mib": round(torch.cuda.memory_reserved(device) / 2**20, 1),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 2**20, 1),
    }


def _debug_event(args, name: str, elapsed_s: Optional[float] = None) -> None:
    if not _cuda_debug_enabled(args):
        return
    _debug_sync(args)
    context = getattr(args, "_debug_context", {})
    fields = [
        f"epoch={context.get('epoch', '?')}", f"phase={context.get('phase', '?')}",
        f"batch={context.get('batch', '?')}/{context.get('num_batches', '?')}",
        f"step={context.get('rollout_step', '-')}", f"stage={name}",
    ]
    if elapsed_s is not None:
        context.setdefault("profile", {}).setdefault("times", {}).setdefault(name, []).append(elapsed_s)
    profile = context.setdefault("profile", {})
    profile["memory"] = _debug_memory(args)
    if getattr(args, "debug_cuda_verbose", False):
        fields.append(f"seconds={elapsed_s:.4f}" if elapsed_s is not None else "")
        fields.extend(f"{key}={value}MiB" for key, value in profile["memory"].items())
        print("[V5][cuda-debug] " + " ".join(part for part in fields if part), flush=True)


def format_cuda_debug_postfix(context: Dict) -> Dict[str, object]:
    """Compact stage timings and memory values for one tqdm postfix."""
    profile = context.get("profile", {})
    times = profile.get("times", {})
    memory = profile.get("memory", {})
    result: Dict[str, object] = {}

    def latest(name: str) -> Optional[float]:
        values = times.get(name, [])
        return values[-1] if values else None

    def add_time(label: str, value: Optional[float]) -> None:
        if value is not None:
            result[label] = f"{value:.3f}"

    add_time("fwd_s", latest("forward"))
    rollout_forward = times.get("forward_rollout_step", [])
    if rollout_forward:
        result["roll_fwd_s"] = "/".join(f"{value:.3f}" for value in rollout_forward)
    reconstruction = times.get("loss_reconstruction", [])
    if reconstruction:
        result["loss_s"] = f"{sum(reconstruction):.3f}"
    rollout_loss = times.get("reconstruct_rollout_step", [])
    if rollout_loss:
        result["roll_loss_s"] = "/".join(f"{value:.3f}" for value in rollout_loss)
    add_time("bwd_s", latest("backward"))
    add_time("opt_s", latest("optimizer_step"))
    for label, key in (("alloc_mib", "allocated_mib"), ("reserved_mib", "reserved_mib"), ("peak_mib", "peak_allocated_mib")):
        if memory.get(key) is not None:
            result[label] = memory[key]
    return result


def _write_debug_failure(args, stage: str, exc: BaseException) -> None:
    if not getattr(args, "debug_cuda", False):
        return
    context = getattr(args, "_debug_context", {})
    already_recorded = getattr(args, "_debug_failure_context", None)
    if already_recorded == context:
        print(f"[V5][cuda-debug] propagated failure stage={stage} context={context}", flush=True)
        return
    sync_error = None
    try:
        _debug_sync(args)
    except BaseException as cuda_error:
        sync_error = repr(cuda_error)
    try:
        memory = _debug_memory(args)
    except BaseException as memory_error:
        memory = {"unavailable": repr(memory_error)}
    record = {
        "stage": stage,
        "context": context,
        "memory": memory,
        "sync_error": sync_error,
        "exception": repr(exc),
        "traceback": traceback.format_exc(),
    }
    path = Path(getattr(args, "run_dir", ".")) / "cuda_debug_last_error.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as log_error:
        print(f"[V5][cuda-debug] failed to write {path}: {log_error}", flush=True)
    print(
        f"[V5][cuda-debug] failure stage={stage} context={record['context']} "
        f"memory={record['memory']} exception={exc!r}", flush=True,
    )
    args._debug_failure_context = dict(context)


def _profile_call(args, name: str, function):
    if not _cuda_debug_enabled(args):
        return function()
    started = time.perf_counter()
    context = getattr(args, "_debug_context", {})
    context["stage"] = name
    args._debug_context = context
    try:
        _debug_sync(args)
        result = function()
        _debug_sync(args)
    except BaseException as exc:
        _write_debug_failure(args, name, exc)
        raise
    _debug_event(args, name, time.perf_counter() - started)
    return result


def indices(cols: List[str], suffixes: List[str]) -> List[int]:
    result = []
    for suffix in suffixes:
        found = next((i for i, col in enumerate(cols) if col.endswith(suffix)), None)
        if found is not None:
            result.append(found)
    return result


def omega_index(cols: List[str]) -> Optional[int]:
    for suffix in ("omega", "ang_vel_y", "ang_vel_z", "ang_vel_x"):
        found = next((i for i, col in enumerate(cols) if col.endswith(suffix)), None)
        if found is not None:
            return found
    return None


def scaler_values(scaler, key: str, device: torch.device):
    current = scaler.scalers[key]
    mean = torch.as_tensor(current.mean_, device=device, dtype=torch.float32)
    std = torch.as_tensor(current.std_, device=device, dtype=torch.float32).clamp_min(1e-8)
    mask = torch.as_tensor(current.apply_mask_, device=device, dtype=torch.bool)
    return mean, std, mask


def inverse(x: torch.Tensor, scaler, key: str) -> torch.Tensor:
    mean, std, mask = scaler_values(scaler, key, x.device)
    y = x.float().clone()
    y[..., mask] = y[..., mask] * std[mask] + mean[mask]
    return y


def transform(x: torch.Tensor, scaler, key: str) -> torch.Tensor:
    mean, std, mask = scaler_values(scaler, key, x.device)
    y = x.float().clone()
    y[..., mask] = (y[..., mask] - mean[mask]) / std[mask]
    return y


def group_std(scaler, key: str, ix: List[int], ref: torch.Tensor) -> torch.Tensor:
    _, std, _ = scaler_values(scaler, key, ref.device)
    return std[ix].view(*([1] * (ref.ndim - 1)), -1)


def residual_raw(pred_scaled: torch.Tensor, res_cols: List[str], hf_cols: List[str], res_key: str, scaler) -> torch.Tensor:
    raw = inverse(pred_scaled, scaler, res_key)
    out = raw.new_zeros(*raw.shape[:-1], len(hf_cols))
    lookup = {name: i for i, name in enumerate(res_cols)}
    for j, col in enumerate(hf_cols):
        res_name = "res_" + col[3:] if col.startswith("hf_") else ""
        if res_name in lookup:
            out[..., j] = raw[..., lookup[res_name]]
    return out


def quat_matrix(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y*y + z*z), 2 * (x*y-z*w), 2 * (x*z+y*w),
        2 * (x*y+z*w), 1 - 2 * (x*x+z*z), 2 * (y*z-x*w),
        2 * (x*z-y*w), 2 * (y*z+x*w), 1 - 2 * (x*x+y*y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


def rotvec_matrix(v: torch.Tensor) -> torch.Tensor:
    if v.shape[-1] != 3:
        raise ValueError(f"rotvec_matrix expects [...,3], got {tuple(v.shape)}")
    theta2 = v.square().sum(dim=-1)
    theta = theta2.clamp_min(1e-12).sqrt()
    theta2_safe = theta2.clamp_min(1e-12)
    small = theta2 < 1e-6

    # Taylor limits keep Rodrigues coefficients smooth and stable near zero.
    a_taylor = 1.0 - theta2 / 6.0 + theta2.square() / 120.0
    b_taylor = 0.5 - theta2 / 24.0 + theta2.square() / 720.0
    a = torch.where(small, a_taylor, torch.sin(theta) / theta)
    b = torch.where(small, b_taylor, (1.0 - torch.cos(theta)) / theta2_safe)

    x, y, z = v.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1)
    skew = skew.reshape(*v.shape[:-1], 3, 3)
    identity = torch.eye(3, dtype=v.dtype, device=v.device).expand(*v.shape[:-1], 3, 3)
    return identity + a[..., None, None] * skew + b[..., None, None] * torch.matmul(skew, skew)


def _rotation_debug_check(name: str, value: torch.Tensor, args, sync: bool = False) -> None:
    if not (getattr(args, "debug_rotations", False) or getattr(args, "debug_cuda_verbose", False)):
        return
    if sync and value.is_cuda and (getattr(args, "debug_cuda_sync", False) or getattr(args, "debug_cuda", False)):
        torch.cuda.synchronize(value.device)
    finite = bool(torch.isfinite(value.detach()).all().item())
    logged = getattr(args, "_rotation_debug_logged", set())
    if name not in logged:
        print(
            f"[V5][rotation] {name}: shape={tuple(value.shape)} dtype={value.dtype} "
            f"device={value.device} isfinite={finite}",
            flush=True,
        )
        logged.add(name)
        args._rotation_debug_logged = logged
    if not finite:
        raise FloatingPointError(f"Non-finite tensor in rotation path: {name}")


def _single_step_vector3(name: str, value: torch.Tensor, batch_size: int, args) -> torch.Tensor:
    if value.ndim == 3 and value.shape[1] == 1:
        value = value[:, 0, :]
    if tuple(value.shape) != (batch_size, 3):
        raise RuntimeError(f"{name} must be [B,3], got {tuple(value.shape)} (B={batch_size})")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point, got {value.dtype}")
    _rotation_debug_check(name, value, args)
    return value


def _single_step_rotation(name: str, value: torch.Tensor, batch_size: int, args) -> torch.Tensor:
    if value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0, :, :]
    if tuple(value.shape) != (batch_size, 3, 3):
        raise RuntimeError(f"{name} must be [B,3,3], got {tuple(value.shape)} (B={batch_size})")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point, got {value.dtype}")
    _rotation_debug_check(name, value, args)
    return value


def _rotation_matmul(name: str, left: torch.Tensor, right: torch.Tensor, batch_size: int, args) -> torch.Tensor:
    left = _single_step_rotation(f"{name}.left", left, batch_size, args)
    right = _single_step_rotation(f"{name}.right", right, batch_size, args)
    if left.device != right.device or left.dtype != right.dtype:
        raise RuntimeError(
            f"{name} rotation operands differ: left={left.dtype}@{left.device}, "
            f"right={right.dtype}@{right.device}"
        )
    result = torch.matmul(left, right)
    if (getattr(args, "debug_cuda", False) or (getattr(args, "debug_rotations", False) and getattr(args, "debug_cuda_sync", False))) and result.is_cuda:
        torch.cuda.synchronize(result.device)
    result = _single_step_rotation(name, result, batch_size, args)
    _rotation_debug_check(name, result, args)
    return result


def so3_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    rel = torch.matmul(pred.transpose(-1, -2), target)
    cosine = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5).clamp(-1 + 1e-6, 1 - 1e-6)
    return F.smooth_l1_loss(torch.acos(cosine), torch.zeros_like(cosine))


def gate_target(raw_delta: torch.Tensor, true_delta: torch.Tensor, std: Optional[torch.Tensor] = None, tau: float = 0.5, temp: float = 0.2) -> torch.Tensor:
    direction = ((raw_delta * true_delta) / (raw_delta.square() + 1e-8)).clamp(0, 1)
    if std is None:
        return direction
    needed = torch.sigmoid(((true_delta / std).abs() - tau) / max(temp, 1e-8))
    return (direction * needed).clamp(0, 1)


def assembly_point(pos: torch.Tensor, rot: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
    return pos + torch.matmul(rot, local.view(local.shape[0],1,3,1)).squeeze(-1)


def assembly_rollout_loss(batch, body_pos, body_rot, rockers, wheels):
    terms=[]
    for side, cfg in ASSEMBLY_SIDES.items():
        if not all(name in rockers for name in (cfg["front"],cfg["rear"],cfg["sub"])):
            continue
        if not all(wheel in wheels for wheel in (cfg["front_wheel"],cfg["mid_wheel"],cfg["rear_wheel"])):
            continue
        key=lambda name: batch.get(f"assembly_{side}_{name}")
        meta={name:key(name) for name in ("A_front_body","A_rear_body","B_front_local","D_main_local","D_sub_local","E_local","F_local")}
        refs={name:key(f"ref_{name}") for name in ASSEMBLY_DISTANCE_CONSTRAINTS}
        if any(value is None for value in (*meta.values(),*refs.values())):
            continue
        af=assembly_point(body_pos,body_rot,meta["A_front_body"])
        ar=assembly_point(body_pos,body_rot,meta["A_rear_body"])
        bf=assembly_point(rockers[cfg["front"]]["pos"],rockers[cfg["front"]]["rot"],meta["B_front_local"])
        dm=assembly_point(rockers[cfg["rear"]]["pos"],rockers[cfg["rear"]]["rot"],meta["D_main_local"])
        ds=assembly_point(rockers[cfg["sub"]]["pos"],rockers[cfg["sub"]]["rot"],meta["D_sub_local"])
        em=assembly_point(rockers[cfg["sub"]]["pos"],rockers[cfg["sub"]]["rot"],meta["E_local"])
        fr=assembly_point(rockers[cfg["sub"]]["pos"],rockers[cfg["sub"]]["rot"],meta["F_local"])
        pairs={"front_AB":(af,bf),"front_BC":(bf,wheels[cfg["front_wheel"]]),"front_AC":(af,wheels[cfg["front_wheel"]]),"rear_main_AD":(ar,dm),"bogie_DE":(dm,em),"bogie_DF":(dm,fr),"middle_upright":(em,wheels[cfg["mid_wheel"]]),"rear_upright":(fr,wheels[cfg["rear_wheel"]]),"middle_rear_wheel":(wheels[cfg["mid_wheel"]],wheels[cfg["rear_wheel"]])}
        for name,(p0,p1) in pairs.items():
            ref=refs[name].view(refs[name].shape[0],1).clamp_min(1e-6)
            terms.append(F.smooth_l1_loss((torch.linalg.norm(p0-p1,dim=-1)-ref)/ref,torch.zeros_like(ref)))
        terms.append(F.smooth_l1_loss(dm-ds,torch.zeros_like(dm)))
    return torch.stack(terms).mean() if terms else body_pos.new_tensor(0.0)


def raw_body_delta(out, spec, scaler):
    return residual_raw(out["pred_res_body"], spec.res_groups.body_cols, spec.target_groups.body_cols, "res_body", scaler)


def reconstruct_rollout_state(batch, outputs, spec, scaler, args):
    body_cols = spec.target_groups.body_cols
    body_pos = indices(body_cols, ["pos_x", "pos_y", "pos_z"])
    body_vel = indices(body_cols, ["vel_x", "vel_y", "vel_z"])
    lf_current = batch["lf_body_current"].float()
    batch_size = int(lf_current.shape[0])
    prev_raw = batch["lf_body_prev_raw"].float()
    prev_pos, prev_vel = prev_raw[..., body_pos], prev_raw[..., body_vel]
    body_gates, delta_pos, positions, props, velocities = [], [], [], [], []
    pos_losses, vel_losses, kin_losses, gate_losses = [], [], [], []
    rocker_pos_losses, rocker_local_losses, wheel_pos_losses, wheel_local_losses = [], [], [], []
    wheel_omega_losses, wheel_local_regs = [], []
    attitude_losses, assembly_losses = [], []
    body_harmful_steps, wheel_harmful_steps = [], []
    corrected_rockers: Dict[str, Dict[str, torch.Tensor]] = {}
    corrected_wheels: Dict[int, torch.Tensor] = {}
    for step, out in enumerate(outputs):
        step_started = time.perf_counter() if _cuda_debug_enabled(args) else None
        if _cuda_debug_enabled(args):
            args._debug_context["rollout_step"] = step
            _debug_sync(args)
        lf = batch["rollout_body"][:, step:step+1].float()
        hf = batch["rollout_hf_body"][:, step:step+1].float()
        delta = raw_body_delta(out, spec, scaler)
        vel = lf[..., body_vel] + delta[..., body_vel]
        dt = batch["rollout_dt"][:, step].view(-1, 1, 1).to(lf).clamp_min(1e-6)
        prop = prev_pos + 0.5 * (prev_vel + vel) * dt
        gate = out["gate_body_pos"]
        final = prop + gate * delta[..., body_pos]
        stdp = group_std(scaler, "hf_body", body_pos, final)
        stdv = group_std(scaler, "hf_body", body_vel, vel)
        pos_losses.append(F.smooth_l1_loss((final-hf[..., body_pos])/stdp, torch.zeros_like(final)))
        vel_losses.append(F.smooth_l1_loss((vel-hf[..., body_vel])/stdv, torch.zeros_like(vel)))
        body_pred_err=((final-hf[...,body_pos])/stdp).square().mean(dim=(-1,-2))
        body_lf_err=((lf[...,body_pos]-hf[...,body_pos])/stdp).square().mean(dim=(-1,-2))
        body_harmful_steps.append(body_pred_err>body_lf_err)
        true_correction = hf[..., body_pos] - prop
        gate_t = gate_target(delta[..., body_pos].detach(), true_correction, stdp, args.trust_lf_error_tau, args.trust_lf_error_temp)
        gate_losses.append(F.smooth_l1_loss(gate, gate_t.detach()))
        body_gates.append(gate)
        delta_pos.append(delta[..., body_pos])
        positions.append(final)
        props.append(prop)
        velocities.append(vel)
        actual = final - prev_pos
        velocity_step = 0.5 * (prev_vel + vel) * dt
        kin_losses.append(F.smooth_l1_loss(actual - velocity_step - gate * delta[..., body_pos], torch.zeros_like(actual)))
        body_delta = final - lf[..., body_pos]
        if "rollout_lf_body_pose" in batch:
            lf_body_pose = batch["rollout_lf_body_pose"][:, step:step+1].float()
            hf_body_pose = batch["rollout_hf_body_pose"][:, step:step+1].float()
            r_lf = quat_matrix(lf_body_pose[..., 3:7])
            r_hf = quat_matrix(hf_body_pose[..., 3:7])
            r_lf_step = _single_step_rotation("body.r_lf", r_lf, batch_size, args)
            r_hf_step = _single_step_rotation("body.r_hf", r_hf, batch_size, args)
            body_rotvec = _single_step_vector3(
                "body.gated_rotvec", out["gate_body_att"] * out["pred_delta_rot_body"], batch_size, args
            )
            body_delta_rot = rotvec_matrix(body_rotvec)
            body_delta_rot = _single_step_rotation("body.rotvec_matrix", body_delta_rot, batch_size, args)
            r_body_step = _rotation_matmul("body.R_lf_times_delta", r_lf_step, body_delta_rot, batch_size, args)
            body_lf_relative = _rotation_matmul(
                "body.R_pred_times_R_lf_transpose", r_body_step, r_lf_step.transpose(-1, -2), batch_size, args
            )
            r_body = r_body_step.unsqueeze(1)
            attitude_losses.append(so3_loss(r_body,r_hf))
        else:
            r_lf_step = torch.eye(3, dtype=lf.dtype, device=lf.device).expand(batch_size, 3, 3)
            r_hf_step = r_lf_step
            r_body_step = r_lf_step
            body_lf_relative = r_lf_step
            r_lf, r_hf, r_body = r_lf_step.unsqueeze(1), r_hf_step.unsqueeze(1), r_body_step.unsqueeze(1)
        rocker_step, wheel_step = {}, {}
        for name in ROCKER_NAMES:
            lf_key, hf_key = f"rollout_lf_rocker_{name}_pose", f"rollout_hf_rocker_{name}_pose"
            if lf_key not in batch or hf_key not in batch:
                continue
            lf_pose = batch[lf_key][:, step:step+1].float()
            hf_pose = batch[hf_key][:, step:step+1].float()
            local = out[f"pred_rocker_{name}_local_delta_pos"]
            part_gate = out[f"gate_rocker_{name}_pos"]
            local_step = _single_step_vector3(f"rocker_{name}.local_delta", local, batch_size, args)
            gate_step = _single_step_vector3(f"rocker_{name}.position_gate", part_gate, batch_size, args)
            local_world = torch.matmul(r_body_step, (gate_step*local_step).unsqueeze(-1)).squeeze(-1).unsqueeze(1)
            base = lf_pose[..., :3] + body_delta
            pred = base + local_world
            local_target_world = hf_pose[..., :3] - lf_pose[..., :3] - (hf[..., body_pos]-lf[..., body_pos])
            local_target_step = _single_step_vector3(f"rocker_{name}.local_target_world", local_target_world, batch_size, args)
            local_target = torch.matmul(r_hf_step.transpose(-1,-2), local_target_step.unsqueeze(-1)).squeeze(-1).unsqueeze(1)
            rocker_std=group_std(scaler,"hf_body",body_pos,pred)
            rocker_pos_losses.append(F.smooth_l1_loss((pred-hf_pose[..., :3])/rocker_std,torch.zeros_like(pred)))
            rocker_local_losses.append(F.smooth_l1_loss(local/rocker_std,local_target/rocker_std))
            gate_t = gate_target(local.detach(), local_target, tau=args.trust_lf_error_tau, temp=args.trust_lf_error_temp)
            gate_losses.append(F.smooth_l1_loss(part_gate, gate_t.detach()))
            rocker_rot_lf = _single_step_rotation(
                f"rocker_{name}.R_lf", quat_matrix(lf_pose[...,3:7]), batch_size, args
            )
            rocker_rot_hf = _single_step_rotation(
                f"rocker_{name}.R_hf", quat_matrix(hf_pose[...,3:7]), batch_size, args
            ).unsqueeze(1)
            rocker_rot_base = _rotation_matmul(
                f"rocker_{name}.R_body_relative_times_R_lf",
                body_lf_relative,
                rocker_rot_lf,
                batch_size,
                args,
            )
            rocker_rotvec = _single_step_vector3(
                f"rocker_{name}.gated_rotvec",
                out[f"gate_rocker_{name}_att"] * out[f"pred_rocker_{name}_delta_rot"],
                batch_size,
                args,
            )
            rocker_delta_rot = _single_step_rotation(
                f"rocker_{name}.rotvec_matrix", rotvec_matrix(rocker_rotvec), batch_size, args
            )
            rocker_rot_pred = _rotation_matmul(
                f"rocker_{name}.R_base_times_delta", rocker_rot_base, rocker_delta_rot, batch_size, args
            ).unsqueeze(1)
            attitude_losses.append(so3_loss(rocker_rot_pred,rocker_rot_hf))
            rocker_step[name]={"pos":pred,"rot":rocker_rot_pred}
            corrected_rockers[name] = rocker_step[name]
        for wheel_id in WHEEL_IDS:
            lf_w = batch[f"rollout_lf_wheel{wheel_id}_kin"][:, step:step+1].float()
            hf_w = batch[f"rollout_hf_wheel{wheel_id}_kin"][:, step:step+1].float()
            cols = spec.target_groups.wheel_kin_cols[wheel_id]
            pos_idx = indices(cols, ["pos_x", "pos_y", "pos_z"])
            omega_idx = omega_index(cols)
            res = residual_raw(out[f"pred_res_wheel{wheel_id}_kin"], spec.res_groups.wheel_kin_cols[wheel_id], cols, f"res_wheel{wheel_id}_kin", scaler)
            local = res[..., pos_idx]
            pg = out[f"gate_wheel{wheel_id}_pos"]
            local_step = _single_step_vector3(f"wheel{wheel_id}.local_delta", local, batch_size, args)
            gate_step = _single_step_vector3(f"wheel{wheel_id}.position_gate", pg, batch_size, args)
            local_world = torch.matmul(r_body_step, (gate_step*local_step).unsqueeze(-1)).squeeze(-1).unsqueeze(1)
            pred_pos = lf_w[..., pos_idx] + body_delta + local_world
            local_target_world = hf_w[..., pos_idx]-lf_w[..., pos_idx]-(hf[..., body_pos]-lf[..., body_pos])
            local_target_step = _single_step_vector3(f"wheel{wheel_id}.local_target_world", local_target_world, batch_size, args)
            local_target = torch.matmul(r_hf_step.transpose(-1, -2), local_target_step.unsqueeze(-1)).squeeze(-1).unsqueeze(1)
            wheel_std=group_std(scaler,f"hf_wheel{wheel_id}_kin",pos_idx,pred_pos)
            wheel_pos_losses.append(F.smooth_l1_loss((pred_pos-hf_w[...,pos_idx])/wheel_std,torch.zeros_like(pred_pos)))
            wheel_local_losses.append(F.smooth_l1_loss(local/wheel_std,local_target/wheel_std))
            wheel_local_regs.append((local/wheel_std).square().mean())
            wheel_pred_err=((pred_pos-hf_w[...,pos_idx])/wheel_std).square().mean(dim=(-1,-2))
            wheel_lf_err=((lf_w[...,pos_idx]-hf_w[...,pos_idx])/wheel_std).square().mean(dim=(-1,-2))
            wheel_harmful_steps.append(wheel_pred_err>wheel_lf_err)
            gate_t = gate_target(local.detach(), local_target, tau=args.trust_lf_error_tau, temp=args.trust_lf_error_temp)
            gate_losses.append(F.smooth_l1_loss(pg, gate_t.detach()))
            if omega_idx is not None:
                omega_pred = lf_w[..., omega_idx:omega_idx+1] + out[f"gate_wheel{wheel_id}_omega"] * res[..., omega_idx:omega_idx+1]
                std_om = group_std(scaler, f"hf_wheel{wheel_id}_kin", [omega_idx], omega_pred)
                wheel_omega_losses.append(F.smooth_l1_loss((omega_pred-hf_w[...,omega_idx:omega_idx+1])/std_om, torch.zeros_like(omega_pred)))
                omega_gate_target=gate_target(res[...,omega_idx:omega_idx+1].detach(),hf_w[...,omega_idx:omega_idx+1]-lf_w[...,omega_idx:omega_idx+1],std_om,args.trust_lf_error_tau,args.trust_lf_error_temp)
                gate_losses.append(F.smooth_l1_loss(out[f"gate_wheel{wheel_id}_omega"],omega_gate_target.detach()))
            pose_lf_key,pose_hf_key=f"rollout_lf_wheel{wheel_id}_pose",f"rollout_hf_wheel{wheel_id}_pose"
            if pose_lf_key in batch and pose_hf_key in batch:
                wheel_pose_lf=batch[pose_lf_key][:,step:step+1].float()
                wheel_pose_hf=batch[pose_hf_key][:,step:step+1].float()
                wheel_rot_lf = _single_step_rotation(
                    f"wheel{wheel_id}.R_lf", quat_matrix(wheel_pose_lf[...,3:7]), batch_size, args
                )
                wheel_rot_hf = _single_step_rotation(
                    f"wheel{wheel_id}.R_hf", quat_matrix(wheel_pose_hf[...,3:7]), batch_size, args
                ).unsqueeze(1)
                wheel_rot_base = _rotation_matmul(
                    f"wheel{wheel_id}.R_body_relative_times_R_lf",
                    body_lf_relative,
                    wheel_rot_lf,
                    batch_size,
                    args,
                )
                wheel_rotvec = _single_step_vector3(
                    f"wheel{wheel_id}.gated_rotvec",
                    out[f"gate_wheel{wheel_id}_att"] * out[f"pred_wheel{wheel_id}_delta_rot"],
                    batch_size,
                    args,
                )
                wheel_delta_rot = _single_step_rotation(
                    f"wheel{wheel_id}.rotvec_matrix", rotvec_matrix(wheel_rotvec), batch_size, args
                )
                wheel_rot_pred = _rotation_matmul(
                    f"wheel{wheel_id}.R_base_times_delta", wheel_rot_base, wheel_delta_rot, batch_size, args
                ).unsqueeze(1)
                attitude_losses.append(so3_loss(wheel_rot_pred,wheel_rot_hf))
            else:
                wheel_rot_pred=r_body
            wheel_step[wheel_id]=pred_pos
            corrected_wheels[wheel_id] = pred_pos
        assembly_losses.append(assembly_rollout_loss(batch,final,r_body,rocker_step,wheel_step))
        prev_pos, prev_vel = final, vel
        if step_started is not None:
            _debug_event(args, "reconstruct_rollout_step", time.perf_counter() - step_started)
    if not positions:
        raise ValueError("V5 requires at least one body rollout output")
    if _cuda_debug_enabled(args):
        args._debug_context["rollout_step"] = "all"
    delta_seq = torch.cat(delta_pos, dim=1)
    smooth = F.huber_loss(delta_seq[:, 1:] - delta_seq[:, :-1], torch.zeros_like(delta_seq[:, 1:])) if delta_seq.shape[1] > 1 else delta_seq.new_tensor(0.0)
    body_pos_loss = torch.stack(pos_losses).mean()
    body_vel_loss = torch.stack(vel_losses).mean()
    kin = torch.stack(kin_losses).mean()
    wheel_position_state = torch.stack(wheel_pos_losses).mean() if wheel_pos_losses else body_pos_loss.new_tensor(0.0)
    wheel_state = torch.stack(wheel_pos_losses + wheel_omega_losses).mean() if wheel_pos_losses or wheel_omega_losses else body_pos_loss.new_tensor(0.0)
    rocker_state = torch.stack(rocker_pos_losses).mean() if rocker_pos_losses else body_pos_loss.new_tensor(0.0)
    wheel_local = torch.stack(wheel_local_losses).mean() if wheel_local_losses else body_pos_loss.new_tensor(0.0)
    rocker_local = torch.stack(rocker_local_losses).mean() if rocker_local_losses else body_pos_loss.new_tensor(0.0)
    gate_loss = torch.stack(gate_losses).mean() if gate_losses else body_pos_loss.new_tensor(0.0)
    attitude = torch.stack(attitude_losses).mean() if attitude_losses else body_pos_loss.new_tensor(0.0)
    assembly = torch.stack(assembly_losses).mean() if assembly_losses else body_pos_loss.new_tensor(0.0)
    state = torch.stack([body_pos_loss, body_vel_loss, wheel_state, rocker_state + rocker_local]).mean()
    metrics = {
        "state": state, "body_pos": body_pos_loss, "body_vel": body_vel_loss,
        "wheel_state": wheel_state, "rocker_state": rocker_state, "wheel_local": wheel_local,
        "wheel_position_state":wheel_position_state,"wheel_local_reg":torch.stack(wheel_local_regs).mean() if wheel_local_regs else body_pos_loss.new_tensor(0.0),
        "body_delta_reg":delta_seq.square().mean(),
        "attitude":attitude,"assembly":assembly,"wheel_local_reg":wheel_local.square(),
        "body_delta_smooth": smooth, "kin": kin, "gate_state": gate_loss,
        "body_final_pos_rmse": torch.sqrt((torch.cat(positions, dim=1)-batch["rollout_hf_body"][:,:,body_pos]).square().mean()),
        "body_prop_pos_rmse": torch.sqrt((torch.cat(props, dim=1)-batch["rollout_hf_body"][:,:,body_pos]).square().mean()),
        "body_delta_abs": delta_seq.abs().mean(),
        "body_delta_x_abs": delta_seq[...,0].abs().mean(), "body_delta_y_abs": delta_seq[...,1].abs().mean(), "body_delta_z_abs": delta_seq[...,2].abs().mean(),
        "body_gate_mean": torch.cat(body_gates, dim=1).mean(),
        "body_harmful_ratio":torch.stack(body_harmful_steps,dim=1).float().mean(),
        "wheel_kin_harmful_ratio":torch.stack(wheel_harmful_steps,dim=1).float().mean() if wheel_harmful_steps else body_pos_loss.new_tensor(0.0),
        "state_output_score":torch.stack([body_pos_loss,body_vel_loss,wheel_state,rocker_state,attitude]).mean(),
        "rollout_score":torch.stack([body_pos_loss,body_vel_loss,wheel_position_state,rocker_state]).mean(),
    }
    metrics["harmful_ratio"]=(metrics["body_harmful_ratio"]+metrics["wheel_kin_harmful_ratio"])*0.5
    return metrics, {"position": torch.cat(positions, dim=1), "velocity": torch.cat(velocities, dim=1), "rockers": corrected_rockers, "wheels": corrected_wheels}


def prepare_rollout_outputs(model, batch, stage: str, args, need_current: bool = False):
    prefix = "teacher" if stage == "state_teacher" else "student"
    current = None
    if need_current:
        current_full = _profile_call(
            args, "forward_current",
            lambda: model.forward_state(batch, input_prefix=prefix, current_index=args.history_len),
        )
        current = {"z": current_full["z"]}
        del current_full
    rollout_keys = ["pred_res_body", "pred_delta_rot_body", "gate_body_pos", "gate_body_att"]
    for name in ROCKER_NAMES:
        rollout_keys.extend((
            f"pred_rocker_{name}_local_delta_pos", f"pred_rocker_{name}_delta_rot",
            f"gate_rocker_{name}_pos", f"gate_rocker_{name}_att",
        ))
    for wheel_id in WHEEL_IDS:
        rollout_keys.extend((
            f"pred_res_wheel{wheel_id}_kin", f"pred_wheel{wheel_id}_delta_rot",
            f"gate_wheel{wheel_id}_pos", f"gate_wheel{wheel_id}_omega", f"gate_wheel{wheel_id}_att",
        ))
    outputs = []
    for step in range(args.body_rollout_len):
        if _cuda_debug_enabled(args):
            args._debug_context["rollout_step"] = step
        step_batch = dict(batch)
        for key in ["system", "body", *ROCKER_NAMES, *[f"wheel{i}_kin" for i in WHEEL_IDS], *[f"wheel{i}_contact" for i in WHEEL_IDS]]:
            src = f"rollout_{prefix}_{key}"
            if src in batch:
                step_batch[f"{prefix}_{key}"] = batch[src][:, step]
        full_output = _profile_call(
            args, "forward_rollout_step",
            lambda: model.forward_state(step_batch, input_prefix=prefix, current_index=args.history_len),
        )
        outputs.append({key: full_output[key] for key in rollout_keys})
        del full_output, step_batch
    if _cuda_debug_enabled(args):
        args._debug_context["rollout_step"] = "all"
    return current, outputs


def build_force_state(model, batch, spec, scaler, params, args, detach: bool) -> Tuple[Dict, Dict]:
    h = args.history_len
    prefix = "student"
    out_prev = model.forward_state(batch, input_prefix=prefix, current_index=h-1)
    out_cur = model.forward_state(batch, input_prefix=prefix, current_index=h)
    body_cols = spec.target_groups.body_cols
    bpos, bvel = indices(body_cols,["pos_x","pos_y","pos_z"]), indices(body_cols,["vel_x","vel_y","vel_z"])
    body_sequence = batch["lf_body_raw_sequence"].float()
    body_prev_lf = body_sequence[:, h-1:h]
    body_cur_lf = body_sequence[:, h:h+1]
    body_init_lf = body_sequence[:, h-2:h-1]
    dprev, dcur = raw_body_delta(out_prev,spec,scaler), raw_body_delta(out_cur,spec,scaler)
    vprev = body_prev_lf[...,bvel] + dprev[...,bvel]
    vcur = body_cur_lf[...,bvel] + dcur[...,bvel]
    dt = batch["dt"].view(-1,1,1).to(body_cur_lf).clamp_min(1e-6)
    dt_prev = batch["state_dt_prev"].view(-1,1,1).to(body_cur_lf).clamp_min(1e-6)
    v_init = body_init_lf[...,bvel]
    pprev = body_init_lf[...,bpos] + 0.5*(v_init+vprev)*dt_prev + out_prev["gate_body_pos"]*dprev[...,bpos]
    pcur = pprev + 0.5*(vprev+vcur)*dt + out_cur["gate_body_pos"]*dcur[...,bpos]
    body_pose_sequence = batch["lf_body_pose_sequence"].float()
    prev_pose = body_pose_sequence[:,h-1:h]
    cur_pose = body_pose_sequence[:,h:h+1]
    rprev_lf = quat_matrix(prev_pose[...,3:7])
    rcur_lf = quat_matrix(cur_pose[...,3:7])
    rprev = torch.matmul(rprev_lf,rotvec_matrix(out_prev["gate_body_att"]*out_prev["pred_delta_rot_body"]))
    rcur = torch.matmul(rcur_lf,rotvec_matrix(out_cur["gate_body_att"]*out_cur["pred_delta_rot_body"]))
    wheel_pos, wheel_omega, force_base, fphy_z, wheel_vz_values = {}, {}, {}, {}, {}
    acc = (vcur-vprev)/dt
    roll = torch.atan2(rcur[...,2,1],rcur[...,2,2])
    pitch = torch.asin((-rcur[...,2,0]).clamp(-1,1)).unsqueeze(-1)
    roll = roll.unsqueeze(-1)
    for i in WHEEL_IDS:
        cols = spec.target_groups.wheel_kin_cols[i]
        pidx, oidx = indices(cols,["pos_x","pos_y","pos_z"]), omega_index(cols)
        res = residual_raw(out_cur[f"pred_res_wheel{i}_kin"],spec.res_groups.wheel_kin_cols[i],cols,f"res_wheel{i}_kin",scaler)
        res_prev = residual_raw(out_prev[f"pred_res_wheel{i}_kin"],spec.res_groups.wheel_kin_cols[i],cols,f"res_wheel{i}_kin",scaler)
        lf_w_sequence = batch[f"lf_wheel{i}_kin_raw_sequence"].float()
        lf_w_prev, lf_w = lf_w_sequence[:,h-1:h], lf_w_sequence[:,h:h+1]
        body_delta_prev = pprev-body_prev_lf[...,bpos]
        body_delta_cur = pcur-body_cur_lf[...,bpos]
        local_prev = res_prev[...,pidx]
        local_prev_world = torch.matmul(rprev,(out_prev[f"gate_wheel{i}_pos"]*local_prev).unsqueeze(-1)).squeeze(-1)
        wheel_pos_prev = lf_w_prev[...,pidx]+body_delta_prev+local_prev_world
        local = res[...,pidx]
        local_world = torch.matmul(rcur,(out_cur[f"gate_wheel{i}_pos"]*local).unsqueeze(-1)).squeeze(-1)
        wheel_pos[i] = lf_w[...,pidx]+body_delta_cur+local_world
        if oidx is not None:
            wheel_omega_prev = lf_w_prev[...,oidx:oidx+1]+out_prev[f"gate_wheel{i}_omega"]*res_prev[...,oidx:oidx+1]
            wheel_omega[i] = lf_w[...,oidx:oidx+1]+out_cur[f"gate_wheel{i}_omega"]*res[...,oidx:oidx+1]
        else:
            wheel_omega_prev = lf_w.new_zeros(*lf_w.shape[:-1],1)
            wheel_omega[i] = lf_w.new_zeros(*lf_w.shape[:-1],1)
        wheel_vz = (wheel_pos[i][...,2:3]-wheel_pos_prev[...,2:3])/dt
        wheel_vz_values[i]=wheel_vz
        fz = load_transfer_fz_torch(roll,pitch,acc,wheel_vz,i,params)
        lf_force = batch[f"wheel{i}_lf_force_current"].float()
        force_base[i] = torch.cat([lf_force[...,:2],fz],dim=-1)
        fphy_z[i] = fz
    state = {
        "z": out_cur["z"], "body_node": out_cur["body_node"],
        "wheel_kin_nodes": out_cur["wheel_kin_nodes"], "wheel_contact_nodes": out_cur["wheel_contact_nodes"],
        "body_velocity": vcur, "body_acceleration":acc, "body_rotation": rcur, "body_roll":roll,"body_pitch":pitch,"wheel_position": wheel_pos,
        "wheel_omega": wheel_omega, "force_base": force_base, "fphy_z": fphy_z,"wheel_vz":wheel_vz_values,
    }
    if detach:
        state = {k: ({i: v.detach() for i,v in val.items()} if isinstance(val,dict) else val.detach() if isinstance(val,torch.Tensor) else val) for k,val in state.items()}
    return state, {"state_prev":out_prev,"state_current":out_cur,"roll":roll,"pitch":pitch,"acc":acc}


def force_losses(model, batch, state, spec, scaler, params, args) -> Dict[str, torch.Tensor]:
    out = model.forward_force(state,batch,input_prefix="student",current_index=args.history_len)
    terms, residual_regs, harmful = [], [], []
    metrics = {}
    for i in WHEEL_IDS:
        cols = spec.target_groups.wheel_contact_cols[i]
        fidx = indices(cols,["Fx","Fy","Fz"])
        target = inverse(batch[f"hf_wheel{i}_contact"],scaler,f"hf_wheel{i}_contact")[...,fidx]
        pred = out[f"force_pred_wheel{i}"]
        base = state["force_base"][i]
        lf_force=batch[f"wheel{i}_lf_force_current"].float()
        std = group_std(scaler,f"hf_wheel{i}_contact",fidx,pred)
        terms.append(F.smooth_l1_loss((pred-target)/std,torch.zeros_like(pred)))
        residual_regs.append(((out[f"delta_force_wheel{i}"]/std)**2).mean())
        pred_norm=(pred-target).div(std).square().mean(dim=(-1,-2))
        lf_norm=(lf_force-target).div(std).square().mean(dim=(-1,-2))
        harmful.append(pred_norm>lf_norm)
        metrics[f"wheel{i}_force_harmful_ratio"]=(pred_norm>lf_norm).float().mean()
        metrics[f"wheel{i}_F_LF_rmse"] = torch.sqrt((lf_force-target).square().mean())
        metrics[f"wheel{i}_F_base_rmse"] = torch.sqrt((base-target).square().mean())
        metrics[f"wheel{i}_F_final_rmse"] = torch.sqrt((pred-target).square().mean())
        metrics[f"wheel{i}_Fx_rmse"] = torch.sqrt((pred[...,0]-target[...,0]).square().mean())
        metrics[f"wheel{i}_Fy_rmse"] = torch.sqrt((pred[...,1]-target[...,1]).square().mean())
        metrics[f"wheel{i}_Fz_rmse"] = torch.sqrt((pred[...,2]-target[...,2]).square().mean())
        metrics[f"wheel{i}_Fphy_z_rmse"] = torch.sqrt((state["fphy_z"][i]-target[...,2:3]).square().mean())
    metrics["force"] = torch.stack(terms).mean()
    metrics["force_residual_reg"] = torch.stack(residual_regs).mean()
    metrics["F_final_rmse"] = torch.stack([metrics[f"wheel{i}_F_final_rmse"] for i in WHEEL_IDS]).mean()
    metrics["F_base_rmse"] = torch.stack([metrics[f"wheel{i}_F_base_rmse"] for i in WHEEL_IDS]).mean()
    metrics["F_LF_rmse"] = torch.stack([metrics[f"wheel{i}_F_LF_rmse"] for i in WHEEL_IDS]).mean()
    metrics["Fphy_z_rmse"] = torch.stack([metrics[f"wheel{i}_Fphy_z_rmse"] for i in WHEEL_IDS]).mean()
    metrics["wheel_force_harmful_ratio"] = torch.stack([v.float().mean() for v in harmful]).mean()
    metrics["harmful_ratio"] = metrics["wheel_force_harmful_ratio"]
    metrics["pred_roll_mean"] = state["body_roll"].mean()
    metrics["pred_pitch_mean"] = state["body_pitch"].mean()
    metrics["pred_acc_abs_mean"] = state["body_acceleration"].abs().mean()
    metrics["pred_wheel_vz_abs_mean"] = torch.stack([v.abs().mean() for v in state["wheel_vz"].values()]).mean()
    total_force = torch.stack([out[f"force_pred_wheel{i}"] for i in WHEEL_IDS]).sum(dim=0)
    gravity=total_force.new_tensor([0.0,0.0,-9.81]).view(1,1,3)
    acc_from_force=total_force/max(args.mass,1e-6)+gravity
    metrics["dyn"] = F.smooth_l1_loss(state["body_acceleration"],acc_from_force)
    metrics["total"] = metrics["force"] + args.lambda_force_residual_reg*metrics["force_residual_reg"] + args.lambda_dyn*metrics["dyn"]
    return metrics


def state_total(metrics: Dict[str,torch.Tensor], args) -> torch.Tensor:
    return (
        metrics["state"] + args.lambda_wheel_local*metrics["wheel_local"]
        + args.lambda_attitude*metrics["attitude"] + args.lambda_assembly*metrics["assembly"]
        + args.lambda_body_delta_reg*metrics["body_delta_reg"]
        + args.lambda_body_delta_smooth*metrics["body_delta_smooth"]
        + args.lambda_wheel_local_reg*metrics["wheel_local_reg"]
        + args.lambda_gate*metrics["gate_state"] + args.lambda_kin*metrics["kin"]
    )


def fit_one_batch(model, batch, stage, teacher, params, spec, scaler, args):
    if stage == "force":
        with torch.no_grad():
            corrected, _ = build_force_state(model,batch,spec,scaler,params,args,detach=True)
        metrics=force_losses(model,batch,corrected,spec,scaler,params,args)
        return metrics
    current, rollout = _profile_call(
        args, "forward",
        lambda: prepare_rollout_outputs(model,batch,stage,args,need_current=teacher is not None),
    )
    state_metrics, _ = _profile_call(
        args, "loss_reconstruction",
        lambda: reconstruct_rollout_state(batch,rollout,spec,scaler,args),
    )
    total = _profile_call(args, "loss_total", lambda: state_total(state_metrics,args))
    if teacher is not None:
        def distillation_loss():
            with torch.no_grad():
                teacher_out = teacher.forward_state(batch,input_prefix="teacher",current_index=args.history_len)
            state_metrics["distill"] = F.smooth_l1_loss(current["z"],teacher_out["z"])
            return total + args.lambda_distill_z*state_metrics["distill"]
        total = _profile_call(args, "loss_distillation", distillation_loss)
    if stage == "joint":
        corrected, aux = build_force_state(model,batch,spec,scaler,params,args,detach=False)
        force_metric = force_losses(model,batch,corrected,spec,scaler,params,args)
        if stage == "force":
            total = force_metric["total"]
        else:
            total = total + force_metric["total"]
        state_metrics.update(force_metric)
    del rollout, current
    state_metrics["total"] = total
    return state_metrics


def run_epoch(model, loader, stage, teacher, params, spec, scaler, args, optimizer=None, epoch: int = 0, max_batches: Optional[int] = None):
    training = optimizer is not None
    grad_accum_steps = max(1, int(getattr(args, "grad_accum_steps", 1)))
    num_batches = min(len(loader), max_batches) if max_batches is not None else len(loader)
    phase_name = "train" if training else "val"
    model.train(training)
    if stage == "force":
        for module in (model.input_encoders,model.hgt_layers,model.temporal_reduce,model.tcn,model.temporal_net,
                       model.body_position_correction_head,model.body_velocity_head,model.body_att_head,
                       model.rocker_pos_head,model.rocker_att_head,model.wheel_local_position_head,
                       model.wheel_omega_head,model.wheel_att_head,model.body_pos_gate_head,
                       model.body_att_gate_head,model.rocker_pos_gate_head,model.rocker_att_gate_head,
                       model.wheel_pos_gate_head,model.wheel_omega_gate_head,model.wheel_att_gate_head):
            module.eval()
    if teacher is not None:
        teacher.eval()
    totals: Dict[str,float] = {}
    n = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    if args.device_obj.type == "cuda" and (args.debug_rotations or args.debug_cuda):
        torch.cuda.reset_peak_memory_stats(args.device_obj)
    progress = None
    iterator = islice(loader, num_batches) if max_batches is not None else loader
    if tqdm is not None and args.progress_bar:
        progress = tqdm(
            iterator,
            total=num_batches,
            desc=f"{stage} {phase_name} epoch={epoch:04d}",
            dynamic_ncols=True,
            leave=False,
            file=sys.stdout,
        )
        iterator = progress
    progress_keys = (
        "total", "state", "body_pos", "body_vel", "wheel_state", "rocker_state",
        "wheel_local", "rollout_score", "body_delta_smooth", "distill",
        "force", "F_final_rmse", "F_base_rmse", "Fphy_z_rmse", "dyn",
    )
    for batch_idx, batch in enumerate(iterator):
        batch = move_batch(batch,args.device_obj)
        if getattr(args, "debug_cuda", False):
            args._debug_context = {
                "epoch": epoch,
                "phase": phase_name,
                "batch": batch_idx + 1,
                "num_batches": num_batches,
                "rollout_step": None,
            }
            args._debug_failure_context = None
        with torch.set_grad_enabled(training):
            metrics = fit_one_batch(model,batch,stage,teacher,params,spec,scaler,args)
            if _cuda_debug_enabled(args):
                bad_losses = [
                    key for key, value in metrics.items()
                    if isinstance(value, torch.Tensor) and value.is_floating_point()
                    and not bool(torch.isfinite(value.detach()).all().item())
                ]
                if bad_losses:
                    error = FloatingPointError(f"Non-finite training values: {bad_losses}")
                    _write_debug_failure(args, "finite_loss_check", error)
                    raise error
            if training:
                batch_size = int(batch["system"].shape[0])
                loader_batch_size = int(loader.batch_size or args.batch_size)
                group_start = (batch_idx // grad_accum_steps) * grad_accum_steps
                group_end = min(group_start + grad_accum_steps, num_batches)
                sample_weight = accumulation_sample_weight(
                    batch_idx, batch_size, loader_batch_size, len(loader.dataset),
                    grad_accum_steps, num_batches,
                )
                _profile_call(
                    args, "backward",
                    lambda: (metrics["total"] * sample_weight).backward(),
                )
                if _cuda_debug_enabled(args):
                    bad_grad = next((
                        name for name, parameter in model.named_parameters()
                        if parameter.requires_grad and parameter.grad is not None
                        and not bool(torch.isfinite(parameter.grad.detach()).all().item())
                    ), None)
                    if bad_grad is not None:
                        error = FloatingPointError(f"Non-finite gradient in {bad_grad}")
                        _write_debug_failure(args, "finite_gradient_check", error)
                        raise error
                if batch_idx + 1 == group_end:
                    def optimizer_update():
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad], args.grad_clip,
                        )
                        if _cuda_debug_enabled(args) and not bool(torch.isfinite(grad_norm.detach()).all().item()):
                            raise FloatingPointError(f"Non-finite gradient norm: {grad_norm}")
                        optimizer.step()
                    _profile_call(args, "optimizer_step", optimizer_update)
                    optimizer.zero_grad(set_to_none=True)
        n += 1
        for key,val in metrics.items():
            if isinstance(val,torch.Tensor) and val.numel()==1 and torch.isfinite(val):
                totals[key]=totals.get(key,0.0)+float(val.detach().cpu())
        if args.device_obj.type == "cuda" and getattr(args, "debug_rotations", False) and not getattr(args, "debug_cuda", False):
            allocated = torch.cuda.memory_allocated(args.device_obj) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(args.device_obj) / (1024 ** 2)
            peak = torch.cuda.max_memory_allocated(args.device_obj) / (1024 ** 2)
            print(
                f"[V5][cuda-memory] {stage} {phase_name} batch={n}/{num_batches} "
                f"allocated={allocated:.1f}MiB reserved={reserved:.1f}MiB peak={peak:.1f}MiB",
                flush=True,
            )
        if _cuda_debug_enabled(args):
            _debug_event(args, "batch_complete")
        if args.progress_interval > 0 and (
            n == 1 or n % args.progress_interval == 0 or n == num_batches
        ):
            current = {
                key: float(metrics[key].detach().cpu())
                for key in progress_keys
                if key in metrics and isinstance(metrics[key],torch.Tensor)
                and metrics[key].numel() == 1 and torch.isfinite(metrics[key])
            }
            if _cuda_debug_enabled(args) and not getattr(args, "debug_cuda_verbose", False):
                current.update(format_cuda_debug_postfix(args._debug_context))
            if progress is not None:
                progress.set_postfix(current, refresh=True)
            elif args.progress_bar:
                percentage = 100.0 * n / max(num_batches, 1)
                print(
                    f"[V5] {stage} {phase_name} epoch={epoch:04d} "
                    f"batch={n}/{num_batches} ({percentage:5.1f}%) "
                    + " ".join(f"{key}={value:.5g}" for key,value in current.items()),
                    flush=True,
                )
        del metrics, batch
    if progress is not None:
        progress.close()
    return {key:value/max(n,1) for key,value in totals.items()}


def checkpoint_payload(model, scaler, spec, args, epoch, metric, params):
    return {
        "model_state_dict":model.state_dict(), "args":vars(args), "epoch":epoch,
        "best_metric":metric, "v5_role":args.train_stage, "train_stage":args.train_stage,
        "state_architecture_args":{"hidden_dim":args.hidden_dim,"graph_layers":args.graph_layers,"tcn_dim":args.tcn_dim,"lstm_dim":args.lstm_dim,"lstm_layers":args.lstm_layers,"history_len":args.history_len,"teacher_future_len":args.teacher_future_len,"body_rollout_len":args.body_rollout_len},
        "force_architecture_args":{"force_hidden_dim":args.force_hidden_dim,"contact_history_dim":7},
        "load_transfer_params":None if params is None else np.asarray(params,dtype=np.float32),
        "scaler_state":{k:v.to_dict() for k,v in scaler.scalers.items()},
        "column_spec":spec.to_dict(),
    }


def save_ckpt(path, model, scaler, spec, args, epoch, metric, params):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    torch.save(checkpoint_payload(model,scaler,spec,args,epoch,metric,params), temporary)
    os.replace(temporary, target)


def save_epoch_resume_checkpoint(path, model, optimizer, scaler, spec, args, epoch, params, best_scores):
    payload = checkpoint_payload(model, scaler, spec, args, epoch, best_scores["total"], params)
    payload.update({
        "resume_format": "v5_complete_epoch_v1",
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "best_metric": float(best_scores["total"]),
        "best_state_output": float(best_scores["state_output"]),
        "best_rollout": float(best_scores["rollout"]),
    })
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)


def compatible_load(model, state):
    current=model.state_dict()
    compatible={k:v for k,v in state.items() if k in current and current[k].shape==v.shape}
    missing,_=model.load_state_dict(compatible,strict=False)
    print(f"[V5] compatible init: loaded={len(compatible)} missing={len(missing)}",flush=True)


def log_epoch_metrics(epoch, stage, phase, metrics):
    groups = (
        ("state",("state","body_pos","body_vel","wheel_state","rocker_state","wheel_local")),
        ("rollout",("rollout_score","body_prop_pos_rmse","body_final_pos_rmse","body_delta_smooth")),
        ("attitude/assembly/gate/kin",("attitude","assembly","gate_state","kin","distill")),
        ("force/physics/residual",("force","dyn","force_residual_reg","F_LF_rmse","F_base_rmse","Fphy_z_rmse","F_final_rmse","pred_roll_mean","pred_pitch_mean","pred_acc_abs_mean","pred_wheel_vz_abs_mean")),
    )
    print(f"[V5] epoch={epoch:04d} stage={stage} {phase}",flush=True)
    for name,keys in groups:
        parts=[f"{key}={metrics[key]:.5g}" for key in keys if key in metrics and math.isfinite(metrics[key])]
        if name.startswith("force"):
            parts += [f"wheel{i}_F{x}_rmse={metrics[f'wheel{i}_F{x}_rmse']:.5g}" for i in WHEEL_IDS for x in ("x","y","z") if f"wheel{i}_F{x}_rmse" in metrics]
        if parts:
            print(f"  {name}: {' '.join(parts)}",flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--feature_dir",default=str(ROOT/"Feature_Selection"/"DataSet"))
    parser.add_argument("--merged_csv",default=str(ROOT/"Feature_Selection"/"DataSet"/"merged_error_dataset.csv"))
    parser.add_argument("--save_dir",default=str(ROOT/"results_v5"))
    parser.add_argument("--train_stage",choices=STAGES,default="state_teacher")
    parser.add_argument("--state_checkpoint",default=None)
    parser.add_argument("--force_checkpoint",default=None)
    parser.add_argument("--init_ckpt",default=None)
    parser.add_argument("--resume_ckpt",default=None,help="从最近一个已完成 Epoch 的 last_epoch.pt 恢复模型、优化器和最佳指标")
    parser.add_argument("--history_len",type=int,default=9)
    parser.add_argument("--teacher_future_len",type=int,default=3)
    parser.add_argument("--body_rollout_len",type=int,default=5)
    parser.add_argument("--batch_size",type=int,default=128)
    parser.add_argument("--grad_accum_steps",type=int,default=1)
    parser.add_argument("--epochs",type=int,default=100)
    parser.add_argument("--lr",type=float,default=3e-4)
    parser.add_argument("--weight_decay",type=float,default=1e-5)
    parser.add_argument("--hidden_dim",type=int,default=192)
    parser.add_argument("--graph_layers",type=int,default=3)
    parser.add_argument("--tcn_dim",type=int,default=256)
    parser.add_argument("--lstm_dim",type=int,default=192)
    parser.add_argument("--lstm_layers",type=int,default=2)
    parser.add_argument("--dropout",type=float,default=0.1)
    parser.add_argument("--force_hidden_dim",type=int,default=48)
    parser.add_argument("--lambda_distill_z",type=float,default=1.0)
    parser.add_argument("--lambda_wheel_local",type=float,default=0.75)
    parser.add_argument("--lambda_attitude",type=float,default=0.2)
    parser.add_argument("--lambda_assembly",type=float,default=0.2)
    parser.add_argument("--lambda_body_delta_reg",type=float,default=1e-3)
    parser.add_argument("--lambda_body_delta_smooth",type=float,default=0.01)
    parser.add_argument("--lambda_wheel_local_reg",type=float,default=1e-4)
    parser.add_argument("--lambda_gate",type=float,default=0.2)
    parser.add_argument("--lambda_kin",type=float,default=0.02)
    parser.add_argument("--lambda_dyn",type=float,default=0.0)
    parser.add_argument("--mass",type=float,default=240.0)
    parser.add_argument("--lambda_force_residual_reg",type=float,default=1e-5)
    parser.add_argument("--trust_lf_error_tau",type=float,default=0.5)
    parser.add_argument("--trust_lf_error_temp",type=float,default=0.2)
    parser.add_argument("--state_trainable_layers",nargs="+",default=["hgt_layers.2","temporal_reduce","tcn","temporal_net"])
    parser.add_argument("--num_workers",type=int,default=4)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--grad_clip",type=float,default=5.0)
    parser.add_argument("--progress_bar",action=argparse.BooleanOptionalAction,default=True,help="显示 train/val batch 进度条")
    parser.add_argument("--progress_interval",type=int,default=10,help="每隔多少个 batch 更新 loss；<=0 表示只更新进度")
    parser.add_argument("--debug_rotations",action=argparse.BooleanOptionalAction,default=False,help="检查姿态 shape/dtype/device/finite，并记录显存")
    parser.add_argument("--debug_cuda_sync",action=argparse.BooleanOptionalAction,default=False,help="debug_rotations 下在旋转矩阵操作后同步 CUDA")
    parser.add_argument("--debug_cuda",action=argparse.BooleanOptionalAction,default=False,help="逐阶段同步并记录 Epoch/Batch/Rollout Step、耗时、显存及异常上下文")
    parser.add_argument("--debug_cuda_verbose",action=argparse.BooleanOptionalAction,default=False,help="debug_cuda 下逐阶段打印详细日志；默认将数据汇总进度条")
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args=parser.parse_args()
    args.debug_cuda = args.debug_cuda or args.debug_cuda_verbose
    if args.body_rollout_len<2: raise ValueError("body_rollout_len must be >=2")
    if args.batch_size<1 or args.grad_accum_steps<1: raise ValueError("batch_size and grad_accum_steps must be >=1")
    if args.lambda_body_delta_smooth<=0: raise ValueError("lambda_body_delta_smooth must be >0")
    args.device_obj=torch.device(args.device)
    set_seed(args.seed)
    if args.train_stage in {"force","joint"} and not (args.state_checkpoint or args.force_checkpoint or args.resume_ckpt):
        raise ValueError(f"{args.train_stage} requires --state_checkpoint (force) or --force_checkpoint (joint)")
    resume_payload = None
    if args.resume_ckpt:
        resume_path = Path(args.resume_ckpt).resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
        resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        saved_args = resume_payload.get("args", {})
        runtime_keys = {
            "resume_ckpt", "epochs", "train_stage", "device", "debug_cuda", "debug_rotations",
            "debug_cuda_sync", "debug_cuda_verbose", "progress_bar", "progress_interval", "num_workers",
            "device_obj", "run_dir", "start_epoch",
        }
        explicit_flags = {token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")}
        for key, value in saved_args.items():
            if key in runtime_keys or not hasattr(args, key):
                continue
            setattr(args, key, value)
        if "--epochs" not in explicit_flags:
            args.epochs = int(saved_args.get("epochs", args.epochs))
        if resume_payload.get("train_stage") != args.train_stage:
            raise ValueError(
                f"resume stage mismatch: checkpoint={resume_payload.get('train_stage')} cli={args.train_stage}"
            )
        if resume_payload.get("resume_format") != "v5_complete_epoch_v1":
            raise ValueError("--resume_ckpt must point to a V5 last_epoch.pt complete-epoch checkpoint")
        args.run_dir=str(resume_path.parent)
    else:
        stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_dir=os.path.join(args.save_dir,args.train_stage,stamp)
    os.makedirs(args.run_dir,exist_ok=True)
    spec,scaler,df_train,df_val,_,train_ds,val_ds,_=prepare_datasets_and_scaler(
        args.feature_dir,args.merged_csv,history_len=args.history_len,teacher_future_len=args.teacher_future_len,body_rollout_len=args.body_rollout_len,seed=args.seed
    )
    save_column_spec_json(spec,os.path.join(args.run_dir,"column_spec_v5.json"))
    scaler.save(os.path.join(args.run_dir,"group_scaler_v5.joblib"))
    body_res=spec.res_groups.body_cols
    body_pos=indices(body_res,["pos_x","pos_y","pos_z"])
    body_vel=indices(body_res,["vel_x","vel_y","vel_z"])
    wheel_res=spec.res_groups.wheel_kin_cols[0]
    wheel_pos=indices(wheel_res,["pos_x","pos_y","pos_z"])
    wheel_omega=omega_index(wheel_res)
    if len(body_pos)!=3 or len(body_vel)!=3 or len(wheel_pos)!=3:
        raise ValueError("V5 state heads require body xyz position/velocity and wheel xyz residual columns")
    if wheel_omega is None:
        print("[V5] Wheel omega residual columns are absent; wheel omega supervision is disabled for this dataset",flush=True)
    for wheel_id in WHEEL_IDS:
        cols=spec.res_groups.wheel_kin_cols[wheel_id]
        if len(cols)!=len(wheel_res) or indices(cols,["pos_x","pos_y","pos_z"])!=wheel_pos or omega_index(cols)!=wheel_omega:
            raise ValueError(f"V5 shared Wheel heads require matching residual column layout; wheel {wheel_id} differs")
    model=HGTGraphTemporalCompensationModelV5(
        get_group_dims(spec),node_hidden_dim=args.hidden_dim,graph_layers=args.graph_layers,
        tcn_hidden_dim=args.tcn_dim,lstm_hidden_dim=args.lstm_dim,lstm_layers=args.lstm_layers,
        dropout=args.dropout,role="teacher" if args.train_stage=="state_teacher" else "student",
        body_pos_indices=body_pos,body_vel_indices=body_vel,wheel_pos_indices=wheel_pos,
        wheel_omega_index=wheel_omega,force_hidden_dim=args.force_hidden_dim,
    ).to(args.device_obj)
    model.configure_rotation_debug(args.debug_rotations or args.debug_cuda_verbose, args.debug_cuda_sync or args.debug_cuda)
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state_dict"], strict=True)
    load_path=args.state_checkpoint if args.train_stage=="force" else args.force_checkpoint if args.train_stage=="joint" else args.init_ckpt
    teacher=None
    params=None
    if args.train_stage=="state_student":
        if not load_path: raise ValueError("state_student requires --init_ckpt for state_teacher")
        ckpt=torch.load(load_path,map_location="cpu",weights_only=False)
        teacher=HGTGraphTemporalCompensationModelV5(
            get_group_dims(spec),node_hidden_dim=int(ckpt["args"]["hidden_dim"]),graph_layers=int(ckpt["args"]["graph_layers"]),
            tcn_hidden_dim=int(ckpt["args"]["tcn_dim"]),lstm_hidden_dim=int(ckpt["args"]["lstm_dim"]),lstm_layers=int(ckpt["args"]["lstm_layers"]),
            role="teacher",body_pos_indices=body_pos,body_vel_indices=body_vel,wheel_pos_indices=wheel_pos,wheel_omega_index=wheel_omega,
        )
        teacher.configure_rotation_debug(args.debug_rotations or args.debug_cuda_verbose, args.debug_cuda_sync or args.debug_cuda)
        compatible_load(teacher,ckpt["model_state_dict"]); teacher.to(args.device_obj).eval()
        for p in teacher.parameters(): p.requires_grad_(False)
        if resume_payload is None:
            compatible_load(model,ckpt["model_state_dict"])
    elif args.train_stage in {"force","joint"}:
        if resume_payload is not None:
            if resume_payload.get("load_transfer_params") is not None:
                params=np.asarray(resume_payload["load_transfer_params"],dtype=np.float32)
            elif args.train_stage=="force":
                params=fit_load_transfer_params(df_train)
            else:
                raise ValueError("joint resume checkpoint is missing train-fitted load_transfer_params")
        else:
            ckpt=torch.load(load_path,map_location="cpu",weights_only=False)
            compatible_load(model,ckpt["model_state_dict"])
            if args.train_stage=="force":
                params=fit_load_transfer_params(df_train)
            else:
                params=np.asarray(ckpt["load_transfer_params"],dtype=np.float32)
    elif load_path and resume_payload is None:
        ckpt=torch.load(load_path,map_location="cpu",weights_only=False)
        compatible_load(model,ckpt["model_state_dict"])
    if params is not None:
        np.save(os.path.join(args.run_dir,"load_transfer_params_train.npy"),params)
    model_prefixes=("force_history_encoder.","force_residual_head.")
    if args.train_stage=="force":
        for name,p in model.named_parameters(): p.requires_grad_(name.startswith(model_prefixes))
    elif args.train_stage=="joint":
        for name,p in model.named_parameters():
            p.requires_grad_(name.startswith(model_prefixes) or any(name.startswith(prefix) for prefix in args.state_trainable_layers))
    else:
        for name,p in model.named_parameters(): p.requires_grad_(not name.startswith(model_prefixes))
    force_params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith(model_prefixes)]
    state_params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith(model_prefixes)]
    groups=[]
    if state_params:
        groups.append({"params":state_params,"lr":args.lr*(0.1 if args.train_stage=="joint" else 1.0)})
    if force_params:
        groups.append({"params":force_params,"lr":args.lr})
    optimizer=torch.optim.AdamW(groups,weight_decay=args.weight_decay)
    if resume_payload is not None:
        args.start_epoch, best_scores = restore_training_checkpoint(model, optimizer, resume_payload)
        print(
            f"[V5] resumed stage={args.train_stage} epoch={resume_payload['epoch']} "
            f"next_epoch={args.start_epoch} checkpoint={args.resume_ckpt}", flush=True,
        )
        resume_payload = None
    else:
        args.start_epoch=1
        best_scores={"total":math.inf,"state_output":math.inf,"rollout":math.inf}
    kwargs={"batch_size":args.batch_size,"num_workers":args.num_workers,"collate_fn":graph_temporal_collate_fn,"pin_memory":args.device_obj.type=="cuda"}
    train_loader=DataLoader(train_ds,shuffle=True,**kwargs)
    val_loader=DataLoader(val_ds,shuffle=False,**kwargs) if val_ds is not None else None
    metric_name={"state_teacher":"state_total","state_student":"state_total","force":"normalized_force_loss","joint":"joint_total"}[args.train_stage]
    if args.device_obj.type == "cuda" and args.debug_cuda:
        torch.cuda.reset_peak_memory_stats(args.device_obj)
    for epoch in range(args.start_epoch,args.epochs+1):
        train_metrics=run_epoch(model,train_loader,args.train_stage,teacher,params,spec,scaler,args,optimizer,epoch)
        valid_metrics=run_epoch(model,val_loader,args.train_stage,teacher,params,spec,scaler,args,epoch=epoch) if val_loader else train_metrics
        log_epoch_metrics(epoch,args.train_stage,"train",train_metrics)
        log_epoch_metrics(epoch,args.train_stage,"val",valid_metrics)
        value=float(valid_metrics.get("total",math.inf))
        if args.train_stage in {"state_teacher","state_student"}:
            state_output=float(valid_metrics.get("state_output_score",math.inf))
            rollout_score=float(valid_metrics.get("rollout_score",math.inf))
            output_path=os.path.join(args.run_dir,"best_state_output.pt")
            if state_output<best_scores["state_output"]:
                best_scores["state_output"]=state_output; save_ckpt(output_path,model,scaler,spec,args,epoch,"state_output_score",params)
            rollout_path=os.path.join(args.run_dir,"best_rollout.pt")
            if rollout_score<best_scores["rollout"]:
                best_scores["rollout"]=rollout_score; save_ckpt(rollout_path,model,scaler,spec,args,epoch,"normalized_rollout_score",params)
            total_path=os.path.join(args.run_dir,"best_state_total.pt")
        elif args.train_stage=="force":
            value=float(valid_metrics.get("force",math.inf))
            total_path=os.path.join(args.run_dir,"best_force.pt")
        else: total_path=os.path.join(args.run_dir,"best_joint.pt")
        if value<best_scores["total"]:
            best_scores["total"]=value; save_ckpt(total_path,model,scaler,spec,args,epoch,metric_name,params)
        save_epoch_resume_checkpoint(
            os.path.join(args.run_dir,"last_epoch.pt"), model, optimizer, scaler, spec,
            args, epoch, params, best_scores,
        )
        if args.device_obj.type == "cuda" and args.debug_cuda:
            _debug_event(args, "epoch_complete")
    print(f"[V5] complete stage={args.train_stage} best={best_scores['total']:.6g} dir={args.run_dir}",flush=True)


if __name__=="__main__":
    main()
