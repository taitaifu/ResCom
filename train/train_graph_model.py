from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import json
import math
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from tqdm.auto import tqdm
import numpy as np
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] # 根目录
sys.path.append(str(ROOT))

from models.data_utils import (  # noqa: E402
    ROCKER_NAMES,
    WHEEL_IDS,
    get_group_dims,
    graph_temporal_collate_fn,
    prepare_datasets_and_scaler,
    save_column_spec_json,
)
from models.graph_temporal_compensation import (  # noqa: E402
    GraphTemporalCompensationModel,
    apply_rotvec_to_quat,
    huber_or_mse,
    quat_geodesic_loss,
    finite_diff_consistency_loss,
    smoothness_loss
)

from models.graph_temporal_hgt_compensation import (  # noqa: E402
    GraphTemporalHGTCompensationModel,
    TeacherGateNet,
    apply_rotvec_to_quat,
    huber_or_mse,
    quat_geodesic_loss,
    finite_diff_consistency_loss,
    smoothness_loss
)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_huber_or_mse(pred: torch.Tensor, target: torch.Tensor, use_huber: bool = True) -> torch.Tensor:
    if pred.numel() == 0 or target.numel() == 0:
        return pred.new_tensor(0.0)
    return huber_or_mse(pred, target, use_huber=use_huber)


def get_quat_indices(cols: List[str]) -> List[int]:
    idx: List[int] = []
    for name in ["q0", "q1", "q2", "q3"]:
        for i, c in enumerate(cols):
            if c.endswith(name):
                idx.append(i)
                break
    return idx


def get_att_indices(cols: List[str]) -> List[int]:
    idx: List[int] = []
    for name in ["att_x", "att_y", "att_z"]:
        for i, c in enumerate(cols):
            if c.endswith(name):
                idx.append(i)
                break
    return idx


def get_contact_indices(cols: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for key in ["Fz", "sinkage", "in_contact"]:
        for i, c in enumerate(cols):
            if c.endswith(key):
                out[key] = i
                break
    return out


def masked_huber_or_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
    use_huber: bool = True,
) -> torch.Tensor:
    if pred.numel() == 0 or target.numel() == 0:
        return pred.new_tensor(0.0)
    if mask is None:
        return safe_huber_or_mse(pred, target, use_huber=use_huber)
    if mask.ndim == pred.ndim - 1:
        mask = mask.unsqueeze(-1)
    mask = mask.to(device=pred.device, dtype=pred.dtype)
    if mask.shape != pred.shape:
        mask = mask.expand_as(pred)
    active = mask > 0.5
    if not torch.any(active):
        return pred.new_tensor(0.0)
    pred_sel = pred[active]
    target_sel = target[active]
    return safe_huber_or_mse(pred_sel, target_sel, use_huber=use_huber)


def masked_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if pred.numel() == 0 or target.numel() == 0:
        return pred.new_tensor(0.0)
    if mask is None:
        return torch.sqrt(((pred - target) ** 2).mean())
    if mask.ndim == pred.ndim - 1:
        mask = mask.unsqueeze(-1)
    mask = mask.to(device=pred.device, dtype=pred.dtype)
    if mask.shape != pred.shape:
        mask = mask.expand_as(pred)
    active = mask > 0.5
    if not torch.any(active):
        return pred.new_tensor(0.0)
    sqerr = (pred - target) ** 2
    return torch.sqrt(sqerr[active].mean())


def masked_weighted_axis_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    axis_weights: List[float],
    mask: Optional[torch.Tensor] = None,
    use_huber: bool = True,
) -> torch.Tensor:
    if pred.numel() == 0 or target.numel() == 0:
        return pred.new_tensor(0.0)
    if pred.shape[-1] != len(axis_weights):
        raise ValueError(f"axis_weights 长度 {len(axis_weights)} 与最后一维 {pred.shape[-1]} 不一致")
    weights = pred.new_tensor(axis_weights, dtype=pred.dtype)
    weights = weights / weights.sum().clamp_min(1e-12)
    loss = pred.new_tensor(0.0)
    for axis_idx, axis_weight in enumerate(weights):
        loss = loss + axis_weight * masked_huber_or_mse(
            pred[..., axis_idx:axis_idx + 1],
            target[..., axis_idx:axis_idx + 1],
            mask,
            use_huber=use_huber,
        )
    return loss


def build_contact_active_mask(batch: Dict[str, torch.Tensor], spec, wheel_id: int, pred_like: torch.Tensor) -> Optional[torch.Tensor]:
    contact_seq = batch.get(f"wheel{wheel_id}_contact")
    if contact_seq is None or contact_seq.numel() == 0:
        return None
    input_cols = spec.input_groups.wheel_contact_cols[wheel_id]
    idx = get_contact_indices(input_cols)
    in_contact_idx = idx.get("in_contact")
    if in_contact_idx is None or in_contact_idx >= contact_seq.shape[-1]:
        return None
    mask = contact_seq[:, -1, in_contact_idx]
    if pred_like.ndim == 3:
        mask = mask.unsqueeze(1).expand(-1, pred_like.shape[1])
    return mask


def build_model_group_columns(spec) -> Dict[str, List[str]]:
    # 训练脚本把输入组列名整理给模型，供 edge_gate 做逐列物理量匹配。
    group_columns: Dict[str, List[str]] = {
        "system": list(spec.input_groups.system_cols),
        "body": list(spec.input_groups.body_cols),
    }
    for name in ROCKER_NAMES:
        group_columns[name] = list(spec.input_groups.rocker_cols[name])
    for i in WHEEL_IDS:
        group_columns[f"wheel{i}_kin"] = list(spec.input_groups.wheel_kin_cols[i])
        group_columns[f"wheel{i}_contact"] = list(spec.input_groups.wheel_contact_cols[i])
    return group_columns


def get_suffix_indices(cols: List[str], suffixes: List[str]) -> List[int]:
    indices: List[int] = []
    for suffix in suffixes:
        for idx, col in enumerate(cols):
            if col.endswith(suffix):
                indices.append(idx)
                break
    return indices


def _frame_mean_abs(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_zeros((x.shape[0], x.shape[1]))
    return x.abs().mean(dim=-1)


def _frame_feature_std(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_zeros((x.shape[0], x.shape[1]))
    if x.shape[-1] <= 1:
        return x.new_zeros((x.shape[0], x.shape[1]))
    return x.std(dim=-1)


def _frame_temporal_diff(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_zeros((x.shape[0], x.shape[1]))
    diff = x.new_zeros((x.shape[0], x.shape[1]))
    if x.shape[1] > 1:
        diff[:, 1:] = (x[:, 1:] - x[:, :-1]).abs().mean(dim=-1)
    return diff


def _sequence_frame_summary(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_zeros((x.shape[0], x.shape[1], 3))
    return torch.stack(
        [
            _frame_mean_abs(x),
            _frame_feature_std(x),
            _frame_temporal_diff(x),
        ],
        dim=-1,
    )


def _control_frame_summary(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_zeros((x.shape[0], x.shape[1], 4))
    max_abs = x.abs().amax(dim=-1) if x.shape[-1] > 0 else x.new_zeros((x.shape[0], x.shape[1]))
    return torch.stack(
        [
            _frame_mean_abs(x),
            _frame_feature_std(x),
            _frame_temporal_diff(x),
            max_abs,
        ],
        dim=-1,
    )


def _expand_target_summary(x: torch.Tensor, seq_len: int) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_zeros((x.shape[0], seq_len, 3))
    if x.ndim == 2:
        x = x.unsqueeze(1)
    mean_abs = x.abs().mean(dim=tuple(range(1, x.ndim)))
    feat_std = x.std(dim=-1).mean(dim=tuple(range(1, x.ndim - 1))) if x.shape[-1] > 1 else x.new_zeros((x.shape[0],))
    max_abs = x.abs().amax(dim=tuple(range(1, x.ndim)))
    summary = torch.stack([mean_abs, feat_std, max_abs], dim=-1)
    return summary.unsqueeze(1).expand(-1, seq_len, -1)


def _strength_to_gate(strength: torch.Tensor, scale: float = 1.5) -> torch.Tensor:
    return (1.0 - torch.exp(-scale * strength)).clamp(1e-4, 1.0)


def _reduce_feature_strength(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        # 空组统一返回 0 强度，避免某组缺失时破坏 gate_target 构造。
        return x.new_zeros((x.shape[0],))
    # 对除 batch 外的所有维度求平均绝对值，得到一个 batch 级强度标量。
    dims = tuple(range(1, x.ndim))
    return x.abs().mean(dim=dims)


def build_teacher_gate_summary(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    # 用逐时刻输入摘要 + 全局任务摘要构造 [B, T, 31] 的 teacher summary。
    wheel_kin_lf = torch.cat([batch[f"wheel{i}_kin"] for i in WHEEL_IDS], dim=-1)
    wheel_contact_lf = torch.cat([batch[f"wheel{i}_contact"] for i in WHEEL_IDS], dim=-1)
    hf_wheel_kin = torch.cat([batch[f"hf_wheel{i}_kin"] for i in WHEEL_IDS], dim=-1)
    hf_wheel_contact = torch.cat([batch[f"hf_wheel{i}_contact"] for i in WHEEL_IDS], dim=-1)
    res_wheel_kin = torch.cat([batch[f"res_wheel{i}_kin"] for i in WHEEL_IDS], dim=-1)
    res_wheel_contact = torch.cat([batch[f"res_wheel{i}_contact"] for i in WHEEL_IDS], dim=-1)
    seq_len = batch["body"].shape[1]
    parts = [
        _control_frame_summary(batch["system"]),
        _sequence_frame_summary(batch["body"]),
        _sequence_frame_summary(wheel_kin_lf),
        _sequence_frame_summary(wheel_contact_lf),
        _expand_target_summary(batch["hf_body"], seq_len),
        _expand_target_summary(hf_wheel_kin, seq_len),
        _expand_target_summary(hf_wheel_contact, seq_len),
        _expand_target_summary(batch["res_body"], seq_len),
        _expand_target_summary(res_wheel_kin, seq_len),
        _expand_target_summary(res_wheel_contact, seq_len),
    ]
    return torch.cat(parts, dim=-1)


def build_gate_target(
    batch: Dict[str, torch.Tensor],
    relation_names: List[str],
) -> torch.Tensor:
    device = batch["res_body"].device
    bsz, seq_len = batch["body"].shape[:2]
    target = torch.zeros(bsz, seq_len, len(relation_names), device=device, dtype=batch["res_body"].dtype)

    body_seq_strength = _frame_mean_abs(batch["body"])
    wheel_kin_seq_strengths = torch.stack(
        [_frame_mean_abs(batch[f"wheel{i}_kin"]) for i in WHEEL_IDS],
        dim=-1,
    )
    wheel_contact_seq_strengths = torch.stack(
        [_frame_mean_abs(batch[f"wheel{i}_contact"]) for i in WHEEL_IDS],
        dim=-1,
    )
    wheel_kin_seq_strength = wheel_kin_seq_strengths.mean(dim=-1)
    wheel_contact_seq_strength = wheel_contact_seq_strengths.mean(dim=-1)
    control_strength = _frame_temporal_diff(batch["system"])

    body_task_strength = _reduce_feature_strength(batch["res_body"]).unsqueeze(1)
    wheel_kin_task_strength = torch.stack(
        [_reduce_feature_strength(batch[f"res_wheel{i}_kin"]) for i in WHEEL_IDS],
        dim=-1,
    )
    wheel_contact_task_strength = torch.stack(
        [_reduce_feature_strength(batch[f"res_wheel{i}_contact"]) for i in WHEEL_IDS],
        dim=-1,
    )
    wheel_kin_task_mean = wheel_kin_task_strength.mean(dim=-1, keepdim=True)
    wheel_contact_task_mean = wheel_contact_task_strength.mean(dim=-1, keepdim=True)

    longitudinal_pairs = [(0, 2), (2, 4), (1, 3), (3, 5)]
    lateral_pairs = [(0, 1), (2, 3), (4, 5)]
    longitudinal_input_strength = torch.stack(
        [
            0.5 * (wheel_kin_seq_strengths[:, :, a] + wheel_kin_seq_strengths[:, :, b]) +
            0.5 * (wheel_contact_seq_strengths[:, :, a] + wheel_contact_seq_strengths[:, :, b])
            for a, b in longitudinal_pairs
        ],
        dim=-1
    ).mean(dim=-1)
    longitudinal_task_strength = torch.stack(
        [
            0.5 * (wheel_kin_task_strength[:, a] + wheel_kin_task_strength[:, b]) +
            0.5 * (wheel_contact_task_strength[:, a] + wheel_contact_task_strength[:, b])
            for a, b in longitudinal_pairs
        ],
        dim=-1
    ).mean(dim=-1, keepdim=True)
    lateral_input_strength = torch.stack(
        [
            (wheel_kin_seq_strengths[:, :, a] - wheel_kin_seq_strengths[:, :, b]).abs() +
            (wheel_contact_seq_strengths[:, :, a] - wheel_contact_seq_strengths[:, :, b]).abs()
            for a, b in lateral_pairs
        ],
        dim=-1
    ).mean(dim=-1)
    lateral_task_strength = torch.stack(
        [
            (wheel_kin_task_strength[:, a] - wheel_kin_task_strength[:, b]).abs() +
            (wheel_contact_task_strength[:, a] - wheel_contact_task_strength[:, b]).abs()
            for a, b in lateral_pairs
        ],
        dim=-1
    ).mean(dim=-1, keepdim=True)

    relation_strengths = {
        "self_loop": torch.ones_like(body_seq_strength),
        "control_excitation": _strength_to_gate(control_strength * (1.0 + 0.5 * wheel_kin_task_mean)),
        "kinematic_transfer": _strength_to_gate(
            0.6 * body_seq_strength * (1.0 + body_task_strength)
            + 0.4 * wheel_kin_seq_strength * (1.0 + wheel_kin_task_mean)
        ),
        "motion_to_contact": _strength_to_gate(wheel_contact_seq_strength * (1.0 + wheel_contact_task_mean)),
        "contact_to_motion": _strength_to_gate(
            0.5 * wheel_kin_seq_strength * (1.0 + wheel_kin_task_mean)
            + 0.5 * wheel_contact_seq_strength * (1.0 + wheel_contact_task_mean)
        ),
        "longitudinal_coupling": _strength_to_gate(
            longitudinal_input_strength * (1.0 + longitudinal_task_strength)
        ),
        "lateral_coupling": _strength_to_gate(
            lateral_input_strength * (1.0 + lateral_task_strength)
        ),
        "state_to_target": _strength_to_gate(
            0.5 * body_seq_strength * (1.0 + body_task_strength)
            + 0.25 * wheel_kin_seq_strength * (1.0 + wheel_kin_task_mean)
            + 0.25 * wheel_contact_seq_strength * (1.0 + wheel_contact_task_mean)
        ),
    }

    for rel_idx, rel_name in enumerate(relation_names):
        rel_target = relation_strengths.get(rel_name, torch.zeros_like(body_seq_strength))
        target[:, :, rel_idx] = rel_target.clamp(1e-4, 1.0)

    return target


def build_teacher_relation_gate(
    teacher_gate_net: TeacherGateNet,
    batch: Dict[str, torch.Tensor],
    seq_len: int,
) -> torch.Tensor:
    summary = build_teacher_gate_summary(batch)
    teacher_gate = teacher_gate_net(summary)
    if teacher_gate.ndim == 2:
        return teacher_gate.unsqueeze(1).expand(-1, seq_len, -1)
    return teacher_gate


def forward_with_stage_gate(
    model,
    batch: Dict[str, torch.Tensor],
    part_cfg: "TrainingPartConfig",
    teacher_gate_net: Optional[TeacherGateNet],
    teacher_requires_grad: bool,
) -> Dict[str, torch.Tensor]:
    if part_cfg.gate_mode == "teacher":
        if teacher_gate_net is None:
            raise ValueError("teacher_full 阶段要求 teacher_gate_net 存在")
        teacher_latent = build_teacher_gate_summary(batch)
        teacher_gate = build_teacher_relation_gate(teacher_gate_net, batch, batch["body"].shape[1])
        output = model(batch, relation_gate_override=teacher_gate)
        output.pop("student_relation_gate", None)
        output.pop("student_gate_summary", None)
        output.pop("student_inferred_relation_gate", None)
        output.pop("student_inferred_relation_gate_stats", None)
        output["teacher_latent"] = teacher_latent
        output["teacher_relation_gate"] = teacher_gate
        return output

    output = model(batch)
    if part_cfg.use_teacher_supervision:
        if teacher_gate_net is None:
            raise ValueError("student_full 阶段要求 teacher_gate_net 存在")
        teacher_ctx = nullcontext() if teacher_requires_grad else torch.no_grad()
        with teacher_ctx:
            teacher_latent = build_teacher_gate_summary(batch)
            teacher_gate = build_teacher_relation_gate(teacher_gate_net, batch, batch["body"].shape[1])
        output["teacher_latent"] = teacher_latent.detach()
        output["teacher_relation_gate"] = teacher_gate.detach()
    return output


def apply_stage_losses(
    losses: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    output: Dict[str, torch.Tensor],
    part_cfg: "TrainingPartConfig",
    lambda_latent: float,
    lambda_gate: float,
) -> Dict[str, torch.Tensor]:
    zero = losses["total"].new_tensor(0.0)
    losses["teacher_full_task_loss"] = zero
    losses["student_full_task_loss"] = zero
    losses["student_full_distill_loss"] = zero
    losses["student_latent_loss"] = zero
    losses["student_gate_loss"] = zero
    losses["teacher_gate_loss"] = zero

    task_loss = losses["total"]
    if part_cfg.key == "teacher_full":
        losses["teacher_full_task_loss"] = task_loss
        losses["total"] = task_loss
        return losses

    if part_cfg.key != "student_full":
        losses["total"] = task_loss
        return losses

    student_latent = output.get("student_gate_summary")
    student_gate = output.get("student_relation_gate")
    teacher_latent = output.get("teacher_latent")
    teacher_gate = output.get("teacher_relation_gate")
    if student_latent is None or student_gate is None:
        raise KeyError("student_full 阶段要求模型输出 student_gate_summary 和 student_relation_gate")
    if teacher_latent is None or teacher_gate is None:
        raise KeyError("student_full 阶段要求 teacher_latent 和 teacher_relation_gate 监督")

    latent_loss = F.mse_loss(student_latent, teacher_latent.detach())
    gate_loss = F.mse_loss(student_gate, teacher_gate.detach())
    distill_loss = lambda_latent * latent_loss + lambda_gate * gate_loss
    losses["student_full_task_loss"] = task_loss
    losses["student_full_distill_loss"] = distill_loss
    losses["student_latent_loss"] = latent_loss
    losses["student_gate_loss"] = gate_loss
    if part_cfg.optimize_task_loss:
        losses["total"] = task_loss + distill_loss
    else:
        losses["total"] = distill_loss
    return losses


# 10 组摘要：1 个 control 用 4 维，其余 9 组各 3 维，共 31 维。
TEACHER_GATE_SUMMARY_DIM = 31
BASE_METRIC_KEYS = (
    "total",
    "selection_score",
    "raw_task_monitor",
    "monitor_body_pos_raw",
    "monitor_body_vel_raw",
    "monitor_contact_force_raw",
    "monitor_contact_moment_raw",
    "res",
    "hf",
    "quat",
    "contact",
    "contact_physics",
    "kin",
    "smooth",
    "body_pos_xyz_loss_scaled",
    "body_pos_xyz_loss_raw",
    "body_pos_fused_xyz_loss_scaled",
    "body_pos_fused_xyz_loss_raw",
    "body_pos_fused_xyz_rmse_raw",
    "body_vel_xyz_loss_scaled",
    "body_vel_xyz_loss_raw",
    "wheel_contact_loss_scaled",
    "wheel_contact_loss_raw",
    "wheel_contact_rmse_raw",
    "wheel_contact_force_loss_scaled",
    "wheel_contact_force_loss_raw",
    "wheel_contact_force_rmse_raw",
    "wheel_contact_moment_loss_scaled",
    "wheel_contact_moment_loss_raw",
    "wheel_contact_moment_rmse_raw",
    "body_future_kin",
    "body_pos_x_rmse_raw",
    "body_pos_y_rmse_raw",
    "body_pos_z_rmse_raw",
    "body_pos_xyz_rmse_raw",
    "wheel_pos_z_loss_scaled",
    "wheel_pos_z_loss_raw",
    "wheel_pos_z_rmse_raw",
    "body_vel_xyz_rmse_raw",
    "res_body_posxyz",
    "weighted_body_pos",
    "weighted_body_vel",
    "weighted_wheel_pos_z",
    "weighted_contact",
    "weighted_contact_scaled",
    "weighted_contact_raw",
    "weighted_contact_force",
    "weighted_contact_moment",
    "weighted_kin",
    "weighted_smooth",
    "teacher_full_task_loss",
    "student_full_task_loss",
    "student_full_distill_loss",
    "student_latent_loss",
    "student_gate_loss",
    "teacher_gate_loss",
)

TRAIN_STAGE_ALIASES = {
    "teacher": "teacher_full",
    "student": "student_full",
}


@dataclass(frozen=True)
class TrainingPartConfig:
    key: str
    display_name: str
    enable_relation_gate: bool
    enable_edge_gate: bool
    gate_mode: str
    use_teacher_supervision: bool
    optimize_task_loss: bool
    freeze_model_modules: Tuple[str, ...]
    train_model_modules: Tuple[str, ...]
    freeze_teacher_gate: bool
    max_lr: float
    warmup_epochs: int
    optimizer_group_lrs: Tuple[Tuple[str, float], ...] = ()
    lr_floor_ratio: float = 0.1
    second_phase_max_lr: Optional[float] = None
    second_phase_warmup_epochs: int = 0


PREVIOUS_TRAIN_STAGE = {
    "student_full": "teacher_full",
}


def canonicalize_train_stage(train_stage: str) -> str:
    return TRAIN_STAGE_ALIASES.get(train_stage, train_stage)


def build_training_part_config(args: argparse.Namespace) -> TrainingPartConfig:
    train_stage = canonicalize_train_stage(args.train_stage)
    train_main_in_stage2 = bool(getattr(args, "train_main_in_stage2", False))

    if train_stage == "teacher_full":
        return TrainingPartConfig(
            key=train_stage,
            display_name="teacher_full",
            enable_relation_gate=True,
            enable_edge_gate=True,
            gate_mode="teacher",
            use_teacher_supervision=False,
            optimize_task_loss=True,
            freeze_model_modules=(),
            train_model_modules=(
                "input_encoders",
                "virtual_node_embeds",
                "hgt_layers",
                "edge_gate_feature_mlps",
                "edge_gate_hidden_mlps",
                "tcn",
                "bilstm",
                "horizon_embed",
                "body_head",
                "wheel_head",
                "contact_head",
            ),
            freeze_teacher_gate=False,
            max_lr=3e-4,
            warmup_epochs=5,
        )
    if train_stage == "student_full":
        if train_main_in_stage2:
            freeze_model_modules: Tuple[str, ...] = ()
            train_model_modules: Tuple[str, ...] = (
                "input_encoders",
                "virtual_node_embeds",
                "hgt_layers",
                "student_summary_predictor",
                "student_summary_gate_net",
                "edge_gate_feature_mlps",
                "edge_gate_hidden_mlps",
                "tcn",
                "bilstm",
                "horizon_embed",
                "body_head",
                "wheel_head",
                "contact_head",
            )
            max_lr = 2e-5
            warmup_epochs = 2
        else:
            freeze_model_modules = (
                "input_encoders",
                "virtual_node_embeds",
                "hgt_layers",
                "tcn",
                "bilstm",
                "horizon_embed",
                "body_head",
                "wheel_head",
                "contact_head",
            )
            train_model_modules = (
                "student_summary_predictor",
                "student_summary_gate_net",
                "edge_gate_feature_mlps",
                "edge_gate_hidden_mlps",
            )
            max_lr = 5e-5
            warmup_epochs = 2
        return TrainingPartConfig(
            key=train_stage,
            display_name="student_full",
            enable_relation_gate=True,
            enable_edge_gate=True,
            gate_mode="student",
            use_teacher_supervision=True,
            optimize_task_loss=train_main_in_stage2,
            freeze_model_modules=freeze_model_modules,
            train_model_modules=train_model_modules,
            freeze_teacher_gate=True,
            max_lr=max_lr,
            warmup_epochs=warmup_epochs,
        )
    raise ValueError(f"不支持的 train_stage: {train_stage}")


def get_module_param_names(module: torch.nn.Module, module_name: str) -> List[str]:
    try:
        submodule = getattr(module, module_name)
    except AttributeError:
        return []
    if isinstance(submodule, torch.nn.Parameter):
        return [module_name]
    return [f"{module_name}.{name}" for name, _ in submodule.named_parameters()]


def freeze_named_modules(module: torch.nn.Module, module_names: Tuple[str, ...]) -> List[str]:
    frozen_param_names: List[str] = []
    for module_name in module_names:
        for param_name in get_module_param_names(module, module_name):
            param = dict(module.named_parameters())[param_name]
            param.requires_grad = False
            frozen_param_names.append(param_name)
    return frozen_param_names


def set_trainable_subset(module: torch.nn.Module, module_names: Tuple[str, ...]) -> List[str]:
    named_params = dict(module.named_parameters())
    trainable_names: List[str] = []
    for param in named_params.values():
        param.requires_grad = False
    for module_name in module_names:
        for param_name in get_module_param_names(module, module_name):
            named_params[param_name].requires_grad = True
            trainable_names.append(param_name)
    return trainable_names


def apply_train_part_freeze_policy(
    model: torch.nn.Module,
    teacher_gate_net: Optional[torch.nn.Module],
    part_cfg: TrainingPartConfig,
) -> Dict[str, List[str]]:
    set_trainable_subset(model, part_cfg.train_model_modules)
    extra_frozen = freeze_named_modules(model, part_cfg.freeze_model_modules)
    frozen_model_names = sorted(
        {name for name, param in model.named_parameters() if not param.requires_grad}.union(extra_frozen)
    )
    trainable_model_names = sorted(name for name, param in model.named_parameters() if param.requires_grad)

    trainable_teacher_names: List[str] = []
    frozen_teacher_names: List[str] = []
    if teacher_gate_net is not None:
        for name, param in teacher_gate_net.named_parameters():
            param.requires_grad = not part_cfg.freeze_teacher_gate
            if param.requires_grad:
                trainable_teacher_names.append(f"teacher_gate_net.{name}")
            else:
                frozen_teacher_names.append(f"teacher_gate_net.{name}")

    return {
        "trainable_model_params": trainable_model_names,
        "frozen_model_params": frozen_model_names,
        "trainable_teacher_params": trainable_teacher_names,
        "frozen_teacher_params": frozen_teacher_names,
    }


def set_frozen_modules_eval(model: torch.nn.Module, part_cfg: TrainingPartConfig) -> None:
    base_model = unwrap_model(model)
    for module_name in part_cfg.freeze_model_modules:
        submodule = getattr(base_model, module_name, None)
        if isinstance(submodule, torch.nn.Module):
            submodule.eval()
            # cuDNN RNN 在参与反向传播时必须保持 training mode，即使参数被冻结。
            for child in submodule.modules():
                if isinstance(child, torch.nn.RNNBase):
                    child.train()


def load_init_checkpoint(
    model: torch.nn.Module,
    teacher_gate_net: Optional[torch.nn.Module],
    ckpt_path: Optional[str],
    device: torch.device,
) -> Optional[Dict]:
    if not ckpt_path:
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    load_result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if load_result.missing_keys:
        preview = ", ".join(load_result.missing_keys[:8])
        suffix = " ..." if len(load_result.missing_keys) > 8 else ""
        print(
            f"初始化 checkpoint 未覆盖 {len(load_result.missing_keys)} 个模型参数，"
            f"例如: {preview}{suffix}"
        )
    if load_result.unexpected_keys:
        preview = ", ".join(load_result.unexpected_keys[:8])
        suffix = " ..." if len(load_result.unexpected_keys) > 8 else ""
        print(
            f"初始化 checkpoint 包含 {len(load_result.unexpected_keys)} 个当前模型未使用参数，"
            f"例如: {preview}{suffix}"
        )
    teacher_state = ckpt.get("teacher_gate_state_dict")
    if teacher_gate_net is not None and teacher_state is not None:
        teacher_gate_net.load_state_dict(teacher_state, strict=False)
    return ckpt


def resolve_init_checkpoint(train_stage: str, init_ckpt: Optional[str], base_save_dir: str) -> Optional[str]:
    if init_ckpt:
        return init_ckpt
    prev_stage = PREVIOUS_TRAIN_STAGE.get(train_stage)
    if prev_stage is None:
        return None

    prev_dir = Path(base_save_dir) / prev_stage
    ckpt_candidates = [
        path for path in prev_dir.glob("*/best_model.pt")
        if path.is_file()
    ]
    if not ckpt_candidates:
        raise FileNotFoundError(
            f"{train_stage} 未指定 --init_ckpt，且未在 {prev_dir} 下找到可用的 best_model.pt"
        )

    latest_ckpt = max(
        ckpt_candidates,
        key=lambda path: (path.parent.name, path.stat().st_mtime),
    )
    return str(latest_ckpt)


def count_parameters(parameters: List[torch.nn.Parameter]) -> int:
    return int(sum(p.numel() for p in parameters))


def collect_trainable_parameters(
    model: torch.nn.Module,
    teacher_gate_net: Optional[torch.nn.Module],
) -> List[torch.nn.Parameter]:
    params = [p for p in model.parameters() if p.requires_grad]
    if teacher_gate_net is not None:
        params.extend(p for p in teacher_gate_net.parameters() if p.requires_grad)
    return params


def build_optimizer_param_groups(
    model: torch.nn.Module,
    teacher_gate_net: Optional[torch.nn.Module],
    part_cfg: TrainingPartConfig,
    weight_decay: float,
) -> List[Dict]:
    named_params = dict(model.named_parameters())
    seen_names: Set[str] = set()
    param_groups: List[Dict] = []

    for module_name, group_lr in part_cfg.optimizer_group_lrs:
        params: List[torch.nn.Parameter] = []
        for param_name in get_module_param_names(model, module_name):
            param = named_params.get(param_name)
            if param is None or not param.requires_grad or param_name in seen_names:
                continue
            params.append(param)
            seen_names.add(param_name)
        if params:
            param_groups.append(
                {
                    "params": params,
                    "lr": group_lr,
                    "max_lr": group_lr,
                    "group_name": module_name,
                    "weight_decay": weight_decay,
                }
            )

    default_params = [
        param
        for name, param in named_params.items()
        if param.requires_grad and name not in seen_names
    ]
    if default_params:
        param_groups.append(
            {
                "params": default_params,
                "lr": part_cfg.max_lr,
                "max_lr": part_cfg.max_lr,
                "group_name": "default",
                "weight_decay": weight_decay,
            }
        )

    if teacher_gate_net is not None:
        teacher_params = [p for p in teacher_gate_net.parameters() if p.requires_grad]
        if teacher_params:
            param_groups.append(
                {
                    "params": teacher_params,
                    "lr": part_cfg.max_lr,
                    "max_lr": part_cfg.max_lr,
                    "group_name": "teacher_gate_net",
                    "weight_decay": weight_decay,
                }
            )

    return param_groups


def _warmup_cosine_lr(
    epoch_idx: int,
    total_epochs: int,
    warmup_epochs: int,
    max_lr: float,
    min_lr_ratio: float,
) -> float:
    if total_epochs <= 0:
        return max_lr

    warmup_epochs = max(0, min(warmup_epochs, total_epochs))
    min_lr = max_lr * min_lr_ratio

    if warmup_epochs > 0 and epoch_idx <= warmup_epochs:
        return max_lr * float(epoch_idx) / float(warmup_epochs)

    if total_epochs == warmup_epochs:
        return max_lr

    progress = float(epoch_idx - warmup_epochs) / float(max(total_epochs - warmup_epochs, 1))
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine


def get_scheduled_group_lr(
    part_cfg: TrainingPartConfig,
    group_max_lr: float,
    epoch_idx: int,
    total_epochs: int,
) -> float:
    if part_cfg.second_phase_max_lr is None or total_epochs <= 1:
        return _warmup_cosine_lr(
            epoch_idx=epoch_idx,
            total_epochs=total_epochs,
            warmup_epochs=part_cfg.warmup_epochs,
            max_lr=group_max_lr,
            min_lr_ratio=part_cfg.lr_floor_ratio,
        )

    first_phase_epochs = max(total_epochs // 2, 1)
    second_phase_epochs = max(total_epochs - first_phase_epochs, 1)
    lr_scale = group_max_lr / max(part_cfg.max_lr, 1e-12)
    second_phase_group_max_lr = part_cfg.second_phase_max_lr * lr_scale

    if epoch_idx <= first_phase_epochs:
        return _warmup_cosine_lr(
            epoch_idx=epoch_idx,
            total_epochs=first_phase_epochs,
            warmup_epochs=part_cfg.warmup_epochs,
            max_lr=group_max_lr,
            min_lr_ratio=part_cfg.lr_floor_ratio,
        )

    return _warmup_cosine_lr(
        epoch_idx=epoch_idx - first_phase_epochs,
        total_epochs=second_phase_epochs,
        warmup_epochs=part_cfg.second_phase_warmup_epochs,
        max_lr=second_phase_group_max_lr,
        min_lr_ratio=part_cfg.lr_floor_ratio,
    )


def apply_active_lr_schedule(
    optimizer: torch.optim.Optimizer,
    part_cfg: TrainingPartConfig,
    epoch_idx: int,
    total_epochs: int,
) -> Dict[str, float]:
    group_lr_map: Dict[str, float] = {}
    for group_idx, param_group in enumerate(optimizer.param_groups):
        group_name = str(param_group.get("group_name", f"group_{group_idx}"))
        group_max_lr = float(param_group.get("max_lr", param_group["lr"]))
        target_lr = get_scheduled_group_lr(
            part_cfg=part_cfg,
            group_max_lr=group_max_lr,
            epoch_idx=epoch_idx,
            total_epochs=total_epochs,
        )
        param_group["lr"] = target_lr
        group_lr_map[group_name] = target_lr
    return group_lr_map


@torch.no_grad()
def run_dummy_forward_check(
    model: torch.nn.Module,
    group_dims: Dict[str, int],
    seq_len: int,
    pred_seq_len: int,
    device: torch.device,
    batch_size: int = 2,
) -> None:
    batch: Dict[str, torch.Tensor] = {}

    # 摇臂节点只作为中间输入节点，不构造任何摇臂预测目标。
    input_group_names = [
        "system",
        "body",
        *ROCKER_NAMES,
        *[f"wheel{i}_kin" for i in WHEEL_IDS],
        *[f"wheel{i}_contact" for i in WHEEL_IDS],
    ]
    for group_name in input_group_names:
        dim = int(group_dims.get(group_name, 0))
        batch[group_name] = torch.zeros(batch_size, seq_len, dim, dtype=torch.float32, device=device)

    was_training = model.training
    model.eval()
    output = model(batch)
    if was_training:
        model.train()

    expected_shapes = {
        "pred_res_body": (batch_size, pred_seq_len, int(group_dims["res_body"])),
        "pred_res_wheel0_kin": (batch_size, pred_seq_len, int(group_dims["res_wheel0_kin"])),
        "pred_res_wheel0_contact": (batch_size, pred_seq_len, int(group_dims["res_wheel0_contact"])),
    }
    print("Dummy forward check:")
    for key, expected_shape in expected_shapes.items():
        actual_shape = tuple(output[key].shape)
        print(f"{key} shape = {actual_shape}")
        if actual_shape != expected_shape:
            raise RuntimeError(f"{key} shape mismatch: expected {expected_shape}, got {actual_shape}")


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    non_blocking = device.type == "cuda"

    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=non_blocking)
        else:
            out[k] = v

    return out

def assert_finite_tensor(name: str, x: torch.Tensor) -> None:
    """
    检查单个 Tensor 是否包含 NaN 或 Inf。
    只用于调试，不改变原始数据。
    """
    if not isinstance(x, torch.Tensor):
        return

    if not x.dtype.is_floating_point:
        return

    if x.numel() == 0:
        return

    if torch.isfinite(x).all():
        return

    with torch.no_grad():
        nan_count = torch.isnan(x).sum().item()
        inf_count = torch.isinf(x).sum().item()
        finite_x = x[torch.isfinite(x)]

        print("\n================ 数值异常 ================")
        print(f"位置: {name}")
        print(f"shape: {tuple(x.shape)}")
        print(f"NaN 数量: {nan_count}")
        print(f"Inf 数量: {inf_count}")

        if finite_x.numel() > 0:
            print(f"有限值 min: {finite_x.min().item():.6e}")
            print(f"有限值 max: {finite_x.max().item():.6e}")
            print(f"有限值 mean: {finite_x.mean().item():.6e}")
        else:
            print("该 Tensor 中没有任何有限值")

        print("==========================================\n")

    raise RuntimeError(f"{name} contains NaN or Inf")


def check_tensor_dict(prefix: str, data: Dict) -> None:
    """
    检查 batch / output / losses 这类字典中的所有浮点 Tensor。
    """
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            assert_finite_tensor(f"{prefix}[{k}]", v)


def _scaler_tensors(scaler, group_name: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s = scaler.scalers[group_name]
    mean = torch.as_tensor(s.mean_, dtype=torch.float32, device=device)
    std = torch.as_tensor(s.std_, dtype=torch.float32, device=device)
    mask = torch.as_tensor(s.apply_mask_, dtype=torch.bool, device=device)
    return mean, std, mask


def mean_scale_for_suffixes(
    scaler,
    group_name: str,
    cols: List[str],
    suffixes: List[str],
    device: torch.device,
) -> torch.Tensor:
    _, std, mask = _scaler_tensors(scaler, group_name, device)
    idx = get_suffix_indices(cols, suffixes)
    if not idx:
        return std.new_tensor(1.0)
    idx_tensor = torch.as_tensor(idx, dtype=torch.long, device=device)
    selected_std = std[idx_tensor]
    selected_mask = mask[idx_tensor]
    if bool(selected_mask.any()):
        selected_std = selected_std[selected_mask]
    if selected_std.numel() == 0:
        return std.new_tensor(1.0)
    return selected_std.abs().mean().clamp_min(1e-6)


def inverse_transform_tensor(x: torch.Tensor, scaler, group_name: str) -> torch.Tensor:
    mean, std, mask = _scaler_tensors(scaler, group_name, x.device)
    y = x.float().clone()
    if y.shape[-1] > 0 and bool(mask.any()):
        y[..., mask] = y[..., mask] * std[mask] + mean[mask]
    return y

def inverse_transform_sequence(
    seq_scaled: torch.Tensor,
    scaler,
    group_name: str,
) -> torch.Tensor:
    """
    将标准化后的序列反标准化。
    seq_scaled: [B, T, D]
    """
    if seq_scaled.ndim != 3:
        return seq_scaled

    b, t, d = seq_scaled.shape
    flat = seq_scaled.reshape(b * t, d)
    flat_raw = inverse_transform_tensor(flat, scaler, group_name)
    return flat_raw.reshape(b, t, d)

def transform_tensor(x: torch.Tensor, scaler, group_name: str) -> torch.Tensor:
    mean, std, mask = _scaler_tensors(scaler, group_name, x.device)
    y = x.float().clone()
    if y.shape[-1] > 0 and bool(mask.any()):
        y[..., mask] = (y[..., mask] - mean[mask]) / std[mask]
    return y


def reconstruct_hf_scaled(
    pred_res_scaled: torch.Tensor,
    lf_current_raw: torch.Tensor,
    res_cols: List[str],
    hf_cols: List[str],
    res_group_name: str,
    hf_group_name: str,
    scaler,
) -> torch.Tensor:
    """
    输入:
        pred_res_scaled: 模型输出的标准化残差
        lf_current_raw: Dataset 中保存的当前时刻低保真原始状态
    输出:
        pred_hf_scaled: 与 batch[hf_*] 同一标准化空间的预测高保真状态
    """
    if pred_res_scaled.ndim not in (2, 3):
        raise ValueError(
            f"pred_res_scaled 应为 [B, D] 或 [B, H, D]，实际 shape={tuple(pred_res_scaled.shape)}"
        )
    if lf_current_raw.ndim != pred_res_scaled.ndim:
        raise ValueError(
            f"lf_current_raw 与 pred_res_scaled 维度必须一致，"
            f"实际为 {tuple(lf_current_raw.shape)} vs {tuple(pred_res_scaled.shape)}"
        )

    if len(hf_cols) == 0:
        return pred_res_scaled.new_zeros((*pred_res_scaled.shape[:-1], 0))

    pred_res_raw = inverse_transform_tensor(pred_res_scaled, scaler, res_group_name)
    pred_hf_raw = lf_current_raw.clone()

    res_index = {c: j for j, c in enumerate(res_cols)}

    for j, hf_col in enumerate(hf_cols):
        if not hf_col.startswith("hf_"):
            continue
        suffix = hf_col[len("hf_"):]
        res_col = "res_" + suffix
        if res_col in res_index:
            pred_hf_raw[..., j] = lf_current_raw[..., j] + pred_res_raw[..., res_index[res_col]]

    quat_idx = get_quat_indices(hf_cols)
    att_idx = get_att_indices(res_cols)
    if len(quat_idx) == 4 and len(att_idx) == 3:
        lf_quat = lf_current_raw[..., quat_idx]
        pred_rotvec = pred_res_raw[..., att_idx]
        pred_quat = apply_rotvec_to_quat(lf_quat, pred_rotvec, left_multiply=True)
        pred_hf_raw[..., quat_idx] = pred_quat

    return transform_tensor(pred_hf_raw, scaler, hf_group_name)


def get_xyz_indices(cols: List[str], name: str) -> List[int]:
    """
    从列名中提取某一类三轴变量的索引。
    例如 name='pos' 时，寻找 pos_x, pos_y, pos_z。
    """
    idx: List[int] = []

    for axis in ["x", "y", "z"]:
        key = f"{name}_{axis}"
        for i, c in enumerate(cols):
            if c.endswith(key):
                idx.append(i)
                break

    return idx


def kinematic_loss_from_state_sequence(
    pred_hf_seq_scaled: torch.Tensor,
    cols: List[str],
    scaler,
    group_name: str,
    dt: float,
) -> torch.Tensor:
    """
    从高保真状态序列中提取 pos/vel，计算运动学一致性损失。
    pred_hf_seq_scaled: [B, T, D]
    """
    if pred_hf_seq_scaled.ndim != 3:
        return pred_hf_seq_scaled.new_tensor(0.0)

    pos_idx = get_xyz_indices(cols, "pos")
    vel_idx = get_xyz_indices(cols, "vel")

    if len(pos_idx) != 3 or len(vel_idx) != 3:
        return pred_hf_seq_scaled.new_tensor(0.0)

    pred_hf_seq_raw = inverse_transform_sequence(
        pred_hf_seq_scaled,
        scaler,
        group_name,
    )

    pos = pred_hf_seq_raw[:, :, pos_idx]
    vel = pred_hf_seq_raw[:, :, vel_idx]

    if pos.shape[1] < 2:
        return pred_hf_seq_scaled.new_tensor(0.0)

    dt_value = float(dt)
    if abs(dt_value) < 1e-12:
        return pred_hf_seq_scaled.new_tensor(0.0)

    pos_fd = (pos[:, 1:, :] - pos[:, :-1, :]) / dt_value
    vel_mid = vel[:, :-1, :]
    return safe_huber_or_mse(pos_fd, vel_mid)


def build_hf_sequence_with_pred_current(
    hf_hist_scaled: torch.Tensor,
    pred_hf_current_scaled: torch.Tensor,
) -> torch.Tensor:
    """
    用前若干个真实高保真历史点 + 当前预测高保真点组成序列。

    hf_hist_scaled: [B, T_hist, D]
    pred_hf_current_scaled: [B, D] 或 [B, H, D]
    返回: [B, T_hist + 1, D] 或 [B, T_hist + H, D]
    """
    if hf_hist_scaled.ndim != 3:
        return pred_hf_current_scaled.unsqueeze(1) if pred_hf_current_scaled.ndim == 2 else pred_hf_current_scaled

    if pred_hf_current_scaled.ndim == 2:
        pred_hf_current_scaled = pred_hf_current_scaled.unsqueeze(1)
    elif pred_hf_current_scaled.ndim != 3:
        raise ValueError(
            "pred_hf_current_scaled 应为 [B, D] 或 [B, H, D]，"
            f"实际 shape={tuple(pred_hf_current_scaled.shape)}"
        )

    return torch.cat(
        [
            hf_hist_scaled,
            pred_hf_current_scaled,
        ],
        dim=1,
    )

def weighted_state_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    cols: List[str],
    acc_weight: float = 3.0,
    use_huber: bool = True,
) -> torch.Tensor:
    """
    对状态量计算加权损失。
    acc_x、acc_y、acc_z 单独加权，避免加速度目标被位置、速度、姿态等低频目标稀释。
    """
    if pred.numel() == 0 or target.numel() == 0:
        return pred.new_tensor(0.0)

    base_loss = safe_huber_or_mse(pred, target)

    acc_idx = []
    for k, c in enumerate(cols):
        if c.endswith("acc_x") or c.endswith("acc_y") or c.endswith("acc_z"):
            acc_idx.append(k)

    if len(acc_idx) == 0:
        return base_loss

    pred_acc = pred[..., acc_idx]
    target_acc = target[..., acc_idx]
    acc_loss = safe_huber_or_mse(pred_acc, target_acc)

    return base_loss + acc_weight * acc_loss


def compute_body_supervision_metrics(
    pred_hf_body: torch.Tensor,
    true_hf_body: torch.Tensor,
    cols: List[str],
    scaler,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    pred_raw = inverse_transform_tensor(pred_hf_body, scaler, "hf_body")
    true_raw = inverse_transform_tensor(true_hf_body, scaler, "hf_body")
    metrics: Dict[str, torch.Tensor] = {}

    pos_idx = get_xyz_indices(cols, "pos")
    vel_idx = get_xyz_indices(cols, "vel")

    if len(pos_idx) == 3:
        metrics["body_pos_xyz_loss_scaled"] = safe_huber_or_mse(pred_hf_body[..., pos_idx], true_hf_body[..., pos_idx])
        pred_pos = pred_raw[..., pos_idx]
        true_pos = true_raw[..., pos_idx]
        pos_err = pred_pos - true_pos
        metrics["body_pos_xyz_loss_raw"] = safe_huber_or_mse(pred_pos, true_pos)
        metrics["body_pos_x_rmse_raw"] = torch.sqrt((pos_err[..., 0] ** 2).mean())
        metrics["body_pos_y_rmse_raw"] = torch.sqrt((pos_err[..., 1] ** 2).mean())
        metrics["body_pos_z_rmse_raw"] = torch.sqrt((pos_err[..., 2] ** 2).mean())
        metrics["body_pos_xyz_rmse_raw"] = torch.sqrt((pos_err ** 2).mean())
    else:
        zero = pred_hf_body.new_tensor(0.0)
        metrics["body_pos_xyz_loss_scaled"] = zero
        metrics["body_pos_xyz_loss_raw"] = zero
        metrics["body_pos_x_rmse_raw"] = zero
        metrics["body_pos_y_rmse_raw"] = zero
        metrics["body_pos_z_rmse_raw"] = zero
        metrics["body_pos_xyz_rmse_raw"] = zero

    if len(vel_idx) == 3:
        metrics["body_vel_xyz_loss_scaled"] = safe_huber_or_mse(pred_hf_body[..., vel_idx], true_hf_body[..., vel_idx])
        pred_vel = pred_raw[..., vel_idx]
        true_vel = true_raw[..., vel_idx]
        vel_err = pred_vel - true_vel
        metrics["body_vel_xyz_loss_raw"] = safe_huber_or_mse(pred_vel, true_vel)
        metrics["body_vel_xyz_rmse_raw"] = torch.sqrt((vel_err ** 2).mean())
    else:
        zero = pred_hf_body.new_tensor(0.0)
        metrics["body_vel_xyz_loss_scaled"] = zero
        metrics["body_vel_xyz_loss_raw"] = zero
        metrics["body_vel_xyz_rmse_raw"] = zero

    return metrics, pred_raw, true_raw


def compute_body_fused_position_metrics(
    pred_hf_body_scaled: torch.Tensor,
    true_hf_body_scaled: torch.Tensor,
    pred_hf_body_raw: torch.Tensor,
    true_hf_body_raw: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    cols: List[str],
    scaler,
    alpha: float,
    dt: float,
) -> Dict[str, torch.Tensor]:
    metrics: Dict[str, torch.Tensor] = {}
    pos_idx = get_xyz_indices(cols, "pos")
    vel_idx = get_xyz_indices(cols, "vel")
    zero = pred_hf_body_scaled.new_tensor(0.0)

    if len(pos_idx) != 3 or len(vel_idx) != 3:
        metrics["body_pos_fused_xyz_loss_scaled"] = zero
        metrics["body_pos_fused_xyz_loss_raw"] = zero
        metrics["body_pos_fused_xyz_rmse_raw"] = zero
        return metrics

    alpha = float(np.clip(alpha, 0.0, 1.0))
    pred_pos = pred_hf_body_raw[..., pos_idx]
    pred_vel = pred_hf_body_raw[..., vel_idx]
    true_pos = true_hf_body_raw[..., pos_idx]

    if "hf_body_hist" in batch and batch["hf_body_hist"].ndim == 3 and batch["hf_body_hist"].shape[1] > 0:
        hist_raw = inverse_transform_sequence(batch["hf_body_hist"], scaler, "hf_body")
        prev_pos = hist_raw[:, -1, pos_idx]
    else:
        lf_current = batch["lf_body_current"]
        if lf_current.ndim == 2:
            lf_current = lf_current.unsqueeze(1)
        prev_pos = lf_current[:, 0, pos_idx].float()

    dt_value = float(dt)
    if abs(dt_value) < 1e-12:
        metrics["body_pos_fused_xyz_loss_scaled"] = zero
        metrics["body_pos_fused_xyz_loss_raw"] = zero
        metrics["body_pos_fused_xyz_rmse_raw"] = zero
        return metrics

    pos_from_vel = torch.zeros_like(pred_pos)
    running_pos = prev_pos
    for step_idx in range(pred_pos.shape[1]):
        running_pos = running_pos + pred_vel[:, step_idx, :] * dt_value
        pos_from_vel[:, step_idx, :] = running_pos

    fused_pos = alpha * pred_pos + (1.0 - alpha) * pos_from_vel
    fused_body_raw = pred_hf_body_raw.clone()
    fused_body_raw[..., pos_idx] = fused_pos
    fused_body_scaled = transform_tensor(fused_body_raw, scaler, "hf_body")

    metrics["body_pos_fused_xyz_loss_scaled"] = safe_huber_or_mse(
        fused_body_scaled[..., pos_idx],
        true_hf_body_scaled[..., pos_idx],
    )
    metrics["body_pos_fused_xyz_loss_raw"] = safe_huber_or_mse(fused_pos, true_pos)
    metrics["body_pos_fused_xyz_rmse_raw"] = torch.sqrt(((fused_pos - true_pos) ** 2).mean())
    return metrics


def compute_wheel_pos_z_supervision_metrics(
    pred_hf_wheel_kin: Dict[int, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    spec,
    scaler,
) -> Dict[str, torch.Tensor]:
    metrics: Dict[str, torch.Tensor] = {}
    scaled_losses: List[torch.Tensor] = []
    raw_losses: List[torch.Tensor] = []
    raw_rmses: List[torch.Tensor] = []

    ref = next(iter(pred_hf_wheel_kin.values()), None)
    if ref is None:
        zero = batch["res_body"].new_tensor(0.0)
        metrics["wheel_pos_z_loss_scaled"] = zero
        metrics["wheel_pos_z_loss_raw"] = zero
        metrics["wheel_pos_z_rmse_raw"] = zero
        return metrics

    for i in WHEEL_IDS:
        cols = spec.target_groups.wheel_kin_cols[i]
        pos_z_idx = get_suffix_indices(cols, ["pos_z"])
        if not pos_z_idx:
            continue
        pred_scaled = pred_hf_wheel_kin[i][..., pos_z_idx]
        true_scaled = batch[f"hf_wheel{i}_kin"][..., pos_z_idx]
        scaled_losses.append(safe_huber_or_mse(pred_scaled, true_scaled))

        pred_raw = inverse_transform_tensor(pred_hf_wheel_kin[i], scaler, f"hf_wheel{i}_kin")[..., pos_z_idx]
        true_raw = inverse_transform_tensor(batch[f"hf_wheel{i}_kin"], scaler, f"hf_wheel{i}_kin")[..., pos_z_idx]
        raw_losses.append(safe_huber_or_mse(pred_raw, true_raw))
        raw_rmses.append(torch.sqrt(((pred_raw - true_raw) ** 2).mean()))

    metrics["wheel_pos_z_loss_scaled"] = torch.stack(scaled_losses).mean() if scaled_losses else ref.new_tensor(0.0)
    metrics["wheel_pos_z_loss_raw"] = torch.stack(raw_losses).mean() if raw_losses else ref.new_tensor(0.0)
    metrics["wheel_pos_z_rmse_raw"] = torch.stack(raw_rmses).mean() if raw_rmses else ref.new_tensor(0.0)
    return metrics


def compute_wheel_contact_supervision_metrics(
    pred_hf_wheel_contact: Dict[int, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    spec,
    scaler,
) -> Dict[str, torch.Tensor]:
    metrics: Dict[str, torch.Tensor] = {}
    scaled_losses: List[torch.Tensor] = []
    raw_losses: List[torch.Tensor] = []
    raw_sqerrs: List[torch.Tensor] = []
    scaled_force_losses: List[torch.Tensor] = []
    raw_force_losses: List[torch.Tensor] = []
    raw_force_sqerrs: List[torch.Tensor] = []
    scaled_moment_losses: List[torch.Tensor] = []
    raw_moment_losses: List[torch.Tensor] = []
    raw_moment_sqerrs: List[torch.Tensor] = []

    ref = next(iter(pred_hf_wheel_contact.values()), None)
    if ref is None:
        zero = batch["res_body"].new_tensor(0.0)
        metrics["wheel_contact_loss_scaled"] = zero
        metrics["wheel_contact_loss_raw"] = zero
        metrics["wheel_contact_rmse_raw"] = zero
        metrics["wheel_contact_force_loss_scaled"] = zero
        metrics["wheel_contact_force_loss_raw"] = zero
        metrics["wheel_contact_force_rmse_raw"] = zero
        metrics["wheel_contact_moment_loss_scaled"] = zero
        metrics["wheel_contact_moment_loss_raw"] = zero
        metrics["wheel_contact_moment_rmse_raw"] = zero
        return metrics

    for i in WHEEL_IDS:
        pred_scaled = pred_hf_wheel_contact[i]
        true_scaled = batch[f"hf_wheel{i}_contact"]
        cols = spec.target_groups.wheel_contact_cols[i]
        force_idx = get_suffix_indices(cols, ["Fx", "Fy", "Fz"])
        moment_idx = get_suffix_indices(cols, ["Mx", "My", "Mz"])
        contact_mask = build_contact_active_mask(batch, spec, i, pred_scaled)
        scaled_losses.append(masked_huber_or_mse(pred_scaled, true_scaled, contact_mask))

        pred_raw = inverse_transform_tensor(pred_scaled, scaler, f"hf_wheel{i}_contact")
        true_raw = inverse_transform_tensor(true_scaled, scaler, f"hf_wheel{i}_contact")
        raw_losses.append(masked_huber_or_mse(pred_raw, true_raw, contact_mask))
        raw_sqerrs.append(masked_rmse(pred_raw, true_raw, contact_mask) ** 2)

        if force_idx:
            scaled_force_losses.append(masked_weighted_axis_loss(
                pred_scaled[..., force_idx],
                true_scaled[..., force_idx],
                axis_weights=[0.3, 0.2, 0.5],
                mask=contact_mask,
            ))
            raw_force_losses.append(masked_weighted_axis_loss(
                pred_raw[..., force_idx],
                true_raw[..., force_idx],
                axis_weights=[0.3, 0.2, 0.5],
                mask=contact_mask,
            ))
            raw_force_sqerrs.append(masked_rmse(pred_raw[..., force_idx], true_raw[..., force_idx], contact_mask) ** 2)
        if moment_idx:
            scaled_moment_losses.append(masked_huber_or_mse(pred_scaled[..., moment_idx], true_scaled[..., moment_idx], contact_mask))
            raw_moment_losses.append(masked_huber_or_mse(pred_raw[..., moment_idx], true_raw[..., moment_idx], contact_mask))
            raw_moment_sqerrs.append(masked_rmse(pred_raw[..., moment_idx], true_raw[..., moment_idx], contact_mask) ** 2)

    metrics["wheel_contact_loss_scaled"] = torch.stack(scaled_losses).mean()
    metrics["wheel_contact_loss_raw"] = torch.stack(raw_losses).mean()
    metrics["wheel_contact_rmse_raw"] = torch.sqrt(torch.stack(raw_sqerrs).mean())
    metrics["wheel_contact_force_loss_scaled"] = torch.stack(scaled_force_losses).mean() if scaled_force_losses else ref.new_tensor(0.0)
    metrics["wheel_contact_force_loss_raw"] = torch.stack(raw_force_losses).mean() if raw_force_losses else ref.new_tensor(0.0)
    metrics["wheel_contact_force_rmse_raw"] = torch.sqrt(torch.stack(raw_force_sqerrs).mean()) if raw_force_sqerrs else ref.new_tensor(0.0)
    metrics["wheel_contact_moment_loss_scaled"] = torch.stack(scaled_moment_losses).mean() if scaled_moment_losses else ref.new_tensor(0.0)
    metrics["wheel_contact_moment_loss_raw"] = torch.stack(raw_moment_losses).mean() if raw_moment_losses else ref.new_tensor(0.0)
    metrics["wheel_contact_moment_rmse_raw"] = torch.sqrt(torch.stack(raw_moment_sqerrs).mean()) if raw_moment_sqerrs else ref.new_tensor(0.0)
    return metrics


def select_horizon_step(x: torch.Tensor, step_idx: int) -> torch.Tensor:
    if x.ndim == 3:
        return x[:, step_idx, :]
    return x


def get_autocast_context(
    device: torch.device,
    enabled: bool,
    amp_dtype: torch.dtype,
):
    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def unwrap_model(model):
    return getattr(model, "_orig_mod", model)


def get_model_attr(model, name: str, default=None):
    base_model = unwrap_model(model)
    if hasattr(model, name):
        return getattr(model, name)
    if hasattr(base_model, name):
        return getattr(base_model, name)
    return default


def should_enable_tqdm() -> bool:
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False

def compute_losses(
    batch: Dict,
    output: Dict,
    spec,
    scaler,
    lambda_res: float = 0.0,
    lambda_hf: float = 0.0,
    lambda_quat: float = 0.0,
    lambda_contact: float = 1.0,
    lambda_contact_force: Optional[float] = None,
    lambda_contact_moment: Optional[float] = 0.0,
    lambda_contact_raw: float = 0.0,
    lambda_contact_physics: float = 0.02,
    lambda_kin: float = 0.0,
    lambda_smooth: float = 0.0,
    lambda_body_pos: float = 1.0,
    lambda_wheel_pos_z: float = 1.0,
    lambda_body_vel: float = 0.15,
    body_pos_blend_alpha: float = 0.2,
    lambda_body_future_kin: float = 0.05,
    raw_monitor_body_weight: float = 1.0,
    raw_monitor_force_weight: float = 1.0,
    raw_monitor_moment_weight: float = 1.0,
    acc_weight: float = 2.0,
    dt: float = 0.015,
) -> Dict[str, torch.Tensor]:
    losses: Dict[str, torch.Tensor] = {}
    if lambda_contact_force is None:
        lambda_contact_force = lambda_contact
    if lambda_contact_moment is None:
        lambda_contact_moment = lambda_contact

    loss_res = weighted_state_loss(
        output["pred_res_body"],
        batch["res_body"],
        spec.res_groups.body_cols,
        acc_weight=acc_weight,
    )
    loss_res_count = 1
    for i in WHEEL_IDS:
        loss_res = loss_res + weighted_state_loss(
            output[f"pred_res_wheel{i}_kin"],
            batch[f"res_wheel{i}_kin"],
            spec.res_groups.wheel_kin_cols[i],
            acc_weight=acc_weight,
        )
        loss_res = loss_res + safe_huber_or_mse(output[f"pred_res_wheel{i}_contact"], batch[f"res_wheel{i}_contact"])
        loss_res_count += 2
    losses["res"] = loss_res / loss_res_count

    pred_hf_body = reconstruct_hf_scaled(
        output["pred_res_body"],
        batch["lf_body_current"],
        spec.res_groups.body_cols,
        spec.target_groups.body_cols,
        "res_body",
        "hf_body",
        scaler,
    )
    body_metrics, pred_hf_body_raw, true_hf_body_raw = compute_body_supervision_metrics(
        pred_hf_body,
        batch["hf_body"],
        spec.target_groups.body_cols,
        scaler,
    )
    losses.update(body_metrics)
    losses.update(
        compute_body_fused_position_metrics(
            pred_hf_body_scaled=pred_hf_body,
            true_hf_body_scaled=batch["hf_body"],
            pred_hf_body_raw=pred_hf_body_raw,
            true_hf_body_raw=true_hf_body_raw,
            batch=batch,
            cols=spec.target_groups.body_cols,
            scaler=scaler,
            alpha=body_pos_blend_alpha,
            dt=dt,
        )
    )

    pred_hf_wheel_kin: Dict[int, torch.Tensor] = {}
    pred_hf_wheel_contact: Dict[int, torch.Tensor] = {}
    for i in WHEEL_IDS:
        pred_hf_wheel_kin[i] = reconstruct_hf_scaled(
            output[f"pred_res_wheel{i}_kin"],
            batch[f"lf_wheel{i}_kin_current"],
            spec.res_groups.wheel_kin_cols[i],
            spec.target_groups.wheel_kin_cols[i],
            f"res_wheel{i}_kin",
            f"hf_wheel{i}_kin",
            scaler,
        )
        pred_hf_wheel_contact[i] = reconstruct_hf_scaled(
            output[f"pred_res_wheel{i}_contact"],
            batch[f"lf_wheel{i}_contact_current"],
            spec.res_groups.wheel_contact_cols[i],
            spec.target_groups.wheel_contact_cols[i],
            f"res_wheel{i}_contact",
            f"hf_wheel{i}_contact",
            scaler,
        )
    losses.update(
        compute_wheel_contact_supervision_metrics(
            pred_hf_wheel_contact=pred_hf_wheel_contact,
            batch=batch,
            spec=spec,
            scaler=scaler,
        )
    )
    losses.update(
        compute_wheel_pos_z_supervision_metrics(
            pred_hf_wheel_kin=pred_hf_wheel_kin,
            batch=batch,
            spec=spec,
            scaler=scaler,
        )
    )
    losses["hf"] = pred_hf_body.new_tensor(0.0)
    losses["quat"] = pred_hf_body.new_tensor(0.0)

    loss_contact_physics = pred_hf_body.new_tensor(0.0)
    contact_count = 0
    for i in WHEEL_IDS:
        idx = get_contact_indices(spec.target_groups.wheel_contact_cols[i])
        if "Fz" in idx and "sinkage" in idx and pred_hf_wheel_contact[i].shape[-1] > 0:
            contact_mask = build_contact_active_mask(batch, spec, i, pred_hf_wheel_contact[i])
            contact_raw = inverse_transform_tensor(pred_hf_wheel_contact[i], scaler, f"hf_wheel{i}_contact")
            fz = contact_raw[..., idx["Fz"]]
            sink = contact_raw[..., idx["sinkage"]]
            if contact_mask is not None:
                active = contact_mask.to(device=fz.device, dtype=torch.bool)
                if active.any():
                    loss_contact_physics = loss_contact_physics + torch.relu(-sink[active]).mean() + 0.1 * torch.relu(-fz[active]).mean()
                    contact_count += 1
            else:
                loss_contact_physics = loss_contact_physics + torch.relu(-sink).mean() + 0.1 * torch.relu(-fz).mean()
                contact_count += 1
    if contact_count > 0:
        loss_contact_physics = loss_contact_physics / contact_count
    losses["contact_physics"] = loss_contact_physics
    losses["contact"] = losses["wheel_contact_rmse_raw"]

    loss_kin = pred_hf_body.new_tensor(0.0)
    loss_smooth = pred_hf_body.new_tensor(0.0)
    losses["body_future_kin"] = pred_hf_body.new_tensor(0.0)

    kin_count = 0
    smooth_count = 0

    if "hf_body_hist" in batch:
        pred_hf_body_seq = build_hf_sequence_with_pred_current(
            batch["hf_body_hist"],
            pred_hf_body,
        )

        loss_kin = loss_kin + kinematic_loss_from_state_sequence(
            pred_hf_body_seq,
            spec.target_groups.body_cols,
            scaler,
            "hf_body",
            dt,
        )
        kin_count += 1

        body_smooth_idx = []
        body_smooth_idx += get_xyz_indices(spec.target_groups.body_cols, "pos")
        body_smooth_idx += get_xyz_indices(spec.target_groups.body_cols, "vel")
        body_smooth_idx += get_xyz_indices(spec.target_groups.body_cols, "acc")

        if len(body_smooth_idx) > 0:
            loss_smooth = loss_smooth + smoothness_loss(pred_hf_body_seq[:, :, body_smooth_idx])
            smooth_count += 1

    if pred_hf_body.ndim == 3 and pred_hf_body.shape[1] > 1:
        losses["body_future_kin"] = kinematic_loss_from_state_sequence(
            pred_hf_body,
            spec.target_groups.body_cols,
            scaler,
            "hf_body",
            dt,
        )

    for i in WHEEL_IDS:
        hist_key = f"hf_wheel{i}_kin_hist"

        if hist_key not in batch:
            continue

        pred_hf_wheel_kin_seq = build_hf_sequence_with_pred_current(
            batch[hist_key],
            pred_hf_wheel_kin[i],
        )

        loss_kin = loss_kin + kinematic_loss_from_state_sequence(
            pred_hf_wheel_kin_seq,
            spec.target_groups.wheel_kin_cols[i],
            scaler,
            f"hf_wheel{i}_kin",
            dt,
        )
        kin_count += 1

        wheel_smooth_idx = []
        wheel_smooth_idx += get_xyz_indices(spec.target_groups.wheel_kin_cols[i], "pos")
        wheel_smooth_idx += get_xyz_indices(spec.target_groups.wheel_kin_cols[i], "vel")
        wheel_smooth_idx += get_xyz_indices(spec.target_groups.wheel_kin_cols[i], "acc")

        if len(wheel_smooth_idx) > 0:
            loss_smooth = loss_smooth + smoothness_loss(pred_hf_wheel_kin_seq[:, :, wheel_smooth_idx])
            smooth_count += 1

    if kin_count > 0:
        loss_kin = loss_kin / kin_count

    if smooth_count > 0:
        loss_smooth = loss_smooth / smooth_count

    losses["kin"] = loss_kin
    losses["smooth"] = loss_smooth
    losses["res_body_posxyz"] = losses["body_pos_fused_xyz_rmse_raw"]
    losses["weighted_body_pos"] = lambda_body_pos * losses["body_pos_fused_xyz_loss_scaled"]
    losses["weighted_body_vel"] = lambda_body_vel * losses["body_vel_xyz_loss_scaled"]
    losses["weighted_wheel_pos_z"] = lambda_wheel_pos_z * losses["wheel_pos_z_loss_scaled"]
    losses["weighted_contact_force"] = lambda_contact_force * losses["wheel_contact_force_loss_scaled"]
    losses["weighted_contact_moment"] = lambda_contact_moment * losses["wheel_contact_moment_loss_scaled"]
    losses["weighted_contact_scaled"] = losses["weighted_contact_force"] + losses["weighted_contact_moment"]
    losses["weighted_contact_raw"] = lambda_contact_raw * losses["wheel_contact_rmse_raw"]
    losses["weighted_contact"] = (
        losses["weighted_contact_scaled"]
        + lambda_contact_physics * losses["contact_physics"]
    )
    losses["weighted_kin"] = lambda_kin * losses["kin"]
    losses["weighted_smooth"] = lambda_smooth * losses["smooth"]
    losses["selection_score"] = (
        losses["weighted_body_pos"]
        + losses["weighted_body_vel"]
        + losses["weighted_wheel_pos_z"]
        + losses["weighted_contact"]
    )

    body_pos_scale = mean_scale_for_suffixes(
        scaler,
        "hf_body",
        spec.target_groups.body_cols,
        ["pos_x", "pos_y", "pos_z"],
        pred_hf_body.device,
    )
    body_vel_scale = mean_scale_for_suffixes(
        scaler,
        "hf_body",
        spec.target_groups.body_cols,
        ["vel_x", "vel_y", "vel_z"],
        pred_hf_body.device,
    )
    force_scales = [
        mean_scale_for_suffixes(
            scaler,
            f"hf_wheel{i}_contact",
            spec.target_groups.wheel_contact_cols[i],
            ["Fx", "Fy", "Fz"],
            pred_hf_body.device,
        )
        for i in WHEEL_IDS
    ]
    moment_scales = [
        mean_scale_for_suffixes(
            scaler,
            f"hf_wheel{i}_contact",
            spec.target_groups.wheel_contact_cols[i],
            ["Mx", "My", "Mz"],
            pred_hf_body.device,
        )
        for i in WHEEL_IDS
    ]
    force_scale = torch.stack(force_scales).mean().clamp_min(1e-6)
    moment_scale = torch.stack(moment_scales).mean().clamp_min(1e-6)
    losses["monitor_body_pos_raw"] = losses["body_pos_fused_xyz_rmse_raw"] / body_pos_scale
    losses["monitor_body_vel_raw"] = losses["body_vel_xyz_rmse_raw"] / body_vel_scale
    losses["monitor_contact_force_raw"] = losses["wheel_contact_force_rmse_raw"] / force_scale
    losses["monitor_contact_moment_raw"] = losses["wheel_contact_moment_rmse_raw"] / moment_scale
    losses["raw_task_monitor"] = (
        raw_monitor_body_weight * losses["monitor_body_vel_raw"]
        + raw_monitor_force_weight * losses["monitor_contact_force_raw"]
        + raw_monitor_moment_weight * losses["monitor_contact_moment_raw"]
    )

    losses["total"] = losses["selection_score"] + losses["weighted_kin"] + losses["weighted_smooth"]
    return losses

# @torch.no_grad()
# def log_validation_prediction_figures(
#     writer: SummaryWriter,
#     model,
#     val_loader,
#     device,
#     spec,
#     scaler,
#     epoch: int,
#     max_samples: int = 64,
# ) -> None:
#     """
#     在 TensorBoard 中记录部分验证集样本的预测效果。

#     显示内容：
#     1. 车身高保真真实值 hf
#     2. 低保真输入 lf
#     3. 模型补偿后的预测值 pred_hf

#     横轴是当前取出的验证集样本编号，不一定代表连续物理时间。
#     """
#     if val_loader is None:
#         return

#     model.eval()

#     try:
#         batch = next(iter(val_loader))
#     except StopIteration:
#         return

#     batch = move_batch_to_device(batch, device)

#     output = model(batch)

#     n = min(max_samples, batch["res_body"].shape[0])
#     if n <= 0:
#         return
    
#     def _to_np(x: torch.Tensor) -> np.ndarray:
#         return x[:n].detach().cpu().numpy()

#     def _find_col(cols: List[str], suffix: str):
#         for idx, c in enumerate(cols):
#             if c.endswith(suffix):
#                 return idx
#         return None

#     def _plot_group(
#         tag_prefix: str,
#         pred_hf_scaled: torch.Tensor,
#         true_hf_scaled: torch.Tensor,
#         lf_current_raw: torch.Tensor,
#         hf_cols: List[str],
#         hf_group_name: str,
#         plot_suffixes: List[str],
#     ) -> None:
#         pred_raw = inverse_transform_tensor(pred_hf_scaled, scaler, hf_group_name)
#         true_raw = inverse_transform_tensor(true_hf_scaled, scaler, hf_group_name)

#         for suffix in plot_suffixes:
#             j = _find_col(hf_cols, suffix)
#             if j is None:
#                 continue

#             fig = plt.figure(figsize=(10, 4))
#             x = np.arange(n)

#             plt.plot(x, _to_np(true_raw[:, j]), label="HF true", linewidth=2.0)
#             plt.plot(x, _to_np(pred_raw[:, j]), label="Pred HF", linewidth=1.8)
#             plt.plot(x, _to_np(lf_current_raw[:, j]), label="LF current", linewidth=1.5)

#             plt.xlabel("validation sample index")
#             plt.ylabel(suffix)
#             plt.title(f"{tag_prefix} | {suffix}")
#             plt.legend()
#             plt.grid(True, alpha=0.3)
#             plt.tight_layout()

#             writer.add_figure(f"val_effect/{tag_prefix}/{suffix}", fig, epoch)
#             plt.close(fig)

#     pred_hf_body = reconstruct_hf_scaled(
#         output["pred_res_body"],
#         batch["lf_body_current"],
#         spec.res_groups.body_cols,
#         spec.target_groups.body_cols,
#         "res_body",
#         "hf_body",
#         scaler,
#     )

#     _plot_group(
#         tag_prefix="body",
#         pred_hf_scaled=pred_hf_body,
#         true_hf_scaled=batch["hf_body"],
#         lf_current_raw=batch["lf_body_current"],
#         hf_cols=spec.target_groups.body_cols,
#         hf_group_name="hf_body",
#         plot_suffixes=[
#             "pos_x", "pos_y", "pos_z",
#             "vel_x", "vel_y", "vel_z",
#             "acc_x", "acc_y", "acc_z",
#         ],
#     )

#     for i in WHEEL_IDS:
#         pred_hf_wheel_kin = reconstruct_hf_scaled(
#             output[f"pred_res_wheel{i}_kin"],
#             batch[f"lf_wheel{i}_kin_current"],
#             spec.res_groups.wheel_kin_cols[i],
#             spec.target_groups.wheel_kin_cols[i],
#             f"res_wheel{i}_kin",
#             f"hf_wheel{i}_kin",
#             scaler,
#         )

#         _plot_group(
#             tag_prefix=f"wheel{i}_kin",
#             pred_hf_scaled=pred_hf_wheel_kin,
#             true_hf_scaled=batch[f"hf_wheel{i}_kin"],
#             lf_current_raw=batch[f"lf_wheel{i}_kin_current"],
#             hf_cols=spec.target_groups.wheel_kin_cols[i],
#             hf_group_name=f"hf_wheel{i}_kin",
#             plot_suffixes=[
#                 "pos_x", "pos_y", "pos_z",
#                 "vel_x", "vel_y", "vel_z",
#                 "acc_x", "acc_y", "acc_z",
#             ],
#         )

#         pred_hf_wheel_contact = reconstruct_hf_scaled(
#             output[f"pred_res_wheel{i}_contact"],
#             batch[f"lf_wheel{i}_contact_current"],
#             spec.res_groups.wheel_contact_cols[i],
#             spec.target_groups.wheel_contact_cols[i],
#             f"res_wheel{i}_contact",
#             f"hf_wheel{i}_contact",
#             scaler,
#         )

#         _plot_group(
#             tag_prefix=f"wheel{i}_contact",
#             pred_hf_scaled=pred_hf_wheel_contact,
#             true_hf_scaled=batch[f"hf_wheel{i}_contact"],
#             lf_current_raw=batch[f"lf_wheel{i}_contact_current"],
#             hf_cols=spec.target_groups.wheel_contact_cols[i],
#             hf_group_name=f"hf_wheel{i}_contact",
#             plot_suffixes=[
#                 "Fx", "Fy", "Fz",
#                 "Mx", "My", "Mz",
#                 "sinkage",
#                 "slip_long",
#                 "slip_lat",
#                 "in_contact",
#             ],
#         )
@torch.no_grad()
def log_validation_prediction_figures(
    writer: SummaryWriter,
    model,
    val_loader,
    device,
    spec,
    scaler,
    part_cfg: TrainingPartConfig,
    teacher_gate_net: Optional[TeacherGateNet],
    epoch: int,
    max_samples: int = 0,
    max_cases: int = 3,
    selected_case: str = None,
    run_label: str = "",
) -> None:
    """
    在 TensorBoard 中记录某一个验证集 case 的连续时间预测效果。

    显示内容：
    1. 车身高保真真实值 hf
    2. 低保真输入 lf
    3. 模型补偿后的预测值 pred_hf

    横轴是真实物理时间 time。
    当 pred_seq_len > 1 时，固定可视化 horizon 第 1 个预测步。
    若 selected_case 为 None，则随机选择一个验证集中样本数足够的 case。
    默认展示该 case 的全部可预测时刻；仅当 max_samples > 0 时才截断。
    """
    if val_loader is None:
        return

    model.eval()

    all_batches = []

    for batch in val_loader:
        all_batches.append(batch)

    if len(all_batches) == 0:
        return

    def _concat_tensor(key: str):
        values = []
        for b in all_batches:
            if key in b and torch.is_tensor(b[key]):
                values.append(b[key])
        if len(values) == 0:
            return None
        return torch.cat(values, dim=0)

    def _concat_meta(key: str):
        values = []
        for b in all_batches:
            if key not in b:
                continue

            v = b[key]

            if torch.is_tensor(v):
                values.extend(v.detach().cpu().numpy().tolist())
            elif isinstance(v, np.ndarray):
                values.extend(v.tolist())
            elif isinstance(v, list):
                values.extend(v)
            else:
                values.extend(list(v))

        return values

    case_names = _concat_meta("case_name")
    times = _concat_meta("time")

    if len(case_names) == 0 or len(times) == 0:
        return

    case_names = np.asarray(case_names).astype(str)
    times = np.asarray(times, dtype=np.float64)

    vis_horizon_idx = 0

    def _to_np(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()

    def _find_col(cols: List[str], suffix: str):
        for idx, c in enumerate(cols):
            if c.endswith(suffix):
                return idx
        return None

    def _plot_group(
        case_name: str,
        case_times: np.ndarray,
        tag_prefix: str,
        pred_hf_scaled: torch.Tensor,
        true_hf_scaled: torch.Tensor,
        lf_current_raw: torch.Tensor,
        hf_cols: List[str],
        hf_group_name: str,
        plot_suffixes: List[str],
    ) -> None:
        pred_hf_scaled = select_horizon_step(pred_hf_scaled, vis_horizon_idx)
        true_hf_scaled = select_horizon_step(true_hf_scaled, vis_horizon_idx)
        lf_current_raw = select_horizon_step(lf_current_raw, vis_horizon_idx)

        pred_raw = inverse_transform_tensor(pred_hf_scaled, scaler, hf_group_name)
        true_raw = inverse_transform_tensor(true_hf_scaled, scaler, hf_group_name)

        x = case_times

        for suffix in plot_suffixes:
            j = _find_col(hf_cols, suffix)
            if j is None:
                continue

            fig = plt.figure(figsize=(10, 4))

            plt.plot(x, _to_np(true_raw[:, j]), label="HF true", linewidth=2.0)
            plt.plot(x, _to_np(pred_raw[:, j]), label="Compensated", linewidth=1.8)
            plt.plot(x, _to_np(lf_current_raw[:, j]), label="LF current", linewidth=1.5)

            plt.xlabel("time / s")
            plt.ylabel(suffix)
            title_suffix = f" | {run_label}" if run_label else ""
            plt.title(f"{tag_prefix} | {suffix} | {case_name} | horizon_step=1{title_suffix}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            writer.add_figure(f"val_effect/{case_name}/{tag_prefix}/horizon_1/{suffix}", fig, epoch)
            plt.close(fig)
    unique_cases = list(dict.fromkeys(case_names.tolist()))
    if selected_case is not None:
        selected_cases = [str(selected_case)]
    else:
        candidate_cases = [
            c for c in unique_cases
            if np.sum(case_names == c) >= 2
        ]
        if len(candidate_cases) == 0:
            selected_cases = []
        else:
            rng = np.random.default_rng(seed=epoch)
            selected_cases = [str(rng.choice(candidate_cases))]

    if not selected_cases:
        return

    writer.add_text(
        "val_effect/selected_cases",
        json.dumps(selected_cases, ensure_ascii=False),
        epoch,
    )

    first_batch = all_batches[0]
    vis_forward_batch_size = int(getattr(val_loader, "batch_size", 0) or 16)
    vis_forward_batch_size = max(1, vis_forward_batch_size)
    vis_amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    for case_name in selected_cases:
        case_mask = case_names == case_name
        case_indices = np.where(case_mask)[0]
        if len(case_indices) <= 1:
            continue

        case_times = times[case_indices]
        order = np.argsort(case_times)
        case_indices = case_indices[order]
        case_times = case_times[order]

        n = len(case_indices) if max_samples <= 0 else min(max_samples, len(case_indices))
        case_indices = case_indices[:n]
        case_times = case_times[:n]
        index_tensor = torch.as_tensor(case_indices, dtype=torch.long)

        full_batch = {}
        for key in first_batch.keys():
            if torch.is_tensor(first_batch[key]):
                full_value = _concat_tensor(key)
                if full_value is not None:
                    full_batch[key] = full_value[index_tensor]

        full_batch["case_name"] = case_names[case_indices].tolist()
        full_batch["time"] = torch.as_tensor(case_times, dtype=torch.float32)

        pred_keys = ["pred_res_body"]
        for i in WHEEL_IDS:
            pred_keys.append(f"pred_res_wheel{i}_kin")
            pred_keys.append(f"pred_res_wheel{i}_contact")

        pred_chunks: Dict[str, List[torch.Tensor]] = {key: [] for key in pred_keys}
        for start_idx in range(0, n, vis_forward_batch_size):
            end_idx = min(start_idx + vis_forward_batch_size, n)
            chunk_batch = {
                key: value[start_idx:end_idx]
                for key, value in full_batch.items()
                if torch.is_tensor(value)
            }

            batch = move_batch_to_device(chunk_batch, device)
            with get_autocast_context(device, enabled=device.type == "cuda", amp_dtype=vis_amp_dtype):
                output = forward_with_stage_gate(
                    model=model,
                    batch=batch,
                    part_cfg=part_cfg,
                    teacher_gate_net=teacher_gate_net,
                    teacher_requires_grad=False,
                )

            for key in pred_keys:
                pred_chunks[key].append(output[key].detach().cpu())

            del batch
            del output

        pred_output = {
            key: torch.cat(chunks, dim=0)
            for key, chunks in pred_chunks.items()
            if len(chunks) > 0
        }
        if device.type == "cuda":
            torch.cuda.empty_cache()

        pred_hf_body = reconstruct_hf_scaled(
            pred_output["pred_res_body"],
            full_batch["lf_body_current"],
            spec.res_groups.body_cols,
            spec.target_groups.body_cols,
            "res_body",
            "hf_body",
            scaler,
        )

        _plot_group(
            case_name=case_name,
            case_times=case_times,
            tag_prefix="body",
            pred_hf_scaled=pred_hf_body,
            true_hf_scaled=full_batch["hf_body"],
            lf_current_raw=full_batch["lf_body_current"],
            hf_cols=spec.target_groups.body_cols,
            hf_group_name="hf_body",
            plot_suffixes=[
                "pos_x", "pos_y", "pos_z",
            ],
        )

        for i in WHEEL_IDS:
            pred_hf_wheel_kin = reconstruct_hf_scaled(
                pred_output[f"pred_res_wheel{i}_kin"],
                full_batch[f"lf_wheel{i}_kin_current"],
                spec.res_groups.wheel_kin_cols[i],
                spec.target_groups.wheel_kin_cols[i],
                f"res_wheel{i}_kin",
                f"hf_wheel{i}_kin",
                scaler,
            )

            _plot_group(
                case_name=case_name,
                case_times=case_times,
                tag_prefix=f"wheel{i}_kin",
                pred_hf_scaled=pred_hf_wheel_kin,
                true_hf_scaled=full_batch[f"hf_wheel{i}_kin"],
                lf_current_raw=full_batch[f"lf_wheel{i}_kin_current"],
                hf_cols=spec.target_groups.wheel_kin_cols[i],
                hf_group_name=f"hf_wheel{i}_kin",
                plot_suffixes=[
                    "pos_x", "pos_y", "pos_z",
                ],
            )

            pred_hf_wheel_contact = reconstruct_hf_scaled(
                pred_output[f"pred_res_wheel{i}_contact"],
                full_batch[f"lf_wheel{i}_contact_current"],
                spec.res_groups.wheel_contact_cols[i],
                spec.target_groups.wheel_contact_cols[i],
                f"res_wheel{i}_contact",
                f"hf_wheel{i}_contact",
                scaler,
            )

            _plot_group(
                case_name=case_name,
                case_times=case_times,
                tag_prefix=f"wheel{i}_contact",
                pred_hf_scaled=pred_hf_wheel_contact,
                true_hf_scaled=full_batch[f"hf_wheel{i}_contact"],
                lf_current_raw=full_batch[f"lf_wheel{i}_contact_current"],
                hf_cols=spec.target_groups.wheel_contact_cols[i],
                hf_group_name=f"hf_wheel{i}_contact",
                plot_suffixes=[
                    "Fx", "Fy", "Fz",
                    "Mx", "My", "Mz",
                ],
            )


@torch.no_grad()
def log_validation_gate_stats(
    writer: SummaryWriter,
    model,
    val_loader,
    device,
    part_cfg: TrainingPartConfig,
    teacher_gate_net: Optional[TeacherGateNet],
    epoch: int,
    max_batches: int = 4,
) -> None:
    if val_loader is None:
        return

    model.eval()
    relation_totals: Dict[str, float] = {}
    edge_totals: Dict[str, float] = {}
    batch_count = 0
    sample_count = 0
    relation_names = get_model_attr(model, "relation_names", None)
    edge_specs = get_model_attr(model, "edge_specs", None)
    relation_to_edge_idx: Dict[int, List[int]] = {}
    if relation_names is not None and edge_specs is not None:
        for edge_idx, edge_spec in enumerate(edge_specs):
            relation_to_edge_idx.setdefault(edge_spec.relation_id, []).append(edge_idx)

    for first_batch in val_loader:
        batch_count += 1
        if batch_count > max_batches:
            break
        if first_batch is None:
            continue

        batch = move_batch_to_device(first_batch, device)
        output = forward_with_stage_gate(
            model=model,
            batch=batch,
            part_cfg=part_cfg,
            teacher_gate_net=teacher_gate_net,
            teacher_requires_grad=False,
        )
        bs = int(batch["res_body"].shape[0])
        sample_count += bs

        relation_gate = output.get("relation_gate_stats")
        if relation_gate is not None:
            relation_totals["mean"] = relation_totals.get("mean", 0.0) + float(relation_gate.mean().item()) * bs
            relation_totals["std"] = relation_totals.get("std", 0.0) + float(relation_gate.std().item()) * bs
            if relation_names is not None:
                for rel_idx, rel_name in enumerate(relation_names):
                    relation_totals[rel_name] = relation_totals.get(rel_name, 0.0) + float(relation_gate[:, :, rel_idx].mean().item()) * bs

        edge_gate = output.get("edge_gate_stats")
        if edge_gate is None:
            continue

        edge_totals["mean"] = edge_totals.get("mean", 0.0) + float(edge_gate.mean().item()) * bs
        if relation_names is None or edge_specs is None:
            continue
        for relation_id, relation_name in enumerate(relation_names):
            edge_indices = relation_to_edge_idx.get(relation_id, [])
            if len(edge_indices) == 0:
                continue
            rel_edge_gate = edge_gate[:, :, edge_indices, :]
            edge_totals[relation_name] = edge_totals.get(relation_name, 0.0) + float(rel_edge_gate.mean().item()) * bs

    if sample_count == 0:
        return

    if relation_totals:
        writer.add_scalar("gate/val_relation_gate_mean", relation_totals["mean"] / sample_count, epoch)
        writer.add_scalar("gate/val_relation_gate_std", relation_totals["std"] / sample_count, epoch)
        if relation_names is not None:
            for relation_name in relation_names:
                if relation_name not in relation_totals:
                    continue
                writer.add_scalar(
                    f"gate/val_relation_gate_by_relation/{relation_name}",
                    relation_totals[relation_name] / sample_count,
                    epoch,
                )

    if edge_totals:
        writer.add_scalar("gate/val_edge_gate_mean", edge_totals["mean"] / sample_count, epoch)
        if relation_names is not None:
            for relation_name in relation_names:
                if relation_name not in edge_totals:
                    continue
                writer.add_scalar(
                    f"gate/val_edge_gate_by_relation/{relation_name}",
                    edge_totals[relation_name] / sample_count,
                    epoch,
                )


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    spec,
    scaler,
    part_cfg: TrainingPartConfig,
    teacher_gate_net: Optional[TeacherGateNet] = None,
    lambda_res: float = 0.0,
    lambda_hf: float = 0.0,
    lambda_quat: float = 0.0,
    lambda_contact: float = 1.0,
    lambda_contact_force: Optional[float] = None,
    lambda_contact_moment: Optional[float] = 0.0,
    lambda_contact_raw: float = 0.0,
    lambda_contact_physics: float = 0.02,
    lambda_kin: float = 0.0,
    lambda_smooth: float = 0.0,
    lambda_body_pos: float = 1.0,
    lambda_wheel_pos_z: float = 1.0,
    lambda_body_vel: float = 0.15,
    body_pos_blend_alpha: float = 0.2,
    lambda_body_future_kin: float = 0.05,
    raw_monitor_body_weight: float = 1.0,
    raw_monitor_force_weight: float = 1.0,
    raw_monitor_moment_weight: float = 1.0,
    acc_weight: float = 2.0,
    dt: float = 0.015,
    lambda_latent: float = 1.0,
    lambda_gate: float = 1.0,
    desc: str = "Val",
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
):
    model.eval()
    meter = {key: 0.0 for key in BASE_METRIC_KEYS}
    n = 0

    pbar = tqdm(
        loader,
        desc=desc,
        leave=False,
        dynamic_ncols=True,
        disable=not should_enable_tqdm(),
    )

    for batch in pbar:
        batch = move_batch_to_device(batch, device)
        with get_autocast_context(device, use_amp, amp_dtype):
            output = forward_with_stage_gate(
                model=model,
                batch=batch,
                part_cfg=part_cfg,
                teacher_gate_net=teacher_gate_net,
                teacher_requires_grad=False,
            )
            losses = compute_losses(
                batch,
                output,
                spec,
                scaler,
                lambda_res=lambda_res,
                lambda_hf=lambda_hf,
                lambda_quat=lambda_quat,
                lambda_contact=lambda_contact,
                lambda_contact_force=lambda_contact_force,
                lambda_contact_moment=lambda_contact_moment,
                lambda_contact_raw=lambda_contact_raw,
                lambda_contact_physics=lambda_contact_physics,
                lambda_kin=lambda_kin,
                lambda_smooth=lambda_smooth,
                lambda_body_pos=lambda_body_pos,
                lambda_wheel_pos_z=lambda_wheel_pos_z,
                lambda_body_vel=lambda_body_vel,
                body_pos_blend_alpha=body_pos_blend_alpha,
                lambda_body_future_kin=lambda_body_future_kin,
                raw_monitor_body_weight=raw_monitor_body_weight,
                raw_monitor_force_weight=raw_monitor_force_weight,
                raw_monitor_moment_weight=raw_monitor_moment_weight,
                acc_weight=acc_weight,
                dt=dt,
            )
            losses = apply_stage_losses(
                losses=losses,
                batch=batch,
                output=output,
                part_cfg=part_cfg,
                lambda_latent=lambda_latent,
                lambda_gate=lambda_gate,
            )

        bs = batch["res_body"].shape[0]
        n += bs

        for k in meter:
            meter[k] += float(losses[k].item()) * bs

        avg_total = meter["total"] / max(n, 1)
        avg_res = meter["res"] / max(n, 1)
        avg_contact = meter["contact"] / max(n, 1)

        pbar.set_postfix({
            "total": f"{avg_total:.5f}",
            "w_bvel": f"{meter['weighted_body_vel'] / max(n, 1):.5f}",
            "w_wpz": f"{meter['weighted_wheel_pos_z'] / max(n, 1):.5f}",
            "w_cF": f"{meter['weighted_contact_force'] / max(n, 1):.5f}",
            "w_cM": f"{meter['weighted_contact_moment'] / max(n, 1):.5f}",
            "w_cPhy": f"{(meter['weighted_contact'] - meter['weighted_contact_scaled']) / max(n, 1):.5f}",
        })

    if n == 0:
        return {k: 0.0 for k in meter}

    return {k: v / n for k, v in meter.items()}


def get_val_gate_mode(part_cfg: TrainingPartConfig) -> str:
    return "teacher" if part_cfg.gate_mode == "teacher" else "student"


def get_best_monitor_name(part_cfg: TrainingPartConfig) -> str:
    # teacher_full 直接按优化目标本身选 best/early-stop，避免监控指标与训练目标错位。
    if part_cfg.key == "teacher_full":
        return "total"
    return "raw_task_monitor"


def build_checkpoint_payload(
    model: torch.nn.Module,
    teacher_gate_net: Optional[torch.nn.Module],
    group_dims: Dict[str, int],
    args: argparse.Namespace,
    part_cfg: TrainingPartConfig,
    freeze_info: Dict[str, List[str]],
    scaler,
    best_val: float,
    best_epoch: int,
    best_monitor_name: str,
) -> Dict:
    scaler_state = {name: scaler_obj.to_dict() for name, scaler_obj in scaler.scalers.items()}
    return {
        "model_state_dict": unwrap_model(model).state_dict(),
        "teacher_gate_state_dict": teacher_gate_net.state_dict() if teacher_gate_net is not None else None,
        "group_dims": group_dims,
        "args": vars(args),
        "train_stage": args.train_stage,
        "val_gate_mode": get_val_gate_mode(part_cfg),
        "source_ckpt": args.init_ckpt,
        "frozen_modules": freeze_info,
        "best_val": best_val,
        "best_epoch": best_epoch,
        "best_monitor_name": best_monitor_name,
        "scaler_state": scaler_state,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", type=str, default=str(ROOT / "Feature_Selection" / "DataSet"))
    parser.add_argument("--merged_csv", type=str, default=str(ROOT / "Feature_Selection" / "DataSet" / "merged_error_dataset.csv"))
    parser.add_argument("--save_dir", type=str, default=str(ROOT / "results" / "0708"))
    parser.add_argument("--log_dir", type=str, default=None, help="TensorBoard 日志目录，默认保存到 save_dir/tensorboard")
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--pred_horizon", type=int, default=0)
    parser.add_argument("--pred_seq_len", type=int, default=1)
    parser.add_argument(
        "--train_stage",
        type=str,
        default="teacher_full",
        help="整体训练阶段，可选: teacher_full, student_full",
    )
    parser.add_argument("--train_part", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--init_ckpt", type=str, default=None, help="初始化 checkpoint 路径；student_full 未提供时自动回溯 teacher_full 最新 best_model.pt")
    parser.add_argument("--train_main_in_stage2", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lambda_latent", type=float, default=1.0)
    parser.add_argument("--lambda_gate", type=float, default=1.0)
    parser.add_argument("--teacher_hidden_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--graph_layers", type=int, default=3)
    parser.add_argument("--tcn_dim", type=int, default=256)
    parser.add_argument("--lstm_dim", type=int, default=128)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min_epochs_before_stop", type=int, default=40)
    parser.add_argument("--lambda_res", type=float, default=0.0)
    parser.add_argument("--lambda_hf", type=float, default=0.0)
    parser.add_argument("--lambda_quat", type=float, default=0.0)
    parser.add_argument("--lambda_contact", type=float, default=0.0)
    parser.add_argument("--lambda_contact_force", type=float, default=None)
    parser.add_argument("--lambda_contact_moment", type=float, default=0.0)
    parser.add_argument("--lambda_contact_raw", type=float, default=0.0)
    parser.add_argument("--lambda_contact_physics", type=float, default=0.0)
    parser.add_argument("--lambda_kin", type=float, default=0.05)
    parser.add_argument("--lambda_smooth", type=float, default=0.0)
    parser.add_argument("--lambda_body_pos", type=float, default=0.2)
    parser.add_argument("--lambda_wheel_pos_z", type=float, default=1.0)
    parser.add_argument("--lambda_body_vel", type=float, default=0.8)
    parser.add_argument("--body_pos_blend_alpha", type=float, default=0.2)
    parser.add_argument("--lambda_body_future_kin", type=float, default=0.0)
    parser.add_argument("--raw_monitor_body_weight", type=float, default=1.0)
    parser.add_argument("--raw_monitor_force_weight", type=float, default=1.0)
    parser.add_argument("--raw_monitor_moment_weight", type=float, default=1.0)
    parser.add_argument("--acc_weight", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=0.015)
    parser.add_argument("--eval_interval", type=int, default=1, help="每隔多少个 epoch 跑一次完整验证")
    parser.add_argument("--gate_log_interval", type=int, default=5, help="每隔多少个 epoch 记录一次 gate 统计")
    parser.add_argument("--gate_log_batches", type=int, default=4, help="每次 gate 统计使用多少个验证 batch")
    parser.add_argument("--vis_interval", type=int, default=20, help="每隔多少个 epoch 记录一次验证集预测效果图")
    parser.add_argument("--vis_max_samples", type=int, default=0, help="每次可视化使用的验证样本上限；<= 0 表示展示随机验证工况的全部可预测时刻")
    parser.add_argument("--vis_num_cases", type=int, default=1, help="保留参数；当前每次可视化随机选择 1 个验证工况")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="是否启用 CUDA AMP 混合精度训练")
    parser.add_argument("--compile_model", action=argparse.BooleanOptionalAction, default=False, help="是否使用 torch.compile 编译模型")

    args = parser.parse_args()
    if args.train_part:
        args.train_stage = args.train_part
    args.train_stage = canonicalize_train_stage(args.train_stage)
    part_cfg = build_training_part_config(args)
    args.lr = part_cfg.max_lr

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    print("Using device:", device)

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = args.save_dir
    args.init_ckpt = resolve_init_checkpoint(args.train_stage, args.init_ckpt, base_save_dir)
    args.save_dir = os.path.join(base_save_dir, args.train_stage, run_name)
    ensure_dir(args.save_dir)
    if args.log_dir is not None:
        tb_dir = os.path.join(args.log_dir, args.train_stage, run_name)
    else:
        tb_dir = os.path.join(args.save_dir, "tensorboard")
    ensure_dir(tb_dir)
    writer = SummaryWriter(log_dir=tb_dir)
    print(f"训练阶段: {args.train_stage}")
    if args.init_ckpt is not None:
        print(f"初始化 checkpoint: {args.init_ckpt}")
    print(f"模型和日志将保存到: {args.save_dir}")

    spec, scaler, _, df_val, _, train_ds, val_ds, _ = prepare_datasets_and_scaler(
        feature_dir=args.feature_dir,
        merged_csv_path=args.merged_csv,
        seq_len=args.seq_len,
        pred_horizon=args.pred_horizon,
        pred_seq_len=args.pred_seq_len,
        seed=args.seed,
    )
    save_column_spec_json(spec, os.path.join(args.save_dir, "column_spec.json"))
    scaler.save(os.path.join(args.save_dir, "group_scaler.joblib"))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=graph_temporal_collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
        drop_last=True,
    )
    val_loader = None
    if val_ds is not None and len(df_val) > 0 and len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=graph_temporal_collate_fn,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.num_workers > 0,
            prefetch_factor=4 if args.num_workers > 0 else None,
            drop_last=False,
        )

    group_dims = get_group_dims(spec)
    group_columns = build_model_group_columns(spec)

    # rocker_dims = {name: group_dims.get(name, 0) for name in ROCKER_NAMES}
    # print("Rocker input dims:", rocker_dims)
    # writer.add_text("config/rocker_dims", json.dumps(rocker_dims, ensure_ascii=False, indent=2), 0)

    model = GraphTemporalHGTCompensationModel(
        group_dims=group_dims,
        node_hidden_dim=args.hidden_dim,
        graph_layers=args.graph_layers,
        tcn_hidden_dim=args.tcn_dim,
        lstm_hidden_dim=args.lstm_dim,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        enable_relation_gate=part_cfg.enable_relation_gate,
        enable_edge_gate=part_cfg.enable_edge_gate,
        pred_seq_len=args.pred_seq_len,
        group_columns=group_columns,
        gate_summary_dim=TEACHER_GATE_SUMMARY_DIM,
    ).to(device)

    teacher_gate_net = TeacherGateNet(
        in_dim=TEACHER_GATE_SUMMARY_DIM,
        num_relations=model.num_relations,
        hidden_dim=args.teacher_hidden_dim,
        dropout=args.dropout,
    ).to(device)

    loaded_ckpt = load_init_checkpoint(model, teacher_gate_net, args.init_ckpt, device)
    freeze_info = apply_train_part_freeze_policy(model, teacher_gate_net, part_cfg)
    optimizer_param_groups = build_optimizer_param_groups(
        model,
        teacher_gate_net,
        part_cfg,
        args.weight_decay,
    )
    trainable_params = [param for group in optimizer_param_groups for param in group["params"]]
    if not trainable_params:
        raise RuntimeError(f"{args.train_stage} 没有可训练参数，请检查冻结策略")

    optimizer = torch.optim.AdamW(optimizer_param_groups)
    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    grad_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    history = []
    best_val = float("inf")
    best_epoch = 0
    best_monitor_name = get_best_monitor_name(part_cfg)
    best_path = os.path.join(args.save_dir, "best_model.pt")
    last_val_metrics = None
    last_val_epoch = 0
    trainable_param_count = count_parameters(trainable_params)
    frozen_param_count = count_parameters([p for p in model.parameters() if not p.requires_grad])
    if teacher_gate_net is not None:
        frozen_param_count += count_parameters([p for p in teacher_gate_net.parameters() if not p.requires_grad])

    writer.add_text("config/args", json.dumps(vars(args), ensure_ascii=False, indent=2), 0)
    writer.add_text("config/train_stage", part_cfg.display_name, 0)
    writer.add_text("config/val_gate_mode", get_val_gate_mode(part_cfg), 0)
    writer.add_text("config/best_monitor_name", best_monitor_name, 0)
    writer.add_text("config/student_stage_optimize_task_loss", json.dumps(part_cfg.optimize_task_loss, ensure_ascii=False), 0)
    writer.add_text("config/group_dims", json.dumps(group_dims, ensure_ascii=False, indent=2), 0)
    writer.add_text("config/group_columns", json.dumps(group_columns, ensure_ascii=False, indent=2), 0)
    writer.add_text("config/freeze_info", json.dumps(freeze_info, ensure_ascii=False, indent=2), 0)
    writer.add_scalar("config/trainable_params", trainable_param_count, 0)
    writer.add_scalar("config/frozen_params", frozen_param_count, 0)
    writer.add_scalar("data/train_samples", len(train_ds), 0)
    writer.add_scalar("data/val_samples", len(val_ds) if val_ds is not None else 0, 0)
    print(f"可训练参数量: {trainable_param_count}")
    print(f"冻结参数量: {frozen_param_count}")
    if loaded_ckpt is not None:
        print(f"已加载初始化 checkpoint: {args.init_ckpt}")
    run_dummy_forward_check(
        model=model,
        group_dims=group_dims,
        seq_len=args.seq_len,
        pred_seq_len=args.pred_seq_len,
        device=device,
    )
    relation_names = list(get_model_attr(model, "relation_names", []))
    if args.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")

    if val_loader is not None and loaded_ckpt is not None:
        pre_val_metrics = evaluate(
            model,
            val_loader,
            device,
            spec,
            scaler,
            part_cfg=part_cfg,
            teacher_gate_net=teacher_gate_net,
            lambda_res=args.lambda_res,
            lambda_hf=args.lambda_hf,
            lambda_quat=args.lambda_quat,
            lambda_contact=args.lambda_contact,
            lambda_contact_force=args.lambda_contact_force,
            lambda_contact_moment=args.lambda_contact_moment,
            lambda_contact_raw=args.lambda_contact_raw,
            lambda_contact_physics=args.lambda_contact_physics,
            lambda_kin=args.lambda_kin,
            lambda_smooth=args.lambda_smooth,
            lambda_body_pos=args.lambda_body_pos,
            lambda_wheel_pos_z=args.lambda_wheel_pos_z,
            lambda_body_vel=args.lambda_body_vel,
            body_pos_blend_alpha=args.body_pos_blend_alpha,
            lambda_body_future_kin=args.lambda_body_future_kin,
            raw_monitor_body_weight=args.raw_monitor_body_weight,
            raw_monitor_force_weight=args.raw_monitor_force_weight,
            raw_monitor_moment_weight=args.raw_monitor_moment_weight,
            acc_weight=args.acc_weight,
            dt=args.dt,
            lambda_latent=args.lambda_latent,
            lambda_gate=args.lambda_gate,
            desc="Pre-eval loaded checkpoint",
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        last_val_metrics = pre_val_metrics.copy()
        last_val_epoch = 0
        history.append({"epoch": 0, "phase": "pre_eval", "train": None, "val": pre_val_metrics})
        for name, value in pre_val_metrics.items():
            writer.add_scalar(f"loss/val_{name}", value, 0)
        writer.add_scalars("loss/total_compare", {"val": pre_val_metrics["total"]}, 0)
        writer.add_scalars("metric/wheel_contact_force_rmse_raw", {"val": pre_val_metrics["wheel_contact_force_rmse_raw"]}, 0)
        writer.add_scalars("metric/wheel_contact_moment_rmse_raw", {"val": pre_val_metrics["wheel_contact_moment_rmse_raw"]}, 0)
        writer.add_scalars("metric/body_pos_xyz_rmse_raw", {"val": pre_val_metrics["body_pos_xyz_rmse_raw"]}, 0)
        writer.add_scalars("metric/raw_task_monitor", {"val": pre_val_metrics["raw_task_monitor"]}, 0)
        best_val = pre_val_metrics[best_monitor_name]
        best_epoch = 0
        torch.save(
            build_checkpoint_payload(
                model=model,
                teacher_gate_net=teacher_gate_net,
                group_dims=group_dims,
                args=args,
                part_cfg=part_cfg,
                freeze_info=freeze_info,
                scaler=scaler,
                best_val=best_val,
                best_epoch=best_epoch,
                best_monitor_name=best_monitor_name,
            ),
            best_path,
        )
        writer.add_scalar(f"best/{best_monitor_name}", best_val, 0)
        writer.add_scalar("best/best_epoch", best_epoch, 0)
        print(
            "Pre-eval loaded checkpoint | "
            f"val_total={pre_val_metrics['total']:.6f} | "
            f"val_w_body={pre_val_metrics['weighted_body_pos']:.6f} | "
            f"val_w_wpz={pre_val_metrics['weighted_wheel_pos_z']:.6f} | "
            f"val_w_cF={pre_val_metrics['weighted_contact_force']:.6f} | "
            f"val_w_cM={pre_val_metrics['weighted_contact_moment']:.6f} | "
            f"val_w_cPhy={pre_val_metrics['weighted_contact'] - pre_val_metrics['weighted_contact_scaled']:.6f} | "
            f"val_w_kin={pre_val_metrics['weighted_kin']:.6f} | "
            f"val_w_smooth={pre_val_metrics['weighted_smooth']:.6f}"
        )

    for epoch in range(1, args.epochs + 1):
        lr_map = apply_active_lr_schedule(
            optimizer=optimizer,
            part_cfg=part_cfg,
            epoch_idx=epoch,
            total_epochs=args.epochs,
        )
        model.train()
        set_frozen_modules_eval(model, part_cfg)
        if teacher_gate_net is not None:
            if part_cfg.freeze_teacher_gate:
                teacher_gate_net.eval()
            else:
                teacher_gate_net.train()
        meter = {key: 0.0 for key in BASE_METRIC_KEYS}
        gate_meter = {
            "student_gate_mean": 0.0,
            "teacher_gate_mean": 0.0,
            "student_gate_std": 0.0,
            "teacher_gate_std": 0.0,
        }
        relation_gate_meter = {f"student_{name}": 0.0 for name in relation_names}
        relation_gate_meter.update({f"teacher_{name}": 0.0 for name in relation_names})
        n = 0

        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:03d}/{args.epochs:03d} Train",
            leave=True,
            dynamic_ncols=True,
            disable=not should_enable_tqdm(),
        )

        for batch in train_pbar:
            batch = move_batch_to_device(batch, device)
            # 1. 检查 Dataset / DataLoader 输出的数据
            # check_tensor_dict("batch", batch)

            optimizer.zero_grad(set_to_none=True)

            with get_autocast_context(device, use_amp, amp_dtype):
                output = forward_with_stage_gate(
                    model=model,
                    batch=batch,
                    part_cfg=part_cfg,
                    teacher_gate_net=teacher_gate_net,
                    teacher_requires_grad=not part_cfg.freeze_teacher_gate,
                )
                losses = compute_losses(
                    batch,
                    output,
                    spec,
                    scaler,
                    lambda_res=args.lambda_res,
                    lambda_hf=args.lambda_hf,
                    lambda_quat=args.lambda_quat,
                    lambda_contact=args.lambda_contact,
                    lambda_contact_force=args.lambda_contact_force,
                    lambda_contact_moment=args.lambda_contact_moment,
                    lambda_contact_raw=args.lambda_contact_raw,
                    lambda_contact_physics=args.lambda_contact_physics,
                    lambda_kin=args.lambda_kin,
                    lambda_smooth=args.lambda_smooth,
                    lambda_body_pos=args.lambda_body_pos,
                    lambda_wheel_pos_z=args.lambda_wheel_pos_z,
                    lambda_body_vel=args.lambda_body_vel,
                    body_pos_blend_alpha=args.body_pos_blend_alpha,
                    lambda_body_future_kin=args.lambda_body_future_kin,
                    raw_monitor_body_weight=args.raw_monitor_body_weight,
                    raw_monitor_force_weight=args.raw_monitor_force_weight,
                    raw_monitor_moment_weight=args.raw_monitor_moment_weight,
                    acc_weight=args.acc_weight,
                    dt=args.dt,
                )
                losses = apply_stage_losses(
                    losses=losses,
                    batch=batch,
                    output=output,
                    part_cfg=part_cfg,
                    lambda_latent=args.lambda_latent,
                    lambda_gate=args.lambda_gate,
                )
            grad_scaler.scale(losses["total"]).backward()
            grad_scaler.unscale_(optimizer)
            trainable_model_params = [p for p in model.parameters() if p.requires_grad]
            if trainable_model_params:
                torch.nn.utils.clip_grad_norm_(trainable_model_params, max_norm=5.0)
            if teacher_gate_net is not None and any(p.requires_grad for p in teacher_gate_net.parameters()):
                torch.nn.utils.clip_grad_norm_(teacher_gate_net.parameters(), max_norm=5.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            bs = batch["res_body"].shape[0]
            n += bs

            for k in meter:
                meter[k] += float(losses[k].item()) * bs

            avg_total = meter["total"] / max(n, 1)
            current_lr = max(lr_map.values()) if lr_map else optimizer.param_groups[0]["lr"]

            student_gate = output.get("student_relation_gate")
            teacher_gate = output.get("teacher_relation_gate")
            if student_gate is not None:
                gate_meter["student_gate_mean"] += float(student_gate.mean().item()) * bs
                gate_meter["student_gate_std"] += float(student_gate.std().item()) * bs
                for rel_idx, rel_name in enumerate(relation_names):
                    relation_gate_meter[f"student_{rel_name}"] += float(student_gate[:, :, rel_idx].mean().item()) * bs
            if teacher_gate is not None:
                gate_meter["teacher_gate_mean"] += float(teacher_gate.mean().item()) * bs
                gate_meter["teacher_gate_std"] += float(teacher_gate.std().item()) * bs
                for rel_idx, rel_name in enumerate(relation_names):
                    relation_gate_meter[f"teacher_{rel_name}"] += float(teacher_gate[:, :, rel_idx].mean().item()) * bs

            train_pbar.set_postfix({
                "total": f"{avg_total:.5f}",
                "w_bvel": f"{meter['weighted_body_vel'] / max(n, 1):.5f}",
                "w_wpz": f"{meter['weighted_wheel_pos_z'] / max(n, 1):.5f}",
                "w_cF": f"{meter['weighted_contact_force'] / max(n, 1):.5f}",
                "w_cM": f"{meter['weighted_contact_moment'] / max(n, 1):.5f}",
                "w_cPhy": f"{(meter['weighted_contact'] - meter['weighted_contact_scaled']) / max(n, 1):.5f}",
                "lr": f"{current_lr:.2e}",
            })

        train_metrics = {k: v / max(n, 1) for k, v in meter.items()}
        should_eval = val_loader is not None and args.eval_interval > 0 and epoch % args.eval_interval == 0
        if should_eval:
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                spec,
                scaler,
                part_cfg=part_cfg,
                teacher_gate_net=teacher_gate_net,
                lambda_res=args.lambda_res,
                lambda_hf=args.lambda_hf,
                lambda_quat=args.lambda_quat,
                lambda_contact=args.lambda_contact,
                lambda_contact_force=args.lambda_contact_force,
                lambda_contact_moment=args.lambda_contact_moment,
                lambda_contact_raw=args.lambda_contact_raw,
                lambda_contact_physics=args.lambda_contact_physics,
                lambda_kin=args.lambda_kin,
                lambda_smooth=args.lambda_smooth,
                lambda_body_pos=args.lambda_body_pos,
                lambda_wheel_pos_z=args.lambda_wheel_pos_z,
                lambda_body_vel=args.lambda_body_vel,
                body_pos_blend_alpha=args.body_pos_blend_alpha,
                lambda_body_future_kin=args.lambda_body_future_kin,
                raw_monitor_body_weight=args.raw_monitor_body_weight,
                raw_monitor_force_weight=args.raw_monitor_force_weight,
                raw_monitor_moment_weight=args.raw_monitor_moment_weight,
                acc_weight=args.acc_weight,
                dt=args.dt,
                lambda_latent=args.lambda_latent,
                lambda_gate=args.lambda_gate,
                desc=f"Epoch {epoch:03d}/{args.epochs:03d} Val",
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
            last_val_metrics = val_metrics
            last_val_epoch = epoch
        elif val_loader is not None and last_val_metrics is not None:
            val_metrics = last_val_metrics.copy()
        else:
            val_metrics = train_metrics.copy()

        if should_eval and args.gate_log_interval > 0 and epoch % args.gate_log_interval == 0:
            log_validation_gate_stats(
                writer=writer,
                model=model,
                val_loader=val_loader,
                device=device,
                part_cfg=part_cfg,
                teacher_gate_net=teacher_gate_net,
                epoch=epoch,
                max_batches=args.gate_log_batches,
            )

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        current_lr = max(lr_map.values()) if lr_map else optimizer.param_groups[0]["lr"]
        writer.add_scalar("lr/max", current_lr, epoch)
        for group_name, group_lr in lr_map.items():
            writer.add_scalar(f"lr/{group_name}", group_lr, epoch)
        for name, value in train_metrics.items():
            writer.add_scalar(f"loss/train_{name}", value, epoch)
        for name, value in val_metrics.items():
            writer.add_scalar(f"loss/val_{name}", value, epoch)
        writer.add_scalars(
            "loss_weighted/body_pos_compare",
            {"train": train_metrics["weighted_body_pos"], "val": val_metrics["weighted_body_pos"]},
            epoch,
        )
        writer.add_scalars(
            "loss_weighted/body_vel_compare",
            {"train": train_metrics["weighted_body_vel"], "val": val_metrics["weighted_body_vel"]},
            epoch,
        )
        writer.add_scalars(
            "loss_weighted/wheel_pos_z_compare",
            {"train": train_metrics["weighted_wheel_pos_z"], "val": val_metrics["weighted_wheel_pos_z"]},
            epoch,
        )
        writer.add_scalars(
            "loss_weighted/contact_compare",
            {"train": train_metrics["weighted_contact"], "val": val_metrics["weighted_contact"]},
            epoch,
        )
        writer.add_scalars(
            "loss_weighted/contact_force_compare",
            {"train": train_metrics["weighted_contact_force"], "val": val_metrics["weighted_contact_force"]},
            epoch,
        )
        writer.add_scalars(
            "loss_weighted/contact_moment_compare",
            {"train": train_metrics["weighted_contact_moment"], "val": val_metrics["weighted_contact_moment"]},
            epoch,
        )
        writer.add_scalars(
            "loss_weighted/contact_scaled_compare",
            {"train": train_metrics["weighted_contact_scaled"], "val": val_metrics["weighted_contact_scaled"]},
            epoch,
        )
        writer.add_scalars(
            "loss_weighted/contact_raw_compare",
            {"train": train_metrics["weighted_contact_raw"], "val": val_metrics["weighted_contact_raw"]},
            epoch,
        )
        writer.add_scalars(
            "loss_weighted/kin_compare",
            {"train": train_metrics["weighted_kin"], "val": val_metrics["weighted_kin"]},
            epoch,
        )
        writer.add_scalars(
            "loss_weighted/smooth_compare",
            {"train": train_metrics["weighted_smooth"], "val": val_metrics["weighted_smooth"]},
            epoch,
        )
        writer.add_scalars("loss/total_compare", {"train": train_metrics["total"], "val": val_metrics["total"]}, epoch)
        writer.add_scalars("loss/contact_compare", {"train": train_metrics["contact"], "val": val_metrics["contact"]}, epoch)
        writer.add_scalars("loss/kinematic_compare", {"train": train_metrics["kin"], "val": val_metrics["kin"]}, epoch)
        writer.add_scalars("loss/smooth_compare", {"train": train_metrics["smooth"], "val": val_metrics["smooth"]}, epoch)
        writer.add_scalars("loss/body_pos_xyz_raw_compare", {"train": train_metrics["body_pos_xyz_loss_raw"], "val": val_metrics["body_pos_xyz_loss_raw"]}, epoch)
        writer.add_scalars("loss/body_pos_xyz_scaled_compare", {"train": train_metrics["body_pos_xyz_loss_scaled"], "val": val_metrics["body_pos_xyz_loss_scaled"]}, epoch)
        writer.add_scalars("loss/body_pos_fused_xyz_raw_compare", {"train": train_metrics["body_pos_fused_xyz_loss_raw"], "val": val_metrics["body_pos_fused_xyz_loss_raw"]}, epoch)
        writer.add_scalars("loss/body_pos_fused_xyz_scaled_compare", {"train": train_metrics["body_pos_fused_xyz_loss_scaled"], "val": val_metrics["body_pos_fused_xyz_loss_scaled"]}, epoch)
        writer.add_scalars("loss/body_vel_xyz_raw_compare", {"train": train_metrics["body_vel_xyz_loss_raw"], "val": val_metrics["body_vel_xyz_loss_raw"]}, epoch)
        writer.add_scalars("loss/body_vel_xyz_scaled_compare", {"train": train_metrics["body_vel_xyz_loss_scaled"], "val": val_metrics["body_vel_xyz_loss_scaled"]}, epoch)
        writer.add_scalars("loss/wheel_contact_scaled_compare", {"train": train_metrics["wheel_contact_loss_scaled"], "val": val_metrics["wheel_contact_loss_scaled"]}, epoch)
        writer.add_scalars("loss/wheel_contact_raw_compare", {"train": train_metrics["wheel_contact_loss_raw"], "val": val_metrics["wheel_contact_loss_raw"]}, epoch)
        writer.add_scalars("metric/wheel_contact_rmse_raw", {"train": train_metrics["wheel_contact_rmse_raw"], "val": val_metrics["wheel_contact_rmse_raw"]}, epoch)
        writer.add_scalars("metric/wheel_contact_force_rmse_raw", {"train": train_metrics["wheel_contact_force_rmse_raw"], "val": val_metrics["wheel_contact_force_rmse_raw"]}, epoch)
        writer.add_scalars("metric/wheel_contact_moment_rmse_raw", {"train": train_metrics["wheel_contact_moment_rmse_raw"], "val": val_metrics["wheel_contact_moment_rmse_raw"]}, epoch)
        writer.add_scalars("metric/body_pos_xyz_rmse_raw", {"train": train_metrics["body_pos_xyz_rmse_raw"], "val": val_metrics["body_pos_xyz_rmse_raw"]}, epoch)
        writer.add_scalars("metric/body_pos_fused_xyz_rmse_raw", {"train": train_metrics["body_pos_fused_xyz_rmse_raw"], "val": val_metrics["body_pos_fused_xyz_rmse_raw"]}, epoch)
        writer.add_scalars("metric/body_vel_xyz_rmse_raw", {"train": train_metrics["body_vel_xyz_rmse_raw"], "val": val_metrics["body_vel_xyz_rmse_raw"]}, epoch)
        writer.add_scalars("metric/raw_task_monitor", {"train": train_metrics["raw_task_monitor"], "val": val_metrics["raw_task_monitor"]}, epoch)
        writer.add_scalars("metric/body_pos_x_rmse_raw", {"train": train_metrics["body_pos_x_rmse_raw"], "val": val_metrics["body_pos_x_rmse_raw"]}, epoch)
        writer.add_scalars("metric/body_pos_y_rmse_raw", {"train": train_metrics["body_pos_y_rmse_raw"], "val": val_metrics["body_pos_y_rmse_raw"]}, epoch)
        writer.add_scalars("metric/body_pos_z_rmse_raw", {"train": train_metrics["body_pos_z_rmse_raw"], "val": val_metrics["body_pos_z_rmse_raw"]}, epoch)
        if n > 0:
            writer.add_scalar("gate/student_relation_gate_mean", gate_meter["student_gate_mean"] / n, epoch)
            writer.add_scalar("gate/teacher_relation_gate_mean", gate_meter["teacher_gate_mean"] / n, epoch)
            writer.add_scalar("gate/student_relation_gate_std", gate_meter["student_gate_std"] / n, epoch)
            writer.add_scalar("gate/teacher_relation_gate_std", gate_meter["teacher_gate_std"] / n, epoch)
            for rel_name in relation_names:
                writer.add_scalar(f"gate/student_relation_by_relation/{rel_name}", relation_gate_meter[f"student_{rel_name}"] / n, epoch)
                writer.add_scalar(f"gate/teacher_relation_by_relation/{rel_name}", relation_gate_meter[f"teacher_{rel_name}"] / n, epoch)
        val_mode = get_val_gate_mode(part_cfg)
        writer.add_scalar(
            f"metric/{val_mode}_gate_validation_raw_task_monitor",
            val_metrics["raw_task_monitor"],
            epoch,
        )
        writer.add_scalar(
            f"metric/{val_mode}_gate_validation_body_pos_xyz_rmse_raw",
            val_metrics["body_pos_xyz_rmse_raw"],
            epoch,
        )
        writer.add_scalar(
            f"metric/{val_mode}_gate_validation_contact_force_rmse_raw",
            val_metrics["wheel_contact_force_rmse_raw"],
            epoch,
        )
        writer.add_scalar(
            f"metric/{val_mode}_gate_validation_contact_moment_rmse_raw",
            val_metrics["wheel_contact_moment_rmse_raw"],
            epoch,
        )

        if should_eval:
            summary_parts = [
                f"Epoch {epoch:03d}/{args.epochs}",
                f"train_total={train_metrics['total']:.6f}",
                f"val_total={val_metrics['total']:.6f}",
                f"train_w_bvel={train_metrics['weighted_body_vel']:.6f}",
                f"val_w_bvel={val_metrics['weighted_body_vel']:.6f}",
                f"train_w_wpz={train_metrics['weighted_wheel_pos_z']:.6f}",
                f"val_w_wpz={val_metrics['weighted_wheel_pos_z']:.6f}",
                f"train_w_cF={train_metrics['weighted_contact_force']:.6f}",
                f"val_w_cF={val_metrics['weighted_contact_force']:.6f}",
                # f"train_w_cM={train_metrics['weighted_contact_moment']:.6f}",
                # f"val_w_cM={val_metrics['weighted_contact_moment']:.6f}",
                # f"train_w_cPhy={train_metrics['weighted_contact'] - train_metrics['weighted_contact_scaled']:.6f}",
                # f"val_w_cPhy={val_metrics['weighted_contact'] - val_metrics['weighted_contact_scaled']:.6f}",
            ]
            if part_cfg.key == "student_full":
                summary_parts.extend(
                    [
                        f"train_student_task={train_metrics['student_full_task_loss']:.6f}",
                        f"val_student_task={val_metrics['student_full_task_loss']:.6f}",
                        f"train_distill={train_metrics['student_full_distill_loss']:.6f}",
                        f"val_distill={val_metrics['student_full_distill_loss']:.6f}",
                    ]
                )
            summary_parts.extend(
                [
                    # f"train_w_kin={train_metrics['weighted_kin']:.6f}",
                    # f"val_w_kin={val_metrics['weighted_kin']:.6f}",
                    # f"train_w_smooth={train_metrics['weighted_smooth']:.6f}",
                    # f"val_w_smooth={val_metrics['weighted_smooth']:.6f}",
                    f"lr={current_lr:.2e}",
                ]
            )
            print(" | ".join(summary_parts))
        else:
            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"train_total={train_metrics['total']:.6f} | "
                f"lr={current_lr:.2e} | "
                f"val_skipped(next_eval_every={args.eval_interval}, last_val_epoch={last_val_epoch})"
            )

        if should_eval and val_metrics[best_monitor_name] < best_val:
            best_val = val_metrics[best_monitor_name]
            best_epoch = epoch
            torch.save(
                build_checkpoint_payload(
                    model=model,
                    teacher_gate_net=teacher_gate_net,
                    group_dims=group_dims,
                    args=args,
                    part_cfg=part_cfg,
                    freeze_info=freeze_info,
                    scaler=scaler,
                    best_val=best_val,
                    best_epoch=best_epoch,
                    best_monitor_name=best_monitor_name,
                ),
                best_path,
            )
            writer.add_scalar(f"best/{best_monitor_name}", best_val, epoch)
            writer.add_scalar("best/best_epoch", best_epoch, epoch)

        if should_eval and epoch >= args.min_epochs_before_stop and epoch - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch}, best epoch = {best_epoch}")
            break

    if val_loader is not None and args.vis_interval > 0 and os.path.exists(best_path):
        best_ckpt = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(best_ckpt["model_state_dict"], strict=False)
        log_validation_prediction_figures(
            writer=writer,
            model=model,
            val_loader=val_loader,
            device=device,
            spec=spec,
            scaler=scaler,
            part_cfg=part_cfg,
            teacher_gate_net=teacher_gate_net,
            epoch=best_epoch if best_epoch > 0 else len(history),
            max_samples=args.vis_max_samples,
            max_cases=args.vis_num_cases,
            selected_case=None,
            run_label=f"{args.train_stage}_best",
        )

    writer.flush()
    writer.close()

    with open(os.path.join(args.save_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"训练完成，最佳模型已保存到: {best_path}")
    print(f"TensorBoard 日志已保存到: {tb_dir}")


if __name__ == "__main__":
    main()
