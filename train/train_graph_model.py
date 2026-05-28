from __future__ import annotations

import argparse
import json
import os
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple
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


def inverse_transform_tensor(x: torch.Tensor, scaler, group_name: str) -> torch.Tensor:
    mean, std, mask = _scaler_tensors(scaler, group_name, x.device)
    y = x.clone()
    if y.shape[-1] > 0 and bool(mask.any()):
        y[:, mask] = y[:, mask] * std[mask] + mean[mask]
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
    y = x.clone()
    if y.shape[-1] > 0 and bool(mask.any()):
        y[:, mask] = (y[:, mask] - mean[mask]) / std[mask]
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
    if len(hf_cols) == 0:
        return pred_res_scaled.new_zeros((pred_res_scaled.shape[0], 0))

    pred_res_raw = inverse_transform_tensor(pred_res_scaled, scaler, res_group_name)
    pred_hf_raw = lf_current_raw.clone()

    res_index = {c: j for j, c in enumerate(res_cols)}

    for j, hf_col in enumerate(hf_cols):
        if not hf_col.startswith("hf_"):
            continue
        suffix = hf_col[len("hf_"):]
        res_col = "res_" + suffix
        if res_col in res_index:
            pred_hf_raw[:, j] = lf_current_raw[:, j] + pred_res_raw[:, res_index[res_col]]

    quat_idx = get_quat_indices(hf_cols)
    att_idx = get_att_indices(res_cols)
    if len(quat_idx) == 4 and len(att_idx) == 3:
        lf_quat = lf_current_raw[:, quat_idx]
        pred_rotvec = pred_res_raw[:, att_idx]
        pred_quat = apply_rotvec_to_quat(lf_quat, pred_rotvec, left_multiply=True)
        pred_hf_raw[:, quat_idx] = pred_quat

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
    从高保真状态序列中提取 pos/vel/acc，计算运动学一致性损失。
    pred_hf_seq_scaled: [B, T, D]
    """
    if pred_hf_seq_scaled.ndim != 3:
        return pred_hf_seq_scaled.new_tensor(0.0)

    pos_idx = get_xyz_indices(cols, "pos")
    vel_idx = get_xyz_indices(cols, "vel")
    acc_idx = get_xyz_indices(cols, "acc")

    if len(pos_idx) != 3 or len(vel_idx) != 3 or len(acc_idx) != 3:
        return pred_hf_seq_scaled.new_tensor(0.0)

    pred_hf_seq_raw = inverse_transform_sequence(
        pred_hf_seq_scaled,
        scaler,
        group_name,
    )

    pos = pred_hf_seq_raw[:, :, pos_idx]
    vel = pred_hf_seq_raw[:, :, vel_idx]
    acc = pred_hf_seq_raw[:, :, acc_idx]

    return finite_diff_consistency_loss(pos, vel, acc, dt)


def build_hf_sequence_with_pred_current(
    hf_hist_scaled: torch.Tensor,
    pred_hf_current_scaled: torch.Tensor,
) -> torch.Tensor:
    """
    用前若干个真实高保真历史点 + 当前预测高保真点组成序列。

    hf_hist_scaled: [B, T_hist, D]
    pred_hf_current_scaled: [B, D]
    返回: [B, T_hist + 1, D]
    """
    if hf_hist_scaled.ndim != 3:
        return pred_hf_current_scaled.unsqueeze(1)

    return torch.cat(
        [
            hf_hist_scaled,
            pred_hf_current_scaled.unsqueeze(1),
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

    pred_acc = pred[:, acc_idx]
    target_acc = target[:, acc_idx]
    acc_loss = safe_huber_or_mse(pred_acc, target_acc)

    return base_loss + acc_weight * acc_loss

def compute_losses(
    batch: Dict,
    output: Dict,
    spec,
    scaler,
    lambda_res: float = 1.0,
    lambda_hf: float = 0.5,
    lambda_quat: float = 0.05,
    lambda_contact: float = 0.05,
    lambda_kin: float = 0.01,
    lambda_smooth: float = 0.001,
    acc_weight: float = 2.0,
    dt: float = 0.015,
) -> Dict[str, torch.Tensor]:
    losses: Dict[str, torch.Tensor] = {}

    loss_res = safe_huber_or_mse(output["pred_res_body"], batch["res_body"])
    loss_res_count = 1
    for i in WHEEL_IDS:
        loss_res = loss_res + safe_huber_or_mse(output[f"pred_res_wheel{i}_kin"], batch[f"res_wheel{i}_kin"])
        loss_res = loss_res + safe_huber_or_mse(output[f"pred_res_wheel{i}_contact"], batch[f"res_wheel{i}_contact"])
        loss_res_count += 2
    losses["res"] = loss_res / loss_res_count

    # loss_res = weighted_state_loss(output["pred_res_body"], batch["res_body"], spec.res_groups.body_cols, acc_weight=acc_weight)
    # loss_res_count = 1
    # for i in WHEEL_IDS:
    #     loss_res = loss_res + weighted_state_loss(output[f"pred_res_wheel{i}_kin"], batch[f"res_wheel{i}_kin"], spec.res_groups.wheel_kin_cols[i], acc_weight=acc_weight)
    #     loss_res = loss_res + safe_huber_or_mse(output[f"pred_res_wheel{i}_contact"], batch[f"res_wheel{i}_contact"])
    #     loss_res_count += 2
    # losses["res"] = loss_res / loss_res_count

    pred_hf_body = reconstruct_hf_scaled(
        output["pred_res_body"],
        batch["lf_body_current"],
        spec.res_groups.body_cols,
        spec.target_groups.body_cols,
        "res_body",
        "hf_body",
        scaler,
    )
    loss_hf = safe_huber_or_mse(pred_hf_body, batch["hf_body"])
    # loss_hf = weighted_state_loss(pred_hf_body, batch["hf_body"], spec.target_groups.body_cols, acc_weight=acc_weight)
    loss_hf_count = 1

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
        loss_hf = loss_hf + safe_huber_or_mse(pred_hf_wheel_kin[i], batch[f"hf_wheel{i}_kin"])
        # loss_hf = loss_hf + weighted_state_loss(pred_hf_wheel_kin[i], batch[f"hf_wheel{i}_kin"], spec.target_groups.wheel_kin_cols[i], acc_weight=acc_weight)
        loss_hf = loss_hf + safe_huber_or_mse(pred_hf_wheel_contact[i], batch[f"hf_wheel{i}_contact"])
        loss_hf_count += 2
    losses["hf"] = loss_hf / loss_hf_count

    quat_body_idx = get_quat_indices(spec.target_groups.body_cols)

    loss_quat = pred_hf_body.new_tensor(0.0)
    quat_count = 0

    if len(quat_body_idx) == 4:
        pred_hf_body_raw = inverse_transform_tensor(pred_hf_body, scaler, "hf_body")
        true_hf_body_raw = inverse_transform_tensor(batch["hf_body"], scaler, "hf_body")

        pred_q_body = pred_hf_body_raw[:, quat_body_idx]
        true_q_body = true_hf_body_raw[:, quat_body_idx]

        loss_quat = loss_quat + quat_geodesic_loss(pred_q_body, true_q_body)
        quat_count += 1

    for i in WHEEL_IDS:
        quat_w_idx = get_quat_indices(spec.target_groups.wheel_kin_cols[i])

        if len(quat_w_idx) == 4:
            pred_hf_wheel_kin_raw = inverse_transform_tensor(
                pred_hf_wheel_kin[i],
                scaler,
                f"hf_wheel{i}_kin"
            )
            true_hf_wheel_kin_raw = inverse_transform_tensor(
                batch[f"hf_wheel{i}_kin"],
                scaler,
                f"hf_wheel{i}_kin"
            )

            pred_q_wheel = pred_hf_wheel_kin_raw[:, quat_w_idx]
            true_q_wheel = true_hf_wheel_kin_raw[:, quat_w_idx]

            loss_quat = loss_quat + quat_geodesic_loss(pred_q_wheel, true_q_wheel)
            quat_count += 1

    if quat_count > 0:
        losses["quat"] = loss_quat / quat_count
    else:
        losses["quat"] = pred_hf_body.new_tensor(0.0)

    loss_contact = pred_hf_body.new_tensor(0.0)
    contact_count = 0
    for i in WHEEL_IDS:
        idx = get_contact_indices(spec.target_groups.wheel_contact_cols[i])
        if "Fz" in idx and "sinkage" in idx and pred_hf_wheel_contact[i].shape[-1] > 0:
            contact_raw = inverse_transform_tensor(pred_hf_wheel_contact[i], scaler, f"hf_wheel{i}_contact")
            fz = contact_raw[:, idx["Fz"]]
            sink = contact_raw[:, idx["sinkage"]]
            loss_contact = loss_contact + torch.relu(-sink).mean() + 0.1 * torch.relu(-fz).mean()
            contact_count += 1
    if contact_count > 0:
        loss_contact = loss_contact / contact_count
    losses["contact"] = loss_contact

    loss_kin = pred_hf_body.new_tensor(0.0)
    loss_smooth = pred_hf_body.new_tensor(0.0)

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

    losses["total"] = (
        lambda_res * losses["res"]
        + lambda_hf * losses["hf"]
        + lambda_quat * losses["quat"]
        + lambda_contact * losses["contact"]
        + lambda_kin * losses["kin"]
        + lambda_smooth * losses["smooth"]
    )
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
    epoch: int,
    max_samples: int = 64,
    selected_case: str = None,
) -> None:
    """
    在 TensorBoard 中记录某一个验证集 case 的连续时间预测效果。

    显示内容：
    1. 车身高保真真实值 hf
    2. 低保真输入 lf
    3. 模型补偿后的预测值 pred_hf

    横轴是真实物理时间 time。
    若 selected_case 为 None，则自动选择验证集中第一个样本数足够的 case。
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

    unique_cases = list(dict.fromkeys(case_names.tolist()))

    if selected_case is None:
        selected_case = None
        for c in unique_cases:
            if np.sum(case_names == c) >= 2:
                selected_case = c
                break

    if selected_case is None:
        return

    case_mask = case_names == str(selected_case)
    case_indices = np.where(case_mask)[0]

    if len(case_indices) <= 1:
        return

    case_times = times[case_indices]
    order = np.argsort(case_times)
    case_indices = case_indices[order]
    case_times = case_times[order]

    n = min(max_samples, len(case_indices))
    case_indices = case_indices[:n]
    case_times = case_times[:n]

    index_tensor = torch.as_tensor(case_indices, dtype=torch.long)

    full_batch = {}

    first_batch = all_batches[0]

    for key in first_batch.keys():
        if torch.is_tensor(first_batch[key]):
            full_value = _concat_tensor(key)
            if full_value is not None:
                full_batch[key] = full_value[index_tensor]

    full_batch["case_name"] = case_names[case_indices].tolist()
    full_batch["time"] = torch.as_tensor(case_times, dtype=torch.float32)

    batch = move_batch_to_device(full_batch, device)

    output = model(batch)

    def _to_np(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()

    def _find_col(cols: List[str], suffix: str):
        for idx, c in enumerate(cols):
            if c.endswith(suffix):
                return idx
        return None

    def _plot_group(
        tag_prefix: str,
        pred_hf_scaled: torch.Tensor,
        true_hf_scaled: torch.Tensor,
        lf_current_raw: torch.Tensor,
        hf_cols: List[str],
        hf_group_name: str,
        plot_suffixes: List[str],
    ) -> None:
        pred_raw = inverse_transform_tensor(pred_hf_scaled, scaler, hf_group_name)
        true_raw = inverse_transform_tensor(true_hf_scaled, scaler, hf_group_name)

        x = case_times

        for suffix in plot_suffixes:
            j = _find_col(hf_cols, suffix)
            if j is None:
                continue

            fig = plt.figure(figsize=(10, 4))

            plt.plot(x, _to_np(true_raw[:, j]), label="HF true", linewidth=2.0)
            plt.plot(x, _to_np(pred_raw[:, j]), label="Pred HF", linewidth=1.8)
            plt.plot(x, _to_np(lf_current_raw[:, j]), label="LF current", linewidth=1.5)

            plt.xlabel("time / s")
            plt.ylabel(suffix)
            plt.title(f"{tag_prefix} | {suffix} | {selected_case}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            writer.add_figure(f"val_effect/{selected_case}/{tag_prefix}/{suffix}", fig, epoch)
            plt.close(fig)

    pred_hf_body = reconstruct_hf_scaled(
        output["pred_res_body"],
        batch["lf_body_current"],
        spec.res_groups.body_cols,
        spec.target_groups.body_cols,
        "res_body",
        "hf_body",
        scaler,
    )

    _plot_group(
        tag_prefix="body",
        pred_hf_scaled=pred_hf_body,
        true_hf_scaled=batch["hf_body"],
        lf_current_raw=batch["lf_body_current"],
        hf_cols=spec.target_groups.body_cols,
        hf_group_name="hf_body",
        plot_suffixes=[
            "pos_x", "pos_y", "pos_z",
            "vel_x", "vel_y", "vel_z",
            "acc_x", "acc_y", "acc_z",
        ],
    )

    for i in WHEEL_IDS:
        pred_hf_wheel_kin = reconstruct_hf_scaled(
            output[f"pred_res_wheel{i}_kin"],
            batch[f"lf_wheel{i}_kin_current"],
            spec.res_groups.wheel_kin_cols[i],
            spec.target_groups.wheel_kin_cols[i],
            f"res_wheel{i}_kin",
            f"hf_wheel{i}_kin",
            scaler,
        )

        _plot_group(
            tag_prefix=f"wheel{i}_kin",
            pred_hf_scaled=pred_hf_wheel_kin,
            true_hf_scaled=batch[f"hf_wheel{i}_kin"],
            lf_current_raw=batch[f"lf_wheel{i}_kin_current"],
            hf_cols=spec.target_groups.wheel_kin_cols[i],
            hf_group_name=f"hf_wheel{i}_kin",
            plot_suffixes=[
                "pos_x", "pos_y", "pos_z",
                "vel_x", "vel_y", "vel_z",
                "acc_x", "acc_y", "acc_z",
            ],
        )

        pred_hf_wheel_contact = reconstruct_hf_scaled(
            output[f"pred_res_wheel{i}_contact"],
            batch[f"lf_wheel{i}_contact_current"],
            spec.res_groups.wheel_contact_cols[i],
            spec.target_groups.wheel_contact_cols[i],
            f"res_wheel{i}_contact",
            f"hf_wheel{i}_contact",
            scaler,
        )

        _plot_group(
            tag_prefix=f"wheel{i}_contact",
            pred_hf_scaled=pred_hf_wheel_contact,
            true_hf_scaled=batch[f"hf_wheel{i}_contact"],
            lf_current_raw=batch[f"lf_wheel{i}_contact_current"],
            hf_cols=spec.target_groups.wheel_contact_cols[i],
            hf_group_name=f"hf_wheel{i}_contact",
            plot_suffixes=[
                "Fx", "Fy", "Fz",
                "Mx", "My", "Mz",
                "sinkage",
                "slip_long",
                "slip_lat",
                "in_contact",
            ],
        )

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    spec,
    scaler,
    lambda_res: float = 1.0,
    lambda_hf: float = 0.5,
    lambda_quat: float = 0.05,
    lambda_contact: float = 0.05,
    lambda_kin: float = 0.01,
    lambda_smooth: float = 0.001,
    acc_weight: float = 2.0,
    dt: float = 0.015,
    desc: str = "Val",
):
    model.eval()
    meter = {"total": 0.0, "res": 0.0, "hf": 0.0, "quat": 0.0, "contact": 0.0, "kin": 0.0, "smooth": 0.0}
    n = 0

    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)

    for batch in pbar:
        batch = move_batch_to_device(batch, device)
        output = model(batch)

        losses = compute_losses(
            batch,
            output,
            spec,
            scaler,
            lambda_res=lambda_res,
            lambda_hf=lambda_hf,
            lambda_quat=lambda_quat,
            lambda_contact=lambda_contact,
            lambda_kin=lambda_kin,
            lambda_smooth=lambda_smooth,
            acc_weight=acc_weight,
            dt=dt,
        )

        bs = batch["res_body"].shape[0]
        n += bs

        for k in meter:
            meter[k] += float(losses[k].item()) * bs

        avg_total = meter["total"] / max(n, 1)
        avg_res = meter["res"] / max(n, 1)
        avg_hf = meter["hf"] / max(n, 1)
        avg_quat = meter["quat"] / max(n, 1)
        avg_contact = meter["contact"] / max(n, 1)
        avg_kin = meter["kin"] / max(n, 1)
        avg_smooth = meter["smooth"] / max(n, 1)

        pbar.set_postfix({
            "total": f"{avg_total:.5f}",
            "res": f"{avg_res:.5f}",
            "hf": f"{avg_hf:.5f}",
            # "quat": f"{avg_quat:.5f}",
            # "contact": f"{avg_contact:.5f}",
            # "kin": f"{avg_kin:.5f}",
            # "smooth": f"{avg_smooth:.5f}",
        })

    if n == 0:
        return {k: 0.0 for k in meter}

    return {k: v / n for k, v in meter.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", type=str, default=str(ROOT / "Feature_Selection" / "DataSet"))
    parser.add_argument("--merged_csv", type=str, default=str(ROOT / "Feature_Selection" / "DataSet" / "merged_error_dataset.csv"))
    parser.add_argument("--save_dir", type=str, default=str(ROOT / "results" / "hgt_graph_temporal"))
    parser.add_argument("--log_dir", type=str, default=None, help="TensorBoard 日志目录，默认保存到 save_dir/tensorboard")
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--pred_horizon", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--graph_layers", type=int, default=2)
    parser.add_argument("--tcn_dim", type=int, default=128)
    parser.add_argument("--lstm_dim", type=int, default=128)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lambda_res", type=float, default=1.0)
    parser.add_argument("--lambda_hf", type=float, default=0.6)
    parser.add_argument("--lambda_quat", type=float, default=0.05)
    parser.add_argument("--lambda_contact", type=float, default=0.05)
    parser.add_argument("--lambda_kin", type=float, default=0.01)
    parser.add_argument("--lambda_smooth", type=float, default=0.01)
    parser.add_argument("--acc_weight", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=0.015)
    parser.add_argument("--vis_interval", type=int, default=5, help="每隔多少个 epoch 记录一次验证集预测效果图")
    parser.add_argument("--vis_max_samples", type=int, default=64, help="每次可视化最多使用多少个验证样本")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = args.save_dir
    args.save_dir = os.path.join(base_save_dir, run_name)
    ensure_dir(args.save_dir)
    if args.log_dir is not None:
        tb_dir = os.path.join(args.log_dir, run_name)
    else:
        tb_dir = os.path.join(args.save_dir, "tensorboard")
    ensure_dir(tb_dir)
    writer = SummaryWriter(log_dir=tb_dir)
    print(f"模型和日志将保存到: {args.save_dir}")

    spec, scaler, _, df_val, _, train_ds, val_ds, _ = prepare_datasets_and_scaler(
        feature_dir=args.feature_dir,
        merged_csv_path=args.merged_csv,
        seq_len=args.seq_len,
        pred_horizon=args.pred_horizon,
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
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

    history = []
    best_val = float("inf")
    best_epoch = 0
    best_path = os.path.join(args.save_dir, "best_model.pt")

    writer.add_text("config/args", json.dumps(vars(args), ensure_ascii=False, indent=2), 0)
    writer.add_text("config/group_dims", json.dumps(group_dims, ensure_ascii=False, indent=2), 0)
    writer.add_scalar("data/train_samples", len(train_ds), 0)
    writer.add_scalar("data/val_samples", len(val_ds) if val_ds is not None else 0, 0)

    for epoch in range(1, args.epochs + 1):
        model.train()
        meter = {"total": 0.0, "res": 0.0, "hf": 0.0, "quat": 0.0, "contact": 0.0, "kin": 0.0, "smooth": 0.0}
        n = 0

        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:03d}/{args.epochs:03d} Train",
            leave=True,
            dynamic_ncols=True,
        )

        for batch in train_pbar:
            batch = move_batch_to_device(batch, device)
            # 1. 检查 Dataset / DataLoader 输出的数据
            # check_tensor_dict("batch", batch)

            optimizer.zero_grad(set_to_none=True)

            output = model(batch)
            # 2. 检查模型前向输出
            # check_tensor_dict("output", output)

            losses = compute_losses(
                batch,
                output,
                spec,
                scaler,
                lambda_res=args.lambda_res,
                lambda_hf=args.lambda_hf,
                lambda_quat=args.lambda_quat,
                lambda_contact=args.lambda_contact,
                lambda_kin=args.lambda_kin,
                lambda_smooth=args.lambda_smooth,
                acc_weight=args.acc_weight,
                dt=args.dt,
            )
            # 3. 检查各项 loss
            # check_tensor_dict("losses", losses)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            bs = batch["res_body"].shape[0]
            n += bs

            for k in meter:
                meter[k] += float(losses[k].item()) * bs

            avg_total = meter["total"] / max(n, 1)
            avg_res = meter["res"] / max(n, 1)
            avg_hf = meter["hf"] / max(n, 1)
            avg_quat = meter["quat"] / max(n, 1)
            avg_contact = meter["contact"] / max(n, 1)
            avg_kin = meter["kin"] / max(n, 1)
            avg_smooth = meter["smooth"] / max(n, 1)

            current_lr = optimizer.param_groups[0]["lr"]

            train_pbar.set_postfix({
                "total": f"{avg_total:.5f}",
                "res": f"{avg_res:.5f}",
                "hf": f"{avg_hf:.5f}",
                # "quat": f"{avg_quat:.5f}",
                # "contact": f"{avg_contact:.5f}",
                # "kin": f"{avg_kin:.5f}",
                # "smooth": f"{avg_smooth:.5f}",
                # "lr": f"{current_lr:.2e}",
            })

        train_metrics = {k: v / max(n, 1) for k, v in meter.items()}
        val_metrics = (
            evaluate(
                model,
                val_loader,
                device,
                spec,
                scaler,
                lambda_res=args.lambda_res,
                lambda_hf=args.lambda_hf,
                lambda_quat=args.lambda_quat,
                lambda_contact=args.lambda_contact,
                lambda_kin=args.lambda_kin,
                lambda_smooth=args.lambda_smooth,
                acc_weight=args.acc_weight,
                dt=args.dt,
                desc=f"Epoch {epoch:03d}/{args.epochs:03d} Val",
            )
            if val_loader is not None
            else train_metrics.copy()
        )
        if val_loader is not None and args.vis_interval > 0 and epoch % args.vis_interval == 0:
            log_validation_prediction_figures(
                writer=writer,
                model=model,
                val_loader=val_loader,
                device=device,
                spec=spec,
                scaler=scaler,
                epoch=epoch,
                max_samples=args.vis_max_samples,
            )
        scheduler.step(val_metrics["total"])

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("lr", current_lr, epoch)
        for name, value in train_metrics.items():
            writer.add_scalar(f"loss/train_{name}", value, epoch)
        for name, value in val_metrics.items():
            writer.add_scalar(f"loss/val_{name}", value, epoch)
        writer.add_scalars("loss/total_compare", {"train": train_metrics["total"], "val": val_metrics["total"]}, epoch)
        writer.add_scalars("loss/res_compare", {"train": train_metrics["res"], "val": val_metrics["res"]}, epoch)
        writer.add_scalars("loss/hf_compare", {"train": train_metrics["hf"], "val": val_metrics["hf"]}, epoch)
        writer.add_scalars("loss/quaternion_compare", {"train": train_metrics["quat"], "val": val_metrics["quat"]}, epoch)
        writer.add_scalars("loss/contact_compare", {"train": train_metrics["contact"], "val": val_metrics["contact"]}, epoch)
        writer.add_scalars("loss/kinematic_compare", {"train": train_metrics["kin"], "val": val_metrics["kin"]}, epoch)
        writer.add_scalars("loss/smooth_compare", {"train": train_metrics["smooth"], "val": val_metrics["smooth"]}, epoch)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_total={train_metrics['total']:.6f} | val_total={val_metrics['total']:.6f} | "
            f"res={val_metrics['res']:.6f} | hf={val_metrics['hf']:.6f} | "
            # f"quat={val_metrics['quat']:.6f} | contact={val_metrics['contact']:.6f} | "
            # f"kin={val_metrics['kin']:.6f} | smooth={val_metrics['smooth']:.6f}"
        )

        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "group_dims": group_dims,
                    "args": vars(args),
                    "best_val": best_val,
                    "best_epoch": best_epoch,
                },
                best_path,
            )
            writer.add_scalar("best/val_total", best_val, epoch)
            writer.add_scalar("best/best_epoch", best_epoch, epoch)

        if epoch - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch}, best epoch = {best_epoch}")
            break

    writer.flush()
    writer.close()

    with open(os.path.join(args.save_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"训练完成，最佳模型已保存到: {best_path}")
    print(f"TensorBoard 日志已保存到: {tb_dir}")


if __name__ == "__main__":
    main()