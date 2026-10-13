from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data_utils_v6 import ROCKER_NAMES, WHEEL_IDS
from .config_v6 import V6Config
from .physics_modules_v6 import (ContactEdgeDecoder, DynamicsHead, contact_features, aggregate_contact, refine_hierarchy, world_to_local, so3_log)


class MLPEncoder(nn.Module):
    """把不同输入特征组映射到统一节点隐藏维度。"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_dim = int(in_dim)
        if self.in_dim == 0:
            self.net = None
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == 0:
            raise ValueError(
                "MLPEncoder 收到 0 维输入。请检查该节点对应的输入列是否为空，"
                "或者为该节点提供有效特征。"
            )
        return self.net(x)


class TemporalConvBlock(nn.Module):
    """用于提取局部时序变化的 TCN 模块，输入输出均为 [B, T, D]。"""

    def __init__(self, in_dim: int, hidden_dim: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size - 1
        self.conv1 = nn.Conv1d(in_dim, hidden_dim, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.res_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.res_proj(x)
        y = x.transpose(1, 2)
        y = self.conv1(y)
        y = y[:, :, : x.shape[1]]
        y = F.relu(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = y[:, :, : x.shape[1]]
        y = y.transpose(1, 2)
        y = self.norm(y)
        return F.relu(y + res)


class WheelContactTCNBlock(nn.Module):
    """Small temporal residual block for wheel contact history."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float, causal: bool):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.causal = bool(causal)
        self.conv = nn.Conv1d(channels, channels, self.kernel_size, dilation=self.dilation)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.dilation * (self.kernel_size - 1)
        y = x.transpose(1, 2)
        if self.causal:
            y = F.pad(y, (pad, 0))
        else:
            left = pad // 2
            right = pad - left
            y = F.pad(y, (left, right))
        y = self.conv(y).transpose(1, 2)
        y = self.norm(y)
        y = F.gelu(y)
        y = self.dropout(y)
        return x + y


class ResidualHead(nn.Module):
    """把全局时序特征和局部节点特征映射为目标残差。"""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def gate_to_attention_multiplier(gate: torch.Tensor) -> torch.Tensor:
    # gate 的数值语义保留在 [0, 1]，但 attention 里的中性点应为 0.5。
    return (1.0 + (gate - 0.5)).clamp(1e-4, 1.5)


class HGTLayer(nn.Module):
    """
    V6 本地 Heterogeneous Graph Transformer 层。

    输入 x 的形状为 [B, T, N, H]。
    该层根据节点类型使用不同 Q、K、V 投影，根据关系类型使用不同注意力变换和消息变换。
    """

    def __init__(
        self,
        hidden_dim: int,
        node_type_names: List[str],
        node_type_ids: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        num_relations: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} 必须能被 num_heads={num_heads} 整除")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_nodes = int(node_type_ids.numel())
        self.num_relations = num_relations
        self.scale = self.head_dim ** -0.5

        self.node_type_names = list(node_type_names)
        self.register_buffer("node_type_ids", node_type_ids.long())
        self.register_buffer("edge_index", edge_index.long())
        self.register_buffer("edge_type", edge_type.long())
        self.register_buffer("edge_dst_index", edge_index[1].long())

        self.q_proj = nn.ModuleDict({t: nn.Linear(hidden_dim, hidden_dim) for t in node_type_names})
        self.k_proj = nn.ModuleDict({t: nn.Linear(hidden_dim, hidden_dim) for t in node_type_names})
        self.v_proj = nn.ModuleDict({t: nn.Linear(hidden_dim, hidden_dim) for t in node_type_names})
        self.out_proj = nn.ModuleDict({t: nn.Linear(hidden_dim, hidden_dim) for t in node_type_names})
        self.norm = nn.ModuleDict({t: nn.LayerNorm(hidden_dim) for t in node_type_names})

        self.rel_att = nn.Parameter(torch.empty(num_relations, num_heads, self.head_dim, self.head_dim))
        self.rel_msg = nn.Parameter(torch.empty(num_relations, num_heads, self.head_dim, self.head_dim))
        self.rel_prior = nn.Parameter(torch.ones(num_relations, num_heads))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module_dict in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            for layer in module_dict.values():
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.rel_att)
        nn.init.xavier_uniform_(self.rel_msg)
        nn.init.ones_(self.rel_prior)

    def _project_by_type(self, x: torch.Tensor, proj: nn.ModuleDict) -> torch.Tensor:
        bt, n, _ = x.shape
        out: Optional[torch.Tensor] = None
        for type_id, type_name in enumerate(self.node_type_names):
            idx = torch.nonzero(self.node_type_ids == type_id, as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            projected = proj[type_name](x[:, idx, :])
            if out is None:
                out = torch.zeros(
                    bt,
                    n,
                    self.num_heads,
                    self.head_dim,
                    device=x.device,
                    dtype=projected.dtype,
                )
            out[:, idx, :, :] = projected.view(bt, idx.numel(), self.num_heads, self.head_dim)
        if out is None:
            return x.new_zeros(bt, n, self.num_heads, self.head_dim)
        return out

    def forward(
        self,
        x: torch.Tensor,
        relation_gate: Optional[torch.Tensor] = None,
        edge_gate: Optional[torch.Tensor] = None,
        edge_message: Optional[torch.Tensor] = None,
        physical_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, num_nodes, hidden_dim = x.shape
        if num_nodes != self.num_nodes:
            raise ValueError(f"节点数量不一致，输入为 {num_nodes}，HGTLayer 初始化为 {self.num_nodes}")

        bt = bsz * seq_len
        flat = x.reshape(bt, num_nodes, hidden_dim)

        q = self._project_by_type(flat, self.q_proj)
        k = self._project_by_type(flat, self.k_proj)
        v = self._project_by_type(flat, self.v_proj)

        src = self.edge_index[0]
        dst = self.edge_index[1]
        rel = self.edge_type
        num_edges = int(rel.numel())

        q_e = q[:, dst, :, :]
        k_e = k[:, src, :, :]
        v_e = v[:, src, :, :]

        rel_att = self.rel_att[rel]
        rel_msg = self.rel_msg[rel]
        k_rel = torch.einsum("behd,ehdf->behf", k_e, rel_att)
        v_rel = torch.einsum("behd,ehdf->behf", v_e, rel_msg)

        score = (q_e * k_rel).sum(dim=-1) * self.scale
        score = score * self.rel_prior[rel].unsqueeze(0)
        if physical_bias is not None:
            score = score + physical_bias.reshape(bt, num_edges, self.num_heads)
        if edge_message is not None:
            v_rel = v_rel + edge_message.reshape(bt, num_edges, self.num_heads, self.head_dim)

        if relation_gate is not None:
            gate = relation_gate.reshape(bt, self.num_relations).clamp_min(1e-4)
            score = score + torch.log(gate_to_attention_multiplier(gate[:, rel])).unsqueeze(-1)

        if edge_gate is not None:
            if edge_gate.ndim == 3:
                gate = edge_gate.reshape(bt, num_edges).clamp(1e-4, 1.0)
                score = score + torch.log(gate_to_attention_multiplier(gate)).unsqueeze(-1)
            elif edge_gate.ndim == 4:
                gate = edge_gate.reshape(bt, num_edges, self.num_heads).clamp(1e-4, 1.0)
                score = score + torch.log(gate_to_attention_multiplier(gate))
            else:
                raise ValueError(
                    f"edge_gate 应为 [B, T, E] 或 [B, T, E, heads]，实际 shape={tuple(edge_gate.shape)}"
                )

        dst_index = self.edge_dst_index.view(1, num_edges, 1).expand(bt, -1, self.num_heads)
        score_max = score.new_full((bt, num_nodes, self.num_heads), float("-inf"))
        score_max.scatter_reduce_(1, dst_index, score, reduce="amax", include_self=True)
        score_centered = score - score_max.gather(1, dst_index)
        score_exp = torch.exp(score_centered)
        score_denom = score.new_zeros(bt, num_nodes, self.num_heads)
        score_denom.scatter_add_(1, dst_index, score_exp)
        attn = score_exp / score_denom.gather(1, dst_index).clamp_min(1e-12)
        attn = self.dropout(attn)

        msg = v_rel * attn.unsqueeze(-1)
        out = torch.zeros(
            bt,
            num_nodes,
            self.num_heads,
            self.head_dim,
            device=msg.device,
            dtype=msg.dtype,
        )
        dst_msg_index = self.edge_dst_index.view(1, num_edges, 1, 1).expand(bt, -1, self.num_heads, self.head_dim)
        out.scatter_add_(1, dst_msg_index, msg)
        out = out.reshape(bt, num_nodes, hidden_dim)

        updated: Optional[torch.Tensor] = None
        for type_id, type_name in enumerate(self.node_type_names):
            idx = torch.nonzero(self.node_type_ids == type_id, as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            y = self.out_proj[type_name](out[:, idx, :])
            y = self.dropout(y)
            normalized = self.norm[type_name](flat[:, idx, :] + F.gelu(y))
            if updated is None:
                updated = torch.empty(
                    bt,
                    num_nodes,
                    hidden_dim,
                    device=normalized.device,
                    dtype=normalized.dtype,
                )
            updated[:, idx, :] = normalized

        if updated is None:
            return flat.reshape(bsz, seq_len, num_nodes, hidden_dim)
        return updated.reshape(bsz, seq_len, num_nodes, hidden_dim)


def build_hgt_vehicle_graph_v6(include_self_loops: bool = True) -> Tuple[Dict[str, int], List[str], torch.Tensor, torch.Tensor, List[str]]:
    node_names: List[str] = ["control_context", "body"]
    node_names += list(ROCKER_NAMES)
    node_names += [f"wheel{i}_kin" for i in WHEEL_IDS]
    node_names += [f"wheel{i}_contact" for i in WHEEL_IDS]
    node_map = {name: idx for idx, name in enumerate(node_names)}
    node_types: List[str] = []
    for name in node_names:
        if name == "control_context":
            node_types.append("control_context")
        elif name == "body":
            node_types.append("body")
        elif name in ROCKER_NAMES:
            node_types.append("rocker")
        elif name.endswith("_kin"):
            node_types.append("wheel_kin")
        elif name.endswith("_contact"):
            node_types.append("wheel_contact")
        else:
            raise ValueError(f"unknown node: {name}")

    relation_names = [
        "self_loop",
        "control_excitation",
        "kinematic_transfer",
        "motion_to_contact",
        "contact_to_motion",
        "longitudinal_coupling",
        "lateral_coupling",
    ]
    rel_id = {name: i for i, name in enumerate(relation_names)}
    edges: List[Tuple[int, int, int]] = []

    def add(src: str, dst: str, rel: str) -> None:
        edges.append((node_map[src], node_map[dst], rel_id[rel]))

    def add_ud(a: str, b: str, rel: str) -> None:
        add(a, b, rel)
        add(b, a, rel)

    for rocker_name in ROCKER_NAMES:
        add("control_context", rocker_name, "control_excitation")
    for i in WHEEL_IDS:
        add("control_context", f"wheel{i}_kin", "control_excitation")

    for front, rear, secondary, wheels in [("lf", "lm", "lb", [0, 2, 4]), ("rf", "rm", "rb", [1, 3, 5])]:
        add_ud("body", front, "kinematic_transfer")
        add_ud("body", rear, "kinematic_transfer")
        add_ud(rear, secondary, "kinematic_transfer")
        add_ud(front, f"wheel{wheels[0]}_kin", "kinematic_transfer")
        add_ud(secondary, f"wheel{wheels[1]}_kin", "kinematic_transfer")
        add_ud(secondary, f"wheel{wheels[2]}_kin", "kinematic_transfer")

    for i in WHEEL_IDS:
        add(f"wheel{i}_kin", f"wheel{i}_contact", "motion_to_contact")
        add(f"wheel{i}_contact", f"wheel{i}_kin", "contact_to_motion")
    for a, b in [(0, 2), (2, 4), (1, 3), (3, 5)]:
        add_ud(f"wheel{a}_kin", f"wheel{b}_kin", "longitudinal_coupling")
        add_ud(f"wheel{a}_contact", f"wheel{b}_contact", "longitudinal_coupling")
    for a, b in [(0, 1), (2, 3), (4, 5)]:
        add_ud(f"wheel{a}_kin", f"wheel{b}_kin", "lateral_coupling")
        add_ud(f"wheel{a}_contact", f"wheel{b}_contact", "lateral_coupling")
    if include_self_loops:
        for name in node_names:
            add(name, name, "self_loop")

    edge_index = torch.tensor([[s for s, _, _ in edges], [d for _, d, _ in edges]], dtype=torch.long)
    edge_type = torch.tensor([r for _, _, r in edges], dtype=torch.long)
    return node_map, node_types, edge_index, edge_type, relation_names


class WheelContactHistoryEncoder(nn.Module):
    """Shared compact causal encoder for local tire/terrain history only."""

    def __init__(self, input_dim: int = 7, hidden_dim: int = 48, dropout: float = 0.1, causal: bool = True):
        super().__init__()
        self.input_dim = int(input_dim)
        self.input = nn.Linear(self.input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            WheelContactTCNBlock(hidden_dim, 3, dilation, dropout, causal) for dilation in (1, 2)
        ])

    def forward(self, x: torch.Tensor, current_index: int = -1) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"contact history dim mismatch: got {x.shape[-1]}, expected {self.input_dim}")
        y = self.input(x)
        for block in self.blocks:
            y = block(y)
        index = current_index if current_index >= 0 else y.shape[1] + current_index
        if index < 0 or index >= y.shape[1]:
            raise IndexError(f"current_index={current_index} out of range for contact history length {y.shape[1]}")
        return y[:, index:index + 1]


class HGTGraphTemporalCompensationModelV6(nn.Module):
    def __init__(
        self,
        group_dims: Dict[str, int],
        node_hidden_dim: int = 64,
        graph_layers: int = 3,
        hgt_heads: int = 4,
        tcn_hidden_dim: int = 256,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        role: str = "student",
        body_pos_indices: Optional[List[int]] = None,
        body_vel_indices: Optional[List[int]] = None,
        wheel_pos_indices: Optional[List[int]] = None,
        wheel_omega_index: Optional[int] = None,
        contact_history_dim: int = 7,
        force_hidden_dim: int = 48,
        config: Optional[V6Config] = None,
    ):
        super().__init__()
        if role not in {"teacher", "student"}:
            raise ValueError(f"role must be 'teacher' or 'student', got {role!r}")
        self.config = (config or V6Config()).validate()
        self.group_dims = dict(group_dims)
        self.node_hidden_dim = int(node_hidden_dim)
        self.role = role
        self.debug_rotation_shapes = False
        self.debug_cuda_sync = False
        self._rotation_debug_logged = set()
        self.body_pos_indices = list(body_pos_indices or [0, 1, 2])
        self.body_vel_indices = list(body_vel_indices or [3, 4, 5])
        self.wheel_pos_indices = list(wheel_pos_indices or [0, 1, 2])
        self.wheel_omega_index = int(wheel_omega_index) if wheel_omega_index is not None else None
        self.contact_history_dim = int(contact_history_dim)
        self.force_hidden_dim = int(force_hidden_dim)

        self.input_encoders = nn.ModuleDict()
        self.virtual_node_embeds = nn.ParameterDict()

        def add_encoder(node_name: str, group_name: str, hidden_width: int) -> None:
            in_dim = int(group_dims.get(group_name, 0))
            if in_dim > 0:
                self.input_encoders[node_name] = MLPEncoder(in_dim, hidden_width, node_hidden_dim, dropout)
            else:
                self.virtual_node_embeds[node_name] = nn.Parameter(torch.zeros(1, 1, node_hidden_dim))

        add_encoder("control_context", "system", max(32, node_hidden_dim))
        add_encoder("body", "body", max(64, node_hidden_dim))
        for name in ROCKER_NAMES:
            add_encoder(name, name, max(32, node_hidden_dim))
        for i in WHEEL_IDS:
            add_encoder(f"wheel{i}_kin", f"wheel{i}_kin", max(64, node_hidden_dim))
            add_encoder(f"wheel{i}_contact", f"wheel{i}_contact", max(64, node_hidden_dim))

        node_map, node_types, edge_index, edge_type, relation_names = build_hgt_vehicle_graph_v6(True)
        self.node_map = node_map
        self.node_types = node_types
        self.relation_names = relation_names
        self.num_nodes = len(node_map)
        self.num_relations = len(relation_names)
        self.control_idx = node_map["control_context"]
        self.body_idx = node_map["body"]
        self.rocker_indices = [node_map[name] for name in ROCKER_NAMES]
        self.wheel_kin_indices = [node_map[f"wheel{i}_kin"] for i in WHEEL_IDS]
        self.wheel_contact_indices = [node_map[f"wheel{i}_contact"] for i in WHEEL_IDS]
        self.wheel_to_rocker = {0: "lf", 1: "rf", 2: "lb", 3: "rb", 4: "lb", 5: "rb"}
        self.readout_dim = node_hidden_dim * 5
        self.node_to_batch_key = {"control_context": "system", "body": "body"}
        self.node_to_batch_key.update({name: name for name in ROCKER_NAMES})
        self.node_to_batch_key.update({f"wheel{i}_kin": f"wheel{i}_kin" for i in WHEEL_IDS})
        self.node_to_batch_key.update({f"wheel{i}_contact": f"wheel{i}_contact" for i in WHEEL_IDS})

        unique_node_types: List[str] = []
        for t in node_types:
            if t not in unique_node_types:
                unique_node_types.append(t)
        type_to_id = {t: i for i, t in enumerate(unique_node_types)}
        node_type_ids = torch.tensor([type_to_id[t] for t in node_types], dtype=torch.long)
        self.hgt_layers = nn.ModuleList([
            HGTLayer(node_hidden_dim, unique_node_types, node_type_ids, edge_index, edge_type, self.num_relations, hgt_heads, dropout)
            for _ in range(graph_layers)
        ])

        self.temporal_reduce = nn.Sequential(
            nn.Linear(self.readout_dim, tcn_hidden_dim),
            nn.LayerNorm(tcn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.tcn = TemporalConvBlock(tcn_hidden_dim, tcn_hidden_dim, 3, dropout)
        if self.role == "teacher":
            temporal_hidden = lstm_hidden_dim // 2
            bidirectional = True
        else:
            temporal_hidden = lstm_hidden_dim
            bidirectional = False
        self.temporal_net = nn.LSTM(
            input_size=tcn_hidden_dim,
            hidden_size=temporal_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.z_dim = lstm_hidden_dim
        head_hidden = max(192, node_hidden_dim * 2)
        head_in = lstm_hidden_dim + node_hidden_dim + self.readout_dim
        self.head_in = head_in
        wheel_head_in = lstm_hidden_dim + node_hidden_dim * 3 + self.readout_dim
        self.body_position_correction_head = ResidualHead(head_in, 3, head_hidden, dropout)
        self.body_velocity_head = ResidualHead(head_in, 3, head_hidden, dropout)
        self.body_att_head = ResidualHead(head_in, 3, head_hidden, dropout)
        self.rocker_pos_head = ResidualHead(head_in, 3, head_hidden, dropout)
        self.rocker_att_head = ResidualHead(head_in, 3, head_hidden, dropout)
        self.wheel_local_position_head = ResidualHead(wheel_head_in, 3, head_hidden, dropout)
        self.wheel_omega_head = ResidualHead(wheel_head_in, 1, head_hidden, dropout)
        self.wheel_att_head = ResidualHead(wheel_head_in, 3, head_hidden, dropout)
        self.force_history_encoder = WheelContactHistoryEncoder(
            self.contact_history_dim, self.force_hidden_dim, dropout, causal=self.role == "student"
        )
        force_in = lstm_hidden_dim + node_hidden_dim * 3 + 23 + self.force_hidden_dim
        self.force_residual_head = ResidualHead(force_in, 3, head_hidden, dropout)
        self.body_pos_gate_head = ResidualHead(head_in, 3, head_hidden, dropout)
        self.body_att_gate_head = ResidualHead(head_in, 3, head_hidden, dropout)
        self.rocker_pos_gate_head = ResidualHead(head_in, 3, head_hidden, dropout)
        self.rocker_att_gate_head = ResidualHead(head_in, 3, head_hidden, dropout)
        self.wheel_pos_gate_head = ResidualHead(wheel_head_in, 3, head_hidden, dropout)
        self.wheel_omega_gate_head = ResidualHead(wheel_head_in, 1, head_hidden, dropout)
        self.wheel_att_gate_head = ResidualHead(wheel_head_in, 3, head_hidden, dropout)

        small_dim = lstm_hidden_dim + node_hidden_dim
        self.body_angular_velocity_head = ResidualHead(small_dim,3,head_hidden,dropout)
        self.body_velocity_gate_head = ResidualHead(small_dim,3,head_hidden,dropout)
        self.body_angular_gate_head = ResidualHead(small_dim,3,head_hidden,dropout)
        self.wheel_spin_head = ResidualHead(small_dim,1,head_hidden,dropout)
        self.part_velocity_head = ResidualHead(small_dim,3,head_hidden,dropout)
        self.part_angular_head = ResidualHead(small_dim,3,head_hidden,dropout)
        nn.init.zeros_(self.body_velocity_gate_head.net[-1].weight)
        nn.init.constant_(self.body_velocity_gate_head.net[-1].bias,6.)
        nn.init.zeros_(self.body_angular_gate_head.net[-1].weight)
        nn.init.zeros_(self.body_angular_gate_head.net[-1].bias)
        for module in (self.body_angular_velocity_head,self.wheel_spin_head,self.part_velocity_head,self.part_angular_head):
            nn.init.zeros_(module.net[-1].weight); nn.init.zeros_(module.net[-1].bias)
        if self.config.enable_physics_edge:
            self.physics_edge_mlps = nn.ModuleList([nn.Sequential(nn.Linear(32,node_hidden_dim),nn.GELU(),nn.Linear(node_hidden_dim,node_hidden_dim)) for _ in range(graph_layers)])
            for m in self.physics_edge_mlps:
                nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias)
        if self.config.enable_physics_attention:
            self.physics_attention_mlps = nn.ModuleList([nn.Sequential(nn.Linear(32,32),nn.GELU(),nn.Linear(32,hgt_heads)) for _ in range(graph_layers)])
            for m in self.physics_attention_mlps:
                nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias)
        if self.config.enable_contact_history:
            self.enhanced_history = EnhancedContactHistory(15,force_hidden_dim,self.config.history_len+1,dropout)
        edge_dim = lstm_hidden_dim+2*node_hidden_dim+15+force_hidden_dim
        if self.config.enable_contact_edge_decoder:
            self.contact_edge_decoder = ContactEdgeDecoder(edge_dim,head_hidden)
        if self.config.enable_force_gate:
            self.force_gate_head = nn.Sequential(nn.Linear(edge_dim,force_hidden_dim),nn.GELU(),nn.Linear(force_hidden_dim,3))
        if self.config.enable_dynamics_update:
            self.dynamics_head = DynamicsHead(small_dim+21,head_hidden)

    def configure_rotation_debug(self, enabled: bool = False, cuda_sync: bool = False) -> None:
        self.debug_rotation_shapes = bool(enabled)
        self.debug_cuda_sync = bool(cuda_sync)
        self._rotation_debug_logged = set()

    def _check_rocker_rotation_tensor(self, name: str, value: torch.Tensor, expected_shape: Tuple[int, ...], sync: bool = False) -> None:
        if tuple(value.shape) != expected_shape:
            raise RuntimeError(
                f"{name} shape mismatch: got {tuple(value.shape)}, expected {expected_shape}; "
                "single-step rocker rotation tensors must not broadcast over batch"
            )
        if not value.is_floating_point():
            raise TypeError(f"{name} must be floating point, got {value.dtype}")
        if sync and self.debug_cuda_sync and value.is_cuda:
            torch.cuda.synchronize(value.device)
        if self.debug_rotation_shapes:
            finite = bool(torch.isfinite(value.detach()).all().item())
            if name not in self._rotation_debug_logged:
                print(
                    f"[V6][rotation] {name}: shape={tuple(value.shape)} dtype={value.dtype} "
                    f"device={value.device} isfinite={finite}",
                    flush=True,
                )
                self._rotation_debug_logged.add(name)
            if not finite:
                raise FloatingPointError(f"Non-finite tensor in {name}")

    def _prefixed_key(self, key: str, input_prefix: Optional[str], batch: Dict[str, torch.Tensor]) -> str:
        prefixed = f"{input_prefix}_{key}" if input_prefix else key
        if prefixed in batch:
            return prefixed
        return key

    def _encode_or_virtual(self, node_name: str, batch_key: str, batch: Dict[str, torch.Tensor], ref: torch.Tensor, input_prefix: Optional[str] = None) -> torch.Tensor:
        if node_name in self.input_encoders:
            return self.input_encoders[node_name](batch[self._prefixed_key(batch_key, input_prefix, batch)])
        return self.virtual_node_embeds[node_name].expand(ref.shape[0], ref.shape[1], -1)

    def _mean_nodes(self, x: torch.Tensor, indices: List[int]) -> torch.Tensor:
        idx = torch.as_tensor(indices, device=x.device, dtype=torch.long)
        return x[:, :, idx, :].mean(dim=2)

    def _five_group_readout(self, x: torch.Tensor) -> torch.Tensor:
        control = x[:, :, self.control_idx, :]
        body = x[:, :, self.body_idx, :]
        rocker = self._mean_nodes(x, self.rocker_indices)
        wheel_kin = self._mean_nodes(x, self.wheel_kin_indices)
        wheel_contact = self._mean_nodes(x, self.wheel_contact_indices)
        return torch.cat([control, body, rocker, wheel_kin, wheel_contact], dim=-1)

    def encode_nodes(self, batch: Dict[str, torch.Tensor], input_prefix: Optional[str] = None):
        system_key = self._prefixed_key("system", input_prefix, batch)
        ref_key = system_key if system_key in batch else self._prefixed_key("body", input_prefix, batch)
        ref = batch[ref_key]
        tensors = [None] * self.num_nodes
        for name, idx in self.node_map.items():
            key = self.node_to_batch_key[name]
            tensors[idx] = self._encode_or_virtual(name, key, batch, ref, input_prefix)
        x = torch.stack(tensors, dim=2)
        edge_features = None
        if self.config.enable_physics_edge or self.config.enable_physics_attention:
            key = self._prefixed_key('physics_nodes', input_prefix, batch)
            physical = batch[key]
            if physical.shape[:3] != x.shape[:3] or physical.shape[-1] != 32:
                raise ValueError('physics_nodes must match [B,T,N,32]')
            src, dst = self.hgt_layers[0].edge_index
            edge_features = physical[:,:,dst]-physical[:,:,src]
            edge_features = edge_features.clone()
            edge_features[...,18:] = physical[:,:,dst,18:]
            rs = physical[:,:,src,6:15].reshape(*physical.shape[:2],-1,3,3)
            rd = physical[:,:,dst,6:15].reshape(*physical.shape[:2],-1,3,3)
            edge_features[...,6:9] = so3_log(rs.transpose(-1,-2)@rd)
        for index, layer in enumerate(self.hgt_layers):
            msg = self.physics_edge_mlps[index](edge_features) if self.config.enable_physics_edge else None
            bias = self.physics_attention_mlps[index](edge_features) if self.config.enable_physics_attention else None
            x = layer(x, edge_message=msg, physical_bias=bias)
        return x, {}

    def forward_features(
        self,
        batch: Dict[str, torch.Tensor],
        input_prefix: Optional[str] = None,
        current_index: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        nodes, stats = self.encode_nodes(batch, input_prefix=input_prefix)
        readout_seq = self._five_group_readout(nodes)
        y = self.temporal_reduce(readout_seq)
        y = self.tcn(y)
        y, _ = self.temporal_net(y)
        if current_index is None:
            current_index = y.shape[1] - 1
        if current_index < 0 or current_index >= y.shape[1]:
            raise IndexError(f"current_index={current_index} out of range for sequence length {y.shape[1]}")
        z_t = y[:, current_index:current_index + 1, :]
        readout_t = readout_seq[:, current_index:current_index + 1, :]
        return z_t, nodes, readout_t, stats

    def forward_state(
        self, batch: Dict[str, torch.Tensor], input_prefix: Optional[str] = None,
        current_index: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        z_t, nodes, readout_t, stats = self.forward_features(batch, input_prefix, current_index)
        target_idx = nodes.shape[1] - 1 if current_index is None else current_index

        def node_feat(name: str) -> torch.Tensor:
            node = nodes[:, target_idx:target_idx + 1, self.node_map[name], :]
            return torch.cat([z_t, node, readout_t], dim=-1)

        def wheel_feat(wheel_id: int) -> torch.Tensor:
            body = nodes[:, target_idx:target_idx + 1, self.body_idx, :]
            rocker = nodes[:, target_idx:target_idx + 1, self.node_map[self.wheel_to_rocker[wheel_id]], :]
            wheel = nodes[:, target_idx:target_idx + 1, self.node_map[f"wheel{wheel_id}_kin"], :]
            return torch.cat([z_t, body, rocker, wheel, readout_t], dim=-1)

        body_feat = node_feat("body")
        body_width = int(self.group_dims["res_body"])
        body_delta = body_feat.new_zeros(*body_feat.shape[:-1], body_width)
        body_delta = body_delta.index_copy(-1, torch.as_tensor(self.body_pos_indices, device=body_feat.device), self.body_position_correction_head(body_feat))
        body_delta = body_delta.index_copy(-1, torch.as_tensor(self.body_vel_indices, device=body_feat.device), self.body_velocity_head(body_feat))
        out: Dict[str, torch.Tensor] = {
            "pred_res_body": body_delta,
            "pred_delta_rot_body": self.body_att_head(body_feat),
            "gate_body_pos": torch.sigmoid(self.body_pos_gate_head(body_feat)),
            "gate_body_att": torch.sigmoid(self.body_att_gate_head(body_feat)),
            "z": z_t,
            "body_node": nodes[:, target_idx:target_idx + 1, self.body_idx, :],
            "wheel_kin_nodes": {
                i: nodes[:, target_idx:target_idx + 1, self.node_map[f"wheel{i}_kin"], :] for i in WHEEL_IDS
            },
            "wheel_contact_nodes": {
                i: nodes[:, target_idx:target_idx + 1, self.node_map[f"wheel{i}_contact"], :] for i in WHEEL_IDS
            },
        }
        for name in ROCKER_NAMES:
            feat = node_feat(name)
            batch_size = int(feat.shape[0])
            self._check_rocker_rotation_tensor(
                f"rocker_{name}.head_input", feat, (batch_size, 1, self.head_in), sync=True
            )
            out[f"pred_rocker_{name}_local_delta_pos"] = self.rocker_pos_head(feat)
            pred_rot = self.rocker_att_head(feat)
            self._check_rocker_rotation_tensor(
                f"rocker_{name}.pred_delta_rot", pred_rot, (batch_size, 1, 3), sync=True
            )
            gate_att_logits = self.rocker_att_gate_head(feat)
            self._check_rocker_rotation_tensor(
                f"rocker_{name}.gate_att_logits", gate_att_logits, (batch_size, 1, 3), sync=True
            )
            out[f"pred_rocker_{name}_delta_rot"] = pred_rot
            out[f"gate_rocker_{name}_pos"] = torch.sigmoid(self.rocker_pos_gate_head(feat))
            out[f"gate_rocker_{name}_att"] = torch.sigmoid(gate_att_logits)
        for i in WHEEL_IDS:
            wf = wheel_feat(i)
            wheel_width = int(self.group_dims[f"res_wheel{i}_kin"])
            wheel_delta = wf.new_zeros(*wf.shape[:-1], wheel_width)
            wheel_delta = wheel_delta.index_copy(-1, torch.as_tensor(self.wheel_pos_indices, device=wf.device), self.wheel_local_position_head(wf))
            if self.wheel_omega_index is not None:
                if self.wheel_omega_index < 0 or self.wheel_omega_index >= wheel_width:
                    raise IndexError(
                        f"wheel_omega_index={self.wheel_omega_index} out of range for wheel residual width {wheel_width}"
                    )
                wheel_delta = wheel_delta.index_copy(-1, torch.as_tensor([self.wheel_omega_index], device=wf.device), self.wheel_omega_head(wf))
            out[f"pred_res_wheel{i}_kin"] = wheel_delta
            out[f"pred_wheel{i}_delta_rot"] = self.wheel_att_head(wf)
            out[f"gate_wheel{i}_pos"] = torch.sigmoid(self.wheel_pos_gate_head(wf))
            out[f"gate_wheel{i}_omega"] = torch.sigmoid(self.wheel_omega_gate_head(wf))
            out[f"gate_wheel{i}_att"] = torch.sigmoid(self.wheel_att_gate_head(wf))
        small_body_feat = torch.cat([z_t, out['body_node']],-1)
        out['delta_body_angular_velocity'] = self.body_angular_velocity_head(small_body_feat)
        out['gate_body_velocity'] = torch.sigmoid(self.body_velocity_gate_head(small_body_feat))
        out['gate_body_angular_velocity'] = torch.sigmoid(self.body_angular_gate_head(small_body_feat))
        for i in WHEEL_IDS:
            feat = torch.cat([z_t,out['wheel_kin_nodes'][i]],-1)
            out[f'delta_wheel{i}_omega_raw'] = self.wheel_spin_head(feat)
            out[f'delta_wheel{i}_velocity'] = self.part_velocity_head(feat)
            out[f'delta_wheel{i}_angular_velocity'] = self.part_angular_head(feat)
        for name in ROCKER_NAMES:
            feat = torch.cat([z_t,nodes[:,target_idx:target_idx+1,self.node_map[name]]],-1)
            out[f'delta_rocker_{name}_velocity'] = self.part_velocity_head(feat)
            out[f'delta_rocker_{name}_angular_velocity'] = self.part_angular_head(feat)
        out.update({k: v for k, v in stats.items() if v is not None})
        return out

    def forward_force_v5_path(
        self, corrected_state: Dict[str, object], batch: Dict[str, torch.Tensor],
        input_prefix: Optional[str] = None, current_index: int = -1,
    ) -> Dict[str, torch.Tensor]:
        required = ("z", "body_node", "wheel_kin_nodes", "wheel_contact_nodes", "body_velocity", "body_acceleration", "body_rotation", "wheel_position", "wheel_omega", "force_base", "fphy_z")
        missing = [key for key in required if key not in corrected_state]
        if missing:
            raise KeyError(f"corrected_state missing force-stage fields: {missing}")
        z_t = corrected_state["z"]
        wheel_kin_nodes = corrected_state["wheel_kin_nodes"]
        wheel_contact_nodes = corrected_state["wheel_contact_nodes"]
        body_node = corrected_state["body_node"]
        out: Dict[str, torch.Tensor] = {}
        for i in WHEEL_IDS:
            key = f"force_history_wheel{i}"
            if input_prefix and f"{input_prefix}_{key}" in batch:
                key = f"{input_prefix}_{key}"
            if key not in batch:
                raise KeyError(f"V6 force stage requires batch[{key!r}]")
            history = batch[key]
            if self.config.enable_contact_history:
                history_feature = self.enhanced_history(torch.cat([history,batch[f'shear_history_wheel{i}']],-1),current_index)
            else:
                history_feature = self.force_history_encoder(history, current_index)
            wheel_pos = corrected_state["wheel_position"][i]
            wheel_omega = corrected_state["wheel_omega"][i]
            force_base = corrected_state["force_base"][i]
            fphy_z = corrected_state["fphy_z"][i]
            features = torch.cat([
                z_t, body_node, wheel_kin_nodes[i], wheel_contact_nodes[i], history_feature,
                corrected_state["body_velocity"], corrected_state["body_acceleration"],
                corrected_state["body_rotation"].reshape(*z_t.shape[:-1], 9),
                wheel_pos, wheel_omega, force_base, fphy_z,
            ], dim=-1)
            delta_force = self.force_residual_head(features)
            out[f"force_latent_wheel{i}"] = features
            out[f"delta_force_wheel{i}"] = delta_force
            out[f"force_pred_wheel{i}"] = force_base + delta_force
        out["force_latent"] = torch.cat([out[f"force_latent_wheel{i}"] for i in WHEEL_IDS], dim=1)
        return out

    def forward_force(self, corrected_state, batch, input_prefix=None, current_index=-1):
        cfg = self.config
        edge, frames, points, active = contact_features(corrected_state,batch,cfg)
        if not cfg.enable_contact_edge_decoder:
            out = self.forward_force_v5_path(corrected_state,batch,input_prefix,current_index)
        else:
            out = {}
        for i in WHEEL_IDS:
            history = batch[f'force_history_wheel{i}']
            if cfg.enable_contact_edge_decoder or cfg.enable_force_gate:
                if cfg.enable_contact_history:
                    h = self.enhanced_history(torch.cat([history,batch[f'shear_history_wheel{i}']],-1),current_index)
                else:
                    h = self.force_history_encoder(history,current_index)
                e = edge[i].clone()
                e = torch.cat([e[...,:3]/cfg.position_scale,e[...,3:6]/cfg.velocity_scale,
                    e[...,6:9],e[...,9:10]/cfg.wheel_radius,e[...,10:12]/cfg.velocity_scale,
                    e[...,12:13]/cfg.omega_scale,e[...,13:]],-1)
                features = torch.cat([corrected_state['z'],corrected_state['wheel_kin_nodes'][i],
                    corrected_state['wheel_contact_nodes'][i],e,h],-1)
                if cfg.enable_contact_edge_decoder:
                    out[f'delta_force_wheel{i}'] = self.contact_edge_decoder(features)*cfg.force_scale
                    out[f'force_latent_wheel{i}'] = features
                gate = torch.sigmoid(self.force_gate_head(features)) if cfg.enable_force_gate else torch.ones_like(out[f'delta_force_wheel{i}'])
            else:
                gate = torch.ones_like(out[f'delta_force_wheel{i}'])
            raw = out[f'delta_force_wheel{i}']
            pred = corrected_state['force_base'][i]+gate*raw
            # Exact zero out of contact; projection only applies to normal force.
            pred = torch.cat([pred[...,:2],pred[...,2:3].clamp_min(0)],-1)*active[i]
            out[f'gate_force_wheel{i}'] = gate
            out[f'effective_force_wheel{i}'] = gate*raw*active[i]
            out[f'force_pred_wheel{i}'] = pred
            out[f'contact_frame_wheel{i}'] = frames[i]
            out[f'contact_point_wheel{i}'] = points[i]
            out[f'contact_active_wheel{i}'] = active[i]
        out['force_latent'] = torch.cat([out[f'force_latent_wheel{i}'] for i in WHEEL_IDS],1)
        return out

    def coupled_update(self, state, batch, current_index=-1):
        initial = {k:v for k,v in state.items() if k!='initial_state'}
        force = self.forward_force(state,batch,current_index=current_index)
        if not self.config.enable_force_state_coupling:
            return state,force
        cfg = self.config
        for iteration in range(cfg.physics_iterations):
            frames = torch.stack([force[f'contact_frame_wheel{i}'] for i in WHEEL_IDS],-3)
            points = torch.stack([force[f'contact_point_wheel{i}'] for i in WHEEL_IDS],-2)
            predicted = torch.stack([force[f'force_pred_wheel{i}'] for i in WHEEL_IDS],-2)
            lf = torch.stack([state['lf_force'][i]*force[f'contact_active_wheel{i}'] for i in WHEEL_IDS],-2)
            fp,tp,_ = aggregate_contact(predicted,frames,points,state['body_position'])
            fl,tl,_ = aggregate_contact(lf,frames,points,state['body_position'])
            feedback = torch.cat([(fp-fl)/cfg.force_scale,(tp-tl)/cfg.torque_scale,
                state['body_velocity']/cfg.velocity_scale,state['body_angular_velocity']/cfg.angular_velocity_scale,
                state['body_rotation'].flatten(-2)],-1)
            force['contact_resultant_world'] = fp
            force['contact_torque_world'] = tp
            force['contact_resultant_delta_world'] = fp-fl
            force['contact_torque_delta_world'] = tp-tl
            if not cfg.enable_dynamics_update: break
            correction,raw,gate = self.dynamics_head(torch.cat([state['z'],state['body_node'],feedback],-1))
            state = refine_hierarchy(state,correction,cfg)
            force['delta_state_dyn_raw'],force['gate_dyn'] = raw,gate
            if cfg.enable_physics_refinement:
                if 'dt' in state:
                    from .load_transfer_force_v6 import load_transfer_fz_torch
                    state['body_acceleration']=(state['body_velocity']-state['previous_body_velocity'])/state['dt']
                    for i in WHEEL_IDS:
                        state['wheel_vz'][i]=(state['wheel_position'][i][...,2:3]-state['previous_wheel_position'][i][...,2:3])/state['dt']
                        fz=load_transfer_fz_torch(state['body_roll'],state['body_pitch'],state['body_acceleration'],state['wheel_vz'][i],i,state['load_transfer_params'])
                        state['fphy_z'][i]=fz
                        state['force_base'][i]=torch.cat([state['lf_force'][i][...,:2],fz],-1)
                state['recalculate_contact'] = True
                aux = {k:v for k,v in force.items() if k.startswith('contact_resultant') or k.startswith('contact_torque') or k in ('delta_state_dyn_raw','gate_dyn')}
                force = self.forward_force(state,batch,current_index=current_index)
                force.update(aux)
        state['initial_state'] = initial
        return state,force

    def forward(
        self, batch: Dict[str, torch.Tensor], input_prefix: Optional[str] = None,
        current_index: Optional[int] = None, mode: str = "state",
        corrected_state: Optional[Dict[str, object]] = None,
    ) -> Dict[str, torch.Tensor]:
        if mode == "state":
            return self.forward_state(batch, input_prefix=input_prefix, current_index=current_index)
        if mode == "full":
            if corrected_state is None:
                raise ValueError("mode='full' requires an explicitly reconstructed corrected_state")
            state_out = self.forward_state(batch, input_prefix=input_prefix, current_index=current_index)
            force_index = -1 if current_index is None else current_index
            return {**state_out, **self.forward_force(corrected_state, batch, input_prefix, force_index)}
        raise ValueError(f"mode must be 'state' or 'full', got {mode!r}")


GraphTemporalHGTCompensationModelV6 = HGTGraphTemporalCompensationModelV6


class EnhancedContactHistory(nn.Module):
    def __init__(self,input_dim,hidden_dim,window,dropout):
        super().__init__()
        self.input = nn.Linear(input_dim,hidden_dim)
        dilations=[]; receptive=1; d=1
        while receptive < window:
            dilations.append(d); receptive += 2*d; d*=2
        self.receptive_field = receptive
        self.blocks = nn.ModuleList([WheelContactTCNBlock(hidden_dim,3,d,dropout,True) for d in dilations])
    def forward(self,x,current_index=-1):
        y=self.input(x)
        for b in self.blocks: y=b(y)
        i=current_index if current_index>=0 else y.shape[1]+current_index
        return y[:,i:i+1]
