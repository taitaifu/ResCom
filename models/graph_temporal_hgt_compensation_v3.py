from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data_utils_v3 import ROCKER_NAMES, WHEEL_IDS


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
    V3 本地 Heterogeneous Graph Transformer 层。

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


def build_hgt_vehicle_graph_v3(include_self_loops: bool = True) -> Tuple[Dict[str, int], List[str], torch.Tensor, torch.Tensor, List[str]]:
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


class HGTGraphTemporalCompensationModelV3(nn.Module):
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
    ):
        super().__init__()
        if role not in {"teacher", "student"}:
            raise ValueError(f"role must be 'teacher' or 'student', got {role!r}")
        self.group_dims = dict(group_dims)
        self.node_hidden_dim = int(node_hidden_dim)
        self.role = role

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

        node_map, node_types, edge_index, edge_type, relation_names = build_hgt_vehicle_graph_v3(True)
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
        wheel_head_in = lstm_hidden_dim + node_hidden_dim * 3 + self.readout_dim
        # Body position channels are interpreted as small integration-error deltas.
        # Rocker and wheel position channels are interpreted as body-frame xyz local deltas.
        self.body_head = ResidualHead(head_in, int(group_dims["res_body"]), head_hidden, dropout)
        self.body_att_head = ResidualHead(head_in, 3, head_hidden, dropout)
        self.rocker_pos_head = ResidualHead(head_in, 3, head_hidden, dropout)
        self.rocker_att_head = ResidualHead(head_in, 3, head_hidden, dropout)
        self.wheel_head = ResidualHead(wheel_head_in, int(group_dims["res_wheel0_kin"]), head_hidden, dropout)
        self.wheel_att_head = ResidualHead(wheel_head_in, 3, head_hidden, dropout)
        self.force_delta_head = ResidualHead(head_in, 3, head_hidden, dropout)
        self.body_pos_gate_head = ResidualHead(head_in, 3, head_hidden, dropout)
        self.body_att_gate_head = ResidualHead(head_in, 3, head_hidden, dropout)
        self.rocker_pos_gate_head = ResidualHead(head_in, 3, head_hidden, dropout)
        self.rocker_att_gate_head = ResidualHead(head_in, 3, head_hidden, dropout)
        self.wheel_kin_gate_head = ResidualHead(wheel_head_in, 4, head_hidden, dropout)
        self.wheel_att_gate_head = ResidualHead(wheel_head_in, 3, head_hidden, dropout)
        self.force_gate_head = ResidualHead(head_in, 3, head_hidden, dropout)

    def _prefixed_key(self, key: str, input_prefix: Optional[str]) -> str:
        if input_prefix and f"{input_prefix}_{key}" in self._active_batch:
            return f"{input_prefix}_{key}"
        return key

    def _encode_or_virtual(self, node_name: str, batch_key: str, batch: Dict[str, torch.Tensor], ref: torch.Tensor, input_prefix: Optional[str] = None) -> torch.Tensor:
        if node_name in self.input_encoders:
            return self.input_encoders[node_name](batch[self._prefixed_key(batch_key, input_prefix)])
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
        self._active_batch = batch
        ref_key = self._prefixed_key("system", input_prefix) if self._prefixed_key("system", input_prefix) in batch else self._prefixed_key("body", input_prefix)
        ref = batch[ref_key]
        tensors = [None] * self.num_nodes
        for name, idx in self.node_map.items():
            key = self.node_to_batch_key[name]
            tensors[idx] = self._encode_or_virtual(name, key, batch, ref, input_prefix)
        x = torch.stack(tensors, dim=2)
        for layer in self.hgt_layers:
            x = layer(x)
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

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        input_prefix: Optional[str] = None,
        current_index: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        z_t, nodes, readout_t, stats = self.forward_features(batch, input_prefix=input_prefix, current_index=current_index)
        target_idx = nodes.shape[1] - 1 if current_index is None else current_index

        def target_feat(node_name: str) -> torch.Tensor:
            node = nodes[:, target_idx:target_idx + 1, self.node_map[node_name], :]
            return torch.cat([z_t, node, readout_t], dim=-1)

        def wheel_feat(wheel_id: int) -> torch.Tensor:
            body = nodes[:, target_idx:target_idx + 1, self.body_idx, :]
            rocker = nodes[:, target_idx:target_idx + 1, self.node_map[self.wheel_to_rocker[wheel_id]], :]
            wheel = nodes[:, target_idx:target_idx + 1, self.node_map[f"wheel{wheel_id}_kin"], :]
            return torch.cat([z_t, body, rocker, wheel, readout_t], dim=-1)

        body_feat = target_feat("body")
        pred_body = self.body_head(body_feat)
        out = {
            "pred_res_body": pred_body,
            "pred_delta_rot_body": self.body_att_head(body_feat),
            "gate_body_pos": torch.sigmoid(self.body_pos_gate_head(body_feat)),
            "gate_body_att": torch.sigmoid(self.body_att_gate_head(body_feat)),
            "z": z_t,
        }
        for name in ROCKER_NAMES:
            feat = target_feat(name)
            out[f"pred_rocker_{name}_local_delta_pos"] = self.rocker_pos_head(feat)
            out[f"pred_rocker_{name}_delta_rot"] = self.rocker_att_head(feat)
            out[f"gate_rocker_{name}_pos"] = torch.sigmoid(self.rocker_pos_gate_head(feat))
            out[f"gate_rocker_{name}_att"] = torch.sigmoid(self.rocker_att_gate_head(feat))
        for i in WHEEL_IDS:
            contact_node = nodes[:, target_idx:target_idx + 1, self.node_map[f"wheel{i}_contact"], :]
            contact_feat = torch.cat([z_t, contact_node, readout_t], dim=-1)
            wf = wheel_feat(i)
            out[f"pred_res_wheel{i}_kin"] = self.wheel_head(wf)
            out[f"pred_wheel{i}_delta_rot"] = self.wheel_att_head(wf)
            out[f"pred_force_delta_wheel{i}"] = self.force_delta_head(contact_feat)
            wheel_gate = torch.sigmoid(self.wheel_kin_gate_head(wf))
            out[f"gate_wheel{i}_pos"] = wheel_gate[..., :3]
            out[f"gate_wheel{i}_omega"] = wheel_gate[..., 3:4]
            out[f"gate_wheel{i}_att"] = torch.sigmoid(self.wheel_att_gate_head(wf))
            out[f"gate_force_wheel{i}"] = torch.sigmoid(self.force_gate_head(contact_feat))
        out.update({k: v for k, v in stats.items() if v is not None})
        return out


GraphTemporalHGTCompensationModelV3 = HGTGraphTemporalCompensationModelV3
