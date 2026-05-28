from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .data_utils import ROCKER_NAMES, WHEEL_IDS
except ImportError:
    from data_utils import ROCKER_NAMES, WHEEL_IDS
except Exception:
    WHEEL_IDS = list(range(6))
    ROCKER_NAMES = ["lf", "lm", "lb", "rf", "rm", "rb"]


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


def build_hgt_vehicle_graph(include_self_loops: bool = True) -> Tuple[Dict[str, int], List[str], torch.Tensor, torch.Tensor, List[str]]:
    """
    构建适配 HGT 的异构车辆图。

    图中显式区分控制上下文、车身、摇臂、车轮运动学、车轮接触和目标误差节点。
    摇臂节点接收 CSV 中的 susp_rocker_lf/lm/lb/rf/rm/rb 状态，仅作为中间传播节点，
    不对应最终残差预测头。
    """
    node_names: List[str] = ["control_context", "body"]
    node_names += list(ROCKER_NAMES)
    node_names += [f"wheel{i}_kin" for i in WHEEL_IDS]
    node_names += [f"wheel{i}_contact" for i in WHEEL_IDS]
    node_names += ["target_error_body"]
    node_names += [f"target_error_wheel{i}_kin" for i in WHEEL_IDS]
    node_names += [f"target_error_wheel{i}_contact" for i in WHEEL_IDS]

    node_map = {name: idx for idx, name in enumerate(node_names)}

    node_types: List[str] = []
    for name in node_names:
        if name == "control_context":
            node_types.append("control_context")
        elif name == "body":
            node_types.append("body")
        elif name in ROCKER_NAMES:
            node_types.append("rocker")
        elif name.startswith("wheel") and name.endswith("_kin"):
            node_types.append("wheel_kin")
        elif name.startswith("wheel") and name.endswith("_contact"):
            node_types.append("wheel_contact")
        elif name.startswith("target_error"):
            node_types.append("target_error")
        else:
            raise ValueError(f"无法识别节点类型: {name}")

    relation_names = [
        "self_loop",
        "control_excitation",
        "kinematic_transfer",
        "motion_to_contact",
        "contact_to_motion",
        "longitudinal_coupling",
        "lateral_coupling",
        "state_to_target",
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

    # 左侧 wheel0、wheel2、wheel4；右侧 wheel1、wheel3、wheel5。
    # 摇臂节点名称直接使用 CSV 中 susp_rocker_ 后面的编码。
    side_specs = [
        ("lf", "lm", "lb", [0, 2, 4]),
        ("rf", "rm", "rb", [1, 3, 5]),
    ]
    for front, rear, secondary, wheels in side_specs:
        if front not in node_map or rear not in node_map or secondary not in node_map:
            continue

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

    add("body", "target_error_body", "state_to_target")
    for i in WHEEL_IDS:
        add(f"wheel{i}_kin", f"target_error_wheel{i}_kin", "state_to_target")
        add(f"wheel{i}_contact", f"target_error_wheel{i}_contact", "state_to_target")

    if include_self_loops:
        for name in node_names:
            add(name, name, "self_loop")

    edge_index = torch.tensor([[s for s, _, _ in edges], [d for _, d, _ in edges]], dtype=torch.long)
    edge_type = torch.tensor([r for _, _, r in edges], dtype=torch.long)
    return node_map, node_types, edge_index, edge_type, relation_names


class HGTLayer(nn.Module):
    """
    轻量 Heterogeneous Graph Transformer 层。

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
        # x: [BT, N, H]
        bt, n, _ = x.shape
        out = x.new_zeros(bt, n, self.num_heads, self.head_dim)
        for type_id, type_name in enumerate(self.node_type_names):
            idx = torch.nonzero(self.node_type_ids == type_id, as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            projected = proj[type_name](x[:, idx, :])
            out[:, idx, :, :] = projected.view(bt, idx.numel(), self.num_heads, self.head_dim)
        return out

    def forward(self, x: torch.Tensor, relation_gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, T, N, H]
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

        q_e = q[:, dst, :, :]  # [BT, E, heads, D]
        k_e = k[:, src, :, :]
        v_e = v[:, src, :, :]

        rel_att = self.rel_att[rel]
        rel_msg = self.rel_msg[rel]
        k_rel = torch.einsum("behd,ehdf->behf", k_e, rel_att)
        v_rel = torch.einsum("behd,ehdf->behf", v_e, rel_msg)

        score = (q_e * k_rel).sum(dim=-1) * self.scale
        score = score * self.rel_prior[rel].unsqueeze(0)

        if relation_gate is not None:
            # relation_gate: [B, T, R]，每个样本和时间步对关系类型给出动态门控
            gate = relation_gate.reshape(bt, self.num_relations).clamp_min(1e-4)
            score = score + torch.log(gate[:, rel]).unsqueeze(-1)

        attn = score.new_zeros(bt, num_edges, self.num_heads)
        for node_idx in range(num_nodes):
            mask = dst == node_idx
            if torch.any(mask):
                attn[:, mask, :] = torch.softmax(score[:, mask, :], dim=1)
        attn = self.dropout(attn)

        msg = v_rel * attn.unsqueeze(-1)
        out = flat.new_zeros(bt, num_nodes, self.num_heads, self.head_dim)
        for edge_pos in range(num_edges):
            out[:, dst[edge_pos], :, :] += msg[:, edge_pos, :, :]
        out = out.reshape(bt, num_nodes, hidden_dim)

        updated = flat.new_empty(bt, num_nodes, hidden_dim)
        for type_id, type_name in enumerate(self.node_type_names):
            idx = torch.nonzero(self.node_type_ids == type_id, as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            y = self.out_proj[type_name](out[:, idx, :])
            y = self.dropout(y)
            updated[:, idx, :] = self.norm[type_name](flat[:, idx, :] + F.gelu(y))

        return updated.reshape(bsz, seq_len, num_nodes, hidden_dim)


class HGTGraphTemporalCompensationModel(nn.Module):
    """
    知识图谱引导的 HGT 图时序误差补偿模型。

    与原 GraphTemporalCompensationModel 保持相同 batch 输入和输出 key，
    因此训练脚本中的 loss 计算逻辑一般可以继续复用。
    """

    def __init__(
        self,
        group_dims: Dict[str, int],
        node_hidden_dim: int = 64,
        graph_layers: int = 2,
        hgt_heads: int = 4,
        tcn_hidden_dim: int = 256,
        lstm_hidden_dim: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        enable_relation_gate: bool = True,
    ):
        super().__init__()
        self.group_dims = group_dims
        self.node_hidden_dim = node_hidden_dim
        self.enable_relation_gate = enable_relation_gate

        self.input_encoders = nn.ModuleDict()
        self.virtual_node_embeds = nn.ParameterDict()

        def add_input_encoder(node_name: str, group_name: str, hidden_width: int) -> None:
            in_dim = int(group_dims.get(group_name, 0))
            if in_dim > 0:
                self.input_encoders[node_name] = MLPEncoder(in_dim, hidden_width, node_hidden_dim, dropout)
            else:
                self.virtual_node_embeds[node_name] = nn.Parameter(torch.zeros(1, 1, node_hidden_dim))

        add_input_encoder("control_context", "system", max(32, node_hidden_dim))
        add_input_encoder("body", "body", max(64, node_hidden_dim))
        for rocker_name in ROCKER_NAMES:
            add_input_encoder(rocker_name, rocker_name, max(32, node_hidden_dim))
        for i in WHEEL_IDS:
            add_input_encoder(f"wheel{i}_kin", f"wheel{i}_kin", max(64, node_hidden_dim))
            add_input_encoder(f"wheel{i}_contact", f"wheel{i}_contact", max(64, node_hidden_dim))

        node_map, node_types, edge_index, edge_type, relation_names = build_hgt_vehicle_graph(include_self_loops=True)
        self.node_map = node_map
        self.node_types = node_types
        self.relation_names = relation_names
        self.num_nodes = len(node_map)
        self.num_relations = len(relation_names)

        unique_node_types = []
        for t in node_types:
            if t not in unique_node_types:
                unique_node_types.append(t)
        self.node_type_names = unique_node_types
        node_type_to_id = {t: i for i, t in enumerate(unique_node_types)}
        node_type_ids = torch.tensor([node_type_to_id[t] for t in node_types], dtype=torch.long)

        self.hgt_layers = nn.ModuleList([
            HGTLayer(
                hidden_dim=node_hidden_dim,
                node_type_names=unique_node_types,
                node_type_ids=node_type_ids,
                edge_index=edge_index,
                edge_type=edge_type,
                num_relations=self.num_relations,
                num_heads=hgt_heads,
                dropout=dropout,
            )
            for _ in range(graph_layers)
        ])

        if enable_relation_gate:
            self.relation_gate_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(node_hidden_dim, node_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(node_hidden_dim, self.num_relations),
                    nn.Sigmoid(),
                )
                for _ in range(graph_layers)
            ])
        else:
            self.relation_gate_layers = nn.ModuleList([nn.Identity() for _ in range(graph_layers)])

        fused_dim = self.num_nodes * node_hidden_dim
        self.tcn = TemporalConvBlock(fused_dim, tcn_hidden_dim, kernel_size=3, dropout=dropout)
        self.bilstm = nn.LSTM(
            input_size=tcn_hidden_dim,
            hidden_size=lstm_hidden_dim // 2,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )

        temporal_dim = lstm_hidden_dim
        head_in_dim = temporal_dim + node_hidden_dim
        self.body_head = ResidualHead(head_in_dim, group_dims["res_body"], hidden_dim=128, dropout=dropout)
        self.wheel_head = ResidualHead(head_in_dim, group_dims["res_wheel0_kin"], hidden_dim=128, dropout=dropout)
        self.contact_head = ResidualHead(head_in_dim, group_dims["res_wheel0_contact"], hidden_dim=128, dropout=dropout)

    def _encode_or_virtual(
        self,
        node_name: str,
        batch_key: Optional[str],
        batch: Dict[str, torch.Tensor],
        ref: torch.Tensor,
    ) -> torch.Tensor:
        if node_name in self.input_encoders:
            if batch_key is None or batch_key not in batch:
                raise KeyError(f"batch 中缺少节点 {node_name} 对应的输入: {batch_key}")
            return self.input_encoders[node_name](batch[batch_key])
        if node_name in self.virtual_node_embeds:
            bsz, seq_len = ref.shape[0], ref.shape[1]
            return self.virtual_node_embeds[node_name].expand(bsz, seq_len, -1)
        # target_error 节点没有原始输入，初始为零向量，通过 state_to_target 接收信息。
        return ref.new_zeros(ref.shape[0], ref.shape[1], self.node_hidden_dim)

    def encode_nodes(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        ref = batch["system"] if "system" in batch else batch["body"]
        encoded_by_name: Dict[str, torch.Tensor] = {}

        encoded_by_name["control_context"] = self._encode_or_virtual("control_context", "system", batch, ref)
        encoded_by_name["body"] = self._encode_or_virtual("body", "body", batch, ref)

        for rocker_name in ROCKER_NAMES:
            encoded_by_name[rocker_name] = self._encode_or_virtual(rocker_name, rocker_name, batch, ref)

        for i in WHEEL_IDS:
            encoded_by_name[f"wheel{i}_kin"] = self._encode_or_virtual(
                f"wheel{i}_kin", f"wheel{i}_kin", batch, ref
            )
            encoded_by_name[f"wheel{i}_contact"] = self._encode_or_virtual(
                f"wheel{i}_contact", f"wheel{i}_contact", batch, ref
            )

        encoded_by_name["target_error_body"] = self._encode_or_virtual(
            "target_error_body", None, batch, ref
        )
        for i in WHEEL_IDS:
            encoded_by_name[f"target_error_wheel{i}_kin"] = self._encode_or_virtual(
                f"target_error_wheel{i}_kin", None, batch, ref
            )
            encoded_by_name[f"target_error_wheel{i}_contact"] = self._encode_or_virtual(
                f"target_error_wheel{i}_contact", None, batch, ref
            )

        node_tensors = [None] * self.num_nodes
        for name, idx in self.node_map.items():
            node_tensors[idx] = encoded_by_name[name]
        x = torch.stack(node_tensors, dim=2)

        control_idx = self.node_map["control_context"]
        for layer_idx, layer in enumerate(self.hgt_layers):
            if self.enable_relation_gate:
                relation_gate = self.relation_gate_layers[layer_idx](x[:, :, control_idx, :])
            else:
                relation_gate = None
            x = layer(x, relation_gate=relation_gate)
        return x

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nodes = self.encode_nodes(batch)
        bsz, seq_len, num_nodes, hidden_dim = nodes.shape

        fused = nodes.reshape(bsz, seq_len, num_nodes * hidden_dim)
        y = self.tcn(fused)
        y, _ = self.bilstm(y)
        h_t = y[:, -1, :]
        node_t = nodes[:, -1, :, :]

        body_feat = torch.cat([h_t, node_t[:, self.node_map["target_error_body"], :]], dim=-1)
        pred_body = self.body_head(body_feat)

        pred_wheel_kin = []
        pred_wheel_contact = []
        for i in WHEEL_IDS:
            kin_idx = self.node_map[f"target_error_wheel{i}_kin"]
            contact_idx = self.node_map[f"target_error_wheel{i}_contact"]
            kin_feat = torch.cat([h_t, node_t[:, kin_idx, :]], dim=-1)
            contact_feat = torch.cat([h_t, node_t[:, contact_idx, :]], dim=-1)
            pred_wheel_kin.append(self.wheel_head(kin_feat))
            pred_wheel_contact.append(self.contact_head(contact_feat))

        return {
            "pred_res_body": pred_body,
            **{f"pred_res_wheel{i}_kin": pred_wheel_kin[i] for i in WHEEL_IDS},
            **{f"pred_res_wheel{i}_contact": pred_wheel_contact[i] for i in WHEEL_IDS},
        }


# 兼容不同命名习惯
GraphTemporalHGTCompensationModel = HGTGraphTemporalCompensationModel


# 以下工具函数保持与原模型文件一致，便于训练脚本从新模型文件中统一导入。
def normalize_quat(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / torch.sqrt((q ** 2).sum(dim=-1, keepdim=True) + eps)


def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def rotvec_to_quat(rotvec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    angle = torch.sqrt((rotvec ** 2).sum(dim=-1, keepdim=True) + eps)
    half_angle = 0.5 * angle
    axis = rotvec / angle
    qw = torch.cos(half_angle)
    qxyz = axis * torch.sin(half_angle)
    return normalize_quat(torch.cat([qw, qxyz], dim=-1))


def apply_rotvec_to_quat(
    lf_quat: torch.Tensor,
    rotvec: torch.Tensor,
    left_multiply: bool = True,
) -> torch.Tensor:
    lf_quat = normalize_quat(lf_quat)
    delta_q = rotvec_to_quat(rotvec)
    if left_multiply:
        pred_q = quat_multiply(delta_q, lf_quat)
    else:
        pred_q = quat_multiply(lf_quat, delta_q)
    return normalize_quat(pred_q)


def huber_or_mse(pred: torch.Tensor, target: torch.Tensor, use_huber: bool = True) -> torch.Tensor:
    return F.huber_loss(pred, target) if use_huber else F.mse_loss(pred, target)


def quat_geodesic_loss(pred_q: torch.Tensor, target_q: torch.Tensor) -> torch.Tensor:
    if pred_q.numel() == 0 or target_q.numel() == 0:
        return pred_q.new_tensor(0.0)
    pred_q = normalize_quat(pred_q)
    target_q = normalize_quat(target_q)
    dot = torch.sum(pred_q * target_q, dim=-1).abs()
    dot = torch.clamp(dot, 0.0, 1.0)
    return torch.mean(1.0 - dot)


def finite_diff_consistency_loss(pos: torch.Tensor, vel: torch.Tensor, acc: torch.Tensor, dt: float) -> torch.Tensor:
    if pos.shape[1] < 2:
        return pos.new_tensor(0.0)
    vel_fd = (pos[:, 1:] - pos[:, :-1]) / dt
    acc_fd = (vel[:, 1:] - vel[:, :-1]) / dt
    return F.mse_loss(vel_fd, vel[:, 1:]) + F.mse_loss(acc_fd, acc[:, 1:])


def smoothness_loss(seq: torch.Tensor) -> torch.Tensor:
    if seq.shape[1] < 2:
        return seq.new_tensor(0.0)
    return torch.mean(torch.abs(seq[:, 1:] - seq[:, :-1]))