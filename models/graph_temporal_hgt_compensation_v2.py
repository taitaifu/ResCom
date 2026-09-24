from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .data_utils_v2 import ROCKER_NAMES, WHEEL_IDS
from .graph_temporal_hgt_compensation import (
    EdgeGateMLP,
    EdgeSpec,
    HGTLayer,
    MLPEncoder,
    RelationGateMLP,
    ResidualHead,
    StudentGateSummaryNet,
    TeacherGateNet,
    TemporalConvBlock,
)


def build_hgt_vehicle_graph_v2(include_self_loops: bool = True) -> Tuple[Dict[str, int], List[str], torch.Tensor, torch.Tensor, List[str], List[EdgeSpec]]:
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
    specs: List[EdgeSpec] = []

    def add(src: str, dst: str, rel: str) -> None:
        edges.append((node_map[src], node_map[dst], rel_id[rel]))
        specs.append(EdgeSpec(src, dst, rel, rel_id[rel]))

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
    return node_map, node_types, edge_index, edge_type, relation_names, specs


class HGTGraphTemporalCompensationModelV2(nn.Module):
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
        enable_relation_gate: bool = True,
        enable_edge_gate: bool = True,
        group_columns: Optional[Dict[str, List[str]]] = None,
        gate_summary_dim: int = 19,
    ):
        super().__init__()
        self.group_dims = dict(group_dims)
        self.node_hidden_dim = int(node_hidden_dim)
        self.enable_relation_gate = bool(enable_relation_gate)
        self.enable_edge_gate = bool(enable_edge_gate)
        self.group_columns = group_columns or {}
        self.gate_summary_dim = int(gate_summary_dim)

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

        node_map, node_types, edge_index, edge_type, relation_names, edge_specs = build_hgt_vehicle_graph_v2(True)
        self.node_map = node_map
        self.node_types = node_types
        self.relation_names = relation_names
        self.edge_specs = edge_specs
        self.num_nodes = len(node_map)
        self.num_relations = len(relation_names)
        self.num_edges = len(edge_specs)
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

        student_hidden = max(64, node_hidden_dim * 2)
        self.student_summary_predictor = StudentGateSummaryNet(self.readout_dim, self.gate_summary_dim, student_hidden, dropout)

        self.relation_to_edge_positions: Dict[str, List[int]] = {}
        for pos, spec in enumerate(edge_specs):
            self.relation_to_edge_positions.setdefault(spec.relation_name, []).append(pos)
        self.edge_feature_dims = {
            "kinematic_transfer": 9,
            "motion_to_contact": 11,
            "contact_to_motion": 11,
            "longitudinal_coupling": 10,
            "lateral_coupling": 10,
        }
        self.hidden_gate_relations = {"control_excitation"}
        self.edge_gate_feature_mlps = nn.ModuleDict({
            rel: EdgeGateMLP(dim, hgt_heads, max(32, node_hidden_dim), dropout)
            for rel, dim in self.edge_feature_dims.items()
        }) if self.enable_edge_gate else nn.ModuleDict()
        self.edge_gate_hidden_mlps = nn.ModuleDict({
            rel: EdgeGateMLP(node_hidden_dim, hgt_heads, max(32, node_hidden_dim), dropout)
            for rel in self.hidden_gate_relations
        }) if self.enable_edge_gate else nn.ModuleDict()

        self.temporal_reduce = nn.Sequential(
            nn.Linear(self.readout_dim, tcn_hidden_dim),
            nn.LayerNorm(tcn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.tcn = TemporalConvBlock(tcn_hidden_dim, tcn_hidden_dim, 3, dropout)
        self.bilstm = nn.LSTM(
            input_size=tcn_hidden_dim,
            hidden_size=lstm_hidden_dim // 2,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        head_hidden = max(192, node_hidden_dim * 2)
        head_in = lstm_hidden_dim + node_hidden_dim + self.readout_dim
        wheel_head_in = lstm_hidden_dim + node_hidden_dim * 3 + self.readout_dim
        # Body position channels are interpreted as small integration-error deltas.
        # Wheel position channels are interpreted as local deltas; only x/z are used
        # by the shared reconstruction logic and local y is fixed to zero.
        self.body_head = ResidualHead(head_in, int(group_dims["res_body"]), head_hidden, dropout)
        self.wheel_head = ResidualHead(wheel_head_in, int(group_dims["res_wheel0_kin"]), head_hidden, dropout)
        self.force_bias_head = ResidualHead(node_hidden_dim + self.readout_dim, 3, head_hidden, dropout)
        self.force_dynamic_head = ResidualHead(head_in, 3, head_hidden, dropout)

    def _encode_or_virtual(self, node_name: str, batch_key: str, batch: Dict[str, torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
        if node_name in self.input_encoders:
            return self.input_encoders[node_name](batch[batch_key])
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

    def _get_group_cols(self, group_name: str) -> List[str]:
        return list(self.group_columns.get(group_name, []))

    def _find_suffix_index(self, cols: List[str], suffix: str) -> Optional[int]:
        for idx, col in enumerate(cols):
            if col.endswith(suffix):
                return idx
        return None

    def _node_tensor_cols(self, batch: Dict[str, torch.Tensor], name: str):
        key = self.node_to_batch_key.get(name)
        return (batch.get(key), self._get_group_cols(key or "")) if key else (None, [])

    def _stack_named(self, tensor, cols: List[str], suffixes: List[str], ref: torch.Tensor) -> torch.Tensor:
        vals = []
        for suffix in suffixes:
            idx = self._find_suffix_index(cols, suffix)
            vals.append(ref.new_zeros(ref.shape[:2]) if tensor is None or idx is None else tensor[..., idx])
        return torch.stack(vals, dim=-1)

    def _stack_delta(self, src, src_cols: List[str], dst, dst_cols: List[str], suffixes: List[str], ref: torch.Tensor) -> torch.Tensor:
        vals = []
        for suffix in suffixes:
            si = self._find_suffix_index(src_cols, suffix)
            di = self._find_suffix_index(dst_cols, suffix)
            vals.append(ref.new_zeros(ref.shape[:2]) if src is None or dst is None or si is None or di is None else src[..., si] - dst[..., di])
        return torch.stack(vals, dim=-1)

    def _physical_edge_feature(self, batch: Dict[str, torch.Tensor], spec: EdgeSpec, ref: torch.Tensor) -> Optional[torch.Tensor]:
        src, src_cols = self._node_tensor_cols(batch, spec.src_name)
        dst, dst_cols = self._node_tensor_cols(batch, spec.dst_name)
        if spec.relation_name == "kinematic_transfer":
            return self._stack_delta(src, src_cols, dst, dst_cols, ["pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z", "acc_x", "acc_y", "acc_z"], ref)
        if spec.relation_name in {"motion_to_contact", "contact_to_motion"}:
            node = spec.dst_name if spec.dst_name.endswith("_contact") else spec.src_name
            tensor, cols = self._node_tensor_cols(batch, node)
            return self._stack_named(tensor, cols, ["Fx", "Fy", "Fz", "Mx", "My", "Mz", "slip_long", "slip_lat", "sinkage", "in_contact", "contact_switch"], ref)
        if spec.relation_name in {"longitudinal_coupling", "lateral_coupling"}:
            if spec.src_name.endswith("_kin") and spec.dst_name.endswith("_kin"):
                first = self._stack_delta(src, src_cols, dst, dst_cols, ["vel_x", "vel_y", "vel_z", "acc_x", "acc_y", "acc_z"], ref)
                return torch.cat([first, ref.new_zeros(*first.shape[:2], 4)], dim=-1)
            if spec.src_name.endswith("_contact") and spec.dst_name.endswith("_contact"):
                return self._stack_delta(src, src_cols, dst, dst_cols, ["Fx", "Fy", "Fz", "Mx", "My", "Mz", "slip_long", "slip_lat", "sinkage", "in_contact"], ref)
        return None

    def compute_edge_gate(self, batch: Dict[str, torch.Tensor], x: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.enable_edge_gate:
            return None
        ref = batch["system"] if "system" in batch else batch["body"]
        num_heads = self.hgt_layers[0].num_heads
        edge_gate = torch.full((ref.shape[0], ref.shape[1], self.num_edges, num_heads), 0.5, device=ref.device, dtype=ref.dtype)
        for rel, positions in self.relation_to_edge_positions.items():
            if rel == "self_loop":
                edge_gate[:, :, positions, :] = 1.0
                continue
            pos_t = torch.as_tensor(positions, device=ref.device, dtype=torch.long)
            if rel in self.hidden_gate_relations:
                src_idx = torch.as_tensor([self.node_map[self.edge_specs[p].src_name] for p in positions], device=x.device, dtype=torch.long)
                hidden = x[:, :, src_idx, :]
                gate = self.edge_gate_hidden_mlps[rel](hidden.reshape(-1, hidden.shape[-1])).reshape(hidden.shape[0], hidden.shape[1], hidden.shape[2], num_heads)
            else:
                feats = [self._physical_edge_feature(batch, self.edge_specs[p], ref) for p in positions]
                if any(f is None for f in feats):
                    continue
                feat = torch.stack(feats, dim=2)
                gate = self.edge_gate_feature_mlps[rel](feat.reshape(-1, feat.shape[-1])).reshape(feat.shape[0], feat.shape[1], feat.shape[2], num_heads)
            edge_gate[:, :, pos_t, :] = gate.clamp(1e-4, 1.0)
        return edge_gate

    def encode_nodes(self, batch: Dict[str, torch.Tensor], relation_gate_override: Optional[torch.Tensor] = None):
        ref = batch["system"] if "system" in batch else batch["body"]
        tensors = [None] * self.num_nodes
        for name, idx in self.node_map.items():
            key = self.node_to_batch_key[name]
            tensors[idx] = self._encode_or_virtual(name, key, batch, ref)
        x = torch.stack(tensors, dim=2)
        gate_stats: Dict[str, torch.Tensor] = {}
        edge_gate_stats = None
        relation_gate = relation_gate_override
        if relation_gate is not None:
            gate_stats["teacher_relation_gate"] = relation_gate
        for layer in self.hgt_layers:
            if relation_gate is None and self.enable_relation_gate:
                summary = self.student_summary_predictor(self._five_group_readout(x))
                gate_stats["student_gate_summary"] = summary
                gate_stats["student_relation_gate"] = gate_stats.get("teacher_gate_net")(summary) if "teacher_gate_net" in gate_stats else None
            layer_relation_gate = relation_gate if relation_gate is not None else gate_stats.get("student_relation_gate")
            edge_gate = self.compute_edge_gate(batch, x)
            edge_gate_stats = edge_gate
            x = layer(x, relation_gate=layer_relation_gate, edge_gate=edge_gate)
        if edge_gate_stats is not None:
            gate_stats["edge_gate_stats"] = edge_gate_stats.detach()
        return x, gate_stats

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        relation_gate_override: Optional[torch.Tensor] = None,
        teacher_gate_net: Optional[TeacherGateNet] = None,
    ) -> Dict[str, torch.Tensor]:
        nodes, stats = self.encode_nodes(batch, relation_gate_override=relation_gate_override)
        if relation_gate_override is None and teacher_gate_net is not None and "student_gate_summary" in stats:
            stats["student_relation_gate"] = teacher_gate_net(stats["student_gate_summary"])
            nodes, stats2 = self.encode_nodes(batch, relation_gate_override=stats["student_relation_gate"])
            stats.update(stats2)
        readout_seq = self._five_group_readout(nodes)
        y = self.temporal_reduce(readout_seq)
        y = self.tcn(y)
        y, _ = self.bilstm(y)
        h_t = y[:, -1:, :]
        readout_t = readout_seq[:, -1:, :]

        def target_feat(node_name: str) -> torch.Tensor:
            node = nodes[:, -1:, self.node_map[node_name], :]
            return torch.cat([h_t, node, readout_t], dim=-1)

        def wheel_feat(wheel_id: int) -> torch.Tensor:
            body = nodes[:, -1:, self.body_idx, :]
            rocker = nodes[:, -1:, self.node_map[self.wheel_to_rocker[wheel_id]], :]
            wheel = nodes[:, -1:, self.node_map[f"wheel{wheel_id}_kin"], :]
            return torch.cat([h_t, body, rocker, wheel, readout_t], dim=-1)

        pred_body = self.body_head(target_feat("body"))
        out = {"pred_res_body": pred_body}
        for i in WHEEL_IDS:
            contact_node = nodes[:, -1:, self.node_map[f"wheel{i}_contact"], :]
            contact_feat = torch.cat([h_t, contact_node, readout_t], dim=-1)
            contact_spatial_feat = torch.cat([contact_node, readout_t], dim=-1)
            out[f"pred_res_wheel{i}_kin"] = self.wheel_head(wheel_feat(i))
            out[f"pred_force_bias_wheel{i}"] = self.force_bias_head(contact_spatial_feat)
            out[f"pred_force_dynamic_wheel{i}"] = self.force_dynamic_head(contact_feat)
        out.update({k: v for k, v in stats.items() if v is not None})
        return out


GraphTemporalHGTCompensationModelV2 = HGTGraphTemporalCompensationModelV2
