from __future__ import annotations

from dataclasses import dataclass
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


class RelationGateMLP(nn.Module):
    """根据全局图状态生成 relation gate。"""

    def __init__(
        self,
        in_dim: int,
        num_relations: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_relations),
            nn.Sigmoid(),
        )
        self._init_neutral_gate_head()

    def _init_neutral_gate_head(self) -> None:
        # 让 gate 在首次启用时稳定落在 0.5，避免随机初始化立即扰动已训练好的 base 模型。
        last_linear = next(module for module in reversed(self.net) if isinstance(module, nn.Linear))
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).clamp(1e-4, 1.0)


@dataclass(frozen=True)
class EdgeSpec:
    # src_name / dst_name: 这条边两端的节点名。
    src_name: str
    # relation_name: 这条边所属的关系类型字符串。
    dst_name: str
    # relation_id: 这条边所属的关系类型编号，和 edge_type 对齐。
    relation_name: str
    relation_id: int


class EdgeGateMLP(nn.Module):
    """将关系相关的边特征映射为逐边 gate。"""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        # in_dim 是当前关系对应的物理边特征维度。
        self.in_dim = int(in_dim)
        if self.in_dim <= 0:
            # 没有有效物理特征时，不构建 MLP，由上层逻辑兜底。
            self.net = None
        else:
            # 输出 out_dim 个 gate；在当前实现里 out_dim = num_heads。
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
                nn.Sigmoid(),
            )
            self._init_neutral_gate_head()

    def _init_neutral_gate_head(self) -> None:
        if self.net is None:
            return
        # 让逐边 gate 的默认输出为 0.5，对应 attention 的中性倍率 1.0。
        last_linear = next(module for module in reversed(self.net) if isinstance(module, nn.Linear))
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.net is None:
            raise RuntimeError("EdgeGateMLP 收到空输入维度，请检查关系特征配置。")
        return self.net(x)


class TeacherGateNet(nn.Module):
    """训练阶段使用高保真/残差摘要生成教师 relation gate。"""

    def __init__(
        self,
        in_dim: int,
        num_relations: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_relations),
            nn.Sigmoid(),
        )
        self._init_neutral_gate_head()

    def _init_neutral_gate_head(self) -> None:
        # 教师 gate 也从 0.5 起步，避免蒸馏一开始就强推某种关系分布。
        last_linear = next(module for module in reversed(self.net) if isinstance(module, nn.Linear))
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        # 教师 gate 永远压到稳定区间，避免蒸馏或取 log 时数值不稳。
        return self.net(summary).clamp(1e-4, 1.0)


class StudentGateSummaryNet(nn.Module):
    """根据学生可见的全局状态预测教师侧全局摘要参数。"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def strength_to_gate(strength: torch.Tensor, scale: float = 1.5) -> torch.Tensor:
    return (1.0 - torch.exp(-scale * strength)).clamp(1e-4, 1.0)


def gate_to_attention_multiplier(gate: torch.Tensor) -> torch.Tensor:
    # gate 的数值语义保留在 [0, 1]，但 attention 里的中性点应为 0.5 而不是 1.0。
    # 这样新启用 gate 时，默认输出不会把已训练好的消息传递整体压小一截。
    return (1.0 + (gate - 0.5)).clamp(1e-4, 1.5)


def build_hgt_vehicle_graph(
    include_self_loops: bool = True,
) -> Tuple[Dict[str, int], List[str], torch.Tensor, torch.Tensor, List[str], List[EdgeSpec]]:
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
    # edges 存数值版边；edge_specs 存可解释的边元数据，后续 edge_gate 和日志都依赖它。
    edges: List[Tuple[int, int, int]] = []
    edge_specs: List[EdgeSpec] = []

    def add(src: str, dst: str, rel: str) -> None:
        # 数值图结构供 HGT 使用。
        edges.append((node_map[src], node_map[dst], rel_id[rel]))
        # 可解释边结构供物理特征构造和分 relation 统计使用。
        edge_specs.append(EdgeSpec(src_name=src, dst_name=dst, relation_name=rel, relation_id=rel_id[rel]))

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
    return node_map, node_types, edge_index, edge_type, relation_names, edge_specs


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
        # x: [BT, N, H]
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
            # 每条边按自己的 relation 类型取 gate，再转成对数偏置加到 score 上。
            score = score + torch.log(gate_to_attention_multiplier(gate[:, rel])).unsqueeze(-1)

        if edge_gate is not None:
            if edge_gate.ndim == 3:
                gate = edge_gate.reshape(bt, num_edges).clamp(1e-4, 1.0)
                # [B, T, E] 情况下，所有 head 共用同一个逐边 gate。
                score = score + torch.log(gate_to_attention_multiplier(gate)).unsqueeze(-1)
            elif edge_gate.ndim == 4:
                gate = edge_gate.reshape(bt, num_edges, self.num_heads).clamp(1e-4, 1.0)
                # [B, T, E, heads] 情况下，每个 head 都有自己的逐边 gate。
                score = score + torch.log(gate_to_attention_multiplier(gate))
            else:
                raise ValueError(
                    f"edge_gate 应为 [B, T, E] 或 [B, T, E, heads]，实际 shape={tuple(edge_gate.shape)}"
                )

        # 按目标节点分组做 softmax；先做 group-wise max，再做 group-wise sum。
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
        # 按目标节点把所有边消息一次性 scatter_add 回节点张量。
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
        enable_edge_gate: bool = False,
        pred_seq_len: int = 1,
        group_columns: Optional[Dict[str, List[str]]] = None,
        gate_summary_dim: int = 31,
    ):
        super().__init__()
        self.group_dims = group_dims
        self.node_hidden_dim = node_hidden_dim
        self.enable_relation_gate = enable_relation_gate
        self.enable_edge_gate = enable_edge_gate
        self.pred_seq_len = int(pred_seq_len)
        self.gate_summary_dim = int(gate_summary_dim)
        # group_columns 保留每个输入组的原始列名，供物理 edge feature 逐列匹配。
        self.group_columns = group_columns or {}
        if self.pred_seq_len < 1:
            raise ValueError(f"pred_seq_len 必须 >= 1，实际为 {self.pred_seq_len}")

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

        node_map, node_types, edge_index, edge_type, relation_names, edge_specs = build_hgt_vehicle_graph(include_self_loops=True)
        self.node_map = node_map
        self.node_types = node_types
        self.relation_names = relation_names
        self.edge_specs = edge_specs
        self.num_nodes = len(node_map)
        self.num_relations = len(relation_names)
        self.num_edges = len(edge_specs)
        self.edge_gate_eps = 1e-4
        self.control_idx = node_map["control_context"]
        self.body_idx = node_map["body"]
        self.rocker_indices = [node_map[name] for name in ROCKER_NAMES if name in node_map]
        self.wheel_kin_indices = [node_map[f"wheel{i}_kin"] for i in WHEEL_IDS if f"wheel{i}_kin" in node_map]
        self.wheel_contact_indices = [node_map[f"wheel{i}_contact"] for i in WHEEL_IDS if f"wheel{i}_contact" in node_map]
        self.observed_indices = [self.control_idx, self.body_idx] + self.rocker_indices + self.wheel_kin_indices + self.wheel_contact_indices
        self.relation_gate_context_dim = node_hidden_dim * 5
        self.graph_readout_dim = node_hidden_dim * 4
        # 把图节点名映射回 batch key，后面用它从 batch 中取该节点对应的 LF 输入。
        self.node_to_batch_key = {"control_context": "system", "body": "body"}
        self.node_to_batch_key.update({name: name for name in ROCKER_NAMES})
        self.node_to_batch_key.update({f"wheel{i}_kin": f"wheel{i}_kin" for i in WHEEL_IDS})
        self.node_to_batch_key.update({f"wheel{i}_contact": f"wheel{i}_contact" for i in WHEEL_IDS})
        self.node_to_batch_key.update({f"target_error_wheel{i}_kin": None for i in WHEEL_IDS})
        self.node_to_batch_key.update({f"target_error_wheel{i}_contact": None for i in WHEEL_IDS})
        self.node_to_batch_key["target_error_body"] = None
        self.target_error_to_res_batch_key = {"target_error_body": "res_body"}
        self.target_error_to_res_batch_key.update({f"target_error_wheel{i}_kin": f"res_wheel{i}_kin" for i in WHEEL_IDS})
        self.target_error_to_res_batch_key.update({f"target_error_wheel{i}_contact": f"res_wheel{i}_contact" for i in WHEEL_IDS})
        self.relation_to_edge_positions: Dict[str, List[int]] = {}
        for edge_pos, edge_spec in enumerate(edge_specs):
            self.relation_to_edge_positions.setdefault(edge_spec.relation_name, []).append(edge_pos)

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
                RelationGateMLP(
                    in_dim=self.relation_gate_context_dim,
                    num_relations=self.num_relations,
                    hidden_dim=max(64, node_hidden_dim * 2),
                    dropout=dropout,
                )
                for _ in range(graph_layers)
            ])
        else:
            self.relation_gate_layers = nn.ModuleList([nn.Identity() for _ in range(graph_layers)])

        student_gate_hidden_dim = max(64, node_hidden_dim * 2)
        self.student_summary_predictor = StudentGateSummaryNet(
            in_dim=self.relation_gate_context_dim,
            out_dim=self.gate_summary_dim,
            hidden_dim=student_gate_hidden_dim,
            dropout=dropout,
        )
        self.student_summary_gate_net = TeacherGateNet(
            in_dim=self.gate_summary_dim,
            num_relations=self.num_relations,
            hidden_dim=student_gate_hidden_dim,
            dropout=dropout,
        )

        self.edge_feature_dims = {
            # 这几类关系使用显式物理边特征。
            "kinematic_transfer": 9,
            "motion_to_contact": 11,
            "contact_to_motion": 11,
            "longitudinal_coupling": 10,
            "lateral_coupling": 10,
        }
        # 这几类关系不用显式物理差分，直接由源节点隐藏状态生成 gate。
        self.hidden_gate_relations = {"control_excitation", "state_to_target"}
        if self.enable_edge_gate:
            self.edge_gate_feature_mlps = nn.ModuleDict({
                rel_name: EdgeGateMLP(
                    in_dim=feat_dim,
                    out_dim=hgt_heads,
                    hidden_dim=max(32, node_hidden_dim),
                    dropout=dropout,
                )
                for rel_name, feat_dim in self.edge_feature_dims.items()
            })
            self.edge_gate_hidden_mlps = nn.ModuleDict({
                rel_name: EdgeGateMLP(
                    in_dim=node_hidden_dim,
                    out_dim=hgt_heads,
                    hidden_dim=max(32, node_hidden_dim),
                    dropout=dropout,
                )
                for rel_name in self.hidden_gate_relations
            })
        else:
            self.edge_gate_feature_mlps = nn.ModuleDict()
            self.edge_gate_hidden_mlps = nn.ModuleDict()

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
        self.horizon_embed = nn.Parameter(torch.zeros(self.pred_seq_len, temporal_dim))
        head_in_dim = temporal_dim + node_hidden_dim + self.graph_readout_dim
        head_hidden_dim = max(192, node_hidden_dim * 2)
        self.body_head = ResidualHead(head_in_dim, group_dims["res_body"], hidden_dim=head_hidden_dim, dropout=dropout)
        self.wheel_head = ResidualHead(head_in_dim, group_dims["res_wheel0_kin"], hidden_dim=head_hidden_dim, dropout=dropout)
        self.contact_head = ResidualHead(head_in_dim, group_dims["res_wheel0_contact"], hidden_dim=head_hidden_dim, dropout=dropout)

    def _get_group_cols(self, group_name: str) -> List[str]:
        # 返回该输入组的列名列表；不存在时返回空列表而不是报错。
        return list(self.group_columns.get(group_name, []))

    def _get_node_tensor_and_cols(
        self,
        batch: Dict[str, torch.Tensor],
        node_name: str,
    ) -> Tuple[Optional[torch.Tensor], List[str]]:
        batch_key = self.node_to_batch_key.get(node_name)
        if batch_key is None or batch_key not in batch:
            # target_error 节点或缺失输入节点不直接参与物理列匹配。
            return None, []
        return batch[batch_key], self._get_group_cols(batch_key)

    def _zero_feature(self, ref: torch.Tensor, feat_dim: int) -> torch.Tensor:
        # 所有缺失列、缺失节点或不适用关系都统一补零，保证前向安全。
        return ref.new_zeros(ref.shape[0], ref.shape[1], feat_dim)

    def _find_suffix_index(self, cols: List[str], suffix: str) -> Optional[int]:
        # 按列后缀匹配物理量，例如 pos_x / slip_long / Fz。
        for idx, col in enumerate(cols):
            if col.endswith(suffix):
                return idx
        return None

    def _stack_named_features(
        self,
        tensor: Optional[torch.Tensor],
        cols: List[str],
        suffixes: List[str],
        ref: torch.Tensor,
    ) -> torch.Tensor:
        values: List[torch.Tensor] = []
        for suffix in suffixes:
            if tensor is None:
                # 整个节点输入缺失时，该物理量直接补零。
                values.append(ref.new_zeros(ref.shape[0], ref.shape[1]))
                continue
            idx = self._find_suffix_index(cols, suffix)
            if idx is None:
                # 某一列缺失时只补当前维，不影响其他物理量。
                values.append(ref.new_zeros(ref.shape[0], ref.shape[1]))
            else:
                values.append(tensor[..., idx])
        return torch.stack(values, dim=-1)

    def _stack_delta_features(
        self,
        src_tensor: Optional[torch.Tensor],
        src_cols: List[str],
        dst_tensor: Optional[torch.Tensor],
        dst_cols: List[str],
        suffixes: List[str],
        ref: torch.Tensor,
    ) -> torch.Tensor:
        values: List[torch.Tensor] = []
        for suffix in suffixes:
            if src_tensor is None or dst_tensor is None:
                # 任一端节点输入缺失时，该差分物理量补零。
                values.append(ref.new_zeros(ref.shape[0], ref.shape[1]))
                continue
            src_idx = self._find_suffix_index(src_cols, suffix)
            dst_idx = self._find_suffix_index(dst_cols, suffix)
            if src_idx is None or dst_idx is None:
                # 源端或目标端任一列不存在时，不抛错，直接补零。
                values.append(ref.new_zeros(ref.shape[0], ref.shape[1]))
            else:
                # 差分方向固定为 src - dst，保持每条有向边的物理方向性。
                values.append(src_tensor[..., src_idx] - dst_tensor[..., dst_idx])
        return torch.stack(values, dim=-1)

    def _build_kinematic_transfer_features(
        self,
        batch: Dict[str, torch.Tensor],
        edge_spec: EdgeSpec,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        suffixes = [
            # 车身-摇臂、摇臂-车轮运动链只使用运动学差分。
            "pos_x", "pos_y", "pos_z",
            "vel_x", "vel_y", "vel_z",
            "acc_x", "acc_y", "acc_z",
        ]
        src_tensor, src_cols = self._get_node_tensor_and_cols(batch, edge_spec.src_name)
        dst_tensor, dst_cols = self._get_node_tensor_and_cols(batch, edge_spec.dst_name)
        return self._stack_delta_features(src_tensor, src_cols, dst_tensor, dst_cols, suffixes, ref)

    def _build_contact_exchange_features(
        self,
        batch: Dict[str, torch.Tensor],
        edge_spec: EdgeSpec,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        suffixes = [
            "Fx", "Fy", "Fz",
            "Mx", "My", "Mz",
            "slip_long", "slip_lat",
            "sinkage", "in_contact", "contact_switch",
        ]
        # motion/contact 双向耦合都只看对应车轮的 contact 输入，不混入其他轮信息。
        contact_node = edge_spec.dst_name if edge_spec.dst_name.endswith("_contact") else edge_spec.src_name
        contact_tensor, contact_cols = self._get_node_tensor_and_cols(batch, contact_node)
        return self._stack_named_features(contact_tensor, contact_cols, suffixes, ref)

    def _build_coupling_features(
        self,
        batch: Dict[str, torch.Tensor],
        edge_spec: EdgeSpec,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        feat = self._zero_feature(ref, 10)
        src_tensor, src_cols = self._get_node_tensor_and_cols(batch, edge_spec.src_name)
        dst_tensor, dst_cols = self._get_node_tensor_and_cols(batch, edge_spec.dst_name)

        if edge_spec.src_name.endswith("_kin") and edge_spec.dst_name.endswith("_kin"):
            # 运动学节点之间优先使用速度/加速度差异。
            kin_suffixes = ["vel_x", "vel_y", "vel_z", "acc_x", "acc_y", "acc_z"]
            feat[..., :6] = self._stack_delta_features(src_tensor, src_cols, dst_tensor, dst_cols, kin_suffixes, ref)
        elif edge_spec.src_name.endswith("_contact") and edge_spec.dst_name.endswith("_contact"):
            # 接触节点之间优先使用接触状态差异。
            contact_suffixes = ["Fx", "Fy", "Fz", "Mx", "My", "Mz", "slip_long", "slip_lat", "sinkage", "in_contact"]
            feat[...] = self._stack_delta_features(src_tensor, src_cols, dst_tensor, dst_cols, contact_suffixes, ref)
        return feat

    def _build_physical_edge_feature(
        self,
        batch: Dict[str, torch.Tensor],
        edge_spec: EdgeSpec,
        ref: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if edge_spec.relation_name == "kinematic_transfer":
            return self._build_kinematic_transfer_features(batch, edge_spec, ref)
        if edge_spec.relation_name in {"motion_to_contact", "contact_to_motion"}:
            return self._build_contact_exchange_features(batch, edge_spec, ref)
        if edge_spec.relation_name in {"longitudinal_coupling", "lateral_coupling"}:
            return self._build_coupling_features(batch, edge_spec, ref)
        return None

    def _build_physical_edge_features_for_relation(
        self,
        batch: Dict[str, torch.Tensor],
        relation_name: str,
        edge_positions: List[int],
        ref: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        features: List[torch.Tensor] = []
        for edge_pos in edge_positions:
            feat = self._build_physical_edge_feature(batch, self.edge_specs[edge_pos], ref)
            if feat is None:
                return None
            features.append(feat)
        if not features:
            return None
        # [B, T, E_rel, D_rel]
        return torch.stack(features, dim=2)

    def _mean_node_features(self, x: torch.Tensor, node_indices: List[int]) -> torch.Tensor:
        if not node_indices:
            return x.new_zeros(x.shape[0], x.shape[1], x.shape[-1])
        index_tensor = torch.as_tensor(node_indices, dtype=torch.long, device=x.device)
        return x[:, :, index_tensor, :].mean(dim=2)

    def _build_relation_gate_context(self, x: torch.Tensor) -> torch.Tensor:
        control_feat = x[:, :, self.control_idx, :]
        body_feat = x[:, :, self.body_idx, :]
        wheel_kin_mean = self._mean_node_features(x, self.wheel_kin_indices)
        wheel_contact_mean = self._mean_node_features(x, self.wheel_contact_indices)
        observed_mean = self._mean_node_features(x, self.observed_indices)
        return torch.cat(
            [control_feat, body_feat, wheel_kin_mean, wheel_contact_mean, observed_mean],
            dim=-1,
        )

    def _build_graph_readout(self, node_t: torch.Tensor) -> torch.Tensor:
        wheel_kin = self._mean_node_features(node_t.unsqueeze(1), self.wheel_kin_indices).squeeze(1)
        wheel_contact = self._mean_node_features(node_t.unsqueeze(1), self.wheel_contact_indices).squeeze(1)
        observed = self._mean_node_features(node_t.unsqueeze(1), self.observed_indices).squeeze(1)
        body_feat = node_t[:, self.body_idx, :]
        return torch.cat([body_feat, wheel_kin, wheel_contact, observed], dim=-1)

    def _node_strength_over_time(
        self,
        batch: Dict[str, torch.Tensor],
        node_name: str,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        tensor, _ = self._get_node_tensor_and_cols(batch, node_name)
        if tensor is None or tensor.numel() == 0:
            return ref.new_zeros(ref.shape[0], ref.shape[1])
        return tensor.abs().mean(dim=-1)

    def _build_hidden_edge_strength(
        self,
        batch: Dict[str, torch.Tensor],
        edge_spec: EdgeSpec,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        src_strength = self._node_strength_over_time(batch, edge_spec.src_name, ref)
        if edge_spec.relation_name == "control_excitation":
            system = batch.get("system")
            if system is None or system.numel() == 0:
                return src_strength
            diff_strength = ref.new_zeros(ref.shape[0], ref.shape[1])
            if system.shape[1] > 1:
                diff_strength[:, 1:] = (system[:, 1:] - system[:, :-1]).abs().mean(dim=-1)
            return 0.6 * src_strength + 0.4 * diff_strength
        if edge_spec.relation_name == "state_to_target":
            res_key = self.target_error_to_res_batch_key.get(edge_spec.dst_name)
            if res_key is None or res_key not in batch:
                return src_strength
            res_tensor = batch[res_key]
            if res_tensor.numel() == 0:
                return src_strength
            res_strength = res_tensor.abs().mean(dim=tuple(range(1, res_tensor.ndim)))
            return src_strength * (1.0 + res_strength.unsqueeze(1))
        return src_strength

    def build_edge_distill_target(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        ref = batch["system"] if "system" in batch else batch["body"]
        num_heads = self.hgt_layers[0].num_heads
        edge_target = ref.new_full((ref.shape[0], ref.shape[1], self.num_edges), 0.5)

        for edge_idx, edge_spec in enumerate(self.edge_specs):
            if edge_spec.relation_name == "self_loop":
                edge_target[:, :, edge_idx] = 1.0
                continue

            feat = self._build_physical_edge_feature(batch, edge_spec, ref)
            if feat is not None:
                strength = feat.abs().mean(dim=-1)
            else:
                strength = self._build_hidden_edge_strength(batch, edge_spec, ref)

            edge_target[:, :, edge_idx] = strength_to_gate(strength)

        return edge_target.unsqueeze(-1).expand(-1, -1, -1, num_heads)

    def compute_edge_gate(
        self,
        batch: Dict[str, torch.Tensor],
        x: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if not self.enable_edge_gate:
            # 关闭 edge_gate 时完全退回第二阶段前的逻辑。
            return None

        ref = batch["system"] if "system" in batch else batch["body"]
        num_heads = self.hgt_layers[0].num_heads
        edge_gate: Optional[torch.Tensor] = None

        # 自环保持常数 1，不需要额外计算。
        for rel_name, edge_positions in self.relation_to_edge_positions.items():
            if rel_name == "self_loop":
                continue

            edge_pos_tensor = torch.as_tensor(edge_positions, dtype=torch.long, device=ref.device)

            if rel_name in self.hidden_gate_relations:
                # 同一 relation 的边一次性取出源节点隐藏状态，再批量过同一个 MLP。
                src_indices = torch.as_tensor(
                    [self.node_map[self.edge_specs[pos].src_name] for pos in edge_positions],
                    dtype=torch.long,
                    device=x.device,
                )
                hidden = x[:, :, src_indices, :]  # [B, T, E_rel, H]
                gate = self.edge_gate_hidden_mlps[rel_name](hidden.reshape(-1, hidden.shape[-1]))
                gate = gate.reshape(hidden.shape[0], hidden.shape[1], hidden.shape[2], num_heads)
            else:
                # 同一 relation 的物理边特征先批量构造，再一次性过该 relation 的 MLP。
                feat = self._build_physical_edge_features_for_relation(batch, rel_name, edge_positions, ref)
                if feat is None:
                    continue
                gate = self.edge_gate_feature_mlps[rel_name](feat.reshape(-1, feat.shape[-1]))
                gate = gate.reshape(feat.shape[0], feat.shape[1], feat.shape[2], num_heads)

            if edge_gate is None:
                edge_gate = torch.full(
                    (ref.shape[0], ref.shape[1], self.num_edges, num_heads),
                    0.5,
                    device=ref.device,
                    dtype=gate.dtype,
                )

            # 按 relation 回填到全局 edge 维；这样 edge_gate 仍和 edge_index 一一对应。
            edge_gate[:, :, edge_pos_tensor, :] = gate.clamp(self.edge_gate_eps, 1.0)
        if edge_gate is None:
            return None
        return edge_gate

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

    def encode_nodes(
        self,
        batch: Dict[str, torch.Tensor],
        relation_gate_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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

        # relation_gate_values 保留可反传版本，供第三阶段蒸馏使用。
        relation_gate_values: List[torch.Tensor] = []
        summary_values: List[torch.Tensor] = []
        edge_gate_values: List[torch.Tensor] = []
        # edge_gate_stats 只做日志观察，不参与额外损失。
        edge_gate_stats: List[torch.Tensor] = []
        for layer_idx, layer in enumerate(self.hgt_layers):
            if relation_gate_override is not None:
                relation_gate = relation_gate_override
                relation_gate_values.append(relation_gate)
            elif self.enable_relation_gate:
                relation_gate_context = self._build_relation_gate_context(x)
                student_gate_summary = self.student_summary_predictor(relation_gate_context)
                relation_gate = self.student_summary_gate_net(student_gate_summary)
                summary_values.append(student_gate_summary)
                relation_gate_values.append(relation_gate)
            else:
                relation_gate = None
            edge_gate = self.compute_edge_gate(batch, x)
            if edge_gate is not None:
                edge_gate_values.append(edge_gate)
                edge_gate_stats.append(edge_gate.detach())
            x = layer(x, relation_gate=relation_gate, edge_gate=edge_gate)
        stats: Dict[str, torch.Tensor] = {}
        if relation_gate_values:
            stats["student_relation_gate"] = relation_gate_values[-1]
            stats["relation_gate_stats"] = relation_gate_values[-1].detach()
            if summary_values:
                stats["student_gate_summary"] = summary_values[-1]
                stats["student_inferred_relation_gate"] = relation_gate_values[-1]
                stats["student_inferred_relation_gate_stats"] = relation_gate_values[-1].detach()
        if edge_gate_values:
            stats["student_edge_gate"] = edge_gate_values[-1]
        if edge_gate_stats:
            stats["edge_gate_stats"] = edge_gate_stats[-1]
        return x, stats

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        relation_gate_override: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        nodes, gate_stats = self.encode_nodes(batch, relation_gate_override=relation_gate_override)
        bsz, seq_len, num_nodes, hidden_dim = nodes.shape

        fused = nodes.reshape(bsz, seq_len, num_nodes * hidden_dim)
        y = self.tcn(fused)
        y, _ = self.bilstm(y)
        h_t = y[:, -1, :]
        node_t = nodes[:, -1, :, :]
        graph_readout = self._build_graph_readout(node_t)

        horizon_feat = h_t.unsqueeze(1) + self.horizon_embed.unsqueeze(0)

        def build_target_feature(node_name: str) -> torch.Tensor:
            node_feat = node_t[:, self.node_map[node_name], :].unsqueeze(1).expand(-1, self.pred_seq_len, -1)
            readout_feat = graph_readout.unsqueeze(1).expand(-1, self.pred_seq_len, -1)
            return torch.cat([horizon_feat, node_feat, readout_feat], dim=-1)

        body_feat = build_target_feature("target_error_body")
        pred_body = self.body_head(body_feat)

        pred_wheel_kin = []
        pred_wheel_contact = []
        for i in WHEEL_IDS:
            kin_feat = build_target_feature(f"target_error_wheel{i}_kin")
            contact_feat = build_target_feature(f"target_error_wheel{i}_contact")
            pred_wheel_kin.append(self.wheel_head(kin_feat))
            pred_wheel_contact.append(self.contact_head(contact_feat))

        output = {
            "pred_res_body": pred_body,
            **{f"pred_res_wheel{i}_kin": pred_wheel_kin[i] for i in WHEEL_IDS},
            **{f"pred_res_wheel{i}_contact": pred_wheel_contact[i] for i in WHEEL_IDS},
        }
        output.update(gate_stats)
        return output


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
