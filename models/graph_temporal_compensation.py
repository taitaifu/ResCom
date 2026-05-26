from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data_utils import WHEEL_IDS, build_vehicle_graph_edges

# 多层感知机MLP编码器，把某一类输入特征映射到统一的隐藏维度
class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        '''
        in_dim: 输入特征维度
        hidden_dim: 中间隐藏层维度
        out_dim: 输出特征维度
        dropout: 随机失活概率
        '''
        super().__init__()
        # 定义一个顺序网络，里面的层按照写入顺序依次执行
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),  # 全连接层，输入特征到隐藏层
            nn.LayerNorm(hidden_dim),       # 隐藏层层归一化，训练更稳定
            nn.ReLU(),                      # ReLU 激活函数，引入非线性
            nn.Dropout(dropout),            # 随机失活，防止过拟合
            nn.Linear(hidden_dim, out_dim), # 全连接层，隐藏层到输出层
            nn.LayerNorm(out_dim),          # 输出层归一化
            nn.ReLU(),                      # 输出端再做一次 ReLU 激活
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: # 前向传播
        if x.shape[-1] == 0:
            shape = list(x.shape)
            shape[-1] = 0
            return x       # 特征维度为0时，直接返回输入
        return self.net(x) # 正常情况下，把输入送入前面定义的网络

# 定义图消息传递层。用于在车辆拓扑图上做节点间的信息传播，
class GraphMessagePassing(nn.Module):
    def __init__(self, hidden_dim: int, edges: List[Tuple[int, int]], num_nodes: int):
        '''
        hidden_dim: 节点特征维度
        edges: 图中边的列表
        num_nodes: 图中节点数量
        '''
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        adj = torch.zeros(num_nodes, num_nodes) # 创建一个邻接矩阵，大小为(num_nodes, num_nodes)，初始值为0
        for i, j in edges:  # 遍历边列表，构建邻接矩阵
            adj[i, j] = 1.0 # 在邻接矩阵中标记边的存在，表示节点i和j之间存在信息传递关系
        deg = adj.sum(dim=1, keepdim=True).clamp_min(1.0) # 计算每个节点的出度，也就是邻接矩阵中每一行的和，并且确保最小值为1.0，避免除以零
        adj = adj / deg # 对邻接矩阵进行归一化处理，使得每个节点的邻居信息在消息传递时被平均化
        self.register_buffer("adj", adj) # 把邻接矩阵注册为 buffer。它不是可训练参数，但是会随着模型一起保存和移动到 GPU。
        self.self_proj = nn.Linear(hidden_dim, hidden_dim) # 定义节点自身特征的线性变换层
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim)  # 定义邻居消息特征的线性变换层
        self.out_norm = nn.LayerNorm(hidden_dim)           # 定义输出特征的层归一化层

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义图消息传递的前向传播
        # x: [B, T, N, H]                                # 输入形状，batch_size x seq_len x num_nodes x hidden_dim
        msg = torch.einsum("ij,btjn->btin", self.adj, x) # 图邻居聚合。输入 x 的维度是 [B,T,N,H]，邻接矩阵是 [N,N]。这里的目标是对每个节点聚合相邻节点的信息。
        out = self.self_proj(x) + self.msg_proj(msg)     # 对节点自身特征和邻居消息特征进行线性变换后相加
        out = self.out_norm(out)                         # 对输出特征进行层归一化
        return F.relu(out)                               # 对输出特征进行 ReLU 激活

# 定义时间卷积模块。用于在时间维度上提取局部时序特征。
class TemporalConvBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, kernel_size: int = 3, dropout: float = 0.1):
        '''
        in_dim: 输入特征维度
        hidden_dim: 隐藏层特征维度
        kernel_size: 卷积核大小
        dropout: 随机失活概率
        '''
        super().__init__()
        # padding的作用是让 TCN 在提取短时序变化特征时，不改变时间序列长度
        padding = kernel_size - 1                                                    # 卷积填充大小
        self.conv1 = nn.Conv1d(in_dim, hidden_dim, kernel_size, padding=padding)     # 第一层卷积
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding) # 第二层卷积
        self.norm = nn.LayerNorm(hidden_dim)                                         # 定义层归一化
        self.dropout = nn.Dropout(dropout)                                           # 定义随机失活层
        self.res_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity() # 定义残差分支，如果输入和输出维度不一致，则使用线性变换，否则使用恒等映射

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]             # 输入张量形状，batch_size x seq_len x feature_dim
        res = self.res_proj(x)     # 残差分支，后面会和卷积结果相加
        y = x.transpose(1, 2)      # 把输入从 [B, T, D] 转换为 [B, D, T]，已适配Conv1d的输入形状要求 [B, C, T]，其中 C 是通道数
        y = self.conv1(y)          # 第一层卷积
        y = y[:, :, :x.shape[1]]   # 把卷积结果裁剪回原始时间长度
        y = F.relu(y)              # ReLU 激活
        y = self.dropout(y)        # 随机失活
        y = self.conv2(y)          # 第二层卷积
        y = y[:, :, :x.shape[1]]   # 把卷积结果裁剪回原始时间长度
        y = y.transpose(1, 2)      # 把张量从 [B, D, T] 转换回 [B, T, D]
        y = self.norm(y)           # 对最后一个维度进行层归一化，即对每个时间步的特征D进行归一化
        y = F.relu(y + res)        # 卷积输出加上残差分支，再经过 ReLU。这样可以缓解深层网络训练困难，也有利于保留原始输入信息。
        return y

# 定义残差预测头。它用于把时序特征和局部节点特征映射成最终的残差输出。
# 一个简单的MLP，主要承担的是最后一步映射，主体建模能力由其他模块提供
class ResidualHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),     # 定义第一层线性变换
            nn.LayerNorm(hidden_dim),          # 定义层归一化
            nn.ReLU(),                         # ReLU 激活
            nn.Dropout(dropout),               # 随机失活
            nn.Linear(hidden_dim, out_dim),    # 定义第二层线性变换
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: # 前向传播
        return self.net(x)

# 主模型，即图时序误差补偿模型
class GraphTemporalCompensationModel(nn.Module):
    def __init__(
        self,
        group_dims: Dict[str, int],         # 每组特征维度字典
        node_hidden_dim: int = 64,          # 每个图节点的隐藏层特征维度
        graph_layers: int = 2,              # 图卷积层数，图消息传递层数
        tcn_hidden_dim: int = 256,          # 时间卷积网络输出维度
        lstm_hidden_dim: int = 256,         # LSTM 输出维度
        lstm_layers: int = 2,               # LSTM 层数
        dropout: float = 0.1,
    ):
        super().__init__()
        self.group_dims = group_dims
        self.node_hidden_dim = node_hidden_dim

        # 系统特征编码器，把系统特征映射到节点特征空间
        self.system_encoder = MLPEncoder(group_dims["system"], max(32, node_hidden_dim), node_hidden_dim, dropout)
        # 车身节点特征编码器，把车身特征映射到节点特征空间
        self.body_encoder = MLPEncoder(group_dims["body"], max(64, node_hidden_dim), node_hidden_dim, dropout)

        wheel_in_dim = group_dims["wheel0_kin"] + group_dims["wheel0_contact"] # 定义单个车轮节点输入维度
        # 车轮节点特征编码器，把车轮特征映射到节点特征空间
        # 所有车轮共享同一个编码器，假设不同车轮的特征形式一致
        self.wheel_encoder = MLPEncoder(wheel_in_dim, max(64, node_hidden_dim), node_hidden_dim, dropout)

        # 构建车辆图边和节点编号映射。include_self_loops=True 表示图中包含自环，即每个节点也连接到自己。
        edges, node_map = build_vehicle_graph_edges(include_self_loops=True)
        self.node_map = node_map
        self.num_nodes = len(node_map)
        # 创建多个图消息传递层。ModuleList 可以让 PyTorch 正确注册这些子模块。每一层都会做一次节点间信息传播。
        self.graph_layers = nn.ModuleList([
            GraphMessagePassing(node_hidden_dim, edges, self.num_nodes) for _ in range(graph_layers)
        ])

        fused_dim = self.num_nodes * node_hidden_dim # 图节点特征融合后的维度，即把所有节点的特征拼接在一起
        # 定义时间卷积模块。输入是展平后的图节点特征，输出是 tcn_hidden_dim 维时序特征。
        self.tcn = TemporalConvBlock(fused_dim, tcn_hidden_dim, kernel_size=3, dropout=dropout)
        # 定义双向 LSTM，用于进一步提取长期时序依赖。
        self.bilstm = nn.LSTM(
            input_size=tcn_hidden_dim,                    # 输入维度=TCN网络输出维度
            hidden_size=lstm_hidden_dim // 2,             # 双向 LSTM，两个方向的输出会拼接。这里每个方向的隐藏维度是输出维度/2
            num_layers=lstm_layers,                       # 层数
            dropout=dropout if lstm_layers > 1 else 0.0,  # 如果层数 > 1 则在层间使用 dropout，否则为 0
            batch_first=True,                             # 输入数据的第一维是 batch_size，即 [B, T, D]
            bidirectional=True,                           # 启用双向
        )
        temporal_dim = lstm_hidden_dim   # 时序特征维度 = LSTM输出维度

        # 定义三个预测头的输入维度。每个预测头都会接收两部分特征：全局时序特征 h_t 和对应节点的局部图特征 node_t。
        body_head_in = temporal_dim + node_hidden_dim
        wheel_head_in = temporal_dim + node_hidden_dim
        contact_head_in = temporal_dim + node_hidden_dim

        # 定义三个预测头
        self.body_head = ResidualHead(body_head_in, group_dims["res_body"], hidden_dim=128, dropout=dropout)
        self.wheel_head = ResidualHead(wheel_head_in, group_dims["res_wheel0_kin"], hidden_dim=128, dropout=dropout)
        self.contact_head = ResidualHead(contact_head_in, group_dims["res_wheel0_contact"], hidden_dim=128, dropout=dropout)

    # 节点编码函数，输入字典batch，按照特征组存放张量，返回 [B, T, N, H]
    def encode_nodes(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 取出各组特征
        sys_feat = batch["system"]
        body_feat = batch["body"]
        wheel_feats = [torch.cat([batch[f"wheel{i}_kin"], batch[f"wheel{i}_contact"]], dim=-1) for i in WHEEL_IDS]
        # 各节点特征编码
        sys_node = self.system_encoder(sys_feat)
        body_node = self.body_encoder(body_feat)
        wheel_nodes = [self.wheel_encoder(wf) for wf in wheel_feats]
        nodes = [sys_node, body_node] + wheel_nodes
        x = torch.stack(nodes, dim=2) # 把节点列表堆叠成一个张量。原来每个节点是 [B,T,H]，堆叠后变成 [B,T,N,H]。
        for layer in self.graph_layers: # 对节点特征进行图消息传递
            x = layer(x)
        return x

    # 主模型前向传播函数
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nodes = self.encode_nodes(batch)      # [B,T,N,H]  把节点特征编码成图结构
        B, T, N, Hn = nodes.shape             # 获取各维度大小
        fused = nodes.reshape(B, T, N * Hn)   # 展平图节点特征，形状从 [B,T,N,H] 变成 [B,T,N*H]。送入时序模型。
        y = self.tcn(fused)                   # 通过时间卷积网络提取时序特征
        y, _ = self.bilstm(y)                 # 通过双向 LSTM 提取长期时序特征
        h_t = y[:, -1, :]                     # 取最后一个时间步的全局时序特征。它代表整个输入序列压缩后的时序表达。
        node_t = nodes[:, -1, :, :]           # 取最后一个时间步的所有图节点特征。形状是 [B,N,H]

        body_feat = torch.cat([h_t, node_t[:, self.node_map["body"], :]], dim=-1) # 车身预测头输入特征是全局时序特征和车身节点特征的拼接
        pred_body = self.body_head(body_feat) # 通过车身预测头预测车身残差

        # 预测每个车轮的残差
        pred_wheel_kin = []
        pred_wheel_contact = []
        for i in WHEEL_IDS:
            node_idx = self.node_map[f"wheel{i}"]
            local_feat = torch.cat([h_t, node_t[:, node_idx, :]], dim=-1)
            pred_wheel_kin.append(self.wheel_head(local_feat))
            pred_wheel_contact.append(self.contact_head(local_feat))

        return {
            "pred_res_body": pred_body,
            **{f"pred_res_wheel{i}_kin": pred_wheel_kin[i] for i in WHEEL_IDS},
            **{f"pred_res_wheel{i}_contact": pred_wheel_contact[i] for i in WHEEL_IDS},
        }

# 四元数归一化
def normalize_quat(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor: 
    return q / torch.sqrt((q ** 2).sum(dim=-1, keepdim=True) + eps)

# 四元数乘法
def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    q1, q2: [..., 4]
    四元数顺序: [q0, q1, q2, q3] = [w, x, y, z]
    返回: q1 ⊗ q2
    """
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)

# 旋转向量转四元数
def rotvec_to_quat(rotvec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    rotvec: [..., 3]
    返回单位四元数 [w, x, y, z]
    """
    angle = torch.sqrt((rotvec ** 2).sum(dim=-1, keepdim=True) + eps)
    half_angle = 0.5 * angle

    axis = rotvec / angle
    qw = torch.cos(half_angle)
    qxyz = axis * torch.sin(half_angle)

    return normalize_quat(torch.cat([qw, qxyz], dim=-1))

# 根据低保真四元数和预测旋转向量计算高保真四元数
def apply_rotvec_to_quat(
    lf_quat: torch.Tensor,
    rotvec: torch.Tensor,
    left_multiply: bool = True,
) -> torch.Tensor:
    """
    lf_quat: [B, 4]
    rotvec: [B, 3]
    返回预测高保真四元数

    left_multiply=True:
        q_pred = delta_q ⊗ q_lf

    left_multiply=False:
        q_pred = q_lf ⊗ delta_q
    """
    lf_quat = normalize_quat(lf_quat)
    delta_q = rotvec_to_quat(rotvec)

    if left_multiply:
        pred_q = quat_multiply(delta_q, lf_quat)
    else:
        pred_q = quat_multiply(lf_quat, delta_q)

    return normalize_quat(pred_q)

# 损失函数选择器。输入是预测值、目标值，以及是否使用 Huber 损失
def huber_or_mse(pred: torch.Tensor, target: torch.Tensor, use_huber: bool = True) -> torch.Tensor:
    # 如果 use_huber=True，使用 Huber loss。
    # 如果为 False，使用 MSE loss。
    # Huber loss 对异常值更稳健，适合高低保真残差中存在突变或局部接触冲击的情况
    return F.huber_loss(pred, target) if use_huber else F.mse_loss(pred, target)

# # 四元数归一化损失函数，约束预测状态中的四元数模长接近 1
# def quat_norm_loss_from_state(pred_state: torch.Tensor, quat_indices: List[int]) -> torch.Tensor:
#     if len(quat_indices) != 4:
#         return pred_state.new_tensor(0.0)
#     q = pred_state[:, quat_indices]
#     norm = torch.sqrt((q ** 2).sum(dim=-1) + 1e-8)
#     return torch.mean(torch.abs(norm - 1.0))

# 四元数间距损失函数，约束预测状态中的四元数与目标四元数之间的距离
def quat_geodesic_loss(pred_q: torch.Tensor, target_q: torch.Tensor) -> torch.Tensor:
    """
    pred_q, target_q: shape = [B, 4]
    四元数格式默认是 [q0, q1, q2, q3]
    """
    if pred_q.numel() == 0 or target_q.numel() == 0:
        return pred_q.new_tensor(0.0)

    pred_q = pred_q / torch.sqrt((pred_q ** 2).sum(dim=-1, keepdim=True) + 1e-8)
    target_q = target_q / torch.sqrt((target_q ** 2).sum(dim=-1, keepdim=True) + 1e-8)

    dot = torch.sum(pred_q * target_q, dim=-1).abs()
    dot = torch.clamp(dot, 0.0, 1.0)

    return torch.mean(1.0 - dot)

# 有限差分一致性损失。用于约束位置、速度、加速度之间满足基本运动学关系。
def finite_diff_consistency_loss(pos: torch.Tensor, vel: torch.Tensor, acc: torch.Tensor, dt: float) -> torch.Tensor:
    # pos/vel/acc: [B, T, D]
    if pos.shape[1] < 2:
        return pos.new_tensor(0.0)
    vel_fd = (pos[:, 1:] - pos[:, :-1]) / dt
    acc_fd = (vel[:, 1:] - vel[:, :-1]) / dt
    return F.mse_loss(vel_fd, vel[:, 1:]) + F.mse_loss(acc_fd, acc[:, 1:])

# 平滑性损失，用于抑制序列预测的高频抖动。
def smoothness_loss(seq: torch.Tensor) -> torch.Tensor:
    if seq.shape[1] < 2:
        return seq.new_tensor(0.0)
    return torch.mean(torch.abs(seq[:, 1:] - seq[:, :-1]))


