from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(self, hidden_dim: int = 64, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv = CausalConv1d(hidden_dim, kernel_size=kernel_size, dilation=dilation)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = x.transpose(1, 2)
        y = self.conv(y).transpose(1, 2)
        y = self.norm(y)
        y = F.gelu(y)
        y = self.dropout(y)
        return residual + y


class ResidualTCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.1,
        wheel_embedding_dim: int = 8,
        num_wheels: int = 6,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.wheel_embedding_dim = int(wheel_embedding_dim)
        self.wheel_embedding = nn.Embedding(num_wheels, wheel_embedding_dim) if wheel_embedding_dim > 0 else None
        in_dim = self.input_dim + (wheel_embedding_dim if self.wheel_embedding is not None else 0)
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [TCNBlock(hidden_dim, kernel_size=kernel_size, dilation=d, dropout=dropout) for d in dilations]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x: torch.Tensor, wheel_id: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"ResidualTCN expects [B, T, C], got {tuple(x.shape)}")
        if self.wheel_embedding is not None:
            if wheel_id is None:
                raise ValueError("wheel_id is required when wheel_embedding_dim > 0")
            emb = self.wheel_embedding(wheel_id.long()).unsqueeze(1).expand(-1, x.shape[1], -1)
            x = torch.cat([x, emb], dim=-1)
        y = self.input_proj(x)
        for block in self.blocks:
            y = block(y)
        return self.head(y[:, -1])
