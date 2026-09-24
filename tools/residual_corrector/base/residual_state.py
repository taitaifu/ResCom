from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .residual_tcn import TCNBlock


class ResidualStateTargetEncoder(nn.Module):
    def __init__(self, state_dim: int = 64, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, residual_scaled: torch.Tensor) -> torch.Tensor:
        return self.net(residual_scaled)


class StateResidualCorrector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        state_dim: int = 64,
        hidden_dim: int = 64,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.1,
        wheel_embedding_dim: int = 8,
        num_wheels: int = 6,
        enable_target_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.state_dim = int(state_dim)
        self.wheel_embedding_dim = int(wheel_embedding_dim)
        self.wheel_embedding = nn.Embedding(num_wheels, wheel_embedding_dim) if wheel_embedding_dim > 0 else None
        in_dim = self.input_dim + (wheel_embedding_dim if self.wheel_embedding is not None else 0)
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([TCNBlock(hidden_dim, kernel_size=kernel_size, dilation=d, dropout=dropout) for d in dilations])
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + self.state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.state_dim),
        )
        self.fx_head = nn.Sequential(nn.Linear(self.state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.fy_head = nn.Sequential(nn.Linear(self.state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.fz_head = nn.Sequential(nn.Linear(self.state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.target_encoder = ResidualStateTargetEncoder(state_dim=self.state_dim, hidden_dim=hidden_dim) if enable_target_encoder else None

    def encode_true_state(self, residual_scaled: torch.Tensor) -> torch.Tensor:
        if self.target_encoder is None:
            raise RuntimeError("ResidualStateTargetEncoder is only enabled for training supervision")
        return self.target_encoder(residual_scaled)

    def forward(self, x: torch.Tensor, z_prev: torch.Tensor, wheel_id: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"StateResidualCorrector expects x [B, T, C], got {tuple(x.shape)}")
        if z_prev.ndim != 2 or z_prev.shape[-1] != self.state_dim:
            raise ValueError(f"z_prev must be [B, {self.state_dim}], got {tuple(z_prev.shape)}")
        if self.wheel_embedding is not None:
            if wheel_id is None:
                raise ValueError("wheel_id is required when wheel_embedding_dim > 0")
            emb = self.wheel_embedding(wheel_id.long()).unsqueeze(1).expand(-1, x.shape[1], -1)
            x = torch.cat([x, emb], dim=-1)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h_now = h[:, -1]
        z_hat = self.fusion(torch.cat([h_now, z_prev], dim=-1))
        residual_hat = torch.cat([self.fx_head(z_hat), self.fy_head(z_hat), self.fz_head(z_hat)], dim=-1)
        return z_hat, residual_hat
