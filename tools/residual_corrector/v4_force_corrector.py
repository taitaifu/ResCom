"""Feed-forward residual corrector attached to a frozen V4 Student."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class V4ForceCorrector(nn.Module):
    """Shared wheel MLP with independent Fx/Fy/Fz residual heads."""

    def __init__(self, z_force_dim: int, hidden_dim: int = 128, dropout: float = .1) -> None:
        super().__init__()
        self.z_force_dim = int(z_force_dim)
        self.input_dim = self.z_force_dim + 3
        self.shared = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.fx_head = nn.Linear(hidden_dim, 1)
        self.fy_head = nn.Linear(hidden_dim, 1)
        self.fz_head = nn.Linear(hidden_dim, 1)

    def forward(self, z_force: torch.Tensor, force_raw: torch.Tensor) -> torch.Tensor:
        if z_force.ndim != 3 or z_force.shape[1] != 6 or z_force.shape[2] != self.z_force_dim:
            raise ValueError(f"z_force must be [B,6,{self.z_force_dim}], got {tuple(z_force.shape)}")
        if force_raw.shape != (z_force.shape[0], 6, 3):
            raise ValueError(f"force_raw must be [B,6,3], got {tuple(force_raw.shape)}")
        shared = self.shared(torch.cat([z_force, force_raw], dim=-1))
        return torch.cat([self.fx_head(shared), self.fy_head(shared), self.fz_head(shared)], dim=-1)


def corrected_force_loss(force_raw: torch.Tensor, residual_pred: torch.Tensor, force_hf: torch.Tensor, scale: torch.Tensor | None = None) -> torch.Tensor:
    corrected = force_raw + residual_pred
    if scale is not None:
        scale = scale.to(device=corrected.device, dtype=corrected.dtype).clamp_min(1e-6)
        if scale.ndim == 1:
            scale = scale.reshape(1, 1, 3)
        elif scale.ndim == 2:
            scale = scale.reshape(1, 6, 3)
        else:
            raise ValueError(f"force scale must be [3] or [6,3], got {tuple(scale.shape)}")
        corrected, force_hf = corrected / scale, force_hf / scale
    return F.huber_loss(corrected, force_hf, reduction="mean")


def freeze_v4_student(student: nn.Module) -> None:
    student.eval()
    for parameter in student.parameters():
        parameter.requires_grad_(False)
