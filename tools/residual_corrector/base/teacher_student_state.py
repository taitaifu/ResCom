from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def hidden_distillation_loss(student_state: torch.Tensor, teacher_state: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(F.normalize(student_state, dim=-1), F.normalize(teacher_state.detach(), dim=-1))


class _ResidualDecoder(nn.Module):
    def __init__(self, state_dim: int, context_dim: int, wheel_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        dim = state_dim + context_dim + wheel_dim
        self.trunk = nn.Sequential(nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.fx = nn.Linear(hidden_dim, 1)
        self.fy = nn.Linear(hidden_dim, 1)
        self.fz = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, context: torch.Tensor, wheel: torch.Tensor) -> torch.Tensor:
        h = self.trunk(torch.cat([state, context, wheel], dim=-1))
        return torch.cat([self.fx(h), self.fy(h), self.fz(h)], dim=-1)


class TeacherStudentResidualCorrector(nn.Module):
    """Shared wheel-cell parameters with separate per-case six-wheel hidden states."""
    def __init__(self, input_dim: int, body_indices: list[int], wheel_state_dim: int = 32, wheel_embedding_dim: int = 8, body_context_dim: int = 32, num_wheels: int = 6) -> None:
        super().__init__()
        self.input_dim, self.wheel_state_dim, self.num_wheels = int(input_dim), int(wheel_state_dim), int(num_wheels)
        self.register_buffer("body_indices", torch.tensor(body_indices, dtype=torch.long), persistent=False)
        wheel_indices = [i for i in range(input_dim) if i not in set(body_indices)]
        self.register_buffer("wheel_indices", torch.tensor(wheel_indices, dtype=torch.long), persistent=False)
        self.body_encoder = nn.Sequential(nn.Linear(len(body_indices), body_context_dim), nn.LayerNorm(body_context_dim), nn.GELU())
        self.wheel_embedding = nn.Embedding(num_wheels, wheel_embedding_dim)
        wheel_dim = len(wheel_indices) + wheel_embedding_dim
        self.student_cell = nn.GRUCell(wheel_dim + body_context_dim, wheel_state_dim)
        self.teacher_cell = nn.GRUCell(wheel_dim + body_context_dim + 3, wheel_state_dim)
        self.student_decoder = _ResidualDecoder(wheel_state_dim, body_context_dim, wheel_dim)
        self.teacher_decoder = _ResidualDecoder(wheel_state_dim, body_context_dim, wheel_dim)

    def _inputs(self, x: torch.Tensor, wheel_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 3:
            x = x[:, -1]
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"expected deployable x [B,{self.input_dim}], got {tuple(x.shape)}")
        context = self.body_encoder(x.index_select(1, self.body_indices))
        wheel = torch.cat([x.index_select(1, self.wheel_indices), self.wheel_embedding(wheel_id.long())], dim=-1)
        return context, wheel

    def _update(self, cell: nn.GRUCell, decoder: _ResidualDecoder, x: torch.Tensor, state: torch.Tensor, wheel_id: torch.Tensor, previous_residual: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if state.ndim != 3 or state.shape[1:] != (self.num_wheels, self.wheel_state_dim):
            raise ValueError(f"state must be [B,{self.num_wheels},{self.wheel_state_dim}]")
        if state.shape[0] != x.shape[0]:
            raise ValueError("state and input batch differ")
        context, wheel = self._inputs(x, wheel_id)
        old = state[torch.arange(len(state), device=state.device), wheel_id.long()]
        cell_input = torch.cat([wheel, context], dim=-1)
        if previous_residual is not None:
            if previous_residual.shape != (len(state), 3):
                raise ValueError("Teacher previous_residual must be [B,3]")
            cell_input = torch.cat([cell_input, previous_residual], dim=-1)
        new = cell(cell_input, old)
        next_state = state.clone()
        next_state[torch.arange(len(state), device=state.device), wheel_id.long()] = new
        return next_state, decoder(new, context, wheel)

    def student_step(self, x: torch.Tensor, student_state: torch.Tensor, wheel_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._update(self.student_cell, self.student_decoder, x, student_state, wheel_id)

    def teacher_step(self, x: torch.Tensor, teacher_state: torch.Tensor, wheel_id: torch.Tensor, previous_residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._update(self.teacher_cell, self.teacher_decoder, x, teacher_state, wheel_id, previous_residual)

    def freeze_teacher(self) -> None:
        for module in (self.body_encoder, self.teacher_cell, self.teacher_decoder):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def student_step_all(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[1] != self.num_wheels:
            raise ValueError(f"x must be [B,{self.num_wheels},C]")
        next_state, predictions = state.clone(), []
        for wheel in range(self.num_wheels):
            ids = torch.full((x.shape[0],), wheel, dtype=torch.long, device=x.device)
            updated, pred = self.student_step(x[:, wheel], next_state, ids)
            next_state[:, wheel] = updated[:, wheel]
            predictions.append(pred)
        return next_state, torch.stack(predictions, dim=1)

    def teacher_step_all(self, x: torch.Tensor, state: torch.Tensor, previous_residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if previous_residual.shape[:2] != x.shape[:2] or previous_residual.shape[-1] != 3:
            raise ValueError("previous_residual must be [B,6,3]")
        next_state, predictions = state.clone(), []
        for wheel in range(self.num_wheels):
            ids = torch.full((x.shape[0],), wheel, dtype=torch.long, device=x.device)
            updated, pred = self.teacher_step(x[:, wheel], next_state, ids, previous_residual[:, wheel])
            next_state[:, wheel] = updated[:, wheel]
            predictions.append(pred)
        return next_state, torch.stack(predictions, dim=1)
