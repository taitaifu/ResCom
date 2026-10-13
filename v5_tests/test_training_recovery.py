from types import SimpleNamespace
import random

import numpy as np
import torch

from train.train_graph_model_v5 import (
    accumulation_sample_weight,
    format_cuda_debug_postfix,
    restore_training_checkpoint,
    save_epoch_resume_checkpoint,
)
from models.graph_temporal_hgt_compensation_v5 import HGTGraphTemporalCompensationModelV5


def test_accumulation_weight_normalizes_full_and_partial_groups():
    weights = [
        accumulation_sample_weight(i, batch_size, 64, 129, 2, 3)
        for i, batch_size in enumerate((64, 64, 1))
    ]
    assert weights == [0.5, 0.5, 1.0]


def test_restore_checkpoint_recovers_next_epoch_and_best_scores():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    model(torch.ones(2, 2)).sum().backward()
    optimizer.step()
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    expected_torch_random = torch.rand(1)
    checkpoint = {
        "model_state_dict": {k: torch.full_like(v, 3.0) for k, v in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": 4,
        "best_metric": 0.7,
        "best_state_output": 0.8,
        "best_rollout": 0.6,
        "rng_state": rng_state,
    }
    torch.rand(1)
    start_epoch, best = restore_training_checkpoint(model, optimizer, checkpoint)
    assert start_epoch == 5
    assert best == {"total": 0.7, "state_output": 0.8, "rollout": 0.6}
    assert all(torch.equal(value, torch.full_like(value, 3.0)) for value in model.state_dict().values())
    assert torch.equal(torch.rand(1), expected_torch_random)


def test_complete_epoch_checkpoint_is_written_atomically(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model(torch.ones(2, 2)).sum().backward()
    optimizer.step()
    args = SimpleNamespace(
        train_stage="state_student", hidden_dim=4, graph_layers=3, tcn_dim=5,
        lstm_dim=6, lstm_layers=1, history_len=9, teacher_future_len=3,
        body_rollout_len=5, force_hidden_dim=8,
    )
    scaler = SimpleNamespace(scalers={})
    spec = SimpleNamespace(to_dict=lambda: {"version": 5})
    path = tmp_path / "last_epoch.pt"
    save_epoch_resume_checkpoint(
        path, model, optimizer, scaler, spec, args, 3, None,
        {"total": 0.4, "state_output": 0.5, "rollout": 0.3},
    )
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["resume_format"] == "v5_complete_epoch_v1"
    assert saved["optimizer_state_dict"]
    assert "rng_state" in saved
    assert saved["epoch"] == 3
    assert not path.with_name(path.name + ".tmp").exists()


def test_prefix_lookup_does_not_retain_batch_on_model():
    holder = type("ModelHolder", (), {})()
    batch = {"student_system": torch.ones(1, 2, 3), "system": torch.zeros(1, 2, 3)}
    key = HGTGraphTemporalCompensationModelV5._prefixed_key(holder, "system", "student", batch)
    assert key == "student_system"
    assert not hasattr(holder, "_active_batch")


def test_cuda_debug_summary_fits_progress_postfix():
    context = {
        "profile": {
            "times": {
                "forward": [0.2],
                "forward_rollout_step": [0.01, 0.02],
                "loss_reconstruction": [0.1],
                "backward": [0.3],
                "optimizer_step": [0.004],
            },
            "memory": {
                "allocated_mib": 200.0,
                "reserved_mib": 500.0,
                "peak_allocated_mib": 450.0,
                "peak_reserved_mib": 500.0,
            },
        }
    }
    summary = format_cuda_debug_postfix(context)
    assert summary["fwd_s"] == "0.200"
    assert summary["roll_fwd_s"] == "0.010/0.020"
    assert summary["bwd_s"] == "0.300"
    assert summary["alloc_mib"] == 200.0
    assert summary["reserved_mib"] == 500.0
    assert summary["peak_mib"] == 450.0
