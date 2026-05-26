from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models.data_utils import (  # noqa: E402
    WHEEL_IDS,
    GraphTemporalSequenceDataset,
    GroupStandardizer,
    get_group_dims,
    graph_temporal_collate_fn,
    load_column_spec,
    load_merged_dataset,
    split_train_val_test_by_traj,
)
from models.graph_temporal_compensation import GraphTemporalCompensationModel  # noqa: E402


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def inverse_group(scaler: GroupStandardizer, arr: np.ndarray, name: str) -> np.ndarray:
    return scaler.inverse_transform_group(arr, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", type=str, default=str(ROOT / "Feature_Selection" / "DataSet"))
    parser.add_argument("--merged_csv", type=str, default=str(ROOT / "Feature_Selection" / "DataSet" / "merged_error_dataset.csv"))
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--scaler", type=str, default="")
    parser.add_argument("--split", type=str, choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--pred_horizon", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_csv", type=str, default=str(ROOT / "infer" / "predictions.csv"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = load_column_spec(args.feature_dir)
    df = load_merged_dataset(args.merged_csv, spec)
    df_train, df_val, df_test = split_train_val_test_by_traj(df, seed=args.seed)

    if args.split == "train":
        df_use = df_train
    elif args.split == "val":
        df_use = df_val
    elif args.split == "test":
        df_use = df_test
    else:
        df_use = df

    scaler_path = args.scaler if args.scaler else os.path.join(os.path.dirname(args.ckpt), "group_scaler.joblib")
    scaler = GroupStandardizer.load(scaler_path)

    dataset = GraphTemporalSequenceDataset(
        df=df_use,
        spec=spec,
        scaler=scaler,
        seq_len=args.seq_len,
        pred_horizon=args.pred_horizon,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=graph_temporal_collate_fn)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    group_dims = ckpt.get("group_dims", get_group_dims(spec))
    train_args = ckpt.get("args", {})
    model = GraphTemporalCompensationModel(
        group_dims=group_dims,
        node_hidden_dim=train_args.get("hidden_dim", 64),
        tcn_hidden_dim=train_args.get("tcn_dim", 256),
        lstm_hidden_dim=train_args.get("lstm_dim", 256),
        lstm_layers=train_args.get("lstm_layers", 2),
        dropout=train_args.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rows: List[Dict] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            out = model(batch)

            pred_res_body = out["pred_res_body"].cpu().numpy()
            pred_res_body_raw = inverse_group(scaler, pred_res_body, "res_body")
            hf_body_pred_raw = batch["lf_body_current"].cpu().numpy() + pred_res_body_raw

            for b in range(pred_res_body.shape[0]):
                row = {
                    "traj_id": batch["traj_id"][b],
                    "time": float(batch["time"][b].cpu().item()) if isinstance(batch["time"], torch.Tensor) else float(batch["time"][b]),
                }
                for j, c in enumerate(spec.res_groups.body_cols):
                    row[f"pred_{c}"] = float(pred_res_body_raw[b, j])
                for j, c in enumerate(spec.target_groups.body_cols):
                    row[f"pred_{c}"] = float(hf_body_pred_raw[b, j])

                for i in WHEEL_IDS:
                    pred_res_wk = out[f"pred_res_wheel{i}_kin"].cpu().numpy()
                    pred_res_wc = out[f"pred_res_wheel{i}_contact"].cpu().numpy()
                    pred_res_wk_raw = inverse_group(scaler, pred_res_wk, f"res_wheel{i}_kin")
                    pred_res_wc_raw = inverse_group(scaler, pred_res_wc, f"res_wheel{i}_contact")
                    hf_wk_pred_raw = batch[f"lf_wheel{i}_kin_current"].cpu().numpy() + pred_res_wk_raw
                    hf_wc_pred_raw = batch[f"lf_wheel{i}_contact_current"].cpu().numpy() + pred_res_wc_raw

                    for j, c in enumerate(spec.res_groups.wheel_kin_cols[i]):
                        row[f"pred_{c}"] = float(pred_res_wk_raw[b, j])
                    for j, c in enumerate(spec.res_groups.wheel_contact_cols[i]):
                        row[f"pred_{c}"] = float(pred_res_wc_raw[b, j])
                    for j, c in enumerate(spec.target_groups.wheel_kin_cols[i]):
                        row[f"pred_{c}"] = float(hf_wk_pred_raw[b, j])
                    for j, c in enumerate(spec.target_groups.wheel_contact_cols[i]):
                        row[f"pred_{c}"] = float(hf_wc_pred_raw[b, j])
                rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["traj_id", "time"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"推理完成，结果已保存到: {args.out_csv}")


if __name__ == "__main__":
    main()
