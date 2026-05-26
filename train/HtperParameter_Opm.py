from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import optuna
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from train_graph_model import (
    ROOT,
    set_seed,
    ensure_dir,
    move_batch_to_device,
    compute_losses,
    evaluate,
    get_group_dims,
    graph_temporal_collate_fn,
    prepare_datasets_and_scaler,
    save_column_spec_json,
    GraphTemporalCompensationModel,
)


def suggest_hparams(trial: optuna.Trial, args: argparse.Namespace) -> None:
    """
    定义贝叶斯优化搜索空间。
    这里会直接修改 args 中对应的超参数。
    """

    args.lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    args.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)

    args.hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    args.graph_layers = trial.suggest_int("graph_layers", 1, 3)

    args.tcn_dim = trial.suggest_categorical("tcn_dim", [128, 256, 384])
    args.lstm_dim = trial.suggest_categorical("lstm_dim", [128, 256, 384])
    args.lstm_layers = trial.suggest_int("lstm_layers", 1, 3)

    args.dropout = trial.suggest_float("dropout", 0.0, 0.3)

    args.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    args.seq_len = trial.suggest_categorical("seq_len", [20, 30, 40, 60])

    args.lambda_res = trial.suggest_float("lambda_res", 0.5, 2.0)
    args.lambda_hf = trial.suggest_float("lambda_hf", 0.05, 1.0)
    args.lambda_quat = trial.suggest_float("lambda_quat", 0.0, 0.1)
    args.lambda_contact = trial.suggest_float("lambda_contact", 0.0, 0.1)

    args.lambda_kin = trial.suggest_float("lambda_kin", 0.0, 0.05)
    args.lambda_smooth = trial.suggest_float("lambda_smooth", 0.0, 0.01)

    args.acc_weight = trial.suggest_float("acc_weight", 1.0, 5.0)


def train_one_trial(
    trial: optuna.Trial,
    base_args: argparse.Namespace,
    device: torch.device,
) -> float:
    """
    单次 trial 训练。
    返回验证集目标值，供 Optuna 最小化。
    """

    args = copy.deepcopy(base_args)
    suggest_hparams(trial, args)

    set_seed(args.seed + trial.number)

    trial_name = f"trial_{trial.number:04d}"
    trial_dir = os.path.join(args.save_dir, trial_name)
    ensure_dir(trial_dir)

    tb_dir = os.path.join(trial_dir, "tensorboard")
    writer = SummaryWriter(log_dir=tb_dir)

    with open(os.path.join(trial_dir, "trial_params.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    spec, scaler, _, df_val, _, train_ds, val_ds, _ = prepare_datasets_and_scaler(
        feature_dir=args.feature_dir,
        merged_csv_path=args.merged_csv,
        seq_len=args.seq_len,
        pred_horizon=args.pred_horizon,
        seed=args.seed,
    )

    save_column_spec_json(spec, os.path.join(trial_dir, "column_spec.json"))
    scaler.save(os.path.join(trial_dir, "group_scaler.joblib"))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=graph_temporal_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = None
    if val_ds is not None and len(df_val) > 0 and len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=graph_temporal_collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

    group_dims = get_group_dims(spec)

    model = GraphTemporalCompensationModel(
        group_dims=group_dims,
        node_hidden_dim=args.hidden_dim,
        graph_layers=args.graph_layers,
        tcn_hidden_dim=args.tcn_dim,
        lstm_hidden_dim=args.lstm_dim,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.scheduler_patience,
    )

    best_value = float("inf")
    best_epoch = 0
    history = []

    best_path = os.path.join(trial_dir, "best_model.pt")

    writer.add_text("config/args", json.dumps(vars(args), ensure_ascii=False, indent=2), 0)
    writer.add_text("config/group_dims", json.dumps(group_dims, ensure_ascii=False, indent=2), 0)
    writer.add_scalar("data/train_samples", len(train_ds), 0)
    writer.add_scalar("data/val_samples", len(val_ds) if val_ds is not None else 0, 0)

    for epoch in range(1, args.epochs + 1):
        model.train()

        meter = {
            "total": 0.0,
            "res": 0.0,
            "hf": 0.0,
            "quat": 0.0,
            "contact": 0.0,
            "kin": 0.0,
            "smooth": 0.0,
        }
        n = 0

        pbar = tqdm(
            train_loader,
            desc=f"{trial_name} Epoch {epoch:03d}/{args.epochs:03d}",
            leave=False,
            dynamic_ncols=True,
        )

        for batch in pbar:
            batch = move_batch_to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)

            output = model(batch)

            losses = compute_losses(
                batch,
                output,
                spec,
                scaler,
                lambda_res=args.lambda_res,
                lambda_hf=args.lambda_hf,
                lambda_quat=args.lambda_quat,
                lambda_contact=args.lambda_contact,
                lambda_kin=args.lambda_kin,
                lambda_smooth=args.lambda_smooth,
                acc_weight=args.acc_weight,
                dt=args.dt,
            )

            losses["total"].backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.grad_clip,
            )

            optimizer.step()

            bs = batch["res_body"].shape[0]
            n += bs

            for k in meter:
                meter[k] += float(losses[k].item()) * bs

            avg_total = meter["total"] / max(n, 1)
            avg_res = meter["res"] / max(n, 1)
            avg_hf = meter["hf"] / max(n, 1)

            pbar.set_postfix(
                {
                    "total": f"{avg_total:.5f}",
                    "res": f"{avg_res:.5f}",
                    "hf": f"{avg_hf:.5f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
            )

        train_metrics = {k: v / max(n, 1) for k, v in meter.items()}

        if val_loader is not None:
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                spec,
                scaler,
                lambda_res=args.lambda_res,
                lambda_hf=args.lambda_hf,
                lambda_quat=args.lambda_quat,
                lambda_contact=args.lambda_contact,
                lambda_kin=args.lambda_kin,
                lambda_smooth=args.lambda_smooth,
                acc_weight=args.acc_weight,
                dt=args.dt,
                desc=f"{trial_name} Val",
            )
        else:
            val_metrics = train_metrics.copy()

        scheduler.step(val_metrics["total"])

        if args.opt_metric == "total":
            objective_value = val_metrics["total"]
        elif args.opt_metric == "hf":
            objective_value = val_metrics["hf"]
        elif args.opt_metric == "res":
            objective_value = val_metrics["res"]
        else:
            raise ValueError(f"未知优化指标: {args.opt_metric}")

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "objective": objective_value,
            }
        )

        current_lr = optimizer.param_groups[0]["lr"]

        writer.add_scalar("lr", current_lr, epoch)

        for name, value in train_metrics.items():
            writer.add_scalar(f"loss/train_{name}", value, epoch)

        for name, value in val_metrics.items():
            writer.add_scalar(f"loss/val_{name}", value, epoch)

        writer.add_scalar("objective/value", objective_value, epoch)

        print(
            f"[{trial_name}] Epoch {epoch:03d}/{args.epochs} | "
            f"train_total={train_metrics['total']:.6f} | "
            f"val_total={val_metrics['total']:.6f} | "
            f"val_res={val_metrics['res']:.6f} | "
            f"val_hf={val_metrics['hf']:.6f} | "
            f"objective={objective_value:.6f}"
        )

        trial.report(objective_value, epoch)

        if trial.should_prune():
            writer.flush()
            writer.close()

            with open(os.path.join(trial_dir, "history.json"), "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)

            raise optuna.exceptions.TrialPruned()

        if objective_value < best_value:
            best_value = objective_value
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "group_dims": group_dims,
                    "args": vars(args),
                    "best_value": best_value,
                    "best_epoch": best_epoch,
                    "opt_metric": args.opt_metric,
                    "trial_number": trial.number,
                    "trial_params": trial.params,
                },
                best_path,
            )

            writer.add_scalar("best/value", best_value, epoch)
            writer.add_scalar("best/epoch", best_epoch, epoch)

        if epoch - best_epoch >= args.patience:
            print(
                f"[{trial_name}] Early stopping at epoch {epoch}, "
                f"best epoch = {best_epoch}, best value = {best_value:.6f}"
            )
            break

    writer.flush()
    writer.close()

    with open(os.path.join(trial_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    return best_value


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feature_dir",
        type=str,
        default=str(ROOT / "Feature_Selection" / "DataSet"),
    )
    parser.add_argument(
        "--merged_csv",
        type=str,
        default=str(ROOT / "Feature_Selection" / "DataSet" / "merged_error_dataset.csv"),
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(ROOT / "results" / "Parameter_Optimization"),
    )

    parser.add_argument("--pred_horizon", type=int, default=1)        # 预测时间步长
    parser.add_argument("--epochs", type=int, default=50)             # 最大训练轮数
    parser.add_argument("--n_trials", type=int, default=30)           # 超参数组合数
    parser.add_argument("--num_workers", type=int, default=0)         # 数据加载线程数
    parser.add_argument("--seed", type=int, default=42)               # 随机种子
    parser.add_argument("--patience", type=int, default=10)           # 耐心值，训练过程中如果验证集指标不再提升，则提前停止训练
    parser.add_argument("--scheduler_patience", type=int, default=5)  # 学习率调度器的耐心值，如果在一定轮数内验证集指标不再提升，则降低学习率


    parser.add_argument("--dt", type=float, default=0.015)
    parser.add_argument("--grad_clip", type=float, default=5.0)       # 梯度裁剪阈值

    parser.add_argument(
        "--opt_metric",
        type=str,
        default="hf",
        choices=["total", "hf", "res"],
        help="Optuna 优化目标，可选 total、hf、res",
    )

    parser.add_argument(
        "--study_name",
        type=str,
        default="graph_temporal_hparam_search",
    )

    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///optuna_graph_temporal.db",                       # 只要 storage 和 study_name 一致，再次运行时，Optuna 就会加载已有 study，然后继续新增 trial
        help="例如 sqlite:///optuna_graph_temporal.db；为空则使用内存数据库",
    )

    # 下列参数会被搜索空间覆盖
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--graph_layers", type=int, default=2)

    parser.add_argument("--tcn_dim", type=int, default=256)
    parser.add_argument("--lstm_dim", type=int, default=256)
    parser.add_argument("--lstm_layers", type=int, default=2)

    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=30)

    parser.add_argument("--lambda_res", type=float, default=1.0)
    parser.add_argument("--lambda_hf", type=float, default=0.1)
    parser.add_argument("--lambda_quat", type=float, default=0.05)
    parser.add_argument("--lambda_contact", type=float, default=0.05)
    parser.add_argument("--lambda_kin", type=float, default=0.0)
    parser.add_argument("--lambda_smooth", type=float, default=0.0)
    parser.add_argument("--acc_weight", type=float, default=2.0)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.save_dir = os.path.join(args.save_dir, run_name)
    ensure_dir(args.save_dir)

    with open(os.path.join(args.save_dir, "base_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        n_startup_trials=5,
        multivariate=True,
    )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=8,
        interval_steps=1,
    )

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=True if args.storage is not None else False,
    )

    def objective(trial: optuna.Trial) -> float:
        return train_one_trial(
            trial=trial,
            base_args=args,
            device=device,
        )

    study.optimize(
        objective,
        n_trials=args.n_trials,
        gc_after_trial=True,
    )

    best_result: Dict[str, Any] = {
        "best_value": study.best_value,
        "best_trial_number": study.best_trial.number,
        "best_params": study.best_trial.params,
        "opt_metric": args.opt_metric,
    }

    with open(os.path.join(args.save_dir, "best_result.json"), "w", encoding="utf-8") as f:
        json.dump(best_result, f, ensure_ascii=False, indent=2)

    df = study.trials_dataframe()
    df.to_csv(os.path.join(args.save_dir, "optuna_trials.csv"), index=False)

    print("=" * 80)
    print("超参数优化完成")
    print(f"优化指标: {args.opt_metric}")
    print(f"最佳 trial: {study.best_trial.number}")
    print(f"最佳目标值: {study.best_value:.8f}")
    print("最佳超参数:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    print(f"结果保存目录: {args.save_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()