"""Learning curve: train at different dataset fractions, measure test performance.

Trains the model at e.g. 10%, 25%, 50%, 75%, 100% of training data and plots
test accuracy vs number of samples to determine if the model is data-limited.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from train import (
    CovarianceAwareLoss,
    CovarianceSurrogateModel,
    build_model,
    chol_vectors_to_covariance,
    get_feature_target_columns,
    run_epoch,
)


def train_at_fraction(
    fraction: float,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols,
    chol_cols,
    mean_cols,
    args,
    device,
    seed: int,
):
    """Train model on a fraction of training data, return test metrics."""
    rng = np.random.RandomState(seed)

    # Subsample training data
    n_total = len(train_df)
    n_use = max(1, int(n_total * fraction))
    idx = rng.choice(n_total, size=n_use, replace=False)
    sub_df = train_df.iloc[idx]

    # Compute scalers from subsampled training data
    X_train_raw = sub_df[feature_cols].values.astype(np.float32)
    y_train_raw = sub_df[chol_cols].values.astype(np.float32)

    x_mean = X_train_raw.mean(axis=0)
    x_std = X_train_raw.std(axis=0)
    x_std[x_std == 0] = 1.0
    y_mean = y_train_raw.mean(axis=0)
    y_std = y_train_raw.std(axis=0)
    y_std[y_std == 0] = 1.0

    # Covariance normalizers
    y_train_t = torch.from_numpy(y_train_raw)
    train_cov = chol_vectors_to_covariance(y_train_t).numpy().reshape(-1, 36)
    cov_mean = train_cov.mean(axis=0).astype(np.float32)
    cov_std = train_cov.std(axis=0).astype(np.float32)
    cov_std[cov_std == 0] = 1.0

    # Mean beam scalers
    has_mean = len(mean_cols) > 0
    if has_mean:
        mean_train_raw = sub_df[mean_cols].values.astype(np.float32)
        mean_y_mean = mean_train_raw.mean(axis=0)
        mean_y_std = mean_train_raw.std(axis=0)
        mean_y_std[mean_y_std == 0] = 1.0
    else:
        mean_y_mean = np.zeros(0, dtype=np.float32)
        mean_y_std = np.ones(0, dtype=np.float32)

    def make_dataset(df):
        X = (df[feature_cols].values.astype(np.float32) - x_mean) / x_std
        y_chol = (df[chol_cols].values.astype(np.float32) - y_mean) / y_std
        tensors = [torch.from_numpy(X), torch.from_numpy(y_chol)]
        if has_mean:
            y_m = (df[mean_cols].values.astype(np.float32) - mean_y_mean) / mean_y_std
            tensors.append(torch.from_numpy(y_m))
        return TensorDataset(*tensors)

    train_ds = make_dataset(sub_df)
    val_ds = make_dataset(val_df)
    test_ds = make_dataset(test_df)

    # Adjust batch size if subset is very small
    batch_size = min(args.batch_size, n_use)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # Build model
    n_inputs = len(feature_cols)
    n_chol = len(chol_cols)
    n_mean = len(mean_cols)
    model = build_model(
        n_inputs, n_chol, n_mean_outputs=n_mean,
        y_mean=torch.from_numpy(y_mean), y_std=torch.from_numpy(y_std),
        mean_y_mean=torch.from_numpy(mean_y_mean) if has_mean else None,
        mean_y_std=torch.from_numpy(mean_y_std) if has_mean else None,
    ).to(device)

    criterion = CovarianceAwareLoss(
        model,
        cov_mean=torch.from_numpy(cov_mean).reshape(6, 6).to(device),
        cov_std=torch.from_numpy(cov_std).reshape(6, 6).to(device),
        cov_loss=args.cov_loss,
        mean_loss_weight=args.mean_loss_weight,
        has_mean_outputs=has_mean,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )

    # Train with early stopping
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, None, device, train=False)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if args.patience > 0 and patience_counter >= args.patience:
            break

    # Fine-tuning stages
    if args.finetune_batch_sizes and best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        ft_lr = args.finetune_lr
        for ft_bs in args.finetune_batch_sizes:
            ft_bs = min(ft_bs, n_use)
            ft_loader = DataLoader(train_ds, batch_size=ft_bs, shuffle=True)
            ft_optimizer = torch.optim.Adam(model.parameters(), lr=ft_lr)
            ft_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                ft_optimizer, mode="min", factor=0.5,
                patience=args.finetune_plateau_patience,
                min_lr=args.finetune_min_lr,
            )
            for _ in range(args.finetune_epochs_per_stage):
                run_epoch(model, ft_loader, criterion, ft_optimizer, device, train=True)
                vl = run_epoch(model, val_loader, criterion, None, device, train=False)
                ft_scheduler.step(vl)
                if vl < best_val_loss:
                    best_val_loss = vl
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            ft_lr *= args.finetune_lr_decay

    # Evaluate best model on test set
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    test_loss = run_epoch(model, test_loader, criterion, None, device, train=False)

    # Compute MAPE on test covariance predictions
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            X_batch = batch[0].to(device)
            output = model(X_batch)
            pred_cov = output[0].cpu().numpy() if has_mean else output.cpu().numpy()
            all_preds.append(pred_cov)
    preds_cov = np.concatenate(all_preds)

    # True covariance from test set raw Cholesky
    y_test_raw = test_df[chol_cols].values.astype(np.float32)
    targets_cov = chol_vectors_to_covariance(torch.from_numpy(y_test_raw)).numpy()

    preds_flat = preds_cov.reshape(-1, 36)
    targets_flat = targets_cov.reshape(-1, 36)

    abs_error = np.abs(preds_flat - targets_flat)
    abs_true = np.abs(targets_flat)
    target_scale = abs_true.std(axis=0)
    pct_mask = abs_true > 0.01 * target_scale[None, :]
    pct_errors = np.where(pct_mask, abs_error / abs_true * 100, np.nan)
    mape = np.nanmean(pct_errors)

    # R² per element
    ss_res = np.sum((preds_flat - targets_flat) ** 2, axis=0)
    ss_tot = np.sum((targets_flat - targets_flat.mean(axis=0)) ** 2, axis=0)
    r_squared = 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, 1.0)
    mean_r2 = np.mean(r_squared)

    return {
        "n_samples": n_use,
        "fraction": fraction,
        "test_loss": test_loss,
        "mape": mape,
        "mean_r2": mean_r2,
        "best_val_loss": best_val_loss,
    }


def main():
    parser = argparse.ArgumentParser(description="Learning curve: samples vs accuracy.")
    parser.add_argument("--train-csv", default="dataset-train.csv")
    parser.add_argument("--val-csv", default="dataset-val.csv")
    parser.add_argument("--test-csv", default="dataset-test.csv")
    parser.add_argument("--output-dir", default="learning-curve-571")
    parser.add_argument(
        "--fractions", type=float, nargs="+",
        default=[0.1, 0.25, 0.5, 0.75, 1.0],
        help="Fractions of training data to use (default: 0.1 0.25 0.5 0.75 1.0)",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cov-loss", choices=["mse", "l1"], default="l1")
    parser.add_argument("--mean-loss-weight", type=float, default=1.0)
    parser.add_argument("--finetune-batch-sizes", type=int, nargs="+", default=[32, 8, 2])
    parser.add_argument("--finetune-epochs-per-stage", type=int, default=300)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--finetune-lr-decay", type=float, default=0.5)
    parser.add_argument("--finetune-plateau-patience", type=int, default=5)
    parser.add_argument("--finetune-min-lr", type=float, default=1e-6)
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: fewer epochs, no finetuning (for testing the script)",
    )
    args = parser.parse_args()

    if args.quick:
        args.epochs = 30
        args.patience = 10
        args.finetune_batch_sizes = None
        args.finetune_epochs_per_stage = 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run] Device: {device}", flush=True)

    # Load data
    print("[run] Loading datasets ...", flush=True)
    train_df = pd.read_csv(args.train_csv, low_memory=False)
    val_df = pd.read_csv(args.val_csv, low_memory=False)
    test_df = pd.read_csv(args.test_csv, low_memory=False)

    feature_cols, target_cols, chol_cols, mean_cols = get_feature_target_columns(train_df)
    print(f"[run] Training set: {len(train_df)} samples", flush=True)
    print(f"[run] Features: {len(feature_cols)}, Chol targets: {len(chol_cols)}, Mean targets: {len(mean_cols)}")
    print(f"[run] Fractions to evaluate: {args.fractions}", flush=True)
    print(f"[run] Fine-tuning: {'enabled' if args.finetune_batch_sizes else 'disabled'}", flush=True)

    # Run training at each fraction
    results = []
    for frac in sorted(args.fractions):
        n_use = max(1, int(len(train_df) * frac))
        print(f"\n{'='*60}", flush=True)
        print(f"[run] Training with {frac*100:.0f}% of data ({n_use} samples) ...", flush=True)
        print(f"{'='*60}", flush=True)
        t0 = time.time()

        result = train_at_fraction(
            fraction=frac,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            feature_cols=feature_cols,
            chol_cols=chol_cols,
            mean_cols=mean_cols,
            args=args,
            device=device,
            seed=args.seed,
        )
        elapsed = time.time() - t0
        result["elapsed_sec"] = elapsed
        results.append(result)

        print(f"[result] n={result['n_samples']:>6d}  "
              f"MAPE={result['mape']:.2f}%  "
              f"R²={result['mean_r2']:.4f}  "
              f"loss={result['test_loss']:.6f}  "
              f"({elapsed:.0f}s)", flush=True)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "learning_curve_results.csv", index=False)
    print(f"\n[run] Results saved to {output_dir}/learning_curve_results.csv")

    # Plot learning curve
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # MAPE vs samples
    ax = axes[0]
    ax.plot(results_df["n_samples"], results_df["mape"], "o-", color="steelblue", linewidth=2)
    ax.set_xlabel("Number of Training Samples")
    ax.set_ylabel("Test MAPE (%)")
    ax.set_title("Learning Curve: MAPE")
    ax.grid(True, alpha=0.3)

    # R² vs samples
    ax = axes[1]
    ax.plot(results_df["n_samples"], results_df["mean_r2"], "o-", color="forestgreen", linewidth=2)
    ax.set_xlabel("Number of Training Samples")
    ax.set_ylabel("Mean R²")
    ax.set_title("Learning Curve: R²")
    ax.grid(True, alpha=0.3)

    # Test loss vs samples
    ax = axes[2]
    ax.plot(results_df["n_samples"], results_df["test_loss"], "o-", color="tomato", linewidth=2)
    ax.set_xlabel("Number of Training Samples")
    ax.set_ylabel("Test Loss")
    ax.set_title("Learning Curve: Test Loss")
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Learning Curve (loss={args.cov_loss}, total_train={len(train_df)})",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "learning_curve.png", dpi=150, bbox_inches="tight")
    print(f"[run] Plot saved to {output_dir}/learning_curve.png")

    # Summary
    print(f"\n{'='*60}")
    print("LEARNING CURVE SUMMARY")
    print(f"{'='*60}")
    print(results_df[["n_samples", "fraction", "mape", "mean_r2", "test_loss", "elapsed_sec"]].to_string(index=False))

    # Check if data-limited
    if len(results) >= 2:
        last_mape = results[-1]["mape"]
        second_last_mape = results[-2]["mape"]
        improvement = second_last_mape - last_mape
        if improvement > 1.0:
            print(f"\n[conclusion] MAPE still improving at 100% data "
                  f"(Δ={improvement:.1f}% from {results[-2]['fraction']*100:.0f}%→100%). "
                  f"Model is likely DATA-LIMITED — more samples should help.")
        else:
            print(f"\n[conclusion] MAPE plateaued "
                  f"(Δ={improvement:.1f}% from {results[-2]['fraction']*100:.0f}%→100%). "
                  f"Model may be CAPACITY-LIMITED — try larger architecture or better loss.")

    print(f"\n[run] Done. Results in {output_dir}/")


if __name__ == "__main__":
    main()
