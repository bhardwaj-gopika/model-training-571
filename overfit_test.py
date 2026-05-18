"""Overfit test: train with no dropout / no early stopping to see if the model
can memorize the training data. If train loss → 0, model has enough capacity
and the problem is generalization (need more data). If it can't memorize,
the model is too small.
"""

import argparse
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
    chol_vectors_to_covariance,
    get_feature_target_columns,
)


class OverfitModel(nn.Module):
    """Same architecture as CovarianceSurrogateModel but with configurable dropout."""

    def __init__(self, n_inputs, n_chol_outputs, n_mean_outputs=0,
                 y_mean=None, y_std=None, mean_y_mean=None, mean_y_std=None,
                 dropout=0.0, hidden_mult=1.0):
        super().__init__()
        self.n_chol_outputs = n_chol_outputs
        self.n_mean_outputs = n_mean_outputs

        # Scale hidden layer widths
        def w(base):
            return max(16, int(base * hidden_mult))

        layers = [
            nn.Linear(n_inputs, w(100)), nn.ELU(),
            nn.Linear(w(100), w(200)), nn.ELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers += [nn.Linear(w(200), w(200)), nn.ELU()]
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers += [nn.Linear(w(200), w(300)), nn.ELU()]
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers += [nn.Linear(w(300), w(300)), nn.ELU()]
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers += [nn.Linear(w(300), w(200)), nn.ELU()]
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers += [nn.Linear(w(200), w(100)), nn.ELU()]
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers += [nn.Linear(w(100), w(100)), nn.ELU()]
        layers += [nn.Linear(w(100), w(100)), nn.ELU()]
        self.backbone = nn.Sequential(*layers)

        self.chol_head = nn.Linear(w(100), n_chol_outputs)

        if y_mean is None:
            y_mean = torch.zeros(n_chol_outputs, dtype=torch.float32)
        if y_std is None:
            y_std = torch.ones(n_chol_outputs, dtype=torch.float32)
        self.register_buffer("y_mean", y_mean)
        self.register_buffer("y_std", y_std)

        if n_mean_outputs > 0:
            self.mean_head = nn.Linear(w(100), n_mean_outputs)
            if mean_y_mean is None:
                mean_y_mean = torch.zeros(n_mean_outputs, dtype=torch.float32)
            if mean_y_std is None:
                mean_y_std = torch.ones(n_mean_outputs, dtype=torch.float32)
            self.register_buffer("mean_y_mean", mean_y_mean)
            self.register_buffer("mean_y_std", mean_y_std)
        else:
            self.mean_head = None

    def forward(self, x):
        features = self.backbone(x)
        chol_norm = self.chol_head(features)
        chol_raw = chol_norm * self.y_std + self.y_mean
        batch_size = chol_raw.shape[0]
        L = torch.zeros((batch_size, 6, 6), dtype=chol_raw.dtype, device=chol_raw.device)
        tril_idx = torch.tril_indices(row=6, col=6, offset=0, device=chol_raw.device)
        L[:, tril_idx[0], tril_idx[1]] = chol_raw
        cov = L @ L.transpose(1, 2)

        if self.mean_head is not None:
            mean_norm = self.mean_head(features)
            mean_pred = mean_norm * self.mean_y_std + self.mean_y_mean
            return cov, mean_pred
        return cov

    def chol_norm_to_cov(self, chol_norm):
        chol_raw = chol_norm * self.y_std + self.y_mean
        return chol_vectors_to_covariance(chol_raw)


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(train)
    total_loss = 0.0
    n_samples = 0
    has_mean = len(loader.dataset.tensors) == 3
    with torch.set_grad_enabled(train):
        for batch in loader:
            X = batch[0].to(device)
            y_chol = batch[1].to(device)
            y_mean = batch[2].to(device) if has_mean else None
            pred = model(X)
            loss = criterion(pred, y_chol, y_mean)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(X)
            n_samples += len(X)
    return total_loss / n_samples


def main():
    parser = argparse.ArgumentParser(description="Overfit test: can the model memorize training data?")
    parser.add_argument("--train-csv", default="dataset-train.csv")
    parser.add_argument("--val-csv", default="dataset-val.csv")
    parser.add_argument("--output-dir", default="overfit-test-571")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cov-loss", choices=["mse", "l1"], default="l1")
    parser.add_argument("--mean-loss-weight", type=float, default=1.0)
    parser.add_argument("--hidden-mult", type=float, default=1.0,
                        help="Multiply all hidden widths by this factor (default: 1.0 = same as train.py)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run] Device: {device}")
    print(f"[run] Overfit test: dropout=0, no early stopping, {args.epochs} epochs")
    print(f"[run] Hidden width multiplier: {args.hidden_mult}x")

    # Load data
    train_df = pd.read_csv(args.train_csv, low_memory=False)
    val_df = pd.read_csv(args.val_csv, low_memory=False)
    feature_cols, _, chol_cols, mean_cols = get_feature_target_columns(train_df)
    has_mean = len(mean_cols) > 0

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[chol_cols].values.astype(np.float32)
    x_mean, x_std = X_train.mean(0), X_train.std(0)
    x_std[x_std == 0] = 1.0
    y_mean, y_std = y_train.mean(0), y_train.std(0)
    y_std[y_std == 0] = 1.0

    # Covariance normalizers
    train_cov = chol_vectors_to_covariance(torch.from_numpy(y_train)).numpy().reshape(-1, 36)
    cov_mean = train_cov.mean(0).astype(np.float32)
    cov_std = train_cov.std(0).astype(np.float32)
    cov_std[cov_std == 0] = 1.0

    # Mean beam scalers
    if has_mean:
        mean_raw = train_df[mean_cols].values.astype(np.float32)
        mean_y_mean, mean_y_std = mean_raw.mean(0), mean_raw.std(0)
        mean_y_std[mean_y_std == 0] = 1.0
    else:
        mean_y_mean = np.zeros(0, dtype=np.float32)
        mean_y_std = np.ones(0, dtype=np.float32)

    def make_ds(df):
        X = (df[feature_cols].values.astype(np.float32) - x_mean) / x_std
        y = (df[chol_cols].values.astype(np.float32) - y_mean) / y_std
        tensors = [torch.from_numpy(X), torch.from_numpy(y)]
        if has_mean:
            m = (df[mean_cols].values.astype(np.float32) - mean_y_mean) / mean_y_std
            tensors.append(torch.from_numpy(m))
        return TensorDataset(*tensors)

    train_ds = make_ds(train_df)
    val_ds = make_ds(val_df)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Model with NO dropout
    n_inputs = len(feature_cols)
    n_chol = len(chol_cols)
    n_mean = len(mean_cols)

    model = OverfitModel(
        n_inputs, n_chol, n_mean_outputs=n_mean,
        y_mean=torch.from_numpy(y_mean).to(device),
        y_std=torch.from_numpy(y_std).to(device),
        mean_y_mean=torch.from_numpy(mean_y_mean).to(device) if has_mean else None,
        mean_y_std=torch.from_numpy(mean_y_std).to(device) if has_mean else None,
        dropout=0.0,
        hidden_mult=args.hidden_mult,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[run] Model parameters: {n_params:,}")
    print(f"[run] Training samples: {len(train_df)}")
    print(f"[run] Params/samples ratio: {n_params/len(train_df):.1f}")

    criterion = CovarianceAwareLoss(
        model=model,
        cov_mean=torch.from_numpy(cov_mean).reshape(6, 6).to(device),
        cov_std=torch.from_numpy(cov_std).reshape(6, 6).to(device),
        cov_loss=args.cov_loss,
        mean_loss_weight=args.mean_loss_weight,
        has_mean_outputs=has_mean,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train without early stopping
    history = {"epoch": [], "train_loss": [], "val_loss": []}
    print(f"\n[run] Training for {args.epochs} epochs (no early stopping) ...")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, None, device, train=False)
        elapsed = time.time() - t0

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 25 == 0 or epoch == 1:
            gap = val_loss / train_loss if train_loss > 0 else float("inf")
            print(f"[epoch {epoch:04d}] train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"gap={gap:.2f}x  ({elapsed:.1f}s)", flush=True)

    # Results
    final_train = history["train_loss"][-1]
    final_val = history["val_loss"][-1]
    min_train = min(history["train_loss"])
    gap = final_val / final_train if final_train > 0 else float("inf")

    print(f"\n{'='*60}")
    print("OVERFIT TEST RESULTS")
    print(f"{'='*60}")
    print(f"Final train loss: {final_train:.6f}")
    print(f"Min train loss:   {min_train:.6f}")
    print(f"Final val loss:   {final_val:.6f}")
    print(f"Val/Train gap:    {gap:.2f}x")

    if min_train < 0.01:
        print(f"\n[conclusion] Model CAN memorize training data (train loss → {min_train:.6f}).")
        print("  → Model has sufficient capacity. Problem is generalization.")
        print("  → More data likely needed, or better regularization.")
    elif min_train < 0.05:
        print(f"\n[conclusion] Model partially memorizes (train loss = {min_train:.6f}).")
        print("  → Capacity is marginal. Could benefit from a larger model OR more training.")
    else:
        print(f"\n[conclusion] Model CANNOT memorize training data (train loss = {min_train:.6f}).")
        print("  → Model is too small or learning rate/architecture needs tuning.")
        print("  → Try --hidden-mult 2.0 or 3.0 for a larger model.")

    # Save
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(output_dir / "overfit_history.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history["epoch"], history["train_loss"], label="Train Loss", linewidth=2)
    ax.plot(history["epoch"], history["val_loss"], label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Loss ({args.cov_loss})")
    ax.set_title(f"Overfit Test (dropout=0, hidden_mult={args.hidden_mult}x, "
                 f"params={n_params:,}, samples={len(train_df)})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(output_dir / "overfit_curve.png", dpi=150)
    print(f"\n[run] Plot saved to {output_dir}/overfit_curve.png")
    print(f"[run] History saved to {output_dir}/overfit_history.csv")


if __name__ == "__main__":
    main()
