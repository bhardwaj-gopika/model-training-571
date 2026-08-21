"""Train a plain MLP to predict 2D phase-space projections from 18 scalar inputs.

This is a sanity check: if this model can predict the beam's 2D density but the
covariance model cannot get its summary statistics right, the problem is in the
covariance representation, not the data.

Expects an HDF5 dataset produced by build_projection_dataset.py with:
    /inputs       (N, 18) float32
    /projections  (N, 3, 64, 64) float32 (normalized histograms)

Example:
    python train_cnn.py \\
        --train-h5 projections_train.h5 \\
        --val-h5 projections_val.h5 \\
        --test-h5 projections_test.h5 \\
        --output-dir model-output-cnn \\
        --epochs 200 --batch-size 128
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class ProjectionDataset(Dataset):
    """Lazy-loading dataset from HDF5."""

    def __init__(self, h5_path: str, x_mean=None, x_std=None):
        self.h5_path = h5_path
        self.x_mean = x_mean
        self.x_std = x_std
        with h5py.File(h5_path, "r") as f:
            self.length = f["inputs"].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, "r") as f:
            x = f["inputs"][idx].astype(np.float32)
            y = f["projections"][idx].astype(np.float32)
        if self.x_mean is not None:
            x = (x - self.x_mean) / self.x_std
        return torch.from_numpy(x), torch.from_numpy(y)


class ProjectionMLP(nn.Module):
    """Plain MLP with 3 output heads predicting 64x64 histograms."""

    def __init__(self, n_inputs: int = 18, n_bins: int = 64, hidden_sizes=None):
        super().__init__()
        self.n_bins = n_bins
        if hidden_sizes is None:
            hidden_sizes = [256, 512, 1024, 2048]

        layers = []
        in_size = n_inputs
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ELU())
            in_size = h
        self.backbone = nn.Sequential(*layers)

        n_out = n_bins * n_bins
        self.head_xpx = nn.Linear(in_size, n_out)
        self.head_ypy = nn.Linear(in_size, n_out)
        self.head_tpz = nn.Linear(in_size, n_out)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (batch, 3, 64, 64) normalized histograms."""
        features = self.backbone(x)
        batch_size = x.shape[0]

        heads = [self.head_xpx, self.head_ypy, self.head_tpz]
        projections = []
        for head in heads:
            raw = self.softplus(head(features))
            img = raw.view(batch_size, self.n_bins, self.n_bins)
            # Normalize each image to sum=1
            img = img / (img.sum(dim=(-2, -1), keepdim=True) + 1e-10)
            projections.append(img)

        return torch.stack(projections, dim=1)  # (batch, 3, 64, 64)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    n_samples = 0
    with torch.set_grad_enabled(train):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(X_batch)
            n_samples += len(X_batch)
    return total_loss / n_samples


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-h5", required=True, help="Training HDF5 dataset")
    ap.add_argument("--val-h5", required=True, help="Validation HDF5 dataset")
    ap.add_argument("--test-h5", default=None, help="Test HDF5 dataset (optional)")
    ap.add_argument("--output-dir", default="model-output-cnn")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4,
                    help="DataLoader workers (default: 4)")
    return ap


def main():
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run] Device: {device}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute input normalization from training set
    print(f"[run] Computing input statistics from {args.train_h5}", flush=True)
    with h5py.File(args.train_h5, "r") as f:
        X_train = f["inputs"][:].astype(np.float32)
        n_inputs = X_train.shape[1]
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)
    x_std[x_std == 0] = 1.0

    input_transformers = {
        "x_mean": torch.from_numpy(x_mean),
        "x_std": torch.from_numpy(x_std),
    }
    torch.save(input_transformers, output_dir / "input_transformers.pt")
    print(f"[run] Input transformers saved ({n_inputs} features)", flush=True)

    # Copy bin edges from training file
    with h5py.File(args.train_h5, "r") as f:
        bin_edges = {}
        for key in f.attrs:
            if key.startswith("bin_edges_"):
                var = key.replace("bin_edges_", "")
                bin_edges[var] = f.attrs[key]
        n_bins = int(f.attrs.get("n_bins", 64))
    torch.save(bin_edges, output_dir / "bin_edges.pt")
    print(f"[run] Bin edges saved (n_bins={n_bins})", flush=True)

    # Datasets
    train_ds = ProjectionDataset(args.train_h5, x_mean, x_std)
    val_ds = ProjectionDataset(args.val_h5, x_mean, x_std)
    print(f"[run] Train: {len(train_ds)} samples, Val: {len(val_ds)} samples", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            num_workers=args.num_workers, pin_memory=True)

    # Model
    model = ProjectionMLP(n_inputs=n_inputs, n_bins=n_bins).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[run] Model parameters: {n_params:,}", flush=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10)

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    print(f"\n[run] Training for up to {args.epochs} epochs ...", flush=True)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        elapsed = time.time() - t0
        print(
            f"[epoch {epoch:04d}/{args.epochs}] "
            f"train_loss={train_loss:.8f}  val_loss={val_loss:.8f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  t={elapsed:.1f}s",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "model_cnn.pt")
        else:
            patience_counter += 1

        if args.patience > 0 and patience_counter >= args.patience:
            print(f"[run] Early stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)", flush=True)
            break

    # Test evaluation
    if args.test_h5:
        print("\n[run] Evaluating on test set ...", flush=True)
        test_ds = ProjectionDataset(args.test_h5, x_mean, x_std)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 num_workers=args.num_workers, pin_memory=True)
        model.load_state_dict(torch.load(output_dir / "model_cnn.pt", weights_only=True))
        test_loss = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
        print(f"[run] Test MSE loss: {test_loss:.8f}", flush=True)

    # Save history
    import pandas as pd
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    print(f"\n[run] History saved to {output_dir}/training_history.csv", flush=True)
    print(f"[run] Model saved to {output_dir}/model_cnn.pt", flush=True)


if __name__ == "__main__":
    main()
