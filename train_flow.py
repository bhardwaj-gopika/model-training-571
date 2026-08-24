"""Train a conditional normalizing flow on 6D particle phase-space data.

Uses a Neural Spline Flow (NSF) from the zuko library, conditioned on 18
machine settings. The flow learns the full 6D distribution p(x,px,y,py,t,pz | inputs)
and can sample particle beams at inference time.

Expects HDF5 from build_particle_dataset.py with:
    /inputs     (N_shots, 18) float32
    /particles  (N_shots, N_sub, 6) float32

Example:
    python train_flow.py \
        --train-h5 particles_train.h5 \
        --val-h5 particles_val.h5 \
        --output-dir model-output-flow \
        --epochs 100 --batch-size 64 --n-transforms 8
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

try:
    import zuko
except ImportError:
    raise ImportError("zuko is required: pip install zuko")


class ParticleFlowDataset(Dataset):
    """Dataset that loads shots and their particles from HDF5."""

    def __init__(self, h5_path: str, x_mean=None, x_std=None,
                 p_mean=None, p_std=None):
        self.h5_path = h5_path
        self.x_mean = x_mean
        self.x_std = x_std
        self.p_mean = p_mean
        self.p_std = p_std
        with h5py.File(h5_path, "r") as f:
            self.length = f["inputs"].shape[0]
            self.n_particles = f["particles"].shape[1]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, "r") as f:
            inputs = f["inputs"][idx].astype(np.float32)
            particles = f["particles"][idx].astype(np.float32)

        # Normalize inputs
        if self.x_mean is not None:
            inputs = (inputs - self.x_mean) / self.x_std

        # Normalize particles
        if self.p_mean is not None:
            particles = (particles - self.p_mean) / self.p_std

        return torch.from_numpy(inputs), torch.from_numpy(particles)


def compute_particle_stats(h5_path: str, max_shots: int = 5000):
    """Compute mean/std of particle coordinates from training set."""
    with h5py.File(h5_path, "r") as f:
        n_total = f["particles"].shape[0]
        n_use = min(n_total, max_shots)
        # Read a subset for statistics
        particles = f["particles"][:n_use]  # (n_use, n_particles, 6)

    # Flatten and ignore NaN-padded entries
    flat = particles.reshape(-1, 6)
    valid = ~np.isnan(flat).any(axis=1)
    flat = flat[valid]

    p_mean = flat.mean(axis=0).astype(np.float32)
    p_std = flat.std(axis=0).astype(np.float32)
    p_std[p_std == 0] = 1.0

    return p_mean, p_std


def train_epoch(flow, loader, optimizer, device, max_grad_norm=1.0):
    """Train one epoch. Returns average NLL per particle."""
    flow.train()
    total_nll = 0.0
    total_particles = 0

    for inputs_batch, particles_batch in loader:
        inputs_batch = inputs_batch.to(device)     # (B, 18)
        particles_batch = particles_batch.to(device)  # (B, N_sub, 6)

        B, N_sub, D = particles_batch.shape

        # Create valid mask (non-NaN particles)
        valid_mask = ~torch.isnan(particles_batch).any(dim=-1)  # (B, N_sub)

        # Expand context to match particles: (B, 18) -> (B*N_sub, 18)
        context = inputs_batch.unsqueeze(1).expand(B, N_sub, -1).reshape(B * N_sub, -1)
        x = particles_batch.reshape(B * N_sub, D)
        mask_flat = valid_mask.reshape(B * N_sub)

        # Only compute log_prob for valid (non-padded) particles
        if mask_flat.sum() == 0:
            continue

        context_valid = context[mask_flat]
        x_valid = x[mask_flat]

        log_prob = flow(context_valid).log_prob(x_valid)
        loss = -log_prob.mean()

        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_grad_norm)
        optimizer.step()

        n_valid = int(mask_flat.sum().item())
        total_nll += -log_prob.sum().item()
        total_particles += n_valid

    return total_nll / max(total_particles, 1)


@torch.no_grad()
def val_epoch(flow, loader, device):
    """Validate. Returns average NLL per particle."""
    flow.eval()
    total_nll = 0.0
    total_particles = 0

    for inputs_batch, particles_batch in loader:
        inputs_batch = inputs_batch.to(device)
        particles_batch = particles_batch.to(device)

        B, N_sub, D = particles_batch.shape
        valid_mask = ~torch.isnan(particles_batch).any(dim=-1)

        context = inputs_batch.unsqueeze(1).expand(B, N_sub, -1).reshape(B * N_sub, -1)
        x = particles_batch.reshape(B * N_sub, D)
        mask_flat = valid_mask.reshape(B * N_sub)

        if mask_flat.sum() == 0:
            continue

        context_valid = context[mask_flat]
        x_valid = x[mask_flat]

        log_prob = flow(context_valid).log_prob(x_valid)

        total_nll += -log_prob.sum().item()
        total_particles += int(mask_flat.sum().item())

    return total_nll / max(total_particles, 1)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-h5", required=True)
    ap.add_argument("--val-h5", required=True)
    ap.add_argument("--test-h5", default=None)
    ap.add_argument("--output-dir", default="model-output-flow")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Shots per batch (each shot has N_sub particles)")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-transforms", type=int, default=8,
                    help="Number of flow transform layers (default: 8)")
    ap.add_argument("--hidden-features", type=int, nargs="+", default=[128, 128],
                    help="Hidden layer sizes in each transform (default: 128 128)")
    ap.add_argument("--bins", type=int, default=8,
                    help="Spline bins per dimension (default: 8)")
    ap.add_argument("--num-workers", type=int, default=4)
    return ap


def main():
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run] Device: {device}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute normalization statistics from training set
    print(f"[run] Computing statistics from {args.train_h5}", flush=True)
    with h5py.File(args.train_h5, "r") as f:
        X_train = f["inputs"][:].astype(np.float32)
        n_inputs = X_train.shape[1]
        n_particles = f["particles"].shape[1]

    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)
    x_std[x_std == 0] = 1.0

    p_mean, p_std = compute_particle_stats(args.train_h5)

    print(f"[run] Inputs: {n_inputs}, Particles/shot: {n_particles}", flush=True)
    print(f"[run] Particle mean: {p_mean}", flush=True)
    print(f"[run] Particle std:  {p_std}", flush=True)

    # Save transformers
    input_transformers = {"x_mean": torch.from_numpy(x_mean),
                          "x_std": torch.from_numpy(x_std)}
    particle_transformers = {"p_mean": torch.from_numpy(p_mean),
                             "p_std": torch.from_numpy(p_std),
                             "phase_space_vars": ["x", "px", "y", "py", "t", "pz"]}
    torch.save(input_transformers, output_dir / "input_transformers.pt")
    torch.save(particle_transformers, output_dir / "particle_transformers.pt")

    # Datasets
    train_ds = ParticleFlowDataset(args.train_h5, x_mean, x_std, p_mean, p_std)
    val_ds = ParticleFlowDataset(args.val_h5, x_mean, x_std, p_mean, p_std)
    print(f"[run] Train: {len(train_ds)} shots, Val: {len(val_ds)} shots", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            num_workers=args.num_workers, pin_memory=True)

    # Build flow
    flow = zuko.flows.NSF(
        features=6,
        context=n_inputs,
        transforms=args.n_transforms,
        hidden_features=args.hidden_features,
        bins=args.bins,
    ).to(device)

    n_params = sum(p.numel() for p in flow.parameters())
    print(f"[run] Flow parameters: {n_params:,}", flush=True)
    print(f"[run] Architecture: NSF(transforms={args.n_transforms}, "
          f"hidden={args.hidden_features}, bins={args.bins})", flush=True)

    # Save config for reconstruction
    flow_config = {
        "features": 6,
        "context": n_inputs,
        "transforms": args.n_transforms,
        "hidden_features": args.hidden_features,
        "bins": args.bins,
    }
    with open(output_dir / "flow_config.json", "w") as f:
        json.dump(flow_config, f, indent=2)

    optimizer = torch.optim.Adam(flow.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10)

    # Training loop
    best_val_nll = float("inf")
    patience_counter = 0
    history = {"train_nll": [], "val_nll": []}

    print(f"\n[run] Training for up to {args.epochs} epochs ...", flush=True)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_nll = train_epoch(flow, train_loader, optimizer, device,
                                max_grad_norm=args.grad_clip)
        val_nll = val_epoch(flow, val_loader, device)
        scheduler.step(val_nll)

        history["train_nll"].append(train_nll)
        history["val_nll"].append(val_nll)

        elapsed = time.time() - t0
        print(
            f"[epoch {epoch:04d}/{args.epochs}] "
            f"train_nll={train_nll:.4f}  val_nll={val_nll:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  t={elapsed:.1f}s",
            flush=True,
        )

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            patience_counter = 0
            torch.save(flow.state_dict(), output_dir / "flow_model.pt")
        else:
            patience_counter += 1

        if args.patience > 0 and patience_counter >= args.patience:
            print(f"[run] Early stopping at epoch {epoch}", flush=True)
            break

    # Test evaluation
    if args.test_h5:
        print("\n[run] Evaluating on test set ...", flush=True)
        test_ds = ParticleFlowDataset(args.test_h5, x_mean, x_std, p_mean, p_std)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 num_workers=args.num_workers, pin_memory=True)
        flow.load_state_dict(torch.load(output_dir / "flow_model.pt", weights_only=True))
        test_nll = val_epoch(flow, test_loader, device)
        print(f"[run] Test NLL: {test_nll:.4f}", flush=True)

    # Save history
    import pandas as pd
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    print(f"\n[run] History saved to {output_dir}/training_history.csv", flush=True)
    print(f"[run] Model saved to {output_dir}/flow_model.pt", flush=True)


if __name__ == "__main__":
    main()
