"""Train with emittance-weighted loss: samples with lower emittance get higher weight.

Supervisor rationale: "weight MAE with a factor inversely proportional to
that sample's emittance" — so the model focuses on fitting good-beam samples
(low emittance) rather than spending capacity on outlier beams.

Usage:
  python train_emittance_weighted.py --cov-loss l1 \
      --output-dir model-output-571-emitw \
      --finetune-batch-sizes 32 8 2 --finetune-epochs-per-stage 50 \
      --finetune-lr 1e-4
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from train import (
    CovarianceSurrogateModel,
    build_model,
    build_parser,
    chol_vectors_to_covariance,
    get_feature_target_columns,
    load_split,
    run_epoch,
    MEAN_TARGET_COLS,
    TARGET_PREFIX,
)

# ── M-normalization constants ─────────────────────────────────────────────────
M_DIAG = np.array([1e3, 1e-6, 1e3, 1e-6, 1e12, 1e-6])
M_INV = np.diag(1.0 / M_DIAG)
M_E_C = 511e3  # eV/c  (electron rest mass × c)


def compute_emittance_weights(
    chol_raw: np.ndarray,
    mean_energy: np.ndarray | None = None,
    method: str = "emit_avg",
    clip_min: float = 1e-3,
    temperature: float = 1.0,
) -> np.ndarray:
    """Compute per-sample weights inversely proportional to beam quality metric.

    Parameters
    ----------
    chol_raw : (N, 21) array in RAW units (not normalized)
        Lower-triangular Cholesky elements.
    mean_energy : (N,) array, optional
        Mean beam energy. Only required for method='combined'.
    method : str
        'emit_avg' — inverse of average (x,y) normalized emittance (default)
        'emit_max' — inverse of max (x,y) normalized emittance
        'combined' — inverse of combined beam quality score
    clip_min : float
        Minimum emittance (µm) to clip before inverting, prevents inf weights.
    temperature : float
        Controls sharpness of weighting.  weight ∝ 1/emit^temperature.
        =1.0 for linear inverse, <1.0 softer, >1.0 sharper.

    Returns
    -------
    weights : (N,) normalized so they average to 1.0.
    """
    # Cholesky → M-normalized covariance → physical covariance
    cov_norm = chol_vectors_to_covariance(torch.from_numpy(chol_raw)).numpy()
    cov_phys = np.einsum("ij,njk,lk->nil", M_INV, cov_norm, M_INV)

    # Normalized projected emittance (x, y)
    det_x = np.abs(cov_phys[:, 0, 0] * cov_phys[:, 1, 1] - cov_phys[:, 0, 1] ** 2)
    det_y = np.abs(cov_phys[:, 2, 2] * cov_phys[:, 3, 3] - cov_phys[:, 2, 3] ** 2)
    emit_x = np.sqrt(det_x) / M_E_C * 1e6  # µm
    emit_y = np.sqrt(det_y) / M_E_C * 1e6  # µm

    if method == "emit_avg":
        metric = (emit_x + emit_y) / 2.0
    elif method == "emit_max":
        metric = np.maximum(emit_x, emit_y)
    elif method == "combined":
        # Include beam size and energy spread in a combined score
        sigma_x = np.sqrt(np.abs(cov_phys[:, 0, 0])) * 1000  # mm
        sigma_y = np.sqrt(np.abs(cov_phys[:, 2, 2])) * 1000  # mm
        std_pz = np.sqrt(np.abs(cov_phys[:, 5, 5]))
        if mean_energy is None:
            raise ValueError("mean_energy required for method='combined'")
        rel_espread = std_pz / mean_energy
        # Combine: geometric mean of normalized metrics
        emit_norm = (emit_x + emit_y) / 2.0
        metric = (emit_norm * sigma_x * sigma_y * rel_espread * 1e3) ** 0.25
    else:
        raise ValueError(f"Unknown method: {method}")

    # Clip to avoid division by zero / extreme weights
    metric = np.clip(metric, clip_min, None)

    # Inverse weighting with temperature
    raw_weights = 1.0 / (metric ** temperature)

    # Normalize so mean weight = 1.0 (preserves overall loss magnitude)
    weights = raw_weights / raw_weights.mean()

    return weights


class EmittanceWeightedLoss(nn.Module):
    """CovarianceAwareLoss with per-sample emittance-based weighting."""

    def __init__(
        self,
        model: CovarianceSurrogateModel,
        cov_mean: torch.Tensor,
        cov_std: torch.Tensor,
        cov_loss: str = "l1",
        mean_loss_weight: float = 1.0,
        has_mean_outputs: bool = False,
    ):
        super().__init__()
        self.model = model
        self.has_mean_outputs = has_mean_outputs
        self.mean_loss_weight = mean_loss_weight
        self.register_buffer("cov_mean", cov_mean)
        self.register_buffer("cov_std", cov_std)
        # Use 'none' reduction so we can apply per-sample weights
        if cov_loss == "l1":
            self.loss_fn = nn.L1Loss(reduction="none")
        elif cov_loss == "mse":
            self.loss_fn = nn.MSELoss(reduction="none")
        else:
            raise ValueError(f"Unsupported cov_loss: {cov_loss}")
        self.mean_loss_fn = nn.L1Loss(reduction="none")

    def forward(
        self,
        model_output,
        target_chol_norm: torch.Tensor,
        target_mean_norm: torch.Tensor = None,
        sample_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.has_mean_outputs:
            pred_cov, pred_mean = model_output
        else:
            pred_cov = model_output

        # Covariance loss (unreduced: shape [batch, 6, 6])
        target_cov = self.model.chol_norm_to_cov(target_chol_norm)
        pred_cov_norm = (pred_cov - self.cov_mean) / self.cov_std
        target_cov_norm = (target_cov - self.cov_mean) / self.cov_std
        cov_loss_unreduced = self.loss_fn(pred_cov_norm, target_cov_norm)

        # Per-sample loss: mean over 36 matrix elements → shape [batch]
        per_sample_cov_loss = cov_loss_unreduced.view(cov_loss_unreduced.shape[0], -1).mean(dim=1)

        # Apply sample weights
        if sample_weights is not None:
            cov_loss = (per_sample_cov_loss * sample_weights).mean()
        else:
            cov_loss = per_sample_cov_loss.mean()

        # Mean beam loss
        if self.has_mean_outputs and target_mean_norm is not None:
            mean_y_mean = self.model.mean_y_mean
            mean_y_std = self.model.mean_y_std
            target_mean_raw = target_mean_norm * mean_y_std + mean_y_mean
            pred_mean_norm = (pred_mean - mean_y_mean) / mean_y_std
            target_mean_renorm = (target_mean_raw - mean_y_mean) / mean_y_std
            mean_loss_unreduced = self.mean_loss_fn(pred_mean_norm, target_mean_renorm)
            per_sample_mean_loss = mean_loss_unreduced.mean(dim=1)
            if sample_weights is not None:
                mean_loss = (per_sample_mean_loss * sample_weights).mean()
            else:
                mean_loss = per_sample_mean_loss.mean()
            return cov_loss + self.mean_loss_weight * mean_loss

        return cov_loss


def run_epoch_weighted(model, loader, criterion, optimizer, device, train: bool):
    """Training loop that passes per-sample weights to the loss."""
    model.train(train)
    total_loss = 0.0
    n_samples = 0
    n_tensors = len(loader.dataset.tensors)
    # Dataset tensors: X, y_chol, [y_mean,] weights
    has_mean = n_tensors >= 4 or (n_tensors == 3 and loader.dataset.tensors[2].shape[1] == 1)

    with torch.set_grad_enabled(train):
        for batch in loader:
            X_batch = batch[0].to(device)
            y_chol_batch = batch[1].to(device)

            if n_tensors == 4:
                # X, y_chol, y_mean, weights
                y_mean_batch = batch[2].to(device)
                w_batch = batch[3].to(device).squeeze()
            elif n_tensors == 3:
                # Could be (X, y_chol, weights) or (X, y_chol, y_mean)
                # Distinguish by shape: weights are 1D, y_mean is 2D
                if batch[2].dim() == 1 or batch[2].shape[1] == 1:
                    y_mean_batch = None
                    w_batch = batch[2].to(device).squeeze()
                else:
                    y_mean_batch = batch[2].to(device)
                    w_batch = None
            else:
                y_mean_batch = None
                w_batch = None

            pred = model(X_batch)
            loss = criterion(pred, y_chol_batch, y_mean_batch, w_batch)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(X_batch)
            n_samples += len(X_batch)
    return total_loss / n_samples


def load_split_weighted(
    path, feature_cols, chol_cols, mean_cols,
    x_mean, x_std, y_mean, y_std, mean_y_mean, mean_y_std,
    weight_method="emit_avg", weight_temperature=1.0,
):
    """Load a data split and compute per-sample emittance weights."""
    df = pd.read_csv(path, low_memory=False)
    X = df[feature_cols].values.astype(np.float32)
    y_chol = df[chol_cols].values.astype(np.float32)
    y_chol_raw = y_chol.copy()  # keep raw for weight computation

    # Normalize
    X = (X - x_mean) / x_std
    y_chol_norm = (y_chol - y_mean) / y_std

    # Compute weights from raw Cholesky targets
    mean_energy = None
    if "mean_energy" in mean_cols and "mean_energy" in df.columns:
        mean_energy = df["mean_energy"].values.astype(np.float32)
    weights = compute_emittance_weights(
        y_chol_raw, mean_energy=mean_energy,
        method=weight_method, temperature=weight_temperature,
    )

    if mean_cols:
        y_mean_vals = df[mean_cols].values.astype(np.float32)
        y_mean_vals = (y_mean_vals - mean_y_mean) / mean_y_std
        return TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(y_chol_norm),
            torch.from_numpy(y_mean_vals),
            torch.from_numpy(weights).float(),
        )
    return TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y_chol_norm),
        torch.from_numpy(weights).float(),
    )


def build_weighted_parser():
    """Extend the base parser with emittance-weighting options."""
    parser = build_parser()
    parser.description = (
        "Train the covariance MLP with emittance-weighted loss. "
        "Samples with lower emittance (better beams) receive higher loss weight."
    )
    parser.add_argument(
        "--weight-method",
        choices=["emit_avg", "emit_max", "combined"],
        default="emit_avg",
        help="Beam quality metric for weighting (default: emit_avg).",
    )
    parser.add_argument(
        "--weight-temperature",
        type=float,
        default=1.0,
        help="Weight sharpness: weight ∝ 1/emit^T.  T=1 linear, T<1 softer, T>1 sharper (default: 1.0).",
    )
    return parser


def main():
    args = build_weighted_parser().parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run] Device: {device}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Compute scalers from training set ─────────────────────────────────────
    print(f"[run] Reading training CSV: {args.train_csv}", flush=True)
    train_df = pd.read_csv(args.train_csv, low_memory=False)
    feature_cols, target_cols, chol_cols, mean_cols = get_feature_target_columns(train_df)
    n_inputs = len(feature_cols)
    n_chol_outputs = len(chol_cols)
    n_mean_outputs = len(mean_cols)
    print(f"[run] Features: {n_inputs}  Cholesky targets: {n_chol_outputs}  Mean targets: {n_mean_outputs}", flush=True)
    if mean_cols:
        print(f"[run] Mean target columns: {mean_cols}", flush=True)

    if n_chol_outputs != 21:
        raise SystemExit("Covariance-space loss requires 21 Cholesky targets.")

    X_train_raw = train_df[feature_cols].values.astype(np.float32)
    y_train_raw = train_df[chol_cols].values.astype(np.float32)

    x_mean = X_train_raw.mean(axis=0)
    x_std = X_train_raw.std(axis=0)
    x_std[x_std == 0] = 1.0

    y_mean_arr = y_train_raw.mean(axis=0)
    y_std_arr = y_train_raw.std(axis=0)
    y_std_arr[y_std_arr == 0] = 1.0

    # Covariance-element normalizers
    y_train_raw_t = torch.from_numpy(y_train_raw)
    train_cov = chol_vectors_to_covariance(y_train_raw_t).numpy().reshape(-1, 36)
    cov_mean = train_cov.mean(axis=0).astype(np.float32)
    cov_std = train_cov.std(axis=0).astype(np.float32)
    cov_std[cov_std == 0] = 1.0

    # Save transformers
    torch.save({
        "x_mean": torch.from_numpy(x_mean),
        "x_std": torch.from_numpy(x_std),
        "feature_cols": feature_cols,
    }, output_dir / "input_transformers.pt")
    torch.save({
        "y_mean": torch.from_numpy(y_mean_arr),
        "y_std": torch.from_numpy(y_std_arr),
        "target_cols": chol_cols,
    }, output_dir / "output_transformers.pt")
    torch.save({
        "cov_mean": torch.from_numpy(cov_mean),
        "cov_std": torch.from_numpy(cov_std),
        "cov_labels": [f"cov_{i}{j}" for i in range(6) for j in range(6)],
    }, output_dir / "covariance_transformers.pt")

    # Mean beam scalers
    if mean_cols:
        mean_train_raw = train_df[mean_cols].values.astype(np.float32)
        mean_y_mean = mean_train_raw.mean(axis=0)
        mean_y_std = mean_train_raw.std(axis=0)
        mean_y_std[mean_y_std == 0] = 1.0
        torch.save({
            "mean_y_mean": torch.from_numpy(mean_y_mean),
            "mean_y_std": torch.from_numpy(mean_y_std),
            "mean_cols": mean_cols,
        }, output_dir / "mean_transformers.pt")
    else:
        mean_y_mean = np.zeros(0, dtype=np.float32)
        mean_y_std = np.ones(0, dtype=np.float32)

    # ── DataLoaders with emittance weights ────────────────────────────────────
    print(f"[run] Emittance weighting: method={args.weight_method}, temperature={args.weight_temperature}", flush=True)

    load_kwargs = dict(
        feature_cols=feature_cols, chol_cols=chol_cols, mean_cols=mean_cols,
        x_mean=x_mean, x_std=x_std, y_mean=y_mean_arr, y_std=y_std_arr,
        mean_y_mean=mean_y_mean, mean_y_std=mean_y_std,
        weight_method=args.weight_method,
        weight_temperature=args.weight_temperature,
    )
    train_ds = load_split_weighted(Path(args.train_csv), **load_kwargs)
    val_ds = load_split_weighted(Path(args.val_csv), **load_kwargs)
    test_ds = load_split_weighted(Path(args.test_csv), **load_kwargs)

    # Print weight statistics
    if mean_cols:
        train_weights = train_ds.tensors[3].numpy()
    else:
        train_weights = train_ds.tensors[2].numpy()
    print(f"[run] Train weight stats: min={train_weights.min():.4f}  max={train_weights.max():.4f}  "
          f"median={np.median(train_weights):.4f}  mean={train_weights.mean():.4f}", flush=True)
    pct = np.percentile(train_weights, [10, 25, 50, 75, 90])
    print(f"[run] Train weight percentiles (10/25/50/75/90): "
          f"{pct[0]:.3f} / {pct[1]:.3f} / {pct[2]:.3f} / {pct[3]:.3f} / {pct[4]:.3f}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # ── Model, loss, optimizer ────────────────────────────────────────────────
    y_mean_t = torch.from_numpy(y_mean_arr).to(device)
    y_std_t = torch.from_numpy(y_std_arr).to(device)
    cov_mean_t = torch.from_numpy(cov_mean).to(device).view(1, 6, 6)
    cov_std_t = torch.from_numpy(cov_std).to(device).view(1, 6, 6)

    mean_y_mean_t = torch.from_numpy(mean_y_mean).to(device) if mean_cols else None
    mean_y_std_t = torch.from_numpy(mean_y_std).to(device) if mean_cols else None

    model = build_model(
        n_inputs, n_chol_outputs, n_mean_outputs=n_mean_outputs,
        y_mean=y_mean_t, y_std=y_std_t,
        mean_y_mean=mean_y_mean_t, mean_y_std=mean_y_std_t,
    ).to(device)
    print(f"[run] Model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    criterion = EmittanceWeightedLoss(
        model=model,
        cov_mean=cov_mean_t,
        cov_std=cov_std_t,
        cov_loss=args.cov_loss,
        mean_loss_weight=args.mean_loss_weight,
        has_mean_outputs=n_mean_outputs > 0,
    )
    print(f"[run] Loss: EmittanceWeightedLoss (cov objective={args.cov_loss})", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    print(f"\n[run] Training for up to {args.epochs} epochs ...", flush=True)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch_weighted(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch_weighted(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        elapsed = time.time() - t0
        print(
            f"[epoch {epoch:04d}/{args.epochs}] "
            f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  t={elapsed:.1f}s",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "model.pt")
        else:
            patience_counter += 1

        if args.patience > 0 and patience_counter >= args.patience:
            print(f"[run] Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)", flush=True)
            break

    # ── Fine-tuning stages ────────────────────────────────────────────────────
    do_finetune = (
        args.finetune_batch_sizes is not None
        and args.finetune_epochs_per_stage > 0
        and len(args.finetune_batch_sizes) > 0
    )
    if do_finetune:
        print(
            f"\n[run] Starting fine-tuning stages "
            f"(batch_sizes={args.finetune_batch_sizes}, "
            f"epochs_per_stage={args.finetune_epochs_per_stage})",
            flush=True,
        )
        model.load_state_dict(torch.load(output_dir / "model.pt", weights_only=True))
        stage_lr = args.finetune_lr

        for stage_idx, stage_bs in enumerate(args.finetune_batch_sizes, start=1):
            stage_train_loader = DataLoader(train_ds, batch_size=stage_bs, shuffle=True)
            stage_val_loader = DataLoader(val_ds, batch_size=stage_bs)

            stage_optimizer = torch.optim.Adam(model.parameters(), lr=stage_lr)
            stage_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                stage_optimizer, mode="min", factor=0.5,
                patience=args.finetune_plateau_patience,
                min_lr=args.finetune_min_lr,
            )

            print(f"[finetune stage {stage_idx}] batch_size={stage_bs} lr={stage_lr:.2e}", flush=True)

            for stage_epoch in range(1, args.finetune_epochs_per_stage + 1):
                t0 = time.time()
                train_loss = run_epoch_weighted(
                    model, stage_train_loader, criterion, stage_optimizer, device, train=True,
                )
                val_loss = run_epoch_weighted(
                    model, stage_val_loader, criterion, stage_optimizer, device, train=False,
                )
                stage_scheduler.step(val_loss)
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                elapsed = time.time() - t0
                print(
                    f"[finetune {stage_idx}:{stage_epoch:03d}/{args.finetune_epochs_per_stage}] "
                    f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                    f"lr={stage_optimizer.param_groups[0]['lr']:.2e}  t={elapsed:.1f}s",
                    flush=True,
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), output_dir / "model.pt")

            stage_lr = max(stage_lr * args.finetune_lr_decay, args.finetune_min_lr)

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\n[run] Loading best checkpoint for test evaluation ...", flush=True)
    model.load_state_dict(torch.load(output_dir / "model.pt", weights_only=True))
    test_loss = run_epoch_weighted(model, test_loader, criterion, optimizer, device, train=False)
    print(f"[run] Test loss (emittance-weighted, {args.cov_loss}): {test_loss:.6f}", flush=True)

    # Unweighted test loss for fair comparison with baseline
    from train import CovarianceAwareLoss as UnweightedLoss
    unweighted_criterion = UnweightedLoss(
        model=model, cov_mean=cov_mean_t, cov_std=cov_std_t,
        cov_loss=args.cov_loss, mean_loss_weight=args.mean_loss_weight,
        has_mean_outputs=n_mean_outputs > 0,
    )
    # Need unweighted loaders for fair comparison
    test_ds_unw = load_split(
        Path(args.test_csv), feature_cols, chol_cols, mean_cols,
        x_mean, x_std, y_mean_arr, y_std_arr, mean_y_mean, mean_y_std,
    )
    test_loader_unw = DataLoader(test_ds_unw, batch_size=args.batch_size)
    test_loss_unw = run_epoch(model, test_loader_unw, unweighted_criterion, optimizer, device, train=False)
    print(f"[run] Test loss (unweighted, {args.cov_loss}, for comparison): {test_loss_unw:.6f}", flush=True)

    # MAE in original covariance units
    model.eval()
    all_preds_cov, all_preds_mean, all_targets_chol = [], [], []
    with torch.no_grad():
        for batch in test_loader_unw:
            X_batch = batch[0].to(device)
            output = model(X_batch)
            if n_mean_outputs > 0:
                pred_cov, pred_mean = output
                all_preds_mean.append(pred_mean.cpu().numpy())
            else:
                pred_cov = output
            all_preds_cov.append(pred_cov.cpu().numpy())
            all_targets_chol.append(batch[1].numpy())

    preds_cov = np.concatenate(all_preds_cov)
    targets_chol_raw = np.concatenate(all_targets_chol) * y_std_arr + y_mean_arr
    targets_cov = chol_vectors_to_covariance(torch.from_numpy(targets_chol_raw)).numpy()

    preds_flat = preds_cov.reshape(preds_cov.shape[0], -1)
    targets_flat = targets_cov.reshape(targets_cov.shape[0], -1)
    mae_per_element = np.abs(preds_flat - targets_flat).mean(axis=0)
    mae_overall = mae_per_element.mean()

    print(f"[run] Test MAE (covariance units, mean over 36 elements): {mae_overall:.6e}", flush=True)
    print("[run] Test MAE per covariance element:", flush=True)
    for i in range(6):
        for j in range(6):
            idx = i * 6 + j
            print(f"       cov_{i}{j}: {mae_per_element[idx]:.6e}", flush=True)

    # MAPE on filtered (good-beam) subset
    target_cov_reshaped = targets_cov.reshape(-1, 6, 6)
    mean_energy_test = None
    if n_mean_outputs > 0:
        preds_mean = np.concatenate(all_preds_mean)
        targets_mean_norm = np.concatenate(
            [batch[2].numpy() if len(batch) > 2 else np.array([]) for batch in
             DataLoader(test_ds_unw, batch_size=args.batch_size)]
        )
        targets_mean_raw = targets_mean_norm * mean_y_std + mean_y_mean
        mean_mae = np.abs(preds_mean - targets_mean_raw).mean(axis=0)
        for col_name, mae_val in zip(mean_cols, mean_mae):
            print(f"[run] Test MAE {col_name}: {mae_val:.6e}", flush=True)

    # Save history
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    print(f"\n[run] Training history saved to {output_dir}/training_history.csv", flush=True)
    print(f"[run] Model saved to {output_dir}/model.pt", flush=True)
    print(f"[run] Weight method: {args.weight_method}, temperature: {args.weight_temperature}", flush=True)


if __name__ == "__main__":
    main()
