"""Train the covariance prediction MLP on the prepared dataset splits."""
# .venv/bin/python train.py

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Column definitions ────────────────────────────────────────────────────────
TARGET_PREFIX = "cov_chol_"
PHASE_SPACE_VARS = ["x", "px", "y", "py", "t", "pz"]
MEAN_TARGET_COLS = [f"mean_{v}" for v in PHASE_SPACE_VARS]


def get_feature_target_columns(df: pd.DataFrame):
    chol_cols = [c for c in df.columns if c.startswith(TARGET_PREFIX)]
    mean_cols = [c for c in MEAN_TARGET_COLS if c in df.columns]
    target_cols = mean_cols + chol_cols
    feature_cols = [c for c in df.columns if c not in target_cols]
    return feature_cols, target_cols, chol_cols, mean_cols


# ── Model ─────────────────────────────────────────────────────────────────────
def _make_activation(name: str) -> nn.Module:
    """Create an activation module by name."""
    activations = {"elu": nn.ELU, "tanh": nn.Tanh, "relu": nn.ReLU, "silu": nn.SiLU}
    if name.lower() not in activations:
        raise ValueError(f"Unsupported activation: {name}. Choose from {list(activations)}")
    return activations[name.lower()]()


class CovarianceSurrogateModel(nn.Module):
    """NN predicts Cholesky factors → 6x6 covariance, plus optional mean beam outputs."""

    def __init__(self, n_inputs: int, n_chol_outputs: int, n_mean_outputs: int = 0,
                 y_mean=None, y_std=None,
                 mean_y_mean=None, mean_y_std=None,
                 dropout: float = 0.05, activation: str = "elu"):
        super().__init__()
        self.n_chol_outputs = n_chol_outputs
        self.n_mean_outputs = n_mean_outputs
        layers = [
            nn.Linear(n_inputs, 100),
            _make_activation(activation),
        ]
        hidden_sizes = [100, 200, 200, 300, 300, 200, 100, 100, 100]
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(_make_activation(activation))
            if dropout > 0 and i < len(hidden_sizes) - 3:
                layers.append(nn.Dropout(p=dropout))
        self.backbone = nn.Sequential(*layers)

        # Cholesky head
        self.chol_head = nn.Linear(100, n_chol_outputs)

        if y_mean is None:
            y_mean = torch.zeros(n_chol_outputs, dtype=torch.float32)
        if y_std is None:
            y_std = torch.ones(n_chol_outputs, dtype=torch.float32)
        self.register_buffer("y_mean", y_mean)
        self.register_buffer("y_std", y_std)

        # Mean beam head (energy, time)
        if n_mean_outputs > 0:
            self.mean_head = nn.Linear(100, n_mean_outputs)
            if mean_y_mean is None:
                mean_y_mean = torch.zeros(n_mean_outputs, dtype=torch.float32)
            if mean_y_std is None:
                mean_y_std = torch.ones(n_mean_outputs, dtype=torch.float32)
            self.register_buffer("mean_y_mean", mean_y_mean)
            self.register_buffer("mean_y_std", mean_y_std)
        else:
            self.mean_head = None

    def forward(self, x: torch.Tensor):
        """Forward pass: returns (cov_6x6, mean_preds) or just cov_6x6."""
        features = self.backbone(x)

        # ── Cholesky → covariance ─────────────────────────────────────
        chol_norm = self.chol_head(features)
        chol_raw = chol_norm * self.y_std + self.y_mean

        batch_size = chol_raw.shape[0]
        L = torch.zeros(
            (batch_size, 6, 6),
            dtype=chol_raw.dtype,
            device=chol_raw.device,
        )
        tril_idx = torch.tril_indices(row=6, col=6, offset=0, device=chol_raw.device)
        L[:, tril_idx[0], tril_idx[1]] = chol_raw
        cov = L @ L.transpose(1, 2)

        # ── Mean beam outputs ─────────────────────────────────────────
        if self.mean_head is not None:
            mean_norm = self.mean_head(features)
            mean_pred = mean_norm * self.mean_y_std + self.mean_y_mean
            return cov, mean_pred

        return cov

    def chol_norm_to_cov(self, chol_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalized Cholesky factors to covariance matrices."""
        chol_raw = chol_norm * self.y_std + self.y_mean
        return chol_vectors_to_covariance(chol_raw)


def build_model(n_inputs: int, n_chol_outputs: int, n_mean_outputs: int = 0,
                y_mean=None, y_std=None,
                mean_y_mean=None, mean_y_std=None,
                dropout: float = 0.05, activation: str = "elu") -> CovarianceSurrogateModel:
    """Factory retained for compatibility with analysis/inference utilities."""
    return CovarianceSurrogateModel(
        n_inputs, n_chol_outputs, n_mean_outputs=n_mean_outputs,
        y_mean=y_mean, y_std=y_std,
        mean_y_mean=mean_y_mean, mean_y_std=mean_y_std,
        dropout=dropout, activation=activation,
    )


# ── Data loading ───────────────────────────────────────────────────────────────
def load_split(path: Path, feature_cols, chol_cols, mean_cols,
               x_mean, x_std, y_mean, y_std, mean_y_mean, mean_y_std):
    df = pd.read_csv(path, low_memory=False)
    X = df[feature_cols].values.astype(np.float32)
    y_chol = df[chol_cols].values.astype(np.float32)
    X = (X - x_mean) / x_std
    y_chol = (y_chol - y_mean) / y_std
    if mean_cols:
        y_mean_vals = df[mean_cols].values.astype(np.float32)
        y_mean_vals = (y_mean_vals - mean_y_mean) / mean_y_std
        return TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(y_chol),
            torch.from_numpy(y_mean_vals),
        )
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y_chol))


def chol_vectors_to_covariance(chol_vectors: torch.Tensor) -> torch.Tensor:
    """Convert batch of stored lower-triangular Cholesky vectors to 6x6 covariance."""
    batch_size = chol_vectors.shape[0]
    L = torch.zeros(
        (batch_size, 6, 6),
        dtype=chol_vectors.dtype,
        device=chol_vectors.device,
    )
    tril_idx = torch.tril_indices(row=6, col=6, offset=0, device=chol_vectors.device)
    L[:, tril_idx[0], tril_idx[1]] = chol_vectors
    return L @ L.transpose(1, 2)


class CovarianceAwareLoss(nn.Module):
    """Loss computed in normalized covariance space, plus optional mean beam loss."""

    def __init__(
        self,
        model: CovarianceSurrogateModel,
        cov_mean: torch.Tensor,
        cov_std: torch.Tensor,
        cov_loss: str = "mse",
        mean_loss_weight: float = 1.0,
        has_mean_outputs: bool = False,
    ):
        super().__init__()
        self.model = model
        self.has_mean_outputs = has_mean_outputs
        self.mean_loss_weight = mean_loss_weight
        self.register_buffer("cov_mean", cov_mean)
        self.register_buffer("cov_std", cov_std)
        if cov_loss == "mse":
            self.loss_fn = nn.MSELoss()
        elif cov_loss == "l1":
            self.loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported cov_loss: {cov_loss}")
        self.mean_loss_fn = nn.L1Loss()

    def forward(self, model_output, target_chol_norm: torch.Tensor,
                target_mean_norm: torch.Tensor = None) -> torch.Tensor:
        if self.has_mean_outputs:
            pred_cov, pred_mean = model_output
        else:
            pred_cov = model_output

        # Covariance loss
        target_cov = self.model.chol_norm_to_cov(target_chol_norm)
        pred_cov_norm = (pred_cov - self.cov_mean) / self.cov_std
        target_cov_norm = (target_cov - self.cov_mean) / self.cov_std
        cov_loss = self.loss_fn(pred_cov_norm, target_cov_norm)

        # Mean beam loss (in normalized space)
        if self.has_mean_outputs and target_mean_norm is not None:
            # pred_mean is in raw units, denormalize target to match
            mean_y_mean = self.model.mean_y_mean
            mean_y_std = self.model.mean_y_std
            target_mean_raw = target_mean_norm * mean_y_std + mean_y_mean
            # Compute loss in normalized space for scale balance
            pred_mean_norm = (pred_mean - mean_y_mean) / mean_y_std
            target_mean_renorm = (target_mean_raw - mean_y_mean) / mean_y_std
            mean_loss = self.mean_loss_fn(pred_mean_norm, target_mean_renorm)
            return cov_loss + self.mean_loss_weight * mean_loss

        return cov_loss


# ── Training ───────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, train: bool, max_grad_norm=None):
    model.train(train)
    total_loss = 0.0
    n_samples = 0
    has_mean = len(loader.dataset.tensors) == 3
    with torch.set_grad_enabled(train):
        for batch in loader:
            X_batch = batch[0].to(device)
            y_chol_batch = batch[1].to(device)
            y_mean_batch = batch[2].to(device) if has_mean else None
            pred = model(X_batch)
            loss = criterion(pred, y_chol_batch, y_mean_batch)
            if train:
                optimizer.zero_grad()
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            total_loss += loss.item() * len(X_batch)
            n_samples += len(X_batch)
    return total_loss / n_samples


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train the covariance MLP on prepared dataset splits."
    )
    parser.add_argument("--train-csv", default="dataset-train.csv")
    parser.add_argument("--val-csv", default="dataset-val.csv")
    parser.add_argument("--test-csv", default="dataset-test.csv")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=None,
                        help="Max gradient norm for clipping (default: None = no clipping)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="model-output",
        help="Directory to save model checkpoint and scalers (default: model-output)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early-stopping patience in epochs (default: 20; 0 disables)",
    )
    parser.add_argument(
        "--finetune-batch-sizes",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional fine-tuning batch sizes to run sequentially after base training "
            "(e.g. --finetune-batch-sizes 32 8 2)"
        ),
    )
    parser.add_argument(
        "--finetune-epochs-per-stage",
        type=int,
        default=0,
        help="Fine-tuning epochs to run at each batch-size stage (default: 0 disables)",
    )
    parser.add_argument(
        "--finetune-lr",
        type=float,
        default=1e-4,
        help="Initial learning rate for fine-tuning stages (default: 1e-4)",
    )
    parser.add_argument(
        "--finetune-lr-decay",
        type=float,
        default=0.5,
        help="Multiply LR by this factor after each fine-tuning stage (default: 0.5)",
    )
    parser.add_argument(
        "--finetune-plateau-patience",
        type=int,
        default=5,
        help="ReduceLROnPlateau patience during fine-tuning stages (default: 5)",
    )
    parser.add_argument(
        "--finetune-min-lr",
        type=float,
        default=1e-6,
        help="Minimum LR for ReduceLROnPlateau during fine-tuning (default: 1e-6)",
    )
    parser.add_argument(
        "--cov-loss",
        choices=["mse", "l1"],
        default="mse",
        help="Covariance-space objective function (default: mse).",
    )
    parser.add_argument(
        "--mean-loss-weight",
        type=float,
        default=1.0,
        help="Weight for mean beam (energy/time) loss relative to covariance loss (default: 1.0)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.05,
        help="Dropout rate for hidden layers (default: 0.05; 0 disables)",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="elu",
        choices=["elu", "tanh", "relu", "silu"],
        help="Activation function for hidden layers (default: elu)",
    )
    return parser


def main():
    args = build_parser().parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run] Device: {device}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Compute scalers from training set only ────────────────────────────────
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
        raise SystemExit(
            "Covariance-space loss requires 21 Cholesky targets (cov_chol_0..cov_chol_20)."
        )
    X_train_raw = train_df[feature_cols].values.astype(np.float32)
    y_train_raw = train_df[chol_cols].values.astype(np.float32)

    x_mean = X_train_raw.mean(axis=0)
    x_std = X_train_raw.std(axis=0)
    x_std[x_std == 0] = 1.0  # avoid divide-by-zero for constant columns

    y_mean = y_train_raw.mean(axis=0)
    y_std = y_train_raw.std(axis=0)
    y_std[y_std == 0] = 1.0

    # Compute covariance-element normalizers from train targets in raw units.
    y_train_raw_t = torch.from_numpy(y_train_raw)
    train_cov = chol_vectors_to_covariance(y_train_raw_t).numpy().reshape(-1, 36)
    cov_mean = train_cov.mean(axis=0).astype(np.float32)
    cov_std = train_cov.std(axis=0).astype(np.float32)
    cov_std[cov_std == 0] = 1.0

    # Save input and output transformers separately
    input_transformers = {
        "x_mean": torch.from_numpy(x_mean),
        "x_std": torch.from_numpy(x_std),
        "feature_cols": feature_cols,
    }
    output_transformers = {
        "y_mean": torch.from_numpy(y_mean),
        "y_std": torch.from_numpy(y_std),
        "target_cols": chol_cols,
    }
    covariance_transformers = {
        "cov_mean": torch.from_numpy(cov_mean),
        "cov_std": torch.from_numpy(cov_std),
        "cov_labels": [f"cov_{i}{j}" for i in range(6) for j in range(6)],
    }
    torch.save(input_transformers, output_dir / "input_transformers.pt")
    torch.save(output_transformers, output_dir / "output_transformers.pt")
    torch.save(covariance_transformers, output_dir / "covariance_transformers.pt")
    print(f"[run] Input transformers saved to {output_dir}/input_transformers.pt", flush=True)
    print(f"[run] Output transformers saved to {output_dir}/output_transformers.pt", flush=True)
    print(f"[run] Covariance transformers saved to {output_dir}/covariance_transformers.pt", flush=True)

    # ── Mean beam target scalers ──────────────────────────────────────────────
    if mean_cols:
        mean_train_raw = train_df[mean_cols].values.astype(np.float32)
        mean_y_mean = mean_train_raw.mean(axis=0)
        mean_y_std = mean_train_raw.std(axis=0)
        mean_y_std[mean_y_std == 0] = 1.0
        mean_transformers = {
            "mean_y_mean": torch.from_numpy(mean_y_mean),
            "mean_y_std": torch.from_numpy(mean_y_std),
            "mean_cols": mean_cols,
        }
        torch.save(mean_transformers, output_dir / "mean_transformers.pt")
        print(f"[run] Mean transformers saved to {output_dir}/mean_transformers.pt", flush=True)
    else:
        mean_y_mean = np.zeros(0, dtype=np.float32)
        mean_y_std = np.ones(0, dtype=np.float32)

    # ── DataLoaders ────────────────────────────────────────────────────────────
    train_ds = load_split(
        Path(args.train_csv), feature_cols, chol_cols, mean_cols,
        x_mean, x_std, y_mean, y_std, mean_y_mean, mean_y_std,
    )
    val_ds = load_split(
        Path(args.val_csv), feature_cols, chol_cols, mean_cols,
        x_mean, x_std, y_mean, y_std, mean_y_mean, mean_y_std,
    )
    test_ds = load_split(
        Path(args.test_csv), feature_cols, chol_cols, mean_cols,
        x_mean, x_std, y_mean, y_std, mean_y_mean, mean_y_std,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # ── Model, loss, optimizer ─────────────────────────────────────────────────
    y_mean_t = torch.from_numpy(y_mean).to(device)
    y_std_t = torch.from_numpy(y_std).to(device)
    cov_mean_t = torch.from_numpy(cov_mean).to(device).view(1, 6, 6)
    cov_std_t = torch.from_numpy(cov_std).to(device).view(1, 6, 6)

    mean_y_mean_t = torch.from_numpy(mean_y_mean).to(device) if mean_cols else None
    mean_y_std_t = torch.from_numpy(mean_y_std).to(device) if mean_cols else None

    model = build_model(
        n_inputs, n_chol_outputs, n_mean_outputs=n_mean_outputs,
        y_mean=y_mean_t, y_std=y_std_t,
        mean_y_mean=mean_y_mean_t, mean_y_std=mean_y_std_t,
        dropout=args.dropout, activation=args.activation,
    ).to(device)
    print(f"[run] Model architecture:\n{model}", flush=True)

    # Save model config so analysis/inference scripts can reconstruct the architecture
    model_config = {"dropout": args.dropout, "activation": args.activation}
    with open(output_dir / "model_config.json", "w") as f:
        json.dump(model_config, f, indent=2)
    print(f"[run] Model config saved to {output_dir}/model_config.json", flush=True)

    criterion = CovarianceAwareLoss(
        model=model,
        cov_mean=cov_mean_t,
        cov_std=cov_std_t,
        cov_loss=args.cov_loss,
        mean_loss_weight=args.mean_loss_weight,
        has_mean_outputs=n_mean_outputs > 0,
    )
    print(f"[run] Loss mode: cov (per-element normalized), objective={args.cov_loss}", flush=True)
    if n_mean_outputs > 0:
        print(f"[run] Mean beam loss weight: {args.mean_loss_weight}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    # ── Training loop ──────────────────────────────────────────────────────────
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
            f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  t={elapsed:.1f}s",
            flush=True,
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "model.pt")
        else:
            patience_counter += 1

        if args.patience > 0 and patience_counter >= args.patience:
            print(
                f"[run] Early stopping triggered at epoch {epoch} "
                f"(no improvement for {args.patience} epochs)",
                flush=True,
            )
            break

    # ── Optional staged fine-tuning with smaller batches / lower LR ───────────
    do_finetune = (
        args.finetune_batch_sizes is not None
        and args.finetune_epochs_per_stage > 0
        and len(args.finetune_batch_sizes) > 0
    )
    if do_finetune:
        print(
            "\n[run] Starting fine-tuning stages "
            f"(batch_sizes={args.finetune_batch_sizes}, "
            f"epochs_per_stage={args.finetune_epochs_per_stage}, "
            f"initial_lr={args.finetune_lr:.2e})",
            flush=True,
        )

        # Resume from best base checkpoint before fine-tuning.
        model.load_state_dict(torch.load(output_dir / "model.pt", weights_only=True))
        stage_lr = args.finetune_lr

        for stage_idx, stage_bs in enumerate(args.finetune_batch_sizes, start=1):
            stage_train_loader = DataLoader(train_ds, batch_size=stage_bs, shuffle=True)
            stage_val_loader = DataLoader(val_ds, batch_size=stage_bs)

            stage_optimizer = torch.optim.Adam(model.parameters(), lr=stage_lr)
            stage_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                stage_optimizer,
                mode="min",
                factor=0.5,
                patience=args.finetune_plateau_patience,
                min_lr=args.finetune_min_lr,
            )

            print(
                f"[finetune stage {stage_idx}] batch_size={stage_bs} "
                f"lr={stage_lr:.2e} epochs={args.finetune_epochs_per_stage}",
                flush=True,
            )

            for stage_epoch in range(1, args.finetune_epochs_per_stage + 1):
                t0 = time.time()
                train_loss = run_epoch(
                    model,
                    stage_train_loader,
                    criterion,
                    stage_optimizer,
                    device,
                    train=True,
                )
                val_loss = run_epoch(
                    model,
                    stage_val_loader,
                    criterion,
                    stage_optimizer,
                    device,
                    train=False,
                )
                stage_scheduler.step(val_loss)

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)

                elapsed = time.time() - t0
                print(
                    f"[finetune {stage_idx}:{stage_epoch:03d}/"
                    f"{args.finetune_epochs_per_stage}] "
                    f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                    f"lr={stage_optimizer.param_groups[0]['lr']:.2e}  t={elapsed:.1f}s",
                    flush=True,
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), output_dir / "model.pt")

            stage_lr = max(stage_lr * args.finetune_lr_decay, args.finetune_min_lr)

    # ── Final evaluation on test set ───────────────────────────────────────────
    print("\n[run] Loading best checkpoint for test evaluation ...", flush=True)
    model.load_state_dict(torch.load(output_dir / "model.pt", weights_only=True))
    test_loss = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    print(f"[run] Test objective loss (cov, {args.cov_loss}): {test_loss:.6f}", flush=True)

    # MAE in original covariance units
    model.eval()
    all_preds_cov, all_preds_mean, all_targets_chol = [], [], []
    with torch.no_grad():
        for batch in test_loader:
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
    targets_chol_raw = np.concatenate(all_targets_chol) * y_std + y_mean
    targets_cov = chol_vectors_to_covariance(torch.from_numpy(targets_chol_raw)).numpy()

    preds_cov_flat = preds_cov.reshape(preds_cov.shape[0], -1)
    targets_cov_flat = targets_cov.reshape(targets_cov.shape[0], -1)
    mae_per_element = np.abs(preds_cov_flat - targets_cov_flat).mean(axis=0)
    mae_overall = mae_per_element.mean()

    print(
        f"[run] Test MAE (covariance units, mean over 36 matrix elements): {mae_overall:.6e}",
        flush=True,
    )
    print(f"[run] Test MAE per covariance element:", flush=True)
    for i in range(6):
        for j in range(6):
            idx = i * 6 + j
            print(f"       cov_{i}{j}: {mae_per_element[idx]:.6e}", flush=True)

    # Mean beam MAE
    if n_mean_outputs > 0:
        preds_mean = np.concatenate(all_preds_mean)
        targets_mean_norm = np.concatenate([batch[2].numpy() for batch in
                                            DataLoader(test_ds, batch_size=args.batch_size)])
        targets_mean_raw = targets_mean_norm * mean_y_std + mean_y_mean
        mean_mae = np.abs(preds_mean - targets_mean_raw).mean(axis=0)
        for col_name, mae_val in zip(mean_cols, mean_mae):
            print(f"[run] Test MAE {col_name}: {mae_val:.6e}", flush=True)

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    print(f"\n[run] Training history saved to {output_dir}/training_history.csv", flush=True)
    print(f"[run] Model saved to {output_dir}/model.pt", flush=True)


if __name__ == "__main__":
    main()
