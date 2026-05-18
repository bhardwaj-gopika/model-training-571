"""Run covariance inference from sim-parameter or machine-PV inputs for screen 571.

The script accepts either simulator-parameter columns or machine-facing PV-unit
columns. Machine inputs are first mapped into simulator parameter space and then
normalized using the saved training transformers before model evaluation.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from lume_torch.variables import TorchScalarVariable, TorchNDVariable
from lume_torch.models import TorchModel, TorchModule
from botorch.models.transforms.input import AffineInputTransform

from train import build_model
from pv_mapping import (
    build_pv_to_sim_transform,
    machine_input_names,
    machine_to_sim_array,
    sim_to_machine_array,
)


# M-normalization diagonal used during training.
# The model predicts covariance in M-normalized space; this converts to physical units.
M_DIAG = torch.tensor([1e3, 1e-6, 1e3, 1e-6, 1e12, 1e-6], dtype=torch.float32)


class CovarianceDenormTransform(torch.nn.Module):
    """Output transformer: M-normalized covariance -> physical units.

    Applies C_phys = M_inv @ C_norm @ M_inv^T where M = diag(M_DIAG).
    """

    def __init__(self, m_diag: torch.Tensor = M_DIAG):
        super().__init__()
        self.register_buffer("m_inv_diag", 1.0 / m_diag)

    def forward(self, cov: torch.Tensor) -> torch.Tensor:
        m_inv = torch.diag(self.m_inv_diag)
        return m_inv @ cov @ m_inv.T


def covariance_labels():
    return [f"cov_{row}{col}" for row in range(6) for col in range(6)]


def load_model_and_transformers(model_dir: Path, device: torch.device):
    input_tr = torch.load(model_dir / "input_transformers.pt")
    output_tr = torch.load(model_dir / "output_transformers.pt")

    feature_cols = list(input_tr["feature_cols"])
    n_inputs = len(feature_cols)
    n_chol_outputs = len(output_tr["target_cols"])
    y_mean = output_tr["y_mean"].to(device)
    y_std = output_tr["y_std"].to(device)

    # Check for mean beam transformers
    mean_tr_path = model_dir / "mean_transformers.pt"
    if mean_tr_path.exists():
        mean_tr = torch.load(mean_tr_path)
        mean_cols = mean_tr["mean_cols"]
        n_mean_outputs = len(mean_cols)
        mean_y_mean = mean_tr["mean_y_mean"].to(device)
        mean_y_std = mean_tr["mean_y_std"].to(device)
    else:
        mean_cols = []
        n_mean_outputs = 0
        mean_y_mean = None
        mean_y_std = None

    model = build_model(
        n_inputs, n_chol_outputs, n_mean_outputs=n_mean_outputs,
        y_mean=y_mean, y_std=y_std,
        mean_y_mean=mean_y_mean, mean_y_std=mean_y_std,
    )
    model.load_state_dict(torch.load(model_dir / "model.pt", weights_only=True, map_location=device))
    model.to(device)
    model.eval()
    return model, input_tr, mean_cols


def build_parser():
    parser = argparse.ArgumentParser(
        description="Infer 6x6 covariance matrices from sim-parameter or machine-PV input CSV rows (screen 571)."
    )
    parser.add_argument(
        "--model-dir",
        default="model-output-571",
        help="Directory containing model.pt and transformer files (default: model-output-571)",
    )
    parser.add_argument(
        "--input-csv",
        default="dataset-test.csv",
        help="CSV with either sim-parameter columns or machine PV columns (default: dataset-test.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="inference-output",
        help="Directory for inference outputs (default: inference-output)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for inference (default: 256)",
    )
    parser.add_argument(
        "--print-row",
        type=int,
        default=0,
        help="Row index whose predicted covariance matrix should be printed (default: 0)",
    )
    parser.add_argument(
        "--input-space",
        choices=["auto", "sim", "pv"],
        default="auto",
        help="Interpret input CSV columns as sim parameters, machine PVs, or auto-detect (default: auto)",
    )
    return parser


def resolve_input_space(df: pd.DataFrame, feature_cols, requested_space: str):
    sim_cols = list(feature_cols)
    pv_cols = machine_input_names(feature_cols)
    has_sim = all(col in df.columns for col in sim_cols)
    has_pv = all(col in df.columns for col in pv_cols)

    if requested_space == "sim":
        if not has_sim:
            missing = [col for col in sim_cols if col not in df.columns]
            raise SystemExit("Input CSV is missing required sim columns: " + ", ".join(missing))
        return "sim", sim_cols, pv_cols

    if requested_space == "pv":
        if not has_pv:
            missing = [col for col in pv_cols if col not in df.columns]
            raise SystemExit("Input CSV is missing required PV columns: " + ", ".join(missing))
        return "pv", sim_cols, pv_cols

    if has_pv:
        return "pv", sim_cols, pv_cols
    if has_sim:
        return "sim", sim_cols, pv_cols

    missing_sim = [col for col in sim_cols if col not in df.columns]
    missing_pv = [col for col in pv_cols if col not in df.columns]
    raise SystemExit(
        "Input CSV does not match either supported schema. "
        f"Missing sim columns: {missing_sim}. Missing PV columns: {missing_pv}."
    )


def create_lume_torch_sim(model, input_tr, dump_dir="lumetorchyaml-sim"):
    """Create LUME-torch model that takes simulator-parameter inputs."""
    feature_cols = list(input_tr["feature_cols"])
    x_mean = input_tr["x_mean"].to(dtype=torch.float32)
    x_std = input_tr["x_std"].to(dtype=torch.float32)

    input_variables = [
        TorchScalarVariable(name=col, default_value=float(x_mean[idx]))
        for idx, col in enumerate(feature_cols)
    ]
    output_variables = [TorchNDVariable(name="covariance_matrix", shape=(6, 6))]

    normalization_transform = AffineInputTransform(
        d=len(feature_cols), coefficient=x_std, offset=x_mean
    )

    denorm_transform = CovarianceDenormTransform(M_DIAG)

    torch_model = TorchModel(
        model=model,
        input_variables=input_variables,
        output_variables=output_variables,
        input_transformers=[normalization_transform],
        output_transformers=[denorm_transform],
        precision="single",
    )

    Path(dump_dir).mkdir(parents=True, exist_ok=True)
    torch_model.dump(f"{dump_dir}/injector_simulator.yaml")

    return TorchModule(model=torch_model)


def create_lume_torch_machine(model, input_tr, dump_dir="lumetorchyaml-machine"):
    """Create LUME-torch model that takes machine-PV inputs (wraps sim model)."""
    feature_cols = list(input_tr["feature_cols"])
    x_mean = input_tr["x_mean"].to(dtype=torch.float32)
    x_std = input_tr["x_std"].to(dtype=torch.float32)
    pv_cols = machine_input_names(feature_cols)
    pv_defaults = sim_to_machine_array(x_mean.cpu().numpy()[None, :], feature_cols)[0]

    input_variables = [
        TorchScalarVariable(name=col, default_value=float(pv_defaults[idx]))
        for idx, col in enumerate(pv_cols)
    ]
    output_variables = [TorchNDVariable(name="covariance_matrix", shape=(6, 6))]

    pv_to_sim_transform = build_pv_to_sim_transform(feature_cols)
    normalization_transform = AffineInputTransform(
        d=len(feature_cols), coefficient=x_std, offset=x_mean
    )

    denorm_transform = CovarianceDenormTransform(M_DIAG)

    torch_model = TorchModel(
        model=model,
        input_variables=input_variables,
        output_variables=output_variables,
        input_transformers=[pv_to_sim_transform, normalization_transform],
        output_transformers=[denorm_transform],
        precision="single",
    )

    Path(dump_dir).mkdir(parents=True, exist_ok=True)
    torch_model.dump(f"{dump_dir}/injector_machine.yaml")

    return TorchModule(model=torch_model)


def main():
    args = build_parser().parse_args()

    model_dir = Path(args.model_dir)
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run] Device: {device}", flush=True)
    print(f"[run] Loading model from {model_dir}", flush=True)
    model, input_tr, mean_cols = load_model_and_transformers(model_dir, device)
    has_mean = len(mean_cols) > 0

    feature_cols = list(input_tr["feature_cols"])
    x_mean = input_tr["x_mean"].cpu().numpy().astype(np.float32)
    x_std = input_tr["x_std"].cpu().numpy().astype(np.float32)

    print(f"[run] Reading input CSV: {input_csv}", flush=True)
    df = pd.read_csv(input_csv, low_memory=False)
    input_space, sim_cols, pv_cols = resolve_input_space(df, feature_cols, args.input_space)

    if input_space == "sim":
        X_sim = df[sim_cols].values.astype(np.float32)
        print("[run] Detected simulator-parameter input columns", flush=True)
    else:
        X_machine = df[pv_cols].values.astype(np.float32)
        X_sim = machine_to_sim_array(X_machine, feature_cols)
        print("[run] Detected machine-PV input columns; applying PV -> sim transform", flush=True)

    X_norm = (X_sim - x_mean) / x_std
    loader1 = DataLoader(TensorDataset(torch.from_numpy(X_norm)), batch_size=args.batch_size)

    print(f"[run] Running inference for {len(df)} rows", flush=True)
    pred_batches = []
    pred_mean_batches = []
    m_inv = np.diag(1.0 / M_DIAG.numpy())
    with torch.no_grad():
        for (X_batch,) in loader1:
            output = model(X_batch.to(device))
            if has_mean:
                pred_cov, pred_mean = output
                pred_mean_batches.append(pred_mean.cpu().numpy())
            else:
                pred_cov = output
            pred_batches.append(pred_cov.cpu().numpy())

    # --- Sim-input LUME-torch model validation ---
    lume_sim_model = create_lume_torch_sim(model, input_tr)
    loader_sim = DataLoader(TensorDataset(torch.from_numpy(X_sim)), batch_size=args.batch_size)

    pred_batches_lume_sim = []
    with torch.no_grad():
        for (X_batch,) in loader_sim:
            output = lume_sim_model(X_batch.to(device))
            pred_cov_lume = output.cpu().numpy() if not has_mean else output[0].cpu().numpy() if isinstance(output, tuple) else output.cpu().numpy()
            pred_batches_lume_sim.append(pred_cov_lume)

    # Reference predictions with M denormalization applied
    loader_ref = DataLoader(TensorDataset(torch.from_numpy(X_norm)), batch_size=args.batch_size)
    pred_batches_ref = []
    with torch.no_grad():
        for (X_batch,) in loader_ref:
            output = model(X_batch.to(device))
            pred_cov = output[0].cpu().numpy() if has_mean else output.cpu().numpy()
            # Apply M-denormalization to match the lume-torch output
            pred_cov = np.array([m_inv @ c @ m_inv.T for c in pred_cov])
            pred_batches_ref.append(pred_cov)

    preds_cov_ref = np.concatenate(pred_batches_ref, axis=0)
    preds_cov_lume_sim = np.concatenate(pred_batches_lume_sim, axis=0)

    if not np.allclose(preds_cov_ref, preds_cov_lume_sim, rtol=1e-5, atol=1e-5):
        max_abs_diff = float(np.max(np.abs(preds_cov_ref - preds_cov_lume_sim)))
        raise AssertionError(
            f"Sim-input LUME model mismatch vs direct model; max abs diff={max_abs_diff:.6e}"
        )
    print("[run] Sim-input LUME-torch model validated successfully", flush=True)

    # --- Machine-input LUME-torch model validation ---
    X_machine = sim_to_machine_array(X_sim, feature_cols)
    # Compute reference through same roundtrip path to avoid float32 precision loss
    X_sim_roundtrip = machine_to_sim_array(X_machine, feature_cols)
    X_norm_roundtrip = (X_sim_roundtrip - x_mean) / x_std
    loader_machine_ref = DataLoader(
        TensorDataset(torch.from_numpy(X_norm_roundtrip)), batch_size=args.batch_size
    )
    pred_batches_machine_ref = []
    with torch.no_grad():
        for (X_batch,) in loader_machine_ref:
            output = model(X_batch.to(device))
            pred_cov = output[0].cpu().numpy() if has_mean else output.cpu().numpy()
            # Apply M-denormalization to match the lume-torch output
            pred_cov = np.array([m_inv @ c @ m_inv.T for c in pred_cov])
            pred_batches_machine_ref.append(pred_cov)
    preds_cov_machine_ref = np.concatenate(pred_batches_machine_ref, axis=0)

    lume_machine_model = create_lume_torch_machine(model, input_tr)
    loader_machine = DataLoader(TensorDataset(torch.from_numpy(X_machine)), batch_size=args.batch_size)

    pred_batches_lume_machine = []
    with torch.no_grad():
        for (X_batch,) in loader_machine:
            output = lume_machine_model(X_batch.to(device))
            pred_cov_lume = output.cpu().numpy() if not has_mean else output[0].cpu().numpy() if isinstance(output, tuple) else output.cpu().numpy()
            pred_batches_lume_machine.append(pred_cov_lume)

    preds_cov_lume_machine = np.concatenate(pred_batches_lume_machine, axis=0)

    if not np.allclose(preds_cov_machine_ref, preds_cov_lume_machine, rtol=1e-5, atol=1e-5):
        max_abs_diff = float(np.max(np.abs(preds_cov_machine_ref - preds_cov_lume_machine)))
        raise AssertionError(
            f"Machine-input LUME model mismatch vs direct model; max abs diff={max_abs_diff:.6e}"
        )
    print("[run] Machine-input LUME-torch model validated successfully", flush=True)

    preds_cov = np.concatenate(pred_batches, axis=0)

    pred_flat = preds_cov.reshape(len(df), 36)
    cov_cols = covariance_labels()

    base_df = df[sim_cols].copy()
    base_df.insert(0, "sample_index", np.arange(len(base_df), dtype=np.int64))
    mapped_sim_df = pd.DataFrame(X_sim, columns=[f"sim_{col}" for col in feature_cols], index=df.index)
    pred_df = pd.DataFrame(
        {f"pred_{col}": pred_flat[:, idx] for idx, col in enumerate(cov_cols)}
    )
    result_parts = [base_df, mapped_sim_df, pred_df]

    if has_mean:
        preds_mean = np.concatenate(pred_mean_batches, axis=0)
        mean_pred_df = pd.DataFrame(
            {f"pred_{col}": preds_mean[:, idx] for idx, col in enumerate(mean_cols)}
        )
        result_parts.append(mean_pred_df)

    result_df = pd.concat(result_parts, axis=1)
    result_df.to_csv(output_dir / "predicted_covariances.csv", index=False)
    np.save(output_dir / "predicted_covariances.npy", preds_cov)

    row_index = args.print_row
    if row_index < 0 or row_index >= len(df):
        raise SystemExit(f"--print-row must be between 0 and {len(df) - 1}")

    np.set_printoptions(precision=6, suppress=False)
    print(f"[run] Saved flat predictions to {output_dir / 'predicted_covariances.csv'}", flush=True)
    print(f"[run] Saved 3D covariance array to {output_dir / 'predicted_covariances.npy'}", flush=True)
    print(f"[run] Predicted covariance matrix for row {row_index}:", flush=True)
    print(preds_cov[row_index], flush=True)
    print(
        "[run] Flow: machine PV inputs -> PV-to-sim affine transform -> normalization -> surrogate model -> predicted 6x6 covariance matrix.",
        flush=True,
    )


if __name__ == "__main__":
    main()
