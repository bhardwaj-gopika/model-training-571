"""Custom lume-torch model utilities for the 571 injector model.

This module defines the custom transforms and TorchModel subclass needed for
loading/evaluating the full model (covariance + mean beam outputs) via lume-torch.
"""

import torch
import torch.nn as nn
from lume_torch.models import TorchModel, TorchModule
from lume_torch.variables import TorchScalarVariable, TorchNDVariable


M_DIAG = torch.tensor([1e3, 1e-6, 1e3, 1e-6, 1e12, 1e-6], dtype=torch.float32)


class CovOnlyWrapper(nn.Module):
    """Wraps a model that returns (cov, mean) tuple to return only cov."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, tuple):
            return output[0]
        return output


class CovarianceDenormTransform(nn.Module):
    """Output transformer: M-normalized covariance -> physical units.

    Applies C_phys = M_inv @ C_norm @ M_inv^T where M = diag(M_DIAG).
    """

    def __init__(self, m_diag: torch.Tensor = M_DIAG):
        super().__init__()
        self.register_buffer("m_inv_diag", 1.0 / m_diag)

    def forward(self, cov: torch.Tensor) -> torch.Tensor:
        m_inv = torch.diag(self.m_inv_diag)
        return m_inv @ cov @ m_inv.T


class FullOutputDenormTransform(nn.Module):
    """Output transformer: denorm covariance + pass through mean predictions.

    Expects input from a model that returns (cov_6x6, mean_vec) as a tuple.
    Returns a flat tensor: [cov_6x6_flat (36), mean_x, mean_px, mean_y, mean_py, mean_t, mean_pz] = 42 elements.
    """

    def __init__(self, m_diag: torch.Tensor = M_DIAG):
        super().__init__()
        self.register_buffer("m_inv_diag", 1.0 / m_diag)

    def forward(self, output):
        if isinstance(output, tuple):
            cov, mean_pred = output
        else:
            cov = output
            mean_pred = None
        m_inv = torch.diag(self.m_inv_diag)
        cov_phys = m_inv @ cov @ m_inv.T
        cov_flat = cov_phys.reshape(cov_phys.shape[0], -1)  # (batch, 36)
        if mean_pred is not None:
            return torch.cat([cov_flat, mean_pred], dim=-1)  # (batch, 42)
        return cov_flat


class CovMeanTorchModel(TorchModel):
    """TorchModel subclass that correctly parses mixed NDVariable + scalar outputs.

    Expects the output transformer to return a flat (batch, 42) tensor where
    the first 36 elements are flattened 6x6 covariance and the remaining 6 are
    scalar mean outputs (x, px, y, py, t, pz).
    """

    def __init__(self, *args, n_cov_elements: int = 36, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_cov_elements = n_cov_elements

    def _parse_outputs(self, output_tensor: torch.Tensor) -> dict:
        """Split flat (batch, 42) tensor into covariance_matrix + scalars."""
        if output_tensor.dim() == 1:
            output_tensor = output_tensor.unsqueeze(0)

        parsed_outputs = {}
        idx = 0
        for var in self.output_variables:
            if isinstance(var, TorchNDVariable):
                n_elements = 1
                for s in var.shape:
                    n_elements *= s
                flat_chunk = output_tensor[:, idx:idx + n_elements]
                parsed_outputs[var.name] = flat_chunk.reshape(-1, *var.shape)
                idx += n_elements
            else:
                # TorchScalarVariable
                parsed_outputs[var.name] = output_tensor[:, idx:idx + 1]
                idx += 1
        return parsed_outputs


class CovMeanTorchModule(TorchModule):
    """TorchModule subclass that handles mixed NDVariable + scalar output stacking.

    Standard TorchModule._dictionary_to_tensor fails when outputs have different
    shapes (e.g. covariance_matrix (batch, 6, 6) vs mean_x (batch, 1)).
    This subclass flattens NDVariables before stacking.
    """

    def _dictionary_to_tensor(self, y_model: dict) -> torch.Tensor:
        output_list = []
        for output_name in self.output_order:
            output = y_model[output_name]
            if output.dim() > 2:
                # Flatten NDVariable: (batch, 6, 6) -> (batch, 36)
                batch_size = output.shape[0]
                output = output.reshape(batch_size, -1)
            elif output.dim() == 1:
                # Scalar squeezed to (batch,) -> (batch, 1)
                output = output.unsqueeze(-1)
            elif output.dim() == 2 and output.shape[-1] == 1:
                # Keep (batch, 1) as is
                pass
            output_list.append(output)
        return torch.cat(output_list, dim=-1)
