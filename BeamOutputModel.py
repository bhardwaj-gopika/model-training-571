from pathlib import Path
import os
import tempfile
from typing import Any, Iterable, Mapping
from distgen import Generator

import numpy as np
import yaml
from lume.model import LUMEModel
from lume.staged_model import FinalParticlesMixIn
from lume_torch.base import LUMETorchModel
from lume_torch.models.torch_model import TorchModel
from scipy import constants
import torch
import beamphysics

def _tensor_to_numpy(value: Any) -> np.ndarray:
    """Return a NumPy view/copy from tensor-like input on CPU without gradients."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _nearest_pd_cov(cov: np.ndarray, base_jitter: float = 1e-10,
                    max_tries: int = 10) -> np.ndarray:
    """Return a symmetric positive-definite version of a (near-)PSD covariance.

    The surrogate reconstructs Sigma = L @ L^T (PSD by construction), but strongly
    correlated phase-space blocks (e.g. a converging beam where x and px are ~99%
    correlated) make Sigma numerically singular, and after M-denormalization the
    element scales span ~1e-24..1e10, so float rounding can push the smallest
    eigenvalue slightly negative. distgen's Cholesky sampler then rejects it with
    "Matrix is not positive definite".

    Strategy: symmetrize, then apply escalating *relative* diagonal loading
    (Sigma_ii *= 1 + eps) until Cholesky succeeds. Relative loading preserves the
    per-dimension scale structure, so the added variance is physically negligible.
    Falls back to eigenvalue clipping if loading is somehow insufficient.
    """
    cov = 0.5 * (cov + cov.T)
    diag = np.abs(np.diag(cov)).copy()
    positive = diag[diag > 0]
    floor = positive.min() * 1e-12 if positive.size else 1e-30
    diag = np.maximum(diag, floor)
    for k in range(max_tries):
        try:
            np.linalg.cholesky(cov)
            return cov
        except np.linalg.LinAlgError:
            eps = base_jitter * (10.0 ** k)
            cov = cov + np.diag(diag * eps)
    # Last resort: clip negative eigenvalues to a small positive floor.
    w, V = np.linalg.eigh(0.5 * (cov + cov.T))
    w = np.clip(w, a_min=float(diag.min()) * 1e-12, a_max=None)
    return (V * w) @ V.T

OTR2_BEAM_ENERGY = 135.0e6  # eV

class BeamOutputModel(LUMEModel, FinalParticlesMixIn):
    """
    LUME wrapper around a surrogate model that adds an openPMD beam
    output variable based on a model predicting the beam covariance matrix.

    The surrogate model is expected to support at least the following variables:
    covariance_matrix: TorchNDVariable
        6x6 covariance matrix in the surrogate convention/order [x, px, y, py, t, pz].
        Units are [m, eV/c, m, eV/c, s, eV/c]. This wrapper converts the t-axis to z (meters)
        using z = -c * t before generating the output ParticleGroup.
    """

    def __init__(
        self,
        surrogate: TorchModel,
        n_particles: int = 10000,
        p0c: float = 1e8,
        t0: float = 0.0,
        z0: float = 0.0,
        total_charge: float = 1e-9,
    ) -> None:
        """
         Initialize wrapper with surrogate model and internal cache copy.

         Parameters
         ----------
        surrogate: TorchModel
            The surrogate model to wrap, which must support the required input variables.
        n_particles: int, optional
            The number of particles to generate in the output beam distribution (default: 10000).
        p0c: float, optional
            The reference momentum in eV/c to use for generating the output beam distribution (default: 1e8).
        t0: float, optional
            The reference time in seconds to use for generating the output beam distribution (default: 0.0).
        z0: float, optional
            The reference position in meters to use for generating the output beam distribution (default: 0.0).
        total_charge: float, optional
            The total charge in Coulombs to use for generating the output beam distribution (default: 1e-9 C).

        """
        super().__init__()
        self.surrogate = LUMETorchModel(surrogate)
        self.n_particles = n_particles
        self.p0c = p0c
        self.t0 = t0
        self.z0 = z0
        self.total_charge = total_charge
        self._cache: dict[str, Any] = {"output_beam": None}
        self.set({})  # Initializing with defaults of NN model
        self.update_state()

    def _get(self, names: Iterable[str]) -> dict[str, Any]:
        return {name: self._cache[name] for name in names}

    def _set(self, values: Mapping[str, Any]) -> None:
        """Update model state and regenerate exported output beam."""
        # handle updates to input variables
        for name, value in values.items():
            self._cache[name] = value

        # update surrogate model with new input variables
        self.surrogate.set(dict(values))

        self.update_state()

    @property
    def supported_variables(self) -> dict[str, Any]:
        """Return supported variables without mutating wrapped model metadata."""
        return self.surrogate.supported_variables

    def reset(self):
        self.surrogate.reset()
        self._cache = {"output_beam": None}

    def update_state(self):
        """Update internal cache from surrogate model and regenerate output beam."""
        self._cache.update(
            self.surrogate.get(list(self.surrogate.supported_variables.keys()))
        )
        self._generate_output_beam()

    def _generate_output_beam(self):
        """Generate the output beam ParticleGroup from cached surrogate outputs and class attributes."""

        # get the covariance matrix from the cache
        # units and variable order: [x, px, y, py, z, pz]
        # units: [m, eV/c, m, eV/c, s, eV/c]
        covariance_matrix = self._cache["covariance_matrix"]

        # some surrogates (e.g. the HAAI 571 model) emit a batched (1, 6, 6) tensor;
        # collapse any leading singleton batch dims down to the bare (6, 6) matrix.
        while covariance_matrix.ndim > 2 and covariance_matrix.shape[0] == 1:
            covariance_matrix = covariance_matrix[0]

        # convert covariance matrix time axis to z using speed of light units for the
        # surrogate and openPMD ParticleBeam convention
        scaled_covariance_matrix = covariance_matrix.clone()
        scaled_covariance_matrix[4, :] *= -constants.speed_of_light
        scaled_covariance_matrix[:, 4] *= -constants.speed_of_light

        # The surrogate covariance is PSD by construction but can be numerically
        # indefinite for strongly correlated / wide-dynamic-range beams; regularize
        # to the nearest PD matrix so distgen's Cholesky sampler does not reject it.
        cov_np = _nearest_pd_cov(
            _tensor_to_numpy(scaled_covariance_matrix).astype(np.float64)
        )

        # Build centroid from predicted means if available, otherwise use defaults.
        # The full model predicts mean_x, mean_px, mean_y, mean_py, mean_t, mean_pz
        # in physical units. Convert mean_t to mean_z (z = -c * t) for distgen.
        mean_x = _tensor_to_numpy(self._cache["mean_x"]).item() if "mean_x" in self._cache else 0.0
        mean_px = _tensor_to_numpy(self._cache["mean_px"]).item() if "mean_px" in self._cache else 0.0
        mean_y = _tensor_to_numpy(self._cache["mean_y"]).item() if "mean_y" in self._cache else 0.0
        mean_py = _tensor_to_numpy(self._cache["mean_py"]).item() if "mean_py" in self._cache else 0.0
        mean_t = _tensor_to_numpy(self._cache["mean_t"]).item() if "mean_t" in self._cache else 0.0
        mean_pz = _tensor_to_numpy(self._cache["mean_pz"]).item() if "mean_pz" in self._cache else self.p0c
        mean_z = -constants.speed_of_light * mean_t + self.z0

        mean = np.array(
            [mean_x, mean_px, mean_y, mean_py, mean_z, mean_pz], dtype=np.float64
        )  # units: [m, eV/c, m, eV/c, m, eV/c]
        inputs = {
            "n_particle": self.n_particles,
            "species": "electron",
            "nd_gaussian_dist": {
                "method": "cholesky",
                "centroid": {
                    "x": str(mean[0]) + " m",
                    "px": str(mean[1]) + " eV/c",
                    "y": str(mean[2]) + " m",
                    "py": str(mean[3]) + " eV/c",
                    "z": str(mean[4]) + " m",
                    "pz": str(mean[5]) + " eV/c",
                },
                "cov_matrix": cov_np.tolist(),
            },
            "start": {"tstart": str(self.t0) + " s", "type": "time"},
            "total_charge": str(self.total_charge) + " C",
        }
        # convert to yaml
        inputs_yaml = yaml.safe_dump(inputs, sort_keys=False)

        generator = Generator(inputs_yaml)

        particle_group = generator.run()
        self._cache["output_beam"] = particle_group

    @property
    def final_particles(self) -> beamphysics.ParticleGroup:
        """Return the final particle distribution as an openPMD ParticleGroup."""
        return self._cache["output_beam"]