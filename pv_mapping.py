"""Affine mapping helpers between machine PV units and simulator parameters for screen 571."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from botorch.models.transforms.input import AffineInputTransform


PV_MAPPING_BY_SIM_PARAM = {
    "CQ10121:b1_gradient": {
        "experimental_pv": "QUAD:IN10:121:BACT",
        "pv_precision": 4,
        "sim_scaling": -2.1,
        "sim_offset": 0.0,
    },
    "GUNF:rf_field_scale": {
        "experimental_pv": "KLYS:LI10:21:AMPL",
        "pv_precision": 6,
        "sim_scaling": 7.89830881e-7,
        "sim_offset": 0.0,
    },
    "GUNF:theta0_deg": {
        "experimental_pv": "KLYS:LI10:21:PHAS",
        "pv_precision": 2,
        "sim_scaling": 1.0,
        "sim_offset": 152.3,
    },
    "SOL10111:solenoid_field_scale": {
        "experimental_pv": "SOLN:IN10:121:BACT",
        "pv_precision": 4,
        "sim_scaling": 1.6,
        "sim_offset": 0.0,
    },
    "SQ10122:b1_gradient": {
        "experimental_pv": "QUAD:IN10:122:BACT",
        "pv_precision": 4,
        "sim_scaling": -2.1,
        "sim_offset": 0.0,
    },
    "distgen:t_dist:sigma_t:value": {
        "experimental_pv": None,
        "pv_precision": None,
        "sim_scaling": 1.0,
        "sim_offset": -1.17,
    },
    "distgen:total_charge:value": {
        "experimental_pv": "TORO:IN10:591:TMIT_PC",
        "pv_precision": 2,
        "sim_scaling": 1.0,
        "sim_offset": 0.0,
    },
    "L0AF_scale:rf_field_scale": {
        "experimental_pv": None,
        "pv_precision": 2,
        "sim_scaling": 1.0,
        "sim_offset": -62380013.198590204,
    },
    "L0AF_phase:theta0_deg": {
        "experimental_pv": "KLYS:LI10:31:PHAS",
        "pv_precision": 2,
        "sim_scaling": 1.0,
        "sim_offset": 25.5,
    },
    "L0BF_scale:rf_field_scale": {
        "experimental_pv": None,
        "pv_precision": 2,
        "sim_scaling": 1.0,
        "sim_offset": -59886109.36180495,
    },
    "L0BF_phase:theta0_deg": {
        "experimental_pv": "KLYS:LI10:41:PHAS",
        "pv_precision": 2,
        "sim_scaling": 1.0,
        "sim_offset": 137.5,
    },
    "QA10361": {
        "experimental_pv": "QUAD:IN10:361:BACT",
        "pv_precision": 4,
        "sim_scaling": -1.08,
        "sim_offset": 0.0,
    },
    "QA10371": {
        "experimental_pv": "QUAD:IN10:371:BACT",
        "pv_precision": 4,
        "sim_scaling": -1.08,
        "sim_offset": 0.0,
    },
    "QE10425": {
        "experimental_pv": "QUAD:IN10:425:BACT",
        "pv_precision": 4,
        "sim_scaling": -1.08,
        "sim_offset": 0.0,
    },
    "QE10441": {
        "experimental_pv": "QUAD:IN10:441:BACT",
        "pv_precision": 4,
        "sim_scaling": -1.08,
        "sim_offset": 0.0,
    },
    "QE10511": {
        "experimental_pv": "QUAD:IN10:511:BACT",
        "pv_precision": 4,
        "sim_scaling": -1.08,
        "sim_offset": 0.0,
    },
    "QE10525": {
        "experimental_pv": "QUAD:IN10:525:BACT",
        "pv_precision": 4,
        "sim_scaling": -1.08,
        "sim_offset": 0.0,
    },
    "distgen:VCC": {
        "experimental_pv": None,
        "pv_precision": None,
        "sim_scaling": 1.0,
        "sim_offset": 0.0,
    },
    "impact_VCC_Cal": {
        "experimental_pv": None,
        "pv_precision": None,
        "sim_scaling": 1.0,
        "sim_offset": -0.00000702,
    },
}


def ordered_pv_mapping(feature_cols: Iterable[str]) -> list[dict]:
    """Return PV mapping specs ordered to match the trained feature columns."""
    ordered_specs = []
    missing = []
    for feature_col in feature_cols:
        spec = PV_MAPPING_BY_SIM_PARAM.get(feature_col)
        if spec is None:
            missing.append(feature_col)
            continue
        ordered_specs.append({"sim_param": feature_col, **spec})
    if missing:
        raise KeyError("Missing PV mapping definitions for: " + ", ".join(missing))
    return ordered_specs


def machine_input_names(feature_cols: Iterable[str]) -> list[str]:
    """Return machine-facing input names, falling back to the sim param when no PV exists."""
    names = []
    for spec in ordered_pv_mapping(feature_cols):
        names.append(spec["experimental_pv"] or spec["sim_param"])
    return names


def build_pv_to_sim_transform(feature_cols: Iterable[str]) -> AffineInputTransform:
    """Build affine transform for machine PV values -> simulator parameters."""
    specs = ordered_pv_mapping(feature_cols)
    coefficient = torch.tensor([spec["sim_scaling"] for spec in specs], dtype=torch.float32)
    offset = torch.tensor([spec["sim_offset"] for spec in specs], dtype=torch.float32)
    return AffineInputTransform(d=len(specs), coefficient=coefficient, offset=offset)


def machine_to_sim_array(machine_values: np.ndarray, feature_cols: Iterable[str]) -> np.ndarray:
    """Convert machine PV values into simulator parameters."""
    specs = ordered_pv_mapping(feature_cols)
    scales = np.asarray([spec["sim_scaling"] for spec in specs], dtype=np.float32)
    offsets = np.asarray([spec["sim_offset"] for spec in specs], dtype=np.float32)
    return (np.asarray(machine_values, dtype=np.float32) - offsets) / scales


def sim_to_machine_array(sim_values: np.ndarray, feature_cols: Iterable[str]) -> np.ndarray:
    """Convert simulator parameters into machine PV units."""
    specs = ordered_pv_mapping(feature_cols)
    scales = np.asarray([spec["sim_scaling"] for spec in specs], dtype=np.float32)
    offsets = np.asarray([spec["sim_offset"] for spec in specs], dtype=np.float32)
    return np.asarray(sim_values, dtype=np.float32) * scales + offsets
