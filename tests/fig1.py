"""
tests/fig1.py

Figure 1 reproduction using the current project pipeline:

    sfc/pipelines/figure1_pipeline.py

This script is intentionally representational only.

It does NOT use:
- channel capacity
- P
- N0
- SNR
- SFCChannel
- MAC / FDMA
- semantic error detection

Figure 1 studies the RbCP representation/reconstruction error as a function of:
- N
- M_RbCP

The Monte Carlo flow is delegated to the trusted pipeline:
    generate_figure1_data(cfg)

Output
------
Saves:

    data/results/fig1.dat

with columns:
- N
- M_RbCP
- Q
- upper_bound
- mse_star
- mse_mc_mean
- mse_mc_std
- num_trials
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


# =============================================================================
# PATH HANDLING
# =============================================================================

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# =============================================================================
# TRUSTED PIPELINE IMPORTS
# =============================================================================

from sfc.pipelines.figure1_pipeline import (
    generate_figure1_data,
    save_dat_file,
)


# =============================================================================
# OUTPUT
# =============================================================================

OUTPUT_DIR = ROOT_DIR / "data" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "fig1.dat"


# =============================================================================
# CONFIGURATION
# =============================================================================

def build_fig1_config() -> dict:
    """
    Build the in-script configuration for Figure 1.

    Notes
    -----
    Figure 1 is a representational experiment. Therefore, no physical-channel
    parameters are needed here.

    M_RbCP is swept directly and is NOT derived from a communication budget.
    """

    cfg = {
        # ---------------------------------------------------------------------
        # Monte Carlo
        # ---------------------------------------------------------------------
        "monte_carlo": {
            "seed": 47,
            "interactions": 1000,
        },

        # ---------------------------------------------------------------------
        # Sweeps
        # ---------------------------------------------------------------------
        "sweep": {
            "m_rbcp_values": [5, 9, 17],
            "n": {
                "start": 2,
                "stop": 22,
                "step": 2,
            },
        },

        # ---------------------------------------------------------------------
        # Signal model
        # ---------------------------------------------------------------------
        "signal": {
            # Legacy Figure-1 convention accepted by the pipeline.
            "T": 1.0,

            # Discrete-time step.
            "Tt": 0.001,

            # Manuscript-style random signal generation.
            "distribution": "uniform",

            # Peak-to-peak value after filtering.
            # Keep this explicit to avoid hidden scaling assumptions.
            "peak_to_peak": 2.0,

            # Fourier / DFT normalization policy.
            "normalize_dft": True,
            "normalization_target": 3.99,

            # Optional DFT periods parameter used by FourierCoefficientCore.
            "dft_signal_periods": 1,

            # Threshold used by the phase coefficient logic when needed.
            "threshold_harmonics": 0.001,
        },

        # ---------------------------------------------------------------------
        # DC handling
        # ---------------------------------------------------------------------
        "dc": {
            # For Figure 1 faithful reproduction, DC is removed before RbCP.
            "enabled": False,
        },
    }

    return cfg


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    cfg = build_fig1_config()

    print("[INFO] Figure 1 simulation using current trusted pipeline")
    print(f"[INFO] Output file: {OUTPUT_FILE}")
    print(f"[INFO] Monte Carlo seed: {cfg['monte_carlo']['seed']}")
    print(f"[INFO] Trials: {cfg['monte_carlo']['interactions']}")
    print(f"[INFO] M_RbCP values: {cfg['sweep']['m_rbcp_values']}")
    print(
        "[INFO] N sweep: "
        f"{cfg['sweep']['n']['start']}:"
        f"{cfg['sweep']['n']['stop']}:"
        f"{cfg['sweep']['n']['step']}"
    )

    df = generate_figure1_data(cfg)

    save_dat_file(
        df=df,
        path=str(OUTPUT_FILE),
        delimiter="\t",
    )

    print("[INFO] Done.")
    print(f"[INFO] Saved: {OUTPUT_FILE}")
    print("[INFO] Preview:")
    print(df.head())


if __name__ == "__main__":
    main()
