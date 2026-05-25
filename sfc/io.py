# sfc/io.py

"""
I/O utilities for saving results in a reproducible and organized way.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional, Union, Dict, Any

import numpy as np

from sfc.paths import RESULTS_DIR


def save_dat_with_metadata(
        data: np.ndarray,
        *,
        experiment: str,
        config: Optional[Dict[str, Any]] = None,
        header: str = "",
        fmt: str = "%.8f",
) -> Path:
    """
    Save a .dat file and a metadata text file in:
        data/results/<experiment>/

    Parameters
    ----------
    data : np.ndarray
        Array to save.
    experiment : str
        Experiment folder name.
    config : dict, optional
        Configuration dictionary to be saved as metadata (string dump).
    header : str
        Header string for the .dat file.
    fmt : str
        Numeric format for np.savetxt.

    Returns
    -------
    Path
        Path to the saved .dat file.
    """
    exp_dir = RESULTS_DIR / experiment
    exp_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    data_file = exp_dir / f"{experiment}_{timestamp}.dat"
    meta_file = exp_dir / f"{experiment}_{timestamp}_meta.txt"

    np.savetxt(
        data_file,
        data,
        fmt=fmt,
        delimiter="\t",
        header=header,
        comments=""
    )

    if config is not None:
        meta_file.write_text(str(config), encoding="utf-8")

    print(f"[INFO] Results saved to: {data_file}")
    if config is not None:
        print(f"[INFO] Metadata saved to: {meta_file}")

    return data_file
