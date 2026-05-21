# sfc/io.py

from pathlib import Path
import numpy as np
import datetime

from sfc.paths import RESULTS_DIR


def save_results(results, prefix="results"):
    """
    Save results matrix to the results directory with a timestamp.

    Parameters
    ----------
    results : np.ndarray
        Data to be saved
    prefix : str
        Prefix for the filename

    Returns
    -------
    file_path : Path
        Full path to saved file
    """

    # Create timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build filename
    file_name = f"{prefix}_{timestamp}.dat"

    file_path = RESULTS_DIR / file_name

    # Save file
    np.savetxt(file_path, results, delimiter="\t")

    print(f"[INFO] Results saved to: {file_path}")

    return file_path
