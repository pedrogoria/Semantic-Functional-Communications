# sfc/experiments.py

"""
Centralized experiment dispatcher.

This module connects experiment names to their corresponding
execution functions and forwards the configuration path.
"""
from typing import Optional

from simulations.run_main import main as run_main
from simulations.run_error_only import main as run_error_only
from simulations.run_rbcp import main as run_rbcp


def run_experiment(name: str, config_path: Optional[str] = None):
    """
    Run a selected experiment by name.

    Parameters
    ----------
    name : str
        Name of the experiment ("main", "error_only", "rbcp")
    config_path : str, optional
        Path to configuration file
    """

    if name == "main":
        print("[INFO] Running MAIN experiment")
        run_main(config_path)

    elif name == "error_only":
        print("[INFO] Running ERROR-ONLY experiment")
        run_error_only(config_path)

    elif name == "rbcp":
        print("[INFO] Running RBCP experiment")
        run_rbcp(config_path)

    else:
        raise ValueError(f"Unknown experiment: {name}")