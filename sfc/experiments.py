# sfc/experiments.py

"""
Centralized experiment dispatcher.

This module maps experiment names to pipeline functions.
"""

from sfc.pipelines.main_pipeline import run as run_main
from sfc.pipelines.error_only_pipeline import run as run_error_only
from sfc.pipelines.rbcp_pipeline import run as run_rbcp


def run_experiment(name: str, config_path: str):
    """
    Dispatch and run an experiment pipeline by name.
    """
    if name == "main":
        return run_main(config_path)

    if name == "error_only":
        return run_error_only(config_path)

    if name == "rbcp":
        return run_rbcp(config_path)

    raise ValueError(f"Unknown experiment: {name}")
