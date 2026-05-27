"""
scripts/run_figure1.py

Entry-point script for reproducing Figure 1.

Responsibilities:
- load YAML configuration
- call pipeline
- save .dat results
- generate plot
- save metadata and config copy

This file is intentionally thin.
All scientific logic is delegated to the pipeline and core modules.

Usage from terminal
-------------------
python scripts/run_figure1.py --config experiments/configs/figures/fig01.yaml

Usage from Python console
-------------------------
from scripts.run_figure1 import run_fig01
run_fig01("experiments/configs/figures/fig01.yaml")
"""

import argparse
import datetime
import os
import shutil
import subprocess

import matplotlib.pyplot as plt
import yaml

from sfc.pipelines.figure1_pipeline import (
    generate_figure1_data,
    save_dat_file,
)


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file"
    )
    return parser.parse_args()


# =============================================================================
# CONFIG LOADING
# =============================================================================

def load_config(path):
    """
    Load YAML configuration from disk.

    Parameters
    ----------
    path : str
        Path to YAML config file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

def prepare_output_dir(cfg):
    """
    Prepare a timestamped output directory.

    Supported YAML formats
    ----------------------
    Format A:
        output:
          base_dir: "data/results"
          figure_dir: "fig01"

    Format B:
        output:
          directory: "data/results/fig01"

    Parameters
    ----------
    cfg : dict
        Parsed YAML configuration.

    Returns
    -------
    tuple
        (output_dir, timestamp)
    """

    if "output" not in cfg:
        raise KeyError(
            "Missing 'output' block in YAML configuration. "
            "Expected either:\n"
            "output:\n"
            "  base_dir: \"data/results\"\n"
            "  figure_dir: \"fig01\"\n"
            "or:\n"
            "output:\n"
            "  directory: \"data/results/fig01\""
        )

    output_cfg = cfg["output"]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # -------------------------------------------------------------------------
    # Preferred split format:
    #   output.base_dir + output.figure_dir
    # -------------------------------------------------------------------------
    if "base_dir" in output_cfg and "figure_dir" in output_cfg:
        base = output_cfg["base_dir"]
        fig = output_cfg["figure_dir"]
        output_dir = os.path.join(base, fig, timestamp)

    # -------------------------------------------------------------------------
    # Alternative flat format:
    #   output.directory
    # -------------------------------------------------------------------------
    elif "directory" in output_cfg:
        output_dir = os.path.join(output_cfg["directory"], timestamp)

    else:
        raise KeyError(
            "Invalid 'output' block in YAML configuration. "
            "Expected either:\n"
            "output:\n"
            "  base_dir: \"data/results\"\n"
            "  figure_dir: \"fig01\"\n"
            "or:\n"
            "output:\n"
            "  directory: \"data/results/fig01\""
        )

    os.makedirs(output_dir, exist_ok=True)

    return output_dir, timestamp


# =============================================================================
# GIT METADATA
# =============================================================================

def get_git_commit():
    """
    Try to retrieve the current Git commit hash.

    Returns
    -------
    str
        Commit hash or 'unknown' if unavailable.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return commit
    except Exception:
        return "unknown"


# =============================================================================
# PLOTTING
# =============================================================================

def generate_plot(df, cfg, output_dir, timestamp):
    """
    Generate and save Figure 1 plot.

    Parameters
    ----------
    df : pandas.DataFrame
        Final figure dataset.

    cfg : dict
        Parsed YAML configuration.

    output_dir : str
        Destination directory.

    timestamp : str
        Timestamp string used in filenames.

    Returns
    -------
    None
    """

    plot_cfg = cfg["plot"]

    plt.figure()

    m_values = sorted(df["M_RbCP"].unique())

    # -------------------------------------------------------------------------
    # Plot curves for each M_RbCP value.
    # -------------------------------------------------------------------------
    for M in m_values:
        df_sub = df[df["M_RbCP"] == M]

        # ---------------------------------------------------------------------
        # Theory: upper bound
        # ---------------------------------------------------------------------
        plt.plot(
            df_sub["N"],
            df_sub["upper_bound"],
            label=f"M={M} Upper bound",
            linestyle="-"
        )

        # ---------------------------------------------------------------------
        # Theory: MSE*
        # ---------------------------------------------------------------------
        plt.plot(
            df_sub["N"],
            df_sub["mse_star"],
            label=f"M={M} MSE*",
            linestyle="--"
        )

        # ---------------------------------------------------------------------
        # Monte Carlo markers
        # ---------------------------------------------------------------------
        plt.scatter(
            df_sub["N"],
            df_sub["mse_mc_mean"],
            label=f"M={M} Monte Carlo",
            marker="o"
        )

    # -------------------------------------------------------------------------
    # Axis labels
    # -------------------------------------------------------------------------
    plt.xlabel(r"$N$")
    plt.ylabel(plot_cfg["y_axis"])

    # -------------------------------------------------------------------------
    # Axis limits
    # -------------------------------------------------------------------------
    if "x_min" in plot_cfg and "x_max" in plot_cfg:
        plt.xlim(plot_cfg["x_min"], plot_cfg["x_max"])

    if "y_min" in plot_cfg and "y_max" in plot_cfg:
        plt.ylim(plot_cfg["y_min"], plot_cfg["y_max"])

    # -------------------------------------------------------------------------
    # Axis ticks
    # -------------------------------------------------------------------------
    if "x_ticks" in plot_cfg:
        plt.xticks(plot_cfg["x_ticks"])

    # -------------------------------------------------------------------------
    # Scale
    # -------------------------------------------------------------------------
    if plot_cfg.get("y_scale", "linear") == "log":
        plt.yscale("log")

    # -------------------------------------------------------------------------
    # Grid and legend
    # -------------------------------------------------------------------------
    if plot_cfg.get("show_grid", False):
        plt.grid(True, which="major", axis="both")

    if plot_cfg.get("legend", False):
        plt.legend()

    # -------------------------------------------------------------------------
    # Save figure in all requested formats
    # -------------------------------------------------------------------------
    plot_formats = ["png"]
    if "output" in cfg and "formats" in cfg["output"] and "plot" in cfg["output"]["formats"]:
        plot_formats = cfg["output"]["formats"]["plot"]

    for fmt in plot_formats:
        filename = f"fig01_{timestamp}.{fmt}"
        path = os.path.join(output_dir, filename)
        plt.savefig(path, bbox_inches="tight")

    plt.show(block=True)
    plt.close()


# =============================================================================
# METADATA
# =============================================================================

def save_metadata(cfg, output_dir, timestamp, config_path):
    """
    Save metadata for reproducibility.

    Parameters
    ----------
    cfg : dict
        Parsed YAML configuration.

    output_dir : str
        Destination directory.

    timestamp : str
        Timestamp string.

    config_path : str
        Path to original config file.

    Returns
    -------
    None
    """
    meta_path = os.path.join(output_dir, f"metadata_{timestamp}.txt")

    with open(meta_path, "w") as f:
        f.write("Figure: fig01\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Git commit: {get_git_commit()}\n")
        f.write(f"Config file: {config_path}\n")
        f.write("\n--- CONFIG SNAPSHOT ---\n")
        f.write(yaml.dump(cfg, sort_keys=False))


# =============================================================================
# COPY CONFIG
# =============================================================================

def copy_config_file(config_path, output_dir):
    """
    Copy the YAML config used for this run into the output directory.

    Parameters
    ----------
    config_path : str
        Original config path.

    output_dir : str
        Destination directory.

    Returns
    -------
    None
    """
    dst = os.path.join(output_dir, "config_used.yaml")
    shutil.copy(config_path, dst)


# =============================================================================
# MAIN RUNNER LOGIC
# =============================================================================

def run_fig01(config_path):
    """
    Run Figure 1 directly from a Python session.

    Parameters
    ----------
    config_path : str
        Path to YAML config file.

    Returns
    -------
    pandas.DataFrame
        Final dataset generated by the pipeline.
    """

    cfg = load_config(config_path)

    output_dir, timestamp = prepare_output_dir(cfg)

    print(f"[INFO] Output directory: {output_dir}")

    # -------------------------------------------------------------------------
    # Run pipeline
    # -------------------------------------------------------------------------
    print("[INFO] Running Figure 1 pipeline...")
    df = generate_figure1_data(cfg)

    # -------------------------------------------------------------------------
    # Save table (.dat)
    # -------------------------------------------------------------------------
    save_dat = cfg.get("output", {}).get("save_dat", True)

    if save_dat:
        filename = f"fig01_{timestamp}.dat"
        path = os.path.join(output_dir, filename)

        delimiter = "\t"
        if "data_format" in cfg and "delimiter" in cfg["data_format"]:
            delimiter = cfg["data_format"]["delimiter"]

        save_dat_file(
            df,
            path,
            delimiter=delimiter
        )

        print(f"[INFO] Saved data: {path}")

    # -------------------------------------------------------------------------
    # Save plot
    # -------------------------------------------------------------------------
    save_plot = cfg.get("output", {}).get("save_plot", True)

    if save_plot:
        generate_plot(df, cfg, output_dir, timestamp)
        print("[INFO] Plot saved")

    # -------------------------------------------------------------------------
    # Save metadata
    # -------------------------------------------------------------------------
    save_metadata_flag = cfg.get("output", {}).get("save_metadata", True)

    if save_metadata_flag:
        save_metadata(cfg, output_dir, timestamp, config_path)
        print("[INFO] Metadata saved")

    # -------------------------------------------------------------------------
    # Save config copy
    # -------------------------------------------------------------------------
    save_config_copy = cfg.get("output", {}).get("save_config_copy", True)

    if save_config_copy:
        copy_config_file(config_path, output_dir)
        print("[INFO] Config copy saved")

    print("[INFO] Figure 1 generation complete")

    return df


def main():
    """
    Command-line entry point.
    """
    args = parse_args()
    run_fig01(args.config)


# =============================================================================

if __name__ == "__main__":
    main()
