"""
scripts/run_figure2.py

Entry-point script for reproducing Figure 2.

Responsibilities:
- load YAML configuration
- call pipeline
- save .dat results
- generate plot
- save metadata and config copy

Usage from Python console:
--------------------------
from scripts.run_figure2 import run_fig02
run_fig02("experiments/configs/figures/fig02.yaml")
"""

import argparse
import datetime
import os
import shutil
import subprocess

import matplotlib.pyplot as plt
import yaml

from sfc.pipelines.figure2_pipeline import (
    generate_figure2_data,
    save_dat_file,
)


# =============================================================================
# CONFIG
# =============================================================================

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# =============================================================================
# OUTPUT DIR
# =============================================================================

def prepare_output_dir(cfg):

    output_cfg = cfg["output"]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if "base_dir" in output_cfg and "figure_dir" in output_cfg:
        output_dir = os.path.join(
            output_cfg["base_dir"],
            output_cfg["figure_dir"],
            timestamp
        )
    elif "directory" in output_cfg:
        output_dir = os.path.join(
            output_cfg["directory"],
            timestamp
        )
    else:
        raise KeyError("Invalid output configuration")

    os.makedirs(output_dir, exist_ok=True)

    return output_dir, timestamp


# =============================================================================
# GIT META
# =============================================================================

def get_git_commit():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return commit
    except Exception:
        return "unknown"


# =============================================================================
# PLOT
# =============================================================================

def generate_plot(df, cfg, output_dir, timestamp):

    plot_cfg = cfg["plot"]

    plt.figure()

    n_values = sorted(df["N"].unique())

    # -------------------------------------------------------------------------
    # Plot RbCP curves (theory + MC)
    # -------------------------------------------------------------------------
    for N in n_values:

        df_sub = df[df["N"] == N]

        # Upper bound
        plt.plot(
            df_sub["M_RbCP"],
            df_sub["upper_bound"],
            label=f"N={N} Upper",
            linestyle="-"
        )

        # MSE*
        plt.plot(
            df_sub["M_RbCP"],
            df_sub["mse_star"],
            label=f"N={N} MSE*",
            linestyle="--"
        )

        # Monte Carlo
        plt.scatter(
            df_sub["M_RbCP"],
            df_sub["mse_mc_mean"],
            label=f"N={N} MC",
            marker="o"
        )

    # -------------------------------------------------------------------------
    # Benchmark curve
    # Only one curve (independent of N)
    # -------------------------------------------------------------------------
    if cfg["benchmark"]["enabled"]:

        df_bench = df[df["N"] == cfg["benchmark"]["compare_with_n"]]

        plt.plot(
            df_bench["M_RbCP"],
            df_bench["benchmark_mse"],
            color="red",
            linewidth=10,
            label="Benchmark"
        )

    # -------------------------------------------------------------------------
    # AXES
    # -------------------------------------------------------------------------
    plt.xlabel(r"$M_{RbCP}$")
    plt.ylabel(plot_cfg["y_axis"])

    if "x_min" in plot_cfg and "x_max" in plot_cfg:
        plt.xlim(plot_cfg["x_min"], plot_cfg["x_max"])

    if "y_min" in plot_cfg and "y_max" in plot_cfg:
        plt.ylim(plot_cfg["y_min"], plot_cfg["y_max"])

    if "x_ticks" in plot_cfg:
        plt.xticks(plot_cfg["x_ticks"])

    if plot_cfg.get("y_scale", "linear") == "log":
        plt.yscale("log")

    if plot_cfg.get("show_grid", False):
        plt.grid(True)

    if plot_cfg.get("legend", False):
        plt.legend()

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------
    formats = cfg["output"]["formats"]["plot"]

    for fmt in formats:
        path = os.path.join(output_dir, f"fig02_{timestamp}.{fmt}")
        plt.savefig(path, bbox_inches="tight")

    plt.show(block=True)
    plt.close()


# =============================================================================
# METADATA
# =============================================================================

def save_metadata(cfg, output_dir, timestamp, config_path):

    meta_path = os.path.join(output_dir, f"metadata_{timestamp}.txt")

    with open(meta_path, "w") as f:
        f.write("Figure: fig02\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Git commit: {get_git_commit()}\n")
        f.write(f"Config file: {config_path}\n")
        f.write("\n--- CONFIG SNAPSHOT ---\n")
        f.write(yaml.dump(cfg, sort_keys=False))


def copy_config_file(config_path, output_dir):
    dst = os.path.join(output_dir, "config_used.yaml")
    shutil.copy(config_path, dst)


# =============================================================================
# MAIN
# =============================================================================

def run_fig02(config_path):

    cfg = load_config(config_path)

    output_dir, timestamp = prepare_output_dir(cfg)

    print("[INFO] Output:", output_dir)

    # -------------------------------------------------------------------------
    # PIPELINE
    # -------------------------------------------------------------------------
    print("[INFO] Running Figure 2 pipeline...")
    df = generate_figure2_data(cfg)

    # -------------------------------------------------------------------------
    # SAVE DATA
    # -------------------------------------------------------------------------
    filename = f"fig02_{timestamp}.dat"
    path = os.path.join(output_dir, filename)

    delimiter = cfg.get("data_format", {}).get("delimiter", "\t")

    save_dat_file(df, path, delimiter)
    print("[INFO] Data saved:", path)

    # -------------------------------------------------------------------------
    # PLOT
    # -------------------------------------------------------------------------
    generate_plot(df, cfg, output_dir, timestamp)
    print("[INFO] Plot saved")

    # -------------------------------------------------------------------------
    # METADATA
    # -------------------------------------------------------------------------
    save_metadata(cfg, output_dir, timestamp, config_path)
    copy_config_file(config_path, output_dir)

    print("[INFO] Done")

    return df


# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    run_fig02(args.config)


if __name__ == "__main__":
    main()
