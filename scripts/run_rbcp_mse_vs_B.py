"""
scripts/run_rbcp_mse_vs_B.py

Runner for manuscript Figure 5-style experiment:

    MSE versus B for:
    - Benchmark Approach
    - RbCP
    - RbCP_time
    - SFC

This runner:
- reads the YAML config
- runs the MSE-vs-B pipeline
- saves the .dat output
- saves metadata and a copy of the config
- generates the plot

Usage (Python Console - PyCharm)
--------------------------------
from scripts.run_rbcp_mse_vs_B import run_rbcp_mse_vs_B

df = run_rbcp_mse_vs_B(
    "experiments/configs/figures/rbcp_mse_vs_B.yaml"
)
"""

import argparse
import datetime
import os
import shutil
import subprocess

import matplotlib.pyplot as plt
import yaml

from sfc.pipelines.rbcp_mse_vs_B import (
    generate_rbcp_mse_vs_B_data,
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

    output_dir = os.path.join(
        output_cfg["base_dir"],
        output_cfg["figure_dir"],
        timestamp
    )

    os.makedirs(output_dir, exist_ok=True)

    return output_dir, timestamp


# =============================================================================
# GIT META
# =============================================================================

def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        return "unknown"


# =============================================================================
# DATA SAVE
# =============================================================================

def save_data(df, cfg, output_dir, timestamp):
    """
    Save the generated dataset in the requested formats.
    """

    data_cfg = cfg["output"].get("formats", {}).get("data", ["dat"])
    delimiter = cfg.get("data_format", {}).get("delimiter", "\t")

    base_name = f"rbcp_mse_vs_B_{timestamp}"

    for fmt in data_cfg:
        path = os.path.join(output_dir, f"{base_name}.{fmt}")

        if fmt == "dat":
            save_dat_file(df, path, delimiter=delimiter)
        elif fmt == "csv":
            df.to_csv(path, index=False)
        else:
            raise ValueError(f"Unsupported data format: {fmt}")


# =============================================================================
# PLOTTING
# =============================================================================

def generate_plot(df, cfg, output_dir, timestamp):
    """
    Generate the Figure 5-style MSE-vs-B plot.
    """

    plot_cfg = cfg.get("plot", {})
    labels = plot_cfg.get("label_map", {})

    plt.figure(figsize=(9, 5))

    # -------------------------------------------------------------------------
    # Benchmark Approach
    # -------------------------------------------------------------------------
    if "mse_benchmark" in df.columns and df["mse_benchmark"].notna().any():
        plt.plot(
            df["B"],
            df["mse_benchmark"],
            linestyle="-",
            linewidth=2,
            label=labels.get("benchmark", "Benchmark Approach")
        )

    # -------------------------------------------------------------------------
    # RbCP
    # -------------------------------------------------------------------------
    if "mse_rbcp" in df.columns and df["mse_rbcp"].notna().any():
        plt.plot(
            df["B"],
            df["mse_rbcp"],
            linestyle="--",
            linewidth=2,
            label=labels.get("rbcp", "RbCP")
        )

    # -------------------------------------------------------------------------
    # RbCP_time
    # -------------------------------------------------------------------------
    if "mse_rbcp_time" in df.columns and df["mse_rbcp_time"].notna().any():
        plt.plot(
            df["B"],
            df["mse_rbcp_time"],
            linestyle="-.",
            linewidth=2,
            label=labels.get("rbcp_time", "RbCP_time")
        )

    # -------------------------------------------------------------------------
    # SFC
    # -------------------------------------------------------------------------
    if "mse_sfc" in df.columns and df["mse_sfc"].notna().any():
        plt.plot(
            df["B"],
            df["mse_sfc"],
            linestyle=":",
            linewidth=2,
            label=labels.get("sfc", "SFC")
        )

    plt.xlabel(plot_cfg.get("x_axis", "B"))
    plt.ylabel(plot_cfg.get("y_axis", "MSE"))

    if plot_cfg.get("x_scale", "linear") == "log":
        plt.xscale("log")

    if plot_cfg.get("y_scale", "linear") == "log":
        plt.yscale("log")

    if plot_cfg.get("show_grid", True):
        plt.grid(True)

    if plot_cfg.get("legend", True):
        plt.legend()

    plt.title(cfg.get("figure", {}).get(
        "title",
        "MSE versus B for RbCP, RbCP_time, SFC, and Benchmark Approach"
    ))

    for fmt in cfg["output"]["formats"]["plot"]:
        plt.savefig(
            os.path.join(output_dir, f"rbcp_mse_vs_B_{timestamp}.{fmt}"),
            bbox_inches="tight"
        )

    plt.show(block=True)
    plt.close()


# =============================================================================
# METADATA
# =============================================================================

def save_metadata(cfg, df, output_dir, timestamp, config_path):
    """
    Save metadata and a small summary of the generated dataset.
    """

    meta_path = os.path.join(output_dir, f"metadata_{timestamp}.txt")

    with open(meta_path, "w") as f:
        f.write("Experiment: rbcp_mse_vs_B\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Git commit: {get_git_commit()}\n")
        f.write(f"Config file: {config_path}\n\n")

        f.write("--- DATA SUMMARY ---\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Columns: {list(df.columns)}\n\n")

        if "B" in df.columns:
            f.write(f"B min: {df['B'].min()}\n")
            f.write(f"B max: {df['B'].max()}\n\n")

        if "P_derived" in df.columns:
            f.write(f"P_derived min: {df['P_derived'].min()}\n")
            f.write(f"P_derived max: {df['P_derived'].max()}\n\n")

        f.write("--- CONFIG SNAPSHOT ---\n\n")
        f.write(yaml.dump(cfg, sort_keys=False))


def copy_config_file(config_path, output_dir):
    shutil.copy(config_path, os.path.join(output_dir, "config_used.yaml"))


# =============================================================================
# MAIN
# =============================================================================

def run_rbcp_mse_vs_B(config_path):
    """
    Run the Figure 5-style MSE-vs-B experiment.

    Parameters
    ----------
    config_path : str
        Path to the YAML config.

    Returns
    -------
    pandas.DataFrame
        Generated dataset.
    """

    cfg = load_config(config_path)

    output_dir, timestamp = prepare_output_dir(cfg)

    print("[INFO] Output directory:", output_dir)
    print("[INFO] Running rbcp_mse_vs_B...\n")
    print(f"[INFO] Figure title: {cfg.get('figure', {}).get('title', 'rbcp_mse_vs_B')}")
    print(f"[INFO] Config path: {config_path}")

    # -------------------------------------------------------------------------
    # RUN PIPELINE
    # -------------------------------------------------------------------------
    df = generate_rbcp_mse_vs_B_data(cfg)

    print("\n[INFO] Generated dataframe preview:")
    print(df.head())

    # -------------------------------------------------------------------------
    # SAVE DATA
    # -------------------------------------------------------------------------
    if cfg["output"].get("save_dat", True):
        save_data(df, cfg, output_dir, timestamp)
        print("[INFO] Data saved")

    # -------------------------------------------------------------------------
    # PLOT
    # -------------------------------------------------------------------------
    if cfg["output"].get("save_plot", True):
        generate_plot(df, cfg, output_dir, timestamp)
        print("[INFO] Plot saved")

    # -------------------------------------------------------------------------
    # METADATA
    # -------------------------------------------------------------------------
    if cfg["output"].get("save_metadata", True):
        save_metadata(cfg, df, output_dir, timestamp, config_path)

    if cfg["output"].get("save_config_copy", True):
        copy_config_file(config_path, output_dir)

    print("\n[INFO] Done ✅")

    return df


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    run_rbcp_mse_vs_B(args.config)


if __name__ == "__main__":
    main()

# from scripts.run_rbcp_mse_vs_B import run_rbcp_mse_vs_B
#
# df_rbcp_B = run_rbcp_mse_vs_B(
#     "experiments/configs/figures/rbcp_mse_vs_B.yaml"
# )
