"""
scripts/run_sfc_mse_throughput_vs_B.py

Runner for Figure 7-style experiment:

    - MSE versus B for RbCP_SFC and RbCP_time
    - Throughput of RbCP_SFC versus B

This runner:
- reads the YAML config
- runs the MSE/throughput-vs-B pipeline
- saves the .dat output
- saves metadata and a copy of the config
- generates a two-panel plot:
    1. MSE versus B
    2. Throughput versus B

Usage (Python Console - PyCharm)
--------------------------------
from scripts.run_sfc_mse_throughput_vs_B import run_sfc_mse_throughput_vs_B

df = run_sfc_mse_throughput_vs_B(
    "experiments/configs/figures/sfc_mse_throughput_vs_B.yaml"
)
"""

import argparse
import datetime
import os
import shutil
import subprocess

import matplotlib.pyplot as plt
import yaml

from sfc.pipelines.sfc_mse_throughput_vs_B import (
    generate_sfc_mse_throughput_vs_B_data,
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

    base_name = f"sfc_mse_throughput_vs_B_{timestamp}"

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
    Generate the Figure-7-style two-panel plot:
      1. MSE versus B
      2. Throughput versus B
    """

    plot_cfg = cfg.get("plot", {})
    labels = plot_cfg.get("label_map", {})

    mse_panel_cfg = plot_cfg.get("mse_panel", {})
    throughput_panel_cfg = plot_cfg.get("throughput_panel", {})

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    # -------------------------------------------------------------------------
    # Panel 1: MSE versus B
    # -------------------------------------------------------------------------
    ax = axes[0]

    if "mse_sfc" in df.columns and df["mse_sfc"].notna().any():
        ax.plot(
            df["B"],
            df["mse_sfc"],
            linestyle="-",
            linewidth=2,
            label=labels.get("sfc", "RbCP_SFC")
        )

    if "mse_rbcp_time" in df.columns and df["mse_rbcp_time"].notna().any():
        ax.plot(
            df["B"],
            df["mse_rbcp_time"],
            linestyle="--",
            linewidth=2,
            label=labels.get("rbcp_time", "RbCP_time")
        )

    ax.set_xlabel(mse_panel_cfg.get("x_axis", "B"))
    ax.set_ylabel(mse_panel_cfg.get("y_axis", "MSE"))

    if mse_panel_cfg.get("x_scale", "linear") == "log":
        ax.set_xscale("log")

    if mse_panel_cfg.get("y_scale", "linear") == "log":
        ax.set_yscale("log")

    if mse_panel_cfg.get("show_grid", True):
        ax.grid(True)

    if mse_panel_cfg.get("legend", True):
        ax.legend()

    ax.set_title(mse_panel_cfg.get("title", "MSE versus B"))

    # -------------------------------------------------------------------------
    # Panel 2: Throughput versus B
    # -------------------------------------------------------------------------
    ax = axes[1]

    if "throughput_sfc" in df.columns and df["throughput_sfc"].notna().any():
        ax.plot(
            df["B"],
            df["throughput_sfc"],
            linestyle="-",
            linewidth=2,
            label=labels.get("throughput_sfc", "Throughput (RbCP_SFC)")
        )

    if "throughput_max" in df.columns and df["throughput_max"].notna().any():
        ax.plot(
            df["B"],
            df["throughput_max"],
            linestyle="--",
            linewidth=2,
            label=labels.get("throughput_max", "Maximum throughput (S / tau)")
        )

    ax.set_xlabel(throughput_panel_cfg.get("x_axis", "B"))
    ax.set_ylabel(throughput_panel_cfg.get("y_axis", "Throughput"))

    if throughput_panel_cfg.get("x_scale", "linear") == "log":
        ax.set_xscale("log")

    if throughput_panel_cfg.get("y_scale", "linear") == "log":
        ax.set_yscale("log")

    if throughput_panel_cfg.get("show_grid", True):
        ax.grid(True)

    if throughput_panel_cfg.get("legend", True):
        ax.legend()

    ax.set_title(throughput_panel_cfg.get("title", "Throughput versus B"))

    # -------------------------------------------------------------------------
    # Figure title / layout
    # -------------------------------------------------------------------------
    fig.suptitle(cfg.get(
        "figure", {}
    ).get(
        "title",
        "MSE versus B for RbCP_SFC and RbCP_time, and throughput versus B"
    ))
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    for fmt in cfg["output"]["formats"]["plot"]:
        fig.savefig(
            os.path.join(output_dir, f"sfc_mse_throughput_vs_B_{timestamp}.{fmt}"),
            bbox_inches="tight"
        )

    plt.show(block=True)
    plt.close(fig)


# =============================================================================
# METADATA
# =============================================================================

def save_metadata(cfg, df, output_dir, timestamp, config_path):
    """
    Save metadata and a small summary of the generated dataset.
    """

    meta_path = os.path.join(output_dir, f"metadata_{timestamp}.txt")

    with open(meta_path, "w") as f:
        f.write("Experiment: sfc_mse_throughput_vs_B\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Git commit: {get_git_commit()}\n")
        f.write(f"Config file: {config_path}\n\n")

        f.write("--- DATA SUMMARY ---\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Columns: {list(df.columns)}\n\n")

        if "B" in df.columns:
            f.write(f"B min: {df['B'].min()}\n")
            f.write(f"B max: {df['B'].max()}\n\n")

        if "SNR_dB_derived" in df.columns:
            f.write(f"SNR_dB_derived min: {df['SNR_dB_derived'].min()}\n")
            f.write(f"SNR_dB_derived max: {df['SNR_dB_derived'].max()}\n\n")

        if "mse_sfc" in df.columns:
            f.write(f"mse_sfc min: {df['mse_sfc'].min()}\n")
            f.write(f"mse_sfc max: {df['mse_sfc'].max()}\n\n")

        if "mse_rbcp_time" in df.columns:
            f.write(f"mse_rbcp_time min: {df['mse_rbcp_time'].min()}\n")
            f.write(f"mse_rbcp_time max: {df['mse_rbcp_time'].max()}\n\n")

        if "throughput_sfc" in df.columns:
            f.write(f"throughput_sfc min: {df['throughput_sfc'].min()}\n")
            f.write(f"throughput_sfc max: {df['throughput_sfc'].max()}\n\n")

        if "valid_period_fraction" in df.columns:
            f.write(f"valid_period_fraction min: {df['valid_period_fraction'].min()}\n")
            f.write(f"valid_period_fraction max: {df['valid_period_fraction'].max()}\n\n")

        f.write("--- CONFIG SNAPSHOT ---\n\n")
        f.write(yaml.dump(cfg, sort_keys=False))


def copy_config_file(config_path, output_dir):
    shutil.copy(config_path, os.path.join(output_dir, "config_used.yaml"))


# =============================================================================
# MAIN
# =============================================================================

def run_sfc_mse_throughput_vs_B(config_path):
    """
    Run the Figure-7-style MSE/throughput-vs-B experiment.

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
    print("[INFO] Running sfc_mse_throughput_vs_B...\n")
    print(f"[INFO] Figure title: {cfg.get('figure', {}).get('title', 'sfc_mse_throughput_vs_B')}")
    print(f"[INFO] Config path: {config_path}")

    # -------------------------------------------------------------------------
    # RUN PIPELINE
    # -------------------------------------------------------------------------
    df = generate_sfc_mse_throughput_vs_B_data(cfg)

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

    run_sfc_mse_throughput_vs_B(args.config)


if __name__ == "__main__":
    main()
