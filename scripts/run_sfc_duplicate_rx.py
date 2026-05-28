"""
scripts/run_sfc_duplicate_rx.py

Runner for the Figure 4-style experiment:
average probability of receiving duplicate values (epsilon) versus B.

This runner:
- reads the YAML config
- runs the SFC duplicate-RX pipeline
- saves the .dat output
- saves metadata and a copy of the config
- generates the plot

Usage (Python Console - PyCharm)
--------------------------------
from scripts.run_sfc_duplicate_rx import run_sfc_duplicate_rx

df = run_sfc_duplicate_rx(
    "experiments/configs/figures/sfc_duplicate_rx.yaml"
)
"""

import argparse
import datetime
import os
import shutil
import subprocess

import matplotlib.pyplot as plt
import yaml

from sfc.pipelines.sfc_duplicate_rx import (
    generate_sfc_duplicate_rx_data,
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

    base_name = f"sfc_duplicate_rx_{timestamp}"

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
    Generate the Figure 4-style epsilon-vs-B plot with five curves:
    1. upper bound
    2. random signals / clean
    3. random signals / awgn
    4. uniform t / clean
    5. uniform t / awgn
    """

    plot_cfg = cfg.get("plot", {})
    labels = plot_cfg.get("label_map", {})

    plt.figure(figsize=(9, 5))

    # -------------------------------------------------------------------------
    # Upper bound
    # -------------------------------------------------------------------------
    if "epsilon_upper_bound" in df.columns and df["epsilon_upper_bound"].notna().any():
        plt.plot(
            df["B"],
            df["epsilon_upper_bound"],
            linestyle="-",
            linewidth=2,
            label=labels.get("upper_bound", "Upper bound")
        )

    # -------------------------------------------------------------------------
    # Random signals / clean
    # -------------------------------------------------------------------------
    if "epsilon_random_clean" in df.columns and df["epsilon_random_clean"].notna().any():
        plt.plot(
            df["B"],
            df["epsilon_random_clean"],
            linestyle="-",
            marker="o",
            linewidth=2,
            markersize=4,
            label=labels.get("random_clean", "Random signals (clean)")
        )

    # -------------------------------------------------------------------------
    # Random signals / awgn
    # -------------------------------------------------------------------------
    if "epsilon_random_awgn" in df.columns and df["epsilon_random_awgn"].notna().any():
        plt.plot(
            df["B"],
            df["epsilon_random_awgn"],
            linestyle="--",
            marker="o",
            linewidth=2,
            markersize=4,
            label=labels.get("random_awgn", "Random signals (awgn)")
        )

    # -------------------------------------------------------------------------
    # Uniform t / clean
    # -------------------------------------------------------------------------
    if "epsilon_uniform_clean" in df.columns and df["epsilon_uniform_clean"].notna().any():
        plt.plot(
            df["B"],
            df["epsilon_uniform_clean"],
            linestyle="-",
            marker="s",
            linewidth=2,
            markersize=4,
            label=labels.get("uniform_clean", "Uniform t (clean)")
        )

    # -------------------------------------------------------------------------
    # Uniform t / awgn
    # -------------------------------------------------------------------------
    if "epsilon_uniform_awgn" in df.columns and df["epsilon_uniform_awgn"].notna().any():
        plt.plot(
            df["B"],
            df["epsilon_uniform_awgn"],
            linestyle="--",
            marker="s",
            linewidth=2,
            markersize=4,
            label=labels.get("uniform_awgn", "Uniform t (awgn)")
        )

    plt.xlabel(plot_cfg.get("x_axis", "B"))
    plt.ylabel(plot_cfg.get("y_axis", "epsilon"))

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
        "Average probability of receiving duplicate values versus B"
    ))

    # Save
    for fmt in cfg["output"]["formats"]["plot"]:
        plt.savefig(
            os.path.join(output_dir, f"sfc_duplicate_rx_{timestamp}.{fmt}"),
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
        f.write("Experiment: sfc_duplicate_rx\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Git commit: {get_git_commit()}\n")
        f.write(f"Config file: {config_path}\n\n")

        f.write("--- DATA SUMMARY ---\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Columns: {list(df.columns)}\n\n")

        if "B" in df.columns:
            f.write(f"B min: {df['B'].min()}\n")
            f.write(f"B max: {df['B'].max()}\n\n")

        f.write("--- CONFIG SNAPSHOT ---\n\n")
        f.write(yaml.dump(cfg, sort_keys=False))


def copy_config_file(config_path, output_dir):
    shutil.copy(config_path, os.path.join(output_dir, "config_used.yaml"))


# =============================================================================
# MAIN
# =============================================================================

def run_sfc_duplicate_rx(config_path):
    """
    Run the Figure 4-style duplicate-RX experiment.

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
    print("[INFO] Running sfc_duplicate_rx...\n")
    print(f"[INFO] Figure title: {cfg.get('figure', {}).get('title', 'sfc_duplicate_rx')}")
    print(f"[INFO] Config path: {config_path}")

    # -------------------------------------------------------------------------
    # RUN PIPELINE
    # -------------------------------------------------------------------------
    df = generate_sfc_duplicate_rx_data(cfg)

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

    run_sfc_duplicate_rx(args.config)


if __name__ == "__main__":
    main()
