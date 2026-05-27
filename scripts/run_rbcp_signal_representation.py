"""
scripts/run_rbcp_signal_representation.py

Runner for RbCP signal representation vs Nyquist benchmark.

This experiment:
- uses physical system parameters (S, B, R, SNR_dB, W, tau)
- derives RbCP bins via channel capacity
- computes Nyquist benchmark via Shannon limit
- produces visual comparison plots

Usage (Python Console - PyCharm):
--------------------------------
from scripts.run_rbcp_signal_representation import run_rbcp_signal_representation

run_rbcp_signal_representation(
    "experiments/configs/figures/rbcp_signal_representation.yaml"
)
"""

import argparse
import datetime
import os
import shutil
import subprocess

import matplotlib.pyplot as plt
import yaml
import numpy as np

from sfc.pipelines.rbcp_signal_representation import (
    generate_rbcp_signal_representation,
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
# PLOTTING
# =============================================================================

def generate_plot(data, cfg, output_dir, timestamp):

    t = data["t"]
    x_filtered = data["x_filtered"]
    x_zero_mean = data["x_zero_mean"]
    x_rbcp = data["x_rbcp"]
    x_benchmark = data["x_benchmark"]

    plot_cfg = cfg["plot"]

    # ------------------------------------------------------------
    # MAIN PLOT
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 5))

    plt.plot(
        t, x_filtered,
        plot_cfg["styles"]["filtered"],
        label=plot_cfg["labels"]["filtered"],
        linewidth=2
    )

    plt.plot(
        t, x_zero_mean,
        plot_cfg["styles"]["filtered"],
        label=plot_cfg["labels"]["zero_mean"],
        linewidth=2
    )

    plt.plot(
        t, x_rbcp,
        plot_cfg["styles"]["rbcp"],
        label=plot_cfg["labels"]["rbcp"],
        linewidth=2
    )

    plt.plot(
        t, x_benchmark,
        plot_cfg["styles"]["benchmark"],
        label=plot_cfg["labels"]["benchmark"],
        linewidth=2
    )

    plt.xlabel(r"$t$")
    plt.ylabel("Amplitude")

    if plot_cfg.get("show_grid", False):
        plt.grid(True)

    if plot_cfg.get("legend", False):
        plt.legend()

    plt.title("RbCP vs Nyquist Benchmark")

    # Save
    for fmt in cfg["output"]["formats"]["plot"]:
        plt.savefig(
            os.path.join(output_dir, f"rbcp_representation_{timestamp}.{fmt}"),
            bbox_inches="tight"
        )

    plt.show(block=True)
    plt.close()

    # ------------------------------------------------------------
    # ERROR PLOT
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 4))

    err_rbcp = x_zero_mean - x_rbcp
    err_bench = x_zero_mean - x_benchmark

    plt.plot(t, err_rbcp, label="RbCP error", linewidth=2)
    plt.plot(t, err_bench, label="Benchmark error", linewidth=2)

    plt.xlabel(r"$t$")
    plt.ylabel("Error")

    if plot_cfg.get("show_grid", False):
        plt.grid(True)

    plt.legend()
    plt.title("Reconstruction Error")

    for fmt in cfg["output"]["formats"]["plot"]:
        plt.savefig(
            os.path.join(output_dir, f"rbcp_error_{timestamp}.{fmt}"),
            bbox_inches="tight"
        )

    plt.show(block=True)
    plt.close()

    # ------------------------------------------------------------
    # MSE PRINT (important diagnostic)
    # ------------------------------------------------------------
    mse_rbcp = np.mean(err_rbcp**2)
    mse_bench = np.mean(err_bench**2)

    print("\n[RESULTS]")
    print(f"MSE RbCP       = {mse_rbcp:.6e}")
    print(f"MSE Benchmark  = {mse_bench:.6e}")


# =============================================================================
# METADATA
# =============================================================================

def save_metadata(cfg, output_dir, timestamp, config_path):

    meta_path = os.path.join(output_dir, f"metadata_{timestamp}.txt")

    with open(meta_path, "w") as f:
        f.write("Experiment: rbcp_signal_representation\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Git commit: {get_git_commit()}\n")
        f.write(f"Config file: {config_path}\n\n")
        f.write("--- CONFIG SNAPSHOT ---\n\n")
        f.write(yaml.dump(cfg, sort_keys=False))


def copy_config_file(config_path, output_dir):
    shutil.copy(config_path, os.path.join(output_dir, "config_used.yaml"))


# =============================================================================
# MAIN
# =============================================================================

def run_rbcp_signal_representation(config_path):

    cfg = load_config(config_path)

    output_dir, timestamp = prepare_output_dir(cfg)

    print("[INFO] Output directory:", output_dir)
    print("[INFO] Starting simulation...\n")

    # ------------------------------------------------------------
    # RUN CORE EXPERIMENT
    # ------------------------------------------------------------
    data = generate_rbcp_signal_representation(cfg)

    # ------------------------------------------------------------
    # PLOT RESULTS
    # ------------------------------------------------------------
    generate_plot(data, cfg, output_dir, timestamp)

    # ------------------------------------------------------------
    # SAVE METADATA
    # ------------------------------------------------------------
    if cfg["output"].get("save_metadata", True):
        save_metadata(cfg, output_dir, timestamp, config_path)

    if cfg["output"].get("save_config_copy", True):
        copy_config_file(config_path, output_dir)

    print("\n[INFO] Done ✅")

    return data


# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    run_rbcp_signal_representation(args.config)


if __name__ == "__main__":
    main()
