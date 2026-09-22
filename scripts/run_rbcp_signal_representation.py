"""
scripts/run_rbcp_signal_representation.py

Runner for RbCP signal representation vs Nyquist benchmark vs SFC.

This figure:
- uses physical system parameters (S, P, B, R, L, SNR_dB, W, tau)
- derives RbCP bins via channel capacity
- computes Nyquist benchmark via Shannon limit
- propagates the representative signal through the SFC stack
- plots:
    - band-limited signal
    - zero-mean signal
    - RbCP reconstruction
    - SFC reconstruction
    - Nyquist benchmark reconstruction

Usage (Python Console)
---------------------
from scripts.run_rbcp_signal_representation import run_rbcp_signal_representation

run_rbcp_signal_representation(
    "experiments/configs/figures/rbcp_signal_representation.yaml"
)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

# =============================================================================
# PATH HANDLING
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# PIPELINE
# =============================================================================

from sfc.pipelines.rbcp_signal_representation import (
    generate_rbcp_signal_representation_data,
)

# =============================================================================
# CONSTANTS
# =============================================================================

BASE_NAME = "rbcp_signal_representation"

DEFAULT_CONFIG_PATH = (
        PROJECT_ROOT
        / "experiments"
        / "configs"
        / "figures"
        / "rbcp_signal_representation.yaml"
)


# =============================================================================
# CONFIG
# =============================================================================

def load_config(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"Empty YAML config: {path}")

    return cfg


# =============================================================================
# OUTPUT DIR
# =============================================================================

def prepare_output_dir(cfg):
    output_cfg = cfg["output"]

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    base_dir = Path(output_cfg["base_dir"])
    if not base_dir.is_absolute():
        base_dir = PROJECT_ROOT / base_dir

    figure_dir = output_cfg.get("figure_dir", BASE_NAME)

    output_dir = base_dir / figure_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir, timestamp


# =============================================================================
# GIT META
# =============================================================================

def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


# =============================================================================
# PLOTTING
# =============================================================================

def _figure_formats(cfg):
    return cfg["output"].get("formats", {}).get("plot", ["png", "pdf"])


def generate_plot(data, cfg, output_dir, timestamp, show=False):
    """
    Generate main plot + error plot.
    """

    t = data["t"]
    x_filtered = data["x_filtered"]
    x_zero_mean = data["x_zero_mean"]
    x_rbcp = data["x_rbcp"]
    x_sfc = data["x_sfc"]
    x_benchmark = data["x_benchmark"]

    plot_cfg = cfg.get("plot", {})
    styles = plot_cfg.get("styles", {})
    labels = plot_cfg.get("labels", {})

    # -------------------------------------------------------------------------
    # MAIN SIGNAL PLOT
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(t, x_filtered, styles.get("filtered", "-"), label=labels.get("filtered", "Filtered"), linewidth=2)
    ax.plot(t, x_zero_mean, styles.get("zero_mean", "--"), label=labels.get("zero_mean", "Zero-mean"), linewidth=2)
    ax.plot(t, x_rbcp, styles.get("rbcp", "-"), label=labels.get("rbcp", "RbCP"), linewidth=2)
    ax.plot(t, x_sfc, styles.get("sfc", "-."), label=labels.get("sfc", "SFC"), linewidth=2)
    ax.plot(t, x_benchmark, styles.get("benchmark", ":"), label=labels.get("benchmark", "Benchmark"), linewidth=2)

    ax.set_xlabel(r"$t$")
    ax.set_ylabel("Amplitude")

    if plot_cfg.get("show_grid", True):
        ax.grid(True, linestyle=":", linewidth=0.7)

    if plot_cfg.get("legend", True):
        ax.legend()

    ax.set_title("RbCP vs SFC vs Nyquist Benchmark")

    fig.tight_layout()

    generated = {}

    for fmt in _figure_formats(cfg):
        path = output_dir / f"{BASE_NAME}_{timestamp}.{fmt}"
        fig.savefig(path, bbox_inches="tight", dpi=300 if fmt == "png" else None)
        generated[f"signal_{fmt}"] = str(path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    # -------------------------------------------------------------------------
    # ERROR PLOT
    # -------------------------------------------------------------------------
    err_rbcp = x_zero_mean - x_rbcp
    err_sfc = x_zero_mean - x_sfc
    err_bench = x_zero_mean - x_benchmark

    fig2, ax2 = plt.subplots(figsize=(10, 4))

    ax2.plot(t, err_rbcp, label="RbCP error", linewidth=2)
    ax2.plot(t, err_sfc, label="SFC error", linewidth=2)
    ax2.plot(t, err_bench, label="Benchmark error", linewidth=2)

    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel("Error")

    if plot_cfg.get("show_grid", True):
        ax2.grid(True, linestyle=":", linewidth=0.7)

    ax2.legend()
    ax2.set_title("Reconstruction Error")

    fig2.tight_layout()

    for fmt in _figure_formats(cfg):
        path = output_dir / f"{BASE_NAME}_{timestamp}_error.{fmt}"
        fig2.savefig(path, bbox_inches="tight", dpi=300 if fmt == "png" else None)
        generated[f"error_{fmt}"] = str(path)

    if show:
        plt.show()
    else:
        plt.close(fig2)

    # -------------------------------------------------------------------------
    # PRINT MSE
    # -------------------------------------------------------------------------
    print("\n[RESULTS]")
    print(f"MSE RbCP      = {np.mean(err_rbcp ** 2):.6e}")
    print(f"MSE SFC       = {np.mean(err_sfc ** 2):.6e}")
    print(f"MSE Benchmark = {np.mean(err_bench ** 2):.6e}")

    return generated


# =============================================================================
# METADATA
# =============================================================================

def save_metadata(cfg, output_dir, timestamp, config_path, generated_files):
    meta_path = output_dir / f"{BASE_NAME}_{timestamp}_metadata.json"

    metadata = {
        "created_at": _dt.datetime.utcnow().isoformat() + "Z",
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "git_commit": get_git_commit(),
        "platform": platform.platform(),
        "generated_files": generated_files,
        "config": cfg,
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return str(meta_path)


def copy_config_file(config_path, output_dir, timestamp):
    dst = output_dir / f"{BASE_NAME}_{timestamp}.yaml"
    shutil.copy(config_path, dst)
    return str(dst)


# =============================================================================
# MAIN
# =============================================================================

def run_rbcp_signal_representation(
        config_path=DEFAULT_CONFIG_PATH,
        show_plots=False,
):
    config_path = Path(config_path)

    cfg = load_config(config_path)

    output_dir, timestamp = prepare_output_dir(cfg)

    print("[INFO] Output directory:", output_dir)
    print("[INFO] Starting simulation...\n")

    # -------------------------------------------------------------------------
    # RUN PIPELINE
    # -------------------------------------------------------------------------
    data = generate_rbcp_signal_representation_data(cfg)

    # -------------------------------------------------------------------------
    # PLOTS
    # -------------------------------------------------------------------------
    generated_files = generate_plot(
        data,
        cfg,
        output_dir,
        timestamp,
        show=show_plots,
    )

    # -------------------------------------------------------------------------
    # METADATA
    # -------------------------------------------------------------------------
    if cfg["output"].get("save_metadata", True):
        meta_path = save_metadata(
            cfg,
            output_dir,
            timestamp,
            config_path,
            generated_files,
        )
        generated_files["metadata"] = meta_path

    # -------------------------------------------------------------------------
    # CONFIG COPY
    # -------------------------------------------------------------------------
    if cfg["output"].get("save_config_copy", True):
        cfg_copy = copy_config_file(config_path, output_dir, timestamp)
        generated_files["config"] = cfg_copy

    print("\n[INFO] Generated files:")
    for k, v in generated_files.items():
        print(f"       {k}: {v}")

    print("\n[INFO] Done ✅")

    return data


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run RbCP signal representation experiment."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively",
    )

    args = parser.parse_args()

    run_rbcp_signal_representation(
        config_path=args.config,
        show_plots=args.show,
    )


if __name__ == "__main__":
    main()

# from scripts.run_rbcp_signal_representation import run_rbcp_signal_representation
#
# data_repr = run_rbcp_signal_representation(
#     "experiments/configs/figures/rbcp_signal_representation.yaml"
# )
