"""
scripts/run_rbcp_benchmark_truncation_mse_vs_B.py

Runner for the truncation-com config;Runner for the truncation-comparison figure:
- runs the truncation-comparison pipeline;
- saves the dataset;
- saves metadata and a copy of the config;
- generates the main MSE plot;
- generates a diagnostics plot for M and M_RbCP.

This runner:
     MSE versus B for:
    - Benchmark (free M)
    - Benchmark (power-of-two M)
    - RbCP (free M_RbCP)
    - RbCP (power-of-two M_RbCP)

Output structure
----------------
This runner follows the project output convention:

    output.base_dir / output.figure_dir / YYYYMMDD_HHMMSS/

Example:

    data/results/rbcp_benchmark_truncation_mse_vs_B/20260525_154501/

Usage (Python Console - PyCharm)
--------------------------------
from scripts.run_rbcp_benchmark_truncation_mse_vs_B import (
    run_rbcp_benchmark_truncation_mse_vs_B,
)

df = run_rbcp_benchmark_truncation_mse_vs_B(
    "experiments/configs/figures/rbcp_benchmark_truncation_mse_vs_B.yaml"
)

Usage (terminal)
----------------
python scripts/run_rbcp_benchmark_truncation_mse_vs_B.py \\
    --config experiments/configs/figures/rbcp_benchmark_truncation_mse_vs_B.yaml
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# =============================================================================
# PATH HANDLING
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from sfc.pipelines.rbcp_benchmark_truncation_mse_vs_B import (
    generate_rbcp_benchmark_truncation_mse_vs_B_data,
    save_dat_file,
)


# =============================================================================
# CONSTANTS
# =============================================================================

BASE_NAME = "rbcp_benchmark_truncation_mse_vs_B"

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "configs"
    / "figures"
    / "rbcp_benchmark_truncation_mse_vs_B.yaml"
)


# =============================================================================
# CONFIG
# =============================================================================

def load_config(path):
    """
    Load YAML config.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"YAML config is empty: {path}")

    return cfg


# =============================================================================
# OUTPUT DIR
# =============================================================================

def prepare_output_dir(cfg):
    """
    Prepare timestamped output directory using the project convention:

        output.base_dir / output.figure_dir / YYYYMMDD_HHMMSS
    """
    output_cfg = cfg["output"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    base_dir = Path(output_cfg["base_dir"])

    if not base_dir.is_absolute():
        base_dir = PROJECT_ROOT / base_dir

    figure_dir = output_cfg.get("figure_dir", BASE_NAME)

    output_dir = base_dir / figure_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    return str(output_dir), timestamp


# =============================================================================
# GIT META
# =============================================================================

def get_git_commit():
    """
    Return current git commit hash if available.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
    except Exception:
        return "unknown"


def get_git_dirty():
    """
    Return whether git working tree has uncommitted changes.
    """
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        return bool(out)
    except Exception:
        return None


# =============================================================================
# JSON SAFE
# =============================================================================

def _json_safe_value(value: Any):
    """
    Convert numpy/pandas values into JSON-safe objects.
    """
    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isfinite(value):
            return float(value)
        return None

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


# =============================================================================
# DATA SAVE
# =============================================================================

def save_data(df, cfg, output_dir, timestamp):
    """
    Save the generated dataset in the requested formats.
    """
    data_cfg = cfg["output"].get("formats", {}).get("data", ["dat"])
    delimiter = cfg.get("data_format", {}).get("delimiter", "\t")

    base_name = f"{BASE_NAME}_{timestamp}"
    generated = {}

    for fmt in data_cfg:
        path = os.path.join(output_dir, f"{base_name}.{fmt}")

        if fmt == "dat":
            save_dat_file(df, path, delimiter=delimiter)
        elif fmt == "csv":
            df.to_csv(path, index=False, float_format="%.8e")
        else:
            raise ValueError(f"Unsupported data format: {fmt}")

        generated[fmt] = path

    return generated


def save_config_copy(config_path, output_dir, timestamp):
    """
    Copy YAML config into output directory.
    """
    dst = os.path.join(output_dir, f"{BASE_NAME}_{timestamp}.yaml")
    shutil.copyfile(config_path, dst)
    return dst


# =============================================================================
# PLOTTING
# =============================================================================

def _finite_positive(values):
    """
    Return finite positive values.
    """
    arr = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
    return arr[np.isfinite(arr) & (arr > 0)]


def _set_log_ylim(ax, series_list):
    """
    Set log-scale limits from finite positive data.
    """
    values = []

    for y in series_list:
        values.extend(_finite_positive(y))

    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return

    ymin = np.min(values)
    ymax = np.max(values)

    ax.set_ylim(
        max(ymin * 0.75, 1e-12),
        ymax * 1.25,
    )


def generate_plot(df, cfg, output_dir, timestamp, show=False):
    """
    Generate the main truncation-comparison MSE-vs-B plot.
    """
    if "B" not in df.columns:
        raise ValueError("Cannot plot main figure. Missing required column: B")

    plot_cfg = cfg.get("plot", {})
    labels = plot_cfg.get("label_map", {})

    B = np.asarray(pd.to_numeric(df["B"], errors="coerce"), dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))

    method_specs = [
        (
            "mse_benchmark_free",
            labels.get("benchmark_free", "Benchmark (free M)"),
            "-",
        ),
        (
            "mse_benchmark_pow2",
            labels.get("benchmark_pow2", "Benchmark (power-of-two M)"),
            "--",
        ),
        (
            "mse_rbcp_free",
            labels.get("rbcp_free", "RbCP (free M_RbCP)"),
            "-.",
        ),
        (
            "mse_rbcp_pow2",
            labels.get("rbcp_pow2", "RbCP (power-of-two M_RbCP)"),
            ":",
        ),
    ]

    plotted = []

    for col, label, linestyle in method_specs:
        if col not in df.columns:
            continue

        y = np.asarray(pd.to_numeric(df[col], errors="coerce"), dtype=float)

        if not np.any(np.isfinite(y)):
            continue

        ax.plot(
            B,
            y,
            linestyle=linestyle,
            linewidth=2,
            label=label,
        )
        plotted.append(y)

    if not plotted:
        raise ValueError(
            "Cannot plot main figure. No valid MSE columns found. "
            f"Available columns: {list(df.columns)}"
        )

    ax.set_xlabel(plot_cfg.get("x_axis", "B"))
    ax.set_ylabel(plot_cfg.get("y_axis", "MSE"))

    if plot_cfg.get("x_scale", "linear") == "log":
        ax.set_xscale("log")

    if plot_cfg.get("y_scale", "linear") == "log":
        ax.set_yscale("log")
        _set_log_ylim(ax, plotted)

    if plot_cfg.get("show_grid", True):
        ax.grid(True, which="both", linestyle=":", linewidth=0.7)

    if plot_cfg.get("legend", True):
        ax.legend(loc="best", frameon=True)

    ax.set_title(
        cfg.get("figure", {}).get(
            "title",
            "Benchmark and RbCP MSE versus B: free vs power-of-two truncation",
        )
    )

    fig.tight_layout()

    formats = cfg["output"].get("formats", {}).get(
        "plot",
        cfg["output"].get("formats", {}).get("figure", ["png", "pdf"]),
    )

    generated = {}
    base_name = f"{BASE_NAME}_{timestamp}"

    for fmt in formats:
        path = os.path.join(output_dir, f"{base_name}.{fmt}")

        if fmt == "png":
            fig.savefig(path, dpi=300, bbox_inches="tight")
        elif fmt == "pdf":
            fig.savefig(path, bbox_inches="tight")
        else:
            raise ValueError(f"Unsupported figure format: {fmt}")

        generated[f"main_{fmt}"] = path

    if show:
        plt.show()
    else:
        plt.close(fig)

    return generated


def generate_diagnostics_plot(df, cfg, output_dir, timestamp, show=False):
    """
    Plot diagnostic quantities versus B:
    - M_benchmark_free
    - M_benchmark_pow2
    - M_rbcp_free
    - M_rbcp_pow2
    """
    if "B" not in df.columns:
        raise ValueError("Cannot plot diagnostics. Missing required column: B")

    B = np.asarray(pd.to_numeric(df["B"], errors="coerce"), dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))

    diagnostic_specs = [
        ("M_benchmark_free", "M_benchmark (free)", "-", "o"),
        ("M_benchmark_pow2", "M_benchmark (power-of-two)", "--", "s"),
        ("M_rbcp_free", "M_RbCP (free)", "-.", "^"),
        ("M_rbcp_pow2", "M_RbCP (power-of-two)", ":", "d"),
    ]

    plotted = []

    for col, label, linestyle, marker in diagnostic_specs:
        if col not in df.columns:
            continue

        y = np.asarray(pd.to_numeric(df[col], errors="coerce"), dtype=float)

        if not np.any(np.isfinite(y)):
            continue

        ax.plot(
            B,
            y,
            linestyle=linestyle,
            marker=marker,
            linewidth=2,
            markersize=4,
            label=label,
        )
        plotted.append(y)

    if not plotted:
        raise ValueError(
            "Cannot plot diagnostics. No valid M columns found. "
            f"Available columns: {list(df.columns)}"
        )

    ax.set_xlabel("B")
    ax.set_ylabel("Value")
    ax.set_yscale("log")
    _set_log_ylim(ax, plotted)
    ax.grid(True, which="both", linestyle=":", linewidth=0.7)
    ax.legend(loc="best", frameon=True)
    ax.set_title("Diagnostic quantities versus B")

    fig.tight_layout()

    formats = cfg["output"].get("formats", {}).get(
        "plot",
        cfg["output"].get("formats", {}).get("figure", ["png", "pdf"]),
    )

    generated = {}
    base_name = f"{BASE_NAME}_{timestamp}_diagnostics"

    for fmt in formats:
        path = os.path.join(output_dir, f"{base_name}.{fmt}")

        if fmt == "png":
            fig.savefig(path, dpi=300, bbox_inches="tight")
        elif fmt == "pdf":
            fig.savefig(path, bbox_inches="tight")
        else:
            raise ValueError(f"Unsupported figure format: {fmt}")

        generated[f"diagnostics_{fmt}"] = path

    if show:
        plt.show()
    else:
        plt.close(fig)

    return generated


# =============================================================================
# METADATA
# =============================================================================

def save_metadata(cfg, df, output_dir, timestamp, config_path, generated_files):
    """
    Save metadata and a small summary of the generated dataset.
    """
    _ = cfg

    metadata_path = os.path.join(
        output_dir,
        f"{BASE_NAME}_{timestamp}_metadata.json",
    )

    metadata = {
        "created_at": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "project_root": str(PROJECT_ROOT),
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_commit": get_git_commit(),
        "git_dirty": get_git_dirty(),
        "columns": list(df.columns),
        "num_rows": int(len(df)),
        "generated_files": generated_files,
        "summary": {},
    }

    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce")
        finite = series[np.isfinite(series)]

        if finite.size > 0:
            metadata["summary"][col] = {
                "min": _json_safe_value(finite.min()),
                "max": _json_safe_value(finite.max()),
                "mean": _json_safe_value(finite.mean()),
            }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata_path


# =============================================================================
# MAIN
# =============================================================================

def run_rbcp_benchmark_truncation_mse_vs_B(config_path):
    """
    Run the truncation-comparison MSE-vs-B experiment.

    Parameters
    ----------
    config_path : str or pathlib.Path
        Path to the YAML config.

    Returns
    -------
    pandas.DataFrame
        Generated dataset.
    """
    config_path = Path(config_path)

    cfg = load_config(config_path)

    output_dir, timestamp = prepare_output_dir(cfg)

    print("[INFO] Output directory:", output_dir)
    print("[INFO] Timestamp:", timestamp)
    print("[INFO] Running rbcp_benchmark_truncation_mse_vs_B...\n")
    print(
        f"[INFO] Figure title: "
        f"{cfg.get('figure', {}).get('title', 'rbcp_benchmark_truncation_mse_vs_B')}"
    )
    print(f"[INFO] Config path: {config_path}")

    # -------------------------------------------------------------------------
    # RUN PIPELINE
    # -------------------------------------------------------------------------
    df = generate_rbcp_benchmark_truncation_mse_vs_B_data(cfg)

    print("\n[INFO] Generated dataframe preview:")
    print(df.head().to_string(index=False))

    print("\n[INFO] Columns:")
    print(list(df.columns))

    generated_files = {}

    # -------------------------------------------------------------------------
    # SAVE DATA
    # -------------------------------------------------------------------------
    if cfg["output"].get("save_dat", True):
        data_files = save_data(df, cfg, output_dir, timestamp)
        generated_files.update(data_files)
        print("[INFO] Data saved")

    # -------------------------------------------------------------------------
    # CONFIG COPY
    # -------------------------------------------------------------------------
    if cfg["output"].get("save_config_copy", True):
        copied_config = save_config_copy(config_path, output_dir, timestamp)
        generated_files["config"] = copied_config
        print("[INFO] Config copy saved")

    # -------------------------------------------------------------------------
    # PLOTS
    # -------------------------------------------------------------------------
    if cfg["output"].get("save_plot", True):
        main_figs = generate_plot(
            df=df,
            cfg=cfg,
            output_dir=output_dir,
            timestamp=timestamp,
            show=False,
        )
        generated_files.update(main_figs)
        print("[INFO] Main plot saved")

        diagnostic_figs = generate_diagnostics_plot(
            df=df,
            cfg=cfg,
            output_dir=output_dir,
            timestamp=timestamp,
            show=False,
        )
        generated_files.update(diagnostic_figs)
        print("[INFO] Diagnostics plot saved")

    # -------------------------------------------------------------------------
    # METADATA
    # -------------------------------------------------------------------------
    if cfg["output"].get("save_metadata", True):
        metadata_filename = os.path.join(
            output_dir,
            f"{BASE_NAME}_{timestamp}_metadata.json",
        )
        generated_files["metadata"] = metadata_filename

        metadata_path = save_metadata(
            cfg=cfg,
            df=df,
            output_dir=output_dir,
            timestamp=timestamp,
            config_path=config_path,
            generated_files=generated_files,
        )
        generated_files["metadata"] = metadata_path
        print("[INFO] Metadata saved")

    print("\n[INFO] Generated files:")
    for key, path in generated_files.items():
        print(f"       {key}: {path}")

    print("\n[INFO] Done ✅")

    return df


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run RbCP/Benchmark truncation MSE-vs-B experiment."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML config.",
    )

    args = parser.parse_args()

    run_rbcp_benchmark_truncation_mse_vs_B(args.config)


if __name__ == "__main__":
    main()


# from scripts.run_rbcp_benchmark_truncation_mse_vs_B import (
#     run_rbcp_benchmark_truncation_mse_vs_B,
# )
#
# df_trunc = run_rbcp_benchmark_truncation_mse_vs_B(
#     "experiments/configs/figures/rbcp_benchmark_truncation_mse_vs_B.yaml"
# )