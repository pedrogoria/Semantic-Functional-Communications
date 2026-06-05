"""
scripts/run_fair_methods_comparison_vs_B.py

Runner for:

    sfc/pipelines/fair_methods_comparison_vs_B.py

This runner loads a YAML configuration, runs the fair comparison pipeline,
saves the resulting data table, metadata, config copy, and figures.

Compared methods
----------------
- Benchmark + FDMA
- CS + FDMA
- PPM + FDMA
- RbCP
- RbCP_time
- SFC
- SFC + SED

Output structure
----------------
This runner follows the project output convention:

    output/base_dir/output/figure_dir/YYYYMMDD_HHMMSS/

Example:

    data/results/fair_methods_comparison_vs_B/20260525_154501/

All generated artifacts are saved inside that timestamped directory.

Files generated
---------------
- fair_methods_comparison_vs_B_<timestamp>.dat
- fair_methods_comparison_vs_B_<timestamp>.csv
- fair_methods_comparison_vs_B_<timestamp>.yaml
- fair_methods_comparison_vs_B_<timestamp>_metadata.json
- fair_methods_comparison_vs_B_<timestamp>.png
- fair_methods_comparison_vs_B_<timestamp>.pdf
- fair_methods_comparison_vs_B_<timestamp>_diagnostics.png
- fair_methods_comparison_vs_B_<timestamp>_diagnostics.pdf
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
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


# =============================================================================
# PATH HANDLING
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# PIPELINE IMPORTS
# =============================================================================

from sfc.pipelines.fair_methods_comparison_vs_B import (
    generate_fair_methods_comparison_vs_B_data,
    save_dat_file,
)


# =============================================================================
# DEFAULT CONFIG
# =============================================================================

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "configs"
    / "figures"
    / "fair_methods_comparison_vs_B.yaml"
)

BASE_NAME = "fair_methods_comparison_vs_B"


# =============================================================================
# YAML
# =============================================================================

def load_yaml_config(config_path: str | Path) -> dict:
    """
    Load YAML config.
    """

    config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =============================================================================
# OUTPUT DIR
# =============================================================================

def prepare_output_dir(cfg):
    """
    Prepare timestamped output directory using the project convention:

        base_dir / figure_dir / YYYYMMDD_HHMMSS
    """

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
    """
    Return current git commit hash if available.
    """

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL
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
            stderr=subprocess.DEVNULL
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


def save_metadata(df, cfg, config_path, output_dir, timestamp, generated_files):
    """
    Save metadata JSON.
    """

    metadata_path = os.path.join(
        output_dir,
        f"{BASE_NAME}_{timestamp}_metadata.json"
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
# PRINT HELPERS
# =============================================================================

def print_result_diagnostics(df: pd.DataFrame):
    """
    Print important result diagnostics explicitly.
    """

    print("\n[INFO] Benchmark + FDMA diagnostics:")

    benchmark_cols = [
        "B",
        "M_benchmark_min",
        "M_benchmark_mean",
        "M_benchmark_max",
        "mse_benchmark_fdma",
    ]
    benchmark_cols = [c for c in benchmark_cols if c in df.columns]

    if benchmark_cols:
        print(df[benchmark_cols].to_string(index=False))
    else:
        print("[WARN] Benchmark columns not found in DataFrame.")

    print("\n[INFO] Method MSE summary:")

    mse_cols = [
        "B",
        "mse_benchmark_fdma",
        "mse_cs_fdma",
        "mse_ppm_fdma",
        "mse_rbcp",
        "mse_rbcp_time",
        "mse_sfc",
        "mse_sfc_sed",
    ]
    mse_cols = [c for c in mse_cols if c in df.columns]

    if mse_cols:
        print(df[mse_cols].to_string(index=False))
    else:
        print("[WARN] MSE columns not found in DataFrame.")

    print("\n[INFO] Resource diagnostics:")

    resource_cols = [
        "B",
        "M_benchmark_min",
        "M_benchmark_mean",
        "M_benchmark_max",
        "M_rbcp",
        "M_time",
        "cs_measurements_min",
        "cs_measurements_max",
        "ppm_fs_msg_min",
        "ppm_fs_msg_max",
        "sfc_sed_valid_fraction",
    ]
    resource_cols = [c for c in resource_cols if c in df.columns]

    if resource_cols:
        print(df[resource_cols].to_string(index=False))
    else:
        print("[WARN] Resource columns not found in DataFrame.")


# =============================================================================
# PLOT HELPERS
# =============================================================================

def _finite_positive(values) -> np.ndarray:
    """
    Return finite positive values.
    """

    arr = np.asarray(values, dtype=float)
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


def plot_main_mse_figure(df, cfg, output_dir, timestamp, show=False):
    """
    Plot main MSE figure and save in requested formats.
    """

    if "B" not in df.columns:
        raise ValueError("Cannot plot main figure. Missing required column: B")

    B = np.asarray(pd.to_numeric(df["B"], errors="coerce"), dtype=float)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    method_specs = [
        ("mse_benchmark_fdma", "Benchmark + FDMA", "x", "-"),
        ("mse_cs_fdma", "CS + FDMA", "o", "-"),
        ("mse_ppm_fdma", "PPM + FDMA", "s", "-"),
        ("mse_rbcp", "RbCP", "^", "-"),
        ("mse_rbcp_time", "RbCP_time", "D", "-"),
        ("mse_sfc", "SFC", "v", "-"),
        ("mse_sfc_sed", "SFC + SED", "P", "-"),
    ]

    plotted = []
    plotted_any = False

    for col, label, marker, linestyle in method_specs:
        if col not in df.columns:
            continue

        y = np.asarray(pd.to_numeric(df[col], errors="coerce"), dtype=float)
        plotted.append(y)
        plotted_any = True

        ax.plot(
            B,
            y,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.5,
            markersize=5,
            label=label,
        )

    if not plotted_any:
        raise ValueError(
            "Cannot plot main figure. No MSE columns found. "
            f"Available columns: {list(df.columns)}"
        )

    ax.set_xlabel("Total bandwidth B")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7)
    ax.legend(loc="best", frameon=True)
    ax.set_title("Fair methods comparison versus total bandwidth")

    _set_log_ylim(ax, plotted)

    fig.tight_layout()

    formats = cfg["output"].get("formats", {}).get("figure", ["png", "pdf"])
    base_name = f"{BASE_NAME}_{timestamp}"
    generated = {}

    for fmt in formats:
        path = os.path.join(output_dir, f"{base_name}.{fmt}")
        if fmt == "png":
            fig.savefig(path, dpi=300)
        elif fmt == "pdf":
            fig.savefig(path)
        else:
            raise ValueError(f"Unsupported figure format: {fmt}")
        generated[f"main_{fmt}"] = path

    if show:
        plt.show()
    else:
        plt.close(fig)

    return generated


def plot_diagnostics_figure(df, cfg, output_dir, timestamp, show=False):
    """
    Plot diagnostics figure and save in requested formats.
    """

    if "B" not in df.columns:
        raise ValueError("Column 'B' is required for diagnostics plot.")

    B = np.asarray(pd.to_numeric(df["B"], errors="coerce"), dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(7.6, 10.2), sharex=True)

    # -------------------------------------------------------------------------
    # 1. Benchmark quantization bins
    # -------------------------------------------------------------------------
    if "M_benchmark_min" in df.columns and "M_benchmark_max" in df.columns:
        M_min = np.asarray(pd.to_numeric(df["M_benchmark_min"], errors="coerce"), dtype=float)
        M_max = np.asarray(pd.to_numeric(df["M_benchmark_max"], errors="coerce"), dtype=float)

        axes[0].plot(
            B,
            M_min,
            "x-",
            linewidth=1.5,
            markersize=5,
            label="Benchmark M min",
        )
        axes[0].plot(
            B,
            M_max,
            "x--",
            linewidth=1.5,
            markersize=5,
            label="Benchmark M max",
        )

    axes[0].set_ylabel("Benchmark M")
    axes[0].grid(True, linestyle=":", linewidth=0.7)
    axes[0].legend(loc="best", frameon=True)

    # -------------------------------------------------------------------------
    # 2. CS measurements
    # -------------------------------------------------------------------------
    if "cs_measurements_min" in df.columns and "cs_measurements_max" in df.columns:
        axes[1].plot(
            B,
            np.asarray(pd.to_numeric(df["cs_measurements_min"], errors="coerce"), dtype=float),
            "o-",
            linewidth=1.5,
            markersize=5,
            label="CS measurements min",
        )
        axes[1].plot(
            B,
            np.asarray(pd.to_numeric(df["cs_measurements_max"], errors="coerce"), dtype=float),
            "s--",
            linewidth=1.5,
            markersize=5,
            label="CS measurements max",
        )

    axes[1].set_ylabel("CS measurements")
    axes[1].grid(True, linestyle=":", linewidth=0.7)
    axes[1].legend(loc="best", frameon=True)

    # -------------------------------------------------------------------------
    # 3. PPM fs_msg
    # -------------------------------------------------------------------------
    if "ppm_fs_msg_min" in df.columns and "ppm_fs_msg_max" in df.columns:
        axes[2].plot(
            B,
            np.asarray(pd.to_numeric(df["ppm_fs_msg_min"], errors="coerce"), dtype=float),
            "o-",
            linewidth=1.5,
            markersize=5,
            label="PPM fs_msg min",
        )
        axes[2].plot(
            B,
            np.asarray(pd.to_numeric(df["ppm_fs_msg_max"], errors="coerce"), dtype=float),
            "s--",
            linewidth=1.5,
            markersize=5,
            label="PPM fs_msg max",
        )

    axes[2].set_ylabel("PPM fs_msg")
    axes[2].grid(True, linestyle=":", linewidth=0.7)
    axes[2].legend(loc="best", frameon=True)

    # -------------------------------------------------------------------------
    # 4. M_time / M_rbcp and SFC+SED valid fraction
    # -------------------------------------------------------------------------
    lines = []
    labels = []

    if "M_time" in df.columns:
        M_time = np.asarray(pd.to_numeric(df["M_time"], errors="coerce"), dtype=float)
        line = axes[3].plot(
            B,
            M_time,
            "o-",
            linewidth=1.5,
            markersize=5,
            label="M_time",
        )
        lines.extend(line)
        labels.extend([line[0].get_label()])

    if "M_rbcp" in df.columns:
        M_rbcp_numeric = pd.to_numeric(df["M_rbcp"], errors="coerce")
        M_rbcp = np.asarray(M_rbcp_numeric, dtype=float)

        if np.any(np.isfinite(M_rbcp)):
            line = axes[3].plot(
                B,
                M_rbcp,
                "^-",
                linewidth=1.5,
                markersize=5,
                label="M_RbCP",
            )
            lines.extend(line)
            labels.extend([line[0].get_label()])

    axes[3].set_xlabel("Total bandwidth B")
    axes[3].set_ylabel("M values")
    axes[3].grid(True, linestyle=":", linewidth=0.7)

    if "sfc_sed_valid_fraction" in df.columns:
        ax2 = axes[3].twinx()
        line = ax2.plot(
            B,
            np.asarray(pd.to_numeric(df["sfc_sed_valid_fraction"], errors="coerce"), dtype=float),
            "s--",
            linewidth=1.5,
            markersize=5,
            color="C1",
            label="SFC+SED valid fraction",
        )
        ax2.set_ylabel("Valid fraction")
        ax2.set_ylim(-0.05, 1.05)

        lines.extend(line)
        labels.extend([line[0].get_label()])

    if lines:
        axes[3].legend(lines, labels, loc="best", frameon=True)

    fig.tight_layout()

    formats = cfg["output"].get("formats", {}).get("figure", ["png", "pdf"])
    base_name = f"{BASE_NAME}_{timestamp}_diagnostics"
    generated = {}

    for fmt in formats:
        path = os.path.join(output_dir, f"{base_name}.{fmt}")
        if fmt == "png":
            fig.savefig(path, dpi=300)
        elif fmt == "pdf":
            fig.savefig(path)
        else:
            raise ValueError(f"Unsupported figure format: {fmt}")
        generated[f"diagnostics_{fmt}"] = path

    if show:
        plt.show()
    else:
        plt.close(fig)

    return generated


# =============================================================================
# MAIN RUNNER FUNCTION
# =============================================================================

def run_fair_methods_comparison_vs_B(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    show_plots: bool = False,
) -> pd.DataFrame:
    """
    Run fair methods comparison versus B following the project output convention.
    """

    config_path = Path(config_path)

    print("\n[RUN FAIR METHODS COMPARISON VS B]")
    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] Config path: {config_path}")

    cfg = load_yaml_config(config_path)

    output_dir, timestamp = prepare_output_dir(cfg)

    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Timestamp: {timestamp}")

    df = generate_fair_methods_comparison_vs_B_data(cfg)

    print_result_diagnostics(df)

    generated_files = {}

    data_files = save_data(df, cfg, output_dir, timestamp)
    generated_files.update(data_files)

    copied_config_path = save_config_copy(config_path, output_dir, timestamp)
    generated_files["config"] = copied_config_path

    if cfg["output"].get("save_figures", True):
        main_figs = plot_main_mse_figure(
            df=df,
            cfg=cfg,
            output_dir=output_dir,
            timestamp=timestamp,
            show=show_plots,
        )
        generated_files.update(main_figs)

        diagnostic_figs = plot_diagnostics_figure(
            df=df,
            cfg=cfg,
            output_dir=output_dir,
            timestamp=timestamp,
            show=show_plots,
        )
        generated_files.update(diagnostic_figs)

    metadata_path = save_metadata(
        df=df,
        cfg=cfg,
        config_path=config_path,
        output_dir=output_dir,
        timestamp=timestamp,
        generated_files=generated_files,
    )
    generated_files["metadata"] = metadata_path

    print("\n[INFO] Generated files:")
    for key, path in generated_files.items():
        print(f"       {key}: {path}")

    print("\n[INFO] Result preview:")
    print(df.head().to_string(index=False))

    print("\n[INFO] Columns:")
    print(list(df.columns))

    return df


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run fair methods comparison versus total bandwidth B."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML config."
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively."
    )

    args = parser.parse_args()

    run_fair_methods_comparison_vs_B(
        config_path=args.config,
        show_plots=args.show,
    )


if __name__ == "__main__":
    main()

# from scripts.run_fair_methods_comparison_vs_B import run_fair_methods_comparison_vs_B
#
# df = run_fair_methods_comparison_vs_B()
