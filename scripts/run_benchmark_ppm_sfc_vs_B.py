"""
scripts/run_benchmark_ppm_sfc_vs_B.py

Runner for:

    sfc/pipelines/.base_dir / output.figure_dir / YYYYMMDD_HHMMSS/    sfc/pipelines/benchmark_ppm_sfc_vs_B.py

This runner loads a YAML configuration, runs the Benchmark/PPM/SFC comparison
pipeline, and saves the resulting data table, metadata, config copy, and figures.

Compared methods
----------------
- Benchmark
- PPM
- SFC

Output structure
----------------
This runner follows the project output convention:

Example:

    data/results/benchmark_ppm_sfc_vs_B/20260525_154501/

All generated artifacts are saved inside that timestamped directory.

Files generated
---------------
Depending on cfg["output"]["formats"], typical files are:

- benchmark_ppm_sfc_vs_B_<timestamp>.dat
- benchmark_ppm_sfc_vs_B_<timestamp>.csv
- benchmark_ppm_sfc_vs_B_<timestamp>.yaml
- benchmark_ppm_sfc_vs_B_<timestamp>_metadata.json
- benchmark_ppm_sfc_vs_B_<timestamp>.png
- benchmark_ppm_sfc_vs_B_<timestamp>.pdf
- benchmark_ppm_sfc_vs_B_<timestamp>_diagnostics.png
- benchmark_ppm_sfc_vs_B_<timestamp>_diagnostics.pdf

Usage
-----
Python console:

    from scripts.run_benchmark_ppm_sfc_vs_B import run_benchmark_ppm_sfc_vs_B

    df = run_benchmark_ppm_sfc_vs_B(
        config_path="experiments/configs/figures/benchmark_ppm_sfc_vs_B.yaml"
    )

Terminal:

    python scripts/run_benchmark_ppm_sfc_vs_B.py \\
        --config experiments/configs/figures/benchmark_ppm_sfc_vs_B.yaml

Optional:

    python scripts/run_benchmark_ppm_sfc_vs_B.py \\
        --config experiments/configs/figures/benchmark_ppm_sfc_vs_B.yaml \\
        --show


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


# =============================================================================
# PIPELINE IMPORTS
# =============================================================================

try:
    from sfc.pipelines.benchmark_ppm_sfc_vs_B import (
        generate_benchmark_ppm_sfc_vs_B_data,
        save_dat_file,
    )
except ImportError:
    from sfc.pipelines.benchmark_ppm_sfc_vs_B import (
        generate_benchmark_ppm_sfc_vs_B_data,
    )

    def save_dat_file(df, path, delimiter="\t"):
        """
        Local fallback if the pipeline does not expose save_dat_file.
        """
        df.to_csv(
            path,
            sep=delimiter,
            index=False,
            float_format="%.8e",
        )


# =============================================================================
# DEFAULT CONFIG
# =============================================================================

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "configs"
    / "figures"
    / "benchmark_ppm_sfc_vs_B.yaml"
)

BASE_NAME = "benchmark_ppm_sfc_vs_B"


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
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"YAML config is empty: {config_path}")

    return cfg


# =============================================================================
# OUTPUT DIR
# =============================================================================

def prepare_output_dir(cfg):
    """
    Prepare timestamped output directory using the project convention:

        base_dir / figure_dir / YYYYMMDD_HHMMSS
    """
    output_cfg = cfg.get("output", {})

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    base_dir = Path(
        output_cfg.get(
            "base_dir",
            PROJECT_ROOT / "data" / "results",
        )
    )

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
    output_cfg = cfg.get("output", {})
    data_cfg = output_cfg.get("formats", {}).get("data", ["dat", "csv"])
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
# PRINT HELPERS
# =============================================================================

def print_result_diagnostics(df: pd.DataFrame):
    """
    Print important result diagnostics explicitly.
    """
    print("\n[INFO] MSE summary:")

    mse_cols = [
        "B",
        "mse_benchmark",
        "mse_benchmark_fdma",
        "mse_ppm",
        "mse_ppm_fdma",
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
        "M_benchmark",
        "M_benchmark_min",
        "M_benchmark_mean",
        "M_benchmark_max",
        "M_ppm",
        "M_ppm_min",
        "M_ppm_max",
        "ppm_fs_msg",
        "ppm_fs_msg_min",
        "ppm_fs_msg_max",
        "M_time",
        "M_rbcp",
        "SNR_dB",
        "SNR_dB_derived",
        "SNR_sensor_dB",
        "SNR_sensor_dB_derived",
    ]
    resource_cols = [c for c in resource_cols if c in df.columns]

    if resource_cols:
        print(df[resource_cols].to_string(index=False))
    else:
        print("[WARN] Resource columns not found in DataFrame.")

    print("\n[INFO] Available columns:")
    print(list(df.columns))


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
        ("mse_benchmark", "Benchmark", "x", "-"),
        ("mse_benchmark_fdma", "Benchmark + FDMA", "x", "--"),
        ("mse_ppm", "PPM", "s", "-"),
        ("mse_ppm_fdma", "PPM + FDMA", "s", "--"),
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
    ax.set_title("Benchmark, PPM, and SFC versus total bandwidth")

    _set_log_ylim(ax, plotted)

    fig.tight_layout()

    formats = cfg.get("output", {}).get("formats", {}).get("figure", ["png", "pdf"])
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

    fig, axes = plt.subplots(3, 1, figsize=(7.6, 8.4), sharex=True)

    # -------------------------------------------------------------------------
    # 1. Benchmark quantization/resource diagnostics.
    # -------------------------------------------------------------------------
    benchmark_specs = [
        ("M_benchmark", "Benchmark M", "x-"),
        ("M_benchmark_min", "Benchmark M min", "x-"),
        ("M_benchmark_mean", "Benchmark M mean", "o-"),
        ("M_benchmark_max", "Benchmark M max", "x--"),
    ]

    for col, label, style in benchmark_specs:
        if col not in df.columns:
            continue

        y = np.asarray(pd.to_numeric(df[col], errors="coerce"), dtype=float)

        if np.any(np.isfinite(y)):
            axes[0].plot(
                B,
                y,
                style,
                linewidth=1.5,
                markersize=5,
                label=label,
            )

    axes[0].set_ylabel("Benchmark M")
    axes[0].grid(True, linestyle=":", linewidth=0.7)

    if axes[0].lines:
        axes[0].legend(loc="best", frameon=True)

    # -------------------------------------------------------------------------
    # 2. PPM diagnostics.
    # -------------------------------------------------------------------------
    ppm_specs = [
        ("M_ppm", "PPM M", "s-"),
        ("M_ppm_min", "PPM M min", "s-"),
        ("M_ppm_max", "PPM M max", "s--"),
        ("ppm_fs_msg", "PPM fs_msg", "o-"),
        ("ppm_fs_msg_min", "PPM fs_msg min", "o-"),
        ("ppm_fs_msg_max", "PPM fs_msg max", "o--"),
    ]

    for col, label, style in ppm_specs:
        if col not in df.columns:
            continue

        y = np.asarray(pd.to_numeric(df[col], errors="coerce"), dtype=float)

        if np.any(np.isfinite(y)):
            axes[1].plot(
                B,
                y,
                style,
                linewidth=1.5,
                markersize=5,
                label=label,
            )

    axes[1].set_ylabel("PPM diagnostics")
    axes[1].grid(True, linestyle=":", linewidth=0.7)

    if axes[1].lines:
        axes[1].legend(loc="best", frameon=True)

    # -------------------------------------------------------------------------
    # 3. SFC / channel diagnostics.
    # -------------------------------------------------------------------------
    sfc_specs = [
        ("M_time", "M_time", "o-"),
        ("M_rbcp", "M_RbCP", "^-"),
        ("sfc_sed_valid_fraction", "SFC+SED valid fraction", "s--"),
        ("valid_period_fraction", "Valid period fraction", "s--"),
    ]

    for col, label, style in sfc_specs:
        if col not in df.columns:
            continue

        y = np.asarray(pd.to_numeric(df[col], errors="coerce"), dtype=float)

        if np.any(np.isfinite(y)):
            axes[2].plot(
                B,
                y,
                style,
                linewidth=1.5,
                markersize=5,
                label=label,
            )

    axes[2].set_xlabel("Total bandwidth B")
    axes[2].set_ylabel("SFC diagnostics")
    axes[2].grid(True, linestyle=":", linewidth=0.7)

    if axes[2].lines:
        axes[2].legend(loc="best", frameon=True)

    fig.tight_layout()

    formats = cfg.get("output", {}).get("formats", {}).get("figure", ["png", "pdf"])
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

def run_benchmark_ppm_sfc_vs_B(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    show_plots: bool = False,
) -> pd.DataFrame:
    """
    Run Benchmark/PPM/SFC comparison versus B following the project output
    convention.
    """
    config_path = Path(config_path)

    print("\n[RUN BENCHMARK PPM SFC VS B]")
    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] Config path: {config_path}")

    cfg = load_yaml_config(config_path)

    output_dir, timestamp = prepare_output_dir(cfg)

    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Timestamp: {timestamp}")

    df = generate_benchmark_ppm_sfc_vs_B_data(cfg)

    print_result_diagnostics(df)

    generated_files = {}

    data_files = save_data(df, cfg, output_dir, timestamp)
    generated_files.update(data_files)

    copied_config_path = save_config_copy(config_path, output_dir, timestamp)
    generated_files["config"] = copied_config_path

    if cfg.get("output", {}).get("save_figures", True):
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

    metadata_filename = os.path.join(
        output_dir,
        f"{BASE_NAME}_{timestamp}_metadata.json",
    )
    generated_files["metadata"] = metadata_filename

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
        description="Run Benchmark/PPM/SFC comparison versus total bandwidth B."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML config.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively.",
    )

    args = parser.parse_args()

    run_benchmark_ppm_sfc_vs_B(
        config_path=args.config,
        show_plots=args.show,
    )


if __name__ == "__main__":
    main()


# from scripts.run_benchmark_ppm_sfc_vs_B import run_benchmark_ppm_sfc_vs_B
#
# df = run_benchmark_ppm_sfc_vs_B()
