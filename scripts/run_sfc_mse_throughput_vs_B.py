"""
scripts/run_sfc_mse_throughput_vs_B.py

Runner for Figure 7-style experiment:

    - MSE versus B for RbCP_SFC and RbCP_time
    - Throughput of RbCP_SFC versus B

This runner:
- reads the YAML config;
- runs the MSE/throughput-vs-B pipeline;
- saves the dataset;
- saves metadata and a copy of the config;
- generates a two-panel plot:
    1. MSE versus B
    2. Throughput versus B

Output structure
----------------
This runner follows the project output convention:

    output.base_dir / output.figure_dir / YYYYMMDD_HHMMSS/

Example:

    data/results/sfc_mse_throughput_vs_B/20260525_154501/

Usage Python Console
--------------------
from scripts.run_sfc_mse_throughput_vs_B import run_sfc_mse_throughput_vs_B

df = run_sfc_mse_throughput_vs_B(
    "experiments/configs/figures/sfc_mse_throughput_vs_B.yaml"
)

Usage terminal
--------------
python scripts/run_sfc_mse_throughput_vs_B.py \\
    --config experiments/configs/figures/sfc_mse_throughput_vs_B.yaml
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
from typing import Any

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

from sfc.pipelines.sfc_mse_throughput_vs_B import (
    generate_sfc_mse_throughput_vs_B_data,
    save_dat_file,
)


# =============================================================================
# CONSTANTS
# =============================================================================

BASE_NAME = "sfc_mse_throughput_vs_B"

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "configs"
    / "figures"
    / "sfc_mse_throughput_vs_B.yaml"
)


# =============================================================================
# CONFIG
# =============================================================================

def load_config(path: str | Path) -> dict:
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
# OUTPUT DIRECTORY
# =============================================================================

def prepare_output_dir(cfg: dict) -> tuple[Path, str]:
    """
    Prepare timestamped output directory using the project convention:

        output.base_dir / output.figure_dir / YYYYMMDD_HHMMSS
    """
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
# GIT METADATA
# =============================================================================

def get_git_commit() -> str:
    """
    Return current git commit hash if available.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def get_git_dirty() -> bool | None:
    """
    Return whether the git working tree has uncommitted changes.
    """
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(out.strip())
    except Exception:
        return None


# =============================================================================
# JSON HELPERS
# =============================================================================

def _json_safe_value(value: Any):
    """
    Convert numpy/pandas values into JSON-safe objects.
    """
    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        if np.isfinite(value):
            return float(value)
        return None

    if isinstance(value, np.bool_):
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

def save_data(
    df: pd.DataFrame,
    cfg: dict,
    output_dir: Path,
    timestamp: str,
) -> dict[str, str]:
    """
    Save the generated dataset in the requested formats.
    """
    data_formats = cfg["output"].get("formats", {}).get("data", ["dat"])
    delimiter = cfg.get("data_format", {}).get("delimiter", "\t")

    base_name = f"{BASE_NAME}_{timestamp}"

    generated_files: dict[str, str] = {}

    for fmt in data_formats:
        path = output_dir / f"{base_name}.{fmt}"

        if fmt == "dat":
            save_dat_file(
                df=df,
                path=str(path),
                delimiter=delimiter,
            )

        elif fmt == "csv":
            df.to_csv(
                path,
                index=False,
                float_format="%.8e",
            )

        else:
            raise ValueError(f"Unsupported data format: {fmt}")

        generated_files[fmt] = str(path)

    return generated_files


def save_config_copy(
    config_path: str | Path,
    output_dir: Path,
    timestamp: str,
) -> str:
    """
    Copy YAML config into output directory.
    """
    dst = output_dir / f"{BASE_NAME}_{timestamp}.yaml"
    shutil.copyfile(config_path, dst)

    return str(dst)


# =============================================================================
# PLOT HELPERS
# =============================================================================

def _to_numeric_array(values) -> np.ndarray:
    """
    Convert a sequence or pandas Series to a float numpy array.
    """
    return np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)


def _finite_positive(values) -> np.ndarray:
    """
    Return finite positive values from a numeric-like sequence.
    """
    arr = _to_numeric_array(values)
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


def _figure_formats(cfg: dict) -> list:
    """
    Return requested figure formats.

    Supports both:
        output.formats.plot
    and:
        output.formats.figure
    """
    formats_cfg = cfg["output"].get("formats", {})

    return formats_cfg.get(
        "plot",
        formats_cfg.get("figure", ["png", "pdf"]),
    )


# =============================================================================
# MAIN PLOT
# =============================================================================

def generate_plot(
    df: pd.DataFrame,
    cfg: dict,
    output_dir: Path,
    timestamp: str,
    show: bool = False,
) -> dict[str, str]:
    """
    Generate the Figure-7-style two-panel plot:

    1. MSE versus B
    2. Throughput versus B
    """
    if "B" not in df.columns:
        raise ValueError("Cannot plot figure. Missing required column: B")

    plot_cfg = cfg.get("plot", {})
    labels = plot_cfg.get("label_map", {})

    mse_panel_cfg = plot_cfg.get("mse_panel", {})
    throughput_panel_cfg = plot_cfg.get("throughput_panel", {})

    B = _to_numeric_array(df["B"])

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    # -------------------------------------------------------------------------
    # Panel 1: MSE versus B.
    # -------------------------------------------------------------------------
    ax_mse = axes[0]

    mse_specs = [
        (
            "mse_sfc",
            labels.get("sfc", "RbCP_SFC"),
            "-",
        ),
        (
            "mse_rbcp_time",
            labels.get("rbcp_time", "RbCP_time"),
            "--",
        ),
    ]

    plotted_mse = []

    for col, label, linestyle in mse_specs:
        if col not in df.columns:
            continue

        y = _to_numeric_array(df[col])

        if not np.any(np.isfinite(y)):
            continue

        ax_mse.plot(
            B,
            y,
            linestyle=linestyle,
            linewidth=2,
            label=label,
        )

        plotted_mse.append(y)

    ax_mse.set_ylabel(mse_panel_cfg.get("y_axis", "MSE"))

    if mse_panel_cfg.get("x_scale", "linear") == "log":
        ax_mse.set_xscale("log")

    if mse_panel_cfg.get("y_scale", "linear") == "log":
        ax_mse.set_yscale("log")
        _set_log_ylim(ax_mse, plotted_mse)

    if mse_panel_cfg.get("show_grid", True):
        ax_mse.grid(True, which="both", linestyle=":", linewidth=0.7)

    if mse_panel_cfg.get("legend", True) and plotted_mse:
        ax_mse.legend(loc="best", frameon=True)

    ax_mse.set_title(mse_panel_cfg.get("title", "MSE versus B"))

    # -------------------------------------------------------------------------
    # Panel 2: Throughput versus B.
    # -------------------------------------------------------------------------
    ax_thr = axes[1]

    throughput_specs = [
        (
            "throughput_sfc",
            labels.get("throughput_sfc", "Throughput (RbCP_SFC)"),
            "-",
        ),
        (
            "throughput_max",
            labels.get("throughput_max", "Maximum throughput (S / tau)"),
            "--",
        ),
    ]

    plotted_thr = []

    for col, label, linestyle in throughput_specs:
        if col not in df.columns:
            continue

        y = _to_numeric_array(df[col])

        if not np.any(np.isfinite(y)):
            continue

        ax_thr.plot(
            B,
            y,
            linestyle=linestyle,
            linewidth=2,
            label=label,
        )

        plotted_thr.append(y)

    ax_thr.set_xlabel(throughput_panel_cfg.get("x_axis", "B"))
    ax_thr.set_ylabel(throughput_panel_cfg.get("y_axis", "Throughput"))

    if throughput_panel_cfg.get("x_scale", "linear") == "log":
        ax_thr.set_xscale("log")

    if throughput_panel_cfg.get("y_scale", "linear") == "log":
        ax_thr.set_yscale("log")
        _set_log_ylim(ax_thr, plotted_thr)

    if throughput_panel_cfg.get("show_grid", True):
        ax_thr.grid(True, which="both", linestyle=":", linewidth=0.7)

    if throughput_panel_cfg.get("legend", True) and plotted_thr:
        ax_thr.legend(loc="best", frameon=True)

    ax_thr.set_title(throughput_panel_cfg.get("title", "Throughput versus B"))

    # -------------------------------------------------------------------------
    # Figure title / layout.
    # -------------------------------------------------------------------------
    fig.suptitle(
        cfg.get("figure", {}).get(
            "title",
            "MSE versus B for RbCP_SFC and RbCP_time, and throughput versus B",
        )
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    generated_files: dict[str, str] = {}
    base_name = f"{BASE_NAME}_{timestamp}"

    for fmt in _figure_formats(cfg):
        path = output_dir / f"{base_name}.{fmt}"

        if fmt == "png":
            fig.savefig(path, dpi=300, bbox_inches="tight")

        elif fmt == "pdf":
            fig.savefig(path, bbox_inches="tight")

        else:
            raise ValueError(f"Unsupported figure format: {fmt}")

        generated_files[f"main_{fmt}"] = str(path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return generated_files


# =============================================================================
# DIAGNOSTICS PLOT
# =============================================================================

def generate_diagnostics_plot(
    df: pd.DataFrame,
    cfg: dict,
    output_dir: Path,
    timestamp: str,
    show: bool = False,
) -> dict[str, str]:
    """
    Generate diagnostics plot for valid fraction, trial counts, and SNR.
    """
    if "B" not in df.columns:
        raise ValueError("Cannot plot diagnostics. Missing required column: B")

    B = _to_numeric_array(df["B"])

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    # -------------------------------------------------------------------------
    # 1. Valid period fraction.
    # -------------------------------------------------------------------------
    if "valid_period_fraction" in df.columns:
        y = _to_numeric_array(df["valid_period_fraction"])

        if np.any(np.isfinite(y)):
            axes[0].plot(
                B,
                y,
                linestyle="-",
                linewidth=2,
                label="Valid period fraction",
            )

    axes[0].set_ylabel("Valid fraction")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, linestyle=":", linewidth=0.7)

    if axes[0].lines:
        axes[0].legend(loc="best", frameon=True)

    # -------------------------------------------------------------------------
    # 2. Valid trial count.
    # -------------------------------------------------------------------------
    trial_specs = [
        ("num_valid_trials", "Valid trials", "-"),
        ("num_trials", "Total trials", "--"),
    ]

    for col, label, linestyle in trial_specs:
        if col not in df.columns:
            continue

        y = _to_numeric_array(df[col])

        if not np.any(np.isfinite(y)):
            continue

        axes[1].plot(
            B,
            y,
            linestyle=linestyle,
            linewidth=2,
            label=label,
        )

    axes[1].set_ylabel("Trials")
    axes[1].grid(True, linestyle=":", linewidth=0.7)

    if axes[1].lines:
        axes[1].legend(loc="best", frameon=True)

    # -------------------------------------------------------------------------
    # 3. SNR diagnostics.
    # -------------------------------------------------------------------------
    snr_specs = [
        ("SNR_dB_derived", "SNR dB", "-"),
        ("SNR_linear_derived", "SNR linear", "--"),
    ]

    for col, label, linestyle in snr_specs:
        if col not in df.columns:
            continue

        y = _to_numeric_array(df[col])

        if not np.any(np.isfinite(y)):
            continue

        axes[2].plot(
            B,
            y,
            linestyle=linestyle,
            linewidth=2,
            label=label,
        )

    axes[2].set_xlabel("B")
    axes[2].set_ylabel("SNR")
    axes[2].grid(True, linestyle=":", linewidth=0.7)

    if axes[2].lines:
        axes[2].legend(loc="best", frameon=True)

    fig.tight_layout()

    generated_files: dict[str, str] = {}
    base_name = f"{BASE_NAME}_{timestamp}_diagnostics"

    for fmt in _figure_formats(cfg):
        path = output_dir / f"{base_name}.{fmt}"

        if fmt == "png":
            fig.savefig(path, dpi=300, bbox_inches="tight")

        elif fmt == "pdf":
            fig.savefig(path, bbox_inches="tight")

        else:
            raise ValueError(f"Unsupported figure format: {fmt}")

        generated_files[f"diagnostics_{fmt}"] = str(path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return generated_files


# =============================================================================
# METADATA
# =============================================================================

def save_metadata(
    df: pd.DataFrame,
    output_dir: Path,
    timestamp: str,
    config_path: str | Path,
    generated_files: dict[str, str],
) -> str:
    """
    Save metadata and a small summary of the generated dataset.
    """
    metadata_path = output_dir / f"{BASE_NAME}_{timestamp}_metadata.json"

    metadata = {
        "created_at": _dt.datetime.utcnow().replace(microsecond=0).isoformat()
        + "Z",
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

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return str(metadata_path)


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_sfc_mse_throughput_vs_B(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    show_plots: bool = False,
) -> pd.DataFrame:
    """
    Run the Figure-7-style MSE/throughput-vs-B experiment.

    Parameters
    ----------
    config_path : str or pathlib.Path
        Path to the YAML config.

    show_plots : bool
        If True, show figures interactively.

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
    print("[INFO] Running sfc_mse_throughput_vs_B...\n")
    print(f"[INFO] Figure title: {cfg.get('figure', {}).get('title', BASE_NAME)}")
    print(f"[INFO] Config path: {config_path}")

    # -------------------------------------------------------------------------
    # Run pipeline.
    # -------------------------------------------------------------------------
    df = generate_sfc_mse_throughput_vs_B_data(cfg)

    print("\n[INFO] Generated dataframe preview:")
    print(df.head().to_string(index=False))

    print("\n[INFO] Columns:")
    print(list(df.columns))

    generated_files: dict[str, str] = {}

    # -------------------------------------------------------------------------
    # Save data.
    # -------------------------------------------------------------------------
    if cfg["output"].get("save_dat", True):
        data_files = save_data(
            df=df,
            cfg=cfg,
            output_dir=output_dir,
            timestamp=timestamp,
        )

        generated_files.update(data_files)
        print("[INFO] Data saved")

    # -------------------------------------------------------------------------
    # Save config copy.
    # -------------------------------------------------------------------------
    if cfg["output"].get("save_config_copy", True):
        copied_config = save_config_copy(
            config_path=config_path,
            output_dir=output_dir,
            timestamp=timestamp,
        )

        generated_files["config"] = copied_config
        print("[INFO] Config copy saved")

    # -------------------------------------------------------------------------
    # Save plots.
    # -------------------------------------------------------------------------
    if cfg["output"].get("save_plot", True):
        main_figs = generate_plot(
            df=df,
            cfg=cfg,
            output_dir=output_dir,
            timestamp=timestamp,
            show=show_plots,
        )

        generated_files.update(main_figs)
        print("[INFO] Plot saved")

        if cfg["output"].get("save_diagnostics_plot", True):
            diagnostic_figs = generate_diagnostics_plot(
                df=df,
                cfg=cfg,
                output_dir=output_dir,
                timestamp=timestamp,
                show=show_plots,
            )

            generated_files.update(diagnostic_figs)
            print("[INFO] Diagnostics plot saved")

    # -------------------------------------------------------------------------
    # Save metadata.
    # -------------------------------------------------------------------------
    if cfg["output"].get("save_metadata", True):
        metadata_expected_path = (
            output_dir / f"{BASE_NAME}_{timestamp}_metadata.json"
        )

        generated_files["metadata"] = str(metadata_expected_path)

        metadata_path = save_metadata(
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
    """
    CLI entry point.
    """
    parser = argparse.ArgumentParser(
        description="Run Figure 7-style SFC MSE/throughput-vs-B experiment."
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

    run_sfc_mse_throughput_vs_B(
        config_path=args.config,
        show_plots=args.show,
    )


if __name__ == "__main__":
    main()


# from scripts.run_sfc_mse_throughput_vs_B import run_sfc_mse_throughput_vs_B
#
# df_sfc_thr = run_sfc_mse_throughput_vs_B(
#     "experiments/configs/figures/sfc_mse_throughput_vs_B.yaml"
# )