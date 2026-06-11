"""
scripts/run_fair_methods_comparison_vs_B.py

Runner for:

    sfc/pipelines/fair_methods_comparison_vs_B.py

This runner loads a YAML configuration, runs the fair comparison pipeline,
saves the resulting data table, metadata, config copy, and figures.

Compared methods
----------------
- Benchmark / Nyquist + FDMA
- CS + FDMA
- PPM + FDMA
- SoD + FDMA
- FRI-inspired + FDMA
- RbCP
- RbCP_time
- SFC
- SFC + SED

Output structure
----------------
This runner follows the project output convention:

    output.base_dir / output.figure_dir / YYYYMMDD_HHMMSS/

Example:

    data/results/fair_methods_comparison_vs_B/20260525_154501/

All generated artifacts are saved inside that timestamped directory.

Files generated
---------------
Depending on output.formats and save flags:

- fair_methods_comparison_vs_B_<timestamp>.dat
- fair_methods_comparison_vs_B_<timestamp>.csv
- fair_methods_comparison_vs_B_<timestamp>.yaml
- fair_methods_comparison_vs_B_<timestamp>_metadata.json
- fair_methods_comparison_vs_B_<timestamp>.png
- fair_methods_comparison_vs_B_<timestamp>.pdf
- fair_methods_comparison_vs_B_<timestamp>_diagnostics.png
- fair_methods_comparison_vs_B_<timestamp>_diagnostics.pdf

Usage Python Console
--------------------
from scripts.run_fair_methods_comparison_vs_B import run_fair_methods_comparison_vs_B

df = run_fair_methods_comparison_vs_B(
    "experiments/configs/figures/fair_methods_comparison_vs_B.yaml"
)

Usage terminal
--------------
python scripts/run_fair_methods_comparison_vs_B.py \\
    --config experiments/configs/figures/fair_methods_comparison_vs_B.yaml
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

from sfc.pipelines.fair_methods_comparison_vs_B import (
    generate_fair_methods_comparison_vs_B_data,
    save_dat_file,
)

# =============================================================================
# CONSTANTS
# =============================================================================

BASE_NAME = "fair_methods_comparison_vs_B"

DEFAULT_CONFIG_PATH = (
        PROJECT_ROOT
        / "experiments"
        / "configs"
        / "figures"
        / "fair_methods_comparison_vs_B.yaml"
)


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
    Return whether git working tree has uncommitted changes.
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
# JSON SAFE
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


def save_metadata(
        df: pd.DataFrame,
        cfg: dict,
        config_path: str | Path,
        output_dir: Path,
        timestamp: str,
        generated_files: dict[str, str],
) -> str:
    """
    Save metadata JSON.
    """
    _ = cfg

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
# PRINT HELPERS
# =============================================================================

def print_result_diagnostics(df: pd.DataFrame):
    """
    Print important result diagnostics explicitly.
    """
    print("\n[INFO] Benchmark / Nyquist + FDMA diagnostics:")

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
        "mse_sod_fdma",
        "mse_fri_fdma",
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

    print("\n[INFO] FDMA budget diagnostics:")

    fdma_cols = [
        "B",
        "fdma_capacity_sensor_min",
        "fdma_capacity_sensor_max",
        "fdma_budget_bits_sensor_min",
        "fdma_budget_bits_sensor_max",
    ]
    fdma_cols = [c for c in fdma_cols if c in df.columns]

    if fdma_cols:
        print(df[fdma_cols].to_string(index=False))
    else:
        print("[WARN] FDMA budget columns not found in DataFrame.")

    print("\n[INFO] CS / PPM diagnostics:")

    cs_ppm_cols = [
        "B",
        "cs_measurements_min",
        "cs_measurements_max",
        "cs_measurement_bits_min",
        "cs_measurement_bits_max",
        "cs_sparsity_eff_min",
        "cs_sparsity_eff_max",
        "cs_sampling_rate",
        "cs_num_samples",
        "cs_quantized_fraction",
        "ppm_fs_msg_min",
        "ppm_fs_msg_max",
    ]
    cs_ppm_cols = [c for c in cs_ppm_cols if c in df.columns]

    if cs_ppm_cols:
        print(df[cs_ppm_cols].to_string(index=False))
    else:
        print("[WARN] CS / PPM diagnostic columns not found in DataFrame.")

    print("\n[INFO] SoD / FRI diagnostics:")

    sod_fri_cols = [
        "B",
        "sod_num_events_min",
        "sod_num_events_mean",
        "sod_num_events_max",
        "sod_payload_bits_mean",
        "sod_payload_budget_bits_mean",
        "sod_fdma_feasible_fraction",
        "fri_K_mean",
        "fri_bits_location_mean",
        "fri_bits_amplitude_mean",
        "fri_budget_bits_mean",
        "fri_fdma_feasible_fraction",
    ]
    sod_fri_cols = [c for c in sod_fri_cols if c in df.columns]

    if sod_fri_cols:
        print(df[sod_fri_cols].to_string(index=False))
    else:
        print("[WARN] SoD / FRI diagnostic columns not found in DataFrame.")

    print("\n[INFO] RbCP / SFC diagnostics:")

    rbcp_sfc_cols = [
        "B",
        "M_rbcp",
        "M_time",
        "sfc_sed_valid_fraction",
    ]
    rbcp_sfc_cols = [c for c in rbcp_sfc_cols if c in df.columns]

    if rbcp_sfc_cols:
        print(df[rbcp_sfc_cols].to_string(index=False))
    else:
        print("[WARN] RbCP / SFC diagnostic columns not found in DataFrame.")


# =============================================================================
# PLOT HELPERS
# =============================================================================

def _figure_formats(cfg: dict) -> list:
    """
    Return requested figure formats.

    Supports both:
        output.formats.figure
    and:
        output.formats.plot
    """
    formats_cfg = cfg["output"].get("formats", {})

    return formats_cfg.get(
        "figure",
        formats_cfg.get("plot", ["png", "pdf"]),
    )


def _to_numeric_array(values) -> np.ndarray:
    """
    Convert a sequence or pandas Series to a float numpy array.
    """
    return np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)


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


def _plot_column_if_present(
        ax,
        df: pd.DataFrame,
        x,
        column: str,
        label: str,
        marker: str,
        linestyle: str = "-",
        linewidth: float = 1.5,
        markersize: float = 5.0,
):
    """
    Plot one column if present and at least one finite value exists.
    """
    if column not in df.columns:
        return None

    y = _to_numeric_array(df[column])

    if not np.any(np.isfinite(y)):
        return None

    line = ax.plot(
        x,
        y,
        marker=marker,
        linestyle=linestyle,
        linewidth=linewidth,
        markersize=markersize,
        label=label,
    )

    return line[0]


# =============================================================================
# MAIN MSE PLOT
# =============================================================================

def plot_main_mse_figure(
        df: pd.DataFrame,
        cfg: dict,
        output_dir: Path,
        timestamp: str,
        show: bool = False,
) -> dict[str, str]:
    """
    Plot main MSE figure and save in requested formats.
    """
    if "B" not in df.columns:
        raise ValueError("Cannot plot main figure. Missing required column: B")

    B = _to_numeric_array(df["B"])

    plot_cfg = cfg.get("plot", {})
    label_map = plot_cfg.get("label_map", {})

    fig, ax = plt.subplots(figsize=(8.2, 5.0))

    method_specs = [
        (
            "mse_benchmark_fdma",
            label_map.get("benchmark_fdma", "Benchmark / Nyquist + FDMA"),
            "x",
            "-",
        ),
        (
            "mse_cs_fdma",
            label_map.get("cs_fdma", "CS + FDMA"),
            "o",
            "-",
        ),
        (
            "mse_ppm_fdma",
            label_map.get("ppm_fdma", "PPM + FDMA"),
            "s",
            "-",
        ),
        (
            "mse_sod_fdma",
            label_map.get("sod_fdma", "SoD + FDMA"),
            "*",
            "-",
        ),
        (
            "mse_fri_fdma",
            label_map.get("fri_fdma", "FRI-inspired + FDMA"),
            "h",
            "-",
        ),
        (
            "mse_rbcp",
            label_map.get("rbcp", "RbCP"),
            "^",
            "-",
        ),
        (
            "mse_rbcp_time",
            label_map.get("rbcp_time", "RbCP_time"),
            "D",
            "-",
        ),
        (
            "mse_sfc",
            label_map.get("sfc", "SFC"),
            "v",
            "-",
        ),
        (
            "mse_sfc_sed",
            label_map.get("sfc_sed", "SFC + SED"),
            "P",
            "-",
        ),
    ]

    plotted = []

    for col, label, marker, linestyle in method_specs:
        line = _plot_column_if_present(
            ax=ax,
            df=df,
            x=B,
            column=col,
            label=label,
            marker=marker,
            linestyle=linestyle,
        )

        if line is not None:
            plotted.append(line.get_ydata())

    if not plotted:
        raise ValueError(
            "Cannot plot main figure. No valid MSE columns found. "
            f"Available columns: {list(df.columns)}"
        )

    ax.set_xlabel(plot_cfg.get("x_axis", "Total bandwidth B"))
    ax.set_ylabel(plot_cfg.get("y_axis", "MSE"))

    if plot_cfg.get("x_scale", "linear") == "log":
        ax.set_xscale("log")

    if plot_cfg.get("y_scale", "log") == "log":
        ax.set_yscale("log")
        _set_log_ylim(ax, plotted)

    if plot_cfg.get("show_grid", True):
        ax.grid(True, which="both", linestyle=":", linewidth=0.7)

    if plot_cfg.get("legend", True):
        ax.legend(loc="best", frameon=True)

    ax.set_title(
        cfg.get("figure", {}).get(
            "title",
            "Fair methods comparison versus total bandwidth",
        )
    )

    fig.tight_layout()

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

def plot_diagnostics_figure(
        df: pd.DataFrame,
        cfg: dict,
        output_dir: Path,
        timestamp: str,
        show: bool = False,
) -> dict[str, str]:
    """
    Plot diagnostics figure and save in requested formats.

    Panels:
    1. FDMA capacity / budget.
    2. Benchmark and RbCP M values.
    3. CS and PPM resources.
    4. SoD and FRI resources / feasibility.
    5. SFC+SED valid fraction.
    """
    if "B" not in df.columns:
        raise ValueError("Column 'B' is required for diagnostics plot.")

    B = _to_numeric_array(df["B"])

    fig, axes = plt.subplots(5, 1, figsize=(8.2, 12.0), sharex=True)

    # -------------------------------------------------------------------------
    # 1. FDMA budget.
    # -------------------------------------------------------------------------
    _plot_column_if_present(
        axes[0],
        df,
        B,
        "fdma_budget_bits_sensor_min",
        "FDMA budget bits min",
        "o",
        "-",
    )
    _plot_column_if_present(
        axes[0],
        df,
        B,
        "fdma_budget_bits_sensor_max",
        "FDMA budget bits max",
        "s",
        "--",
    )

    axes[0].set_ylabel("FDMA bits")
    axes[0].grid(True, linestyle=":", linewidth=0.7)

    if axes[0].lines:
        axes[0].legend(loc="best", frameon=True)

    # -------------------------------------------------------------------------
    # 2. Benchmark / RbCP M values.
    # -------------------------------------------------------------------------
    _plot_column_if_present(
        axes[1],
        df,
        B,
        "M_benchmark_min",
        "Benchmark M min",
        "x",
        "-",
    )
    _plot_column_if_present(
        axes[1],
        df,
        B,
        "M_benchmark_max",
        "Benchmark M max",
        "x",
        "--",
    )
    _plot_column_if_present(
        axes[1],
        df,
        B,
        "M_rbcp",
        "M_RbCP",
        "^",
        "-",
    )
    _plot_column_if_present(
        axes[1],
        df,
        B,
        "M_time",
        "M_time",
        "D",
        "-",
    )

    axes[1].set_ylabel("M values")
    axes[1].grid(True, linestyle=":", linewidth=0.7)

    if axes[1].lines:
        axes[1].legend(loc="best", frameon=True)

    # -------------------------------------------------------------------------
    # 3. CS and PPM resources.
    # -------------------------------------------------------------------------
    _plot_column_if_present(
        axes[2],
        df,
        B,
        "cs_measurements_min",
        "CS measurements min",
        "o",
        "-",
    )
    _plot_column_if_present(
        axes[2],
        df,
        B,
        "cs_measurements_max",
        "CS measurements max",
        "s",
        "--",
    )
    _plot_column_if_present(
        axes[2],
        df,
        B,
        "ppm_fs_msg_min",
        "PPM fs_msg min",
        "^",
        "-",
    )
    _plot_column_if_present(
        axes[2],
        df,
        B,
        "ppm_fs_msg_max",
        "PPM fs_msg max",
        "v",
        "--",
    )

    axes[2].set_ylabel("CS / PPM")
    axes[2].grid(True, linestyle=":", linewidth=0.7)

    if axes[2].lines:
        axes[2].legend(loc="best", frameon=True)

    # -------------------------------------------------------------------------
    # 4. SoD / FRI resources and feasibility.
    # -------------------------------------------------------------------------
    _plot_column_if_present(
        axes[3],
        df,
        B,
        "sod_num_events_mean",
        "SoD events mean",
        "*",
        "-",
    )
    _plot_column_if_present(
        axes[3],
        df,
        B,
        "sod_payload_bits_mean",
        "SoD payload bits mean",
        "o",
        "--",
    )
    _plot_column_if_present(
        axes[3],
        df,
        B,
        "fri_K_mean",
        "FRI K mean",
        "h",
        "-",
    )
    _plot_column_if_present(
        axes[3],
        df,
        B,
        "fri_budget_bits_mean",
        "FRI budget bits mean",
        "s",
        "--",
    )

    axes[3].set_ylabel("SoD / FRI")
    axes[3].grid(True, linestyle=":", linewidth=0.7)

    if axes[3].lines:
        axes[3].legend(loc="best", frameon=True)

    ax3b = axes[3].twinx()

    line_refs = []
    line_labels = []

    for col, label, marker, linestyle in [
        (
                "sod_fdma_feasible_fraction",
                "SoD feasible fraction",
                "P",
                ":",
        ),
        (
                "fri_fdma_feasible_fraction",
                "FRI feasible fraction",
                "X",
                ":",
        ),
    ]:
        if col in df.columns:
            y = _to_numeric_array(df[col])

            if np.any(np.isfinite(y)):
                line = ax3b.plot(
                    B,
                    y,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.5,
                    markersize=5,
                    label=label,
                )[0]
                line_refs.append(line)
                line_labels.append(label)

    ax3b.set_ylabel("Feasible fraction")
    ax3b.set_ylim(-0.05, 1.05)

    if line_refs:
        existing_lines = list(axes[3].lines) + line_refs
        existing_labels = [line.get_label() for line in axes[3].lines] + line_labels
        axes[3].legend(existing_lines, existing_labels, loc="best", frameon=True)

    # -------------------------------------------------------------------------
    # 5. SFC+SED valid fraction.
    # -------------------------------------------------------------------------
    _plot_column_if_present(
        axes[4],
        df,
        B,
        "sfc_sed_valid_fraction",
        "SFC+SED valid fraction",
        "P",
        "-",
    )

    axes[4].set_xlabel("Total bandwidth B")
    axes[4].set_ylabel("Valid fraction")
    axes[4].set_ylim(-0.05, 1.05)
    axes[4].grid(True, linestyle=":", linewidth=0.7)

    if axes[4].lines:
        axes[4].legend(loc="best", frameon=True)

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
        copied_config_path = save_config_copy(
            config_path=config_path,
            output_dir=output_dir,
            timestamp=timestamp,
        )
        generated_files["config"] = copied_config_path
        print("[INFO] Config copy saved")

    # -------------------------------------------------------------------------
    # Save figures.
    # -------------------------------------------------------------------------
    if cfg["output"].get("save_plot", cfg["output"].get("save_figures", True)):
        main_figs = plot_main_mse_figure(
            df=df,
            cfg=cfg,
            output_dir=output_dir,
            timestamp=timestamp,
            show=show_plots,
        )
        generated_files.update(main_figs)
        print("[INFO] Main MSE figure saved")

        if cfg["output"].get("save_diagnostics_plot", True):
            diagnostic_figs = plot_diagnostics_figure(
                df=df,
                cfg=cfg,
                output_dir=output_dir,
                timestamp=timestamp,
                show=show_plots,
            )
            generated_files.update(diagnostic_figs)
            print("[INFO] Diagnostics figure saved")

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
            cfg=cfg,
            config_path=config_path,
            output_dir=output_dir,
            timestamp=timestamp,
            generated_files=generated_files,
        )

        generated_files["metadata"] = metadata_path
        print("[INFO] Metadata saved")

    print("\n[INFO] Generated files:")
    for key, path in generated_files.items():
        print(f"       {key}: {path}")

    print("\n[INFO] Result preview:")
    print(df.head().to_string(index=False))

    print("\n[INFO] Columns:")
    print(list(df.columns))

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
        description="Run fair methods comparison versus total bandwidth B."
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

    run_fair_methods_comparison_vs_B(
        config_path=args.config,
        show_plots=args.show,
    )


if __name__ == "__main__":
    main()

# from scripts.run_fair_methods_comparison_vs_B import run_fair_methods_comparison_vs_B
#
# df = run_fair_methods_comparison_vs_B(
#     "experiments/configs/figures/fair_methods_comparison_vs_B.yaml"
# )
