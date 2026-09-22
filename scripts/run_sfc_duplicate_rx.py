"""
scripts/run_sfc_duplicate_rx.py

Runner for the Figure 4-style experiment:
average probability of receiving duplicate values (epsilon) versus B.

This runner:
- reads the YAML config;
- runs the SFC duplicate-RX pipeline;
- saves the dataset;
- saves metadata and a copy of the config;
- generates the epsilon-vs-B plot.

Output structure
----------------
This runner follows the project output convention:

    output.base_dir / output.figure_dir / YYYYMMDD_HHMMSS/

Example:

    data/results/sfc_duplicate_rx/20260525_154501/

Usage Python Console
--------------------
from scripts.run_sfc_duplicate_rx import run_sfc_duplicate_rx

df = run_sfc_duplicate_rx(
    "experiments/configs/figures/sfc_duplicate_rx.yaml"
)

Usage terminal
--------------
python scripts/run_sfc_duplicate_rx.py \\
    --config experiments/configs/figures/sfc_duplicate_rx.yaml
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

from sfc.pipelines.sfc_duplicate_rx import (
    generate_sfc_duplicate_rx_data,
    save_dat_file,
)

# =============================================================================
# CONSTANTS
# =============================================================================

BASE_NAME = "sfc_duplicate_rx"

DEFAULT_CONFIG_PATH = (
        PROJECT_ROOT
        / "experiments"
        / "configs"
        / "figures"
        / "sfc_duplicate_rx.yaml"
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
        min(ymax * 1.25, 1.25),
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
    Generate the Figure 4-style epsilon-vs-B plot.

    Curves:
    1. upper bound
    2. random signals / clean
    3. random signals / AWGN
    4. uniform t / clean
    5. uniform t / AWGN
    """
    if "B" not in df.columns:
        raise ValueError("Cannot plot main figure. Missing required column: B")

    plot_cfg = cfg.get("plot", {})
    labels = plot_cfg.get("label_map", {})

    B = _to_numeric_array(df["B"])

    fig, ax = plt.subplots(figsize=(9, 5))

    curve_specs = [
        (
            "epsilon_upper_bound",
            labels.get("upper_bound", "Upper bound"),
            "-",
            None,
        ),
        (
            "epsilon_random_clean",
            labels.get("random_clean", "Random signals (clean)"),
            "-",
            "o",
        ),
        (
            "epsilon_random_awgn",
            labels.get("random_awgn", "Random signals (AWGN)"),
            "--",
            "o",
        ),
        (
            "epsilon_uniform_clean",
            labels.get("uniform_clean", "Uniform t (clean)"),
            "-",
            "s",
        ),
        (
            "epsilon_uniform_awgn",
            labels.get("uniform_awgn", "Uniform t (AWGN)"),
            "--",
            "s",
        ),
    ]

    plotted = []

    for col, label, linestyle, marker in curve_specs:
        if col not in df.columns:
            continue

        y = _to_numeric_array(df[col])

        if not np.any(np.isfinite(y)):
            continue

        ax.plot(
            B,
            y,
            linestyle=linestyle,
            marker=marker,
            linewidth=2,
            markersize=4 if marker is not None else 0,
            label=label,
        )

        plotted.append(y)

    if not plotted:
        raise ValueError(
            "Cannot plot main figure. No valid epsilon columns found. "
            f"Available columns: {list(df.columns)}"
        )

    ax.set_xlabel(plot_cfg.get("x_axis", "B"))
    ax.set_ylabel(plot_cfg.get("y_axis", "epsilon"))

    if plot_cfg.get("x_scale", "linear") == "log":
        ax.set_xscale("log")

    if plot_cfg.get("y_scale", "linear") == "log":
        ax.set_yscale("log")
        _set_log_ylim(ax, plotted)
    else:
        ax.set_ylim(-0.05, 1.05)

    if plot_cfg.get("show_grid", True):
        ax.grid(True, which="both", linestyle=":", linewidth=0.7)

    if plot_cfg.get("legend", True):
        ax.legend(loc="best", frameon=True)

    ax.set_title(
        cfg.get("figure", {}).get(
            "title",
            "Average probability of receiving duplicate values versus B",
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

def run_sfc_duplicate_rx(
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        show_plots: bool = False,
) -> pd.DataFrame:
    """
    Run the Figure 4-style duplicate-RX experiment.

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
    print("[INFO] Running sfc_duplicate_rx...\n")
    print(f"[INFO] Figure title: {cfg.get('figure', {}).get('title', BASE_NAME)}")
    print(f"[INFO] Config path: {config_path}")

    # -------------------------------------------------------------------------
    # Run pipeline.
    # -------------------------------------------------------------------------
    df = generate_sfc_duplicate_rx_data(cfg)

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
    # Save plot.
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
        description="Run Figure 4-style SFC duplicate-RX experiment."
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

    run_sfc_duplicate_rx(
        config_path=args.config,
        show_plots=args.show,
    )


if __name__ == "__main__":
    main()

# from scripts.run_sfc_duplicate_rx import run_sfc_duplicate_rx
#
# df_dup = run_sfc_duplicate_rx(
#     "experiments/configs/figures/sfc_duplicate_rx.yaml"
# )
