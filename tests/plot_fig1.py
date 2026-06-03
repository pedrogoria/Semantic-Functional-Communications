"""
plot_fig1.py

Plot script for Figure 1 using results saved in:

    data/results/fig1.dat

This script is compatible with the current trusted Figure 1 pipeline:

    sfc/pipelines/figure1_pipeline.py

Expected current columns
------------------------
The current Figure 1 pipeline saves columns:

- N
- M_RbCP
- Q
- upper_bound
- mse_star
- mse_mc_mean
- mse_mc_std
- num_trials

Legacy compatibility
--------------------
This script also accepts older column names:

- MSE_Monte_Carlo
- MSE_UPPER
- MSE_STAR

The manuscript Figure 1 shows:
- Upper bound of MSE_RbCP
- MSE*_RbCP
- Monte Carlo markers
versus N for several M_RbCP values.

This script:
1. Loads data/results/fig1.dat
2. Plots MC points as markers
3. Plots theoretical curves as lines
4. Uses log scale on the y-axis
5. Saves the figure as PNG and PDF in plots/

Designed to run directly from PyCharm "Run" with no IDE configuration.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =============================================================================
# PATH HELPERS
# =============================================================================

def _project_root_from_this_file():
    """
    Return the project root directory assuming this file is in:

        <root>/tests/plot_fig1.py

    or directly in:

        <root>/plot_fig1.py
    """

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # If this script is inside tests/, root is one level up.
    if os.path.basename(current_dir) == "tests":
        return os.path.abspath(os.path.join(current_dir, ".."))

    # Otherwise assume the script is already in the project root.
    return current_dir


def _ensure_dir(path):
    """
    Create directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


# =============================================================================
# DATA LOADING
# =============================================================================

def _load_fig1_dat(dat_path):
    """
    Load the Figure 1 .dat file.

    The file is expected to be tab-separated with a header row.
    """

    data = np.genfromtxt(
        dat_path,
        delimiter="\t",
        names=True,
        dtype=None,
        encoding=None
    )

    # If the file has a single row, genfromtxt returns a scalar structured array.
    # Convert it to a one-row array for uniform downstream handling.
    if data.shape == ():
        data = np.array([data], dtype=data.dtype)

    return data


def _require_column(data, candidates, logical_name):
    """
    Return the first available column name among candidates.

    Parameters
    ----------
    data : np.ndarray
        Structured array.

    candidates : list[str]
        Candidate column names.

    logical_name : str
        Human-readable name used in error messages.
    """

    available = data.dtype.names

    for c in candidates:
        if c in available:
            return c

    raise ValueError(
        f"Missing required column for '{logical_name}'. "
        f"Tried candidates: {candidates}. "
        f"Found columns: {available}"
    )


def _resolve_fig1_columns(data):
    """
    Resolve current and legacy Figure 1 column names.
    """

    col_N = _require_column(
        data,
        candidates=["N"],
        logical_name="N"
    )

    col_M = _require_column(
        data,
        candidates=["M_RbCP", "M"],
        logical_name="M_RbCP"
    )

    col_mc = _require_column(
        data,
        candidates=["mse_mc_mean", "MSE_Monte_Carlo", "mse_mc", "MSE_MC"],
        logical_name="Monte Carlo MSE"
    )

    col_upper = _require_column(
        data,
        candidates=["upper_bound", "MSE_UPPER", "mse_upper", "upper"],
        logical_name="upper bound"
    )

    col_star = _require_column(
        data,
        candidates=["mse_star", "MSE_STAR", "mse_st"],
        logical_name="MSE star"
    )

    col_std = None
    for candidate in ["mse_mc_std", "MSE_Monte_Carlo_STD", "MSE_MC_STD"]:
        if candidate in data.dtype.names:
            col_std = candidate
            break

    return {
        "N": col_N,
        "M_RbCP": col_M,
        "mse_mc": col_mc,
        "upper_bound": col_upper,
        "mse_star": col_star,
        "mse_mc_std": col_std,
    }


# =============================================================================
# PLOTTING
# =============================================================================

def main():
    """
    Main plotting function.
    """

    root_dir = _project_root_from_this_file()

    dat_path = os.path.join(root_dir, "data", "results", "fig1.dat")

    if not os.path.isfile(dat_path):
        raise FileNotFoundError(
            f"Could not find input file:\n{dat_path}\n\n"
            "Run the Figure 1 simulation first to generate fig1.dat."
        )

    plot_dir = os.path.join(root_dir, "plots")
    _ensure_dir(plot_dir)

    data = _load_fig1_dat(dat_path)
    cols = _resolve_fig1_columns(data)

    m_values = np.unique(data[cols["M_RbCP"]])
    m_values = np.sort(m_values)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if len(color_cycle) == 0:
        color_cycle = ["C0", "C1", "C2", "C3", "C4"]

    marker_cycle = ["o", "s", "^", "D", "v", "P", "X"]

    all_y_values = []

    for idx, m in enumerate(m_values):
        c = color_cycle[idx % len(color_cycle)]
        marker = marker_cycle[idx % len(marker_cycle)]

        sel = data[cols["M_RbCP"]] == m
        d = data[sel]

        order = np.argsort(d[cols["N"]])

        N = np.asarray(d[cols["N"]][order], dtype=float)
        mse_mc = np.asarray(d[cols["mse_mc"]][order], dtype=float)
        mse_upper = np.asarray(d[cols["upper_bound"]][order], dtype=float)
        mse_star = np.asarray(d[cols["mse_star"]][order], dtype=float)

        all_y_values.extend(mse_mc[np.isfinite(mse_mc)])
        all_y_values.extend(mse_upper[np.isfinite(mse_upper)])
        all_y_values.extend(mse_star[np.isfinite(mse_star)])

        # ---------------------------------------------------------------------
        # Monte Carlo markers
        # ---------------------------------------------------------------------
        if cols["mse_mc_std"] is not None:
            mse_std = np.asarray(d[cols["mse_mc_std"]][order], dtype=float)

            ax.errorbar(
                N,
                mse_mc,
                yerr=mse_std,
                linestyle="None",
                marker=marker,
                markersize=5,
                color=c,
                capsize=2,
                linewidth=1.0
            )
        else:
            ax.plot(
                N,
                mse_mc,
                linestyle="None",
                marker=marker,
                markersize=5,
                color=c
            )

        # ---------------------------------------------------------------------
        # Upper bound
        # ---------------------------------------------------------------------
        ax.plot(
            N,
            mse_upper,
            linestyle="-",
            linewidth=1.5,
            color=c
        )

        # ---------------------------------------------------------------------
        # MSE*
        # ---------------------------------------------------------------------
        ax.plot(
            N,
            mse_star,
            linestyle=":",
            linewidth=1.8,
            color=c
        )

    # -------------------------------------------------------------------------
    # Axes
    # -------------------------------------------------------------------------
    ax.set_xlabel("N")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")

    finite_N = np.asarray(data[cols["N"]], dtype=float)
    finite_N = finite_N[np.isfinite(finite_N)]

    if finite_N.size > 0:
        ax.set_xlim(np.min(finite_N), np.max(finite_N))

    all_y_values = np.asarray(all_y_values, dtype=float)
    all_y_values = all_y_values[np.isfinite(all_y_values)]
    all_y_values = all_y_values[all_y_values > 0]

    if all_y_values.size > 0:
        ymin = max(np.min(all_y_values) * 0.75, 1e-8)
        ymax = max(np.max(all_y_values) * 1.25, ymin * 10.0)
        ax.set_ylim(ymin, ymax)

    ax.grid(True, which="both", linestyle=":", linewidth=0.7)

    # -------------------------------------------------------------------------
    # Legend
    # -------------------------------------------------------------------------
    m_handles = []

    for idx, m in enumerate(m_values):
        c = color_cycle[idx % len(color_cycle)]
        marker = marker_cycle[idx % len(marker_cycle)]

        h = Line2D(
            [0],
            [0],
            linestyle="None",
            marker=marker,
            color=c,
            markersize=6,
            label=f"$M_{{RbCP}} = {int(m)}$"
        )
        m_handles.append(h)

    style_handles = [
        Line2D(
            [0],
            [0],
            linestyle="-",
            color="black",
            linewidth=1.5,
            label="Upper bound"
        ),
        Line2D(
            [0],
            [0],
            linestyle=":",
            color="black",
            linewidth=1.8,
            label=r"$\mathrm{MSE}^{\star}_{\mathrm{RbCP}}$"
        ),
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="o",
            color="black",
            markersize=5,
            label="Monte Carlo"
        ),
    ]

    ax.legend(
        handles=m_handles + style_handles,
        loc="best",
        frameon=True
    )

    fig.tight_layout()

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    out_png = os.path.join(plot_dir, "fig1.png")
    out_pdf = os.path.join(plot_dir, "fig1.pdf")

    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)

    print("[INFO] Figure 1 plot generated.")
    print("[INFO] Input:", dat_path)
    print("[INFO] Saved:", out_png)
    print("[INFO] Saved:", out_pdf)


if __name__ == "__main__":
    main()
