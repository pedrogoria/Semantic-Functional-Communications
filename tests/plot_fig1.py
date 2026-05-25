"""
plot_fig1.py

Plot script for Figure 1 using results saved in:
    data/results/fig1.dat

The manuscript Figure 1 shows:
- Upper bound of MSE_RbCP (Lemma 4)
- MSE*_RbCP (Proposition 2)
- Monte Carlo markers
versus N for M_RbCP in {4, 8, 16}.

This script:
1) Loads data/results/fig1.dat (tab-separated with header)
2) Plots MC points as markers
3) Plots theoretical curves (upper bound and MSE*) as lines
4) Uses log scale on the y-axis
5) Saves the figure as PNG and PDF in plots/

Designed to run directly from PyCharm "Run" with no IDE configuration.

Author: SFC Project
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _project_root_from_this_file():
    """
    Return the project root directory assuming this file is in:
        <root>/tests/plot_fig1.py
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, ".."))
    return root_dir


def _load_fig1_dat(dat_path):
    """
    Load a .dat file saved by numpy.savetxt with a header row.
    Expected columns:
        N, M_RbCP, MSE_MC, MSE_UPPER, MSE_STAR

    Returns
    -------
    data : np.ndarray (structured)
        Structured array with named columns.
    """
    # genfromtxt supports header-based named columns
    data = np.genfromtxt(
        dat_path,
        delimiter="\t",
        names=True,
        dtype=None,
        encoding=None
    )
    return data


def _ensure_dir(path):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def main():
    """
    Main plotting function.
    """
    root_dir = _project_root_from_this_file()

    # Input data path
    dat_path = os.path.join(root_dir, "data", "results", "fig1.dat")
    if not os.path.isfile(dat_path):
        raise FileNotFoundError(
            f"Could not find input file:\n{dat_path}\n"
            f"Run the Figure 1 simulation first to generate fig1.dat."
        )

    # Output plot directory
    plot_dir = os.path.join(root_dir, "plots")
    _ensure_dir(plot_dir)

    # Load data
    data = _load_fig1_dat(dat_path)

    # Basic validation of required columns
    required = ["N", "M_RbCP", "MSE_Monte_Carlo", "MSE_UPPER", "MSE_STAR"]
    for col in required:
        if col not in data.dtype.names:
            raise ValueError(
                f"Missing required column '{col}' in {dat_path}. "
                f"Found columns: {data.dtype.names}"
            )

    # Extract unique M values in sorted order
    m_values = np.unique(data["M_RbCP"])
    m_values = np.sort(m_values)

    # Create figure (size chosen to resemble typical paper aspect ratio)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    # Use default color cycle for M groups
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if len(color_cycle) == 0:
        color_cycle = ["C0", "C1", "C2", "C3", "C4"]

    # Plot each M group:
    # - MC markers (no connecting line)
    # - Upper bound as solid line
    # - MSE* as dotted line
    for idx, m in enumerate(m_values):
        c = color_cycle[idx % len(color_cycle)]

        sel = (data["M_RbCP"] == m)
        d = data[sel]

        # Sort by N to ensure correct line plotting order
        order = np.argsort(d["N"])
        N = d["N"][order]
        mse_mc = d["MSE_Monte_Carlo"][order]
        mse_upper = d["MSE_UPPER"][order]
        mse_star = d["MSE_STAR"][order]

        # Monte Carlo markers
        ax.plot(
            N, mse_mc,
            linestyle="None",
            marker="o",
            markersize=5,
            color=c
        )

        # Upper bound (solid)
        ax.plot(
            N, mse_upper,
            linestyle="-",
            linewidth=1.5,
            color=c
        )

        # MSE* (dotted)
        ax.plot(
            N, mse_star,
            linestyle=":",
            linewidth=1.8,
            color=c
        )

    # Axis labels and scale (paper-like)
    ax.set_xlabel("N")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")

    # Match typical axis limits seen in the manuscript figure
    # You can adjust these if your generated values differ.
    ax.set_xlim(min(data["N"]) - 0.5, max(data["N"]) + 0.5)

    # If MSE values are available, define sensible y-limits
    y_all = np.concatenate([data["MSE_Monte_Carlo"], data["MSE_UPPER"], data["MSE_STAR"]])
    y_all = y_all[np.isfinite(y_all)]
    y_all = y_all[y_all > 0]
    if y_all.size > 0:
        ymin = max(np.min(y_all) * 0.8, 1e-6)
        ymax = max(np.max(y_all) * 1.2, ymin * 10)
        ax.set_ylim(ymin, ymax)

    # Build legend similar to the manuscript:
    # - Colors encode M_RbCP values
    # - Line styles encode "Upper bound" and "MSE*"
    m_handles = []
    for idx, m in enumerate(m_values):
        c = color_cycle[idx % len(color_cycle)]
        h = Line2D(
            [0], [0],
            linestyle="None",
            marker="o",
            color=c,
            markersize=6,
            label=f"M_RbCP = {int(m)}"
        )
        m_handles.append(h)

    style_handles = [
        Line2D([0], [0], linestyle="-", color="black", linewidth=1.5, label="Upper bound"),
        Line2D([0], [0], linestyle=":", color="black", linewidth=1.8, label=r"$\mathrm{MSE}^{\star}_{\mathrm{RbCP}}$")
    ]

    # Combine legend entries
    handles = m_handles + style_handles
    ax.legend(handles=handles, loc="best", frameon=True)

    # Tight layout for paper-like compactness
    fig.tight_layout()

    # Save outputs
    out_png = os.path.join(plot_dir, "fig1.png")
    out_pdf = os.path.join(plot_dir, "fig1.pdf")

    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)

    print("[INFO] Figure 1 plot generated.")
    print("[INFO] Saved:", out_png)
    print("[INFO] Saved:", out_pdf)


if __name__ == "__main__":
    main()
