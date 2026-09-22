"""
scripts/make_quantization_error_distribution_table.py

Generate a LaTeX table for the Supplementary Material from an existing CSV
previously produced by:

    tests/quantization_error_distribution_mc.py

Important
---------
This script DOES NOT run any simulation.

It only reads an explicitly specified CSV file and generates a LaTeX table.

Usage from Python console
-------------------------
from scripts.make_quantization_error_distribution_table import make_quantization_error_distribution_table

tex_path = make_quantization_error_distribution_table(
    csv_path="tests/results/quantization_error_distribution/20260605_130414/quantization_error_distribution_metrics_20260605_130414.csv"
)

Optional output path
--------------------
tex_path = make_quantization_error_distribution_table(
    csv_path="tests/results/quantization_error_distribution/20260605_130414/quantization_error_distribution_metrics_20260605_130414.csv",
    output_tex_path="tables/quantization_error_distribution_table.tex"
)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import pandas as pd


DEFAULT_OUTPUT_TEX = Path("tables/quantization_error_distribution_table.tex")


def latex_sci(x, digits: int = 2) -> str:
    """
    Convert a number to compact LaTeX scientific notation.

    Example
    -------
    0.0002548581 -> $2.55\\times 10^{-4}$
    """
    if pd.isna(x):
        return r"--"

    x = float(x)

    if x == 0.0:
        return r"$0$"

    exponent = int(math.floor(math.log10(abs(x))))
    mantissa = x / (10.0 ** exponent)

    return rf"${mantissa:.{digits}f}\times 10^{{{exponent}}}$"


def latex_fixed(x, digits: int = 4) -> str:
    """
    Convert a number to fixed-point LaTeX notation.

    Example
    -------
    0.3332576 -> $0.3333$
    """
    if pd.isna(x):
        return r"--"

    return rf"${float(x):.{digits}f}$"


def latex_int(x) -> str:
    """
    Convert integer-like value to LaTeX math format.
    """
    if pd.isna(x):
        return r"--"

    return rf"${int(x)}$"


def safe_pipeline_label(pipeline: str) -> str:
    """
    Convert pipeline name into a LaTeX-safe texttt label.
    """
    return r"\texttt{" + str(pipeline).replace("_", r"\_") + r"}"


def make_quantization_error_distribution_table(
    csv_path: str | Path,
    output_tex_path: Optional[str | Path] = None,
    pipeline: Optional[str] = "random_signal",
    include_pipeline_column: bool = False,
) -> Path:
    """
    Generate the complete LaTeX table from a specified CSV.

    This function does not execute any Monte Carlo simulation and does not search
    for the most recent result folder. The CSV file must be explicitly provided.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Path to an existing metrics CSV.

    output_tex_path : str, pathlib.Path, or None
        Path where the LaTeX table will be written. If None, the table is saved
        to DEFAULT_OUTPUT_TEX.

    pipeline : str or None
        Pipeline to include in the table.

        Use:
            "random_signal"
        for the main SM table.

        Use:
            "random_coefficients"
        for the controlled coefficient-based pipeline.

        Use:
            None
        to include all pipelines present in the CSV.

    include_pipeline_column : bool
        If True, include a first column identifying the pipeline. This is useful
        when pipeline=None.

    Returns
    -------
    pathlib.Path
        Path to the generated LaTeX table file.
    """
    if csv_path is None:
        raise ValueError(
            "csv_path must be explicitly specified. "
            "This function does not search for the latest CSV."
        )

    csv_path = Path(csv_path)

    if output_tex_path is None:
        output_tex_path = DEFAULT_OUTPUT_TEX
    else:
        output_tex_path = Path(output_tex_path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = [
        "M_RbCP",
        "num_errors",
        "mean",
        "variance",
        "ks_distance",
        "wasserstein1_distance",
        "total_variation_hist",
        "kl_empirical_to_uniform_bits_hist",
        "jensen_shannon_bits_hist",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            "The CSV file does not contain the required columns: "
            + ", ".join(missing)
        )

    if pipeline is not None and "pipeline" in df.columns:
        df = df[df["pipeline"] == pipeline].copy()

        if df.empty:
            raise ValueError(
                f"No rows found for pipeline={pipeline!r} in CSV file: {csv_path}"
            )

    if "pipeline" in df.columns:
        sort_columns = ["pipeline", "M_RbCP"] if pipeline is None else ["M_RbCP"]
        df = df.sort_values(sort_columns)
    else:
        df = df.sort_values("M_RbCP")

    output_tex_path.parent.mkdir(parents=True, exist_ok=True)

    caption = (
        r"Distributional metrics for the normalized RbCP phase-shift "
        r"quantization error. For each value of $M_{\rm RbCP}$, the empirical "
        r"distribution of $E_{\rm norm}=(T-\widehat{T})/(\ell_n/2)$ is compared "
        r"with the reference uniform distribution $\mathcal{U}[-1,1]$. The "
        r"reported metrics include the sample mean, sample variance, "
        r"Kolmogorov--Smirnov distance, Wasserstein-1 distance, total variation "
        r"distance, Kullback--Leibler divergence, and Jensen--Shannon divergence."
    )

    label = r"tab:quantization_error_distribution"

    lines: list[str] = []

    lines.append(r"\begin{table*}[!t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")

    if include_pipeline_column:
        lines.append(r"\begin{tabular}{lccccccccc}")
    else:
        lines.append(r"\begin{tabular}{ccccccccc}")

    lines.append(r"\toprule")

    if include_pipeline_column:
        header = (
            r"Pipeline & "
            r"$M_{\rm RbCP}$ & "
            r"$N_{\rm err}$ & "
            r"$\widehat{\mu}_E$ & "
            r"$\widehat{\sigma}_E^2$ & "
            r"$D_{\rm KS}$ & "
            r"$W_1$ & "
            r"$D_{\rm TV}$ & "
            r"$D_{\rm KL}$ (bits) & "
            r"$D_{\rm JS}$ (bits) \\"
        )
    else:
        header = (
            r"$M_{\rm RbCP}$ & "
            r"$N_{\rm err}$ & "
            r"$\widehat{\mu}_E$ & "
            r"$\widehat{\sigma}_E^2$ & "
            r"$D_{\rm KS}$ & "
            r"$W_1$ & "
            r"$D_{\rm TV}$ & "
            r"$D_{\rm KL}$ (bits) & "
            r"$D_{\rm JS}$ (bits) \\"
        )

    lines.append(header)
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        row_items: list[str] = []

        if include_pipeline_column:
            if "pipeline" in row:
                row_items.append(safe_pipeline_label(row["pipeline"]))
            else:
                row_items.append(r"--")

        row_items.extend(
            [
                latex_int(row["M_RbCP"]),
                latex_int(row["num_errors"]),
                latex_sci(row["mean"], digits=2),
                latex_fixed(row["variance"], digits=4),
                latex_sci(row["ks_distance"], digits=2),
                latex_sci(row["wasserstein1_distance"], digits=2),
                latex_sci(row["total_variation_hist"], digits=2),
                latex_sci(row["kl_empirical_to_uniform_bits_hist"], digits=2),
                latex_sci(row["jensen_shannon_bits_hist"], digits=2),
            ]
        )

        lines.append(" & ".join(row_items) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    lines.append("")

    output_tex_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[INFO] CSV used: {csv_path}")
    print(f"[INFO] Pipeline filter: {pipeline}")
    print(f"[INFO] LaTeX table saved to: {output_tex_path}")

    return output_tex_path


if __name__ == "__main__":
    raise SystemExit(
        "This script requires an explicit csv_path. "
        "Use it from the Python console, e.g.:\n\n"
        "from scripts.make_quantization_error_distribution_table import "
        "make_quantization_error_distribution_table\n\n"
        "make_quantization_error_distribution_table("
        "csv_path='tests/results/quantization_error_distribution/20260605_130414/"
        "quantization_error_distribution_metrics_20260605_130414.csv')"
    )

# from scripts.make_quantization_error_distribution_table import make_quantization_error_distribution_table
#
# tex_path = make_quantization_error_distribution_table(
#     csv_path="tests/results/quantization_error_distribution/20260605_130414/quantization_error_distribution_metrics_20260605_130414.csv"
# )