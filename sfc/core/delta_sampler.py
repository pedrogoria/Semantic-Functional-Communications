"""
sfc/core/delta_sampler.py

Trusted core module for threshold-based and delta-based sampling utilities.

This file isolates the following legacy/trusted functions extracted from
`sfc/sampling.py`:

- print_new_samples(...)
- plot_threshold(...)
- plot_delta(...)
- delta(...)
- print_new_samples_delta(...)

IMPORTANT
---------
This is a structural extraction of trusted code.

Allowed changes:
- file split
- comments and documentation
- clearer organization

Forbidden changes:
- changing formulas
- changing interpolation logic
- changing threshold logic
- changing delta-computation logic
- changing plotting behavior
- changing file-loading patterns
- changing iteration ranges
- changing output structure

The goal is to preserve the original numerical and logical behavior exactly.
"""

import glob as glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import style
from pandas.plotting import register_matplotlib_converters


# --------------------------------------------------------------------------
# Preserve the original converter registration exactly as in the trusted file.
# --------------------------------------------------------------------------
register_matplotlib_converters()


def print_new_samples(dataset, percentage, ts):
    """
    Generate an interpolated dataset and compute a thresholded version of it.

    Parameters
    ----------
    dataset : pandas.DataFrame
        Input dataset. The trusted code expects a column named
        'Air temperature (degC)'.

    percentage : float
        Percentage used to create upper and lower threshold bands around the mean.

    ts : float
        Sampling-time-related parameter used in the line-insertion rule:
            n_lines = round(600 / ts)

    Returns
    -------
    tuple
        (mimi, mama, momo, dfs)

        mimi : pandas.Series or scalar-like
            Upper threshold line.

        mama : pandas.Series or scalar-like
            Mean value.

        momo : pandas.Series or scalar-like
            Lower threshold line.

        dfs : pandas.DataFrame
            Interpolated dataset with an added 'Threshold' column.

    Notes
    -----
    This function preserves exactly the trusted logic:
    - line insertion using NaN rows;
    - interpolation only on 'Air temperature (degC)';
    - threshold interval built from the mean and extrema;
    - replacement of in-band values in the 'Threshold' column by the mean.
    """

    df = dataset

    n_lines = round(600 / ts)

    # ------------------------------------------------------------------
    # Insert extra rows between existing values.
    # This preserves the trusted list-comprehension-based expansion logic.
    # ------------------------------------------------------------------
    line_ins = n_lines
    res_dict = {
        col: [y for val in df[col] for y in [val] + [np.nan] * line_ins][:-line_ins]
        for col in df.columns
    }
    df_new = pd.DataFrame(res_dict)

    # ------------------------------------------------------------------
    # Interpolate only the temperature column, exactly as in the trusted code.
    # ------------------------------------------------------------------
    df_new['Air temperature (degC)'] = df_new['Air temperature (degC)'].interpolate()

    mimi, mama, momo = 0, 0, 0

    dfs = df_new

    max_v1 = dfs.iloc[:].max()
    mama = dfs.iloc[:].mean()
    min_v1 = dfs.iloc[:].min()

    # ------------------------------------------------------------------
    # Percentage UP and DOWN the mean, preserving the original formulas.
    # ------------------------------------------------------------------
    mimi = mama + ((percentage / 100) * abs(max_v1 - mama))
    momo = mama - ((percentage / 100) * abs(min_v1 - mama))

    # ------------------------------------------------------------------
    # Retrieve indices of values inside the threshold interval.
    # The trusted code expects the specific column name below.
    # ------------------------------------------------------------------
    indices_within_interval = dfs[
        (dfs['Air temperature (degC)'] >= float(momo))
        & (dfs['Air temperature (degC)'] <= float(mimi))
    ].index.tolist()

    # ------------------------------------------------------------------
    # Create a new threshold column and replace in-band values by the mean.
    # ------------------------------------------------------------------
    dfs['Threshold'] = dfs['Air temperature (degC)']
    dfs.loc[indices_within_interval, 'Threshold'] = float(mama)

    return mimi, mama, momo, dfs


def plot_threshold(Temperature_threshold, plot_v, mama, mimi, momo):
    """
    Plot the temperature signal or the thresholded signal exactly as in the
    trusted implementation.

    Parameters
    ----------
    Temperature_threshold : pandas.DataFrame
        DataFrame containing at least:
        - 'Air temperature (degC)'
        - 'Threshold'

    plot_v : int
        If 1, plot the original temperature signal.
        Otherwise, plot the thresholded signal.

    mama : pandas.Series or scalar-like
        Mean level.

    mimi : pandas.Series or scalar-like
        Upper threshold level.

    momo : pandas.Series or scalar-like
        Lower threshold level.

    Returns
    -------
    None

    Notes
    -----
    This plotting function preserves exactly:
    - the selected style ('bmh')
    - figure size
    - font size
    - labels
    - horizontal threshold lines
    - hidden x ticks
    """

    t1 = mama
    t2 = mimi
    t3 = momo

    style.use('bmh')
    plt.figure(figsize=(16, 6))
    plt.rcParams.update({'font.size': 18})
    plt.xlabel('Lappeenranta October 2022')
    plt.ylabel('Temperature signal')

    if plot_v == 1:
        plt.plot(Temperature_threshold['Air temperature (degC)'], 'k')
    else:
        plt.plot(Temperature_threshold['Threshold'], 'k')

    plt.xticks([])

    plt.axhline(t2[0], linestyle='--', color='r')
    plt.axhline(t1[0], linestyle='--', color='g')
    plt.axhline(t3[0], linestyle='--', color='r')

    plt.autoscale()
    plt.show()


def plot_delta(Temperature_threshold, plot_v=1):
    """
    Plot the original temperature signal or the delta-based representation.

    Parameters
    ----------
    Temperature_threshold : pandas.DataFrame
        DataFrame containing at least either:
        - 'Air temperature (degC)'
        or
        - 'Delta'

    plot_v : int, optional
        If 1, plot the original temperature signal.
        Otherwise, plot the 'Delta' column.

    Returns
    -------
    None

    Notes
    -----
    This plotting function preserves exactly:
    - the selected style ('bmh')
    - figure size
    - font size
    - labels
    - hidden x ticks
    """

    style.use('bmh')
    plt.figure(figsize=(16, 6))
    plt.rcParams.update({'font.size': 18})
    plt.xlabel('Lappeenranta October 2022')
    plt.ylabel('Temperature signal')

    if plot_v == 1:
        plt.plot(Temperature_threshold['Air temperature (degC)'], 'k')
    else:
        plt.plot(Temperature_threshold['Delta'], 'k')

    plt.xticks([])

    plt.autoscale()
    plt.show()


def delta(dfs, p):
    """
    Apply the trusted delta-based sampling/compression rule to a single-column
    dataset representation.

    Parameters
    ----------
    dfs : pandas.DataFrame
        Input DataFrame. The trusted implementation assumes:
        - the original signal is in column index 0;
        - a 'Delta' column is created and stored at column index 2.

    p : float
        Percentage threshold relative to the maximum observed delta.

    Returns
    -------
    tuple
        (dfs, m_delta, val_delta)

        dfs : pandas.DataFrame
            Updated DataFrame with the 'Delta' column filled according to the
            trusted delta rule.

        m_delta : np.ndarray or scalar-like
            Maximum delta observed in the input signal.

        val_delta : np.ndarray
            Per-sample absolute delta values.

    Notes
    -----
    This function preserves exactly:
    - creation of the 'Delta' column;
    - initialization of the first delta sample;
    - full-delta scan;
    - threshold comparison;
    - copy-forward rule when the delta is below threshold.
    """

    m = len(dfs)
    val_delta = np.zeros((m, 1))
    m_delta = 0

    # ------------------------------------------------------------------
    # Preserve the original DataFrame mutation logic exactly.
    # ------------------------------------------------------------------
    dfs['Delta'] = [0] * len(dfs)
    dfs.iloc[0, 2] = dfs.iloc[0, 0]

    compress = 0

    for d in range(0, m - 1):
        val_delta[d] = abs(dfs.iloc[d, 0] - dfs.iloc[d + 1, 0])

    m_delta = max(val_delta[:])

    for d in range(0, m - 1):
        if abs(dfs.iloc[d, 0] - dfs.iloc[d + 1, 0]) < ((p / 100) * m_delta[0]):
            dfs.iloc[d + 1, 2] = dfs.iloc[d, 2]
        else:
            dfs.iloc[d + 1, 2] = dfs.iloc[d + 1, 0]

    return dfs, m_delta, val_delta


def print_new_samples_delta(fault):
    """
    Apply the trusted delta-sampling logic to a collection of files matching
    the pattern 'd*_te.dat'.

    Parameters
    ----------
    fault : int
        Index of the fault dataset to be processed from the loaded file set.

    Returns
    -------
    None

    Notes
    -----
    This function preserves exactly the trusted logic:
    - percentages = [50]
    - file pattern 'd*_te.dat'
    - fixed 52-column naming scheme
    - fixed matrix sizes for delta, m_delta, and compress
    - same thresholding and compression-rate computation

    The original trusted implementation does not return the computed compression
    metrics because the return line is commented out. That behavior is preserved.
    """

    # ------------------------------------------------------------------
    # Preserve the trusted hard-coded percentage list exactly.
    # ------------------------------------------------------------------
    percentages = [50]

    paths = glob.glob('d*_te.dat')

    dfs = {}
    columns_name = ['Variable_' + str(x) for x in range(1, 53)]

    for mov in range(0, len(paths)):
        data = np.genfromtxt(paths[mov])
        dfs[mov] = pd.DataFrame(data=data, columns=columns_name)

    for p in percentages:

        delta = np.zeros((960, 52))
        m_delta = np.zeros((1, 52))
        compress = np.zeros((1, 52))

        # ------------------------------------------------------------------
        # Compute reference deltas from the normal-operation dataset dfs[0].
        # ------------------------------------------------------------------
        for v in range(0, 52):
            for d in range(0, dfs[0].shape[0] - 1):
                delta[d, v] = abs(dfs[0].iloc[d, v] - dfs[0].iloc[d + 1, v])

            m_delta[0, v] = max(delta[:, v])

        # ------------------------------------------------------------------
        # Apply the delta-based rule to the selected fault dataset.
        # ------------------------------------------------------------------
        for v in range(0, 52):
            count = 0

            for d in range(0, dfs[0].shape[0] - 1):
                if abs(dfs[fault].iloc[d, v] - dfs[fault].iloc[d + 1, v]) < ((p / 100) * m_delta[0, v]):
                    dfs[fault].iloc[d + 1, v] = dfs[fault].iloc[d, v]
                    count += 1

            compress[0, v] = count / 960

    # ------------------------------------------------------------------
    # Preserve the original behavior: no explicit return value.
    # The trusted source contains only a commented-out return line.
    # ------------------------------------------------------------------
    return None


__all__ = [
    "print_new_samples",
    "plot_threshold",
    "plot_delta",
    "delta",
    "print_new_samples_delta",
]
