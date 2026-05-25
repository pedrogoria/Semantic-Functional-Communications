"""
sfc/core/signal_generation.py

Signal generation and preprocessing.

Pure helpers for simulations.
"""

import numpy as np


def generate_random_signal(n_samples, n_periods, n_sensors, p2p):
    """
    EXACT original random generation:

    x = p2p * rand(...) - 0.5 * p2p
    """

    return p2p * np.random.rand(n_samples, n_periods, n_sensors) - 0.5 * p2p


def normalize_signal(x):
    """
    EXACT original normalization:

    x = 2*(x - mean)/max
    """

    for s in range(x.shape[2]):
        for p in range(x.shape[1]):
            x[:, p, s] = 2 * (x[:, p, s] - np.mean(x[:, p, s])) / np.max(x[:, p, s])

    return x