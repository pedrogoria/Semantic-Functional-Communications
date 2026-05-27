"""
sfc/api.py

Public API for the Semantic-Functional Communications (SFC) package.

This module provides a stable and user-facing import surface for the rest of
the repository. Its main purpose is to expose the trusted core components
through a single, clean entry point.

Design goals
------------
- Keep the public API stable even if the internal file organization evolves.
- Avoid forcing experiment scripts to import directly from many core modules.
- Provide thin factory helpers for the most common objects.
- Re-export trusted scientific primitives without duplicating any logic.

IMPORTANT
---------
This file must remain a thin API boundary.

Allowed responsibilities:
- re-export public classes and functions from the core package;
- provide lightweight factory helpers;
- group imports in one place for user-facing convenience;
- expose analytical/theoretical functions used by the manuscript.

Forbidden responsibilities:
- implementing scientific equations;
- duplicating any core mathematical logic;
- embedding experiment-specific behavior;
- parsing configs;
- running pipelines.

In short:
    core modules compute;
    api.py exposes;
    experiments orchestrate.
"""

# ==========================================================================
# Core channel / physical-layer components
# ==========================================================================

from sfc.core.positioning import PositionSensorNodes
from sfc.core.fading import (
    FadingGenerator,
    ricean_fading,
    rician_fading,
    rayleigh_fading,
    nakagami_fading,
    weibull_fading,
)
from sfc.core.propagation import PropagationModel
from sfc.core.mapping import SFCMapping
from sfc.core.channel_legacy import SFCChannel

# ==========================================================================
# Core sampling / representation components
# ==========================================================================

from sfc.core.phase_cof import (
    PhaseCoefficientCore,
    calc_ta_tb,
)
from sfc.core.fourier import (
    FourierCoefficientCore,
    calc_an_bn_dft,
    cpm_sample,
)
from sfc.core.reconstruction import (
    ReconstructionCore,
    recover_signal,
)
from sfc.core.quantization import (
    QuantizationCore,
    quantize,
    quantize_ta_tb,
)
from sfc.core.filters import (
    sinc_filter,
    plot_fft,
    filter_periodic,
)
from sfc.core.nyquist import Nyquist
from sfc.core.delta_sampler import (
    print_new_samples,
    plot_threshold,
    plot_delta,
    delta,
    print_new_samples_delta,
)
from sfc.core.sampling import (
    CosinePhaseCore,
    CPSample,
    SamplingCore,
)

# ==========================================================================
# Core analytical / theoretical components
# ==========================================================================

from sfc.core.theory import (
    shannon_capacity,
    benchmark_rate,
    rbcp_rate,
    benchmark_max_bins,
    rbcp_max_bins,
    compute_q,
    benchmark_mse,
    rbcp_mse_general,
    rbcp_mse_lower_bound,
    rbcp_mse_upper_bound,
    rbcp_mse_star,
    xi_from_bandwidth,
    rbcp_bins_from_benchmark,
    time_bins,
    time_quantization_interval,
    sfc_duplicate_probability_upper_bound,
    rbcp_total_latency,
    benchmark_total_latency,
    rbcp_has_lower_latency_than_benchmark,
)

# ==========================================================================
# Public factory helpers
# ==========================================================================


def create_cp_sampler(
    T=1,
    harmonics=3,
    n_sub_symbol=6,
    resource=7,
    sensor_nodes=5,
    bandwidth=100,
    **options
):
    """
    Create a cosine-phase sampler using the public API.

    Parameters
    ----------
    T : float, optional
        Signal period / observation window.

    harmonics : int, optional
        Number of harmonics.

    n_sub_symbol : int, optional
        Number of SFC sub-symbol rows.

    resource : int, optional
        Number of SFC resources.

    sensor_nodes : int, optional
        Number of sensors.

    bandwidth : float, optional
        Communication bandwidth.

    **options : dict
        Extra options forwarded directly to `CPSample`.

    Returns
    -------
    CPSample
        Configured cosine-phase sampler.

    Notes
    -----
    This helper exists only for convenience. It does not add any logic beyond
    object creation.
    """

    return CPSample(
        T=T,
        harmonics=harmonics,
        n_sub_symbol=n_sub_symbol,
        resource=resource,
        sensor_nodes=sensor_nodes,
        bandwidth=bandwidth,
        **options
    )


def create_nyquist_sampler(
    T=1,
    Tt=0.001,
    sampling_rate=10,
    sensor_nodes=5,
    bandwidth=100,
    **options
):
    """
    Create a Nyquist sampler using the public API.

    Parameters
    ----------
    T : float, optional
        Signal period / observation window.

    Tt : float, optional
        Time step.

    sampling_rate : float, optional
        Sampling rate.

    sensor_nodes : int, optional
        Number of sensors.

    bandwidth : float, optional
        Communication bandwidth.

    **options : dict
        Extra options forwarded directly to `Nyquist`.

    Returns
    -------
    Nyquist
        Configured Nyquist sampling object.
    """

    return Nyquist(
        T=T,
        Tt=Tt,
        sampling_rate=sampling_rate,
        sensor_nodes=sensor_nodes,
        bandwidth=bandwidth,
        **options
    )


def create_sfc_channel(
    base_station=None,
    sensor_nodes=64,
    carrier_frequency=2.4e9,
    n_sub_symbol=6,
    resource=7,
    **options
):
    """
    Create an SFC channel using the public API.

    Parameters
    ----------
    base_station : np.ndarray or None, optional
        Base-station position. If None, the default trusted value [[0, 0]]
        is used by the underlying class.

    sensor_nodes : int, optional
        Number of sensors.

    carrier_frequency : float, optional
        Carrier frequency in Hz.

    n_sub_symbol : int, optional
        Number of sub-symbol rows in the SFC map.

    resource : int, optional
        Number of resource columns in the SFC map.

    **options : dict
        Extra options forwarded directly to `SFCChannel`.

    Returns
    -------
    SFCChannel
        Configured SFC channel object.
    """

    if base_station is None:
        return SFCChannel(
            sensor_nodes=sensor_nodes,
            carrier_frequency=carrier_frequency,
            n_sub_symbol=n_sub_symbol,
            resource=resource,
            **options
        )

    return SFCChannel(
        base_station=base_station,
        sensor_nodes=sensor_nodes,
        carrier_frequency=carrier_frequency,
        n_sub_symbol=n_sub_symbol,
        resource=resource,
        **options
    )


def create_fading_generator(seed=None, rng=None, use_legacy_random_state=False):
    """
    Create a fading generator using the public API.

    Parameters
    ----------
    seed : int or None, optional
        Seed used by the internal random generator.

    rng : numpy.random.Generator or numpy.random.RandomState or None, optional
        External random generator.

    use_legacy_random_state : bool, optional
        If True, use NumPy RandomState instead of the modern default_rng.

    Returns
    -------
    FadingGenerator
        Configured fading generator.
    """

    return FadingGenerator(
        seed=seed,
        rng=rng,
        use_legacy_random_state=use_legacy_random_state
    )


def available_public_components():
    """
    Return a summary of the public API surface.

    Returns
    -------
    dict
        Dictionary grouping the exported public components by subsystem.

    Notes
    -----
    This helper is meant only for introspection/documentation convenience.
    """

    return {
        "channel": [
            "PositionSensorNodes",
            "FadingGenerator",
            "PropagationModel",
            "SFCMapping",
            "SFCChannel",
        ],
        "sampling": [
            "CosinePhaseCore",
            "CPSample",
            "SamplingCore",
            "PhaseCoefficientCore",
            "FourierCoefficientCore",
            "ReconstructionCore",
            "QuantizationCore",
            "Nyquist",
        ],
        "theory": [
            "shannon_capacity",
            "benchmark_rate",
            "rbcp_rate",
            "benchmark_max_bins",
            "rbcp_max_bins",
            "compute_q",
            "benchmark_mse",
            "rbcp_mse_general",
            "rbcp_mse_lower_bound",
            "rbcp_mse_upper_bound",
            "rbcp_mse_star",
            "xi_from_bandwidth",
            "rbcp_bins_from_benchmark",
            "time_bins",
            "time_quantization_interval",
            "sfc_duplicate_probability_upper_bound",
            "rbcp_total_latency",
            "benchmark_total_latency",
            "rbcp_has_lower_latency_than_benchmark",
        ],
        "helpers": [
            "calc_ta_tb",
            "calc_an_bn_dft",
            "cpm_sample",
            "recover_signal",
            "quantize",
            "quantize_ta_tb",
            "sinc_filter",
            "plot_fft",
            "filter_periodic",
            "print_new_samples",
            "plot_threshold",
            "plot_delta",
            "delta",
            "print_new_samples_delta",
        ],
        "factories": [
            "create_cp_sampler",
            "create_nyquist_sampler",
            "create_sfc_channel",
            "create_fading_generator",
        ],
    }


# ==========================================================================
# Public export list
# ==========================================================================

__all__ = [
    # Public channel / physical-layer classes
    "PositionSensorNodes",
    "FadingGenerator",
    "PropagationModel",
    "SFCMapping",
    "SFCChannel",

    # Public fading helpers
    "ricean_fading",
    "rician_fading",
    "rayleigh_fading",
    "nakagami_fading",
    "weibull_fading",

    # Public sampling / representation classes
    "CosinePhaseCore",
    "CPSample",
    "SamplingCore",
    "PhaseCoefficientCore",
    "FourierCoefficientCore",
    "ReconstructionCore",
    "QuantizationCore",
    "Nyquist",

    # Public analytical / theoretical helpers
    "shannon_capacity",
    "benchmark_rate",
    "rbcp_rate",
    "benchmark_max_bins",
    "rbcp_max_bins",
    "compute_q",
    "benchmark_mse",
    "rbcp_mse_general",
    "rbcp_mse_lower_bound",
    "rbcp_mse_upper_bound",
    "rbcp_mse_star",
    "xi_from_bandwidth",
    "rbcp_bins_from_benchmark",
    "time_bins",
    "time_quantization_interval",
    "sfc_duplicate_probability_upper_bound",
    "rbcp_total_latency",
    "benchmark_total_latency",
    "rbcp_has_lower_latency_than_benchmark",

    # Public scientific helpers
    "calc_ta_tb",
    "calc_an_bn_dft",
    "cpm_sample",
    "recover_signal",
    "quantize",
    "quantize_ta_tb",
    "sinc_filter",
    "plot_fft",
    "filter_periodic",
    "print_new_samples",
    "plot_threshold",
    "plot_delta",
    "delta",
    "print_new_samples_delta",

    # Public factory helpers
    "create_cp_sampler",
    "create_nyquist_sampler",
    "create_sfc_channel",
    "create_fading_generator",
    "available_public_components",
]
