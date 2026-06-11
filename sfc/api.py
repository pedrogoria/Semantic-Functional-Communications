"""
sfc/api.py

Public API for the Semantic-Functional Communications (SFC) package.

This module provides a stable and user-facing import surface for the rest of
the repository. Its main purpose is to expose trusted core components through
a single, clean entry point.

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

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import numpy as np


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
from sfc.core.channel.SFCChannel import SFCChannel


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
from sfc.core.acquisition.nyquist import Nyquist
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
# Core modulation / MAC components
# ==========================================================================

from sfc.core.modulation.ppm import PPMCore
from sfc.core.mac.fdma import FDMACore


# ==========================================================================
# System-parameter builder and compatibility wrappers
# ==========================================================================

from sfc.core.system_parameters import (
    DerivedSystemParameters,
    build_derived_system_parameters,
    resolve_power_noise_model,
    compute_M_rbcp_per_sensor,
    compute_M_rbcp,
    compute_benchmark_M_per_sensor,
    compute_benchmark_M_single_sensor,
)


# ==========================================================================
# Core analytical / theoretical components
# ==========================================================================

from sfc.core.theory import (
    # Basic signal/channel theory
    compute_N,
    compute_snr_linear,
    compute_snr_db,
    compute_capacity,
    compute_sensor_snr,

    # RbCP theory
    compute_q,
    rbcp_mse_upper_bound,
    rbcp_mse_lower_bound,
    rbcp_mse_star,

    # M / M_RbCP relation
    compute_M_rbcp_from_M,
    compute_M_from_M_rbcp,

    # Bandwidth sharing
    compute_bandwidth_allocation,
    compute_sensor_bandwidths,

    # Feasible bins
    compute_M_rbcp_single_sensor,
    compute_benchmark_bits_per_sample_single_sensor,
    compute_benchmark_bits_per_sample_per_sensor,

    # SFC time / slot relations
    compute_M_time,
    compute_slot_duration,
    compute_slots_per_period,
    compute_event_slots_total,
    compute_rx_slots_total,

    # Duplicate probability
    epsilon_upper_bound,

    # Physical helpers
    compute_N0,
    compute_total_energy,
    compute_symbol_energy,
    compute_signal_level,
    compute_default_detection_threshold,
)


# ==========================================================================
# Semantic error detection
# ==========================================================================

from sfc.core.semantic_error_detection import detect_semantic_errors


# ==========================================================================
# Backward-compatible aliases
# ==========================================================================

# Older scripts may still use these names. Keep them as aliases, not new logic.
shannon_capacity = compute_capacity
sfc_duplicate_probability_upper_bound = epsilon_upper_bound
time_bins = compute_M_time


def time_quantization_interval(tau: float, B: float, R: int) -> float:
    """
    Backward-compatible alias for the SFC slot duration.

    This helper intentionally delegates to the trusted core function.
    """
    return compute_slot_duration(B=B, R=R)


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

    This helper exists only for convenience. It does not add scientific logic
    beyond object creation.
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


def create_phase_core(
    T=1,
    harmonics=3,
    n_sub_symbol=6,
    resource=7,
    sensor_nodes=5,
    bandwidth=100,
    periods=1,
    detect_errors=False,
    threshold_harmonics=0.001,
    **options
):
    """
    Create a PhaseCoefficientCore object using the public API.
    """

    return PhaseCoefficientCore(
        T=T,
        harmonics=harmonics,
        n_sub_symbol=n_sub_symbol,
        resource=resource,
        sensor_nodes=sensor_nodes,
        bandwidth=bandwidth,
        periods=periods,
        detect_errors=detect_errors,
        threshold_harmonics=threshold_harmonics,
        **options
    )


def create_fourier_core(
    T=1,
    harmonics=3,
    sensor_nodes=1,
    dft_signal_periods=1,
    **options
):
    """
    Create a FourierCoefficientCore object using the public API.
    """

    return FourierCoefficientCore(
        T=T,
        harmonics=harmonics,
        sensor_nodes=sensor_nodes,
        dft_signal_periods=dft_signal_periods,
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
    cfg: Optional[Dict[str, Any]] = None,
    *,
    S=1,
    P=1.0,
    N0=1e-3,
    B=100.0,
    R=7,
    L=6,
    tau=1.0,
    W=1.0,
    sensor_x_event=None,
    collision_mode="sum",
    channel_type="awgn",
    detection_mode="threshold",
    threshold=None,
    threshold_factor=0.5,
    score_threshold=None,
    seed=12345,
    **options
):
    """
    Create an SFCChannel using the current cfg-based channel interface.

    Preferred usage
    ---------------
    Pass a complete configuration dictionary:

        create_sfc_channel(cfg)

    Convenience usage
    -----------------
    If cfg is not provided, this helper builds a minimal configuration from
    physical/system parameters.

    Notes
    -----
    This helper does not implement any channel logic. It only builds or forwards
    a configuration to SFCChannel.
    """

    if cfg is not None:
        cfg_sfc = copy.deepcopy(cfg)
        return SFCChannel(cfg_sfc)

    if sensor_x_event is None:
        sensor_x_event = np.ones((int(S), 1), dtype=float)

    channel_cfg = {
        "sensor_x_event": sensor_x_event,
        "collision_mode": collision_mode,
        "type": channel_type,
        "detection_mode": detection_mode,
        "score_threshold": L if score_threshold is None else score_threshold,
    }

    if threshold is not None:
        channel_cfg["threshold"] = threshold
    else:
        channel_cfg["threshold_factor"] = threshold_factor

    channel_cfg.update(options)

    cfg_sfc = {
        "system": {
            "S": S,
            "P": P,
            "N0": N0,
            "B": B,
            "R": R,
            "L": L,
        },
        "signal": {
            "tau": tau,
            "W": W,
        },
        "channel": channel_cfg,
        "reproducibility": {
            "seed": seed,
        },
    }

    return SFCChannel(cfg_sfc)


def create_fading_generator(seed=None, rng=None, use_legacy_random_state=False):
    """
    Create a fading generator using the public API.
    """

    return FadingGenerator(
        seed=seed,
        rng=rng,
        use_legacy_random_state=use_legacy_random_state
    )


def create_ppm_core(
    fc,
    pulse_width,
    rec_pulse=0.0,
    pulse_type="raised_cosine",
    rolloff=0.99,
    span=12,
    eps_margin=1e-3,
    interp_mode="sinc",
    periodic_replicas=10,
    clip_recovered_to_unit_interval=True,
    **options
):
    """
    Create a PPMCore object using the public API.
    """

    return PPMCore(
        fc=fc,
        pulse_width=pulse_width,
        rec_pulse=rec_pulse,
        pulse_type=pulse_type,
        rolloff=rolloff,
        span=span,
        eps_margin=eps_margin,
        interp_mode=interp_mode,
        periodic_replicas=periodic_replicas,
        clip_recovered_to_unit_interval=clip_recovered_to_unit_interval,
        **options
    )


def create_fdma_core(
    S,
    B_total,
    P_per_sensor,
    tau,
    bandwidth_allocation=None,
    normalize_sensor_power=True,
    return_nonorthogonal_sum_preview=False,
    frequency_axis_centered_at_zero=True,
    **options
):
    """
    Create an FDMACore object using the public API.
    """

    return FDMACore(
        S=S,
        B_total=B_total,
        P_per_sensor=P_per_sensor,
        tau=tau,
        bandwidth_allocation=bandwidth_allocation,
        normalize_sensor_power=normalize_sensor_power,
        return_nonorthogonal_sum_preview=return_nonorthogonal_sum_preview,
        frequency_axis_centered_at_zero=frequency_axis_centered_at_zero,
        **options
    )


def available_public_components():
    """
    Return a summary of the public API surface.

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
        "modulation_mac": [
            "PPMCore",
            "FDMACore",
        ],
        "system_parameters": [
            "DerivedSystemParameters",
            "build_derived_system_parameters",
            "resolve_power_noise_model",
            "compute_M_rbcp_per_sensor",
            "compute_M_rbcp",
            "compute_benchmark_M_per_sensor",
            "compute_benchmark_M_single_sensor",
        ],
        "theory": [
            "compute_N",
            "compute_snr_linear",
            "compute_snr_db",
            "compute_capacity",
            "compute_sensor_snr",
            "compute_q",
            "rbcp_mse_upper_bound",
            "rbcp_mse_lower_bound",
            "rbcp_mse_star",
            "compute_M_rbcp_from_M",
            "compute_M_from_M_rbcp",
            "compute_bandwidth_allocation",
            "compute_sensor_bandwidths",
            "compute_M_rbcp_single_sensor",
            "compute_benchmark_bits_per_sample_single_sensor",
            "compute_benchmark_bits_per_sample_per_sensor",
            "compute_M_time",
            "compute_slot_duration",
            "compute_slots_per_period",
            "compute_event_slots_total",
            "compute_rx_slots_total",
            "epsilon_upper_bound",
            "compute_N0",
            "compute_total_energy",
            "compute_symbol_energy",
            "compute_signal_level",
            "compute_default_detection_threshold",
        ],
        "backward_compatible_aliases": [
            "shannon_capacity",
            "sfc_duplicate_probability_upper_bound",
            "time_bins",
            "time_quantization_interval",
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
            "detect_semantic_errors",
        ],
        "factories": [
            "create_cp_sampler",
            "create_phase_core",
            "create_fourier_core",
            "create_nyquist_sampler",
            "create_sfc_channel",
            "create_fading_generator",
            "create_ppm_core",
            "create_fdma_core",
            "available_public_components",
        ],
    }


# ==========================================================================
# Public export list
# ==========================================================================

__all__ = [
    # Channel / physical-layer classes
    "PositionSensorNodes",
    "FadingGenerator",
    "PropagationModel",
    "SFCMapping",
    "SFCChannel",

    # Fading helpers
    "ricean_fading",
    "rician_fading",
    "rayleigh_fading",
    "nakagami_fading",
    "weibull_fading",

    # Sampling / representation classes
    "CosinePhaseCore",
    "CPSample",
    "SamplingCore",
    "PhaseCoefficientCore",
    "FourierCoefficientCore",
    "ReconstructionCore",
    "QuantizationCore",
    "Nyquist",

    # Modulation / MAC classes
    "PPMCore",
    "FDMACore",

    # System parameters
    "DerivedSystemParameters",
    "build_derived_system_parameters",
    "resolve_power_noise_model",
    "compute_M_rbcp_per_sensor",
    "compute_M_rbcp",
    "compute_benchmark_M_per_sensor",
    "compute_benchmark_M_single_sensor",

    # Basic theory
    "compute_N",
    "compute_snr_linear",
    "compute_snr_db",
    "compute_capacity",
    "compute_sensor_snr",

    # RbCP theory
    "compute_q",
    "rbcp_mse_upper_bound",
    "rbcp_mse_lower_bound",
    "rbcp_mse_star",

    # M / M_RbCP relation
    "compute_M_rbcp_from_M",
    "compute_M_from_M_rbcp",

    # Bandwidth sharing
    "compute_bandwidth_allocation",
    "compute_sensor_bandwidths",

    # Feasible bins
    "compute_M_rbcp_single_sensor",
    "compute_benchmark_bits_per_sample_single_sensor",
    "compute_benchmark_bits_per_sample_per_sensor",

    # SFC time / slot relations
    "compute_M_time",
    "compute_slot_duration",
    "compute_slots_per_period",
    "compute_event_slots_total",
    "compute_rx_slots_total",

    # Duplicate probability
    "epsilon_upper_bound",

    # Physical helpers
    "compute_N0",
    "compute_total_energy",
    "compute_symbol_energy",
    "compute_signal_level",
    "compute_default_detection_threshold",

    # Backward-compatible aliases
    "shannon_capacity",
    "sfc_duplicate_probability_upper_bound",
    "time_bins",
    "time_quantization_interval",

    # Scientific helpers
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
    "detect_semantic_errors",

    # Factory helpers
    "create_cp_sampler",
    "create_phase_core",
    "create_fourier_core",
    "create_nyquist_sampler",
    "create_sfc_channel",
    "create_fading_generator",
    "create_ppm_core",
    "create_fdma_core",
    "available_public_components",
]
