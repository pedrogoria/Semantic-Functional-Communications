"""
sfc/core/modulation/pulse_shaping.py

Generic pulse-shaping utilities for digital modulation experiments.

This module provides:
- rectangular pulse
- raised cosine (RC) pulse
- root raised cosine (RRC) pulse
- symbol upsampling
- pulse shaping by convolution
- matched filtering
- symbol-rate sampling after matched filter

Compatibility note
------------------
This module also provides the wrapper functions expected by PPMCore:

- build_tx_pulse(...)
- build_matched_filter(...)

and accepts pulse-name aliases such as:
- "rectangular"      -> "rect"
- "raised_cosine"    -> "rc"
- "root_raised_cosine" -> "rrc"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np


# =============================================================================
# TYPES / CONFIG
# =============================================================================

PulseKind = Literal["rect", "rc", "rrc"]


@dataclass(frozen=True)
class PulseShapingConfig:
    pulse: PulseKind = "rrc"
    samples_per_symbol: int = 8
    span_symbols: int = 8
    rolloff: float = 0.25
    gain: float = 1.0
    normalize_energy: bool = True


# =============================================================================
# BASIC HELPERS
# =============================================================================

def _validate_sps(samples_per_symbol: int) -> int:
    if not isinstance(samples_per_symbol, int) or samples_per_symbol <= 0:
        raise ValueError("samples_per_symbol must be a positive integer")
    return samples_per_symbol


def _validate_span(span_symbols: int) -> int:
    if not isinstance(span_symbols, int) or span_symbols <= 0:
        raise ValueError("span_symbols must be a positive integer")
    return span_symbols


def _validate_rolloff(rolloff: float) -> float:
    rolloff = float(rolloff)
    if rolloff < 0.0 or rolloff > 1.0:
        raise ValueError("rolloff must satisfy 0 <= rolloff <= 1")
    return rolloff


def _normalize_energy(pulse: np.ndarray) -> np.ndarray:
    energy = np.sum(np.abs(pulse) ** 2)
    if energy <= 0:
        raise ValueError("Pulse energy must be positive")
    return pulse / np.sqrt(energy)


def pulse_energy(pulse: np.ndarray) -> float:
    pulse = np.asarray(pulse, dtype=float)
    return float(np.sum(np.abs(pulse) ** 2))


def pulse_peak(pulse: np.ndarray) -> float:
    pulse = np.asarray(pulse, dtype=float)
    return float(np.max(np.abs(pulse)))


def _canonical_pulse_name(pulse: str) -> str:
    """
    Map aliases to the internal canonical pulse names.
    """
    pulse = str(pulse).strip().lower()

    aliases = {
        "rect": "rect",
        "rectangular": "rect",
        "rc": "rc",
        "raised_cosine": "rc",
        "raised-cosine": "rc",
        "rrc": "rrc",
        "root_raised_cosine": "rrc",
        "root-raised-cosine": "rrc",
    }

    if pulse not in aliases:
        raise ValueError(f"Unsupported pulse type: {pulse}")

    return aliases[pulse]


# =============================================================================
# TIME GRID
# =============================================================================

def pulse_time_vector(span_symbols: int, samples_per_symbol: int) -> np.ndarray:
    sps = _validate_sps(samples_per_symbol)
    span = _validate_span(span_symbols)

    n = np.arange(-span * sps / 2, span * sps / 2 + 1)
    t = n / sps

    return t.astype(float)


# =============================================================================
# PULSE GENERATORS
# =============================================================================

def rectangular_pulse(
    samples_per_symbol: int,
    gain: float = 1.0,
    normalize_energy: bool = True
) -> np.ndarray:
    sps = _validate_sps(samples_per_symbol)

    pulse = np.ones(sps, dtype=float)

    if normalize_energy:
        pulse = _normalize_energy(pulse)

    pulse = gain * pulse
    return pulse


def raised_cosine_pulse(
    samples_per_symbol: int,
    span_symbols: int,
    rolloff: float,
    gain: float = 1.0,
    normalize_energy: bool = True
) -> np.ndarray:
    sps = _validate_sps(samples_per_symbol)
    span = _validate_span(span_symbols)
    beta = _validate_rolloff(rolloff)

    t = pulse_time_vector(span, sps)
    h = np.zeros_like(t, dtype=float)

    if beta == 0.0:
        h = np.sinc(t)
    else:
        for i, ti in enumerate(t):
            denom = 1.0 - (2.0 * beta * ti) ** 2

            if np.isclose(ti, 0.0):
                h[i] = 1.0

            elif np.isclose(np.abs(ti), 1.0 / (2.0 * beta)):
                h[i] = (np.pi / 4.0) * np.sinc(1.0 / (2.0 * beta))

            else:
                h[i] = (
                    np.sinc(ti)
                    * np.cos(np.pi * beta * ti)
                    / denom
                )

    if normalize_energy:
        h = _normalize_energy(h)

    h = gain * h
    return h


def root_raised_cosine_pulse(
    samples_per_symbol: int,
    span_symbols: int,
    rolloff: float,
    gain: float = 1.0,
    normalize_energy: bool = True
) -> np.ndarray:
    sps = _validate_sps(samples_per_symbol)
    span = _validate_span(span_symbols)
    beta = _validate_rolloff(rolloff)

    t = pulse_time_vector(span, sps)
    h = np.zeros_like(t, dtype=float)

    if beta == 0.0:
        h = np.sinc(t)
    else:
        for i, ti in enumerate(t):
            if np.isclose(ti, 0.0):
                h[i] = 1.0 - beta + (4.0 * beta / np.pi)

            elif np.isclose(np.abs(ti), 1.0 / (4.0 * beta)):
                term1 = (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                term2 = (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
                h[i] = (beta / np.sqrt(2.0)) * (term1 + term2)

            else:
                num = (
                    np.sin(np.pi * ti * (1.0 - beta))
                    + 4.0 * beta * ti * np.cos(np.pi * ti * (1.0 + beta))
                )
                den = np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
                h[i] = num / den

    if normalize_energy:
        h = _normalize_energy(h)

    h = gain * h
    return h


def generate_pulse(
    pulse: str,
    samples_per_symbol: int,
    span_symbols: int = 8,
    rolloff: float = 0.25,
    gain: float = 1.0,
    normalize_energy: bool = True
) -> np.ndarray:
    pulse = _canonical_pulse_name(pulse)

    if pulse == "rect":
        return rectangular_pulse(
            samples_per_symbol=samples_per_symbol,
            gain=gain,
            normalize_energy=normalize_energy
        )

    if pulse == "rc":
        return raised_cosine_pulse(
            samples_per_symbol=samples_per_symbol,
            span_symbols=span_symbols,
            rolloff=rolloff,
            gain=gain,
            normalize_energy=normalize_energy
        )

    if pulse == "rrc":
        return root_raised_cosine_pulse(
            samples_per_symbol=samples_per_symbol,
            span_symbols=span_symbols,
            rolloff=rolloff,
            gain=gain,
            normalize_energy=normalize_energy
        )

    raise ValueError(f"Unsupported pulse type: {pulse}")


# =============================================================================
# UPSAMPLING / SHAPING
# =============================================================================

def upsample_symbols(symbols: np.ndarray, samples_per_symbol: int) -> np.ndarray:
    sps = _validate_sps(samples_per_symbol)

    symbols = np.asarray(symbols)
    if symbols.ndim != 1:
        raise ValueError("symbols must be a 1D array")

    up = np.zeros(len(symbols) * sps, dtype=symbols.dtype)
    up[::sps] = symbols

    return up


def pulse_shape(
    symbols: np.ndarray,
    pulse: Optional[np.ndarray] = None,
    *,
    pulse_kind: str = "rrc",
    samples_per_symbol: int = 8,
    span_symbols: int = 8,
    rolloff: float = 0.25,
    gain: float = 1.0,
    normalize_energy: bool = True,
    mode: Literal["full", "same", "valid"] = "full"
) -> Tuple[np.ndarray, np.ndarray]:
    sps = _validate_sps(samples_per_symbol)
    symbols = np.asarray(symbols)
    if symbols.ndim != 1:
        raise ValueError("symbols must be a 1D array")

    if pulse is None:
        pulse = generate_pulse(
            pulse=pulse_kind,
            samples_per_symbol=sps,
            span_symbols=span_symbols,
            rolloff=rolloff,
            gain=gain,
            normalize_energy=normalize_energy
        )
    else:
        pulse = np.asarray(pulse, dtype=float)
        if pulse.ndim != 1:
            raise ValueError("pulse must be a 1D array")

    up = upsample_symbols(symbols, sps)
    tx = np.convolve(up, pulse, mode=mode)

    return tx, pulse


# =============================================================================
# MATCHED FILTER
# =============================================================================

def matched_filter(
    rx: np.ndarray,
    pulse: np.ndarray,
    mode: Literal["full", "same", "valid"] = "full"
) -> np.ndarray:
    rx = np.asarray(rx)
    pulse = np.asarray(pulse)

    if rx.ndim != 1:
        raise ValueError("rx must be a 1D array")
    if pulse.ndim != 1:
        raise ValueError("pulse must be a 1D array")

    mf = np.conjugate(pulse[::-1])
    y = np.convolve(rx, mf, mode=mode)

    return y


def pulse_group_delay(pulse: np.ndarray) -> int:
    pulse = np.asarray(pulse)
    if pulse.ndim != 1:
        raise ValueError("pulse must be a 1D array")

    return (len(pulse) - 1) // 2


def sample_matched_filter_output(
    y_mf: np.ndarray,
    samples_per_symbol: int,
    pulse: Optional[np.ndarray] = None,
    *,
    delay_samples: Optional[int] = None,
    n_symbols: Optional[int] = None
) -> np.ndarray:
    sps = _validate_sps(samples_per_symbol)
    y_mf = np.asarray(y_mf)

    if y_mf.ndim != 1:
        raise ValueError("y_mf must be a 1D array")

    if delay_samples is None:
        if pulse is not None:
            delay_samples = pulse_group_delay(pulse)
        else:
            delay_samples = 0

    samples = y_mf[delay_samples::sps]

    if n_symbols is not None:
        samples = samples[:n_symbols]

    return samples


# =============================================================================
# COMPATIBILITY WRAPPERS FOR PPMCore
# =============================================================================

def build_tx_pulse(
    pulse_type: str,
    pulse_width: float,
    dt: float,
    rolloff: float = 0.25,
    span: int = 8,
    normalize_energy: bool = True,
    gain: float = 1.0,
) -> np.ndarray:
    """
    Compatibility wrapper expected by PPMCore.

    Parameters
    ----------
    pulse_type : str
        Examples:
        - "rect"
        - "rectangular"
        - "rc"
        - "raised_cosine"
        - "rrc"
        - "root_raised_cosine"

    pulse_width : float
        Pulse width in seconds.

    dt : float
        Time resolution in seconds.

    rolloff : float
        Roll-off factor for RC/RRC.

    span : int
        Span in symbols for RC/RRC.

    normalize_energy : bool
        If True, normalize pulse energy.

    gain : float
        Multiplicative gain.

    Returns
    -------
    np.ndarray
        1D pulse-shaping waveform.
    """
    if pulse_width <= 0:
        raise ValueError("pulse_width must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")

    sps = max(1, int(round(pulse_width / dt)))

    pulse_kind = _canonical_pulse_name(pulse_type)

    return generate_pulse(
        pulse=pulse_kind,
        samples_per_symbol=sps,
        span_symbols=span,
        rolloff=rolloff,
        gain=gain,
        normalize_energy=normalize_energy
    )


def build_matched_filter(tx_pulse: np.ndarray) -> np.ndarray:
    """
    Compatibility wrapper expected by PPMCore.

    For a real-valued pulse, the matched filter is:
        h_mf[n] = h_tx[-n]
    """
    tx_pulse = np.asarray(tx_pulse)
    if tx_pulse.ndim != 1:
        raise ValueError("tx_pulse must be a 1D array")

    return np.conjugate(tx_pulse[::-1])


# =============================================================================
# COMBINED TX/RX HELPERS
# =============================================================================

def tx_rx_shaping_chain(
    symbols: np.ndarray,
    *,
    pulse_kind: str = "rrc",
    samples_per_symbol: int = 8,
    span_symbols: int = 8,
    rolloff: float = 0.25,
    gain: float = 1.0,
    normalize_energy: bool = True,
    tx_mode: Literal["full", "same", "valid"] = "full",
    mf_mode: Literal["full", "same", "valid"] = "full"
) -> dict:
    symbols = np.asarray(symbols)
    if symbols.ndim != 1:
        raise ValueError("symbols must be a 1D array")

    tx, pulse = pulse_shape(
        symbols=symbols,
        pulse=None,
        pulse_kind=pulse_kind,
        samples_per_symbol=samples_per_symbol,
        span_symbols=span_symbols,
        rolloff=rolloff,
        gain=gain,
        normalize_energy=normalize_energy,
        mode=tx_mode
    )

    y_mf = matched_filter(
        rx=tx,
        pulse=pulse,
        mode=mf_mode
    )

    total_delay = 2 * pulse_group_delay(pulse)
    samples = sample_matched_filter_output(
        y_mf=y_mf,
        samples_per_symbol=samples_per_symbol,
        delay_samples=total_delay,
        n_symbols=len(symbols)
    )

    return {
        "pulse": pulse,
        "tx": tx,
        "y_mf": y_mf,
        "samples": samples,
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "PulseKind",
    "PulseShapingConfig",
    "pulse_energy",
    "pulse_peak",
    "pulse_time_vector",
    "rectangular_pulse",
    "raised_cosine_pulse",
    "root_raised_cosine_pulse",
    "generate_pulse",
    "upsample_symbols",
    "pulse_shape",
    "matched_filter",
    "pulse_group_delay",
    "sample_matched_filter_output",
    "build_tx_pulse",
    "build_matched_filter",
    "tx_rx_shaping_chain",
]
