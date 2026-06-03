"""
sfc/core/acquisition/__init__.py

Core acquisition package.

Purpose
-------
This package groups source-acquisition / source-representation primitives that
define how continuous-time source signals are converted into compact payloads
before modulation / communication.

In the current architecture:

- acquisition is responsible for:
    * how the source signal is sampled / represented / compressed
    * Nyquist-style sample acquisition
    * compressive sensing acquisition
    * finite-rate-of-innovation acquisition (future)
    * reconstruction back to the source-signal domain

- modulation is responsible for:
    * how the acquired payload becomes a waveform or signaling object
    * pulse shaping
    * modulation / demodulation

- MAC is responsible for:
    * how multiple sensors share the medium

Design principle
----------------
Acquisition must remain separate from modulation and MAC.

Current status
--------------
This package initializer exports:
- the stable acquisition base abstractions
- the first CS acquisition core

Recommended future modules
--------------------------
- sfc.core.acquisition.base
- sfc.core.acquisition.nyquist
- sfc.core.acquisition.cs
- sfc.core.acquisition.fri
"""

from .base import (
    AcquisitionResult,
    ReconstructionResult,
    AcquisitionCoreBase,
    ensure_3d_signal_tensor,
    ensure_1d_time_vector,
    validate_time_axis_length,
)

from .cs import CSAcquisitionCore

__all__ = [
    "AcquisitionResult",
    "ReconstructionResult",
    "AcquisitionCoreBase",
    "ensure_3d_signal_tensor",
    "ensure_1d_time_vector",
    "validate_time_axis_length",
    "CSAcquisitionCore",
]
