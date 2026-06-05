"""
sfc/core/mac/__init__.py

Core MAC package.

Purpose
-------
This package groups medium-access-control (MAC) primitives that define how
multiple sensors share the communication medium under a common physical budget.

In the current architecture:

- modulation is responsible for:
    * message normalization / denormalization when applicable
    * message sampling
    * pulse shaping
    * modulation / demodulation
    * optional continuous-time reconstruction

- MAC is responsible for:
    * how multiple sensors share the medium
    * bandwidth slicing
    * time slicing
    * subcarrier/resource allocation
    * multiplexing and inverse demultiplexing

Design principle
----------------
MAC must remain separate from modulation.

That means this package should implement operations such as:
- FDMA allocation
- TDMA allocation
- OFDM resource mapping
- SFC-specific resource/event access logic

while leaving signal-representation and waveform-generation details to:
    sfc.core.modulation

Current status
--------------
This package initializer is intentionally conservative:

- it exports the stable base abstractions;
- it exports FDMACore, which is the first conventional MAC prepared for the
  fair-comparison stack:
      Benchmark + FDMA
      PPM + FDMA
      SFC
- it does NOT yet import TDMA / OFDM / SFC MAC implementations, because those
  modules may still be under development.

Recommended future modules
--------------------------
- sfc.core.mac.base
- sfc.core.mac.fdma
- sfc.core.mac.tdma
- sfc.core.mac.ofdm
- sfc.core.mac.sfc
"""

from .base import (
    MACInput,
    MACMultiplexResult,
    MACDemultiplexResult,
    MACCoreBase,
    ensure_3d_waveform_tensor,
    ensure_2d_aggregated_waveform,
    validate_sensor_dimension,
    validate_time_dimension,
)

from .fdma import FDMACore

__all__ = [
    "MACInput",
    "MACMultiplexResult",
    "MACDemultiplexResult",
    "MACCoreBase",
    "ensure_3d_waveform_tensor",
    "ensure_2d_aggregated_waveform",
    "validate_sensor_dimension",
    "validate_time_dimension",
    "FDMACore",
]
