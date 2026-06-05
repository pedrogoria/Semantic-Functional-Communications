"""
sfc/core/modulation/__init__.py

Core modulation package.

Purpose
-------
This package groups modulation-related primitives that should remain
independent from MAC-layer decisions.

Planned responsibilities
------------------------
- base abstractions for modulation / demodulation
- reusable pulse shaping utilities
- concrete modem implementations such as PPM

Design principle
----------------
Modulation must remain separate from medium-access logic.

That means this package should implement operations such as:
- signal normalization / denormalization (when part of the modulation method)
- message sampling
- pulse generation / pulse shaping
- modulation
- demodulation
- continuous-time reconstruction (when appropriate)

while leaving decisions such as:
- TDMA
- FDMA
- OFDM
- SFC resource access / map allocation

to the sibling package:
    sfc.core.mac

Current status
--------------
This package initializer is intentionally lightweight so that the package
can be introduced safely before all underlying modules are fully implemented.

Recommended future modules
--------------------------
- sfc.core.modulation.base
- sfc.core.modulation.pulse_shaping
- sfc.core.modulation.ppm
"""

__all__ = []
