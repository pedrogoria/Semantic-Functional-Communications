"""
sfc/core/mac/ofdm.py

Ideal / budget-aware OFDM MAC core.

Purpose
-------
This module implements a first OFDM-aware MAC layer consistent with the new
architecture:

    sfc/core/mac/

and with future comparison stacks such as:

    - Benchmark + OFDM
    - PPM + OFDM

Design philosophy
-----------------
This first OFDM implementation is intentionally resource-grid oriented:

1. It explicitly splits the total bandwidth B_total into N_subcarriers equally
   spaced subcarriers.

2. It allocates disjoint subcarrier groups to sensors according to an allocation
   vector.

3. It optionally maps sensor-wise symbols into a shared OFDM resource grid.

4. It optionally returns a time-domain preview obtained by IFFT across the
   subcarrier axis, with optional cyclic prefix.

This module should currently be understood as:
- a fairness-enforcing OFDM resource allocator,
- a clean integration point for future OFDM-based baselines,
- a provider of subcarrier metadata and ideal resource mapping.

Current scope
-------------
This module does NOT yet implement:
- channel estimation
- equalization
- CFO handling
- full practical OFDM PHY details

Tensor / grid conventions
-------------------------
This MAC is most naturally defined in a resource-grid domain.

The main OFDM grid convention used here is:

    (subcarriers, ofdm_symbols, periods)

Sensor-wise symbol inputs are assumed to be:

    (ofdm_symbols, periods, sensors)

That is:
- one scalar symbol stream per sensor
- each sensor occupies its allocated subcarrier set
- the same scalar symbol is replicated over the sensor-owned subcarriers,
  with per-subcarrier normalization to preserve total stream energy

This is not the only possible OFDM mapping, but it is a clean and useful
starting point for system comparison.

IMPORTANT
---------
This class is still MAC-level and not meant to replace a full OFDM modem.
It defines:
- how subcarriers are allocated
- how sensor streams are placed into the grid
- how they are recovered ideally
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from sfc.core.mac.base import (
    MACCoreBase,
    MACInput,
    MACMultiplexResult,
    MACDemultiplexResult,
    ensure_3d_waveform_tensor,
    validate_sensor_dimension,
    validate_time_dimension,
)


class OFDMCore(MACCoreBase):
    """
    Ideal / budget-aware OFDM MAC core.

    Parameters
    ----------
    S : int
        Number of sensors.

    B_total : float
        Total communication bandwidth.

    P_per_sensor : float
        Average available power per sensor.

    tau : float
        Period / frame duration.

    n_subcarriers : int
        Total number of OFDM subcarriers.

    bandwidth_allocation : array-like or None, optional
        Allocation fractions across sensors.
        If None, equal split among sensors is assumed.
        The same allocation is used to derive subcarrier counts.

    cp_fraction : float, optional
        Cyclic-prefix fraction relative to the IFFT size.
        Example:
            cp_fraction = 0.25
        means:
            cp_len = round(0.25 * n_subcarriers)

    normalize_sensor_power : bool, optional
        If True, normalize each sensor stream / waveform to the average target
        power P_per_sensor when applicable. Default is False.

    return_ifft_preview : bool, optional
        If True, build a time-domain OFDM preview waveform via IFFT.
        Default is False.

    frequency_axis_centered_at_zero : bool, optional
        If True, subcarrier frequencies are reported centered around zero.
        Otherwise, reported on [0, B_total).
        Default is True.
    """

    def __init__(
        self,
        S: int,
        B_total: float,
        P_per_sensor: float,
        tau: float,
        n_subcarriers: int,
        bandwidth_allocation: Optional[np.ndarray] = None,
        cp_fraction: float = 0.0,
        normalize_sensor_power: bool = False,
        return_ifft_preview: bool = False,
        frequency_axis_centered_at_zero: bool = True,
        **kwargs
    ):
        super().__init__(
            S=S,
            B_total=B_total,
            P_per_sensor=P_per_sensor,
            tau=tau,
            bandwidth_allocation=bandwidth_allocation,
            normalize_sensor_power=normalize_sensor_power,
            return_ifft_preview=return_ifft_preview,
            frequency_axis_centered_at_zero=frequency_axis_centered_at_zero,
            n_subcarriers=n_subcarriers,
            cp_fraction=cp_fraction,
            **kwargs,
        )

        if n_subcarriers < 1:
            raise ValueError("n_subcarriers must be >= 1.")

        if cp_fraction < 0:
            raise ValueError("cp_fraction must be >= 0.")

        self.n_subcarriers = int(n_subcarriers)
        self.cp_fraction = float(cp_fraction)
        self.normalize_sensor_power = bool(normalize_sensor_power)
        self.return_ifft_preview = bool(return_ifft_preview)
        self.frequency_axis_centered_at_zero = bool(frequency_axis_centered_at_zero)

        self.subcarrier_counts = _resolve_subcarrier_counts(
            n_subcarriers=self.n_subcarriers,
            allocation=self.bandwidth_allocation
        )
        self.subcarrier_indices = self._build_subcarrier_indices_per_sensor()

    # =========================================================================
    # RESOURCE DESCRIPTION
    # =========================================================================

    def get_subcarrier_width(self) -> float:
        """
        Return the bandwidth per subcarrier.

        Returns
        -------
        float
            B_total / n_subcarriers
        """
        return self.B_total / self.n_subcarriers

    def get_subcarrier_frequencies(self) -> np.ndarray:
        """
        Return subcarrier center frequencies.

        Returns
        -------
        np.ndarray
            Shape:
                (n_subcarriers,)
        """

        df = self.get_subcarrier_width()

        if self.frequency_axis_centered_at_zero:
            # Frequencies centered around zero
            # Example for N carriers:
            #   [-N/2, ..., N/2-1] * df
            idx = np.arange(self.n_subcarriers) - self.n_subcarriers / 2.0
            return idx * df
        else:
            return np.arange(self.n_subcarriers) * df

    def get_subcarrier_counts(self) -> np.ndarray:
        """
        Return the number of subcarriers allocated to each sensor.

        Returns
        -------
        np.ndarray
            Shape:
                (S,)
        """
        return np.array(self.subcarrier_counts, dtype=int)

    def get_subcarrier_indices_per_sensor(self) -> Dict[int, np.ndarray]:
        """
        Return the allocated subcarrier indices for each sensor.

        Returns
        -------
        dict
            Mapping:
                sensor_id -> np.ndarray of indices
        """
        return {
            s: np.array(self.subcarrier_indices[s], dtype=int)
            for s in range(self.S)
        }

    def get_subcarrier_ranges(self) -> np.ndarray:
        """
        Return the min/max subcarrier indices allocated to each sensor.

        Returns
        -------
        np.ndarray
            Shape:
                (S, 2)

            each row:
                [k_min, k_max]

            If a sensor has no subcarriers, returns [-1, -1].
        """

        ranges = np.zeros((self.S, 2), dtype=int)

        for s in range(self.S):
            idx = self.subcarrier_indices[s]
            if len(idx) == 0:
                ranges[s, :] = -1
            else:
                ranges[s, 0] = int(np.min(idx))
                ranges[s, 1] = int(np.max(idx))

        return ranges

    def describe_budget(self) -> Dict[str, Any]:
        """
        Return the OFDM resource budget summary.
        """

        summary = super().describe_budget()
        summary.update({
            "mac_type": "ofdm",
            "n_subcarriers": self.n_subcarriers,
            "subcarrier_width": self.get_subcarrier_width(),
            "subcarrier_frequencies": self.get_subcarrier_frequencies(),
            "subcarrier_counts": self.get_subcarrier_counts(),
            "subcarrier_ranges": self.get_subcarrier_ranges(),
            "subcarrier_indices_per_sensor": self.get_subcarrier_indices_per_sensor(),
            "cp_fraction": self.cp_fraction,
            "cp_length_samples": int(round(self.cp_fraction * self.n_subcarriers)),
            "normalize_sensor_power": self.normalize_sensor_power,
            "return_ifft_preview": self.return_ifft_preview,
            "frequency_axis_centered_at_zero": self.frequency_axis_centered_at_zero,
        })
        return summary

    def _build_subcarrier_indices_per_sensor(self) -> Dict[int, np.ndarray]:
        """
        Build a contiguous subcarrier allocation per sensor.

        Returns
        -------
        dict
            sensor_id -> np.ndarray of subcarrier indices
        """

        indices = {}
        cursor = 0

        for s in range(self.S):
            k_count = int(self.subcarrier_counts[s])
            indices[s] = np.arange(cursor, cursor + k_count, dtype=int)
            cursor += k_count

        return indices

    # =========================================================================
    # MULTIPLEX
    # =========================================================================

    def multiplex(
        self,
        mac_input: MACInput,
        **kwargs
    ) -> MACMultiplexResult:
        """
        Apply OFDM resource allocation.

        Accepted input modes
        --------------------
        1. Symbol-domain mode (preferred):
           mac_input.symbols_per_sensor with shape:
               (n_ofdm_symbols, periods, sensors)

           In this case:
           - the method builds an OFDM resource grid
           - each sensor occupies its allocated subcarrier set
           - the same sensor symbol is replicated across the owned subcarriers,
             scaled by 1/sqrt(K_s), where K_s is the number of subcarriers owned
             by that sensor

        2. Waveform pass-through mode:
           mac_input.tx_waveform_per_sensor with shape:
               (time, periods, sensors)

           In this case:
           - no OFDM grid is built
           - the function only attaches the OFDM allocation metadata
           - this is useful as a placeholder integration step if a waveform-level
             OFDM PHY is not yet available

        Keyword arguments
        -----------------
        normalize_sensor_power : bool, optional
            Overrides the instance default for this call.

        return_ifft_preview : bool, optional
            Overrides the instance default for this call.

        Returns
        -------
        MACMultiplexResult
        """

        normalize_sensor_power = kwargs.get(
            "normalize_sensor_power",
            self.normalize_sensor_power
        )
        return_ifft_preview = kwargs.get(
            "return_ifft_preview",
            self.return_ifft_preview
        )

        allocation_metadata = {
            "mac_type": "ofdm",
            "S": self.S,
            "B_total": self.B_total,
            "P_per_sensor": self.P_per_sensor,
            "tau": self.tau,
            "bandwidth_allocation": np.array(self.bandwidth_allocation, dtype=float),
            "B_per_sensor": self.get_bandwidth_per_sensor(),
            "n_subcarriers": self.n_subcarriers,
            "subcarrier_width": self.get_subcarrier_width(),
            "subcarrier_frequencies": self.get_subcarrier_frequencies(),
            "subcarrier_counts": self.get_subcarrier_counts(),
            "subcarrier_ranges": self.get_subcarrier_ranges(),
            "subcarrier_indices_per_sensor": self.get_subcarrier_indices_per_sensor(),
            "cp_fraction": self.cp_fraction,
            "cp_length_samples": int(round(self.cp_fraction * self.n_subcarriers)),
            "normalize_sensor_power": bool(normalize_sensor_power),
            "ideal_ofdm": True,
            "explicit_channel_equalization": False,
        }

        # ---------------------------------------------------------------------
        # MODE 1: Symbol-domain OFDM grid mapping
        # ---------------------------------------------------------------------
        if mac_input.symbols_per_sensor is not None:
            x = np.asarray(mac_input.symbols_per_sensor, dtype=complex)

            if x.ndim != 3:
                raise ValueError(
                    "mac_input.symbols_per_sensor must have shape "
                    "(n_ofdm_symbols, periods, sensors)"
                )

            n_ofdm_symbols, n_periods, Sdim = x.shape
            if Sdim != self.S:
                raise ValueError(
                    f"Sensor dimension mismatch in symbols_per_sensor: "
                    f"{Sdim} != S={self.S}"
                )

            if normalize_sensor_power:
                x = self._scale_symbol_streams_to_target_power(
                    x,
                    target_power=self.P_per_sensor
                )

            resource_grid = self._build_resource_grid_from_sensor_streams(x)

            aux = {
                "resource_grid": resource_grid,
            }

            multiplexed_waveform = None
            if return_ifft_preview:
                preview = self._build_time_domain_ifft_preview(resource_grid)
                multiplexed_waveform = preview
                aux["ifft_preview"] = preview

            return MACMultiplexResult(
                multiplexed_waveform=multiplexed_waveform,
                per_sensor_allocated_waveform=None,
                allocation_metadata=allocation_metadata,
                aux=aux
            )

        # ---------------------------------------------------------------------
        # MODE 2: Waveform pass-through with OFDM allocation metadata
        # ---------------------------------------------------------------------
        if mac_input.tx_waveform_per_sensor is not None:
            x = ensure_3d_waveform_tensor(mac_input.tx_waveform_per_sensor)
            validate_sensor_dimension(x, self.S)

            if mac_input.t is not None:
                validate_time_dimension(x, mac_input.t)

            x_alloc = np.array(x, dtype=float, copy=True)

            if normalize_sensor_power:
                x_alloc = self._scale_waveforms_to_target_power(
                    x_alloc,
                    target_power=self.P_per_sensor
                )

            return MACMultiplexResult(
                multiplexed_waveform=None,
                per_sensor_allocated_waveform=x_alloc,
                allocation_metadata=allocation_metadata,
                aux={
                    "warning": (
                        "Waveform pass-through mode used. "
                        "No explicit OFDM grid was built."
                    )
                }
            )

        raise ValueError(
            "OFDMCore.multiplex(...) requires either:\n"
            "- mac_input.symbols_per_sensor\n"
            "or\n"
            "- mac_input.tx_waveform_per_sensor"
        )

    def _build_resource_grid_from_sensor_streams(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        """
        Build OFDM resource grid from per-sensor scalar streams.

        Parameters
        ----------
        x : np.ndarray
            Shape:
                (n_ofdm_symbols, periods, sensors)

        Returns
        -------
        np.ndarray
            Resource grid with shape:
                (n_subcarriers, n_ofdm_symbols, periods)
        """

        n_ofdm_symbols, n_periods, _ = x.shape
        grid = np.zeros((self.n_subcarriers, n_ofdm_symbols, n_periods), dtype=complex)

        for s in range(self.S):
            idx = self.subcarrier_indices[s]
            Ks = len(idx)

            if Ks == 0:
                continue

            # Replicate the same stream across the allocated subcarriers.
            # Scale by 1/sqrt(Ks) so total energy of the stream is preserved.
            for k in idx:
                grid[k, :, :] = x[:, :, s] / np.sqrt(Ks)

        return grid

    def _build_time_domain_ifft_preview(
        self,
        resource_grid: np.ndarray
    ) -> np.ndarray:
        """
        Build a time-domain OFDM preview waveform by IFFT across subcarriers.

        Parameters
        ----------
        resource_grid : np.ndarray
            Shape:
                (n_subcarriers, n_ofdm_symbols, periods)

        Returns
        -------
        np.ndarray
            Time-domain preview with shape:
                (n_time_samples, n_ofdm_symbols, periods)

            If cp_fraction > 0, the preview includes cyclic prefix.
        """

        grid = np.asarray(resource_grid, dtype=complex)

        if grid.ndim != 3:
            raise ValueError(
                "resource_grid must have shape (n_subcarriers, n_ofdm_symbols, periods)"
            )

        n_subcarriers, n_ofdm_symbols, n_periods = grid.shape
        if n_subcarriers != self.n_subcarriers:
            raise ValueError(
                f"resource_grid subcarrier mismatch: {n_subcarriers} != {self.n_subcarriers}"
            )

        # IFFT across subcarrier axis
        time_domain = np.fft.ifft(grid, axis=0)

        cp_len = int(round(self.cp_fraction * self.n_subcarriers))
        if cp_len <= 0:
            return time_domain

        preview = np.zeros(
            (self.n_subcarriers + cp_len, n_ofdm_symbols, n_periods),
            dtype=complex
        )

        preview[:cp_len, :, :] = time_domain[-cp_len:, :, :]
        preview[cp_len:, :, :] = time_domain

        return preview

    # =========================================================================
    # DEMULTIPLEX
    # =========================================================================

    def demultiplex(
        self,
        received_signal: Any,
        multiplex_result: Optional[MACMultiplexResult] = None,
        **kwargs
    ) -> MACDemultiplexResult:
        """
        Ideal OFDM inverse allocation.

        Accepted input modes
        --------------------
        1. received_signal is a dict containing:
               {"resource_grid": ...}
           or
               {"recovered_symbols_per_sensor": ...}

        2. received_signal is directly a resource grid with shape:
               (n_subcarriers, n_ofdm_symbols, periods)

        3. received_signal is None and multiplex_result.aux contains "resource_grid"

        Returns
        -------
        MACDemultiplexResult
            In symbol-domain mode:
                recovered_symbols_per_sensor with shape
                    (n_ofdm_symbols, periods, sensors)
        """

        if received_signal is None:
            if multiplex_result is None or "resource_grid" not in multiplex_result.aux:
                raise ValueError(
                    "OFDMCore.demultiplex(...) received_signal is None and no "
                    "resource_grid is available in multiplex_result."
                )
            resource_grid = np.asarray(multiplex_result.aux["resource_grid"], dtype=complex)

        elif isinstance(received_signal, dict):
            if "recovered_symbols_per_sensor" in received_signal:
                return MACDemultiplexResult(
                    recovered_waveform_per_sensor=None,
                    recovered_symbols_per_sensor=np.asarray(
                        received_signal["recovered_symbols_per_sensor"]
                    ),
                    recovered_event_matrix=None,
                    aux={}
                )

            if "resource_grid" not in received_signal:
                raise KeyError(
                    "When received_signal is a dict, it must contain either "
                    "'resource_grid' or 'recovered_symbols_per_sensor'."
                )

            resource_grid = np.asarray(received_signal["resource_grid"], dtype=complex)

        else:
            resource_grid = np.asarray(received_signal, dtype=complex)

        if resource_grid.ndim != 3:
            raise ValueError(
                "resource_grid must have shape (n_subcarriers, n_ofdm_symbols, periods)"
            )

        n_subcarriers, n_ofdm_symbols, n_periods = resource_grid.shape
        if n_subcarriers != self.n_subcarriers:
            raise ValueError(
                f"resource_grid subcarrier mismatch: {n_subcarriers} != {self.n_subcarriers}"
            )

        x_rec = np.zeros((n_ofdm_symbols, n_periods, self.S), dtype=complex)

        for s in range(self.S):
            idx = self.subcarrier_indices[s]
            Ks = len(idx)

            if Ks == 0:
                continue

            # Since multiplex(...) replicated the same stream across all owned
            # subcarriers with a 1/sqrt(Ks) factor, the inverse is:
            #
            #   x_hat = mean_k( grid[k] * sqrt(Ks) )
            #
            # which is exact in the ideal noiseless case.
            sensor_grid = resource_grid[idx, :, :] * np.sqrt(Ks)
            x_rec[:, :, s] = np.mean(sensor_grid, axis=0)

        return MACDemultiplexResult(
            recovered_waveform_per_sensor=None,
            recovered_symbols_per_sensor=x_rec,
            recovered_event_matrix=None,
            aux={
                "subcarrier_indices_per_sensor": self.get_subcarrier_indices_per_sensor()
            }
        )

    # =========================================================================
    # POWER HELPERS
    # =========================================================================

    def _scale_symbol_streams_to_target_power(
        self,
        x: np.ndarray,
        target_power: float
    ) -> np.ndarray:
        """
        Rescale each sensor symbol stream so that its average power equals target_power.

        Parameters
        ----------
        x : np.ndarray
            Shape:
                (n_ofdm_symbols, periods, sensors)

        target_power : float
            Desired average power per sensor.

        Returns
        -------
        np.ndarray
            Power-normalized symbol tensor.
        """

        x = np.asarray(x, dtype=complex)
        x_out = np.array(x, copy=True)

        _, _, S = x.shape

        for s in range(S):
            ps = np.mean(np.abs(x[:, :, s]) ** 2)

            if np.isclose(ps, 0.0):
                continue

            scale = np.sqrt(target_power / ps)
            x_out[:, :, s] *= scale

        return x_out

    def _scale_waveforms_to_target_power(
        self,
        x: np.ndarray,
        target_power: float
    ) -> np.ndarray:
        """
        Rescale each sensor waveform so that its average power equals target_power.

        Parameters
        ----------
        x : np.ndarray
            Canonical waveform tensor:
                (time, periods, sensors)

        target_power : float
            Desired average power per sensor.

        Returns
        -------
        np.ndarray
            Power-normalized waveform tensor.
        """

        x = np.asarray(x, dtype=float)
        x_out = np.array(x, copy=True)

        _, _, S = x.shape

        for s in range(S):
            ps = np.mean(x[:, :, s] ** 2)

            if np.isclose(ps, 0.0):
                continue

            scale = np.sqrt(target_power / ps)
            x_out[:, :, s] *= scale

        return x_out


# =============================================================================
# SUBCARRIER ALLOCATION HELPERS
# =============================================================================

def _resolve_subcarrier_counts(
    n_subcarriers: int,
    allocation: np.ndarray
) -> np.ndarray:
    """
    Convert allocation fractions into integer subcarrier counts.

    Strategy
    --------
    - compute floor(allocation * n_subcarriers)
    - distribute the remaining carriers to the largest residuals

    Parameters
    ----------
    n_subcarriers : int
        Total number of subcarriers.

    allocation : np.ndarray
        Allocation fractions of shape:
            (S,)

    Returns
    -------
    np.ndarray
        Integer subcarrier counts of shape:
            (S,)
    """

    allocation = np.asarray(allocation, dtype=float).reshape(-1)

    raw = allocation * n_subcarriers
    counts = np.floor(raw).astype(int)

    remainder = n_subcarriers - np.sum(counts)
    if remainder > 0:
        residuals = raw - counts
        order = np.argsort(-residuals)

        for i in range(remainder):
            counts[order[i]] += 1

    return counts


__all__ = [
    "OFDMCore",
]
