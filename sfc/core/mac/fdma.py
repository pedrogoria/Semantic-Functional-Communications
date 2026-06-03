"""
sfc/core/mac/fdma.py

Ideal / budget-aware FDMA MAC core.

Purpose
-------
This module implements an FDMA MAC/resource-allocation layer consistent with
the current SFC project convention:

    P  = average transmit power per sensor
    B  = total system bandwidth
    N0 = universal noise spectral-density / noise parameter

The FDMA layer derives per-sensor bandwidth slices:

    B_s = alpha_s * B_total

and, when N0 is provided, also derives per-sensor SNR values:

    SNR_s = P_per_sensor / (B_s * N0)

Design philosophy
-----------------
This FDMA implementation is intentionally "ideal" at the MAC/resource level.

It explicitly provides:
1. bandwidth allocation across sensors;
2. optional per-sensor waveform power normalization to P_per_sensor;
3. budget metadata for Benchmark + FDMA and PPM + FDMA comparisons.

It does NOT yet implement a full explicit spectral FDMA waveform chain:
- no frequency shifting;
- no practical subband filtering;
- no demultiplexing filter bank.

Instead, this class currently acts as:
- a fairness-enforcing resource allocator;
- a clean integration point for Benchmark + FDMA and PPM + FDMA;
- a carrier/subband metadata provider for future explicit spectral FDMA.

Current tensor convention
-------------------------
Sensor-wise waveforms use:

    (time, periods, sensors)

Important separation of responsibilities
----------------------------------------
FDMACore knows the FDMA resource budget:

    B_total, B_s, P_per_sensor, optionally N0 and SNR_s.

The modulation core, e.g. PPMCore, remains MAC-agnostic and does not know
P, B, B_s, or N0.

The physical channel or pipeline is responsible for adding AWGN. This FDMA
class can provide the relevant derived SNR_s when N0 is available.
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


class FDMACore(MACCoreBase):
    """
    Ideal / budget-aware FDMA MAC core.

    Parameters
    ----------
    S : int
        Number of sensors.

    B_total : float
        Total system bandwidth.

    P_per_sensor : float
        Average available transmit power per sensor.

    tau : float
        Period / frame duration.

    bandwidth_allocation : array-like or None, optional
        Allocation fractions across sensors.
        If None, equal split is assumed.

    N0 : float or None, optional
        Universal noise parameter. If provided, this class can derive:

            SNR_s = P_per_sensor / (B_s * N0)

        If None, SNR diagnostics are returned as None.

    normalize_sensor_power : bool, optional
        If True, rescale each sensor waveform so that its average power equals
        P_per_sensor. Default is True, consistent with the physical convention.

    return_nonorthogonal_sum_preview : bool, optional
        If True, also return a simple sum across sensors as a preview waveform.

        Important:
        this preview is NOT a physically correct explicit FDMA superposition,
        because no spectral translation is applied.

    frequency_axis_centered_at_zero : bool, optional
        If True, subband edges are reported over:

            [-B_total/2, +B_total/2]

        Otherwise:

            [0, B_total]
    """

    def __init__(
        self,
        S: int,
        B_total: float,
        P_per_sensor: float,
        tau: float,
        bandwidth_allocation: Optional[np.ndarray] = None,
        N0: Optional[float] = None,
        normalize_sensor_power: bool = True,
        return_nonorthogonal_sum_preview: bool = False,
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
            return_nonorthogonal_sum_preview=return_nonorthogonal_sum_preview,
            frequency_axis_centered_at_zero=frequency_axis_centered_at_zero,
            **kwargs,
        )

        if self.S <= 0:
            raise ValueError("S must be positive.")

        if self.B_total <= 0:
            raise ValueError("B_total must be positive.")

        if self.P_per_sensor <= 0:
            raise ValueError("P_per_sensor must be positive.")

        if self.tau <= 0:
            raise ValueError("tau must be positive.")

        self.N0 = None if N0 is None else float(N0)
        if self.N0 is not None and self.N0 <= 0:
            raise ValueError("N0 must be positive when provided.")

        self.normalize_sensor_power = bool(normalize_sensor_power)
        self.return_nonorthogonal_sum_preview = bool(return_nonorthogonal_sum_preview)
        self.frequency_axis_centered_at_zero = bool(frequency_axis_centered_at_zero)

    # =========================================================================
    # RESOURCE DESCRIPTION
    # =========================================================================

    def get_subband_edges(self) -> np.ndarray:
        """
        Return the FDMA subband edges.

        Returns
        -------
        np.ndarray
            Shape:

                (S, 2)

            where each row is:

                [f_low, f_high]
        """

        widths = self.get_bandwidth_per_sensor()

        if self.frequency_axis_centered_at_zero:
            f_start = -self.B_total / 2.0
        else:
            f_start = 0.0

        edges = np.zeros((self.S, 2), dtype=float)

        cursor = f_start
        for s in range(self.S):
            f_low = cursor
            f_high = cursor + widths[s]

            edges[s, 0] = f_low
            edges[s, 1] = f_high

            cursor = f_high

        return edges

    def get_subband_centers(self) -> np.ndarray:
        """
        Return the center frequency of each FDMA subband.

        Returns
        -------
        np.ndarray
            Shape:

                (S,)
        """

        edges = self.get_subband_edges()
        return np.mean(edges, axis=1)

    def get_snr_per_sensor(self, N0: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Return per-sensor SNR values when N0 is available.

        Formula
        -------
        For each sensor s:

            SNR_s = P_per_sensor / (B_s * N0)

        Parameters
        ----------
        N0 : float or None, optional
            If provided, overrides self.N0 for this computation.

        Returns
        -------
        np.ndarray or None
            Per-sensor SNR values with shape (S,), or None if N0 is not
            available.
        """

        N0_eff = self.N0 if N0 is None else float(N0)

        if N0_eff is None:
            return None

        if N0_eff <= 0:
            raise ValueError("N0 must be positive.")

        B_per_sensor = self.get_bandwidth_per_sensor()

        if np.any(B_per_sensor <= 0):
            raise ValueError("All per-sensor bandwidths must be positive.")

        return self.P_per_sensor / (B_per_sensor * N0_eff)

    def get_snr_per_sensor_db(self, N0: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Return per-sensor SNR values in dB when N0 is available.
        """

        snr = self.get_snr_per_sensor(N0=N0)

        if snr is None:
            return None

        return 10.0 * np.log10(np.maximum(snr, np.finfo(float).tiny))

    def get_capacity_per_sensor(self, N0: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Return per-sensor Shannon capacities when N0 is available.

        Formula
        -------
        For each sensor s:

            C_s = B_s * log2(1 + SNR_s)

        with:

            SNR_s = P_per_sensor / (B_s * N0)

        Returns
        -------
        np.ndarray or None
            Per-sensor capacities in bit/s, or None if N0 is not available.
        """

        snr = self.get_snr_per_sensor(N0=N0)

        if snr is None:
            return None

        B_per_sensor = self.get_bandwidth_per_sensor()

        return B_per_sensor * np.log2(1.0 + snr)

    def describe_budget(self) -> Dict[str, Any]:
        """
        Return the FDMA resource budget summary.

        Extends the base class summary with:
        - subband descriptors;
        - optional SNR diagnostics;
        - optional capacity diagnostics.
        """

        summary = super().describe_budget()

        snr_per_sensor = self.get_snr_per_sensor()
        snr_per_sensor_db = self.get_snr_per_sensor_db()
        capacity_per_sensor = self.get_capacity_per_sensor()

        summary.update({
            "subband_edges": self.get_subband_edges(),
            "subband_centers": self.get_subband_centers(),
            "normalize_sensor_power": self.normalize_sensor_power,
            "return_nonorthogonal_sum_preview": self.return_nonorthogonal_sum_preview,
            "frequency_axis_centered_at_zero": self.frequency_axis_centered_at_zero,
            "N0": self.N0,
            "SNR_per_sensor": snr_per_sensor,
            "SNR_per_sensor_dB": snr_per_sensor_db,
            "capacity_per_sensor": capacity_per_sensor,
        })

        return summary

    # =========================================================================
    # MULTIPLEX
    # =========================================================================

    def multiplex(
        self,
        mac_input: MACInput,
        **kwargs
    ) -> MACMultiplexResult:
        """
        Apply ideal/budget-aware FDMA allocation.

        Parameters
        ----------
        mac_input : MACInput
            Expected main field:

                tx_waveform_per_sensor

            with shape:

                (time, periods, sensors)

        Keyword arguments
        -----------------
        normalize_sensor_power : bool, optional
            Overrides the instance default for this call.

        return_nonorthogonal_sum_preview : bool, optional
            Overrides the instance default for this call.

        Returns
        -------
        MACMultiplexResult
            In this ideal FDMA version:
            - per_sensor_allocated_waveform is the main output;
            - multiplexed_waveform is returned only if the preview sum is enabled.
        """

        if mac_input.tx_waveform_per_sensor is None:
            raise ValueError(
                "FDMACore.multiplex(...) expects mac_input.tx_waveform_per_sensor "
                "to be provided."
            )

        x = ensure_3d_waveform_tensor(mac_input.tx_waveform_per_sensor)
        validate_sensor_dimension(x, self.S)

        if mac_input.t is not None:
            validate_time_dimension(x, mac_input.t)

        normalize_sensor_power = kwargs.get(
            "normalize_sensor_power",
            self.normalize_sensor_power
        )

        return_nonorthogonal_sum_preview = kwargs.get(
            "return_nonorthogonal_sum_preview",
            self.return_nonorthogonal_sum_preview
        )

        x_alloc = np.array(x, dtype=float, copy=True)

        if normalize_sensor_power:
            x_alloc = self._scale_waveforms_to_target_power(
                x_alloc,
                target_power=self.P_per_sensor
            )

        average_power_per_sensor = self._average_power_per_sensor(x_alloc)

        snr_per_sensor = self.get_snr_per_sensor()
        snr_per_sensor_db = self.get_snr_per_sensor_db()
        capacity_per_sensor = self.get_capacity_per_sensor()

        allocation_metadata = {
            "mac_type": "fdma",
            "S": self.S,
            "B_total": self.B_total,
            "P_per_sensor": self.P_per_sensor,
            "tau": self.tau,
            "N0": self.N0,
            "bandwidth_allocation": np.array(self.bandwidth_allocation, dtype=float),
            "B_per_sensor": self.get_bandwidth_per_sensor(),
            "subband_edges": self.get_subband_edges(),
            "subband_centers": self.get_subband_centers(),
            "SNR_per_sensor": snr_per_sensor,
            "SNR_per_sensor_dB": snr_per_sensor_db,
            "capacity_per_sensor": capacity_per_sensor,
            "average_power_per_sensor": average_power_per_sensor,
            "normalize_sensor_power": bool(normalize_sensor_power),
            "ideal_fdma": True,
            "explicit_spectral_synthesis": False,
        }

        if return_nonorthogonal_sum_preview:
            # IMPORTANT:
            # This is NOT a physically correct explicit FDMA superposition.
            # It is only a convenience preview waveform for diagnostics.
            multiplexed = np.sum(x_alloc, axis=2)
        else:
            multiplexed = None

        return MACMultiplexResult(
            multiplexed_waveform=multiplexed,
            per_sensor_allocated_waveform=x_alloc,
            allocation_metadata=allocation_metadata,
            aux={
                "warning": (
                    "This FDMA implementation is ideal/budget-aware. "
                    "No explicit spectral frequency translation is applied yet."
                )
            }
        )

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
        Ideal FDMA inverse allocation.

        Parameters
        ----------
        received_signal : Any
            Preferred accepted forms in this first implementation:

            1. np.ndarray with shape (time, periods, sensors)
               -> interpreted as already separated sensor-wise waveforms.

            2. dict containing:
                   {"recovered_waveform_per_sensor": ...}

            3. None, together with multiplex_result containing
               per_sensor_allocated_waveform
               -> useful for fully ideal/self-contained tests.

        multiplex_result : MACMultiplexResult or None
            Optional transmit-side FDMA metadata.

        Keyword arguments
        -----------------
        do_power_diagnostics : bool, optional
            If True, include recovered average power per sensor in aux.

        Returns
        -------
        MACDemultiplexResult
            Recovered waveform per sensor.
        """

        do_power_diagnostics = kwargs.get("do_power_diagnostics", True)

        x_rec = None

        if received_signal is None:
            if (
                multiplex_result is None
                or multiplex_result.per_sensor_allocated_waveform is None
            ):
                raise ValueError(
                    "FDMACore.demultiplex(...) received_signal is None and no "
                    "per_sensor_allocated_waveform is available in multiplex_result."
                )

            x_rec = ensure_3d_waveform_tensor(
                multiplex_result.per_sensor_allocated_waveform
            )

        elif isinstance(received_signal, dict):
            if "recovered_waveform_per_sensor" not in received_signal:
                raise KeyError(
                    "When received_signal is a dict, it must contain "
                    "'recovered_waveform_per_sensor'."
                )

            x_rec = ensure_3d_waveform_tensor(
                received_signal["recovered_waveform_per_sensor"]
            )

        else:
            x_rec = ensure_3d_waveform_tensor(received_signal)

        validate_sensor_dimension(x_rec, self.S)

        aux = {}

        if do_power_diagnostics:
            aux["recovered_average_power_per_sensor"] = self._average_power_per_sensor(
                x_rec
            )

        return MACDemultiplexResult(
            recovered_waveform_per_sensor=x_rec,
            recovered_symbols_per_sensor=None,
            recovered_event_matrix=None,
            aux=aux
        )

    # =========================================================================
    # POWER HELPERS
    # =========================================================================

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

        if target_power <= 0:
            raise ValueError("target_power must be positive.")

        x = np.asarray(x, dtype=float)
        x_out = np.array(x, copy=True)

        _, _, S = x.shape

        for s in range(S):
            ps = np.mean(x[:, :, s] ** 2)

            if np.isclose(ps, 0.0):
                # Leave null signals unchanged to avoid division by zero.
                continue

            scale = np.sqrt(target_power / ps)
            x_out[:, :, s] *= scale

        return x_out

    def _average_power_per_sensor(self, x: np.ndarray) -> np.ndarray:
        """
        Compute average waveform power per sensor.

        Parameters
        ----------
        x : np.ndarray
            Canonical waveform tensor:

                (time, periods, sensors)

        Returns
        -------
        np.ndarray
            Shape:

                (S,)
        """

        x = np.asarray(x, dtype=float)

        if x.ndim != 3:
            raise ValueError("x must have shape (time, periods, sensors).")

        _, _, S = x.shape

        p = np.zeros(S, dtype=float)

        for s in range(S):
            p[s] = np.mean(x[:, :, s] ** 2)

        return p


__all__ = [
    "FDMACore",
]
