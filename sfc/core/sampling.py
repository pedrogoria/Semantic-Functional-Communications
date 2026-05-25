"""
sfc/core/sampling.py

Core façade for sampling-related functionality.

This module should act as the main access point for the sampling subsystem,
while delegating the actual scientific logic to the specialized core modules.

Extracted scientific logic now lives in:
- sfc/core/phase_cof.py
- sfc/core/fourier.py
- sfc/core/reconstruction.py
- sfc/core/quantization.py
- sfc/core/filters.py
- sfc/core/nyquist.py
- sfc/core/delta_sampler.py

CRITICAL
--------
This file must remain a thin orchestration / compatibility layer.

Allowed responsibilities:
- parameter bookkeeping;
- module composition;
- compatibility wrappers;
- re-export of trusted functionality.

Forbidden responsibilities:
- re-implementing extracted equations;
- duplicating Fourier logic;
- duplicating phase logic;
- duplicating reconstruction logic;
- duplicating quantization logic.

The goal is to preserve compatibility with the original trusted sampling API
while avoiding duplicated scientific logic.
"""

import numpy as np

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


class CosinePhaseCore(PhaseCoefficientCore):
    """
    Backward-compatible phase-only core.

    Why this class exists
    ---------------------
    Earlier refactoring work placed the trusted phase computation directly in
    `sfc/core/sampling.py` under the name `CosinePhaseCore`.

    Now that the trusted phase equations have been extracted to
    `sfc/core/phase_cof.py`, this class remains only as a compatibility wrapper.

    Important
    ---------
    No scientific equations are implemented here.
    All scientific behavior is inherited from `PhaseCoefficientCore`.
    """

    def __init__(self, harmonics, sensors, n, w0, threshold_harmonics=1e-6):
        """
        Initialize the backward-compatible phase core.

        Parameters
        ----------
        harmonics : int
            Number of harmonics.

        sensors : int
            Number of sensors.

        n : np.ndarray
            Harmonic index vector.

        w0 : float
            Fundamental angular frequency.

        threshold_harmonics : float, optional
            Threshold used to zero-out small Fourier coefficients.

        Notes
        -----
        This wrapper preserves the public constructor signature of the current
        `CosinePhaseCore` implementation while delegating the actual phase logic
        to `PhaseCoefficientCore`.
        """

        super().__init__(
            T=2 * np.pi / w0,
            harmonics=harmonics,
            sensor_nodes=sensors,
            threshold_harmonics=threshold_harmonics
        )

        # ------------------------------------------------------------------
        # Preserve public attributes exactly as expected by external code.
        # ------------------------------------------------------------------
        self.harmonics = harmonics
        self.sensors = sensors
        self.n = n
        self.w0 = w0
        self.threshold_harmonics = threshold_harmonics


class CPSample:
    """
    Trusted compatibility façade for the cosine-phase sampling pipeline.

    This class preserves the overall public behavior of the original trusted
    `CPSample` class, but delegates all scientific logic to the extracted
    core modules.

    Delegation map
    --------------
    - Fourier coefficient computation:
        sfc.core.fourier.FourierCoefficientCore
    - Phase extraction and event conversion:
        sfc.core.phase_cof.PhaseCoefficientCore
    - Signal reconstruction:
        sfc.core.reconstruction.ReconstructionCore
    - ta/tb quantization:
        sfc.core.quantization.QuantizationCore

    Important
    ---------
    This class should not internally duplicate any scientific equations.
    """

    def __init__(
        self,
        T=1,
        harmonics=3,
        n_sub_symbol=6,
        resource=7,
        sensor_nodes=5,
        bandwidth=100,
        **options
    ):
        """
        Initialize the cosine-phase sampling façade.

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
            Bandwidth used to derive the sub-symbol time.

        **options : dict
            Additional trusted options preserved from the original CPSample:
            - name
            - detect_errors
            - periods
            - threshold_harmonics
            - dft_signal_periods
        """

        # ------------------------------------------------------------------
        # Preserve the original public naming default.
        # ------------------------------------------------------------------
        self.name = options.pop('name', 'CPM')

        # ------------------------------------------------------------------
        # Preserve the original scalar configuration.
        # ------------------------------------------------------------------
        self.T = T
        self.harmonics = harmonics
        self.sensors = sensor_nodes
        self.n_sub_symbol = n_sub_symbol
        self.resources = resource
        self.bandwidth = bandwidth

        # ------------------------------------------------------------------
        # Preserve the trusted sub-symbol timing definition.
        # ------------------------------------------------------------------
        self.sub_symbol_time = 1 / (self.bandwidth / self.resources)

        # ------------------------------------------------------------------
        # Preserve the trusted sensor-event incidence matrix allocation.
        # ------------------------------------------------------------------
        self.sensors_x_event = np.zeros((self.sensors, 2 * self.sensors * self.harmonics))

        # ------------------------------------------------------------------
        # Preserve the trusted angular-frequency and harmonic-index definitions.
        # ------------------------------------------------------------------
        self.w0 = 2 * np.pi / T
        self.n = np.arange(harmonics) + 1

        # ------------------------------------------------------------------
        # Preserve trusted configuration flags and defaults.
        # ------------------------------------------------------------------
        self.detect_errors = options.pop('detect_errors', False)
        self.rs_per_period = np.floor(T / self.sub_symbol_time)
        self.n_periods = options.pop('periods', 'empty')
        self.threshold_harmonics = options.pop('threshold_harmonics', 0.001)
        self.dft_signal_periods = int(options.pop('dft_signal_periods', 1))

        # ------------------------------------------------------------------
        # Preserve the original assertion exactly.
        # ------------------------------------------------------------------
        assert self.rs_per_period > 1, 'error in parameter rs_per_period '

        # ------------------------------------------------------------------
        # Preserve the trusted sensor-to-event allocation pattern exactly.
        # ------------------------------------------------------------------
        for i in range(self.sensors):
            self.sensors_x_event[
                i,
                (2 * i * self.harmonics):(2 * (i + 1) * self.harmonics)
            ] = np.ones((1, 2 * self.harmonics))

        # ------------------------------------------------------------------
        # Build specialized core modules.
        # ------------------------------------------------------------------
        self._build_modules()

    # ======================================================================
    # Internal helpers
    # ======================================================================

    def _build_modules(self):
        """
        Instantiate the specialized core modules.

        This method should be called once during initialization.
        """

        # ------------------------------------------------------------------
        # Fourier module: handles DFT-based coefficient computation.
        # ------------------------------------------------------------------
        self.fourier = FourierCoefficientCore(
            T=self.T,
            harmonics=self.harmonics,
            sensor_nodes=self.sensors,
            dft_signal_periods=self.dft_signal_periods,
            name=self.name
        )

        # ------------------------------------------------------------------
        # Phase module: handles ta/tb extraction and event conversion.
        # ------------------------------------------------------------------
        self.phase = PhaseCoefficientCore(
            T=self.T,
            harmonics=self.harmonics,
            n_sub_symbol=self.n_sub_symbol,
            resource=self.resources,
            sensor_nodes=self.sensors,
            bandwidth=self.bandwidth,
            detect_errors=self.detect_errors,
            periods=self.n_periods,
            threshold_harmonics=self.threshold_harmonics,
            name=self.name
        )

        # ------------------------------------------------------------------
        # Reconstruction module: handles waveform reconstruction.
        # ------------------------------------------------------------------
        self.reconstruction = ReconstructionCore(
            T=self.T,
            harmonics=self.harmonics,
            sensor_nodes=self.sensors,
            name=self.name
        )

        # ------------------------------------------------------------------
        # Quantization module: handles scalar and ta/tb quantization.
        # ------------------------------------------------------------------
        self.quantization = QuantizationCore(
            T=self.T,
            harmonics=self.harmonics,
            sensor_nodes=self.sensors,
            name=self.name
        )

    def _sync_modules(self):
        """
        Synchronize mutable public attributes with the delegated core modules.

        Why this is needed
        ------------------
        External code may modify attributes such as:
        - detect_errors
        - n_periods
        - threshold_harmonics
        - dft_signal_periods

        Since the specialized modules were instantiated earlier, this method
        updates their internal state before delegation.
        """

        # ------------------------------------------------------------------
        # Keep Fourier-module configuration synchronized.
        # ------------------------------------------------------------------
        self.fourier.T = self.T
        self.fourier.harmonics = self.harmonics
        self.fourier.sensors = self.sensors
        self.fourier.w0 = self.w0
        self.fourier.dft_signal_periods = self.dft_signal_periods

        # ------------------------------------------------------------------
        # Keep phase-module configuration synchronized.
        # ------------------------------------------------------------------
        self.phase.T = self.T
        self.phase.harmonics = self.harmonics
        self.phase.sensors = self.sensors
        self.phase.n_sub_symbol = self.n_sub_symbol
        self.phase.resources = self.resources
        self.phase.bandwidth = self.bandwidth
        self.phase.sub_symbol_time = self.sub_symbol_time
        self.phase.w0 = self.w0
        self.phase.n = self.n
        self.phase.detect_errors = self.detect_errors
        self.phase.rs_per_period = self.rs_per_period
        self.phase.n_periods = self.n_periods
        self.phase.threshold_harmonics = self.threshold_harmonics

        # ------------------------------------------------------------------
        # Keep reconstruction-module configuration synchronized.
        # ------------------------------------------------------------------
        self.reconstruction.T = self.T
        self.reconstruction.harmonics = self.harmonics
        self.reconstruction.sensors = self.sensors
        self.reconstruction.w0 = self.w0

        # ------------------------------------------------------------------
        # Keep quantization-module configuration synchronized.
        # ------------------------------------------------------------------
        self.quantization.T = self.T
        self.quantization.harmonics = self.harmonics
        self.quantization.sensors = self.sensors
        self.quantization.w0 = self.w0

    # ======================================================================
    # Public API: main call
    # ======================================================================

    def __call__(self, x, Tt, **options):
        """
        Convert a signal (or phase parameters) into SFC events.

        This method preserves the orchestration pattern of the original trusted
        `CPSample.__call__` implementation.

        Parameters
        ----------
        x : np.ndarray
            Input signal array.

            Accepted shapes (preserved from the trusted implementation):
            - 1D: (time,)
            - 2D: (time, periods)
            - 3D: (time, periods, sensors)

        Tt : float
            Time step.

        **options : dict
            Trusted options:
            - ta_tb : bool
            - ta : precomputed ta
            - tb : precomputed tb
            - any options accepted by Fourier/phase modules

        Returns
        -------
        np.ndarray
            Event matrix.

        Notes
        -----
        This method preserves:
        - the trusted shape-promotion behavior for x;
        - the trusted use of `np.real` on ta/tb after sampling;
        - the trusted ta/tb to events conversion logic,
          now delegated to `PhaseCoefficientCore.ta_tb_to_events(...)`.
        """

        self._sync_modules()

        if options.pop('ta_tb', False):
            ta = options.pop('ta')
            tb = options.pop('tb')
        else:
            # --------------------------------------------------------------
            # Preserve the original input-shape assertion exactly.
            # --------------------------------------------------------------
            assert len(x.shape) < 4, 'error in parameter shape: x '

            # --------------------------------------------------------------
            # Preserve trusted 1D -> 2D promotion exactly.
            # --------------------------------------------------------------
            if len(x.shape) == 1:
                x = x.reshape(-1, 1)

            # --------------------------------------------------------------
            # Preserve trusted 2D -> 3D promotion exactly.
            # --------------------------------------------------------------
            if len(x.shape) == 2:
                xx = np.zeros(np.append(x.shape, self.sensors))
                xx[:, :, 0] = x
                x = xx

            ta, tb, x = self.sample(x, Tt, **options)
            ta = np.real(ta)
            tb = np.real(tb)

        events = self.phase.ta_tb_to_events(ta, tb)

        # ------------------------------------------------------------------
        # Keep the public n_periods attribute consistent with the phase module.
        # ------------------------------------------------------------------
        self.n_periods = self.phase.n_periods

        return events

    # ======================================================================
    # Public API: delegated scientific operations
    # ======================================================================

    def calc_ta_tb(self, an, bn, **options):
        """
        Delegate phase extraction to the extracted phase module.
        """

        self._sync_modules()
        return self.phase.calc_ta_tb(an, bn, **options)

    def calc_an_bn_dft(self, x, Tt, **options):
        """
        Delegate DFT-based coefficient computation to the extracted Fourier module.
        """

        self._sync_modules()
        return self.fourier.calc_an_bn_dft(x, Tt, **options)

    def sample(self, x, Tt, **options):
        """
        Perform the full cosine-phase sampling step.

        This method preserves the orchestration pattern of the trusted
        `CPSample.sample(...)` method:

            an, bn, x = calc_an_bn_dft(...)
            ta, tb = calc_ta_tb(...)
            return ta, tb, x

        Parameters
        ----------
        x : np.ndarray
            Input signal array.

        Tt : float
            Time step.

        **options : dict
            Optional parameters forwarded to the delegated modules.

        Returns
        -------
        tuple
            (ta, tb, x)

        Notes
        -----
        The returned signal `x` may be modified by the DFT normalization loop,
        exactly as in the trusted implementation.
        """

        self._sync_modules()

        an, bn, x = self.fourier.calc_an_bn_dft(x, Tt, **options)
        ta, tb = self.phase.calc_ta_tb(an, bn, **options)

        return ta, tb, x

    def ta_tb_to_events(self, ta, tb):
        """
        Delegate ta/tb-to-events conversion to the extracted phase module.
        """

        self._sync_modules()
        events = self.phase.ta_tb_to_events(ta, tb)
        self.n_periods = self.phase.n_periods
        return events

    def event_to_ta_tb(self, events, **options):
        """
        Delegate events-to-ta/tb conversion to the extracted phase module.
        """

        self._sync_modules()
        result = self.phase.event_to_ta_tb(events, **options)
        self.n_periods = self.phase.n_periods
        return result

    def recover_signal(self, ta, tb, t, **options):
        """
        Delegate signal reconstruction to the extracted reconstruction module.
        """

        self._sync_modules()
        return self.reconstruction.recover_signal(ta, tb, t, **options)

    def quantize_ta_tb(self, ta, tb, bins, poss_ta=1 / 2, poss_tb=1 / 2):
        """
        Delegate ta/tb quantization to the extracted quantization module.
        """

        self._sync_modules()
        return self.quantization.quantize_ta_tb(
            ta,
            tb,
            bins,
            poss_ta=poss_ta,
            poss_tb=poss_tb
        )


class SamplingCore(CPSample):
    """
    Alias-style façade for the full cosine-phase sampling pipeline.

    Why this class exists
    ---------------------
    Some codebases prefer a more generic name than `CPSample`. This subclass
    provides that name without changing behavior.

    Important
    ---------
    No logic is added here.
    All behavior comes from `CPSample`.
    """

    pass


# ======================================================================
# Backward-compatible re-exports
# ======================================================================

__all__ = [
    # Main façade classes
    "CosinePhaseCore",
    "CPSample",
    "SamplingCore",

    # Extracted classes
    "PhaseCoefficientCore",
    "FourierCoefficientCore",
    "ReconstructionCore",
    "QuantizationCore",
    "Nyquist",

    # Scientific helper functions
    "calc_ta_tb",
    "calc_an_bn_dft",
    "cpm_sample",
    "recover_signal",
    "quantize",
    "quantize_ta_tb",

    # Filter / spectral helpers
    "sinc_filter",
    "plot_fft",
    "filter_periodic",

    # Legacy delta / threshold helpers
    "print_new_samples",
    "plot_threshold",
    "plot_delta",
    "delta",
    "print_new_samples_delta",
]
