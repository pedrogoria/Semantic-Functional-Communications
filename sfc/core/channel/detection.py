"""
sfc/core/channel/detection.py

Detection block for SFC.

Role
----
Recover event-related map hypotheses from the final received SFC channel frame.

The detector has two stages:

1. Local candidate generation
   --------------------------
   The received frame is thresholded and scanned with L-row sliding windows.

   For each candidate start slot t0 and each reference map, a score is computed:

       score = sum(y_bin[t0:t0+L, :] * reference_map)

   Depending on detection_mode, this produces a candidate tensor:

       maps_candidates.shape = (event_slots_total, num_event_ids, L, R)

2. Optional global candidate selection
   -----------------------------------
   The local detector may produce false positives when multiple events start in
   the same symbol slot. This is expected: complete same-slot overlaps can create
   combinatorial map ambiguities.

   By default, this detector applies a global frame-fit selector. For each
   event ID, the global selector chooses either:

       - one candidate start slot, or
       - the empty decision, if allow_empty=True

   The selected set minimizes a global frame reconstruction objective:

       || observed_frame - reconstructed_frame ||^2

   with optional event penalty or residual-improvement rule.

Physical-scale convention
-------------------------
The SFC physical channel produces a matched-filter/resource-output frame:

    y = sqrt(E_chip) * superposed + n

where:

    E_chip = P tau / (2 N L)

and:

    n ~ CN(0, N0)

The detector therefore uses a physical threshold:

    threshold = threshold_factor * sqrt(E_chip)

unless an explicit threshold is provided in the YAML.

The detector does NOT generate noise. It also does NOT use SNR as an operational
noise parameter. SNR is retained only for diagnostics and channel-capacity
reporting.

YAML interface
--------------
Recommended default:

    channel:
      detection_mode: threshold
      threshold_factor: 0.5
      score_threshold: 4

      candidate_selection: global_frame_fit

      global:
        allow_empty: true
        strategy: local_only       # local_only | event_penalty | residual_rule
        restarts: 100
        max_iter: 80
        seed: 12345
        observed_mode: abs         # abs | real | binary
        event_penalty: 0.0
        residual_threshold: 0.0

To recover the old local-only behavior:

    channel:
      candidate_selection: none
"""

from __future__ import annotations

import numpy as np

from sfc.core.system_parameters import build_derived_system_parameters


class MapDetector:
    """
    Recover event-related map hypotheses from the final received channel frame.
    """

    def __init__(self, cfg):
        """
        Initialize detector configuration.

        Parameters
        ----------
        cfg : dict
            Configuration dictionary.
        """
        self.cfg = cfg
        det_cfg = cfg.get("channel", {})

        # ------------------------------------------------------------------
        # Local detection mode:
        # - strict
        # - loose
        # - threshold
        # ------------------------------------------------------------------
        self.mode = str(det_cfg.get("detection_mode", "threshold")).lower()

        valid_modes = {"strict", "loose", "threshold"}
        if self.mode not in valid_modes:
            raise ValueError(
                f"Invalid detection_mode: {self.mode}. "
                f"Expected one of {sorted(valid_modes)}."
            )

        # Optional score threshold in map-matching domain.
        # If absent, use L at runtime.
        self.score_threshold = det_cfg.get("score_threshold", None)

        if self.score_threshold is not None:
            self.score_threshold = int(self.score_threshold)
            if self.score_threshold < 1:
                raise ValueError("channel.score_threshold must be >= 1.")

        # ------------------------------------------------------------------
        # Global candidate-selection configuration.
        #
        # Default is global_frame_fit, with empty decision allowed.
        # Use candidate_selection="none" to recover old local-only behavior.
        # ------------------------------------------------------------------
        self.candidate_selection = str(
            det_cfg.get("candidate_selection", "global_frame_fit")
        ).lower()

        valid_candidate_selection = {
            "none",
            "local",
            "local_only",
            "global_frame_fit",
        }

        if self.candidate_selection not in valid_candidate_selection:
            raise ValueError(
                f"Invalid candidate_selection: {self.candidate_selection}. "
                f"Expected one of {sorted(valid_candidate_selection)}."
            )

        global_cfg = det_cfg.get("global", {})

        self.global_allow_empty = bool(global_cfg.get("allow_empty", True))

        self.global_strategy = str(
            global_cfg.get("strategy", "local_only")
        ).lower()

        valid_global_strategies = {
            "local_only",
            "event_penalty",
            "residual_rule",
        }

        if self.global_strategy not in valid_global_strategies:
            raise ValueError(
                f"Invalid global strategy: {self.global_strategy}. "
                f"Expected one of {sorted(valid_global_strategies)}."
            )

        self.global_restarts = int(global_cfg.get("restarts", 100))
        self.global_max_iter = int(global_cfg.get("max_iter", 80))

        self.global_seed = int(
            global_cfg.get(
                "seed",
                cfg.get("reproducibility", {}).get(
                    "seed",
                    cfg.get("monte_carlo", {}).get("seed", 12345),
                ),
            )
        )

        self.global_observed_mode = str(
            global_cfg.get("observed_mode", "abs")
        ).lower()

        valid_observed_modes = {"abs", "real", "binary"}

        if self.global_observed_mode not in valid_observed_modes:
            raise ValueError(
                f"Invalid global observed_mode: {self.global_observed_mode}. "
                f"Expected one of {sorted(valid_observed_modes)}."
            )

        self.global_event_penalty = float(
            global_cfg.get("event_penalty", 0.0)
        )

        self.global_residual_threshold = float(
            global_cfg.get("residual_threshold", 0.0)
        )

        if self.global_restarts < 1:
            raise ValueError("global.restarts must be >= 1.")

        if self.global_max_iter < 1:
            raise ValueError("global.max_iter must be >= 1.")

        if self.global_event_penalty < 0:
            raise ValueError("global.event_penalty must be nonnegative.")

        if self.global_residual_threshold < 0:
            raise ValueError("global.residual_threshold must be nonnegative.")

        # ------------------------------------------------------------------
        # Centralized physical/system-derived parameters.
        # ------------------------------------------------------------------
        self.params = build_derived_system_parameters(cfg)

        self.P = float(self.params.P)
        self.B = float(self.params.B)
        self.N0 = float(self.params.N0)
        self.tau = float(self.params.tau)
        self.L = int(self.params.L)
        self.R = int(self.params.R)

        # Classical wideband reference SNR, useful for diagnostics only.
        # It is not used as the detector's operational noise parameter.
        self.SNR = float(self.params.SNR)
        self.SNR_dB = float(self.params.SNR_dB)

        # ------------------------------------------------------------------
        # Number of harmonics N.
        # ------------------------------------------------------------------
        signal_cfg = cfg.get("signal", {})

        if signal_cfg.get("N_override", None) is not None:
            self.N = int(signal_cfg["N_override"])
        else:
            self.N = int(self.params.N)

        if self.N < 1:
            raise ValueError("MapDetector requires N >= 1.")

        if self.L < 1:
            raise ValueError("MapDetector requires L >= 1.")

        if self.R < 1:
            raise ValueError("MapDetector requires R >= 1.")

        if self.P <= 0:
            raise ValueError("MapDetector requires P > 0.")

        if self.tau <= 0:
            raise ValueError("MapDetector requires tau > 0.")

        if self.N0 <= 0:
            raise ValueError("MapDetector requires N0 > 0.")

        # ------------------------------------------------------------------
        # SFC matched-filter/resource-output energy normalization.
        # Same convention as physical_channel.py.
        #
        # Each sensor transmits:
        #     2N events per period
        #
        # Each event map has:
        #     L active chips
        #
        # Therefore:
        #     E_sensor = P * tau
        #     E_chip   = E_sensor / (2 * N * L)
        # ------------------------------------------------------------------
        self.E_sensor = self.P * self.tau
        self.num_events_per_sensor = 2 * self.N
        self.num_active_chips_per_sensor = self.num_events_per_sensor * self.L

        self.E_event = self.E_sensor / self.num_events_per_sensor
        self.E_chip = self.E_sensor / self.num_active_chips_per_sensor

        self.sfc_signal_level = float(np.sqrt(self.E_chip))

        if self.sfc_signal_level <= 0:
            raise ValueError("sfc_signal_level must be positive.")

        self.sfc_pulse_amplitude = float(
            np.sqrt((self.tau * self.P * self.B) / (4.0 * self.L * self.R * self.N))
        )

        # ------------------------------------------------------------------
        # Threshold configuration.
        #
        # If explicit threshold is not supplied, use:
        #
        #     threshold = threshold_factor * sqrt(E_chip)
        #
        # This is consistent with the physical matched-filter output:
        #
        #     y = sqrt(E_chip) * superposed + n.
        # ------------------------------------------------------------------
        self.threshold_factor = float(det_cfg.get("threshold_factor", 0.5))

        if self.threshold_factor < 0:
            raise ValueError("channel.threshold_factor must be nonnegative.")

        if "threshold" in det_cfg:
            self.threshold = float(det_cfg["threshold"])
            self.threshold_source = "explicit"
        else:
            self.threshold = float(self.threshold_factor * self.sfc_signal_level)
            self.threshold_source = "threshold_factor_times_sfc_signal_level"

        if self.threshold < 0:
            raise ValueError("channel.threshold must be nonnegative.")

        # Last-run diagnostics.
        self.last_global_objective = None
        self.last_global_active_events = None
        self.last_global_restarts_used = None
        self.last_global_exact_solution_found = None

    # =========================================================================
    # PUBLIC DETECTION METHOD
    # =========================================================================

    def detect(self, y, reference_maps=None):
        """
        Detect maps from the final received channel frame.

        Parameters
        ----------
        y : np.ndarray
            Final received channel frame with shape:

                (rx_slots_total, R)

            May be complex-valued.

        reference_maps : np.ndarray or None, optional
            Fixed codebook with shape:

                (num_event_ids, L, R)

        Returns
        -------
        np.ndarray
            If reference_maps is None:

                binary thresholded frame with shape:
                    (rx_slots_total, R)

            If reference_maps is provided:

                estimated per-event maps with shape:
                    (event_slots_total, num_event_ids, L, R)
        """
        y = np.asarray(y)

        if y.ndim != 2:
            raise ValueError("y must have shape (rx_slots_total, R).")

        if y.shape[1] != self.R:
            raise ValueError(
                f"y.shape[1] must equal R={self.R}, got {y.shape[1]}."
            )

        # ------------------------------------------------------------------
        # Threshold the magnitude of the complex matched-filter output.
        #
        # y is physical-scale. The threshold is physical-scale too.
        # ------------------------------------------------------------------
        y_bin = (np.abs(y) > self.threshold).astype(float)

        if reference_maps is None:
            return y_bin

        reference_maps = np.asarray(reference_maps, dtype=float)

        if reference_maps.ndim != 3:
            raise ValueError(
                "reference_maps must have shape (num_event_ids, L, R)."
            )

        num_event_ids, L, R = reference_maps.shape

        if L != self.L:
            raise ValueError(
                f"reference_maps.shape[1] must equal L={self.L}, got {L}."
            )

        if R != self.R:
            raise ValueError(
                f"reference_maps.shape[2] must equal R={self.R}, got {R}."
            )

        rx_slots_total = y_bin.shape[0]
        event_slots_total = rx_slots_total - L + 1

        if event_slots_total <= 0:
            raise ValueError(
                "Invalid dimensions: rx_slots_total - L + 1 must be positive."
            )

        # ------------------------------------------------------------------
        # Stage 1: local candidate generation.
        # ------------------------------------------------------------------
        maps_candidates = self._local_candidate_detection(
            y_bin=y_bin,
            reference_maps=reference_maps,
        )

        # ------------------------------------------------------------------
        # Stage 2: optional global candidate selection.
        # ------------------------------------------------------------------
        if self.candidate_selection in {"none", "local", "local_only"}:
            self.last_global_objective = None
            self.last_global_active_events = None
            self.last_global_restarts_used = 0
            self.last_global_exact_solution_found = False
            return maps_candidates

        if self.candidate_selection == "global_frame_fit":
            observed_frame = self._build_observed_frame(
                y=y,
                y_bin=y_bin,
            )

            maps_selected = self._global_frame_fit_selection(
                maps_candidates=maps_candidates,
                observed_frame=observed_frame,
                reference_maps=reference_maps,
            )

            return maps_selected

        raise ValueError(
            f"Unknown candidate_selection: {self.candidate_selection}"
        )

    # =========================================================================
    # LOCAL CANDIDATE DETECTION
    # =========================================================================

    def _local_candidate_detection(self, y_bin, reference_maps):
        """
        Generate local map candidates by sliding-window support matching.
        """
        num_event_ids, L, R = reference_maps.shape
        rx_slots_total = y_bin.shape[0]
        event_slots_total = rx_slots_total - L + 1

        score_threshold = (
            int(self.score_threshold)
            if self.score_threshold is not None
            else int(L)
        )

        maps_est = np.zeros(
            (event_slots_total, num_event_ids, L, R),
            dtype=float,
        )

        for t0 in range(event_slots_total):
            rec_win = y_bin[t0:t0 + L, :]

            for event_id in range(num_event_ids):
                ref = reference_maps[event_id]
                score = float(np.sum(rec_win * ref))

                if self.mode == "strict":
                    if score == L:
                        maps_est[t0, event_id] = ref

                elif self.mode == "loose":
                    if score > 0:
                        maps_est[t0, event_id] = ref

                elif self.mode == "threshold":
                    if score >= score_threshold:
                        maps_est[t0, event_id] = ref

                else:
                    raise ValueError(f"Unknown detection mode: {self.mode}")

        return maps_est

    # =========================================================================
    # GLOBAL FRAME-FIT SELECTION
    # =========================================================================

    def _build_observed_frame(self, y, y_bin):
        """
        Build the normalized observed frame used by the global selector.

        Modes
        -----
        abs:
            observed = abs(y) / sfc_signal_level

        real:
            observed = real(y / sfc_signal_level)

        binary:
            observed = y_bin

        Notes
        -----
        The reconstructed frame used by the selector is dimensionless because it
        is constructed from reference maps. Therefore, abs/real observed modes
        divide by sfc_signal_level to place the physical received frame back in
        the dimensionless map-superposition scale.
        """
        if self.global_observed_mode == "abs":
            return np.abs(y) / self.sfc_signal_level

        if self.global_observed_mode == "real":
            return np.real(y / self.sfc_signal_level)

        if self.global_observed_mode == "binary":
            return np.asarray(y_bin, dtype=float)

        raise ValueError(
            f"Unknown observed_mode: {self.global_observed_mode}"
        )

    @staticmethod
    def _candidate_slots_from_maps(maps_candidates):
        """
        Convert candidate map tensor into candidate slots per event ID.
        """
        event_slots_total, num_event_ids, _, _ = maps_candidates.shape

        candidate_slots = []

        for event_id in range(num_event_ids):
            active_per_slot = np.sum(
                maps_candidates[:, event_id, :, :],
                axis=(1, 2),
            )

            slots = np.where(active_per_slot > 0)[0].astype(int)
            candidate_slots.append(slots)

        return candidate_slots

    @staticmethod
    def _events_from_chosen(chosen, event_slots_total):
        """
        Convert chosen slot vector into event matrix.
        """
        num_event_ids = len(chosen)
        events = np.zeros((event_slots_total, num_event_ids), dtype=float)

        for event_id, t0 in enumerate(chosen):
            if t0 >= 0:
                events[t0, event_id] = 1.0

        return events

    @staticmethod
    def _maps_from_chosen(chosen, event_slots_total, reference_maps):
        """
        Convert chosen slot vector into maps_est tensor.
        """
        num_event_ids, L, R = reference_maps.shape

        maps_est = np.zeros(
            (event_slots_total, num_event_ids, L, R),
            dtype=float,
        )

        for event_id, t0 in enumerate(chosen):
            if t0 >= 0:
                maps_est[t0, event_id] = reference_maps[event_id]

        return maps_est

    @staticmethod
    def _frame_from_chosen(chosen, event_slots_total, reference_maps):
        """
        Build reconstructed frame from chosen event slots.
        """
        num_event_ids, L, R = reference_maps.shape
        frame = np.zeros((event_slots_total + L - 1, R), dtype=float)

        for event_id, t0 in enumerate(chosen):
            if t0 >= 0:
                frame[t0:t0 + L, :] += reference_maps[event_id]

        return frame

    def _objective(self, observed_frame, reconstructed_frame, active_count):
        """
        Compute global objective according to selected strategy.
        """
        residual_cost = float(np.sum((observed_frame - reconstructed_frame) ** 2))

        if self.global_strategy == "event_penalty":
            return residual_cost + self.global_event_penalty * float(active_count)

        return residual_cost

    def _global_frame_fit_selection(
        self,
        maps_candidates,
        observed_frame,
        reference_maps,
    ):
        """
        Select a physically consistent subset of local candidates.

        Decision variable
        -----------------
        For each event_id, choose either:

            - one candidate slot, or
            - empty decision, represented by -1, when allow_empty=True.

        Strategies
        ----------
        local_only:
            Minimize squared frame residual.

        event_penalty:
            Minimize squared frame residual + lambda * number_of_active_events.

        residual_rule:
            Same squared residual objective, but each coordinate update is only
            accepted if the improvement is larger than residual_threshold.
        """
        event_slots_total, num_event_ids, _, _ = maps_candidates.shape

        candidate_slots = self._candidate_slots_from_maps(maps_candidates)

        rng = np.random.default_rng(self.global_seed)

        best_chosen = None
        best_obj = np.inf
        best_active_count = 0
        exact_solution_found = False
        restarts_used = 0

        for restart in range(self.global_restarts):
            restarts_used = restart + 1

            chosen = np.full(num_event_ids, -1, dtype=int)

            # --------------------------------------------------------------
            # Initialization.
            # --------------------------------------------------------------
            for event_id in range(num_event_ids):
                slots = candidate_slots[event_id]

                if len(slots) == 0:
                    chosen[event_id] = -1
                    continue

                if restart == 0:
                    # Deterministic first initialization.
                    chosen[event_id] = int(slots[0])
                else:
                    if self.global_allow_empty:
                        choices = np.concatenate((np.array([-1], dtype=int), slots))
                        chosen[event_id] = int(rng.choice(choices))
                    else:
                        chosen[event_id] = int(rng.choice(slots))

            chosen, obj = self._local_improve_chosen(
                chosen=chosen,
                candidate_slots=candidate_slots,
                observed_frame=observed_frame,
                reference_maps=reference_maps,
                rng=rng,
            )

            active_count = int(np.sum(chosen >= 0))

            if obj < best_obj:
                best_obj = obj
                best_chosen = chosen.copy()
                best_active_count = active_count

            if best_obj < 1e-12:
                exact_solution_found = True
                break

        if best_chosen is None:
            best_chosen = np.full(num_event_ids, -1, dtype=int)
            best_obj = float(np.sum(observed_frame ** 2))
            best_active_count = 0

        maps_selected = self._maps_from_chosen(
            chosen=best_chosen,
            event_slots_total=event_slots_total,
            reference_maps=reference_maps,
        )

        self.last_global_objective = float(best_obj)
        self.last_global_active_events = int(best_active_count)
        self.last_global_restarts_used = int(restarts_used)
        self.last_global_exact_solution_found = bool(exact_solution_found)

        return maps_selected

    def _local_improve_chosen(
        self,
        chosen,
        candidate_slots,
        observed_frame,
        reference_maps,
        rng,
    ):
        """
        Coordinate-descent improvement of a chosen slot vector.

        The RNG is passed from _global_frame_fit_selection so that random
        restarts and coordinate orders do not reset to the same sequence on
        every restart.
        """
        num_event_ids, L, _ = reference_maps.shape
        event_slots_total = observed_frame.shape[0] - L + 1

        current_frame = self._frame_from_chosen(
            chosen=chosen,
            event_slots_total=event_slots_total,
            reference_maps=reference_maps,
        )

        current_active = int(np.sum(chosen >= 0))
        current_obj = self._objective(
            observed_frame=observed_frame,
            reconstructed_frame=current_frame,
            active_count=current_active,
        )

        for _ in range(self.global_max_iter):
            improved = False
            order = rng.permutation(num_event_ids)

            for event_id in order:
                slots = candidate_slots[event_id]

                if len(slots) == 0 and not self.global_allow_empty:
                    continue

                old_slot = int(chosen[event_id])

                trial_slot_options = slots

                if self.global_allow_empty:
                    trial_slot_options = np.concatenate(
                        (np.array([-1], dtype=int), slots)
                    )

                if len(trial_slot_options) <= 1:
                    continue

                frame_without = np.array(current_frame, copy=True)

                if old_slot >= 0:
                    frame_without[old_slot:old_slot + L, :] -= reference_maps[event_id]

                best_slot = old_slot
                best_obj = current_obj
                best_frame = current_frame

                for cand_slot in trial_slot_options:
                    cand_slot = int(cand_slot)

                    trial_frame = np.array(frame_without, copy=True)

                    if cand_slot >= 0:
                        trial_frame[cand_slot:cand_slot + L, :] += reference_maps[event_id]

                    trial_active = current_active

                    if old_slot >= 0:
                        trial_active -= 1

                    if cand_slot >= 0:
                        trial_active += 1

                    trial_obj = self._objective(
                        observed_frame=observed_frame,
                        reconstructed_frame=trial_frame,
                        active_count=trial_active,
                    )

                    improvement = current_obj - trial_obj

                    if self.global_strategy == "residual_rule":
                        accept = improvement > self.global_residual_threshold
                    else:
                        accept = improvement > 1e-12

                    if accept and trial_obj < best_obj:
                        best_obj = trial_obj
                        best_slot = cand_slot
                        best_frame = trial_frame

                if best_slot != old_slot:
                    if old_slot >= 0:
                        current_active -= 1

                    if best_slot >= 0:
                        current_active += 1

                    chosen[event_id] = best_slot
                    current_frame = best_frame
                    current_obj = best_obj
                    improved = True

            if not improved:
                break

            if current_obj < 1e-12:
                break

        return chosen, float(current_obj)

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    def diagnostics(self):
        """
        Return detector diagnostic quantities.
        """
        return {
            "mode": self.mode,
            "threshold": self.threshold,
            "threshold_source": self.threshold_source,
            "threshold_factor": self.threshold_factor,
            "score_threshold": self.score_threshold,
            "candidate_selection": self.candidate_selection,
            "global_allow_empty": self.global_allow_empty,
            "global_strategy": self.global_strategy,
            "global_restarts": self.global_restarts,
            "global_max_iter": self.global_max_iter,
            "global_seed": self.global_seed,
            "global_observed_mode": self.global_observed_mode,
            "global_event_penalty": self.global_event_penalty,
            "global_residual_threshold": self.global_residual_threshold,
            "last_global_objective": self.last_global_objective,
            "last_global_active_events": self.last_global_active_events,
            "last_global_restarts_used": self.last_global_restarts_used,
            "last_global_exact_solution_found": self.last_global_exact_solution_found,
            "P": self.P,
            "B": self.B,
            "N0": self.N0,
            "tau": self.tau,
            "R": self.R,
            "L": self.L,
            "N": self.N,
            "SNR_reference": self.SNR,
            "SNR_reference_dB": self.SNR_dB,
            "E_sensor": self.E_sensor,
            "num_events_per_sensor": self.num_events_per_sensor,
            "num_active_chips_per_sensor": self.num_active_chips_per_sensor,
            "E_event": self.E_event,
            "E_chip": self.E_chip,
            "sfc_signal_level": self.sfc_signal_level,
            "sfc_pulse_amplitude": self.sfc_pulse_amplitude,
        }


__all__ = [
    "MapDetector",
]
