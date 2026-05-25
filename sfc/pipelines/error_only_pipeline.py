# sfc/pipelines/error_only_pipeline.py

"""
Error-only pipeline: computes event error rate across parameter sweeps.
"""

from __future__ import annotations

import numpy as np

from sfc import channel as sfc_c
from sfc.sampling import CPSample, filter_periodic
from sfc.config import load_config
from sfc.params import build_error_only_params
from sfc.io import save_dat_with_metadata


def run(config_path: str, *, save: bool = True):
    """
    Run the error-only experiment.

    Returns
    -------
    dict
        Contains 'results' array with columns [bw_channel, snr_dB, event_error_rate].
    """
    cfg = load_config(config_path)
    params = build_error_only_params(cfg)

    bw_channel_lim = params["bw_channel_lim"]
    n_periods = params["n_periods"]
    n_sensors = params["n_sensors"]

    n_resource = params["n_resource"]
    n_sub_symbol = params["n_sub_symbol"]

    snr_dB_lim = params["snr_dB_lim"]
    average_power = params["average_power"]

    W = params["W"]
    T = params["T"]
    NN = params["NN"]
    Tt = params["Tt"]
    p2p = params["p2p"]

    results = []

    for bw_channel in bw_channel_lim:
        for snr_dB in snr_dB_lim:

            N0 = average_power / ((10 ** (snr_dB / 10)) * bw_channel / n_sensors)

            sfc_tx_amplitude = np.sqrt(
                T * average_power * bw_channel / (4 * n_sub_symbol * n_resource * NN)
            ) * np.ones((n_sensors, 1))

            t = np.arange(n_periods * T / Tt) * Tt - T * n_periods / 2
            t_1p = np.where(np.logical_and(t >= -T / 2, t < T / 2))

            x = np.random.rand(len(t_1p[0]), n_periods, n_sensors) - 0.5
            for ind2 in range(x.shape[2]):
                for ind1 in range(x.shape[1]):
                    x[:, ind1, ind2] -= np.mean(x[:, ind1, ind2])
                    x[:, ind1, ind2] = p2p * (
                        filter_periodic(x[:, ind1, ind2], W, Tt, T)
                    ) / (x[:, ind1, ind2].max() - x[:, ind1, ind2].min())

            s = CPSample(
                T=T,
                harmonics=NN,
                n_sub_symbol=n_sub_symbol,
                resource=n_resource,
                sensor_nodes=n_sensors,
                bandwidth=bw_channel,
                detect_errors=True
            )

            ch = sfc_c.SFCChannel(
                sensor_nodes=n_sensors,
                resource=n_resource,
                n_sub_symbol=n_sub_symbol,
                sensor_x_event=s.sensors_x_event,
                N0=N0,
                tx_amplitude=sfc_tx_amplitude,
                power_at_receiver=True,
                bandwidth=bw_channel
            )

            # Sampling and event generation
            _ta, _tb, x = s.sample(x, Tt, t=t[t_1p])
            events = s(x, Tt, t=t[t_1p])

            # Channel transmission
            rx_events, _rx_map, _received_signal = ch(events)

            # Event error rate
            event_error_rate = np.sum(events != rx_events) / events.size
            results.append([bw_channel, snr_dB, event_error_rate])

            print(f"[INFO] bw={bw_channel}, snr={snr_dB}, error={event_error_rate:.6f}")

    results = np.array(results, dtype=float)

    if save:
        save_dat_with_metadata(
            results,
            experiment="error_only",
            config=cfg,
            header="bw_channel\tsnr_dB\tevent_error_rate",
            fmt="%.8f"
        )

    return {"results": results, "config": cfg}
