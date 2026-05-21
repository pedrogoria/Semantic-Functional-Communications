import numpy as np


def build_main_params(cfg: dict) -> dict:
    sim = cfg["simulation"]
    sfc = cfg["sfc_channel"]
    trad = cfg["traditional"]
    sig = cfg["signal"]
    samp = cfg["sampling"]

    params = {}

    # simulation parameters
    params["bw_channel"] = sim["bw_channel"]
    params["n_periods"] = int(sim["n_periods"])
    params["n_sensors"] = int(sim["n_sensors"])

    # semantic-functional channel parameters
    params["n_resource"] = int(sfc["n_resource"])
    params["n_sub_symbol"] = int(sfc["n_sub_symbol"])
    params["detect_errors"] = bool(sfc["detect_errors"])

    # nyquist sampling and traditional communication
    params["snr_dB"] = trad["snr_dB"]
    params["average_power"] = trad["average_power"]

    # other parameters
    params["bw_signal"] = sig["bw_signal"]
    params["T"] = sig["T"]
    params["NN"] = int(sig["NN"])
    params["Tt"] = sig["Tt"]
    params["p2p"] = sig["p2p"]

    params["W"] = params["bw_signal"] * 2
    params["sampling_rate"] = samp["nyquist_factor"] * params["W"]
    params["w0"] = 2 * np.pi / params["T"]

    # xi = pi*bw_signal/w0 - floor(pi*bw_signal/w0)
    aux = np.pi * params["bw_signal"] / params["w0"]
    params["xi"] = aux - np.floor(aux)

    return params
