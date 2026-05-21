import argparse
import numpy as np
import datetime

# Project imports
from sfc.sampling import *
from sfc.config import load_config
from sfc.paths import RESULTS_DIR


def parse_args():
    """
    Parse command-line arguments.
    """
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default="experiments/configs/rbcp.yaml",
        help="Path to configuration file"
    )
    return p.parse_args()


def save_results(results, prefix="rbcp", config=None):
    """
    Save results and metadata in structured directories.
    """

    experiment_dir = RESULTS_DIR / prefix
    experiment_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    data_file = experiment_dir / f"{prefix}_{timestamp}.dat"
    meta_file = experiment_dir / f"{prefix}_{timestamp}_meta.txt"

    np.savetxt(
        data_file,
        results,
        fmt="%.8f",
        delimiter="\t",
        header='upBound_MSE, lowBound_MSE, MSE_rbcp, MSE, MSExy, Mc, Bins, M, bw, sensors, snr dB, W, sampling_rate, N, tau, p2p, mean_p2p',
        comments=""
    )

    if config is not None:
        with open(meta_file, "w") as f:
            f.write(str(config))

    print(f"[INFO] Results saved to: {data_file}")


def main(config_path=None):
    """
    RbCP simulation with hybrid config-driven parameters.
    """

    # --------------------------------------------------
    # Load configuration
    # --------------------------------------------------
    if config_path is None:
        args = parse_args()
        config_path = args.config

    cfg = load_config(config_path)

    # --------------------------------------------------
    # Config-driven parameters
    # --------------------------------------------------
    sim_cfg = cfg["simulation"]
    sig_cfg = cfg["signal"]
    samp_cfg = cfg["sampling"]
    comm_cfg = cfg["communication"]
    time_cfg = cfg["time"]

    bw_channel = sim_cfg["bw_channel"]
    n_sensors = sim_cfg["n_sensors"]
    n_periods = sim_cfg["n_periods"]
    interactions = sim_cfg["interactions"]

    snr_dB = comm_cfg["snr_dB"]

    bw_signal_lim = sig_cfg["bw_signal_lim"]

    p2p_lim = np.arange(
        samp_cfg["p2p_lim_start"],
        samp_cfg["p2p_lim_stop"],
        samp_cfg["p2p_lim_step"]
    )

    T = time_cfg["T"]
    Tt = time_cfg["Tt"]

    # Derived fixed parameter
    w0 = 2 * np.pi / T

    # --------------------------------------------------
    # Initialize results
    # --------------------------------------------------
    all_results = []

    # --------------------------------------------------
    # Main loops
    # --------------------------------------------------
    for bw_signal in bw_signal_lim:

        results = np.zeros((len(p2p_lim), 17))
        indx_res = 0

        for p2p in p2p_lim:

            MSE_rbcp = np.zeros(interactions)
            MSE = np.zeros(interactions)
            mean_p2p = np.zeros(interactions)

            # ------------------------------------------
            # Derived model parameters (DO NOT move to config)
            # ------------------------------------------
            W = 2 * bw_signal
            NN_local = int(np.floor(np.pi * W / w0))
            sampling_rate = 1.2 * W

            N_sample = Nyquist(
                T=T,
                Tt=Tt,
                sampling_rate=sampling_rate,
                sensor_nodes=n_sensors,
                bandwidth=bw_channel,
                snr_dB=snr_dB
            )

            M = N_sample.bins
            Mc = np.floor(M ** (T * sampling_rate / (2 * NN_local)))

            # ------------------------------------------
            # Time vector
            # ------------------------------------------
            t = np.arange(n_periods * T / Tt) * Tt - T * n_periods / 2
            t_1p = np.where(np.logical_and(t >= -T / 2, t < T / 2))

            # ------------------------------------------
            # Monte Carlo loop
            # ------------------------------------------
            for interac in range(interactions):

                s = CPSample(
                    T=T,
                    harmonics=NN_local,
                    sensor_nodes=n_sensors,
                    bandwidth=bw_channel
                )

                x = np.random.rand(len(t_1p[0]), n_periods, n_sensors) - 0.5

                # Normalize signals
                for ind2 in range(x.shape[2]):
                    for ind1 in range(x.shape[1]):
                        x[:, ind1, ind2] -= np.mean(x[:, ind1, ind2])
                        x[:, ind1, ind2] = p2p * (
                            filter_periodic(x[:, ind1, ind2], W, Tt, T)
                        ) / (
                            x[:, ind1, ind2].max() - x[:, ind1, ind2].min()
                        )

                ta, tb, x = s.sample(x, Tt, t=t)

                # Stability loop
                cont_tr = 0
                while (np.max(np.imag(ta)) > 0.001 or np.max(np.imag(tb)) > 0.001) and cont_tr < 100:
                    cont_tr += 1
                    x = np.random.rand(len(t_1p[0]), n_periods, n_sensors) - 0.5

                    for ind2 in range(x.shape[2]):
                        for ind1 in range(x.shape[1]):
                            x[:, ind1, ind2] -= np.mean(x[:, ind1, ind2])
                            x[:, ind1, ind2] = p2p * (
                                filter_periodic(x[:, ind1, ind2], W, Tt, T)
                            ) / (
                                x[:, ind1, ind2].max() - x[:, ind1, ind2].min()
                            )

                    x *= (100 - cont_tr) / 100
                    ta, tb, x = s.sample(x, Tt, t=t)

                ta = np.real(ta)
                tb = np.real(tb)

                # ------------------------------------------
                # RbCP reconstruction
                # ------------------------------------------
                ta_q, tb_q = quantize_ta_tb(ta, tb, w0, Mc)
                xs = s.recover_signal(ta_q, tb_q, t[t_1p])

                # Nyquist reference
                y = N_sample(x, t, quantize=True, peak2peak=p2p)
                yr = N_sample.recover_signal(y)

                # Errors
                MSE_rbcp[interac] = np.mean((x - xs) ** 2)
                MSE[interac] = np.mean((x - yr) ** 2)
                mean_p2p[interac] = p2p * (100 - cont_tr) / 100

            # ------------------------------------------
            # Theoretical bounds
            # ------------------------------------------
            Q = (Mc / (2 * np.pi)) * np.sin(np.pi / Mc)
            upBound_MSE = 4 * NN_local * (0.5 - Q) * (1.5 - Q)
            lowBound_MSE = 1 - 4 * (Q ** 2)

            results[indx_res, :] = np.array([
                upBound_MSE,
                lowBound_MSE,
                np.mean(MSE_rbcp),
                np.mean(MSE),
                ((np.mean(mean_p2p)) ** 2) / (12 * M ** 2),
                Mc,
                N_sample.bins,
                M,
                bw_channel,
                n_sensors,
                snr_dB,
                W,
                sampling_rate,
                NN_local,
                T,
                p2p,
                np.mean(mean_p2p)
            ])

            print(f"[INFO] bw_signal={bw_signal}, p2p={p2p} finished")

            indx_res += 1

        all_results.append(results)

    all_results = np.vstack(all_results)

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------
    save_results(all_results, prefix="rbcp", config=cfg)


# Entry point
if __name__ == "__main__":
    main()