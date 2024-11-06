import numpy as np

import sfc_channel as sfc_c
from sampling_methods import *

# simulation parameters
bw_channel = 2400  # total bandwidth for communicate
n_periods = int(1)  # number of signal periods considered in the simulation
n_sensors = 2  # number of signals, we assume that each sensor only monitors a single signal.

# semantic-functional channel parameters
n_resource = 12  # sub-band number assigned to the SF Channel.
n_sub_symbol = 4  # length (number of rows) of the SF matrix for the SF channel
detect_errors = True

# nyquist sampling and traditional communication
snr_dB = -15
average_power = 1  # average power transmission per sensor in watts

# other parameters
bw_signal = 5
W = bw_signal * 2
sampling_rate = 1.1 * W
T = 1  # window for sampling. It is the period assigned to signals
NN = 5  # int(bw_signal * T)  # number of harmonics, i.e., number of DFT terms
Tt = 0.01  # time increment
w0 = 2 * np.pi / T
xi = np.pi * bw_signal / w0 - np.floor(np.pi * bw_signal / w0)
p2p = 4

# preliminary calculations
N0 = average_power / ((10 ** (snr_dB / 10)) * bw_channel / n_sensors)
# N0c = (2 * n_sub_symbol * (np.mean(sf_tx_power) ** 2) * NN) / (bw_channel * (10 ** (snr_dB / 10)) * T)

sfc_tx_amplitude = np.sqrt(T * average_power * bw_channel / (4 * n_sub_symbol * n_resource * NN)) * np.ones((n_sensors, 1))
# sfc_tx_power = 10 * np.ones((n_sensors, 1))

# time vectors
t = np.arange(n_periods * T / Tt) * Tt - T * n_periods / 2
t_1p = np.where(np.logical_and(t >= -T / 2, t < T / 2))

# generates random signal and filters them
x_s1 = np.cos(2 * np.pi * 3 * t)  # + np.cos(2 * np.pi * 7 * t + np.pi / 5)
x_s1 = x_s1.reshape(-1, len(t_1p[0]))
x_s1 = np.transpose(x_s1)
# x = np.random.normal(0, 0.1, (len(t_1p[0]), n_periods, n_sensors))
x = np.random.rand(len(t_1p[0]), n_periods, n_sensors) - 0.5
# x = sinc_filter(x, Tt=Tt, BW=bw_signal)
for ind2 in range(x.shape[2]):
    for ind1 in range(x.shape[1]):
        x[:, ind1, ind2] = x[:, ind1, ind2] - np.mean(x[:, ind1, ind2])
        x[:, ind1, ind2] = p2p * (filter_periodic(x[:, ind1, ind2], W, Tt, T)) / (x[:, ind1, ind2].max() - x[:, ind1, ind2].min())
x[:, :, 0] = x_s1

# Temperature = pd.read_csv("Data/Temperature_oct_22_lapp.csv", dayfirst=True, sep=",",
#                           header=0, decimal=b".", index_col=0,
#                           parse_dates=[[0, 1, 2, 3]], usecols=[0, 1, 2, 3, 5])

s = CPSample(T=T, harmonics=NN, n_sub_symbol=n_sub_symbol, resource=n_resource, sensor_nodes=n_sensors, bandwidth=bw_channel,
             detect_errors=detect_errors)
ch = sfc_c.SFCChannel(sensor_nodes=n_sensors, resource=n_resource, n_sub_symbol=n_sub_symbol, sensor_x_event=s.sensors_x_event,
                      N0=N0, tx_amplitude=sfc_tx_amplitude, power_at_receiver=True, bandwidth=bw_channel)
N_sample = Nyquist(T=T, Tt=Tt, sampling_rate=sampling_rate, sensor_nodes=n_sensors, bandwidth=bw_channel, snr_dB=snr_dB)
# events = np.array([rnd.randint(0, 1) for _ in range(0, N * n_sensors)]).reshape((-1, n_sensors))
# events = np.array([1 if rnd.random() < event_rate else 0 for _ in range(0, N * n_sensors * NN)]).reshape((-1, n_sensors * NN))

# Mc = np.floor(N_sample.bins ** ((np.pi * bw_signal) / (np.pi * bw_signal - xi * w0)))
Mc = np.floor((1 + (10 ** (snr_dB / 10))) ** ((T * bw_channel) / (2 * NN * n_sensors)))
ta, tb, x = s.sample(x, Tt, t=t[t_1p])

cont_tr = 0
while (np.max(np.imag(ta)) > 0.001 or np.max(np.imag(tb)) > 0.001) and cont_tr < 100:
    cont_tr = cont_tr + 1
    x = np.random.rand(len(t_1p[0]), n_periods, n_sensors) - 0.5
    # x = sinc_filter(x, Tt=Tt, BW=bw_signal)
    for ind2 in range(x.shape[2]):
        for ind1 in range(x.shape[1]):
            x[:, ind1, ind2] = x[:, ind1, ind2] - np.mean(x[:, ind1, ind2])
            x[:, ind1, ind2] = p2p * (filter_periodic(x[:, ind1, ind2], W, Tt, T)) / (x[:, ind1, ind2].max() - x[:, ind1, ind2].min())

    x = x * (100 - cont_tr) / 100
    ta, tb, x = s.sample(x, Tt, t=t)

ta = np.real(ta)
tb = np.real(tb)
events = s(x, Tt, t=t[t_1p])
rx_events, rx_map, received_signal = ch(events)
#
if detect_errors:
    ta_time, tb_time, _ = s.event_to_ta_tb(events)
    ta_cbcp, tb_cbcp = quantize_ta_tb(ta, tb, w0, Mc)
    ta_sfc, tb_sfc, error_sfc = s.event_to_ta_tb(rx_events)
    x_time = s.recover_signal(ta_time, tb_time, t[t_1p])
    x_cbcp = s.recover_signal(ta_cbcp, tb_cbcp, t[t_1p])
    x_sfc = s.recover_signal(ta_sfc, tb_sfc, t[t_1p])
    indx_error = np.argwhere(error_sfc == 1)
    for i in range(indx_error.shape[0]):
        x_sfc[:, indx_error[i, 0], indx_error[i, 1]] = x_time[:, indx_error[i, 0], indx_error[i, 1]]
else:
    ta_time, tb_time = s.event_to_ta_tb(events)
    ta_cbcp, tb_cbcp = quantize_ta_tb(ta, tb, w0, Mc)
    ta_sfc, tb_sfc = s.event_to_ta_tb(rx_events)
    x_time = s.recover_signal(ta_time, tb_time, t[t_1p])
    x_cbcp = s.recover_signal(ta_cbcp, tb_cbcp, t[t_1p])
    x_sfc = s.recover_signal(ta_sfc, tb_sfc, t[t_1p])

y = N_sample(x, t[t_1p], quantize=True)
yr = N_sample.recover_signal(y)

x_perf = s.recover_signal(ta, tb, t[t_1p])

signal_plt = 1
fig0, plot_sfc = plt.subplots()
plt.title('SFC with cosine phase sampling')
plot_sfc.plot(t[t_1p], x[:, 0, signal_plt], label='Original')
plot_sfc.plot(t[t_1p], x_cbcp[:, 0, signal_plt], label='CbCP')
# plot_sfc.plot(t[t_1p], x_perf[:, 0, signal_plt], label='Perf')
# plot_sfc.plot(t[t_1p], x_sfc[:, 0, signal_plt], label='SFC')
# plot_sfc.plot(t[t_1p], x_time[:, 0, 0], label='CbCP_{time}')
legend0 = plot_sfc.legend(loc='upper right')
plt.xlabel('time (s)')
plt.show()

fig1, plot_nyquist = plt.subplots()
plt.title('Traditional communication with Nyquist sampling')
plot_nyquist.plot(t[t_1p], x[:, 0, signal_plt], label='Original')
plot_nyquist.plot(t[t_1p], yr[:, 0, signal_plt], label='Nyquist')
legend1 = plot_nyquist.legend(loc='upper right')
plt.xlabel('time (s)')
plt.show()

# event_error_rate = np.sum(events != rx_events) / events.size
# print("event error rate: {}".format(event_error_rate))

# overlapping_tx = np.sum(np.sum(events, 1) > 1)
# print("number of fully overlapping transmissions: {}".format(overlapping_tx))

# if detect_errors:
#     indx_no_error = np.argwhere(error_sfc == 0)
#     MSE_sfc = np.zeros(indx_no_error.shape[0])
#     for i in range(indx_no_error.shape[0]):
#         MSE_sfc[i] = np.mean((x[:, indx_no_error[i, 0], indx_no_error[i, 1]]-x_sfc[:, indx_no_error[i, 0], indx_no_error[i, 1]]) ** 2)
#     MSE_sfc = np.mean(MSE_sfc)
# else:
#     MSE_sfc = np.mean((x - x_sfc) ** 2)
# MSE_cbcp = np.mean((x - x_cbcp) ** 2)
# MSE_time = np.mean((x - x_time) ** 2)
# MSE = np.mean((x - yr) ** 2)
# print("MSE of sfc: {}".format(MSE_sfc))
# print("MSE of time: {}".format(MSE_time))
# print("MSE of CbCP: {}".format(MSE_cbcp))
# print("MSE: {}".format(MSE))

a = np.concatenate((t.reshape(-1, 1), x[:, 0, signal_plt].reshape(-1, 1).reshape(-1, 1), x_cbcp[:, 0, signal_plt].reshape(-1, 1).reshape(-1, 1),
                    yr[:, 0, signal_plt].reshape(-1, 1).reshape(-1, 1)), axis=1)
np.savetxt('signals_comp.dat', a, header='time, original, CbCP sampling, Nyquist')
