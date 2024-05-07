import numpy as np

import sfc_channel as sfc_c
from sampling_methods import *

# simulation parameters
bw_channel = 1e5  # total bandwidth for communicate
n_periods = int(2)  # number of signal periods considered in the simulation
n_sensors = 8  # number of signals, we assume that each sensor only monitors a single signal.

# semantic-functional channel parameters
n_resource = 9  # subband number assigned to the SF Channel.
n_sub_symbol = 5  # length (number of rows) of the SF matrix for the SF channel

# nyquist sampling and traditional communication
snr_dB = -10
average_power = 1  # average power transmission per sensor in watts

# other parameters
bw_signal = 10
sampling_rate = 20
T = 0.5  # window for sampling. It is the period assigned to signals
NN = 1  # int(bw_signal * T)  # number of harmonics, i.e., number of DFT terms
Tt = 0.01  # time increment

# preliminary calculations
N0 = average_power / ((10 ** (snr_dB / 10)) * bw_channel / n_sensors)
# N0c = (2 * n_sub_symbol * (np.mean(sf_tx_power) ** 2) * NN) / (bw_channel * (10 ** (snr_dB / 10)) * T)
AWGN_std = np.sqrt(N0 / 2)  # noise
# AWGN_std = 0
sfc_tx_power = np.sqrt((average_power * T / (2 * n_sub_symbol * NN))) * np.ones((n_sensors, 1))
# sfc_tx_power = 10 * np.ones((n_sensors, 1))

# time vectors
t = np.arange(-T * n_periods / 2, T * n_periods / 2, Tt)
t_1p = np.where(np.logical_and(t >= 0, t < T))

# generates random signal and filters them
x_s1 = np.cos(2 * np.pi * 2 * t) + np.cos(2 * np.pi * 4 * t + np.pi / 5)
x_s1 = x_s1.reshape(-1, len(t_1p[0]))
x_s1 = np.transpose(x_s1)
x = np.random.normal(0, 0.1, (len(t_1p[0]), n_periods, n_sensors))
x = sinc_filter(x, Tt=Tt, BW=bw_signal)
for ind2 in range(x.shape[2]):
    for ind1 in range(x.shape[1]):
        x[:, ind1, ind2] = x[:, ind1, ind2] - np.mean(x[:, ind1, ind2])
x[:, :, 0] = x_s1

# Temperature = pd.read_csv("Data/Temperature_oct_22_lapp.csv", dayfirst=True, sep=",",
#                           header=0, decimal=b".", index_col=0,
#                           parse_dates=[[0, 1, 2, 3]], usecols=[0, 1, 2, 3, 5])

s = CPSample(T=T, harmonics=NN, n_sub_symbol=n_sub_symbol, resource=n_resource, sensor_nodes=n_sensors, bandwidth=bw_channel)
ch = sfc_c.SFCChannel(sensor_nodes=n_sensors, resource=n_resource, n_sub_symbol=n_sub_symbol, sensor_x_event=s.sensors_x_event,
                      AWGN_std=AWGN_std, sn_tx_power=sfc_tx_power, power_at_receiver=True, bandwidth=bw_channel)
N_sample = Nyquist(T=T, Tt=Tt, sampling_rate=sampling_rate, sensor_nodes=n_sensors, bandwidth=bw_channel, snr_dB=snr_dB)
# events = np.array([rnd.randint(0, 1) for _ in range(0, N * n_sensors)]).reshape((-1, n_sensors))
# events = np.array([1 if rnd.random() < event_rate else 0 for _ in range(0, N * n_sensors * NN)]).reshape((-1, n_sensors * NN))

_, _, x = s.sample(x, Tt)
events = s(x, Tt)
rx_events, rx_map, received_signal = ch(events)
#
ta, tb = s.event_to_ta_tb(events)
xs = s.recover_signal(ta, tb, t[t_1p])
r_ta, r_tb = s.event_to_ta_tb(rx_events)
xr = s.recover_signal(r_ta, r_tb, t[t_1p])

y = N_sample(x, t)
yr = N_sample.recover_signal(y)

signal_plt = 1
fig0, plot_sfc = plt.subplots()
plt.title('SFC with cosine phase sampling')
plot_sfc.plot(t[t_1p], x[:, 0, signal_plt], label='Original')
plot_sfc.plot(t[t_1p], xs[:, 0, signal_plt], label='Proposed sampling')
plot_sfc.plot(t[t_1p], xr[:, 0, signal_plt], label='Recovered after transmission')
# plot_sfc.plot(t[t_1p], yr[:, 0, 0], label='Nyquist')
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

event_error_rate = np.sum(events != rx_events) / events.size
print("event error rate: {}".format(event_error_rate))


