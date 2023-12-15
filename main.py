import sfc_channel as sfc_c
from sampling_methods import *

# simulation parameters
n_periods = int(2)  # number of signal periods considered in the simulation
n_sensors = 2  # number of signals, we assume that each sensor only monitors a single signal.
# semantic-functional channel parameters
n_resource = 7  # subband number assigned to the SF Channel.
n_sub_symbol = 6  # length (number of rows) of the SF matrix for the SF channel
AWGN_std = 0  # noise
bw_channel = 1000  # total bandwidth for communicate
# other parameters
bw_signal = 10
sampling_rate = 10
T = 0.5  # window for sampling. It is the period assigned to signals
NN = 2  # int(bw_signal * T)  # number of harmonics, i.e., number of DFT terms
Tt = 0.001  # time increment

# time vectors
t = np.arange(-T * n_periods / 2, T * n_periods / 2, Tt)
t_1p = np.where(np.logical_and(t >= 0, t < T))

# generates random signal and filters them
x_s1 = np.cos(2 * np.pi * 2 * t) + np.cos(2 * np.pi * 4 * t + np.pi / 5)
x_s1 = x_s1.reshape(-1, len(t_1p[0]))
x_s1 = np.transpose(x_s1)
x = np.random.normal(0, 0.1, (len(t_1p[0]), n_periods, n_sensors))
x = sinc_filter(x, BW=bw_signal)
x[:, :, 0] = x_s1

# Temperature = pd.read_csv("Data/Temperature_oct_22_lapp.csv", dayfirst=True, sep=",",
#                           header=0, decimal=b".", index_col=0,
#                           parse_dates=[[0, 1, 2, 3]], usecols=[0, 1, 2, 3, 5])

s = CPMSample(T=T, harmonics=NN, n_sub_symbol=n_sub_symbol, resource=n_resource, sensor_nodes=n_sensors, bandwidth=bw_channel)
ch = sfc_c.SFCChannel(sensor_nodes=2 * n_sensors * s.harmonics, resource=n_resource, n_sub_symbol=n_sub_symbol, sensor_x_event=s.sensors_x_event,
                      AWGN_std=AWGN_std)
# events = np.array([rnd.randint(0, 1) for _ in range(0, N * n_sensors)]).reshape((-1, n_sensors))
# events = np.array([1 if rnd.random() < event_rate else 0 for _ in range(0, N * n_sensors * NN)]).reshape((-1, n_sensors * NN))

ta, tb, x = s.sample(x, Tt)
xs = s.recover_signal(ta, tb, t[t_1p])
events = s(x, Tt)
rx_events, rx_map, received_signal = ch(events)
#
r_ta, r_tb = s.event_to_ta_tb(rx_events)
xr = s.recover_signal(r_ta, r_tb, t[t_1p])

N_sample = Nyquist(T=T, Tt=Tt, sampling_rate=sampling_rate)
y = N_sample(x, t)
yr = N_sample.recover_signal(y)

fig, ax = plt.subplots()
# ax.title('A sine wave with a gap of NaNs between 0.4 and 0.6')
ax.plot(t[t_1p], x[:, 0, 0], label='Original')
ax.plot(t[t_1p], xs[:, 0, 0], label='Proposed sampling')
ax.plot(t[t_1p], xr[:, 0, 0], label='Recovered after transmission')
ax.plot(t[t_1p], yr[:, 0, 0], label='Nyquist')
legend = ax.legend(loc='upper right')
plt.xlabel('time (s)')
plt.show()

event_error_rate = np.sum(events != rx_events) / events.size
print("event error rate: {}".format(event_error_rate))


