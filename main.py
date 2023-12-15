import sfc_channel as sfc_c
import numpy as np
import random as rnd

N = 1000
event_rate = 0.1
n_sensors = 3
n_resource = 7
n_sub_symbol = 6

ch = sfc_c.BBChannel(sensor_nodes=n_sensors, resource=n_resource, n_sub_symbol=n_sub_symbol)
# events = np.array([rnd.randint(0, 1) for _ in range(0, N * n_sensors)]).reshape((-1, n_sensors))
events = np.array([1 if rnd.random() < event_rate else 0 for _ in range(0, N * n_sensors)]).reshape((-1, n_sensors))

rx_events = ch(events)

event_error_rate = np.sum(events != rx_events) / events.size
