import numpy as np
import matplotlib.pyplot as plt

from sfc.core.filters import filter_periodic
from sfc.core.fourier import FourierCoefficientCore
from sfc.core.phase_cof import calc_ta_tb
from sfc.core.quantization import quantize_ta_tb
from sfc.core.reconstruction import recover_signal
from sfc.core.theory import compute_q, rbcp_mse_upper_bound, rbcp_mse_star

# Parameters
T = 1.0
Tt = 0.01
N = 5
M_RbCP = 8
peak_to_peak = 8.0

w0 = 2 * np.pi / T
W = 2 * N / T
n_vec = np.arange(1, N + 1)
t = np.arange(0, T, Tt)

# Raw signal
rng = np.random.default_rng(12345)
x_raw = rng.uniform(-1, 1, size=len(t))   # or rng.normal(0, 1, size=len(t))

# Band-limit
x_filtered = filter_periodic(x_raw, W, Tt, T)

dc_value = Tt * np.sum(x_filtered) / T
x_filtered = x_filtered - dc_value

# Adjust peak-to-peak AFTER filtering
current_p2p = np.max(x_filtered) - np.min(x_filtered)
if current_p2p != 0:
    x_filtered = x_filtered * (peak_to_peak / current_p2p)

print("Peak-to-peak filtered =", np.max(x_filtered) - np.min(x_filtered))

# Fourier with normalization logic
fourier_core = FourierCoefficientCore(T=T, harmonics=N, sensor_nodes=1, dft_signal_periods=1)
an, bn, x_used = fourier_core.calc_an_bn_dft(
    x_filtered,
    Tt,
    normalize=True,
    norm=3.9
)

x_used_1d = x_used[:, 0, 0]

print("max(an^2 + bn^2) =", np.max(an[0, :, 0]**2 + bn[0, :, 0]**2))
print("Peak-to-peak used =", np.max(x_used_1d) - np.min(x_used_1d))

# ta/tb via complex log only
ta, tb = calc_ta_tb(an[0, :, 0], bn[0, :, 0], n_vec, w0)
ta = np.real(ta)
tb = np.real(tb)

# Quantization
ta_q, tb_q = quantize_ta_tb(ta, tb, w0, M_RbCP)

# Reconstruction
x_r = recover_signal(ta, tb, t, w0)
x_hat = recover_signal(ta_q, tb_q, t, w0)

# MSE
mse = np.mean((x_used_1d - x_hat)**2)

# Theory
Q = compute_q(M_RbCP)
mse_up = rbcp_mse_upper_bound(N, Q)
mse_st = rbcp_mse_star(N, Q)

print("ta =", ta)
print("tb =", tb)
print("ta_q =", ta_q)
print("tb_q =", tb_q)
print("Q =", Q)
print("MSE =", mse)
print("MSE* =", mse_st)
print("Upper bound =", mse_up)

# Plot signals
plt.figure(figsize=(11, 5))
plt.plot(t, x_raw, label="x_raw", alpha=0.4)
plt.plot(t, x_filtered, label="x_filtered (p2p adjusted)", linewidth=2)
plt.plot(t, x_used_1d, label="x_used (after DFT normalization)", linewidth=2)
plt.plot(t, x_hat, label="x_hat (reconstructed)", linestyle="--", linewidth=2)
plt.plot(t, x_r, label="x_r (reconstructed)", linestyle="--", linewidth=2)
plt.xlabel(r"$t$")
plt.ylabel("Amplitude")
plt.title(f"RbCP single run | N={N}, M_RbCP={M_RbCP}, MSE={mse:.3e}")
plt.grid(True)
plt.legend()
plt.show()

# Plot reconstruction error
plt.figure(figsize=(11, 4))
plt.plot(t, x_used_1d - x_hat, color="red")
plt.xlabel(r"$t$")
plt.ylabel(r"$x_{\mathrm{used}} - \hat{x}$")
plt.title("Reconstruction error")
plt.grid(True)
plt.show()

# Plot ta/tb before and after quantization
plt.figure(figsize=(9, 4))
plt.plot(n_vec, ta, "o-", label="ta")
plt.plot(n_vec, ta_q, "s--", label="ta_q")
plt.plot(n_vec, tb, "o-", label="tb")
plt.plot(n_vec, tb_q, "s--", label="tb_q")
plt.xlabel("Harmonic index")
plt.ylabel("Phase parameter")
plt.title("Phase parameters before/after quantization")
plt.grid(True)
plt.legend()
plt.show()