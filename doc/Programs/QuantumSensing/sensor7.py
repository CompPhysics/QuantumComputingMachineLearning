import numpy as np
import matplotlib.pyplot as plt

# -------------------- Prior Parameter Grid --------------------
B0_vals = np.linspace(0.5, 1.1, 60)
omega_vals = np.linspace(0.7, 1.3, 60)
phi0_vals = np.linspace(-np.pi, np.pi, 60)

# -------------------- Simulation Parameters --------------------
n_qubits = 3
gamma = 1.0
T2 = 3.0
sigma_noise = 0.05
shots = 100
times = np.linspace(0.1, 5, 100)

# True values of the magnetic field
B0_true = 0.8
omega_true = 1.0
phi0_true = 0.0

# -------------------- Field and Measurement Functions --------------------
def B_field(t, B0, omega):
    return B0 * np.sin(omega * t)

def ideal_expectation(t, B0, omega, phi0):
    phase = gamma * n_qubits * B0 * np.sin(omega * t) * t + phi0
    return np.cos(phase)

def decohered_expectation(t, B0, omega, phi0, T2):
    return ideal_expectation(t, B0, omega, phi0) * np.exp(-t / T2)

def noisy_measurement(expectation, shots, sigma):
    p = (1 + expectation) / 2
    counts = np.random.binomial(shots, p)
    mean = 2 * (counts / shots) - 1
    return mean + np.random.normal(0, sigma)

# -------------------- Generate Simulated Measurements --------------------
measured_expectations = []
for t in times:
    exp_t = decohered_expectation(t, B0_true, omega_true, phi0_true, T2)
    meas_t = noisy_measurement(exp_t, shots, sigma_noise)
    measured_expectations.append(meas_t)
measured_expectations = np.array(measured_expectations)

# -------------------- Compute Likelihood Grid --------------------
posterior = np.zeros((len(B0_vals), len(omega_vals), len(phi0_vals)))

for i, B0 in enumerate(B0_vals):
    for j, omega in enumerate(omega_vals):
        for k, phi0 in enumerate(phi0_vals):
            preds = decohered_expectation(times, B0, omega, phi0, T2)
            log_likelihood = -np.sum((measured_expectations - preds)**2) / (2 * sigma_noise**2)
            posterior[i, j, k] = np.exp(log_likelihood)

# Normalize the posterior
posterior /= np.sum(posterior)

# -------------------- Marginals and MAP Estimate --------------------
posterior_B0 = np.sum(posterior, axis=(1, 2))
posterior_omega = np.sum(posterior, axis=(0, 2))
posterior_phi0 = np.sum(posterior, axis=(0, 1))

i_max, j_max, k_max = np.unravel_index(np.argmax(posterior), posterior.shape)
B0_map = B0_vals[i_max]
omega_map = omega_vals[j_max]
phi0_map = phi0_vals[k_max]

print(f"MAP Estimates: B0 = {B0_map:.3f}, omega = {omega_map:.3f}, phi0 = {phi0_map:.3f}")

# -------------------- Posterior Marginal Plots --------------------
fig, axs = plt.subplots(3, 1, figsize=(8, 8))

axs[0].plot(B0_vals, posterior_B0, label='Posterior p(B₀)', color='navy')
axs[0].axvline(B0_true, color='green', linestyle='--', label='True B₀')
axs[0].axvline(B0_map, color='red', linestyle=':', label='MAP B₀')
axs[0].legend()
axs[0].set_title("Posterior Distribution for B₀")

axs[1].plot(omega_vals, posterior_omega, label='Posterior p(ω)', color='purple')
axs[1].axvline(omega_true, color='green', linestyle='--', label='True ω')
axs[1].axvline(omega_map, color='red', linestyle=':', label='MAP ω')
axs[1].legend()
axs[1].set_title("Posterior Distribution for ω")

axs[2].plot(phi0_vals, posterior_phi0, label='Posterior p(φ₀)', color='darkorange')
axs[2].axvline(phi0_true, color='green', linestyle='--', label='True φ₀')
axs[2].axvline(phi0_map, color='red', linestyle=':', label='MAP φ₀')
axs[2].legend()
axs[2].set_title("Posterior Distribution for φ₀")

plt.tight_layout()
plt.show()
