"""
Models a time-dependent magnetic field
Uses Bell-state entangled qubits
Simulates decoherence
Computes density matrices for mixed states
Calculates expectation values
Estimates field parameters using Bayesian inference
Visualizes fidelity and posterior distributions
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh

class QuantumSensorBayesian:
    def __init__(self, n_qubits=2, gamma=1.0, T2=5.0, shots=100, sigma_noise=0.02):
        self.n_qubits = n_qubits
        self.gamma = gamma
        self.T2 = T2
        self.shots = shots
        self.sigma_noise = sigma_noise

        self.times = np.linspace(0.1, 5, 100)
        self.B0_true = 1.0
        self.omega_true = 1.0
        self.phi0_true = 0.0

    def B_field(self, t, B0, omega):
        return B0 * np.sin(omega * t)

    def phase(self, t, B0, omega, phi0):
        return self.n_qubits * self.gamma * self.B_field(t, B0, omega) * t + phi0

    def bell_density_matrix(self, t, B0, omega, phi0):
        theta = self.phase(t, B0, omega, phi0)
        decay = np.exp(-t / self.T2)
        rho = np.zeros((4, 4), dtype=complex)
        rho[0, 0] = rho[3, 3] = 0.5
        rho[0, 3] = 0.5 * decay * np.exp(-1j * theta)
        rho[3, 0] = 0.5 * decay * np.exp(1j * theta)
        return rho

    def observable_ZZ(self):
        return np.diag([1, -1, -1, 1])

    def expectation(self, rho):
        ZZ = self.observable_ZZ()
        return np.real(np.trace(rho @ ZZ))

    def simulate_measurement(self, exp_val):
        p = (1 + exp_val) / 2
        counts = np.random.binomial(self.shots, p)
        return 2 * (counts / self.shots) - 1 + np.random.normal(0, self.sigma_noise)

    def generate_data(self):
        self.measurements = []
        for t in self.times:
            rho = self.bell_density_matrix(t, self.B0_true, self.omega_true, self.phi0_true)
            exp_val = self.expectation(rho)
            meas = self.simulate_measurement(exp_val)
            self.measurements.append(meas)
        self.measurements = np.array(self.measurements)

    def sqrtm(self, matrix):
        eigvals, eigvecs = eigh(matrix)
        sqrt_vals = np.sqrt(np.maximum(eigvals, 0))
        return eigvecs @ np.diag(sqrt_vals) @ eigvecs.T.conj()

    def fidelity(self, rho_true, rho_est):
        sqrt_rho = self.sqrtm(rho_true)
        inter = sqrt_rho @ rho_est @ sqrt_rho
        sqrt_inter = self.sqrtm(inter)
        return np.real(np.trace(sqrt_inter)) ** 2

    def compute_fidelity_curve(self, B0_est, omega_est, phi0_est):
        fids = []
        for t in self.times:
            r_true = self.bell_density_matrix(t, self.B0_true, self.omega_true, self.phi0_true)
            r_est = self.bell_density_matrix(t, B0_est, omega_est, phi0_est)
            fids.append(self.fidelity(r_true, r_est))
        return np.array(fids)

    def plot_fidelity_vs_data(self, B0, omega, phi0):
        preds = []
        for t in self.times:
            rho = self.bell_density_matrix(t, B0, omega, phi0)
            preds.append(self.expectation(rho))
        preds = np.array(preds)
        fids = self.compute_fidelity_curve(B0, omega, phi0)

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.times, self.measurements, label='Data', alpha=0.7)
        plt.plot(self.times, preds, label='Model Prediction', color='crimson')
        plt.xlabel("Time")
        plt.ylabel("⟨Z⊗Z⟩")
        plt.title("Signal Comparison")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.times, fids, color='seagreen')
        plt.ylim(0, 1.05)
        plt.xlabel("Time")
        plt.ylabel("Fidelity")
        plt.title("Fidelity of Bell State Over Time")

        plt.tight_layout()
        plt.show()

    def bayesian_posterior(self, B0_range, omega_range, phi0_range):
        B_vals = np.linspace(*B0_range)
        omega_vals = np.linspace(*omega_range)
        phi_vals = np.linspace(*phi0_range)
        posterior = np.zeros((len(B_vals), len(omega_vals), len(phi_vals)))

        for i, B0 in enumerate(B_vals):
            for j, omega in enumerate(omega_vals):
                for k, phi0 in enumerate(phi_vals):
                    likelihood = 0
                    for t, meas in zip(self.times, self.measurements):
                        rho = self.bell_density_matrix(t, B0, omega, phi0)
                        pred = self.expectation(rho)
                        likelihood += -((meas - pred) ** 2) / (2 * self.sigma_noise ** 2)
                    posterior[i, j, k] = np.exp(likelihood)

        posterior /= np.sum(posterior)
        self.posterior = posterior
        self.param_grids = (B_vals, omega_vals, phi_vals)

    def plot_posterior_slices(self):
        B_vals, omega_vals, phi_vals = self.param_grids
        B_slice = np.argmax(np.sum(np.sum(self.posterior, axis=2), axis=1))
        omega_slice = np.argmax(np.sum(np.sum(self.posterior, axis=2), axis=0))
        phi_slice = np.argmax(np.sum(np.sum(self.posterior, axis=0), axis=0))

        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        axs[0].imshow(self.posterior[:, :, phi_slice], aspect='auto',
                      extent=[omega_vals[0], omega_vals[-1], B_vals[0], B_vals[-1]], origin='lower')
        axs[0].set_title(f"Posterior Slice at ϕ₀ = {phi_vals[phi_slice]:.2f}")
        axs[0].set_xlabel("ω"), axs[0].set_ylabel("B₀")

        axs[1].imshow(self.posterior[:, omega_slice, :], aspect='auto',
                      extent=[phi_vals[0], phi_vals[-1], B_vals[0], B_vals[-1]], origin='lower')
        axs[1].set_title(f"Posterior Slice at ω = {omega_vals[omega_slice]:.2f}")
        axs[1].set_xlabel("ϕ₀"), axs[1].set_ylabel("B₀")

        axs[2].imshow(self.posterior[B_slice, :, :], aspect='auto',
                      extent=[phi_vals[0], phi_vals[-1], omega_vals[0], omega_vals[-1]], origin='lower')
        axs[2].set_title(f"Posterior Slice at B₀ = {B_vals[B_slice]:.2f}")
        axs[2].set_xlabel("ϕ₀"), axs[2].set_ylabel("ω")

        plt.tight_layout()
        plt.show()


sensor = QuantumSensorBayesian()
sensor.generate_data()
sensor.plot_fidelity_vs_data(B0=1.0, omega=1.0, phi0=0.0)

# Posterior estimation
sensor.bayesian_posterior(
    B0_range=(0.8, 1.2, 20),
    omega_range=(0.8, 1.2, 20),
    phi0_range=(-np.pi, np.pi, 20)
)
sensor.plot_posterior_slices()
