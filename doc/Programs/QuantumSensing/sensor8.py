import numpy as np
import matplotlib.pyplot as plt

class QuantumSensor:
    def __init__(self, n_qubits=3, gamma=1.0, T2=3.0, sigma_noise=0.05, shots=100):
        self.n_qubits = n_qubits
        self.gamma = gamma
        self.T2 = T2
        self.sigma_noise = sigma_noise
        self.shots = shots

        self.times = np.linspace(0.1, 5, 100)

        # True field parameters
        self.B0_true = 0.8
        self.omega_true = 1.0
        self.phi0_true = 0.0

        # Bayesian grid
        self.B0_vals = np.linspace(0.5, 1.1, 60)
        self.omega_vals = np.linspace(0.7, 1.3, 60)
        self.phi0_vals = np.linspace(-np.pi, np.pi, 60)

        self.posterior = None
        self.posterior_B0 = None
        self.posterior_omega = None
        self.posterior_phi0 = None
        self.map_estimates = None

    def B_field(self, t, B0, omega):
        return B0 * np.sin(omega * t)

    def ideal_expectation(self, t, B0, omega, phi0):
        phase = self.gamma * self.n_qubits * self.B_field(t, B0, omega) * t + phi0
        return np.cos(phase)

    def decohered_expectation(self, t, B0, omega, phi0):
        return self.ideal_expectation(t, B0, omega, phi0) * np.exp(-t / self.T2)

    def noisy_measurement(self, expectation):
        p = (1 + expectation) / 2
        counts = np.random.binomial(self.shots, p)
        mean = 2 * (counts / self.shots) - 1
        return mean + np.random.normal(0, self.sigma_noise)

    def generate_measurements(self):
        self.measurements = []
        for t in self.times:
            exp_val = self.decohered_expectation(t, self.B0_true, self.omega_true, self.phi0_true)
            meas = self.noisy_measurement(exp_val)
            self.measurements.append(meas)
        self.measurements = np.array(self.measurements)

    def compute_posterior(self):
        B0_vals, omega_vals, phi0_vals = self.B0_vals, self.omega_vals, self.phi0_vals
        posterior = np.zeros((len(B0_vals), len(omega_vals), len(phi0_vals)))

        for i, B0 in enumerate(B0_vals):
            for j, omega in enumerate(omega_vals):
                for k, phi0 in enumerate(phi0_vals):
                    preds = self.decohered_expectation(self.times, B0, omega, phi0)
                    log_like = -np.sum((self.measurements - preds)**2) / (2 * self.sigma_noise**2)
                    posterior[i, j, k] = np.exp(log_like)

        posterior /= np.sum(posterior)
        self.posterior = posterior

        self.posterior_B0 = np.sum(posterior, axis=(1, 2))
        self.posterior_omega = np.sum(posterior, axis=(0, 2))
        self.posterior_phi0 = np.sum(posterior, axis=(0, 1))

        i_max, j_max, k_max = np.unravel_index(np.argmax(posterior), posterior.shape)
        self.map_estimates = (
            B0_vals[i_max],
            omega_vals[j_max],
            phi0_vals[k_max]
        )

    def print_results(self):
        B0_map, omega_map, phi0_map = self.map_estimates
        print("MAP Estimates:")
        print(f"  B₀   = {B0_map:.3f} (true = {self.B0_true})")
        print(f"  ω    = {omega_map:.3f} (true = {self.omega_true})")
        print(f"  φ₀   = {phi0_map:.3f} (true = {self.phi0_true})")

    def plot_posteriors(self):
        fig, axs = plt.subplots(3, 1, figsize=(8, 8))
        axs[0].plot(self.B0_vals, self.posterior_B0, label='Posterior p(B₀)', color='navy')
        axs[0].axvline(self.B0_true, color='green', linestyle='--', label='True B₀')
        axs[0].axvline(self.map_estimates[0], color='red', linestyle=':', label='MAP B₀')
        axs[0].legend()
        axs[0].set_title("Posterior Distribution for B₀")

        axs[1].plot(self.omega_vals, self.posterior_omega, label='Posterior p(ω)', color='purple')
        axs[1].axvline(self.omega_true, color='green', linestyle='--', label='True ω')
        axs[1].axvline(self.map_estimates[1], color='red', linestyle=':', label='MAP ω')
        axs[1].legend()
        axs[1].set_title("Posterior Distribution for ω")

        axs[2].plot(self.phi0_vals, self.posterior_phi0, label='Posterior p(φ₀)', color='darkorange')
        axs[2].axvline(self.phi0_true, color='green', linestyle='--', label='True φ₀')
        axs[2].axvline(self.map_estimates[2], color='red', linestyle=':', label='MAP φ₀')
        axs[2].legend()
        axs[2].set_title("Posterior Distribution for φ₀")

        plt.tight_layout()
        plt.show()

    def estimate_fidelity(self):
        def evolved_state_vector(B0, omega, phi0, t):
           phase = self.gamma * self.n_qubits * B0 * np.sin(omega * t) * t + phi0
           return np.array([np.cos(phase), np.sin(phase)])

        fidelities = []
        for t in self.times:
            true_state = evolved_state_vector(self.B0_true, self.omega_true, self.phi0_true, t)
            est_state = evolved_state_vector(*self.map_estimates, t)
            fid = np.abs(np.dot(true_state, est_state))**2
            fidelities.append(fid)

        avg_fidelity = np.mean(fidelities)
        print(f"Average Fidelity between true and estimated quantum state: {avg_fidelity:.4f}")

        # Optional: plot fidelity over time
        plt.figure(figsize=(7, 3))
        plt.plot(self.times, fidelities, label="Fidelity over time", color="teal")
        plt.xlabel("Time")
        plt.ylabel("Fidelity")
        plt.title("Quantum State Fidelity vs Time")
        plt.ylim(0, 1.05)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


# -------------------- Run the Sensor Simulation --------------------
sensor = QuantumSensor()
sensor.generate_measurements()
sensor.compute_posterior()
sensor.print_results()
sensor.plot_posteriors()
sensor.estimate_fidelity()
