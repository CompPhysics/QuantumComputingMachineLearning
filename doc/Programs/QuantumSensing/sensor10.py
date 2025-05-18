import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class QuantumSensorAnimated:
    def __init__(self, n_qubits=2, gamma=1.0, T2=5.0, shots=100, sigma_noise=0.02):
        self.n_qubits = n_qubits
        self.gamma = gamma
        self.T2 = T2
        self.shots = shots
        self.sigma_noise = sigma_noise

        # Time points for simulation
        self.times = np.linspace(0.1, 5, 100)

        # True field parameters
        self.B0_true = 1.0
        self.omega_true = 1.0
        self.phi0_true = 0.0
    def B_field(self, t):
        """Time-dependent magnetic field."""
        return self.B0_true * np.sin(self.omega_true * t)

    def phase(self, t):
        """Total accumulated phase for entangled qubits."""
        return self.n_qubits * self.gamma * self.B_field(t) * t + self.phi0_true

    def animate_true_phase(self):
        """Create an animation of phase evolution over time."""
        fig, ax = plt.subplots()
        ax.set_xlim(self.times[0], self.times[-1])
        ax.set_ylim(-10, 10)
        ax.set_xlabel("Time")
        ax.set_ylabel("Phase (radians)")
        ax.set_title(f"Quantum Phase Accumulation for {self.n_qubits} Qubits")

        line, = ax.plot([], [], lw=2, color='blue')
        phase_vals = [self.phase(t) for t in self.times]

        def init():
            line.set_data([], [])
            return line,

        def update(frame):
            t_vals = self.times[:frame]
            y_vals = phase_vals[:frame]
            line.set_data(t_vals, y_vals)
            return line,

        ani = FuncAnimation(fig, update, frames=len(self.times),
                            init_func=init, blit=True, repeat=False)
        plt.show()


# ----------------------- Run Example -----------------------

sensor = QuantumSensorAnimated(n_qubits=4)  # use any number of entangled qubits
sensor.animate_true_phase()

"""
Notes:
•
You can change n_qubits=4 to any integer for different entanglement levels.
•
The phase scales linearly with the number of entangled qubits — as expected in quantum metrology.
•
The animation runs in-place in a Jupyter notebook or can be saved using ani.save(...).
"""
