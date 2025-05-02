import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

class QuantumFourierTransform:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.N = 2 ** n_qubits
        self.state = np.zeros(self.N, dtype=complex)

    def initialize_basis_state(self, index):
        """Set state to basis state |index⟩"""
        self.state = np.zeros(self.N, dtype=complex)
        self.state[index] = 1.0

    def initialize_custom_state(self, state_vector):
        """Set state to custom normalized state vector"""
        assert len(state_vector) == self.N, "State vector length mismatch."
        norm = np.linalg.norm(state_vector)
        assert np.isclose(norm, 1.0), "State vector must be normalized."
        self.state = np.array(state_vector, dtype=complex)

    def initialize_superposition(self):
        """Create |+⟩^n superposition state"""
        self.state = np.ones(self.N, dtype=complex) / np.sqrt(self.N)
    def initialize_bell_pair(self):
        """2-qubit Bell state: (|00⟩ + |11⟩)/√2"""
        if self.n_qubits != 2:
            raise ValueError("Bell pair requires exactly 2 qubits.")
        self.state = np.zeros(self.N, dtype=complex)
        self.state[0] = 1 / np.sqrt(2)
        self.state[3] = 1 / np.sqrt(2)

    def initialize_ghz_state(self):
        """n-qubit GHZ state: (|00...0⟩ + |11...1⟩)/√2"""
        self.state = np.zeros(self.N, dtype=complex)
        self.state[0] = 1 / np.sqrt(2)
        self.state[-1] = 1 / np.sqrt(2)

    def hadamard(self):
        return np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    def apply_single_qubit_gate(self, gate, target):
        I = np.eye(2)
        ops = [I] * self.n_qubits
        ops[target] = gate
        U = ops[0]
        for op in ops[1:]:
            U = np.kron(U, op)
        self.state = U @ self.state
    def apply_controlled_phase(self, control, target, theta):
        U = np.eye(self.N, dtype=complex)
        for i in range(self.N):
            b = format(i, f'0{self.n_qubits}b')
            if b[self.n_qubits - 1 - control] == '1' and b[self.n_qubits - 1 - target] == '1':
                U[i, i] *= np.exp(1j * theta)
        self.state = U @ self.state

    def swap_registers(self):
        perm = [int(format(i, f'0{self.n_qubits}b')[::-1], 2) for i in range(self.N)]
        self.state = self.state[perm]
    def apply_qft(self, inverse=False):
        for target in range(self.n_qubits):
            idx = self.n_qubits - 1 - target
            for offset in range(1, self.n_qubits - target):
                control = self.n_qubits - 1 - (target + offset)
                angle = np.pi / (2 ** offset)
                if inverse:
                    angle *= -1
                self.apply_controlled_phase(control, idx, angle)
            self.apply_single_qubit_gate(self.hadamard(), idx)
        self.swap_registers()
    def measure(self, shots=1024):
        probs = np.abs(self.state) ** 2
        outcomes = np.random.choice(self.N, size=shots, p=probs)
        counts = {}
        for o in outcomes:
            bitstring = format(o, f'0{self.n_qubits}b')
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts

    def print_amplitudes(self, title="Quantum State"):
        print(f"\n{title}:")
        for i, amp in enumerate(self.state):
            print(f"|{i:0{self.n_qubits}b}>: {amp:.4f}")


# Inside the class
    def reset_animation_log(self):
        self.animation_frames = []

    def record_probability_frame(self):
        """Store current probabilities for animation."""
        probs = np.abs(self.state) ** 2
        self.animation_frames.append(probs.copy())

    def apply_qft_with_recording(self, inverse=False):
        self.reset_animation_log()
        self.record_probability_frame()  # Record initial state

        for target in range(self.n_qubits):
            idx = self.n_qubits - 1 - target
            for offset in range(1, self.n_qubits - target):
                control = self.n_qubits - 1 - (target + offset)
                angle = np.pi / (2 ** offset)
                if inverse:
                    angle *= -1
                self.apply_controlled_phase(control, idx, angle)
                self.record_probability_frame()
            self.apply_single_qubit_gate(self.hadamard(), idx)
            self.record_probability_frame()

        self.swap_registers()
        self.record_probability_frame()

    def animate_probability_evolution(self, save_path="qft_animation.gif", interval=600):
        labels = [format(i, f'0{self.n_qubits}b') for i in range(self.N)]
        fig, ax = plt.subplots()
        bar = ax.bar(labels, [0]*self.N)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("QFT Probability Evolution")

        def update(frame_probs):
            for rect, prob in zip(bar, frame_probs):
                rect.set_height(prob)
            return bar

        ani = FuncAnimation(fig, update, frames=self.animation_frames,
                            interval=interval, blit=False, repeat=False)

        if save_path.endswith(".gif"):
            ani.save(save_path, writer='pillow')
        elif save_path.endswith(".mp4"):
            ani.save(save_path, writer='ffmpeg')
        else:
            raise ValueError("Unsupported format. Use .gif or .mp4")

        print(f"Animation saved to {save_path}")

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
   qft = QuantumFourierTransform(n_qubits=3)
   qft.initialize_basis_state(5)

# Apply QFT with intermediate state recording
   qft.apply_qft_with_recording()

# Animate the probability distribution over steps
   qft.animate_probability_evolution(save_path="qft_probs.gif", interval=800)



"""
Quick Tips on Usage


Superposition inputs (e.g. Hadamards to create |+\rangle^{\otimes n})
Custom state initialization (e.g. any normalized complex vector)
Entangled state preparation (e.g. Bell states, GHZ states, etc.)



initialize_superposition() gives |+\rangle^{\otimes n}
initialize_bell_pair() is for 2-qubit entangled states
initialize_ghz_state() creates a GHZ state
initialize_custom_state(vec) lets you pass any normalized state
"""
