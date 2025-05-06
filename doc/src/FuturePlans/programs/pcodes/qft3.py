import numpy as np
import matplotlib.pyplot as plt
#import matplotlib as plt
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

    def plot_amplitudes(self, title="State Vector Amplitudes", save_path=None):
        labels = [format(i, f'0{self.n_qubits}b') for i in range(self.N)]
        reals = [self.state[i].real for i in range(self.N)]
        imags = [self.state[i].imag for i in range(self.N)]
        x = np.arange(self.N)
        width = 0.35
        fig, ax = plt.subplots()
        ax.bar(x - width/2, reals, width, label='Real')
        ax.bar(x + width/2, imags, width, label='Imaginary')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_ylabel('Amplitude')
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()

        if save_path:
           plt.savefig(save_path)
           print(f"Plot saved to: {save_path}")
        else:
           plt.show()

    def plot_probabilities(self, shots=1024, title="Measurement Probabilities", save_path=None):
        results = self.measure(shots=shots)
        bitstrings = sorted(results.keys())
        counts = [results[b] for b in bitstrings]
        fig, ax = plt.subplots()
        ax.bar(bitstrings, counts)
        ax.set_xlabel("Bitstring")
        ax.set_ylabel("Counts")
        ax.set_title(title)
        ax.set_xticklabels(bitstrings, rotation=45)
        plt.tight_layout()

        if save_path:
           plt.savefig(save_path)
           print(f"Histogram saved to: {save_path}")
        else:
           plt.show()

            
# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
   qft = QuantumFourierTransform(3)
   qft.initialize_basis_state(5)

   qft.apply_qft()

# Save amplitude plot
   qft.plot_amplitudes("QFT Amplitudes", save_path="qft_amplitudes.png")

# Save measurement histogram
   qft.plot_probabilities(shots=1024, title="QFT Output Distribution", save_path="qft_histogram.png")

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
