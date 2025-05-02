import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm



def unitary_operator():
    # Controlled Phase Gate (CP) with a phase of pi/4
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, np.exp(1j * np.pi / 4), 0],
                     [0, 0, 0, np.exp(1j * np.pi / 4)]], dtype=complex)

def eigenstate():
    # Superposition state |ψ⟩ = (|0⟩ + |1⟩)/√2
    return np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)

class QuantumPhaseEstimation:
    def __init__(self, n_qubits, unitary, eigenstate, inverse=False):
        self.n_qubits = n_qubits
        self.N = 2 ** n_qubits
        self.unitary = unitary  # Unitary operator U
        self.eigenstate = eigenstate  # Eigenstate |ψ⟩
        self.inverse = inverse  # Whether to reverse the QFT
        self.state = np.zeros(self.N, dtype=complex)
        self.state[0] = 1.0  # Initialize ancillary qubits to |0...0⟩

    def apply_single_qubit_gate(self, gate, target):
        I = np.eye(2)
        ops = [I] * self.n_qubits
        ops[target] = gate
        U = ops[0]
        for op in ops[1:]:
            U = np.kron(U, op)
        self.state = U @ self.state

    def hadamard(self):
        return np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    def apply_controlled_unitary(self, control, target, exp_power):
        """Apply controlled unitary operation U^(2^j)."""
        U = np.eye(self.N, dtype=complex)
        for i in range(self.N):
            b = format(i, f'0{self.n_qubits}b')
            if b[self.n_qubits - 1 - control] == '1':
                U[i, i] = np.exp(1j * (np.pi / (2 ** exp_power)))
        self.state = U @ self.state

    def apply_qft(self, inverse=False):
        """Apply Quantum Fourier Transform on ancillary qubits."""
        for target in range(self.n_qubits):
            idx = self.n_qubits - 1 - target
            for offset in range(1, self.n_qubits - target):
                control = self.n_qubits - 1 - (target + offset)
                angle = np.pi / (2 ** offset)
                if inverse:
                    angle *= -1
                self.apply_controlled_unitary(control, idx, angle)
            self.apply_single_qubit_gate(self.hadamard(), idx)

        self.swap_registers()

    def swap_registers(self):
        perm = [int(format(i, f'0{self.n_qubits}b')[::-1], 2) for i in range(self.N)]
        self.state = self.state[perm]

    def measure(self):
        """Measure the ancillary qubits."""
        probs = np.abs(self.state) ** 2
        outcomes = np.random.choice(self.N, size=1024, p=probs)
        counts = {}
        for o in outcomes:
            bitstring = format(o, f'0{self.n_qubits}b')
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts

    def plot_probability_distribution(self, results):
        bitstrings = sorted(results.keys())
        counts = [results[b] for b in bitstrings]
        plt.bar(bitstrings, counts)
        plt.xlabel("Bitstring")
        plt.ylabel("Counts")
        plt.title("QPE Measurement Results")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def apply_phase_estimation(self):
        """Perform Quantum Phase Estimation (QPE)."""
        # Apply Hadamard to all ancillary qubits
        for target in range(self.n_qubits):
            self.apply_single_qubit_gate(self.hadamard(), target)

        # Apply controlled unitaries (U^(2^j))
        for j in range(self.n_qubits):
            self.apply_controlled_unitary(j, self.n_qubits, j)

        # Apply QFT
        self.apply_qft(inverse=True)

        # Measure the ancillary qubits
        results = self.measure()
        self.plot_probability_distribution(results)

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    n_qubits = 3  # Number of ancillary qubits
    qpe = QuantumPhaseEstimation(n_qubits, unitary_operator(), eigenstate())
    qpe.apply_phase_estimation()
