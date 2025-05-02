import numpy as np

class QuantumFourierTransform:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.N = 2 ** n_qubits
        self.state = np.zeros(self.N, dtype=complex)

    def initialize_basis_state(self, index):
        """Initialize the quantum state to |index⟩"""
        self.state = np.zeros(self.N, dtype=complex)
        self.state[index] = 1.0

    def hadamard(self):
        return np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    def apply_single_qubit_gate(self, gate, target):
        """Apply a single-qubit gate to the target qubit."""
        I = np.eye(2)
        ops = [I] * self.n_qubits
        ops[target] = gate
        U = ops[0]
        for op in ops[1:]:
            U = np.kron(U, op)
        self.state = U @ self.state

    def apply_controlled_phase(self, control, target, theta):
        """Apply a controlled phase rotation gate."""
        size = self.N
        U = np.eye(size, dtype=complex)
        for i in range(size):
            b = format(i, f'0{self.n_qubits}b')
            if b[self.n_qubits - 1 - control] == '1' and b[self.n_qubits - 1 - target] == '1':
                U[i, i] *= np.exp(1j * theta)
        self.state = U @ self.state

    def swap_registers(self):
        """Swap qubit order (bit-reversal permutation)."""
        perm = [int(format(i, f'0{self.n_qubits}b')[::-1], 2) for i in range(self.N)]
        self.state = self.state[perm]

    def apply_qft(self, inverse=False):
        """Apply (inverse) QFT to the current state."""
        for target in range(self.n_qubits):
            idx = self.n_qubits - 1 - target
            for ctrl_offset in range(1, self.n_qubits - target):
                control = self.n_qubits - 1 - (target + ctrl_offset)
                angle = np.pi / (2 ** ctrl_offset)
                if inverse:
                    angle *= -1
                self.apply_controlled_phase(control, idx, angle)
            self.apply_single_qubit_gate(self.hadamard(), idx)
        self.swap_registers()
    def measure(self, shots=1024):
        """Simulate measurement outcomes."""
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

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    qft = QuantumFourierTransform(n_qubits=3)
    qft.initialize_basis_state(5)  # Set initial state to |5⟩

    qft.print_amplitudes("Initial State")

    qft.apply_qft()  # Apply QFT
    qft.print_amplitudes("After QFT")

    samples = qft.measure(shots=1024)
    print("\nMeasurement results:")
    for k, v in sorted(samples.items()):
        print(f"{k}: {v}")

    qft.apply_qft(inverse=True)  # Apply inverse QFT to recover original
    qft.print_amplitudes("After Inverse QFT (should be |5⟩)")



"""
reusable Python class called QuantumFourierTransform that encapsulates everything:



Initialization of state
QFT and inverse QFT
Measurement sampling
Easy customization of qubit number and input state

Benefits of This Class Design
Reusable and clean.
Makes it easy to switch between QFT and inverse QFT.
Includes state visualization and measurement simulation.
Fully self-contained — just run the script as-is.
"""
