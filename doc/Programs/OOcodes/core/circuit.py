import random
from collections import Counter

class Gate:
    def __init__(self, matrix, targets):
        self.matrix = matrix
        self.targets = targets

class Circuit:
    def __init__(self, num_qubits):
        self.n = num_qubits
        self.reset()

    def reset(self):
        self.state = np.zeros(2**self.n, dtype=np.complex128)
        self.state[0] = 1.0
        self.gates = []

    def add_gate(self, gate):
        self.gates.append(gate)

    def run(self):
        for gate in self.gates:
            self.apply_gate(gate)

    def apply_gate(self, gate):
        full_U = self.expand_gate(gate)
        self.state = full_U @ self.state

    def expand_gate(self, gate):
        if len(gate.targets) == 1:
            return self.tensor_product_gate(gate)
        elif len(gate.targets) == 2:
            return self.tensor_product_two_qubit_gate(gate)
        else:
            raise ValueError("Only 1 or 2 qubit gates supported.")

    def tensor_product_gate(self, gate):
        ops = [np.eye(2) for _ in range(self.n)]
        ops[gate.targets[0]] = gate.matrix
        U = ops[0]
        for op in ops[1:]:
            U = np.kron(U, op)
        return U

    def tensor_product_two_qubit_gate(self, gate):
        ops = [np.eye(2) for _ in range(self.n)]
        idx = sorted(gate.targets)
        # naive expansion for generality:
        full_U = np.eye(1)
        for i in range(self.n):
            if i in gate.targets:
                continue
            full_U = np.kron(full_U, np.eye(2))
        full_U = np.kron(gate.matrix, full_U)
        return full_U  # for small N this works; we can optimize later

    def get_statevector(self):
        return self.state

    def get_probabilities(self):
        return np.abs(self.state)**2

    def measure(self, shots=1024):
        probs = self.get_probabilities()
        basis_states = [format(i, f'0{self.n}b') for i in range(2**self.n)]
        samples = random.choices(basis_states, weights=probs, k=shots)
        return dict(Counter(samples))

