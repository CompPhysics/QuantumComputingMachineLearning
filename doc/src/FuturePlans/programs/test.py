import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter

# =============================== #
#       Quantum Gate Classes      #
# =============================== #

class Gate:
    def __init__(self, matrix, targets):
        self.matrix = np.array(matrix, dtype=np.complex128)
        self.targets = targets

class OneQubitGate(Gate):
    def __init__(self, matrix, target):
        super().__init__(matrix, [target])

class TwoQubitGate(Gate):
    def __init__(self, matrix, control, target):
        super().__init__(matrix, [control, target])

# One-qubit standard gates
def I():  return np.eye(2)
def X():  return np.array([[0,1],[1,0]])
def Y():  return np.array([[0,-1j],[1j,0]])
def Z():  return np.array([[1,0],[0,-1]])
def H():  return (1/np.sqrt(2))*np.array([[1,1],[1,-1]])
def S():  return np.array([[1,0],[0,1j]])
def T():  return np.array([[1,0],[0,np.exp(1j*np.pi/4)]])

def Rx(theta):
    return np.array([
        [np.cos(theta/2), -1j*np.sin(theta/2)],
        [-1j*np.sin(theta/2), np.cos(theta/2)]
    ])

def Ry(theta):
    return np.array([
        [np.cos(theta/2), -np.sin(theta/2)],
        [np.sin(theta/2),  np.cos(theta/2)]
    ])

def Rz(theta):
    return np.array([
        [np.exp(-1j*theta/2), 0],
        [0, np.exp(1j*theta/2)]
    ])

# Two-qubit gates
def CNOT():
    return np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,0,1],
        [0,0,1,0]
    ])

def CZ():
    return np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,-1]
    ])

def SWAP():
    return np.array([
        [1,0,0,0],
        [0,0,1,0],
        [0,1,0,0],
        [0,0,0,1]
    ])

# =============================== #
#      Quantum Circuit Class      #
# =============================== #

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
        n = self.n
        if len(gate.targets) == 1:
            # One-qubit gate
            target = gate.targets[0]
            ops = [I()]*n
            ops[target] = gate.matrix
            return self.tensor_product(ops)
        elif len(gate.targets) == 2:
            # Two-qubit gate
            t0, t1 = gate.targets
            ops = [I()]*n
            # Insert identity first, apply 4x4 gate manually:
            full = np.eye(1, dtype=np.complex128)
            for i in range(n):
                if i == t0:
                    full = np.kron(full, np.eye(2))
                elif i == t1:
                    full = np.kron(full, np.eye(2))
                else:
                    full = np.kron(full, I())

            # Reshape state space to insert 4x4 gate
            axes = list(range(n))
            axes.remove(t0)
            axes.remove(t1)
            axes = [t0, t1] + axes

            perm = np.argsort(axes)
            U = gate.matrix

            full_gate = np.tensordot(U, full.reshape([2]*2*n), axes=0)
            full_gate = np.moveaxis(full_gate, list(range(2*n)), perm*2 + perm*2)
            return full_gate.reshape(2**n, 2**n)
        else:
            raise ValueError("Only 1- and 2-qubit gates supported")

    def tensor_product(self, matrices):
        result = matrices[0]
        for m in matrices[1:]:
            result = np.kron(result, m)
        return result

    def get_statevector(self):
        return self.state

    def get_probabilities(self):
        return np.abs(self.state)**2

    def measure(self, shots=1024):
        probs = self.get_probabilities()
        basis_states = [format(i, f'0{self.n}b') for i in range(2**self.n)]
        samples = random.choices(basis_states, weights=probs, k=shots)
        return dict(Counter(samples))

    def visualize_probabilities(self, title="State Probabilities"):
        probs = self.get_probabilities()
        basis = [format(i, f'0{self.n}b') for i in range(2**self.n)]
        plt.bar(basis, probs, color='teal')
        plt.xlabel("Basis States")
        plt.ylabel("Probability")
        plt.title(title)
        plt.show()

# =============================== #
#            Noise models         #
# =============================== #

def apply_bit_flip(state, p):
    noisy_state = state.copy()
    for i in range(len(state)):
        if random.random() < p:
            flipped = i ^ 1  # flip LSB (bit flip on qubit 0 for example)
            noisy_state[flipped] += state[i]
            noisy_state[i] = 0
    return noisy_state / np.linalg.norm(noisy_state)

def apply_depolarizing(state, p):
    d = len(state)
    noisy_state = (1 - p) * state + p / d * np.ones(d)
    return noisy_state / np.linalg.norm(noisy_state)

# =============================== #
#       Bell state generator      #
# =============================== #

def bell_state(label="Phi+"):
    c = Circuit(2)
    c.add_gate(OneQubitGate(H(), 0))
    c.add_gate(TwoQubitGate(CNOT(), 0, 1))
    if label == "Phi+":
        pass
    elif label == "Phi-":
        c.add_gate(OneQubitGate(Z(), 0))
    elif label == "Psi+":
        c.add_gate(OneQubitGate(X(), 1))
    elif label == "Psi-":
        c.add_gate(OneQubitGate(X(), 1))
        c.add_gate(OneQubitGate(Z(), 0))
    else:
        raise ValueError("Unknown Bell state")
    c.run()
    return c

# =============================== #
#         Demonstration           #
# =============================== #

if __name__ == "__main__":

    labels = ["Phi+", "Phi-", "Psi+", "Psi-"]
    shots = 1000

    for label in labels:
        print(f"\n{label} state:")
        c = bell_state(label)
        print("Statevector:", c.get_statevector())
        results = c.measure(shots=shots)
        print(f"Measurement (shots={shots}):", results)
        c.visualize_probabilities(title=f"{label} state probabilities")


