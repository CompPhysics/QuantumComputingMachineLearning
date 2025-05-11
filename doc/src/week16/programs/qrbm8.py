import pennylane as qml
from pennylane import numpy as np

# Model setup
num_visible = 2
num_hidden = 2
num_qubits = num_visible + num_hidden
wires = list(range(num_qubits))

dev = qml.device("default.qubit", wires=num_qubits)

# Variational ansatz
def vqbm_ansatz(params):
    for i in range(num_qubits):
        qml.RY(params[i], wires=i)
    for i in range(num_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    for i in range(num_qubits):
        qml.RZ(params[i + num_qubits], wires=i)


# Define the Hamiltonian (Ising-like)
def generate_hamiltonian():
    coeffs = []
    observables = []

    # Local Z terms (biases)
    for i in range(num_qubits):
        coeffs.append(np.random.uniform(-1, 1))  # Random bias
        observables.append(qml.PauliZ(wires=i))

    # ZZ interactions (couplings)
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            coeffs.append(np.random.uniform(-1, 1))  # Random interaction
            observables.append(qml.PauliZ(wires=i) @ qml.PauliZ(wires=j))

    return qml.Hamiltonian(coeffs, observables)

H = generate_hamiltonian()


# QNode that returns the energy
@qml.qnode(dev)
def energy_expectation(params):
    vqbm_ansatz(params)
    return qml.expval(H)

# Training
params = 0.01 * np.random.randn(2 * num_qubits, requires_grad=True)
opt = qml.AdamOptimizer(stepsize=0.1)
epochs = 100

for i in range(epochs):
    params = opt.step(energy_expectation, params)
    if i % 10 == 0:
        energy = energy_expectation(params)
        print(f"Epoch {i}: Energy = {energy:.4f}")

"""
Constructs a random Ising-like Hamiltonian (bias + coupling terms).
Trains the quantum circuit to minimize the expected energy.
The final state approximates a low-energy configuration of a quantum Boltzmann distribution.
"""
