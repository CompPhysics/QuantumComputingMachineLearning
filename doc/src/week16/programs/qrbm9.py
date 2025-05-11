import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Configuration
num_visible = 2
num_hidden = 2
num_qubits = num_visible + num_hidden
wires = list(range(num_qubits))

dev = qml.device("default.qubit", wires=num_qubits, shots=1000)

# Ansatz
def vqbm_ansatz(params):
    for i in range(num_qubits):
        qml.RY(params[i], wires=i)
    for i in range(num_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    for i in range(num_qubits):
        qml.RZ(params[i + num_qubits], wires=i)

# Hamiltonian
def generate_hamiltonian():
    coeffs = []
    observables = []
    for i in range(num_qubits):
        coeffs.append(np.random.uniform(-1, 1))
        observables.append(qml.PauliZ(wires=i))
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            coeffs.append(np.random.uniform(-1, 1))
            observables.append(qml.PauliZ(wires=i) @ qml.PauliZ(wires=j))
    return qml.Hamiltonian(coeffs, observables)

H = generate_hamiltonian()

# Energy expectation value
@qml.qnode(dev)
def energy_expectation(params):
    vqbm_ansatz(params)
    return qml.expval(H)

# Sampling function (returns bitstrings)
@qml.qnode(dev)
def sample_circuit(params):
    vqbm_ansatz(params)
    return qml.sample(wires=range(num_visible))  # only visible units

# Training
params = 0.01 * np.random.randn(2 * num_qubits, requires_grad=True)
opt = qml.AdamOptimizer(stepsize=0.1)
epochs = 100

for i in range(epochs):
    params = opt.step(energy_expectation, params)
    if i % 10 == 0:
        energy = energy_expectation(params)
        print(f"Epoch {i}: Energy = {energy:.4f}")

# Generate samples
samples = sample_circuit(params)
bitstrings = ["".join(str(bit) for bit in sample) for sample in samples]

# Count frequencies
counts = Counter(bitstrings)
total = sum(counts.values())
probs = {state: count / total for state, count in counts.items()}

# Plot
states = sorted(probs.keys())
values = [probs[s] for s in states]

plt.bar(states, values, color='skyblue')
plt.xlabel("Visible states")
plt.ylabel("Probability")
plt.title("Learned Distribution from VQBM")
plt.show()
