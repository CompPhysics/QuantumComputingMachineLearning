import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import Counter

# Config
num_visible = 2
num_hidden = 2
num_qubits = num_visible + num_hidden
epochs = 50
shots = 1000

dev = qml.device("default.qubit", wires=num_qubits, shots=shots)

# Target data (biased toward '11' and '00')
target_bitstrings = ['11', '11', '11', '00', '00', '01']
target_counts = Counter(target_bitstrings)
target_probs = {
    format(i, f'0{num_visible}b'): target_counts.get(format(i, f'0{num_visible}b'), 0) / len(target_bitstrings)
    for i in range(2**num_visible)
}


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


@qml.qnode(dev)
def energy_expectation(params):
    vqbm_ansatz(params)
    return qml.expval(H)

@qml.qnode(dev)
def sample_circuit(params):
    vqbm_ansatz(params)
    return qml.sample(wires=range(num_visible))

# Helper: Convert samples to bitstring histogram
def get_distribution(samples):
    bitstrings = ["".join(str(bit) for bit in s) for s in samples]
    counts = Counter(bitstrings)
    total = sum(counts.values())
    return {
        format(i, f'0{num_visible}b'): counts.get(format(i, f'0{num_visible}b'), 0) / total
        for i in range(2**num_visible)
    }

# Training and storing distributions
params = 0.01 * np.random.randn(2 * num_qubits, requires_grad=True)
opt = qml.AdamOptimizer(stepsize=0.1)
history = []

for epoch in range(epochs):
    params = opt.step(energy_expectation, params)
    learned_dist = get_distribution(sample_circuit(params))
    history.append(learned_dist)
    if epoch % 10 == 0:
        print(f"Epoch {epoch} energy: {energy_expectation(params):.4f}")

# Animation setup
states = [format(i, f'0{num_visible}b') for i in range(2**num_visible)]

fig, ax = plt.subplots()
bar1 = ax.bar(states, [0]*len(states), color='skyblue', label="VQBM")
bar2 = ax.bar(states, [target_probs[s] for s in states], color='orange', alpha=0.6, label="Target")
ax.set_ylim(0, 1)
ax.set_ylabel("Probability")
ax.set_title("VQBM Learning Over Epochs")
ax.legend()

def update(frame):
    dist = history[frame]
    for i, state in enumerate(states):
        bar1[i].set_height(dist[state])
    ax.set_title(f"Epoch {frame}")

ani = FuncAnimation(fig, update, frames=len(history), repeat=False)
plt.show()


"""
Define a target distribution (e.g., classical binary data).
At each epoch:
Train the model.
Sample the VQBM output.
Compute histogram probabilities.

Store results for animation.
Use matplotlib.animation to animate VQBM’s learned distribution converging to the target.
Trains a VQBM using an energy-based loss.
Samples from the circuit at each epoch.
Animates how the model’s output distribution converges toward the target.
"""
