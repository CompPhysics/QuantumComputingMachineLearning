import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# Number of counting qubits (controls precision)
n_counting = 3
dev = qml.device("default.qubit", wires=n_counting + 1)

# Phase to estimate
phi = 0.125  # Target phase (1/8)

def apply_controlled_unitary(phi, control, target, power):
    """Apply controlled-U^{2^power} = Rz(2πφ * 2^power)"""
    angle = 2 * np.pi * phi * (2 ** power)
    qml.ctrl(qml.RZ, control=control)(angle, wires=target)

@qml.qnode(dev)
def qpe_circuit():
    # Step 1: Apply Hadamards to counting register
    for i in range(n_counting):
        qml.Hadamard(wires=i)
    
    # Step 2: Prepare eigenstate |ψ> = |1>
    qml.PauliX(wires=n_counting)

    # Step 3: Apply controlled-U^{2^j}
    for i in range(n_counting):
        apply_controlled_unitary(phi, control=i, target=n_counting, power=n_counting - 1 - i)
    
    # Step 4: Apply inverse QFT
    qml.adjoint(qml.templates.QFT)(wires=range(n_counting))

    # Step 5: Measure the counting register
    return qml.probs(wires=range(n_counting))

# Run the circuit
probs = qpe_circuit()
# Compute estimated phase
estimated_bin = np.argmax(probs)
estimated_phi = estimated_bin / (2 ** n_counting)

# Plot the probability distribution
x = np.arange(2**n_counting)
labels = [format(i, f'0{n_counting}b') for i in x]

plt.figure(figsize=(8, 4))
plt.bar(x, probs, tick_label=labels, color='skyblue', edgecolor='black')
plt.xlabel("Measured binary outcome")
plt.ylabel("Probability")
plt.title(f"Quantum Phase Estimation (φ = {phi})")
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Print results
print(f"Exact phase φ: {phi}")
print(f"Most likely outcome (binary): {format(estimated_bin, f'0{n_counting}b')}")
print(f"Estimated phase φ: {estimated_phi}")
