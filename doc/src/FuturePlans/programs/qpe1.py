import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# Number of counting qubits
n_counting = 4
total_wires = n_counting + 1
dev = qml.device("default.qubit", wires=total_wires)

# ------------------------------
# Define the eigenstate |ψ⟩ preparation
# ------------------------------
def prepare_eigenstate():
    """Prepares an eigenstate of the unitary U"""
    # Example: |ψ⟩ = |+⟩ (eigenstate of Hadamard gate)
    qml.Hadamard(wires=n_counting)

# ------------------------------
# Define the unitary U
# ------------------------------
def unitary():
    """The unitary operator U whose eigenvalue we want to estimate"""
    # Example: Phase shift of angle θ -> eigenvalue = exp(iθ)
    theta = 2 * np.pi * 0.3125  # target φ = 0.3125
    qml.PhaseShift(theta, wires=n_counting)

# ------------------------------
# Apply controlled-U^{2^k}
# ------------------------------
def apply_controlled_powers(unitary, control, target, power):
    """Apply controlled-U^{2^power}"""
    for _ in range(2 ** power):
        qml.ctrl(unitary, control=control)()

# ------------------------------
# QPE Circuit
# ------------------------------
@qml.qnode(dev)
def qpe_generalized():
    # Step 1: Apply Hadamards to counting register
    for i in range(n_counting):
        qml.Hadamard(wires=i)

    # Step 2: Prepare eigenstate |ψ⟩ on the last qubit
    prepare_eigenstate()

    # Step 3: Apply controlled-U^{2^j} gates
    for i in range(n_counting):
        power = n_counting - 1 - i
        apply_controlled_powers(unitary, control=i, target=n_counting, power=power)

    # Step 4: Apply inverse QFT
    qml.adjoint(qml.templates.QFT)(wires=range(n_counting))

    # Step 5: Measurement
    return qml.probs(wires=range(n_counting))

# ------------------------------
# Run and Analyze
# ------------------------------
probs = qpe_generalized()
estimated_bin = np.argmax(probs)
estimated_phi = estimated_bin / (2 ** n_counting)

# Plot
x = np.arange(2**n_counting)
labels = [format(i, f'0{n_counting}b') for i in x]

plt.figure(figsize=(8, 4))
plt.bar(x, probs, tick_label=labels, color='lightgreen', edgecolor='black')
plt.xlabel("Measured binary outcome")
plt.ylabel("Probability")
plt.title(f"Generalized QPE — Estimated φ ≈ {estimated_phi:.4f}")
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Print results
print(f"Most likely binary outcome: {format(estimated_bin, f'0{n_counting}b')}")
print(f"Estimated phase φ: {estimated_phi}")
