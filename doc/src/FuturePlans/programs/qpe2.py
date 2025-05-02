import pennylane as qml
from pennylane import numpy as np

# Number of qubits in the counting register (controls the precision)
n_counting = 3
dev = qml.device("default.qubit", wires=n_counting + 1)

# Phase to estimate (should be between 0 and 1)
phi = 0.125  # exact phase is 1/8

def apply_controlled_unitary(phi, control, target, power):
    """Apply controlled unitary U^{2^power} = Rz(2πφ * 2^power)"""
    angle = 2 * np.pi * phi * (2 ** power)
    qml.ctrl(qml.RZ, control=control)(angle, wires=target)

@qml.qnode(dev)
def qpe_circuit():
    # Counting register (first n_counting qubits)
    for i in range(n_counting):
        qml.Hadamard(wires=i)

    # Eigenstate |ψ> = |1> on the last qubit
    qml.PauliX(wires=n_counting)

    # Apply controlled-U^{2^j}
    for i in range(n_counting):
        apply_controlled_unitary(phi, control=i, target=n_counting, power=n_counting - 1 - i)

    # Apply inverse QFT on the counting register
    qml.adjoint(qml.templates.QFT)(wires=range(n_counting))

    # Measurement
    return qml.probs(wires=range(n_counting))

# Run the circuit
probs = qpe_circuit()
estimated_bin = np.argmax(probs)
estimated_phi = estimated_bin / (2 ** n_counting)

# Print results
print(f"Exact phase φ: {phi}")
print(f"Estimated binary outcome: {format(estimated_bin, f'0{n_counting}b')}")
print(f"Estimated phase φ: {estimated_phi}")
