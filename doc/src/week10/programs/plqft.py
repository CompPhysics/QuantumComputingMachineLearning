import pennylane as qml
from pennylane import numpy as np

# Define the number of qubits
n_qubits = 3

# Create a device using default.qubit
dev = qml.device("default.qubit", wires=n_qubits)

# Define the Quantum Fourier Transform as a Quantum Function
@qml.qnode(dev)
def qft_circuit():
    for j in range(n_qubits):
        # Apply the Hadamard gate to the j-th qubit
        qml.Hadamard(wires=j)
        # Apply controlled phase shifts
        for k in range(j + 1, n_qubits):
            qml.ControlledPhaseShift(np.pi / 2 ** (k - j), wires=[k, j])
    
    # PennyLane natively returns the state, but if you want measurement in computational basis:
    return qml.probs(wires=range(n_qubits))

# Execute the QFT circuit
probabilities = qft_circuit()

# Print the resulting state probabilities
print("Probabilities after performing QFT:")
print(probabilities)
