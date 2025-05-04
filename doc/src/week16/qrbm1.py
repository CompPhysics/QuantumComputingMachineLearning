import pennylane as qml
from pennylane import numpy as np

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qbm_circuit(params, beta=1.0):
   h, J = params
   # Apply transverse field and Ising couplings
   for i in range(n_qubits):
       qml.RX(beta * h[i], wires=i)
   for i in range(n_qubits):
       for j in range(i+1, n_qubits):
           qml.IsingZZ(2 * beta * J[i][j], wires=[i, j])
   return qml.probs(wires=range(n_qubits))

# Initialize random parameters
h = np.random.rand(n_qubits)
J = np.random.rand(n_qubits, n_qubits)
probs = qbm_circuit((h, J), beta=2.0)
print("Thermal probabilities:", probs)
