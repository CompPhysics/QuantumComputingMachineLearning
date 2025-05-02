import pennylane as qml
from pennylane import numpy as np
n_qubits = 2
dev = qml.device(``default.qubit'', wires=n_qubits + 1)

def controlled_unitary(t):
def unitary():
qml.PhaseShift(2 * np.pi * 0.3 * t, wires=n_qubits)
return unitary
@qml.qnode(dev)
def energy_estimation():
  # Apply Hadamards to estimation qubits
  for i in range(n_qubits):
      qml.Hadamard(wires=i)
  # Prepare eigenstate for target qubit
      qml.PauliX(wires=n_qubits)
  # Apply controlled-U^{2^j}
  for i in range(n_qubits):
      unitary = controlled_unitary(2**i)
      qml.ctrl(unitary, control=i)()
  # Inverse QFT
  inverse_qft(range(n_qubits))
  return qml.probs(wires=range(n_qubits))


probs = energy_estimation()
print(``Measured energy phase:'', probs)
