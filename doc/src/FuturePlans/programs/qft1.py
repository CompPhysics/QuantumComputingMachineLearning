import pennylane as qml
from pennylane import numpy as np

n_qubits = 3
dev = qml.device("default.qubit", wires=n_qubits)

def apply_qft(wires):
   qml.templates.QFT(wires=wires)

@qml.qnode(dev)
def qft_circuit():
   qml.BasisState(np.array([1, 0, 1]), wires=range(n_qubits))
   apply_qft(wires=range(n_qubits))
   return qml.state()

state = qft_circuit()
print("QFT output state:", state)
