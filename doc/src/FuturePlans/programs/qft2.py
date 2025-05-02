import pennylane as qml
from pennylane import numpy as np
import math

dev = qml.device("default.qubit", wires=3)

def qft_rotations(wires):
   if len(wires) == 0:
       return
   head, *tail = wires
   qml.Hadamard(wires=head)
   for i, qubit in enumerate(tail):
       qml.ctrl(qml.PhaseShift, control=qubit)(math.pi / 2**(i+1), wires=head)
   qft_rotations(tail)

def swap_registers(wires):
   for i in range(len(wires) // 2):
       qml.SWAP(wires=[wires[i], wires[-i - 1]])

@qml.qnode(dev)
def qft_custom():
   qml.BasisState(np.array([1, 0, 1]), wires=range(3))
   qft_rotations(wires=[0, 1, 2])
   swap_registers(wires=[0, 1, 2])
   return qml.state()

print("Custom QFT state:", qft_custom())
