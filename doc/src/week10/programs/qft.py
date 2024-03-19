import matplotlib.pyplot as plt
import numpy as np
from math import pi
from qiskit import *
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.tools.visualization import circuit_drawer

q = QuantumRegister(4)
qc = QuantumCircuit(q)
#-------------------
qc.h(0)
qc.cu1(pi/2,q[1],q[0])
qc.cu1(pi/4,q[2],q[0])
qc.cu1(pi/8,q[3],q[0])
#-------------------
qc.barrier()
qc.h(1)
qc.cu1(pi/2,q[2],q[1])
qc.cu1(pi/4,q[3],q[1])
#-------------------
qc.barrier()
qc.h(2)
qc.cu1(pi/2,q[3],q[2])
#-------------------
qc.barrier()
qc.h(3)
#-------------------
qc.draw(output='mpl')

# Import Aer
from qiskit import Aer

# Run the quantum circuit on a statevector simulator backend
backend = Aer.get_backend('statevector_simulator')

# Create a Quantum Program for execution
job = execute(qc, backend)
result = job.result()

outputstate = result.get_statevector(qc, decimals=3)
print(outputstate)
