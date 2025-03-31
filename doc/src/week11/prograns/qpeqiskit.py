from qiskit import QuantumCircuit, Aer, transpile, assemble, execute
from qiskit.visualization import plot_histogram
import numpy as np

def qpe(unitary, num_counting_qubits):
   """Quantum Phase Estimation Algorithm.

   Args:
       unitary (QuantumCircuit): The unitary operator whose phase we estimate.
       num_counting_qubits (int): Number of qubits in the counting register.

   Returns:
       QuantumCircuit: QPE quantum circuit
   """
   n = num_counting_qubits
   qc = QuantumCircuit(n + 1, n)  # n counting qubits + 1 eigenstate qubit

   # Step 1: Apply Hadamard to counting qubits
   for qubit in range(n):
       qc.h(qubit)

   # Step 2: Apply controlled-U^2^j operations
   for j in range(n):
       power = 2**j
       controlled_U = unitary.control(1).power(power)
       qc.append(controlled_U, [j] + [n])  # Control: j, Target: n

   # Step 3: Apply inverse QFT
   qc.append(qft_dagger(n), range(n))

   # Step 4: Measure counting qubits
   qc.measure(range(n), range(n))

   return qc

def qft_dagger(n):
   """Creates an inverse Quantum Fourier Transform (QFT†) circuit."""
   qc = QuantumCircuit(n)
   for qubit in range(n//2):
       qc.swap(qubit, n-qubit-1)

   for j in range(n):
       for m in range(j):
           qc.cp(-np.pi / (2**(j-m)), m, j)
       qc.h(j)

   return qc

# Define the unitary U with phase phi = 1/3
phi = 1/3
U = QuantumCircuit(1)
U.p(2 * np.pi * phi, 0)  # Phase gate

# Number of counting qubits
num_counting_qubits = 3

# Generate QPE circuit
qpe_circuit = qpe(U, num_counting_qubits)

# Simulate the circuit
simulator = Aer.get_backend('qasm_simulator')
compiled_circuit = transpile(qpe_circuit, simulator)
qobj = assemble(compiled_circuit)
result = simulator.run(qobj).result()

# Get measurement results
counts = result.get_counts()

# Plot histogram of results
plot_histogram(counts)
