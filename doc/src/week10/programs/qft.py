import numpy as np
from qiskit import QuantumCircuit, Aer, execute

def qft(circuit, n):
    """Apply the Quantum Fourier Transform to the first n qubits in the circuit."""
    # Apply Hadamard gates and controlled rotations
    for j in range(n):
        circuit.h(j)
        for k in range(j + 1, n):
            circuit.cp(np.pi / 2**(k - j), k, j)

    # Swap the qubits to reverse their order
    for i in range(n // 2):
        circuit.swap(i, n - i - 1)

# Number of qubits
n = 4

# Create a quantum circuit with n qubits
qc = QuantumCircuit(n)

# Apply QFT to the quantum circuit
qft(qc, n)

# Draw the resulting circuit
print("Quantum Circuit for QFT:")
print(qc.draw(output='text'))

# Run the quantum circuit on a statevector simulator backend                                                                                                                                  
backend = Aer.get_backend('statevector_simulator')

# Execute the quantum circuit and get results                                                                                                                                                      
job = execute(qc, backend)
result = job.result()

output_state_vector = result.get_statevector()
print("\nOutput State Vector:")
print(output_state_vector)
