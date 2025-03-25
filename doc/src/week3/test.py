from qiskit_aer import Aer
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# Create a quantum circuit with two qubits
bell_circuit = QuantumCircuit(2, 2)

# Apply Hadamard gate to the first qubit
bell_circuit.h(0)

# Apply a CNOT gate with the first qubit as control and second qubit as target
bell_circuit.cx(0, 1)

# Add measurements to the circuit
bell_circuit.measure([0, 1], [0, 1])

# Visualize the circuit
print("Quantum Circuit:")
print(bell_circuit.draw())

# Number of shots
num_shots = 10000

# Simulate the circuit using the Aer simulator
simulator = Aer.get_backend('qasm_simulator')

# Transpile the circuit for optimization on the simulator backend
transpiled_circuit = transpile(bell_circuit, simulator)

# Execute the transpiled circuit on the simulator with specified number of shots
result = simulator.run(transpiled_circuit, shots=num_shots).result()

# Get measurement counts from results 
counts = result.get_counts(bell_circuit)
print("\nMeasurement Results:")
print(counts)

# Plot histogram using Matplotlib's built-in function for histograms in Qiskit visualization module.
plot_histogram(counts)
plt.title('Bell State Measurement Results')
plt.ylabel('Counts')
plt.show()
