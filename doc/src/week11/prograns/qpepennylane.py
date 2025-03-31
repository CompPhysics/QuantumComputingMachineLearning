import pennylane as qml
import numpy as np

def qft(n):
   """Applies the inverse Quantum Fourier Transform (QFT†) circuit."""
   for i in range(n):
       qml.Hadamard(wires=i)
       for j in range(i):
           qml.CPhase(-np.pi / (2 ** (i - j)), wires=[j, i])
   for i in range(n // 2):
       qml.SWAP(wires=[i, n - i - 1])

def controlled_unitary(U, control, target):
   """Applies a controlled-unitary operation."""
   qml.ctrl(U, control=control)(wires=target)


def qpe(phi, num_counting_qubits):
   """Quantum Phase Estimation circuit in PennyLane."""
   total_qubits = num_counting_qubits + 1

   dev = qml.device("default.qubit", wires=total_qubits, shots=1000)

   @qml.qnode(dev)
   def circuit():
       # Initialize the counting register in |0> and eigenstate register in |1>
       qml.PauliX(wires=num_counting_qubits)  # Set last qubit to |1>

       # Apply Hadamard to counting qubits
       for qubit in range(num_counting_qubits):
           qml.Hadamard(wires=qubit)

       # Apply controlled-U^2^j operations
       for j in range(num_counting_qubits):
           power = 2**j
           U = qml.RZ(2 * np.pi * phi, wires=num_counting_qubits)  # Phase shift
           controlled_unitary(U, control=j, target=num_counting_qubits)

       # Apply inverse QFT
       qft(num_counting_qubits)

       # Measure counting qubits
       return qml.sample(wires=range(num_counting_qubits))

   # Run the circuit
   samples = circuit()

   # Convert measurement results to decimal phase estimate
   binary_result = "".join(map(str, samples[0]))  # Take the first sample
   estimated_phi = int(binary_result, 2) / (2 ** num_counting_qubits)

   return estimated_phi

# Define the phase φ
phi = 1/3

# Number of counting qubits (higher gives better precision)
num_counting_qubits = 3

# Run QPE
estimated_phi = qpe(phi, num_counting_qubits)

print(f"Estimated phase: {estimated_phi}")
print(f"Actual phase: {phi}")
