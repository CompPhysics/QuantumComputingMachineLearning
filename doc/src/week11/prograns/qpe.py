import numpy as np

def hadamard(n):
   """Creates an n-qubit Hadamard gate as a matrix."""
   H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
   H_n = H
   for _ in range(n - 1):
       H_n = np.kron(H_n, H)  # Tensor product to expand Hadamard gate
   return H_n

def qft(n):
   """Creates an n-qubit Quantum Fourier Transform (QFT) matrix."""
   N = 2**n
   omega = np.exp(2j * np.pi / N)
   qft_matrix = np.array([[omega**(i * j) for j in range(N)] for i in range(N)]) / np.sqrt(N)
   return qft_matrix

def inverse_qft(n):
   """Creates an n-qubit inverse Quantum Fourier Transform (QFT†) matrix."""
   return np.conj(qft(n)).T  # Hermitian transpose of QFT

def controlled_unitary(U, control, target, n):
   """Creates an n-qubit controlled unitary matrix."""
   I = np.eye(2**n)  # Identity matrix
   CU = np.copy(I)
   for i in range(2**n):
       if (i >> control) & 1:  # Check if control qubit is |1⟩
           CU[i, :] = np.kron(np.eye(2**target), U).dot(I[i, :])
   return CU

def apply_gate(state, gate):
   """Applies a gate (matrix) to a quantum state (vector)."""
   return gate @ state

def measure(state):
   """Simulates measurement by computing probability distribution."""
   probabilities = np.abs(state) ** 2
   return np.argmax(probabilities)  # Return the most probable outcome

def qpe(unitary, phi, num_counting_qubits):
   """Simulates the Quantum Phase Estimation algorithm."""
   n = num_counting_qubits
   total_qubits = n + 1
   dim = 2**total_qubits

   # Step 1: Initialize state |0...0⟩ ⊗ |ψ⟩
   state = np.zeros(dim, dtype=complex)
   state[0] = 1  # |00...0⟩

   # Step 2: Apply Hadamard to counting qubits
   H_n = hadamard(n)
   state = apply_gate(state.reshape(2**n, 2), H_n).reshape(dim)

   # Step 3: Apply controlled-U^2^j operations
   for j in range(n):
       power = 2**j
       U_power = np.linalg.matrix_power(unitary, power)
       CU = controlled_unitary(U_power, j, n, total_qubits)
       state = apply_gate(state, CU)

   # Step 4: Apply inverse QFT
   IQFT = inverse_qft(n)
   state = apply_gate(state.reshape(2**n, 2), IQFT).reshape(dim)

   # Step 5: Measure and return estimated phase
   measurement_result = measure(state)
   return measurement_result / (2**n)  # Convert binary to decimal

# Define the unitary U with phase φ = 1/3
phi = 1/3
U = np.array([[1, 0], [0, np.exp(2j * np.pi * phi)]])  # Phase gate

# Run QPE
num_counting_qubits = 3  # More qubits give higher precision
estimated_phi = qpe(U, phi, num_counting_qubits)

print(f"Estimated phase: {estimated_phi}")
print(f"Actual phase: {phi}")
