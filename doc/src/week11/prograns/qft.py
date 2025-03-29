import numpy as np
def qft(n):
   """
   Creates the Quantum Fourier Transform (QFT) matrix for n qubits.

   Parameters:
   - n: Number of qubits (QFT acts on 2^n dimensions)

   Returns:
   - QFT matrix of size (2^n, 2^n)
   """
   dim = 2 ** n  # Dimension of the Hilbert space
   omega = np.exp(2j * np.pi / dim)  # Primitive root of unity

   # Initialize QFT matrix
   QFT = np.zeros((dim, dim), dtype=complex)

   # Construct the QFT matrix
   for i in range(dim):
       for j in range(dim):
           QFT[i, j] = omega ** (i * j) / np.sqrt(dim)

   return QFT

def apply_qft(state):
   """
   Applies QFT to a given quantum state vector.

   Parameters:
   - state: Input state vector (1D NumPy array)

   Returns:
   - Transformed state vector after QFT
   """
   n = int(np.log2(len(state)))  # Number of qubits
   qft_matrix = qft(n)
   return np.dot(qft_matrix, state)

# Example: 3-qubit system
n_qubits = 3  # QFT on 3 qubits (8 dimensions)
dim = 2 ** n_qubits

# Define an example quantum state (|5⟩ in computational basis)
state = np.zeros(dim, dtype=complex)
state[5] = 1  # |5⟩ = [0,0,0,0,0,1,0,0]

print("Initial State |5⟩:", state)

# Apply QFT
qft_state = apply_qft(state)

print("\nState after QFT:")
print(np.round(qft_state, 4))  # Rounded for better readability

# Verify QFT is unitary (QFT * QFT† = I)
qft_matrix = qft(n_qubits)
identity = np.dot(qft_matrix, qft_matrix.conj().T)
print("\nQFT * QFT† = Identity Matrix (Rounded):")
print(np.round(identity, 4))
