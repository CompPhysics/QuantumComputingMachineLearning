import numpy as np
def qft(n):

   # Inputs n qubits and returns QFT matrix of size (2^n, 2^n)
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
   n = int(np.log2(len(state)))  # Number of qubits
   qft_matrix = qft(n)
   return qft_matrix @ state
# Example: 3-qubit system
n_qubits = 4  # QFT on 3 qubits (8 dimensions)
dim = 2 ** n_qubits

# Define an example quantum state (|5⟩ in computational basis)
state = np.zeros(dim, dtype=complex)
state[0] = 1  # |5⟩ = [0,0,0,0,0,1,0,0]

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
