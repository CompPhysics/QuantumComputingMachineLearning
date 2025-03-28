import numpy as np

# Define basic quantum gates
def hadamard(n):
   """Creates an n-qubit Hadamard gate."""
   H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
   return np.linalg.matrix_power(np.kron(H, np.eye(2 ** (n - 1))), 1)

def phase_shift(theta):
   """Single-qubit phase shift gate U = e^(2πiθ)."""
   return np.array([[1, 0], [0, np.exp(2j * np.pi * theta)]])

def controlled_U(U, n):
   """Controlled-U gate for n qubits."""
   # Identity for control 0
   dim = 2 ** (n + 1)
   CU = np.eye(dim, dtype=complex)
   # Apply U when control is 1
   for i in range(2 ** n):
       if (i >> (n - 1)) & 1:  # Check if the last qubit is 1
           CU[i, i] = U[1, 1]
   return CU

def inverse_qft(n):
   """Inverse Quantum Fourier Transform (QFT†)."""
   dim = 2 ** n
   QFT = np.zeros((dim, dim), dtype=complex)
   omega = np.exp(-2j * np.pi / dim)
   for i in range(dim):
       for j in range(dim):
           QFT[i, j] = omega ** (i * j) / np.sqrt(dim)
   return QFT

# Initialize parameters
t = 4  # Number of counting qubits (precision)
theta = 0.3125  # Phase we are trying to estimate (0.3125 = 5/16)
n_qubits = t + 1  # Total qubits (t counting + 1 target)

# Create the initial state |0...0⟩|1⟩
state_dim = 2 ** n_qubits
state = np.zeros(state_dim, dtype=complex)
state[-1] = 1  # |0...01⟩

# Create Hadamard on counting qubits
for i in range(t):
   H_i = np.eye(2 ** i) if i != 0 else 1
   H = np.kron(H_i, np.array([[1, 1], [1, -1]]) / np.sqrt(2))
   H = np.kron(H, np.eye(2 ** (t - i - 1)))
   state = H @ state

# Apply controlled U gates with increasing powers
U = phase_shift(theta)
for i in range(t):
   CU = controlled_U(np.linalg.matrix_power(U, 2 ** i), t)
   state = CU @ state

# Apply inverse QFT on counting qubits
QFT_inv = inverse_qft(t)
QFT_inv = np.kron(QFT_inv, np.eye(2))  # Do not apply on target qubit
state = QFT_inv @ state

# Measure probabilities
probabilities = np.abs(state) ** 2
most_likely_state = np.argmax(probabilities)
binary_result = bin(most_likely_state >> 1)[2:].zfill(t)
estimated_phase = int(binary_result, 2) / (2 ** t)

# Output results
print(f"Estimated Phase (binary): {binary_result}")
print(f"Estimated Phase (decimal): {estimated_phase}")
print(f"Actual Phase: {theta}")
