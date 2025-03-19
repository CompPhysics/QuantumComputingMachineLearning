import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

# Define single-qubit operations, Identity, Pauli, Hadamard and S matrices
Id = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
Had = np.array([[1, 1],[1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0],[0, 1j]], dtype=complex)
# Define two-qubit gates
CNOT01 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
CNOT10 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)

times = np.linspace(0, 1, 100)  # Time range from 0 to 1 seconds

# Initial state for each qubit
psi_1 =  np.array([1, 0], dtype=complex) # start with |0> for qubits 1 and 2
psi_2 =  np.array([1, 0], dtype=complex) # start with |0> for qubits 1 and 2
# possible basis states for measurements
basis_00 = np.array([1, 0, 0, 0], dtype=complex)
basis_01 = np.array([0, 1, 0, 0], dtype=complex)
basis_10 = np.array([0, 0, 1, 0], dtype=complex)
basis_11 = np.array([0, 0, 0, 1], dtype=complex)
# then act with Hadamard on first qubit only
psi_1 = Had @ psi_1
# Initial two-qubit state
Psi_0 = np.kron(psi_1,psi_2)
# Define parameters
# Then we act on this state in order to get a Bell state 1/sqrt(2)(|00>+|11)
Psi_0 = CNOT01 @ Psi_0
B = 1.0  # Strength of the magnetic field (in arbitrary units)
omega = B  # Frequency associated with the magnetic field
# Constructing the Hamiltonian H = -omega/2 * (Z * I + I * Z)
H_z_I = -omega / 2 * np.kron(Z, Id)  # Z * I
I_H_z = -omega / 2 * np.kron(Id, Z)    # I * Z
# Total Hamiltonian, try a more complicated one
H = H_z_I + I_H_z   
# Lists to store expectation values
expect_00, expect_11 = [], []
for t in times:
   # Calculate time evolution operator
   U = expm(-1j * H * t)
   # Evolve the initial state
   Psi_t = U @ Psi_0
   # Calculate probabilities of measuring specific states: P(|00>) and P(|11>)
   # Calculate expectation values
   expect_00.append(abs(np.dot(basis_00.conj(), Psi_t))**2)
   expect_11.append(abs(np.dot(basis_11.conj(), Psi_t))**2)

# Plotting results using matplotlib
plt.figure(figsize=(10, 6))
plt.plot(times, expect_00, label='Probability |00>')
plt.plot(times, expect_11, label='Probability |11>')
plt.xlabel('Time')
plt.ylabel('Probability')
plt.title('Quantum Sensing Simulation')
plt.legend()
plt.show()
