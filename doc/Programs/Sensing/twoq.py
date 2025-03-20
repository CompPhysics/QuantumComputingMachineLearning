import numpy as np
import matplotlib.pyplot as plt

# Define parameters
B = 1.0  # Strength of the magnetic field (in arbitrary units)
omega = B  # Frequency associated with the magnetic field

# Create basis states |00>, |01>, |10>, |11>
basis_00 = np.array([1, 0, 0, 0])   # |00>
basis_01 = np.array([0, 1, 0, 0])   # |01>
basis_10 = np.array([0, 0, 1, 0])   # |10>
basis_11 = np.array([0, 0, 0, 1])   # |11>

# Create an entangled state (Bell state) |Φ+> = (|00> + |11>) / sqrt(2)
entangled_state = (basis_00 + basis_11) / np.sqrt(2)

# Define Pauli matrices
sigma_z = np.array([[1, 0], [0, -1]])
identity = np.eye(2)

# Constructing the Hamiltonian H = -ω/2 * (σz ⊗ I + I ⊗ σz)
H_z_I = -omega / 2 * np.kron(sigma_z, identity)    # σz ⊗ I
I_H_z = -omega / 2 * np.kron(identity, sigma_z)    # I ⊗ σz
H = H_z_I + I_H_z                                   # Total Hamiltonian

# Time evolution parameters
t_list = np.linspace(0, 10, num=100)               # Time from t=0 to t=10
dt = t_list[1] - t_list[0]

def evolve(state):
    """Evolve quantum state under Hamiltonian."""
    return np.dot(np.linalg.expm(-1j * H * dt), state)

# Evolve the initial state over time
result_states = []
current_state = entangled_state.copy()

for _ in t_list:
    result_states.append(current_state)
    current_state = evolve(current_state)

# Calculate probabilities of measuring specific states: P(|00>) and P(|11>)
probabilities_00 = [abs(np.dot(basis_00.conj(), state))**2 for state in result_states]
probabilities_11 = [abs(np.dot(basis_11.conj(), state))**2 for state in result_states]

# Output results
print("Probabilities of measuring |00>: ", probabilities_00)
print("Probabilities of measuring |11>: ", probabilities_11)

# Plotting results using matplotlib
plt.plot(t_list, probabilities_00,label='Probability of |00>')
plt.plot(t_list, probabilities_11,label='Probability of |11>')
plt.xlabel('Time')
plt.ylabel('Probability')
plt.title('Quantum Sensing Simulation')
plt.legend()
plt.show()

"""
### Explanation:
- **Basis States:** We define four basis states corresponding to two qubits.
- **Entanglement Creation:** The Bell state \(|\Phi^+\rangle\) is created manually.
- **Hamiltonian Definition:** The Hamiltonian \(H\) combines contributions from both qubits affected by a static magnetic field.
- **State Evolution:** A function `evolve` computes the new quantum state at each time step using matrix exponentiation to apply the unitary operator derived from the Hamiltonian.
- **Measurement Probabilities:** Finally we compute measurement probabilities for \(|00\rangle\) and \(|11\rangle\).
"""
