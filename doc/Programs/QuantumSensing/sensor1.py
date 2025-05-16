import numpy as np
import matplotlib.pyplot as plt

# --- Quantum Gates and Helpers ---

I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])

def kron_n(*ops):
    """Kronecker product of a sequence of gates."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

def GHZ_state(n):
    """Create GHZ state |00..0> + |11..1>."""
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1
    state[-1] = 1
    return state / np.sqrt(2)

def evolve_magnetic(state, B_t, t, gamma, n):
    """Evolve under H = γ B(t) ΣZ_i for time t."""
    phi = gamma * B_t * t
    U = np.eye(1, dtype=complex)
    for _ in range(n):
        U = np.kron(U, expm(-1j * phi * Z / 2))
    return U @ state


def expm(matrix):
    """Matrix exponential for 2x2 Hermitian matrix."""
    eigvals, eigvecs = np.linalg.eigh(matrix)
    return eigvecs @ np.diag(np.exp(eigvals)) @ eigvecs.conj().T

def measure_in_X_basis(state, n):
    """Measure expectation value of X⊗X⊗...⊗X."""
    Xn = kron_n(*([X] * n))
    return np.real(np.vdot(state, Xn @ state))

# --- Parameters ---

n = 3  # Number of qubits (entangled sensor)
gamma = 1.0  # gyromagnetic ratio
B0 = 1.0     # field amplitude
omega = 1.0  # frequency of magnetic field
times = np.linspace(0, 5, 100)

# --- Sensing Simulation ---

expectations = []
true_fields = []

for t in times:
    B_t = B0 * np.sin(omega * t)
    true_fields.append(B_t)

    # Prepare GHZ state
    psi0 = GHZ_state(n)

    # Evolve under collective B field
    psi_t = evolve_magnetic(psi0, B_t, t, gamma, n)

    # Apply inverse GHZ preparation (H and CNOTs)
    # For ideal simulation, we just measure in X^⊗n basis
    expX = measure_in_X_basis(psi_t, n)
    expectations.append(expX)

plt.figure(figsize=(10, 5))
plt.plot(times, true_fields, label='B(t) = sin(ωt)', color='orange')
plt.plot(times, expectations, label=r'$\langle X^{\otimes n} \rangle$', color='blue')
plt.xlabel("Time")
plt.ylabel("Field / Expectation")
plt.title(f"Quantum Sensor with {n}-Qubit GHZ State")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
