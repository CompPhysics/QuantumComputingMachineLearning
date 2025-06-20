import numpy as np
from scipy.optimize import minimize

# --- Generate a random Hermitian Hamiltonian (8x8) ---
np.random.seed(0)
A = np.random.randn(8,8) + 1j*np.random.randn(8,8)
H = (A + A.conj().T) / 2    # Hermitian Hamiltonian

# --- Define single-qubit and two-qubit gates ---
def ry(theta):
    """2x2 rotation around Y by angle theta."""
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                     [np.sin(theta/2),  np.cos(theta/2)]], dtype=complex)

def apply_single_qubit_gate(state, gate, target, n_qubits=3):
    """Apply a single-qubit gate to 'target' qubit of the statevector."""
    # Build full operator as tensor product of identities and the gate
    op = 1
    for q in range(n_qubits):
        op = np.kron(op, gate if q==target else np.eye(2))
    return op.dot(state)

def apply_cnot(state, control, target, n_qubits=3):
    """Apply a CNOT gate with given control and target on the statevector."""
    new_state = np.zeros_like(state)
    for idx, amp in enumerate(state):
        bits = list(map(int, format(idx, f'0{n_qubits}b')))
        # If control qubit is |1>, flip the target bit
        if bits[control] == 1:
            bits[target] ^= 1
        new_idx = int("".join(str(b) for b in bits), 2)
        new_state[new_idx] += amp
    return new_state

# --- Ansatz state preparation ---
def prepare_state(params):
    """
    Prepare the 3-qubit statevector from parameters.
    Ansatz: 2 layers of Ry + chain of CNOTs.
    params: list or array of length 6 [θ0,...,θ5].
    """
    # Start in |000>
    state = np.zeros(8, dtype=complex)
    state[0] = 1.0
    
    # Layer 1: RY on each qubit 0,1,2
    for q in range(3):
        state = apply_single_qubit_gate(state, ry(params[q]), q)
    # Entangling CNOTs
    state = apply_cnot(state, control=0, target=1)
    state = apply_cnot(state, control=1, target=2)
    
    # Layer 2: another RY on each qubit
    for q in range(3):
        state = apply_single_qubit_gate(state, ry(params[3+q]), q)
    # Another round of CNOTs
    state = apply_cnot(state, control=0, target=1)
    state = apply_cnot(state, control=1, target=2)
    
    return state

# --- Energy expectation function ---
def energy_expectation(params):
    """
    Return the expectation value <psi(θ)|H|psi(θ)> for the ansatz state.
    """
    psi = prepare_state(params)
    # ⟨psi|H|psi⟩ = psi.conj().T @ H @ psi
    return np.real(np.vdot(psi, H.dot(psi)))

# --- Optimization with COBYLA and convergence tracking ---
energy_history = []  # to record energy at each iteration

def callback(params):
    """Callback to store energy at each iteration."""
    energy_history.append(energy_expectation(params))

# Random initial parameters (e.g., in [0,2π])
init_params = np.random.rand(6) * 2*np.pi
energy_history.append(energy_expectation(init_params))  # record initial energy

res = minimize(energy_expectation, init_params, method='COBYLA',
               callback=callback, options={'maxiter': 200, 'tol': 1e-6})

print("Optimization success:", res.success)
print("Minimum energy found:", res.fun)
print("Ground-state eigenvalue (True):", np.min(np.linalg.eigvalsh(H)))

# --- Display convergence curve (matplotlib can be used here) ---
import matplotlib.pyplot as plt

plt.plot(energy_history, marker='o')
plt.title("VQE Convergence (Energy vs Iteration)")
plt.xlabel("Iteration")
plt.ylabel("Energy ⟨ψ|H|ψ⟩")
plt.grid(True)
plt.show()

# --- Print ansatz structure (text) ---
print("\nAnsatz circuit (text form):")
print("  Layer 1: RY(θ0) on qubit 0, RY(θ1) on qubit 1, RY(θ2) on qubit 2")
print("           CNOT(0→1), CNOT(1→2)")
print("  Layer 2: RY(θ3) on qubit 0, RY(θ4) on qubit 1, RY(θ5) on qubit 2")
print("           CNOT(0→1), CNOT(1→2)")
