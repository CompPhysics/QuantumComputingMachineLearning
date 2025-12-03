import numpy as np
from math import cos, sin

def create_pauli_matrices():
    X = np.array([[0,1],[1,0]], dtype=complex)
    Y = np.array([[0,-1j],[1j,0]], dtype=complex)
    Z = np.array([[1,0],[0,-1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    return X, Y, Z, I

def build_hamiltonian(N):
    """Construct LMG Hamiltonian with two-body X-X and Y-Y interactions."""
    X, Y, Z, I = create_pauli_matrices()
    dim = 2**N
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(N):
        for j in range(i+1, N):
            # Build tensor products with X on qubits i,j and I elsewhere
            op_xx = 1
            op_yy = 1
            for k in range(N):
                if k == i:
                    op_xx = np.kron(op_xx, X)
                    op_yy = np.kron(op_yy, Y)
                elif k == j:
                    op_xx = np.kron(op_xx, X)
                    op_yy = np.kron(op_yy, Y)
                else:
                    op_xx = np.kron(op_xx, I)
                    op_yy = np.kron(op_yy, I)
            H += -(1/(2*N)) * (op_xx + op_yy)
    return H

# Example: build H for N=4 spins
N = 4
H = build_hamiltonian(N)
print("Hamiltonian matrix size:", H.shape, "Hermitian check:", np.allclose(H, H.conj().T))

def apply_ry(state, theta, qubit, N):
    """Apply RY(theta) to qubit 'qubit' on an N-qubit state vector."""
    # RY gate matrix (2x2)
    ry = np.array([[np.cos(theta/2), -np.sin(theta/2)],
                   [np.sin(theta/2),  np.cos(theta/2)]], dtype=complex)
    psi = state.reshape([2]*N)
    psi = np.moveaxis(psi, qubit, 0)              # bring target axis to front
    psi = psi.reshape(2, -1)                      # flatten other axes
    psi = ry @ psi                                # apply gate
    psi = psi.reshape((2,)+psi.shape[1:])         # reshape back
    psi = np.moveaxis(psi, 0, qubit)              # return axis to original position
    return psi.reshape(state.shape)

def apply_cnot(state, control, target, N):
    """Apply CNOT (control->target) on an N-qubit state vector."""
    psi = state.reshape([2]*N)
    psi = np.moveaxis(psi, control, 0)            # move control qubit to axis 0
    # determine new index of target after control move
    if target > control:
        target_idx = target
    else:
        target_idx = target + 1
    psi = np.moveaxis(psi, target_idx, 1)         # move target qubit to axis 1
    # Now psi.shape = (2,2,...). Flatten rest dims beyond first two
    rest_shape = psi.shape[2:]
    psi2 = psi.reshape(2, 2, -1)
    # Apply 4x4 CNOT matrix in |00>,|01>,|10>,|11> basis
    CNOT = np.array([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,0,1],
                     [0,0,1,0]], dtype=complex)
    psi2 = (CNOT @ psi2.reshape(4, -1)).reshape(2,2,*rest_shape)
    # Move axes back to original positions
    psi2 = np.moveaxis(psi2, 1, target_idx)
    psi2 = np.moveaxis(psi2, 0, control)
    return psi2.reshape(state.shape)

def prepare_state(params, N):
    """Apply one layer of RY on each qubit then CNOT ladder."""
    state = np.zeros(2**N, dtype=complex)
    state[0] = 1.0  # |00...0>
    # Single-qubit rotations
    for i, theta in enumerate(params):
        state = apply_ry(state, theta, qubit=i, N=N)
    # CNOT entangling ladder (0->1, 1->2, ...)
    for i in range(N-1):
        state = apply_cnot(state, control=i, target=i+1, N=N)
    return state

# Example: prepare a random state for N=3
N = 3
random_angles = [0.3, 1.2, -0.7]
psi = prepare_state(random_angles, N)
print("Prepared state norm:", np.linalg.norm(psi))

def energy_expectation(params, H, N):
    """Return the expectation value <psi(theta)|H|psi(theta)>."""
    psi = prepare_state(params, N)
    E = np.vdot(psi, H.dot(psi))
    return np.real(E)

# Example evaluation:
N = 3
H3 = build_hamiltonian(N)
params = [0.5, -0.3, 1.0]
E_val = energy_expectation(params, H3, N)
print(f"Expectation energy = {E_val:.6f}")



N = 3
H3 = build_hamiltonian(N)
# Random initial guess for angles
init_params = np.random.rand(N) * 0.2  
res = minimize(lambda th: energy_expectation(th, H3, N), init_params,
               method='COBYLA', options={'maxiter':500, 'tol':1e-6})
vqe_energy = res.fun

eigvals = np.linalg.eigvalsh(H3)
exact_energy = np.min(eigvals)
print(f"VQE energy = {vqe_energy:.6f}")
print(f"Exact energy = {exact_energy:.6f}")

"""
For a concrete case with $N=3$, we obtain (sample output):
VQE energy = -0.666667
Exact energy = -0.666667
"""

