import numpy as np

# Define 1-qubit Pauli matrices
I = np.array([[1,0],[0,1]], dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)

def get_neighbors(L):
    """List of nearest-neighbor pairs (i,j) for an LxL square lattice (open boundaries)."""
    neighbors = []
    for i in range(L):
        for j in range(L):
            idx = i*L + j
            if j < L-1:
                neighbors.append((idx, idx+1))
            if i < L-1:
                neighbors.append((idx, idx+L))
    return neighbors

def build_ising_hamiltonian(L, J, h):
    """Construct the 2^N x 2^N transverse-field Ising Hamiltonian for LxL spins."""
    N = L*L
    dim = 2**N
    H = np.zeros((dim, dim), dtype=complex)
    for (i,j) in get_neighbors(L):
        # Build tensor-product for -J * Z_i Z_j
        ops = [Z if k==i or k==j else I for k in range(N)]
        term = ops[0]
        for op in ops[1:]:
            term = np.kron(term, op)
        H += -J * term
    for i in range(N):
        # -h * X_i term
        ops = [X if k==i else I for k in range(N)]
        term = ops[0]
        for op in ops[1:]:
            term = np.kron(term, op)
        H += -h * term
    return H

def build_heisenberg_hamiltonian(L, J):
    """Construct the 2^N x 2^N isotropic Heisenberg Hamiltonian for LxL spins."""
    N = L*L
    dim = 2**N
    H = np.zeros((dim, dim), dtype=complex)
    for (i,j) in get_neighbors(L):
        # Add J*(X_iX_j + Y_iY_j + Z_iZ_j)
        for P in (X, Y, Z):
            ops = [P if k==i or k==j else I for k in range(N)]
            term = ops[0]
            for op in ops[1:]:
                term = np.kron(term, op)
            H += J * term
    return H

def apply_RY(state, qubit, theta):
    """Apply a single-qubit RY(theta) rotation on the specified qubit of the state."""
    N = int(np.log2(len(state)))
    new_state = np.zeros_like(state, dtype=complex)
    cos = np.cos(theta/2); sin = np.sin(theta/2)
    # Loop over basis states in binary; if qubit bit is 0, mix with its partner state
    for b in range(len(state)):
        if ((b >> qubit) & 1) == 0:
            partner = b | (1 << qubit)     # flip that qubit bit
            new_state[b]     += cos * state[b] - sin * state[partner]
            new_state[partner] += sin * state[b] + cos * state[partner]
    return new_state

def apply_CNOT(state, control, target):
    """Apply a CNOT gate (control -> target) on the specified qubits of the state."""
    new_state = np.zeros_like(state, dtype=complex)
    for b in range(len(state)):
        # If control bit is 1, flip the target bit
        if ((b >> control) & 1) == 1:
            b_new = b ^ (1 << target)
        else:
            b_new = b
        new_state[b_new] += state[b]
    return new_state

def ansatz_state(params, L):
    """Construct the variational ansatz state for parameters `params` on an LxL lattice."""
    N = L*L
    psi = np.zeros(2**N, dtype=complex)
    psi[0] = 1.0     # start in |00...0>
    # Apply RY on each qubit with corresponding parameter
    for i, theta in enumerate(params):
        psi = apply_RY(psi, i, theta)
    # Apply CNOT entanglers on each neighbor pair
    for (i,j) in get_neighbors(L):
        psi = apply_CNOT(psi, i, j)
    return psi

import scipy.optimize as opt

def energy_expectation(params, H, L):
    """Compute expectation <psi(params)|H|psi(params)> for Hamiltonian H."""
    psi = ansatz_state(params, L)
    # <psi|H|psi> using complex conjugate transpose
    return np.vdot(psi, H.dot(psi)).real

# Example: find ground energy of 2x2 transverse Ising (J=1.0, h=0.5)
L = 2
J = 1.0
h = 0.5
H_ising = build_ising_hamiltonian(L, J, h)
# Initial random parameters
np.random.seed(0)
init_params = 0.1 * np.random.randn(L*L)
# Optimize parameters to minimize energy
result = opt.minimize(lambda th: energy_expectation(th, H_ising, L),
                      init_params, method='COBYLA')
estimated_energy = result.fun



import matplotlib.pyplot as plt

# Example: Vary h for 2x2 Ising, J=1.0
hs = np.linspace(0.0, 2.0, 5)
energies_vqe = []
for h_val in hs:
    H2 = build_ising_hamiltonian(2, 1.0, h_val)
    res = opt.minimize(lambda th: energy_expectation(th, H2, 2),
                       init_params, method='COBYLA')
    energies_vqe.append(res.fun)

plt.plot(hs, energies_vqe, marker='o')
plt.xlabel('Transverse field h')
plt.ylabel('Ground-state energy (VQE)')
plt.title('2x2 Transverse-Field Ising (J=1)')
plt.show()

