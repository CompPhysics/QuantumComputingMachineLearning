import numpy as np
from scipy.optimize import minimize

# Pauli matrices (2x2) and 2x2 identity
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)

def tensor_op(op_list):
    """Compute the tensor product of a list of operators."""
    result = op_list[0]
    for op in op_list[1:]:
        result = np.kron(result, op)
    return result


def heisenberg_xyz_hamiltonian(N, Jx, Jy, Jz, periodic=False):
    """Construct the Heisenberg XYZ Hamiltonian for N qubits."""
    dim = 2**N
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(N):
        # Determine neighbor index (periodic or open chain)
        j = (i+1) % N if periodic else i+1
        if j >= N: 
            continue
        # Add Jx * X_i X_j
        ops = [I]*N
        ops[i], ops[j] = X, X
        H += Jx * tensor_op(ops)
        # Add Jy * Y_i Y_j
        ops = [I]*N
        ops[i], ops[j] = Y, Y
        H += Jy * tensor_op(ops)
        # Add Jz * Z_i Z_j
        ops = [I]*N
        ops[i], ops[j] = Z, Z
        H += Jz * tensor_op(ops)
    return H

def apply_single_qubit_gate(state, gate, qubit, N):
    """Apply a single-qubit gate to 'state' on the given qubit index."""
    # Build full operator by tensoring gate at qubit, identity elsewhere
    ops = [I]*N
    ops[qubit] = gate
    full_op = tensor_op(ops)
    return full_op.dot(state)


def apply_cnot(state, control, target, N):
    """Apply a CNOT to the state vector (control->target)."""
    new_state = np.zeros_like(state)
    for index, amp in enumerate(state):
        if amp == 0:
            continue
        # Check if control qubit is |1> in basis index
        if (index >> control) & 1:
            # Flip the target bit
            new_index = index ^ (1 << target)
            new_state[new_index] += amp
        else:
            # Leave state unchanged if control is |0>
            new_state[index] += amp
    return new_state


def rotation_y(theta):
    """Return the 2x2 R_y rotation matrix for angle theta."""
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                     [np.sin(theta/2),  np.cos(theta/2)]], dtype=complex)

def ansatz_state(params, N, layers):
    """Prepare the ansatz state |Psi(theta)> from parameters (initial |00..0>)."""
    # Start in |00..0>
    state = np.zeros(2**N, dtype=complex)
    state[0] = 1.0
    # Apply each layer of rotations and entanglers
    for layer in range(layers):
        # Apply R_y on each qubit
        for qubit in range(N):
            gate = rotation_y(params[layer*N + qubit])
            state = apply_single_qubit_gate(state, gate, qubit, N)
        # Entangle chain: CNOT between i->i+1
        for qubit in range(N-1):
            state = apply_cnot(state, qubit, qubit+1, N)
    return state


# Example parameters
N = 4  # number of qubits
Jx, Jy, Jz = 1.0, 1.0, 1.0
H = heisenberg_xyz_hamiltonian(N, Jx, Jy, Jz, periodic=False)

layers = 4  # number of ansatz layers
param_size = layers * N
# Random initial parameters in [0,2π)
initial_params = np.random.rand(param_size) * 2 * np.pi

def expectation_value(params):
    """Compute <Psi(params)| H |Psi(params)>."""
    psi = ansatz_state(params, N, layers)
    return np.real(np.vdot(psi, H.dot(psi)))

# Classical optimization: find parameters that minimize the energy
result = minimize(expectation_value, initial_params, method='COBYLA')
vqe_energy = result.fun


# Exact diagonalization for ground-state energy
eigvals = np.linalg.eigvalsh(H)
exact_energy = np.min(eigvals)

print("VQE estimated ground energy =", vqe_energy)
print("Exact ground energy =", exact_energy)
