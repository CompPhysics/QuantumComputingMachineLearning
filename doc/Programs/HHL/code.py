import numpy as np
from scipy.linalg import expm

# --- Problem setup: Define A (2x2 Hermitian) and b (normalized) ---
P = np.array([[1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), -1/np.sqrt(2)]], dtype=complex)

# Diagonal eigenvalues 0.25 and 0.75
D = np.diag([0.25, 0.75])
A = P @ D @ P.conj().T
b = np.array([1.0, 2.0], dtype=complex)
b = b / np.linalg.norm(b)

# Classical solution (for comparison, unnormalized)
x_classical = np.linalg.solve(A, b)

# Check Hermitian and eigenvalues (sanity)
assert np.allclose(A, A.conj().T), "A must be Hermitian"
eigvals, eigvecs = np.linalg.eigh(A)
print("Eigenvalues of A:", eigvals)

# --- Compute and display the classical solution ---
print("Classical solution x (unnormalized):", x_classical)

# Solve Ax = b classically for comparison
x_classical = np.linalg.solve(A, b)
print("Classical solution (unnormalized):", x_classical)

# Build the initial quantum state |00>_phase ⊗ |b>_main ⊗ |0>_ancilla
phase0 = np.zeros(4, dtype=complex);
phase0[0] = 1.0  # |00> in 2-qubit space (dim=4)
anc0 = np.array([1.0+0j, 0+0j]) # single-qubit |0>
state = np.kron(phase0, np.kron(b, anc0))

# Apply H on both phase qubits: (H⊗H⊗I⊗I) on 4-qubit state
H = np.array([[1,1],[1,-1]], dtype=complex)/np.sqrt(2)
I2 = np.eye(2, dtype=complex)
U_hadamard = np.kron(np.kron(H,H), np.kron(I2, I2))
state = U_hadamard @ state
# Controlled U-rotations: construct U = exp(2πi A) and U^2 = (exp(2πi A))^2
U = expm(2j * np.pi * A)
U2 = expm(2j * np.pi * A * 2) # which equals U^2

# Controlled-U^2 by phase qubit 0 (MSB): if bit0=1 apply U2 on main
P0 = np.array([[1,0],[0,0]], dtype=complex); P1 = np.array([[0,0],[0,1]], dtype=complex)
CU2 = np.kron(P0, np.kron(I2, np.kron(I2, I2))) + np.kron(P1, np.kron(I2, np.kron(U2, I2)))
state = CU2 @ state

# Controlled-U^1 by phase qubit 1 (LSB): if bit1=1 apply U on main
CU1 = np.kron(np.kron(I2, P0), np.kron(I2, I2)) + np.kron(np.kron(I2, P1), np.kron(U, I2))
state = CU1 @ state
# Inverse QFT on 2 phase qubits (4x4 FFT matrix, conjugate-transpose of QFT)
N = 4
F = np.zeros((N,N), complex)
for k in range(N):
    for l in range(N):
        F[k,l] = np.exp(2j*np.pi*k*l/N)
F = F/np.sqrt(N)
F_dag = F.conj().T
IQFT = np.kron(F_dag, np.kron(I2, I2))
state = IQFT @ state

# Compute eigenvalues and rotation angles
eigvals, _ = np.linalg.eigh(A)
lam1, lam2 = eigvals
C = lam1 # smallest eigenvalue = 0.25
theta1 = 2*np.arcsin(C/lam1)
theta2 = 2*np.arcsin(C/lam2)

# Rotation matrices on ancilla
def R_y(angle):
    return np.array([[np.cos(angle/2), -np.sin(angle/2)],[np.sin(angle/2), np.cos(angle/2)]], dtype=complex)
R1 = R_y(theta1)
R2 = R_y(theta2)

# Build controlled-Ry: project phase states and apply R_y or I
rot_op = np.zeros((16,16), complex)
for k in range(4):
    # Projector |k><k| on 2-qubit phase register
    e = np.zeros(4); e[k]=1
    Pk = np.outer(e,e)
    if k == 1:
        rot_op += np.kron(Pk, np.kron(I2, R1))
    elif k == 3:
        rot_op += np.kron(Pk, np.kron(I2, R2))
    else:
        rot_op += np.kron(Pk, np.kron(I2, I2))
state = rot_op @ state

# Forward QFT on phase
QFT = np.kron(F, np.kron(I2, I2))
state = QFT @ state

# Controlled U^{-1}: U_inv = U†, U2_inv = (U2)†
U_inv = U.conj().T
U2_inv = U2.conj().T
CU1_inv = np.kron(np.kron(I2, P0), np.kron(I2, I2)) + np.kron(np.kron(I2, P1), np.kron(U_inv, I2))
state = CU1_inv @ state
CU2_inv = np.kron(P0, np.kron(I2, np.kron(I2, I2))) + np.kron(P1, np.kron(I2, np.kron(U2_inv, I2)))
state = CU2_inv @ state

# Hadamards on phase qubits to complete inverse QPE
state = np.kron(np.kron(H,H), np.kron(I2,I2)) @ state

# Extract main register state for ancilla = 1
final_state = state
main_dim = 2
solution_state = np.zeros(main_dim, complex)
for idx, amp in enumerate(final_state):
    anc_bit = idx % 2
    if anc_bit == 1:
        # main index is bit 2 of idx in our ordering
        main_index = (idx // 2) % 2
        solution_state[main_index] += amp

# Normalize the resulting main register state
solution_state /= np.linalg.norm(solution_state)
print("HHL result state |x>:", np.round(solution_state, 3))
print("Normalized classical x:", np.round(x_classical/np.linalg.norm(x_classical), 3))

