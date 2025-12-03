import numpy as np
from numpy import linalg
from scipy.linalg import expm

# Step 1: Define A and b, ensure A is Hermitian and scale its eigenvalues.
A = np.array([[4, 1],
              [1, 3]], dtype=complex)      # Example 2×2 Hermitian matrix A.
b = np.array([1, 0], dtype=complex)         # Example right-hand side vector.

# Check Hermiticity of A; embed in 2N×2N if needed (per [23†L105-L113]).
if not np.allclose(A.conj().T, A):
    # Build block matrix [ 0 A; A† 0 ] (not needed here since A is already Hermitian)
    n = A.shape[0]
    A_big = np.zeros((2*n, 2*n), dtype=complex)
    A_big[:n, n:] = A
    A_big[n:, :n] = A.conj().T
    A = A_big
    b = np.concatenate([b, np.zeros(n)])  # Expand b with zeros for the new block
    # (Now solve extended system [A 0;0 A†][x;0] = [b;0], so solution x is original.)

# Normalize |b> as a quantum state: 
b_norm = np.linalg.norm(b)
if b_norm == 0:
    raise ValueError("The input vector b must be nonzero.")
b = b / b_norm

# Scale A so that its eigenvalues lie in [0,2π] (one can choose a scaling factor).
# As in [23], we multiply A by 2π/max_eig to fit range. This does not change the solution up to normalization.
eigvals = np.linalg.eigvals(A)
max_eig = np.max(np.abs(eigvals))
A = A * (2*np.pi / max_eig)

# Step 2: Prepare registers for QPE. We use T time-steps (clock register size).
T = 8  # number of time steps (e.g. clock register of 3 qubits since 2^3=8)
# Initialize the clock register in uniform superposition (state |psi>).
psi_clock = np.ones(T) / np.sqrt(T)  # uniform amplitudes for simplicity
# Form the initial state |psi>_clock ⊗ |b>_memory:
registers = np.kron(psi_clock, b)     # shape = (T * dim(A),)

# Build block-diagonal evolution operators U^i = exp(i*A*t_i) for each clock time i.
H_blocks = np.array([expm(1j * A * i * (1.0/T)) for i in range(T)])
# (Here time step = (2π * i / T) is effectively included by scaling A earlier.)

# Apply Hamiltonian evolution to each block of the registers.
n = b.size
state = np.zeros((T, n), dtype=complex)
registers_reshaped = registers.reshape((T, n))
for i in range(T):
    # Evolve the memory state by U^i = exp(i*A*i/T) for clock index i.
    state[i] = H_blocks[i] @ registers_reshaped[i]

# Simulate QFT (the inverse of QPE's Fourier step) on the clock register using FFT.
# (We use numpy's FFT to simulate perfect QFT [oai_citation:10‡mediatum.ub.tum.de](https://mediatum.ub.tum.de/doc/1610762/4rcz2u1o0h61wji7mimto382h.BA-Kotil.pdf#:~:text=Listing%205,state%20stores%20the%20following%3A%20%EF%A3%AB).)
state = np.fft.fft(state, axis=0, norm="ortho")

# Step 3: Controlled rotation on ancilla.
# We will simulate the effect of applying a controlled R_y rotation that attaches |1>_ancilla with amplitude ~1/λ.
# For each clock index i, identify the corresponding eigenvalue. Here we approximate by finding the peaks.
probs = np.linalg.norm(state, axis=1)         # probability weights for clock indices
indices = np.argsort(probs)[::-1][:2]        # pick two largest components (for 2 eigenvalues)
eig_vals = np.sort(np.linalg.eigvals(A))[::-1]  # sorted eigenvalues of A

# Map the two most likely indices to the two eigenvalues (simple heuristic for demonstration).
mapping = {indices[0]: eig_vals[0], indices[1]: eig_vals[1]}

# Controlled rotation constant c (must be ≤ min(|λ_i|) to keep sqrt positive in [23†L168-L170]).
c = 0.1  # Example chosen c < min(eigenvalues).
one_state = np.zeros((T, n), dtype=complex)
for i in range(T):
    lam = mapping.get(i, None)
    if lam is not None and abs(c) <= abs(lam):
        # Apply rotation factor c/λ to the |1> component (simulate amplitude multiplication).
        one_state[i] = (c/lam) * state[i]
    else:
        one_state[i] = 0

# Step 4: Inverse QPE (Undo QFT and Hamiltonian).
# Inverse FFT on clock register:
one_state = np.fft.ifft(one_state, axis=0, norm="ortho")
# Apply inverse Hamiltonian evolution (U^† = exp(-iA i/T)).
for i in range(T):
    one_state[i] = H_blocks[i].conj().T @ one_state[i]

# Step 5: Extract the solution vector from the state.
# After inverse QPE, the state is |psi_clock>⊗|x> (up to amplitude). We choose a clock index i to read out.
# In practice one picks the index with significant amplitude; here we take indices[0] (highest weight).
chosen = indices[0]
# Undo the entanglement: divide by the original clock amplitude psi_clock[i] and c factor:
x_state = one_state[chosen] / (psi_clock[chosen] * c)
# Normalize the extracted solution state:
x_state /= np.linalg.norm(x_state)

print("Simulated solution state |x> ≈", np.real_if_close(x_state))
# For comparison, compute the classical solution (normalized):
x_classical = np.linalg.inv(A) @ b
x_classical /= np.linalg.norm(x_classical)
print("Classical solution (normalized) =", x_classical.real)

