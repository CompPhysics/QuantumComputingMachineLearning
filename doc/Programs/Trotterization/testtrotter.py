import numpy as np
from scipy.linalg import expm

# --- Simulation Parameters (user configurable) ---
num_qubits = 2            # Number of qubits in the system
num_terms = 3 * num_qubits  # Number of random Pauli terms in H (e.g., 3*n)
total_time = 1.0          # Total evolution time
trotter_steps = 10        # Number of Trotter steps (N)
track_error = False       # If True, track fidelity at each step (prints overlaps)

np.random.seed(42)  # Set a random seed for reproducibility (optional)

# Define Pauli matrices (2x2 complex NumPy arrays)
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
paulis = [I, X, Y, Z]
pauli_labels = ['I', 'X', 'Y', 'Z']  # for reference/printing

# Randomly generate a Hamiltonian as a sum of Pauli strings
terms = []  # list to hold (coeff, matrix, label) for each term
dim = 2 ** num_qubits
H = np.zeros((dim, dim), dtype=complex)  # Hamiltonian matrix
for term_index in range(num_terms):
    # Randomly pick a Pauli for each qubit to form a tensor-product term
    matrices = []
    labels = []
    for q in range(num_qubits):
        choice = np.random.randint(0, 4)  # 0=I, 1=X, 2=Y, 3=Z
        matrices.append(paulis[choice])
        labels.append(pauli_labels[choice])
    # Ensure the term is not Identity on all qubits (if so, replace one factor with X)
    if all(lbl == 'I' for lbl in labels):
        rand_q = np.random.randint(0, num_qubits)
        matrices[rand_q] = X
        labels[rand_q] = 'X'
    # Construct the term matrix via Kronecker (tensor) product
    term_matrix = matrices[0]
    for mat in matrices[1:]:
        term_matrix = np.kron(term_matrix, mat)
    # Random real coefficient for this term
    coeff = np.random.uniform(-1.0, 1.0)
    # Add to Hamiltonian
    H += coeff * term_matrix
    terms.append((coeff, term_matrix, "".join(labels)))

# Print out the generated Hamiltonian terms (for information)
print(f"Hamiltonian has {len(terms)} terms on {num_qubits} qubits:")
for coeff, _, label in terms:
    print(f"  {coeff:+.3f} * ({label})")

# Initial state |00...0>: a vector of size 2^n with 1 in the 0th element
initial_state = np.zeros(dim, dtype=complex)
initial_state[0] = 1.0

# Prepare for Trotter evolution
dt = total_time / trotter_steps  # time step size
state = initial_state.copy()     # will hold the evolving state vector

# (Optional) prepare for exact intermediate evolution tracking
if track_error:
    U_step = expm(-1j * H * dt)        # exact propagator for one step
    exact_state = initial_state.copy()  # will hold exact state as we step through

# --- Main Trotter Evolution Loop ---
for step in range(trotter_steps):
    # Apply each term's evolution for this time slice
    for (coeff, term_matrix, label) in terms:
        theta = coeff * dt  # the angle for this term's evolution
        # Using exp(-i * coeff * P * dt) = I * cos(theta) - i * P * sin(theta)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        # Update state: new_state = cos(theta)*state - i * sin(theta) * (P * state)
        state = cos_theta * state - 1j * sin_theta * (term_matrix.dot(state))
    # After applying all terms, one full Trotter step is complete.
    if track_error:
        # Update exact state by one full step and compute fidelity
        exact_state = U_step.dot(exact_state)
        overlap = np.vdot(exact_state, state)  # inner product <psi_exact | psi_trotter>
        fidelity = np.abs(overlap)**2
        print(f"Step {step+1:2d}/{trotter_steps}: fidelity = {fidelity:.6f}")

# Compute exact final state for comparison using full matrix exponential
psi_exact_final = expm(-1j * H * total_time).dot(initial_state)

# Compute overlap and fidelity between Trotter result and exact result
final_overlap = np.vdot(psi_exact_final, state)
final_fidelity = np.abs(final_overlap)**2

print("\nFinal state (Trotter approximation):")
print(state)
print("\nFinal state (Exact evolution):")
print(psi_exact_final)
print(f"\nOverlap = {final_overlap:.6f}")
print(f"Fidelity = {final_fidelity:.6f}")
