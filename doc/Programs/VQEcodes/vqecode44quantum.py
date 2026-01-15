import numpy as np

# =========================
# Pauli matrices & utilities
# =========================
I2 = np.array([[1, 0], [0, 1]], dtype=complex)
X2 = np.array([[0, 1], [1, 0]], dtype=complex)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z2 = np.array([[1, 0], [0, -1]], dtype=complex)

PAULI = {"I": I2, "X": X2, "Y": Y2, "Z": Z2}

def kron2(A, B):
    return np.kron(A, B)

def safe_real(x):
    return float(np.real_if_close(x))

# =========================
# Build the 4x4 Hamiltonian
# =========================
def build_hamiltonian(e1, e2, e3, e4, Vx, Vz, Hx):
    """
    Matrix in computational basis |00>,|01>,|10>,|11>:
    [ e1+Vz,   0,     0,     Vx ]
    [  0,    e2-Vz,  Vx,     0  ]
    [  0,     Hx,   e3-Vz,   0  ]
    [  Hx,    0,     0,    e4+Vz]
    """
    H = np.zeros((4, 4), dtype=complex)
    H[0, 0] = e1 + Vz
    H[1, 1] = e2 - Vz
    H[2, 2] = e3 - Vz
    H[3, 3] = e4 + Vz
    H[0, 3] = Vx
    H[3, 0] = Hx
    H[1, 2] = Vx
    H[2, 1] = Hx
    return H

# ==========================================
# Decompose H into Pauli strings c_{ij} P_i⊗P_j
# ==========================================
def pauli_decomposition(H, tol=1e-10):
    """
    c_{ij} = (1/4) Tr[(P_i ⊗ P_j) H]
    Returns dict {("I","Z"): coeff, ...}
    """
    coeffs = {}
    for a, Pa in PAULI.items():
        for b, Pb in PAULI.items():
            P = kron2(Pa, Pb)
            c = np.trace(P @ H) / 4.0
            # For Hermitian H, coefficients should be real (within numeric tolerance)
            c = np.real_if_close(c)
            if abs(c) > tol:
                coeffs[(a, b)] = safe_real(c)
    return coeffs

# =========================
# Two-qubit ansatz simulation
# =========================
def RY(theta):
    return np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                     [np.sin(theta / 2),  np.cos(theta / 2)]], dtype=complex)

def RZ(theta):
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0, np.exp(1j * theta / 2)]], dtype=complex)

# CNOT: control qubit 0 -> target qubit 1 (basis |00>,|01>,|10>,|11|)
CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]], dtype=complex)

def prepare_state(params):
    """
    |ψ(θ)> = CNOT · (RZ(phi0)RY(theta0) ⊗ RZ(phi1)RY(theta1)) · |00>
    params = [theta0, theta1, phi0, phi1]
    """
    theta0, theta1, phi0, phi1 = params
    psi0 = np.array([1, 0, 0, 0], dtype=complex)  # |00>

    U_local = kron2(RZ(phi0) @ RY(theta0), RZ(phi1) @ RY(theta1))
    psi = U_local @ psi0
    psi = CNOT @ psi
    return psi

# ==========================================
# Measurement: sample both qubits to estimate <P⊗Q>
# ==========================================
H_gate = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
Sdg = np.array([[1, 0], [0, -1j]], dtype=complex)  # S†

def basis_rotation_for_pauli(p):
    """
    Returns U such that measuring Z after applying U gives measurement in Pauli p basis.
      Z: U = I
      X: U = H
      Y: U = H S†  (since S† maps Y-basis to X-basis then H to Z-basis)
      I: U = I (we still measure both qubits, but eigenvalue factor will be 1)
    """
    if p == "I":
        return I2
    if p == "Z":
        return I2
    if p == "X":
        return H_gate
    if p == "Y":
        return H_gate @ Sdg
    raise ValueError(f"Unknown Pauli {p}")

def sample_expectation_pauli_string(state, p0, p1, shots, rng):
    """
    Estimate <p0 ⊗ p1> by:
      - rotating each qubit into Z-basis for that Pauli
      - sampling computational basis outcomes (both qubits)
      - mapping outcomes -> eigenvalues (+1/-1) and multiplying
    """
    U = kron2(basis_rotation_for_pauli(p0), basis_rotation_for_pauli(p1))
    psi_rot = U @ state
    probs = np.abs(psi_rot)**2
    probs = probs / probs.sum()

    # Sample outcomes in {0,1,2,3} corresponding to |00>,|01>,|10>,|11|
    outcomes = rng.choice(4, size=shots, p=probs)

    # Bits: outcome = 2*b0 + b1 (b0=qubit0, b1=qubit1)
    b0 = (outcomes >> 1) & 1
    b1 = outcomes & 1

    # Eigenvalue for Z measurement is +1 for |0>, -1 for |1>
    z0 = 1 - 2*b0
    z1 = 1 - 2*b1

    # If Pauli is I, eigenvalue factor is 1 regardless of measured bit
    ev0 = z0 if p0 != "I" else np.ones_like(z0, dtype=int)
    ev1 = z1 if p1 != "I" else np.ones_like(z1, dtype=int)

    return float(np.mean(ev0 * ev1))

def estimate_energy_from_measurements(state, coeffs, shots_per_term, rng):
    """
    E ≈ sum_{(p0,p1)} c_{p0,p1} * <p0⊗p1>_estimated
    Each expectation is estimated by sampling both qubits.
    """
    E = 0.0
    for (p0, p1), c in coeffs.items():
        exp_est = sample_expectation_pauli_string(state, p0, p1, shots_per_term, rng)
        E += c * exp_est
    return float(E)

# =========================
# VQE optimization via SPSA (NumPy-only, measurement-based energy)
# =========================
def vqe_spsa(coeffs, iters=250, shots_per_term=200, seed=0,
             a=0.2, c=0.15, alpha=0.602, gamma=0.101):
    """
    SPSA minimizes noisy objective with 2 evaluations/iter.
    - coeffs: Pauli decomposition dict
    - iters: optimization steps
    - shots_per_term: measurement shots per Pauli term per energy evaluation
    """
    rng = np.random.default_rng(seed)
    p = 4  # parameters: theta0, theta1, phi0, phi1
    x = rng.uniform(0, 2*np.pi, size=p)

    best_x = x.copy()
    best_E = np.inf

    for k in range(1, iters + 1):
        ak = a / (k ** alpha)
        ck = c / (k ** gamma)

        delta = rng.choice([-1.0, 1.0], size=p)

        x_plus  = x + ck * delta
        x_minus = x - ck * delta

        psi_plus  = prepare_state(x_plus)
        psi_minus = prepare_state(x_minus)

        E_plus  = estimate_energy_from_measurements(psi_plus,  coeffs, shots_per_term, rng)
        E_minus = estimate_energy_from_measurements(psi_minus, coeffs, shots_per_term, rng)

        ghat = (E_plus - E_minus) / (2.0 * ck) * delta  # elementwise

        x = x - ak * ghat

        # Track best (re-evaluate occasionally for stability if desired)
        psi = prepare_state(x)
        E = estimate_energy_from_measurements(psi, coeffs, shots_per_term, rng)
        if E < best_E:
            best_E = E
            best_x = x.copy()

    return best_x, best_E

# =========================
# Example usage
# =========================
if __name__ == "__main__":
    # --- Set constants (edit these) ---
    e1, e2, e3, e4 = 0.5, 1.0, 1.5, 2.0
    Vx, Vz, Hx = 0.2, 0.3, 0.4

    # Build H and Pauli decomposition
    H = build_hamiltonian(e1, e2, e3, e4, Vx, Vz, Hx)
    coeffs = pauli_decomposition(H)

    # Run measurement-based VQE
    opt_params, opt_E = vqe_spsa(
        coeffs,
        iters=300,
        shots_per_term=300,   # increase for better accuracy
        seed=42
    )

    # Final energy estimate with more shots (still measurement-based)
    rng = np.random.default_rng(123)
    psi_opt = prepare_state(opt_params)
    E_final = estimate_energy_from_measurements(psi_opt, coeffs, shots_per_term=3000, rng=rng)

    print("Optimized params:", opt_params)
    print("VQE energy estimate (noisy):", opt_E)
    print("VQE energy estimate (final, more shots):", E_final)

    # (Optional) exact ground state energy for sanity check
    exact = np.min(np.linalg.eigvalsh(H))
    print("Exact ground state energy:", float(np.real(exact)))
