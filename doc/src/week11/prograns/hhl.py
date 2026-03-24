#!/usr/bin/env python3
"""
HHL Algorithm — Pure NumPy implementation
==========================================
Solves the linear system  A x = b  using the quantum HHL algorithm.
No quantum computing libraries are used: everything is explicit matrix
algebra on state vectors, so every step can be read directly alongside
the equations in the accompanying LaTeX slides.

Algorithm stages (matching the slide sections)
-----------------------------------------------
1. State preparation   : encode b as a normalised quantum state |b⟩
2. Quantum Phase Estim.: apply controlled-U^(2^k) to stamp eigenvalues
                         into the clock register  →  Σ β_j |u_j⟩|λ̃_j⟩
3. Controlled rotation : ancilla rotation  C/λ_j |1⟩ + sqrt(1-C²/λ_j²)|0⟩
4. Inverse QPE         : uncompute the clock register (unitary adjoint)
5. Post-selection      : measure ancilla = 1; extract the solution state

Physical system used
--------------------
We solve a 2×2 Hermitian system (so the system register needs 1 qubit):

    A = [[1.5, 0.5],       b = [1, 0]   →  exact solution x = [2/3, -1/3]
         [0.5, 1.5]]

Eigenvalues: λ₁ = 1, λ₂ = 2.
We use m=3 clock qubits, giving a phase resolution of 1/2³ = 0.125.

Total state space: 1 ancilla + m clock + 1 system = 1+3+1 = 5 qubits
Hilbert space dimension: 2^5 = 32.

Key equations from the slides
-------------------------------
Spectral expansion:   |b⟩ = Σ_j β_j |u_j⟩
QPE output:           Σ_j β_j |u_j⟩ |λ̃_j⟩
After rotation:       Σ_j β_j |u_j⟩ |λ̃_j⟩ (√(1-C²/λ_j²)|0⟩ + C/λ_j|1⟩)
After inv-QPE:        Σ_j β_j/λ_j |u_j⟩ |0⟩ |1⟩   (post-selected)
Solution:             |x⟩ ∝ A⁻¹|b⟩ = Σ_j β_j/λ_j |u_j⟩
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(0)

# =============================================================================
# PARAMETERS
# =============================================================================
M_CLOCK = 3            # number of QPE clock qubits (phase resolution = 1/2^M)
T_QPE   = 2 * np.pi    # evolution time for Hamiltonian simulation
                       # chosen so that e^{i A t} encodes eigenvalues exactly

# =============================================================================
# SECTION 1 — PROBLEM SETUP
# =============================================================================
print("=" * 65)
print("HHL ALGORITHM — pure NumPy simulation")
print("=" * 65)

# --- Matrix and right-hand side -------------------------------------------
A = np.array([[1.5, 0.5],
              [0.5, 1.5]])

b_vec = np.array([1.0, 0.0])

# Classical solution (ground truth)
x_exact = np.linalg.solve(A, b_vec)

print(f"\nMatrix A:\n{A}")
print(f"\nRight-hand side b = {b_vec}")
print(f"\nClassical solution x = A⁻¹b = {x_exact}")

# --- Spectral decomposition ------------------------------------------------
# A = U diag(λ) U†  with U columns = eigenvectors |u_j⟩
eigenvalues, eigenvectors = np.linalg.eigh(A)   # eigh: real eigenvalues, unitary U
print(f"\nEigenvalues  λ = {eigenvalues}")
print(f"Eigenvectors (columns = |u_j⟩):\n{eigenvectors}")

# Expand |b⟩ in the eigenbasis: β_j = ⟨u_j|b⟩
b_norm  = np.linalg.norm(b_vec)
b_state = b_vec / b_norm           # normalised quantum state |b⟩

betas = eigenvectors.conj().T @ b_state   # coefficients β_j
print(f"\n|b⟩ = {b_state}")
print(f"Expansion coefficients β_j = {betas}")

# Condition number
kappa = eigenvalues.max() / eigenvalues.min()
print(f"\nCondition number κ = {kappa:.4f}")
print(f"HHL complexity: O(κ² log N) vs classical O(N³)")

# =============================================================================
# SECTION 2 — QUBIT REGISTER LAYOUT
# =============================================================================
# We concatenate three registers in the tensor-product order:
#
#   |ancilla⟩ ⊗ |clock_0 ... clock_{m-1}⟩ ⊗ |system⟩
#
# Qubit index assignments (most-significant first in bitstrings):
#   qubit 0          : ancilla
#   qubits 1 .. m    : clock (QPE) register
#   qubit m+1        : system register (1 qubit for a 2×2 matrix)
#
# With m=3 clock qubits + 1 ancilla + 1 system = 5 qubits total.
# Full Hilbert space dimension: 2^5 = 32.

N_SYS    = int(np.log2(len(A)))   # system qubits (= 1 for 2×2 A)
N_TOTAL  = 1 + M_CLOCK + N_SYS   # total qubits
DIM      = 2 ** N_TOTAL           # full Hilbert space dimension

# Index ranges in the full state vector
# (we address bits MSB→LSB so qubit 0 is the most significant)
def qubit_index_to_basis_contribution(q, val, n_qubits=N_TOTAL):
    """
    Contribution of qubit q having value val ∈ {0,1} to a basis index.
    Qubit 0 is the MSB, qubit n_qubits-1 is the LSB.
    """
    return val << (n_qubits - 1 - q)

print(f"\n{'='*65}")
print(f"REGISTER LAYOUT")
print(f"{'='*65}")
print(f"  Qubit 0          : ancilla")
print(f"  Qubits 1..{M_CLOCK}      : clock (QPE) register  [{M_CLOCK} qubits]")
print(f"  Qubit {M_CLOCK+1}          : system register      [1 qubit for 2×2 A]")
print(f"  Total qubits     : {N_TOTAL}")
print(f"  Hilbert space dim: {DIM}")

# =============================================================================
# SECTION 3 — STATE PREPARATION  (slide: "Quantum Goal")
# =============================================================================
# Initialise |ψ₀⟩ = |0⟩_anc ⊗ |0...0⟩_clock ⊗ |b⟩_sys
#
# In our basis the system register occupies the least-significant qubit.
# |b⟩ = b_state[0]|0⟩ + b_state[1]|1⟩
# so the full initial state has support only on indices 0 (sys=0) and 1 (sys=1).

psi = np.zeros(DIM, dtype=complex)
for sys_bit in range(len(b_state)):
    idx = qubit_index_to_basis_contribution(N_TOTAL - 1, sys_bit)
    psi[idx] = b_state[sys_bit]

assert abs(np.linalg.norm(psi) - 1.0) < 1e-12, "Initial state not normalised"

print(f"\n{'='*65}")
print(f"STAGE 1 — STATE PREPARATION")
print(f"{'='*65}")
print(f"|ψ₀⟩ = |0⟩_anc ⊗ |0⟩_clock ⊗ |b⟩_sys")
print(f"  Non-zero amplitudes:")
for i, amp in enumerate(psi):
    if abs(amp) > 1e-12:
        bits = format(i, f'0{N_TOTAL}b')
        print(f"    |{bits}⟩  amplitude = {amp:.6f}")

# =============================================================================
# SECTION 4 — QUANTUM PHASE ESTIMATION  (slides: "Hamiltonian Encoding",
#             "QPE Action", "State After QPE")
# =============================================================================
# QPE stamps the eigenphase φ_j = λ_j * t / (2π) into the clock register.
#
# For clock qubit k (0-indexed within the clock block) we apply:
#   controlled-U^(2^k)  where  U = e^{iAt}
#
# controlled means: if clock qubit k = 1, apply U^(2^k) to system.
#
# The standard QPE circuit is:
#   (a) Hadamard on all clock qubits
#   (b) For each clock qubit k: controlled-U^(2^k) on the system
#   (c) Inverse QFT on the clock register
#
# We implement each step as an explicit 2^N × 2^N unitary.

def tensor(*ops):
    """Tensor (Kronecker) product of a sequence of matrices."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

I1 = np.eye(2, dtype=complex)
I_anc   = I1                              # ancilla identity
I_clock = np.eye(2**M_CLOCK, dtype=complex)
I_sys   = np.eye(2**N_SYS, dtype=complex)

H1 = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)  # single-qubit Hadamard

# --- (a) Hadamard layer on all clock qubits --------------------------------
# Apply H to each of the M_CLOCK clock qubits independently.
# Full unitary: I_anc ⊗ H^⊗M ⊗ I_sys

H_clock = H1
for _ in range(M_CLOCK - 1):
    H_clock = np.kron(H_clock, H1)

U_hadamard = tensor(I_anc, H_clock, I_sys)
psi = U_hadamard @ psi

print(f"\n{'='*65}")
print(f"STAGE 2a — HADAMARDS ON CLOCK REGISTER")
print(f"{'='*65}")
print(f"  Clock qubits now in uniform superposition 1/√{2**M_CLOCK} Σ|k⟩")

# --- (b) Controlled-U^(2^k) for each clock qubit --------------------------
# U = e^{iAt}.  For each clock qubit k the corresponding power is 2^k.
# The full controlled-U^p unitary on the complete state space is:
#
#   |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ U^p
#
# embedded at the position of clock qubit k, with identities elsewhere.

def expm_hermitian(H, t):
    """
    Matrix exponential exp(i H t) for a Hermitian H.
    Uses the spectral decomposition: exp(i H t) = U diag(exp(i λ t)) U†
    This is exact (no Trotter approximation).
    """
    vals, vecs = np.linalg.eigh(H)
    return vecs @ np.diag(np.exp(1j * vals * t)) @ vecs.conj().T

# Base unitary U = e^{iAt}
U_base = expm_hermitian(A, T_QPE)

proj0 = np.array([[1, 0], [0, 0]], dtype=complex)   # |0⟩⟨0|
proj1 = np.array([[0, 0], [0, 1]], dtype=complex)   # |1⟩⟨1|

for k in range(M_CLOCK):
    # Clock qubit k is at position (1 + k) in the full register
    # (0=ancilla, 1..M_CLOCK=clock, M_CLOCK+1=system)
    power = 2 ** k
    U_power = expm_hermitian(A, T_QPE * power)

    # Number of identity qubits before and after clock qubit k
    n_before = 1 + k              # ancilla + clock qubits 0..k-1
    n_after  = M_CLOCK - k - 1   # remaining clock qubits after k
    n_after_total = n_after + N_SYS

    I_before = np.eye(2**n_before, dtype=complex)
    I_after  = np.eye(2**n_after_total, dtype=complex)

    # Controlled-U^(2^k): if clock_k = 0 → identity; if clock_k = 1 → U^p
    ctrl_U = (tensor(I_before, proj0, I_after) +
              tensor(I_before, proj1, np.kron(np.eye(2**n_after, dtype=complex),
                                               U_power)))

    psi = ctrl_U @ psi

print(f"\n{'='*65}")
print(f"STAGE 2b — CONTROLLED-U^(2^k) FOR k = 0..{M_CLOCK-1}")
print(f"{'='*65}")
print(f"  U = exp(i A t) with t = {T_QPE/(2*np.pi):.4f} × 2π")
print(f"  Eigenphases: φ_j = λ_j × t/(2π) = {eigenvalues * T_QPE / (2*np.pi)}")

# --- (c) Inverse QFT on the clock register --------------------------------
# QFT on n qubits: (QFT)_jk = 1/√N × exp(2πi j k / N)
# We apply QFT†.  This is an exact 2^m × 2^m unitary.

def qft_matrix(n_qubits):
    """Exact QFT matrix on n_qubits qubits."""
    N = 2 ** n_qubits
    idx = np.arange(N)
    return np.exp(2j * np.pi * np.outer(idx, idx) / N) / np.sqrt(N)

QFT  = qft_matrix(M_CLOCK)
IQFT = QFT.conj().T        # inverse QFT = QFT†

# Embed IQFT in the full space: I_anc ⊗ IQFT ⊗ I_sys
U_iqft = tensor(I_anc, IQFT, I_sys)
psi = U_iqft @ psi

print(f"\n{'='*65}")
print(f"STAGE 2c — INVERSE QFT ON CLOCK REGISTER")
print(f"{'='*65}")
print(f"  Clock register now encodes λ̃_j ≈ λ_j in binary")

# Verify: find the dominant clock basis states
probs = np.abs(psi)**2
print(f"  Largest amplitudes (threshold > 0.05):")
for i, p in enumerate(probs):
    if p > 0.05:
        bits = format(i, f'0{N_TOTAL}b')
        anc_b   = bits[0]
        clock_b = bits[1:1+M_CLOCK]
        sys_b   = bits[1+M_CLOCK:]
        clock_int = int(clock_b, 2)
        lambda_est = clock_int / (2**M_CLOCK) * (2 * np.pi / T_QPE)
        print(f"    |anc={anc_b}⟩|clock={clock_b}({clock_int:2d})⟩|sys={sys_b}⟩  "
              f"p={p:.4f}  λ̃ = {lambda_est:.4f}")

# =============================================================================
# SECTION 5 — CONTROLLED ROTATION  (slides: "Key Step: Inversion",
#             "State After Rotation")
# =============================================================================
# For each eigenvalue λ_j the ancilla qubit is rotated:
#
#   |0⟩_anc → √(1 - C²/λ_j²) |0⟩ + C/λ_j |1⟩
#
# C is a scaling constant; we choose C = min(|λ_j|) so that C/λ_j ≤ 1.
#
# In a real quantum computer this would be implemented as a sequence of
# multi-controlled Ry rotations conditioned on each clock register state.
# Here we implement it exactly: for every basis state where the clock
# register encodes a value k (corresponding to λ̃ = 2πk / (2^m t)) we
# apply a 2×2 rotation matrix to the ancilla subspace.

C_scale = min(abs(eigenvalues))   # ensures C/λ_j ≤ 1 for all j

print(f"\n{'='*65}")
print(f"STAGE 3 — CONTROLLED ROTATION")
print(f"{'='*65}")
print(f"  Rotation: |0⟩ → √(1-C²/λ²)|0⟩ + C/λ|1⟩")
print(f"  Scaling constant C = {C_scale:.4f}")

# Build the controlled-rotation operator.
# Strategy: iterate over all basis states; for those with ancilla=0,
# pair them with the corresponding ancilla=1 state and apply the 2×2
# rotation in that 2-dimensional subspace.
#
# Basis index layout (MSB → LSB):
#   bit N_TOTAL-1 : ancilla
#   bits N_TOTAL-2 .. N_TOTAL-1-M_CLOCK : clock
#   bits N_TOTAL-2-M_CLOCK .. 0 : system

anc_bit_pos    = N_TOTAL - 1          # ancilla is the MSB
clock_bit_pos  = range(N_TOTAL - 2, N_TOTAL - 2 - M_CLOCK, -1)  # MSB→LSB within clock
sys_bit_pos    = range(N_TOTAL - 2 - M_CLOCK, -1, -1)            # system bits

def extract_clock_value(idx):
    """Extract the integer value stored in the clock register from basis index idx."""
    clock_val = 0
    for bit_pos in clock_bit_pos:
        clock_val = (clock_val << 1) | ((idx >> bit_pos) & 1)
    return clock_val

def extract_ancilla_bit(idx):
    return (idx >> anc_bit_pos) & 1

def flip_ancilla(idx):
    """Return the basis index with the ancilla bit flipped."""
    return idx ^ (1 << anc_bit_pos)

# Apply the rotation directly to the state vector
psi_rot = psi.copy()

for idx in range(DIM):
    if extract_ancilla_bit(idx) == 0:
        idx1 = flip_ancilla(idx)   # same state with ancilla=1

        clock_val  = extract_clock_value(idx)
        if clock_val == 0:
            continue    # λ̃=0 would require infinite rotation; skip

        # Eigenvalue estimate from clock register
        lambda_est = (clock_val / 2**M_CLOCK) * (2 * np.pi / T_QPE)

        ratio = C_scale / lambda_est   # C / λ̃_j
        if abs(ratio) > 1.0:
            ratio = 1.0   # numerical safety clamp (shouldn't trigger)

        cos_theta = np.sqrt(max(0.0, 1.0 - ratio**2))
        sin_theta = ratio

        # 2×2 rotation in the {|0⟩_anc, |1⟩_anc} subspace:
        #   |0⟩_anc → cos_theta |0⟩_anc + sin_theta |1⟩_anc
        #   |1⟩_anc → -sin_theta|0⟩_anc + cos_theta |1⟩_anc  (unitary completion)
        a0 = psi[idx]
        a1 = psi[idx1]
        psi_rot[idx]  = cos_theta * a0 - sin_theta * a1
        psi_rot[idx1] = sin_theta * a0 + cos_theta * a1

psi = psi_rot

print(f"  Dominant amplitudes after rotation:")
for i, amp in enumerate(psi):
    if abs(amp)**2 > 0.01:
        bits = format(i, f'0{N_TOTAL}b')
        print(f"    |{bits[0]}⟩_anc |{bits[1:1+M_CLOCK]}⟩_clk |{bits[1+M_CLOCK:]}⟩_sys  "
              f"amp={amp:.4f}  p={abs(amp)**2:.4f}")

# =============================================================================
# SECTION 6 — INVERSE QPE  (slide: "Inverse QPE", "Uncomputation")
# =============================================================================
# Apply QFT (not IQFT) to the clock register, then undo the controlled-U
# sequence. This is the exact adjoint of the QPE forward pass:
#   (IQFT controlled-U)† = controlled-U† × QFT

# (a) QFT on clock
U_qft_fwd = tensor(I_anc, QFT, I_sys)
psi = U_qft_fwd @ psi

# (b) Undo controlled-U^(2^k) in reverse order, applying U† = e^{-iAt p}
for k in reversed(range(M_CLOCK)):
    power = 2 ** k
    U_power_dag = expm_hermitian(A, -T_QPE * power)  # U†^(2^k) = e^{-iAt 2^k}

    n_before       = 1 + k
    n_after        = M_CLOCK - k - 1
    n_after_total  = n_after + N_SYS
    I_before = np.eye(2**n_before, dtype=complex)
    I_after  = np.eye(2**n_after_total, dtype=complex)

    ctrl_Udag = (tensor(I_before, proj0, I_after) +
                 tensor(I_before, proj1, np.kron(np.eye(2**n_after, dtype=complex),
                                                  U_power_dag)))
    psi = ctrl_Udag @ psi

# (c) Hadamard again on all clock qubits (H = H†)
psi = U_hadamard @ psi

print(f"\n{'='*65}")
print(f"STAGE 4 — INVERSE QPE (UNCOMPUTATION)")
print(f"{'='*65}")
print(f"  Clock register disentangled from system register")

# Verify clock is back to |0...0⟩
clock_0_prob = 0.0
for i, amp in enumerate(psi):
    bits = format(i, f'0{N_TOTAL}b')
    if bits[1:1+M_CLOCK] == '0' * M_CLOCK:
        clock_0_prob += abs(amp)**2
print(f"  Probability of clock = |0⟩^⊗m: {clock_0_prob:.6f}  (should be ≈ 1.0)")

# =============================================================================
# SECTION 7 — POST-SELECTION  (slides: "Post-selection", "Final State")
# =============================================================================
# Measure ancilla qubit.  Keep only the component with ancilla = 1.
# After renormalisation this gives |x⟩ ∝ A⁻¹|b⟩.

print(f"\n{'='*65}")
print(f"STAGE 5 — POST-SELECTION ON ANCILLA = |1⟩")
print(f"{'='*65}")

# Project onto ancilla=1 subspace
psi_post = np.zeros(DIM, dtype=complex)
for i, amp in enumerate(psi):
    if extract_ancilla_bit(i) == 1:
        psi_post[i] = amp

# Success probability
success_prob = np.sum(np.abs(psi_post)**2)
print(f"  Success probability P(ancilla=1) = {success_prob:.6f}")
print(f"  (Scales as O(1/κ²) = {1/kappa**2:.6f} in the worst case)")

# Renormalise
psi_post /= np.sqrt(success_prob)

# =============================================================================
# SECTION 8 — EXTRACT SOLUTION AND COMPARE
# =============================================================================
# The system register (least-significant qubit) now holds |x⟩ ∝ A⁻¹|b⟩.
# Extract the two amplitudes (for sys=0 and sys=1) from the post-selected state.

print(f"\n{'='*65}")
print(f"SOLUTION EXTRACTION")
print(f"{'='*65}")

# Collect amplitudes where ancilla=1, clock=0...0
x_amps = np.zeros(2**N_SYS, dtype=complex)
for i, amp in enumerate(psi_post):
    bits = format(i, f'0{N_TOTAL}b')
    if bits[0] == '1' and bits[1:1+M_CLOCK] == '0' * M_CLOCK:
        sys_val = int(bits[1+M_CLOCK:], 2)
        x_amps[sys_val] = amp

print(f"  Quantum state amplitudes: {x_amps}")

# The state encodes |x⟩ up to a global phase and norm.
# To compare with the classical solution we must account for:
#   - the normalisation of |b⟩ (we encoded b/||b||)
#   - the normalisation of |x⟩ (it is itself a normalised state)
#
# The amplitudes are proportional to the entries of A⁻¹b.
# The ratio between amplitudes equals the ratio of solution components.

x_hhl_raw = x_amps.real              # drop any imaginary noise
x_hhl_raw /= np.linalg.norm(x_hhl_raw)  # renormalise

# Determine the overall scale from the classical solution
# (in a real quantum computer we would estimate expectation values,
# not the raw amplitude, but here we can compare directly)
scale = np.dot(x_exact, x_hhl_raw) / np.dot(x_hhl_raw, x_hhl_raw)
x_hhl = scale * x_hhl_raw

print(f"\n  Classical solution:          x = {x_exact}")
print(f"  HHL solution (rescaled):     x = {x_hhl}")
print(f"  Relative error (L2):       {np.linalg.norm(x_hhl - x_exact)/np.linalg.norm(x_exact):.2e}")

# Verify A x ≈ b
residual = A @ x_hhl - b_vec
print(f"\n  Residual A x_HHL - b = {residual}")
print(f"  ||A x_HHL - b||₂    = {np.linalg.norm(residual):.2e}")

# =============================================================================
# SECTION 9 — EXTENDED TEST WITH DIFFERENT SYSTEMS
# =============================================================================
print(f"\n{'='*65}")
print(f"ADDITIONAL TEST CASES")
print(f"{'='*65}")

def hhl_simulate(A_in, b_in, m_clock=3):
    """
    Run the full HHL simulation for an arbitrary 2x2 Hermitian matrix.
    Returns the unnormalised HHL solution (matched to scale of b_in).
    """
    N_sys = int(np.log2(len(A_in)))
    assert 2**N_sys == len(A_in), "A must be 2^n × 2^n"
    N_tot = 1 + m_clock + N_sys
    dim   = 2**N_tot

    # State preparation
    b_n = b_in / np.linalg.norm(b_in)
    psi = np.zeros(dim, dtype=complex)
    for s in range(len(b_n)):
        psi[s] = b_n[s]

    I_a = np.eye(2, dtype=complex)
    I_c = np.eye(2**m_clock, dtype=complex)
    I_s = np.eye(2**N_sys, dtype=complex)
    H1_ = np.array([[1,1],[1,-1]], dtype=complex)/np.sqrt(2)
    H_c = H1_
    for _ in range(m_clock-1):
        H_c = np.kron(H_c, H1_)

    t = 2*np.pi
    evals, _ = np.linalg.eigh(A_in)
    C = min(abs(evals))
    p0 = np.array([[1,0],[0,0]], dtype=complex)
    p1 = np.array([[0,0],[0,1]], dtype=complex)

    # Hadamards
    psi = tensor(I_a, H_c, I_s) @ psi

    # Controlled-U^(2^k)
    for k in range(m_clock):
        U_p = expm_hermitian(A_in, t * 2**k)
        nb = 1+k; na = m_clock-k-1
        I_b = np.eye(2**nb, dtype=complex)
        I_af = np.eye(2**(na+N_sys), dtype=complex)
        ctrl_U = (tensor(I_b, p0, I_af) +
                  tensor(I_b, p1, np.kron(np.eye(2**na, dtype=complex), U_p)))
        psi = ctrl_U @ psi

    # IQFT
    N_ = 2**m_clock
    idx_ = np.arange(N_)
    QFT_ = np.exp(2j*np.pi*np.outer(idx_,idx_)/N_)/np.sqrt(N_)
    IQFT_ = QFT_.conj().T
    psi = tensor(I_a, IQFT_, I_s) @ psi

    # Controlled rotation
    psi_r = psi.copy()
    anc_bp = N_tot-1
    clk_bp = range(N_tot-2, N_tot-2-m_clock, -1)
    for idx in range(dim):
        if (idx >> anc_bp) & 1 == 0:
            idx1 = idx ^ (1 << anc_bp)
            cv = sum(((idx >> bp) & 1) << (m_clock-1-ki) for ki,bp in enumerate(clk_bp))
            if cv == 0: continue
            le = cv/(2**m_clock) * (2*np.pi/t)
            r = C/le
            if abs(r) > 1.0: r = 1.0
            ct = np.sqrt(max(0.0, 1-r**2)); st = r
            a0,a1 = psi[idx], psi[idx1]
            psi_r[idx]  = ct*a0 - st*a1
            psi_r[idx1] = st*a0 + ct*a1
    psi = psi_r

    # Inverse QPE
    psi = tensor(I_a, QFT_, I_s) @ psi
    for k in reversed(range(m_clock)):
        U_pd = expm_hermitian(A_in, -t*2**k)
        nb = 1+k; na = m_clock-k-1
        I_b = np.eye(2**nb, dtype=complex)
        I_af = np.eye(2**(na+N_sys), dtype=complex)
        ctrl_Ud = (tensor(I_b, p0, I_af) +
                   tensor(I_b, p1, np.kron(np.eye(2**na, dtype=complex), U_pd)))
        psi = ctrl_Ud @ psi
    psi = tensor(I_a, H_c, I_s) @ psi

    # Post-select on ancilla=1
    psi_ps = np.array([amp if (i >> anc_bp)&1 else 0.0+0j
                        for i,amp in enumerate(psi)], dtype=complex)
    sp = np.sum(np.abs(psi_ps)**2)
    if sp < 1e-14:
        return None, 0.0
    psi_ps /= np.sqrt(sp)

    # Extract solution amplitudes  (ancilla=1, clock=000)
    xa = np.zeros(2**N_sys, dtype=complex)
    for i,amp in enumerate(psi_ps):
        bits = format(i, f'0{N_tot}b')
        if bits[0]=='1' and bits[1:1+m_clock]=='0'*m_clock:
            xa[int(bits[1+m_clock:],2)] = amp
    return xa.real, sp

test_cases = [
    ("Diagonal  [[2,0],[0,3]]",
     np.array([[2.0,0.0],[0.0,3.0]]),
     np.array([1.0,1.0])),
    ("Identity  [[1,0],[0,1]]",
     np.eye(2),
     np.array([0.5, 0.5])),
    ("Original  [[1.5,0.5],[0.5,1.5]]",
     A, b_vec),
]

for name, A_t, b_t in test_cases:
    x_cl = np.linalg.solve(A_t, b_t)
    xa_raw, sp = hhl_simulate(A_t, b_t)
    if xa_raw is not None and np.linalg.norm(xa_raw) > 1e-10:
        xa_raw /= np.linalg.norm(xa_raw)
        sc = np.dot(x_cl, xa_raw)/(np.dot(xa_raw,xa_raw)+1e-30)
        x_hhl_t = sc * xa_raw
        err = np.linalg.norm(x_hhl_t-x_cl)/(np.linalg.norm(x_cl)+1e-30)
        print(f"\n  {name}")
        print(f"    b = {b_t}")
        print(f"    Classical x = {x_cl}")
        print(f"    HHL x       = {x_hhl_t}")
        print(f"    Rel. error  = {err:.2e}    P(success) = {sp:.4f}")

# =============================================================================
# SECTION 10 — VISUALISATION
# =============================================================================
print(f"\n{'='*65}")
print(f"GENERATING VISUALISATION")
print(f"{'='*65}")

fig = plt.figure(figsize=(16, 10))
fig.suptitle("HHL Algorithm — Pure NumPy Simulation\n"
             r"Solving $A\mathbf{x}=\mathbf{b}$ via Quantum Phase Estimation",
             fontsize=13, fontweight='bold')

gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.38)

# ── Panel 1: Matrix A and spectrum ─────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
im = ax1.imshow(A, cmap='coolwarm', aspect='equal',
                vmin=-np.abs(A).max(), vmax=np.abs(A).max())
for i in range(2):
    for j in range(2):
        ax1.text(j, i, f"{A[i,j]:.2f}", ha='center', va='center',
                 fontsize=12, fontweight='bold', color='white')
ax1.set_xticks([0,1]); ax1.set_yticks([0,1])
ax1.set_xticklabels(['col 0','col 1']); ax1.set_yticklabels(['row 0','row 1'])
ax1.set_title('Matrix A\n(Hermitian, 2×2)', fontsize=10, fontweight='bold')
plt.colorbar(im, ax=ax1, shrink=0.8)

# ── Panel 2: Eigenvalues and condition number ──────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
bars = ax2.bar([0, 1], eigenvalues, color=['steelblue', 'firebrick'],
               width=0.4, edgecolor='k')
ax2.set_xticks([0, 1])
ax2.set_xticklabels([r'$\lambda_1$', r'$\lambda_2$'], fontsize=12)
ax2.set_ylabel('Eigenvalue', fontsize=10)
ax2.set_title(f'Spectrum of A\n'
              r'$\kappa = \lambda_{\max}/\lambda_{\min}$' + f' = {kappa:.2f}',
              fontsize=10, fontweight='bold')
for bar, lam in zip(bars, eigenvalues):
    ax2.text(bar.get_x() + bar.get_width()/2, lam + 0.02,
             f'{lam:.2f}', ha='center', va='bottom', fontsize=11)
ax2.set_ylim(0, max(eigenvalues) * 1.3)
ax2.grid(True, alpha=0.3, axis='y')

# ── Panel 3: HHL algorithm stages ─────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.axis('off')
stages = [
    ("Stage 1", "State prep",
     r"$|b\rangle = \sum_j \beta_j |u_j\rangle$"),
    ("Stage 2", "QPE",
     r"$\sum_j \beta_j |u_j\rangle |\tilde\lambda_j\rangle$"),
    ("Stage 3", "Ctrl rotation",
     r"$\left(\sqrt{1-\frac{C^2}{\lambda_j^2}}|0\rangle + \frac{C}{\lambda_j}|1\rangle\right)$"),
    ("Stage 4", "Inv. QPE",
     r"uncompute clock register"),
    ("Stage 5", "Post-select",
     r"$|x\rangle \propto A^{-1}|b\rangle$"),
]
y = 0.95
for num, name, eq in stages:
    ax3.text(0.03, y, f"● {num}: {name}", transform=ax3.transAxes,
             fontsize=9, fontweight='bold', va='top',
             color='steelblue')
    ax3.text(0.08, y - 0.045, eq, transform=ax3.transAxes,
             fontsize=8.5, va='top', color='#333333',
             style='italic')
    y -= 0.175
ax3.set_title('Algorithm Stages\n(matching slide sections)',
              fontsize=10, fontweight='bold')

# ── Panel 4: QPE clock register probabilities ──────────────────────────────
# Recompute the state just after the forward QPE (before rotation) for display
psi_qpe_check = np.zeros(DIM, dtype=complex)
for sys_bit in range(len(b_state)):
    psi_qpe_check[sys_bit] = b_state[sys_bit]
psi_qpe_check = U_hadamard @ psi_qpe_check
for k in range(M_CLOCK):
    power = 2**k
    U_pow = expm_hermitian(A, T_QPE * power)
    nb = 1+k; na = M_CLOCK-k-1
    I_b = np.eye(2**nb, dtype=complex)
    I_af = np.eye(2**(na+N_SYS), dtype=complex)
    cU = (tensor(I_b, proj0, I_af) +
          tensor(I_b, proj1, np.kron(np.eye(2**na, dtype=complex), U_pow)))
    psi_qpe_check = cU @ psi_qpe_check
psi_qpe_check = U_iqft @ psi_qpe_check

# Marginalise over ancilla and system to get clock register distribution
clock_probs = np.zeros(2**M_CLOCK)
for i, amp in enumerate(psi_qpe_check):
    bits = format(i, f'0{N_TOTAL}b')
    clock_val = int(bits[1:1+M_CLOCK], 2)
    clock_probs[clock_val] += abs(amp)**2

ax4 = fig.add_subplot(gs[1, 0])
x_ticks = np.arange(2**M_CLOCK)
ax4.bar(x_ticks, clock_probs, color='steelblue', edgecolor='k', alpha=0.85)
# Mark the expected eigenvalue positions
for lam in eigenvalues:
    phi = lam * T_QPE / (2 * np.pi)
    k_exp = phi * 2**M_CLOCK
    ax4.axvline(k_exp, color='red', lw=2, ls='--',
                label=f'λ={lam:.1f} → k≈{k_exp:.1f}')
ax4.set_xlabel('Clock register value k', fontsize=9)
ax4.set_ylabel('Probability', fontsize=9)
ax4.set_title(f'QPE Clock Distribution\n'
              f'(m={M_CLOCK} qubits, resolution=1/{2**M_CLOCK})',
              fontsize=10, fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3, axis='y')

# ── Panel 5: Solution comparison bar chart ─────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
x_idx   = np.array([0, 1])
width   = 0.32
bars_cl = ax5.bar(x_idx - width/2, x_exact,  width,
                  label='Classical', color='steelblue',
                  edgecolor='k', alpha=0.85)
bars_hh = ax5.bar(x_idx + width/2, x_hhl,    width,
                  label='HHL', color='firebrick',
                  edgecolor='k', alpha=0.85)
ax5.axhline(0, color='k', lw=0.8)
ax5.set_xticks(x_idx)
ax5.set_xticklabels([r'$x_0$', r'$x_1$'], fontsize=12)
ax5.set_ylabel('Value', fontsize=10)
ax5.set_title('Solution Comparison\n'
              r'$A\mathbf{x}=\mathbf{b}$',
              fontsize=10, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')
rel_err = np.linalg.norm(x_hhl - x_exact) / np.linalg.norm(x_exact)
ax5.text(0.5, 0.02, f'Rel. error = {rel_err:.2e}',
         ha='center', va='bottom', transform=ax5.transAxes,
         fontsize=9, color='darkgreen', fontweight='bold')

# ── Panel 6: Success probability vs condition number ──────────────────────
ax6 = fig.add_subplot(gs[1, 2])
kappas = np.linspace(1.0, 5.0, 200)
p_succ_theory = 1.0 / kappas**2   # P(success) ∝ 1/κ²
ax6.plot(kappas, p_succ_theory, 'b-', lw=2, label=r'$P \propto 1/\kappa^2$')
ax6.axvline(kappa, color='red', ls='--', lw=1.5,
            label=f'This problem κ={kappa:.2f}')
ax6.scatter([kappa], [success_prob], color='red', s=80, zorder=5,
            label=f'Measured P={success_prob:.4f}')
ax6.set_xlabel('Condition number κ', fontsize=10)
ax6.set_ylabel('Success probability', fontsize=10)
ax6.set_title('Post-selection Success\n'
              r'$P(\text{ancilla}=|1\rangle) \propto 1/\kappa^2$',
              fontsize=10, fontweight='bold')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)
ax6.set_ylim(0, 1.1)

plt.savefig('hhl_result.png', dpi=140, bbox_inches='tight')
plt.show()
print("Figure saved to hhl_result.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print(f"\n{'='*65}")
print(f"FINAL SUMMARY")
print(f"{'='*65}")
print(f"""
Problem:  A x = b
  A = {A.tolist()}
  b = {b_vec.tolist()}

Eigenvalues:  λ = {eigenvalues}
Condition number:  κ = {kappa:.4f}

Register layout:
  1 ancilla + {M_CLOCK} clock + {N_SYS} system = {N_TOTAL} qubits total
  Hilbert space dimension = {DIM}

Classical solution:          x = {x_exact}
HHL solution (rescaled):     x = {np.round(x_hhl, 6)}
Residual ||Ax_HHL - b||₂:  {np.linalg.norm(A @ x_hhl - b_vec):.2e}
Relative error:              {rel_err:.2e}

Post-selection success probability: {success_prob:.6f}
  (Theory: C/λ_min = {C_scale/min(abs(eigenvalues)):.4f})

Key equations from the slides implemented here:
  |b⟩ = Σ β_j |u_j⟩                         [spectral expansion]
  QPE: Σ β_j |u_j⟩|λ̃_j⟩                     [eigenvalue stamping]
  Rotation: C/λ_j |1⟩ + √(1-C²/λ_j²)|0⟩     [inversion step]
  Post-select: |x⟩ ∝ Σ β_j/λ_j |u_j⟩ = A⁻¹|b⟩  [solution]
""")
