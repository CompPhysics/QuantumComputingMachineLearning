#!/usr/bin/env python3
"""
HHL Algorithm Applied to the 1-D Poisson Equation
===================================================
Pure NumPy / SciPy — no Qiskit or any quantum library.

Problem
-------
  -d²u/dx² = f(x),   x ∈ [0,1],   u(0) = u(1) = 0.

With n interior grid points and step h = 1/(n+1), the second-order
finite-difference discretisation gives the symmetric tridiagonal system

    A u = b,

where A is the n×n tridiagonal matrix

    A = (1/h²) * tridiag(-1, 2, -1),

and  b_i = f(x_i).

We use the manufactured right-hand side  f(x) = sin(πx)  for which the
exact solution is  u(x) = sin(πx) / π²,  giving a clean error benchmark.

Solvers implemented (no third-party quantum libraries):
  1. Thomas' algorithm  — O(n) tridiagonal solver (classical reference)
  2. HHL statevector simulation  — exact 2ⁿ-dimensional quantum circuit
     simulation in NumPy; works for n = 2 and n = 4 (1 and 2 system qubits)

Contents
--------
  §1   Finite-difference discretisation and exact solution
  §2   Thomas' algorithm (tridiagonal solver)
  §3   Grid-convergence study for Thomas' algorithm
  §4   HHL building blocks (QPE, QFT, controlled rotation, inverse QPE)
  §5   Full HHL pipeline for the Poisson matrix
  §6   HHL applied to n=2 and n=4 Poisson systems
  §7   HHL clock-precision study: accuracy vs number of clock qubits
  §8   Comparison: HHL vs Thomas vs exact solution
  §9   Visualisation (6-panel figure)
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)

SEP = "=" * 68

# =============================================================================
# §1 — FINITE-DIFFERENCE DISCRETISATION AND EXACT SOLUTION
# =============================================================================

def build_poisson_system(n):
    """
    Build the finite-difference Poisson matrix and RHS for
        -u'' = f(x) = sin(πx),   u(0)=u(1)=0,
    on n interior points  x_i = i/(n+1),  i=1,...,n.

    Returns
    -------
    A     : (n,n) symmetric tridiagonal matrix  (1/h²)*tridiag(-1,2,-1)
    b     : (n,)  RHS vector   b_i = f(x_i)
    x_int : (n,)  interior grid
    h     : grid spacing
    """
    h = 1.0 / (n + 1)
    x_int = np.linspace(h, 1.0 - h, n)

    # Tridiagonal Poisson matrix
    A = (2.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)) / h**2

    # RHS: f(x) = sin(πx)
    b = np.sin(np.pi * x_int)

    return A, b, x_int, h


def exact_solution(x):
    """Exact solution of  -u'' = sin(πx):  u(x) = sin(πx)/π²."""
    return np.sin(np.pi * x) / np.pi**2


print(SEP)
print("HHL SOLVER FOR THE 1-D POISSON EQUATION")
print("  -d²u/dx² = sin(πx),   u(0) = u(1) = 0")
print(f"  Exact solution: u(x) = sin(πx) / π²")
print(SEP)

# Demonstrate the n=4 system (will also be used for HHL)
A4, b4, x4, h4 = build_poisson_system(4)
u_exact_4 = exact_solution(x4)

print(f"\nPoisson matrix A (n=4, h={h4:.4f}):\n{np.round(A4,2)}")
print(f"\nRHS b = sin(πx_i):\n{np.round(b4,6)}")
print(f"\nExact solution u(x_i):\n{np.round(u_exact_4,6)}")
print(f"\nEigenvalues of A (n=4):")
evals4 = np.linalg.eigvalsh(A4)
for i, ev in enumerate(evals4):
    print(f"  λ_{i} = {ev:.6f}")
kappa4 = evals4.max() / evals4.min()
print(f"Condition number κ = {kappa4:.4f}")

# =============================================================================
# §2 — THOMAS' ALGORITHM (TRIDIAGONAL SOLVER)
# =============================================================================

def thomas_algorithm(lower, main, upper, rhs):
    """
    Thomas' algorithm: O(n) direct solver for the tridiagonal system

        [main_i  upper_i                      ] [x_0]   [rhs_0]
        [lower_i  main_i  upper_i             ] [x_1] = [rhs_1]
        [          ...    ...    ...          ] [ . ]   [  .  ]
        [               lower_i  main_i       ] [x_n]   [rhs_n]

    This is Gaussian elimination specialised to the tridiagonal structure.
    It performs exactly 2(n-1) divisions and 3(n-1) multiplications.

    Parameters
    ----------
    lower : (n-1,) subdiagonal  (a_i, i=1,...,n-1)
    main  : (n,)   main diagonal (b_i)
    upper : (n-1,) superdiagonal (c_i, i=0,...,n-2)
    rhs   : (n,)   right-hand side (d_i)

    Returns
    -------
    x : (n,) solution vector

    Algorithm
    ---------
    Forward sweep (eliminate subdiagonal):
        w_i = a_i / b'_{i-1}
        b'_i = b_i - w_i * c_{i-1}
        d'_i = d_i - w_i * d'_{i-1}

    Back substitution:
        x_{n-1} = d'_{n-1} / b'_{n-1}
        x_i     = (d'_i - c_i * x_{i+1}) / b'_i    (i = n-2, ..., 0)
    """
    n  = len(main)
    b_ = main.copy().astype(float)
    d_ = rhs.copy().astype(float)
    c_ = upper.copy().astype(float)

    # Forward sweep
    for i in range(1, n):
        w     = lower[i-1] / b_[i-1]
        b_[i] = b_[i] - w * c_[i-1]
        d_[i] = d_[i] - w * d_[i-1]

    # Back substitution
    x    = np.zeros(n)
    x[-1] = d_[-1] / b_[-1]
    for i in range(n-2, -1, -1):
        x[i] = (d_[i] - c_[i] * x[i+1]) / b_[i]

    return x


def solve_poisson_thomas(n):
    """
    Solve  -u'' = sin(πx)  with n interior points using Thomas' algorithm.

    For the uniform Poisson matrix the diagonals are:
      main  = [2/h², 2/h², ..., 2/h²]
      upper = [-1/h², ..., -1/h²]   (n-1 entries)
      lower = [-1/h², ..., -1/h²]   (n-1 entries)
    """
    h     = 1.0 / (n + 1)
    x_int = np.linspace(h, 1.0 - h, n)
    b     = np.sin(np.pi * x_int)

    main  = np.full(n,    2.0 / h**2)
    upper = np.full(n-1, -1.0 / h**2)
    lower = np.full(n-1, -1.0 / h**2)

    u = thomas_algorithm(lower, main, upper, b)
    return x_int, u


print(f"\n{SEP}")
print("§2 — THOMAS' ALGORITHM  (n=4)")
print(SEP)

x_th4, u_th4 = solve_poisson_thomas(4)
print(f"Thomas solution (n=4):  {np.round(u_th4, 8)}")
print(f"Exact  solution:        {np.round(u_exact_4, 8)}")
print(f"Max error:              {np.max(np.abs(u_th4 - u_exact_4)):.4e}")

# =============================================================================
# §3 — GRID-CONVERGENCE STUDY: THOMAS' ALGORITHM
# =============================================================================

print(f"\n{SEP}")
print("§3 — GRID-CONVERGENCE STUDY: THOMAS' ALGORITHM")
print(SEP)
print(f"\n  {'n':>6}  {'h':>10}  {'Max error':>12}  {'L2 error':>12}  {'Order':>8}")
print("  " + "-"*56)

grid_sizes = [4, 8, 16, 32, 64, 128, 256]
thomas_errors_max = []
thomas_errors_l2  = []
thomas_hs         = []

prev_err = None
for n in grid_sizes:
    x_t, u_t   = solve_poisson_thomas(n)
    u_ex        = exact_solution(x_t)
    err_max     = np.max(np.abs(u_t - u_ex))
    err_l2      = np.sqrt(np.mean((u_t - u_ex)**2))
    thomas_errors_max.append(err_max)
    thomas_errors_l2.append(err_l2)
    thomas_hs.append(1.0 / (n + 1))
    order = (f"{math.log2(prev_err/err_max):.2f}"
             if prev_err is not None else "  —")
    print(f"  {n:>6}  {1/(n+1):>10.6f}  {err_max:>12.4e}  {err_l2:>12.4e}  {order:>8}")
    prev_err = err_max

print(f"\n  → Convergence order ≈ 2.00  (O(h²) as expected for 2nd-order FD)")

# =============================================================================
# §4 — HHL BUILDING BLOCKS
# =============================================================================
#
# All operations are explicit matrix / vector products on the full
# 2^(1 + M_CLOCK + N_SYS) dimensional Hilbert space.
#
# Register layout  (MSB → LSB in bitstring notation):
#   qubit 0            : ancilla
#   qubits 1 .. M      : clock register (QPE)
#   qubits M+1 .. M+S  : system register (S = N_SYS qubits for 2^S × 2^S A)
#
# QPE evolution parameter t = 2π / λ_max so that the largest eigenvalue
# maps to phase 1 exactly when it is a dyadic fraction of λ_max.

def tensor(*ops):
    """Kronecker product of a sequence of matrices."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def expm_hermitian(H, t):
    """
    Exact matrix exponential  exp(i H t)  for a Hermitian H.
    Uses the spectral decomposition: exp(iHt) = U diag(exp(iλt)) U†.
    This is exact — no Trotter approximation.
    """
    vals, vecs = np.linalg.eigh(H)
    return vecs @ np.diag(np.exp(1j * vals * t)) @ vecs.conj().T


def qft_matrix(n_qubits):
    """Exact QFT matrix on n_qubits qubits (DFT normalised by 1/√N)."""
    N   = 2 ** n_qubits
    idx = np.arange(N)
    return np.exp(2j * np.pi * np.outer(idx, idx) / N) / np.sqrt(N)


# One-qubit building blocks
I2    = np.eye(2, dtype=complex)
H1    = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
proj0 = np.array([[1, 0], [0, 0]], dtype=complex)   # |0⟩⟨0|
proj1 = np.array([[0, 0], [0, 1]], dtype=complex)   # |1⟩⟨1|


# =============================================================================
# §5 — FULL HHL PIPELINE
# =============================================================================

def hhl_poisson(A, b_vec, M_CLOCK, verbose=True):
    """
    Solve  A x = b  using the HHL algorithm.

    The function is self-contained: it builds every unitary from scratch
    using NumPy Kronecker products and matrix exponentials.  No quantum
    libraries are used.

    Parameters
    ----------
    A       : (n, n) Hermitian positive-definite matrix  (n must be a power of 2)
    b_vec   : (n,) right-hand side vector
    M_CLOCK : number of clock qubits (more → better phase resolution)
    verbose : print stage-by-stage diagnostics

    Returns
    -------
    x_hhl      : (n,) reconstructed solution (rescaled to match b_vec units)
    success_p  : post-selection probability  P(ancilla = |1⟩)
    info       : dict with intermediate diagnostics

    Algorithm stages
    ----------------
    Stage 1  State preparation   |ψ₀⟩ = |0⟩_anc ⊗ |0⟩_clk ⊗ |b/‖b‖⟩_sys
    Stage 2a Hadamards           H^⊗M on clock register
    Stage 2b Controlled-U^(2^k)  stamp eigenphases into clock register
    Stage 2c Inverse QFT         clock register → computational basis
    Stage 3  Controlled rotation  ancilla rotation C/λ̃ |1⟩ + √(1-C²/λ̃²)|0⟩
    Stage 4  Inverse QPE          uncompute clock register (exact adjoint)
    Stage 5  Post-selection       keep ancilla=1 component; renormalise
    Stage 6  Extract solution     read system register amplitudes
    """

    n      = len(b_vec)
    N_SYS  = int(round(math.log2(n)))
    assert 2**N_SYS == n, "n must be a power of 2"

    N_TOT  = 1 + M_CLOCK + N_SYS
    DIM    = 2**N_TOT

    # ── Spectral data (used throughout) ──────────────────────────────────────
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    lambda_min = eigenvalues.min()
    lambda_max = eigenvalues.max()
    kappa      = lambda_max / lambda_min

    # Evolution time: t = 2π / λ_max  so λ_max maps to phase 1.
    # With M_CLOCK bits the phase resolution is  Δφ = 1/2^M.
    # λ_j maps to clock value  k_j = round(λ_j * 2^M / λ_max).
    T_QPE = 2.0 * np.pi / lambda_max

    # Scaling constant for the ancilla rotation.
    # We need  C / λ̃_j ≤ 1  for all j.  Choose C = λ_min / λ_max,
    # expressed in clock-register units where eigenvalues run from
    # C_clock = λ_min/λ_max * 2^M to 2^M.
    C_scale = lambda_min / lambda_max   # in [0,1], relative to λ_max

    if verbose:
        print(f"\n  Matrix size   : {n}×{n}  ({N_SYS} system qubit{'s' if N_SYS>1 else ''})")
        print(f"  Clock qubits  : {M_CLOCK}  (phase resolution = 1/{2**M_CLOCK})")
        print(f"  Total qubits  : {N_TOT}  (Hilbert dim = {DIM})")
        print(f"  λ range       : [{lambda_min:.4f}, {lambda_max:.4f}]")
        print(f"  κ             : {kappa:.4f}")
        print(f"  T_QPE         : {T_QPE:.4f}")

    # ── Identity matrices for each register ──────────────────────────────────
    I_anc = I2
    I_clk = np.eye(2**M_CLOCK, dtype=complex)
    I_sys = np.eye(2**N_SYS,   dtype=complex)

    # Hadamard layer on clock: I_anc ⊗ H^⊗M ⊗ I_sys
    H_clk = H1
    for _ in range(M_CLOCK - 1):
        H_clk = np.kron(H_clk, H1)
    U_hadamard = tensor(I_anc, H_clk, I_sys)

    # Inverse / forward QFT on clock register
    QFT_M  = qft_matrix(M_CLOCK)
    IQFT_M = QFT_M.conj().T
    U_iqft = tensor(I_anc, IQFT_M, I_sys)
    U_qft  = tensor(I_anc, QFT_M,  I_sys)

    # ── Stage 1: State preparation ───────────────────────────────────────────
    b_norm  = np.linalg.norm(b_vec)
    b_state = b_vec / b_norm          # normalised |b⟩

    # |ψ₀⟩ = |0⟩_anc ⊗ |0⟩_clk ⊗ |b⟩_sys
    # In the full basis the system register is the LEAST significant block.
    # Basis index for |0⟩_anc |0⟩_clk |s⟩_sys  is just  s  (s=0,...,n-1).
    psi = np.zeros(DIM, dtype=complex)
    for s in range(n):
        psi[s] = b_state[s]

    assert abs(np.linalg.norm(psi) - 1.0) < 1e-12, "State preparation failed"

    # ── Stage 2a: Hadamards on clock ─────────────────────────────────────────
    psi = U_hadamard @ psi

    # ── Stage 2b: Controlled-U^(2^k) for k = 0,...,M-1 ──────────────────────
    # U = exp(i A T_QPE).  Clock qubit k is at position (1+k) from the MSB,
    # so it sits between  (1+k) identity qubits on its left  and
    # (M_CLOCK - k - 1 + N_SYS) identity qubits on its right.
    for k in range(M_CLOCK):
        U_pow = expm_hermitian(A, T_QPE * 2**k)
        n_left  = 1 + k
        n_right = M_CLOCK - k - 1   # clock qubits to the right of qubit k
        I_left  = np.eye(2**n_left,  dtype=complex)
        I_right = np.eye(2**n_right, dtype=complex)
        I_right_sys = np.kron(I_right, I_sys)
        # Controlled-U: |0⟩⟨0| ⊗ I_rest  +  |1⟩⟨1| ⊗ U^(2^k) ⊗ I_right
        ctrl_U = (tensor(I_left, proj0, I_right_sys) +
                  tensor(I_left, proj1,
                         np.kron(I_right, U_pow)))
        psi = ctrl_U @ psi

    # ── Stage 2c: Inverse QFT on clock ───────────────────────────────────────
    psi = U_iqft @ psi

    # Diagnostics: marginalise over ancilla and system to see clock distribution
    clock_probs = np.zeros(2**M_CLOCK)
    for idx, amp in enumerate(psi):
        bits      = format(idx, f'0{N_TOT}b')
        clk_val   = int(bits[1:1+M_CLOCK], 2)
        clock_probs[clk_val] += abs(amp)**2

    # ── Stage 3: Controlled ancilla rotation ─────────────────────────────────
    # For each basis state with ancilla=0, find its clock value k,
    # compute  λ̃ = k / 2^M * λ_max,
    # and apply the rotation  C/λ̃ |1⟩ + √(1-C²/λ̃²) |0⟩.
    #
    # The ancilla is the MSB: bit N_TOT-1 in the index.
    # Clock occupies bits N_TOT-2 down to N_TOT-1-M_CLOCK.
    # System occupies the N_SYS least significant bits.
    anc_bit_pos  = N_TOT - 1
    clk_bit_high = N_TOT - 2
    clk_bit_low  = N_TOT - 1 - M_CLOCK

    psi_rot = psi.copy()
    for idx in range(DIM):
        if (idx >> anc_bit_pos) & 1:
            continue    # already ancilla=1: handled as the partner of idx0

        idx1 = idx ^ (1 << anc_bit_pos)   # flip ancilla bit

        # Extract clock integer value
        bits    = format(idx, f'0{N_TOT}b')
        clk_int = int(bits[1:1+M_CLOCK], 2)
        if clk_int == 0:
            continue    # λ̃ = 0 → can't rotate; leave alone

        # Eigenvalue estimate from clock register
        lambda_tilde = clk_int / 2**M_CLOCK * lambda_max

        # Rotation angle
        ratio = C_scale * lambda_max / lambda_tilde  # = C_scale if λ̃ = λ_max
        ratio = min(ratio, 1.0)                       # numerical safety

        cos_t = math.sqrt(max(0.0, 1.0 - ratio**2))
        sin_t = ratio

        # Apply 2×2 rotation in {|0⟩_anc, |1⟩_anc} subspace
        a0 = psi[idx]
        a1 = psi[idx1]
        psi_rot[idx]  = cos_t * a0 - sin_t * a1
        psi_rot[idx1] = sin_t * a0 + cos_t * a1

    psi = psi_rot

    # ── Stage 4: Inverse QPE (uncompute clock) ────────────────────────────────
    # Apply the exact adjoint of Stage 2: QFT → controlled-U† in reverse order → H
    psi = U_qft @ psi

    for k in reversed(range(M_CLOCK)):
        U_pow_dag = expm_hermitian(A, -T_QPE * 2**k)
        n_left    = 1 + k
        n_right   = M_CLOCK - k - 1
        I_left    = np.eye(2**n_left,  dtype=complex)
        I_right   = np.eye(2**n_right, dtype=complex)
        I_right_sys = np.kron(I_right, I_sys)
        ctrl_Udag   = (tensor(I_left, proj0, I_right_sys) +
                       tensor(I_left, proj1,
                              np.kron(I_right, U_pow_dag)))
        psi = ctrl_Udag @ psi

    psi = U_hadamard @ psi   # H† = H for Hadamard

    # Verify clock is back to |0⟩^⊗M
    clk_zero_prob = sum(
        abs(psi[idx])**2
        for idx in range(DIM)
        if int(format(idx, f'0{N_TOT}b')[1:1+M_CLOCK], 2) == 0
    )
    if verbose:
        print(f"\n  Clock disentanglement check: P(clock=0) = {clk_zero_prob:.6f}"
              f"  (should be ≈ 1)")

    # ── Stage 5: Post-selection on ancilla = 1 ────────────────────────────────
    psi_post = np.zeros(DIM, dtype=complex)
    for idx, amp in enumerate(psi):
        if (idx >> anc_bit_pos) & 1:
            psi_post[idx] = amp

    success_p = float(np.sum(np.abs(psi_post)**2))
    if success_p < 1e-14:
        if verbose:
            print("  WARNING: success probability is effectively zero.")
        return None, 0.0, {}

    psi_post /= math.sqrt(success_p)

    # ── Stage 6: Extract system register amplitudes ───────────────────────────
    # Keep only ancilla=1, clock=0...0 components for the solution
    x_amps = np.zeros(n, dtype=complex)
    for idx, amp in enumerate(psi_post):
        bits = format(idx, f'0{N_TOT}b')
        if bits[0] == '1' and bits[1:1+M_CLOCK] == '0' * M_CLOCK:
            sys_val        = int(bits[1+M_CLOCK:], 2)
            x_amps[sys_val] = amp

    # The state encodes |x⟩ ∝ A⁻¹|b/‖b‖⟩.
    # To recover the physical solution we rescale:
    #   x_phys = ‖b‖ * x_amps / ‖x_amps‖  * ‖A⁻¹b‖ / ‖A⁻¹(b/‖b‖)‖
    # Since we only know the direction from the quantum register, we
    # determine the overall scale by a dot-product projection against
    # the classical solution (or any non-zero reference).
    x_raw = x_amps.real
    norm_raw = np.linalg.norm(x_raw)
    if norm_raw < 1e-12:
        return None, success_p, {}

    x_raw /= norm_raw   # unit vector in the direction of A⁻¹b

    # Rescale: the full physical solution is  x = (b_norm / ‖A b/‖b‖‖) * x_raw
    # A⁻¹b/‖b‖ has norm ‖A⁻¹b‖/‖b‖, so x_phys = ‖b‖ * ‖A⁻¹b/‖b‖‖ * x_raw
    # We estimate ‖A⁻¹b/‖b‖‖ from the post-selected state norm before
    # normalisation — but the cleanest approach for a simulation is to use
    # the known direction and rescale via a least-squares projection:
    x_exact_ref = np.linalg.solve(A, b_vec)   # classical reference for scale
    scale = np.dot(x_exact_ref, x_raw) / np.dot(x_raw, x_raw)
    x_hhl = scale * x_raw

    if verbose:
        print(f"  Post-selection P(ancilla=1) = {success_p:.6f}")
        print(f"  Relative error vs exact:      "
              f"{np.linalg.norm(x_hhl - x_exact_ref)/np.linalg.norm(x_exact_ref):.4e}")

    info = {
        'clock_probs': clock_probs,
        'success_p'  : success_p,
        'eigenvalues': eigenvalues,
        'kappa'      : kappa,
        'T_QPE'      : T_QPE,
        'N_SYS'      : N_SYS,
        'M_CLOCK'    : M_CLOCK,
        'N_TOT'      : N_TOT,
        'DIM'        : DIM,
        'lambda_min' : lambda_min,
        'lambda_max' : lambda_max,
    }
    return x_hhl, success_p, info


# =============================================================================
# §6 — HHL APPLIED TO n=2 AND n=4 POISSON SYSTEMS
# =============================================================================

M_CLOCK_DEFAULT = 5   # 5 clock qubits  →  phase resolution 1/32

print(f"\n{SEP}")
print(f"§6 — HHL FOR THE POISSON EQUATION  (M_CLOCK={M_CLOCK_DEFAULT})")
print(SEP)

hhl_results = {}

for n_pts in [2, 4]:
    print(f"\n{'─'*60}")
    print(f"  n = {n_pts} interior points  "
          f"({'1 system qubit' if n_pts==2 else '2 system qubits'})")
    print(f"{'─'*60}")

    A_p, b_p, x_grid, h_p = build_poisson_system(n_pts)
    x_th, u_th = solve_poisson_thomas(n_pts)
    u_ex       = exact_solution(x_grid)

    x_hhl, sp, info = hhl_poisson(A_p, b_p, M_CLOCK_DEFAULT, verbose=True)

    if x_hhl is not None:
        err_hhl    = np.max(np.abs(x_hhl - u_ex))
        err_thomas = np.max(np.abs(u_th  - u_ex))
        print(f"\n  Thomas error (max):  {err_thomas:.4e}")
        print(f"  HHL    error (max):  {err_hhl:.4e}")
        print(f"  Thomas solution:     {np.round(u_th, 8)}")
        print(f"  HHL    solution:     {np.round(x_hhl, 8)}")
        print(f"  Exact  solution:     {np.round(u_ex,  8)}")
        hhl_results[n_pts] = {
            'x_grid' : x_grid,
            'u_exact': u_ex,
            'u_thomas': u_th,
            'u_hhl'  : x_hhl,
            'info'   : info,
            'err_hhl': err_hhl,
            'err_th' : err_thomas,
            'sp'     : sp,
        }

# =============================================================================
# §7 — HHL CLOCK-PRECISION STUDY  (n=2, varying M_CLOCK)
# =============================================================================

print(f"\n{SEP}")
print("§7 — HHL CLOCK-PRECISION STUDY  (n=2, varying M_CLOCK)")
print(SEP)
print(f"\n  {'M_CLOCK':>8}  {'Resolution':>12}  {'HHL max err':>13}  "
      f"{'P(success)':>12}  {'DIM':>6}")
print("  " + "-"*58)

A2, b2, x2, h2 = build_poisson_system(2)
u_ex2 = exact_solution(x2)

clock_study = []
for m in range(3, 9):
    x_hhl_m, sp_m, info_m = hhl_poisson(A2, b2, m, verbose=False)
    if x_hhl_m is not None:
        err_m = np.max(np.abs(x_hhl_m - u_ex2))
        clock_study.append((m, err_m, sp_m, info_m['DIM']))
        print(f"  {m:>8}  {1/2**m:>12.6f}  {err_m:>13.4e}  "
              f"{sp_m:>12.6f}  {info_m['DIM']:>6}")

# =============================================================================
# §8 — COMPREHENSIVE COMPARISON TABLE
# =============================================================================

print(f"\n{SEP}")
print("§8 — COMPARISON: HHL vs THOMAS vs EXACT  (n=4)")
print(SEP)

if 4 in hhl_results:
    r = hhl_results[4]
    x4_g = r['x_grid']
    print(f"\n  {'x_i':>8}  {'Exact':>12}  {'Thomas':>12}  {'HHL':>12}  "
          f"{'|Th−Ex|':>10}  {'|HHL−Ex|':>10}")
    print("  " + "-"*72)
    for xi, ue, ut, uh in zip(x4_g, r['u_exact'], r['u_thomas'], r['u_hhl']):
        print(f"  {xi:>8.4f}  {ue:>12.8f}  {ut:>12.8f}  {uh:>12.8f}  "
              f"{abs(ut-ue):>10.3e}  {abs(uh-ue):>10.3e}")

    info4 = r['info']
    print(f"\n  Summary (n=4, M_CLOCK={M_CLOCK_DEFAULT}):")
    print(f"    Qubits    : {info4['N_TOT']}  "
          f"(1 anc + {info4['M_CLOCK']} clock + {info4['N_SYS']} system)")
    print(f"    Hilbert   : 2^{info4['N_TOT']} = {info4['DIM']} dimensions")
    print(f"    κ(A)      : {info4['kappa']:.4f}")
    print(f"    P(success): {r['sp']:.6f}")
    print(f"    Thomas err: {r['err_th']:.4e}  (O(h²) finite-difference error)")
    print(f"    HHL    err: {r['err_hhl']:.4e}  "
          f"(phase resolution + FD discretisation)")

print(f"""
Algorithm comparison:
┌─────────────────┬──────────────────────────┬──────────────────────────────┐
│ Property        │ Thomas' algorithm        │ HHL algorithm                │
├─────────────────┼──────────────────────────┼──────────────────────────────┤
│ Complexity      │ O(n)  classical          │ O(κ² log n)  quantum         │
│ Error source    │ O(h²) truncation only    │ O(h²) + QPE phase resolution │
│ Output          │ Full classical vector    │ Quantum state ∝ A⁻¹b         │
│ Scaling limit   │ Dense systems slow       │ Advantage for large sparse   │
│ Simulation cost │ O(n) ops                 │ O(2^(M+S)) explicit ops      │
│ Implementation  │ Straightforward          │ Full statevector simulation  │
└─────────────────┴──────────────────────────┴──────────────────────────────┘
""")

# =============================================================================
# §9 — VISUALISATION
# =============================================================================

fig = plt.figure(figsize=(18, 12))
fig.suptitle(
    "HHL Algorithm vs Thomas' Algorithm — 1-D Poisson Equation\n"
    r"$-u''=\sin(\pi x)$,  $u(0)=u(1)=0$,  exact: $u(x)=\sin(\pi x)/\pi^2$",
    fontsize=13, fontweight='bold'
)
gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.40)

# ── Panel 1: Poisson matrix (n=4) ──────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(A4 * h4**2, cmap='RdBu_r', aspect='equal',
               vmin=-1.5, vmax=2.5)
plt.colorbar(im, ax=ax, shrink=0.8, label=r'$h^2 \cdot A_{ij}$')
for i in range(4):
    for j in range(4):
        val = A4[i,j] * h4**2
        ax.text(j, i, f"{val:.0f}", ha='center', va='center',
                fontsize=11, fontweight='bold',
                color='white' if abs(val) > 1 else 'black')
ax.set_xticks(range(4)); ax.set_yticks(range(4))
ax.set_title(r'Poisson matrix $h^2 A$  ($n=4$)' + '\n'
             r'$A = h^{-2}\,\mathrm{tridiag}(-1,2,-1)$',
             fontsize=10, fontweight='bold')

# ── Panel 2: Grid convergence (Thomas) ─────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
hs_arr   = np.array(thomas_hs)
errs_arr = np.array(thomas_errors_max)
ax.loglog(hs_arr, errs_arr, 'bo-', lw=2, ms=7, label="Thomas' algorithm")
# O(h²) reference line
ref = errs_arr[0] * (hs_arr / hs_arr[0])**2
ax.loglog(hs_arr, ref, 'k--', lw=1.5, alpha=0.6, label=r'$O(h^2)$ reference')
ax.set_xlabel('Grid spacing $h$', fontsize=10)
ax.set_ylabel(r'Max error $\|u_h - u_\mathrm{ex}\|_\infty$', fontsize=10)
ax.set_title("Grid Convergence: Thomas' Algorithm\n"
             r"$O(h^2)$ as expected for 2nd-order FD",
             fontsize=10, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which='both')

# ── Panel 3: HHL clock precision (n=2) ─────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
if clock_study:
    m_vals   = [c[0] for c in clock_study]
    err_vals = [c[1] for c in clock_study]
    sp_vals  = [c[2] for c in clock_study]
    ax2_twin = ax.twinx()
    ax.semilogy(m_vals, err_vals, 'r-s', lw=2, ms=7, label='HHL max error')
    ax2_twin.plot(m_vals, sp_vals, 'g--^', lw=1.5, ms=6, label='P(success)')
    ax.set_xlabel('Clock qubits $M$', fontsize=10)
    ax.set_ylabel('Max error (log scale)', fontsize=10, color='firebrick')
    ax2_twin.set_ylabel('Post-selection probability', fontsize=10,
                        color='darkgreen')
    ax.tick_params(axis='y', labelcolor='firebrick')
    ax2_twin.tick_params(axis='y', labelcolor='darkgreen')
    ax.set_title('HHL Precision vs Clock Qubits\n'
                 r'($n=2$, varying $M$)',
                 fontsize=10, fontweight='bold')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.grid(True, alpha=0.3)

# ── Panel 4: Solution comparison (n=4) ─────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
if 4 in hhl_results:
    r = hhl_results[4]
    x_dense = np.linspace(0, 1, 300)
    ax.plot(x_dense, exact_solution(x_dense), 'k-', lw=2,
            label=r'Exact: $\sin(\pi x)/\pi^2$')
    ax.plot(r['x_grid'], r['u_thomas'], 'bs--', ms=10, lw=1.5,
            label=f"Thomas' (err={r['err_th']:.2e})")
    ax.plot(r['x_grid'], r['u_hhl'], 'r^:', ms=10, lw=1.5,
            label=f"HHL (err={r['err_hhl']:.2e})")
    ax.set(xlabel='$x$', ylabel='$u(x)$',
           title='Solution Comparison  ($n=4$)\n'
                 f'M_CLOCK={M_CLOCK_DEFAULT}, '
                 f'{hhl_results[4]["info"]["N_TOT"]} qubits total')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# ── Panel 5: QPE clock distribution (n=4) ──────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
if 4 in hhl_results:
    info4 = hhl_results[4]['info']
    cp    = info4['clock_probs']
    evals = info4['eigenvalues']
    T     = info4['T_QPE']
    M     = info4['M_CLOCK']
    lmax  = info4['lambda_max']
    x_pos = np.arange(len(cp))
    ax.bar(x_pos, cp, color='steelblue', edgecolor='k', alpha=0.85, width=0.8)
    for lam in evals:
        k_exp = lam / lmax * 2**M
        ax.axvline(k_exp, color='firebrick', lw=2, ls='--',
                   label=f'$\\lambda={lam:.2f}$ → $k\\approx{k_exp:.1f}$')
    ax.set_xlabel('Clock register value $k$', fontsize=10)
    ax.set_ylabel('Probability', fontsize=10)
    ax.set_title(f'QPE Clock Distribution  ($n=4$, $M={M}$)\n'
                 f'Resolution = $1/{2**M}$ of $\\lambda_{{\\max}}$',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis='y')

# ── Panel 6: Error comparison bar chart ────────────────────────────────────
ax = fig.add_subplot(gs[1, 2])
cases  = []
labels = []
colors_bar = []

# Thomas for various n
for n_t in [4, 8, 16, 32]:
    x_t, u_t = solve_poisson_thomas(n_t)
    e = np.max(np.abs(u_t - exact_solution(x_t)))
    cases.append(e)
    labels.append(f"Thomas\nn={n_t}")
    colors_bar.append('steelblue')

# HHL for n=2 and n=4
for n_h in [2, 4]:
    if n_h in hhl_results:
        cases.append(hhl_results[n_h]['err_hhl'])
        labels.append(f"HHL\nn={n_h}")
        colors_bar.append('firebrick')

x_bar = np.arange(len(cases))
ax.bar(x_bar, cases, color=colors_bar, edgecolor='k', alpha=0.85)
ax.set_yscale('log')
ax.set_xticks(x_bar)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel(r'Max error $\|u_h - u_\mathrm{ex}\|_\infty$', fontsize=10)
ax.set_title('Error Summary: Thomas (blue) vs HHL (red)\n'
             'Both sources: FD discretisation + HHL phase resolution',
             fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Annotate bar values
for xi, val in zip(x_bar, cases):
    ax.text(xi, val * 1.5, f'{val:.1e}', ha='center', va='bottom',
            fontsize=7, fontweight='bold')

plt.savefig('hhl_poisson.png', dpi=140, bbox_inches='tight')
plt.show()
print("\n✓ Figure saved to hhl_poisson.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print(f"\n{SEP}")
print("FINAL SUMMARY")
print(SEP)
print(f"""
Problem:  -u'' = sin(πx),  u(0)=u(1)=0
Exact:    u(x) = sin(πx)/π²

THOMAS' ALGORITHM
  Solver : tridiagonal forward-sweep + back-substitution  (O(n) operations)
  Error  : O(h²) = O(1/(n+1)²) — pure finite-difference truncation error
  n= 4:  max err ≈ {np.max(np.abs(solve_poisson_thomas(4)[1]-exact_solution(solve_poisson_thomas(4)[0]))):.2e}
  n=16:  max err ≈ {np.max(np.abs(solve_poisson_thomas(16)[1]-exact_solution(solve_poisson_thomas(16)[0]))):.2e}
  n=64:  max err ≈ {np.max(np.abs(solve_poisson_thomas(64)[1]-exact_solution(solve_poisson_thomas(64)[0]))):.2e}

HHL ALGORITHM  (M_CLOCK={M_CLOCK_DEFAULT})
  n=2  (1 system qubit): {hhl_results[2]['err_hhl']:.2e}  error,  P(success)={hhl_results[2]['sp']:.4f}
  n=4  (2 system qubit): {hhl_results[4]['err_hhl']:.2e}  error,  P(success)={hhl_results[4]['sp']:.4f}

Register layout for n=4 (M_CLOCK={M_CLOCK_DEFAULT}):
  1 ancilla + {M_CLOCK_DEFAULT} clock + 2 system = {1+M_CLOCK_DEFAULT+2} qubits
  Hilbert space dimension = 2^{1+M_CLOCK_DEFAULT+2} = {2**(1+M_CLOCK_DEFAULT+2)}

Key equations implemented:
  A u = b              [FD Poisson: A = h⁻² tridiag(-1,2,-1)]
  |b⟩ = b/‖b‖          [state preparation]
  QPE: |u_j⟩|0⟩ → |u_j⟩|λ̃_j⟩  [eigenvalue stamping via e^(iAt)]
  Rotation: C/λ̃_j|1⟩ + √(1-C²/λ̃_j²)|0⟩  [inversion step]
  |x⟩ ∝ A⁻¹|b⟩ = Σ_j β_j/λ_j |u_j⟩   [post-selected solution]

Thomas' algorithm is exact (within O(h²)) and O(n).
HHL adds a QPE phase-resolution error on top of the O(h²) FD error,
but has theoretical complexity advantage O(κ² log n) for large sparse
systems when only expectation values of the solution are needed.
""")
