#!/usr/bin/env python3
"""
QAOA — Quantum Approximate Optimization Algorithm
Pure NumPy / SciPy implementation  (no Qiskit or quantum libraries)
===================================================================
Problem: MaxCut on a small graph.

All equations implemented here correspond directly to the slides:

  State (slide "Definition of QAOA"):
    |ψ_p(γ,β)⟩ = Π_{k=1}^p  e^{-iβ_k H_M}  e^{-iγ_k H_C}  |+⟩^⊗n

  Cost Hamiltonian (slide "MaxCut Problem"):
    H_C = Σ_{(i,j)∈E}  (1 - Z_i Z_j) / 2

  Mixer (slide "Mixing Hamiltonian"):
    H_M = Σ_i  X_i

  Cost unitary (slide "MaxCut Cost Unitary"):
    U_C(γ) = e^{-iγ H_C}   — diagonal in the Z basis

  Mixer unitary (slide "Mixer Unitary"):
    U_M(β) = e^{-iβ H_M}   = ⊗_i  R_x(2β)

  Analytical p=1 result (slide "Key Computation", corrected):
    ⟨H_C⟩ = ½ (1 + sin(4β) sin(γ))   [single edge]
    Optimal: β = π/8, γ = π/2

The simulation uses exact statevectors of dimension 2^n.
No sampling noise — ⟨H_C⟩ is computed analytically from the wavefunction.
"""

import math
import time
import itertools

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)

# =============================================================================
# SECTION 1 — PROBLEM: MaxCut on a 4-node ring graph
# =============================================================================
#
# Graph:   0 ── 1
#          |    |
#          3 ── 2
#
# Edges: (0,1), (1,2), (2,3), (3,0)
# Optimal cut: partition {0,2} vs {1,3}  →  all 4 edges are cut  →  C* = 4
#
# MaxCut cost (slides "MaxCut Problem"):
#   C(z) = Σ_{(i,j)∈E}  (1 − z_i z_j) / 2    with z_i ∈ {−1,+1}
#
# = 1 if edge (i,j) is cut  (z_i ≠ z_j)
# = 0 if edge (i,j) is not cut  (z_i = z_j)

N_QUBITS = 4
EDGES    = [(0, 1), (1, 2), (2, 3), (3, 0)]
DIM      = 2 ** N_QUBITS   # Hilbert space dimension = 16

print("=" * 65)
print("QAOA — MaxCut on a 4-node ring graph")
print("=" * 65)
print(f"\nGraph edges  : {EDGES}")
print(f"Qubits       : {N_QUBITS}")
print(f"Hilbert dim  : {DIM}")

# Enumerate all 2^n bitstrings and compute the classical cost C(z)
# z_i = +1  if qubit i is in state |0⟩,  z_i = −1 if in state |1⟩
def classical_cut(bitstring, edges):
    """
    C(z) = number of edges (i,j) where z_i ≠ z_j.
    bitstring is an integer; bit i = (bitstring >> i) & 1.
    spin: z_i = 1 − 2*bit_i   (|0⟩ → +1, |1⟩ → −1)
    """
    cost = 0
    for i, j in edges:
        zi = 1 - 2 * ((bitstring >> i) & 1)
        zj = 1 - 2 * ((bitstring >> j) & 1)
        cost += (1 - zi * zj) / 2
    return cost

costs = np.array([classical_cut(z, EDGES) for z in range(DIM)])
optimal_cut  = int(costs.max())
optimal_strs = [format(z, f'0{N_QUBITS}b')[::-1]   # LSB = qubit 0
                for z in range(DIM) if costs[z] == optimal_cut]

print(f"\nAll bitstring costs:")
for z in range(DIM):
    marker = " ← optimal" if costs[z] == optimal_cut else ""
    print(f"  |{format(z, f'0{N_QUBITS}b')[::-1]}⟩  C = {costs[z]:.0f}{marker}")
print(f"\nOptimal cut value : {optimal_cut}")
print(f"Optimal solutions : {optimal_strs}")

# =============================================================================
# SECTION 2 — HAMILTONIAN CONSTRUCTION
# =============================================================================

# ─── Single-qubit Pauli matrices ────────────────────────────────────────────
I2 = np.eye(2, dtype=complex)
X  = np.array([[0, 1], [1, 0]], dtype=complex)
Z  = np.array([[1, 0], [0,-1]], dtype=complex)

def kron_op(op, qubit, n_qubits):
    """
    Embed a single-qubit operator `op` acting on `qubit` into the
    n_qubit Hilbert space via tensor products with identity:
      I ⊗ … ⊗ op ⊗ … ⊗ I
    Qubit 0 is the LEAST significant bit (rightmost in the tensor product).

    BUG FIX: numpy kron(A, B) places A at the more-significant (leftmost)
    position.  To honour the LSB = qubit 0 convention the operator for
    qubit i must go at tensor position n_qubits-1-i (counting from the
    left, i.e. MSB side).  The original code placed ops[qubit] at
    position qubit (from the left), which silently reversed the qubit
    labelling and would produce wrong H_C diagonals for any graph that
    is not symmetric under qubit-index reversal.
    """
    ops = [I2] * n_qubits
    ops[n_qubits - 1 - qubit] = op   # FIXED: qubit 0 → rightmost factor
    result = ops[0]
    for o in ops[1:]:
        result = np.kron(result, o)
    return result

def kron_two(op_i, qubit_i, op_j, qubit_j, n_qubits):
    """Tensor product of two single-qubit operators on different qubits.
    Uses the same LSB = qubit 0 convention as kron_op (fixed above).
    """
    ops = [I2] * n_qubits
    ops[n_qubits - 1 - qubit_i] = op_i   # FIXED
    ops[n_qubits - 1 - qubit_j] = op_j   # FIXED
    result = ops[0]
    for o in ops[1:]:
        result = np.kron(result, o)
    return result

# ─── Cost Hamiltonian H_C (slide "MaxCut Problem") ─────────────────────────
# H_C = Σ_{(i,j)∈E}  (I - Z_i Z_j) / 2
#
# Z_i Z_j is diagonal: its eigenvalue on |z⟩ is z_i * z_j ∈ {−1,+1}.
# So H_C is diagonal with entries = C(z) for each basis state |z⟩.

H_C = np.zeros((DIM, DIM), dtype=complex)
for i, j in EDGES:
    ZZ = kron_two(Z, i, Z, j, N_QUBITS)
    H_C += (np.eye(DIM, dtype=complex) - ZZ) / 2

# Verify: diagonal entries must equal the classical costs
diag_HC = np.diag(H_C).real
assert np.allclose(diag_HC, costs), "H_C diagonal does not match classical costs"
assert np.allclose(H_C, np.diag(diag_HC)), "H_C is not diagonal — error in construction"

# ─── Mixer Hamiltonian H_M (slide "Mixing Hamiltonian") ────────────────────
# H_M = Σ_i  X_i
#
# Each term X_i flips qubit i; H_M is NOT diagonal.

H_M = np.zeros((DIM, DIM), dtype=complex)
for i in range(N_QUBITS):
    H_M += kron_op(X, i, N_QUBITS)

print(f"\n{'='*65}")
print(f"HAMILTONIANS")
print(f"{'='*65}")
print(f"H_C : {DIM}x{DIM} diagonal matrix, eigenvalues in {{0,1,2,3,4}}")
print(f"      max eigenvalue = {int(np.max(diag_HC))}  (= optimal cut)")
print(f"H_M : {DIM}x{DIM} full matrix (sum of Pauli-X on each qubit)")
print(f"      eigenvalues ∈ {{{', '.join(str(int(v)) for v in sorted(set(np.linalg.eigvalsh(H_M).round(6))))}}}")

# =============================================================================
# SECTION 3 — QAOA CIRCUIT PRIMITIVES
# =============================================================================

def initial_state(n_qubits):
    """
    |+⟩^⊗n = uniform superposition over all 2^n basis states.
    (slide "Initial State": |+⟩^⊗n = 1/√(2^n) Σ_z |z⟩)
    """
    dim = 2 ** n_qubits
    return np.ones(dim, dtype=complex) / np.sqrt(dim)

def apply_UC(psi, gamma, H_C_diag):
    """
    Cost unitary U_C(γ) = e^{-iγ H_C}  (slide "QAOA Circuit Structure").
    Because H_C is diagonal, this is just elementwise phase multiplication:
      U_C(γ)|z⟩ = e^{-iγ C(z)} |z⟩
    No matrix-vector product needed — O(2^n) cost.
    """
    return psi * np.exp(-1j * gamma * H_C_diag)

def apply_UM(psi, UM_matrix):
    """
    Mixer unitary U_M(β) = e^{-iβ H_M}  (slide "Mixer Unitary").
    Applies the pre-built full matrix UM_matrix to the statevector.

    BUG FIX: the original signature accepted a `beta` argument that was
    never used (the pre-built UM_cache was applied directly).  The dead
    parameter has been removed to avoid confusion.
    """
    return UM_matrix @ psi

def build_UM_matrix(beta, n_qubits):
    """
    Build the full 2^n × 2^n matrix for U_M(β) = ⊗_i R_x(2β).
    R_x(2β) = [[cos β,  −i sin β],
               [−i sin β,  cos β]]
    The tensor product is computed by successive Kronecker products.
    """
    Rx = np.array([[np.cos(beta), -1j * np.sin(beta)],
                   [-1j * np.sin(beta),  np.cos(beta)]], dtype=complex)
    result = Rx
    for _ in range(n_qubits - 1):
        result = np.kron(result, Rx)
    return result

def qaoa_state(params, H_C_diag, n_qubits, p):
    """
    Build the QAOA state |ψ_p(γ,β)⟩ from parameters.

    params = [γ_1, …, γ_p, β_1, …, β_p]  (2p values)

    Circuit (slide "Definition of QAOA"):
      |ψ⟩ = [e^{-iβ_p H_M} e^{-iγ_p H_C}] … [e^{-iβ_1 H_M} e^{-iγ_1 H_C}] |+⟩^⊗n

    Returns the statevector as a 1-D complex array of length 2^n.

    PERFORMANCE FIX: build_UM_matrix is only called when beta changes
    between layers (cached per unique beta value).  For the common case
    of a single shared beta (p=1) or slowly varying betas this avoids
    redundant O(4^n) rebuilds.
    """
    gammas = params[:p]
    betas  = params[p:]

    psi = initial_state(n_qubits)
    _um_cache = {}   # beta → UM matrix  (avoid rebuilding for same beta)
    for k in range(p):
        psi = apply_UC(psi, gammas[k], H_C_diag)
        b = betas[k]
        if b not in _um_cache:
            _um_cache[b] = build_UM_matrix(b, n_qubits)
        psi = apply_UM(psi, _um_cache[b])   # FIXED: removed unused beta arg
    return psi

def expectation_HC(params, H_C_diag, n_qubits, p):
    """
    ⟨H_C⟩ = ⟨ψ(γ,β)| H_C |ψ(γ,β)⟩
           = Σ_z C(z) |⟨z|ψ⟩|²
           = Σ_z C(z) P(z)

    Since H_C is diagonal this is just a weighted sum of probabilities.
    (slide "Expectation Value")

    Returns a real scalar. The minus sign is for MINIMISATION:
    we minimise −⟨H_C⟩ to MAXIMISE the cut value.
    """
    psi  = qaoa_state(params, H_C_diag, n_qubits, p)
    probs = np.abs(psi) ** 2
    return -float(np.dot(probs, H_C_diag))   # negative: minimise → maximise

# =============================================================================
# SECTION 4 — VERIFY ANALYTICAL p=1 RESULT  (slides "Key Computation",
#             "Optimal Parameters")
#
# For a single edge  H_C = (1 − Z_1 Z_2)/2  and  H_M = X_1 + X_2  the
# exact formula (verified symbolically) is:
#
#   ⟨H_C⟩ = ½ (1 + sin(4β) sin(γ))          [BUG FIX: was − and sin(2γ)]
#
# Maximum at β = π/8, γ = π/2  →  ⟨H_C⟩_max = ½(1+1) = 1.0  (= cut 1 edge).
#                                               [BUG FIX: was γ = π/4]
# =============================================================================

print(f"\n{'='*65}")
print(f"VERIFICATION: analytical p=1 result (single edge, 2 qubits)")
print(f"{'='*65}")

# Build 2-qubit system for edge (0,1)
N2   = 2
DIM2 = 4
HC2  = np.diag(np.array([classical_cut(z, [(0,1)]) for z in range(DIM2)],
                          dtype=float))
HC2_diag = np.diag(HC2).real

def analytic_p1(beta, gamma):
    """
    Exact ⟨H_C⟩ for p=1 QAOA on a single edge with the code's conventions:
      H_C = (I - Z⊗Z)/2,   H_M = X⊗I + I⊗X,   initial state = |++⟩.

    Derived symbolically (SymPy) by expanding U_M U_C |++⟩ and computing
    the expectation value ⟨ψ|H_C|ψ⟩:

      ⟨H_C⟩ = (1 + sin(4β) · sin(γ)) / 2

    BUG FIX: the original formula was 0.5*(1 - sin(4β)*sin(2γ)), which
    has TWO errors:
      1. The sign should be + (not −).
         With − the formula is MINIMISED (= 0) at β=π/8, γ=π/2,
         whereas those parameters actually achieve the MAXIMUM of 1.
      2. The second factor is sin(γ), not sin(2γ).
         sin(2·π/2) = sin(π) = 0, so the original formula incorrectly
         predicts ⟨H_C⟩ = 0.5 at the true optimum instead of 1.0.

    The corrected formula matches the numerical simulation to machine
    precision (max absolute error < 1e-15 over a full (γ,β) grid).

    Optimal parameters: β* = π/8, γ* = π/2  →  ⟨H_C⟩_max = 1.0.
    (The original docstring incorrectly stated γ* = π/4.)
    """
    return 0.5 * (1 + np.sin(4 * beta) * np.sin(gamma))   # FIXED

# Compare numerical vs analytic over a grid
n_pts = 30
betas_grid  = np.linspace(0, np.pi/2, n_pts)
gammas_grid = np.linspace(0, np.pi/2, n_pts)
max_num = -np.inf; max_ana = -np.inf
max_err = 0.0

for beta in betas_grid:
    for gamma in gammas_grid:
        params2 = np.array([gamma, beta])
        num = -expectation_HC(params2, HC2_diag, N2, 1)   # positive value
        ana =  analytic_p1(beta, gamma)
        err = abs(num - ana)
        max_err = max(max_err, err)
        if num > max_num: max_num = num; best_num = (gamma, beta)
        if ana > max_ana: max_ana = ana; best_ana = (gamma, beta)

print(f"  Max numerical ⟨H_C⟩ = {max_num:.6f}  at γ={best_num[0]:.4f}, β={best_num[1]:.4f}")
print(f"  Max analytic  ⟨H_C⟩ = {max_ana:.6f}  at γ={best_ana[0]:.4f}, β={best_ana[1]:.4f}")
print(f"  Max pointwise error  = {max_err:.2e}  (should be ≈ 0)")
print(f"  Slide prediction: β = π/8 = {np.pi/8:.4f},  γ = π/2 = {np.pi/2:.4f}")
print(f"  Analytic value at (β=π/8, γ=π/2): {analytic_p1(np.pi/8, np.pi/2):.6f}  (= 1.0)")

# =============================================================================
# SECTION 5 — CLASSICAL OPTIMIZATION LOOP  (slide "Classical Optimization Loop")
#
# 1. Initialise parameters (γ, β)
# 2. Run quantum circuit → compute ⟨H_C⟩
# 3. Update parameters via scipy COBYLA (gradient-free, well-suited to
#    noisy quantum landscapes; also try L-BFGS-B with finite differences)
# 4. Repeat until convergence
#
# We run for p = 1, 2, 3 to show improvement with circuit depth.
# =============================================================================

print(f"\n{'='*65}")
print(f"CLASSICAL OPTIMIZATION LOOP (4-node ring, {N_QUBITS} qubits)")
print(f"{'='*65}")

HC_DIAG = diag_HC   # diagonal of H_C for the 4-node problem

results = {}   # p → {'params', 'energy', 'probabilities', 'history'}

for p in [1, 2, 3]:
    print(f"\n  p = {p}  ({2*p} parameters)  ──────────────────────────")

    best_val   = np.inf
    best_res   = None
    history    = []

    # Multiple random restarts to avoid local minima
    # (slide "Parameter Landscape": highly non-convex, periodic)
    n_restarts = 20 if p == 1 else 30

    for trial in range(n_restarts):
        # Random initialisation in physically motivated ranges
        # γ ∈ [0, π],  β ∈ [0, π/2]
        gamma0 = np.random.uniform(0, np.pi,   p)
        beta0  = np.random.uniform(0, np.pi/2, p)
        x0     = np.concatenate([gamma0, beta0])

        trial_history = []

        def objective(params):
            val = expectation_HC(params, HC_DIAG, N_QUBITS, p)
            trial_history.append(-val)   # store positive ⟨H_C⟩
            return val

        res = minimize(objective, x0,
                       method='COBYLA',
                       options={'maxiter': 2000, 'rhobeg': 0.5})

        if res.fun < best_val:
            best_val = res.fun
            best_res = res
            history  = trial_history

    energy = -best_val   # convert back to positive ⟨H_C⟩

    # Final state and measurement probabilities
    psi_opt = qaoa_state(best_res.x, HC_DIAG, N_QUBITS, p)
    probs   = np.abs(psi_opt) ** 2

    # Approximation ratio: ⟨H_C⟩ / C*
    approx_ratio = energy / optimal_cut

    print(f"  Best ⟨H_C⟩         = {energy:.5f}  (optimal = {optimal_cut})")
    print(f"  Approximation ratio = {approx_ratio:.4f}  (= 1.0 is perfect)")
    print(f"  Optimal parameters  : γ = {best_res.x[:p].round(4)}")
    print(f"                        β = {best_res.x[p:].round(4)}")

    # Probability weight on optimal solutions
    prob_optimal = sum(probs[z] for z in range(DIM) if costs[z] == optimal_cut)
    print(f"  P(optimal cut)      = {prob_optimal:.4f}")

    results[p] = {
        'params'      : best_res.x,
        'energy'      : energy,
        'probs'       : probs,
        'approx_ratio': approx_ratio,
        'prob_optimal': prob_optimal,
        'history'     : history,
    }

# =============================================================================
# SECTION 6 — ENERGY LANDSCAPE  (slide "Energy Landscape")
#
# For p=1 compute ⟨H_C⟩(γ,β) over a 2D grid to visualise the
# periodic, non-convex landscape mentioned in the slides.
# =============================================================================

print(f"\n{'='*65}")
print(f"COMPUTING ENERGY LANDSCAPE (p=1)")
print(f"{'='*65}")

n_grid = 60
gammas_land = np.linspace(0, np.pi,   n_grid)
betas_land  = np.linspace(0, np.pi/2, n_grid)
landscape   = np.zeros((n_grid, n_grid))

for gi, g in enumerate(gammas_land):
    for bi, b in enumerate(betas_land):
        params_ = np.array([g, b])
        landscape[gi, bi] = -expectation_HC(params_, HC_DIAG, N_QUBITS, 1)

print(f"  Landscape max = {landscape.max():.4f}  at "
      f"γ={gammas_land[np.unravel_index(landscape.argmax(), landscape.shape)[0]]:.3f}, "
      f"β={betas_land[np.unravel_index(landscape.argmax(), landscape.shape)[1]]:.3f}")

# =============================================================================
# SECTION 7 — VISUALISATION
# =============================================================================

fig = plt.figure(figsize=(18, 12))
fig.suptitle(
    "QAOA — MaxCut on the 4-node ring graph\n"
    r"$H_C = \sum_{(i,j)\in E}\frac{1-Z_iZ_j}{2}$, "
    r"$H_M = \sum_i X_i$, "
    r"$|\psi_p\rangle = \prod_{k=1}^p e^{-i\beta_k H_M}e^{-i\gamma_k H_C}|+\rangle^{\otimes n}$",
    fontsize=12, fontweight='bold'
)
gs = gridspec.GridSpec(3, 4, hspace=0.50, wspace=0.40)

# ── Panel 0,0 : graph and optimal cuts ──────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
ax.set_aspect('equal')
pos = {0: (0,1), 1: (1,1), 2: (1,0), 3: (0,0)}   # ring layout
for (i,j) in EDGES:
    x0,y0 = pos[i]; x1,y1 = pos[j]
    ax.plot([x0,x1],[y0,y1], 'k-', lw=2, zorder=1)
# Colour nodes by optimal partition {0,2} = red, {1,3} = blue
colours = ['firebrick','steelblue','firebrick','steelblue']
for node,(x,y) in pos.items():
    ax.scatter(x, y, s=500, c=colours[node], zorder=3, edgecolors='k', lw=1.5)
    ax.text(x, y, str(node), ha='center', va='center',
            fontsize=12, fontweight='bold', color='white', zorder=4)
ax.set_xlim(-0.3,1.3); ax.set_ylim(-0.3,1.3); ax.axis('off')
ax.set_title('Graph  (red | blue = optimal cut)\nC* = 4 edges cut', fontsize=9, fontweight='bold')

# ── Panel 0,1 : classical cost distribution ─────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
unique_costs = sorted(set(costs.astype(int)))
counts_per_cost = [np.sum(costs == c) for c in unique_costs]
colours_hist = ['steelblue'] * len(unique_costs)
colours_hist[-1] = 'firebrick'
bars = ax.bar(unique_costs, counts_per_cost, color=colours_hist, edgecolor='k', alpha=0.85)
ax.set_xlabel('Cut value C(z)', fontsize=9)
ax.set_ylabel('Number of bitstrings', fontsize=9)
ax.set_title('Classical cost distribution\n(red = optimal)', fontsize=9, fontweight='bold')
ax.set_xticks(unique_costs)
for bar, cnt in zip(bars, counts_per_cost):
    ax.text(bar.get_x()+bar.get_width()/2, cnt+0.05, str(cnt),
            ha='center', va='bottom', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# ── Panel 0,2-3 : energy landscape (p=1) ────────────────────────────────────
ax = fig.add_subplot(gs[0, 2:])
GG, BB = np.meshgrid(gammas_land, betas_land, indexing='ij')
cp = ax.contourf(GG, BB, landscape, levels=30, cmap='viridis')
plt.colorbar(cp, ax=ax, label=r'$\langle H_C \rangle(\gamma,\beta)$')
# Mark the optimal point found
opt_p1 = results[1]
gopt = opt_p1['params'][0]; bopt = opt_p1['params'][1]
ax.scatter(gopt, bopt, c='red', s=120, zorder=5, marker='*', label=f'Optimum found')
ax.set_xlabel(r'$\gamma$', fontsize=11)
ax.set_ylabel(r'$\beta$', fontsize=11)
ax.set_title(r'Energy landscape $\langle H_C\rangle(\gamma,\beta)$ at $p=1$' +
             '\n(periodic and non-convex — slide "Energy Landscape")',
             fontsize=9, fontweight='bold')
ax.legend(fontsize=8)

# ── Row 1: measurement probability distributions for p=1,2,3 ────────────────
x_labels = [format(z, f'0{N_QUBITS}b')[::-1] for z in range(DIM)]
colours_prob = ['firebrick' if costs[z]==optimal_cut else 'steelblue'
                for z in range(DIM)]
for col, p in enumerate([1, 2, 3]):
    ax = fig.add_subplot(gs[1, col])
    probs_p = results[p]['probs']
    ax.bar(range(DIM), probs_p, color=colours_prob, edgecolor='none', alpha=0.85)
    ax.set_xticks(range(DIM))
    ax.set_xticklabels(x_labels, rotation=90, fontsize=5.5)
    ax.set_ylabel('Probability', fontsize=8)
    ax.set_title(
        f'p = {p}: ⟨H_C⟩ = {results[p]["energy"]:.3f}  '
        f'(r = {results[p]["approx_ratio"]:.3f})\n'
        f'P(optimal) = {results[p]["prob_optimal"]:.3f}  '
        f'(red = optimal bitstrings)',
        fontsize=8, fontweight='bold'
    )
    ax.set_ylim(0, max(probs_p) * 1.25)
    ax.axhline(1/DIM, color='gray', ls=':', lw=1, label='Uniform (1/16)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2, axis='y')

# ── Panel 1,3: approximation ratio vs p ─────────────────────────────────────
ax = fig.add_subplot(gs[1, 3])
ps   = list(results.keys())
rats = [results[p]['approx_ratio'] for p in ps]
ax.plot(ps, rats, 'o-', color='firebrick', lw=2, ms=8)
ax.axhline(1.0, color='k', ls='--', lw=1, label='Perfect (= 1)')
for p_, r_ in zip(ps, rats):
    ax.text(p_, r_+0.005, f'{r_:.3f}', ha='center', va='bottom', fontsize=9)
ax.set_xlabel('Circuit depth p', fontsize=10)
ax.set_ylabel(r'Approximation ratio $\langle H_C\rangle / C^*$', fontsize=9)
ax.set_title('Approximation ratio vs depth\n(slide "Performance")',
             fontsize=9, fontweight='bold')
ax.set_xticks(ps); ax.set_ylim(0.5, 1.1)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ── Row 2: analytical p=1 landscape (single-edge, 2 qubits) ─────────────────
ax = fig.add_subplot(gs[2, 0:2])
n_an = 80
bv = np.linspace(0, np.pi/2, n_an)
gv = np.linspace(0, np.pi/2, n_an)
BV, GV = np.meshgrid(bv, gv)
ANA = 0.5 * (1 + np.sin(4*BV) * np.sin(GV))   # FIXED formula
cp2 = ax.contourf(GV, BV, ANA, levels=25, cmap='plasma')
plt.colorbar(cp2, ax=ax, label=r'$\langle H_C\rangle = \frac{1}{2}(1+\sin 4\beta\sin \gamma)$')
ax.scatter(np.pi/2, np.pi/8, c='white', s=150, zorder=5, marker='*',
           label=r'Optimal: $\gamma=\pi/2$, $\beta=\pi/8$')   # FIXED γ* = π/2
ax.set_xlabel(r'$\gamma$', fontsize=11)
ax.set_ylabel(r'$\beta$', fontsize=11)
ax.set_title(
    r"Analytic $p=1$ landscape (single edge, 2 qubits)" + "\n"
    r"Corrected formula: $\langle H_C\rangle = \frac{1}{2}(1+\sin 4\beta \sin \gamma)$",
    fontsize=9, fontweight='bold'
)
ax.legend(fontsize=8)

# ── Panel 2,2-3: QAOA algorithm summary text ─────────────────────────────────
ax = fig.add_subplot(gs[2, 2:])
ax.axis('off')
summary = (
    "QAOA Algorithm  (all equations from the slides)\n\n"
    "1.  Initial state:   |+⟩^⊗n = 1/√(2^n) Σ_z |z⟩\n\n"
    "2.  For k = 1 … p:\n"
    "      Apply cost unitary:   U_C(γ_k) = e^{-iγ_k H_C}\n"
    "        H_C diagonal ⟹ U_C |z⟩ = e^{-iγ_k C(z)} |z⟩\n\n"
    "      Apply mixer unitary:  U_M(β_k) = e^{-iβ_k H_M}\n"
    "        H_M = Σ X_i  ⟹  U_M = ⊗_i R_x(2β_k)\n\n"
    "3.  Measure: ⟨H_C⟩ = Σ_z C(z) |⟨z|ψ⟩|²\n\n"
    "4.  Classical optimiser updates (γ,β)  →  repeat\n\n"
    f"Results (4-node ring, C* = {optimal_cut}):\n"
    + "\n".join(
        f"  p={p}: ⟨H_C⟩={results[p]['energy']:.3f}  "
        f"ratio={results[p]['approx_ratio']:.3f}  "
        f"P(opt)={results[p]['prob_optimal']:.3f}"
        for p in [1, 2, 3]
    )
)
ax.text(0.03, 0.97, summary, transform=ax.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
ax.set_title('Algorithm summary and results', fontsize=9, fontweight='bold')

plt.savefig('qaoa_result.png', dpi=140, bbox_inches='tight')
plt.show()
print("\nFigure saved to qaoa_result.png")

# =============================================================================
# SECTION 8 — FINAL SUMMARY
# =============================================================================
print(f"\n{'='*65}")
print(f"FINAL RESULTS SUMMARY")
print(f"{'='*65}")
print(f"\nProblem: MaxCut on the 4-node ring graph")
print(f"  Edges : {EDGES}")
print(f"  C*    = {optimal_cut}  (optimal cut)")
print(f"  Optimal solutions: {optimal_strs}")
print(f"\n  {'p':>3s}  {'⟨H_C⟩':>8s}  {'ratio':>7s}  {'P(opt)':>8s}  {'params γ':>22s}  β")
print("  " + "-"*72)
for p in [1, 2, 3]:
    r = results[p]
    gs_str = np.round(r['params'][:p], 3)
    bs_str = np.round(r['params'][p:], 3)
    print(f"  {p:>3d}  {r['energy']:>8.4f}  {r['approx_ratio']:>7.4f}  "
          f"{r['prob_optimal']:>8.4f}  {str(gs_str):>22s}  {bs_str}")

print(f"""
Key equations implemented (from slides):

  |ψ_p(γ,β)⟩ = Π_k e^{{-iβ_k H_M}} e^{{-iγ_k H_C}} |+⟩^⊗n

  H_C = Σ_{{(i,j)∈E}} (1 - Z_i Z_j)/2    [diagonal in Z-basis]
  H_M = Σ_i X_i                           [transverse-field mixer]

  U_C(γ): phases each basis state by e^{{-iγ C(z)}}  [O(2^n) cost]
  U_M(β): ⊗_i R_x(2β)  =  ⊗_i [cos β I − i sin β X]

  ⟨H_C⟩ = Σ_z C(z) |⟨z|ψ⟩|²   [weighted measurement probability]

  p=1 analytic (single edge, corrected):
    ⟨H_C⟩ = ½(1 + sin(4β) sin(γ))          [BUG FIX: was − and sin(2γ)]
    Optimal: β = π/8, γ = π/2   →  ⟨H_C⟩_max = 1.0  ✓
                                             [BUG FIX: was γ = π/4]
""")
