"""
Quantum Fourier Transform — Circuit-Level Implementation
=========================================================

Builds the QFT and IQFT exactly as they are decomposed on a quantum computer:
  • Hadamard gates  H
  • Controlled phase-rotation gates  CR_k  (= controlled-R_k)
  • SWAP gates to reverse qubit order

All gates are represented as full 2^n × 2^n unitary matrices obtained by
tensoring single-qubit and two-qubit gates with identity blocks, so every
circuit matrix can be compared directly to the analytic QFT matrix from the
original qft.py.

Structure
---------
  gate_H(n, qubit)          – n-qubit Hadamard on `qubit`
  gate_R(k)                 – single-qubit phase rotation by 2π/2^k
  gate_CR(n, ctrl, tgt, k)  – controlled-R_k in n-qubit space
  gate_SWAP(n, i, j)        – SWAP on qubits i and j
  qft_circuit(n)            – full QFT unitary built from gates
  iqft_circuit(n)           – full IQFT unitary (reverse circuit, R → R†)
  analytic_qft(n)           – closed-form matrix for comparison
  run_all_tests(n)           – unitarity + equivalence tests with plots
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Single-qubit gate matrices
# ═══════════════════════════════════════════════════════════════════════════════

def _H():
    """2×2 Hadamard matrix."""
    return np.array([[1,  1],
                     [1, -1]], dtype=complex) / np.sqrt(2)


def _R(k):
    """
    Single-qubit phase-rotation gate R_k:

        R_k = | 1    0          |
              | 0    e^{2πi/2^k}|

    In the QFT circuit qubit j receives R_k from a control qubit k steps
    away, contributing a phase of 2π/2^k to the |1⟩ component.
    """
    return np.array([[1, 0],
                     [0, np.exp(2j * np.pi / (2 ** k))]], dtype=complex)


def _Rd(k):
    """Conjugate-transpose of R_k  (used in the IQFT)."""
    return _R(k).conj().T


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Embedding single-qubit and two-qubit gates into the n-qubit space
# ═══════════════════════════════════════════════════════════════════════════════

def _embed_single(n: int, qubit: int, gate2: np.ndarray) -> np.ndarray:
    """
    Embed a 2×2 gate acting on `qubit` into the full 2^n × 2^n space via
    Kronecker products:

        I ⊗ … ⊗ gate2 ⊗ … ⊗ I   (gate2 is at position `qubit`)

    Qubit 0 is the most-significant (leftmost) qubit.
    """
    ops = [np.eye(2, dtype=complex)] * n
    ops[qubit] = gate2
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def _embed_controlled(n: int, ctrl: int, tgt: int,
                      gate2: np.ndarray) -> np.ndarray:
    """
    Embed a controlled-gate into the 2^n space.

    The full matrix is built by projecting the control qubit onto |0⟩ and |1⟩:

        U = P0_ctrl ⊗ I_tgt  +  P1_ctrl ⊗ gate2_tgt

    where P0 = |0⟩⟨0| and P1 = |1⟩⟨1|.
    """
    P0 = np.array([[1, 0], [0, 0]], dtype=complex)
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)

    # Branch 1: ctrl = |0⟩ → identity on target
    ops0 = [np.eye(2, dtype=complex)] * n
    ops0[ctrl] = P0
    branch0 = ops0[0]
    for op in ops0[1:]:
        branch0 = np.kron(branch0, op)

    # Branch 2: ctrl = |1⟩ → gate2 on target
    ops1 = [np.eye(2, dtype=complex)] * n
    ops1[ctrl] = P1
    ops1[tgt]  = gate2
    branch1 = ops1[0]
    for op in ops1[1:]:
        branch1 = np.kron(branch1, op)

    return branch0 + branch1


def _embed_swap(n: int, i: int, j: int) -> np.ndarray:
    """
    SWAP gate on qubits i and j in the n-qubit space.

    SWAP is decomposed as three CNOTs:
        SWAP(i,j) = CNOT(i,j) · CNOT(j,i) · CNOT(i,j)

    where CNOT(ctrl, tgt) is a controlled-X gate.
    """
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    cnot_ij = _embed_controlled(n, i, j, X)
    cnot_ji = _embed_controlled(n, j, i, X)
    return cnot_ij @ cnot_ji @ cnot_ij


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Public gate constructors (n-qubit space)
# ═══════════════════════════════════════════════════════════════════════════════

def gate_H(n: int, qubit: int) -> np.ndarray:
    """
    n-qubit Hadamard gate acting on `qubit`.

    H = (1/√2) |0+1⟩⟨0| + (1/√2)|0-1⟩⟨1|

    In the circuit diagram this is the first gate applied to each qubit
    in the QFT layer.
    """
    return _embed_single(n, qubit, _H())


def gate_CR(n: int, ctrl: int, tgt: int, k: int) -> np.ndarray:
    """
    Controlled phase-rotation CR_k in the n-qubit space.

    If the control qubit is |1⟩, apply R_k to the target:
        R_k = diag(1, e^{2πi/2^k})

    In the standard QFT circuit qubit `qubit` receives CR_k gates from
    all subsequent qubits `qubit+1, qubit+2, …` with k = 2, 3, …
    """
    return _embed_controlled(n, ctrl, tgt, _R(k))


def gate_CRd(n: int, ctrl: int, tgt: int, k: int) -> np.ndarray:
    """Controlled R_k† (used in the IQFT)."""
    return _embed_controlled(n, ctrl, tgt, _Rd(k))


def gate_SWAP(n: int, i: int, j: int) -> np.ndarray:
    """SWAP gate on qubits i and j in the n-qubit space."""
    return _embed_swap(n, i, j)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  QFT circuit
# ═══════════════════════════════════════════════════════════════════════════════

def qft_circuit(n: int) -> np.ndarray:
    """
    Build the n-qubit QFT unitary from Hadamard, controlled-rotation, and
    SWAP gates — exactly as implemented on a real quantum computer.

    Circuit structure (qubit 0 = most significant):
    ─────────────────────────────────────────────────
    For each qubit j = 0, 1, …, n-1:
      1. Apply H on qubit j
      2. For each qubit m = j+1, j+2, …, n-1:
            Apply CR_{m-j+1} with control=m, target=j
            (rotation angle 2π / 2^{m-j+1})

    Then reverse the qubit order with n//2 SWAP gates.

    Gate count:  n Hadamards  +  n(n-1)/2 CR gates  +  ⌊n/2⌋ SWAPs
    ─────────────────────────────────────────────────
    Returns the 2^n × 2^n unitary matrix U_QFT.
    """
    dim = 2 ** n
    U   = np.eye(dim, dtype=complex)

    for j in range(n):
        # Hadamard on qubit j
        U = gate_H(n, j) @ U
        # Controlled rotations from all later qubits
        for m in range(j + 1, n):
            k = m - j + 1          # rotation order: R_2, R_3, …
            U = gate_CR(n, m, j, k) @ U

    # Bit-reversal: swap qubit 0↔n-1, 1↔n-2, …
    for i in range(n // 2):
        U = gate_SWAP(n, i, n - 1 - i) @ U

    return U


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  IQFT circuit
# ═══════════════════════════════════════════════════════════════════════════════

def iqft_circuit(n: int) -> np.ndarray:
    """
    Build the n-qubit inverse QFT from the reversed and conjugated circuit.

    The IQFT is U_QFT†.  On a quantum computer this is implemented by:
      1. Undo the SWAP layer (SWAPs are self-inverse, same gates)
      2. For each qubit j = n-1, n-2, …, 0  (reverse order):
            For each qubit m = n-1, …, j+1  (reverse order):
               Apply CR_k†  with control=m, target=j
            Apply H on qubit j

    Gate count: same as QFT (all gates are inverted, SWAPs unchanged).
    ─────────────────────────────────────────────────
    Returns the 2^n × 2^n unitary matrix U_IQFT = U_QFT†.
    """
    dim = 2 ** n
    U   = np.eye(dim, dtype=complex)

    # Step 1: undo the SWAP layer first (SWAPs are self-inverse)
    for i in range(n // 2):
        U = gate_SWAP(n, i, n - 1 - i) @ U

    # Step 2: reverse the rotation + Hadamard layers
    for j in range(n - 1, -1, -1):
        # Inverse controlled rotations (reverse order)
        for m in range(n - 1, j, -1):
            k = m - j + 1
            U = gate_CRd(n, m, j, k) @ U
        # Hadamard (self-inverse)
        U = gate_H(n, j) @ U

    return U


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Analytic reference matrix (from original qft.py)
# ═══════════════════════════════════════════════════════════════════════════════

def analytic_qft(n: int) -> np.ndarray:
    """
    Closed-form QFT matrix:  F[j,k] = ω^{jk} / √(2^n),  ω = e^{2πi/2^n}.

    Used as the ground-truth reference to verify the circuit construction.
    """
    dim   = 2 ** n
    omega = np.exp(2j * np.pi / dim)
    idx   = np.arange(dim)
    return np.power(omega, np.outer(idx, idx)) / np.sqrt(dim)


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Unitarity and equivalence tests
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_tests(n: int, n_random: int = 300,
                  seed: int = 0, savename: str = "qft_circuit_tests.png"):
    """
    Run a full test suite and produce diagnostic plots.

    Tests
    -----
    1.  U_circ · U_circ†  ≈  I           (QFT circuit left-unitarity)
    2.  U_circ† · U_circ  ≈  I           (QFT circuit right-unitarity)
    3.  U_iqft · U_iqft†  ≈  I           (IQFT circuit unitarity)
    4.  U_iqft · U_circ   ≈  I           (IQFT is inverse of QFT)
    5.  |U_circ - U_analytic|  ≈  0      (circuit ≡ analytic formula)
    6.  Norm preservation: ‖U|ψ⟩‖ = 1    for random states
    7.  Round-trip fidelity:  |⟨ψ|IQFT·QFT|ψ⟩|² ≈ 1
    8.  Eigenvalues of U_circ on unit circle
    """
    print(f"\n{'='*62}")
    print(f"  Gate-Circuit QFT/IQFT Tests  —  n={n} qubits  (dim={2**n})")
    print(f"{'='*62}")

    dim   = 2 ** n
    I     = np.eye(dim, dtype=complex)
    rng   = np.random.default_rng(seed)

    print(f"  Building QFT circuit …", end='', flush=True)
    U_qft  = qft_circuit(n)
    print(f"  done.")
    print(f"  Building IQFT circuit …", end='', flush=True)
    U_iqft = iqft_circuit(n)
    print(f"  done.")
    U_ref  = analytic_qft(n)

    # ── Compute all errors ─────────────────────────────────────────────
    err_UUd   = np.abs(U_qft @ U_qft.conj().T - I)
    err_UdU   = np.abs(U_qft.conj().T @ U_qft - I)
    err_iqft  = np.abs(U_iqft @ U_iqft.conj().T - I)
    err_inv   = np.abs(U_iqft @ U_qft - I)
    err_ref   = np.abs(U_qft - U_ref)

    # ── Norm preservation ──────────────────────────────────────────────
    norm_errors = []
    for _ in range(n_random):
        v  = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        v /= np.linalg.norm(v)
        norm_errors.append(abs(np.linalg.norm(U_qft @ v) - 1.0))
    norm_errors = np.array(norm_errors)

    # ── Round-trip fidelities ──────────────────────────────────────────
    fidelities = []
    for _ in range(n_random):
        v   = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        v  /= np.linalg.norm(v)
        vrt = U_iqft @ (U_qft @ v)
        fidelities.append(abs(np.vdot(v, vrt)) ** 2)
    fidelities = np.array(fidelities)

    # ── Eigenvalues ────────────────────────────────────────────────────
    eigvals   = np.linalg.eigvals(U_qft)
    ev_dev    = np.max(np.abs(np.abs(eigvals) - 1.0))

    # ── Singular values ────────────────────────────────────────────────
    svs    = np.linalg.svd(U_qft, compute_uv=False)
    sv_dev = np.max(np.abs(svs - 1.0))

    # ── Print summary ──────────────────────────────────────────────────
    tol = 1e-10
    rows = [
        ("U·U† = I  (max error)",            err_UUd.max(),    tol),
        ("U†·U = I  (max error)",             err_UdU.max(),    tol),
        ("IQFT·IQFT† = I  (max error)",       err_iqft.max(),   tol),
        ("IQFT·QFT = I  (max error)",          err_inv.max(),    tol),
        ("Circuit vs analytic  (max error)",   err_ref.max(),    tol),
        ("Norm preservation  (max error)",     norm_errors.max(),tol),
        ("Round-trip fidelity  (min)",         fidelities.min(), None),
        ("Singular values ≈ 1  (max dev)",     sv_dev,           tol),
        ("|eigenvalue| ≈ 1  (max dev)",        ev_dev,           tol),
    ]
    for name, val, threshold in rows:
        if threshold is None:
            ok = abs(val - 1.0) < 1e-10
        else:
            ok = val < threshold
        print(f"  {'✓' if ok else '✗'}  {name:<44s}  {val:.3e}")
    print(f"{'='*62}\n")

    # ── Plots ──────────────────────────────────────────────────────────
    _plot_tests(n, dim, U_qft, U_iqft,
                err_UUd, err_UdU, err_iqft, err_inv, err_ref,
                norm_errors, fidelities, eigvals, svs,
                savename=savename)


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Plotting
# ═══════════════════════════════════════════════════════════════════════════════

# Colour palette
BG     = "#0d0f1a"
PANEL  = "#131629"
ACCENT = "#4fc3f7"
WARM   = "#ff7043"
GREEN  = "#69f0ae"
PURPLE = "#ce93d8"
GRID   = "#1e2340"
TEXT   = "#e8eaf6"


def _plot_tests(n, dim, U_qft, U_iqft,
                err_UUd, err_UdU, err_iqft, err_inv, err_ref,
                norm_errors, fidelities, eigvals, svs,
                savename):

    plt.rcParams.update({
        'figure.facecolor':  BG,   'axes.facecolor':  PANEL,
        'axes.edgecolor':    GRID, 'axes.labelcolor': TEXT,
        'xtick.color':       TEXT, 'ytick.color':     TEXT,
        'text.color':        TEXT, 'grid.color':      GRID,
        'grid.linewidth':    0.6,  'legend.facecolor':PANEL,
        'legend.edgecolor':  GRID,
    })

    fig = plt.figure(figsize=(22, 16), facecolor=BG)
    fig.suptitle(
        f"Gate-Circuit QFT / IQFT  —  {n} qubits  (dim = {dim})",
        fontsize=17, fontweight='bold', color=TEXT, y=0.98)

    gs = gridspec.GridSpec(3, 4, figure=fig,
                           hspace=0.52, wspace=0.42,
                           left=0.05, right=0.97, top=0.93, bottom=0.05)

    def ax(row, col, **kw):
        a = fig.add_subplot(gs[row, col], **kw)
        a.grid(True, alpha=0.35)
        return a

    # ── 1: U·U† error heatmap ─────────────────────────────────────────
    a1 = ax(0, 0)
    im1 = a1.imshow(err_UUd, cmap='inferno', vmin=0,
                    vmax=max(err_UUd.max(), 1e-16),
                    interpolation='nearest', aspect='auto')
    plt.colorbar(im1, ax=a1, label='|error|')
    a1.set_title(f"U·U† − I\n(max {err_UUd.max():.2e})",
                 color=TEXT, fontweight='bold')
    a1.set_xlabel("col j"); a1.set_ylabel("row i")

    # ── 2: U†·U error heatmap ─────────────────────────────────────────
    a2 = ax(0, 1)
    im2 = a2.imshow(err_UdU, cmap='inferno', vmin=0,
                    vmax=max(err_UdU.max(), 1e-16),
                    interpolation='nearest', aspect='auto')
    plt.colorbar(im2, ax=a2, label='|error|')
    a2.set_title(f"U†·U − I\n(max {err_UdU.max():.2e})",
                 color=TEXT, fontweight='bold')
    a2.set_xlabel("col j"); a2.set_ylabel("row i")

    # ── 3: IQFT·QFT = I heatmap ───────────────────────────────────────
    a3 = ax(0, 2)
    im3 = a3.imshow(err_inv, cmap='plasma', vmin=0,
                    vmax=max(err_inv.max(), 1e-16),
                    interpolation='nearest', aspect='auto')
    plt.colorbar(im3, ax=a3, label='|error|')
    a3.set_title(f"IQFT·QFT − I\n(max {err_inv.max():.2e})",
                 color=TEXT, fontweight='bold')
    a3.set_xlabel("col j"); a3.set_ylabel("row i")

    # ── 4: Circuit vs analytic ────────────────────────────────────────
    a4 = ax(0, 3)
    im4 = a4.imshow(err_ref, cmap='viridis', vmin=0,
                    vmax=max(err_ref.max(), 1e-16),
                    interpolation='nearest', aspect='auto')
    plt.colorbar(im4, ax=a4, label='|error|')
    a4.set_title(f"Circuit − Analytic\n(max {err_ref.max():.2e})",
                 color=TEXT, fontweight='bold')
    a4.set_xlabel("col j"); a4.set_ylabel("row i")

    # ── 5: Norm preservation histogram ───────────────────────────────
    a5 = ax(1, 0)
    a5.hist(norm_errors, bins=40, color=ACCENT, edgecolor=BG, alpha=0.85)
    a5.axvline(norm_errors.mean(), color=WARM, lw=2,
               label=f"mean={norm_errors.mean():.2e}")
    a5.axvline(norm_errors.max(), color=GREEN, lw=2, ls='--',
               label=f"max={norm_errors.max():.2e}")
    a5.set_title("Norm preservation\n(random states)",
                 color=TEXT, fontweight='bold')
    a5.set_xlabel("|‖U|ψ⟩‖ − 1|"); a5.set_ylabel("count")
    a5.legend(fontsize=9)

    # ── 6: Round-trip fidelity histogram ─────────────────────────────
    a6 = ax(1, 1)
    infid = 1 - fidelities
    a6.hist(infid, bins=40, color=GREEN, edgecolor=BG, alpha=0.85)
    a6.axvline(infid.mean(), color=WARM, lw=2,
               label=f"mean={infid.mean():.2e}")
    a6.set_title("Round-trip infidelity\n1 − |⟨ψ|IQFT·QFT|ψ⟩|²",
                 color=TEXT, fontweight='bold')
    a6.set_xlabel("1 − fidelity"); a6.set_ylabel("count")
    a6.legend(fontsize=9)

    # ── 7: Singular values ────────────────────────────────────────────
    a7 = ax(1, 2)
    sv_dev = np.max(np.abs(svs - 1.0))
    a7.scatter(range(len(svs)), svs, color=ACCENT, s=14, alpha=0.75, zorder=3)
    a7.axhline(1.0, color=GREEN, lw=1.5, ls='--', label='σ = 1  (ideal)')
    a7.fill_between(range(len(svs)), 1-1e-10, 1+1e-10,
                    color=GREEN, alpha=0.12)
    a7.set_title(f"Singular values\n(max dev: {sv_dev:.2e})",
                 color=TEXT, fontweight='bold')
    a7.set_xlabel("index"); a7.set_ylabel("σ")
    a7.legend(fontsize=9)

    # ── 8: Eigenvalues on unit circle ────────────────────────────────
    a8 = ax(1, 3, aspect='equal')
    th = np.linspace(0, 2*np.pi, 400)
    a8.plot(np.cos(th), np.sin(th), color=GRID, lw=1.5, zorder=1)
    sc = a8.scatter(eigvals.real, eigvals.imag,
                    c=np.angle(eigvals), cmap='hsv',
                    s=28, alpha=0.85, zorder=3,
                    vmin=-np.pi, vmax=np.pi)
    plt.colorbar(sc, ax=a8, label='arg(λ) [rad]')
    a8.axhline(0, color=GRID, lw=0.8); a8.axvline(0, color=GRID, lw=0.8)
    ev_dev = np.max(np.abs(np.abs(eigvals) - 1.0))
    a8.set_title(f"Eigenvalues on unit circle\n(max |λ|−1: {ev_dev:.2e})",
                 color=TEXT, fontweight='bold')
    a8.set_xlabel("Re(λ)"); a8.set_ylabel("Im(λ)")

    # ── 9: QFT circuit matrix — |U| ───────────────────────────────────
    a9 = ax(2, 0)
    im9 = a9.imshow(np.abs(U_qft), cmap='magma', vmin=0,
                    interpolation='nearest', aspect='auto')
    plt.colorbar(im9, ax=a9, label='|U_{jk}|')
    a9.set_title("|U_QFT|  (circuit)\nAll entries = 1/√dim",
                 color=TEXT, fontweight='bold')
    a9.set_xlabel("col k"); a9.set_ylabel("row j")

    # ── 10: IQFT matrix — |U†| ────────────────────────────────────────
    a10 = ax(2, 1)
    im10 = a10.imshow(np.abs(U_iqft), cmap='magma', vmin=0,
                      interpolation='nearest', aspect='auto')
    plt.colorbar(im10, ax=a10, label='|U_{jk}|')
    a10.set_title("|U_IQFT|  (circuit)\nAll entries = 1/√dim",
                  color=TEXT, fontweight='bold')
    a10.set_xlabel("col k"); a10.set_ylabel("row j")

    # ── 11: Phase of QFT matrix ───────────────────────────────────────
    a11 = ax(2, 2)
    im11 = a11.imshow(np.angle(U_qft), cmap='hsv',
                      vmin=-np.pi, vmax=np.pi,
                      interpolation='nearest', aspect='auto')
    plt.colorbar(im11, ax=a11, label='arg(U_{jk}) [rad]')
    a11.set_title("arg(U_QFT)  — phase pattern\n(circuit)",
                  color=TEXT, fontweight='bold')
    a11.set_xlabel("col k"); a11.set_ylabel("row j")

    # ── 12: Phase difference circuit vs analytic ──────────────────────
    a12 = ax(2, 3)
    phase_diff = np.abs(np.angle(U_qft) - np.angle(analytic_qft(n)))
    # Wrap to [-π, π]
    phase_diff = np.where(phase_diff > np.pi, 2*np.pi - phase_diff, phase_diff)
    im12 = a12.imshow(phase_diff, cmap='inferno', vmin=0,
                      interpolation='nearest', aspect='auto')
    plt.colorbar(im12, ax=a12, label='|Δarg| [rad]')
    a12.set_title(f"Phase diff: circuit vs analytic\n(max {phase_diff.max():.2e})",
                  color=TEXT, fontweight='bold')
    a12.set_xlabel("col k"); a12.set_ylabel("row j")

    plt.savefig(savename, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)
    print(f"Saved {savename}")


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  Circuit diagram printer
# ═══════════════════════════════════════════════════════════════════════════════

def print_circuit(n: int):
    """Print a text schematic of the QFT circuit for n qubits."""
    print(f"\n  QFT Circuit  —  {n} qubits")
    print("  " + "─" * (8 + 9 * n))
    for j in range(n):
        row = f"  q{j} ─── H"
        for m in range(j + 1, n):
            k = m - j + 1
            row += f" ─ CR{k}(q{m})"
        for _ in range(n - 1 - j):
            row += "           " [:(11 if j == 0 else 11)]
        row += " ─── (swap)"
        print(row)
    print()
    print("  Bit-reversal SWAP layer:")
    for i in range(n // 2):
        print(f"    SWAP(q{i}, q{n-1-i})")
    print("  " + "─" * (8 + 9 * n))
    print(f"  Gate count:  {n} H  +  {n*(n-1)//2} CR  +  {n//2} SWAP\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Demo
# ═══════════════════════════════════════════════════════════════════════════════

def demo(n: int = 3):
    print(f"\n{'═'*62}")
    print(f"  Gate-Circuit QFT Demo  —  n={n}  (dim={2**n})")
    print(f"{'═'*62}")

    print_circuit(n)

    U_qft  = qft_circuit(n)
    U_iqft = iqft_circuit(n)
    U_ref  = analytic_qft(n)

    # ── Verify circuit == analytic ─────────────────────────────────────
    max_diff = np.max(np.abs(U_qft - U_ref))
    print(f"  Max |circuit − analytic|        = {max_diff:.3e}  (expect ~0)")

    # ── Verify IQFT = QFT† ─────────────────────────────────────────────
    max_inv = np.max(np.abs(U_iqft - U_qft.conj().T))
    print(f"  Max |IQFT − QFT†|               = {max_inv:.3e}  (expect ~0)")

    # ── Apply to a specific state ──────────────────────────────────────
    state = np.zeros(2**n, dtype=complex)
    state[5 % (2**n)] = 1.0               # |5 mod 2^n⟩
    label = 5 % (2**n)

    out     = U_qft  @ state
    rec     = U_iqft @ out
    fid     = abs(np.vdot(state, rec))**2
    print(f"\n  Initial state: |{label}⟩")
    print(f"  After QFT  (amplitudes rounded):")
    print(f"    {np.round(out, 4)}")
    print(f"  Round-trip fidelity |⟨{label}|IQFT·QFT|{label}⟩|² = {fid:.15f}")

    run_all_tests(n, savename=f"qft_circuit_tests_n{n}.png")


if __name__ == "__main__":
    # Run demo + tests for 3 and 4 qubits
    demo(n=3)
    demo(n=4)
