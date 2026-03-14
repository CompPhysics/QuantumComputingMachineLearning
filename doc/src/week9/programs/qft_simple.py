"""
Quantum Fourier Transform (QFT) — Simple Functional Implementation
==================================================================
Functions:
  - qft_matrix(n)         : build the n-qubit QFT matrix
  - iqft_matrix(n)        : build the n-qubit inverse QFT matrix
  - apply_qft(state)      : apply QFT to a state vector
  - apply_iqft(state)     : apply IQFT to a state vector
  - basis_state(idx, n)   : make a computational basis vector |idx⟩
  - random_state(n)       : make a random normalised state vector
  - run_unitarity_tests(n): run all tests and print + plot results
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Matrix construction
# ─────────────────────────────────────────────────────────────────────────────

def qft_matrix(n):
    """
    Return the 2^n × 2^n QFT matrix.
    F[j,k] = exp(+2πi·jk / 2^n) / sqrt(2^n)
    """
    dim   = 2 ** n
    omega = np.exp(2j * np.pi / dim)
    idx   = np.arange(dim)
    return np.power(omega, np.outer(idx, idx)) / np.sqrt(dim)


def iqft_matrix(n):
    """
    Return the 2^n × 2^n inverse QFT matrix.
    F†[j,k] = exp(−2πi·jk / 2^n) / sqrt(2^n)  =  conj(F[j,k])
    """
    return qft_matrix(n).conj()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Applying the transforms
# ─────────────────────────────────────────────────────────────────────────────

def apply_qft(state):
    """Apply the QFT to a state vector. Returns transformed vector."""
    n = int(round(np.log2(len(state))))
    return qft_matrix(n) @ np.asarray(state, dtype=complex)


def apply_iqft(state):
    """Apply the inverse QFT to a state vector. Returns transformed vector."""
    n = int(round(np.log2(len(state))))
    return iqft_matrix(n) @ np.asarray(state, dtype=complex)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Convenience state builders
# ─────────────────────────────────────────────────────────────────────────────

def basis_state(idx, n):
    """Return the computational basis vector |idx⟩ for an n-qubit system."""
    v = np.zeros(2 ** n, dtype=complex)
    v[idx] = 1.0
    return v


def random_state(n, seed=None):
    """Return a Haar-random normalised state vector for an n-qubit system."""
    rng = np.random.default_rng(seed)
    v   = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
    return v / np.linalg.norm(v)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Unitarity tests
# ─────────────────────────────────────────────────────────────────────────────

def run_unitarity_tests(n, n_random=500, seed=0, plot=True,
                         savename_tests="unitarity_tests_simple.png",
                         savename_matrix="qft_matrix_simple.png"):
    """
    Run a full suite of unitarity tests on the n-qubit QFT, print a
    pass/fail summary, and optionally save two diagnostic figures.
    """
    dim  = 2 ** n
    F    = qft_matrix(n)
    Fd   = iqft_matrix(n)
    I    = np.eye(dim)
    rng  = np.random.default_rng(seed)

    # ── 1 & 2: product matrices ────────────────────────────────────────────
    FFd_err   = np.abs(F @ Fd - I)
    FdF_err   = np.abs(Fd @ F - I)

    # ── 3 & 4: column / row orthonormality ────────────────────────────────
    col_err   = np.abs(F.conj().T @ F - I)
    row_err   = np.abs(F @ F.conj().T - I)

    # ── 5: norm preservation over random states ────────────────────────────
    def _rand_norm():
        v = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        v /= np.linalg.norm(v)
        return v

    norm_errors = np.array([
        abs(np.linalg.norm(apply_qft(_rand_norm())) - 1.0)
        for _ in range(n_random)
    ])

    # ── 6: round-trip fidelity ─────────────────────────────────────────────
    fidelities = []
    for _ in range(n_random):
        v    = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        v   /= np.linalg.norm(v)
        vrt  = apply_iqft(apply_qft(v))
        fidelities.append(abs(np.vdot(v, vrt)) ** 2)
    fidelities = np.array(fidelities)

    # ── 7: singular values ─────────────────────────────────────────────────
    svs       = np.linalg.svd(F, compute_uv=False)
    sv_dev    = np.max(np.abs(svs - 1.0))

    # ── 8: eigenvalues ─────────────────────────────────────────────────────
    eigvals   = np.linalg.eigvals(F)
    ev_dev    = np.max(np.abs(np.abs(eigvals) - 1.0))

    # ── Print summary ──────────────────────────────────────────────────────
    tol = 1e-10
    print("=" * 60)
    print(f"  Unitarity Tests — {n}-qubit QFT  (dim = {dim})")
    print("=" * 60)
    rows = [
        ("F·F† = I  (max error)",          FFd_err.max(),     tol,  True),
        ("F†·F = I  (max error)",           FdF_err.max(),     tol,  True),
        ("Column orthonormality (max err)", col_err.max(),     tol,  True),
        ("Row orthonormality (max err)",    row_err.max(),     tol,  True),
        ("Norm preservation (max err)",     norm_errors.max(), tol,  True),
        ("Round-trip fidelity (min)",       fidelities.min(),  None, False),
        ("Singular values ≈ 1 (max dev)",   sv_dev,            tol,  True),
        ("|eigenvalue| ≈ 1 (max dev)",      ev_dev,            tol,  True),
    ]
    for name, val, threshold, low_is_good in rows:
        if threshold is None:
            ok = abs(val - 1.0) < 1e-10
        else:
            ok = val < threshold
        print(f"  {'✓' if ok else '✗'}  {name:<42s}  {val:.3e}")
    print("=" * 60)

    if not plot:
        return

    # ── Styling ────────────────────────────────────────────────────────────
    BG, PANEL, ACCENT = "#0d0f1a", "#131629", "#4fc3f7"
    WARM, GREEN, GRID  = "#ff7043", "#69f0ae", "#1e2340"
    TEXT = "#e8eaf6"

    plt.rcParams.update({
        'figure.facecolor': BG,   'axes.facecolor':  PANEL,
        'axes.edgecolor':   GRID, 'axes.labelcolor': TEXT,
        'xtick.color':      TEXT, 'ytick.color':     TEXT,
        'text.color':       TEXT, 'grid.color':      GRID,
        'grid.linewidth':   0.6,
    })

    # ════════════════════════════════════════════════════════════════════════
    # Figure 1: nine-panel diagnostic grid
    # ════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(20, 15), facecolor=BG)
    fig.suptitle(f"Unitarity Tests — {n}-qubit QFT  (dim = {dim})",
                 fontsize=17, fontweight='bold', color=TEXT, y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.52, wspace=0.38,
                           left=0.06, right=0.97, top=0.93, bottom=0.06)

    def make_ax(row, col, **kw):
        a = fig.add_subplot(gs[row, col], **kw)
        a.grid(True, alpha=0.35)
        return a

    # 1 — F·F† heatmap
    a1 = make_ax(0, 0)
    im1 = a1.imshow(FFd_err, cmap='inferno', vmin=0,
                    vmax=max(FFd_err.max(), 1e-16),
                    interpolation='nearest', aspect='auto')
    plt.colorbar(im1, ax=a1, label='|error|')
    a1.set_title(f"F·F† − I  (max {FFd_err.max():.2e})",
                 color=TEXT, fontweight='bold')
    a1.set_xlabel("column j"); a1.set_ylabel("row i")

    # 2 — F†·F heatmap
    a2 = make_ax(0, 1)
    im2 = a2.imshow(FdF_err, cmap='inferno', vmin=0,
                    vmax=max(FdF_err.max(), 1e-16),
                    interpolation='nearest', aspect='auto')
    plt.colorbar(im2, ax=a2, label='|error|')
    a2.set_title(f"F†·F − I  (max {FdF_err.max():.2e})",
                 color=TEXT, fontweight='bold')
    a2.set_xlabel("column j"); a2.set_ylabel("row i")

    # 3 — column orthonormality heatmap
    a3 = make_ax(0, 2)
    im3 = a3.imshow(col_err, cmap='plasma', vmin=0,
                    vmax=max(col_err.max(), 1e-16),
                    interpolation='nearest', aspect='auto')
    plt.colorbar(im3, ax=a3, label='|error|')
    a3.set_title("Column orthonormality error",
                 color=TEXT, fontweight='bold')
    a3.set_xlabel("column j"); a3.set_ylabel("column i")

    # 4 — norm preservation histogram
    a4 = make_ax(1, 0)
    a4.hist(norm_errors, bins=40, color=ACCENT, edgecolor=BG, alpha=0.85)
    a4.axvline(norm_errors.mean(), color=WARM, lw=2,
               label=f"mean={norm_errors.mean():.2e}")
    a4.axvline(norm_errors.max(), color=GREEN, lw=2, ls='--',
               label=f"max={norm_errors.max():.2e}")
    a4.set_title(f"Norm preservation error\n({n_random} random states)",
                 color=TEXT, fontweight='bold')
    a4.set_xlabel("|‖F|ψ⟩‖ − 1|"); a4.set_ylabel("count")
    a4.legend(fontsize=9)

    # 5 — round-trip infidelity histogram
    a5 = make_ax(1, 1)
    infid = 1 - fidelities
    a5.hist(infid, bins=40, color=GREEN, edgecolor=BG, alpha=0.85)
    a5.axvline(infid.mean(), color=WARM, lw=2,
               label=f"mean={infid.mean():.2e}")
    a5.set_title("Round-trip infidelity  1−F\n(IQFT∘QFT, random states)",
                 color=TEXT, fontweight='bold')
    a5.set_xlabel("1 − |⟨ψ|IQFT(QFT|ψ⟩)|²"); a5.set_ylabel("count")
    a5.legend(fontsize=9)

    # 6 — singular values scatter
    a6 = make_ax(1, 2)
    a6.scatter(range(len(svs)), svs, color=ACCENT, s=14, alpha=0.75, zorder=3)
    a6.axhline(1.0, color=GREEN, lw=1.5, ls='--', label='σ = 1  (ideal)')
    a6.fill_between(range(len(svs)), 1-1e-10, 1+1e-10,
                    color=GREEN, alpha=0.12)
    a6.set_title(f"Singular values of F\n(max dev: {sv_dev:.2e})",
                 color=TEXT, fontweight='bold')
    a6.set_xlabel("index"); a6.set_ylabel("σ")
    a6.legend(fontsize=9)

    # 7 — eigenvalues on unit circle
    a7 = make_ax(2, 0, aspect='equal')
    th = np.linspace(0, 2*np.pi, 400)
    a7.plot(np.cos(th), np.sin(th), color=GRID, lw=1.5, zorder=1)
    sc = a7.scatter(eigvals.real, eigvals.imag,
                    c=np.angle(eigvals), cmap='hsv',
                    s=28, alpha=0.85, zorder=3,
                    vmin=-np.pi, vmax=np.pi)
    plt.colorbar(sc, ax=a7, label='arg(λ)  [rad]')
    a7.axhline(0, color=GRID, lw=0.8); a7.axvline(0, color=GRID, lw=0.8)
    a7.set_title(f"Eigenvalues on unit circle\n(max |λ|−1: {ev_dev:.2e})",
                 color=TEXT, fontweight='bold')
    a7.set_xlabel("Re(λ)"); a7.set_ylabel("Im(λ)")

    # 8 — QFT of basis states
    a8 = make_ax(2, 1)
    for idx in range(min(4, dim)):
        v = basis_state(idx, n)
        a8.plot(np.abs(apply_qft(v))**2, lw=1.5, alpha=0.8, label=f"|{idx}⟩")
    a8.set_title("QFT probability distribution\n(basis states |0⟩…|3⟩)",
                 color=TEXT, fontweight='bold')
    a8.set_xlabel("output basis state"); a8.set_ylabel("probability")
    a8.legend(fontsize=9)

    # 9 — row orthonormality heatmap
    a9 = make_ax(2, 2)
    im9 = a9.imshow(row_err, cmap='magma', vmin=0,
                    vmax=max(row_err.max(), 1e-16),
                    interpolation='nearest', aspect='auto')
    plt.colorbar(im9, ax=a9, label='|error|')
    a9.set_title("Row orthonormality error",
                 color=TEXT, fontweight='bold')
    a9.set_xlabel("row j"); a9.set_ylabel("row i")

    plt.savefig(savename_tests, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Saved {savename_tests}")

    # ════════════════════════════════════════════════════════════════════════
    # Figure 2: QFT matrix visualisation (Re, Im, phase)
    # ════════════════════════════════════════════════════════════════════════
    fig2, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
    for ax_i, (data, cmap, title) in zip(axes, [
        (F.real,      'RdBu_r', 'Re(F_{jk})'),
        (F.imag,      'RdBu_r', 'Im(F_{jk})'),
        (np.angle(F), 'hsv',    'arg(F_{jk})  [rad]'),
    ]):
        vabs = max(np.abs(data).max(), 1e-16)
        im = ax_i.imshow(data, cmap=cmap, vmin=-vabs, vmax=vabs,
                         interpolation='nearest', aspect='auto')
        plt.colorbar(im, ax=ax_i)
        ax_i.set_title(title, color=TEXT, fontweight='bold', fontsize=13)
        ax_i.set_xlabel("column k"); ax_i.set_ylabel("row j")

    fig2.suptitle(f"QFT Matrix — {n} qubits  (dim = {dim})",
                  fontsize=15, fontweight='bold', color=TEXT, y=1.01)
    plt.tight_layout()
    plt.savefig(savename_matrix, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Saved {savename_matrix}")

    plt.rcParams.update(plt.rcParamsDefault)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    n = 4
    dim = 2 ** n

    print("=" * 60)
    print(f"  QFT / IQFT Demo  —  {n} qubits  (dim = {dim})")
    print("=" * 60)

    # Basis state |5⟩
    s5  = basis_state(5, n)
    fs5 = apply_qft(s5)
    rs5 = apply_iqft(fs5)
    fid = abs(np.vdot(s5, rs5)) ** 2
    print(f"\n|5⟩ round-trip fidelity : {fid:.15f}")
    print(f"QFT(|5⟩) amplitudes     : {np.round(fs5, 4)}")

    # Random state round-trip
    rnd  = random_state(n, seed=42)
    fid2 = abs(np.vdot(rnd, apply_iqft(apply_qft(rnd)))) ** 2
    print(f"\nRandom state round-trip : {fid2:.15f}")

    # Confirm IQFT = QFT†
    diff = np.max(np.abs(iqft_matrix(n) - qft_matrix(n).conj().T))
    print(f"‖IQFT − QFT†‖_max       : {diff:.3e}  (should be ~0)")

    # Full unitarity test suite + plots
    print()
    run_unitarity_tests(n, n_random=500, seed=0)
