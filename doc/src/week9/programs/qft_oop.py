"""
Quantum Fourier Transform (QFT) — Object-Oriented Implementation
================================================================
Provides:
  - QuantumState    : wrapper around a complex state vector
  - QFTOperator     : builds and applies the QFT matrix
  - IQFTOperator    : builds and applies the inverse QFT matrix
  - UnitarityTester : suite of unitarity diagnostics with matplotlib plots
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.cm as cm


# ─────────────────────────────────────────────────────────────────────────────
# 1.  QuantumState
# ─────────────────────────────────────────────────────────────────────────────

class QuantumState:
    """
    Represents a normalised (or un-normalised) n-qubit quantum state vector.

    Parameters
    ----------
    amplitudes : array-like of complex
        State vector of length 2^n.  Need not be normalised.
    label : str, optional
        Human-readable name for plots / printing.
    """

    def __init__(self, amplitudes, label: str = ""):
        self._vec = np.asarray(amplitudes, dtype=complex)
        n = np.log2(len(self._vec))
        if not n.is_integer() or n < 1:
            raise ValueError("State vector length must be a power of 2 (≥ 2).")
        self._n = int(n)
        self.label = label

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n_qubits(self) -> int:
        return self._n

    @property
    def dim(self) -> int:
        return len(self._vec)

    @property
    def amplitudes(self) -> np.ndarray:
        return self._vec.copy()

    @property
    def probabilities(self) -> np.ndarray:
        return np.abs(self._vec) ** 2

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self._vec))

    def is_normalised(self, tol: float = 1e-10) -> bool:
        return abs(self.norm - 1.0) < tol

    def normalise(self) -> "QuantumState":
        """Return a new normalised QuantumState."""
        n = self.norm
        if n < 1e-15:
            raise ValueError("Cannot normalise the zero vector.")
        return QuantumState(self._vec / n, label=self.label)

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def computational_basis(cls, index: int, n_qubits: int,
                             label: str = "") -> "QuantumState":
        """
        Create the computational-basis state |index⟩ for an n-qubit system.
        """
        dim = 2 ** n_qubits
        if not (0 <= index < dim):
            raise ValueError(f"index must be in [0, {dim-1}].")
        vec = np.zeros(dim, dtype=complex)
        vec[index] = 1.0
        lbl = label or f"|{index}⟩"
        return cls(vec, label=lbl)

    @classmethod
    def uniform_superposition(cls, n_qubits: int) -> "QuantumState":
        """Create the equal-superposition state (H^⊗n |0⟩)."""
        dim = 2 ** n_qubits
        vec = np.ones(dim, dtype=complex) / np.sqrt(dim)
        return cls(vec, label="|+…+⟩")

    @classmethod
    def random(cls, n_qubits: int, seed: int = None) -> "QuantumState":
        """Create a Haar-random normalised state."""
        rng = np.random.default_rng(seed)
        dim = 2 ** n_qubits
        real = rng.standard_normal(dim)
        imag = rng.standard_normal(dim)
        vec  = real + 1j * imag
        vec /= np.linalg.norm(vec)
        return cls(vec, label="random")

    # ── Dunder helpers ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (f"QuantumState(n_qubits={self._n}, "
                f"dim={self.dim}, label='{self.label}', "
                f"norm={self.norm:.6f})")

    def __len__(self) -> int:
        return self.dim


# ─────────────────────────────────────────────────────────────────────────────
# 2.  _FourierOperatorBase  (shared machinery)
# ─────────────────────────────────────────────────────────────────────────────

class _FourierOperatorBase:
    """Internal base class — do not instantiate directly."""

    _sign: int   # subclasses set +1 (QFT) or -1 (IQFT)

    def __init__(self, n_qubits: int):
        if n_qubits < 1:
            raise ValueError("n_qubits must be ≥ 1.")
        self._n   = n_qubits
        self._dim = 2 ** n_qubits
        self._matrix: np.ndarray | None = None   # lazy build

    # ── Matrix construction ────────────────────────────────────────────────

    def _build(self) -> np.ndarray:
        """Construct the (I)QFT matrix using vectorised outer product."""
        dim   = self._dim
        omega = np.exp(self._sign * 2j * np.pi / dim)
        idx   = np.arange(dim)
        # M[i,j] = omega^(i*j) / sqrt(dim)
        return np.power(omega, np.outer(idx, idx)) / np.sqrt(dim)

    @property
    def matrix(self) -> np.ndarray:
        """Return (and cache) the full unitary matrix."""
        if self._matrix is None:
            self._matrix = self._build()
        return self._matrix

    @property
    def n_qubits(self) -> int:
        return self._n

    @property
    def dim(self) -> int:
        return self._dim

    # ── Application ────────────────────────────────────────────────────────

    def apply(self, state: QuantumState) -> QuantumState:
        """
        Apply this operator to *state* and return the transformed QuantumState.
        """
        if state.n_qubits != self._n:
            raise ValueError(
                f"Operator is {self._n}-qubit but state has "
                f"{state.n_qubits} qubits.")
        new_vec = self.matrix @ state.amplitudes
        lbl     = f"{self._label}({state.label})"
        return QuantumState(new_vec, label=lbl)

    def __call__(self, state: QuantumState) -> QuantumState:
        """Syntactic sugar: operator(state) == operator.apply(state)."""
        return self.apply(state)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_qubits={self._n}, dim={self._dim})"

    # subclasses must define _label
    _label = "F"


# ─────────────────────────────────────────────────────────────────────────────
# 3.  QFTOperator
# ─────────────────────────────────────────────────────────────────────────────

class QFTOperator(_FourierOperatorBase):
    """
    Quantum Fourier Transform operator for *n_qubits* qubits.

    The QFT matrix is:
        F_{jk} = exp(+2πi·jk / 2^n) / sqrt(2^n)

    Usage
    -----
    >>> qft = QFTOperator(3)
    >>> state_in  = QuantumState.computational_basis(5, 3)
    >>> state_out = qft(state_in)
    """
    _sign  = +1
    _label = "QFT"


# ─────────────────────────────────────────────────────────────────────────────
# 4.  IQFTOperator
# ─────────────────────────────────────────────────────────────────────────────

class IQFTOperator(_FourierOperatorBase):
    """
    Inverse Quantum Fourier Transform operator for *n_qubits* qubits.

    The IQFT matrix is:
        F†_{jk} = exp(−2πi·jk / 2^n) / sqrt(2^n)

    This is the conjugate-transpose of the QFT matrix:  IQFT = QFT†.

    Usage
    -----
    >>> iqft = IQFTOperator(3)
    >>> recovered = iqft(qft(state_in))   # should equal state_in
    """
    _sign  = -1
    _label = "IQFT"


# ─────────────────────────────────────────────────────────────────────────────
# 5.  UnitarityTester
# ─────────────────────────────────────────────────────────────────────────────

class UnitarityTester:
    """
    Suite of unitarity diagnostics for QFT / IQFT operators.

    Tests performed
    ---------------
    1. **U U† ≈ I**          — forward-inverse product is identity
    2. **U† U ≈ I**          — inverse-forward product is identity
    3. **Column orthonormality** — columns of U form an orthonormal set
    4. **Row orthonormality**    — rows of U form an orthonormal set
    5. **Norm preservation**  — ‖U|ψ⟩‖ = 1 for many random states
    6. **Round-trip fidelity** — F = |⟨ψ|IQFT(QFT(|ψ⟩))|² ≈ 1
    7. **Singular values**    — all singular values should equal 1
    8. **Eigenvalue phases**  — eigenvalues should lie on the unit circle

    Parameters
    ----------
    n_qubits : int
        Number of qubits.  Matrices grow as 2^n so keep n ≤ 10 for speed.
    n_random_states : int
        Number of random states used for norm-preservation / round-trip tests.
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(self, n_qubits: int, n_random_states: int = 200, seed: int = 0):
        self._n     = n_qubits
        self._dim   = 2 ** n_qubits
        self._nrand = n_random_states
        self._rng   = np.random.default_rng(seed)
        self.qft    = QFTOperator(n_qubits)
        self.iqft   = IQFTOperator(n_qubits)
        self._results: dict = {}

    # ── Individual diagnostics ─────────────────────────────────────────────

    def _product_error(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Return |A @ B − I| (element-wise absolute value)."""
        return np.abs(A @ B - np.eye(self._dim))

    def _random_states(self) -> list[QuantumState]:
        states = []
        for _ in range(self._nrand):
            r = self._rng.standard_normal(self._dim)
            i = self._rng.standard_normal(self._dim)
            v = r + 1j * i
            v /= np.linalg.norm(v)
            states.append(QuantumState(v))
        return states

    def run_all(self) -> dict:
        """Execute all tests and store results in self._results."""
        F  = self.qft.matrix
        Fd = self.iqft.matrix

        # 1 & 2 — product matrices
        FFd = F @ Fd
        FdF = Fd @ F
        self._results['FF†_error']    = np.abs(FFd - np.eye(self._dim))
        self._results['F†F_error']    = np.abs(FdF - np.eye(self._dim))
        self._results['FF†_max_err']  = float(np.max(self._results['FF†_error']))
        self._results['F†F_max_err']  = float(np.max(self._results['F†F_error']))

        # 3 & 4 — column / row norms and cross-products
        col_dots = np.abs(np.conj(F.T) @ F - np.eye(self._dim))
        row_dots = np.abs(F @ np.conj(F.T) - np.eye(self._dim))
        self._results['col_ortho_error'] = col_dots
        self._results['row_ortho_error'] = row_dots

        # 5 — norm preservation
        rand_states  = self._random_states()
        norms_before = np.array([s.norm for s in rand_states])
        norms_after  = np.array([self.qft(s).norm for s in rand_states])
        norm_errors  = np.abs(norms_after - norms_before)
        self._results['norm_errors']     = norm_errors
        self._results['norm_max_err']    = float(np.max(norm_errors))

        # 6 — round-trip fidelity
        fidelities = []
        for s in rand_states:
            s_rt = self.iqft(self.qft(s))
            fid  = float(np.abs(np.vdot(s.amplitudes, s_rt.amplitudes)) ** 2)
            fidelities.append(fid)
        self._results['fidelities']      = np.array(fidelities)
        self._results['fidelity_min']    = float(np.min(fidelities))

        # 7 — singular values
        svs = np.linalg.svd(F, compute_uv=False)
        self._results['singular_values'] = svs
        self._results['sv_max_deviation'] = float(np.max(np.abs(svs - 1.0)))

        # 8 — eigenvalues
        eigvals = np.linalg.eigvals(F)
        self._results['eigenvalues']     = eigvals
        ev_radii = np.abs(eigvals)
        self._results['ev_radius_max_dev'] = float(np.max(np.abs(ev_radii - 1.0)))

        return self._results

    def print_summary(self):
        """Print a concise pass/fail summary."""
        r = self._results
        tol = 1e-10
        print("=" * 60)
        print(f"  Unitarity Tests — {self._n}-qubit QFT  (dim = {self._dim})")
        print("=" * 60)
        tests = [
            ("F·F† = I  (max error)",        r.get('FF†_max_err',   np.nan), tol),
            ("F†·F = I  (max error)",        r.get('F†F_max_err',   np.nan), tol),
            ("Norm preservation (max err)",  r.get('norm_max_err',  np.nan), tol),
            ("Round-trip fidelity (min)",     r.get('fidelity_min',  np.nan), None),
            ("Singular values ≈ 1 (max dev)",r.get('sv_max_deviation', np.nan), tol),
            ("|eigenvalue| ≈ 1 (max dev)",   r.get('ev_radius_max_dev', np.nan), tol),
        ]
        for name, val, threshold in tests:
            if threshold is None:
                status = "✓" if abs(val - 1.0) < 1e-10 else "✗"
            else:
                status = "✓" if val < threshold else "✗"
            print(f"  {status}  {name:<40s}  {val:.3e}")
        print("=" * 60)

    # ── Plotting ───────────────────────────────────────────────────────────

    def plot_all(self, savename: str = "unitarity_tests.png"):
        """
        Produce a single figure with eight diagnostic sub-plots and save it.
        """
        if not self._results:
            self.run_all()
        r = self._results

        # ── Colour palette ─────────────────────────────────────────────────
        BG     = "#0d0f1a"
        PANEL  = "#131629"
        ACCENT = "#4fc3f7"
        WARM   = "#ff7043"
        GREEN  = "#69f0ae"
        GRID   = "#1e2340"
        TEXT   = "#e8eaf6"

        plt.rcParams.update({
            'figure.facecolor':  BG,
            'axes.facecolor':    PANEL,
            'axes.edgecolor':    GRID,
            'axes.labelcolor':   TEXT,
            'xtick.color':       TEXT,
            'ytick.color':       TEXT,
            'text.color':        TEXT,
            'grid.color':        GRID,
            'grid.linewidth':    0.6,
        })

        fig = plt.figure(figsize=(20, 15), facecolor=BG)
        fig.suptitle(
            f"Unitarity Tests — {self._n}-qubit QFT  (dim = {self._dim})",
            fontsize=17, fontweight='bold', color=TEXT, y=0.98)

        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38,
                               left=0.06, right=0.97, top=0.93, bottom=0.06)

        # ── Helper to make a labelled axes ────────────────────────────────
        def ax(row, col, **kw):
            a = fig.add_subplot(gs[row, col], **kw)
            a.grid(True, alpha=0.35)
            return a

        # ── 1: Heat-map  F·F† error ───────────────────────────────────────
        a1 = ax(0, 0)
        err_mat = r['FF†_error']
        vmax    = max(err_mat.max(), 1e-16)
        im1 = a1.imshow(err_mat, cmap='inferno', vmin=0, vmax=vmax,
                        interpolation='nearest', aspect='auto')
        plt.colorbar(im1, ax=a1, label='|error|')
        a1.set_title(f"F·F† − I  (max {r['FF†_max_err']:.2e})",
                     color=TEXT, fontweight='bold')
        a1.set_xlabel("column j");  a1.set_ylabel("row i")

        # ── 2: Heat-map  F†·F error ───────────────────────────────────────
        a2 = ax(0, 1)
        err2 = r['F†F_error']
        im2  = a2.imshow(err2, cmap='inferno', vmin=0,
                         vmax=max(err2.max(), 1e-16),
                         interpolation='nearest', aspect='auto')
        plt.colorbar(im2, ax=a2, label='|error|')
        a2.set_title(f"F†·F − I  (max {r['F†F_max_err']:.2e})",
                     color=TEXT, fontweight='bold')
        a2.set_xlabel("column j");  a2.set_ylabel("row i")

        # ── 3: Column orthonormality heat-map ─────────────────────────────
        a3 = ax(0, 2)
        ce   = r['col_ortho_error']
        im3  = a3.imshow(ce, cmap='plasma', vmin=0,
                         vmax=max(ce.max(), 1e-16),
                         interpolation='nearest', aspect='auto')
        plt.colorbar(im3, ax=a3, label='|error|')
        a3.set_title("Column orthonormality error",
                     color=TEXT, fontweight='bold')
        a3.set_xlabel("column j");  a3.set_ylabel("column i")

        # ── 4: Norm preservation histogram ────────────────────────────────
        a4 = ax(1, 0)
        ne = r['norm_errors']
        a4.hist(ne, bins=40, color=ACCENT, edgecolor=BG, alpha=0.85)
        a4.axvline(ne.mean(), color=WARM, lw=2,
                   label=f"mean={ne.mean():.2e}")
        a4.axvline(ne.max(), color=GREEN, lw=2, ls='--',
                   label=f"max={ne.max():.2e}")
        a4.set_title("Norm preservation error\n"
                     f"({self._nrand} random states)",
                     color=TEXT, fontweight='bold')
        a4.set_xlabel("|‖F|ψ⟩‖ − 1|"); a4.set_ylabel("count")
        a4.legend(fontsize=9)

        # ── 5: Round-trip fidelity histogram ──────────────────────────────
        a5 = ax(1, 1)
        fi = r['fidelities']
        a5.hist(1 - fi, bins=40, color=GREEN, edgecolor=BG, alpha=0.85)
        a5.axvline((1 - fi).mean(), color=WARM, lw=2,
                   label=f"mean={((1-fi).mean()):.2e}")
        a5.set_title("Round-trip infidelity  1−F\n"
                     "(IQFT∘QFT, random states)",
                     color=TEXT, fontweight='bold')
        a5.set_xlabel("1 − |⟨ψ|IQFT(QFT|ψ⟩)|²")
        a5.set_ylabel("count")
        a5.legend(fontsize=9)

        # ── 6: Singular values ────────────────────────────────────────────
        a6 = ax(1, 2)
        svs = r['singular_values']
        a6.scatter(range(len(svs)), svs, color=ACCENT, s=14, alpha=0.75,
                   zorder=3)
        a6.axhline(1.0, color=GREEN, lw=1.5, ls='--', label='σ = 1  (ideal)')
        a6.fill_between(range(len(svs)),
                        1 - 1e-10, 1 + 1e-10, color=GREEN, alpha=0.12)
        a6.set_title(f"Singular values of F\n"
                     f"(max dev from 1: {r['sv_max_deviation']:.2e})",
                     color=TEXT, fontweight='bold')
        a6.set_xlabel("index");  a6.set_ylabel("σ")
        a6.legend(fontsize=9)

        # ── 7: Eigenvalues on the unit circle ─────────────────────────────
        a7 = ax(2, 0, aspect='equal')
        ev   = r['eigenvalues']
        th   = np.linspace(0, 2 * np.pi, 400)
        a7.plot(np.cos(th), np.sin(th), color=GRID, lw=1.5, zorder=1)
        sc = a7.scatter(ev.real, ev.imag,
                        c=np.angle(ev), cmap='hsv',
                        s=28, alpha=0.85, zorder=3,
                        vmin=-np.pi, vmax=np.pi)
        plt.colorbar(sc, ax=a7, label='arg(λ)  [rad]')
        a7.axhline(0, color=GRID, lw=0.8); a7.axvline(0, color=GRID, lw=0.8)
        a7.set_title(f"Eigenvalues on unit circle\n"
                     f"(max |λ|−1: {r['ev_radius_max_dev']:.2e})",
                     color=TEXT, fontweight='bold')
        a7.set_xlabel("Re(λ)");  a7.set_ylabel("Im(λ)")

        # ── 8: QFT of computational-basis states (prob. distribution) ─────
        a8 = ax(2, 1)
        for idx in range(min(4, self._dim)):
            s    = QuantumState.computational_basis(idx, self._n)
            sf   = self.qft(s)
            probs = sf.probabilities
            a8.plot(probs, lw=1.5, alpha=0.8, label=f"|{idx}⟩")
        a8.set_title("QFT probability distribution\n(basis states |0⟩…|3⟩)",
                     color=TEXT, fontweight='bold')
        a8.set_xlabel("output basis state");  a8.set_ylabel("probability")
        a8.legend(fontsize=9)

        # ── 9: Row orthonormality heat-map ────────────────────────────────
        a9 = ax(2, 2)
        re   = r['row_ortho_error']
        im9  = a9.imshow(re, cmap='magma', vmin=0,
                         vmax=max(re.max(), 1e-16),
                         interpolation='nearest', aspect='auto')
        plt.colorbar(im9, ax=a9, label='|error|')
        a9.set_title("Row orthonormality error",
                     color=TEXT, fontweight='bold')
        a9.set_xlabel("row j");  a9.set_ylabel("row i")

        plt.savefig(savename, dpi=150, bbox_inches='tight',
                    facecolor=BG)
        plt.close()
        plt.rcParams.update(plt.rcParamsDefault)
        print(f"Saved {savename}")

    def plot_matrix(self, savename: str = "qft_matrix.png"):
        """
        Visualise the real and imaginary parts of the QFT matrix side-by-side,
        plus the phase (argument) of each entry.
        """
        F   = self.qft.matrix
        BG  = "#0d0f1a"; TEXT = "#e8eaf6"; GRID = "#1e2340"

        plt.rcParams.update({
            'figure.facecolor': BG, 'axes.facecolor': "#131629",
            'axes.edgecolor': GRID, 'axes.labelcolor': TEXT,
            'xtick.color': TEXT, 'ytick.color': TEXT, 'text.color': TEXT,
        })

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
        parts = [
            (F.real,     'RdBu_r', 'Re(F_{jk})'),
            (F.imag,     'RdBu_r', 'Im(F_{jk})'),
            (np.angle(F), 'hsv',   'arg(F_{jk})  [rad]'),
        ]
        for ax_i, (data, cmap, title) in zip(axes, parts):
            vabs = np.max(np.abs(data))
            im = ax_i.imshow(data, cmap=cmap,
                             vmin=-vabs, vmax=vabs,
                             interpolation='nearest', aspect='auto')
            plt.colorbar(im, ax=ax_i)
            ax_i.set_title(title, color=TEXT, fontweight='bold', fontsize=13)
            ax_i.set_xlabel("column k");  ax_i.set_ylabel("row j")

        fig.suptitle(
            f"QFT Matrix — {self._n} qubits  (dim = {self._dim})",
            fontsize=15, fontweight='bold', color=TEXT, y=1.01)
        plt.tight_layout()
        plt.savefig(savename, dpi=150, bbox_inches='tight', facecolor=BG)
        plt.close()
        plt.rcParams.update(plt.rcParamsDefault)
        print(f"Saved {savename}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Demo / __main__
# ─────────────────────────────────────────────────────────────────────────────

def run_demo(n_qubits: int = 4):
    print("=" * 60)
    print(f"  QFT / IQFT Demo  —  {n_qubits} qubits  (dim = {2**n_qubits})")
    print("=" * 60)

    qft_op  = QFTOperator(n_qubits)
    iqft_op = IQFTOperator(n_qubits)

    # ── (a) Computational basis state |5⟩ ─────────────────────────────────
    s5  = QuantumState.computational_basis(5, n_qubits)
    fs5 = qft_op(s5)
    r5  = iqft_op(fs5)
    print(f"\nInitial state:          {s5}")
    print(f"After QFT:              {fs5}")
    print(f"After IQFT (round-trip):{r5}")
    fid = np.abs(np.vdot(s5.amplitudes, r5.amplitudes)) ** 2
    print(f"Round-trip fidelity:    {fid:.15f}")

    # ── (b) Uniform superposition ──────────────────────────────────────────
    sup = QuantumState.uniform_superposition(n_qubits)
    fs  = qft_op(sup)
    print(f"\nUniform superposition → QFT → {fs}")

    # ── (c) Random state ──────────────────────────────────────────────────
    rnd  = QuantumState.random(n_qubits, seed=42)
    frnd = qft_op(rnd)
    rrnd = iqft_op(frnd)
    fid2 = np.abs(np.vdot(rnd.amplitudes, rrnd.amplitudes)) ** 2
    print(f"\nRandom state round-trip fidelity: {fid2:.15f}")

    # ── (d) Verify IQFT.matrix == QFT.matrix† ─────────────────────────────
    diff = np.max(np.abs(iqft_op.matrix - qft_op.matrix.conj().T))
    print(f"\n‖IQFT − QFT†‖_max = {diff:.3e}  (should be ~0)")

    # ── (e) Unitarity tests ────────────────────────────────────────────────
    tester = UnitarityTester(n_qubits, n_random_states=500, seed=0)
    tester.run_all()
    tester.print_summary()
    tester.plot_all(savename="unitarity_tests.png")
    tester.plot_matrix(savename="qft_matrix.png")


if __name__ == "__main__":
    run_demo(n_qubits=4)
