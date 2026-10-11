"""Chapter 6, Sec. 6.9: the Harrow-Hassidim-Lloyd algorithm on the state-vector simulator.

``HHL`` solves A x = b for a Hermitian A on n qubits by the circuit of
Fig. 6.hhl: phase estimation of U = exp(i A t) with m clock qubits, a
rotation of one ancilla by an angle whose sine is C / lambda for every
eigenvalue label lambda in the clock register, the inverse of the phase
estimation, and post-selection of the ancilla on |1>.  Register layout:
qubit 0 the ancilla, qubits 1..m the clock (qubit 1 the most significant
bit), qubits m+1..m+n the system.  The clock is read with the signed
convention of Eq. (6.signedphase), so that A may be indefinite, which the
resolvent application of Sec. 6.9.4 needs.

Run as a script it works through the 2 x 2 example of Eq. (6.hhlexample)
with exactly representable phases, the effect of an inexact clock register
(Table 6.hhlclock), the discretised Poisson equation of Eq. (6.poisson) on
two system qubits (Table 6.poisson, Fig. 6.hhlpoisson), and the resolvent
(z - H)^{-1} of the two-level pairing Hamiltonian (Fig. 6.hhlresolvent).
About ten seconds.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np
import gates as G
from statevector import StateVector
from pauli import PauliSum
from qft import qft_circuit, inverse_qft_circuit
from phase_estimation import controlled, signed_phase
from pauli_circuits import compact_pairing
from models import PairingModel
from bookplot import savefig, plt


class HHL:
    """Quantum linear-system solver for a Hermitian matrix A on n system qubits."""

    def __init__(self, A, m, t=None, C=None, s=0.9, window="sine"):
        self.A = np.asarray(A, dtype=complex)
        self.window = window
        self.n = int(round(np.log2(self.A.shape[0])))
        self.m = m
        self.w, self.V = np.linalg.eigh(self.A)
        # |lambda| t < pi for every eigenvalue: t from the spectral norm (or from a Pauli 1-norm on hardware)
        self.t = s * np.pi / np.abs(self.w).max() if t is None else t
        # the rotation constant: C must not exceed any eigenvalue label that carries
        # weight, and an inexact register spreads the weight over several grid
        # points around lambda_min, so C is placed ten per cent below it
        self.grid = 2 * np.pi / (2 ** m * self.t)
        self.C = 0.9 * np.abs(self.w).min() if C is None else C
        self.power = lambda p: (self.V * np.exp(1j * self.w * self.t * p)) @ self.V.conj().T   # U^p, U = exp(i A t)
        self.state = None

    # ------------------------------------------------------------ pieces
    def clock_preparation(self):
        """The unitary W with W|0...0> = |w>: Hadamards for the uniform clock state of
        ordinary phase estimation, or a reflection preparing the sine window
        w_tau = sqrt(2/T) sin(pi (tau + 1/2) / T) of Eq. (6.sinewindow)."""
        T = 2 ** self.m
        if self.window == "uniform":
            return G.kron(*([G.H] * self.m))
        w = np.sqrt(2 / T) * np.sin(np.pi * (np.arange(T) + 0.5) / T)
        v = np.eye(T)[0] - w                                        # Householder reflection e_0 -> w
        v /= np.linalg.norm(v)
        return np.eye(T) - 2 * np.outer(v, v)

    def _qpe(self, sv, inverse=False):
        """Phase estimation on clock + system, or its exact inverse."""
        m, n = self.m, self.n
        clock, system = list(range(1, m + 1)), list(range(m + 1, m + n + 1))
        W = self.clock_preparation()
        if not inverse:
            sv.apply(W, clock)
            for j, q in enumerate(clock):                           # clock qubit j controls U^(2^(m-1-j))
                sv.apply(controlled(self.power(2 ** (m - 1 - j))), [q] + system)
            for g in inverse_qft_circuit(m).gates:
                sv.apply(g.full_matrix(), [clock[i] for i in g.qubits])
        else:
            for g in qft_circuit(m).gates:
                sv.apply(g.full_matrix(), [clock[i] for i in g.qubits])
            for j, q in reversed(list(enumerate(clock))):
                sv.apply(controlled(self.power(2 ** (m - 1 - j)).conj().T), [q] + system)
            sv.apply(W.conj().T, clock)
        return sv

    def eigenvalue_labels(self):
        """The eigenvalue lambda_k = 2 pi phi_k / t that clock outcome k stands for (signed)."""
        return 2 * np.pi * signed_phase(np.arange(2 ** self.m), self.m) / self.t

    def _rotation(self, sv):
        """R_y(theta_k) on the ancilla for every clock value k, sin(theta_k / 2) = C / lambda_k, Eq. (6.hhlrotation)."""
        m, n = self.m, self.n
        amp = sv.psi.reshape(2, 2 ** m, 2 ** n)
        lam = self.eigenvalue_labels()
        for k in range(1, 2 ** m):                                  # k = 0 (lambda = 0) is left alone
            ratio = self.C / lam[k]
            if abs(ratio) > 1:
                continue                                            # |lambda| below C: no valid rotation, skip
            theta = 2 * np.arcsin(ratio)
            amp[:, k, :] = G.Ry(theta) @ amp[:, k, :]
        sv.psi = amp.reshape(-1)
        return sv

    # --------------------------------------------------------------- run
    def run(self, b):
        m, n = self.m, self.n
        b = np.asarray(b, dtype=complex).reshape(2 ** n)
        self.b = b / np.linalg.norm(b)
        sv = StateVector(1 + m + n)
        full = np.zeros(2 ** (1 + m + n), dtype=complex)
        full[:2 ** n] = self.b                                      # ancilla and clock in |0>
        sv.psi = full
        self._qpe(sv)
        self.clock_distribution = sv.probabilities().reshape(2, 2 ** m, 2 ** n).sum(axis=(0, 2))
        self._rotation(sv)
        self._qpe(sv, inverse=True)
        self.state = sv
        amp = sv.psi.reshape(2, 2 ** m, 2 ** n)
        self.success = float(np.sum(np.abs(amp[1]) ** 2))         # ancilla read as |1>
        self.clock_reset = float(np.sum(np.abs(amp[1, 0]) ** 2)) / self.success   # clock back in |0...0>
        self.x_unnormalised = amp[1, 0].copy()                     # C sum_j beta_j / lambda_j |u_j>
        self.x = self.x_unnormalised / np.linalg.norm(self.x_unnormalised)
        x_exact = np.linalg.solve(self.A, self.b)
        self.x_exact = x_exact / np.linalg.norm(x_exact)
        self.fidelity = abs(np.vdot(self.x_exact, self.x)) ** 2
        return self

    def solution(self, norm_b=1.0):
        """The classical vector A^{-1} b with its scale, from the output state and the
        success probability: x = sqrt(p) |x> / C times the norm of b."""
        return norm_b * np.sqrt(self.success) * self.x / self.C

    def bracket(self):
        """<b| A^{-1} |b> from the run: <b|x~> / C, Eq. (6.hhlresolvent)."""
        return np.vdot(self.b, self.x_unnormalised) / self.C


def poisson_system(n_points):
    """-u'' = sin(pi x) on (0, 1) with u(0) = u(1) = 0 on n interior points, Eq. (6.poisson)."""
    h = 1.0 / (n_points + 1)
    x = np.arange(1, n_points + 1) * h
    A = (2 * np.eye(n_points) - np.eye(n_points, k=1) - np.eye(n_points, k=-1)) / h ** 2
    return A, np.sin(np.pi * x), x, h


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True, linewidth=120)

    # ------------------------------------------------------ the 2 x 2 example
    print("=== A = [[1.5, 0.5], [0.5, 1.5]], b = (1, 0): eigenvalues 1 and 2, t = pi/4, m = 3 (Eq. 6.hhlexample)")
    A = np.array([[1.5, 0.5], [0.5, 1.5]])
    b = np.array([1.0, 0.0])
    hhl = HHL(A, m=3, t=np.pi / 4, C=1.0, window="uniform").run(b)
    print(f"  clock distribution after phase estimation: {hhl.clock_distribution}")
    print(f"    (lambda = 1 -> phi = 1/8 -> |001>, lambda = 2 -> phi = 2/8 -> |010>, each with weight |beta_j|^2 = 1/2)")
    print(f"  success probability {hhl.success:.6f} (C^2 sum beta_j^2 / lambda_j^2 = {0.5 + 0.5 / 4:.6f})")
    print(f"  clock returned to |000>: {hhl.clock_reset:.6f}; output state {hhl.x}; exact {hhl.x_exact}; fidelity {hhl.fidelity:.10f}")
    print(f"  classical solution recovered with its scale: {hhl.solution().real}, exact {np.linalg.solve(A, b)}")
    assert abs(hhl.success - 0.625) < 1e-12 and abs(hhl.fidelity - 1) < 1e-12 and abs(hhl.clock_reset - 1) < 1e-12
    assert np.allclose(hhl.solution().real, [0.75, -0.25])
    # the success probability is 1/kappa^2 when b is the eigenvector of the largest eigenvalue
    for kappa in (2, 5, 10):
        Ak = np.diag([1.0, float(kappa)])
        h = HHL(Ak, m=5, t=np.pi / 16, C=1.0, window="uniform").run([0, 1])      # phi = kappa/32, exact
        assert abs(h.success - 1 / kappa ** 2) < 1e-10
    print("  b = |u_max>: success probability 1/kappa^2 for kappa = 2, 5, 10")

    print("\n=== The same system with a generic t = 0.9 pi / 2: the clock no longer represents the eigenvalues exactly (Table 6.hhlclock)")
    print(f"{'m':>3s} {'C':>7s} {'window':>8s} {'success':>8s} {'clock reset':>12s} {'1 - fidelity':>13s} {'x (scaled)':>20s}")
    rows = {}
    for window in ("uniform", "sine"):
        for m in (3, 4, 6, 8, 10):
            h = HHL(A, m=m, window=window).run(b)
            rows[(window, m)] = (h.success, h.clock_reset, 1 - h.fidelity, h.solution().real)
            print(f"{m:3d} {h.C:7.4f} {window:>8s} {h.success:8.4f} {h.clock_reset:12.6f} {1 - h.fidelity:13.2e} {str(np.round(h.solution().real, 4)):>20s}")
    assert rows[("sine", 10)][2] < 1e-9 and rows[("uniform", 10)][2] > 100 * rows[("sine", 10)][2]
    assert np.allclose(rows[("sine", 10)][3], [0.75, -0.25], atol=2e-3)

    # ------------------------------------------------------ Poisson
    print("\n=== -u'' = sin(pi x), four interior points, two system qubits (Table 6.poisson)")
    Ap, bp, xp, hp = poisson_system(4)
    wp = np.linalg.eigvalsh(Ap)
    print(f"  eigenvalues {wp}, condition number {wp[-1] / wp[0]:.3f}")
    u_fd = np.linalg.solve(Ap, bp)
    u_ex = np.sin(np.pi * xp) / np.pi ** 2
    print(f"  finite-difference solution {u_fd}, exact {u_ex}, discretisation error {np.abs(u_fd - u_ex).max():.2e}")
    print(f"{'m':>3s} {'success':>9s} {'fidelity':>10s} {'max |u_HHL - u_FD|':>20s} {'max |u_HHL - u|':>17s}")
    prows = []
    for m in (4, 5, 6, 7, 8, 9, 10):
        h = HHL(Ap, m=m).run(bp)
        u = h.solution(np.linalg.norm(bp)).real
        prows.append((m, h.success, h.fidelity, np.abs(u - u_fd).max(), np.abs(u - u_ex).max(), u))
        print(f"{m:3d} {h.success:9.4f} {h.fidelity:10.6f} {np.abs(u - u_fd).max():20.2e} {np.abs(u - u_ex).max():17.2e}")
    assert prows[-1][3] < 1e-3
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    xx = np.linspace(0, 1, 200)
    axes[0].plot(xx, np.sin(np.pi * xx) / np.pi ** 2, "k-", lw=1, label=r"$\sin(\pi x)/\pi^2$")
    axes[0].plot(xp, u_fd, "ks", ms=7, mfc="none", label="finite differences")
    for m, mk in ((5, "o"), (7, "^"), (10, "d")):
        u = [r[5] for r in prows if r[0] == m][0]
        axes[0].plot(xp, u, mk, ms=5, label=f"HHL, $m={m}$")
    axes[0].set_xlabel("$x$"); axes[0].set_ylabel("$u(x)$"); axes[0].legend(fontsize=8)
    axes[1].semilogy([r[0] for r in prows], [r[3] for r in prows], "o-", label=r"$\max|u_{\rm HHL}-u_{\rm FD}|$")
    axes[1].semilogy([r[0] for r in prows], [r[4] for r in prows], "s--", label=r"$\max|u_{\rm HHL}-u|$")
    axes[1].axhline(np.abs(u_fd - u_ex).max(), color="gray", ls=":", lw=0.8)
    axes[1].set_xlabel("clock qubits $m$"); axes[1].legend(fontsize=8)
    savefig(fig, 6, "hhl_poisson")

    # ------------------------------------------------------ the resolvent of a Hamiltonian
    print("\n=== The resolvent (z - H)^{-1} of the two-level pairing Hamiltonian, b = |10> (Fig. 6.hhlresolvent)")
    H = compact_pairing(PairingModel(2, g=1.0)).matrix().real
    w, V = np.linalg.eigh(H)
    bres = StateVector.from_bits("10").psi
    beta2 = np.abs(V.T @ bres) ** 2
    print(f"  eigenvalues {w}, |beta_j|^2 = {beta2}")
    zs = -2.5 + 0.05 * (np.arange(110) + 0.37)                    # a grid that avoids the eigenvalues
    exact = np.array([np.vdot(bres, np.linalg.solve(z * np.eye(4) - H, bres)).real for z in zs])
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(zs, exact, "k-", lw=1, label=r"exact $\langle b|(z-H)^{-1}|b\rangle$")
    worst = {}
    for m, mk in ((6, "o"), (9, "s")):
        gh = []
        for z in zs:
            Az = z * np.eye(4) - H
            h = HHL(Az, m=m).run(bres)
            gh.append(h.bracket().real)
        gh = np.array(gh)
        ax.plot(zs, gh, mk, ms=3, label=f"HHL, $m={m}$")
        far = np.min(np.abs(zs[:, None] - w[None, :]), axis=1) > 0.3
        worst[m] = (np.abs(gh - exact) / np.abs(exact))[far].max()
        print(f"  m = {m}: largest relative error at least 0.3 away from every pole {worst[m]:.2e}; success probability at z = 2: "
              f"{HHL(2 * np.eye(4) - H, m=m).run(bres).success:.4f}, at z = -1: {HHL(-np.eye(4) - H, m=m).run(bres).success:.4f}")
    print(f"{'z':>6s} {'exact G':>9s} {'HHL m=6':>9s} {'HHL m=9':>9s} {'success m=9':>12s}")
    for z in (-1.5, -0.25, 0.5, 2.5):
        g = np.vdot(bres, np.linalg.solve(z * np.eye(4) - H, bres)).real
        h6 = HHL(z * np.eye(4) - H, m=6).run(bres); h9 = HHL(z * np.eye(4) - H, m=9).run(bres)
        print(f"{z:6.2f} {g:9.4f} {h6.bracket().real:9.4f} {h9.bracket().real:9.4f} {h9.success:12.4f}")
        assert abs(h9.bracket().real - g) < 0.02 * abs(g)
    for e in w:
        ax.axvline(e, color="gray", ls=":", lw=0.7)
    ax.set_ylim(-6, 6); ax.set_xlabel("$z$"); ax.set_ylabel(r"$G_b(z)$"); ax.legend(fontsize=8)
    savefig(fig, 6, "hhl_resolvent")
    assert worst[9] < worst[6]
    print("\nHHL checks passed")
