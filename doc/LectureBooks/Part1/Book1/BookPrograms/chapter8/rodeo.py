"""Chapter 8, Sec. 8.3: the rodeo algorithm.

One cycle is the one-ancilla circuit of Fig. 8.rodeocircuit: Hadamard,
controlled exp(-i H t), phase gate P(E t) on the ancilla, Hadamard,
measurement.  The outcome 0 applies the filter of Eq. (8.filter),

    F(t, E) = (1 + exp(i E t) exp(-i H t)) / 2,

to the system register; an eigenstate |E_j> survives with probability
cos^2((E_j - E) t / 2), Eq. (8.cos2).

``RodeoFilter``  * ``cycle_ancilla``: the circuit on the joint register, gate by gate
                 * ``cycle``: the filter applied directly (the two agree to 1e-15)
                 * ``all_zero_probability``: Eq. (8.allzero) for a given set of times
                 * ``scan``: the energy scan averaged over Gaussian time sets
                 * ``prepare``: fidelity and acceptance against the number of cycles
                 * ``run``: a stochastic run with sampled ancilla outcomes
``gaussian_factor``  the averaged one-cycle survival (1 + exp(-x^2 sigma^2/2))/2, Eq. (8.gaussian)

As a script: the one-qubit tapered pairing Hamiltonian of Eq. (4.tapered) (checks of
Eqs. 8.cos2-8.gaussian, Fig. 8.onequbit), the explicit circuit against the filter on
a random Hamiltonian, a sampled run, and the infidelity bound of Eq. (8.fidelity).
About 5 s.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np
import gates as G
from bookplot import savefig, plt


def gaussian_factor(x, sigma):
    """Average of cos^2(x t / 2) over t ~ N(0, sigma^2), Eq. (8.gaussian)."""
    return 0.5 * (1 + np.exp(-0.5 * (x * sigma) ** 2))


class RodeoFilter:
    """The rodeo algorithm for a Hermitian matrix H (any basis, any sector)."""

    def __init__(self, H, sigma=4.0, rng=None, evolution=None):
        self.H = np.asarray(H)
        self.sigma = sigma
        self.rng = np.random.default_rng(1) if rng is None else rng
        self.w, self.V = np.linalg.eigh(self.H)
        # evolution(t) -> matrix of exp(-i H t); replace it by a product formula to study Trotter errors
        self.evolution = self._exact if evolution is None else evolution

    def _exact(self, t):
        return (self.V * np.exp(-1j * self.w * t)) @ self.V.conj().T

    def times(self, M):
        return self.rng.normal(0.0, self.sigma, size=M)

    # ---- one cycle
    def cycle_ancilla(self, psi, t, E):
        """The circuit on ancilla (first factor) + system.  Returns the normalised system
        states and probabilities of the two ancilla outcomes."""
        d = len(psi)
        joint = np.kron(np.array([1, 0], dtype=complex), psi)
        Id = np.eye(d)
        joint = np.kron(G.H, Id) @ joint
        U = self.evolution(t)
        cU = np.block([[Id, np.zeros((d, d))], [np.zeros((d, d)), U]])
        joint = cU @ joint
        joint = np.kron(G.phase(E * t), Id) @ joint            # |1> -> e^{iEt} |1>
        joint = np.kron(G.H, Id) @ joint
        out = []
        for a in (0, 1):
            branch = joint[a * d:(a + 1) * d]
            p = float(np.vdot(branch, branch).real)
            out.append((branch / np.sqrt(p) if p > 1e-300 else branch, p))
        return out

    def cycle(self, psi, t, E):
        """Outcome 0: the filter (1 + e^{iEt} e^{-iHt})/2.  Returns (state, probability)."""
        phi = 0.5 * (psi + np.exp(1j * E * t) * (self.evolution(t) @ psi))
        p = float(np.vdot(phi, phi).real)
        return phi / np.sqrt(p), p

    # ---- many cycles
    def all_zero_probability(self, psi, times, E):
        """P(0...0) for the given times: sum_j |c_j|^2 prod_k cos^2((E_j - E) t_k / 2)."""
        c2 = np.abs(self.V.conj().T @ psi) ** 2
        f = np.ones_like(self.w)
        for t in times:
            f = f * np.cos(0.5 * (self.w - E) * t) ** 2
        return float(c2 @ f)

    def scan(self, psi, energies, M, n_sets=40):
        """Energy scan: P(0...0)(E) averaged over n_sets Gaussian time sets."""
        sets = [self.times(M) for _ in range(n_sets)]
        return np.array([np.mean([self.all_zero_probability(psi, ts, E) for ts in sets]) for E in energies])

    def prepare(self, psi, E, M, target, n_sets=200):
        """Infidelity of the accepted runs with `target` and mean acceptance after 1..M cycles.
        A time set that is accepted with probability P contributes to the accepted ensemble
        in proportion to P, so the infidelity is the P-weighted mean over time sets."""
        infid, acc = np.zeros(M), np.zeros(M)
        for _ in range(n_sets):
            phi, P = psi.astype(complex), 1.0
            for k, t in enumerate(self.times(M)):
                phi, p = self.cycle(phi, t, E)
                P *= p
                infid[k] += P * (1 - abs(np.vdot(target, phi)) ** 2)
                acc[k] += P
        return infid / acc, acc / n_sets

    def expected(self, psi, E, M):
        """Exact Gaussian averages for cycles 1..M, Eq. (8.PM): the acceptance
        P_M = sum_j |c_j|^2 f_j^M with f_j = (1 + exp(-(E_j-E)^2 sigma^2/2))/2, and the
        infidelity 1 - |c_0|^2 f_0^M / P_M of the accepted runs with the eigenstate nearest E."""
        c2 = np.abs(self.V.conj().T @ psi) ** 2
        f = gaussian_factor(self.w - E, self.sigma)
        j0 = int(np.argmin(np.abs(self.w - E)))
        Ms = np.arange(1, M + 1)[:, None]
        PM = (c2 * f ** Ms).sum(axis=1)
        return 1 - c2[j0] * f[j0] ** Ms[:, 0] / PM, PM

    def run(self, psi, E, M):
        """One run as on a device: every ancilla outcome is sampled.  Returns (outcomes, state)."""
        phi, record = psi.astype(complex), []
        for t in self.times(M):
            (phi0, p0), (phi1, p1) = self.cycle_ancilla(phi, t, E)
            a = int(self.rng.random() > p0)
            record.append(a)
            phi = phi1 if a else phi0
        return record, phi


def infidelity_bound(p, M, gap_sigma=np.inf):
    """(1-p)/p f^M with f = (1 + exp(-(gap sigma)^2/2))/2, Eq. (8.fidelity); f = 1/2 when
    sigma is large against the inverse gap to the nearest level that carries weight."""
    return (1 - p) / p * gaussian_factor(gap_sigma, 1.0) ** M


if __name__ == "__main__":
    rng = np.random.default_rng(8)

    # ---- one qubit: the tapered two-level pairing Hamiltonian of Eq. (4.tapered),
    # H = I/2 + Z - X/2 with eigenvalues (1 -+ sqrt 5)/2
    H1 = np.array([[1.5, -0.5], [-0.5, -0.5]])                 # I/2 + Z - X/2
    rf = RodeoFilter(H1, sigma=3.0, rng=rng)
    E0, E1 = rf.w
    print(f"One-qubit Hamiltonian: E0 = {E0:.6f}, E1 = {E1:.6f}")
    ground, excited = rf.V[:, 0], rf.V[:, 1]

    # Eq. (8.cos2): an eigenstate survives one cycle with probability cos^2((E_j - E) t / 2)
    t, E = 1.7, 0.3
    (s0, p0), (s1, p1) = rf.cycle_ancilla(excited, t, E)
    print(f"  excited state, t = {t}, E = {E}: P(0) = {p0:.12f},  cos^2 = {np.cos(0.5 * (E1 - E) * t) ** 2:.12f}")
    assert abs(p0 - np.cos(0.5 * (E1 - E) * t) ** 2) < 1e-13 and abs(p0 + p1 - 1) < 1e-13

    # the explicit circuit against the filter, on a random 16-dimensional Hamiltonian
    A = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    big = RodeoFilter(A + A.conj().T, sigma=2.0, rng=rng)
    psi = rng.normal(size=16) + 1j * rng.normal(size=16); psi /= np.linalg.norm(psi)
    (c0, q0), _ = big.cycle_ancilla(psi, 0.83, -1.1)
    f0, r0 = big.cycle(psi, 0.83, -1.1)
    print(f"  circuit against filter, 16 x 16 random H: |dpsi| = {np.linalg.norm(c0 - f0):.1e}, |dP| = {abs(q0 - r0):.1e}")
    assert np.linalg.norm(c0 - f0) < 1e-13 and abs(q0 - r0) < 1e-13

    # Eq. (8.gaussian): the Gaussian average of one cycle
    x = E1 - E0
    ts = rng.normal(0, 3.0, size=200000)
    print(f"  Gaussian average of cos^2(x t/2), x = E1 - E0: sampled {np.mean(np.cos(0.5 * x * ts) ** 2):.4f}, "
          f"formula {gaussian_factor(x, 3.0):.4f}")

    # ---- Fig. 8.onequbit: energy scans of |+> for M = 1, 3, 6 and the formula
    psi = np.array([1.0, 1.0]) / np.sqrt(2)                       # |+>
    c2 = np.abs(rf.V.T @ psi) ** 2
    energies = np.linspace(-2.5, 3.5, 241)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for M, col in ((1, "C0"), (3, "C1"), (6, "C3")):
        a.plot(energies, rf.scan(psi, energies, M, n_sets=300), col, label=f"$M={M}$")
        formula = sum(c2[j] * gaussian_factor(rf.w[j] - energies, 3.0) ** M for j in range(2))
        a.plot(energies, formula, col, ls=":", lw=1)
    for Ej in rf.w:
        a.axvline(Ej, color="k", lw=0.5, ls="--")
    a.set_xlabel("target energy $E$"); a.set_ylabel("$P_{0^M}(E)$"); a.legend()
    a.set_title(f"input $|+\\rangle$: weights {c2[0]:.3f}, {c2[1]:.3f}")
    print(f"  input |+>: weights {c2[0]:.4f} (ground) and {c2[1]:.4f}; peak heights -> the weights as M grows")
    for M in (1, 3, 6, 10):
        print(f"    M = {M:2d}: P(E0) = {c2[0] + c2[1] * gaussian_factor(x, 3.0) ** M:.4f}, "
              f"P(E1) = {c2[1] + c2[0] * gaussian_factor(x, 3.0) ** M:.4f}, background 2^-M = {2.0 ** -M:.4f}")

    # preparation of the ground state from |+>: infidelity against the bound
    M = 10
    infid, acc = rf.prepare(psi, E0, M, ground, n_sets=2000)
    exact_inf, exact_acc = rf.expected(psi, E0, M)
    b.semilogy(range(1, M + 1), infid, "o", ms=4, label="2000 sampled time sets")
    b.semilogy(range(1, M + 1), exact_inf, "C0-", lw=1, label="Gaussian average, Eq. (8.PM)")
    b.semilogy(range(1, M + 1), infidelity_bound(c2[0], np.arange(1, M + 1)), "k:", label="$\\frac{1-p}{p}2^{-M}$")
    b.set_xlabel("cycles $M$"); b.set_ylabel("$1-\\mathcal{F}_M$"); b.legend()
    b.set_title(f"acceptance $\\to p={c2[0]:.3f}$")
    fig.tight_layout(); savefig(fig, 8, "onequbit")
    print("\n   M   infidelity (sampled, exact average, bound)      acceptance (sampled, exact)")
    for k in range(M):
        print(f"  {k + 1:2d}   {infid[k]:.3e}   {exact_inf[k]:.3e}   {infidelity_bound(c2[0], k + 1):.3e}        {acc[k]:.4f}   {exact_acc[k]:.4f}")
    assert np.all(exact_inf <= infidelity_bound(c2[0], np.arange(1, M + 1)) * (1 + 1e-6))
    assert abs(acc[-1] - c2[0]) < 0.02

    # ---- a run as on a device
    print("\nSampled runs at E = E0 (M = 6): ancilla record, fidelity with the ground state")
    accepted = 0
    for r in range(8):
        rec, phi = rf.run(psi, E0, 6)
        ok = not any(rec); accepted += ok
        print(f"  {''.join(map(str, rec))}  {'accepted' if ok else 'rejected'}  F = {abs(ground @ phi) ** 2:.6f}")
    n_runs = 4000
    hits = sum(not any(rf.run(psi, E0, 6)[0]) for _ in range(n_runs))
    print(f"  accepted fraction over {n_runs} runs: {hits / n_runs:.3f}  (Eq. 8.PM: {exact_acc[5]:.3f}; p = {c2[0]:.3f})")
