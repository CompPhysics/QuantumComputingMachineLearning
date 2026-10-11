"""Chapter 8, Sec. 8.7.2: the pipeline with product formulas instead of exact exponentials.

The pairing Hamiltonian is split as the gate compilation splits it,
Eq. (8.split): a diagonal part D (single-particle energies and the p = q pairing
term: phases and ZZ rotations) and one pair-hopping term h_pq per pair of
levels.  The eight Pauli strings of one h_pq commute, so exp(-i h_pq dt) is
exact and the only Trotter error is the non-commutativity between the terms.

* ``PairingSplit``: the terms in the N-particle sector, first- and second-order
  steps, the spectral-norm error of the product formula (Table 8.trottererror);
* the Trotterised refinement k = 2 -> 4 against the number of steps (Fig. 8.trotter, left);
* the rodeo filter with a Trotterised controlled evolution at fixed step size
  (Fig. 8.trotter, right; Table 8.trotterrodeo): too coarse a step is rejected by
  the filter itself.

About 15 s.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np
from models import PairingModel, add_term
from matrixfunctions import expm_hermitian
from refinement import pairing_refinement
from rodeo import RodeoFilter
from bookplot import savefig, plt


class PairingSplit:
    """H = D + sum_{p<q} h_pq in the N-particle sector of the k-level pairing model."""

    def __init__(self, k, N=4, g=1.0):
        pm = PairingModel(k, g=g)
        self.basis, self.H = pm.sector_hamiltonian(N)
        n = pm.n_modes
        self.hops = []
        for p in range(k):
            for q in range(p + 1, k):
                v = np.zeros((n, n, n, n))
                add_term(v, 2 * p, 2 * p + 1, 2 * q, 2 * q + 1, -g / 2)
                add_term(v, 2 * q, 2 * q + 1, 2 * p, 2 * p + 1, -g / 2)
                self.hops.append(self.basis.hamiltonian_matrix(np.zeros((n, n)), v))
        self.D = self.H - sum(self.hops)
        assert np.allclose(self.D, np.diag(np.diag(self.D)))            # D is diagonal
        self.d = np.diag(self.D).copy()
        self._eig = [np.linalg.eigh(h) for h in self.hops]

    def hop(self, j, tau):
        w, V = self._eig[j]
        return (V * np.exp(-1j * w * tau)) @ V.T

    def step(self, tau, order=2, diag=None, hop_scale=1.0):
        """One step of length tau.  ``diag`` replaces D (the refinement adds the coarse
        projector to it), ``hop_scale`` multiplies the hopping terms (sin^2 of the path)."""
        d = self.d if diag is None else diag
        J = range(len(self.hops))
        if order == 1:
            U = np.diag(np.exp(-1j * d * tau))
            for j in J:
                U = self.hop(j, hop_scale * tau) @ U
            return U
        half = np.diag(np.exp(-0.5j * d * tau))
        U = half.copy()
        for j in J:
            U = self.hop(j, 0.5 * hop_scale * tau) @ U
        for j in reversed(J):
            U = self.hop(j, 0.5 * hop_scale * tau) @ U
        return half @ U

    def evolution(self, dt, order=2):
        """t -> product formula for exp(-iHt) with ceil(|t|/dt) equal steps (fixed step size)."""
        def U(t):
            n = max(1, int(np.ceil(abs(t) / dt)))
            return np.linalg.matrix_power(self.step(t / n, order), n)
        return U


def trotter_stepper(split, rr, order):
    """Product-formula slice of the path Hamiltonian for a one-dimensional coarse space:
    cos^2 (E_low - mu)|Phi_0><Phi_0| is diagonal and joins D."""
    proj = np.abs(rr.phi0) ** 2                      # |Phi_0> is a basis determinant
    assert np.isclose(proj.max(), 1.0)

    def stepper(s, dt, psi):
        c, si = np.cos(0.5 * np.pi * s) ** 2, np.sin(0.5 * np.pi * s) ** 2
        diag = c * (rr.E_low - rr.mu) * proj + si * (split.d - rr.mu)
        return split.step(dt, order, diag=diag, hop_scale=si) @ psi
    return stepper


if __name__ == "__main__":
    split = PairingSplit(4)
    print("Spectral-norm error of the product formula for exp(-iH), t = 1 (k = 4, N = 4, g = 1)")
    exact = expm_hermitian(split.H, -1j)
    print("  steps   first order   second order")
    for n in (1, 2, 4, 8, 16):
        e1 = np.linalg.norm(np.linalg.matrix_power(split.step(1 / n, 1), n) - exact, 2)
        e2 = np.linalg.norm(np.linalg.matrix_power(split.step(1 / n, 2), n) - exact, 2)
        print(f"  {n:4d}    {e1:.2e}      {e2:.2e}")

    # ---- refinement with product formulas
    rr = pairing_refinement(2, 4, 4)
    T = 30
    p_exact = rr.overlap(rr.evolve(T, 160))
    print(f"\nTrotterised refinement k = 2 -> 4, T = {T}: exact slices give p = {p_exact:.4f}")
    print("   n_s   first order   second order")
    ns_list = [20, 40, 80, 160, 320, 640]
    inf = {1: [], 2: []}
    for ns in ns_list:
        for order in (1, 2):
            psi = rr.evolve(T, ns, stepper=trotter_stepper(split, rr, order))
            inf[order].append(1 - rr.overlap(psi))
        print(f"  {ns:4d}    {1 - inf[1][-1]:.4f}        {1 - inf[2][-1]:.4f}")
    refined = rr.evolve(T, 80, stepper=trotter_stepper(split, rr, 2))
    p_in = rr.overlap(refined)

    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    a.loglog(ns_list, inf[1], "s-", ms=4, label="first order")
    a.loglog(ns_list, inf[2], "o-", ms=4, label="second order")
    a.axhline(1 - p_exact, color="k", ls=":", lw=0.8)
    a.set_xlabel("refinement steps $n_s$"); a.set_ylabel("$1-p$"); a.legend()

    # ---- rodeo with a Trotterised controlled evolution
    M, sigma = 6, 4.0
    print(f"\nRodeo on the n_s = 80 second-order refined state (p = {p_in:.4f}), sigma = {sigma}, "
          f"mean steps per cycle = sigma sqrt(2/pi)/dt")
    print("    dt     steps/cycle   infidelity M = 2   M = 6     acceptance M = 6")
    exact_rf = RodeoFilter(split.H, sigma=sigma, rng=np.random.default_rng(5))
    inf_x, acc_x = exact_rf.expected(refined, rr.E_high, M)
    b.semilogy(range(1, M + 1), inf_x, "k-", label="exact evolution")
    print(f"   exact        -        {inf_x[1]:.2e}         {inf_x[5]:.2e}   {acc_x[5]:.4f}")
    for dt, col in ((1.0, "C3"), (0.5, "C1"), (0.25, "C0"), (0.125, "C2")):
        rf = RodeoFilter(split.H, sigma=sigma, rng=np.random.default_rng(5), evolution=split.evolution(dt))
        infid, acc = rf.prepare(refined, rr.E_high, M, rr.psi_high, n_sets=300)
        b.semilogy(range(1, M + 1), infid, col + "o-", ms=4, label=f"$\\delta t={dt}$")
        print(f"   {dt:5.3f}     {sigma * np.sqrt(2 / np.pi) / dt:6.1f}      {infid[1]:.2e}         {infid[5]:.2e}   {acc[5]:.4f}")
    b.set_xlabel("cycles $M$"); b.set_ylabel("$1-\\mathcal{F}_M$"); b.legend(fontsize=7)
    fig.tight_layout(); savefig(fig, 8, "trotter")

    # the floor: the filter projects on the ground state of the effective Hamiltonian
    print("\nGround state of the effective Hamiltonian of one second-order step against the true one")
    for dt in (1.0, 0.5, 0.25, 0.125):
        U = split.step(dt, 2)
        ph, V = np.linalg.eig(U)
        Eeff = -np.angle(ph) / dt
        j = np.argmax(np.abs(V.conj().T @ rr.psi_high))
        print(f"   dt = {dt:5.3f}: 1 - |<E0|E0_eff>|^2 = {1 - abs(V[:, j].conj() @ rr.psi_high) ** 2 / np.vdot(V[:, j], V[:, j]).real:.2e}, "
              f"E0_eff - E0 = {Eeff[j] - rr.E_high:+.2e}")
