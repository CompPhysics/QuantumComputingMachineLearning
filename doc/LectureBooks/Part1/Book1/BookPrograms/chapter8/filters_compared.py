"""Chapter 8, Sec. 8.5: the rodeo filter against phase estimation and VQE on the pairing model.

Phase estimation with m control qubits is also a filter: reading the outcome k
multiplies the component |E_j> of the input by the Dirichlet amplitude
a_k(phi_j) of Eq. (6.qpeamplitude).  Both algorithms are therefore compared as
state preparation, at equal total evolution time of the controlled
exp(-iHt), on the same input (the prolonged k = 4 pairing state, p = 0.868,
and the refined one, p = 0.9988):

* infidelity and acceptance of phase estimation (modal outcome) for m = 2..10
  and of the rodeo algorithm for M = 1..12 (Table 8.qpevsrodeo, Fig. 8.qpevsrodeo);
* the energy read off by both;
* the UCCSD state of Table 8.uccsd as a third preparation: its energy error and
  its infidelity, and what two rodeo cycles do to it (Table 8.vqepolish).

About 5 s.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np
from phase_estimation import dirichlet_probability
from refinement import pairing_refinement
from rodeo import RodeoFilter, infidelity_bound
from uccsd_sector import pairing_references
from bookplot import savefig, plt


def dirichlet_amplitude(phi, m, k):
    """a_k(phi) = 2^-m sum_y exp(2 pi i y (phi - k/2^m)), Eq. (6.qpeamplitude)."""
    y = np.arange(2 ** m)
    return np.exp(2j * np.pi * np.outer(np.atleast_1d(phi) - k / 2 ** m, y)).sum(axis=1) / 2 ** m


class PhaseEstimationFilter:
    """Phase estimation of U = exp(-i (H - shift) t) seen as a filter on the system register."""

    def __init__(self, H, safety=0.9):
        self.w, self.V = np.linalg.eigh(H)
        self.shift = 0.5 * (self.w[0] + self.w[-1])              # centre the spectrum
        self.t = safety * np.pi / (0.5 * (self.w[-1] - self.w[0]))
        self.phi = (-(self.w - self.shift) * self.t / (2 * np.pi)) % 1.0

    def filter(self, psi, m, j0=0):
        """Accept the grid point nearest to the phase of level j0.  Returns
        (infidelity with level j0, acceptance, energy read off, total evolution time)."""
        c = self.V.conj().T @ psi
        k = int(np.round(self.phi[j0] * 2 ** m)) % 2 ** m
        a = dirichlet_amplitude(self.phi, m, k)
        out = c * a
        P = float(np.sum(np.abs(out) ** 2))
        signed = k / 2 ** m if k < 2 ** (m - 1) else k / 2 ** m - 1
        E = self.shift - 2 * np.pi * signed / self.t
        return 1 - abs(out[j0]) ** 2 / P, P, E, self.t * (2 ** m - 1)


if __name__ == "__main__":
    rr = pairing_refinement(2, 4, 4)
    refined = rr.evolve(30)
    H, E0, ground = rr.H_high, rr.E_high, rr.psi_high
    qpe = PhaseEstimationFilter(H)
    print(f"Sector spectrum from {qpe.w[0]:.4f} to {qpe.w[-1]:.4f}; time step t = {qpe.t:.4f}")
    # consistency with chapter 6: |a_k|^2 is the Dirichlet probability
    assert np.allclose(np.abs(dirichlet_amplitude(0.3, 5, 9)) ** 2, dirichlet_probability(0.3, 5, 9))

    sigma = 4.0
    rf = RodeoFilter(H, sigma=sigma)
    mean_t = sigma * np.sqrt(2 / np.pi)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for ax, (psi, lab) in zip(axes, ((rr.phi0, "prolonged"), (refined, "refined"))):
        p = rr.overlap(psi)
        print(f"\nInput: {lab} state, p = {p:.4f}")
        print("   phase estimation                                        rodeo, sigma = 4")
        print("    m   total time   infidelity   acceptance   E - E0         M   total time   infidelity   acceptance")
        ms, Ms = range(2, 11), np.arange(1, 13)
        q = np.array([qpe.filter(psi, m) for m in ms])
        inf_r, acc_r = rf.expected(psi, E0, 12)
        for i in range(max(len(ms), len(Ms))):
            left = (f"   {ms[i]:2d}   {q[i, 3]:9.1f}    {q[i, 0]:.2e}     {q[i, 1]:.4f}     {q[i, 2] - E0:+.1e}"
                    if i < len(ms) else " " * 60)
            print(left + f"      {Ms[i]:2d}   {Ms[i] * mean_t:9.1f}    {inf_r[i]:.2e}     {acc_r[i]:.4f}")
        ax.loglog(q[:, 3], q[:, 0], "s-", ms=4, label="phase estimation, modal outcome")
        ax.loglog(Ms * mean_t, inf_r, "o-", ms=4, label="rodeo, Gaussian times")
        ax.set_xlabel("total evolution time"); ax.set_ylabel("infidelity of accepted runs")
        ax.set_title(f"{lab} input, $p={p:.3f}$"); ax.legend(fontsize=8)
    fig.tight_layout(); savefig(fig, 8, "qpevsrodeo")

    # ---- the variational state as the input
    ref = pairing_references(4)
    psi_u = ref["solver"].state(ref["theta"])
    p_u = rr.overlap(psi_u)
    print(f"\nUCCSD at k = 4: E - E_FCI = {ref['uccsd'] - E0:.2e}, infidelity 1 - p = {1 - p_u:.2e}")
    inf_u, acc_u = rf.expected(psi_u, E0, 6)
    print("   rodeo cycles on the UCCSD state:  M   infidelity   acceptance   energy error of the filtered state")
    for M in (1, 2, 4, 6):
        # energy of the mean accepted state: weights |c_j|^2 f_j^M
        c2 = np.abs(rf.V.T @ psi_u) ** 2
        f = 0.5 * (1 + np.exp(-0.5 * ((rf.w - E0) * sigma) ** 2))
        wts = c2 * f ** M; wts /= wts.sum()
        print(f"                                     {M}   {inf_u[M - 1]:.2e}     {acc_u[M - 1]:.4f}       {wts @ rf.w - E0:.2e}")
