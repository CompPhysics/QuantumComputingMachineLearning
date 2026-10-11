"""Chapter 8, Sec. 8.4.1: the coarse solver, a Trotterised UCCSD ansatz in a number sector.

The ansatz of Eq. (8.uccsd) is the product of exp(theta_mu (tau_mu - tau_mu^dagger))
over all spin-conserving single and double excitations of the Hartree-Fock
determinant, singles first.  Every generator A = tau - tau^dagger satisfies
A^3 = -A, so exp(theta A) = 1 + sin(theta) A + (1 - cos(theta)) A^2 is applied
to a vector of the N-particle sector without ever forming a matrix; the
gradient is exact, from one forward and one backward sweep through the
product (the two-term parameter-shift rule fails here, Sec. 7.4.2).  This is
what lets the same solver run from k = 2 (four qubits) to k = 8 (sixteen
qubits, sector dimension 1820).

``UCCSDSector``  pool, state, energy, gradient, BFGS optimisation from theta = 0
``pairing_references(k, N, g)``  FCI, CCD (chapter 3) and UCCSD energies

As a script: Table 8.uccsd and Fig. 8.uccsd.  About 5 s.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import itertools, time
import numpy as np
from fockspace import SlaterBasis
from models import PairingModel
from coupledcluster import CCD
from optimize import minimize
from bookplot import savefig, plt


class UCCSDSector:
    """Product-form UCCSD on the reference with the first N modes occupied (mode = 2 level + spin)."""

    def __init__(self, basis, H, singles=True):
        self.basis, self.H = basis, H
        n, N = basis.n, basis.N
        occ, virt = range(N), range(N, n)
        self.pool = []
        if singles:
            self.pool += [((i,), (a,)) for i in occ for a in virt if i % 2 == a % 2]
        self.pool += [((i, j), (a, b)) for i, j in itertools.combinations(occ, 2)
                      for a, b in itertools.combinations(virt, 2) if i % 2 + j % 2 == a % 2 + b % 2]
        self.n_singles = sum(1 for o, v in self.pool if len(o) == 1)
        self.generators = [self._generator(o, v) for o, v in self.pool]
        self.reference = basis.vector(list(occ))

    def _generator(self, occ, virt):
        """tau = prod a_a^dagger prod a_i as (source, destination, sign) index arrays."""
        ops = [("+", a) for a in virt] + [("-", i) for i in occ]
        src, dst, sgn = [], [], []
        for k, state in enumerate(self.basis.states):
            s, new = self.basis.apply(ops, state)
            if s:
                src.append(k); dst.append(self.basis.position[new]); sgn.append(s)
        return np.array(src), np.array(dst), np.array(sgn, dtype=float)

    @staticmethod
    def _A(gen, psi):
        src, dst, sgn = gen
        out = np.zeros_like(psi)
        out[dst] = sgn * psi[src]
        out[src] = -sgn * psi[dst]
        return out

    def _exp(self, gen, theta, psi):
        A1 = self._A(gen, psi)
        return psi + np.sin(theta) * A1 + (1 - np.cos(theta)) * self._A(gen, A1)

    def state(self, theta):
        psi = self.reference.copy()
        for gen, th in zip(self.generators, theta):
            psi = self._exp(gen, th, psi)
        return psi

    def energy(self, theta):
        psi = self.state(theta)
        return float(psi @ self.H @ psi)

    def gradient(self, theta):
        psi = self.state(theta)
        phi = self.H @ psi
        g = np.zeros(len(theta))
        for m in range(len(theta) - 1, -1, -1):
            gen = self.generators[m]
            g[m] = 2 * phi @ self._A(gen, psi)
            psi = self._exp(gen, -theta[m], psi)
            phi = self._exp(gen, -theta[m], phi)
        return g

    def optimise(self):
        theta, E, it = minimize(self.energy, np.zeros(len(self.pool)), grad=self.gradient, tol=1e-12, gtol=1e-7)
        return E, theta, it


def pairing_references(k, N=4, g=1.0):
    pm = PairingModel(k, g=g)
    basis, H = pm.sector_hamiltonian(N)
    w = np.linalg.eigvalsh(H)
    if k * 2 == N:                                   # the sector is a single determinant
        return dict(basis=basis, H=H, fci=w[0], ccd=w[0], uccsd=w[0], theta=np.zeros(0), solver=None)
    ccd = CCD(pm.h, pm.v, N)
    E_ccd = ccd.E_ref + ccd.solve()[0]
    solver = UCCSDSector(basis, H)
    E_u, theta, _ = solver.optimise()
    return dict(basis=basis, H=H, fci=w[0], ccd=E_ccd, uccsd=E_u, theta=theta, solver=solver)


if __name__ == "__main__":
    # gradient check
    pm = PairingModel(3, g=1.0)
    basis, H = pm.sector_hamiltonian(4)
    sol = UCCSDSector(basis, H)
    th = np.random.default_rng(0).normal(scale=0.3, size=len(sol.pool))
    num = np.array([(sol.energy(th + 1e-6 * e) - sol.energy(th - 1e-6 * e)) / 2e-6 for e in np.eye(len(th))])
    print(f"k = 3: {len(sol.pool)} operators ({sol.n_singles} singles); "
          f"analytic against numerical gradient: {np.max(np.abs(num - sol.gradient(th))):.1e}")
    assert np.max(np.abs(num - sol.gradient(th))) < 1e-7

    print("\nGround-state energies, N = 4, g = 1")
    print("  k  qubits  dim    FCI        CCD        UCCSD      UCCSD-FCI   CCD-FCI   #par  max|singles|   s")
    rows = []
    for k in (2, 3, 4, 5, 6, 7, 8):
        t0 = time.time()
        r = pairing_references(k)
        ns = r["solver"].n_singles if r["solver"] else 0
        smax = np.max(np.abs(r["theta"][:ns])) if ns else 0.0
        rows.append((k, r["fci"], r["ccd"], r["uccsd"]))
        print(f"  {k}  {2 * k:4d}  {r['basis'].dim:5d}  {r['fci']:.6f}  {r['ccd']:.6f}  {r['uccsd']:.6f}  "
              f"{r['uccsd'] - r['fci']:+.2e}  {r['ccd'] - r['fci']:+.2e}  {len(r['theta']):4d}  {smax:.1e}  {time.time() - t0:5.1f}")
        assert r["uccsd"] >= r["fci"] - 1e-10          # variational
    rows = np.array(rows)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for col, lab, mk in ((1, "FCI", "k-"), (2, "CCD", "s--"), (3, "UCCSD", "o-")):
        a.plot(rows[:, 0], rows[:, col], mk, ms=4, label=lab)
    a.set_xlabel("levels $k$"); a.set_ylabel("$E_0$"); a.legend()
    b.semilogy(rows[1:, 0], np.abs(rows[1:, 2] - rows[1:, 1]), "s--", ms=4, label="$|E_{\\rm CCD}-E_{\\rm FCI}|$")
    b.semilogy(rows[1:, 0], np.abs(rows[1:, 3] - rows[1:, 1]) + 1e-16, "o-", ms=4, label="$E_{\\rm UCCSD}-E_{\\rm FCI}$")
    b.set_xlabel("levels $k$"); b.legend()
    fig.tight_layout(); savefig(fig, 8, "uccsd")
