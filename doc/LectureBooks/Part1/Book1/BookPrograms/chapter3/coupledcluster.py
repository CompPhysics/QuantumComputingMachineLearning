"""Chapter 3, Secs. 3.7-3.8: coupled-cluster theory, standard and unitary.

``CCD`` solves the coupled-cluster doubles equations (Eq. 3.ccd) for any
Hamiltonian given as (h, v) in a spin-orbital basis with the first N
orbitals occupied, by iteration from the MP2 amplitudes, and evaluates the
correlation energy (Eq. 3.ccenergy).  ``UnitaryCC`` builds the anti-Hermitian
generator sigma = sum_k t_k (A_k - A_k^dagger) of Eq. (3.uccgenerators) as a
matrix in Fock space, evaluates the variational energy (Eq. 3.uccenergy),
minimises it, and Trotterises the exponential (Eq. 3.ucctrotter).

Applied to the pairing model with four levels and two pairs: Table 3.ccd
(MP2, CCD, FCI), Table 3.uccd (UCCD above the exact energy, CCD below),
the generator-redundancy table, the re-optimised Trotter table and the
frozen-amplitude Trotter table (Table 3.ucctrotter); Fig. 3.cc.  Runs in a few seconds.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import itertools
import numpy as np
from fockspace import FockSpace
from models import PairingModel
from matrixfunctions import expm_antihermitian
from optimize import minimize
from bookplot import savefig, plt


class CCD:
    """Coupled-cluster doubles in a spin-orbital basis; the first N orbitals are occupied."""

    def __init__(self, h, v, N):
        self.h, self.v, self.N = np.asarray(h, float), np.asarray(v, float), N
        self.n = self.h.shape[0]
        o, u = slice(0, N), slice(N, self.n)
        self.f = self.h + np.einsum("piqi->pq", self.v[:, o, :, o])       # Fock matrix
        self.E_ref = np.trace(self.h[o, o]) + 0.5 * np.einsum("ijij->", self.v[o, o, o, o])
        self.oooo = self.v[o, o, o, o]
        self.vvvv = self.v[u, u, u, u]
        self.oovv = self.v[o, o, u, u]      # <ij||ab>
        self.vvoo = self.v[u, u, o, o]      # <ab||ij>
        self.ovvo = self.v[o, u, u, o]      # <kb||cj>
        eo, ev = np.diag(self.f)[o], np.diag(self.f)[u]
        self.D = eo[:, None, None, None] + eo[None, :, None, None] - ev[None, None, :, None] - ev[None, None, None, :]
        self.fo = self.f[o, o] - np.diag(np.diag(self.f[o, o]))          # off-diagonal parts
        self.fv = self.f[u, u] - np.diag(np.diag(self.f[u, u]))

    def energy(self, t):
        return 0.25 * np.einsum("ijab,ijab->", self.oovv, t)

    def residual(self, t):
        """The right-hand side of Eq. (3.ccd); the new amplitudes are residual / D."""
        v, P = self, lambda X: X - X.transpose(1, 0, 2, 3) - X.transpose(0, 1, 3, 2) + X.transpose(1, 0, 3, 2)
        R = np.einsum("abij->ijab", v.vvoo).copy()
        X = np.einsum("bc,ijac->ijab", v.fv, t); R += X - X.transpose(0, 1, 3, 2)
        X = np.einsum("kj,ikab->ijab", v.fo, t); R -= X - X.transpose(1, 0, 2, 3)
        R += 0.5 * np.einsum("abcd,ijcd->ijab", v.vvvv, t)
        R += 0.5 * np.einsum("klij,klab->ijab", v.oooo, t)
        R += P(np.einsum("kbcj,ikac->ijab", v.ovvo, t))
        R += 0.25 * np.einsum("klcd,ijcd,klab->ijab", v.oovv, t, t)
        R += 0.5 * P(np.einsum("klcd,ikac,jlbd->ijab", v.oovv, t, t))
        X = np.einsum("klcd,jlcd,ikab->ijab", v.oovv, t, t); R -= 0.5 * (X - X.transpose(1, 0, 2, 3))
        X = np.einsum("klcd,klbd,ijac->ijab", v.oovv, t, t); R -= 0.5 * (X - X.transpose(0, 1, 3, 2))
        return R

    def solve(self, tol=1e-12, maxiter=500, mixing=0.0):
        t = np.einsum("abij->ijab", self.vvoo) / self.D          # MP2 amplitudes
        E_mp2 = self.energy(t)
        E_old = E_mp2
        for it in range(1, maxiter + 1):
            t_new = self.residual(t) / self.D
            t = (1 - mixing) * t_new + mixing * t
            E = self.energy(t)
            if abs(E - E_old) < tol:
                break
            E_old = E
        return E, t, it, E_mp2


class UnitaryCC:
    """|Psi(t)> = exp(sum_k t_k (A_k - A_k^dagger)) |Phi_0> as matrices in Fock space."""

    def __init__(self, H, generators, reference, sector=None):
        """`sector`: indices of the particle-number block everything is restricted to."""
        idx = np.arange(len(reference)) if sector is None else np.asarray(sector)
        self.H = np.asarray(H)[np.ix_(idx, idx)]
        self.A = [np.asarray(A)[np.ix_(idx, idx)] for A in generators]
        self.ref = np.asarray(reference, dtype=complex)[idx]
        self.kappa = [A - A.T for A in self.A]                 # anti-Hermitian, real A

    def sigma(self, t):
        return sum(tk * k for tk, k in zip(t, self.kappa))

    def state(self, t):
        return expm_antihermitian(self.sigma(t)) @ self.ref

    def energy(self, t):
        psi = self.state(t)
        return float(np.real(np.vdot(psi, self.H @ psi)))

    def trotter_state(self, t, n):
        U = np.eye(len(self.ref), dtype=complex)
        for tk, k in zip(t, self.kappa):
            U = U @ expm_antihermitian(tk * k / n)
        return np.linalg.matrix_power(U, n) @ self.ref

    def trotter_energy(self, t, n):
        psi = self.trotter_state(t, n)
        return float(np.real(np.vdot(psi, self.H @ psi)))

    def optimise(self, t0=None, energy=None):
        energy = self.energy if energy is None else energy
        t0 = np.zeros(len(self.A)) if t0 is None else t0
        t, E, it = minimize(energy, t0, tol=1e-13, gtol=1e-9)
        return t, E, it


def pairing_generators(pm, n_pairs, fs, kind="pairs"):
    """Excitation operators for the pairing model: pair moves, all doubles, or singles + doubles."""
    occ = list(range(2 * n_pairs)); vir = list(range(2 * n_pairs, pm.n_modes))
    gens = []
    if kind == "pairs":
        for i in range(n_pairs):
            for a in range(n_pairs, pm.L):
                gens.append(fs.create(2 * a) @ fs.create(2 * a + 1) @ fs.annihilate(2 * i + 1) @ fs.annihilate(2 * i))
        return gens
    if kind == "singles+doubles":
        for i in occ:
            for a in vir:
                gens.append(fs.create(a) @ fs.annihilate(i))
    for i, j in itertools.combinations(occ, 2):
        for a, b in itertools.combinations(vir, 2):
            gens.append(fs.create(a) @ fs.create(b) @ fs.annihilate(j) @ fs.annihilate(i))
    return gens


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    L, n_pairs, N = 4, 2, 4
    fs = FockSpace(2 * L)
    ref = fs.determinant(PairingModel(L).reference(n_pairs))
    sec = fs.sector(N)                       # the 70 four-particle states

    print("=== CCD for the pairing model, L = 4, two pairs (Table 3.ccd)")
    print(f"{'g':>5s} {'E_ref':>8s} {'MP2':>12s} {'CCD':>12s} {'FCI':>12s} {'CCD error':>10s} {'iter':>5s}")
    ccd_rows = []
    for g in (0.25, 0.5, 1.0, 1.5, 2.0):
        pm = PairingModel(L, g=g)
        cc = CCD(pm.h, pm.v, N)
        dE, t, it, Emp2 = cc.solve()
        Eccd = cc.E_ref + dE
        Efci = pm.exact(N)[0]
        ccd_rows.append((g, Eccd, Efci))
        print(f"{g:5.2f} {cc.E_ref:8.5f} {cc.E_ref + Emp2:12.8f} {Eccd:12.8f} {Efci:12.8f} {Eccd - Efci:10.1e} {it:5d}")
    assert abs(ccd_rows[2][1] - 0.63044275) < 1e-7

    print("\n=== Unitary coupled cluster, g = 1: generator redundancy")
    pm = PairingModel(L, g=1.0)
    H = fs.hamiltonian(pm.h, pm.v)
    for kind in ("pairs", "doubles", "singles+doubles"):
        gens = pairing_generators(pm, n_pairs, fs, kind)
        ucc = UnitaryCC(H, gens, ref, sec)
        t, E, it = ucc.optimise()
        print(f"  {kind:17s}: {len(gens):3d} amplitudes, E(UCC) = {E:.10f} ({it} BFGS iterations)")
        if kind == "pairs":
            E_pairs, t_pairs = E, t
    assert abs(E_pairs - 0.6369869810) < 1e-8

    print("\n=== UCCD against CCD and FCI (Table 3.uccd)")
    print(f"{'g':>5s} {'CCD':>12s} {'UCCD':>12s} {'FCI':>12s} {'CCD error':>10s} {'UCCD error':>11s}")
    ucc_rows = []
    for (g, Eccd, Efci) in ccd_rows:
        pm = PairingModel(L, g=g)
        ucc = UnitaryCC(fs.hamiltonian(pm.h, pm.v), pairing_generators(pm, n_pairs, fs), ref, sec)
        t, Eucc, _ = ucc.optimise()
        ucc_rows.append((g, Eccd, Eucc, Efci, t))
        print(f"{g:5.2f} {Eccd:12.8f} {Eucc:12.8f} {Efci:12.8f} {Eccd - Efci:10.1e} {Eucc - Efci:11.1e}")
        assert Eucc >= Efci - 1e-10

    print("\n=== Trotterised UCCD, g = 1: re-optimised amplitudes at each n")
    pm = PairingModel(L, g=1.0)
    H = fs.hamiltonian(pm.h, pm.v)
    ucc = UnitaryCC(H, pairing_generators(pm, n_pairs, fs), ref, sec)
    Efci = pm.exact(N)[0]
    print(f"  exact exponential: E = {E_pairs:.10f}, error {E_pairs - Efci:.2e}")
    for n in (1, 2, 4, 8):
        t, E, _ = ucc.optimise(t0=t_pairs, energy=lambda tt, n=n: ucc.trotter_energy(tt, n))
        print(f"  n = {n}: E = {E:.10f}, error {E - Efci:.2e}")

    print("\n=== Trotterised UCCD with the amplitudes frozen (Table 3.ucctrotter)")
    psi_exact = ucc.state(t_pairs)
    print(f"{'n':>4s} {'||psi_n - psi||':>15s} {'ratio':>6s} {'energy error':>13s} {'ratio':>6s}")
    old = None
    tr = []
    for n in (1, 2, 4, 8, 16, 32, 64):
        psi_n = ucc.trotter_state(t_pairs, n)
        dpsi = np.linalg.norm(psi_n - psi_exact)
        dE = ucc.trotter_energy(t_pairs, n) - E_pairs
        tr.append((n, dpsi, dE))
        r = f"{old[0]/dpsi:6.2f} {dE:13.2e} {old[1]/dE:6.2f}" if old else f"   --- {dE:13.2e}    ---"
        print(f"{n:4d} {dpsi:15.2e} {r}")
        old = (dpsi, dE)

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.3))
    gs = np.linspace(0.1, 2.2, 15)
    errs = []
    for g in gs:
        pm = PairingModel(L, g=g)
        cc = CCD(pm.h, pm.v, N); dE, t, it, _ = cc.solve()
        ucc = UnitaryCC(fs.hamiltonian(pm.h, pm.v), pairing_generators(pm, n_pairs, fs), ref, sec)
        tt, Eu, _ = ucc.optimise()
        Ef = pm.exact(N)[0]
        errs.append((cc.E_ref + dE - Ef, Eu - Ef))
    errs = np.array(errs)
    axes[0].plot(gs, errs[:, 0], "o-", label="CCD $-$ FCI")
    axes[0].plot(gs, errs[:, 1], "s-", label="UCCD $-$ FCI")
    axes[0].axhline(0, color="k", lw=0.6); axes[0].set_xlabel("$g$"); axes[0].set_ylabel("energy error")
    axes[0].legend(fontsize=8); axes[0].set_title("pairing model, $L=4$, two pairs", fontsize=9)
    ns = np.array([r[0] for r in tr])
    axes[1].loglog(ns, [r[1] for r in tr], "o-", label=r"$\|\Psi_n-\Psi\|$")
    axes[1].loglog(ns, [r[2] for r in tr], "s-", label="$E_n - E$")
    axes[1].loglog(ns, tr[0][1] / ns, "k:", lw=0.8); axes[1].loglog(ns, tr[0][2] / ns ** 2, "k:", lw=0.8)
    axes[1].set_xlabel("Trotter number $n$"); axes[1].legend(fontsize=8)
    axes[1].set_title("frozen-amplitude Trotter error, $g=1$", fontsize=9)
    print("figure written to", savefig(fig, 3, "coupled_cluster"))
