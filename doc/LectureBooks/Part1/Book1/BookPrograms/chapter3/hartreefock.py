"""Chapter 3, Sec. 3.5: Hartree-Fock theory.

The class HartreeFock solves the self-consistent field equations
(Eqs. 3.hfequations, 3.fockdensity) for any Hamiltonian given as (h, v) in a
spin-orbital basis, by repeated diagonalisation of the Fock matrix built
from the one-body density matrix.  It also evaluates the stability matrix
(Eq. 3.stabilitymatrix) and performs Thouless rotations (Eq. 3.thouless).

Applications in the text: the Lipkin model, where the spherical solution
turns into a saddle point at chi = 1 and a deformed solution appears
(Table 3.lipkinhf, Fig. 3.lipkinhf); the pairing model, where the mean field
does nothing (Table 3.pairinghf); and the Hubbard ring, where a
symmetry-broken (antiferromagnetic) mean field lowers the energy at large
U/t (Table 3.hubbardhf).  Runs in a few seconds.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np
from fockspace import FockSpace
from models import LipkinModel, PairingModel, HubbardModel
from matrixfunctions import expm
from bookplot import savefig, plt


class HartreeFock:
    """Self-consistent field for H = sum h_pq a+_p a_q + 1/4 sum v_pqrs a+a+aa."""

    def __init__(self, h, v, n_particles):
        self.h, self.v, self.N = np.asarray(h, dtype=float), np.asarray(v, dtype=float), n_particles
        self.n = self.h.shape[0]

    def density_matrix(self, C):
        """rho_{gamma delta} = sum_i C_{gamma i} C*_{delta i}, columns of C = orbitals."""
        occ = C[:, :self.N]
        return occ @ occ.conj().T

    def fock_matrix(self, rho):
        """f_{alpha beta} = h_{alpha beta} + sum_{gamma delta} rho_{delta gamma} <alpha gamma|v|beta delta>."""
        return self.h + np.einsum("dg,agbd->ab", rho, self.v)

    def energy(self, rho):
        """E = sum h rho + 1/2 sum <ag|v|bd> rho_ba rho_dg."""
        return float(np.einsum("ab,ba->", self.h, rho)
                     + 0.5 * np.einsum("agbd,ba,dg->", self.v, rho, rho))

    def solve(self, C0=None, tol=1e-10, maxiter=500, mixing=0.0):
        """Iterate to self-consistency; returns orbitals, energies, HF energy, iterations."""
        C = np.eye(self.n) if C0 is None else np.asarray(C0, dtype=float)
        eps_old = np.zeros(self.n)
        for it in range(1, maxiter + 1):
            rho = self.density_matrix(C)
            f = self.fock_matrix(rho)
            eps, Cnew = np.linalg.eigh(f)
            if mixing:
                rho_new = self.density_matrix(Cnew)
                rho = (1 - mixing) * rho_new + mixing * rho
                # re-diagonalise the mixed Fock matrix to keep orbitals orthonormal
                eps, Cnew = np.linalg.eigh(self.fock_matrix(rho))
            C = Cnew
            if np.abs(eps - eps_old).mean() < tol:
                break
            eps_old = eps
        rho = self.density_matrix(C)
        return C, eps, self.energy(rho), it

    def brillouin(self, C):
        """max |f_ai| in the HF basis: zero at self-consistency."""
        rho = self.density_matrix(C)
        f = C.T @ self.fock_matrix(rho) @ C
        return np.abs(f[self.N:, :self.N]).max()

    def stability_matrix(self, C):
        """The matrix M of Eq. (3.stabilitymatrix) in the HF basis of the columns of C."""
        rho = self.density_matrix(C)
        f = C.T @ self.fock_matrix(rho) @ C
        eps = np.diag(f)
        vhf = np.einsum("ap,bq,cr,ds,abcd->pqrs", C, C, C, C, self.v)   # transformed elements
        occ, vir = range(self.N), range(self.N, self.n)
        pairs = [(a, i) for a in vir for i in occ]
        m = len(pairs)
        A = np.zeros((m, m)); B = np.zeros((m, m))
        for x, (a, i) in enumerate(pairs):
            for y, (b, j) in enumerate(pairs):
                A[x, y] = (eps[a] - eps[i]) * (a == b) * (i == j) - vhf[a, j, b, i]
                B[x, y] = vhf[a, b, i, j]
        return np.block([[A, B], [B, A]])

    def thouless_state(self, C, amplitudes, fs):
        """exp(sum_ai t_ai a+_a a_i)|Phi_0> in the Fock space of `fs`, HF orbitals = columns of C."""
        # rotate the operators to the HF basis: a+_p(HF) = sum_alpha C_{alpha p} a+_alpha
        cr = [sum(C[al, p] * fs.create(al) for al in range(self.n)) for p in range(self.n)]
        an = [m.T for m in cr]
        T = sum(amplitudes[a - self.N, i] * cr[a] @ an[i] for a in range(self.N, self.n) for i in range(self.N))
        ref = fs.vacuum()
        for i in reversed(range(self.N)):
            ref = cr[i] @ ref
        return (expm(T) @ ref).real, ref


def lipkin_mean_field_energy(alpha, N, eps, chi):
    """E(alpha)/N = -eps/2 cos(alpha) - eps chi/4 sin^2(alpha), Eq. (3.lipkinmf)."""
    return -eps / 2 * np.cos(alpha) - eps * chi / 4 * np.sin(alpha) ** 2


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True, linewidth=120)
    rng = np.random.default_rng(3)

    # ------------------------------------------------------------ Lipkin
    print("=== Lipkin model, N = 4, eps = 1, W = 0 (Table 3.lipkinhf)")
    print(f"{'chi':>5s} {'E_spherical':>12s} {'E_HF':>12s} {'E_mf formula':>13s} {'lambda_min':>11s} {'E_exact':>10s}  solution")
    N, eps = 4, 1.0
    lip_rows = []
    for chi in (0.25, 0.5, 0.9, 1.0, 1.5, 2.0, 2.5):
        V = -chi * eps / (N - 1)
        lip = LipkinModel(N, eps=eps, V=V, W=0.0)
        hf = HartreeFock(lip.h, lip.v, N)
        # the spherical solution: start from the level-ordered basis (lower level first)
        order = np.argsort(np.diag(lip.h), kind="stable")
        C_sph = np.eye(lip.n_modes)[:, order]
        Csph, epss, Esph, _ = hf.solve(C_sph)
        lam_min = np.linalg.eigvalsh(hf.stability_matrix(Csph))[0]
        # a symmetry-broken start: rotate every doublet by the same random angle
        best = Esph
        for trial in range(6):
            Q, _ = np.linalg.qr(rng.normal(size=(lip.n_modes, lip.n_modes)))
            C, e, E, it = hf.solve(Q, mixing=0.3)
            best = min(best, E)
        alpha = np.linspace(0, np.pi, 2001)
        Emf = N * lipkin_mean_field_energy(alpha, N, eps, chi).min()
        Eex = np.linalg.eigvalsh(lip.quasispin_matrix())[0]
        kind = "spherical" if chi < 1 else ("critical" if chi == 1 else "deformed")
        lip_rows.append((chi, Esph, best, Emf, lam_min, Eex))
        print(f"{chi:5.2f} {Esph:12.6f} {best:12.6f} {Emf:13.6f} {lam_min:11.4f} {Eex:10.6f}  {kind}")
        assert abs(Emf - best) < 1e-6 and abs(lam_min - eps * (1 - chi)) < 1e-8

    # ----------------------------------------------------------- pairing
    print("\n=== Pairing model, L = 4, N = 4: the mean field does nothing (Table 3.pairinghf)")
    print(f"{'g':>5s} {'E_ref':>11s} {'E_HF':>11s} {'E_exact':>11s} {'iterations':>10s} {'max|f_ai|':>10s}")
    for g in (0.25, 0.5, 1.0):
        pm = PairingModel(4, g=g)
        hf = HartreeFock(pm.h, pm.v, 4)
        C, e, E, it = hf.solve()
        Eex = pm.exact(4)[0]
        print(f"{g:5.2f} {2 - g:11.8f} {E:11.8f} {Eex:11.8f} {it:10d} {hf.brillouin(C):10.1e}")
        assert abs(E - (2 - g)) < 1e-10

    # ------------------------------------------------------------ Hubbard
    print("\n=== Half-filled Hubbard ring, 4 sites: restricted vs unrestricted mean field (Table 3.hubbardhf)")
    print(f"{'U/t':>4s} {'E_para':>10s} {'E_UHF':>10s} {'E_exact':>10s} {'m_stagger':>10s}")
    for U in (1, 2, 4, 8):
        hub = HubbardModel(4, t=1.0, U=float(U))
        hf = HartreeFock(hub.h, hub.v, 4)
        # the uniform (paramagnetic) determinant: plane waves, E = -4t + U N/4
        Erhf = -4.0 + U
        # UHF: start from a staggered spin density
        stag = np.zeros((8, 8))
        for i in range(4):
            stag[2 * i, 2 * i] = -0.5 * (-1) ** i; stag[2 * i + 1, 2 * i + 1] = 0.5 * (-1) ** i
        e0, Cs = np.linalg.eigh(hub.h + stag)
        Cuhf, e, Euhf, _ = hf.solve(Cs, mixing=0.2)
        rho = hf.density_matrix(Cuhf)
        m = np.mean([(-1) ** i * (rho[2 * i, 2 * i] - rho[2 * i + 1, 2 * i + 1]) / 2 for i in range(4)])
        Eex = hub.exact(4)[0]
        print(f"{U:4d} {Erhf:10.6f} {Euhf:10.6f} {Eex:10.6f} {abs(m):10.4f}")

    # ------------------------------------------------------------ Thouless
    print("\n=== Thouless' theorem: the rotated state is a single determinant")
    pm = PairingModel(3, g=0.5)
    hf = HartreeFock(pm.h, pm.v, 2)
    C = np.eye(6)
    fs = FockSpace(6)
    t = 0.3 * rng.normal(size=(4, 2))
    psi, ref = hf.thouless_state(C, t, fs)
    # the same state from the rotated creation operators b+_i = a+_i + sum_a t_ai a+_a
    b = [fs.create(i) + sum(t[a - 2, i] * fs.create(a) for a in range(2, 6)) for i in range(2)]
    det = b[0] @ b[1] @ fs.vacuum()
    assert np.allclose(psi, det)
    print(f"  ||exp(T)|c> - prod_i b_i^dagger|0>|| = {np.linalg.norm(psi - det):.1e}, <c|c'> = {ref @ psi:.6f}")
    lin = ref + sum(t[a - 2, i] * fs.create(a) @ fs.annihilate(i) @ ref for a in range(2, 6) for i in range(2))
    print(f"  ||exp(T)|c> - (1+T)|c>|| = {np.linalg.norm(psi - lin):.3e} (the 2p-2h cross terms)")

    # ------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.3))
    alpha = np.linspace(-np.pi, np.pi, 400)
    for chi in (0.5, 1.0, 1.5, 2.5):
        axes[0].plot(alpha, lipkin_mean_field_energy(alpha, N, eps, chi), label=f"$\\chi={chi}$")
    axes[0].set_xlabel(r"$\alpha$"); axes[0].set_ylabel(r"$E(\alpha)/N\varepsilon$"); axes[0].legend(fontsize=8)
    axes[0].set_title("Lipkin mean-field energy surface", fontsize=9)
    chis = np.linspace(0.05, 2.5, 40)
    lam = []
    for chi in chis:
        lip = LipkinModel(N, eps=eps, V=-chi * eps / (N - 1))
        hf = HartreeFock(lip.h, lip.v, N)
        Csph = np.eye(lip.n_modes)[:, np.argsort(np.diag(lip.h), kind="stable")]
        lam.append(np.linalg.eigvalsh(hf.stability_matrix(Csph))[0])
    axes[1].plot(chis, lam, label=r"$\lambda_{\min}(M)$")
    axes[1].plot([r[0] for r in lip_rows], [r[2] - r[5] for r in lip_rows], "s--", label=r"$E_{\rm HF}-E_{\rm exact}$")
    axes[1].axhline(0, color="k", lw=0.5); axes[1].set_xlabel(r"$\chi$"); axes[1].legend(fontsize=8)
    axes[1].set_title("stability of the spherical solution", fontsize=9)
    print("figure written to", savefig(fig, 3, "lipkin_hf"))
