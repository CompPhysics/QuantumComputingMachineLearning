"""Chapter 3, Secs. 3.3-3.4: exact diagonalisation of the model Hamiltonians.

Produces every table of the model and full-CI sections: the Lipkin spectrum
in the quasispin block against the 70-dimensional determinant space
(Table 3.lipkin), the pairing model in the seniority-zero and full spaces
(Table 3.pairing), truncated CI for the pairing model (Table 3.citrunc), the
size-consistency failure of CID (Table 3.sizecons), the four-site Hubbard
ring against U/t and its Heisenberg limit (Tables 3.hubbard, 3.hubbardstrong),
the Heisenberg ring against the Bethe-ansatz limit (Table 3.heisenberg),
the growth of the Hilbert space (Table 3.wall) and the convergence of the
Lanczos iteration (Table 3.lanczos).  Fig. 3.models collects the scans.
Runs in a few seconds (the N = 14 Heisenberg ring dominates).
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import itertools
from math import comb, sqrt, pi, log
import numpy as np
from fockspace import SlaterBasis
from models import LipkinModel, PairingModel, HubbardModel, HeisenbergModel
from bookplot import savefig, plt


def truncated_ci(model, n_particles, reference, max_level):
    """Lowest eigenvalue in the space of determinants at most `max_level`
    excitations away from `reference` (a list of occupied modes)."""
    basis, H = model.sector_hamiltonian(n_particles)
    ref = sum(basis.bit(p) for p in reference)
    keep = [k for k, s in enumerate(basis.states) if basis.excitation_level(s, ref) <= max_level]
    return np.linalg.eigvalsh(H[np.ix_(keep, keep)])[0], len(keep)


def lanczos(H, v0, m):
    """m steps of the Lanczos iteration from v0 with full reorthogonalisation;
    returns the Ritz values (eigenvalues of the m x m tridiagonal matrix)."""
    n = len(v0)
    V = np.zeros((m, n)); alpha = np.zeros(m); beta = np.zeros(m)
    v = v0 / np.linalg.norm(v0)
    for k in range(m):
        V[k] = v
        w = H @ v
        alpha[k] = v @ w
        w -= alpha[k] * v + (beta[k - 1] * V[k - 1] if k else 0)
        w -= V[:k + 1].T @ (V[:k + 1] @ w)          # reorthogonalise
        beta[k] = np.linalg.norm(w)
        if beta[k] < 1e-12:
            return np.linalg.eigvalsh(np.diag(alpha[:k + 1]) + np.diag(beta[:k], 1) + np.diag(beta[:k], -1))
        v = w / beta[k]
    return np.linalg.eigvalsh(np.diag(alpha) + np.diag(beta[:-1], 1) + np.diag(beta[:-1], -1))


class NonInteractingCopies:
    """m copies of a two-level, one-pair pairing system that do not interact."""

    def __init__(self, m, g):
        self.m, self.g = m, g
        self.n_modes = 4 * m
        n = self.n_modes
        single = PairingModel(2, g=g)
        self.h = np.zeros((n, n)); self.v = np.zeros((n, n, n, n))
        for c in range(m):
            sl = slice(4 * c, 4 * c + 4)
            self.h[sl, sl] = single.h
            self.v[sl, sl, sl, sl] = single.v
        self.single = single

    def sector_hamiltonian(self, N):
        basis = SlaterBasis(self.n_modes, N)
        return basis, basis.hamiltonian_matrix(self.h, self.v)


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True, linewidth=120)

    # ------------------------------------------------------------ Lipkin
    print("=== Lipkin model, N = 4, eps = 2 (Table 3.lipkin)")
    for V, W in ((-1 / 3, -1 / 4), (-4 / 3, -1.0)):
        lip = LipkinModel(4, eps=2.0, V=V, W=W)
        Hq = lip.quasispin_matrix()
        Eq, Vq = np.linalg.eigh(Hq)
        E0, v0, basis = lip.exact(4)
        assert abs(Eq[0] - E0) < 1e-10
        w = Vq[:, 0] ** 2
        print(f"V = {V:+.4f}, W = {W:+.4f}: eigenvalues {Eq}")
        print(f"   ground state weights on |2,-2>,|2,0>,|2,2>: {w[0]:.5f} {w[2]:.5f} {w[4]:.5f}; "
              f"70x70 space gives E0 = {E0:.5f}")
    Vscan = -np.linspace(0.0, 3.0, 61)
    lip_E, lip_w = [], []
    for V in Vscan:
        Eq, Vq = np.linalg.eigh(LipkinModel(4, eps=2.0, V=V, W=-1 / 4).quasispin_matrix())
        lip_E.append(Eq[0]); lip_w.append(Vq[0, 0] ** 2)
    lip_E, lip_w = np.array(lip_E), np.array(lip_w)
    k = np.argmin(np.abs(lip_w - 0.5))
    print(f"   reference weight passes 1/2 at -V = {-Vscan[k]:.2f}")

    # ----------------------------------------------------------- pairing
    print("\n=== Pairing model, L = 4, two pairs, xi = 1 (Table 3.pairing)")
    print(f"{'g':>6s} {'E_ref':>10s} {'E0 (6x6)':>12s} {'E0 (70x70)':>12s} {'E_corr':>12s}")
    pair_scan = {}
    for g in (-1.0, -0.5, 0.0, 0.5, 1.0):
        pm = PairingModel(4, g=g)
        _, H6 = pm.seniority_zero_matrix(2)
        E6 = np.linalg.eigvalsh(H6)[0]
        E0, v0, basis = pm.exact(4)
        Eref = H6[0, 0]
        assert abs(E6 - E0) < 1e-10
        print(f"{g:6.2f} {Eref:10.6f} {E6:12.8f} {E0:12.8f} {E0 - Eref:12.8f}")
    for L, n in ((4, 2), (6, 3), (8, 4)):
        gs = np.linspace(-1, 1, 41)
        corr = []
        for g in gs:
            _, H = PairingModel(L, g=g).seniority_zero_matrix(n)
            corr.append(np.linalg.eigvalsh(H)[0] - H[0, 0])
        pair_scan[(L, n)] = (gs, np.array(corr))
        print(f"   L = {L}, n = {n}: dim {comb(L, n)}, E_corr(g=1) = {corr[-1]:.4f}, per pair {corr[-1]/n:.4f}")

    print("\n=== Truncated CI for the pairing model, L = 4, N = 4 (Table 3.citrunc)")
    print(f"{'g':>5s} {'E_ref':>9s} {'CIS':>12s} {'CID':>12s} {'CISD':>12s} {'FCI':>12s}")
    ref = PairingModel(4).reference(2)
    for g in (0.25, 0.5, 1.0, 2.0):
        pm = PairingModel(4, g=g)
        row = []
        for lvl in (0, 1, 2, 4):
            E, dim = truncated_ci(pm, 4, ref, lvl)
            row.append(E)
        cid, _ = truncated_ci(pm, 4, ref, 2)
        # CID: only doubles (no singles) -- for the pairing model singles are inert, so CID = CISD
        print(f"{g:5.2f} {row[0]:9.6f} {row[1]:12.8f} {cid:12.8f} {row[2]:12.8f} {row[3]:12.8f}")
    dims = [truncated_ci(PairingModel(4, g=1), 4, ref, lvl)[1] for lvl in range(5)]
    print("   dimensions of the truncated spaces:", dims)

    print("\n=== Size consistency: two non-interacting copies of a two-level system (Table 3.sizecons)")
    print(f"{'g':>5s} {'2E(A)':>13s} {'FCI':>13s} {'CID':>13s} {'CID error':>11s}")
    for g in (0.25, 0.5, 1.0):
        single = PairingModel(2, g=g)
        EA = single.exact(2)[0]
        two = NonInteractingCopies(2, g)
        basis, H = two.sector_hamiltonian(4)
        Efci = np.linalg.eigvalsh(H)[0]
        refstate = sum(basis.bit(p) for p in (0, 1, 4, 5))
        keep = [k for k, s in enumerate(basis.states) if basis.excitation_level(s, refstate) <= 2]
        Ecid = np.linalg.eigvalsh(H[np.ix_(keep, keep)])[0]
        print(f"{g:5.2f} {2*EA:13.8f} {Efci:13.8f} {Ecid:13.8f} {Ecid - Efci:11.2e}")
        assert abs(Efci - 2 * EA) < 1e-10

    # ------------------------------------------------------------ Hubbard
    print("\n=== Half-filled four-site Hubbard ring (Tables 3.hubbard, 3.hubbardstrong)")
    print(f"{'U/t':>5s} {'E0/t':>13s} {'E0/(Nt)':>10s} {'double occ':>11s} {'E0/J':>9s}")
    hub_scan = []
    for U in (0, 2, 4, 8, 16, 32, 64, 128):
        hub = HubbardModel(4, t=1.0, U=float(U))
        E0, v0, basis = hub.exact(4)
        docc = hub.double_occupancy(basis, v0)
        EJ = E0 / (4 / U) if U else float("nan")
        hub_scan.append((U, E0, docc))
        print(f"{U:5d} {E0:13.8f} {E0/4:10.6f} {docc:11.6f} {EJ:9.5f}")

    # --------------------------------------------------------- Heisenberg
    print("\n=== Antiferromagnetic Heisenberg ring, J = 1 (Table 3.heisenberg)")
    bethe = 0.25 - log(2)
    print(f"{'N':>3s} {'dim':>6s} {'E0/J':>13s} {'E0/(NJ)':>12s} {'deviation':>10s}")
    heis_scan = []
    for N in (4, 6, 8, 10, 12):
        states, H = HeisenbergModel(N).sector_matrix()
        E0 = np.linalg.eigvalsh(H)[0]
        heis_scan.append((N, E0 / N))
        print(f"{N:3d} {len(states):6d} {E0:13.8f} {E0/N:12.8f} {E0/N - bethe:10.6f}")
    print(f"  Bethe ansatz: E0/(NJ) -> 1/4 - ln 2 = {bethe:.8f}")
    states, H = HeisenbergModel(4).sector_matrix()
    E, V = np.linalg.eigh(H)
    hm = HeisenbergModel(4)
    print(f"  four-site ring: <S_i.S_i+1> = {hm.correlation(states, V[:,0], 0, 1):.4f}, "
          f"<S_i.S_i+2> = {hm.correlation(states, V[:,0], 0, 2):.4f}")

    # ------------------------------------------------------------- wall
    print("\n=== The exponential wall (Table 3.wall)")
    print(f"{'n':>4s} {'C(n,n/2)':>12s} {'2^n/sqrt(n)':>12s}")
    for n in (10, 20, 40, 80, 160):
        print(f"{n:4d} {comb(n, n // 2):12.3e} {2 ** n / sqrt(n):12.3e}")
    print(f"  oxygen-16 neutrons, 8 in 40 states: C(40,8) = {comb(40, 8):.3e}; protons and neutrons: {comb(40, 8) ** 2:.3e}")

    # ---------------------------------------------------------- Lanczos
    print("\n=== Lanczos iteration for the 14-site Heisenberg ring (Table 3.lanczos)")
    states, H = HeisenbergModel(14).sector_matrix()
    E_exact = np.linalg.eigvalsh(H)[0]
    rng = np.random.default_rng(0)
    v0 = rng.normal(size=len(states))
    print(f"  dimension {len(states)}, exact E0 = {E_exact:.8f}")
    print(f"{'m':>4s} {'lowest Ritz value':>18s} {'error':>10s}")
    for m in (5, 10, 20, 40, 80):
        Em = lanczos(H, v0, m)[0]
        print(f"{m:4d} {Em:18.8f} {Em - E_exact:10.2e}")
    assert abs(lanczos(H, v0, 80)[0] - E_exact) < 1e-8

    # ----------------------------------------------------------- figure
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.6), gridspec_kw={"wspace": 0.5, "hspace": 0.45})
    ax = axes[0, 0]
    ax.plot(-Vscan, lip_E, label="$E_0$"); ax.set_xlabel("$-V$"); ax.set_ylabel("$E_0$")
    ax2 = ax.twinx(); ax2.plot(-Vscan, lip_w, "C3--", label="weight of $|2,-2\\rangle$")
    ax2.set_ylabel("reference weight", color="C3"); ax2.axhline(0.5, color="gray", lw=0.5)
    ax.set_title("Lipkin, $N=4$, $\\varepsilon=2$, $W=-1/4$", fontsize=9)
    ax = axes[0, 1]
    for (L, n), (gs, corr) in pair_scan.items():
        ax.plot(gs, corr, label=f"$L={L}$, $n={n}$")
    ax.set_xlabel("$g$"); ax.set_ylabel("$E_{\\mathrm{corr}}$"); ax.legend(fontsize=8)
    ax.set_title("pairing model, correlation energy", fontsize=9)
    ax = axes[1, 0]
    Us = [u for u, _, _ in hub_scan if u > 0]
    ax.semilogx(Us, [d for u, _, d in hub_scan if u > 0], "o-", label="double occupancy")
    ax.semilogx(Us, [e / 4 for u, e, _ in hub_scan if u > 0], "s-", label="$E_0/(Nt)$")
    ax.set_xlabel("$U/t$"); ax.legend(fontsize=8); ax.set_title("Hubbard ring, 4 sites, half filling", fontsize=9)
    ax = axes[1, 1]
    Ns = np.array([n for n, _ in heis_scan]); es = np.array([e for _, e in heis_scan])
    ax.plot(1 / Ns ** 2, es, "o-", label="$E_0/(NJ)$")
    ax.axhline(bethe, color="k", ls="--", lw=1, label="Bethe ansatz")
    ax.set_xlabel("$1/N^2$"); ax.set_ylabel("$E_0/(NJ)$"); ax.legend(fontsize=8)
    ax.set_title("Heisenberg ring, finite-size scaling", fontsize=9)
    print("figure written to", savefig(fig, 3, "models"))
