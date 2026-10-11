"""Chapter 8, Sec. 8.6: refinement and rodeo filtering for the Lipkin, Heisenberg and Hubbard models.

Each model gets the resolution hierarchy that fits it, with the target
Hamiltonians of Table 6.models:

* Lipkin (N = 2 -> 4, one qubit per doublet): the coarse ground state is embedded
  with the two new doublets in their lower level, P = 1 (x) |00>;
* Heisenberg ring (2 -> 4 and 4 -> 8 sites, S^z = 0): down spins are magnons on a
  lattice; the fully filled coarse lattice is one state and every magnon is
  spread over its two fine sites with the Marshall sign, (|du> - |ud>)/sqrt 2,
  which makes the prolonged state the singlet covering of chapter 7;
* Hubbard ring (2 -> 4 sites, N = 4, U = 4t): every coarse orbital becomes the
  bonding orbital of its two fine sites, Eq. (8.prolongdet);
* two particles on an open chain, 2 -> 4 -> 8 sites with t -> 4t and U -> 2U at
  every halving of the lattice spacing: a chain of two refinements with a
  non-trivial coarse space, the setting of the original proposal.

For every case: overlap of the prolonged state, minimum gap, T* for overlap
0.99, the overlap at T = 10 and 20 (Table 8.othermodels, Fig. 8.othermodels),
then the rodeo filter on the prolonged and refined states (Table 8.otherrodeo,
Fig. 8.othermodels, right).  About 10 s.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import itertools
import numpy as np
from models import LipkinModel, HubbardModel, HeisenbergModel
from mapped_models import lipkin_level_encoding
from refinement import Prolongation, ResolutionRefinement
from rodeo import RodeoFilter, infidelity_bound
from bookplot import savefig, plt

BUFFER = 1.0


# ------------------------------------------------------------------ Lipkin
def lipkin_refinement(V=-1 / 3, W=-1 / 4, eps=2.0, buffer=BUFFER):
    hi = lipkin_level_encoding(LipkinModel(4, eps=eps, V=V, W=W)).matrix().real
    lo = lipkin_level_encoding(LipkinModel(2, eps=eps, V=V, W=W)).matrix().real
    P = np.kron(np.eye(4), np.array([[1.0], [0.0], [0.0], [0.0]]))        # |psi> -> |psi>|00>
    return ResolutionRefinement(lo, hi, P, buffer=buffer, match_energy=True)


# -------------------------------------------------------------- Heisenberg
def singlet_covering(n, states):
    """prod over blocks (2b, 2b+1) of (|ud> - |du>)/sqrt 2 in the S^z = 0 basis (bit 1 = up)."""
    pos = {s: k for k, s in enumerate(states)}
    psi = np.zeros(len(states))
    for choice in itertools.product((0, 1), repeat=n // 2):
        state, sign = 0, 1
        for b, c in enumerate(choice):
            up = 2 * b + c                                   # which site of the block is up
            state |= 1 << (n - 1 - up)
            sign *= (-1) ** c
        psi[pos[state]] = sign
    return psi / np.linalg.norm(psi)


def heisenberg_refinement(n, buffer=BUFFER):
    states, H = HeisenbergModel(n).sector_matrix()
    H_low = HeisenbergModel(n // 2).sector_matrix(n_up=0)[1]      # the filled coarse lattice: one state
    P = singlet_covering(n, states)[:, None]
    return ResolutionRefinement(H_low, H, P, buffer=buffer, match_energy=True)


# ----------------------------------------------------------------- Hubbard
def spread(n_coarse, sign=+1):
    """W of Eq. (8.prolongdet): coarse site c -> (|2c> + sign |2c+1>)/sqrt 2, for both spins."""
    W = np.zeros((4 * n_coarse, 2 * n_coarse))
    for c in range(n_coarse):
        for s in (0, 1):
            W[2 * (2 * c) + s, 2 * c + s] = 1 / np.sqrt(2)
            W[2 * (2 * c + 1) + s, 2 * c + s] = sign / np.sqrt(2)
    return W


def hubbard_refinement(U=4.0, buffer=BUFFER):
    lo, hi = HubbardModel(2, t=1.0, U=U), HubbardModel(4, t=1.0, U=U)
    b_lo, H_lo = lo.sector_hamiltonian(4)
    b_hi, H_hi = hi.sector_hamiltonian(4)
    P = Prolongation.orbital(b_lo, b_hi, spread(2))
    return ResolutionRefinement(H_lo, H_hi, P, buffer=buffer, match_energy=True)


def box(n_sites, t, U):
    """Open Hubbard chain with the on-site energy 2t of the discretised kinetic energy."""
    m = HubbardModel(n_sites, t=t, U=U, periodic=False)
    m.h = m.h + 2 * t * np.eye(2 * n_sites)
    return m


def continuum_chain(t0=1.0, U0=2.0, buffer=4.0, T=2.0):
    """Two particles, 2 -> 4 -> 8 sites; the refined state of one stage is the input of the next."""
    out, psi = [], None
    for stage, (n_lo, n_hi) in enumerate(((2, 4), (4, 8))):
        lo = box(n_lo, t0 * 4 ** stage, U0 * 2 ** stage)
        hi = box(n_hi, t0 * 4 ** (stage + 1), U0 * 2 ** (stage + 1))
        b_lo, H_lo = lo.sector_hamiltonian(2)
        b_hi, H_hi = hi.sector_hamiltonian(2)
        P = Prolongation.orbital(b_lo, b_hi, spread(n_lo))
        rr = ResolutionRefinement(H_lo, H_hi, P, psi_low=psi, buffer=buffer * 4 ** stage, match_energy=True)
        psi = np.real_if_close(rr.evolve(T / 4 ** stage))
        out.append((n_lo, n_hi, b_hi.dim, rr, psi))
    return out


if __name__ == "__main__":
    cases = [("Lipkin $N=2\\to4$", lipkin_refinement()),
             ("Lipkin, $V=-1$", lipkin_refinement(V=-1.0)),
             ("Heisenberg $2\\to4$", heisenberg_refinement(4)),
             ("Heisenberg $4\\to8$", heisenberg_refinement(8)),
             ("Hubbard $2\\to4$", hubbard_refinement())]
    Ts = np.arange(0, 41)
    print(f"Refinement, buffer = {BUFFER}")
    print("   case                  dim    E_high      <H>_prol    p(0)     min gap   T*(0.99)   p(10)    p(20)")
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    refined = {}
    for (name, rr), mk in zip(cases, ("o-", "o--", "s-", "s--", "^-")):
        res = rr.scan(Ts)[:, 0]
        below = np.where(res < 0.99)[0]
        Tstar = Ts[below[-1] + 1] if len(below) and below[-1] + 1 < len(Ts) else (0 if not len(below) else np.inf)
        refined[name] = rr.evolve(20)
        plain = name.replace("$", "").replace("\\to", "->")
        print(f"   {plain:20s} {rr.H_high.shape[0]:4d}   {rr.E_high:9.5f}   {rr.energy(rr.phi0):9.5f}   {res[0]:.4f}   "
              f"{rr.min_gap():.4f}    {Tstar:4}      {res[10]:.4f}   {res[20]:.4f}")
        a.semilogy(Ts, 1 - res, mk, ms=3, label=name)
        assert res[-1] > 0.999
    a.set_xlabel("total adiabatic time $T$"); a.set_ylabel("$1-p$"); a.legend(fontsize=7)

    # the role of the sign of the prolongation for the antiferromagnet and for the Hubbard ring
    states, H = HeisenbergModel(4).sector_matrix()
    w, V = np.linalg.eigh(H)
    sym = np.abs(singlet_covering(4, states))                 # (|ud> + |du>) on every block
    sym /= np.linalg.norm(sym)
    print(f"\nHeisenberg 2 -> 4: overlap of the prolonged state with the ground state, "
          f"Marshall sign {abs(V[:, 0] @ singlet_covering(4, states)) ** 2:.4f}, plus sign {abs(V[:, 0] @ sym) ** 2:.4f}")

    # ---- rodeo on prolonged and refined (T = 20) states
    print("\nRodeo at E = E_0, sigma = 4: infidelity of the accepted runs (exact Gaussian average)")
    print("   case                  input       p        M = 2      M = 4      M = 8      acceptance (M = 8)")
    for name, rr in cases:
        rf = RodeoFilter(rr.H_high, sigma=4.0)
        plain = name.replace("$", "").replace("\\to", "->")
        for lab, psi in (("prolonged", rr.phi0), ("refined", refined[name])):
            inf, acc = rf.expected(psi, rr.E_high, 8)
            p = rr.overlap(psi)
            print(f"   {plain:20s} {lab:10s} {p:.4f}   {inf[1]:.2e}   {inf[3]:.2e}   {inf[7]:.2e}   {acc[7]:.4f}")
            assert np.all(np.diff(inf) < 0)
    name, rr = cases[-1]
    rf = RodeoFilter(rr.H_high, sigma=3.0, rng=np.random.default_rng(2))
    Es = np.linspace(rr.E_high - 1.5, rr.E_high + 9, 300)
    for lab, psi, col in (("prolonged", rr.phi0, "C1"), ("refined, $T=20$", refined[name], "C0")):
        b.semilogy(Es, rf.scan(psi, Es, M=5, n_sets=150), col, label=f"{lab}, $p={rr.overlap(psi):.3f}$")
    b.axhline(2.0 ** -5, color="k", ls="--", lw=0.8)
    b.set_xlabel("target energy $E$"); b.set_ylabel("$P_{0^5}(E)$"); b.set_title("Hubbard ring, $\\sigma=3$"); b.legend(fontsize=7)
    fig.tight_layout(); savefig(fig, 8, "othermodels")
    wts = np.abs(rf.V.T @ rr.phi0) ** 2
    print("\nHubbard ring: levels carrying weight in the prolonged state:",
          ", ".join(f"{e:.3f} ({x:.3f})" for e, x in zip(rf.w, wts) if x > 5e-3))

    # ---- a chain of two refinements with a non-trivial coarse space
    print("\nTwo particles on an open chain, t -> 4t, U -> 2U per stage, buffer 4 (scaled), T = 2 (scaled)")
    print("   stage    dim   E_low      <H>_prol    E_high     p(0)     p(T)     rodeo M = 4")
    for n_lo, n_hi, dim, rr, psi in continuum_chain():
        rf = RodeoFilter(rr.H_high, sigma=4.0 / (rr.spectrum_high[1] - rr.spectrum_high[0]))
        inf, acc = rf.expected(psi, rr.E_high, 4)
        print(f"   {n_lo} -> {n_hi}   {dim:4d}   {rr.E_low:8.4f}   {rr.energy(rr.phi0):8.4f}   {rr.E_high:8.4f}   "
              f"{rr.overlap(rr.phi0):.4f}   {rr.overlap(psi):.4f}   {inf[3]:.1e}")
