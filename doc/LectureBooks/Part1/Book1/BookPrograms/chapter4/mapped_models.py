"""Chapter 4, Sec. 4.4: the four model Hamiltonians on a qubit register.

Builds the Pauli-string form of every model of chapter 3 and verifies each
against the exact diagonalisation of chapter 3, eigenvalue by eigenvalue:

* Lipkin model in four encodings -- one qubit per spin-orbital (2 Omega
  qubits, Jordan-Wigner), one qubit per doublet (Omega qubits, quasispin
  Pauli map), one-hot |J,J_z> states (2J+1 qubits) and binary-compressed
  (ceil(log2(2J+1)) qubits) -- Table 4.lipkinencodings;
* pairing model, one qubit per spin-orbital (Eq. 4.jwpairingfull) and one
  qubit per level (Eqs. 4.jwpairingcompact, 4.jwpairingtwolevels), with and
  without the particle-hole term -- Table 4.paulicounts and Fig. 4.encodings;
* Hubbard ring on eight qubits with two orbital orderings -- Table 4.hubbardjw;
* Heisenberg ring, which is already a Pauli sum -- and its fermionic image.

Runs in about one second.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

from math import ceil, log2, comb
import numpy as np
from pauli import PauliSum
from jordanwigner import JordanWigner
from fockspace import FockSpace
from models import LipkinModel, PairingModel, HubbardModel, HeisenbergModel
from bookplot import savefig, plt


def sector_spectrum(H_pauli, N):
    """Eigenvalues of a Pauli sum restricted to the N-particle (N-excitation) sector."""
    fs = FockSpace(H_pauli.n)
    idx = fs.sector(N)
    M = H_pauli.matrix()[np.ix_(idx, idx)]
    return np.linalg.eigvalsh(M)


# ---------------------------------------------------------------- Lipkin
def lipkin_level_encoding(model):
    """One qubit per doublet at half filling: j_z -> -Z/2, j_+ -> (X - iY)/2 (|0> = lower level)."""
    n = model.Omega
    Jz = PauliSum(n_qubits=n); Jp = PauliSum(n_qubits=n)
    for p in range(n):
        Jz = Jz + PauliSum.single(n, p, "Z", -0.5)
        Jp = Jp + PauliSum.single(n, p, "X", 0.5) + PauliSum.single(n, p, "Y", -0.5j)
    Jm = Jp.dagger()
    Nop = model.N * PauliSum.identity(n)
    H = (model.eps * Jz + 0.5 * model.V * (Jp * Jp + Jm * Jm)
         + 0.5 * model.W * (Jp * Jm + Jm * Jp - Nop))
    return H.simplify()


def one_hot_encoding(Hmat):
    """|m><m'| -> sigma^+_m sigma^-_m' on a register with one qubit per basis state."""
    d = Hmat.shape[0]
    H = PauliSum(n_qubits=d)
    for m in range(d):
        H = H + Hmat[m, m] * (0.5 * PauliSum.identity(d) - PauliSum.single(d, m, "Z", 0.5))
        for mp in range(m + 1, d):
            if abs(Hmat[m, mp]) > 1e-14:
                XX = PauliSum.single(d, m, "X") * PauliSum.single(d, mp, "X")
                YY = PauliSum.single(d, m, "Y") * PauliSum.single(d, mp, "Y")
                H = H + 0.5 * Hmat[m, mp] * (XX + YY)
    return H.simplify()


def binary_encoding(Hmat, pad=0.0):
    """Store the (2J+1)-dimensional block on ceil(log2(2J+1)) qubits; unused
    basis states are given the energy `pad` (choose it above the spectrum)."""
    d = Hmat.shape[0]
    k = max(1, ceil(log2(d)))
    M = np.eye(2 ** k) * pad
    M[:d, :d] = Hmat
    return PauliSum.from_matrix(M), k


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True, linewidth=120)

    print("=== Lipkin model, N = Omega = 4, eps = 2, V = -1/3, W = -1/4 (Table 4.lipkinencodings)")
    lip = LipkinModel(4, eps=2.0, V=-1 / 3, W=-1 / 4)
    Hq = lip.quasispin_matrix()
    Eq = np.linalg.eigvalsh(Hq)
    print(f"{'encoding':>22s} {'qubits':>7s} {'strings':>8s} {'max weight':>11s} {'E0':>10s}")
    rows = []
    # (a) occupation: Jordan-Wigner on 2 Omega qubits, N-particle sector
    Hocc = JordanWigner(lip.n_modes).map_hamiltonian(lip.h, lip.v)
    E = sector_spectrum(Hocc, lip.N)
    rows.append(("one per spin-orbital", Hocc.n, Hocc.n_strings(), Hocc.max_weight(), E[0]))
    assert abs(E[0] - Eq[0]) < 1e-10
    # (b) level encoding on Omega qubits
    Hlev = lipkin_level_encoding(lip)
    E = np.linalg.eigvalsh(Hlev.matrix())
    rows.append(("one per doublet", Hlev.n, Hlev.n_strings(), Hlev.max_weight(), E[0]))
    assert abs(E[0] - Eq[0]) < 1e-10
    # the J = 2 block sits inside the 16-dimensional level register: all five quasispin eigenvalues appear
    assert all(np.min(np.abs(E - e)) < 1e-10 for e in Eq)
    # (c) one-hot on 2J+1 qubits
    Hhot = one_hot_encoding(Hq)
    fs = FockSpace(Hhot.n)
    E = np.linalg.eigvalsh(Hhot.matrix()[np.ix_(fs.sector(1), fs.sector(1))])
    rows.append(("one-hot |J,J_z>", Hhot.n, Hhot.n_strings(), Hhot.max_weight(), E[0]))
    assert np.allclose(E, Eq)
    # (d) binary on ceil(log2(2J+1)) qubits
    Hbin, k = binary_encoding(Hq, pad=Eq[-1] + 1)
    E = np.linalg.eigvalsh(Hbin.matrix())
    rows.append(("binary, J = 2 block", k, Hbin.n_strings(), Hbin.max_weight(), E[0]))
    assert np.allclose(E[:5], Eq)
    for name, nq, ns, w, e0 in rows:
        print(f"{name:>22s} {nq:7d} {ns:8d} {w:11d} {e0:10.5f}")
    print("level encoding, Pauli form:")
    print(Hlev)

    # ---------------------------------------------------------------- pairing
    print("\n=== Pairing model, L = 2, xi = 1, g = 1, one qubit per spin-orbital (Eq. 4.jwpairingfull)")
    pm = PairingModel(2, g=1.0)
    Hfull = JordanWigner(4).map_hamiltonian(pm.h, pm.v)
    print(Hfull)
    assert Hfull.n_strings() == 15 and abs(Hfull.terms["IIII"] - 0.75) < 1e-12
    assert abs(Hfull.terms["IIZI"] + 3 / 8) < 1e-12 and abs(Hfull.terms["ZIII"] - 1 / 8) < 1e-12
    assert abs(Hfull.terms["XXXX"] + 1 / 16) < 1e-12 and abs(Hfull.terms["XXYY"] - 1 / 16) < 1e-12

    def compact_pairing(pm):
        """Eq. (4.jwpairingcompact): sum_p [xi(p-1) - g/4](I - Z_p) - g/4 sum_{p<q} (X_pX_q + Y_pY_q)."""
        L = pm.L
        H = PauliSum(n_qubits=L)
        for p in range(L):
            c = pm.xi * p - pm.g / 4
            H = H + c * (PauliSum.identity(L) - PauliSum.single(L, p, "Z"))
            for q in range(p + 1, L):
                H = H - pm.g / 4 * (PauliSum.single(L, p, "X") * PauliSum.single(L, q, "X")
                                    + PauliSum.single(L, p, "Y") * PauliSum.single(L, q, "Y"))
        return H.simplify()

    print("\n=== Pairing model, one qubit per level (Eq. 4.jwpairingtwolevels)")
    Hc = compact_pairing(pm)
    print(Hc)
    assert Hc.terms == {"II": 0.5, "ZI": 0.25, "IZ": -0.75, "XX": -0.25, "YY": -0.25}
    for L in (2, 3):
        pmL = PairingModel(L, g=1.0)
        HcL = compact_pairing(pmL)
        HfL = JordanWigner(2 * L).map_hamiltonian(pmL.h, pmL.v)
        for n_pairs in range(L + 1):
            _, Hs = pmL.seniority_zero_matrix(n_pairs)
            Es = np.linalg.eigvalsh(Hs)
            Ec = sector_spectrum(HcL, n_pairs)           # n_pairs occupied level-qubits
            Ef = sector_spectrum(HfL, 2 * n_pairs)       # all determinants with 2 n_pairs particles
            assert np.allclose(Ec, Es)
            assert all(np.min(np.abs(Ef - e)) < 1e-10 for e in Es)
        print(f"  L = {L}: compact spectrum = seniority-zero spectrum for every pair number "
              f"(dimensions {[comb(L, n) for n in range(L + 1)]}), and a subset of the full one")

    print("\n=== The cost of the pairing model (Table 4.paulicounts)")
    print(f"{'Hamiltonian':>22s} {'encoding':>16s} {'qubits':>7s} {'strings':>8s} {'max weight':>11s}")
    table = {}
    for L in (2, 3):
        pmL = PairingModel(L, g=1.0)
        Hc = compact_pairing(pmL)
        Hf = JordanWigner(2 * L).map_hamiltonian(pmL.h, pmL.v)
        pmf = PairingModel(L, g=1.0, f=0.5)
        Hph = JordanWigner(2 * L).map_hamiltonian(pmf.h, pmf.v)
        for name, H in ((f"pairing, L = {L}", Hc), (f"pairing, L = {L}", Hf), (f"pairing + p-h, L = {L}", Hph)):
            enc = "one per level" if H.n == L else "one per orbital"
            table[(name, enc)] = (H.n, H.n_strings(), H.max_weight())
            print(f"{name:>22s} {enc:>16s} {H.n:7d} {H.n_strings():8d} {H.max_weight():11d}")
        # the particle-hole Hamiltonian leaks out of the seniority-zero space
        E_ph = sector_spectrum(Hph, 2)
        E0, _, _ = pmf.exact(2)
        assert abs(E_ph[0] - E0) < 1e-10
    assert table[("pairing, L = 2", "one per level")][1:] == (5, 2)
    assert table[("pairing, L = 2", "one per orbital")][1:] == (15, 4)
    assert table[("pairing + p-h, L = 2", "one per orbital")][1:] == (27, 4)
    assert table[("pairing, L = 3", "one per level")][1:] == (10, 2)
    assert table[("pairing, L = 3", "one per orbital")][1:] == (34, 4)
    assert table[("pairing + p-h, L = 3", "one per orbital")][1:] == (118, 6)
    # cost against L for the figure
    Ls = list(range(1, 9))
    n_full = [4 * L * (L - 1) + 3 * L + 1 for L in Ls]
    n_comp = [L * L + 1 for L in Ls]
    for L in (4, 5, 6):
        pmL = PairingModel(L, g=1.0)
        assert JordanWigner(2 * L).map_hamiltonian(pmL.h, pmL.v).n_strings() == 4 * L * (L - 1) + 3 * L + 1
        assert compact_pairing(pmL).n_strings() == L * L + 1
    print("  string counts 4L(L-1)+3L+1 (full) and L^2+1 (compact) verified up to L = 6")

    # ---------------------------------------------------------------- Hubbard
    print("\n=== Half-filled Hubbard ring, 4 sites, U = 4t, eight qubits (Table 4.hubbardjw)")
    hub = HubbardModel(4, t=1.0, U=4.0)
    E0, _, _ = hub.exact(4)
    print(f"{'ordering':>28s} {'strings':>8s} {'max weight':>11s} {'hopping strings by weight':>26s}")
    # (a) site-major ordering (up, down of site 0, then site 1, ...): modes 2i+s -- as in models.py
    # (b) spin-major ordering: all spin-up sites first, then spin-down
    perm_spin = [2 * i for i in range(4)] + [2 * i + 1 for i in range(4)]   # new position k holds old mode perm[k]
    for name, perm in (("site-major (i up, i down, ...)", list(range(8))), ("spin-major (all up, all down)", perm_spin)):
        h = hub.h[np.ix_(perm, perm)]
        v = hub.v[np.ix_(perm, perm, perm, perm)]
        H = JordanWigner(8).map_hamiltonian(h, v)
        Hhop = JordanWigner(8).one_body(h)
        E = sector_spectrum(H, 4)
        assert abs(E[0] - E0) < 1e-9
        print(f"{name:>28s} {H.n_strings():8d} {H.max_weight():11d} {str(Hhop.weights()):>26s}")
    print(f"  ground-state energy from the qubit Hamiltonian {E[0]:.8f} (chapter 3: {E0:.8f})")

    # ------------------------------------------------------------- Heisenberg
    print("\n=== Heisenberg ring, 4 sites: already a Pauli sum")
    heis = HeisenbergModel(4)
    Hh = PauliSum.from_matrix(heis.full_matrix())
    print(Hh)
    assert Hh.n_strings() == 12 and Hh.max_weight() == 2
    states, Hs = heis.sector_matrix()
    assert abs(np.linalg.eigvalsh(Hh.matrix())[0] - np.linalg.eigvalsh(Hs)[0]) < 1e-10
    # the fermionic image, Eq. (4.heisenbergfermions): S^+_i = a_i^dagger with the parity string,
    # S^z_i = n_i - 1/2 ... for nearest neighbours the strings cancel
    jw = JordanWigner(4)
    Hf = PauliSum(n_qubits=4)
    for i, j in [(0, 1), (1, 2), (2, 3)]:
        hop = jw.create(i) * jw.annihilate(j) + jw.create(j) * jw.annihilate(i)
        zz = (jw.number(i) - 0.5) * (jw.number(j) - 0.5)
        Hf = Hf + 0.5 * heis.J * hop + heis.J * zz
    # the bond (3,0) closes the ring and drags the parity of sites 1 and 2 along
    hop = jw.create(3) * jw.annihilate(0) + jw.create(0) * jw.annihilate(3)
    zz = (jw.number(3) - 0.5) * (jw.number(0) - 0.5)
    Hf = (Hf + 0.5 * heis.J * hop + heis.J * zz).simplify()
    print("fermionic form: the open bonds give the same Pauli strings; the closing bond (3,0) carries a parity string:")
    print({k: c for k, c in Hf.terms.items() if k[1] == "Z" and k[2] == "Z"})
    Ef = np.linalg.eigvalsh(Hf.matrix()[np.ix_(FockSpace(4).sector(2), FockSpace(4).sector(2))])
    print(f"  lowest eigenvalues with two fermions: {Ef[:3]} -- not the spin ring's -2:")
    print("  the parity string twists the boundary condition (Sec. 4.4); the spin form is the one to use")

    # ------------------------------------------------- symmetries as commutators
    print("\n=== Symmetries as vanishing commutators of Pauli sums (Table 4.symmetries)")
    pm4 = PairingModel(4, g=1.0)
    hubH = JordanWigner(8).map_hamiltonian(hub.h[np.ix_(perm_spin, perm_spin)],
                                           hub.v[np.ix_(perm_spin, perm_spin, perm_spin, perm_spin)])
    jw8 = JordanWigner(8)
    Nop = jw8.total_number()
    Sz = PauliSum(n_qubits=8)
    for i in range(4):                                       # spin-major ordering: qubits 0-3 up, 4-7 down
        Sz = Sz + 0.5 * (jw8.number(i) - jw8.number(4 + i))
    parity = PauliSum({"Z" * 8: 1.0})
    Hp8 = JordanWigner(8).map_hamiltonian(pm4.h, pm4.v)
    pmf4 = PairingModel(4, g=1.0, f=0.5)
    Hph8 = JordanWigner(8).map_hamiltonian(pmf4.h, pmf4.v)
    seniority = PauliSum(n_qubits=8)                         # sum_p (n_{p+} - n_{p-})^2 = number of unpaired particles
    for pp in range(4):
        d = jw8.number(2 * pp) - jw8.number(2 * pp + 1)
        seniority = seniority + d * d
    checks = [("Hubbard ring", hubH, [("N", Nop), ("S^z", Sz), ("parity", parity)]),
              ("pairing L=4", Hp8, [("N", Nop), ("parity", parity), ("seniority", seniority)]),
              ("pairing + p-h", Hph8, [("N", Nop), ("parity", parity), ("seniority", seniority)])]
    for name, H, ops in checks:
        row = []
        for oname, O in ops:
            c = H.commutator(O).simplify()
            row.append(f"[H, {oname}] = {'0' if not c.terms else f'{c.n_strings()} strings, norm bound {c.norm_bound():.3f}'}")
        print(f"  {name:>14s}: " + "; ".join(row))

    # ------------------------------------------------- matrix-free application
    print("\n=== Matrix-free Lanczos on the eight-qubit Hubbard Hamiltonian")
    import time
    from fockspace import FockSpace
    sec = FockSpace(8).sector(4)
    v0 = np.zeros(256); v0[sec] = np.random.default_rng(1).normal(size=len(sec)); v0 /= np.linalg.norm(v0)
    Hop = lambda v: hubH.apply(v)
    # Lanczos with the string-by-string product (same loop as chapter 3)
    m = 30; V = np.zeros((m, 256), complex); al = np.zeros(m); be = np.zeros(m); v = v0.astype(complex)
    t0 = time.time()
    for k in range(m):
        V[k] = v; w = Hop(v); al[k] = np.real(np.vdot(v, w)); w -= al[k] * v + (be[k - 1] * V[k - 1] if k else 0)
        w -= V[:k + 1].T @ (V[:k + 1].conj() @ w); be[k] = np.linalg.norm(w); v = w / be[k]
    T = np.diag(al) + np.diag(be[:-1], 1) + np.diag(be[:-1], -1)
    E_lanczos = np.linalg.eigvalsh(T)[0]
    print(f"  30 Lanczos steps with H.apply: E0 = {E_lanczos:.8f} (exact {E0:.8f}), {time.time() - t0:.2f} s, "
          f"{hubH.n_strings()} strings x 2^8 amplitudes per product")
    assert abs(E_lanczos - E0) < 1e-6

    # ------------------------------------------------- Trotter error bounds
    print("\n=== Norms and commutator bounds of the Pauli sums (Table 4.bounds)")
    print(f"{'Hamiltonian':>34s} {'sum|h|':>8s} {'||H||':>8s} {'commutator bound':>17s}")
    for name, H in (("Lipkin, one per doublet", Hlev), ("pairing L=4, one per level", compact_pairing(pm4)),
                    ("pairing L=4, one per orbital", Hp8), ("Hubbard ring, spin-major", hubH), ("Heisenberg ring", Hh)):
        nb = H.norm_bound(); nrm = np.linalg.norm(H.matrix(), 2); cb = H.commutator_bound()
        print(f"{name:>34s} {nb:8.3f} {nrm:8.3f} {cb:17.3f}")

    # ------------------------------------------------- measurement settings
    print("\n=== Measurement settings: groups of qubit-wise commuting strings (Table 4.settings)")
    print(f"{'Hamiltonian':>34s} {'qubits':>7s} {'strings':>8s} {'settings':>9s}")
    for name, H in (("Lipkin, one per doublet", Hlev),
                    ("pairing L=4, one per level", compact_pairing(pm4)),
                    ("pairing L=4, one per orbital", JordanWigner(8).map_hamiltonian(pm4.h, pm4.v)),
                    ("Hubbard ring, spin-major", hubH),
                    ("Heisenberg ring", Hh)):
        g = H.commuting_groups(qubitwise=True)
        print(f"{name:>34s} {H.n:7d} {H.n_strings():8d} {len(g):9d}")
    print("  Heisenberg groups:", Hh.commuting_groups(qubitwise=True))

    # ------------------------------------------------------ qubit tapering
    print("\n=== Tapering a Z2 symmetry: the two-level pairing model on one qubit (Eq. 4.tapered)")
    H2 = compact_pairing(PairingModel(2, g=1.0))
    sym = PauliSum({"ZZ": 1.0})                              # pair-number parity, commutes with H
    assert not H2.commutator(sym).simplify().terms
    import gates as G
    U = G.CNOT                                               # maps Z0Z1 -> Z1
    Hrot = PauliSum.from_matrix(U @ H2.matrix() @ U.conj().T)
    print("  after CNOT_01:", {k: float(c) for k, c in Hrot.terms.items()})
    assert all(k[1] in "IZ" for k in Hrot.terms)             # qubit 1 carries only Z's
    tapered = {}
    for k, c in Hrot.terms.items():                           # one pair: Z0Z1 = -1, i.e. Z1 -> -1
        tapered[k[0]] = tapered.get(k[0], 0) + c * (-1 if k[1] == "Z" else 1)
    Ht = PauliSum(tapered, 1)
    print("  one-qubit Hamiltonian:", {k: float(c) for k, c in Ht.terms.items()})
    E1 = np.linalg.eigvalsh(Ht.matrix())
    _, Hs = PairingModel(2, g=1.0).seniority_zero_matrix(1)
    assert np.allclose(E1, np.linalg.eigvalsh(Hs))
    print(f"  eigenvalues {E1} = seniority-zero spectrum with one pair")

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), gridspec_kw={"wspace": 0.35})
    ax = axes[0]
    ax.plot(Ls, n_full, "o-", label="one qubit per orbital, $4L(L-1)+3L+1$")
    ax.plot(Ls, n_comp, "s-", label="one qubit per level, $L^2+1$")
    ax.set_xlabel("levels $L$"); ax.set_ylabel("Pauli strings"); ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(Ls, [4] * len(Ls), "o-", label="one qubit per orbital")
    ax.plot(Ls, [2] * len(Ls), "s-", label="one qubit per level")
    ax.plot([2, 3], [table[("pairing + p-h, L = 2", "one per orbital")][2],
                     table[("pairing + p-h, L = 3", "one per orbital")][2]], "^--", label="with particle-hole term")
    ax.set_xlabel("levels $L$"); ax.set_ylabel("largest Pauli weight"); ax.set_ylim(0, 7); ax.legend(fontsize=8)
    print("figure written to", savefig(fig, 4, "encodings"))
