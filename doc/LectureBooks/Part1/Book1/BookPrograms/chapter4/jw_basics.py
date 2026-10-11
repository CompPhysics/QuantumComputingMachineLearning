"""Chapter 4, Secs. 4.2-4.3: the Jordan-Wigner transformation, checked term by term.

Shows why the naive map a_p -> sigma^-_p fails (Eq. 4.naive), verifies the
anticommutators of the Jordan-Wigner operators symbolically, prints the
Pauli-string identities for number, hopping, pair-creation and the general
two-body term (Eqs. 4.jwnumber-4.jwtwobody), counts the strings of a generic
one- plus two-body Hamiltonian on M orbitals (Table 4.paulicount), maps the
two-qubit Hamiltonian of Exercise 4.x, and produces the weight histogram of
the pairing Hamiltonian in the full encoding (Fig. 4.pauliweights).
Runs in about one second.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np
from pauli import PauliSum
from jordanwigner import JordanWigner, ParityMapping, BravyiKitaev
from models import PairingModel, HubbardModel, add_term
from bookplot import savefig, plt


def random_hamiltonian(M, rng):
    """Generic real, symmetric h and antisymmetrised v on M orbitals."""
    h = rng.normal(size=(M, M)); h = (h + h.T) / 2
    v = np.zeros((M, M, M, M))
    for p in range(M):
        for q in range(p + 1, M):
            for r in range(M):
                for s in range(r + 1, M):
                    if (p, q) <= (r, s):
                        c = rng.normal()
                        add_term(v, p, q, r, s, c)
                        if (p, q) != (r, s):
                            add_term(v, r, s, p, q, c)
    return h, v


if __name__ == "__main__":
    # ---------------------------------------------- the naive map fails
    print("=== Naive map a_p -> sigma^-_p (no parity string), Eq. (4.naive)")
    n = 2
    sm = [PauliSum.single(n, p, "X", 0.5) + PauliSum.single(n, p, "Y", 0.5j) for p in range(n)]
    anti = (sm[0] * sm[1] + sm[1] * sm[0]).simplify()
    print("  {sigma^-_0, sigma^-_1} =", anti.terms)
    print(f"  spectral norm {np.linalg.norm(anti.matrix(), 2):.1f} instead of 0")
    jw = JordanWigner(n)
    anti = (jw.annihilate(0) * jw.annihilate(1) + jw.annihilate(1) * jw.annihilate(0)).simplify()
    assert not anti.terms
    print("  with the parity string, {a_0, a_1} = 0")

    # ------------------------------------------------ operator identities
    print("\n=== Jordan-Wigner images on four qubits, Eqs. (4.jwnumber)-(4.jwtwobody)")
    jw = JordanWigner(4)
    print("a_2^dagger      :", jw.create(2).terms)
    print("n_2             :", jw.number(2).terms)
    hop = (jw.create(0) * jw.annihilate(3) + jw.create(3) * jw.annihilate(0)).simplify()
    print("a_0^+ a_3 + h.c.:", hop.terms)
    assert hop.terms == {"XZZX": 0.5, "YZZY": 0.5}
    pair = (jw.create(0) * jw.create(3) + jw.annihilate(3) * jw.annihilate(0)).simplify()
    print("a_0^+ a_3^+ + h.c.:", pair.terms)
    assert pair.terms == {"XZZX": 0.5, "YZZY": -0.5}
    two = jw.create(0) * jw.create(1) * jw.annihilate(3) * jw.annihilate(2)
    two = (two + two.dagger()).simplify()
    print("a_0^+ a_1^+ a_3 a_2 + h.c.:")
    print(two)
    assert two.n_strings() == 8 and all(abs(abs(c) - 0.125) < 1e-12 for c in two.terms.values())
    assert two.max_weight() == 4

    # ------------------------------------------------ generic string count
    print("\n=== Generic one- plus two-body Hamiltonian on M orbitals (Table 4.paulicount)")
    print(f"{'M':>3s} {'4^M':>6s} {'strings':>8s} {'max weight':>11s}")
    rng = np.random.default_rng(1)
    counts = {}
    for M in (2, 3, 4, 5, 6):
        h, v = random_hamiltonian(M, rng)
        H = JordanWigner(M).map_hamiltonian(h, v)
        counts[M] = (H.n_strings(), H.max_weight())
        print(f"{M:3d} {4 ** M:6d} {H.n_strings():8d} {H.max_weight():11d}")
        if M <= 4:   # cross-check against the exact trace decomposition of the Fock matrix
            from fockspace import FockSpace
            Hm = FockSpace(M).hamiltonian(h, v)
            assert np.allclose(H.matrix(), Hm)
    assert [counts[M][0] for M in (2, 3, 4, 5, 6)] == [6, 19, 55, 136, 292]

    # ----------------------------------------------- two-qubit Hamiltonian
    print("\n=== Two-qubit Hamiltonian of Exercise 4.twoqubit")
    e0, e1, U, t = 1.0, 2.0, 0.5, 0.3
    jw = JordanWigner(2)
    H = (e0 * jw.number(0) + e1 * jw.number(1) + U * (jw.number(0) * jw.number(1))
         + t * (jw.create(0) * jw.annihilate(1) + jw.create(1) * jw.annihilate(0))).simplify()
    print(H)
    expected = {"II": (e0 + e1) / 2 + U / 4, "ZI": -(2 * e0 + U) / 4, "IZ": -(2 * e1 + U) / 4,
                "ZZ": U / 4, "XX": t / 2, "YY": t / 2}
    assert all(abs(H.terms[k] - c) < 1e-12 for k, c in expected.items())
    M = H.matrix().real
    print("matrix in the basis |00>,|01>,|10>,|11>:\n", np.round(M, 4))
    Epm = np.linalg.eigvalsh(M[1:3, 1:3])
    print(f"one-particle block eigenvalues {Epm}, formula "
          f"{(e0 + e1) / 2 - np.hypot((e0 - e1) / 2, t):.4f}, {(e0 + e1) / 2 + np.hypot((e0 - e1) / 2, t):.4f}")

    # ------------------------------------------- three mappings compared
    print("\n=== Jordan-Wigner, parity and Bravyi-Kitaev mappings (Table 4.mappings)")
    print(f"{'M':>3s} {'mapping':>14s} {'weights of a+_p, p = 0..M-1':>40s}")
    for M in (4, 8):
        for name, cls in (("Jordan-Wigner", JordanWigner), ("parity", ParityMapping), ("Bravyi-Kitaev", BravyiKitaev)):
            m = cls(M)
            w = [m.create(p).max_weight() for p in range(M)]
            print(f"{M:3d} {name:>14s} {str(w):>40s}")
    assert [BravyiKitaev(8).create(p).max_weight() for p in range(8)] == [4] * 8
    print("  structured Hamiltonians on eight qubits (strings, largest weight):")
    pm4 = PairingModel(4, g=1.0)
    hub = HubbardModel(4, t=1.0, U=4.0)
    perm = [0, 2, 4, 6, 1, 3, 5, 7]                       # spin-major ordering
    hh, hv = hub.h[np.ix_(perm, perm)], hub.v[np.ix_(perm, perm, perm, perm)]
    for name, cls in (("Jordan-Wigner", JordanWigner), ("parity", ParityMapping), ("Bravyi-Kitaev", BravyiKitaev)):
        Hp = cls(8).map_hamiltonian(pm4.h, pm4.v)
        Hh = cls(8).map_hamiltonian(hh, hv)
        print(f"  {name:>14s}: pairing L=4 ({Hp.n_strings()}, {Hp.max_weight()}), "
              f"Hubbard ring ({Hh.n_strings()}, {Hh.max_weight()})")

    # ------------------------------------------------ a bosonic mode
    print("\n=== A bosonic mode truncated to d levels in binary encoding (Table 4.boson)")
    print(f"{'d':>3s} {'qubits':>7s} {'strings in b':>13s} {'strings in n':>13s} {'strings in x':>13s}")
    for d in (2, 4, 8, 16):
        k = int(np.log2(d))
        b = np.diag(np.sqrt(np.arange(1, d)), 1)            # <m-1| b |m> = sqrt(m)
        Pb = PauliSum.from_matrix(b)
        Pn = PauliSum.from_matrix(b.T @ b)
        Px = PauliSum.from_matrix((b + b.T) / np.sqrt(2))
        print(f"{d:3d} {k:7d} {Pb.n_strings():13d} {Pn.n_strings():13d} {Px.n_strings():13d}")
    print("  d = 4: position operator x = (b + b^dagger)/sqrt(2) =")
    b4 = np.diag(np.sqrt(np.arange(1, 4)), 1)
    print(PauliSum.from_matrix((b4 + b4.T) / np.sqrt(2)))

    # ------------------------------------------- pairing weight histogram
    print("\n=== Pairing Hamiltonian, g = 1, full encoding: Pauli weights (Fig. 4.pauliweights)")
    hist = {}
    for L in (1, 2, 3):
        pm = PairingModel(L, g=1.0)
        H = JordanWigner(2 * L).map_hamiltonian(pm.h, pm.v)
        hist[L] = H.weights()
        print(f"  L = {L} ({2 * L} qubits): {H.n_strings()} strings, weights {hist[L]}, "
              f"formula 4L(L-1)+3L+1 = {4 * L * (L - 1) + 3 * L + 1}")
        assert H.n_strings() == 4 * L * (L - 1) + 3 * L + 1
    assert hist[3].get(4, 0) == 24 and 3 not in hist[3]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    w = np.arange(5)
    for k, L in enumerate((1, 2, 3)):
        ax.bar(w + 0.25 * (k - 1), [hist[L].get(x, 0) for x in w], width=0.25, label=f"$L={L}$")
    ax.set_xlabel("Pauli weight"); ax.set_ylabel("number of strings"); ax.set_xticks(w)
    ax.set_title("pairing Hamiltonian, full encoding ($2L$ qubits)", fontsize=9); ax.legend()
    print("figure written to", savefig(fig, 4, "pauliweights"))
