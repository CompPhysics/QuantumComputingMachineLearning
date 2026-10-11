"""Chapter 3, Secs. 3.1-3.2: the algebra of creation and annihilation operators.

Verifies, with the explicit matrices of FockSpace, the anticommutation
relations (Eq. 3.anticommutation), the sign rule of Eq. (3.phase) on the
sixteen-orbital example of the text, the one- and two-body operators
(Eqs. 3.onebody, 3.twobody) against a direct construction, the vacuum
expectation values that Wick's theorem organises (Eq. 3.wickvev) and the
normal-ordered Hamiltonian E_ref + F_N + V_N (Eq. 3.HNO) for a random
interaction.  Instantaneous.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import itertools
import numpy as np
from fockspace import FockSpace, SlaterBasis


def apply_creation(bits, p):
    """a_p^dagger on a bit string (mode 0 leftmost): (sign, new bits) or (0, None)."""
    if bits[p] == 1:
        return 0, None
    sign = (-1) ** sum(bits[:p])
    new = list(bits); new[p] = 1
    return sign, new


def vacuum_expectation(fs, ops):
    """<0| op_1 op_2 ... |0> for ops given left to right as ('+', p) / ('-', p)."""
    M = np.eye(fs.dim)
    for kind, p in ops:
        M = M @ (fs.create(p) if kind == "+" else fs.annihilate(p))
    return M[0, 0]


def pairings(items):
    """All perfect matchings of a list, for the counting in Eq. (3.wickcount)."""
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for k in range(len(rest)):
        for m in pairings(rest[:k] + rest[k + 1:]):
            yield [(first, rest[k])] + m


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    # --- the sixteen-orbital example: modes 3, 6, 10, 13 occupied
    bits = [0] * 16
    for p in (3, 6, 10, 13):
        bits[p] = 1
    print("determinant |3 6 10 13> as bits:", "".join(map(str, bits)))
    for p in (4, 6, 12):
        sign, new = apply_creation(bits, p)
        print(f"  a^dagger_{p:<2d} ->", (f"{sign:+d} x |" + "".join(map(str, new)) + ">") if sign else "0")
    # --- anticommutators as matrix identities, and the Z string
    fs = FockSpace(4)
    print("largest violation of the anticommutation relations, 4 modes:", fs.check_anticommutators())
    # --- a two-particle overlap the hard way and by Wick's theorem, Eq. (3.fourresult)
    for r, s, p, q in ((0, 1, 0, 1), (0, 1, 1, 0), (0, 2, 0, 1)):
        vev = vacuum_expectation(fs, [("-", s), ("-", r), ("+", p), ("+", q)])
        wick = (r == p) * (s == q) - (s == p) * (r == q)
        assert vev == wick
        print(f"<0|a_{s} a_{r} a^dag_{p} a^dag_{q}|0> = {vev:+.0f}  (Wick: {wick:+d})")
    # --- counting: (M-1)!! pairings for M operators
    for M in (2, 4, 6, 8):
        print(f"M = {M}: {sum(1 for _ in pairings(list(range(M))))} pairings")
    # --- one- and two-body operators: matrix elements between determinants
    rng = np.random.default_rng(1)
    n = 4
    h = rng.normal(size=(n, n)); h = (h + h.T) / 2
    v = rng.normal(size=(n, n, n, n))
    v = v - v.transpose(1, 0, 2, 3); v = v - v.transpose(0, 1, 3, 2); v = (v + v.transpose(2, 3, 0, 1)) / 2
    H = fs.hamiltonian(h, v)
    ij = fs.determinant([0, 1])
    # <ij|H|ij> = h_ii + h_jj + <ij|v|ij>_AS, Eq. (3.twobodyresult)
    lhs = ij @ H @ ij
    rhs = h[0, 0] + h[1, 1] + v[0, 1, 0, 1]
    assert abs(lhs - rhs) < 1e-12
    print(f"<01|H|01> = {lhs:.6f} = h_00 + h_11 + <01|v|01> = {rhs:.6f}")
    # <ik|H|ij> = h_kj + <ik|v|ij> for a single replacement j -> k
    ik = fs.determinant([0, 2])
    lhs = ik @ H @ ij
    rhs = h[2, 1] + v[0, 2, 0, 1]
    assert abs(lhs - rhs) < 1e-12
    print(f"<02|H|01> = {lhs:.6f} = h_21 + <02|v|01> = {rhs:.6f}")
    # --- normal ordering: E_ref and the Fock matrix, Eqs. (3.Eref), (3.FN)
    N = 2
    holes, parts = [0, 1], [2, 3]
    E_ref = sum(h[i, i] for i in holes) + 0.5 * sum(v[i, j, i, j] for i in holes for j in holes)
    ref = fs.determinant(holes)
    assert abs(ref @ H @ ref - E_ref) < 1e-12
    f = h + sum(v[:, i, :, i] for i in holes)
    print(f"E_ref = {E_ref:.6f};  Fock matrix f_pq = h_pq + sum_i <pi|v|qi>:\n", f)
    # Brillouin-type elements: <Phi_i^a|H|c> = f_ai, <Phi_ij^ab|H|c> = <ab|v|ij>
    for i in holes:
        for a in parts:
            phi = fs.create(a) @ fs.annihilate(i) @ ref
            assert abs(phi @ H @ ref - f[a, i]) < 1e-12
    phi2 = fs.create(2) @ fs.create(3) @ fs.annihilate(1) @ fs.annihilate(0) @ ref
    assert abs(phi2 @ H @ ref - v[2, 3, 0, 1]) < 1e-12
    print("<Phi_i^a|H|c> = f_ai and <Phi_ij^ab|H|c> = <ab|v|ij>: verified")
    # --- the two-mode Hamiltonian of Exercise 3.x in the Fock basis |00>,|01>,|10>,|11>
    fs2 = FockSpace(2)
    e0, e1, U, t = 1.0, 2.0, 0.5, 0.3
    H2 = (e0 * fs2.number(0) + e1 * fs2.number(1) + U * fs2.number(0) @ fs2.number(1)
          + t * (fs2.create(0) @ fs2.annihilate(1) + fs2.create(1) @ fs2.annihilate(0)))
    print("two-mode Hamiltonian (basis |00>,|01>,|10>,|11>):\n", H2)
    Epm = (e0 + e1) / 2 + np.array([-1, 1]) * np.sqrt(((e0 - e1) / 2) ** 2 + t ** 2)
    assert np.allclose(np.sort(np.linalg.eigvalsh(H2)), np.sort([0, Epm[0], Epm[1], e0 + e1 + U]))
    print("eigenvalues:", np.linalg.eigvalsh(H2))
