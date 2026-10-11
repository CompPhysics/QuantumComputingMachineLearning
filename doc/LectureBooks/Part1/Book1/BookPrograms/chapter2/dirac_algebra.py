"""Chapter 2, Sec. 2.2: the worked linear-algebra examples of the text.

Inner and outer products in Dirac notation (Eqs. 2.innerex-2.outer),
projection operators and their action on a superposition (Eq. 2.project),
Pauli algebra (Eqs. 2.pauliprod-2.paulicomm), tensor products of states and
matrices (Eqs. 2.tensorstate-2.tensormatrix) and the spectral decomposition
of a Hermitian 2x2 matrix (Eq. 2.spectral).  Runs in well under a second.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]          # BookPrograms/
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np
import gates as G


def ket(*amplitudes):
    return np.array(amplitudes, dtype=complex)


def bra(v):
    return v.conj()


def inner(u, v):
    """<u|v> = sum_i u_i^* v_i."""
    return np.vdot(u, v)


def outer(u, v):
    """|u><v|, the matrix u_i v_j^*."""
    return np.outer(u, v.conj())


def spectral_decomposition(A):
    """Eigenvalues lambda_i and projectors P_i = |i><i| with A = sum_i lambda_i P_i."""
    lam, V = np.linalg.eigh(A)
    projectors = [outer(V[:, i], V[:, i]) for i in range(len(lam))]
    return lam, projectors


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    # --- Eq. (2.innerex): a two-component complex vector and its norm
    x = ket(1 - 1j, 2 + 1j)
    print("<x|x> =", inner(x, x).real, " normalised x =", x / np.sqrt(inner(x, x).real))
    # --- Eq. (2.innerasym): <x|y>^* = <y|x>
    x = ket(-1, 2j, 1); y = ket(1, 0, 1j)
    print("<x|y> =", inner(x, y), "  <y|x> =", inner(y, x))
    assert inner(x, y) == np.conj(inner(y, x))
    # --- computational basis, identity as a sum of projectors, Eq. (2.completeness)
    k0, k1 = ket(1, 0), ket(0, 1)
    P, Q = outer(k0, k0), outer(k1, k1)
    assert np.allclose(P + Q, np.eye(2)) and np.allclose(P @ P, P) and np.allclose(P @ Q, 0)
    print("P = |0><0| =\n", P.real, "\nQ = |1><1| =\n", Q.real)
    # --- projecting a superposition, Eq. (2.project)
    alpha, beta = 0.6, 0.8j
    psi = alpha * k0 + beta * k1
    print("P|psi> =", P @ psi, "  Q|psi> =", Q @ psi)
    # --- the density matrix of the superposition, Eq. (2.rhopure2)
    rho = outer(psi, psi)
    print("rho = |psi><psi| =\n", rho, "\ntrace =", np.trace(rho).real)
    # --- Pauli algebra, Eqs. (2.pauliprod)-(2.paulicomm)
    for a, b, c in (("X", "Y", "Z"), ("Y", "Z", "X"), ("Z", "X", "Y")):
        assert np.allclose(G.commutator(G.PAULIS[a], G.PAULIS[b]), 2j * G.PAULIS[c])
        assert np.allclose(G.PAULIS[a] @ G.PAULIS[b], 1j * G.PAULIS[c])
    print("[X,Y] = 2iZ, XY = iZ, and cyclic permutations: verified")
    # --- Hadamard on the basis states, Eq. (2.hadamardaction)
    print("H|0> =", G.H @ k0, "  H|1> =", G.H @ k1)
    # --- tensor products, Eqs. (2.tensorstate)-(2.tensormatrix)
    for a, ka in ((0, k0), (1, k1)):
        for b, kb in ((0, k0), (1, k1)):
            print(f"|{a}{b}> =", np.kron(ka, kb).real)
    print("|000> =", np.kron(np.kron(k0, k0), k0).real)
    A = np.array([[1, 2], [3, 4]]); B = np.array([[0, 1], [1, 0]])
    print("A (x) B =\n", np.kron(A, B))
    # --- spectral decomposition of a Hermitian matrix, Eq. (2.spectral)
    Aherm = np.array([[2, 1 - 1j], [1 + 1j, 3]])
    lam, projs = spectral_decomposition(Aherm)
    rebuilt = sum(l * Pj for l, Pj in zip(lam, projs))
    assert np.allclose(rebuilt, Aherm)
    print("eigenvalues of A:", lam)
    for l, Pj in zip(lam, projs):
        print(f"lambda = {l:.4f}, projector\n", Pj)
    # a function of the operator through its spectral decomposition
    expA = sum(np.exp(l) * Pj for l, Pj in zip(lam, projs))
    from math import factorial
    series = sum(np.linalg.matrix_power(Aherm, k) / factorial(k) for k in range(40))
    assert np.allclose(expA, series)
    print("exp(A) from the spectral decomposition agrees with the power series")
