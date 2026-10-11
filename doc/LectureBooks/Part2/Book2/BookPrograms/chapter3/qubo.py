"""Chapter 3, Sec. 3.2: quadratic unconstrained binary optimization, the Ising
model and the cost Hamiltonian.

    QUBO     a symmetric matrix Q and an offset: C(x) = x^T Q x + offset for
             x in {0,1}^n, Eq. (3-qubo); built from a graph (MaxCut,
             Eq. 3-maxcutqubo) or from a mean-variance portfolio problem
             (Eq. 3-portfolioqubo); `to_ising()` is the substitution
             x_i = (1 - z_i)/2 of Eq. (3-qubotoising)
    Ising    fields h_i, couplings J_ij and a constant: H(z) = sum_i h_i z_i
             + sum_(i<j) J_ij z_i z_j + const, Eq. (3-ising); `energies()`
             is the diagonal of the cost Hamiltonian H_C of Eq. (3-costham)
             over all 2^n bit strings (qubit 0 = leftmost = most significant
             bit, bit 1 <-> z = -1), `terms()` its Pauli decomposition, used
             by the per-term parameter-shift rule of qaoa.py

Run as a script the module checks the worked example of Table 3-qubo and
the MaxCut Hamiltonian of the four-vertex ring.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]      # BookPrograms/
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import itertools
import numpy as np


def bitstrings(n):
    """All 2^n bit strings as an (2^n, n) integer array, index = binary number."""
    return np.array(list(itertools.product([0, 1], repeat=n)), dtype=int).reshape(2 ** n, n)


class Ising:
    """H(z) = sum_i h_i z_i + sum_(i<j) J_ij z_i z_j + const with z_i = +-1."""

    def __init__(self, h, J, const=0.0):
        self.h = np.asarray(h, float)
        self.J = np.triu(np.asarray(J, float), 1)         # upper triangle only
        self.const = float(const)
        self.n = len(self.h)

    def energy(self, z):
        z = np.asarray(z, float)
        return float(self.h @ z + z @ self.J @ z + self.const)

    def energies(self):
        """The diagonal of H_C = sum h_i Z_i + sum J_ij Z_i Z_j + const over the
        computational basis, Eq. (3-costham): bit 1 is z = -1."""
        z = 1 - 2 * bitstrings(self.n)
        return z @ self.h + np.einsum("ki,ij,kj->k", z, self.J, z) + self.const

    def terms(self):
        """The Pauli decomposition of H_C without the constant: a list of
        (coefficient, +-1 diagonal of the Pauli string, qubits) triples,
        one for every nonzero h_i Z_i and J_ij Z_i Z_j."""
        z = 1 - 2 * bitstrings(self.n)
        out = [(self.h[i], z[:, i].astype(float), (i,)) for i in range(self.n) if self.h[i] != 0]
        out += [(self.J[i, j], (z[:, i] * z[:, j]).astype(float), (i, j))
                for i in range(self.n) for j in range(i + 1, self.n) if self.J[i, j] != 0]
        return out

    def ground_state(self):
        """Brute force: the lowest energy and the bit strings that reach it."""
        E = self.energies()
        E0 = E.min()
        return E0, [k for k in range(len(E)) if np.isclose(E[k], E0)]


class QUBO:
    """C(x) = x^T Q x + offset, x in {0,1}^n, with Q symmetric, Eq. (3-qubo)."""

    def __init__(self, Q, offset=0.0):
        Q = np.asarray(Q, float)
        self.Q = (Q + Q.T) / 2                              # symmetrize
        self.offset = float(offset)
        self.n = Q.shape[0]

    def evaluate(self, x):
        x = np.asarray(x, float)
        return float(x @ self.Q @ x + self.offset)

    def costs(self):
        """C(x) for every bit string, in the order of `bitstrings(n)`."""
        X = bitstrings(self.n).astype(float)
        return np.einsum("ki,ij,kj->k", X, self.Q, X) + self.offset

    def to_ising(self):
        """x_i = (1 - z_i)/2: Eq. (3-qubotoising) for a symmetric Q."""
        Q, n = self.Q, self.n
        J = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                J[i, j] = Q[i, j] / 2                      # (Q_ij + Q_ji)/4
        h = np.array([-Q[i, i] / 2 - sum(Q[i, j] / 2 for j in range(n) if j != i) for i in range(n)])
        const = np.trace(Q) / 2 + sum(Q[i, j] / 2 for i in range(n) for j in range(i + 1, n)) + self.offset
        return Ising(h, J, const)

    # ------------------------------------------------------------ builders
    @classmethod
    def maxcut(cls, n, edges, weights=None):
        """The QUBO whose minimum is the maximum cut: C(x) = -sum_(ij) w_ij
        (x_i + x_j - 2 x_i x_j), Eq. (3-maxcutqubo)."""
        Q = np.zeros((n, n))
        for k, (i, j) in enumerate(edges):
            w = 1.0 if weights is None else weights[k]
            Q[i, i] -= w; Q[j, j] -= w
            Q[i, j] += w; Q[j, i] += w                     # x_i x_j appears twice in x^T Q x
        return cls(Q)

    @classmethod
    def portfolio(cls, mu, Sigma, lam, A, K):
        """Mean-variance selection of K of n assets with a quadratic budget
        penalty: C(x) = -mu^T x + lam x^T Sigma x + A (sum_i x_i - K)^2,
        Eq. (3-portfolioqubo)."""
        mu, Sigma = np.asarray(mu, float), np.asarray(Sigma, float)
        n = len(mu)
        Q = lam * Sigma + A * np.ones((n, n))
        Q[np.diag_indices(n)] += -mu - 2 * A * K         # x_i^2 = x_i absorbs the linear terms
        return cls(Q, offset=A * K ** 2)


def maxcut_value(z_bits, edges, weights=None):
    """The weight of the cut defined by a bit string."""
    return sum((1.0 if weights is None else weights[k]) for k, (i, j) in enumerate(edges)
               if z_bits[i] != z_bits[j])


if __name__ == "__main__":
    # Table 3-qubo: Q = [[1, -2], [-2, 1]], C(x) = x1 + x2 - 4 x1 x2
    q = QUBO([[1, -2], [-2, 1]])
    ising = q.to_ising()
    print("Worked example: Q = [[1,-2],[-2,1]]")
    print(f"  J_12 = {ising.J[0, 1]:+.3f}, h = {ising.h}, const = {ising.const:+.3f}")
    for x, c, e in zip(bitstrings(2), q.costs(), ising.energies()):
        print(f"  x = {x}, z = {1 - 2 * x}: C(x) = {c:+.1f}, H(z) + const = {e:+.1f}")
    assert np.allclose(q.costs(), ising.energies())

    # MaxCut on the four-vertex ring: H_C = -(1/2) sum (I - Z_i Z_j)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    ring = QUBO.maxcut(4, edges)
    isr = ring.to_ising()
    print("\nRing C4: h =", isr.h, " J_01, J_12, J_23, J_03 =",
          [isr.J[i, j] for (i, j) in [(0, 1), (1, 2), (2, 3), (0, 3)]], " const =", isr.const)
    cuts = np.array([maxcut_value(b, edges) for b in bitstrings(4)])
    assert np.allclose(isr.energies(), -cuts)
    E0, gs = isr.ground_state()
    print(f"  ground energy {E0:+.1f} = -(maximum cut), reached by",
          ["".join(map(str, bitstrings(4)[k])) for k in gs])
    print("  Pauli terms of H_C:", len(isr.terms()))
    print("QUBO / Ising checks passed")
