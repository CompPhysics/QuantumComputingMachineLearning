"""Chapter 4, Sec. 4.1: Pauli strings and sums of Pauli strings.

A Pauli string on n qubits is a word over {I, X, Y, Z} such as "ZZXI"; qubit 0
is the leftmost letter, matching the tensor-product convention of chapters
2 and 3.  ``PauliSum`` is a dictionary {string: coefficient} with the
algebra of Eq. (4.paulialgebra) (products, sums, commutators), conversion to
and from dense matrices (Eq. 4.paulicoefficients, by the trace formula
c_P = Tr(P H)/2^n), and the bookkeeping the text uses: number of strings,
weight of a string, grouping into mutually commuting sets, expectation
values in a state vector.

Shared by chapters 4-8; ``from pauli import PauliSum``.
"""
import itertools
import numpy as np

_MAT = {"I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex)}

# single-qubit products: (a, b) -> (phase, c) with a b = phase * c
_PROD = {("I", "I"): (1, "I"), ("I", "X"): (1, "X"), ("I", "Y"): (1, "Y"), ("I", "Z"): (1, "Z"),
         ("X", "I"): (1, "X"), ("Y", "I"): (1, "Y"), ("Z", "I"): (1, "Z"),
         ("X", "X"): (1, "I"), ("Y", "Y"): (1, "I"), ("Z", "Z"): (1, "I"),
         ("X", "Y"): (1j, "Z"), ("Y", "Z"): (1j, "X"), ("Z", "X"): (1j, "Y"),
         ("Y", "X"): (-1j, "Z"), ("Z", "Y"): (-1j, "X"), ("X", "Z"): (-1j, "Y")}


def string_matrix(label):
    out = np.array([[1.0 + 0j]])
    for c in label:
        out = np.kron(out, _MAT[c])
    return out


def weight(label):
    return sum(1 for c in label if c != "I")


def strings_commute(a, b):
    """Two Pauli strings commute iff they anticommute on an even number of qubits."""
    anti = sum(1 for x, y in zip(a, b) if x != "I" and y != "I" and x != y)
    return anti % 2 == 0


def strings_qubitwise_commute(a, b):
    """Stronger: the letters agree wherever both are not I, so both strings can be
    read off from the same single-qubit basis rotations (Sec. 4.5)."""
    return all(x == "I" or y == "I" or x == y for x, y in zip(a, b))


class PauliSum:
    """H = sum_alpha h_alpha P_alpha as {label: coefficient}."""

    def __init__(self, terms=None, n_qubits=None):
        self.terms = {}
        if terms:
            for k, c in dict(terms).items():
                if abs(c) > 1e-14:
                    self.terms[k] = self.terms.get(k, 0) + c
        if n_qubits is None:
            n_qubits = len(next(iter(self.terms))) if self.terms else 0
        self.n = n_qubits

    # ------------------------------------------------------------ builders
    @classmethod
    def identity(cls, n):
        return cls({"I" * n: 1.0}, n)

    @classmethod
    def single(cls, n, qubit, letter, coeff=1.0):
        label = ["I"] * n; label[qubit] = letter
        return cls({"".join(label): coeff}, n)

    @classmethod
    def from_matrix(cls, M, n_qubits=None, tol=1e-12):
        """Exact decomposition c_P = Tr(P M)/2^n for every string P, Eq. (4.paulicoefficients).

        Uses the tensor structure: reshaping M into 2n indices and contracting
        one qubit at a time costs O(4^n n) instead of O(8^n)."""
        M = np.asarray(M, dtype=complex)
        n = int(np.log2(M.shape[0])) if n_qubits is None else n_qubits
        T = M.reshape((2,) * (2 * n))
        # bring row/column indices of each qubit together: (r0,c0,r1,c1,...)
        perm = [k for pair in zip(range(n), range(n, 2 * n)) for k in pair]
        T = T.transpose(perm).reshape((4,) * n)
        # coefficient tensor: contract each qubit index pair with the four Paulis
        basis = np.array([_MAT[c].reshape(4) for c in "IXYZ"]).conj() / 2   # (4 letters, 4 entries)
        for q in range(n):
            T = np.tensordot(T, basis, axes=([0], [1]))   # consumes axis 0, appends letter axis
        terms = {}
        for idx in itertools.product(range(4), repeat=n):
            c = T[idx]
            if abs(c) > tol:
                terms["".join("IXYZ"[k] for k in idx)] = c.real if abs(c.imag) < tol else c
        return cls(terms, n)

    # ------------------------------------------------------------- algebra
    def __add__(self, other):
        if not isinstance(other, PauliSum):
            other = other * PauliSum.identity(self.n)
        out = dict(self.terms)
        for k, c in other.terms.items():
            out[k] = out.get(k, 0) + c
        return PauliSum(out, self.n)

    __radd__ = __add__

    def __neg__(self):
        return PauliSum({k: -c for k, c in self.terms.items()}, self.n)

    def __sub__(self, other):
        return self + (-other if isinstance(other, PauliSum) else -other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if not isinstance(other, PauliSum):
            return PauliSum({k: c * other for k, c in self.terms.items()}, self.n)
        out = {}
        for a, ca in self.terms.items():
            for b, cb in other.terms.items():
                phase, letters = 1, []
                for x, y in zip(a, b):
                    ph, c = _PROD[(x, y)]
                    phase *= ph; letters.append(c)
                key = "".join(letters)
                out[key] = out.get(key, 0) + phase * ca * cb
        return PauliSum(out, self.n)

    __rmul__ = lambda self, other: self.__mul__(other)

    def dagger(self):
        return PauliSum({k: np.conj(c) for k, c in self.terms.items()}, self.n)

    def commutator(self, other):
        return self * other - other * self

    def simplify(self, tol=1e-12):
        return PauliSum({k: (c.real if abs(np.imag(c)) < tol else c)
                         for k, c in self.terms.items() if abs(c) > tol}, self.n)

    # ---------------------------------------------------------- properties
    def matrix(self):
        M = np.zeros((2 ** self.n, 2 ** self.n), dtype=complex)
        for k, c in self.terms.items():
            M += c * string_matrix(k)
        return M

    def n_strings(self):
        return len(self.terms)

    def max_weight(self):
        return max((weight(k) for k in self.terms), default=0)

    def weights(self):
        """Histogram {weight: number of strings}."""
        h = {}
        for k in self.terms:
            h[weight(k)] = h.get(weight(k), 0) + 1
        return dict(sorted(h.items()))

    def apply(self, psi):
        """H psi without forming the matrix: each string acts factor by factor on
        the tensor of amplitudes, O(2^n) per string (Sec. 4.2)."""
        psi = np.asarray(psi, dtype=complex).reshape((2,) * self.n)
        out = np.zeros_like(psi)
        for k, c in self.terms.items():
            phi = psi
            for q, letter in enumerate(k):
                if letter != "I":
                    phi = np.moveaxis(np.tensordot(_MAT[letter], phi, axes=([1], [q])), 0, q)
            out += c * phi
        return out.reshape(-1)

    def commutator_bound(self):
        """sum_{a<b} |h_a h_b| ||[P_a, P_b]|| = 2 sum over anticommuting pairs of |h_a h_b|,
        the quantity that bounds the first-order Trotter error (Sec. 4.5)."""
        keys = list(self.terms)
        total = 0.0
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                if not strings_commute(a, b):
                    total += 2 * abs(self.terms[a] * self.terms[b])
        return total

    def norm_bound(self):
        """sum_a |h_a| >= ||H||."""
        return sum(abs(c) for c in self.terms.values())

    def expectation(self, psi):
        psi = np.asarray(psi, dtype=complex)
        return sum(c * np.vdot(psi, string_matrix(k) @ psi) for k, c in self.terms.items())

    def commuting_groups(self, qubitwise=True):
        """Greedy partition of the strings into sets that commute pairwise;
        with qubitwise=True each set can be measured after one layer of
        single-qubit basis rotations (Sec. 4.5)."""
        test = strings_qubitwise_commute if qubitwise else strings_commute
        groups = []
        for k in sorted(self.terms, key=lambda s: -weight(s)):
            for g in groups:
                if all(test(k, m) for m in g):
                    g.append(k); break
            else:
                groups.append([k])
        return groups

    def __repr__(self):
        parts = []
        for k, c in sorted(self.terms.items(), key=lambda kv: (weight(kv[0]), kv[0])):
            cs = f"{c:+.4f}" if abs(np.imag(c)) < 1e-12 else f"+({c:.4f})"
            parts.append(f"{cs} {k}")
        return "\n".join(parts) if parts else "0"


if __name__ == "__main__":
    X0 = PauliSum.single(2, 0, "X"); Y0 = PauliSum.single(2, 0, "Y"); Z1 = PauliSum.single(2, 1, "Z")
    assert (X0 * Y0).terms == {"ZI": 1j}
    assert (X0 * Z1).terms == {"XZ": 1}
    assert X0.commutator(Y0).terms == {"ZI": 2j} and not X0.commutator(Z1).terms
    rng = np.random.default_rng(0)
    M = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8)); M = M + M.conj().T
    P = PauliSum.from_matrix(M)
    assert np.allclose(P.matrix(), M) and P.n_strings() == 64
    print("random Hermitian 8x8 matrix: 64 Pauli strings, weights", P.weights())
    H = PauliSum({"ZZ": 0.5, "XX": 0.25, "YY": 0.25, "IZ": -0.3})
    print("groups of commuting strings:", H.commuting_groups())
    psi = rng.normal(size=4) + 1j * rng.normal(size=4)
    assert np.allclose(H.apply(psi), H.matrix() @ psi)
    assert abs(H.commutator_bound() - 2 * (0.25 * 0.3 + 0.25 * 0.3)) < 1e-12   # IZ anticommutes with XX and YY
    print("matrix-free application and commutator bound checked")
    print("Pauli-sum checks passed")
