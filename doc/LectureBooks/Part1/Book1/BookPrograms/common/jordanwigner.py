"""Chapter 4, Secs. 4.2-4.3: the Jordan-Wigner transformation as Pauli sums.

``JordanWigner(n_modes)`` returns the creation, annihilation and number
operators of mode p as ``PauliSum`` objects,

    a_p         = Z_0 ... Z_{p-1} (X_p + i Y_p)/2 = Z^{(p)} sigma^-_p,
    a_p^dagger  = Z_0 ... Z_{p-1} (X_p - i Y_p)/2 = Z^{(p)} sigma^+_p,
    n_p         = (I - Z_p)/2,

(Eq. 4.jwmap) with |1> = occupied and qubit 0 the leftmost tensor factor,
the same convention as ``fockspace.FockSpace``.  ``map_hamiltonian(h, v)``
turns the general second-quantised Hamiltonian
    H = sum_pq h_pq a_p^dagger a_q + 1/4 sum_pqrs v_pqrs a_p^dagger a_q^dagger a_s a_r
into a Pauli sum (Eq. 4.jwhamiltonian).  The parity mapping is included for
comparison (Eq. 4.parity).  Shared by chapters 4-8.
"""
import numpy as np
from pauli import PauliSum


class JordanWigner:
    """Fermionic ladder operators on n_modes qubits, Eq. (4.jwmap)."""

    def __init__(self, n_modes):
        self.n = n_modes
        self._cache = {}

    def _zstring(self, p):
        label = ["Z"] * p + ["I"] * (self.n - p)
        return PauliSum({"".join(label): 1.0}, self.n)

    def sigma_plus(self, p):
        """sigma^+_p = (X_p - i Y_p)/2 = |1><0| on qubit p (creates an occupation)."""
        return (PauliSum.single(self.n, p, "X", 0.5) + PauliSum.single(self.n, p, "Y", -0.5j))

    def sigma_minus(self, p):
        return (PauliSum.single(self.n, p, "X", 0.5) + PauliSum.single(self.n, p, "Y", 0.5j))

    def create(self, p):
        if ("+", p) not in self._cache:
            self._cache[("+", p)] = self._zstring(p) * self.sigma_plus(p)
        return self._cache[("+", p)]

    def annihilate(self, p):
        if ("-", p) not in self._cache:
            self._cache[("-", p)] = self._zstring(p) * self.sigma_minus(p)
        return self._cache[("-", p)]

    def number(self, p):
        return 0.5 * PauliSum.identity(self.n) - PauliSum.single(self.n, p, "Z", 0.5)

    def total_number(self):
        N = PauliSum(n_qubits=self.n)
        for p in range(self.n):
            N = N + self.number(p)
        return N

    def one_body(self, h):
        H = PauliSum(n_qubits=self.n)
        for p in range(self.n):
            for q in range(self.n):
                if abs(h[p, q]) > 1e-14:
                    H = H + h[p, q] * (self.create(p) * self.annihilate(q))
        return H.simplify()

    def two_body(self, v):
        """1/4 sum v_pqrs a+_p a+_q a_s a_r, v antisymmetrised."""
        n = self.n
        H = PauliSum(n_qubits=n)
        for p in range(n):
            for q in range(p + 1, n):            # use antisymmetry: p<q, r<s, factor 4/4 = 1
                cpq = self.create(p) * self.create(q)
                for r in range(n):
                    for s in range(r + 1, n):
                        if abs(v[p, q, r, s]) > 1e-14:
                            H = H + v[p, q, r, s] * (cpq * (self.annihilate(s) * self.annihilate(r)))
        return H.simplify()

    def map_hamiltonian(self, h, v=None):
        H = self.one_body(h)
        if v is not None:
            H = H + self.two_body(v)
        return H.simplify()


class ParityMapping(JordanWigner):
    """Parity encoding: qubit j stores the parity of modes 0..j, Eq. (4.parity).
    Then the occupation of mode p is (I - Z_{p-1} Z_p)/2 and the Z-string of
    Jordan-Wigner becomes an X-string on the qubits *after* p (update string)."""

    def _ladder(self, p, sign):
        n = self.n
        # local part: (Z_{p-1} X_p -/+ i Y_p)/2 with Z_{-1} = I
        loc = ["I"] * n; loc[p] = "X"
        if p > 0:
            loc[p - 1] = "Z"
        locy = ["I"] * n; locy[p] = "Y"
        op = PauliSum({"".join(loc): 0.5, "".join(locy): -0.5j * sign}, n)
        upd = ["I"] * p + ["I"] + ["X"] * (n - p - 1)
        return PauliSum({"".join(upd): 1.0}, n) * op

    def create(self, p):
        if ("+", p) not in self._cache:
            self._cache[("+", p)] = self._ladder(p, +1)
        return self._cache[("+", p)]

    def annihilate(self, p):
        if ("-", p) not in self._cache:
            self._cache[("-", p)] = self._ladder(p, -1)
        return self._cache[("-", p)]

    def number(self, p):
        n = self.n
        label = ["I"] * n; label[p] = "Z"
        if p > 0:
            label[p - 1] = "Z"
        return 0.5 * PauliSum.identity(n) - PauliSum({"".join(label): 0.5}, n)


class BravyiKitaev:
    """Bravyi-Kitaev encoding: qubit j stores the parity of the modes in a
    binary-tree set S(j), b = beta n (mod 2) with the recursive matrix
    beta_{2k} = [[beta_k, 0], [1-row, beta_k]] (Seeley, Richard, Love 2012).
    Operators are obtained by permuting the Fock basis and decomposing with
    PauliSum.from_matrix, so the class is exact but limited to n <= 8."""

    def __init__(self, n_modes):
        from fockspace import FockSpace
        self.n = n_modes
        k = 1
        while k < n_modes:
            k *= 2
        beta = np.array([[1]])
        while beta.shape[0] < k:
            m = beta.shape[0]
            top = np.hstack([beta, np.zeros((m, m), int)])
            low = np.hstack([np.zeros((m, m), int), beta]); low[-1, :m] = 1
            beta = np.vstack([top, low])
        self.beta = beta[:n_modes, :n_modes]
        # permutation of basis states: occupation bits n -> encoded bits b
        dim = 2 ** n_modes
        P = np.zeros((dim, dim))
        for idx in range(dim):
            bits = np.array([int(c) for c in format(idx, f"0{n_modes}b")])
            enc = self.beta @ bits % 2
            P[int("".join(map(str, enc)), 2), idx] = 1.0
        self.P = P
        self.fs = FockSpace(n_modes)

    def encode(self, bits):
        return "".join(map(str, self.beta @ np.array([int(c) for c in bits]) % 2))

    def create(self, p):
        return PauliSum.from_matrix(self.P @ self.fs.create(p) @ self.P.T)

    def annihilate(self, p):
        return self.create(p).dagger()

    def number(self, p):
        return PauliSum.from_matrix(self.P @ self.fs.number(p) @ self.P.T)

    def map_hamiltonian(self, h, v=None):
        return PauliSum.from_matrix(self.P @ self.fs.hamiltonian(h, v) @ self.P.T)


if __name__ == "__main__":
    from fockspace import FockSpace
    n = 4
    jw, fs = JordanWigner(n), FockSpace(n)
    for p in range(n):
        assert np.allclose(jw.create(p).matrix(), fs.create(p))
        assert np.allclose(jw.number(p).matrix(), fs.number(p))
    print("Jordan-Wigner a_p^dagger equals the Fock-space matrices of chapter 3 for n =", n)
    # anticommutators as Pauli sums
    for p in range(n):
        for q in range(n):
            ac = jw.annihilate(p) * jw.create(q) + jw.create(q) * jw.annihilate(p)
            assert ac.simplify().terms == ({"I" * n: 1.0} if p == q else {})
            aa = jw.annihilate(p) * jw.annihilate(q) + jw.annihilate(q) * jw.annihilate(p)
            assert not aa.simplify().terms
    print("{a_p, a_q^dagger} = delta_pq, {a_p, a_q} = 0 verified symbolically")
    # parity mapping reproduces the same algebra
    pm = ParityMapping(n)
    for p in range(n):
        for q in range(n):
            ac = (pm.annihilate(p) * pm.create(q) + pm.create(q) * pm.annihilate(p)).simplify()
            assert ac.terms == ({"I" * n: 1.0} if p == q else {})
    print("parity mapping: anticommutators verified")
    bk = BravyiKitaev(n)
    for p in range(n):
        for q in range(n):
            ac = (bk.annihilate(p) * bk.create(q) + bk.create(q) * bk.annihilate(p)).simplify()
            assert ac.terms == ({"I" * n: 1.0} if p == q else {})
    print("Bravyi-Kitaev mapping: anticommutators verified;", "|1011> is stored as", bk.encode("1011"))
    print("Jordan-Wigner checks passed")
