"""Chapter 3, Secs. 3.1-3.2: second quantization as matrices and as bit strings.

Two representations of the same algebra:

* ``FockSpace(n)`` builds the creation and annihilation operators of n
  fermionic modes as explicit 2^n x 2^n matrices, Eq. (3.mattensor):
  a_p = Z (x) ... (x) Z (x) a (x) I (x) ... (x) I with p factors of Z.  Mode 0
  is the leftmost tensor factor (the most significant bit of the basis index),
  the same convention as the StateVector of chapter 2, so that a Fock state
  |n_0 n_1 ... n_{n-1}> and the qubit state with the same bits are literally
  the same vector.  Useful up to n = 12 or so.

* ``SlaterBasis(n, N)`` stores the determinants of N particles in n modes as
  integers (bit strings) and applies a_p^dagger a_q with the sign rule
  (-1)^{l}, l = number of occupied modes between the two, Eq. (3.phase).
  This is how a full configuration interaction code works, and it never
  forms a 2^n-dimensional object.

Both build the one- and two-body operators of Eq. (3.hamiltonian) from a
one-body matrix h_pq and antisymmetrised two-body elements v_pqrs.

Shared by chapters 3, 4 and 7; ``from fockspace import FockSpace, SlaterBasis``.
"""
import itertools
import numpy as np

_a = np.array([[0, 1], [0, 0]], dtype=float)      # annihilates: |1> -> |0>
_Z = np.diag([1.0, -1.0])
_I = np.eye(2)


def kron_all(factors):
    out = np.array([[1.0]])
    for f in factors:
        out = np.kron(out, f)
    return out


class FockSpace:
    """Fermionic operators of n modes as dense 2^n x 2^n matrices."""

    def __init__(self, n_modes):
        self.n = n_modes
        self.dim = 2 ** n_modes
        self._a = [kron_all([_Z] * p + [_a] + [_I] * (self.n - p - 1)) for p in range(self.n)]

    def annihilate(self, p):
        return self._a[p]

    def create(self, p):
        return self._a[p].T

    def number(self, p):
        return self.create(p) @ self.annihilate(p)

    def total_number(self):
        return sum(self.number(p) for p in range(self.n))

    def vacuum(self):
        v = np.zeros(self.dim); v[0] = 1.0
        return v

    def determinant(self, occupied):
        """|p_1 p_2 ... p_N> = a_{p_1}^dagger ... a_{p_N}^dagger |0>, applied right to left."""
        v = self.vacuum()
        for p in reversed(list(occupied)):
            v = self.create(p) @ v
        return v

    def index(self, bits):
        """Basis index of the Fock state with occupation bits n_0 n_1 ... (mode 0 leftmost)."""
        return int("".join(str(int(b)) for b in bits), 2)

    def bits(self, k):
        return [int(c) for c in format(k, f"0{self.n}b")]

    def one_body(self, h):
        """sum_pq h_pq a_p^dagger a_q."""
        H = np.zeros((self.dim, self.dim), dtype=np.asarray(h).dtype)
        for p in range(self.n):
            for q in range(self.n):
                if h[p, q] != 0:
                    H = H + h[p, q] * self.create(p) @ self.annihilate(q)
        return H

    def two_body(self, v):
        """(1/4) sum_pqrs v_pqrs a_p^dagger a_q^dagger a_s a_r, v antisymmetrised."""
        H = np.zeros((self.dim, self.dim), dtype=np.asarray(v).dtype)
        for p, q, r, s in itertools.product(range(self.n), repeat=4):
            if v[p, q, r, s] != 0:
                H = H + 0.25 * v[p, q, r, s] * (self.create(p) @ self.create(q)
                                                 @ self.annihilate(s) @ self.annihilate(r))
        return H

    def hamiltonian(self, h, v=None):
        H = self.one_body(h)
        return H if v is None else H + self.two_body(v)

    def sector(self, N):
        """Indices of the basis states with exactly N particles."""
        return np.array([k for k in range(self.dim) if bin(k).count("1") == N])

    def check_anticommutators(self):
        worst = 0.0
        for p in range(self.n):
            for q in range(self.n):
                a, b = self.annihilate(p), self.annihilate(q)
                worst = max(worst, np.abs(a @ b + b @ a).max(),
                            np.abs(a @ b.T + b.T @ a - (p == q) * np.eye(self.dim)).max())
        return worst


# ------------------------------------------------------------- bit strings
class SlaterBasis:
    """All determinants of N particles in n modes, as integers."""

    def __init__(self, n_modes, n_particles):
        self.n, self.N = n_modes, n_particles
        self.states = [sum(1 << (self.n - 1 - p) for p in occ)
                       for occ in itertools.combinations(range(self.n), self.N)]
        self.states.sort()
        self.position = {s: k for k, s in enumerate(self.states)}
        self.dim = len(self.states)

    def bit(self, p):
        return 1 << (self.n - 1 - p)

    def occupied(self, state):
        return [p for p in range(self.n) if state & self.bit(p)]

    def label(self, state):
        return format(state, f"0{self.n}b")

    def excitation_level(self, state, reference):
        """Number of particle-hole pairs by which `state` differs from `reference`."""
        return bin(state & ~reference).count("1")

    def apply(self, ops, state):
        """Apply a product of operators, given right to left as [('+', p), ('-', q), ...]
        with '+' for a_p^dagger and '-' for a_p.  Returns (sign, new_state) or (0, None)."""
        sign = 1
        for kind, p in reversed(ops):
            b = self.bit(p)
            occ = bool(state & b)
            if (kind == "+" and occ) or (kind == "-" and not occ):
                return 0, None
            # particles to the left of p (modes < p) each contribute a sign
            left = state >> (self.n - p)
            if bin(left).count("1") % 2:
                sign = -sign
            state ^= b
        return sign, state

    def hamiltonian_matrix(self, h, v=None):
        """<K|H|L> for H = sum h_pq a_p^dag a_q + 1/4 sum v_pqrs a_p^dag a_q^dag a_s a_r."""
        H = np.zeros((self.dim, self.dim))
        n = self.n
        nz1 = [(p, q) for p in range(n) for q in range(n) if h[p, q] != 0]
        nz2 = []
        if v is not None:
            nz2 = [(p, q, r, s) for p, q, r, s in itertools.product(range(n), repeat=4) if v[p, q, r, s] != 0]
        for L, state in enumerate(self.states):
            for p, q in nz1:
                sign, new = self.apply([("+", p), ("-", q)], state)
                if sign:
                    H[self.position[new], L] += sign * h[p, q]
            for p, q, r, s in nz2:
                sign, new = self.apply([("+", p), ("+", q), ("-", s), ("-", r)], state)
                if sign:
                    H[self.position[new], L] += 0.25 * sign * v[p, q, r, s]
        return H

    def vector(self, occupied):
        """The determinant with the given occupied modes as a unit vector."""
        state = sum(self.bit(p) for p in occupied)
        v = np.zeros(self.dim); v[self.position[state]] = 1.0
        return v



if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    # one mode: Eq. (3.matsingle)
    fs = FockSpace(1)
    print("a =\n", fs.annihilate(0), "\na^dagger a =\n", fs.number(0))
    # two modes: Eq. (3.mattwo); the sign in a_1 comes from the Z factor
    fs = FockSpace(2)
    print("a_0 =\n", fs.annihilate(0), "\na_1 =\n", fs.annihilate(1))
    v = fs.create(0) @ fs.create(1) @ fs.vacuum()
    w = fs.create(1) @ fs.create(0) @ fs.vacuum()
    assert np.allclose(v, -w) and v[3] == 1
    print("a_0^dag a_1^dag|0> = e_3, a_1^dag a_0^dag|0> = -e_3: verified")
    for n in (2, 3, 4, 5):
        assert FockSpace(n).check_anticommutators() < 1e-14
    print("anticommutation relations hold as matrix identities for n = 2..5")
    # a random one- plus two-body Hamiltonian: matrices vs. bit strings agree
    rng = np.random.default_rng(0)
    n, N = 6, 3
    h = rng.normal(size=(n, n)); h = (h + h.T) / 2
    v = rng.normal(size=(n, n, n, n))
    v = v - v.transpose(1, 0, 2, 3); v = v - v.transpose(0, 1, 3, 2); v = (v + v.transpose(2, 3, 0, 1)) / 2
    fs = FockSpace(n)
    Hfull = fs.hamiltonian(h, v)
    sec = fs.sector(N)
    Hsec = Hfull[np.ix_(sec, sec)]
    sb = SlaterBasis(n, N)
    Hbits = sb.hamiltonian_matrix(h, v)
    assert np.allclose(np.linalg.eigvalsh(Hsec), np.linalg.eigvalsh(Hbits))
    assert np.allclose(Hfull, Hfull.T)
    print(f"N = {N} sector of {n} modes: {sb.dim} determinants; matrices and bit strings agree")
    print("particle number is conserved:", np.allclose(Hfull @ fs.total_number(), fs.total_number() @ Hfull))
