"""Chapter 3, Sec. 3.3: the model Hamiltonians of the book in second quantization.

Every model is the general Hamiltonian
    H = sum_pq h_pq a_p^dagger a_q + 1/4 sum_pqrs v_pqrs a_p^dagger a_q^dagger a_s a_r
with a restricted set of matrix elements (Eq. 3.generalH).  Each class
returns (h, v) in a spin-orbital basis, so that the same FockSpace /
SlaterBasis machinery diagonalises all of them, plus the symmetry-adapted
matrix that the text uses (quasispin block, seniority-zero block, S_z = 0
sector).  Mode orderings:

* Lipkin:   mode 2p + s,  s = 0 for the lower level (sigma = -1), s = 1 for the upper
* pairing:  mode 2(p-1) + s,  s = 0 for spin +, s = 1 for spin -   (partners adjacent)
* Hubbard:  mode 2i + s,  s = 0 for spin up, s = 1 for spin down
* Heisenberg: one qubit per site, no fermions

Shared by chapters 3, 4 and 7; ``from models import PairingModel, ...``.
"""
import itertools
import numpy as np

from fockspace import FockSpace, SlaterBasis


def add_term(v, P, Q, R, S, c):
    """Add c * a_P^dagger a_Q^dagger a_S a_R to the antisymmetrised tensor v."""
    v[P, Q, R, S] += c
    v[Q, P, S, R] += c
    v[Q, P, R, S] -= c
    v[P, Q, S, R] -= c


class Model:
    """Common interface: h, v, n_modes, and exact diagonalisation in a number sector."""

    def fock_hamiltonian(self):
        return FockSpace(self.n_modes).hamiltonian(self.h, self.v)

    def sector_hamiltonian(self, n_particles):
        basis = SlaterBasis(self.n_modes, n_particles)
        return basis, basis.hamiltonian_matrix(self.h, self.v)

    def exact(self, n_particles):
        """Lowest eigenvalue and eigenvector in the N-particle sector."""
        basis, H = self.sector_hamiltonian(n_particles)
        E, V = np.linalg.eigh(H)
        return E[0], V[:, 0], basis


# ------------------------------------------------------------------ Lipkin
class LipkinModel(Model):
    """N fermions in two Omega-fold degenerate levels, Eqs. (3.lipkinH)-(3.lipkinquasispin)."""

    def __init__(self, N, eps=1.0, V=0.0, W=0.0, Omega=None):
        self.N, self.eps, self.V, self.W = N, eps, V, W
        self.Omega = N if Omega is None else Omega
        self.n_modes = 2 * self.Omega
        n = self.n_modes
        self.h = np.zeros((n, n))
        for p in range(self.Omega):
            self.h[2 * p, 2 * p] = -eps / 2          # lower level, sigma = -1
            self.h[2 * p + 1, 2 * p + 1] = eps / 2   # upper level, sigma = +1
        self.v = np.zeros((n, n, n, n))
        for p in range(self.Omega):
            for pp in range(self.Omega):
                for s in (0, 1):                      # sigma = +-1 <-> s = 1, 0
                    ms = 1 - s
                    # H1: V/2 a+_{p s} a+_{p' s} a_{p' -s} a_{p -s}
                    if p != pp:
                        add_term(self.v, 2 * p + s, 2 * pp + s, 2 * p + ms, 2 * pp + ms, V / 2)
                    # H2: W/2 a+_{p s} a+_{p' -s} a_{p' s} a_{p -s}
                    if not (p == pp and s == ms):
                        add_term(self.v, 2 * p + s, 2 * pp + ms, 2 * p + ms, 2 * pp + s, W / 2)

    def mode(self, p, sigma):
        return 2 * p + (1 if sigma > 0 else 0)

    def quasispin_matrix(self, J=None):
        """H in the basis |J, J_z>, J_z = -J..J, Eq. (3.lipkinquasispin) with Eq. (3.ladder)."""
        J = self.N / 2 if J is None else J
        m = np.arange(-J, J + 1)
        d = len(m)
        Jz = np.diag(m)
        Jp = np.zeros((d, d))
        for k in range(d - 1):
            Jp[k + 1, k] = np.sqrt(J * (J + 1) - m[k] * (m[k] + 1))
        Jm = Jp.T
        Nop = self.N * np.eye(d)
        return (self.eps * Jz + 0.5 * self.V * (Jp @ Jp + Jm @ Jm)
                + 0.5 * self.W * (-Nop + Jp @ Jm + Jm @ Jp))

    def reference(self):
        """All particles in the lower level: modes 0, 2, 4, ..."""
        return [2 * p for p in range(self.N)]


# ----------------------------------------------------------------- pairing
class PairingModel(Model):
    """L doubly degenerate levels, pair interaction of strength g, Eq. (3.pairingH),
    optionally the pair-breaking term of strength f, Eq. (3.Vph)."""

    def __init__(self, L, g=1.0, xi=1.0, f=0.0):
        self.L, self.g, self.xi, self.f = L, g, xi, f
        self.n_modes = 2 * L
        n = self.n_modes
        self.h = np.zeros((n, n))
        for p in range(L):
            for s in (0, 1):
                self.h[2 * p + s, 2 * p + s] = xi * p        # level p+1 has energy xi*(p)
        self.v = np.zeros((n, n, n, n))
        for p in range(L):
            for q in range(L):
                # -g/2 a+_{p+} a+_{p-} a_{q-} a_{q+}
                add_term(self.v, 2 * p, 2 * p + 1, 2 * q, 2 * q + 1, -g / 2)
        if f != 0:
            for p in range(L):
                for q in range(L):
                    for r in range(L):
                        # -f/2 a+_{p+} a+_{p-} a_{q-} a_{r+}  + h.c.
                        add_term(self.v, 2 * p, 2 * p + 1, 2 * r, 2 * q + 1, -f / 2)
                        add_term(self.v, 2 * r, 2 * q + 1, 2 * p, 2 * p + 1, -f / 2)

    def plus(self, p): return 2 * p          # spin-up partner of level p (0-based)
    def minus(self, p): return 2 * p + 1

    def reference(self, n_pairs):
        return [m for p in range(n_pairs) for m in (2 * p, 2 * p + 1)]

    def seniority_zero_matrix(self, n_pairs):
        """H in the basis of pair configurations |p_1 ... p_n>, Eq. (3.pairingmatrix):
        diagonal 2 xi sum (p-1) - g n/2 ... written for general L, n."""
        configs = list(itertools.combinations(range(self.L), n_pairs))
        d = len(configs)
        H = np.zeros((d, d))
        for a, ca in enumerate(configs):
            H[a, a] = 2 * self.xi * sum(ca) - 0.5 * self.g * n_pairs
            for b, cb in enumerate(configs):
                if a != b and len(set(ca) & set(cb)) == n_pairs - 1:
                    H[a, b] = -0.5 * self.g
        return configs, H


# ----------------------------------------------------------------- Hubbard
class HubbardModel(Model):
    """Fermi-Hubbard model on a list of bonds, Eq. (3.hubbardH)."""

    def __init__(self, n_sites, t=1.0, U=0.0, periodic=True, bonds=None):
        self.n_sites, self.t, self.U = n_sites, t, U
        if bonds is None:
            bonds = [(i, i + 1) for i in range(n_sites - 1)]
            if periodic and n_sites > 2:
                bonds.append((n_sites - 1, 0))
        self.bonds = bonds
        self.n_modes = 2 * n_sites
        n = self.n_modes
        self.h = np.zeros((n, n))
        for i, j in bonds:
            for s in (0, 1):
                self.h[2 * i + s, 2 * j + s] += -t
                self.h[2 * j + s, 2 * i + s] += -t
        self.v = np.zeros((n, n, n, n))
        for i in range(n_sites):
            # U n_{i up} n_{i down} = U a+_{i up} a+_{i down} a_{i down} a_{i up}
            add_term(self.v, 2 * i, 2 * i + 1, 2 * i, 2 * i + 1, U)

    def up(self, i): return 2 * i
    def down(self, i): return 2 * i + 1

    def double_occupancy(self, basis, vector):
        """<sum_i n_{i up} n_{i down}> in a SlaterBasis eigenvector."""
        total = 0.0
        for k, state in enumerate(basis.states):
            occ = basis.occupied(state)
            d = sum(1 for i in range(self.n_sites) if 2 * i in occ and 2 * i + 1 in occ)
            total += d * vector[k] ** 2
        return total


# -------------------------------------------------------------- Heisenberg
class HeisenbergModel:
    """Spin-1/2 Heisenberg chain/ring, H = J sum S_i.S_j - h sum S_i^z, Eq. (3.heisenbergH)."""

    def __init__(self, n_sites, J=1.0, h=0.0, periodic=True, bonds=None):
        self.n_sites, self.J, self.hfield = n_sites, J, h
        if bonds is None:
            bonds = [(i, i + 1) for i in range(n_sites - 1)]
            if periodic and n_sites > 2:
                bonds.append((n_sites - 1, 0))
        self.bonds = bonds
        self.n_qubits = n_sites

    def full_matrix(self):
        """Dense 2^N x 2^N matrix from Pauli matrices (N <= 12)."""
        import gates as G
        n = self.n_sites
        H = np.zeros((2 ** n, 2 ** n), dtype=complex)
        for i, j in self.bonds:
            for P in (G.X, G.Y, G.Z):
                ops = [G.I] * n; ops[i] = P; ops[j] = P
                H += 0.25 * self.J * G.kron(*ops)
        for i in range(n):
            ops = [G.I] * n; ops[i] = G.Z
            H -= 0.5 * self.hfield * G.kron(*ops)
        return H

    def sector_matrix(self, n_up=None):
        """H in the S^z_tot sector with n_up spins up, built from bit strings."""
        n = self.n_sites
        n_up = n // 2 if n_up is None else n_up
        states = [sum(1 << (n - 1 - i) for i in occ) for occ in itertools.combinations(range(n), n_up)]
        pos = {s: k for k, s in enumerate(states)}
        d = len(states)
        H = np.zeros((d, d))
        for k, s in enumerate(states):
            diag = 0.0
            for i, j in self.bonds:
                bi, bj = (s >> (n - 1 - i)) & 1, (s >> (n - 1 - j)) & 1
                if bi == bj:
                    diag += 0.25 * self.J
                else:
                    diag -= 0.25 * self.J
                    flipped = s ^ (1 << (n - 1 - i)) ^ (1 << (n - 1 - j))
                    H[pos[flipped], k] += 0.5 * self.J
            diag -= 0.5 * self.hfield * (2 * n_up - n)
            H[k, k] += diag
        return states, H

    def correlation(self, states, vector, i, j):
        """<S_i . S_j> in an S^z-sector eigenvector."""
        n = self.n_sites
        pos = {s: k for k, s in enumerate(states)}
        total = 0.0
        for k, s in enumerate(states):
            bi, bj = (s >> (n - 1 - i)) & 1, (s >> (n - 1 - j)) & 1
            if bi == bj:
                total += 0.25 * vector[k] ** 2
            else:
                total -= 0.25 * vector[k] ** 2
                flipped = s ^ (1 << (n - 1 - i)) ^ (1 << (n - 1 - j))
                total += 0.5 * vector[k] * vector[pos[flipped]]
        return total


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True, linewidth=120)
    # Lipkin, N = 4, eps = 2, V = -1/3, W = -1/4: quasispin block vs full diagonalisation
    lip = LipkinModel(4, eps=2.0, V=-1 / 3, W=-1 / 4)
    Hq = lip.quasispin_matrix()
    print("Lipkin J=2 quasispin matrix:\n", Hq)
    Eq = np.linalg.eigvalsh(Hq)
    E0, v0, basis = lip.exact(4)
    print("quasispin eigenvalues:", Eq, " lowest from 70x70 FCI:", E0)
    assert abs(Eq[0] - E0) < 1e-10
    # pairing: seniority-zero 6x6 vs full 70x70
    pm = PairingModel(4, g=1.0)
    configs, H6 = pm.seniority_zero_matrix(2)
    E6 = np.linalg.eigvalsh(H6)
    E0, v0, basis = pm.exact(4)
    print("pairing L=4, g=1: seniority-zero eigenvalues", E6, " full-space lowest", E0)
    assert abs(E6[0] - E0) < 1e-10 and abs(E0 - 0.63554847) < 1e-7
    # pairing + particle-hole term, table value
    E0, _, _ = PairingModel(4, g=1.0, f=0.05).exact(4)
    assert abs(E0 - 0.45058234) < 1e-7
    print("pairing + p-h, g=1, f=0.05:", E0)
    # Hubbard four-site ring at U = 4t
    hub = HubbardModel(4, t=1.0, U=4.0)
    E0, v0, basis = hub.exact(4)
    print("Hubbard ring, 4 sites, U=4t: E0 =", E0, "double occupancy", hub.double_occupancy(basis, v0))
    assert abs(E0 + 2.10274848) < 1e-7
    # Heisenberg ring, 4 sites: E0 = -2J, full matrix and sector matrix agree
    heis = HeisenbergModel(4)
    Ef = np.linalg.eigvalsh(heis.full_matrix())[0]
    states, Hs = heis.sector_matrix()
    Es, Vs = np.linalg.eigh(Hs)
    print("Heisenberg ring, 4 sites: E0 =", Ef, Es[0], " <S1.S2> =", heis.correlation(states, Vs[:, 0], 0, 1))
    assert abs(Ef + 2) < 1e-12 and abs(Es[0] + 2) < 1e-12
    print("model checks passed")
