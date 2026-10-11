"""Chapter 2, Sec. 2.5: density matrices, partial traces and entropies.

Implements the definitions of Eqs. (2.rhodef)-(2.rhoevolve): the density
matrix of a pure state or of an ensemble, its purity Tr(rho^2), the Bloch
vector of a single qubit, Born-rule probabilities Tr(P rho), the
post-measurement state P rho P / p, unitary evolution U rho U^dagger, the
partial trace over a subset of qubits and the von Neumann entropy.

Shared by every chapter; ``from density_matrix import DensityMatrix``.
"""
import numpy as np

import gates as G


def shannon_entropy(p, base=2):
    """H = -sum p log p for a probability vector p (Eq. 2.shannon)."""
    p = np.asarray(p, dtype=float)
    p = p[p > 1e-15]
    return float(max(0.0, -np.sum(p * np.log(p)) / np.log(base)))


class DensityMatrix:
    """rho for n qubits, stored as a 2^n x 2^n complex matrix."""

    def __init__(self, rho, n_qubits=None):
        self.rho = np.asarray(rho, dtype=complex)
        self.dim = self.rho.shape[0]
        self.n = int(np.log2(self.dim)) if n_qubits is None else n_qubits

    # ------------------------------------------------------------ builders
    @classmethod
    def from_state(cls, psi):
        """rho = |psi><psi| for a normalised state vector (or a StateVector)."""
        psi = np.asarray(getattr(psi, "psi", psi), dtype=complex)
        return cls(np.outer(psi, psi.conj()))

    @classmethod
    def from_ensemble(cls, probabilities, states):
        """rho = sum_i p_i |psi_i><psi_i|, Eq. (2.rhodef)."""
        rho = sum(p * np.outer(np.asarray(s, dtype=complex), np.asarray(s).conj())
                  for p, s in zip(probabilities, states))
        return cls(rho)

    @classmethod
    def from_bloch(cls, r):
        """Single-qubit state from its Bloch vector, Eq. (2.rhobloch)."""
        r = np.asarray(r, dtype=float)
        return cls(0.5 * (G.I + r[0] * G.X + r[1] * G.Y + r[2] * G.Z))

    @classmethod
    def maximally_mixed(cls, n_qubits):
        return cls(np.eye(2 ** n_qubits) / 2 ** n_qubits)

    # ---------------------------------------------------------- properties
    def trace(self):
        return float(np.real(np.trace(self.rho)))

    def purity(self):
        """Tr(rho^2): 1 for a pure state, 1/d for the maximally mixed state."""
        return float(np.real(np.trace(self.rho @ self.rho)))

    def is_pure(self, tol=1e-10):
        return abs(self.purity() - 1) < tol

    def is_valid(self, tol=1e-10):
        """Hermitian, unit trace, positive semidefinite."""
        herm = np.allclose(self.rho, self.rho.conj().T, atol=tol)
        eig = np.linalg.eigvalsh(self.rho)
        return herm and abs(self.trace() - 1) < tol and eig.min() > -tol

    def eigenvalues(self):
        return np.clip(np.linalg.eigvalsh(self.rho), 0, None)

    def bloch_vector(self):
        """(<X>, <Y>, <Z>) for a single qubit."""
        assert self.n == 1
        return np.array([self.expectation(P) for P in (G.X, G.Y, G.Z)])

    def expectation(self, op):
        """<O> = Tr(rho O), Eq. (2.rhoexpect)."""
        return float(np.real(np.trace(self.rho @ np.asarray(op))))

    def von_neumann_entropy(self, base=2):
        """S = -Tr(rho log rho) = -sum lambda_i log lambda_i, Eq. (2.vonneumann)."""
        return shannon_entropy(self.eigenvalues(), base)

    # ------------------------------------------------------------ dynamics
    def evolve(self, U):
        """rho -> U rho U^dagger, Eq. (2.rhoevolve)."""
        U = np.asarray(U)
        return DensityMatrix(U @ self.rho @ U.conj().T, self.n)

    def apply(self, U, targets):
        """Apply a gate on a subset of qubits (via the full operator)."""
        from statevector import StateVector   # local import: avoid cycles
        # build the full operator by applying U to each basis vector
        full = np.zeros((self.dim, self.dim), dtype=complex)
        for col in range(self.dim):
            sv = StateVector(self.n, np.eye(self.dim)[col])
            full[:, col] = sv.apply(U, targets).psi
        return self.evolve(full)

    def measure(self, projectors):
        """Probabilities Tr(P_k rho) and post-measurement states P_k rho P_k / p_k."""
        out = []
        for P in projectors:
            P = np.asarray(P)
            p = float(np.real(np.trace(P @ self.rho)))
            post = DensityMatrix(P @ self.rho @ P / p, self.n) if p > 1e-15 else None
            out.append((p, post))
        return out

    def partial_trace(self, keep):
        """Trace out every qubit not in `keep`; Eq. (2.partialtrace)."""
        keep = [keep] if isinstance(keep, int) else sorted(keep)
        trace_out = [q for q in range(self.n) if q not in keep]
        t = self.rho.reshape((2,) * (2 * self.n))
        # contract row index and column index of each traced qubit
        for k, q in enumerate(trace_out):
            axis_row = q - k                      # axes already removed shift left
            axis_col = axis_row + (self.n - k)
            t = np.trace(t, axis1=axis_row, axis2=axis_col)
        d = 2 ** len(keep)
        return DensityMatrix(t.reshape(d, d), len(keep))

    def __repr__(self):
        with np.printoptions(precision=4, suppress=True):
            return f"DensityMatrix({self.n} qubits)\n{self.rho}"


if __name__ == "__main__":
    # a pure state and its Bloch vector
    theta, phi = 1.1, 0.4
    psi = [np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)]
    rho = DensityMatrix.from_state(psi)
    r = rho.bloch_vector()
    assert np.allclose(r, [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    assert rho.is_pure() and rho.is_valid()
    # an ensemble: 50/50 of |0> and |1> is the maximally mixed state
    mixed = DensityMatrix.from_ensemble([0.5, 0.5], [[1, 0], [0, 1]])
    assert np.allclose(mixed.rho, np.eye(2) / 2) and abs(mixed.purity() - 0.5) < 1e-12
    assert abs(mixed.von_neumann_entropy() - 1) < 1e-12
    # Bell state: reduced state is maximally mixed, entanglement entropy 1 bit
    bell = DensityMatrix.from_state(np.array([1, 0, 0, 1]) / np.sqrt(2))
    rhoA = bell.partial_trace(0)
    assert np.allclose(rhoA.rho, np.eye(2) / 2)
    print("rho_A for |Phi+>:\n", rhoA.rho.real)
    print("S(rho_A) =", rhoA.von_neumann_entropy(), "bit")
    # product state: reduced state stays pure
    prod = DensityMatrix.from_state(np.kron(psi, [1 / np.sqrt(2), 1j / np.sqrt(2)]))
    assert prod.partial_trace(0).is_pure() and prod.partial_trace(1).is_pure()
    # measurement of qubit 0 of the Bell state
    P0 = np.kron(np.diag([1, 0]), np.eye(2)); P1 = np.kron(np.diag([0, 1]), np.eye(2))
    for (p, post), expect in zip(bell.measure([P0, P1]), (0, 3)):
        assert abs(p - 0.5) < 1e-12 and abs(post.rho[expect, expect] - 1) < 1e-12
    print("density-matrix checks passed")
