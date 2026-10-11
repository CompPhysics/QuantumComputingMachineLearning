"""Chapter 2, Secs. 2.3-2.6 and 2.9: a state-vector simulator for n qubits.

The state of n qubits is the vector of 2^n complex amplitudes
Psi_{i_1 i_2 ... i_n} of Eq. (2.nqubitstate).  Qubit 0 is the leftmost
label (the most significant bit of the basis-state index), so that
|i_1 i_2 ... i_n> has index i_1 2^{n-1} + ... + i_n, exactly as the tensor
products of Sec. 2.2.7 are written on paper.

A gate on qubit a acts by Eq. (2.onequbitaction): reshape the amplitude
vector into an n-index tensor and contract the index of qubit a with the
gate matrix.  Two-qubit gates contract two indices with a (2,2,2,2) tensor.
Measurement follows the projective postulate, Eqs. (2.bornrule) and
(2.collapse).

Shared by every chapter; ``from statevector import StateVector``.
"""
import numpy as np

import gates as G


class StateVector:
    """Amplitudes of an n-qubit register, with gates and measurements."""

    def __init__(self, n_qubits, amplitudes=None):
        self.n = n_qubits
        self.dim = 2 ** n_qubits
        if amplitudes is None:
            self.psi = np.zeros(self.dim, dtype=complex)
            self.psi[0] = 1.0                     # |00...0>
        else:
            psi = np.asarray(amplitudes, dtype=complex).reshape(self.dim)
            self.psi = psi / np.linalg.norm(psi)

    # ------------------------------------------------------------ builders
    @classmethod
    def from_bits(cls, bits):
        """|b_1 b_2 ... b_n> from a string such as '0110'."""
        n = len(bits)
        sv = cls(n)
        sv.psi[:] = 0
        sv.psi[int(bits, 2)] = 1.0
        return sv

    @classmethod
    def product(cls, *single_qubit_states):
        """Tensor product |psi_1> (x) |psi_2> (x) ... of single-qubit states."""
        psi = np.array([1.0 + 0j])
        for s in single_qubit_states:
            psi = np.kron(psi, np.asarray(s, dtype=complex))
        return cls(len(single_qubit_states), psi)

    def copy(self):
        return StateVector(self.n, self.psi.copy())

    # --------------------------------------------------------------- gates
    def apply(self, U, targets):
        """Apply the unitary U to the qubits listed in `targets`.

        U is a (2^k x 2^k) matrix and targets a list of k qubit labels, in the
        order of U's tensor factors.  Works for any k; used for 1, 2 and 3."""
        if isinstance(targets, (int, np.integer)):
            targets = [int(targets)]
        k = len(targets)
        U = np.asarray(U, dtype=complex).reshape((2,) * (2 * k))
        tensor = self.psi.reshape((2,) * self.n)
        # contract the "input" indices of U (the last k) with the target axes
        tensor = np.tensordot(U, tensor, axes=(list(range(k, 2 * k)), targets))
        # tensordot puts the k output indices first; move them back into place
        tensor = np.moveaxis(tensor, list(range(k)), targets)
        self.psi = tensor.reshape(self.dim)
        return self

    def apply_controlled(self, U, control, target):
        """Controlled-U with the given control and target qubits."""
        return self.apply(G.controlled(U), [control, target])

    def x(self, q): return self.apply(G.X, q)
    def y(self, q): return self.apply(G.Y, q)
    def z(self, q): return self.apply(G.Z, q)
    def h(self, q): return self.apply(G.H, q)
    def s(self, q): return self.apply(G.S, q)
    def t(self, q): return self.apply(G.T, q)
    def rx(self, theta, q): return self.apply(G.Rx(theta), q)
    def ry(self, theta, q): return self.apply(G.Ry(theta), q)
    def rz(self, theta, q): return self.apply(G.Rz(theta), q)
    def cnot(self, c, t): return self.apply(G.CNOT, [c, t])
    def cz(self, c, t): return self.apply(G.CZ, [c, t])
    def swap(self, a, b): return self.apply(G.SWAP, [a, b])

    # ---------------------------------------------------------- observables
    def probabilities(self):
        """|Psi_{i_1...i_n}|^2 for every basis state."""
        return np.abs(self.psi) ** 2

    def marginal(self, qubit):
        """P(qubit = 0), P(qubit = 1): Eq. (2.marginal)."""
        p = self.probabilities().reshape((2,) * self.n)
        axes = tuple(a for a in range(self.n) if a != qubit)
        return p.sum(axis=axes)

    def expectation(self, op, targets=None):
        """<Psi| op |Psi>, with op acting on all qubits (targets=None) or on a subset."""
        if targets is None:
            return float(np.real(np.vdot(self.psi, np.asarray(op) @ self.psi)))
        phi = self.copy().apply(op, targets)
        return float(np.real(np.vdot(self.psi, phi.psi)))

    def density_matrix(self):
        """rho = |Psi><Psi|."""
        return np.outer(self.psi, self.psi.conj())

    def overlap(self, other):
        return np.vdot(self.psi, other.psi)

    # ---------------------------------------------------------- measurement
    def measure(self, qubit, rng=None, outcome=None):
        """Projective measurement of one qubit in the computational basis.

        Draws the outcome with the probabilities of Eq. (2.marginal) (or takes
        the given `outcome`) and collapses the state by Eq. (2.collapse).
        Returns (outcome, probability)."""
        p = self.marginal(qubit)
        if outcome is None:
            rng = np.random.default_rng() if rng is None else rng
            outcome = int(rng.random() < p[1])
        tensor = self.psi.reshape((2,) * self.n)
        keep = [slice(None)] * self.n
        keep[qubit] = 1 - outcome
        tensor = tensor.copy()
        tensor[tuple(keep)] = 0.0
        self.psi = tensor.reshape(self.dim) / np.sqrt(p[outcome])
        return outcome, float(p[outcome])

    def measure_all(self, rng=None):
        """Measure every qubit; returns the bit string and collapses to it."""
        rng = np.random.default_rng() if rng is None else rng
        idx = rng.choice(self.dim, p=self.probabilities())
        self.psi[:] = 0
        self.psi[idx] = 1.0
        return format(idx, f"0{self.n}b")

    def sample(self, shots, rng=None):
        """Sample `shots` bit strings without collapsing the state."""
        rng = np.random.default_rng() if rng is None else rng
        idx = rng.choice(self.dim, size=shots, p=self.probabilities())
        counts = {}
        for i in idx:
            key = format(i, f"0{self.n}b")
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    # ---------------------------------------------------------------- print
    def __str__(self, tol=1e-10):
        terms = []
        for i, a in enumerate(self.psi):
            if abs(a) > tol:
                terms.append(f"({a.real:+.4f}{a.imag:+.4f}j)|{format(i, f'0{self.n}b')}>")
        return " + ".join(terms) if terms else "0"


if __name__ == "__main__":
    # Sec. 2.9: the Bell-state circuit H(0) CNOT(0,1) on |00>
    sv = StateVector(2).h(0).cnot(0, 1)
    print("Bell state |Phi+> =", sv)
    assert np.allclose(sv.psi, [1, 0, 0, 1] / np.sqrt(2))
    print("marginals of qubit 0:", sv.marginal(0))
    rng = np.random.default_rng(1)
    print("1000 shots:", sv.sample(1000, rng))
    # measuring qubit 0 fixes qubit 1 (Eq. 2.bellcollapse)
    post = sv.copy()
    out, p = post.measure(0, outcome=1)
    print(f"outcome {out} with probability {p}; post-measurement state:", post)
    assert np.allclose(post.psi, [0, 0, 0, 1])
    # a gate on the second of three qubits equals I (x) U (x) I on the full vector
    sv3 = StateVector(3, np.random.default_rng(0).normal(size=8) + 1j)
    U = G.Rn(0.9, [1, 2, 3])
    ref = np.kron(np.kron(G.I, U), G.I) @ sv3.psi
    assert np.allclose(sv3.copy().apply(U, 1).psi, ref)
    print("state-vector simulator checks passed")
