"""Chapter 7, Secs. 7.2-7.5: the variational quantum eigensolver.

* ansatz classes that turn a parameter vector into a state of the state-vector
  simulator of chapter 2: ``HardwareEfficient`` (layers of R_y R_z rotations
  and a CNOT ladder, Eq. 7.hea), ``ProductState`` (one R_y per qubit, the
  mean field of Sec. 4.5.5) and ``PauliRotationAnsatz`` (a reference bit
  string followed by exp(-i theta_k G_k) for Hermitian generators G_k given
  as PauliSums: unitary coupled cluster, the Hamiltonian-variational ansatz
  and the operators picked by ADAPT-VQE all have this form);
* ``VQE``: the energy <psi(theta)|H|psi(theta)> from the exact state vector or
  from shots (the measurement circuits of chapter 4), the parameter-shift
  gradient with the two-term rule for generators with two eigenvalues and
  the four-term rule for the three-eigenvalue generators of the excitation
  operators (Eqs. 7.standardshift, 7.generalshift), and three optimisers:
  gradient descent, BFGS (common/optimize.py) and SPSA;
* ``run_restarts``: the best of several random initialisations.

Run as a script it checks the ansatz classes against the circuits of chapter 4,
the parameter-shift rules against finite differences, and the optimisers on
the two-qubit Hamiltonian of Eq. (4.twoqubitH).  A few seconds.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np
import gates as G
from pauli import PauliSum, string_matrix, weight
from statevector import StateVector
from circuit import Circuit
from optimize import minimize
from pauli_circuits import energy_from_shots, pauli_rotation


# ----------------------------------------------------------------- ansätze
class Ansatz:
    """Base class: ``state(theta)`` returns the 2^n amplitudes, ``shift(k)``
    the parameter-shift data of parameter k: ("two", r) for a generator with
    eigenvalues +-r, ("three", 1) for eigenvalues -1, 0, +1."""
    n_params = 0

    def state(self, theta):
        raise NotImplementedError

    def shift(self, k):
        return ("two", 0.5)

    def random_parameters(self, rng, scale=np.pi):
        return rng.uniform(-scale, scale, size=self.n_params)


class HardwareEfficient(Ansatz):
    """L layers of [R_y(theta) R_z(phi) on every qubit] followed by a CNOT ladder,
    Eq. (7.hea); 2 n L parameters, (n - 1) L CNOTs.  The last layer has no ladder."""

    def __init__(self, n, layers, entangler="ladder"):
        self.n, self.layers, self.entangler = n, layers, entangler
        self.n_params = 2 * n * layers

    def circuit(self, theta):
        circ = Circuit(self.n)
        k = 0
        for l in range(self.layers):
            for q in range(self.n):
                circ.ry(theta[k], q); circ.rz(theta[k + 1], q); k += 2
            if l < self.layers - 1:
                if self.entangler == "ladder":
                    for q in range(self.n - 1):
                        circ.cnot(q, q + 1)
                elif self.entangler == "ring":
                    for q in range(self.n):
                        circ.cnot(q, (q + 1) % self.n)
        return circ

    def state(self, theta):
        return self.circuit(theta).run().psi

    def cnots(self):
        per = self.n - 1 if self.entangler == "ladder" else self.n
        return per * (self.layers - 1)


class ProductState(Ansatz):
    """R_y(alpha_q) on every qubit of |0...0>: the mean-field product state of Sec. 4.5.5.
    With ``shared=True`` a single angle is used for all qubits."""

    def __init__(self, n, shared=False):
        self.n, self.shared = n, shared
        self.n_params = 1 if shared else n

    def state(self, theta):
        angles = [theta[0]] * self.n if self.shared else theta
        return StateVector.product(*[G.Ry(a) @ np.array([1, 0]) for a in angles]).psi


class PauliRotationAnsatz(Ansatz):
    """|psi(theta)> = prod_k exp(-i theta_k G_k) |reference>, the generators G_k
    being Hermitian PauliSums applied in the order given (the first generator
    acts first).  Covers unitary coupled cluster (G = i(T - T^dagger)), the
    Hamiltonian-variational ansatz (G = the commuting groups of H) and the
    ADAPT-VQE circuits.  Each exp(-i theta G) is applied through the spectral
    decomposition of the (cached) matrix of G, which is exact and, for the
    register sizes of the book, fast."""

    def __init__(self, n, reference, generators):
        self.n = n
        self.reference = StateVector.from_bits(reference).psi if isinstance(reference, str) else np.asarray(reference, dtype=complex)
        self.generators = []
        self._eig = []
        for g in generators:
            self.add(g)

    def add(self, g):
        self.generators.append(g)
        w, V = np.linalg.eigh(g.matrix())
        self._eig.append((w, V))
        self.n_params = len(self.generators)
        return self

    def apply(self, k, theta, psi):
        w, V = self._eig[k]
        return V @ (np.exp(-1j * theta * w) * (V.conj().T @ psi))

    def state(self, theta):
        psi = self.reference.copy()
        for k, t in enumerate(theta):
            psi = self.apply(k, t, psi)
        return psi

    def shift(self, k):
        w = np.round(self._eig[k][0], 10)
        vals = sorted(set(w))
        if len(vals) == 2 and abs(vals[0] + vals[1]) < 1e-9:
            return ("two", vals[1])
        if len(vals) == 3 and abs(vals[0] + 1) < 1e-9 and abs(vals[1]) < 1e-9 and abs(vals[2] - 1) < 1e-9:
            return ("three", 1.0)
        return ("numerical", None)

    def circuit(self, theta):
        """The same state as a gate sequence (only for generators that are sums of
        mutually commuting Pauli strings, which is the case for all of chapter 7)."""
        circ = Circuit(self.n)
        for q, b in enumerate(format(int(np.argmax(np.abs(self.reference))), f"0{self.n}b")):
            if b == "1":
                circ.x(q)
        for g, t in zip(self.generators, theta):
            for label, c in g.terms.items():
                pauli_rotation(circ, label, float(np.real(c)) * t)
        return circ


# ---------------------------------------------------------------- the loop
class VQE:
    """Minimise <psi(theta)|H|psi(theta)> for a PauliSum H over an Ansatz."""

    def __init__(self, H, ansatz, shots=None, rng=None):
        self.H, self.ansatz = H, ansatz
        self.Hmat = H.matrix()
        self.shots, self.rng = shots, (np.random.default_rng(0) if rng is None else rng)
        self.n_energy = 0
        self.history = []

    def energy(self, theta):
        """Exact expectation value, or the shot estimate of Sec. 4.6 with `shots` per group."""
        self.n_energy += 1
        psi = self.ansatz.state(theta)
        if self.shots is None:
            return float(np.real(np.vdot(psi, self.Hmat @ psi)))
        E, _ = energy_from_shots(self.H, StateVector(self.ansatz.n, psi), self.shots, self.rng)
        return E

    def exact_energy(self, theta):
        psi = self.ansatz.state(theta)
        return float(np.real(np.vdot(psi, self.Hmat @ psi)))

    def gradient(self, theta):
        """Parameter-shift gradient, Eq. (7.standardshift) or (7.generalshift)."""
        theta = np.asarray(theta, dtype=float)
        g = np.zeros_like(theta)
        for k in range(len(theta)):
            kind, r = self.ansatz.shift(k)
            if kind == "two":
                s = np.pi / (4 * r)
                g[k] = r * (self.energy(shifted(theta, k, s)) - self.energy(shifted(theta, k, -s)))
            elif kind == "three":                       # frequencies 1 and 2: R = 2
                R = 2
                for mu in range(1, 2 * R + 1):
                    x = (2 * mu - 1) * np.pi / (2 * R)
                    g[k] += (-1) ** (mu - 1) / (4 * R * np.sin(x / 2) ** 2) * self.energy(shifted(theta, k, x))
            else:
                h = 1e-5
                g[k] = (self.energy(shifted(theta, k, h)) - self.energy(shifted(theta, k, -h))) / (2 * h)
        return g

    # ------------------------------------------------------------ optimisers
    def gradient_descent(self, theta0, lr=0.1, iters=200, tol=1e-12):
        theta = np.array(theta0, dtype=float)
        self.history = [self.energy(theta)]
        for it in range(iters):
            theta = theta - lr * self.gradient(theta)
            self.history.append(self.energy(theta))
            if it > 5 and abs(self.history[-1] - self.history[-2]) < tol:
                break
        return theta, self.history[-1]

    def bfgs(self, theta0, tol=1e-12, gtol=1e-8, maxiter=500):
        theta, E, it = minimize(self.energy, np.array(theta0, dtype=float), grad=self.gradient, tol=tol, gtol=gtol, maxiter=maxiter)
        return theta, E

    def spsa(self, theta0, iters=300, a=0.2, c=0.2, alpha=0.602, gamma=0.101, rng=None):
        """Simultaneous perturbation: two energies per step whatever the number of parameters."""
        rng = self.rng if rng is None else rng
        theta = np.array(theta0, dtype=float)
        self.history = [self.energy(theta)]
        for k in range(1, iters + 1):
            ak, ck = a / k ** alpha, c / k ** gamma
            delta = rng.choice([-1.0, 1.0], size=len(theta))
            gk = (self.energy(theta + ck * delta) - self.energy(theta - ck * delta)) / (2 * ck) * delta
            theta = theta - ak * gk
            self.history.append(self.energy(theta))
        return theta, self.history[-1]


def shifted(theta, k, s):
    t = theta.copy(); t[k] += s
    return t


def run_restarts(H, ansatz, restarts=4, rng=None, method="bfgs", scale=np.pi, **kw):
    """Best of `restarts` random initialisations; returns (theta, E, VQE object)."""
    rng = np.random.default_rng(1) if rng is None else rng
    best = None
    for _ in range(restarts):
        vqe = VQE(H, ansatz)
        theta0 = ansatz.random_parameters(rng, scale)
        theta, E = vqe.bfgs(theta0, **kw) if method == "bfgs" else vqe.gradient_descent(theta0, **kw)
        if best is None or E < best[1]:
            best = (theta, E, vqe)
    return best


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    rng = np.random.default_rng(7)

    # ----------------------------------------------- the ansatz classes against chapter-4 circuits
    from pauli_circuits import compact_pairing, pair_excitation, prepare_reference
    from models import PairingModel
    H4 = compact_pairing(PairingModel(4, g=1.0))
    pairs = [(i, a) for i in (0, 1) for a in (2, 3)]

    def pair_generator(n, i, a):
        """G with exp(-i theta G) = exp(theta (S+_a S-_i - h.c.)): G = -(1/2)(Y_i X_a - X_i Y_a)."""
        return (-0.5) * (PauliSum.single(n, i, "Y") * PauliSum.single(n, a, "X")
                         - PauliSum.single(n, i, "X") * PauliSum.single(n, a, "Y"))
    ucc = PauliRotationAnsatz(4, "1100", [pair_generator(4, i, a) for i, a in pairs])
    t = rng.normal(size=4)
    circ = prepare_reference(4, [0, 1])
    for (i, a), tk in zip(pairs, t):
        pair_excitation(circ, i, a, tk)
    assert np.allclose(ucc.state(t), circ.run().psi)
    assert np.allclose(ucc.circuit(t).run().psi, circ.run().psi)
    print("PauliRotationAnsatz reproduces the pair-excitation circuit of chapter 4")
    hea = HardwareEfficient(3, 2)
    th = rng.normal(size=hea.n_params)
    assert abs(np.linalg.norm(hea.state(th)) - 1) < 1e-12 and hea.cnots() == 2
    prod = ProductState(4, shared=True)
    assert np.allclose(prod.state([0.0]), StateVector.from_bits("0000").psi)

    # ----------------------------------------------- parameter-shift rules against finite differences
    vqe = VQE(H4, ucc)
    g_shift = vqe.gradient(t)
    h = 1e-5
    g_fd = np.array([(vqe.energy(shifted(t, k, h)) - vqe.energy(shifted(t, k, -h))) / (2 * h) for k in range(4)])
    assert ucc.shift(0)[0] == "three" and np.abs(g_shift - g_fd).max() < 1e-8
    print(f"four-term rule for the pair-excitation generators (eigenvalues -1, 0, 1): |shift - fd| = {np.abs(g_shift - g_fd).max():.1e}")
    # the two-term rule applied to the same generator is wrong
    wrong = np.array([1.0 * (vqe.energy(shifted(t, k, np.pi / 4)) - vqe.energy(shifted(t, k, -np.pi / 4))) for k in range(4)])
    print(f"two-term rule on the same generators: |wrong - fd| = {np.abs(wrong - g_fd).max():.2e}")
    assert np.abs(wrong - g_fd).max() > 1e-3
    H2 = PauliSum({"II": 1.625, "ZI": -0.625, "IZ": -1.125, "ZZ": 0.125, "XX": 0.15, "YY": 0.15})
    hea = HardwareEfficient(2, 2)
    vqe = VQE(H2, hea)
    th = rng.normal(size=hea.n_params)
    g_fd = np.array([(vqe.energy(shifted(th, k, h)) - vqe.energy(shifted(th, k, -h))) / (2 * h) for k in range(len(th))])
    assert np.abs(vqe.gradient(th) - g_fd).max() < 1e-8
    print("two-term rule for the R_y / R_z rotations of the hardware-efficient ansatz agrees with finite differences")

    # ----------------------------------------------- the three optimisers on the two-qubit Hamiltonian
    E0 = np.linalg.eigvalsh(H2.matrix())[0]
    theta0 = hea.random_parameters(rng, 0.5)
    for name in ("gradient_descent", "bfgs", "spsa"):
        vqe = VQE(H2, hea)
        theta, E = getattr(vqe, name)(theta0) if name != "gradient_descent" else vqe.gradient_descent(theta0, lr=0.2, iters=300)
        print(f"  {name:>16s}: E = {vqe.exact_energy(theta):.8f} (exact {E0:.8f}), {vqe.n_energy} energy evaluations")
        assert vqe.exact_energy(theta) >= E0 - 1e-9
    print("VQE checks passed")
