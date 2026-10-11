"""Chapter 3, Secs. 3.3-3.5: the quantum approximate optimization algorithm.

The class `QAOA` prepares the depth-p state of Eq. (3-qaoastate),

    |psi_p(gamma, beta)> = prod_k exp(-i beta_k H_M) exp(-i gamma_k H_C) |+>^n,

on the state-vector simulator of common/statevector.py for a cost
Hamiltonian given as an `Ising` object of qubo.py (diagonal in the
computational basis) and the transverse-field mixer H_M = sum_i X_i.  The
phase separator is an elementwise phase on the 2^n amplitudes, the mixer an
R_x(2 beta) on every qubit.  The class evaluates the energy
E = <psi_p|H_C|psi_p> of Eq. (3-objective), its gradient by finite
differences, by the naive global shift of Eq. (3-globalshift) and by the
exact per-term parameter-shift rule of Eq. (3-pertermshift), optimizes the
2p angles with BFGS (common/optimize.py) or with gradient descent from
random restarts, and samples the output distribution.  `gate_count` is the
depth budget of Eq. (3-gatecount).
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]      # BookPrograms/
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np

from statevector import StateVector
from optimize import minimize
from qubo import Ising, bitstrings


class QAOA:
    """Depth-p QAOA for the cost Hamiltonian of an Ising object."""

    def __init__(self, ising, p=1):
        self.ising, self.p, self.n = ising, p, ising.n
        self.diag = ising.energies()                    # C(z) = <z|H_C|z>, incl. the constant
        self.terms = ising.terms()                      # (c_a, d_a, qubits): d_a = +-1 diagonal of P_a
        self.E0, self.optimal = ising.ground_state()
        self.n_params = 2 * p
        self._cost_phase = sum(c * d for c, d, _ in self.terms)   # C(z) without the constant
        self._zsigns = (1 - 2 * bitstrings(self.n)).astype(float)
        Hn = np.array([[1.0]])
        for _ in range(min(self.n, 10)):
            Hn = np.kron(Hn, np.array([[1, 1], [1, -1]]) / np.sqrt(2))
        self._Hn = Hn

    # ------------------------------------------------------------ circuit
    @staticmethod
    def _split(params):
        p = len(params) // 2
        return np.asarray(params[:p], float), np.asarray(params[p:], float)

    def _hadamard_all(self, psi, n):
        """H^{(x)n} psi: for the small registers of this chapter one matrix
        product with the precomputed Hadamard matrix (O(4^n), but a single
        BLAS call); the fast Walsh-Hadamard transform, O(n 2^n), for n > 10."""
        if n <= 10:
            return self._Hn @ psi
        out = psi.copy()
        for q in range(n):
            v = out.reshape(2 ** q, 2, 2 ** (n - q - 1))     # qubit q as the middle axis
            a = v[:, 0, :].copy(); b = v[:, 1, :]
            v[:, 0, :] = a + b; v[:, 1, :] = a - b
        return out / 2 ** (n / 2)

    def state(self, params, shift=None):
        """|psi_p(gamma, beta)> as a StateVector.  `shift=(layer, 'cost'|'mixer',
        index, delta)` adds delta to the angle of one term of the cost unitary
        (index a of self.terms) or of one qubit's mixer rotation in that layer:
        the building block of the parameter-shift rule.

        The phase separator is a phase per bit string.  The mixer uses
        exp(-i beta sum_q X_q) = H^{(x)n} exp(-i beta sum_q Z_q) H^{(x)n}:
        in the X basis it is diagonal too, and the two Hadamard layers are
        fast Walsh-Hadamard transforms (Eq. 3-mixerwht)."""
        gammas, betas = self._split(params)
        n = self.n
        z = self._zsigns                                 # (2^n, n) array of z_q = +-1 per bit string
        psi = np.full(2 ** n, 1 / np.sqrt(2 ** n), dtype=complex)   # |+>^n
        for k in range(self.p):
            # phase separator exp(-i gamma H_C): exp(-i gamma C(z)) per bit string
            # (the constant of H_C is a global phase and is left out)
            phase = self._cost_phase * gammas[k]
            if shift is not None and shift[0] == k and shift[1] == "cost":
                phase = phase + shift[3] * self.terms[shift[2]][1]
            psi = np.exp(-1j * phase) * psi
            # mixer: Hadamards, a phase exp(-i sum_q beta_q z_q), Hadamards
            angles = np.full(n, betas[k])
            if shift is not None and shift[0] == k and shift[1] == "mixer":
                angles[shift[2]] += shift[3]
            psi = self._hadamard_all(psi, n)
            psi = np.exp(-1j * (z @ angles)) * psi
            psi = self._hadamard_all(psi, n)
        sv = StateVector.__new__(StateVector)
        sv.n, sv.dim, sv.psi = n, 2 ** n, psi
        return sv

    def probabilities(self, params):
        return self.state(params).probabilities()

    def energy(self, params, shift=None):
        """E(gamma, beta) = sum_z C(z) |<z|psi>|^2, Eq. (3-objective)."""
        return float(self.state(params, shift).probabilities() @ self.diag)

    def approximation_ratio(self, params):
        """<H_C>/E_0 (both negative for MaxCut: the fraction of the best cut)."""
        return self.energy(params) / self.E0

    def success_probability(self, params):
        """Probability of measuring an optimal bit string."""
        return float(self.probabilities(params)[self.optimal].sum())

    # ---------------------------------------------------------- gradients
    def gradient_fd(self, params, h=1e-6):
        g = np.zeros(len(params))
        for k in range(len(params)):
            e = np.zeros(len(params)); e[k] = h
            g[k] = (self.energy(params + e) - self.energy(params - e)) / (2 * h)
        return g

    def gradient_global_shift(self, params, s=np.pi / 4):
        """The two-point rule applied to gamma_k and beta_k as a whole,
        Eq. (3-globalshift): exact only if the generator has two eigenvalues,
        which neither H_C nor H_M has for more than one qubit."""
        g = np.zeros(len(params))
        for k in range(len(params)):
            e = np.zeros(len(params)); e[k] = s
            g[k] = self.energy(params + e) - self.energy(params - e)
        return g

    def gradient_shift(self, params):
        """The exact parameter-shift gradient, Eq. (3-pertermshift): every gate
        exp(-i theta P) with P^2 = I obeys dE/dtheta = E(theta + pi/4) -
        E(theta - pi/4); gamma multiplies the coefficient c_a of term a and
        beta enters every qubit's R_x(2 beta) once."""
        p, s = self.p, np.pi / 4
        g = np.zeros(2 * p)
        for k in range(p):
            for a, (c, d, _) in enumerate(self.terms):
                g[k] += c * (self.energy(params, (k, "cost", a, +s)) - self.energy(params, (k, "cost", a, -s)))
            for q in range(self.n):
                g[p + k] += self.energy(params, (k, "mixer", q, +s)) - self.energy(params, (k, "mixer", q, -s))
        return g

    # ------------------------------------------------------- optimization
    def random_start(self, rng):
        return np.r_[rng.uniform(0, 2 * np.pi, self.p), rng.uniform(0, np.pi, self.p)]

    def optimize(self, restarts=10, rng=0, method="bfgs", lr=0.1, steps=200, x0=None, warm=None, grad="shift"):
        """Minimize E over the 2p angles from random starts (and, if `warm` is
        the optimum of depth p-1, from that optimum with one layer appended:
        the layerwise warm start of Sec. 3.4); returns (best params, best
        energy, history of the best run).  With `x0` a single run from x0.
        `grad` selects the parameter-shift gradient ("shift", what a quantum
        computer would do) or central finite differences ("fd", cheaper on
        the simulator when there are many Pauli terms)."""
        rng = np.random.default_rng(rng)
        best = (None, np.inf, None)
        starts = [np.array(x0, float)] if x0 is not None else [self.random_start(rng) for _ in range(restarts)]
        if warm is not None and x0 is None:
            g, b = self._split(warm)
            starts.append(np.r_[g, g[-1], b, b[-1]])
        for x in starts:
            if method == "bfgs":
                g = self.gradient_shift if grad == "shift" else None
                x, fx, _ = minimize(self.energy, x, grad=g, tol=1e-12)
                hist = None
            else:                                     # plain gradient descent
                hist = [self.energy(x)]
                for _ in range(steps):
                    x = x - lr * self.gradient_shift(x)
                    hist.append(self.energy(x))
                fx = hist[-1]
            if fx < best[1]:
                best = (x, fx, hist)
        return best

    # ------------------------------------------------------------ output
    def sample(self, params, shots, rng=0):
        """Measured bit strings (as integers) from the QAOA state."""
        rng = np.random.default_rng(rng)
        return rng.choice(2 ** self.n, size=shots, p=self.probabilities(params))

    def gate_count(self):
        """Gates of the circuit of Fig. 3-circuit, Eq. (3-gatecount): for every
        two-body term CNOT, R_z, CNOT; for every field term one R_z; one R_x
        per qubit and layer; plus the n Hadamards of the initial state."""
        n_two = sum(1 for c, d, q in self.terms if len(q) == 2)
        n_one = len(self.terms) - n_two
        return {"hadamard": self.n, "cnot": 2 * n_two * self.p, "rz": (n_two + n_one) * self.p,
                "rx": self.n * self.p}


def analytic_single_edge(beta, gamma):
    """The closed form of Eq. (3-oneedge) for H_C = -(I - Z_0 Z_1)/2, p = 1:
    E = -1/2 + (1/2) sin(4 beta) sin(gamma), minimum -1 at beta = pi/8, gamma = 3 pi/2."""
    return -0.5 + 0.5 * np.sin(4 * beta) * np.sin(gamma)


if __name__ == "__main__":
    from qubo import QUBO
    edge = QUBO.maxcut(2, [(0, 1)]).to_ising()
    qaoa = QAOA(edge, p=1)
    # the closed form against the simulator on a grid
    err = max(abs(qaoa.energy([g, b]) - analytic_single_edge(b, g))
              for b in np.linspace(0, np.pi / 2, 25) for g in np.linspace(0, 2 * np.pi, 25))
    print(f"single edge, p = 1: max |E_simulator - E_closed form| = {err:.1e}")
    assert err < 1e-12
    x_opt = [3 * np.pi / 2, np.pi / 8]
    psi = qaoa.state(x_opt)
    print(f"  at (gamma, beta) = (3pi/2, pi/8): E = {qaoa.energy(x_opt):.6f}, state = {psi}")
    assert np.isclose(qaoa.energy(x_opt), -1.0)
    # gradients: per-term shift equals finite differences, the global shift does not
    x = np.array([0.8, 0.5])
    ring = QAOA(QUBO.maxcut(4, [(0, 1), (1, 2), (2, 3), (3, 0)]).to_ising(), p=1)
    g_fd, g_ps, g_gl = ring.gradient_fd(x), ring.gradient_shift(x), ring.gradient_global_shift(x)
    print(f"ring C4 at (gamma, beta) = {x}: dE = {g_fd} (finite differences)")
    print(f"                                   {g_ps} (per-term parameter shift)")
    print(f"                                   {g_gl} (global shift, wrong)")
    assert np.allclose(g_fd, g_ps, atol=1e-6)
    print("gate count, one layer on the ring:", ring.gate_count())
    print("QAOA checks passed")
