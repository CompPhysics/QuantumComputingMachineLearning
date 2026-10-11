"""Chapter 5, Secs. 5.3-5.5: the quantum Boltzmann machine

    H(theta) = - sum_mu theta_mu Lambda_mu,   rho(theta) = exp(-beta H) / Z,

Eq. (5-qbmham), with Pauli strings Lambda_mu on n_v visible and n_h hidden
qubits.  The model distribution over visible bit strings is
p(v) = Tr[Pi_v rho] with Pi_v = |v><v| x I_h, Eq. (5-visible).

    QBM(n_visible, n_hidden, terms, beta)
        terms: list of dicts {qubit: "X"/"Y"/"Z"}, one per parameter; the
        helpers `rqbm_terms` (restricted QBM: Z and X fields, Z Z couplings
        between layers) and `ising_terms` build the usual sets
        .hamiltonian(theta), .gibbs(theta), .visible_probabilities(theta)
        .nll(theta, target)              -sum_v pi(v) ln p(v)
        .gradient_exact(theta, target)   the Duhamel gradient, Eq. (5-exactgrad)
        .gradient_bound(theta, target)   the clamped-state gradient of Amin et al.,
                                         Eq. (5-boundgrad): <Lambda>_clamped - <Lambda>_model
        .relative_entropy(theta, rho_t)  S(rho_target || rho(theta)) for a quantum target
        .gradient_relent(theta, rho_t)   beta (<Lambda>_theta - <Lambda>_target), Eq. (5-relentgrad)
        .fit(...)                        gradient descent / Adam on either objective

Everything is exact (dense matrices, 2^(n_v+n_h) <= 64), so that the
approximations of the text can be measured against the truth.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]      # BookPrograms/
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np

from gibbs import pauli_string, gibbs_state, partial_trace, exp_derivative, relative_entropy, log_partition
from rbm import patterns, kl_divergence
from optimize import Adam


def rqbm_terms(n_v, n_h, transverse=True):
    """Restricted QBM: Z and (optionally) X on every qubit, Z Z between every
    visible-hidden pair.  Returns (terms, names)."""
    terms, names = [], []
    for q in range(n_v + n_h):
        terms.append({q: "Z"}); names.append(f"Z{q}")
    if transverse:
        for q in range(n_v + n_h):
            terms.append({q: "X"}); names.append(f"X{q}")
    for i in range(n_v):
        for j in range(n_h):
            terms.append({i: "Z", n_v + j: "Z"}); names.append(f"Z{i}Z{n_v + j}")
    return terms, names


def ising_terms(n, fields=("Z", "X"), couplings=("ZZ",), ring=False):
    """Fully visible chain: the given one-qubit fields and nearest-neighbour couplings."""
    terms, names = [], []
    for f in fields:
        for q in range(n):
            terms.append({q: f}); names.append(f"{f}{q}")
    pairs = [(q, q + 1) for q in range(n - 1)] + ([(n - 1, 0)] if ring and n > 2 else [])
    for c in couplings:
        for (i, j) in pairs:
            terms.append({i: c[0], j: c[1]}); names.append(f"{c[0]}{i}{c[1]}{j}")
    return terms, names


class QBM:
    def __init__(self, n_visible, n_hidden, terms, beta=1.0):
        self.n_v, self.n_h, self.beta = n_visible, n_hidden, beta
        self.n = n_visible + n_hidden
        self.terms = list(terms)
        self.ops = [pauli_string(t, self.n) for t in self.terms]
        self.n_params = len(self.ops)
        self.V = patterns(n_visible)
        # projectors Pi_v = |v><v| x I_h as diagonals
        self.proj = np.zeros((2 ** n_visible, 2 ** self.n))
        for k in range(2 ** n_visible):
            self.proj[k, k * 2 ** n_hidden:(k + 1) * 2 ** n_hidden] = 1.0

    def hamiltonian(self, theta):
        return -sum(t * op for t, op in zip(theta, self.ops))

    def gibbs(self, theta):
        return gibbs_state(self.hamiltonian(theta), self.beta)

    def visible_state(self, theta):
        return partial_trace(self.gibbs(theta), self.n_v, self.n_h)

    def visible_probabilities(self, theta):
        return self.proj @ np.diag(self.gibbs(theta)).real

    def nll(self, theta, target):
        return float(-np.sum(target * np.log(np.maximum(self.visible_probabilities(theta), 1e-300))))

    def expectations(self, rho):
        return np.array([np.trace(op @ rho).real for op in self.ops])

    # ---- gradients for a classical target distribution
    def gradient_exact(self, theta, target):
        """d NLL / d theta_mu by the Duhamel formula, Eq. (5-exactgrad):
        d p(v) = Tr[Pi_v d(e^{-beta H})]/Z - p(v) d ln Z, with d ln Z = beta <Lambda_mu>."""
        H = self.hamiltonian(theta)
        rho = gibbs_state(H, self.beta)
        lnZ = log_partition(H, self.beta)
        p = self.proj @ np.diag(rho).real
        model = self.expectations(rho)
        g = np.zeros(self.n_params)
        for mu, op in enumerate(self.ops):
            dE = exp_derivative(H, -op, self.beta) * np.exp(-lnZ)   # d(e^{-beta H})/d theta_mu / Z
            dp = self.proj @ np.diag(dE).real - p * self.beta * model[mu]
            g[mu] = -np.sum(target * dp / np.maximum(p, 1e-300))
        return g

    def clamped_state(self, theta, v):
        """Gibbs state of the hidden qubits with the visible ones clamped to v:
        Z_i -> (1 - 2 v_i), X_i -> 0 on the visible qubits, Eq. (5-clamped)."""
        Hh = np.zeros((2 ** self.n_h, 2 ** self.n_h), complex)
        for t, term in zip(theta, self.terms):
            factor, hidden = 1.0, {}
            for q, P in term.items():
                if q < self.n_v:
                    factor *= (1 - 2 * v[q]) if P == "Z" else 0.0
                else:
                    hidden[q - self.n_v] = P
            if factor != 0.0:
                Hh -= t * factor * pauli_string(hidden, self.n_h)
        return gibbs_state(Hh, self.beta)

    def clamped_expectations(self, theta, v):
        """<Lambda_mu> in the clamped state (visible Z's replaced by their values)."""
        rho_h = self.clamped_state(theta, v)
        out = np.zeros(self.n_params)
        for mu, term in enumerate(self.terms):
            factor, hidden = 1.0, {}
            for q, P in term.items():
                if q < self.n_v:
                    factor *= (1 - 2 * v[q]) if P == "Z" else 0.0
                else:
                    hidden[q - self.n_v] = P
            out[mu] = factor * np.trace(pauli_string(hidden, self.n_h) @ rho_h).real if factor != 0 else 0.0
        return out

    def gradient_bound(self, theta, target):
        """Gradient of the upper bound on the NLL (Golden-Thompson), Eq. (5-boundgrad):
        beta [ <Lambda>_model - sum_v pi(v) <Lambda>_clamped(v) ]."""
        model = self.expectations(self.gibbs(theta))
        clamped = sum(target[k] * self.clamped_expectations(theta, self.V[k])
                      for k in range(len(target)) if target[k] > 0)
        return self.beta * (model - clamped)

    # ---- a quantum target state (no hidden qubits)
    def relative_entropy(self, theta, rho_target):
        assert self.n_h == 0
        return relative_entropy(rho_target, self.gibbs(theta))

    def gradient_relent(self, theta, rho_target):
        """d S(rho_t || rho_theta) / d theta_mu = beta (<Lambda_mu>_theta - <Lambda_mu>_target), Eq. (5-relentgrad)."""
        return self.beta * (self.expectations(self.gibbs(theta)) - self.expectations(rho_target))

    # ---- training
    def fit(self, theta0, target, objective="exact", steps=300, lr=0.1, optimizer="adam", record=10):
        """objective: "exact" (Duhamel NLL gradient), "bound" (clamped-state
        gradient), "relent" (quantum target).  Returns theta and the history
        of the NLL (classical target) or of the relative entropy."""
        theta = np.array(theta0, float)
        opt = Adam(lr=lr) if optimizer == "adam" else None
        hist = []
        for step in range(steps):
            if objective == "exact":
                g, val = self.gradient_exact(theta, target), self.nll(theta, target)
            elif objective == "bound":
                g, val = self.gradient_bound(theta, target), self.nll(theta, target)
            else:
                g, val = self.gradient_relent(theta, target), self.relative_entropy(theta, target)
            if step % record == 0:
                hist.append(val)
            theta = theta - lr * g if opt is None else opt.step(theta, g)
        hist.append(self.nll(theta, target) if objective != "relent" else self.relative_entropy(theta, target))
        return theta, np.array(hist)


if __name__ == "__main__":
    from optimize import numerical_gradient
    from rbm import parity
    rng = np.random.default_rng(0)
    terms, names = rqbm_terms(2, 2)
    q = QBM(2, 2, terms, beta=1.0)
    theta = rng.normal(0, 0.5, q.n_params)
    target = parity(2)
    g = q.gradient_exact(theta, target)
    gn = numerical_gradient(lambda t: q.nll(t, target), theta)
    gb = q.gradient_bound(theta, target)
    print("RQBM, 2 visible + 2 hidden qubits, random parameters, XOR target")
    print(f"{'term':>6}  {'exact (Duhamel)':>16}  {'finite diff.':>13}  {'bound gradient':>14}")
    for nm, a, b, c in zip(names, g, gn, gb):
        print(f"{nm:>6}  {a:>16.6f}  {b:>13.6f}  {c:>14.6f}")
    assert np.max(np.abs(g - gn)) < 1e-6
    # classical limit: with no transverse fields the two gradients agree
    terms_c, _ = rqbm_terms(2, 2, transverse=False)
    qc = QBM(2, 2, terms_c)
    tc = rng.normal(0, 0.5, qc.n_params)
    print(f"no transverse fields: max |exact - bound| = {np.max(np.abs(qc.gradient_exact(tc, target) - qc.gradient_bound(tc, target))):.1e}")
    assert np.max(np.abs(qc.gradient_exact(tc, target) - qc.gradient_bound(tc, target))) < 1e-9
    # relative-entropy gradient
    q3 = QBM(3, 0, ising_terms(3)[0])
    t_star = rng.normal(0, 0.7, q3.n_params); rho_t = q3.gibbs(t_star)
    t0 = rng.normal(0, 0.3, q3.n_params)
    gr = q3.gradient_relent(t0, rho_t); grn = numerical_gradient(lambda t: q3.relative_entropy(t, rho_t), t0)
    print(f"relative-entropy gradient against finite differences: max |diff| = {np.max(np.abs(gr - grn)):.1e}")
    assert np.max(np.abs(gr - grn)) < 1e-6
