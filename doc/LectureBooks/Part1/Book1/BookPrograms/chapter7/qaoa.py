"""Chapter 7, Sec. 7.9: the quantum approximate optimization algorithm.

* QUBO -> Ising -> cost Hamiltonian, Eqs. (7.qubo)-(7.costhamiltonian), with
  the two-variable example of Table 7.qubo checked bit string by bit string;
* MaxCut and its cost Hamiltonian H_C = sum_(ij) (I - Z_i Z_j)/2;
* ``QAOA``: the state of Eq. (7.qaoastate), prepared with the phase separator
  (a diagonal phase per bit string) and the mixer (an R_x(2 beta) on every
  qubit) on the state-vector simulator, its cost expectation, and the
  classical optimisation of (gamma, beta) with random restarts;
* the single-edge case at p = 1, where <Z_0 Z_1> = -sin(4 beta) sin(gamma)
  (Eq. 7.qaoaedge), and the ring, F_1 = 2 + sin(4 beta) sin(2 gamma), both
  checked against the simulator;
* the four-vertex ring and a six-vertex graph: approximation ratio and
  probability of an optimal cut against the depth p (Table 7.qaoa), the p = 1
  landscape of the ring (Fig. 7.qaoalandscape) and the output distributions
  (Fig. 7.qaoa).

About five seconds.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import itertools
import numpy as np
import gates as G
from pauli import PauliSum
from statevector import StateVector
from optimize import minimize
from bookplot import savefig, plt


# ---------------------------------------------------------- QUBO and Ising
def qubo_to_ising(Q):
    """x_i = (1 - s_i)/2 turns x^T Q x into sum_(i<j) J_ij s_i s_j + sum_i h_i s_i + const, Eq. (7.qubotoising)."""
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    J = {(i, j): (Q[i, j] + Q[j, i]) / 4 for i in range(n) for j in range(i + 1, n)}
    h = {i: -Q[i, i] / 2 - sum((Q[i, j] + Q[j, i]) / 4 for j in range(n) if j != i) for i in range(n)}
    const = Q.trace() / 2 + (Q.sum() - Q.trace()) / 4
    return J, h, const


def ising_hamiltonian(n, J, h):
    """H_C = sum J_ij Z_i Z_j + sum h_i Z_i as a PauliSum."""
    H = PauliSum(n_qubits=n)
    for (i, j), c in J.items():
        if c:
            H = H + PauliSum.single(n, i, "Z") * PauliSum.single(n, j, "Z") * c
    for i, c in h.items():
        if c:
            H = H + PauliSum.single(n, i, "Z", c)
    return H.simplify()


def maxcut_hamiltonian(n, edges, weights=None):
    """H_C = sum_(ij) w_ij (I - Z_i Z_j)/2: its eigenvalue on |z> is the weight of the cut z, Eq. (7.maxcut)."""
    weights = weights or {e: 1.0 for e in edges}
    H = PauliSum(n_qubits=n)
    for (i, j) in edges:
        H = H + 0.5 * weights[(i, j)] * (PauliSum.identity(n) - PauliSum.single(n, i, "Z") * PauliSum.single(n, j, "Z"))
    return H.simplify()


def cut_value(z, edges, weights=None):
    weights = weights or {e: 1.0 for e in edges}
    return sum(weights[(i, j)] for (i, j) in edges if z[i] != z[j])


# ------------------------------------------------------------------- QAOA
class QAOA:
    """The depth-p QAOA state for a diagonal cost Hamiltonian, Eq. (7.qaoastate)."""

    def __init__(self, H_cost, p, maximize=True):
        self.H, self.p, self.n = H_cost, p, H_cost.n
        self.diag = np.real(np.diag(H_cost.matrix()))          # C(z) for every bit string
        self.sign = -1.0 if maximize else 1.0                   # the optimiser minimises sign * <H_C>
        self.n_params = 2 * p

    def state(self, params):
        gammas, betas = params[:self.p], params[self.p:]
        sv = StateVector(self.n)
        for q in range(self.n):
            sv.h(q)                                             # |+>^n
        for gamma, beta in zip(gammas, betas):
            sv.psi = np.exp(-1j * gamma * self.diag) * sv.psi   # phase separator exp(-i gamma H_C)
            for q in range(self.n):
                sv.rx(2 * beta, q)                              # mixer exp(-i beta X_q)
        return sv.psi

    def expectation(self, params):
        psi = self.state(params)
        return float(np.sum(self.diag * np.abs(psi) ** 2))

    def objective(self, params):
        return self.sign * self.expectation(params)

    def optimize(self, restarts=8, rng=None):
        rng = np.random.default_rng(0) if rng is None else rng
        best = None
        for _ in range(restarts):
            x0 = np.concatenate([rng.uniform(0, np.pi, self.p), rng.uniform(0, np.pi / 2, self.p)])
            x, f, it = minimize(self.objective, x0, tol=1e-12, gtol=1e-8)
            if best is None or f < best[1]:
                best = (x, f)
        # for an integer-valued cost the state is periodic in gamma (2 pi) and beta (pi)
        x = best[0].copy()
        x[:self.p] = np.mod(x[:self.p], 2 * np.pi)
        x[self.p:] = np.mod(x[self.p:], np.pi)
        self.params = x
        return x, self.sign * best[1]

    def distribution(self, params=None):
        return np.abs(self.state(self.params if params is None else params)) ** 2


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True, linewidth=120)

    # ------------------------------------------------ QUBO to Ising (Table 7.qubo)
    print("=== QUBO x^T Q x with Q = [[1, -2], [-2, 1]] -> Ising (Table 7.qubo)")
    Q = np.array([[1.0, -2.0], [-2.0, 1.0]])
    J, h, const = qubo_to_ising(Q)
    HC = ising_hamiltonian(2, J, h)
    print(f"  J_01 = {J[(0, 1)]}, h = {h}, constant = {const}; H_C = {dict(HC.terms)}")
    diag = np.real(np.diag(HC.matrix()))
    for k, bits in enumerate(itertools.product((0, 1), repeat=2)):
        x = np.array(bits)
        c = x @ Q @ x
        print(f"  x = {bits}: x^T Q x = {c:+.1f}, H_C(z) + const = {diag[k] + const:+.1f}")
        assert abs(c - diag[k] - const) < 1e-12
    assert J[(0, 1)] == -1 and h[0] == 0.5 and const == 0.0

    # ------------------------------------------------ the single edge at p = 1 (Eq. 7.qaoaedge)
    print("\n=== One edge, p = 1: <Z_0 Z_1> = -sin(4 beta) sin(gamma)")
    H1 = maxcut_hamiltonian(2, [(0, 1)])
    q1 = QAOA(H1, 1)
    worst = 0.0
    for gamma in np.linspace(0, 2 * np.pi, 13):
        for beta in np.linspace(0, np.pi, 11):
            psi = q1.state([gamma, beta])
            zz = float(np.real(np.vdot(psi, PauliSum({"ZZ": 1.0}).matrix() @ psi)))
            worst = max(worst, abs(zz + np.sin(4 * beta) * np.sin(gamma)))
    print(f"  largest deviation from the formula over a 13 x 11 grid: {worst:.1e}")
    assert worst < 1e-12
    gb, F = q1.optimize()
    print(f"  optimum F = {F:.6f} at gamma = {gb[0]:.4f}, beta = {gb[1]:.4f} (the cut of one edge is 1: the single edge is solved exactly at p = 1)")
    assert abs(F - 1) < 1e-8

    # ------------------------------------------------ the four-vertex ring and a six-vertex graph (Table 7.qaoa)
    graphs = {"ring $C_4$": (4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
              "prism graph, 6 vertices": (6, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)])}
    table = {}
    dists = {}
    print("\n=== QAOA for MaxCut (Table 7.qaoa)")
    print(f"{'graph':>24s} {'C_max':>5s} {'p':>2s} {'F_p':>8s} {'ratio':>7s} {'P(optimal cut)':>15s} {'gamma, beta':>30s}")
    for name, (n, edges) in graphs.items():
        H = maxcut_hamiltonian(n, edges)
        diag = np.real(np.diag(H.matrix()))
        cmax = diag.max()
        optimal = np.abs(diag - cmax) < 1e-9
        for p in (1, 2, 3):
            q = QAOA(H, p)
            params, F = q.optimize(restarts=10, rng=np.random.default_rng(p))
            prob = float(q.distribution()[optimal].sum())
            table[(name, p)] = (F, F / cmax, prob, params)
            dists[(name, p)] = q.distribution()
            pstr = "(" + ", ".join(f"{x:.3f}" for x in params) + ")"
            print(f"{name:>24s} {cmax:5.0f} {p:2d} {F:8.4f} {F / cmax:7.4f} {prob:15.4f} {pstr:>30s}")
        assert table[(name, 3)][1] > table[(name, 1)][1] - 1e-9
    assert table[("ring $C_4$", 2)][1] > 0.999

    # p = 1 landscape of the ring, and the closed form F_1 = 2 + sin(4 beta) sin(2 gamma) (Eq. 7.qaoaring)
    n, edges = graphs["ring $C_4$"]
    H = maxcut_hamiltonian(n, edges)
    q = QAOA(H, 1)
    worst = max(abs(q.expectation([g, b]) - 2 - np.sin(4 * b) * np.sin(2 * g))
                for g in np.linspace(0, 2 * np.pi, 9) for b in np.linspace(0, np.pi, 7))
    print(f"  ring, p = 1: largest deviation from F_1 = 2 + sin(4 beta) sin(2 gamma): {worst:.1e}")
    assert worst < 1e-12
    gs = np.linspace(0, 2 * np.pi, 121); bs = np.linspace(0, np.pi, 61)
    F = np.array([[q.expectation([g, b]) for g in gs] for b in bs])
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    im = ax.contourf(gs, bs, F, levels=25)
    plt.colorbar(im, ax=ax, label=r"$F_1(\gamma,\beta)$")
    g1, b1 = table[("ring $C_4$", 1)][3]
    ax.plot([g1], [b1], "w*", ms=10)
    ax.set_xlabel(r"$\gamma$"); ax.set_ylabel(r"$\beta$")
    savefig(fig, 7, "qaoa_landscape")
    print(f"  ring, p = 1: largest F on the grid {F.max():.4f} (optimiser {table[('ring $C_4$', 1)][0]:.4f})")

    # output distributions of the prism graph
    n, edges = graphs["prism graph, 6 vertices"]
    H = maxcut_hamiltonian(n, edges)
    diag = np.real(np.diag(H.matrix()))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), sharey=True)
    for ax, p in zip(axes, (1, 2, 3)):
        d = dists[("prism graph, 6 vertices", p)]
        colors = ["C3" if abs(diag[k] - diag.max()) < 1e-9 else "C0" for k in range(len(d))]
        ax.bar(np.arange(len(d)), d, color=colors, width=1.0)
        ax.set_title(f"$p={p}$, $F_p={table[('prism graph, 6 vertices', p)][0]:.3f}$", fontsize=9)
        ax.set_xlabel("bit string $z$")
    axes[0].set_ylabel("probability")
    savefig(fig, 7, "qaoa")
    print("\nQAOA checks passed")
