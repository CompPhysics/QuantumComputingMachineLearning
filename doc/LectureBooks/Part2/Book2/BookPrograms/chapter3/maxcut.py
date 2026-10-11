"""Chapter 3, Secs. 3.4-3.5: QAOA on MaxCut -- the four-vertex ring and a
six-vertex graph -- and the gradient rules.

  * Table 3-ring and Fig. 3-ring: energy, approximation ratio, probability of
    an optimal cut and the optimal angles for p = 1, 2, 3 on the ring C_4,
    the p = 1 landscape and the output distributions;
  * Table 3-sixgraph: the same for the six-vertex graph of Fig. 3-ring
    (a ring with the chords (0,2) and (1,4): a triangle makes it non-bipartite,
    maximum cut 7 of 8 edges);
  * Fig. 3-gradients: dE/dgamma and dE/dbeta along a line through parameter
    space by finite differences, by the naive global shift and by the exact
    per-term parameter-shift rule; and plain gradient descent against BFGS
    on the six-vertex graph.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]      # BookPrograms/
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np

from qubo import QUBO, bitstrings
from qaoa import QAOA

RING = [(0, 1), (1, 2), (2, 3), (3, 0)]
SIX = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 2), (1, 4)]


def run_depths(edges, n, depths=(1, 2, 3), restarts=8, rng=0):
    ising = QUBO.maxcut(n, edges).to_ising()
    rows, warm = [], None
    for p in depths:
        q = QAOA(ising, p)
        x, E, _ = q.optimize(restarts=restarts, rng=rng, warm=warm)
        warm = x
        rows.append(dict(p=p, params=x, E=E, cut=-E, ratio=q.approximation_ratio(x),
                         p_opt=q.success_probability(x), probs=q.probabilities(x), qaoa=q))
    return ising, rows


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from bookplot import savefig

    # ---------------------------------------------------------- the ring
    ising, rows = run_depths(RING, 4)
    q1 = rows[0]["qaoa"]
    print(f"MaxCut on the ring C4: maximum cut {-q1.E0:.0f}, optimal strings",
          ["".join(map(str, bitstrings(4)[k])) for k in q1.optimal])
    print("Table 3-ring")
    for r in rows:
        g, b = r["params"][:r["p"]], r["params"][r["p"]:]
        print(f"  p={r['p']}: <H_C> = {r['E']:+.4f}  cut {r['cut']:.4f}  ratio {r['ratio']:.4f}  "
              f"P(opt) {r['p_opt']:.4f}  gamma = {np.round(np.mod(g, 2*np.pi), 3)}  beta = {np.round(np.mod(b, np.pi), 3)}")
    # the p = 1 closed form of the ring, Eq. (3-ringclosed): E = -2 + sin(4 beta) sin(2 gamma)
    err = max(abs(q1.energy([g, b]) - (-2 + np.sin(4 * b) * np.sin(2 * g)))
              for g in np.linspace(0, np.pi, 15) for b in np.linspace(0, np.pi / 2, 15))
    print(f"  ring closed form E = -2 + sin(4 beta) sin(2 gamma): max error {err:.1e}")
    assert err < 1e-10

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.1))
    # landscape p = 1
    gs, bs = np.linspace(0, np.pi, 70), np.linspace(0, np.pi / 2, 70)
    L = np.array([[q1.energy([g, b]) for b in bs] for g in gs])
    ax = axes[0]
    cp = ax.contourf(gs / np.pi, bs / np.pi, L.T, levels=24, cmap="viridis")
    plt.colorbar(cp, ax=ax, label=r"$E(\gamma,\beta)$")
    gopt, bopt = np.mod(rows[0]["params"][0], np.pi), np.mod(rows[0]["params"][1], np.pi / 2)
    ax.plot(gopt / np.pi, bopt / np.pi, "r*", ms=12)
    ax.set_xlabel(r"$\gamma/\pi$"); ax.set_ylabel(r"$\beta/\pi$"); ax.set_title("ring $C_4$, $p=1$")
    labels = ["".join(map(str, b)) for b in bitstrings(4)]
    for ax, r in zip(axes[1:], rows):
        cols = ["tab:red" if k in q1.optimal else "tab:blue" for k in range(16)]
        ax.bar(range(16), r["probs"], color=cols)
        ax.set_xticks(range(16)); ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_title(f"$p={r['p']}$: ratio {r['ratio']:.3f}, $P_{{\\rm opt}}={r['p_opt']:.2f}$", fontsize=9)
        ax.set_ylim(0, 0.55)
    axes[1].set_ylabel("probability")
    fig.tight_layout()
    savefig(fig, 3, "ring")

    # ----------------------------------------------------- six-vertex graph
    ising6, rows6 = run_depths(SIX, 6, depths=(1, 2, 3, 4), restarts=8)
    q6 = rows6[0]["qaoa"]
    print(f"\nSix-vertex graph (ring + chords (0,2), (1,4)): maximum cut {-q6.E0:.0f} of {len(SIX)} edges, "
          f"{len(q6.optimal)} optimal strings")
    print("Table 3-sixgraph")
    for r in rows6:
        print(f"  p={r['p']}: cut {r['cut']:.4f}  ratio {r['ratio']:.4f}  P(opt) {r['p_opt']:.4f}")
    print("  gates per layer:", q6.gate_count())

    # -------------------------------------------------------- gradients
    print("\nFig. 3-gradients: gradients along the line beta = 0.4, gamma in [0, pi] (six-vertex graph, p = 1)")
    gline = np.linspace(0.05, np.pi - 0.05, 25)
    fd = np.array([q6.gradient_fd([g, 0.4]) for g in gline])
    ps = np.array([q6.gradient_shift([g, 0.4]) for g in gline])
    gl = np.array([q6.gradient_global_shift([g, 0.4]) for g in gline])
    print(f"  max |per-term shift - finite differences| = {np.abs(ps - fd).max():.1e}")
    print(f"  max |global shift   - finite differences| = {np.abs(gl - fd).max():.3f}")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, k, name in ((axes[0], 0, r"$\partial E/\partial\gamma$"), (axes[1], 1, r"$\partial E/\partial\beta$")):
        ax.plot(gline / np.pi, fd[:, k], "k-", lw=2.5, label="finite differences")
        ax.plot(gline / np.pi, ps[:, k], "r--", lw=1.5, label="per-term shift (exact)")
        ax.plot(gline / np.pi, gl[:, k], "b:", lw=1.5, label="global shift")
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_xlabel(r"$\gamma/\pi$"); ax.set_ylabel(name); ax.legend(fontsize=7)
    # gradient descent against BFGS from the same start, p = 2, both charged
    # with the number of energy evaluations (a gradient costs 2p(K + n) of them)
    q62 = rows6[1]["qaoa"]
    per_grad = 2 * q62.p * (len(q62.terms) + q62.n)      # 2(K + n) energies per layer
    rng = np.random.default_rng(3)
    x0 = q62.random_start(rng)
    x_gd, E_gd, hist = q62.optimize(method="gd", lr=0.05, steps=60, x0=x0)
    cost_gd = np.arange(len(hist)) * (per_grad + 1)
    evals, best_trace = [0], [q62.energy(x0)]
    count = [0]
    def E_traced(x):
        count[0] += 1; e = q62.energy(x)
        evals.append(count[0]); best_trace.append(min(best_trace[-1], e)); return e
    def g_traced(x):
        count[0] += per_grad; return q62.gradient_shift(x)
    from optimize import minimize
    x_bf, E_bf, it = minimize(E_traced, x0, grad=g_traced, tol=1e-12)
    print(f"  p = 2 from the same start: gradient descent {E_gd:+.4f} after 60 steps "
          f"({cost_gd[-1]} energy evaluations), BFGS {E_bf:+.4f} after {it} iterations "
          f"({count[0]} evaluations), best of 8 restarts {rows6[1]['E']:+.4f}")
    ax = axes[2]
    ax.plot(cost_gd, hist, label="gradient descent, step 0.05")
    ax.plot(evals, best_trace, label="BFGS (best so far)")
    ax.axhline(q62.E0, color="k", ls="--", lw=1, label="$E_0 = -7$")
    ax.set_xlabel("energy evaluations (gradients included)"); ax.set_ylabel(r"$E$"); ax.legend(fontsize=7)
    ax.set_xlim(0, cost_gd[-1])
    fig.tight_layout()
    savefig(fig, 3, "gradients")
    print("figures ring and gradients written")
