"""Chapter 3, Sec. 3.8: QAOA as a machine-learning model -- two-way clustering
as a weighted MaxCut.

Eight points in the plane are to be split into two clusters.  With the
similarity graph w_ij = ||x_i - x_j||^2 (every pair an edge), the partition
that maximises the weight of the cut separates the points that are farthest
apart: this is the MaxCut clustering of Eq. (3-clustering), the two-cluster
case of spectral clustering.  The data enter QAOA through the cost
Hamiltonian H_C(data), and the chapter uses this to read QAOA as a model
of the hybrid kind of Chapter 1.  The script compares the QAOA solution with
the brute-force optimum and the classical two-means clustering, and draws
Fig. 3-clustering.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]      # BookPrograms/
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np

from qubo import QUBO, bitstrings, maxcut_value
from qaoa import QAOA


def similarity_graph(X):
    """All pairs (i < j) with weight ||x_i - x_j||^2, Eq. (3-clustering)."""
    n = len(X)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    weights = [float(np.sum((X[i] - X[j]) ** 2)) for i, j in edges]
    return edges, weights


def two_means(X, iters=20, rng=0):
    """Lloyd's algorithm with two centres, for comparison."""
    rng = np.random.default_rng(rng)
    c = X[rng.choice(len(X), 2, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((X[:, None, :] - c[None, :, :]) ** 2).sum(-1), axis=1)
        c = np.array([X[lab == k].mean(0) for k in range(2)])
    return lab


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from bookplot import savefig
    from datasets import two_blobs

    X, _ = two_blobs(n=8, separation=3.0, noise=0.7, rng=2)
    edges, weights = similarity_graph(X)
    ising = QUBO.maxcut(len(X), edges, weights).to_ising()
    # the cost is scaled so that the angles live on the same range as for unit weights
    scale = np.mean(weights)
    ising.h, ising.J, ising.const = ising.h / scale, ising.J / scale, ising.const / scale
    E0, gs = ising.ground_state()
    best = bitstrings(8)[gs[0]]
    print(f"8 points, {len(edges)} weighted edges; maximum cut {-E0 * scale:.3f} (scaled {-E0:.3f}) at",
          "".join(map(str, best)), "and its complement")
    labels_km = two_means(X)
    cut_km = maxcut_value(labels_km, edges, weights)
    print(f"two-means clustering cuts {cut_km:.3f}: {'the same partition' if cut_km == -E0 * scale else 'a different partition'}")
    rows = []
    warm = None
    for p in (1, 2, 3):
        q = QAOA(ising, p)
        x, E, _ = q.optimize(restarts=6, rng=0, warm=warm, grad="fd")
        warm = x
        pr = q.probabilities(x)
        k = int(np.argmax(pr))
        rows.append((p, q.approximation_ratio(x), q.success_probability(x), bitstrings(8)[k], pr[k]))
        print(f"  p={p}: ratio {rows[-1][1]:.4f}, P(optimal partition) {rows[-1][2]:.3f}, "
              f"most probable string {''.join(map(str, rows[-1][3]))} (p = {pr[k]:.3f}), cut {maxcut_value(rows[-1][3], edges, weights):.3f}")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    lab = rows[-1][3]
    for k, (m, c) in enumerate((("s", "tab:blue"), ("^", "tab:green"))):
        axes[0].scatter(X[lab == k, 0], X[lab == k, 1], marker=m, color=c, s=60, edgecolor="k")
    for (i, j), w in zip(edges, weights):
        if lab[i] != lab[j]:
            axes[0].plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], color="gray", lw=0.4, alpha=0.5)
    axes[0].set_xlabel("$x_1$"); axes[0].set_ylabel("$x_2$"); axes[0].set_title(f"QAOA $p=3$ partition, cut edges drawn")
    q = rows[-1]
    pr = QAOA(ising, 3).probabilities(warm)
    top = np.argsort(pr)[::-1][:8]
    axes[1].bar(range(8), pr[top], color=["tab:red" if t in gs else "tab:blue" for t in top])
    axes[1].set_xticks(range(8)); axes[1].set_xticklabels(["".join(map(str, bitstrings(8)[t])) for t in top], rotation=90, fontsize=7)
    axes[1].set_ylabel("probability"); axes[1].set_title("eight most probable strings, $p=3$")
    fig.tight_layout()
    savefig(fig, 3, "clustering")
    print("figure clustering written")
