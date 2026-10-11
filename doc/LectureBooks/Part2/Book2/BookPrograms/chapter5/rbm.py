"""Chapter 5, Sec. 5.2: the classical restricted Boltzmann machine.

    RBM(n_visible, n_hidden, rng)
        energy E(v, h) = -a.v - b.h - v.W h, Eq. (5-rbmenergy), binary units in {0, 1}
        .log_probabilities()           ln p(v) for all 2^n_v patterns (exact, small n_v)
        .nll(target)                   -sum_v pi(v) ln p(v)
        .gradient_exact(target)        d NLL / d(a, b, W): data term minus model term, Eq. (5-rbmgrad)
        .gradient_cd(data, k, rng)     contrastive divergence CD-k, Eq. (5-cd)
        .gradient_pcd(data, chains, rng)  persistent contrastive divergence
        .gibbs_step(v, rng)            one block Gibbs step v -> h -> v'
        .fit(target, method, ...)      training loop (plain gradient descent)
    bars_and_stripes()                 the 2x2 bars-and-stripes distribution (6 of 16 patterns)
    parity(n)                          the even-parity distribution on n bits (XOR for n = 2)

All parameters live in one flat vector theta = (a, b, W.ravel()).  The
script trains on the two toy distributions with the exact gradient, CD-1,
CD-10 and PCD and produces Table 5-cd and Fig. 5-rbm.  Runtime about 20 s.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]      # BookPrograms/
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np


def patterns(n):
    """All 2^n bit strings as rows, in the order of their integer value."""
    idx = np.arange(2 ** n)
    return ((idx[:, None] >> (n - 1 - np.arange(n))) & 1).astype(float)


def bars_and_stripes():
    """Visible patterns of the 2x2 bars-and-stripes data set (rows or columns
    uniform): 0000, 0011, 1100, 1111, 0101, 1010, each with probability 1/6."""
    p = np.zeros(16)
    for s in ["0000", "0011", "1100", "1111", "0101", "1010"]:
        p[int(s, 2)] = 1 / 6
    return p


def parity(n=2):
    """Even parity: every pattern with an even number of ones, equally likely."""
    P = patterns(n)
    p = (P.sum(axis=1) % 2 == 0).astype(float)
    return p / p.sum()


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


def kl_divergence(p, q):
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / np.maximum(q[m], 1e-300))))


class RBM:
    def __init__(self, n_visible, n_hidden, rng=0, scale=0.5):
        self.n_v, self.n_h = n_visible, n_hidden
        rng = np.random.default_rng(rng)
        self.theta = np.concatenate([np.zeros(n_visible), np.zeros(n_hidden),
                                     rng.normal(0, scale, n_visible * n_hidden)])
        self.V = patterns(n_visible)

    @property
    def n_params(self):
        return len(self.theta)

    def unpack(self, theta=None):
        t = self.theta if theta is None else theta
        a, b = t[:self.n_v], t[self.n_v:self.n_v + self.n_h]
        W = t[self.n_v + self.n_h:].reshape(self.n_v, self.n_h)
        return a, b, W

    # ---- exact quantities (enumeration over the visible patterns)
    def log_probabilities(self, theta=None):
        """ln p(v) = a.v + sum_j ln(1 + e^{b_j + (W^T v)_j}) - ln Z, Eq. (5-rbmmarginal)."""
        a, b, W = self.unpack(theta)
        f = self.V @ a + np.sum(np.logaddexp(0.0, b + self.V @ W), axis=1)   # -F(v)
        return f - np.logaddexp.reduce(f)

    def probabilities(self, theta=None):
        return np.exp(self.log_probabilities(theta))

    def nll(self, target, theta=None):
        return float(-np.sum(target * self.log_probabilities(theta)))

    def hidden_means(self, V, theta=None):
        a, b, W = self.unpack(theta)
        return sigmoid(b + V @ W)

    def _statistics(self, V, weights):
        """Weighted averages of v, <h|v> and v <h|v>^T: the sufficient statistics."""
        Hm = self.hidden_means(V)
        return np.concatenate([weights @ V, weights @ Hm, ((V.T * weights) @ Hm).ravel()])

    def gradient_exact(self, target):
        """d NLL / d theta = <stats>_model - <stats>_data, Eq. (5-rbmgrad)."""
        return self._statistics(self.V, self.probabilities()) - self._statistics(self.V, target)

    # ---- sampling
    def gibbs_step(self, V, rng):
        a, b, W = self.unpack()
        H = (rng.uniform(size=(len(V), self.n_h)) < sigmoid(b + V @ W)).astype(float)
        return (rng.uniform(size=V.shape) < sigmoid(a + H @ W.T)).astype(float)

    def gradient_cd(self, data, k, rng):
        """CD-k: the model statistics are replaced by those of the k-step
        reconstructions started from the data, Eq. (5-cd)."""
        Vk = data.copy()
        for _ in range(k):
            Vk = self.gibbs_step(Vk, rng)
        w = np.full(len(data), 1 / len(data))
        return self._statistics(Vk, w) - self._statistics(data, w)

    def gradient_pcd(self, data, chains, rng):
        """Persistent CD: the chains are advanced one step per update and kept."""
        chains[:] = self.gibbs_step(chains, rng)
        return self._statistics(chains, np.full(len(chains), 1 / len(chains))) \
            - self._statistics(data, np.full(len(data), 1 / len(data)))

    def fit(self, target, method="exact", steps=3000, lr=0.5, k=1, n_data=200, n_chains=50, rng=0, record=20):
        """Gradient descent on the NLL; CD and PCD see a data set of n_data
        samples drawn from the target.  Returns the KL history (every `record` steps)."""
        rng = np.random.default_rng(rng)
        data = self.V[rng.choice(len(target), size=n_data, p=target)]
        chains = self.V[rng.choice(len(target), size=n_chains, p=target)]
        hist = []
        for step in range(steps):
            if method == "exact":
                g = self.gradient_exact(target)
            elif method == "cd":
                g = self.gradient_cd(data, k, rng)
            elif method == "pcd":
                g = self.gradient_pcd(data, chains, rng)
            else:
                raise ValueError(method)
            self.theta = self.theta - lr * g
            if step % record == 0 or step == steps - 1:
                hist.append(kl_divergence(target, self.probabilities()))
        return np.array(hist)


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    import bookplot
    from optimize import numerical_gradient

    # gradient check
    rbm = RBM(3, 2, rng=1)
    t = parity(3)
    g = rbm.gradient_exact(t)
    gn = numerical_gradient(lambda th: rbm.nll(t, th), rbm.theta)
    print(f"exact RBM gradient against finite differences: max |diff| = {np.max(np.abs(g - gn)):.1e}")
    assert np.max(np.abs(g - gn)) < 1e-6

    print("\nTable 5-cd: KL(target || model) after 3000 gradient steps, lr = 0.5 (PCD: 0.2, 50 chains)")
    print(f"{'target':>18}  {'n_h':>3}  {'exact':>8}  {'CD-1':>8}  {'CD-10':>8}  {'PCD':>8}  {'time':>6}")
    targets = {"XOR (parity, n=2)": (parity(2), 2), "bars and stripes": (bars_and_stripes(), 4)}
    curves = {}
    for name, (target, n_h) in targets.items():
        t0 = time.time()
        row = []
        for method, k in [("exact", 0), ("cd", 1), ("cd", 10), ("pcd", 0)]:
            rbm = RBM(int(np.log2(len(target))), n_h, rng=0)
            hist = rbm.fit(target, method=method, k=k, rng=0, lr=0.2 if method == "pcd" else 0.5)
            row.append(hist[-1])
            curves[(name, method, k)] = (hist, rbm.probabilities())
        print(f"{name:>18}  {n_h:>3}  " + "  ".join(f"{v:>8.4f}" for v in row) + f"  {time.time() - t0:>5.1f}s")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3))
    steps = np.arange(0, 3000, 20).tolist() + [2999]
    for (name, method, k), (hist, p) in curves.items():
        if name.startswith("bars"):
            lab = {"exact": "exact gradient", "cd": f"CD-{k}", "pcd": "PCD"}[method]
            axes[0].semilogy(steps, hist, label=lab)
    axes[0].set_xlabel("gradient step"); axes[0].set_ylabel("KL(target $\\|$ model)")
    axes[0].set_title("bars and stripes, $n_v=4$, $n_h=4$", fontsize=10); axes[0].legend(fontsize=8)
    p_t = bars_and_stripes(); x = np.arange(16); w = 0.27
    axes[1].bar(x - w, p_t, w, color="k", label="target")
    axes[1].bar(x, curves[("bars and stripes", "exact", 0)][1], w, label="exact gradient")
    axes[1].bar(x + w, curves[("bars and stripes", "cd", 1)][1], w, label="CD-1")
    axes[1].set_xticks(x); axes[1].set_xticklabels([format(i, "04b") for i in x], rotation=90, fontsize=7)
    axes[1].set_ylabel("$p(v)$"); axes[1].legend(fontsize=8)
    fig.tight_layout()
    bookplot.savefig(fig, 5, "rbm")
    print("figure rbm written")
