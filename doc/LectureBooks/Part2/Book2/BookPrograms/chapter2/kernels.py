"""Chapter 2, Secs. 2.4-2.5: classical kernels and the kernel trick.

A kernel is a function K(x, x') = phi(x)^T phi(x') for some feature map phi
(Mercer's theorem, Eq. 2-mercer).  Each class below evaluates its kernel on
two sets of points at once and returns the Gram matrix K_ij = K(x_i, x'_j),
which is all the support-vector machine of svm.py needs.  Run as a script
the module draws the two figures of Sec. 2.4: the quadratic feature map
that separates a one-dimensional problem (Fig. 2-kerneltrick) and the
radial-basis landmarks (Fig. 2-rbflandmarks), and checks Mercer's
condition numerically.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]      # BookPrograms/
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np


class Kernel:
    """Base class: subclasses implement `__call__(X, Y)` -> Gram matrix."""
    name = "kernel"

    def __call__(self, X, Y):
        raise NotImplementedError

    def gram(self, X):
        return self(X, X)

    def is_mercer(self, X, tol=1e-10):
        """Numerical Mercer check: is the Gram matrix positive semidefinite?"""
        K = self.gram(X)
        return bool(np.all(np.linalg.eigvalsh((K + K.T) / 2) > -tol))

    def __repr__(self):
        return self.name


class LinearKernel(Kernel):
    """K(x, x') = x^T x', the inner product of the inputs themselves."""
    name = "linear"

    def __call__(self, X, Y):
        return np.asarray(X, float) @ np.asarray(Y, float).T


class PolynomialKernel(Kernel):
    """K(x, x') = (gamma x^T x' + coef0)^degree, Eq. (2-polykernel)."""

    def __init__(self, degree=2, coef0=1.0, gamma=1.0):
        self.degree, self.coef0, self.gamma = degree, coef0, gamma
        self.name = f"poly(d={degree},r={coef0:g})"

    def __call__(self, X, Y):
        return (self.gamma * np.asarray(X, float) @ np.asarray(Y, float).T
                + self.coef0) ** self.degree


class RBFKernel(Kernel):
    """K(x, x') = exp(-gamma ||x - x'||^2), the Gaussian radial basis
    function, Eq. (2-rbfkernel)."""

    def __init__(self, gamma=1.0):
        self.gamma = gamma
        self.name = f"rbf(gamma={gamma:g})"

    def __call__(self, X, Y):
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        d2 = (X ** 2).sum(1)[:, None] + (Y ** 2).sum(1)[None, :] - 2 * X @ Y.T
        return np.exp(-self.gamma * np.maximum(d2, 0.0))


class TanhKernel(Kernel):
    """K(x, x') = tanh(gamma x^T x' + coef0): the sigmoid kernel, which is
    *not* a Mercer kernel for every choice of its parameters."""

    def __init__(self, gamma=1.0, coef0=-1.0):
        self.gamma, self.coef0 = gamma, coef0
        self.name = f"tanh(gamma={gamma:g},r={coef0:g})"

    def __call__(self, X, Y):
        return np.tanh(self.gamma * np.asarray(X, float) @ np.asarray(Y, float).T + self.coef0)


class PrecomputedKernel(Kernel):
    """Wraps a kernel *matrix* so that it can be handed to the SVM when the
    kernel values come from somewhere else, for example a quantum computer
    (Sec. 2.7).  `train` is the N x N training Gram matrix; `test`, when
    given, the M x N matrix between test and training points."""
    name = "precomputed"

    def __init__(self, train, test=None):
        self.train, self.test = np.asarray(train, float), test

    def __call__(self, X, Y):
        # X and Y are index arrays into the stored matrices
        X, Y = np.asarray(X, int).ravel(), np.asarray(Y, int).ravel()
        return self.train[np.ix_(X, Y)]


def quadratic_feature_map(X):
    """phi(x1, x2) = (x1^2, x2^2, sqrt(2) x1 x2), Eq. (2-phi2), for which
    phi(x)^T phi(x') = (x^T x')^2."""
    X = np.asarray(X, float)
    return np.c_[X[:, 0] ** 2, X[:, 1] ** 2, np.sqrt(2) * X[:, 0] * X[:, 1]]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from bookplot import savefig
    from datasets import two_moons

    rng = np.random.default_rng(0)
    # 1. the kernel trick in one line: phi(x)^T phi(x') = (x^T x')^2
    X = rng.normal(size=(5, 2))
    Phi = quadratic_feature_map(X)
    assert np.allclose(Phi @ Phi.T, (X @ X.T) ** 2)
    print("(x^T x')^2 equals phi(x)^T phi(x') for phi of Eq. (2-phi2)")

    # 2. Mercer's condition on the moons data
    Xm, ym = two_moons()
    for k in (LinearKernel(), PolynomialKernel(3, 1.0), RBFKernel(1.0),
              TanhKernel(1.0, -1.0), TanhKernel(0.1, 1.0)):
        print(f"{k!r:24s} Mercer (PSD Gram matrix on the moons): {k.is_mercer(Xm)}")

    # 3. Fig. 2-kerneltrick: a one-dimensional problem no threshold separates,
    #    linearly separable after x -> (x, x^2)
    x1 = np.linspace(-4, 4, 9)
    y1 = np.array([-1, -1, 1, 1, 1, 1, 1, -1, -1])
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.1))
    ax = axes[0]
    ax.axhline(0, color="k", lw=0.8)
    ax.scatter(x1[y1 < 0], 0 * x1[y1 < 0], marker="s", color="tab:blue", edgecolor="k", s=45, zorder=3)
    ax.scatter(x1[y1 > 0], 0 * x1[y1 > 0], marker="^", color="tab:green", edgecolor="k", s=45, zorder=3)
    ax.set_yticks([]); ax.set_xlabel("$x_1$"); ax.set_ylim(-0.3, 0.3)
    ax.set_title("input space: not separable by a threshold")
    ax = axes[1]
    ax.scatter(x1[y1 < 0], x1[y1 < 0] ** 2, marker="s", color="tab:blue", edgecolor="k", s=45, zorder=3)
    ax.scatter(x1[y1 > 0], x1[y1 > 0] ** 2, marker="^", color="tab:green", edgecolor="k", s=45, zorder=3)
    ax.axhline(6.5, color="tab:red", ls="--", lw=2, label="$x_2 = 6.5$")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2 = x_1^2$"); ax.legend(loc="upper center")
    ax.set_title("feature space $(x_1, x_1^2)$: a line separates")
    savefig(fig, 2, "kerneltrick")

    # 4. Fig. 2-rbflandmarks: Gaussian similarity to two landmarks
    gamma = 0.3
    def rbf1(x, l): return np.exp(-gamma * (x - l) ** 2)
    xs = np.linspace(-4.5, 4.5, 200)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.1))
    ax = axes[0]
    ax.plot(xs, rbf1(xs, -2), "g--", label="$K(x,\\ell_1=-2)$")
    ax.plot(xs, rbf1(xs, 1), "b:", label="$K(x,\\ell_2=1)$")
    ax.scatter(x1[y1 < 0], 0 * x1[y1 < 0], marker="s", color="tab:blue", edgecolor="k", s=45, zorder=3)
    ax.scatter(x1[y1 > 0], 0 * x1[y1 > 0], marker="^", color="tab:green", edgecolor="k", s=45, zorder=3)
    ax.scatter([-2, 1], [0, 0], s=160, alpha=0.4, color="red", zorder=2)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("similarity"); ax.legend(loc="upper right")
    ax = axes[1]
    f1, f2 = rbf1(x1, -2), rbf1(x1, 1)
    ax.scatter(f1[y1 < 0], f2[y1 < 0], marker="s", color="tab:blue", edgecolor="k", s=45, zorder=3)
    ax.scatter(f1[y1 > 0], f2[y1 > 0], marker="^", color="tab:green", edgecolor="k", s=45, zorder=3)
    ax.plot([-0.1, 1.1], [0.57, -0.1], "r--", lw=2)
    ax.set_xlabel("$K(x,\\ell_1)$"); ax.set_ylabel("$K(x,\\ell_2)$")
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    savefig(fig, 2, "rbflandmarks")
    print("figures kerneltrick and rbflandmarks written")
