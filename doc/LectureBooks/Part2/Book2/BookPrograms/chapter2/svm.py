"""Chapter 2, Secs. 2.2-2.6: the support-vector machine, trained in its dual
form by sequential minimal optimization.

The class `SVM` solves the dual problem of Eq. (2-dual),

    max_lambda  sum_i lambda_i - (1/2) sum_ij lambda_i lambda_j y_i y_j K(x_i, x_j)
    subject to  0 <= lambda_i <= C,   sum_i lambda_i y_i = 0,

for any kernel object of kernels.py (or a quantum kernel of
feature_maps.py: the solver only ever sees the Gram matrix).  The solver
is Platt's sequential minimal optimization (SMO), which updates two
multipliers at a time so that the equality constraint stays satisfied
(Sec. 2.6); it needs nothing beyond NumPy.  C = numpy.inf gives the
hard-margin machine of Eq. (2-primal), a finite C the soft margin of
Eq. (2-softprimal).  `qp_matrices` returns the matrices (P, q, G, h, A, b)
of the generic quadratic program of Eq. (2-qp), for readers who prefer an
off-the-shelf solver such as CVXOPT (`solve_with_cvxopt`).

Run as a script the module trains the hard-margin machine on two blobs
(Fig. 2-hardmargin), the soft-margin one for several C (Fig. 2-softmargin)
and polynomial and Gaussian kernels on the two moons (Fig. 2-moons).
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]      # BookPrograms/
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np

from kernels import LinearKernel, PolynomialKernel, RBFKernel


class SVM:
    """A binary support-vector classifier with labels y = +1 / -1."""

    def __init__(self, kernel=None, C=np.inf, tol=1e-5, max_passes=3, max_iter=20000, rng=0):
        self.kernel = LinearKernel() if kernel is None else kernel
        self.C, self.tol, self.max_passes, self.max_iter = C, tol, max_passes, max_iter
        self.rng = np.random.default_rng(rng)

    # ------------------------------------------------------------ training
    def fit(self, X, y, K=None):
        """Train on (X, y).  `K` may supply the training Gram matrix
        directly (a precomputed, e.g. quantum, kernel)."""
        X = np.asarray(X, float); y = np.asarray(y, float)
        n = len(y)
        K = self.kernel(X, X) if K is None else np.asarray(K, float)
        self.X_train, self.y_train, self.K_train = X, y, K
        lam, b = np.zeros(n), 0.0
        Q = (y[:, None] * y[None, :]) * K                 # Q_ij = y_i y_j K_ij
        state = {"lam": lam, "b": b}

        def take_step(i, j):
            """Optimize lambda_i and lambda_j jointly, Eqs. (2-smo); returns
            True when the pair moved."""
            lam, b = state["lam"], state["b"]
            E = Q @ lam * y - y + b                       # E_i = f(x_i) - y_i
            eta = K[i, i] + K[j, j] - 2 * K[i, j]
            if eta <= 1e-12:
                return False
            # the box that keeps 0 <= lambda <= C and sum_i lambda_i y_i fixed
            if y[i] != y[j]:
                L, H = max(0.0, lam[j] - lam[i]), min(self.C, self.C + lam[j] - lam[i])
            else:
                L, H = max(0.0, lam[i] + lam[j] - self.C), min(self.C, lam[i] + lam[j])
            if L >= H:
                return False
            lam_j_old, lam_i_old = lam[j], lam[i]
            lam_j = np.clip(lam_j_old + y[j] * (E[i] - E[j]) / eta, L, H)
            if abs(lam_j - lam_j_old) < 1e-12 * (lam_j + lam_j_old + 1e-12):
                return False
            lam_i = lam_i_old + y[i] * y[j] * (lam_j_old - lam_j)
            lam[i], lam[j] = lam_i, lam_j
            # the threshold b from whichever multiplier is unbounded
            b1 = b - E[i] - y[i] * (lam_i - lam_i_old) * K[i, i] - y[j] * (lam_j - lam_j_old) * K[i, j]
            b2 = b - E[j] - y[i] * (lam_i - lam_i_old) * K[i, j] - y[j] * (lam_j - lam_j_old) * K[j, j]
            if 0 < lam_i < self.C:
                state["b"] = b1
            elif 0 < lam_j < self.C:
                state["b"] = b2
            else:
                state["b"] = (b1 + b2) / 2
            return True

        passes, self.n_iter, self.converged = 0, 0, True
        while passes < self.max_passes:
            if self.n_iter >= self.max_iter:      # safety net: a Gram matrix that
                self.converged = False            # is not PSD can make SMO cycle
                break
            changed = 0
            lam = state["lam"]
            E = Q @ lam * y - y + state["b"]
            for i in self.rng.permutation(n):
                # the KKT conditions of Eq. (2-kkt): r_i = y_i E_i = y_i f(x_i) - 1
                r = y[i] * E[i]
                if not ((r < -self.tol and lam[i] < self.C) or (r > self.tol and lam[i] > 0)):
                    continue
                # second multiplier: try the candidates in the order of
                # decreasing |E_i - E_j| until one of them can move
                for j in np.argsort(-np.abs(E - E[i])):
                    if j != i and take_step(i, int(j)):
                        changed += 1
                        self.n_iter += 1
                        E = Q @ lam * y - y + state["b"]
                        break
            passes = passes + 1 if changed == 0 else 0
        b = state["b"]
        self.lam = lam
        self.support = np.flatnonzero(lam > 1e-8)
        # recompute b from the margin support vectors, Eq. (2-bias)
        on_margin = np.flatnonzero((lam > 1e-8) & (lam < self.C - 1e-8))
        if len(on_margin):
            f0 = (lam * y) @ K[:, on_margin]
            b = float(np.mean(y[on_margin] - f0))
        self.b = b
        return self

    # ---------------------------------------------------------- inference
    def decision_function(self, X, K=None):
        """f(x) = sum_i lambda_i y_i K(x_i, x) + b, Eq. (2-decision)."""
        if K is None:
            K = self.kernel(np.asarray(X, float), self.X_train)
        return K @ (self.lam * self.y_train) + self.b

    def predict(self, X, K=None):
        return np.sign(self.decision_function(X, K))

    def score(self, X, y, K=None):
        return float(np.mean(self.predict(X, K) == np.sign(y)))

    # ------------------------------------------------------------ extras
    def weight_vector(self):
        """w = sum_i lambda_i y_i x_i, Eq. (2-wsum); meaningful for the linear
        kernel only."""
        return (self.lam * self.y_train) @ self.X_train

    def margin(self):
        """The geometric margin 1/||w|| of the linear machine."""
        return 1.0 / np.linalg.norm(self.weight_vector())

    def dual_objective(self):
        lam, y, K = self.lam, self.y_train, self.K_train
        return lam.sum() - 0.5 * (lam * y) @ K @ (lam * y)

    @staticmethod
    def qp_matrices(K, y, C=np.inf):
        """The quadratic program of Eq. (2-qp) in the convention
        min (1/2) lam^T P lam + q^T lam,  G lam <= h,  A lam = b."""
        n = len(y)
        P = (y[:, None] * y[None, :]) * K
        q = -np.ones(n)
        if np.isinf(C):
            G, h = -np.eye(n), np.zeros(n)
        else:
            G = np.vstack([-np.eye(n), np.eye(n)])
            h = np.r_[np.zeros(n), C * np.ones(n)]
        A, b = y.reshape(1, n).astype(float), np.zeros(1)
        return P, q, G, h, A, b

    def solve_with_cvxopt(self, X, y):
        """The same dual solved by CVXOPT, if it is installed; returns the
        multipliers for comparison with the SMO result."""
        try:
            from cvxopt import matrix, solvers
        except ImportError:
            return None
        X = np.asarray(X, float); y = np.asarray(y, float)
        P, q, G, h, A, b = self.qp_matrices(self.kernel(X, X), y, self.C)
        solvers.options["show_progress"] = False
        sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b))
        return np.array(sol["x"]).ravel()


def _margin_lines(ax, svm, xlim):
    """Draw the decision line and the two margin lines of a linear SVM."""
    w, b = svm.weight_vector(), svm.b
    xs = np.array(xlim)
    for c, style in ((0, "k-"), (1, "k--"), (-1, "k--")):
        ax.plot(xs, (c - b - w[0] * xs) / w[1], style, lw=1.2 if c == 0 else 0.8)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from bookplot import savefig, scatter_classes, decision_regions
    from datasets import two_blobs, two_moons, Scaler

    # ---------------------------------------------- hard margin on two blobs
    X, y = two_blobs(n=40, separation=3.5, noise=0.6, rng=4)
    svm = SVM(LinearKernel(), C=np.inf).fit(X, y)
    w, b = svm.weight_vector(), svm.b
    print(f"hard margin: w = ({w[0]:.4f}, {w[1]:.4f}), b = {b:.4f}, "
          f"margin 1/||w|| = {svm.margin():.4f}, {len(svm.support)} support vectors, "
          f"{svm.n_iter} SMO updates")
    # every support vector sits on the margin, every other point outside it
    f = svm.decision_function(X)
    assert np.allclose(y[svm.support] * f[svm.support], 1.0, atol=1e-4)
    assert np.all(y * f >= 1 - 1e-4)
    lam_qp = svm.solve_with_cvxopt(X, y)
    if lam_qp is not None:
        print("  max |lambda_SMO - lambda_CVXOPT| =", np.abs(lam_qp - svm.lam).max())
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    xlim, ylim = (-4.2, 4.2), (-2.5, 2.5)
    decision_regions(ax, svm.decision_function, xlim, ylim)
    scatter_classes(ax, X, y)
    _margin_lines(ax, svm, xlim)
    ax.scatter(X[svm.support, 0], X[svm.support, 1], s=120, facecolor="none",
               edgecolor="tab:red", linewidth=1.5, label="support vectors")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.legend(loc="upper left")
    savefig(fig, 2, "hardmargin")

    # --------------------------------------------- soft margin, overlapping
    X, y = two_blobs(n=60, separation=2.2, noise=0.8, rng=7)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    print("soft margin, overlapping blobs:")
    for ax, C in zip(axes, (0.05, 1.0, 100.0)):
        svm = SVM(LinearKernel(), C=C).fit(X, y)
        n_viol = int(np.sum(y * svm.decision_function(X) < 1 - 1e-6))
        print(f"  C = {C:6.2f}: margin {svm.margin():.3f}, {len(svm.support):2d} support vectors, "
              f"{n_viol:2d} margin violations, training accuracy {svm.score(X, y):.2f}")
        decision_regions(ax, svm.decision_function, xlim, ylim)
        scatter_classes(ax, X, y)
        _margin_lines(ax, svm, xlim)
        ax.scatter(X[svm.support, 0], X[svm.support, 1], s=110, facecolor="none",
                   edgecolor="tab:red", linewidth=1.2)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_xlabel("$x_1$"); ax.set_title(f"$C={C:g}$")
    axes[0].set_ylabel("$x_2$")
    savefig(fig, 2, "softmargin")

    # -------------------------------------------- kernels on the two moons
    X, y = two_moons(n=100, noise=0.15)
    X = Scaler().fit_transform(X)          # zero mean, unit variance per feature
    kernels = [LinearKernel(), PolynomialKernel(3, 1.0), RBFKernel(0.5), RBFKernel(5.0)]
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.1), sharey=True)
    print("two moons (standardized), C = 5:")
    for ax, k in zip(axes, kernels):
        svm = SVM(k, C=5.0).fit(X, y)
        print(f"  {k!r:16s} training accuracy {svm.score(X, y):.2f}, {len(svm.support)} support vectors")
        if isinstance(k, RBFKernel):          # optional check against the LIBSVM solver of scikit-learn
            try:
                from sklearn.svm import SVC
                clf = SVC(kernel="rbf", gamma=k.gamma, C=5.0).fit(X, y)
                lam = np.zeros(len(y)); lam[clf.support_] = np.abs(clf.dual_coef_[0])
                K = k(X, X); dual_sk = lam.sum() - 0.5 * (lam * y) @ K @ (lam * y)
                print(f"      dual objective: SMO {svm.dual_objective():.6f}, scikit-learn {dual_sk:.6f}")
            except ImportError:
                pass
        xlim, ylim = (-2.3, 2.3), (-2.0, 2.2)
        decision_regions(ax, svm.decision_function, xlim, ylim)
        scatter_classes(ax, X, y)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_xlabel("$x_1$"); ax.set_title(repr(k))
    axes[0].set_ylabel("$x_2$")
    savefig(fig, 2, "moons")
    print("figures hardmargin, softmargin and moons written")
