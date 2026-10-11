"""Chapter 2, Sec. 2.10: training the feature map by kernel-target alignment.

A feature map with parameters theta, U(x; theta), defines a family of
kernels K_theta.  The kernel-target alignment of Eq. (2-alignment),

    A(theta) = sum_ij K_ij y_i y_j / (||K||_F ||y y^T||_F),

is the cosine of the angle between the Gram matrix and the ideal kernel
y y^T; maximizing it over theta before the support-vector machine is
trained is the kernel-learning scheme of Hubregtsen et al. (2022).  The
gradient of every kernel entry follows from the parameter-shift rule,
Eq. (2-kernelshift): a parameter that appears in both U(x) and U(x')^dagger
is shifted in each of the two factors separately.  Run as a script the
module trains the embedding of feature_maps.TrainableEmbedding on the two
moons, checks the parameter-shift gradient against finite differences, and
draws Fig. 2-alignment (alignment and test accuracy against the training
step).
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]      # BookPrograms/
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np

from feature_maps import TrainableEmbedding
from qsvm import QSVM


class KernelAlignment:
    """Maximizes the alignment of a trainable quantum kernel with y y^T."""

    def __init__(self, embedding, X, y):
        self.fm, self.X, self.y = embedding, np.asarray(X, float), np.asarray(y, float)
        self.yy = np.outer(self.y, self.y)

    def kernel(self, theta_bra, theta_ket=None):
        """K_ij = |<phi(x_i; theta_bra)|phi(x_j; theta_ket)>|^2."""
        Phi_a = self.fm.states(self.X, theta_bra)
        Phi_b = Phi_a if theta_ket is None else self.fm.states(self.X, theta_ket)
        return np.abs(Phi_a.conj() @ Phi_b.T) ** 2

    @staticmethod
    def alignment_of(K, yy):
        return float((K * yy).sum() / (np.linalg.norm(K) * np.linalg.norm(yy)))

    def alignment(self, theta):
        return self.alignment_of(self.kernel(theta), self.yy)

    def kernel_gradient(self, theta, shift=np.pi / 2):
        """dK/dtheta_k by the parameter-shift rule, Eq. (2-kernelshift), as an
        (n_params, N, N) array.  Every gate is a rotation exp(-i a P/2), so
        d/da = [f(a + pi/2) - f(a - pi/2)]/2; the chain factor da/dtheta_k
        (x_q for an input weight) multiplies the row of the shifted bra.
        Since K(theta_a, theta_b) = K(theta_b, theta_a)^T, the ket shift is
        the transpose of the bra shift and costs nothing extra."""
        Phi0 = self.fm.states(self.X, theta)
        grads = []
        for k in range(len(theta)):
            Kp = np.abs(self.fm.states(self.X, theta, shift=(k, +shift)).conj() @ Phi0.T) ** 2
            Km = np.abs(self.fm.states(self.X, theta, shift=(k, -shift)).conj() @ Phi0.T) ** 2
            M = self.fm.chain(self.X, k)[:, None] * 0.5 * (Kp - Km)   # bra factor shifted
            grads.append(M + M.T)                                    # ... and the ket factor
        return np.array(grads)

    def gradient(self, theta):
        """dA/dtheta from the chain rule applied to Eq. (2-alignment)."""
        K = self.kernel(theta)
        yy = self.yy
        nK, nY = np.linalg.norm(K), np.linalg.norm(yy)
        a = (K * yy).sum()
        dK = self.kernel_gradient(theta)
        # d/dK of <K,yy>/(|K||yy|) = yy/(|K||yy|) - <K,yy> K/(|K|^3 |yy|)
        dA_dK = yy / (nK * nY) - a * K / (nK ** 3 * nY)
        return np.array([(dA_dK * g).sum() for g in dK])

    def train(self, theta0, steps=30, lr=0.5, callback=None):
        """Plain gradient ascent; returns the trajectory of (theta, A)."""
        theta = np.array(theta0, float)
        history = [(theta.copy(), self.alignment(theta))]
        for step in range(steps):
            theta = theta + lr * self.gradient(theta)
            history.append((theta.copy(), self.alignment(theta)))
            if callback:
                callback(step + 1, theta, history[-1][1])
        return history


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from bookplot import savefig
    from datasets import two_moons, train_test_split, Scaler

    X, y = two_moons(100, 0.15)
    Xtr, Xte, ytr, yte = train_test_split(X, y, 0.3, rng=0)
    sc = Scaler("angle").fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)

    fm = TrainableEmbedding(2, layers=1, reps=2, rng=0)
    ka = KernelAlignment(fm, Ztr, ytr)
    theta0 = fm.theta.copy()
    # start from badly scaled inputs: weights w = 2.5 instead of 1
    theta0[[k for k in range(fm.n_params) if k % fm.per_rep < fm.n_qubits]] = 2.5

    # the parameter-shift gradient against central finite differences
    g_ps = ka.gradient(theta0)
    h = 1e-5
    g_fd = np.array([(ka.alignment(theta0 + h * e) - ka.alignment(theta0 - h * e)) / (2 * h)
                     for e in np.eye(len(theta0))])
    assert np.allclose(g_ps, g_fd, atol=1e-6), (g_ps, g_fd)
    print(f"parameter-shift gradient of the alignment agrees with finite differences "
          f"(max difference {np.abs(g_ps - g_fd).max():.1e})")

    # train, and record the SVM accuracy along the way
    def evaluate(theta):
        fm.theta = theta
        model = QSVM(fm, C=1.0).fit(Ztr, ytr)
        return model.score(Ztr, ytr), model.score(Zte, yte)

    acc = [evaluate(theta0)]
    print(f"step  0: alignment {ka.alignment(theta0):.4f}  train/test accuracy {acc[0][0]:.2f}/{acc[0][1]:.2f}")
    def report(step, theta, A):
        if step % 5 == 0:
            acc.append(evaluate(theta))
            print(f"step {step:2d}: alignment {A:.4f}  train/test accuracy {acc[-1][0]:.2f}/{acc[-1][1]:.2f}")
    history = ka.train(theta0, steps=40, lr=1.0, callback=report)
    A = [h[1] for h in history]
    w = history[-1][0][[k for k in range(fm.n_params) if k % fm.per_rep < fm.n_qubits]]
    print("input weights after training:", np.round(w, 3))

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    axes[0].plot(range(len(A)), A, "o-", ms=3)
    axes[0].set_xlabel("gradient-ascent step"); axes[0].set_ylabel("kernel-target alignment $A$")
    steps = [0] + list(range(5, 41, 5))
    axes[1].plot(steps, [a[0] for a in acc], "s-", label="training")
    axes[1].plot(steps, [a[1] for a in acc], "o--", label="test")
    axes[1].set_xlabel("gradient-ascent step"); axes[1].set_ylabel("accuracy of the QSVM"); axes[1].legend()
    savefig(fig, 2, "alignment")
    print("figure alignment written")
