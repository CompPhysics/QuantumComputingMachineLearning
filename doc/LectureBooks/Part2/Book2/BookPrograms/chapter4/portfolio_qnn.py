"""Chapter 4, Sec. 4.7.3: a data re-uploading QNN as a scoring function for
the portfolio problem of chapter 3.

Every asset i is described by the features phi_i = (mu_i, sigma_i, 1)
(standardised expected return and volatility, and a bias) and scored by the
same single-qubit circuit, Eq. (4-scorecircuit),

    |psi_i> = prod_{l=1}^{L} R_z(w^z_l . phi_i) R_y(w^y_l . phi_i) |0>,   s_i = <psi_i|Z|psi_i>,

whose weights are shared between the assets.  The scores are turned into
soft selection probabilities p_i = sigma(alpha (s_i - tau)) with the
threshold tau fixed by sum_i p_i = K (Eq. (4-softselect)), and the weights
are trained on the mean-variance cost of chapter 3 evaluated at p,
Eq. (4-softcost).  The gradient is exact: parameter shifts through the
circuits and the chain rule through the threshold.  Data: the synthetic
and the S&P 500 problems of chapter 3 (BookPrograms/chapter3/portfolio.py).
Runtime about 5 s.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]      # BookPrograms/
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np

from optimize import Adam
from portfolio import MU_SYNTH, SIGMA_SYNTH, TICKERS_SYNTH, TICKERS_SP, sp500, feasible_table


def features(mu, Sigma):
    """Standardised (mu_i, sigma_i) plus a bias feature."""
    sig = np.sqrt(np.diag(Sigma))
    f = np.c_[(mu - mu.mean()) / mu.std(), (sig - sig.mean()) / sig.std(), np.ones(len(mu))]
    return f


def ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], complex)


def rz(t):
    return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])


class ScoreQNN:
    """The single-qubit re-uploading circuit with weights W (L, 2, F)."""

    def __init__(self, n_features, n_layers=3):
        self.L, self.F = n_layers, n_features
        self.n_params = 2 * n_layers * n_features

    def score(self, phi, W):
        """<Z> for one feature vector phi and weights W (L, 2, F)."""
        psi = np.array([1.0, 0.0], complex)
        for l in range(self.L):
            psi = rz(W[l, 1] @ phi) @ (ry(W[l, 0] @ phi) @ psi)
        return float(abs(psi[0]) ** 2 - abs(psi[1]) ** 2)

    def scores(self, Phi, w):
        W = w.reshape(self.L, 2, self.F)
        return np.array([self.score(phi, W) for phi in Phi])

    def jacobian(self, Phi, w):
        """ds_i/dw_mu: the angle of gate (l, r) is W[l, r] . phi, so the
        parameter-shift derivative with respect to the angle is multiplied
        by the feature phi_f (chain rule).  Shape (n_assets, P)."""
        W = w.reshape(self.L, 2, self.F)
        J = np.zeros((len(Phi), self.n_params))
        for i, phi in enumerate(Phi):
            for l in range(self.L):
                for r in range(2):
                    # shift the angle of gate (l, r) by +-pi/2 through the bias weight (phi_bias = 1)
                    Wp, Wm = W.copy(), W.copy()
                    Wp[l, r, -1] += np.pi / 2; Wm[l, r, -1] -= np.pi / 2
                    d_angle = 0.5 * (self.score(phi, Wp) - self.score(phi, Wm))
                    J[i, (l * 2 + r) * self.F:(l * 2 + r + 1) * self.F] = d_angle * phi
        return J


def soft_selection(s, K, alpha=5.0):
    """p_i = sigma(alpha (s_i - tau)) with tau such that sum_i p_i = K (bisection);
    returns p and dp/ds (n x n), the derivative through the implicit threshold."""
    lo, hi = s.min() - 10, s.max() + 10
    for _ in range(100):
        tau = 0.5 * (lo + hi)
        p = 1 / (1 + np.exp(-alpha * (s - tau)))
        if p.sum() > K:
            lo = tau
        else:
            hi = tau
    dp = alpha * p * (1 - p)                             # dp_i/d(s_i - tau)
    dtau_ds = dp / dp.sum()                              # from d/ds (sum p) = 0
    return p, np.diag(dp) - np.outer(dp, dtau_ds)


def soft_cost(p, mu, Sigma, lam, A, K):
    """C(p) = -mu.p + lam p.Sigma p + A (sum p - K)^2 and its gradient."""
    c = -mu @ p + lam * p @ Sigma @ p + A * (p.sum() - K) ** 2
    g = -mu + 2 * lam * Sigma @ p + 2 * A * (p.sum() - K)
    return float(c), g


def train(model, Phi, mu, Sigma, lam, A, K, steps=150, lr=0.05, rng=0, alpha=5.0):
    rng = np.random.default_rng(rng)
    w = rng.uniform(-np.pi, np.pi, model.n_params)
    opt = Adam(lr=lr)
    hist = []
    for _ in range(steps):
        s = model.scores(Phi, w)
        p, dp_ds = soft_selection(s, K, alpha)
        c, dc_dp = soft_cost(p, mu, Sigma, lam, A, K)
        hist.append(c)
        grad = (dc_dp @ dp_ds) @ model.jacobian(Phi, w)
        w = opt.step(w, grad)
    return w, np.array(hist)


def run(mu, Sigma, lam, A, K, tickers, name, restarts=4):
    Phi = features(mu, Sigma)
    model = ScoreQNN(Phi.shape[1], n_layers=3)
    best = None
    for seed in range(restarts):
        w, hist = train(model, Phi, mu, Sigma, lam, A, K, rng=seed)
        if best is None or hist[-1] < best[1][-1]:
            best = (w, hist)
    w, hist = best
    s = model.scores(Phi, w)
    p, _ = soft_selection(s, K)
    chosen = np.argsort(s)[::-1][:K]
    x = np.zeros(len(mu)); x[chosen] = 1
    rows = feasible_table(mu, Sigma, lam, K, tickers)
    print(f"\n{name}: lambda = {lam}, A = {A}, K = {K}, {model.n_params} weights, best of {restarts} starts")
    print(f"  {'asset':>6}  {'mu':>7}  {'sigma':>7}  {'score':>7}  {'p':>6}")
    for i, t in enumerate(tickers):
        print(f"  {t:>6}  {mu[i]:>7.3f}  {np.sqrt(Sigma[i, i]):>7.3f}  {s[i]:>+7.3f}  {p[i]:>6.3f}  {'<- selected' if x[i] else ''}")
    cost = -mu @ x + lam * x @ Sigma @ x
    print(f"  soft cost after training {hist[-1]:+.5f}; hard portfolio {'+'.join(t for t, v in zip(tickers, x) if v)} "
          f"with cost {cost:+.5f}; best feasible portfolio {rows[0][0]} with cost {rows[0][2]:+.5f}")
    return hist


if __name__ == "__main__":
    import time
    t0 = time.time()
    hA = run(MU_SYNTH, SIGMA_SYNTH, lam=1.0, A=0.3, K=2, tickers=TICKERS_SYNTH, name="Synthetic data")
    _, _, mu_sp, Sigma_sp = sp500()
    hB = run(mu_sp, Sigma_sp, lam=2.0, A=0.3, K=2, tickers=TICKERS_SP, name="S&P data")
    print(f"({time.time() - t0:.1f} s)")
