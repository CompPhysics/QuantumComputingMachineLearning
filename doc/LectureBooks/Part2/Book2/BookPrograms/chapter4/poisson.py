"""Chapter 4, Sec. 4.8.3: the one-dimensional Poisson equation

    -u''(x) = f(x) = pi^2 sin(pi x),  x in (0, 1),  u(0) = u(1) = 0,

exact solution u*(x) = sin(pi x), solved with a two-qubit Q-PINN (encoding
angle pi x on both qubits, L = 1, 2, 3 re-uploading layers).  The boundary
conditions are built into the trial function, Eq. (4-hardbc),
u_hat(x) = u_theta(x) - (1-x) u_theta(0) - x u_theta(1), so the loss is the
mean-square residual at 20 interior collocation points (strong form,
Eq. (4-pinnloss)), minimised with BFGS and shift-rule gradients; the Ritz
form Eq. (4-ritz) is trained for comparison.  Produces Table 4-poisson and
Fig. 4-poisson.  Runtime about 15 s.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]      # BookPrograms/
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np

from qnn import QNN
from qpinn import QPINN, residual_loss, ritz_loss, train


def source(x):
    return np.pi ** 2 * np.sin(np.pi * x)


def exact(x):
    return np.sin(np.pi * x)


def dirichlet_trial(qp, X):
    """u_hat, u_hat' and u_hat'' at the points X as functionals of the circuit,
    for u_hat(x) = u(x) - (1-x) u(0) - x u(1)."""
    x = np.asarray(X, float).reshape(-1)
    u0, u1 = qp.functional([[0.0]]), qp.functional([[1.0]])
    u = qp.functional(x[:, None]) - (1 - x) * u0 - x * u1
    du = qp.functional(x[:, None], (0,)) - np.ones_like(x) * u1 + np.ones_like(x) * u0
    d2u = qp.functional(x[:, None], (0, 0))
    return u, du, d2u


def solve_poisson(L, x_col, steps=300, rng=0, form="strong", method="bfgs"):
    qnn = QNN(2, L)
    qp = QPINN(qnn, np.full((2, 1), np.pi))
    u, du, d2u = dirichlet_trial(qp, x_col)
    if form == "strong":
        R = -1.0 * d2u - source(x_col)                   # residual -u'' - f
        lg = residual_loss(R)
        n_eval = len(R.points)
    else:
        w = np.full(len(x_col), 1.0 / len(x_col))        # midpoint rule on (0, 1)
        lg = ritz_loss(du, u, source(x_col), w)
        n_eval = len(du.points) + len(u.points)
    theta, hist = train(lg, qnn.random_parameters(rng), method=method, steps=steps)
    return qp, theta, hist, n_eval


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    import bookplot

    N = 20
    x_col = (np.arange(N) + 0.5) / N                     # midpoints of 20 cells
    x_plot = np.linspace(0, 1, 201)
    print("Table 4-poisson: strong-form Q-PINN, 2 qubits, 20 collocation points, BFGS (best of 3 starts)")
    print(f"{'layers':>6}  {'params':>6}  {'iterations':>10}  {'final loss':>10}  {'L2 error':>9}  {'max error':>9}  {'evaluations/gradient':>20}  {'time':>6}")
    solutions = {}
    for L in (1, 2, 3):
        t0 = time.time()
        best = None
        for seed in range(3):
            res = solve_poisson(L, x_col, rng=seed)
            if best is None or res[2][-1] < best[2][-1]:
                best = res
        qp, theta, hist, n_eval = best
        u_hat = dirichlet_trial(qp, x_plot)[0].value(theta)
        err = u_hat - exact(x_plot)
        l2 = np.sqrt(np.mean(err ** 2))
        P = qp.qnn.n_params
        print(f"{L:>6}  {P:>6}  {len(hist) - 1:>10}  {hist[-1]:>10.1e}  {l2:>9.1e}  {np.max(np.abs(err)):>9.1e}  {2 * P * n_eval:>20}  {time.time() - t0:>5.1f}s")
        solutions[L] = (u_hat, hist)
    # Ritz form, L = 2
    t0 = time.time()
    qp, theta, hist_ritz, n_eval = solve_poisson(2, x_col, rng=0, form="ritz")
    u_ritz = dirichlet_trial(qp, x_plot)[0].value(theta)
    l2_ritz = np.sqrt(np.mean((u_ritz - exact(x_plot)) ** 2))
    E_exact = -np.pi ** 2 / 4
    print(f"Ritz form, L=2: energy {hist_ritz[-1]:.5f} (exact {E_exact:.5f}; midpoint rule on the exact solution "
          f"{np.mean(0.5 * (np.pi * np.cos(np.pi * x_col)) ** 2 - source(x_col) * exact(x_col)):.5f}), "
          f"L2 error {l2_ritz:.1e}, {len(hist_ritz) - 1} iterations ({time.time() - t0:.1f} s)")
    # the single layer cannot represent sin(pi x): its spectrum has no k = 1 component
    qnn1 = QNN(2, 1); k, c = qnn1.fourier_coefficients(qnn1.random_parameters(5))
    print(f"single layer, random parameters: |c_k| for k = 0..2: {np.round(np.abs(c[2:]), 4)}")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    axes[0].plot(x_plot, exact(x_plot), "k-", label="$\\sin\\pi x$")
    for L, (u_hat, hist) in solutions.items():
        axes[0].plot(x_plot, u_hat, "--", label=f"$L={L}$")
    axes[0].plot(x_plot, u_ritz, ":", label="Ritz, $L=2$")
    axes[0].plot(x_col, np.zeros_like(x_col), "r|", ms=6)
    axes[0].set_xlabel("$x$"); axes[0].set_ylabel("$u(x)$"); axes[0].legend(fontsize=8)
    for L, (u_hat, hist) in solutions.items():
        axes[1].semilogy(x_plot, np.abs(u_hat - exact(x_plot)) + 1e-16, label=f"$L={L}$")
    axes[1].semilogy(x_plot, np.abs(u_ritz - exact(x_plot)) + 1e-16, ":", label="Ritz, $L=2$")
    axes[1].set_xlabel("$x$"); axes[1].set_ylabel("$|\\hat u_\\theta - u^\\star|$"); axes[1].legend(fontsize=8)
    for L, (u_hat, hist) in solutions.items():
        axes[2].semilogy(hist, label=f"strong form, $L={L}$")
    axes[2].semilogy(np.abs(hist_ritz - E_exact), ":", label="Ritz $|E-E^\\star|$, $L=2$")
    axes[2].set_xlabel("BFGS iteration"); axes[2].set_ylabel("loss"); axes[2].legend(fontsize=7)
    fig.tight_layout()
    bookplot.savefig(fig, 4, "poisson")
    print("figure poisson written")
