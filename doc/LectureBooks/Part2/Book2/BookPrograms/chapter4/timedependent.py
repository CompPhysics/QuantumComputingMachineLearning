"""Chapter 4, Sec. 4.8.4: an initial-value problem and a partial differential
equation in two variables, solved with Q-PINNs whose trial functions satisfy
the initial and boundary conditions exactly (additive corrections, Eq. (4-hardic)).

1. The harmonic oscillator u'' + omega^2 u = 0, u(0) = 0, u'(0) = 1 on [0, 1]
   (omega = 2, exact u = sin(omega t)/omega) with the trial function
   u_hat(t) = u(t) - u(0) + t [1 - u'(0)]; two qubits with the angle pi t/2
   (the interval covers a quarter of the period of the model), L = 2.
2. The wave equation u_tt = u_xx on (0,1) x (0,1], u(x,0) = sin(pi x),
   u_t(x,0) = 0, u(0,t) = u(1,t) = 0, exact u = sin(pi x) cos(pi t), with
   x and t on one qubit each and the trial function
   u_tilde = u_hat(x,t) - u_hat(x,0) - t u_hat_t(x,0) + sin(pi x),
   u_hat = u - (1-x) u(0,t) - x u(1,t); L = 1 and L = 2.

BFGS with shift-rule gradients throughout.  Produces Fig. 4-timedependent
and the numbers quoted in the text.  Runtime about 40 s.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]      # BookPrograms/
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np

from qnn import QNN
from qpinn import QPINN, residual_loss, train

OMEGA = 2.0


# ------------------------------------------------------------------ oscillator
def oscillator_trial(qp, T):
    """u_hat, u_hat', u_hat'' for u_hat(t) = u(t) - u(0) + t (1 - u'(0))."""
    t = np.asarray(T, float).reshape(-1)
    u0, du0 = qp.functional([[0.0]]), qp.functional([[0.0]], (0,))
    u = qp.functional(t[:, None]) - np.ones_like(t) * u0 - t * du0 + t
    du = qp.functional(t[:, None], (0,)) - np.ones_like(t) * du0 + 1.0
    d2u = qp.functional(t[:, None], (0, 0))
    return u, du, d2u


def solve_oscillator(L=2, N=20, rng=0, scale=np.pi / 2):
    qnn = QNN(2, L)
    qp = QPINN(qnn, np.full((2, 1), scale))
    t_col = (np.arange(N) + 0.5) / N
    u, du, d2u = oscillator_trial(qp, t_col)
    lg = residual_loss(d2u + OMEGA ** 2 * u)
    theta, hist = train(lg, qnn.random_parameters(rng))
    return qp, theta, hist


# ------------------------------------------------------------------ wave equation
def wave_trial(qp, X):
    """u_tilde, u_tilde_tt and u_tilde_xx at the points X = (x, t)."""
    X = np.atleast_2d(np.asarray(X, float))
    x, t = X[:, 0], X[:, 1]
    zeros, ones = np.zeros_like(x), np.ones_like(x)

    def hat(T, deriv=()):
        """Derivatives of u_hat = u - (1-x) u(0,t) - x u(1,t) at the points (x, T)."""
        F = qp.functional(np.c_[x, T], deriv)
        left, right = np.c_[zeros, T], np.c_[ones, T]
        if 0 not in deriv:                               # d/dt only: the correction survives
            F = F - (1 - x) * qp.functional(left, deriv) - x * qp.functional(right, deriv)
        elif deriv.count(0) == 1:                        # one d/dx: the linear factors differentiate
            dt = tuple(d for d in deriv if d != 0)
            F = F + ones * qp.functional(left, dt) - ones * qp.functional(right, dt)
        return F                                         # two d/dx: the correction vanishes

    u = hat(t) - hat(zeros) - t * hat(zeros, (1,)) + np.sin(np.pi * x)
    u_tt = hat(t, (1, 1))
    u_xx = hat(t, (0, 0)) - hat(zeros, (0, 0)) - t * hat(zeros, (0, 0, 1)) - np.pi ** 2 * np.sin(np.pi * x)
    return u, u_tt, u_xx


def wave_exact(x, t):
    return np.sin(np.pi * x) * np.cos(np.pi * t)


def solve_wave(L=1, n_col=8, rng=0):
    qnn = QNN(2, L)
    qp = QPINN(qnn, [[np.pi, 0.0], [0.0, np.pi]])        # phi_0 = pi x, phi_1 = pi t
    g = (np.arange(n_col) + 0.5) / n_col
    X = np.array([(x, t) for t in g for x in g])
    u, u_tt, u_xx = wave_trial(qp, X)
    lg = residual_loss(u_tt - u_xx)
    theta, hist = train(lg, qnn.random_parameters(rng))
    return qp, theta, hist, X


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    import bookplot

    # ---- oscillator
    t0 = time.time()
    best = None
    for seed in range(3):
        res = solve_oscillator(rng=seed)
        if best is None or res[2][-1] < best[2][-1]:
            best = res
    qp, theta, hist_osc = best
    t_plot = np.linspace(0, 1, 201)
    u, du, d2u = oscillator_trial(qp, t_plot)
    u_osc = u.value(theta)
    exact_osc = np.sin(OMEGA * t_plot) / OMEGA
    print(f"harmonic oscillator, omega = {OMEGA}, 2 qubits, L = 2: final loss {hist_osc[-1]:.1e} after "
          f"{len(hist_osc) - 1} BFGS iterations, L2 error {np.sqrt(np.mean((u_osc - exact_osc) ** 2)):.1e}, "
          f"max error {np.max(np.abs(u_osc - exact_osc)):.1e}, u_hat(0) = {u.value(theta)[0]:.1e}, "
          f"u_hat'(0) = {du.value(theta)[0]:.6f} ({time.time() - t0:.1f} s)")

    # ---- wave equation
    xg, tg = np.meshgrid(np.linspace(0, 1, 41), np.linspace(0, 1, 41))
    G = np.c_[xg.ravel(), tg.ravel()]
    waves = {}
    for L in (1, 2):
        t0 = time.time()
        best = None
        for seed in range(3):
            res = solve_wave(L, rng=seed)
            if best is None or res[2][-1] < best[2][-1]:
                best = res
        qp_w, theta_w, hist_w, X_col = best
        u_grid = wave_trial(qp_w, G)[0].value(theta_w).reshape(xg.shape)
        err = u_grid - wave_exact(xg, tg)
        waves[L] = (qp_w, theta_w, hist_w, u_grid, err)
        print(f"wave equation, 2 qubits, L = {L} ({qp_w.qnn.n_params} parameters, {len(X_col)} collocation points): "
              f"final loss {hist_w[-1]:.1e} after {len(hist_w) - 1} iterations, "
              f"L2 error {np.sqrt(np.mean(err ** 2)):.1e}, max error {np.max(np.abs(err)):.1e} on a 41 x 41 grid "
              f"({time.time() - t0:.1f} s)")
    # the single-layer output in closed form: a product of cosines, Eq. (4-productform)
    qnn1 = QNN(2, 1); th = qnn1.random_parameters(7)
    phi = np.random.default_rng(0).uniform(0, 2 * np.pi, (5, 2))
    f1 = qnn1.forward(phi, th)
    ty = qnn1.unpack(th)[0, 0, 0]                         # the R_y angles of the single layer
    print(f"single layer, two qubits: f = cos(phi_0 + theta_0) cos(phi_1 + theta_1), max deviation "
          f"{np.max(np.abs(f1 - np.cos(phi[:, 0] + ty[0]) * np.cos(phi[:, 1] + ty[1]))):.1e}")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    axes[0].plot(t_plot, exact_osc, "k-", label="$\\sin(\\omega t)/\\omega$")
    axes[0].plot(t_plot, u_osc, "--", label="Q-PINN, $L=2$")
    axes[0].set_xlabel("$t$"); axes[0].set_ylabel("$u(t)$"); axes[0].legend(fontsize=8)
    axes[0].set_title(f"$u''+\\omega^2u=0$, $\\omega={OMEGA:.0f}$", fontsize=10)
    for t_s, c in zip([0.0, 0.25, 0.5, 0.75], ["C0", "C1", "C2", "C3"]):
        xs = np.linspace(0, 1, 101)
        axes[1].plot(xs, wave_exact(xs, t_s), "-", color=c, lw=1, label=f"$t={t_s}$")
        for L, ls in [(1, "--"), (2, ":")]:
            qp_w, theta_w = waves[L][0], waves[L][1]
            axes[1].plot(xs, wave_trial(qp_w, np.c_[xs, np.full_like(xs, t_s)])[0].value(theta_w), ls, color=c)
    axes[1].set_xlabel("$x$"); axes[1].set_ylabel("$u(x,t)$"); axes[1].legend(fontsize=7)
    axes[1].set_title("wave equation: exact, $L=1$ (dashed), $L=2$ (dotted)", fontsize=9)
    axes[2].semilogy(hist_osc, label="oscillator, $L=2$")
    for L in (1, 2):
        axes[2].semilogy(waves[L][2], label=f"wave equation, $L={L}$")
    axes[2].set_xlabel("BFGS iteration"); axes[2].set_ylabel("residual loss"); axes[2].legend(fontsize=8)
    fig.tight_layout()
    bookplot.savefig(fig, 4, "timedependent")
    print("figure timedependent written")
