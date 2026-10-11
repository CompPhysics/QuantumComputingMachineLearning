"""Chapter 4, Sec. 4.6: barren plateaus.

The variance over random parameters of one component of the gradient of
C(theta) = <psi(theta)|O|psi(theta)> for hardware-efficient circuits (R_y
and R_z rotations, CNOT ladder) of growing width n and depth L, for the
local cost O_L = (1/n) sum_i |0><0|_i (an average of one-qubit projectors)
and the global cost O_G = |0...0><0...0| of Cerezo et al., Eq. (4-bpvariance).
Produces Fig. 4-barren and the numbers of Table 4-barren.  Runtime about 50 s.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]      # BookPrograms/
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np

from qnn import QNN, z_diagonal


def local_cost(n):
    """Diagonal of O_L = (1/n) sum_i (1 + Z_i)/2."""
    return np.mean([(1 + z_diagonal(n, (i,))) / 2 for i in range(n)], axis=0)


def global_cost(n):
    """Diagonal of O_G = |0...0><0...0| = prod_i (1 + Z_i)/2."""
    d = np.zeros(2 ** n); d[0] = 1.0
    return d


def gradient_samples(n, L, observable, n_samples=200, rng=0, parameter=0):
    """Samples of dC/dtheta_mu (parameter-shift rule) at uniformly random
    parameters, for the circuit of n qubits and L layers with the given
    diagonal observable; the input is x = 1/2 on every qubit."""
    rng = np.random.default_rng(rng)
    qnn = QNN(n, L, rotations=("Y", "Z"), observable=observable)
    angles = np.full((1, n), np.pi / 2)
    g = np.zeros(n_samples)
    for s in range(n_samples):
        theta = rng.uniform(-np.pi, np.pi, qnn.n_params)
        tp, tm = theta.copy(), theta.copy()
        tp[parameter] += np.pi / 2; tm[parameter] -= np.pi / 2
        g[s] = 0.5 * (qnn.forward(angles, tp)[0] - qnn.forward(angles, tm)[0])
    return g


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    import bookplot

    t0 = time.time()
    ns = np.arange(2, 11)
    print("Table 4-barren: Var[dC/dtheta_1] over 200 random parameter sets, depth L = n")
    print(f"{'n':>3}  {'local O_L':>14}  {'global O_G':>14}  {'2^-n':>8}  {'4^-n':>8}")
    var_local, var_global = [], []
    for n in ns:
        gl = gradient_samples(n, n, local_cost(n))
        gg = gradient_samples(n, n, global_cost(n))
        var_local.append(gl.var()); var_global.append(gg.var())
        print(f"{n:>3}  {gl.var():>14.2e}  {gg.var():>14.2e}  {2.0 ** -n:>8.1e}  {4.0 ** -n:>8.1e}")
    # fitted decay rates b in Var ~ b^{-n} from the last five points
    for name, v in [("local", var_local), ("global", var_global)]:
        slope = np.polyfit(ns[-5:], np.log(v[-5:]), 1)[0]
        print(f"  {name}: Var ~ b^(-n) with b = {np.exp(-slope):.2f} (fit over n = {ns[-5]}..{ns[-1]})")

    # depth dependence at fixed width
    n_fix = 6
    Ls = [1, 2, 3, 4, 6, 8, 12, 16]
    print(f"\nvariance against depth at n = {n_fix} qubits")
    vl, vg = [], []
    for L in Ls:
        vl.append(gradient_samples(n_fix, L, local_cost(n_fix)).var())
        vg.append(gradient_samples(n_fix, L, global_cost(n_fix)).var())
        print(f"  L = {L:>2}: local {vl[-1]:.2e}  global {vg[-1]:.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].semilogy(ns, var_local, "o-", label="local cost $O_L$")
    axes[0].semilogy(ns, var_global, "s-", label="global cost $O_G$")
    k = list(ns).index(6)
    axes[0].semilogy(ns, var_local[k] * 2.0 ** (6 - ns), "k--", lw=0.8, label=r"$\propto 2^{-n}$")
    axes[0].semilogy(ns, var_global[k] * 4.0 ** (6 - ns), "k:", lw=0.8, label=r"$\propto 4^{-n}$")
    axes[0].set_xlabel("qubits $n$ (depth $L=n$)"); axes[0].set_ylabel(r"Var$[\partial C/\partial\theta_1]$")
    axes[0].legend(fontsize=8)
    axes[1].semilogy(Ls, vl, "o-", label="local"); axes[1].semilogy(Ls, vg, "s-", label="global")
    axes[1].set_xlabel(f"depth $L$ ($n={n_fix}$)"); axes[1].set_ylabel(r"Var$[\partial C/\partial\theta_1]$")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    bookplot.savefig(fig, 4, "barren")
    print(f"figure barren written ({time.time() - t0:.0f} s)")
