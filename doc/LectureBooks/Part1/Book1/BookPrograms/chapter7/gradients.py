"""Chapter 7, Sec. 7.5: gradients on a quantum computer.

* the trigonometric form of Eq. (7.trigpoly): three energies of a two-qubit,
  two-layer ansatz determine the whole dependence on one angle;
* the two-term parameter-shift rule against a converged finite difference;
* its failure for the three-eigenvalue generator (Z_0 + Z_1)/2 and for the
  pair-excitation generators of chapter 4, and the four-term rule that
  repairs it, Eq. (7.generalshift);
* the barren plateau, Fig. 7.barren: the variance of one gradient component
  of a hardware-efficient ansatz with L = n layers over random parameters,
  against the number of qubits, for the Heisenberg-like observable Z_0 Z_1;
* SPSA against parameter-shift gradient descent on the four-qubit pairing
  Hamiltonian, plotted against the number of energy evaluations rather than
  the number of iterations, Fig. 7.spsa.

About twenty seconds.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np
from pauli import PauliSum
from models import PairingModel
from pauli_circuits import compact_pairing
from vqe import VQE, HardwareEfficient, PauliRotationAnsatz, shifted
from bookplot import savefig, plt


def general_shift(energy, theta, k, R):
    """Eq. (7.generalshift): the derivative for a generator whose energy has the
    equidistant frequencies 1, ..., R, from 2R evaluations."""
    g = 0.0
    for mu in range(1, 2 * R + 1):
        x = (2 * mu - 1) * np.pi / (2 * R)
        g += (-1) ** (mu - 1) / (4 * R * np.sin(x / 2) ** 2) * energy(shifted(theta, k, x))
    return g


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    rng = np.random.default_rng(11)
    H2 = PauliSum({"II": 0.5, "ZI": 0.25, "IZ": -0.75, "XX": -0.25, "YY": -0.25})       # two-level pairing, Eq. (4.jwpairingtwolevels)

    # ------------------------------------------------ the trigonometric form
    print("=== E(theta_k) = A + B cos(theta_k) + C sin(theta_k) for an R_y rotation (Eq. 7.trigpoly)")
    hea = HardwareEfficient(2, 2)
    theta = hea.random_parameters(rng)
    vqe = VQE(H2, hea)
    k = 2                                                   # an R_y angle in the first layer
    pts = [0.0, 1.0, 2.5]
    M = np.array([[1, np.cos(x), np.sin(x)] for x in pts])
    E = np.array([vqe.energy(shifted(theta, k, x - theta[k])) for x in pts])
    A, B, C = np.linalg.solve(M, E)
    worst = max(abs(A + B * np.cos(x) + C * np.sin(x) - vqe.energy(shifted(theta, k, x - theta[k]))) for x in np.linspace(-3, 3, 25))
    print(f"  A = {A:.6f}, B = {B:.6f}, C = {C:.6f}; largest deviation of the fitted form on 25 other points {worst:.1e}")
    assert worst < 1e-12

    # ------------------------------------------------ two-term rule against finite differences
    h = 1e-5
    g_shift = vqe.gradient(theta)
    g_fd = np.array([(vqe.energy(shifted(theta, j, h)) - vqe.energy(shifted(theta, j, -h))) / (2 * h) for j in range(len(theta))])
    print(f"=== two-term rule against finite differences (h = 1e-5): largest difference {np.abs(g_shift - g_fd).max():.1e}")
    assert np.abs(g_shift - g_fd).max() < 1e-8

    # ------------------------------------------------ when two terms are not enough
    print("\n=== A generator with three eigenvalues: G = (Z_0 + Z_1)/2")
    Gbad = 0.5 * (PauliSum({"ZI": 1.0}) + PauliSum({"IZ": 1.0}))
    ref = np.array([1, 1, 1, 1]) / 2.0                       # |++>, so that the rotation does something
    ans = PauliRotationAnsatz(2, ref, [Gbad])
    Hx = PauliSum({"XI": 0.7, "IX": 0.4, "XX": 0.5, "YZ": 0.3})
    v = VQE(Hx, ans)
    th = np.array([0.6])
    exact = (v.energy(th + h) - v.energy(th - h)) / (2 * h)
    two = 0.5 * (v.energy(th + np.pi / 2) - v.energy(th - np.pi / 2))
    four = general_shift(v.energy, th, 0, 2)
    print(f"  eigenvalues of G: {np.round(np.linalg.eigvalsh(Gbad.matrix()), 6)}")
    print(f"  true derivative {exact:.6f}; two-term rule {two:.6f}; four-term rule (R = 2) {four:.6f}")
    assert abs(four - exact) < 1e-8
    print("  the pair-excitation generators of chapter 4 have the same spectrum, and VQE uses the four-term rule for them")

    # ------------------------------------------------ barren plateau
    print("\n=== Barren plateau: Var[dE/dtheta_1] for the hardware-efficient ansatz with L = n layers (Fig. 7.barren)")
    ns = [2, 3, 4, 5, 6, 7, 8]
    var = []
    samples = 300
    for n in ns:
        obs = PauliSum.single(n, 0, "Z") * PauliSum.single(n, 1, "Z")
        hea = HardwareEfficient(n, n, entangler="ring")
        v = VQE(obs, hea)
        grads = []
        for _ in range(samples):
            th = hea.random_parameters(rng)
            kind, r = hea.shift(0)
            s = np.pi / (4 * r)
            grads.append(r * (v.energy(shifted(th, 0, s)) - v.energy(shifted(th, 0, -s))))
        var.append(np.var(grads))
        print(f"  n = {n}: variance {var[-1]:.2e}, mean {np.mean(grads):+.3f}")
    slope = np.polyfit(ns[2:], np.log2(var[2:]), 1)[0]
    print(f"  slope of log2(variance) against n for n >= 4: {slope:.2f} (an exponential decay 2^(-c n))")
    assert var[-1] < var[1] / 4
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.semilogy(ns, var, "o-", label=r"Var$[\partial E/\partial\theta_1]$, $300$ random points")
    ax.semilogy(ns, [var[0] * 2.0 ** (slope * (n - ns[0])) for n in ns], "k--", lw=0.8, label=f"$2^{{{slope:.2f}\\,n}}$")
    ax.set_xlabel("qubits $n$ (layers $L=n$)"); ax.legend(fontsize=8)
    savefig(fig, 7, "barren")

    # ------------------------------------------------ SPSA against parameter-shift descent
    print("\n=== SPSA against parameter-shift gradient descent on the four-qubit pairing Hamiltonian, HEA with 3 layers (Fig. 7.spsa)")
    H4 = compact_pairing(PairingModel(4, g=1.0))
    hea = HardwareEfficient(4, 3)
    theta0 = hea.random_parameters(np.random.default_rng(5), 0.3)
    E0 = np.linalg.eigvalsh(H4.matrix())[0]
    curves = {}
    v = VQE(H4, hea); v.gradient_descent(theta0, lr=0.15, iters=60)
    per_iter = 2 * hea.n_params + 1
    curves["parameter shift"] = (np.arange(len(v.history)) * per_iter, np.array(v.history))
    v = VQE(H4, hea); v.spsa(theta0, iters=600, a=0.3, c=0.15)
    curves["SPSA"] = (np.arange(len(v.history)) * 3, np.array(v.history))
    for name, (x, y) in curves.items():
        print(f"  {name:>16s}: {x[-1]:5d} energy evaluations, final energy {y[-1]:.6f} (lowest eigenvalue of the register {E0:.6f})")
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for name, (x, y) in curves.items():
        ax.plot(x, y, label=name)
    ax.axhline(E0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("energy evaluations"); ax.set_ylabel("energy"); ax.legend(fontsize=8)
    savefig(fig, 7, "spsa")
    print("\ngradient checks passed")
