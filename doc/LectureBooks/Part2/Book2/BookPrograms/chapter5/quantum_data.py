"""Chapter 5, Sec. 5.5: learning a quantum state with a QBM.

The target is a density matrix rho_t on three qubits and the model the
Gibbs state of a fully visible QBM; the objective is the quantum relative
entropy S(rho_t || rho_theta), Eq. (5-relent), whose gradient
beta(<Lambda_mu>_theta - <Lambda_mu>_target), Eq. (5-relentgrad), needs
only expectation values of the model and of the target.

Two targets:
  1. the Gibbs state of a transverse-field Ising chain with random couplings
     (inside the model family: the couplings are recovered, Table 5-hamlearn);
  2. the Gibbs state of a Heisenberg XXZ chain in a tilted field with a
     next-nearest-neighbour coupling (outside every family used) learned by
     three models of growing operator content: Z + ZZ (classical), Z + X + ZZ
     (the transverse-field QBM), and Z + X + ZZ + XX + YY, nearest neighbours only.
Produces Fig. 5-quantumdata.  Runtime about 10 s.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]      # BookPrograms/
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np

from gibbs import pauli_string, gibbs_state, fidelity, relative_entropy
from qbm import QBM, ising_terms


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    import bookplot

    rng = np.random.default_rng(4)
    n, beta = 3, 1.0

    # ---- 1. Hamiltonian learning: target inside the family
    terms, names = ising_terms(n)                        # Z_q, X_q, Z_q Z_{q+1}
    model = QBM(n, 0, terms, beta=beta)
    theta_star = rng.normal(0, 0.8, model.n_params)
    rho_t = model.gibbs(theta_star)
    t0 = time.time()
    theta, hist = model.fit(np.zeros(model.n_params), rho_t, "relent", steps=400, lr=0.05)
    print("Table 5-hamlearn: couplings of the target Hamiltonian and of the trained QBM (3 qubits, beta = 1)")
    print(f"{'term':>6}  {'target':>8}  {'learned':>8}")
    for nm, a, b in zip(names, theta_star, theta):
        print(f"{nm:>6}  {a:>8.4f}  {b:>8.4f}")
    print(f"relative entropy {hist[0]:.4f} -> {hist[-1]:.2e}, fidelity {fidelity(rho_t, model.gibbs(theta)):.8f}, "
          f"max |theta - theta*| = {np.max(np.abs(theta - theta_star)):.1e} ({time.time() - t0:.1f} s)")

    # ---- 2. a target outside the family: the XXZ chain
    H_xxz = sum(pauli_string({q: "X", q + 1: "X"}, n) + pauli_string({q: "Y", q + 1: "Y"}, n)
                + 0.5 * pauli_string({q: "Z", q + 1: "Z"}, n) for q in range(n - 1)) \
        + sum(0.3 * pauli_string({q: "Z"}, n) + 0.4 * pauli_string({q: "X"}, n) for q in range(n)) \
        + 0.5 * pauli_string({0: "Z", 2: "Z"}, n)
    rho_xxz = gibbs_state(H_xxz, beta)
    families = [("Z + ZZ (classical)", ising_terms(n, fields=("Z",), couplings=("ZZ",))),
                ("Z + X + ZZ", ising_terms(n, fields=("Z", "X"), couplings=("ZZ",))),
                ("Z + X + ZZ + XX + YY", ising_terms(n, fields=("Z", "X"), couplings=("ZZ", "XX", "YY")))]
    print(f"\nXXZ-chain target (tilted field, next-nearest-neighbour Z Z), beta = {beta}: three model families, 400 Adam steps")
    print(f"{'model':>22}  {'P':>3}  {'S(rho_t||rho)':>13}  {'fidelity':>9}  {'diagonal KL':>11}")
    curves = {}
    p_t = np.diag(rho_xxz).real
    for name, (tm, nm) in families:
        m = QBM(n, 0, tm, beta=beta)
        th, h = m.fit(np.zeros(m.n_params), rho_xxz, "relent", steps=400, lr=0.05)
        rho = m.gibbs(th)
        p = np.diag(rho).real
        kl = float(np.sum(p_t * np.log(p_t / p)))
        curves[name] = h
        print(f"{name:>22}  {m.n_params:>3}  {h[-1]:>13.5f}  {fidelity(rho_xxz, rho):>9.6f}  {kl:>11.5f}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].semilogy(np.arange(0, 400, 10).tolist() + [400], np.maximum(hist, 1e-16), label="transverse-field Ising target")
    for name, h in curves.items():
        axes[0].semilogy(np.arange(0, 400, 10).tolist() + [400], h, "--", label=f"XXZ target, model {name}")
    axes[0].set_xlabel("Adam step"); axes[0].set_ylabel("$S(\\rho_t\\,\\|\\,\\rho_\\theta)$"); axes[0].legend(fontsize=6)
    axes[1].plot(theta_star, theta, "o")
    lim = [min(theta_star.min(), theta.min()) - 0.2, max(theta_star.max(), theta.max()) + 0.2]
    axes[1].plot(lim, lim, "k--", lw=0.8)
    for nm, a, b in zip(names, theta_star, theta):
        axes[1].annotate(nm, (a, b), fontsize=6, xytext=(3, 3), textcoords="offset points")
    axes[1].set_xlabel("target coupling $\\theta^\\star_\\mu$"); axes[1].set_ylabel("learned coupling $\\theta_\\mu$")
    axes[1].set_title("Hamiltonian learning", fontsize=10)
    fig.tight_layout()
    bookplot.savefig(fig, 5, "quantumdata")
    print("figure quantumdata written")
