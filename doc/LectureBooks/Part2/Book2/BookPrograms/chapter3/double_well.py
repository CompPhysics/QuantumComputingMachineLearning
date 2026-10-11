"""Chapter 3, Sec. 3.6: QAOA on two electrons in a double quantum well.

Two spin-1/2 electrons, one in each well, are two qubits coupled by the
exchange interaction H_ex = J S_L.S_R = (J/4)(XX + YY + ZZ), Eq. (3-exchange).
The Ising part (J/4) Z_L Z_R realises the cost unitary of MaxCut on the
single edge, H_C = -(I - Z_L Z_R)/2: evolving for a time t = 2 gamma / J
is exp(-i gamma H_C) up to a phase, Eq. (3-gatetime).  The script

  * prints the singlet/triplet spectrum of H_ex and the cost table of H_C;
  * checks the closed form E(beta, gamma) = -1/2 + sin(4 beta) sin(gamma)/2,
    Eq. (3-oneedge), against the simulator and draws Fig. 3-dwlandscape;
  * shows the output distribution at the optimum (the triplet |T_0>) and
    off the optimum, and 100 simulated single-shot spin readouts
    (Fig. 3-dwprobs);
  * optimizes p = 1, 2, 3 and reports that p = 1 is already exact.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]      # BookPrograms/
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np

import gates as G
from qubo import QUBO
from qaoa import QAOA, analytic_single_edge

LABELS = [r"$|00\rangle=|\!\uparrow\uparrow\rangle$", r"$|01\rangle=|\!\uparrow\downarrow\rangle$",
          r"$|10\rangle=|\!\downarrow\uparrow\rangle$", r"$|11\rangle=|\!\downarrow\downarrow\rangle$"]


def exchange_hamiltonian(J=1.0):
    """H_ex = (J/4)(X X + Y Y + Z Z), Eq. (3-exchange)."""
    return J / 4 * (np.kron(G.X, G.X) + np.kron(G.Y, G.Y) + np.kron(G.Z, G.Z))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from bookplot import savefig

    J = 1.0
    w, v = np.linalg.eigh(exchange_hamiltonian(J))
    print(f"Exchange Hamiltonian, J = {J}: eigenvalues {np.round(w, 4)} (singlet -3J/4, triplet J/4)")
    singlet = v[:, 0]
    print("  singlet amplitudes:", np.round(singlet.real, 4), " (|01> - |10>)/sqrt(2) up to sign")

    edge = QUBO.maxcut(2, [(0, 1)]).to_ising()
    qaoa = QAOA(edge, p=1)
    print("Cost Hamiltonian H_C = -(I - Z_L Z_R)/2: diagonal", qaoa.diag, "(cut states |01>, |10> have energy -1)")
    print(f"  gate time for the cost unitary: t = 2 gamma / J; at gamma* = 3pi/2, t = {3 * np.pi / J:.4f}/J")

    # closed form against the simulator, and the landscape
    bs, gs = np.linspace(0, np.pi / 2, 60), np.linspace(0, 2 * np.pi, 120)
    E_num = np.array([[qaoa.energy([g, b]) for g in gs] for b in bs])
    E_ana = analytic_single_edge(bs[:, None], gs[None, :])
    print(f"  closed form -1/2 + sin(4 beta) sin(gamma)/2: max error {np.abs(E_num - E_ana).max():.1e}")
    assert np.abs(E_num - E_ana).max() < 1e-12
    b_opt, g_opt = np.pi / 8, 3 * np.pi / 2
    x_opt = [g_opt, b_opt]
    print(f"  minimum E = {qaoa.energy(x_opt):.6f} at beta = pi/8, gamma = 3pi/2")
    psi = qaoa.state(x_opt)
    T0 = np.array([0, 1, 1, 0]) / np.sqrt(2)
    print(f"  |<T0|psi*>|^2 = {abs(np.vdot(T0, psi.psi)) ** 2:.6f}: the optimal state is the triplet (|01>+|10>)/sqrt(2)")
    print(f"  |<S|psi*>|^2  = {abs(np.vdot(singlet, psi.psi)) ** 2:.6f}: not the singlet ground state of H_ex")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.3))
    ax = axes[0]
    cp = ax.contourf(gs / np.pi, bs / np.pi, E_num, levels=np.linspace(-1, 0, 21), cmap="RdYlBu_r")
    plt.colorbar(cp, ax=ax, label=r"$E(\gamma,\beta)$")
    ax.plot(g_opt / np.pi, b_opt / np.pi, "k*", ms=12)
    ax.set_xlabel(r"$\gamma/\pi$"); ax.set_ylabel(r"$\beta/\pi$"); ax.set_title("single edge, $p=1$")
    ax = axes[1]
    ax.plot(bs / np.pi, E_num[:, np.argmin(abs(gs - g_opt))], "k-", lw=2, label=r"simulator, $\gamma=3\pi/2$")
    ax.plot(bs / np.pi, analytic_single_edge(bs, g_opt), "r--", label="closed form")
    ax.axvline(b_opt / np.pi, color="gray", ls=":")
    ax.set_xlabel(r"$\beta/\pi$"); ax.set_ylabel(r"$E$"); ax.legend(fontsize=8)
    ax = axes[2]
    ax.plot(gs / np.pi, E_num[np.argmin(abs(bs - b_opt)), :], "k-", lw=2, label=r"simulator, $\beta=\pi/8$")
    ax.plot(gs / np.pi, analytic_single_edge(b_opt, gs), "r--", label="closed form")
    ax.axvline(g_opt / np.pi, color="gray", ls=":")
    ax.set_xlabel(r"$\gamma/\pi$"); ax.set_ylabel(r"$E$"); ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, 3, "dwlandscape")

    # output distributions and simulated spin readouts
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.3), sharey=True)
    cols = ["tab:blue", "tab:red", "tab:red", "tab:blue"]
    for ax, x, title in ((axes[0], x_opt, r"optimum $(\gamma,\beta)=(3\pi/2,\pi/8)$"),
                         (axes[1], [3 * np.pi / 2, np.pi / 16], r"off the optimum $(3\pi/2,\pi/16)$")):
        pr = qaoa.probabilities(x)
        ax.bar(range(4), pr, color=cols)
        for k, pk in enumerate(pr):
            ax.text(k, pk + 0.01, f"{pk:.3f}", ha="center", fontsize=8)
        ax.set_title(f"{title}: $E={qaoa.energy(x):.3f}$", fontsize=9)
        print(f"  {title}: P = {np.round(pr, 4)}, E = {qaoa.energy(x):.4f}")
    shots = 100
    counts = np.bincount(qaoa.sample(x_opt, shots, rng=7), minlength=4)
    axes[2].bar(range(4), counts / shots, color=cols)
    for k in range(4):
        axes[2].plot([k - 0.4, k + 0.4], [qaoa.probabilities(x_opt)[k]] * 2, "k-")
    axes[2].set_title(f"{shots} single-shot spin readouts at the optimum", fontsize=9)
    print(f"  {shots} shots at the optimum: counts {counts}, P(cut state) = {(counts[1] + counts[2]) / shots:.2f}")
    for ax in axes:
        ax.set_xticks(range(4)); ax.set_xticklabels(LABELS, fontsize=7); ax.set_ylim(0, 0.65)
    axes[0].set_ylabel("probability")
    fig.tight_layout()
    savefig(fig, 3, "dwprobs")

    # depth p = 1, 2, 3
    for p in (1, 2, 3):
        q = QAOA(edge, p)
        x, E, _ = q.optimize(restarts=6, rng=1)
        print(f"  p = {p}: E* = {E:.6f}, ratio {q.approximation_ratio(x):.4f}, P(cut) = {q.success_probability(x):.4f}")
    print("figures dwlandscape and dwprobs written")
