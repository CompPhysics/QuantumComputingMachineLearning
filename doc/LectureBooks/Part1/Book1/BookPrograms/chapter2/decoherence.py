"""Chapter 2, Sec. 2.5.5: decoherence seen through the density matrix.

Two elementary noise channels on one qubit, written as Kraus maps
rho -> sum_k E_k rho E_k^dagger (Eq. 2.kraus): dephasing, which shrinks the
off-diagonal elements of rho by a factor (1 - 2p) and leaves the populations
untouched (Eq. 2.dephasing), and amplitude damping, which drives the qubit
towards |0> (Eq. 2.damping).  Fig. 2.decoherence shows the Bloch vector of
|+> under repeated application of each channel and the accompanying loss of
purity.  Runs in a second.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np
import gates as G
from density_matrix import DensityMatrix
from bookplot import savefig, plt


def kraus_apply(rho, kraus):
    """rho -> sum_k E_k rho E_k^dagger."""
    return sum(E @ rho @ E.conj().T for E in kraus)


def dephasing(p):
    """With probability p a Z error hits the qubit."""
    return [np.sqrt(1 - p) * G.I, np.sqrt(p) * G.Z]


def amplitude_damping(gamma):
    """|1> decays to |0> with probability gamma (spontaneous emission)."""
    E0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    E1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return [E0, E1]


def trajectory(rho0, kraus, steps):
    rho = rho0.copy()
    out = [DensityMatrix(rho)]
    for _ in range(steps):
        rho = kraus_apply(rho, kraus)
        out.append(DensityMatrix(rho))
    return out


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    plus = np.array([1, 1]) / np.sqrt(2)
    rho0 = np.outer(plus, plus.conj())
    # Kraus operators satisfy sum_k E_k^dagger E_k = I
    for kraus in (dephasing(0.1), amplitude_damping(0.1)):
        assert np.allclose(sum(E.conj().T @ E for E in kraus), G.I)
    # one dephasing step: off-diagonals scaled by 1 - 2p, Eq. (2.dephasing)
    p = 0.1
    rho1 = kraus_apply(rho0, dephasing(p))
    assert np.allclose(rho1, [[0.5, 0.5 * (1 - 2 * p)], [0.5 * (1 - 2 * p), 0.5]])
    print("rho(|+>) after one dephasing step with p = 0.1:\n", rho1.real)
    steps = 30
    deph = trajectory(rho0, dephasing(0.05), steps)
    damp = trajectory(rho0, amplitude_damping(0.08), steps)
    print(f"{'step':>4s} {'purity (dephasing)':>20s} {'purity (damping)':>18s} {'<Z> (damping)':>14s}")
    for k in (0, 5, 10, 20, 30):
        print(f"{k:4d} {deph[k].purity():20.4f} {damp[k].purity():18.4f} {damp[k].bloch_vector()[2]:14.4f}")
    # the fully dephased state is the maximally mixed state, entropy 1 bit
    final = trajectory(rho0, dephasing(0.5), 1)[-1]
    assert np.allclose(final.rho, np.eye(2) / 2) and abs(final.von_neumann_entropy() - 1) < 1e-12

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    t = np.arange(steps + 1)
    for ax, traj, title in ((axes[0], deph, "dephasing, $p=0.05$ per step"),
                            (axes[1], damp, r"amplitude damping, $\gamma=0.08$ per step")):
        r = np.array([d.bloch_vector() for d in traj])
        ax.plot(t, r[:, 0], label=r"$\langle X\rangle$")
        ax.plot(t, r[:, 2], label=r"$\langle Z\rangle$")
        ax.plot(t, [d.purity() for d in traj], "k--", label=r"purity $\mathrm{Tr}\rho^2$")
        ax.plot(t, [d.von_neumann_entropy() for d in traj], "k:", label=r"entropy $S(\rho)$")
        ax.set_xlabel("channel applications"); ax.set_title(title, fontsize=9)
    axes[0].set_ylim(-0.05, 1.05); axes[1].set_ylim(-0.05, 1.05)
    axes[0].legend(fontsize=8, loc="center right")
    print("figure written to", savefig(fig, 2, "decoherence"))
