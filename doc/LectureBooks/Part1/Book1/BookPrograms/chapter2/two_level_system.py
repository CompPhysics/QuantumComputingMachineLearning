"""Chapter 2, Sec. 2.11: a first Hamiltonian, the two-level system.

H = H_0 + lambda H_I with H_0 = diag(E_1, E_2) and a symmetric interaction
V, rewritten in the Pauli basis as H = (E_+ + lambda V_+) I + (E_- + lambda V_-) Z
+ lambda V_o X (Eq. 2.tlspauli).  The program diagonalises H for
0 <= lambda <= 1 (Fig. 2.tls, left panel: the avoided crossing), and
evaluates the energy of the one-parameter-family ansatz
|psi(theta, phi)> = R_y(phi) R_x(theta) |0> on a grid of angles
(Fig. 2.tls, right panel), the first step towards the variational quantum
eigensolver of Chapter 7.  Parameters as in the text: E_1 = 0, E_2 = 4,
V_11 = -V_22 = 3, V_12 = V_21 = 0.2.  Runs in a second.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np
import gates as G
from bookplot import savefig, plt


class TwoLevelSystem:
    """H(lambda) = H_0 + lambda H_I for a two-level system in the Pauli basis."""

    def __init__(self, E1=0.0, E2=4.0, V11=3.0, V22=-3.0, V12=0.2):
        self.E_plus, self.E_minus = (E1 + E2) / 2, (E1 - E2) / 2
        self.V_plus, self.V_minus, self.V_o = (V11 + V22) / 2, (V11 - V22) / 2, V12
        self.H0 = np.diag([E1, E2]).astype(complex)
        self.HI = np.array([[V11, V12], [V12, V22]], dtype=complex)

    def hamiltonian(self, lam):
        """Matrix form, Eq. (2.tlsmatrix)."""
        return self.H0 + lam * self.HI

    def pauli_form(self, lam):
        """The same operator from the Pauli decomposition, Eq. (2.tlspauli)."""
        return ((self.E_plus + lam * self.V_plus) * G.I
                + (self.E_minus + lam * self.V_minus) * G.Z
                + lam * self.V_o * G.X)

    def eigen(self, lam):
        return np.linalg.eigh(self.hamiltonian(lam))

    def ansatz(self, theta, phi):
        """|psi> = R_y(phi) R_x(theta) |0>."""
        return G.Ry(phi) @ G.Rx(theta) @ np.array([1, 0], dtype=complex)

    def energy(self, theta, phi, lam):
        psi = self.ansatz(theta, phi)
        return float(np.real(np.vdot(psi, self.hamiltonian(lam) @ psi)))


if __name__ == "__main__":
    tls = TwoLevelSystem()
    lam = 1.0
    assert np.allclose(tls.hamiltonian(lam), tls.pauli_form(lam))
    E, V = tls.eigen(lam)
    print(f"lambda = {lam}: eigenvalues {E}")
    print(f"ground state {V[:, 0]}, |0>-component squared {abs(V[0, 0])**2:.4f}")
    # the character of the eigenvectors interchanges near lambda = 2/3
    print(f"\n{'lambda':>7s} {'E_0':>9s} {'E_1':>9s} {'|<0|E_0>|^2':>12s}")
    lams = np.linspace(0, 1, 101)
    spectrum = np.array([tls.eigen(l)[0] for l in lams])
    weight = np.array([abs(tls.eigen(l)[1][0, 0]) ** 2 for l in lams])
    for l in (0.0, 0.3, 0.6, 2 / 3, 0.7, 1.0):
        e, v = tls.eigen(l)
        print(f"{l:7.3f} {e[0]:9.4f} {e[1]:9.4f} {abs(v[0,0])**2:12.4f}")
    # a grid of ansatz angles; energy difference to the exact ground state
    angles = np.deg2rad(np.arange(0, 180, 10))
    grid = np.array([[tls.energy(th, ph, lam) for ph in angles] for th in angles])
    diff = np.abs(grid - E[0])
    i, j = np.unravel_index(np.argmin(diff), diff.shape)
    print(f"\nsingle ansatz theta=pi/2, phi=0.2pi: E = {tls.energy(np.pi/2, 0.2*np.pi, lam):.4f}")
    print(f"best grid point: theta = {np.rad2deg(angles[i]):.0f} deg, phi = {np.rad2deg(angles[j]):.0f} deg, "
          f"E = {grid[i, j]:.4f}, E - E_0 = {diff[i, j]:.4f}")
    # a finer grid (one-degree steps) gets much closer; the exact ground state
    # lies in the x-z plane of the Bloch sphere, i.e. at theta = pi
    fine = np.deg2rad(np.arange(0, 181, 1))
    fine_grid = np.array([[tls.energy(th, ph, lam) for ph in fine] for th in fine])
    fi, fj = np.unravel_index(np.argmin(fine_grid), fine_grid.shape)
    print(f"one-degree grid: theta = {np.rad2deg(fine[fi]):.0f} deg, phi = {np.rad2deg(fine[fj]):.0f} deg, "
          f"E - E_0 = {fine_grid[fi, fj] - E[0]:.2e}")
    assert fine_grid.min() - E[0] > -1e-12                # variational principle

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), gridspec_kw={"wspace": 0.55})
    axes[0].plot(lams, spectrum[:, 0], label="$E_0(\\lambda)$")
    axes[0].plot(lams, spectrum[:, 1], label="$E_1(\\lambda)$")
    axes[0].plot(lams, [tls.hamiltonian(l)[0, 0].real for l in lams], "k:", lw=0.8, label="diagonal elements")
    axes[0].plot(lams, [tls.hamiltonian(l)[1, 1].real for l in lams], "k:", lw=0.8)
    ax2 = axes[0].twinx(); ax2.plot(lams, weight, "C3--", lw=1); ax2.set_ylabel(r"$|\langle 0|E_0\rangle|^2$", color="C3")
    axes[0].set_xlabel(r"$\lambda$"); axes[0].set_ylabel("energy"); axes[0].legend(fontsize=8, loc="center left")
    TH, PH = np.meshgrid(np.rad2deg(angles), np.rad2deg(angles), indexing="ij")
    cs = axes[1].contourf(TH, PH, grid, levels=20, cmap="viridis")
    fig.colorbar(cs, ax=axes[1], label=r"$\langle\psi(\theta,\phi)|H|\psi(\theta,\phi)\rangle$")
    axes[1].plot(np.rad2deg(angles[i]), np.rad2deg(angles[j]), "r*", ms=10)
    axes[1].set_xlabel(r"$\theta$ [deg]"); axes[1].set_ylabel(r"$\phi$ [deg]")
    print("figure written to", savefig(fig, 2, "two_level_system"))
