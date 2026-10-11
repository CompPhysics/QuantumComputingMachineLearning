"""Chapter 2, Sec. 2.3: the Bloch sphere.

Converts between the amplitudes of a qubit and its Bloch angles
(Eq. 2.blochstate), computes the Bloch vector r = (<X>, <Y>, <Z>)
(Eq. 2.blochvector), lists the Bloch angles of the six Pauli eigenstates
(Table 2.paulieigen) and draws Fig. 2.bloch: the sphere with the Pauli
eigenstates, the trajectory of a spin precessing in a magnetic field along
z (Eq. 2.larmor) and the two-rotation path R_y(phi) R_x(theta)|0> used as a
variational ansatz in Sec. 2.11.  Runs in a few seconds.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np
import gates as G
from bookplot import savefig, plt


def bloch_state(theta, phi):
    """|psi> = cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>."""
    return np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])


def bloch_angles(psi):
    """Inverse of bloch_state, removing the global phase first."""
    psi = np.asarray(psi, dtype=complex)
    psi = psi / np.linalg.norm(psi)
    psi = psi * np.exp(-1j * np.angle(psi[0]))       # make the |0> amplitude real
    theta = 2 * np.arccos(np.clip(psi[0].real, -1, 1))
    phi = np.angle(psi[1]) if abs(psi[1]) > 1e-12 else 0.0
    return theta, phi % (2 * np.pi)


def bloch_vector(psi):
    """r = (<X>, <Y>, <Z>) for a pure single-qubit state."""
    psi = np.asarray(psi, dtype=complex)
    return np.array([np.vdot(psi, P @ psi).real for P in (G.X, G.Y, G.Z)])


PAULI_EIGENSTATES = {
    "|0>": ([1, 0], "Z", +1), "|1>": ([0, 1], "Z", -1),
    "|+>": ([1, 1] / np.sqrt(2), "X", +1), "|->": ([1, -1] / np.sqrt(2), "X", -1),
    "|+i>": ([1, 1j] / np.sqrt(2), "Y", +1), "|-i>": ([1, -1j] / np.sqrt(2), "Y", -1),
}


def larmor_trajectory(theta0, phi0, omega, times):
    """Points on the sphere for |psi(t)> = exp(-i H t) |psi(0)>, H = -omega Z/2."""
    psi0 = bloch_state(theta0, phi0)
    pts = []
    for t in times:
        U = G.Rz(-omega * t)                     # exp(-i H t) = exp(+i omega t Z / 2)
        pts.append(bloch_vector(U @ psi0))
    return np.array(pts)


def draw_bloch(ax, elev=20, azim=35):
    u, v = np.mgrid[0:2 * np.pi:60j, 0:np.pi:30j]
    ax.plot_surface(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                    color="lightsteelblue", alpha=0.15, linewidth=0)
    t = np.linspace(0, 2 * np.pi, 200)
    for circ in ((np.cos(t), np.sin(t), 0 * t), (np.cos(t), 0 * t, np.sin(t)), (0 * t, np.cos(t), np.sin(t))):
        ax.plot(*circ, color="gray", lw=0.5)
    for axis, label in (((1.3, 0, 0), "x"), ((0, 1.3, 0), "y"), ((0, 0, 1.3), "z")):
        ax.quiver(0, 0, 0, *axis, color="k", arrow_length_ratio=0.08, lw=0.8)
        ax.text(*(1.4 * np.array(axis)), label)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_zlim(-1.2, 1.2)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    # Table 2.paulieigen
    print(f"{'state':6s} {'operator':9s} {'eigenvalue':>10s} {'theta':>8s} {'phi':>8s}   Bloch vector")
    for name, (psi, op, ev) in PAULI_EIGENSTATES.items():
        th, ph = bloch_angles(psi)
        print(f"{name:6s} {op:9s} {ev:>10d} {th/np.pi:7.2f}π {ph/np.pi:7.2f}π   {bloch_vector(psi)}")
    # round trip amplitudes -> angles -> amplitudes
    theta, phi = 2.0, 0.7
    psi = bloch_state(theta, phi)
    assert np.allclose(bloch_angles(psi), (theta, phi))
    assert np.allclose(bloch_vector(psi), [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    # a global phase leaves the Bloch vector unchanged
    assert np.allclose(bloch_vector(np.exp(0.3j) * psi), bloch_vector(psi))
    print("\nBloch angles of psi =", psi, "->", f"theta={theta}, phi={phi}")

    # Figure 2.bloch
    fig = plt.figure(figsize=(9, 4.4))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    draw_bloch(ax)
    for name, (v, op, ev) in PAULI_EIGENSTATES.items():
        r = bloch_vector(v)
        ax.scatter(*r, color="C3", s=25)
        ax.text(*(1.12 * r), name, fontsize=8)
    r = bloch_vector(psi)
    ax.quiver(0, 0, 0, *r, color="C0", arrow_length_ratio=0.1, lw=2)
    ax.text(*(1.1 * r), r"$|\psi\rangle$", color="C0")
    ax.set_title("(a) Pauli eigenstates and a general state", fontsize=9)

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    draw_bloch(ax)
    traj = larmor_trajectory(theta0=1.0, phi0=0.0, omega=1.0, times=np.linspace(0, 2 * np.pi, 200))
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="C1", lw=2, label="Larmor precession")
    # the path R_y(phi) R_x(theta)|0>: first rotate about x, then about y
    k0 = np.array([1, 0], dtype=complex)
    th_path = np.linspace(0, 2.0, 60)
    p1 = np.array([bloch_vector(G.Rx(t) @ k0) for t in th_path])
    ph_path = np.linspace(0, 1.2, 60)
    p2 = np.array([bloch_vector(G.Ry(p) @ G.Rx(2.0) @ k0) for p in ph_path])
    ax.plot(p1[:, 0], p1[:, 1], p1[:, 2], color="C2", lw=2, label=r"$R_x(\theta)|0\rangle$")
    ax.plot(p2[:, 0], p2[:, 1], p2[:, 2], color="C4", lw=2, label=r"$R_y(\phi)R_x(2)|0\rangle$")
    ax.legend(loc="lower left", fontsize=7)
    ax.set_title("(b) rotations and precession", fontsize=9)
    out = savefig(fig, 2, "bloch_sphere")
    print("figure written to", out)
