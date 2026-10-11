"""Chapter 5, Secs. 5.3 and 5.6: Gibbs states and the tools around them.

    pauli_string(ops, n)            e.g. pauli_string({0: "Z", 2: "X"}, 3) = Z x I x X
    gibbs_state(H, beta)            rho = exp(-beta H) / Tr exp(-beta H), Eq. (5-gibbs)
    partial_trace(rho, n_v, n_h)    the reduced state of the first n_v qubits
    von_neumann_entropy(rho)        S = -Tr rho ln rho
    free_energy(rho, H, beta)       F = Tr[H rho] - S/beta, Eq. (5-freeenergy)
    relative_entropy(rho, sigma)    S(rho || sigma) = Tr rho (ln rho - ln sigma)
    fidelity(rho, sigma)            (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2
    exp_derivative(H, dH, beta)     d/dt exp(-beta (H + t dH)) at t = 0 (Duhamel),
                                    Eq. (5-duhamel), exactly in the eigenbasis of H
    thermofield_double(H, beta)     the purification of Eq. (5-tfd) on 2n qubits

The script checks the identities of the text (Klein's inequality, the
Duhamel derivative against finite differences, the thermofield double)
and draws the single-qubit free-energy landscape, Fig. 5-freeenergy.
Runtime 2 s.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]      # BookPrograms/
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np

PAULI = {"I": np.eye(2, dtype=complex),
         "X": np.array([[0, 1], [1, 0]], complex),
         "Y": np.array([[0, -1j], [1j, 0]], complex),
         "Z": np.array([[1, 0], [0, -1]], complex)}


def pauli_string(ops, n):
    """Tensor product with the given Paulis on the given qubits (qubit 0 leftmost)."""
    M = np.array([[1.0 + 0j]])
    for q in range(n):
        M = np.kron(M, PAULI[ops.get(q, "I")])
    return M


def gibbs_state(H, beta=1.0):
    e, U = np.linalg.eigh(H)
    w = np.exp(-beta * (e - e.min()))
    w /= w.sum()
    return (U * w) @ U.conj().T


def log_partition(H, beta=1.0):
    """ln Tr exp(-beta H), computed stably."""
    e = np.linalg.eigvalsh(H)
    return float(-beta * e.min() + np.log(np.sum(np.exp(-beta * (e - e.min())))))


def partial_trace(rho, n_v, n_h):
    """Trace out the last n_h qubits of an (n_v + n_h)-qubit state."""
    dv, dh = 2 ** n_v, 2 ** n_h
    return np.trace(rho.reshape(dv, dh, dv, dh), axis1=1, axis2=3)


def von_neumann_entropy(rho):
    p = np.linalg.eigvalsh(rho).real
    p = p[p > 1e-15]
    return float(-np.sum(p * np.log(p)))


def free_energy(rho, H, beta=1.0):
    return float(np.trace(H @ rho).real) - von_neumann_entropy(rho) / beta


def _logm(rho):
    p, U = np.linalg.eigh(rho)
    p = np.clip(p.real, 1e-300, None)
    return (U * np.log(p)) @ U.conj().T


def relative_entropy(rho, sigma):
    """S(rho || sigma) = Tr rho (ln rho - ln sigma), Eq. (5-relent); non-negative
    by Klein's inequality, so rounding below zero is clipped."""
    return max(float(np.trace(rho @ (_logm(rho) - _logm(sigma))).real), 0.0)


def _sqrtm(rho):
    p, U = np.linalg.eigh(rho)
    return (U * np.sqrt(np.clip(p.real, 0, None))) @ U.conj().T


def fidelity(rho, sigma):
    s = _sqrtm(rho)
    return float(np.trace(_sqrtm(s @ sigma @ s)).real ** 2)


def exp_derivative(H, dH, beta=1.0):
    """The directional derivative of exp(-beta H) along dH:
        d/dt exp(-beta (H + t dH))|_{t=0} = -int_0^beta e^{-(beta-s)H} dH e^{-sH} ds,
    evaluated exactly in the eigenbasis of H with the divided differences
    (e^{-beta e_k} - e^{-beta e_l}) / (e_l - e_k)."""
    e, U = np.linalg.eigh(H)
    D = U.conj().T @ dH @ U
    w = np.exp(-beta * e)
    de = e[:, None] - e[None, :]                     # e_k - e_l
    with np.errstate(divide="ignore", invalid="ignore"):
        G = np.where(np.abs(de) > 1e-12, (w[:, None] - w[None, :]) / de, -beta * w[:, None])
    return U @ (D * G) @ U.conj().T


def thermofield_double(H, beta=1.0):
    """|TFD> = Z^{-1/2} sum_k e^{-beta e_k/2} |e_k>_S |e_k>_A, Eq. (5-tfd);
    returned as a vector on system x ancilla (system qubits first)."""
    e, U = np.linalg.eigh(H)
    w = np.exp(-beta * (e - e.min()) / 2)
    w /= np.sqrt(np.sum(w ** 2))
    return np.einsum("k,ik,jk->ij", w, U, U.conj()).reshape(-1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import bookplot
    rng = np.random.default_rng(0)
    X, Z = PAULI["X"], PAULI["Z"]

    # ---- single qubit: Gibbs state, Klein's inequality, the free-energy landscape
    H1 = -1.0 * Z - 0.5 * X
    beta = 1.2
    rho_g = gibbs_state(H1, beta)
    F_g = free_energy(rho_g, H1, beta)
    r = np.linspace(0, 1, 101); th = np.linspace(0, np.pi, 121)
    F = np.zeros((len(r), len(th)))
    for i, ri in enumerate(r):
        for j, tj in enumerate(th):
            rho = 0.5 * (np.eye(2) + ri * (np.sin(tj) * X + np.cos(tj) * Z))
            F[i, j] = free_energy(rho, H1, beta)
    nx, nz = np.trace(X @ rho_g).real, np.trace(Z @ rho_g).real
    print(f"single qubit H = -Z - X/2, beta = {beta}: Gibbs state Bloch vector ({nx:.4f}, 0, {nz:.4f}), "
          f"|r| = {np.hypot(nx, nz):.4f}, S = {von_neumann_entropy(rho_g):.4f}, F = {F_g:.6f}; "
          f"minimum of F over the Bloch ball on the grid {F.min():.6f}")
    assert F.min() >= F_g - 1e-9
    # Klein: F(rho) - F(rho_g) = S(rho||rho_g)/beta
    rho = 0.5 * (np.eye(2) + 0.6 * (np.sin(1.0) * X + np.cos(1.0) * Z))
    assert abs(free_energy(rho, H1, beta) - F_g - relative_entropy(rho, rho_g) / beta) < 1e-10
    print("F(rho) - F(rho_Gibbs) = S(rho || rho_Gibbs)/beta verified")

    # ---- the Duhamel derivative against finite differences
    n = 3
    H = sum(rng.normal() * pauli_string({q: "Z", (q + 1) % n: "Z"}, n) for q in range(n)) \
        + sum(rng.normal() * pauli_string({q: "X"}, n) for q in range(n))
    dH = pauli_string({1: "X"}, n)
    h = 1e-5
    e1, U1 = np.linalg.eigh(H + h * dH); e2, U2 = np.linalg.eigh(H - h * dH)
    fd = ((U1 * np.exp(-e1)) @ U1.conj().T - (U2 * np.exp(-e2)) @ U2.conj().T) / (2 * h)
    err = np.max(np.abs(exp_derivative(H, dH, 1.0) - fd))
    print(f"Duhamel derivative of exp(-H) along X_1: max difference from finite differences {err:.1e}")
    assert err < 1e-6
    # ---- the thermofield double
    tfd = thermofield_double(H, 1.0)
    rho_S = partial_trace(np.outer(tfd, tfd.conj()), n, n)
    print(f"thermofield double: norm {np.linalg.norm(tfd):.10f}, max |Tr_A|TFD><TFD| - rho| = "
          f"{np.max(np.abs(rho_S - gibbs_state(H, 1.0))):.1e}, entanglement entropy S_A = "
          f"{von_neumann_entropy(rho_S):.4f} = thermal entropy")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    c = axes[0].contourf(th / np.pi, r, F, levels=30, cmap="RdYlBu_r")
    fig.colorbar(c, ax=axes[0])
    axes[0].plot(np.arctan2(nx, nz) / np.pi, np.hypot(nx, nz), "k*", ms=12, label="Gibbs state")
    axes[0].set_xlabel(r"Bloch angle $\vartheta/\pi$"); axes[0].set_ylabel("Bloch radius $r$")
    axes[0].set_title(r"free energy $F(\rho)$, $H=-Z-X/2$, $\beta=1.2$", fontsize=10); axes[0].legend(fontsize=8)
    S = np.array([[von_neumann_entropy(0.5 * (np.eye(2) + ri * Z)) for tj in th] for ri in r])
    c2 = axes[1].contourf(th / np.pi, r, S, levels=25, cmap="Blues")
    fig.colorbar(c2, ax=axes[1])
    axes[1].plot(np.arctan2(nx, nz) / np.pi, np.hypot(nx, nz), "k*", ms=12)
    axes[1].set_xlabel(r"Bloch angle $\vartheta/\pi$"); axes[1].set_ylabel("Bloch radius $r$")
    axes[1].set_title("von Neumann entropy $S(\\rho)$", fontsize=10)
    fig.tight_layout()
    bookplot.savefig(fig, 5, "freeenergy")
    print("figure freeenergy written")
