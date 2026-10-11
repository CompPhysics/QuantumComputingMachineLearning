"""Chapter 3, Sec. 3.6: the Baker-Campbell-Hausdorff expansion and Trotter splitting.

Checks the BCH series (Eq. 3.bch) order by order on two 2x2 matrices
(Table 3.bch), then the Lie-Trotter and Strang splittings (Eqs. 3.trotter,
3.strang) for a single qubit with H = X + Z (Table 3.trotter) and for the
four-site Heisenberg ring split into even and odd bonds (Fig. 3.trotter),
where every factor is a two-qubit operation.  Instantaneous.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np
import gates as G
from matrixfunctions import expm, expm_hermitian
from models import HeisenbergModel
from bookplot import savefig, plt


def comm(A, B):
    return A @ B - B @ A


def bch_terms(X, Y):
    """The BCH series through fourth order, one term per entry."""
    return [X + Y,
            0.5 * comm(X, Y),
            (comm(X, comm(X, Y)) + comm(Y, comm(Y, X))) / 12,
            -comm(Y, comm(X, comm(X, Y))) / 24]


def lie_trotter(Hs, t, n):
    """(prod_k e^{-i H_k t/n})^n."""
    step = np.eye(Hs[0].shape[0], dtype=complex)
    for H in Hs:
        step = step @ expm_hermitian(H, -1j * t / n)
    return np.linalg.matrix_power(step, n)


def strang(Hs, t, n):
    """Symmetric splitting: e^{-iH_1 dt/2} ... e^{-iH_m dt} ... e^{-iH_1 dt/2}, repeated n times."""
    dt = t / n
    half = [expm_hermitian(H, -1j * dt / 2) for H in Hs[:-1]]
    step = np.eye(Hs[0].shape[0], dtype=complex)
    for U in half:
        step = step @ U
    step = step @ expm_hermitian(Hs[-1], -1j * dt)
    for U in reversed(half):
        step = step @ U
    return np.linalg.matrix_power(step, n)


if __name__ == "__main__":
    # ---------------------------------------------------------------- BCH
    print("=== BCH expansion, X = [[0,0.1],[0,0]], Y = [[0,0],[0.1,0]] (Table 3.bch)")
    X = np.array([[0, 0.1], [0, 0]]); Y = np.array([[0, 0], [0.1, 0]])
    target = expm(X) @ expm(Y)
    Z = np.zeros((2, 2), dtype=complex)
    for order, term in enumerate(bch_terms(X, Y), start=1):
        Z = Z + term
        print(f"  order {order}: ||e^X e^Y - e^Z_n|| = {np.linalg.norm(target - expm(Z)):.3e}")
    # --------------------------------------------------------- one qubit
    print("\n=== Trotter splitting of exp(-i(X+Z)t), t = 1 (Table 3.trotter)")
    exact = expm_hermitian(G.X + G.Z, -1j)
    print(f"{'N':>4s} {'first order':>12s} {'ratio':>6s} {'second order':>13s} {'ratio':>6s}")
    e1_old = e2_old = None
    for n in (1, 2, 4, 8, 16, 32, 64):
        e1 = np.linalg.norm(lie_trotter([G.X, G.Z], 1.0, n) - exact, 2)
        e2 = np.linalg.norm(strang([G.X, G.Z], 1.0, n) - exact, 2)
        r1 = f"{e1_old / e1:6.2f}" if e1_old else "   ---"
        r2 = f"{e2_old / e2:6.2f}" if e2_old else "   ---"
        print(f"{n:4d} {e1:12.3e} {r1} {e2:13.3e} {r2}")
        e1_old, e2_old = e1, e2
    # ----------------------------------------------------- Heisenberg ring
    print("\n=== Heisenberg ring, 4 sites, even/odd bond splitting (Fig. 3.trotter)")
    heis = HeisenbergModel(4)
    H = heis.full_matrix()
    even = HeisenbergModel(4, bonds=[(0, 1), (2, 3)]).full_matrix()
    odd = HeisenbergModel(4, bonds=[(1, 2), (3, 0)]).full_matrix()
    assert np.allclose(even + odd, H)
    t = 2.0
    exact = expm_hermitian(H, -1j * t)
    ns = [1, 2, 4, 8, 16, 32, 64, 128]
    err1 = [np.linalg.norm(lie_trotter([even, odd], t, n) - exact, 2) for n in ns]
    err2 = [np.linalg.norm(strang([even, odd], t, n) - exact, 2) for n in ns]
    for n, a, b in zip(ns, err1, err2):
        print(f"  n = {n:3d}: first order {a:.3e}, second order {b:.3e}")
    slope1 = np.polyfit(np.log(ns[-4:]), np.log(err1[-4:]), 1)[0]
    slope2 = np.polyfit(np.log(ns[-4:]), np.log(err2[-4:]), 1)[0]
    print(f"  asymptotic slopes: {slope1:.2f} (first order), {slope2:.2f} (second order)")
    # also the norm of the commutator that governs the first-order error
    print(f"  ||[H_even, H_odd]|| = {np.linalg.norm(comm(even, odd), 2):.4f}")
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.loglog(ns, err1, "o-", label="Lie-Trotter, $\\mathcal{O}(1/n)$")
    ax.loglog(ns, err2, "s-", label="Strang, $\\mathcal{O}(1/n^2)$")
    ax.loglog(ns, err1[0] / np.array(ns), "k:", lw=0.8); ax.loglog(ns, err2[0] / np.array(ns) ** 2, "k:", lw=0.8)
    ax.set_xlabel("Trotter steps $n$"); ax.set_ylabel(r"$\|U_n - e^{-iHt}\|$")
    ax.set_title("Heisenberg ring, 4 sites, $t=2$", fontsize=9); ax.legend(fontsize=8)
    print("figure written to", savefig(fig, 3, "trotter"))
