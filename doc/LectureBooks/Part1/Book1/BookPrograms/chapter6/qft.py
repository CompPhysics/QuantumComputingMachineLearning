"""Chapter 6, Secs. 6.2-6.3: the quantum Fourier transform.

* ``qft_matrix(n)``            the 2^n x 2^n Fourier matrix, Eq. (6.dftmatrix);
* ``qft_circuit(n)``           the Hadamard / controlled-R_k / SWAP circuit of
                               Fig. 6.qft3 as a ``Circuit`` of chapter 2, with
                               the options of dropping the final swaps and of
                               truncating the rotation ladder (approximate QFT);
* ``qft_product_state(x, n)``  the product form of Eq. (6.product), built with
                               no circuit at all, as the independent check;
* ``gate_count(n, d)``         Hadamards, controlled rotations and swaps.

Run as a script it verifies the circuit four ways (matrix, product form,
numpy.fft, inverse), prints the eight amplitudes of QFT_8 |101>, the gate
counts of Table 6.qftcost, the approximate-QFT errors of Table 6.aqft, the
bit-reversal check, and writes Figs. 6.qftphases and 6.aqfterror.  About one
second.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np
import gates as G
from circuit import Circuit, Gate
from statevector import StateVector
from bookplot import savefig, plt


# ------------------------------------------------------------- definitions
def qft_matrix(n):
    """F_N with N = 2^n: F[k, x] = exp(2 pi i x k / N) / sqrt(N), Eq. (6.dftmatrix)."""
    N = 2 ** n
    k = np.arange(N)
    return np.exp(2j * np.pi * np.outer(k, k) / N) / np.sqrt(N)


def R(k):
    """The phase rotation R_k = diag(1, exp(2 pi i / 2^k)), Eq. (6.rk): R_1 = Z, R_2 = S, R_3 = T."""
    return G.phase(2 * np.pi / 2 ** k)


def qft_circuit(n, swaps=True, max_k=None):
    """The QFT circuit on n qubits (qubit 0 = most significant bit).

    For j = 0..n-1: a Hadamard on qubit j, then for every later qubit k > j a
    controlled R_{k-j+1} with control k and target j; finally the register is
    reversed with floor(n/2) swaps.  ``max_k`` keeps only the rotations R_k
    with k <= max_k (the approximate QFT); ``swaps=False`` leaves the output
    bit-reversed, as many software libraries do."""
    circ = Circuit(n)
    for j in range(n):
        circ.h(j)
        for k in range(j + 1, n):
            order = k - j + 1
            if max_k is not None and order > max_k:
                continue
            circ.add(Gate(f"R{order}", R(order), j, controls=k))
    if swaps:
        for j in range(n // 2):
            circ.swap(j, n - 1 - j)
    return circ


def inverse_qft_circuit(n, **kw):
    """Reverse the gate order and conjugate every R_k (the swaps are their own inverse)."""
    return qft_circuit(n, **kw).inverse()


def binary_fraction(bits):
    """0.b_1 b_2 ... b_k as a number, Eq. (6.binaryfraction)."""
    return sum(int(b) / 2 ** (i + 1) for i, b in enumerate(bits))


def qft_product_state(x, n):
    """QFT_N |x> from the product form of Eq. (6.product): the m-th output qubit
    (m = 1..n, qubit 0 first) carries the phase exp(2 pi i 0.x_{n-m+1} ... x_n)."""
    bits = format(x, f"0{n}b")
    factors = []
    for m in range(1, n + 1):
        phase = np.exp(2j * np.pi * binary_fraction(bits[n - m:]))
        factors.append(np.array([1, phase]) / np.sqrt(2))
    return StateVector.product(*factors).psi


def gate_count(n, max_k=None):
    """(Hadamards, controlled rotations, swaps) of qft_circuit(n, max_k)."""
    rot = sum(1 for j in range(n) for k in range(j + 1, n) if max_k is None or k - j + 1 <= max_k)
    return n, rot, n // 2


def bit_reverse(x, n):
    return int(format(x, f"0{n}b")[::-1], 2)


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True, linewidth=120)

    # ------------------------------------------------- four verifications
    print("=== The QFT circuit against the Fourier matrix, the product form, numpy.fft and its inverse")
    worst = 0.0
    rng = np.random.default_rng(6)
    for n in (1, 2, 3, 4, 5):
        F = qft_matrix(n)
        U = qft_circuit(n).unitary()
        worst = max(worst, np.abs(U - F).max())
        for x in range(2 ** n):
            psi = StateVector.from_bits(format(x, f"0{n}b"))
            out = qft_circuit(n).run(psi).psi
            worst = max(worst, np.abs(out - qft_product_state(x, n)).max())
        v = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
        v /= np.linalg.norm(v)
        # y_k = N^{-1/2} sum_x v_x e^{+2 pi i xk/N} is numpy's *inverse* transform with 'ortho' normalisation
        worst = max(worst, np.abs(qft_circuit(n).run(StateVector(n, v)).psi - np.fft.ifft(v, norm="ortho")).max())
        Uinv = inverse_qft_circuit(n).unitary()
        worst = max(worst, np.abs(Uinv @ U - np.eye(2 ** n)).max())
    print(f"  worst discrepancy over n = 1..5: {worst:.1e}")
    assert worst < 1e-13

    # ------------------------------------------------- the worked example
    print("\n=== QFT_8 |101>: the eight amplitudes (Eq. 6.example101result)")
    out = qft_circuit(3).run(StateVector.from_bits("101")).psi
    for k, a in enumerate(out):
        print(f"  |{format(k, '03b')}>  {a.real:+.6f}{a.imag:+.6f}i   modulus {abs(a):.6f}   phase/2pi {np.angle(a) / (2 * np.pi) % 1:.4f}   5k/8 mod 1 = {5 * k / 8 % 1:.4f}")
    assert np.allclose(np.abs(out), 1 / np.sqrt(8))
    assert np.allclose(np.angle(out) / (2 * np.pi) % 1, (5 * np.arange(8) / 8) % 1)
    # the three one-qubit factors of Eq. (6.example101result)
    f1 = np.array([1, -1]) / np.sqrt(2); f2 = np.array([1, 1j]) / np.sqrt(2); f3 = np.array([1, np.exp(5j * np.pi / 4)]) / np.sqrt(2)
    assert np.allclose(out, StateVector.product(f1, f2, f3).psi)

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th) / np.sqrt(8), np.sin(th) / np.sqrt(8), "k--", lw=0.8)
    for k, a in enumerate(out):
        ax.annotate("", xy=(a.real, a.imag), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1.4))
        ax.text(1.18 * a.real, 1.18 * a.imag, f"$|{format(k, '03b')}\\rangle$", ha="center", va="center", fontsize=9)
    ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5); ax.set_aspect("equal")
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("Re"); ax.set_ylabel("Im")
    savefig(fig, 6, "qft_phases")

    # ------------------------------------------------------- gate counts
    print("\n=== Gate counts (Table 6.qftcost)")
    print(f"{'n':>4s} {'H':>5s} {'cR_k':>6s} {'swaps':>6s} {'direct DFT ops':>16s}")
    for n in (3, 6, 10, 100):
        h, r, s = gate_count(n)
        print(f"{n:4d} {h:5d} {r:6d} {s:6d} {4.0 ** n:16.1e}")
        assert r == n * (n - 1) // 2

    # --------------------------------------------------- approximate QFT
    print("\n=== The approximate QFT on n = 6 qubits, |x> with x = 37 (Table 6.aqft)")
    n, x = 6, 37
    exact = qft_product_state(x, n)
    psi0 = StateVector.from_bits(format(x, f"0{n}b"))
    print(f"{'d':>3s} {'rotations':>10s} {'largest amplitude error':>24s}")
    rows = []
    for d in range(1, n + 1):
        out = qft_circuit(n, max_k=d).run(psi0.copy()).psi
        err = np.abs(out - exact).max()
        rows.append((d, gate_count(n, d)[1], err))
        print(f"{d:3d} {rows[-1][1]:10d} {err:24.2e}")
    assert rows[0][1] == 0 and rows[-1][1] == 15 and rows[-1][2] < 1e-14
    assert abs(rows[0][2] - 0.25) < 1e-3
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.semilogy([r[1] for r in rows], [max(r[2], 1e-17) for r in rows], "o-")
    for d, nr, err in rows:
        ax.annotate(f"$d={d}$", (nr, max(err, 1e-17)), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("controlled rotations kept"); ax.set_ylabel("largest amplitude error")
    savefig(fig, 6, "aqft_error")

    # ------------------------------------------------------- bit reversal
    print("\n=== Bit reversal: the circuit without swaps produces the output register in reversed order")
    n = 3
    perm = [bit_reverse(k, n) for k in range(2 ** n)]
    for x in range(2 ** n):
        a = qft_circuit(n, swaps=False).run(StateVector.from_bits(format(x, f"0{n}b"))).psi
        b = qft_circuit(n).run(StateVector.from_bits(format(x, f"0{n}b"))).psi
        assert np.allclose(a[perm], b)          # amplitude of |k> without swaps = amplitude of |reverse(k)> with swaps
    a = qft_circuit(n, swaps=False).run(StateVector.from_bits("001")).psi
    print("  QFT_8 |001> without the swaps:", " ".join(f"|{format(k, '03b')}>:{np.angle(a[k]) / (2 * np.pi) % 1:.3f}" for k in range(8)))
    print("  the phase k/8 of output |k> sits at |reverse(k)>: 1/8 at |100>, 2/8 at |010>, 4/8 at |001>")
    print("\nQFT checks passed")
