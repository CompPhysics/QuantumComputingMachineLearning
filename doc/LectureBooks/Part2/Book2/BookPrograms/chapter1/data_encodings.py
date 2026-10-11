"""Chapter 1, Sec. 1.4: three ways of loading a classical data vector into
a qubit register, Eqs. (1-basis), (1-angle) and (1-amplitude).

    basis encoding      an integer (or bit string) becomes a basis state
    angle encoding      one feature per qubit, as a rotation angle
    amplitude encoding  the normalized vector becomes the amplitudes

The script prints the three encodings of the same small vector, counts the
qubits each needs for d features (Fig. 1-qubitcount) and shows why angle
encoding needs its inputs rescaled: the encoded state is periodic in the
feature (Fig. 1-periodic).  Only the shared state-vector simulator of
common/ is used.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]      # BookPrograms/
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np

from statevector import StateVector


def basis_encoding(k, n_qubits):
    """|k> for the integer k written with n_qubits bits, Eq. (1-basis)."""
    return StateVector.from_bits(format(k, f"0{n_qubits}b"))


def angle_encoding(x, rotation="Y"):
    """(x) R(x_i)|0>: one rotation per feature, Eq. (1-angle)."""
    sv = StateVector(len(x))
    for q, xq in enumerate(x):
        if rotation == "Y":
            sv.ry(xq, q)
        elif rotation == "X":
            sv.rx(xq, q)
        else:
            sv.h(q); sv.rz(xq, q)
    return sv


def amplitude_encoding(x):
    """sum_k x_k |k> / ||x||, padded with zeros to 2^n amplitudes, Eq. (1-amplitude)."""
    n = int(np.ceil(np.log2(len(x)))) if len(x) > 1 else 1
    amp = np.zeros(2 ** n)
    amp[:len(x)] = x
    return StateVector(n, amp)


def qubits_needed(d, precision_bits=8):
    """Number of qubits for d features in the three encodings."""
    return {"basis (8 bits per feature)": precision_bits * d,
            "angle": d,
            "amplitude": int(np.ceil(np.log2(d)))}


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from bookplot import savefig

    x = np.array([0.3, 1.2, 2.0, 0.7])
    print("data vector x =", x)
    print("basis encoding of k = 5 on 3 qubits:   ", basis_encoding(5, 3))
    print("angle encoding (R_y), 4 qubits:         ", angle_encoding(x))
    amp = amplitude_encoding(x)
    print("amplitude encoding, 2 qubits:           ", amp)
    assert np.allclose(amp.psi.real, x / np.linalg.norm(x))
    # the angle-encoded state is a product state: its kernel factorizes
    xp = x + 0.4
    overlap = abs(angle_encoding(x).overlap(angle_encoding(xp))) ** 2
    assert np.isclose(overlap, np.prod(np.cos((x - xp) / 2) ** 2))
    print(f"|<phi(x)|phi(x')>|^2 for the angle encoding = {overlap:.4f} = prod_i cos^2((x_i-x'_i)/2)")

    # Fig. 1-qubitcount
    ds = np.array([2, 4, 8, 16, 64, 256, 1024, 4096])
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    for name, style in (("basis (8 bits per feature)", "s-"), ("angle", "o-"), ("amplitude", "^-")):
        ax.loglog(ds, [qubits_needed(d)[name] for d in ds], style, label=name)
    ax.set_xlabel("number of features $d$"); ax.set_ylabel("qubits")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    savefig(fig, 1, "qubitcount")

    # Fig. 1-periodic: one feature, one qubit, R_y(x)|0> = cos(x/2)|0> + sin(x/2)|1>
    xs = np.linspace(-2 * np.pi, 4 * np.pi, 400)
    p1 = np.sin(xs / 2) ** 2
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.plot(xs, p1, label="$P(1) = \\sin^2(x/2)$")
    ax.axvspan(0, np.pi, color="tab:green", alpha=0.15, label="rescaled inputs $x\\in[0,\\pi]$")
    ax.set_xlabel("feature value $x$"); ax.set_ylabel("$P(\\mathrm{qubit}=1)$")
    ax.legend(loc="upper right", fontsize=8)
    savefig(fig, 1, "periodic")
    print("figures qubitcount and periodic written")
