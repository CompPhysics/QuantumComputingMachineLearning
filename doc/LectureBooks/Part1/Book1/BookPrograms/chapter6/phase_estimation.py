"""Chapter 6, Secs. 6.4-6.5: quantum phase estimation on the state-vector simulator.

``PhaseEstimation`` runs the circuit of Fig. 6.qpe: Hadamards on m control
qubits, controlled powers U^(2^j), the inverse QFT of ``qft.py``, and the
read-out of the control register.  The control qubits are qubits 0..m-1
(qubit 0 carries the most significant bit of the phase and controls
U^(2^(m-1))), the system register is qubits m..m+n-1.

Two ways of applying the controlled powers are provided and checked against
each other: ``method="circuit"`` applies the (2^(n+1) x 2^(n+1)) controlled
matrix to the register, gate by gate, as a quantum computer would; ``method=
"spectral"`` rotates the system register into the eigenbasis of U once,
where every controlled power is diagonal, and rotates back at the end.  The
two are the same unitary, since the rotation commutes with the controls; the
second is affordable for the eight-qubit Hubbard Hamiltonian of Sec. 6.7.

Also here: the Dirichlet-kernel probability of Eq. (6.dirichlet), the signed
phase-to-energy conversion of Eq. (6.phasetoenergy), and iterative phase
estimation with a single control qubit (Sec. 6.5.4).  Run as a script it
reproduces the exact example phi = 5/8, Table 6.inexact and Fig. 6.dirichlet
for phi = 0.3, the 4/pi^2 bound, and the iterative example.  A few seconds.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np
import gates as G
from statevector import StateVector
from qft import qft_matrix, inverse_qft_circuit
from bookplot import savefig, plt


def controlled(U):
    """|0><0| (x) I + |1><1| (x) U for a unitary of any dimension (the control is the first qubit)."""
    U = np.asarray(U, dtype=complex)
    d = U.shape[0]
    return np.kron(np.diag([1, 0]), np.eye(d)) + np.kron(np.diag([0, 1]), U)


def dirichlet_probability(phi, m, k):
    """Probability of reading k from an m-qubit register when the phase is phi,
    Eq. (6.dirichlet): sin^2(pi 2^m delta) / (2^(2m) sin^2(pi delta)), delta = phi - k/2^m."""
    delta = (phi - np.asarray(k) / 2 ** m + 0.5) % 1 - 0.5
    out = np.ones_like(delta, dtype=float)
    small = np.abs(delta) < 1e-12
    d = np.where(small, 1.0, delta)
    out = np.where(small, 1.0, np.sin(np.pi * 2 ** m * d) ** 2 / (4 ** m * np.sin(np.pi * d) ** 2))
    return out


def signed_phase(k, m):
    """Read the m-bit outcome k as a phase in [-1/2, 1/2): k/2^m for k < 2^(m-1), else k/2^m - 1."""
    phi = np.asarray(k) / 2 ** m
    return np.where(phi >= 0.5, phi - 1, phi)


def phase_to_energy(k, m, t):
    """E = -2 pi phi / t with the signed reading of the phase, Eq. (6.phasetoenergy)."""
    return -2 * np.pi * signed_phase(k, m) / t


def choose_time(H, s=0.9):
    """t = pi s / Lambda with Lambda the Pauli 1-norm of the traceless part of the
    PauliSum H, so that |E| t < pi for every eigenvalue, Eq. (6.timechoice)."""
    lam = sum(abs(c) for k, c in H.terms.items() if any(ch != "I" for ch in k))
    return np.pi * s / lam


class PhaseEstimation:
    """Quantum phase estimation of a unitary on n system qubits with m control qubits."""

    def __init__(self, n_system, m, power, eigen=None, method="circuit"):
        """``power(p)`` returns the matrix U^p; ``eigen = (phases, V)`` with
        U = V diag(exp(2 pi i phases)) V^dagger enables the spectral method."""
        self.n, self.m = n_system, m
        self.power = power
        self.eigen = eigen
        self.method = method
        self.state = None

    # ------------------------------------------------------------ builders
    @classmethod
    def from_hamiltonian(cls, H, t, m, method="circuit"):
        """U = exp(-i H t) for a Hermitian matrix H; powers from the spectral theorem."""
        H = np.asarray(H)
        w, V = np.linalg.eigh(H)
        n = int(round(np.log2(H.shape[0])))
        phases = (-w * t / (2 * np.pi)) % 1.0
        power = lambda p: (V * np.exp(-1j * w * t * p)) @ V.conj().T
        obj = cls(n, m, power, eigen=(phases, V), method=method)
        obj.t = t
        return obj

    @classmethod
    def from_unitary(cls, U, m, method="circuit"):
        """Any unitary matrix, e.g. a Trotterised evolution; powers by repeated squaring."""
        U = np.asarray(U, dtype=complex)
        n = int(round(np.log2(U.shape[0])))
        cache = {1: U}

        def power(p):
            if p not in cache:
                half = power(p // 2)
                cache[p] = half @ half if p % 2 == 0 else half @ half @ U
            return cache[p]
        return cls(n, m, power, method=method)

    # ----------------------------------------------------------------- run
    def run(self, psi_system):
        """Run the circuit on the system state and return the register (control + system)."""
        m, n = self.m, self.n
        psi_system = np.asarray(psi_system, dtype=complex).reshape(2 ** n)
        psi_system = psi_system / np.linalg.norm(psi_system)
        if self.method == "spectral":
            if self.eigen is None:
                raise ValueError("the spectral method needs the eigen-decomposition")
            phases, V = self.eigen
            c = V.conj().T @ psi_system                       # amplitudes in the eigenbasis
            k = np.arange(2 ** m)
            amp = np.exp(2j * np.pi * np.outer(k, phases)) * c / 2 ** (m / 2)   # the Fourier state, Eq. (6.qpefourierstate)
            amp = qft_matrix(m).conj().T @ amp                # inverse QFT on the control index
            amp = amp @ V.T                                   # back to the computational basis of the system
            self.state = StateVector(m + n, amp.reshape(-1))
            return self.state
        sv = StateVector(m + n)
        full = np.zeros(2 ** (m + n), dtype=complex)
        full[:2 ** n] = psi_system                            # controls in |0...0>
        sv.psi = full
        for j in range(m):
            sv.h(j)
        system = list(range(m, m + n))
        for j in range(m):                                    # qubit j controls U^(2^(m-1-j))
            Up = self.power(2 ** (m - 1 - j))
            sv.apply(controlled(Up), [j] + system)
        for g in inverse_qft_circuit(m).gates:                # inverse QFT on the control register
            sv.apply(g.full_matrix(), g.qubits)
        self.state = sv
        return sv

    # ------------------------------------------------------------- readout
    def distribution(self):
        """Probability of every control outcome k = 0..2^m-1 (system traced out)."""
        p = self.state.probabilities().reshape(2 ** self.m, 2 ** self.n)
        return p.sum(axis=1)

    def estimate(self):
        """(k, phase k/2^m, probability) of the modal outcome."""
        p = self.distribution()
        k = int(np.argmax(p))
        return k, k / 2 ** self.m, float(p[k])

    def collapse(self, k):
        """The system register after the control register has been read as k."""
        amp = self.state.psi.reshape(2 ** self.m, 2 ** self.n)[k]
        return amp / np.linalg.norm(amp)

    def energies(self):
        """The energy grid of the register for a run built from a Hamiltonian, Eq. (6.energygrid)."""
        return phase_to_energy(np.arange(2 ** self.m), self.m, self.t)


# ----------------------------------------------------------- iterative QPE
def iterative_phase_estimation(power, n_system, m, psi_system, rng=None, repeats=1):
    """Kitaev's iterative phase estimation with one control qubit, Sec. 6.5.4.

    The bits of phi = 0.b_1 ... b_m are measured from the least significant
    upwards.  For bit j (j = m, ..., 1): prepare the control in |+>, apply the
    controlled U^(2^(j-1)), rotate the control by the phase -2 pi (0.0 b_{j+1}
    ... b_m) already known, apply H and measure.  Each measurement is repeated
    ``repeats`` times and decided by majority.  Returns (bits, phase)."""
    rng = np.random.default_rng(0) if rng is None else rng
    psi_system = np.asarray(psi_system, dtype=complex) / np.linalg.norm(psi_system)
    bits = [0] * (m + 1)                                    # bits[j] = b_j, index 0 unused
    system = list(range(1, n_system + 1))
    for j in range(m, 0, -1):
        correction = sum(bits[l] / 2 ** (l - j + 1) for l in range(j + 1, m + 1))
        votes = 0
        for _ in range(repeats):
            sv = StateVector(1 + n_system)
            full = np.zeros(2 ** (1 + n_system), dtype=complex); full[:2 ** n_system] = psi_system
            sv.psi = full
            sv.h(0)
            sv.apply(controlled(power(2 ** (j - 1))), [0] + system)
            sv.apply(G.phase(-2 * np.pi * correction), 0)
            sv.h(0)
            out, _ = sv.measure(0, rng=rng)
            votes += out
        bits[j] = int(votes > repeats / 2)
    phase = sum(bits[j] / 2 ** j for j in range(1, m + 1))
    return bits[1:], phase


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True, linewidth=120)

    # ------------------------------------------------ exact: phi = 5/8, m = 3
    print("=== Exact case: U = diag(1, e^{2 pi i phi}) with phi = 5/8 = 0.101, eigenstate |1>, m = 3")
    phi = 5 / 8
    U = np.diag([1, np.exp(2j * np.pi * phi)])
    qpe = PhaseEstimation.from_unitary(U, 3)
    qpe.run([0, 1])
    k, est, p = qpe.estimate()
    print(f"  outcome |{format(k, '03b')}> with probability {p:.6f}, estimate {est:.6f}")
    assert k == 5 and abs(p - 1) < 1e-12
    # the two methods agree
    Hd = np.diag([0.0, -2 * np.pi * phi])               # exp(-i H t) with t = 1 has the same eigenphase
    qs = PhaseEstimation.from_hamiltonian(Hd, 1.0, 3, method="spectral"); qs.run([0, 1])
    qc = PhaseEstimation.from_hamiltonian(Hd, 1.0, 3, method="circuit"); qc.run([0, 1])
    assert np.allclose(qs.state.psi, qc.state.psi)
    rng = np.random.default_rng(1)
    Hr = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)); Hr = Hr + Hr.conj().T
    v = rng.normal(size=4) + 1j * rng.normal(size=4)
    for method in ("circuit", "spectral"):
        q = PhaseEstimation.from_hamiltonian(Hr, 0.37, 4, method=method); q.run(v)
        if method == "circuit": ref = q.state.psi.copy()
    assert np.allclose(ref, q.state.psi)
    print("  circuit and spectral implementations agree on a random two-qubit Hamiltonian")

    # ----------------------------------------------- inexact: phi = 0.3
    print("\n=== Inexact case: phi = 0.3 = 0.0100110011..., increasing the register (Table 6.inexact)")
    phi = 0.3
    U = np.diag([1, np.exp(2j * np.pi * phi)])
    print(f"{'m':>3s} {'outcome':>12s} {'probability':>12s} {'estimate':>10s} {'error':>9s} {'two best':>9s}")
    table = {}
    for m in (3, 4, 6, 8):
        qpe = PhaseEstimation.from_unitary(U, m); qpe.run([0, 1])
        p = qpe.distribution()
        k, est, pk = qpe.estimate()
        two = np.sort(p)[-2:].sum()
        table[m] = (k, pk, est, abs(est - phi), two)
        print(f"{m:3d} {'|' + format(k, f'0{m}b') + '>':>12s} {pk:12.4f} {est:10.6f} {abs(est - phi):9.1e} {two:9.3f}")
        assert np.allclose(p, dirichlet_probability(phi, m, np.arange(2 ** m)))
    assert table[3][0] == 2 and table[4][0] == 5 and table[8][0] == 77
    assert abs(table[4][1] - 0.8756) < 1e-4 and abs(table[8][1] - 0.8751) < 1e-4
    print("  every distribution equals the Dirichlet kernel of Eq. (6.dirichlet)")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for ax, m in zip(axes, (4, 8)):
        qpe = PhaseEstimation.from_unitary(U, m); qpe.run([0, 1])
        p = qpe.distribution()
        ax.bar(np.arange(2 ** m) / 2 ** m, p, width=0.8 / 2 ** m)
        ax.axvline(phi, color="k", ls="--", lw=0.8)
        ax.set_xlabel(r"$k/2^m$"); ax.set_title(f"$m={m}$")
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("probability")
    savefig(fig, 6, "qpe_dirichlet")

    # ------------------------------------------------- the 4/pi^2 bound
    print("\n=== The worst case over a whole grid interval, m = 6")
    m = 6
    worst_p, worst_phi = 1.0, None
    for phi in np.linspace(10 / 64, 11 / 64, 201):
        p = dirichlet_probability(phi, m, np.arange(2 ** m))
        if p.max() < worst_p:
            worst_p, worst_phi = p.max(), phi
    print(f"  smallest modal probability {worst_p:.4f} at phi = {worst_phi:.5f} (midpoint 10.5/64 = {10.5 / 64:.5f}); bound 4/pi^2 = {4 / np.pi ** 2:.4f}")
    assert worst_p > 4 / np.pi ** 2 and abs(worst_phi - 10.5 / 64) < 1e-3
    # probability within one grid step of the truth exceeds 8/pi^2 for every phi
    lo = 1.0
    for phi in np.linspace(0, 1, 1001, endpoint=False):
        p = dirichlet_probability(phi, m, np.arange(2 ** m))
        ks = np.arange(2 ** m)
        near = np.abs(((ks / 2 ** m - phi + 0.5) % 1) - 0.5) <= 1 / 2 ** m + 1e-12
        lo = min(lo, p[near].sum())
    print(f"  smallest probability of landing within one grid step: {lo:.4f} >= 8/pi^2 = {8 / np.pi ** 2:.4f}")
    assert lo >= 8 / np.pi ** 2 - 1e-9

    # ------------------------------------------------- iterative QPE
    print("\n=== Iterative phase estimation with one control qubit")
    for phi, m in ((5 / 8, 3), (0.3, 8)):
        U = np.diag([1, np.exp(2j * np.pi * phi)])
        cache = {}

        def power(p, U=U):
            return np.linalg.matrix_power(U, p)
        bits, est = iterative_phase_estimation(power, 1, m, [0, 1], rng=np.random.default_rng(3), repeats=11)
        print(f"  phi = {phi:.4f}, m = {m}: bits {''.join(map(str, bits))}, estimate {est:.6f}, error {abs(est - phi):.1e}")
        if phi == 5 / 8:
            assert bits == [1, 0, 1]
    print("\nphase-estimation checks passed")
