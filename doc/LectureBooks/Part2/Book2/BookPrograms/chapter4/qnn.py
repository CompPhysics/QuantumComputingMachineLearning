"""Chapter 4, Secs. 4.2-4.5: the quantum neural network

    f(x; theta) = <psi(x,theta)| O |psi(x,theta)>,
    |psi(x,theta)> = W_L(theta_L) S(x) ... W_1(theta_1) S(x) |0...0>,

Eq. (4-model), with the angle encoding S(x) = prod_q R_y(phi_q) (one angle
phi_q per qubit, re-uploaded before every layer when reupload=True), the
hardware-efficient layers W_l = [CNOT ladder 0->1->...->n-1] prod_q R_z(theta^z_{lq}) R_y(theta^y_{lq})
of Fig. 4-circuit, and the observable O = Z on the last qubit, the one at
the end of the ladder.  In the Heisenberg picture the ladder maps Z_{n-1}
to the full string Z_0 Z_1 ... Z_{n-1}, so every R_y of the last layer is
seen by the measurement and the output reaches the full Fourier degree n
per encoding block (Sec. 4.3); the R_z of the last layer commute with O
and are redundant (Sec. 4.5).  rotations=("Y",) gives the real-amplitude
circuit with half the parameters.  The class works on
batches: `angles` has shape (N, n) -- one row of n encoding angles per
sample -- or (N, n_enc, n) when every encoding block gets its own angles
(needed for the per-gate shift rules of Sec. 4.8).

    QNN(n_qubits, n_layers, reupload=True, rotations=("Y", "Z"), entangler="ladder")
        .state(angles, theta)        (N, 2^n) amplitudes
        .forward(angles, theta)      (N,) expectation values of O
        .gradient(angles, theta)     (N, P) parameter-shift gradients, Eq. (4-shift)
        .state_derivatives(angles, theta)  (P, N, 2^n): d|psi>/d theta_mu = |psi(theta + pi e_mu)>/2
        .fourier_coefficients(theta, scale)  the coefficients c_k of Eq. (4-fourier)
        .random_parameters(rng)

Also here: trig_weights(G, order), the weights of the general shift rule
Eq. (4-generalshift) (derivatives of a trigonometric polynomial of degree G
from 2G+1 equidistant evaluations).

Conventions: qubit 0 is the leftmost (most significant) bit; R_a(t) =
exp(-i t sigma_a / 2); `observable` may be given as the diagonal (an array
of +-1 over the basis states) of any Z-string.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]      # BookPrograms/
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np


def z_diagonal(n_qubits, qubits=(0,)):
    """Diagonal of the Pauli string prod_{q in qubits} Z_q over the 2^n basis
    states (qubit 0 = most significant bit)."""
    idx = np.arange(2 ** n_qubits)
    d = np.ones(2 ** n_qubits)
    for q in qubits:
        d *= 1.0 - 2.0 * ((idx >> (n_qubits - 1 - q)) & 1)
    return d


def cnot_permutation(n_qubits, control, target):
    """CNOT as a permutation of the basis states: psi_new = psi[perm]."""
    idx = np.arange(2 ** n_qubits)
    cbit = (idx >> (n_qubits - 1 - control)) & 1
    return idx ^ (cbit << (n_qubits - 1 - target))


def trig_weights(G, order):
    """Weights w_m, m = -G..G, such that for every trigonometric polynomial
    u(phi) = sum_{|k|<=G} c_k e^{i k phi} the derivative of the given order is
        u^(order)(phi) = sum_m w_m u(phi + 2 pi m / (2G+1)).
    This is the general parameter-shift rule, Eq. (4-generalshift)."""
    m = np.arange(-G, G + 1)
    phi = 2 * np.pi * m / (2 * G + 1)
    k = np.arange(1, G + 1)
    B = np.hstack([np.ones((2 * G + 1, 1)), np.cos(np.outer(phi, k)), np.sin(np.outer(phi, k))])
    # derivatives of the basis functions 1, cos(k phi), sin(k phi) at phi = 0
    d = np.concatenate([[1.0 if order == 0 else 0.0],
                        k ** order * np.cos(order * np.pi / 2),
                        k ** order * np.sin(order * np.pi / 2)])
    return np.linalg.solve(B.T, d), phi


class QNN:
    def __init__(self, n_qubits, n_layers, reupload=True, rotations=("Y", "Z"),
                 entangler="ladder", observable=None):
        self.n, self.L = n_qubits, n_layers
        self.dim = 2 ** n_qubits
        self.reupload = reupload
        self.n_enc = n_layers if reupload else 1       # encoding blocks
        self.rotations = tuple(rotations)
        self.n_rot = len(self.rotations)
        self.n_params = self.n_rot * n_qubits * n_layers
        self.z = (z_diagonal(n_qubits, (n_qubits - 1,)) if observable is None
                  else np.asarray(observable, float))
        pairs = [(q, q + 1) for q in range(n_qubits - 1)]
        if entangler == "ring" and n_qubits > 2:
            pairs.append((n_qubits - 1, 0))
        elif entangler not in ("ring", "ladder", "none"):
            raise ValueError(entangler)
        if entangler == "none":
            pairs = []
        perm = np.arange(self.dim)
        for (c, t) in pairs:                             # psi <- psi[perm_k] in sequence
            perm = perm[cnot_permutation(n_qubits, c, t)]
        self.entangler, self.perm = entangler, perm

    # ------------------------------------------------------------ gates
    def _apply_1q(self, states, U, q, per_sample=False):
        """Apply a 2x2 matrix to qubit q of the states (S, N, 2^n): U of shape
        (N, 2, 2) with per_sample=True (one matrix per sample, the encoding)
        or (S, 2, 2) (one matrix per parameter set).  Written out as four
        broadcast products, which is much faster than einsum for these sizes."""
        S_, N = states.shape[:2]
        T = states.reshape(S_, N, 2 ** q, 2, 2 ** (self.n - q - 1))
        if per_sample:
            u = U.reshape(1, N, 1, 2, 2, 1)
        else:
            u = U.reshape(S_, 1, 1, 2, 2, 1)
        T0, T1 = T[:, :, :, 0, :], T[:, :, :, 1, :]
        out = np.empty_like(T)
        out[:, :, :, 0, :] = u[:, :, :, 0, 0] * T0 + u[:, :, :, 0, 1] * T1
        out[:, :, :, 1, :] = u[:, :, :, 1, 0] * T0 + u[:, :, :, 1, 1] * T1
        return out.reshape(S_, N, self.dim)

    @staticmethod
    def _ry(t):
        """R_y for a scalar angle or an array of angles (..., 2, 2)."""
        t = np.asarray(t, float)
        c, s = np.cos(t / 2), np.sin(t / 2)
        U = np.zeros(t.shape + (2, 2), complex)
        U[..., 0, 0] = c; U[..., 0, 1] = -s; U[..., 1, 0] = s; U[..., 1, 1] = c
        return U

    @staticmethod
    def _rz(t):
        t = np.asarray(t, float)
        U = np.zeros(t.shape + (2, 2), complex)
        U[..., 0, 0] = np.exp(-0.5j * t); U[..., 1, 1] = np.exp(0.5j * t)
        return U

    @staticmethod
    def _rx(t):
        t = np.asarray(t, float)
        c, s = np.cos(t / 2), np.sin(t / 2)
        U = np.zeros(t.shape + (2, 2), complex)
        U[..., 0, 0] = c; U[..., 0, 1] = -1j * s; U[..., 1, 0] = -1j * s; U[..., 1, 1] = c
        return U

    def _rotation(self, axis, t):
        return {"X": self._rx, "Y": self._ry, "Z": self._rz}[axis](t)

    def _angles(self, angles):
        """Bring the encoding angles to shape (N, n_enc, n)."""
        a = np.asarray(angles, float)
        if a.ndim == 1:
            a = a[:, None]
        if a.ndim == 2:
            a = np.repeat(a[:, None, :], self.n_enc, axis=1)
        assert a.shape[1:] == (self.n_enc, self.n), a.shape
        return a

    def unpack(self, theta):
        """theta (P,) or (S, P) -> array (S, L, n_rot, n): the angle of rotation r
        on qubit q in layer l of parameter set s is theta[s, l, r, q]."""
        theta = np.asarray(theta, float)
        assert theta.shape[-1] == self.n_params, (theta.shape, self.n_params)
        return theta.reshape(-1, self.L, self.n_rot, self.n)

    # ------------------------------------------------------------ the model
    def state(self, angles, theta):
        """Amplitudes (N, 2^n) for one parameter vector, or (S, N, 2^n) for S
        parameter vectors stacked in the rows of theta."""
        a = self._angles(angles)
        N = a.shape[0]
        single = np.ndim(theta) == 1
        t = self.unpack(theta)
        S_ = t.shape[0]
        psi = np.zeros((S_, N, self.dim), complex); psi[:, :, 0] = 1.0
        for l in range(self.L):
            if l == 0 or self.reupload:                 # S(x): one R_y per qubit
                for q in range(self.n):
                    psi = self._apply_1q(psi, self._ry(a[:, min(l, self.n_enc - 1), q]), q, per_sample=True)
            for q in range(self.n):                     # W_l: rotations, then the CNOT ladder
                U = np.broadcast_to(np.eye(2, dtype=complex), (S_, 2, 2))
                for r, axis in enumerate(self.rotations):
                    U = self._rotation(axis, t[:, l, r, q]) @ U
                psi = self._apply_1q(psi, U, q)
            psi = psi[:, :, self.perm]
        return psi[0] if single else psi

    def forward(self, angles, theta):
        return np.abs(self.state(angles, theta)) ** 2 @ self.z

    def gradient(self, angles, theta, shift=np.pi / 2):
        """Parameter-shift rule, Eq. (4-shift): d f / d theta_mu =
        [f(theta_mu + pi/2) - f(theta_mu - pi/2)] / 2, for the whole batch;
        the 2P shifted parameter sets are evaluated in one call."""
        theta = np.asarray(theta, float)
        P = self.n_params
        T = np.repeat(theta[None, :], 2 * P, axis=0)
        T[np.arange(P), np.arange(P)] += shift
        T[P + np.arange(P), np.arange(P)] -= shift
        f = self.forward(angles, T)                     # (2P, N)
        return 0.5 * (f[:P] - f[P:]).T

    def state_derivatives(self, angles, theta):
        """d|psi>/d theta_mu = |psi(theta + pi e_mu)>/2 exactly, since
        dR(t)/dt = R(t + pi)/2 for R(t) = exp(-i t P/2); shape (P, N, 2^n)."""
        theta = np.asarray(theta, float)
        P = self.n_params
        T = np.repeat(theta[None, :], P, axis=0) + np.pi * np.eye(P)
        return 0.5 * self.state(angles, T)

    def random_parameters(self, rng=0, scale=np.pi):
        rng = np.random.default_rng(rng)
        return rng.uniform(-scale, scale, self.n_params)

    # ------------------------------------------------------------ Fourier picture
    def fourier_degree(self):
        """Degree G of f as a trigonometric polynomial in phi when every
        encoding angle equals phi: one unit per encoding gate."""
        return self.n * self.n_enc

    def fourier_coefficients(self, theta, scale=np.pi):
        """The output as a function of a scalar x with all encoding angles
        phi = scale * x is f(x) = sum_{|k|<=G} c_k exp(i k scale x), Eq. (4-fourier).
        The 2G+1 coefficients follow exactly from 2G+1 equidistant samples."""
        G = self.fourier_degree()
        phi = 2 * np.pi * np.arange(2 * G + 1) / (2 * G + 1)
        f = self.forward(np.repeat(phi[:, None], self.n, axis=1), theta)
        k = np.arange(-G, G + 1)
        c = np.exp(-1j * np.outer(k, phi)) @ f / (2 * G + 1)
        return k, c


if __name__ == "__main__":
    from optimize import numerical_gradient
    rng = np.random.default_rng(1)

    # 1. parameter-shift gradient against central differences
    qnn = QNN(2, 2)
    theta = qnn.random_parameters(rng)
    X = rng.uniform(0, np.pi, (5, 2))
    J = qnn.gradient(X, theta)
    Jnum = np.array([numerical_gradient(lambda t: qnn.forward(X[i:i + 1], t)[0], theta)
                     for i in range(5)])
    print(f"parameter shift vs finite differences: max |diff| = {np.max(np.abs(J - Jnum)):.1e}")
    assert np.max(np.abs(J - Jnum)) < 1e-6

    # 2. the output as a trigonometric polynomial: 2G+1 samples reproduce it everywhere
    for L in (1, 2, 3):
        q = QNN(2, L)
        t = q.random_parameters(rng)
        k, c = q.fourier_coefficients(t)
        x = rng.uniform(-3, 3, 50)
        f_fourier = (np.exp(1j * np.outer(x * np.pi, k)) @ c).real
        f_direct = q.forward(np.repeat(np.pi * x[:, None], 2, axis=1), t)
        assert np.allclose(f_fourier, f_direct, atol=1e-12)
        print(f"n=2, L={L}: degree G = {q.fourier_degree()}, |c_k| = "
              + ", ".join(f"{abs(c[G_ + kk]):.3f}" for G_ in [q.fourier_degree()] for kk in range(G_ + 1)))

    # 3. the general shift rule: derivatives of a degree-G trigonometric polynomial
    q = QNN(2, 2); t = q.random_parameters(rng); G = q.fourier_degree()
    x0, s = 0.37, np.pi
    f = lambda x: q.forward(np.repeat(np.asarray(x, float)[:, None] * s, 2, axis=1), t)
    for order in (1, 2):
        w, phi = trig_weights(G, order)
        d_shift = s ** order * (w @ f(x0 + phi / s))
        h = 1e-4
        d_fd = ((f([x0 + h]) - f([x0 - h])) / (2 * h) if order == 1
                else (f([x0 + h]) - 2 * f([x0]) + f([x0 - h])) / h ** 2)[0]
        print(f"d^{order}f/dx^{order} at x={x0}: general shift rule {d_shift:.8f}, finite difference {d_fd:.8f}")
        assert abs(d_shift - d_fd) < 1e-5

    # 4. state derivative identity
    D = qnn.state_derivatives(X[:1], theta)
    h = 1e-6
    for mu in (0, 3):
        tp, tm = theta.copy(), theta.copy(); tp[mu] += h; tm[mu] -= h
        Dnum = (qnn.state(X[:1], tp) - qnn.state(X[:1], tm)) / (2 * h)
        assert np.allclose(D[mu], Dnum, atol=1e-6)
    print("state derivative d|psi>/d theta = |psi(theta + pi e_mu)>/2 verified")

    # 5. Fig. 4-fourier: the output of random two-qubit circuits and its spectrum
    import matplotlib.pyplot as plt
    import bookplot
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    x = np.linspace(-1, 1, 400)
    for L, ls in [(1, "-"), (2, "--"), (4, ":")]:
        q = QNN(2, L); t = q.random_parameters(11)
        k, c = q.fourier_coefficients(t)
        axes[0].plot(x, q.forward(np.repeat(np.pi * x[:, None], 2, axis=1), t), ls, label=f"$L={L}$, $G={q.fourier_degree()}$")
        axes[1].plot(k[k >= 0], np.abs(c[k >= 0]), "o" + ls, ms=4, label=f"$L={L}$")
    axes[0].set_xlabel("$x$ (one period)"); axes[0].set_ylabel("$f(x;\\theta)$"); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("frequency $k$"); axes[1].set_ylabel("$|c_k|$"); axes[1].legend(fontsize=8)
    axes[1].set_xticks(range(0, 9))
    fig.tight_layout()
    bookplot.savefig(fig, 4, "fourier")
    print("figure fourier written")
