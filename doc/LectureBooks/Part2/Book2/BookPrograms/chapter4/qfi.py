"""Chapter 4, Sec. 4.5: the quantum Fisher information of a QNN and the
natural gradient.

    qfi_matrix(qnn, angles, theta)    F_jk = 4 Re(<d_j psi|d_k psi> - <d_j psi|psi><psi|d_k psi>),
                                      Eq. (4-qfi), from the exact state derivatives
                                      d|psi>/d theta_mu = |psi(theta + pi e_mu)>/2,
                                      averaged over the batch of inputs
    qfi_overlap(qnn, angles, theta)   the same matrix from overlaps of shifted
                                      states, Eq. (4-qfishift): F_jk =
                                      -[O(++) - O(+-) - O(-+) + O(--)]/2 with
                                      O(theta') = |<psi(theta)|psi(theta')>|^2
    natural_gradient_step(...)        theta <- theta - eta (F + lam I)^{-1} grad L
    transverse_ising(n, h)            the Hamiltonian -sum Z_i Z_{i+1} - h sum X_i as a matrix
    minimize_energy(...)              variational ground-state search with plain
                                      gradient descent, Adam or the natural gradient
                                      (= projected imaginary-time evolution, Sec. 4.5)

The script checks the two QFI formulas against each other and against the
single-qubit value F = 1, shows the rank deficiency caused by redundant
parameters and the difference between flat directions of the metric and of
the loss (Table 4-qfirank), and compares the three optimisers on the
ground state of the transverse-field Ising chain (Fig. 4-natgrad).
Runtime about 10 s.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]      # BookPrograms/
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np

from qnn import QNN
from optimize import Adam


def qfi_matrix(qnn, angles, theta, average=True):
    """Quantum Fisher information (Fubini-Study metric) of |psi(x, theta)>,
    Eq. (4-qfi), for every input x in the batch; averaged unless average=False."""
    psi = qnn.state(angles, theta)                      # (N, dim)
    D = qnn.state_derivatives(angles, theta)            # (P, N, dim)
    # <d_j psi|d_k psi> and <d_j psi|psi>
    G = np.einsum("jna,kna->njk", D.conj(), D)
    v = np.einsum("jna,na->nj", D.conj(), psi)
    chi = G - np.einsum("nj,nk->njk", v, v.conj())       # quantum geometric tensor
    F = 4 * chi.real
    return F.mean(axis=0) if average else F


def qfi_overlap(qnn, angles, theta, average=True):
    """The QFI from the overlap O(theta') = |<psi(theta)|psi(theta')>|^2 of
    shifted states, Eq. (4-qfishift): F_jk = -2 d_j d_k O at theta' = theta,
    the mixed second derivative taken with the shift rule."""
    theta = np.asarray(theta, float)
    P = qnn.n_params
    psi0 = qnn.state(angles, theta)
    N = psi0.shape[0]

    def overlap(shifts):
        t = theta.copy()
        for mu, s in shifts:
            t[mu] += s
        return np.abs(np.einsum("na,na->n", psi0.conj(), qnn.state(angles, t))) ** 2

    F = np.zeros((N, P, P))
    for j in range(P):
        F[:, j, j] = 1.0 - overlap([(j, np.pi)])
        for k in range(j + 1, P):
            s = np.pi / 2
            F[:, j, k] = -0.5 * (overlap([(j, s), (k, s)]) - overlap([(j, s), (k, -s)])
                                 - overlap([(j, -s), (k, s)]) + overlap([(j, -s), (k, -s)]))
            F[:, k, j] = F[:, j, k]
    return F.mean(axis=0) if average else F


def natural_gradient_step(theta, grad, F, eta=0.1, lam=1e-3):
    """theta <- theta - eta (F + lam I)^{-1} grad, Eq. (4-natgrad); the
    regulator lam makes the step well defined on flat directions."""
    return theta - eta * np.linalg.solve(F + lam * np.eye(len(theta)), grad)


def transverse_ising(n, h=1.0):
    """H = -sum_i Z_i Z_{i+1} - h sum_i X_i on an open chain, as a 2^n x 2^n matrix."""
    Z, X, I = np.diag([1.0, -1.0]), np.array([[0.0, 1.0], [1.0, 0.0]]), np.eye(2)

    def op(P, q):
        M = np.array([[1.0]])
        for k in range(n):
            M = np.kron(M, P if k == q else I)
        return M

    H = np.zeros((2 ** n, 2 ** n))
    for q in range(n - 1):
        H -= op(Z, q) @ op(Z, q + 1)
    for q in range(n):
        H -= h * op(X, q)
    return H


def minimize_energy(qnn, H, theta0, method="natural", steps=100, eta=0.2, lam=1e-3):
    """Variational ground-state search E(theta) = <psi(theta)|H|psi(theta)> with the
    data-free circuit (all encoding angles zero), the parameter-shift gradient
    and one of three optimisers.  Returns the parameters and the energy history."""
    theta = np.array(theta0, float)
    A = np.zeros((1, qnn.n))
    opt = Adam(lr=eta) if method == "adam" else None

    def energy(t):
        psi = qnn.state(A, t)[0]
        return float((psi.conj() @ H @ psi).real)

    def gradient(t):
        g = np.zeros(len(t))
        for mu in range(len(t)):
            tp, tm = t.copy(), t.copy(); tp[mu] += np.pi / 2; tm[mu] -= np.pi / 2
            g[mu] = 0.5 * (energy(tp) - energy(tm))
        return g

    hist = [energy(theta)]
    for it in range(steps):
        g = gradient(theta)
        if method == "gd":
            theta = theta - eta * g
        elif method == "adam":
            theta = opt.step(theta, g)
        elif method == "natural":
            theta = natural_gradient_step(theta, g, qfi_matrix(qnn, A, theta), eta, lam)
        else:
            raise ValueError(method)
        hist.append(energy(theta))
    return theta, np.array(hist)


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    import bookplot
    rng = np.random.default_rng(0)

    # ---- 1. a single qubit: |psi> = R_y(theta)|0>, F = 4 Var(Y/2) = 1
    q1 = QNN(1, 1, rotations=("Y",), entangler="none")
    F1 = qfi_matrix(q1, np.zeros((1, 1)), [0.7])
    print(f"single qubit R_y(theta)|0>: F = {F1[0, 0]:.6f}")
    assert abs(F1[0, 0] - 1) < 1e-10

    # ---- 2. the two formulas agree; the rank counts the independent directions
    print("\nTable 4-qfirank: the QFI matrix of two-qubit circuits, averaged over 8 inputs")
    print(f"{'circuit':>44}  {'P':>3}  {'rank F':>6}  {'zero output gradients':>22}  {'|F_exact - F_shift|':>20}")
    X = rng.uniform(0, np.pi, (8, 2))
    cases = [("Y rotations, L=1", dict(rotations=("Y",), n_layers=1)),
             ("Y rotations, L=2", dict(rotations=("Y",), n_layers=2)),
             ("Y+Z rotations, L=2", dict(rotations=("Y", "Z"), n_layers=2)),
             ("Y rotations, L=2, no re-upload, no CNOT", dict(rotations=("Y",), n_layers=2, reupload=False, entangler="none")),
             ("Y+Z rotations, L=2, no CNOT", dict(rotations=("Y", "Z"), n_layers=2, entangler="none"))]
    for name, kw in cases:
        q = QNN(2, **kw)
        th = q.random_parameters(rng)
        F = qfi_matrix(q, X, th)
        Fs = qfi_overlap(q, X, th)
        ev = np.sort(np.linalg.eigvalsh(F))
        rank = int(np.sum(ev > 1e-8))
        J = q.gradient(X, th)                            # (N, P): output gradients
        dead = int(np.sum(np.max(np.abs(J), axis=0) < 1e-10))
        print(f"{name:>44}  {q.n_params:>3}  {rank:>6}  {dead:>22}  {np.max(np.abs(F - Fs)):>20.1e}")
        assert np.max(np.abs(F - Fs)) < 1e-10
        if kw.get("rotations") == ("Y", "Z") and kw.get("entangler", "ladder") == "ladder":
            assert dead == 2, "the R_z of the last layer commute with Z_1"
        if kw.get("entangler") == "none" and kw.get("reupload", True) is False:
            assert rank == 2, "consecutive rotations about the same axis merge"

    # ---- 3. optimisers on a variational ground-state problem
    n, L, h = 4, 3, 1.0
    H = transverse_ising(n, h)
    E0 = np.linalg.eigvalsh(H)[0]
    ansatz = QNN(n, L, rotations=("Y",))                # real amplitudes suffice: H is real
    theta0 = ansatz.random_parameters(1, scale=0.5)
    print(f"\ntransverse-field Ising chain, n={n}, h={h}: E_0 = {E0:.6f}; ansatz with {ansatz.n_params} parameters")
    results = {}
    for method, eta in [("gd", 0.2), ("adam", 0.1), ("natural", 0.2)]:
        t0 = time.time()
        th, hist = minimize_energy(ansatz, H, theta0, method=method, steps=100, eta=eta)
        results[method] = hist - E0
        print(f"  {method:>8} (eta={eta}): E - E_0 = {hist[0] - E0:.4f} -> {hist[-1] - E0:.2e} after 100 steps"
              f"  ({time.time() - t0:.1f} s)")
    F0 = qfi_matrix(ansatz, np.zeros((1, n)), theta0)
    ev = np.linalg.eigvalsh(F0)
    print(f"  eigenvalues of F at the start: min {ev[0]:.3f}, max {ev[-1]:.3f}")

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    labels = {"gd": "gradient descent", "adam": "Adam", "natural": "natural gradient"}
    for method, hist in results.items():
        ax.semilogy(hist, label=labels[method])
    ax.set_xlabel("iteration"); ax.set_ylabel("$E(\\theta) - E_0$"); ax.legend()
    ax.set_title(f"transverse-field Ising chain, {n} qubits, {L} layers")
    fig.tight_layout()
    bookplot.savefig(fig, 4, "natgrad")
    print("figure natgrad written")
