"""Chapter 2, Secs. 2.7-2.8: quantum feature maps and the quantum kernel.

A quantum feature map is a data-dependent unitary U(x) acting on
|0...0>; the feature vector is the state |phi(x)> = U(x)|0...0> of
Eq. (2-qfeaturemap) and the quantum kernel is the fidelity
K(x, x') = |<phi(x)|phi(x')>|^2 of Eq. (2-qkernel).  The classes:

    AngleEncoding     one rotation per feature, Eq. (2-angle), optionally
                      repeated and entangled (then the 'Pauli' map of
                      Havlicek et al. with first-order terms only)
    ZZFeatureMap      the second-order Pauli-Z map of Havlicek et al.,
                      Eq. (2-zzmap), with phi_ij = (pi - x_i)(pi - x_j)
    AmplitudeEncoding the normalized data vector as the amplitudes,
                      Eq. (2-amplitude)
    TrainableEmbedding an angle encoding followed by a trainable layer,
                      used by kernel_alignment.py
    QuantumKernel     evaluates the kernel exactly from the state vectors,
                      or estimates it with a finite number of shots by the
                      adjoint (inversion) test of Fig. 2-adjointtest or
                      the SWAP test of Fig. 2-swaptest

Everything runs on the state-vector simulator of common/statevector.py.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]      # BookPrograms/
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np

import gates as G
from statevector import StateVector


# ----------------------------------------------------------------- maps
class FeatureMap:
    """Base class: `prepare(x)` applies U(x) to a fresh register and returns
    the StateVector; `state(x)` returns its amplitudes."""
    n_qubits = 1
    name = "feature map"

    def prepare(self, x, sv=None):
        raise NotImplementedError

    def state(self, x):
        return self.prepare(np.asarray(x, float)).psi

    def states(self, X):
        """The matrix whose rows are the feature vectors of all samples."""
        return np.array([self.state(x) for x in np.asarray(X, float)])

    def __repr__(self):
        return self.name


class AngleEncoding(FeatureMap):
    """|phi(x)> = prod_layers [entangler] prod_i R_a(x_i)|0>: one rotation
    about the axis `rotation` ('X', 'Y' or 'Z', preceded by Hadamards for
    'Z') per feature.  With reps=1 and no entangler this is a product
    state and the kernel factorizes, Eq. (2-anglekernel)."""

    def __init__(self, n_features, rotation="Y", reps=1, entangle=False, scale=1.0):
        self.n_qubits = n_features
        self.rotation, self.reps, self.entangle, self.scale = rotation, reps, entangle, scale
        self.name = f"angle({rotation},reps={reps}{',ent' if entangle else ''})"

    def _rotate(self, sv, x):
        for q, xq in enumerate(x):
            if self.rotation == "Y":
                sv.ry(self.scale * xq, q)
            elif self.rotation == "X":
                sv.rx(self.scale * xq, q)
            else:
                sv.h(q); sv.rz(self.scale * xq, q)

    def prepare(self, x, sv=None):
        sv = StateVector(self.n_qubits) if sv is None else sv
        for _ in range(self.reps):
            self._rotate(sv, x)
            if self.entangle:
                for q in range(self.n_qubits - 1):
                    sv.cnot(q, q + 1)
        return sv


class ZZFeatureMap(FeatureMap):
    """The second-order Pauli-Z feature map of Havlicek et al. (2019):

        U(x) = [ U_Phi(x) H^{(x) n} ]^reps,
        U_Phi(x) = exp( i sum_i phi_i(x) Z_i + i sum_{i<j} phi_ij(x) Z_i Z_j ),

    with phi_i = x_i and phi_ij = (pi - x_i)(pi - x_j), Eq. (2-zzmap).  The
    ZZ rotation is realised as CNOT - R_z - CNOT (Fig. 2-zzcircuit); the
    pairs are the nearest neighbours ('linear') or all pairs ('full')."""

    def __init__(self, n_features, reps=2, entanglement="linear", alpha=1.0):
        self.n_qubits = n_features
        self.reps, self.entanglement, self.alpha = reps, entanglement, alpha
        n = n_features
        if entanglement == "linear":
            self.pairs = [(q, q + 1) for q in range(n - 1)]
        elif entanglement == "circular":
            self.pairs = [(q, (q + 1) % n) for q in range(n)] if n > 2 else [(0, 1)]
        else:
            self.pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        self.name = f"zz(reps={reps},{entanglement})"

    @staticmethod
    def phi1(xi):
        return xi

    @staticmethod
    def phi2(xi, xj):
        return (np.pi - xi) * (np.pi - xj)

    def prepare(self, x, sv=None):
        sv = StateVector(self.n_qubits) if sv is None else sv
        x = self.alpha * np.asarray(x, float)
        for _ in range(self.reps):
            for q in range(self.n_qubits):
                sv.h(q)
            # exp(i phi Z) = R_z(-2 phi) up to a global phase
            for q in range(self.n_qubits):
                sv.rz(-2 * self.phi1(x[q]), q)
            for (i, j) in self.pairs:
                # exp(i phi Z_i Z_j) = CNOT_ij R_z(-2 phi)_j CNOT_ij
                sv.cnot(i, j); sv.rz(-2 * self.phi2(x[i], x[j]), j); sv.cnot(i, j)
        return sv


class AmplitudeEncoding(FeatureMap):
    """|phi(x)> = x / ||x||: the 2^n amplitudes *are* the (padded) data,
    Eq. (2-amplitude); the kernel is then (x^T x')^2 / (||x||^2 ||x'||^2)."""

    def __init__(self, n_features):
        self.n_qubits = int(np.ceil(np.log2(max(n_features, 2))))
        self.n_features = n_features
        self.name = "amplitude"

    def prepare(self, x, sv=None):
        amp = np.zeros(2 ** self.n_qubits)
        amp[:len(x)] = x
        return StateVector(self.n_qubits, amp)       # normalized by StateVector


class TrainableEmbedding(FeatureMap):
    """A data re-uploading embedding with trainable parameters, Eq. (2-trainable):
    `reps` repetitions of [R_y(w_q x_q) on every qubit; then `layers` blocks
    of R_z(theta) R_y(theta') on every qubit followed by a CNOT ring].  The
    parameter vector theta holds, repetition by repetition, the input
    weights w (initialised to 1, they set the bandwidth of the kernel) and
    the rotation angles.  `prepare(x, theta, shift=(k, s))` adds s to the
    angle of the gate controlled by theta_k, which is what the
    parameter-shift rule of kernel_alignment.py needs; `chain(X, k)` is the
    factor d(angle)/d(theta_k) per sample (x_q for a weight, 1 otherwise)."""

    def __init__(self, n_features, layers=1, reps=2, theta=None, rng=0):
        self.n_qubits = n_features
        self.layers, self.reps = layers, reps
        n = n_features
        self.per_rep = n + 2 * n * layers
        self.n_params = reps * self.per_rep
        rng = np.random.default_rng(rng)
        if theta is None:
            theta = np.concatenate([np.r_[np.ones(n), rng.uniform(0, 2 * np.pi, 2 * n * layers)]
                                    for _ in range(reps)])
        self.theta = np.asarray(theta, float)
        self.name = f"trainable(layers={layers},reps={reps})"

    def prepare(self, x, sv=None, theta=None, shift=None):
        theta = self.theta if theta is None else theta
        k_shift, s = shift if shift is not None else (-1, 0.0)
        sv = StateVector(self.n_qubits) if sv is None else sv
        n, k = self.n_qubits, 0
        for _ in range(self.reps):
            for q in range(n):                                  # data layer
                sv.ry(theta[k] * x[q] + (s if k == k_shift else 0.0), q); k += 1
            for _ in range(self.layers):                        # trainable layers
                for q in range(n):
                    sv.rz(theta[k] + (s if k == k_shift else 0.0), q); k += 1
                for q in range(n):
                    sv.ry(theta[k] + (s if k == k_shift else 0.0), q); k += 1
                for q in range(n):
                    if n > 1:
                        sv.cnot(q, (q + 1) % n)
        return sv

    def states(self, X, theta=None, shift=None):
        return np.array([self.prepare(x, theta=theta, shift=shift).psi for x in np.asarray(X, float)])

    def chain(self, X, k):
        """d(gate angle)/d(theta_k) for every sample: x_q for an input
        weight of qubit q, 1 for a rotation angle."""
        X = np.asarray(X, float)
        pos = k % self.per_rep
        return X[:, pos] if pos < self.n_qubits else np.ones(len(X))


# --------------------------------------------------------------- kernel
class QuantumKernel:
    """K(x, x') = |<phi(x)|phi(x')>|^2 for a feature map.

    shots=None evaluates the fidelity exactly from the two state vectors.
    With a finite number of shots the kernel entry is *estimated* the way a
    quantum computer would: method='adjoint' prepares U(x')^dagger U(x)|0>
    and counts how often all qubits read 0 (Fig. 2-adjointtest, probability
    = fidelity); method='swap' runs the SWAP test with an ancilla
    (Fig. 2-swaptest, P(ancilla = 0) = (1 + fidelity)/2).  Both are sampled
    from the exact outcome probabilities, which is what any simulator does."""

    def __init__(self, feature_map, shots=None, method="adjoint", rng=0):
        self.feature_map, self.shots, self.method = feature_map, shots, method
        self.rng = np.random.default_rng(rng)
        self.name = repr(feature_map) + ("" if shots is None else f"[{shots} shots,{method}]")

    # exact fidelities between two sets of samples
    def fidelities(self, X, Y=None):
        Phi_X = self.feature_map.states(X)
        Phi_Y = Phi_X if Y is None else self.feature_map.states(Y)
        return np.abs(Phi_X.conj() @ Phi_Y.T) ** 2

    def __call__(self, X, Y=None):
        F = self.fidelities(X, Y)
        if self.shots is None:
            return F
        if self.method == "adjoint":
            # number of all-zero outcomes in `shots` repetitions
            return self.rng.binomial(self.shots, np.clip(F, 0, 1)) / self.shots
        p0 = (1 + np.clip(F, 0, 1)) / 2
        return 2 * self.rng.binomial(self.shots, p0) / self.shots - 1

    def gram(self, X):
        return self(X, X)

    # the two estimation circuits, written out on the simulator
    def adjoint_circuit(self, x, xp):
        """U(x')^dagger U(x)|0>: the register of the adjoint test."""
        fm = self.feature_map
        sv = fm.prepare(np.asarray(x, float))
        # U(x')^dagger = the inverse of the circuit for x'; obtain it from the
        # matrix of U(x') on the full register (fine for a few qubits)
        dim = 2 ** fm.n_qubits
        U = np.column_stack([fm.prepare(np.asarray(xp, float),
                                        StateVector(fm.n_qubits, np.eye(dim)[:, k])).psi
                             for k in range(dim)])
        sv.psi = U.conj().T @ sv.psi
        return sv

    def swap_test_circuit(self, x, xp):
        """Ancilla + |phi(x)> + |phi(x')>, controlled swaps, Hadamards:
        the SWAP test register (ancilla = qubit 0)."""
        fm = self.feature_map
        n = fm.n_qubits
        psi = np.kron(np.array([1, 0], complex),
                      np.kron(fm.state(x), fm.state(xp)))
        sv = StateVector(2 * n + 1, psi)
        sv.h(0)
        P0, P1 = np.diag([1, 0]).astype(complex), np.diag([0, 1]).astype(complex)
        cswap = np.kron(P0, np.eye(4)) + np.kron(P1, G.SWAP)      # Fredkin gate
        for q in range(n):
            sv.apply(cswap, [0, 1 + q, 1 + n + q])
        sv.h(0)
        return sv

    def __repr__(self):
        return self.name


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    x, xp = rng.uniform(0, np.pi, 2), rng.uniform(0, np.pi, 2)

    # 1. angle encoding: the kernel factorizes, Eq. (2-anglekernel)
    fm = AngleEncoding(2, "Y")
    k = QuantumKernel(fm)
    exact = np.prod(np.cos((x - xp) / 2) ** 2)
    assert np.isclose(k([x], [xp])[0, 0], exact)
    print(f"angle encoding: K = {exact:.6f} = prod_i cos^2((x_i - x'_i)/2)")

    # 2. amplitude encoding: K = (x.x')^2/(|x|^2 |x'|^2)
    fa = AmplitudeEncoding(3)
    u, v = rng.normal(size=3), rng.normal(size=3)
    assert np.isclose(QuantumKernel(fa)([u], [v])[0, 0], (u @ v) ** 2 / (u @ u) / (v @ v))
    print("amplitude encoding: K = (x.x')^2 / (|x|^2 |x'|^2)")

    # 3. the ZZ map against an explicit matrix exponential
    zz = ZZFeatureMap(2, reps=1)
    Z = G.Z; I = G.I
    H2 = np.kron(G.H, G.H)
    gen = (zz.phi1(x[0]) * np.kron(Z, I) + zz.phi1(x[1]) * np.kron(I, Z)
           + zz.phi2(x[0], x[1]) * np.kron(Z, Z))
    # the generator is diagonal, so exp(i gen) is the exponential of the diagonal
    U = np.diag(np.exp(1j * np.diag(gen))) @ H2
    psi_ref = U @ np.array([1, 0, 0, 0], complex)
    psi = zz.state(x)
    assert np.isclose(abs(np.vdot(psi_ref, psi)), 1.0)
    print("ZZ feature map agrees with exp(i sum phi Z) H^{(x)2} up to a global phase")

    # 4. the two estimation circuits reproduce the exact fidelity
    kz = QuantumKernel(zz)
    F = kz([x], [xp])[0, 0]
    adj = kz.adjoint_circuit(x, xp)
    assert np.isclose(adj.probabilities()[0], F)
    sw = kz.swap_test_circuit(x, xp)
    assert np.isclose(sw.marginal(0)[0], (1 + F) / 2)
    print(f"fidelity {F:.6f}: adjoint test P(00) = {adj.probabilities()[0]:.6f}, "
          f"SWAP test P(anc=0) = {sw.marginal(0)[0]:.6f} = (1+F)/2")
    counts = adj.sample(1000, np.random.default_rng(3))
    print("1000 shots of the adjoint test:", counts, "-> K ~", counts.get("00", 0) / 1000)

    # 5. the Gram matrix is symmetric, positive semidefinite, with unit diagonal
    X = rng.uniform(0, np.pi, size=(12, 2))
    K = kz.gram(X)
    assert np.allclose(K, K.T) and np.allclose(np.diag(K), 1) and np.linalg.eigvalsh(K).min() > -1e-10
    print("Gram matrix of the ZZ kernel: symmetric, unit diagonal, PSD; quantum-kernel checks passed")
