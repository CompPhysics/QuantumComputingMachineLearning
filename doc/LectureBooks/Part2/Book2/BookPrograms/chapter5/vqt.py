"""Chapter 5, Sec. 5.6: variational preparation of a Gibbs state (the
variational quantum thermalizer of Verdon et al.).

The ansatz is a classical latent distribution over the computational basis
-- a product of Bernoulli factors p(i; w) = prod_q s_q^{i_q} (1 - s_q)^{1 - i_q}
with s_q = sigma(w_q) (latent="product", n parameters) or a general
distribution p = softmax(w) over the 2^n basis states (latent="full") --
followed by the unitary U(phi) of the QNN of chapter 4 (no data encoding),

    rho(w, phi) = sum_i p(i; w) U(phi) |i><i| U(phi)^dagger,   Eq. (5-vqtansatz),

and the objective is the free energy F = sum_i p_i E_i(phi) - S(w)/beta with
E_i = <i|U^dagger H U|i> and S(w) the entropy of p, Eq. (5-vqtfree).  The
gradient with respect to w is analytic and the one with respect to phi is
the parameter-shift rule applied to the energies E_i.  The Gibbs state of
H is the unique minimiser of F (Sec. 5.6), so F - F_exact and the fidelity
with the exact Gibbs state measure the quality of the preparation.

    VQT(H, n_qubits, n_layers, beta, latent="product"|"full")
        .free_energy(params), .gradient(params), .state(params), .fit(...)

Target: the transverse-field Ising chain on 3 qubits at beta = 1 and 3.
Produces Fig. 5-vqt and the numbers of the text.  Runtime about 30 s.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]      # BookPrograms/
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np

from qnn import QNN
from gibbs import gibbs_state, fidelity, von_neumann_entropy, log_partition, pauli_string
from rbm import patterns, sigmoid
from optimize import Adam


class VQT:
    def __init__(self, H, n_qubits, n_layers, beta=1.0, rotations=("Y", "Z"), latent="product"):
        self.H, self.n, self.beta = H, n_qubits, beta
        self.qnn = QNN(n_qubits, n_layers, reupload=False, rotations=rotations)
        self.basis = patterns(n_qubits)                  # (2^n, n) bit strings
        self.angles = np.pi * self.basis                 # R_y(pi)|0> = |1>: encodes |i>
        self.latent_type = latent
        self.n_latent = n_qubits if latent == "product" else 2 ** n_qubits
        self.n_params = self.n_latent + self.qnn.n_params

    def split(self, params):
        return params[:self.n_latent], params[self.n_latent:]

    def latent(self, w):
        """The probabilities p_i and the entropy S of the latent distribution."""
        if self.latent_type == "product":
            s = sigmoid(w)
            p = np.prod(np.where(self.basis == 1, s, 1 - s), axis=1)
            entropy = -np.sum(s * np.log(s) + (1 - s) * np.log(1 - s))
        else:
            p = np.exp(w - w.max()); p /= p.sum()
            entropy = -np.sum(p * np.log(np.maximum(p, 1e-300)))
        return p, entropy

    def energies(self, phi):
        """E_i(phi) = <i|U^dagger H U|i> for all basis states; phi may be (S, P)."""
        psi = self.qnn.state(self.angles, phi)           # (2^n, dim) or (S, 2^n, dim)
        return np.einsum("...ia,ab,...ib->...i", psi.conj(), self.H, psi).real

    def free_energy(self, params):
        w, phi = self.split(params)
        p, S = self.latent(w)
        return float(p @ self.energies(phi) - S / self.beta)

    def gradient(self, params):
        w, phi = self.split(params)
        p, S = self.latent(w)
        E = self.energies(phi)
        if self.latent_type == "product":
            s = sigmoid(w)
            # dp_i/dw_q = p_i (i_q - s_q); dS/dw_q = -w_q s_q (1 - s_q)
            g_w = (p * E) @ (self.basis - s) + w * s * (1 - s) / self.beta
        else:
            # softmax: dp_j/dw_i = p_i (delta_ij - p_j); dS/dw_i = -p_i (ln p_i + S)
            g_w = p * (E - p @ E) + p * (np.log(np.maximum(p, 1e-300)) + S) / self.beta
        # d/dphi: parameter shift on every E_i, all shifted parameter sets in one call
        P = self.qnn.n_params
        T = np.repeat(phi[None, :], 2 * P, axis=0)
        T[np.arange(P), np.arange(P)] += np.pi / 2
        T[P + np.arange(P), np.arange(P)] -= np.pi / 2
        Es = self.energies(T)                            # (2P, 2^n)
        dE = 0.5 * (Es[:P] - Es[P:])                     # (P, 2^n)
        g_phi = dE @ p
        return np.concatenate([g_w, g_phi])

    def state(self, params):
        w, phi = self.split(params)
        p, _ = self.latent(w)
        psi = self.qnn.state(self.angles, phi)
        return np.einsum("i,ia,ib->ab", p, psi, psi.conj())

    def fit(self, params0, steps=300, lr=0.05, record=10):
        params = np.array(params0, float)
        opt = Adam(lr=lr)
        hist = []
        for step in range(steps):
            if step % record == 0:
                hist.append(self.free_energy(params))
            params = opt.step(params, self.gradient(params))
        hist.append(self.free_energy(params))
        return params, np.array(hist)


def transverse_ising(n, h=1.0):
    return -sum(pauli_string({q: "Z", q + 1: "Z"}, n) for q in range(n - 1)) \
        - h * sum(pauli_string({q: "X"}, n) for q in range(n))


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    import bookplot
    from optimize import numerical_gradient

    n = 3
    H = transverse_ising(n, 1.0)
    rng = np.random.default_rng(0)
    v = VQT(H, n, 2, beta=1.0)
    x0 = np.concatenate([rng.normal(0, 0.5, n), v.qnn.random_parameters(1)])
    g = v.gradient(x0); gn = numerical_gradient(v.free_energy, x0)
    print(f"free-energy gradient against finite differences: max |diff| = {np.max(np.abs(g - gn)):.1e}")
    assert np.max(np.abs(g - gn)) < 1e-6

    print(f"\nTable 5-vqt: variational Gibbs states of the transverse-field Ising chain, n = {n}, h = 1 (300 Adam steps, lr 0.05)")
    print(f"{'beta':>5}  {'latent':>7}  {'L':>2}  {'P':>3}  {'F_exact':>9}  {'F_vqt':>9}  {'F - F_exact':>11}  {'fidelity':>9}  {'S_vqt':>6}  {'S_exact':>7}  {'time':>5}")
    curves = {}
    for beta in (1.0, 3.0):
        rho_ex = gibbs_state(H, beta)
        F_ex = -log_partition(H, beta) / beta
        for latent, L in [("product", 1), ("product", 2), ("product", 3), ("full", 1), ("full", 2), ("full", 3)]:
            t0 = time.time()
            vq = VQT(H, n, L, beta=beta, latent=latent)
            best = None
            for seed in range(2):
                x0 = np.concatenate([np.zeros(vq.n_latent), vq.qnn.random_parameters(seed)])
                params, hist = vq.fit(x0)
                if best is None or hist[-1] < best[1][-1]:
                    best = (params, hist)
            params, hist = best
            rho = vq.state(params)
            S = vq.latent(vq.split(params)[0])[1]
            curves[(beta, latent, L)] = hist - F_ex
            print(f"{beta:>5.1f}  {latent:>7}  {L:>2}  {vq.n_params:>3}  {F_ex:>9.5f}  {hist[-1]:>9.5f}  {hist[-1] - F_ex:>11.2e}  "
                  f"{fidelity(rho_ex, rho):>9.6f}  {S:>6.3f}  {von_neumann_entropy(rho_ex):>7.3f}  {time.time() - t0:>4.1f}s")

    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    x = np.arange(0, 300, 10).tolist() + [300]
    for (beta, latent, L), h in curves.items():
        if beta == 1.0:
            ax.semilogy(x, np.maximum(h, 1e-12), "-" if latent == "full" else "--", color=f"C{L - 1}",
                        label=f"{latent} latent, $L={L}$")
    ax.set_xlabel("Adam step"); ax.set_ylabel("$F(w,\\phi) - F_{\\mathrm{exact}}$"); ax.legend(fontsize=7, ncol=2)
    ax.set_title("variational Gibbs state, transverse-field Ising chain, 3 qubits, $\\beta=1$", fontsize=9)
    fig.tight_layout()
    bookplot.savefig(fig, 5, "vqt")
    print("figure vqt written")
