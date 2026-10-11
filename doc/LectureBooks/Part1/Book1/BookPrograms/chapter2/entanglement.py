"""Chapter 2, Sec. 2.6: entanglement, Bell states, Schmidt decomposition, entropy.

Builds the four Bell states (Eq. 2.bell) and checks that they form an
orthonormal basis; measures qubit A of |Phi+> and shows that the outcome
fixes qubit B (Eq. 2.bellcollapse); computes the reduced density matrices
by partial trace (Eq. 2.partialtrace) and their von Neumann entropy
(Eq. 2.vonneumann); performs the Schmidt decomposition of an arbitrary
two-qubit state through the singular value decomposition (Eq. 2.schmidt);
and produces Fig. 2.entropy: the binary Shannon entropy H_2(p) and the
entanglement entropy of cos(theta)|00> + sin(theta)|11>.  Runs in a second.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np
import gates as G
from density_matrix import DensityMatrix, shannon_entropy
from statevector import StateVector
from bookplot import savefig, plt

BELL = {
    "Phi+": np.array([1, 0, 0, 1]) / np.sqrt(2),
    "Phi-": np.array([1, 0, 0, -1]) / np.sqrt(2),
    "Psi+": np.array([0, 1, 1, 0]) / np.sqrt(2),
    "Psi-": np.array([0, 1, -1, 0]) / np.sqrt(2),
}


def schmidt_decomposition(psi, dA=2, dB=2):
    """Schmidt coefficients and bases of a pure bipartite state, via the SVD.

    psi = sum_{ij} c_{ij} |i>_A |j>_B  ->  C = U Sigma V^dagger,
    psi = sum_k sigma_k |u_k>_A |v_k>_B  with the components of |u_k> in
    column k of U and those of |v_k> in row k of V^dagger."""
    C = np.asarray(psi, dtype=complex).reshape(dA, dB)
    U, sigma, Vh = np.linalg.svd(C)
    return sigma, U, Vh


def schmidt_rank(psi, tol=1e-12):
    return int(np.sum(schmidt_decomposition(psi)[0] > tol))


def is_entangled(psi):
    return schmidt_rank(psi) > 1


def entanglement_entropy(psi, keep=0):
    """S(rho_A) for a pure two-qubit state."""
    return DensityMatrix.from_state(psi).partial_trace(keep).von_neumann_entropy()


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    # --- orthonormality of the Bell basis
    Bmat = np.array(list(BELL.values()))
    assert np.allclose(Bmat.conj() @ Bmat.T, np.eye(4))
    print("Bell states form an orthonormal basis: Gram matrix =\n", (Bmat.conj() @ Bmat.T).real)
    # --- each Bell state from the circuit H, CNOT (+ X, Z), Sec. 2.9
    # --- measuring qubit A of |Phi+>, Eq. (2.bellcollapse)
    sv = StateVector(2, BELL["Phi+"])
    for outcome in (0, 1):
        post = sv.copy(); _, p = post.measure(0, outcome=outcome)
        print(f"qubit A = {outcome} with probability {p:.2f}; state afterwards {post}")
    # --- reduced density matrices and entropies
    print(f"\n{'state':6s} {'rho_A':>26s} {'purity':>8s} {'S(rho_A)':>9s} {'Schmidt coeffs':>18s}")
    for name, psi in BELL.items():
        rhoA = DensityMatrix.from_state(psi).partial_trace(0)
        sig = schmidt_decomposition(psi)[0]
        print(f"{name:6s} {str(np.round(rhoA.rho.real, 3).tolist()):>26s} {rhoA.purity():8.3f} {rhoA.von_neumann_entropy():9.3f} {str(np.round(sig,4).tolist()):>18s}")
        assert np.allclose(rhoA.rho, np.eye(2) / 2)
    # --- a product state has Schmidt rank one and zero entropy
    prod = np.kron([np.cos(0.3), np.sin(0.3)], [1 / np.sqrt(2), 1j / np.sqrt(2)])
    print("\nproduct state: Schmidt rank", schmidt_rank(prod), ", S(rho_A) =", round(entanglement_entropy(prod), 6))
    assert not is_entangled(prod) and is_entangled(BELL["Phi+"])
    # --- Schmidt decomposition of a random state reproduces the state
    rng = np.random.default_rng(7)
    psi = rng.normal(size=4) + 1j * rng.normal(size=4); psi /= np.linalg.norm(psi)
    sigma, U, V = schmidt_decomposition(psi)
    rebuilt = sum(sigma[k] * np.kron(U[:, k], V[k, :]) for k in range(2))
    assert np.allclose(rebuilt, psi) and abs(np.sum(sigma ** 2) - 1) < 1e-12
    # the squared Schmidt coefficients are the eigenvalues of rho_A
    rhoA = DensityMatrix.from_state(psi).partial_trace(0)
    assert np.allclose(np.sort(rhoA.eigenvalues()), np.sort(sigma ** 2))
    print("random state: Schmidt coefficients", sigma, "-> S(rho_A) =", round(shannon_entropy(sigma ** 2), 4), "bit")
    # --- the family cos(theta)|00> + sin(theta)|11>, Table 2.enttheta
    print(f"\n{'theta/pi':>8s} {'S(rho_A)':>9s}")
    thetas = np.linspace(0, np.pi / 2, 11)
    for th in thetas:
        print(f"{th/np.pi:8.2f} {entanglement_entropy([np.cos(th), 0, 0, np.sin(th)]):9.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.1))
    p = np.linspace(0, 1, 201)
    axes[0].plot(p, [shannon_entropy([q, 1 - q]) for q in p])
    axes[0].set_xlabel("$p$"); axes[0].set_ylabel("$H_2(p)$ [bit]"); axes[0].set_title("binary Shannon entropy", fontsize=9)
    th = np.linspace(0, np.pi / 2, 201)
    axes[1].plot(th / np.pi, [entanglement_entropy([np.cos(t), 0, 0, np.sin(t)]) for t in th])
    axes[1].set_xlabel(r"$\theta/\pi$"); axes[1].set_ylabel(r"$S(\rho_A)$ [bit]")
    axes[1].set_title(r"entanglement entropy of $\cos\theta|00\rangle+\sin\theta|11\rangle$", fontsize=9)
    print("figure written to", savefig(fig, 2, "entropy"))
