"""Chapter 7, Sec. 7.8: the adaptive variational quantum eigensolver (ADAPT-VQE).

``AdaptVQE`` grows a ``PauliRotationAnsatz`` one operator at a time.  At every
step it evaluates, for every generator G of a pool, the gradient of the
energy with respect to a new angle inserted at the end of the circuit,
dE/dtheta |_(theta=0) = i <psi| [G, H] |psi>, Eq. (7.adaptgradient), appends
the generator with the largest |gradient|, re-optimises all angles, and stops
when the largest gradient drops below a threshold.

Pools used here (Table 7.adapt); every generator is an imaginary matrix (an odd
number of Y's in each string), so that a real reference state stays real:
* pairing model, L = 4: the six pair-transfer generators of Eq. (4.pairgadget);
* Lipkin model: the 28 one- and two-qubit strings with an odd number of Y's
  (the "qubit pool"), and the collective quasispin generators i(J+ - J-),
  i(J+^2 - J-^2) with and without the third-order i(J+^2 J_z - J_z J-^2);
* Heisenberg ring: the bond spin currents X_iY_j - Y_iX_j and the scalar
  chiralities S_i.(S_j x S_k), from a singlet covering and from the Neel state;
* Hubbard ring: the spin-conserving fermionic singles and doubles (162
  generators) from the weak-coupling reference, twelve operators.

Fig. 7.adapt shows the energy error against the number of operators.  About
one minute (the Hubbard pool dominates).
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np
from pauli import PauliSum
from models import PairingModel
from pauli_circuits import compact_pairing
from qpe_models import book_models, hubbard_ring_spin_major, sector_ground_state, split_identity
from vqe import VQE, PauliRotationAnsatz
from vqe_models import pair_generator, overlap_with_ground
from bookplot import savefig, plt


class AdaptVQE:
    """Grow the ansatz from a pool of Hermitian generators by the largest energy gradient."""

    def __init__(self, H, reference, pool, labels=None):
        self.H, self.Hmat = H, H.matrix()
        self.pool = list(pool)
        self.labels = labels or [str(k) for k in range(len(pool))]
        self.pool_matrices = [g.matrix() for g in self.pool]
        self.ansatz = PauliRotationAnsatz(H.n, reference, [])
        self.theta = np.zeros(0)
        self.chosen = []
        self.energies = []
        self.gradients = []

    def pool_gradients(self):
        """dE/dtheta at theta = 0 for every pool operator appended to the current state."""
        psi = self.ansatz.state(self.theta)
        Hpsi = self.Hmat @ psi
        return np.array([2 * np.imag(np.vdot(Hpsi, Gm @ psi)) for Gm in self.pool_matrices])

    def run(self, max_ops=12, tol=1e-4, verbose=False):
        vqe = VQE(self.H, self.ansatz)
        self.energies = [vqe.exact_energy(self.theta)]
        for step in range(max_ops):
            g = self.pool_gradients()
            k = int(np.argmax(np.abs(g)))
            self.gradients.append(np.abs(g).max())
            if np.abs(g).max() < tol:
                break
            self.ansatz.add(self.pool[k])
            self.chosen.append(self.labels[k])
            self.theta = np.append(self.theta, 0.0)
            vqe = VQE(self.H, self.ansatz)
            self.theta, E = vqe.bfgs(self.theta, tol=1e-13, gtol=1e-9, maxiter=300)
            self.energies.append(E)
            if verbose:
                print(f"    step {step + 1}: added {self.labels[k]:>12s} (|grad| = {np.abs(g).max():.2e}), E = {E:.8f}")
        return self.theta, self.energies[-1]


S = PauliSum.single


def hermitian(A):
    """i (A - A^dagger): the Hermitian generator whose exp(-i theta G) is exp(theta (A - A^dagger))."""
    return (1j * (A - A.dagger())).simplify()


def lipkin_pools(n=4):
    """(a) the qubit pool: every one- and two-qubit Pauli string with an odd number
    of Y's (an imaginary matrix, so that exp(-i theta G) keeps real amplitudes real);
    (b) the collective pool i(J+ - J-), i(J+^2 - J-^2), i(J+^2 J_z - J_z J-^2) built
    from the quasispin operators of the doublet encoding, Eq. (4.lipkinpauli)."""
    qubit, ql = [], []
    for p in range(n):
        qubit.append(S(n, p, "Y")); ql.append(f"Y{p}")
    for p in range(n):
        for q in range(p + 1, n):
            for a, b in (("X", "Y"), ("Y", "X"), ("Y", "Z"), ("Z", "Y")):
                qubit.append(S(n, p, a) * S(n, q, b)); ql.append(f"{a}{p}{b}{q}")
    Jp = sum((0.5 * S(n, p, "X") - 0.5j * S(n, p, "Y") for p in range(n)), PauliSum(n_qubits=n))
    Jm = Jp.dagger()
    Jz = sum((S(n, p, "Z", -0.5) for p in range(n)), PauliSum(n_qubits=n))
    collective = [hermitian(Jp), hermitian(Jp * Jp), hermitian(Jp * Jp * Jz)]
    cl = ["i(J+ - J-)", "i(J+^2 - J-^2)", "i(J+^2 Jz - Jz J-^2)"]
    return (qubit, ql), (collective, cl)


def heisenberg_pool(n=4):
    """Bond spin currents X_iY_j - Y_iX_j and scalar chiralities S_i.(S_j x S_k) of the ring."""
    bonds = [(i, (i + 1) % n) for i in range(n)]
    pool, labels = [], []
    for i, j in bonds:
        pool.append(S(n, i, "X") * S(n, j, "Y") - S(n, i, "Y") * S(n, j, "X")); labels.append(f"current {i}{j}")
    for i in range(n):
        j, k = (i + 1) % n, (i + 2) % n
        chi = PauliSum(n_qubits=n)
        for a, b, c, sgn in (("X", "Y", "Z", 1), ("Y", "Z", "X", 1), ("Z", "X", "Y", 1), ("X", "Z", "Y", -1), ("Z", "Y", "X", -1), ("Y", "X", "Z", -1)):
            chi = chi + sgn * 0.125 * S(n, i, a) * S(n, j, b) * S(n, k, c)
        pool.append(chi.simplify()); labels.append(f"chirality {i}{j}{k}")
    return pool, labels


def hubbard_pool(n_sites=4):
    """Spin-conserving fermionic singles and doubles of the ring in the spin-major
    Jordan-Wigner encoding (the pool of fermionic ADAPT-VQE)."""
    from jordanwigner import JordanWigner
    jw = JordanWigner(2 * n_sites)
    pool, labels = [], []
    for off, s in ((0, "up"), (n_sites, "dn")):
        for p in range(n_sites):
            for q in range(p + 1, n_sites):
                pool.append(hermitian(jw.create(off + q) * jw.annihilate(off + p))); labels.append(f"single {s} {p}->{q}")
    modes = range(2 * n_sites)
    spin = lambda m: m // n_sites
    occ = [(p, q) for p in modes for q in modes if p < q]
    seen = set()
    for p, q in occ:
        for r, t in occ:
            if {p, q} == {r, t} or sorted([spin(p), spin(q)]) != sorted([spin(r), spin(t)]):
                continue
            key = frozenset([(p, q), (r, t)])
            if key in seen:
                continue
            seen.add(key)
            G = hermitian(jw.create(r) * jw.create(t) * jw.annihilate(q) * jw.annihilate(p))
            if G.n_strings():
                pool.append(G); labels.append(f"double {p}{q}->{r}{t}")
    return pool, labels


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    models = {name: (H, E0, sector) for name, H, E0, psi, sector in book_models()}
    results = {}

    print("=== Pairing model, L = 4, two pairs: pool of the six pair transfers")
    H, E0, sector = models["pairing, $L=4$"]
    pool = [pair_generator(4, i, a) for i in range(4) for a in range(i + 1, 4)]
    labels = [f"P{i}{a}" for i in range(4) for a in range(i + 1, 4)]
    adapt = AdaptVQE(H, "1100", pool, labels)
    adapt.run(max_ops=8, tol=1e-6, verbose=True)
    results["pairing, six pair transfers"] = (adapt, E0, sector)

    print("\n=== Lipkin model: the qubit pool and the collective pool")
    H, E0, sector = models["Lipkin, $N=4$"]
    (qubit, ql), (collective, cl) = lipkin_pools()
    for name, pool, labels in (("Lipkin, qubit pool", qubit, ql), ("Lipkin, collective pool (2)", collective[:2], cl[:2]),
                               ("Lipkin, collective pool (3)", collective, cl)):
        print(f"  {name}:")
        adapt = AdaptVQE(H, "0000", pool, labels)
        adapt.run(max_ops=12, tol=1e-6, verbose=True)
        results[name] = (adapt, E0, sector)

    print("\n=== Heisenberg ring: spin currents and scalar chiralities")
    H, E0, sector = models["Heisenberg ring"]
    pool, labels = heisenberg_pool()
    singlet = np.array([0, 1, -1, 0]) / np.sqrt(2)
    for name, ref in (("Heisenberg, from singlets", np.kron(singlet, singlet)), ("Heisenberg, from Neel", "0101")):
        print(f"  {name}:")
        adapt = AdaptVQE(H, ref, pool, labels)
        adapt.run(max_ops=8, tol=1e-6, verbose=True)
        results[name] = (adapt, E0, sector)

    print("\n=== Hubbard ring: fermionic singles and doubles from the weak-coupling reference (12 operators)")
    H, E0, sector = models["Hubbard ring"]
    _, H0 = hubbard_ring_spin_major(U=0.01)
    _, ref, _ = sector_ground_state(H0, 4)
    pool, labels = hubbard_pool()
    print(f"  pool of {len(pool)} generators")
    adapt = AdaptVQE(H, ref, pool, labels)
    adapt.run(max_ops=12, tol=1e-6, verbose=True)
    results["Hubbard, fermionic pool"] = (adapt, E0, sector)

    print("\n=== Summary (Table 7.adapt)")
    print(f"{'model / pool':>30s} {'pool':>5s} {'ops':>4s} {'E_ADAPT':>12s} {'error':>9s} {'|<E0|psi>|^2':>12s}")
    for name, (adapt, E0, sector) in results.items():
        ov = overlap_with_ground(adapt.ansatz.state(adapt.theta), adapt.H, sector)
        print(f"{name:>30s} {len(adapt.pool):5d} {len(adapt.chosen):4d} {adapt.energies[-1]:12.8f} {adapt.energies[-1] - E0:9.1e} {ov:12.6f}")
    assert results["pairing, six pair transfers"][0].energies[-1] - 0.63554847 < 1e-7 and len(results["pairing, six pair transfers"][0].chosen) == 5
    assert results["Lipkin, qubit pool"][0].energies[-1] + 4.21287670 < 1e-7
    assert results["Lipkin, collective pool (2)"][0].energies[-1] + 4.21287670 > 1e-4          # the gradient trap
    assert results["Lipkin, collective pool (3)"][0].energies[-1] + 4.21287670 < 1e-7
    assert len(results["Heisenberg, from singlets"][0].chosen) == 1 and results["Heisenberg, from singlets"][0].energies[-1] + 2 < 1e-9
    assert results["Hubbard, fermionic pool"][0].energies[-1] + 2.10274848 < 2e-2
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for name, (adapt, E0, sector) in results.items():
        err = np.maximum(np.array(adapt.energies) - E0, 1e-13)
        ax.semilogy(np.arange(len(err)), err, "o-", ms=4, label=name)
    ax.set_xlabel("operators in the ansatz"); ax.set_ylabel(r"$E - E_0$"); ax.legend(fontsize=7)
    savefig(fig, 7, "adapt")
    print("\nADAPT-VQE checks passed")
