"""Chapter 8, Sec. 8.2: resolution refinement.

An eigenstate of a coarse (low-resolution) Hamiltonian is lifted into a finer
model space by a prolongation operator P, Eq. (8.prolongation), and followed
adiabatically along the interpolated, shifted Hamiltonian of Eq. (8.path),

    H(s) = cos^2(pi s/2) P (H_low - mu) P^dagger + sin^2(pi s/2) (H_high - mu),

to the eigenstate of the fine Hamiltonian.

``Prolongation``          the isometry P between two number sectors:
                          ``embedding`` (new single-particle levels left empty,
                          the basis refinement of the pairing model) and
                          ``orbital`` (every coarse orbital spread over fine
                          orbitals, the lattice refinement), Eq. (8.prolongdet)
``ResolutionRefinement``  the path Hamiltonian, its instantaneous spectrum and
                          minimum gap, the adiabatic evolution by exact slices or
                          by a user-supplied product formula, overlap and energy
                          of the refined state, and the time T* to reach a target
                          overlap.

As a script: the pairing model with k = 2 -> 4 levels (N = 2 and N = 4), the
four-particle refinement to k = 3 and k = 4, the spectrum along the path
(Figs. 8.refine and 8.fourparticle), and the comparison with a plain linear
ramp without the shift, and the long-time tail.  About 15 s.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import itertools
import numpy as np
from fockspace import SlaterBasis
from models import PairingModel
from bookplot import savefig, plt


# ------------------------------------------------------------ prolongation
class Prolongation:
    """The isometry P: coarse N-particle sector -> fine N-particle sector."""

    def __init__(self, matrix):
        self.matrix = np.asarray(matrix)
        d = self.matrix.shape[1]
        assert np.allclose(self.matrix.conj().T @ self.matrix, np.eye(d)), "P is not an isometry"

    @classmethod
    def embedding(cls, basis_low, basis_high, mode_map=None):
        """Coarse mode p becomes fine mode mode_map[p] (default: the same label);
        all other fine modes stay empty.  No gates in the Jordan-Wigner ordering."""
        mode_map = list(range(basis_low.n)) if mode_map is None else mode_map
        W = np.zeros((basis_high.n, basis_low.n))
        for p, f in enumerate(mode_map):
            W[f, p] = 1.0
        return cls.orbital(basis_low, basis_high, W)

    @classmethod
    def orbital(cls, basis_low, basis_high, W):
        """a_c^dagger -> sum_f W[f, c] a_f^dagger with orthonormal columns.  A coarse
        determinant |c_1 < ... < c_N> becomes sum_F det(W[F, C]) |F>, Eq. (8.prolongdet)."""
        W = np.asarray(W)
        P = np.zeros((basis_high.dim, basis_low.dim), dtype=W.dtype)
        occ_high = [basis_high.occupied(s) for s in basis_high.states]
        for b, state in enumerate(basis_low.states):
            C = basis_low.occupied(state)
            for a, F in enumerate(occ_high):
                P[a, b] = np.linalg.det(W[np.ix_(F, C)])
        return cls(P)

    def __call__(self, psi_low):
        return self.matrix @ psi_low


# -------------------------------------------------------------- refinement
class ResolutionRefinement:
    """Adiabatic refinement of P|psi_low> to the ground state of H_high."""

    def __init__(self, H_low, H_high, P, psi_low=None, E_low=None, buffer=0.6, match_energy=False):
        self.H_low, self.H_high = np.atleast_2d(H_low), np.asarray(H_high)
        self.P = P.matrix if isinstance(P, Prolongation) else np.asarray(P)
        w, V = np.linalg.eigh(self.H_low)
        self.psi_low = V[:, 0] if psi_low is None else np.asarray(psi_low)
        # a device knows only the energy of the state it prepared
        self.E_low = float(np.real(np.vdot(self.psi_low, self.H_low @ self.psi_low))) if E_low is None else E_low
        self.mu = self.E_low + buffer
        d_low, d_high = self.H_low.shape[0], self.H_high.shape[0]
        self.phi0 = self.P @ self.psi_low
        # For an embedding <Phi_0|H_high|Phi_0> = E_low and one shift serves both terms.  A
        # lattice prolongation changes the energy scale; the fine Hamiltonian is then shifted
        # by the energy of the prolonged state instead (match_energy=True), Eq. (8.pathlattice).
        self.mu_high = self.mu
        if match_energy:
            self.mu_high = float(np.real(np.vdot(self.phi0, self.H_high @ self.phi0))) + buffer
        self.A = self.P @ (self.H_low - self.mu * np.eye(d_low)) @ self.P.conj().T
        self.B = self.H_high - self.mu_high * np.eye(d_high)
        wh, Vh = np.linalg.eigh(self.H_high)
        self.E_high, self.psi_high, self.spectrum_high = wh[0], Vh[:, 0], wh

    # ---- the path
    def path_hamiltonian(self, s):
        c, si = np.cos(0.5 * np.pi * s) ** 2, np.sin(0.5 * np.pi * s) ** 2
        return c * self.A + si * self.B

    def spectrum(self, s_values, n_levels=None):
        return np.array([np.linalg.eigvalsh(self.path_hamiltonian(s))[:n_levels] for s in s_values])

    def min_gap(self, n_s=201):
        """Smallest gap between the tracked (lowest) level and the next one along the path."""
        w = self.spectrum(np.linspace(0, 1, n_s), 2)
        return float(np.min(w[:, 1] - w[:, 0]))

    # ---- evolution
    def evolve(self, T, n_steps=160, stepper=None):
        """Evolve P|psi_low> from s = 0 to s = 1 in total time T with n_steps midpoint
        slices.  ``stepper(s, dt, psi)`` replaces the exact slice by a product formula."""
        psi = self.phi0.astype(complex)
        dt = T / n_steps
        for j in range(n_steps):
            s = (j + 0.5) / n_steps
            if stepper is None:
                w, V = np.linalg.eigh(self.path_hamiltonian(s))
                psi = V @ (np.exp(-1j * w * dt) * (V.conj().T @ psi))
            else:
                psi = stepper(s, dt, psi)
        return psi / np.linalg.norm(psi)

    def overlap(self, psi):
        return float(abs(np.vdot(self.psi_high, psi)) ** 2)

    def energy(self, psi):
        return float(np.real(np.vdot(psi, self.H_high @ psi)))

    def scan(self, Ts, n_steps=160):
        """Overlap and energy <H_high> of the refined state for every total time T."""
        out = []
        for T in Ts:
            psi = self.phi0 if T == 0 else self.evolve(T, n_steps)
            out.append((self.overlap(psi), self.energy(psi)))
        return np.array(out)

    def time_to_overlap(self, target=0.99, Ts=None, n_steps=160, confirm=3):
        """The first T of the grid at which the overlap reaches the target and stays
        above it for the next `confirm` grid points (the overlap oscillates weakly)."""
        Ts = np.arange(0, 81) if Ts is None else Ts
        ov = {}

        def p(i):
            if i not in ov:
                ov[i] = self.overlap(self.phi0 if Ts[i] == 0 else self.evolve(Ts[i], n_steps))
            return ov[i]
        for i in range(len(Ts)):
            if p(i) >= target and all(p(j) >= target for j in range(i + 1, min(i + 1 + confirm, len(Ts)))):
                return Ts[i]
        return np.inf


# ------------------------------------------------------- the pairing model
def pairing_refinement(k_low, k_high, N, g=1.0, buffer=0.6, psi_low=None):
    """Basis refinement of the pairing model: k_low -> k_high levels at fixed N."""
    lo, hi = PairingModel(k_low, g=g), PairingModel(k_high, g=g)
    b_lo, H_lo = lo.sector_hamiltonian(N)
    b_hi, H_hi = hi.sector_hamiltonian(N)
    P = Prolongation.embedding(b_lo, b_hi)
    return ResolutionRefinement(H_lo, H_hi, P, psi_low=psi_low, buffer=buffer)


def linear_ramp_overlap(H0, H1, psi0, target, T, n_steps=160):
    """Plain adiabatic preparation H = (1-s) H0 + s H1 for comparison."""
    psi = psi0.astype(complex)
    dt = T / n_steps
    for j in range(n_steps):
        s = (j + 0.5) / n_steps
        w, V = np.linalg.eigh((1 - s) * H0 + s * H1)
        psi = V @ (np.exp(-1j * w * dt) * (V.conj().T @ psi))
    return abs(np.vdot(target, psi)) ** 2


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    # ---- the prolongation: <H_high> of the prolonged state is E_low
    print("Prolongation k = 2 -> 4 (basis refinement), g = 1")
    for N in (2, 4):
        rr = pairing_refinement(2, 4, N)
        E_prol = rr.energy(rr.phi0)
        print(f"  N = {N}: dim_low = {rr.H_low.shape[0]:2d}, dim_high = {rr.H_high.shape[0]:2d}, "
              f"E_low = {rr.E_low:.6f}, <H_high> = {E_prol:.6f}, E_high = {rr.E_high:.6f}, "
              f"overlap = {rr.overlap(rr.phi0):.4f}")
        assert abs(E_prol - rr.E_low) < 1e-12

    # ---- Fig. 8.refine: overlap against T and the spectrum along the path
    Ts = np.array([0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 25, 30])
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    print("\nOverlap of the refined state with the exact k = 4 ground state")
    print("   T    " + "    ".join(f"N = {N}" for N in (2, 4)))
    curves = {}
    for N, mk in ((2, "s-"), (4, "o-")):
        rr = pairing_refinement(2, 4, N)
        curves[N] = rr.scan(Ts)[:, 0]
        a.plot(Ts, 1 - curves[N], mk, ms=4, label=f"$N={N}$")
    for i, T in enumerate(Ts):
        print(f"  {T:3d}   {curves[2][i]:.6f}   {curves[4][i]:.6f}")
    a.set_yscale("log"); a.set_xlabel("total adiabatic time $T$"); a.set_ylabel("$1-p$")
    a.set_title("refinement $k=2\\to4$"); a.legend()
    rr = pairing_refinement(2, 4, 4)
    s = np.linspace(0, 1, 101)
    w = rr.spectrum(s, 8)
    b.plot(s, w[:, 0], "C3", lw=2)
    b.plot(s, w[:, 1:], "0.5", lw=0.8)
    b.set_xlabel("$s=t/T$"); b.set_ylabel("eigenvalues of $\\hat{H}(s)$")
    b.set_title(f"$N=4$: minimum gap {rr.min_gap():.3f}")
    fig.tight_layout(); savefig(fig, 8, "refine")
    print(f"  minimum gap along the path (N = 4): {rr.min_gap():.4f}")

    # ---- Fig. 8.fourparticle: overlap and energy, k_high = 3 and 4
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    print("\nFour particles, k = 2 -> k_high: overlap and energy at selected T")
    for kh, mk in ((3, "s-"), (4, "o-")):
        rr = pairing_refinement(2, kh, 4)
        res = rr.scan(Ts)
        a.plot(Ts, res[:, 0], mk, ms=4, label=f"$k_{{\\rm high}}={kh}$")
        b.plot(Ts, res[:, 1], mk, ms=4, label=f"$k_{{\\rm high}}={kh}$")
        b.axhline(rr.E_high, color="k", ls=":", lw=0.8)
        print(f"  k_high = {kh}: E_FCI = {rr.E_high:.6f}, overlap(0) = {res[0, 0]:.4f}, "
              f"overlap(30) = {res[-1, 0]:.4f}, E(30) = {res[-1, 1]:.4f}, T* = {rr.time_to_overlap()}")
        assert res[-1, 0] > 0.995 and res[-1, 1] >= rr.E_high
    a.set_xlabel("$T$"); a.set_ylabel("overlap $p$"); a.legend()
    b.set_xlabel("$T$"); b.set_ylabel("$\\langle\\hat{H}_{\\rm high}\\rangle$"); b.legend()
    fig.tight_layout(); savefig(fig, 8, "fourparticle")

    # ---- the slow tail: beyond T ~ 40 the exponential decay of 1 - p flattens
    rr = pairing_refinement(2, 4, 4)
    print("\nThe tail of the k = 2 -> 4 refinement (8 T slices):  "
          + ",  ".join(f"1-p({T}) = {1 - rr.overlap(rr.evolve(T, 8 * T)):.1e}" for T in (50, 70, 100, 110)))

    # ---- what the shift and the cos^2/sin^2 schedule buy: a plain linear ramp from
    # the Hamiltonian without pairing (g = 0) to g = 1 in the same k = 4 space
    hi0, hi1 = PairingModel(4, g=0.0), PairingModel(4, g=1.0)
    basis, H0 = hi0.sector_hamiltonian(4)
    H1 = hi1.sector_hamiltonian(4)[1]
    # g = 0 is degenerate in the full sector only above the ground state; the HF determinant is unique
    psi0 = basis.vector(hi0.reference(2))
    rr = pairing_refinement(2, 4, 4)
    print("\nRefinement against a linear coupling ramp g: 0 -> 1 (k = 4, N = 4)")
    print("   T    refinement   coupling ramp")
    for T in (5, 10, 20, 30):
        print(f"  {T:3d}   {rr.overlap(rr.evolve(T)):.6f}     {linear_ramp_overlap(H0, H1, psi0, rr.psi_high, T):.6f}")
