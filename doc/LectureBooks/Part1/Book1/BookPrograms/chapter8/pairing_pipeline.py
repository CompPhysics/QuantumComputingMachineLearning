"""Chapter 8, Sec. 8.4: refinement + rodeo, end to end on the pairing model.

N = 4 particles, g = 1, target resolution k = 4 (eight qubits, sector dimension 70).

1. two refinement chains to k = 4: from the gate-free k = 2 determinant and from
   the UCCSD state at k = 3, with the shift referenced to the variational coarse
   energy (Fig. 8.chains);
2. the coupling scan g = 0.3 ... 4 at fixed buffer and the buffer scan at g = 1,
   i.e. T* against the inverse minimum gap of the path (Fig. 8.gapscan);
3. rodeo energy scans of the refined and of the merely prolonged state
   (Fig. 8.scan);
4. rodeo preparation: infidelity and acceptance against the number of cycles
   for both inputs (Fig. 8.convergence, Table 8.pipeline);
5. the same with three times chosen from the scan, t = pi/(E_j - E_0), each of
   which removes one excited level exactly (Table 8.chosentimes).

About 30 s (the T* searches dominate).
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np
from models import PairingModel
from refinement import Prolongation, ResolutionRefinement, pairing_refinement
from rodeo import RodeoFilter, infidelity_bound
from uccsd_sector import pairing_references
from bookplot import savefig, plt


def refined_input(T=30, n_steps=160):
    """The k = 2 -> 4 refined state of Sec. 8.4.2 and the refinement object."""
    rr = pairing_refinement(2, 4, 4)
    return rr, rr.evolve(T, n_steps)


if __name__ == "__main__":
    # ------------------------------------------------ 1. the two chains
    Ts = np.arange(0, 31)
    rr24 = pairing_refinement(2, 4, 4)
    ref3 = pairing_references(3)
    psi3 = ref3["solver"].state(ref3["theta"])                      # the UCCSD state at k = 3
    rr34 = pairing_refinement(3, 4, 4, psi_low=psi3)
    res24, res34 = rr24.scan(Ts), rr34.scan(Ts)
    print("Refinement chains to k = 4 (N = 4, g = 1)")
    print(f"  2 -> 4: E_low = {rr24.E_low:.6f}, prolonged overlap = {res24[0, 0]:.4f}, "
          f"T* = {rr24.time_to_overlap(0.99, Ts)}, min gap = {rr24.min_gap():.3f}")
    print(f"  3 -> 4: E_low = {rr34.E_low:.6f} (UCCSD), prolonged overlap = {res34[0, 0]:.4f}, "
          f"T* = {rr34.time_to_overlap(0.99, Ts)}, min gap = {rr34.min_gap():.3f}")
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for res, mk, lab in ((res24, "o-", "$k=2\\to4$"), (res34, "s-", "$k=3\\to4$ (UCCSD start)")):
        a.plot(Ts, res[:, 0], mk, ms=3, label=lab); b.semilogy(Ts, 1 - res[:, 0], mk, ms=3, label=lab)
    a.axhline(0.99, color="k", ls=":", lw=0.8); b.axhline(0.01, color="k", ls=":", lw=0.8)
    a.set_xlabel("$T$"); a.set_ylabel("overlap $p$"); a.legend()
    b.set_xlabel("$T$"); b.set_ylabel("$1-p$")
    fig.tight_layout(); savefig(fig, 8, "chains")

    # ------------------------------------------------ 2. coupling scan and gap scan
    print("\nCoupling scan, k = 2 -> 4, buffer 0.6")
    print("     g    p(0)    min gap    T*(0.99)")
    gs, Tstar_g, p0_g = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0], [], []
    grid = np.arange(0, 91)
    for g in gs:
        rr = pairing_refinement(2, 4, 4, g=g)
        p0 = rr.overlap(rr.phi0); Tst = rr.time_to_overlap(0.99, grid)
        Tstar_g.append(Tst); p0_g.append(p0)
        print(f"  {g:4.1f}   {p0:.4f}   {rr.min_gap():.4f}    {Tst}")
    print("\nBuffer scan at g = 1")
    print("   buffer   min gap   T*(0.99)")
    bufs, gaps, Tstar_b = [0.3, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5], [], []
    for buf in bufs:
        rr = pairing_refinement(2, 4, 4, buffer=buf)
        gaps.append(rr.min_gap()); Tstar_b.append(rr.time_to_overlap(0.99, grid))
        print(f"   {buf:5.2f}   {gaps[-1]:.4f}    {Tstar_b[-1]}")
    slope = np.polyfit(1 / np.array(gaps), Tstar_b, 1)
    print(f"  linear fit T* = {slope[0]:.1f} / gap {slope[1]:+.1f}")
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    a.plot(gs, Tstar_g, "o-", ms=4)
    for g, T, p0 in zip(gs, Tstar_g, p0_g):
        a.annotate(f"{p0:.2f}", (g, T), textcoords="offset points", xytext=(4, 4), fontsize=8)
    a.set_xlabel("pairing strength $g$"); a.set_ylabel("$T^*$ (overlap 0.99)")
    x = 1 / np.array(gaps)
    b.plot(x, Tstar_b, "s", ms=5); b.plot(x, np.polyval(slope, x), "k--", lw=0.8)
    b.set_xlabel("$1/\\Delta E_{\\min}$"); b.set_ylabel("$T^*$")
    fig.tight_layout(); savefig(fig, 8, "gapscan")

    # ------------------------------------------------ 3. rodeo scans
    rr, refined = refined_input()
    H, E0, ground = rr.H_high, rr.E_high, rr.psi_high
    p_ref, p_pro = rr.overlap(refined), rr.overlap(rr.phi0)
    print(f"\nRodeo inputs: refined p = {p_ref:.4f}, prolonged p = {p_pro:.4f}; E0 = {E0:.6f}, "
          f"first excited state with weight: see scan")
    rng = np.random.default_rng(3)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    coarse = RodeoFilter(H, sigma=2.0, rng=rng)
    Es = np.linspace(-0.5, 8.0, 341)
    for psi, lab, col in ((rr.phi0, f"prolonged, $p={p_pro:.3f}$", "C1"), (refined, f"refined, $p={p_ref:.3f}$", "C0")):
        a.semilogy(Es, coarse.scan(psi, Es, M=4, n_sets=200), col, label=lab)
    c2 = np.abs(rr.psi_high @ rr.phi0) ** 2
    weights = np.abs(coarse.V.T @ rr.phi0) ** 2
    for Ej, wj in zip(coarse.w, weights):
        if wj > 1e-6:
            a.axvline(Ej, color="k", lw=0.4, ls=":")
    a.axhline(2.0 ** -4, color="k", ls="--", lw=0.8)
    a.set_xlabel("target energy $E$"); a.set_ylabel("$P_{0^4}(E)$"); a.legend(); a.set_title("$\\sigma=2$")
    print("  eigenstates carrying weight in the prolonged state:",
          ", ".join(f"{Ej:.3f} ({wj:.4f})" for Ej, wj in zip(coarse.w, weights) if wj > 1e-4))
    fine = RodeoFilter(H, sigma=8.0, rng=rng)
    Ef = np.linspace(E0 - 0.6, E0 + 0.6, 241)
    scan = fine.scan(refined, Ef, M=4, n_sets=400)
    top = Ef[np.argmax(scan)]
    # a parabola through the three highest points refines the peak position
    i = np.argmax(scan); c = np.polyfit(Ef[i - 1:i + 2], scan[i - 1:i + 2], 2); peak = -c[1] / (2 * c[0])
    b.plot(Ef, scan, "C0"); b.axvline(E0, color="k", ls=":", lw=0.8)
    b.set_xlabel("target energy $E$"); b.set_title(f"$\\sigma=8$: peak at {peak:.4f}, exact {E0:.4f}")
    fig.tight_layout(); savefig(fig, 8, "scan")
    print(f"  fine scan (sigma = 8, M = 4): peak at E = {peak:.4f} (grid maximum {top:.4f}), exact {E0:.6f}")

    # ------------------------------------------------ 4. preparation
    M = 8
    rf = RodeoFilter(H, sigma=4.0, rng=rng)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    print("\nRodeo preparation at E = E0, sigma = 4: infidelity of the accepted runs and acceptance")
    print("   M    refined: infid.   bound      accept.     prolonged: infid.   bound      accept.")
    table = {}
    for psi, p, lab, col in ((refined, p_ref, "refined", "C0"), (rr.phi0, p_pro, "prolonged", "C1")):
        infid, acc = rf.expected(psi, E0, M)
        s_inf, s_acc = rf.prepare(psi, E0, M, ground, n_sets=300)
        table[lab] = (infid, acc)
        Ms = np.arange(1, M + 1)
        a.semilogy(Ms, infid, col + "-", label=lab); a.semilogy(Ms, s_inf, col + "o", ms=3)
        a.semilogy(Ms, infidelity_bound(p, Ms), col + ":", lw=1)
        b.plot(Ms, acc, col + "-", label=lab); b.plot(Ms, s_acc, col + "o", ms=3); b.axhline(p, color=col, ls=":", lw=0.8)
        assert np.all(infid <= infidelity_bound(p, Ms) * (1 + 1e-9))
    for k in range(M):
        r, q = table["refined"], table["prolonged"]
        print(f"  {k + 1:2d}    {r[0][k]:.2e}   {infidelity_bound(p_ref, k + 1):.2e}   {r[1][k]:.4f}      "
              f"{q[0][k]:.2e}   {infidelity_bound(p_pro, k + 1):.2e}   {q[1][k]:.4f}")
    a.set_xlabel("cycles $M$"); a.set_ylabel("$1-\\mathcal{F}_M$"); a.legend()
    b.set_xlabel("cycles $M$"); b.set_ylabel("acceptance $P_M$"); b.legend()
    fig.tight_layout(); savefig(fig, 8, "convergence")

    # ------------------------------------------------ 5. times chosen from the spectrum
    print("\nThree cycles with t_j = pi/(E_j - E0) for the three excited levels that carry the most weight")
    for psi, lab in ((refined, "refined"), (rr.phi0, "prolonged")):
        wts = np.abs(rf.V.T @ psi) ** 2
        levels = [j for j in np.argsort(-wts) if j != 0][:3]
        phi, P = psi.astype(complex), 1.0
        for j in levels:
            t = np.pi / (rf.w[j] - E0)
            phi, p = rf.cycle(phi, t, E0); P *= p
            print(f"  {lab:9s}  E_j = {rf.w[j]:.4f}  weight {wts[j]:.2e}  t = {t:.4f}  "
                  f"infidelity {1 - abs(ground @ phi) ** 2:.2e}  acceptance {P:.4f}")
