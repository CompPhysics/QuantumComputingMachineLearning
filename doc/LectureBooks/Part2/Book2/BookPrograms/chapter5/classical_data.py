"""Chapter 5, Sec. 5.4: Boltzmann machines on classical data.

The XOR (even-parity) distribution on two bits and the 2x2 bars-and-stripes
distribution are learned by
  (a) the classical RBM of rbm.py (binary units, exact gradient, 1000 steps),
  (b) the restricted QBM without transverse fields (a classical Boltzmann
      machine written with spins, exact gradient),
  (c) the restricted QBM with transverse fields and the exact Duhamel
      gradient of Eq. (5-exactgrad),
  (d) the same model trained with the clamped-state (bound) gradient of
      Eq. (5-boundgrad),
all with n_h = 2 (XOR, 300 Adam steps, lr 0.1) or n_h = 3 (bars and
stripes, 200 steps, lr 0.05) hidden units.
Produces Table 5-classical and Fig. 5-classical (training curves, the
dependence of the XOR likelihood on a common transverse field, the
learned bars-and-stripes distribution).  Runtime about 100 s.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]      # BookPrograms/
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np

from rbm import RBM, parity, bars_and_stripes, kl_divergence
from qbm import QBM, rqbm_terms
from optimize import Adam


def train_rbm(target, n_h, steps, lr=0.05, rng=0, record=10):
    rbm = RBM(int(np.log2(len(target))), n_h, rng=rng)
    opt = Adam(lr=lr)
    hist = []
    for s in range(steps):
        if s % record == 0:
            hist.append(rbm.nll(target))
        rbm.theta = opt.step(rbm.theta, rbm.gradient_exact(target))
    hist.append(rbm.nll(target))
    return rbm, np.array(hist)


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    import bookplot

    rng = np.random.default_rng(0)
    targets = {"XOR": (parity(2), 2, 2, 300, 0.1), "bars and stripes": (bars_and_stripes(), 4, 3, 200, 0.05)}
    print("Table 5-classical: NLL - entropy of the target = KL(target || model) after training with Adam")
    print(f"{'target':>17}  {'n_v':>3}  {'n_h':>3}  {'steps':>5}  {'lr':>4}  {'RBM (1000 steps)':>16}  {'QBM, no X':>10}  {'QBM exact':>10}  {'QBM bound':>10}  {'time':>6}")
    results = {}
    for name, (target, n_v, n_h, steps, lr) in targets.items():
        t0 = time.time()
        H_t = -np.sum(target[target > 0] * np.log(target[target > 0]))      # entropy of the target
        rbm, h_rbm = train_rbm(target, n_h, 1000, lr=0.1)       # the binary-unit RBM learns more slowly: 1000 steps
        row = [h_rbm[-1] - H_t]
        qc = QBM(n_v, n_h, rqbm_terms(n_v, n_h, transverse=False)[0])
        th0c = rng.normal(0, 0.3, qc.n_params)
        _, h_c = qc.fit(th0c, target, "exact", steps=steps, lr=lr)
        row.append(h_c[-1] - H_t)
        qq = QBM(n_v, n_h, rqbm_terms(n_v, n_h)[0])
        th0 = rng.normal(0, 0.3, qq.n_params)
        th_e, h_e = qq.fit(th0, target, "exact", steps=steps, lr=lr)
        th_b, h_b = qq.fit(th0, target, "bound", steps=steps, lr=lr)
        row += [h_e[-1] - H_t, h_b[-1] - H_t]
        results[name] = dict(rbm=h_rbm, classical=h_c, exact=h_e, bound=h_b, H=H_t, steps=steps,
                             theta_exact=th_e, theta_bound=th_b, model=qq, target=target)
        print(f"{name:>17}  {n_v:>3}  {n_h:>3}  {steps:>5}  {lr:>4}  " + "  ".join(f"{v:>{w}.4f}" for v, w in zip(row, (16, 10, 10, 10)))
              + f"  {time.time() - t0:>5.1f}s")
    # what the transverse fields do to XOR: scan a common field Gamma on top of the trained classical couplings
    xr = results["XOR"]; q = xr["model"]
    th = xr["theta_exact"].copy()
    names = rqbm_terms(2, 2)[1]
    ix = [k for k, nm in enumerate(names) if nm.startswith("X")]
    print(f"\nXOR, model trained with the exact gradient: transverse fields {np.round(th[ix], 3)} (qubits 0-3)")
    gammas = np.linspace(0, 2, 41)
    scan = []
    for G in gammas:
        t = th.copy(); t[ix] = G
        scan.append(q.nll(t, xr["target"]) - xr["H"])
    print(f"KL against a common transverse field Gamma: {scan[0]:.4f} at Gamma = 0, {min(scan):.4f} at "
          f"Gamma = {gammas[int(np.argmin(scan))]:.2f}, {scan[-1]:.4f} at Gamma = 2")
    rho_v = q.visible_state(th)
    off = np.sum(np.abs(rho_v - np.diag(np.diag(rho_v))))
    print(f"visible state of the trained XOR model: diagonal {np.round(np.diag(rho_v).real, 4)}, "
          f"sum of |off-diagonal| elements {off:.4f}")
    rho_vb = q.visible_state(xr["theta_bound"])
    print(f"bound-trained model: transverse fields {np.round(xr['theta_bound'][ix], 3)}, "
          f"visible diagonal {np.round(np.diag(rho_vb).real, 4)}")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for name, ls in [("XOR", "-"), ("bars and stripes", "--")]:
        r = results[name]
        for key, lab, c in [("rbm", "classical RBM", "C0"), ("classical", "QBM, no $X$", "C1"),
                            ("exact", "QBM, exact gradient", "C2"), ("bound", "QBM, bound gradient", "C3")]:
            n_steps = 1000 if key == "rbm" else r["steps"]
            x = np.arange(0, n_steps, 10).tolist() + [n_steps]
            axes[0].semilogy(x[:len(r[key])], r[key] - r["H"], ls, color=c, label=lab if name == "XOR" else None)
    axes[0].set_xlabel("Adam step"); axes[0].set_ylabel("KL(target $\\|$ model)"); axes[0].legend(fontsize=7)
    axes[0].set_title("XOR (solid), bars and stripes (dashed)", fontsize=9)
    axes[1].plot(gammas, scan)
    axes[1].set_xlabel("common transverse field $\\Gamma$"); axes[1].set_ylabel("KL(target $\\|$ model)")
    axes[1].set_title("XOR: transverse field on the trained model", fontsize=9)
    br = results["bars and stripes"]; x16 = np.arange(16); w = 0.27
    axes[2].bar(x16 - w, br["target"], w, color="k", label="target")
    axes[2].bar(x16, br["model"].visible_probabilities(br["theta_exact"]), w, label="QBM, exact gradient")
    axes[2].bar(x16 + w, br["model"].visible_probabilities(br["theta_bound"]), w, label="QBM, bound gradient")
    axes[2].set_xticks(x16); axes[2].set_xticklabels([format(i, "04b") for i in x16], rotation=90, fontsize=6)
    axes[2].set_ylabel("$p(v)$"); axes[2].legend(fontsize=7); axes[2].set_title("bars and stripes", fontsize=9)
    fig.tight_layout()
    bookplot.savefig(fig, 5, "classical")
    print("figure classical written")
