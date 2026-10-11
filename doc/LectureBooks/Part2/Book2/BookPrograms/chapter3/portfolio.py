"""Chapter 3, Sec. 3.7: QAOA for mean-variance portfolio selection.

Select K of n assets so as to minimise -mu^T x + lam x^T Sigma x subject to
sum_i x_i = K, Eq. (3-markowitz), with the budget enforced by a quadratic
penalty, Eq. (3-portfolioqubo).  Two data sets: four synthetic assets
(Table 3-synthetic) and 164 weekly closing prices of AAPL, MSFT, AMZN and
GOOG, 2020-2023, in sp500_weekly.csv (Fig. 3-sp500data).  The script

  * builds the QUBO, the Ising coefficients and the cost Hamiltonian, and
    lists every feasible (budget-K) portfolio with its cost (Table 3-portfolio);
  * runs QAOA for p = 1, 2, 3 on both data sets and decodes the most probable
    bit strings (Table 3-portfolioqaoa, Fig. 3-portfolio);
  * compares the QAOA energies with the classical thermal energy of the same
    Ising model as a function of the inverse temperature (Fig. 3-thermal),
    Eq. (3-thermal);
  * shows what happens when the penalty A is too small (Table 3-penalty).
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]      # BookPrograms/
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np

from qubo import QUBO, bitstrings
from qaoa import QAOA

HERE = pathlib.Path(__file__).resolve().parent

# synthetic data: annualised expected returns and covariance of four assets
TICKERS_SYNTH = ["A", "B", "C", "D"]
MU_SYNTH = np.array([0.12, 0.10, 0.08, 0.15])
SIGMA_SYNTH = np.array([[0.100, 0.020, 0.010, 0.030],
                        [0.020, 0.080, 0.020, 0.010],
                        [0.010, 0.020, 0.060, 0.020],
                        [0.030, 0.010, 0.020, 0.090]])
TICKERS_SP = ["AAPL", "MSFT", "AMZN", "GOOG"]


def sp500():
    """Weekly log returns from sp500_weekly.csv, and their annualised mean
    and covariance (52 weeks per year)."""
    prices = np.loadtxt(HERE / "sp500_weekly.csv", delimiter=",", comments="#")
    r = np.diff(np.log(prices), axis=0)
    return prices, r, 52 * r.mean(0), 52 * np.cov(r.T)


def feasible_table(mu, Sigma, lam, K, tickers):
    """Every portfolio with exactly K assets and its mean-variance cost."""
    rows = []
    for x in bitstrings(len(mu)):
        if x.sum() == K:
            cost = -mu @ x + lam * x @ Sigma @ x
            rows.append(("+".join(t for t, v in zip(tickers, x) if v), "".join(map(str, x)), cost))
    return sorted(rows, key=lambda r: r[2])


def decode(qaoa, params, tickers, top=4):
    """The most probable bit strings of the QAOA state with their costs."""
    pr = qaoa.probabilities(params)
    out = []
    for k in np.argsort(pr)[::-1][:top]:
        x = bitstrings(qaoa.n)[k]
        out.append(("+".join(t for t, v in zip(tickers, x) if v) or "none", "".join(map(str, x)),
                    pr[k], qaoa.diag[k], int(x.sum())))
    return out


def thermal_energy(energies, beta_th):
    """<H>_beta = sum_z H(z) e^{-beta H(z)} / Z for the classical Ising model, Eq. (3-thermal)."""
    E = energies - energies.min()
    w = np.exp(-beta_th * E)
    return float((energies * w).sum() / w.sum())


def run(mu, Sigma, lam, A, K, tickers, name, restarts=6, depths=(1, 2, 3, 4)):
    q = QUBO.portfolio(mu, Sigma, lam, A, K)
    ising = q.to_ising()
    print(f"\n{name}: lambda = {lam}, A = {A}, K = {K}")
    print("  QUBO diagonal:", np.round(np.diag(q.Q), 5))
    print("  Ising h:", np.round(ising.h, 5), " J (upper):", np.round(ising.J[np.triu_indices(4, 1)], 5),
          f" const {ising.const:.5f}")
    rows = feasible_table(mu, Sigma, lam, K, tickers)
    print(f"  feasible portfolios (budget {K}), Table 3-portfolio:")
    for r in rows:
        print(f"    {r[0]:12s} {r[1]}  cost {r[2]:+.6f}")
    E0, gs = ising.ground_state()
    print(f"  ground state of H_C: E0 = {E0:+.6f} at", ["".join(map(str, bitstrings(4)[k])) for k in gs],
          "(= best feasible portfolio + penalty offset 0)" if np.isclose(E0, rows[0][2]) else "(NOT feasible: penalty too small)")
    E_all = ising.energies()
    E_uniform = E_all.mean()                      # the energy of |+>^n, the p = 0 state
    print(f"  energy of the uniform superposition {E_uniform:+.4f}, spectrum [{E_all.min():+.4f}, {E_all.max():+.4f}]")
    results, warm = {}, None
    for p in depths:
        qa = QAOA(ising, p)
        x, E, _ = qa.optimize(restarts=restarts, rng=0, warm=warm)
        warm = x
        top = decode(qa, x, tickers)
        pf = sum(qa.probabilities(x)[k] for k in range(16) if bitstrings(4)[k].sum() == K)
        gap = (E_uniform - E) / (E_uniform - E0)   # fraction of the gap to E0 closed, Eq. (3-gapfraction)
        results[p] = dict(qaoa=qa, params=x, E=E, top=top, p_feasible=pf, gap=gap,
                          p_opt=qa.success_probability(x))
        print(f"  p={p}: E = {E:+.5f}  gap closed {gap:.3f}  P(optimal) {results[p]['p_opt']:.3f}  "
              f"P(feasible) {pf:.3f}  most probable: {top[0][0]} ({top[0][2]:.3f})")
    return q, ising, rows, results


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from bookplot import savefig

    # ------------------------------------------------------------ data
    prices, returns, mu_sp, Sigma_sp = sp500()
    print(f"S&P data: {len(prices)} weekly closes, {len(returns)} returns")
    print("  annualised mean return (%):", np.round(mu_sp * 100, 2))
    print("  annualised volatility (%):", np.round(np.sqrt(np.diag(Sigma_sp)) * 100, 2))
    corr = Sigma_sp / np.sqrt(np.outer(np.diag(Sigma_sp), np.diag(Sigma_sp)))
    print("  correlation matrix:\n", np.round(corr, 3))
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.3))
    for k, t in enumerate(TICKERS_SP):
        axes[0].plot(prices[:, k] / prices[0, k], label=t, lw=1.2)
    axes[0].set_xlabel("week"); axes[0].set_ylabel("price / initial price"); axes[0].legend(fontsize=8)
    im = axes[1].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=axes[1])
    axes[1].set_xticks(range(4)); axes[1].set_yticks(range(4))
    axes[1].set_xticklabels(TICKERS_SP); axes[1].set_yticklabels(TICKERS_SP)
    for i in range(4):
        for j in range(4):
            axes[1].text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8)
    axes[1].set_title("return correlations")
    vol = np.sqrt(np.diag(Sigma_sp)) * 100; ret = mu_sp * 100
    axes[2].scatter(vol, ret, s=60)
    for k, t in enumerate(TICKERS_SP):
        axes[2].annotate(t, (vol[k], ret[k]), textcoords="offset points", xytext=(5, 3), fontsize=8)
    axes[2].set_xlabel("annualised volatility (%)"); axes[2].set_ylabel("annualised return (%)")
    fig.tight_layout()
    savefig(fig, 3, "sp500data")

    # ------------------------------------------------------ experiments
    qA, isA, rowsA, resA = run(MU_SYNTH, SIGMA_SYNTH, lam=1.0, A=0.3, K=2, tickers=TICKERS_SYNTH, name="Synthetic data")
    qB, isB, rowsB, resB = run(mu_sp, Sigma_sp, lam=2.0, A=0.3, K=2, tickers=TICKERS_SP, name="S&P data")

    # Fig. 3-portfolio: output distributions for the S&P data and the p = 1 landscape
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.3))
    q1 = resB[1]["qaoa"]
    gs, bs = np.linspace(0, 2 * np.pi, 45), np.linspace(0, np.pi, 45)
    L = np.array([[q1.energy([g, b]) for b in bs] for g in gs])
    cp = axes[0].contourf(gs / np.pi, bs / np.pi, L.T, levels=24, cmap="viridis")
    plt.colorbar(cp, ax=axes[0], label=r"$E(\gamma,\beta)$")
    axes[0].plot(np.mod(resB[1]["params"][0], 2 * np.pi) / np.pi, np.mod(resB[1]["params"][1], np.pi) / np.pi, "r*", ms=11)
    axes[0].set_xlabel(r"$\gamma/\pi$"); axes[0].set_ylabel(r"$\beta/\pi$"); axes[0].set_title("S&P data, $p=1$")
    labels = ["+".join(t for t, v in zip(TICKERS_SP, x) if v) if x.sum() == 2 else "".join(map(str, x)) for x in bitstrings(4)]
    for ax, p in zip(axes[1:], (1, 2, 4)):
        pr = resB[p]["qaoa"].probabilities(resB[p]["params"])
        cols = ["tab:red" if bitstrings(4)[k].sum() == 2 else "tab:blue" for k in range(16)]
        ax.bar(range(16), pr, color=cols)
        ax.set_xticks(range(16)); ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_title(f"$p={p}$: $P_{{\\rm feasible}}={resB[p]['p_feasible']:.2f}$, $P_{{\\rm opt}}={resB[p]['p_opt']:.2f}$", fontsize=9)
        ax.set_ylim(0, 0.6)
    axes[1].set_ylabel("probability")
    fig.tight_layout()
    savefig(fig, 3, "portfolio")

    # Fig. 3-thermal: classical thermal energy against the QAOA energies
    EB = isB.energies()
    beta_th = np.logspace(-1, 4, 200)
    Eth = [thermal_energy(EB, b) for b in beta_th]
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    ax.semilogx(beta_th, Eth, "k-", lw=2, label=r"classical $\langle H\rangle_{\beta_T}$")
    for p in (1, 2, 3, 4):
        ax.axhline(resB[p]["E"], ls="--", lw=1, color=f"C{p}", label=f"QAOA $p={p}$")
    ax.axhline(EB.min(), color="gray", ls=":", label="$E_0$")
    ax.set_xlabel(r"inverse temperature $\beta_T$"); ax.set_ylabel("energy"); ax.legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, 3, "thermal")
    for p in (1, 2, 3, 4):
        b_eff = beta_th[np.argmin(abs(np.array(Eth) - resB[p]["E"]))]
        print(f"  QAOA p={p} energy {resB[p]['E']:+.5f} equals the thermal energy at beta_T ~ {b_eff:.1f}")

    # Table 3-penalty: the penalty must exceed the cost differences
    print("\nTable 3-penalty (S&P data, p = 2): the penalty strength A")
    for A in (0.03, 0.1, 0.3, 1.0):
        q = QUBO.portfolio(mu_sp, Sigma_sp, 2.0, A, 2).to_ising()
        E0, gs = q.ground_state()
        x0 = bitstrings(4)[gs[0]]
        qa = QAOA(q, 2)
        x, E, _ = qa.optimize(restarts=4, rng=0)
        pf = sum(qa.probabilities(x)[k] for k in range(16) if bitstrings(4)[k].sum() == 2)
        print(f"  A = {A:5.2f}: ground state {''.join(map(str, x0))} (budget {x0.sum()}), "
              f"P(feasible) {pf:.3f}, P(ground state) {qa.success_probability(x):.3f}, "
              f"P(MSFT+GOOG) {qa.probabilities(x)[5]:.3f}")
    print("figures sp500data, portfolio and thermal written")
