# BookPrograms/chapter3 — The Quantum Approximate Optimization Algorithm

| Program | What it computes | Runtime |
|---|---|---|
| `qubo.py` | `QUBO` (builders `maxcut`, `portfolio`; `to_ising`) and `Ising` (energies of all bit strings, Pauli terms, brute-force ground state); checks Table 3-qubo and the ring. | <1 s |
| `qaoa.py` | `QAOA`: the depth-p state on the simulator (phases + Hadamard transforms), energy, probabilities, samples, the finite-difference / global-shift / per-term parameter-shift gradients, BFGS and gradient-descent optimization with restarts and warm start, gate count; checks the one-edge closed form and the gradient rules. | 1 s |
| `maxcut.py` | Ring C4 and the six-vertex graph: Table 3-ring, the ring closed form, figures `ring`, `gradients`. | ~1 min |
| `double_well.py` | Two electrons in a double quantum dot: exchange spectrum, closed-form landscape, figures `dwlandscape`, `dwprobs`, sampled readouts, p = 1-3. | 5 s |
| `portfolio.py` | Mean-variance portfolio selection on synthetic and S&P data (`sp500_weekly.csv`): Tables 3-portfolio, 3-portfolioqaoa, 3-penalty; figures `sp500data`, `portfolio`, `thermal`. | ~40 s |
| `clustering.py` | Two-way clustering as weighted MaxCut on eight points; figure `clustering`. | 10 s |
| `sp500_weekly.csv` | 164 weekly closing prices of AAPL, MSFT, AMZN, GOOG (Jan 2020 - Dec 2023), from the course notebook `qaoa_finance.ipynb`. | data |

Import order: `qubo` <- `qaoa` <- `maxcut`, `double_well`, `portfolio`, `clustering`. Sign convention: QAOA minimises <H_C>; for MaxCut H_C = -(1/2) sum w_ij (I - Z_i Z_j), so the one-edge optimum is at gamma = 3pi/2, beta = pi/8. Mirrors the chapter's closing `Programs for this chapter` list.
