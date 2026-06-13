# pairing-eigenq

Reproducible pipeline for **resolution refinement + the rodeo algorithm** applied
to the constant-pairing Hamiltonian — a controlled demonstration of the EIGEN-Q
eigenstate-preparation workflow, producing a Jupyter notebook and a LaTeX article.

```
pairing-eigenq/
├── CLAUDE.md                 # project memory for Claude Code
├── .claude/skills/           # 4 Agent Skills (pairing-model, vqe-circuits,
│                             #   notebook-builder, physics-paper)
├── src/pairinglib/           # tested Python package (single source of truth)
├── notebooks/                # ResolutionRefPairing.ipynb (build artifact)
├── paper/                    # eigenq_pairing.tex + figs/ (generated)
├── tests/                    # pytest: API + benchmark anchors
├── scripts/pipeline.py       # one-command orchestration
├── Makefile                  # install / test / notebook / check / figures / paper / all
└── .github/workflows/ci.yml  # runs tests + builds notebook + paper
```

## Quickstart
```bash
make install        # pip install -e . + jupyter/latex deps
make test           # library + benchmark anchors
make all            # notebook -> check -> figures -> paper PDF
```

## Working with Claude Code
Open the repo with Claude Code; the skills in `.claude/skills/` load
automatically and `CLAUDE.md` provides project memory. Example requests:
- "Add a Trotterised controlled-evolution variant to the rodeo chapter and
  quantify the Trotter error." (uses `vqe-circuits` + `notebook-builder`)
- "Regenerate the paper figures and rebuild the PDF." (uses `physics-paper`)
- "Extend the four-particle analysis to N=6." (uses `pairing-model` + library)

Authors (alphabetical): Bogner, Glittum, Hergert, Hjorth-Jensen, Lange, LaRose,
Lee, Massel. See `paper/eigenq_pairing.tex` for affiliations.
