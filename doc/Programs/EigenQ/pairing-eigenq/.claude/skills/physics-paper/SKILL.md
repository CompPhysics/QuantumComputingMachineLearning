---
name: physics-paper
description: >
  How to build the LaTeX scientific article from the notebook: extract the
  notebook figures, write/maintain the paper in the project's article style, and
  compile to PDF. Use this skill WHENEVER the task is to create, edit, or compile
  the paper (eigenq_pairing.tex), pull figures out of the notebook into
  paper/figs, update the author list or bibliography, or turn notebook results
  into article prose. Follow `references/article-conventions.md` for the house
  style (article class, authblk author block, figure/caption rules).
---

# Physics paper builder

The article is generated FROM the executed notebook: figures are extracted, and
every quantitative claim must trace back to a notebook output. Do not invent
numbers.

## Workflow
1. **Extract figures** from the executed notebook into `paper/figs/`:
   `python scripts/extract_figures.py notebooks/ResolutionRefPairing.ipynb paper/figs`
   It writes `fig1.png ...` in cell order and prints a cell->figure map; choose
   which to include and reference them with `\includegraphics{figs/figN.png}`.
2. **Write / update** `paper/eigenq_pairing.tex` following
   `references/article-conventions.md` (and reuse `assets/article-template.tex`
   as the starting skeleton for a new paper).
3. **Compile**: `bash scripts/build_paper.sh paper/eigenq_pairing.tex` (runs
   pdflatex twice to resolve references; fails loudly on LaTeX errors or
   undefined citations/references).

## House rules
- Author list in ALPHABETICAL order by surname, with `authblk` affiliation
  superscripts. The canonical list lives in the README / application; keep it in sync.
- Standard `article` class (RevTeX is not assumed installed). Packages known to
  be available: amsmath, amssymb, graphicx, booktabs, authblk, physics, braket,
  hyperref, lmodern, geometry.
- Wide notebook figures (~13:5) go full `\linewidth`, one per `figure`.
- Equations and quoted results must match the notebook and the rodeo reference
  (see the vqe-circuits skill); cite Choi 2021 / Qian 2024 for the rodeo and
  Bogner 2026 (PLB 875) for resolution refinement.
- Keep Sections within the journal page budget if one is specified.
