---
name: notebook-builder
description: >
  How to assemble, execute, and validate the project Jupyter notebook
  programmatically (with nbformat), so the notebook stays reproducible and is
  built the same way every time. Use this skill WHENEVER the task is to create,
  extend, rebuild, re-run, or check a notebook in this repo (e.g.
  ResolutionRefPairing.ipynb), add a section/chapter, regenerate its figures, or
  verify it executes without errors. Always build notebooks with the scripts
  here rather than editing JSON by hand or pasting large code blobs.
---

# Notebook builder

Notebooks are BUILD ARTIFACTS. The source of truth for physics is
`src/pairinglib`; notebook code cells should `import pairinglib as pl` and call
the library, not re-define it. This keeps the tested library and the narrative
in sync.

## Workflow (always in this order)
1. **Plan the cells.** A cell is either a markdown narrative cell or a code cell.
   Group: intro -> model -> coarse VQE -> prolongation -> refinement ->
   four-particle analysis -> rodeo -> summary. Read the `pairing-model` and
   `vqe-circuits` skills first for conventions and the API.
2. **Assemble** with `scripts/build_notebook.py` (uses nbformat; never hand-edit
   the .ipynb JSON). It takes an ordered list of (kind, source) cells.
3. **Execute** with `scripts/execute_notebook.py` (jupyter nbconvert --execute,
   timeout generous: the HEA k=3 optimisation alone is ~60 s).
4. **Check** with `scripts/check_notebook.py`: asserts zero error outputs, a
   minimum figure count, and that the benchmark anchor numbers
   (FCI/CCD/UCCSD at k=2,3; refined overlaps; rodeo acceptance->p) appear in the
   outputs. Treat a failing check as a build failure.

## Rules
- Keep markdown prose first-person-plural and paper-like (this notebook is the
  source for the article). One concept per code cell; a figure-producing cell
  ends in `plt.show()`.
- Numbers quoted in markdown must come from a cell that actually computed them.
- Do not use `localStorage`/browser APIs; this is a Python kernel notebook.
- After any change, re-run execute + check before declaring done.

## Common entry points
- "add a section X to the notebook" -> append cells, execute, check.
- "rebuild the notebook from scratch" -> run the full builder, execute, check.
- "the notebook errors" -> execute, read the first error output, fix the cell or
  the library, repeat.
