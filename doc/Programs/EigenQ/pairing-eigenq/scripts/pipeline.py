#!/usr/bin/env python3
"""End-to-end pipeline: execute notebook -> check -> extract figures -> build paper.
Run from the repo root:  python scripts/pipeline.py"""
import subprocess, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
S = ROOT / ".claude/skills"
NB = ROOT / "notebooks/ResolutionRefPairing.ipynb"
TEX = ROOT / "paper/eigenq_pairing.tex"
steps = [
    ["python3", str(S/"notebook-builder/scripts/execute_notebook.py"), str(NB), "1800"],
    ["python3", str(S/"notebook-builder/scripts/check_notebook.py"), str(NB), "6"],
    ["python3", str(S/"physics-paper/scripts/extract_figures.py"), str(NB), str(ROOT/"paper/figs")],
    ["bash",    str(S/"physics-paper/scripts/build_paper.sh"), str(TEX)],
]
for cmd in steps:
    print("==>", " ".join(cmd))
    if subprocess.call(cmd): sys.exit("pipeline failed at: " + " ".join(cmd))
print("pipeline complete")
