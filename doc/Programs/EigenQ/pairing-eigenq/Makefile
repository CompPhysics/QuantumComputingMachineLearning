NB     := notebooks/ResolutionRefPairing.ipynb
TEX    := paper/eigenq_pairing.tex
FIGS   := paper/figs
SKILLS := .claude/skills
PY     := python3

.PHONY: install test notebook check figures paper all clean

install:
	$(PY) -m pip install -e ".[notebook,dev]"

test:
	$(PY) -m pytest -q

notebook:
	$(PY) $(SKILLS)/notebook-builder/scripts/execute_notebook.py $(NB) 1800

check:
	$(PY) $(SKILLS)/notebook-builder/scripts/check_notebook.py $(NB) 6

figures:
	$(PY) $(SKILLS)/physics-paper/scripts/extract_figures.py $(NB) $(FIGS)

paper: figures
	bash $(SKILLS)/physics-paper/scripts/build_paper.sh $(TEX)

all: test notebook check paper

clean:
	rm -rf paper/*.aux paper/*.log paper/*.out __pycache__ src/*.egg-info
