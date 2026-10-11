# Notebooks and the Jupyter Book of the QML book

## The notebooks

One notebook per chapter, `chapterN/chapterN.ipynb`, **generated** from `../chapterN.tex` and `../BookPrograms/chapterN/` by

    cd doc/Book/BookQML
    python3 BookPrograms/tex_to_notebook.py Notebooks        # all chapters
    python3 BookPrograms/tex_to_notebook.py Notebooks 2      # one chapter

and executed with

    cd Notebooks/chapter2 && jupyter nbconvert --to notebook --execute --inplace chapter2.ipynb

Never edit the notebooks by hand: change the `.tex` or the programs and regenerate. TikZ figures of a chapter are rendered to `../BookFigures/chapterN/<label>.png` by `BookPrograms/render_tikz.py` (needs `pdflatex` with the `standalone` class and `pdftoppm`). The generator needs only the Python standard library; executing the notebooks needs `numpy`, `matplotlib`, `jupyter`/`nbconvert`. Chapter 2 takes about 80 s to execute (the kernel-alignment program is the slowest cell), chapter 3 about 75 s (`maxcut.py` and `portfolio.py`), chapter 4 about 4 min (`barren.py`, `timedependent.py` and `regression.py` are the slowest cells), chapter 5 about 3 min (`classical_data.py` is the slowest cell); chapters 6 and 7 have no code beyond the bootstrap cell.

## The Jupyter Book

This folder is also the source of a [Jupyter Book](https://jupyterbook.org) (version 1.x) made of the chapter notebooks:

    cd doc/Book/BookQML/Notebooks
    pip install -r requirements.txt          # once; pins jupyter-book < 2
    python3 build_book.py                    # writes _build/html/index.html
    python3 build_book.py --clean            # rebuild from scratch
    python3 build_book.py --check            # bibliography + consistency checks only
    python3 build_book.py --regenerate       # regenerate the notebooks of _toc.yml from the LaTeX first
    python3 build_book.py --execute          # re-execute every notebook during the build

| File | Role |
|---|---|
| `_config.yml` | Title, author, repository, MathJax and bibliography settings. `execute_notebooks: "off"`: the notebooks are committed with their outputs and the book shows those |
| `_toc.yml` | Table of contents. Chapter 5 is commented out until its notebook exists: uncomment the line when `chapterN/chapterN.ipynb` has been generated (and update the row in `intro.md`) |
| `intro.md` | Landing page: what the book is, chapter table, how to run the code |
| `references.md`, `references.bib` | Reference list. `references.bib` is **generated** by `build_book.py` from `../chapter*.bib`; edit the chapter files |
| `_ext/bibkeys.py` | Small Sphinx extension: the reference list is labelled by BibTeX key, because the notebooks print citations as `[Key2021]` |
| `build_book.py` | Merges the bibliographies, checks that every `_toc.yml` file and every figure exists, then calls `jupyter-book build .` |
| `requirements.txt` | Packages for building and for executing |
| `.gitignore` | Keeps `_build/` out of the repository |

Everything here is a copy of the corresponding file of `doc/Book/Notebooks` (the quantum-computing book) with the paths and titles of this book; keep the two in step when one of them is improved.
