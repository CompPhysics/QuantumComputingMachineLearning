# Notebooks and the Jupyter Book

## The notebooks

One notebook per chapter, `chapterN/chapterN.ipynb`, **generated** from `../chapterN.tex` and `../BookPrograms/chapterN/` by

    cd doc/Book
    python3 BookPrograms/tex_to_notebook.py Notebooks        # all chapters
    python3 BookPrograms/tex_to_notebook.py Notebooks 2      # one chapter

and executed with

    cd Notebooks/chapter2 && jupyter nbconvert --to notebook --execute --inplace chapter2.ipynb

Never edit the notebooks by hand: change the `.tex` or the programs and regenerate. TikZ figures of a chapter are rendered to `../BookFigures/chapterN/<label>.png` by `BookPrograms/render_tikz.py` (needs `pdflatex` with the `standalone` class and `pdftoppm`). The generator needs only the Python standard library; executing the notebooks needs `numpy`, `matplotlib`, `jupyter`/`nbconvert`. Chapters 1 and 11 are prose only; their notebooks have no outputs to store.

## The Jupyter Book

This folder is also the source of a [Jupyter Book](https://jupyterbook.org) (version 1.x) made of the chapter notebooks:

    cd doc/Book/Notebooks
    pip install -r requirements.txt          # once; pins jupyter-book < 2
    python3 build_book.py                    # writes _build/html/index.html
    python3 build_book.py --clean            # rebuild from scratch
    python3 build_book.py --check            # bibliography + consistency checks only
    python3 build_book.py --regenerate       # regenerate the notebooks of _toc.yml from the LaTeX first
    python3 build_book.py --execute          # re-execute every notebook during the build (~10 min)

| File | Role |
|---|---|
| `_config.yml` | Title, authors, repository, MathJax and bibliography settings. `execute_notebooks: "off"`: the notebooks are committed with their outputs and the book shows those |
| `_toc.yml` | Table of contents in three parts. Chapters 5, 9 and 10 are commented out until their notebooks exist — uncomment the line when `chapterN/chapterN.ipynb` has been generated (and add the row to `intro.md`) |
| `intro.md` | Landing page: what the book is, chapter table, how to run the code |
| `references.md`, `references.bib` | Reference list. `references.bib` is **generated** by `build_book.py` from `../chapter*.bib` (duplicate keys between chapters removed); edit the chapter files |
| `_ext/bibkeys.py` | Small Sphinx extension: the reference list is labelled by BibTeX key, because the notebooks print citations as `[Key2021]` |
| `build_book.py` | Merges the bibliographies, checks that every `_toc.yml` file and every figure exists, then calls `jupyter-book build .` |
| `requirements.txt` | Packages for building and for executing |
| `.gitignore` | Keeps `_build/` out of the repository |

`jupyter-book build .` alone also works once `references.bib` exists. The figures are read from `../BookFigures/chapterN/*.png` (Sphinx copies them into `_build/html/_images`), equation numbers are the `\tag{N.m}` written by the generator (so they agree with the printed book), and cross-references between chapters are plain text, as in the notebooks.

To publish on GitHub Pages: `pip install ghp-import && ghp-import -n -p -f _build/html` from this folder pushes the built HTML to the `gh-pages` branch. This is deliberately not automated, since it makes the draft public.

Jupyter Book 2 (the MyST-MD based rewrite, which is what a plain `pip install jupyter-book` now installs) uses a different configuration file (`myst.yml`); `jupyter book init --migrate`-style tools can convert `_config.yml`/`_toc.yml`, but the key-labelled bibliography of `_ext/bibkeys.py` is a Sphinx extension and would have to be replaced.
