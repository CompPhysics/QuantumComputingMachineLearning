"""Render the TikZ figures of a chapter to PNG for the generated notebooks.

Every ``figure`` environment of chapterN.tex that contains a ``tikzpicture``
but no ``\\includegraphics`` is compiled on its own with pdflatex (standalone
class, same TikZ libraries as bookqml.tex) and converted with pdftoppm to
BookFigures/chapterN/<label>.png, where <label> is the figure's ``fig:``
label without the prefix.  Called by tex_to_notebook.py; can also be run as

    python3 BookPrograms/render_tikz.py 2
"""
import os, re, subprocess, sys, tempfile, shutil

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # doc/Book/BookQML

STANDALONE = r"""\documentclass[border=4pt]{standalone}
\usepackage{amsmath,amssymb}
\usepackage{tikz}
\usetikzlibrary{arrows,snakes,shapes,calc,decorations.pathreplacing,positioning}
\begin{document}
%s
\end{document}
"""


def strip_comments(s):
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", l) for l in s.split("\n"))


def render_chapter(chapter, dpi=160):
    tex = strip_comments(open(f"{SRC}/chapter{chapter}.tex").read())
    outdir = f"{SRC}/BookFigures/chapter{chapter}"
    os.makedirs(outdir, exist_ok=True)
    done = []
    for fm in re.finditer(r"\\begin\{figure\}(.*?)\\end\{figure\}", tex, re.S):
        inner = fm.group(1)
        if "tikzpicture" not in inner or "includegraphics" in inner:
            continue
        lm = re.search(r"\\label\{fig:([A-Za-z0-9_-]+)\}", inner)
        tm = re.search(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", inner, re.S)
        if not (lm and tm):
            continue
        name = lm.group(1)
        target = f"{outdir}/{name}.png"
        if os.path.exists(target) and os.path.getmtime(target) > os.path.getmtime(f"{SRC}/chapter{chapter}.tex"):
            done.append(target); continue
        with tempfile.TemporaryDirectory() as tmp:
            open(f"{tmp}/fig.tex", "w").write(STANDALONE % tm.group(0))
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "fig.tex"], cwd=tmp,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if not os.path.exists(f"{tmp}/fig.pdf"):
                print(f"  could not render fig:{name}"); continue
            subprocess.run(["pdftoppm", "-r", str(dpi), "-png", "-singlefile", "fig.pdf", "fig"], cwd=tmp,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            shutil.copy(f"{tmp}/fig.png", target)
            done.append(target)
    return done


if __name__ == "__main__":
    for c in sys.argv[1:] or range(1, 8):
        if os.path.exists(f"{SRC}/chapter{c}.tex"):
            for f in render_chapter(int(c)):
                print(f)
