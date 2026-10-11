"""Plotting conventions shared by the book programs: figure directory,
matplotlib style, and a helper that saves both PDF (for LaTeX) and PNG (for
the notebooks) of a figure."""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]          # doc/Book
FIGROOT = ROOT / "BookFigures"

plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "legend.fontsize": 9,
                     "figure.dpi": 110, "savefig.bbox": "tight"})


def figdir(chapter):
    d = FIGROOT / f"chapter{chapter}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def savefig(fig, chapter, name):
    d = figdir(chapter)
    fig.savefig(d / f"{name}.pdf")
    fig.savefig(d / f"{name}.png", dpi=160)
    plt.close(fig)
    return d / f"{name}.pdf"
