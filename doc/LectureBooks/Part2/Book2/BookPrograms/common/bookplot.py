"""Plotting conventions shared by the programs of the QML book: figure
directory, matplotlib style, and a helper that saves both PDF (for LaTeX)
and PNG (for the notebooks) of a figure.  Same interface as the module of
the same name in *Quantum Computing for Many-body Physics*."""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]          # doc/Book/BookQML
FIGROOT = ROOT / "BookFigures"

plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "legend.fontsize": 9,
                     "figure.dpi": 110, "savefig.bbox": "tight"})

# the two classes of every binary-classification figure of the book
CLASS_STYLE = {-1: dict(marker="s", color="tab:blue", label="$y=-1$"),
               +1: dict(marker="^", color="tab:green", label="$y=+1$")}


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


def scatter_classes(ax, X, y, **kw):
    """Scatter a two-dimensional labelled data set with the book's markers."""
    for label, style in CLASS_STYLE.items():
        m = y == label
        if m.any():
            ax.scatter(X[m, 0], X[m, 1], s=kw.pop("s", 28), edgecolor="k",
                       linewidth=0.4, **style, **kw)


def decision_regions(ax, predict, xlim, ylim, n=120, levels=(-1e9, 0, 1e9)):
    """Shade the two half spaces of a decision function f on a grid."""
    import numpy as np
    xs = np.linspace(*xlim, n); ys = np.linspace(*ylim, n)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.c_[gx.ravel(), gy.ravel()]
    f = predict(grid).reshape(gx.shape)
    ax.contourf(gx, gy, f, levels=levels, colors=["#cfe0f5", "#d5f0d5"], alpha=0.9)
    ax.contour(gx, gy, f, levels=[0.0], colors="k", linewidths=1.2)
    return gx, gy, f
