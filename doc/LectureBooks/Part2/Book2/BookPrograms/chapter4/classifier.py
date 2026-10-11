"""Chapter 4, Sec. 4.7.2: the variational quantum classifier on the four
two-class problems of chapter 2 (two moons, two circles, XOR, Iris
versicolor/virginica), with the same 70/30 splits as Table 2-qsvmdata.

Two qubits, one feature per qubit rescaled into [0, pi], L = 1 (a single
encoding) and L = 3 re-uploading layers, square loss on the labels +-1,
Adam with parameter-shift gradients; the RBF-kernel and angle-kernel SVMs
of chapter 2 are rerun for comparison.  Produces Table 4-classifier and
Fig. 4-classifier.  Runtime about 40 s.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]      # BookPrograms/
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np

from training import QNNClassifier
from datasets import two_moons, two_circles, xor, iris, train_test_split, Scaler


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    from bookplot import savefig, scatter_classes, decision_regions
    from svm import SVM
    from kernels import RBFKernel
    from feature_maps import AngleEncoding
    from qsvm import QSVM

    datasets = {"two moons": two_moons(100, 0.15), "two circles": two_circles(100),
                "XOR": xor(100), "Iris (versicolor/virginica)": iris((2, 3), (1, 2))}
    configs = [("QNN, $L=1$", dict(n_layers=1)), ("QNN, $L=3$", dict(n_layers=3))]
    print("Table 4-classifier: training / test accuracy, 70/30 split, 300 Adam steps")
    n_par = [QNNClassifier(2, **kw).n_params for _, kw in configs]
    print(f"{'data set':28s}{'RBF SVM':>12}{'angle QSVM':>12}{'QNN L=1 (' + str(n_par[0]) + ')':>14}{'QNN L=3 (' + str(n_par[1]) + ')':>14}")
    fits = {}
    t0 = time.time()
    for name, (X, y) in datasets.items():
        Xtr, Xte, ytr, yte = train_test_split(X, y, 0.3, rng=0)
        row = []
        sc = Scaler("standard").fit(Xtr)
        m = SVM(RBFKernel(1.0), C=1.0).fit(sc.transform(Xtr), ytr)
        row.append((m.score(sc.transform(Xtr), ytr), m.score(sc.transform(Xte), yte)))
        sa = Scaler("angle").fit(Xtr)
        m = QSVM(AngleEncoding(2, "Y"), C=1.0).fit(sa.transform(Xtr), ytr)
        row.append((m.score(sa.transform(Xtr), ytr), m.score(sa.transform(Xte), yte)))
        for label, kw in configs:
            best = None
            for seed in range(3):                        # three random starts, keep the best training loss
                clf = QNNClassifier(2, **kw).fit(sa.transform(Xtr), ytr, steps=300, lr=0.1, rng=seed)
                if best is None or clf.history[-1] < best.history[-1]:
                    best = clf
            row.append((best.score(sa.transform(Xtr), ytr), best.score(sa.transform(Xte), yte)))
            fits[(name, label)] = (best, sa, Xtr, ytr, Xte, yte)
        print(f"{name:28s}" + "".join(f"{tr:>6.2f}/{te:<6.2f}" for tr, te in row[:2])
              + "".join(f"{tr:>8.2f}/{te:<6.2f}" for tr, te in row[2:]))
    print(f"({time.time() - t0:.0f} s)")

    fig, axes = plt.subplots(2, 2, figsize=(8, 6.4))
    for i, name in enumerate(["two moons", "two circles"]):
        for j, (label, kw) in enumerate(configs):
            ax = axes[i, j]
            clf, sa, Xtr, ytr, Xte, yte = fits[(name, label)]
            xlim = (Xtr[:, 0].min() - 0.3, Xtr[:, 0].max() + 0.3)
            ylim = (Xtr[:, 1].min() - 0.3, Xtr[:, 1].max() + 0.3)
            decision_regions(ax, lambda G: clf.decision_function(sa.transform(G)), xlim, ylim)
            scatter_classes(ax, Xtr, ytr)
            ax.scatter(Xte[:, 0], Xte[:, 1], marker="x", c=np.where(yte > 0, "tab:green", "tab:blue"), s=22)
            ax.set_title(f"{name}, {label}: test accuracy {clf.score(sa.transform(Xte), yte):.2f}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    savefig(fig, 4, "classifier")
    print("figure classifier written")
