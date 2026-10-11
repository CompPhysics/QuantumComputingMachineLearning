"""Chapter 2, Secs. 2.8-2.9: the quantum support-vector machine.

`QSVM` is the support-vector machine of svm.py fed with the Gram matrix of
a quantum kernel (feature_maps.py): the quantum computer -- here the
simulator -- produces the kernel entries, the classical SMO solver does the
optimization, Eq. (2-qsvmdual).  As a script the module produces

  * Table 2-qsvmdata: training and test accuracies of classical and quantum
    kernels on the two moons, two circles, XOR and the Iris data;
  * Fig. 2-qsvmmoons: decision regions of the RBF, angle and ZZ kernels;
  * Fig. 2-havlicek and Table 2-havlicek: the data set of Havlicek et al.,
    labelled by the ZZ feature map itself, Eq. (2-havlicekdata), on which
    the matching quantum kernel is exact and the classical kernels are not;
  * Table 2-shots and Fig. 2-shots: the kernel matrix estimated with a
    finite number of shots, its error and the resulting test accuracy.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]      # BookPrograms/
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np

from svm import SVM
from kernels import LinearKernel, RBFKernel
from feature_maps import AngleEncoding, ZZFeatureMap, QuantumKernel


class QSVM:
    """A support-vector machine with a quantum kernel."""

    def __init__(self, feature_map, C=1.0, shots=None, method="adjoint", psd=True, rng=0):
        self.kernel = QuantumKernel(feature_map, shots=shots, method=method, rng=rng)
        self.svm = SVM(C=C)
        self.C, self.psd = C, psd

    @staticmethod
    def nearest_psd(K):
        """Symmetrize K and set its negative eigenvalues to zero: the nearest
        positive-semidefinite matrix in the Frobenius norm, Eq. (2-psdfix)."""
        K = (K + K.T) / 2
        w, V = np.linalg.eigh(K)
        return (V * np.maximum(w, 0.0)) @ V.T

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.X_train = X
        self.K_raw = self.kernel.gram(X)                  # the quantum part
        # a kernel matrix estimated from shots is symmetric only on average
        # and may have small negative eigenvalues; the dual problem is then
        # not concave and SMO need not converge, so project onto the PSD cone
        self.K_train = self.nearest_psd(self.K_raw) if (self.psd and self.kernel.shots) else self.K_raw
        self.svm.fit(X, y, K=self.K_train)                # the classical part
        return self

    def decision_function(self, X):
        K = self.kernel(np.asarray(X, float), self.X_train)
        return self.svm.decision_function(X, K=K)

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.sign(y)))

    @property
    def support(self):
        return self.svm.support


def havlicek_dataset(n=100, delta=0.3, reps=2, rng=5, n_qubits=2):
    """Data labelled by the ZZ feature map itself, after Havlicek et al.:
    x is drawn uniformly from [0, 2 pi)^n, f(x) = <phi(x)| V^dagger Z..Z V |phi(x)>
    with a fixed random unitary V, and y = sign f(x) for the points with
    |f(x)| >= delta (the others are discarded), Eq. (2-havlicekdata)."""
    rng = np.random.default_rng(rng)
    fm = ZZFeatureMap(n_qubits, reps=reps)
    dim = 2 ** n_qubits
    # a Haar-random unitary V from the QR decomposition of a Gaussian matrix
    A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    Q, R = np.linalg.qr(A)
    V = Q @ np.diag(np.diag(R) / np.abs(np.diag(R)))
    ZZ = np.diag([(-1) ** bin(k).count("1") for k in range(dim)]).astype(complex)
    O = V.conj().T @ ZZ @ V
    X, y = [], []
    while len(y) < n:
        x = rng.uniform(0, 2 * np.pi, n_qubits)
        psi = fm.state(x)
        f = float(np.real(np.vdot(psi, O @ psi)))
        if abs(f) >= delta:
            X.append(x); y.append(np.sign(f))
    return np.array(X), np.array(y)


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    from bookplot import savefig, scatter_classes, decision_regions
    from datasets import two_moons, two_circles, xor, iris, train_test_split, Scaler

    C = 1.0
    datasets = {"moons": two_moons(100, 0.15), "circles": two_circles(100),
                "xor": xor(100), "iris (versicolor/virginica)": iris((2, 3), (1, 2))}

    def models(n_features):
        return [("linear", lambda: SVM(LinearKernel(), C=C), "std"),
                ("rbf(gamma=1)", lambda: SVM(RBFKernel(1.0), C=C), "std"),
                ("angle(Y)", lambda: QSVM(AngleEncoding(n_features, "Y"), C=C), "angle"),
                ("angle(Y,reps=2,ent)", lambda: QSVM(AngleEncoding(n_features, "Y", reps=2, entangle=True), C=C), "angle"),
                ("zz(reps=2)", lambda: QSVM(ZZFeatureMap(n_features, reps=2), C=C), "angle"),
                ("zz(reps=2,alpha=0.5)", lambda: QSVM(ZZFeatureMap(n_features, reps=2, alpha=0.5), C=C), "angle")]

    # -------------------------------------------- Table 2-qsvmdata
    print(f"Table 2-qsvmdata: training / test accuracy, C = {C}, 70/30 split")
    print(f"{'data set':28s}" + "".join(f"{m[0]:>22s}" for m in models(2)))
    print(f"{'':28s}" + "".join(f"{'train/test':>22s}" for m in models(2)))
    table = {}
    for name, (X, y) in datasets.items():
        Xtr, Xte, ytr, yte = train_test_split(X, y, 0.3, rng=0)
        row = []
        for mname, make, scaling in models(X.shape[1]):
            sc = Scaler("standard" if scaling == "std" else "angle").fit(Xtr)
            model = make().fit(sc.transform(Xtr), ytr)
            row.append((model.score(sc.transform(Xtr), ytr), model.score(sc.transform(Xte), yte)))
        table[name] = row
        print(f"{name:28s}" + "".join(f"{tr:>11.2f}/{te:<10.2f}" for tr, te in row))

    # -------------------------------------------- Fig. 2-qsvmmoons
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.4), constrained_layout=True)
    short = {"rbf(gamma=1)": "RBF, $\\gamma=1$", "angle(Y)": "angle encoding",
             "zz(reps=2,alpha=0.5)": "ZZ map, $\\alpha=0.5$"}
    for r, (name, (X, y)) in enumerate([("moons", datasets["moons"]), ("circles", datasets["circles"])]):
        Xtr, Xte, ytr, yte = train_test_split(X, y, 0.3, rng=0)
        for c, (mname, make, scaling) in enumerate([m for m in models(2) if m[0] in ("rbf(gamma=1)", "angle(Y)", "zz(reps=2,alpha=0.5)")]):
            sc = Scaler("standard" if scaling == "std" else "angle").fit(Xtr)
            model = make().fit(sc.transform(Xtr), ytr)
            ax = axes[r, c]
            Z = sc.transform(Xtr)
            lims = (Z.min(0) - 0.3, Z.max(0) + 0.3)
            decision_regions(ax, model.decision_function, (lims[0][0], lims[1][0]), (lims[0][1], lims[1][1]), n=70)
            scatter_classes(ax, Z, ytr)
            Zt = sc.transform(Xte)
            ax.scatter(Zt[:, 0], Zt[:, 1], marker="x", c=np.where(yte > 0, "tab:green", "tab:blue"), s=22)
            ax.set_title(f"{short[mname]}: test accuracy {model.score(Zt, yte):.2f}", fontsize=10)
            ax.set_xlabel("$x_1$"); ax.set_ylabel(f"{name}: $x_2$")
    savefig(fig, 2, "qsvmmoons")

    # -------------------------------------------- Havlicek data, Table/Fig. 2-havlicek
    X, y = havlicek_dataset(n=120, delta=0.3, reps=2, rng=5)
    Xtr, Xte, ytr, yte = train_test_split(X, y, 0.25, rng=1)
    print(f"\nTable 2-havlicek: data labelled by the ZZ map (120 points, delta = 0.3, "
          f"{int((y > 0).sum())} with y=+1), C = {C}")
    hav = []
    for mname, make in [("linear", lambda: SVM(LinearKernel(), C=C)),
                        ("rbf(gamma=1)", lambda: SVM(RBFKernel(1.0), C=C)),
                        ("rbf(gamma=0.2)", lambda: SVM(RBFKernel(0.2), C=C)),
                        ("rbf(gamma=5)", lambda: SVM(RBFKernel(5.0), C=C)),
                        ("angle(Y)", lambda: QSVM(AngleEncoding(2, "Y"), C=C)),
                        ("zz(reps=1)", lambda: QSVM(ZZFeatureMap(2, reps=1), C=C)),
                        ("zz(reps=2)", lambda: QSVM(ZZFeatureMap(2, reps=2), C=C)),
                        ("zz(reps=2), C=100", lambda: QSVM(ZZFeatureMap(2, reps=2), C=100.0))]:
        t0 = time.time()
        model = make().fit(Xtr, ytr)
        hav.append((mname, model.score(Xtr, ytr), model.score(Xte, yte), len(model.support)))
        print(f"  {mname:20s} train {hav[-1][1]:.2f}  test {hav[-1][2]:.2f}  "
              f"{hav[-1][3]:3d} support vectors  ({time.time() - t0:.1f} s)")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    for ax, (mname, make) in zip(axes, [("rbf(gamma=1)", lambda: SVM(RBFKernel(1.0), C=C)),
                                        ("zz(reps=2)", lambda: QSVM(ZZFeatureMap(2, reps=2), C=C))]):
        model = make().fit(Xtr, ytr)
        decision_regions(ax, model.decision_function, (0, 2 * np.pi), (0, 2 * np.pi), n=70)
        scatter_classes(ax, Xtr, ytr)
        ax.scatter(Xte[:, 0], Xte[:, 1], marker="x", c=np.where(yte > 0, "tab:green", "tab:blue"), s=22)
        ax.set_title(f"{mname}: test accuracy {model.score(Xte, yte):.2f}")
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.set_xlim(0, 2 * np.pi); ax.set_ylim(0, 2 * np.pi)
    savefig(fig, 2, "havlicek")

    # -------------------------------------------- shots, Table/Fig. 2-shots
    print("\nTable 2-shots: the ZZ kernel of the Havlicek data estimated with shots")
    fm = ZZFeatureMap(2, reps=2)
    K_exact = QuantumKernel(fm).gram(Xtr)
    shot_list = [10, 100, 1000, 10000, 100000]
    rows = []
    for method in ("adjoint", "swap"):
        for shots in shot_list:
            model = QSVM(fm, C=C, shots=shots, method=method, rng=11).fit(Xtr, ytr)
            K = model.K_raw
            err = np.abs(K - K_exact)
            rows.append((method, shots, err.max(), np.sqrt((err ** 2).mean()),
                         np.linalg.eigvalsh((K + K.T) / 2).min(), model.score(Xte, yte)))
            print(f"  {method:8s} {shots:7d} shots: max|dK| {err.max():.4f}  rms {rows[-1][3]:.4f}  "
                  f"min eig {rows[-1][4]:+.3f}  test acc {rows[-1][5]:.2f}")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))
    for method, style in (("adjoint", "o-"), ("swap", "s--")):
        sel = [r for r in rows if r[0] == method]
        axes[0].loglog([r[1] for r in sel], [r[3] for r in sel], style, label=f"{method} test")
        axes[1].semilogx([r[1] for r in sel], [r[5] for r in sel], style, label=f"{method} test")
    s = np.array(shot_list, float)
    axes[0].loglog(s, 0.5 / np.sqrt(s), "k:", label="$1/(2\\sqrt{S})$")
    axes[0].set_xlabel("shots $S$ per kernel entry"); axes[0].set_ylabel("rms error of $K_{ij}$"); axes[0].legend()
    axes[1].set_xlabel("shots $S$ per kernel entry"); axes[1].set_ylabel("test accuracy"); axes[1].legend()
    axes[1].set_ylim(0.4, 1.05)
    savefig(fig, 2, "shots")
    print("figures qsvmmoons, havlicek and shots written")
