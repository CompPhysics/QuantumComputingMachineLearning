"""Small labelled data sets used throughout the book, generated with NumPy
alone so that every program runs without scikit-learn.  Labels are always
y = +1 / -1 (the sign convention of the support-vector machine of
Chapter 2); the class helpers below produce the standard toy problems of
the machine-learning literature (two blobs, two moons, two circles, XOR)
and the Iris flowers, which are stored in ``iris.csv`` next to this file.

    from datasets import two_moons, two_circles, iris, train_test_split, Scaler
"""
import pathlib
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


def two_blobs(n=60, separation=3.0, noise=0.8, rng=0):
    """Two Gaussian clouds, linearly separable for separation >~ 3 noise."""
    rng = np.random.default_rng(rng)
    half = n // 2
    Xa = rng.normal([-separation / 2, 0.0], noise, size=(half, 2))
    Xb = rng.normal([+separation / 2, 0.0], noise, size=(n - half, 2))
    return np.vstack([Xa, Xb]), np.r_[-np.ones(half), np.ones(n - half)]


def two_moons(n=100, noise=0.15, rng=42):
    """Two interleaving half circles (the 'moons' of scikit-learn)."""
    rng = np.random.default_rng(rng)
    n_out, n_in = n // 2, n - n // 2
    t_out = np.linspace(0, np.pi, n_out)
    t_in = np.linspace(0, np.pi, n_in)
    outer = np.c_[np.cos(t_out), np.sin(t_out)]
    inner = np.c_[1 - np.cos(t_in), 1 - np.sin(t_in) - 0.5]
    X = np.vstack([outer, inner]) + rng.normal(0, noise, size=(n, 2))
    y = np.r_[-np.ones(n_out), np.ones(n_in)]
    return X, y


def two_circles(n=100, factor=0.45, noise=0.08, rng=1):
    """A small circle inside a large one."""
    rng = np.random.default_rng(rng)
    n_out, n_in = n // 2, n - n // 2
    t_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    t_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False)
    outer = np.c_[np.cos(t_out), np.sin(t_out)]
    inner = factor * np.c_[np.cos(t_in), np.sin(t_in)]
    X = np.vstack([outer, inner]) + rng.normal(0, noise, size=(n, 2))
    y = np.r_[-np.ones(n_out), np.ones(n_in)]
    return X, y


def xor(n=100, noise=0.25, rng=3):
    """Four Gaussian blobs at the corners of a square, labelled by the XOR
    of the signs of the coordinates: no straight line separates them."""
    rng = np.random.default_rng(rng)
    centres = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]], dtype=float)
    k = n // 4
    X = np.vstack([rng.normal(c, noise, size=(k, 2)) for c in centres])
    y = np.r_[np.ones(2 * k), -np.ones(2 * k)]
    return X, y


def iris(features=(2, 3), classes=(0, 1)):
    """The Iris data (Fisher 1936; 150 flowers, 4 features, 3 species).

    Returns the selected feature columns (default petal length and width)
    and the labels of the two selected species as +1 / -1 (default setosa
    = -1, versicolor = +1).  Column order of iris.csv: sepal length, sepal
    width, petal length, petal width, species (0 setosa, 1 versicolor,
    2 virginica)."""
    data = np.loadtxt(HERE / "iris.csv", delimiter=",", skiprows=1)
    keep = np.isin(data[:, 4], classes)
    X = data[keep][:, list(features)]
    y = np.where(data[keep][:, 4] == classes[0], -1.0, 1.0)
    return X, y


def train_test_split(X, y, test_fraction=0.3, rng=0):
    """A random split into training and test sets."""
    rng = np.random.default_rng(rng)
    idx = rng.permutation(len(y))
    n_test = int(round(test_fraction * len(y)))
    te, tr = idx[:n_test], idx[n_test:]
    return X[tr], X[te], y[tr], y[te]


class Scaler:
    """Affine rescaling of the features, fitted on the training set.

    mode='standard' maps every feature to zero mean and unit variance;
    mode='angle' maps every feature into the interval [lo, hi] (default
    [0, pi]), which is what the angle encodings of Chapter 2 expect."""

    def __init__(self, mode="standard", lo=0.0, hi=np.pi):
        self.mode, self.lo, self.hi = mode, lo, hi

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if self.mode == "standard":
            self.shift, self.scale = X.mean(0), X.std(0)
        else:
            xmin, xmax = X.min(0), X.max(0)
            self.scale = (xmax - xmin) / (self.hi - self.lo)
            self.shift = xmin - self.lo * self.scale
        self.scale = np.where(self.scale == 0, 1.0, self.scale)
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.shift) / self.scale

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        return np.asarray(Z, dtype=float) * self.scale + self.shift


def accuracy(y_true, y_pred):
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


if __name__ == "__main__":
    for name, (X, y) in {"blobs": two_blobs(), "moons": two_moons(),
                         "circles": two_circles(), "xor": xor(),
                         "iris": iris()}.items():
        print(f"{name:8s} {len(y):4d} samples, {X.shape[1]} features, "
              f"{int((y > 0).sum())} with y=+1")
    Xtr, Xte, ytr, yte = train_test_split(*iris(), 0.3)
    assert len(ytr) + len(yte) == 100 and len(yte) == 30
    sc = Scaler("angle").fit(Xtr)
    Z = sc.transform(Xtr)
    assert Z.min() >= -1e-12 and Z.max() <= np.pi + 1e-12
    print("data set checks passed")
