"""Chapter 4, Secs. 4.4 and 4.7: training a QNN on data.

    QNNRegressor(n_qubits, n_layers, reupload=True, affine=True)
        .fit(X, y, steps=300, lr=0.05, rng=0)   mean-squared-error training with
                                                 Adam and parameter-shift gradients
        .predict(X)                               a f(x; theta) + b
        .loss_and_grad(X, y, params)
    QNNClassifier(...)                           the same model with labels y = +-1,
        .predict(X) = sign(a f + b), .score(X, y) = accuracy

The inputs X (N, d) are angles: the caller rescales features into [0, pi]
(classification, Scaler("angle") of common/datasets.py) or maps a scalar
x in [-1, 1] to pi x on every qubit (regression).  With d < n_qubits the
features are repeated cyclically over the qubits.  The classical output
layer a f + b (two parameters, gradients by the chain rule) lets the model
leave the interval [-1, 1] of an expectation value.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]      # BookPrograms/
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np

from qnn import QNN
from optimize import Adam


class QNNRegressor:
    def __init__(self, n_qubits, n_layers, reupload=True, affine=True, rotations=("Y", "Z")):
        self.qnn = QNN(n_qubits, n_layers, reupload=reupload, rotations=rotations)
        self.affine = affine
        self.params = None
        self.history = None

    @property
    def n_params(self):
        return self.qnn.n_params + (2 if self.affine else 0)

    def angles(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        if X.ndim == 2 and X.shape[1] != self.qnn.n:
            X = X[:, np.arange(self.qnn.n) % X.shape[1]]       # repeat features
        return X

    def _split(self, params):
        theta = params[:self.qnn.n_params]
        a, b = (params[-2], params[-1]) if self.affine else (1.0, 0.0)
        return theta, a, b

    def predict(self, X, params=None):
        theta, a, b = self._split(self.params if params is None else params)
        return a * self.qnn.forward(self.angles(X), theta) + b

    def loss_and_grad(self, X, y, params):
        """Mean squared error and its gradient: the chain rule combines the
        parameter-shift Jacobian of f with the derivatives of a f + b."""
        theta, a, b = self._split(params)
        A = self.angles(X)
        f = self.qnn.forward(A, theta)
        r = a * f + b - y
        loss = float(np.mean(r ** 2))
        g_theta = 2.0 * a * (r @ self.qnn.gradient(A, theta)) / len(y)
        if self.affine:
            return loss, np.concatenate([g_theta, [2 * np.mean(r * f), 2 * np.mean(r)]])
        return loss, g_theta

    def fit(self, X, y, steps=300, lr=0.05, rng=0, params0=None):
        y = np.asarray(y, float)
        if params0 is None:
            theta0 = self.qnn.random_parameters(rng)
            params = np.concatenate([theta0, [1.0, 0.0]]) if self.affine else theta0
        else:
            params = np.array(params0, float)
        opt = Adam(lr=lr)
        hist = []
        for _ in range(steps):
            loss, g = self.loss_and_grad(X, y, params)
            hist.append(loss)
            params = opt.step(params, g)
        self.params, self.history = params, np.array(hist)
        return self


class QNNClassifier(QNNRegressor):
    """Labels y = +-1; the square loss (a f + b - y)^2 is minimised and the
    class is the sign of a f + b."""

    def predict(self, X, params=None):
        return np.where(super().predict(X, params) >= 0, 1, -1)

    def decision_function(self, X, params=None):
        return super().predict(X, params)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 40)
    y = np.cos(2 * np.pi * x) + 0.3 * np.sin(np.pi * x)
    model = QNNRegressor(2, 2).fit(np.pi * x[:, None], y, steps=200, lr=0.1)
    print(f"regressor with {model.n_params} parameters: loss {model.history[0]:.3f} -> {model.history[-1]:.2e}")
