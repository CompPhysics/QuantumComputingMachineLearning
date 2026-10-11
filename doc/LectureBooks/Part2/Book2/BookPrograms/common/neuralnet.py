"""A small fully connected neural network (multilayer perceptron) in NumPy,
the classical reference model of chapters 4 and 5.

    MLP(layers=(1, 16, 16, 1), activation="tanh", rng=0)
        .forward(X)                -> outputs, shape (N, n_out)
        .loss_and_grad(X, y)       -> mean squared error and its gradient
                                      with respect to the flat parameter vector
                                      (backpropagation)
        .fit(X, y, steps, lr)      -> trains with Adam, returns the loss history
        .n_params

The parameters are kept as one flat vector so that the same optimisers
(common/optimize.py) serve the classical and the quantum models.
"""
import numpy as np

from optimize import Adam


class MLP:
    def __init__(self, layers=(1, 16, 16, 1), activation="tanh", rng=0):
        self.layers = tuple(layers)
        self.activation = activation
        rng = np.random.default_rng(rng)
        self.shapes = [(layers[i + 1], layers[i]) for i in range(len(layers) - 1)]
        theta = []
        for (m, n) in self.shapes:                       # Glorot initialisation
            theta.append(rng.normal(0, np.sqrt(2.0 / (m + n)), m * n))
            theta.append(np.zeros(m))
        self.theta = np.concatenate(theta)

    @property
    def n_params(self):
        return len(self.theta)

    def _unpack(self, theta):
        Ws, bs, k = [], [], 0
        for (m, n) in self.shapes:
            Ws.append(theta[k:k + m * n].reshape(m, n)); k += m * n
            bs.append(theta[k:k + m]); k += m
        return Ws, bs

    def _act(self, z):
        return np.tanh(z) if self.activation == "tanh" else np.maximum(z, 0.0)

    def _dact(self, z):
        return 1.0 - np.tanh(z) ** 2 if self.activation == "tanh" else (z > 0).astype(float)

    def forward(self, X, theta=None):
        theta = self.theta if theta is None else theta
        Ws, bs = self._unpack(theta)
        a = np.atleast_2d(np.asarray(X, float))
        for W, b in zip(Ws[:-1], bs[:-1]):
            a = self._act(a @ W.T + b)
        return a @ Ws[-1].T + bs[-1]                     # linear output layer

    def loss_and_grad(self, X, y, theta=None):
        """Mean squared error (1/N) sum_i |f(x_i) - y_i|^2 and its gradient."""
        theta = self.theta if theta is None else theta
        Ws, bs = self._unpack(theta)
        X = np.atleast_2d(np.asarray(X, float)); y = np.asarray(y, float).reshape(len(X), -1)
        acts, zs, a = [X], [], X
        for W, b in zip(Ws[:-1], bs[:-1]):               # forward pass, keep activations
            z = a @ W.T + b; zs.append(z); a = self._act(z); acts.append(a)
        out = a @ Ws[-1].T + bs[-1]
        diff = out - y
        loss = float(np.mean(np.sum(diff ** 2, axis=1)))
        delta = 2.0 * diff / len(X)                       # dL/d out
        gW, gb = [None] * len(Ws), [None] * len(Ws)
        for l in range(len(Ws) - 1, -1, -1):             # backpropagation
            gW[l] = delta.T @ acts[l]; gb[l] = delta.sum(axis=0)
            if l > 0:
                delta = (delta @ Ws[l]) * self._dact(zs[l - 1])
        grad = np.concatenate([np.concatenate([gW[l].ravel(), gb[l]]) for l in range(len(Ws))])
        return loss, grad

    def fit(self, X, y, steps=2000, lr=0.01, rng=None):
        opt = Adam(lr=lr)
        hist = []
        for _ in range(steps):
            f, g = self.loss_and_grad(X, y)
            hist.append(f)
            self.theta = opt.step(self.theta, g)
        return np.array(hist)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (200, 1)); y = np.sin(3 * X) + 0.5 * X ** 2
    net = MLP((1, 16, 16, 1))
    from optimize import numerical_gradient
    f, g = net.loss_and_grad(X, y)
    gnum = numerical_gradient(lambda t: net.loss_and_grad(X, y, t)[0], net.theta)
    assert np.max(np.abs(g - gnum)) < 1e-6, "backpropagation gradient wrong"
    hist = net.fit(X, y, steps=1500, lr=0.02)
    print(f"MLP with {net.n_params} parameters: loss {hist[0]:.3f} -> {hist[-1]:.2e}")
