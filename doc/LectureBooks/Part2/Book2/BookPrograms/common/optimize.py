"""A small BFGS minimiser with a backtracking line search, for the variational
calculations of the QML book (no SciPy on the device VM); a copy of the module
of the same name in the quantum-computing book.

minimize(f, x0, grad=None, tol=1e-10, maxiter=500, history=None) -> (x, f(x), n_iter)
The gradient is taken by central differences when none is supplied; a list
passed as `history` receives f(x) after every iteration.
"""
import numpy as np


def numerical_gradient(f, x, h=1e-6):
    g = np.zeros_like(x, dtype=float)
    for k in range(len(x)):
        e = np.zeros_like(x, dtype=float); e[k] = h
        g[k] = (f(x + e) - f(x - e)) / (2 * h)
    return g


def minimize(f, x0, grad=None, tol=1e-10, maxiter=500, gtol=1e-8, history=None):
    x = np.array(x0, dtype=float)
    n = len(x)
    grad = grad or (lambda y: numerical_gradient(f, y))
    B = np.eye(n)                        # inverse Hessian approximation
    fx, g = f(x), grad(x)
    if history is not None:
        history.append(fx)
    for it in range(1, maxiter + 1):
        if np.linalg.norm(g) < gtol:
            break
        d = -B @ g
        if d @ g >= 0:                   # not a descent direction: reset
            B = np.eye(n); d = -g
        alpha = 1.0
        while True:                      # Armijo backtracking
            xn = x + alpha * d
            fn = f(xn)
            if fn <= fx + 1e-4 * alpha * (g @ d) or alpha < 1e-12:
                break
            alpha *= 0.5
        gn = grad(xn)
        s, y = xn - x, gn - g
        sy = s @ y
        if sy > 1e-14:
            rho = 1.0 / sy
            I = np.eye(n)
            B = (I - rho * np.outer(s, y)) @ B @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        converged = abs(fn - fx) < tol and np.linalg.norm(s) < 1e-8
        x, fx, g = xn, fn, gn
        if history is not None:
            history.append(fx)
        if converged:
            break
    return x, fx, it


if __name__ == "__main__":
    rosen = lambda v: (1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2
    x, fx, it = minimize(rosen, [-1.2, 1.0])
    assert np.allclose(x, [1, 1], atol=1e-5)
    print("BFGS reached", x, "in", it, "iterations")


class Adam:
    """The Adam optimiser of Kingma and Ba (2015), used for the quantum neural
    networks of chapter 4: a gradient step with a running mean m of the
    gradient and a running mean v of its square,

        theta <- theta - lr * m_hat / (sqrt(v_hat) + eps),

    m_hat and v_hat being the bias-corrected means.  `step(theta, grad)`
    returns the updated parameters; the optimiser keeps its own state."""

    def __init__(self, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = self.v = None
        self.t = 0

    def step(self, theta, grad):
        theta, grad = np.asarray(theta, float), np.asarray(grad, float)
        if self.m is None:
            self.m, self.v = np.zeros_like(theta), np.zeros_like(theta)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad ** 2
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        return theta - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def gradient_descent(loss_and_grad, x0, lr=0.05, steps=200, optimizer=None):
    """Run `steps` updates of plain gradient descent (optimizer=None) or of the
    optimiser object given (an `Adam` instance, say) and return the final
    parameters and the loss history."""
    x = np.array(x0, float)
    hist = []
    for _ in range(steps):
        f, g = loss_and_grad(x)
        hist.append(f)
        x = x - lr * g if optimizer is None else optimizer.step(x, g)
    return x, np.array(hist)
