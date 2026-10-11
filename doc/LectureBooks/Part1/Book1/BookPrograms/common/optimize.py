"""A small BFGS minimiser with a backtracking line search, for the variational
calculations of chapters 3 and 7 (no SciPy on the device VM).

minimize(f, x0, grad=None, tol=1e-10, maxiter=500) -> (x, f(x), n_iter)
The gradient is taken by central differences when none is supplied.
"""
import numpy as np


def numerical_gradient(f, x, h=1e-6):
    g = np.zeros_like(x, dtype=float)
    for k in range(len(x)):
        e = np.zeros_like(x, dtype=float); e[k] = h
        g[k] = (f(x + e) - f(x - e)) / (2 * h)
    return g


def minimize(f, x0, grad=None, tol=1e-10, maxiter=500, gtol=1e-8):
    x = np.array(x0, dtype=float)
    n = len(x)
    grad = grad or (lambda y: numerical_gradient(f, y))
    B = np.eye(n)                        # inverse Hessian approximation
    fx, g = f(x), grad(x)
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
        if converged:
            break
    return x, fx, it


if __name__ == "__main__":
    rosen = lambda v: (1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2
    x, fx, it = minimize(rosen, [-1.2, 1.0])
    assert np.allclose(x, [1, 1], atol=1e-5)
    print("BFGS reached", x, "in", it, "iterations")
