"""Chapter 4, Sec. 4.8: physics-informed quantum neural networks (Q-PINNs).

A Q-PINN approximates the solution u(x) of a differential equation by the
circuit output u_theta(x) = f(x; theta) of a QNN (chapter 4, Eq. (4-model))
whose encoding angles are linear in the coordinates, phi_q = sum_i s_qi x_i.
Everything that enters the loss -- u_theta at the collocation points, its
derivatives, boundary and initial values -- is a *linear functional* of the
circuit output, hence a weighted sum of circuit evaluations at shifted
inputs (Eq. (4-functional)), and its gradient with respect to theta is the
same weighted sum of parameter-shift gradients.  The classes:

    QPINN(qnn, scales)            scales[q, i]: the angle of qubit q is sum_i scales[q,i] x_i
        .functional(X, deriv=())  the functional u -> d^k u / dx_i dx_j ... at the points X
                                  (deriv = (i,), (i, i), (i, j), ...) built with the
                                  general shift rule, Eq. (4-generalshift), one
                                  variable at a time
        .functional_gates(X, deriv)   the same from the per-gate product rule,
                                  Eq. (4-productrule) (first and second order; used
                                  to verify the general rule)
    Functional                    points, coefficient matrix and constant;
                                  .value(theta), .jacobian(theta); may be added,
                                  subtracted and multiplied by per-point weights
    residual_loss(R)              mean-square loss of a residual functional and its gradient
    ritz_loss(du, u, f, w)        the energy functional sum_k w_k [ (du_k)^2/2 - f_k u_k ]
    train(loss_and_grad, theta0, method="bfgs"|"adam", ...)  -> theta, loss history

The script verifies the derivative rules against finite differences
(Table 4-derivatives).  Runtime 1 s.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]      # BookPrograms/
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np

from qnn import QNN, trig_weights


class Functional:
    """F[u](x_k) = sum_j C[k, j] u(points[j]) + const[k] for the circuit u = f(.; theta)."""
    __array_ufunc__ = None            # so that numpy arrays defer to __rmul__, __rsub__

    def __init__(self, qnn, points, C, const=None):
        self.qnn = qnn
        self.points = np.asarray(points, float)         # (M, n_enc, n)
        self.C = np.asarray(C, float)                   # (N, M)
        self.const = np.zeros(self.C.shape[0]) if const is None else np.asarray(const, float)

    @property
    def N(self):
        return self.C.shape[0]

    def value(self, theta):
        return self.C @ self.qnn.forward(self.points, theta) + self.const

    def jacobian(self, theta):
        return self.C @ self.qnn.gradient(self.points, theta)

    # ---- algebra
    def _combine(self, other, sign):
        if np.isscalar(other):
            return Functional(self.qnn, self.points, self.C, self.const + sign * other)
        if isinstance(other, np.ndarray):
            return Functional(self.qnn, self.points, self.C, self.const + sign * other)
        a, b = self, other
        if a.N == 1 and b.N > 1:
            a = a.broadcast(b.N)
        if b.N == 1 and a.N > 1:
            b = b.broadcast(a.N)
        return Functional(a.qnn, np.concatenate([a.points, b.points]),
                          np.hstack([a.C, sign * b.C]), a.const + sign * b.const)

    def __add__(self, other):
        return self._combine(other, +1.0)

    def __radd__(self, other):
        return self._combine(other, +1.0)

    def __sub__(self, other):
        return self._combine(other, -1.0)

    def __rsub__(self, other):
        return (-1.0 * self)._combine(other, +1.0)

    def __mul__(self, w):
        """Row-wise weights: a scalar or an array (N,); a one-row functional
        multiplied by an array of N weights becomes an N-row functional."""
        w = np.asarray(w, float)
        if w.ndim == 0:
            return Functional(self.qnn, self.points, w * self.C, w * self.const)
        f = self.broadcast(len(w)) if self.N == 1 else self
        return Functional(f.qnn, f.points, w[:, None] * f.C, w * f.const)

    __rmul__ = __mul__

    def __neg__(self):
        return -1.0 * self

    def broadcast(self, N):
        assert self.N == 1
        return Functional(self.qnn, self.points, np.repeat(self.C, N, axis=0), np.repeat(self.const, N))


class QPINN:
    def __init__(self, qnn, scales):
        self.qnn = qnn
        self.scales = np.atleast_2d(np.asarray(scales, float))   # (n, d)
        assert self.scales.shape[0] == qnn.n
        self.d = self.scales.shape[1]

    def angles(self, X):
        """(N, d) coordinates -> (N, n_enc, n) encoding angles."""
        X = np.atleast_2d(np.asarray(X, float))
        a = X @ self.scales.T
        return np.repeat(a[:, None, :], self.qnn.n_enc, axis=1)

    def _scale_and_degree(self, i):
        s = self.scales[:, i]
        nz = s[s != 0]
        assert nz.size > 0 and np.allclose(nz, nz[0]), "every gate carrying x_i must use the same scale"
        return nz[0], self.qnn.n_enc * nz.size

    def functional(self, X, deriv=()):
        """The functional u -> (prod_{i in deriv} d/dx_i) u evaluated at the rows
        of X, by the general shift rule applied to one variable at a time."""
        X = np.atleast_2d(np.asarray(X, float))
        N = X.shape[0]
        # the stencil: a list of (weight, shift vector in x) pairs
        stencil = [(1.0, np.zeros(self.d))]
        for i in sorted(set(deriv)):
            order = list(deriv).count(i)
            s, G = self._scale_and_degree(i)
            w, phi = trig_weights(G, order)
            new = []
            for (c, dx) in stencil:
                for wm, pm in zip(w, phi):
                    if abs(wm) < 1e-14:
                        continue
                    d2 = dx.copy(); d2[i] += pm / s
                    new.append((c * wm * s ** order, d2))
            stencil = new
        M = len(stencil)
        points = np.concatenate([self.angles(X + dx) for (_, dx) in stencil])   # (M*N, n_enc, n)
        C = np.zeros((N, M * N))
        for j, (c, _) in enumerate(stencil):
            C[np.arange(N), j * N + np.arange(N)] = c
        return Functional(self.qnn, points, C)

    def functional_gates(self, X, deriv=()):
        """The same functional from the product rule over the encoding gates,
        Eq. (4-productrule): first and second derivatives only."""
        X = np.atleast_2d(np.asarray(X, float))
        N = X.shape[0]
        base = self.angles(X)
        gates = lambda i: [(l, q) for l in range(self.qnn.n_enc) for q in range(self.qnn.n)
                           if self.scales[q, i] != 0]

        def shifted(shifts):
            a = base.copy()
            for (l, q), delta in shifts:
                a[:, l, q] += delta
            return a

        terms = []                                       # (coefficient, angles)
        if len(deriv) == 0:
            terms.append((1.0, base))
        elif len(deriv) == 1:
            i = deriv[0]
            for g in gates(i):
                s = self.scales[g[1], i]
                terms.append((s / 2, shifted([(g, np.pi / 2)])))
                terms.append((-s / 2, shifted([(g, -np.pi / 2)])))
        elif len(deriv) == 2:
            i, j = deriv
            for g in gates(i):
                for h in gates(j):
                    s = self.scales[g[1], i] * self.scales[h[1], j]
                    if g == h:
                        terms.append((s / 4, shifted([(g, np.pi)])))
                        terms.append((-s / 2, base))
                        terms.append((s / 4, shifted([(g, -np.pi)])))
                    else:
                        for sg in (1, -1):
                            for sh in (1, -1):
                                terms.append((sg * sh * s / 4, shifted([(g, sg * np.pi / 2), (h, sh * np.pi / 2)])))
        else:
            raise ValueError("first and second derivatives only")
        M = len(terms)
        points = np.concatenate([a for (_, a) in terms])
        C = np.zeros((N, M * N))
        for j, (c, _) in enumerate(terms):
            C[np.arange(N), j * N + np.arange(N)] = c
        return Functional(self.qnn, points, C)


def residual_loss(R):
    """L(theta) = (1/N) sum_k R[u_theta](x_k)^2 and its gradient."""
    def loss_and_grad(theta):
        r = R.value(theta)
        return float(np.mean(r ** 2)), 2.0 * r @ R.jacobian(theta) / len(r)
    return loss_and_grad


def ritz_loss(du, u, f, w):
    """E(theta) = sum_k w_k [ (du_k)^2 / 2 - f_k u_k ], Eq. (4-ritz), and its gradient."""
    def loss_and_grad(theta):
        d, v = du.value(theta), u.value(theta)
        E = float(np.sum(w * (0.5 * d ** 2 - f * v)))
        g = (w * d) @ du.jacobian(theta) - (w * f) @ u.jacobian(theta)
        return E, g
    return loss_and_grad


def train(loss_and_grad, theta0, method="bfgs", steps=300, lr=0.05):
    """Minimise a loss with BFGS (common/optimize.py, gradients from the shift
    rules; the history holds the loss after every iteration) or with Adam."""
    from optimize import minimize, Adam, gradient_descent
    if method == "bfgs":
        hist = []
        theta, f, it = minimize(lambda t: loss_and_grad(t)[0], theta0, grad=lambda t: loss_and_grad(t)[1],
                                maxiter=steps, history=hist)
        return theta, np.array(hist)
    return gradient_descent(loss_and_grad, theta0, steps=steps, optimizer=Adam(lr=lr))


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    print("Table 4-derivatives: derivatives of the circuit output by the shift rules")
    print(f"{'circuit':>22}  {'derivative':>12}  {'general rule':>13}  {'per-gate rule':>13}  {'finite diff.':>13}  {'evaluations':>24}")
    for n, L in [(2, 1), (2, 3), (3, 2)]:
        qnn = QNN(n, L)
        theta = qnn.random_parameters(rng)
        qp = QPINN(qnn, np.full((n, 1), np.pi))        # phi_q = pi x on every qubit
        x = np.array([[0.37]])
        u = lambda xv: qnn.forward(qp.angles(np.atleast_2d(xv)), theta)[0]
        h = 1e-4
        fd = {(0,): (u(x + h) - u(x - h)) / (2 * h),
              (0, 0): (u(x + h) - 2 * u(x) + u(x - h)) / h ** 2}
        for deriv, name in [((0,), "du/dx"), ((0, 0), "d2u/dx2")]:
            F1 = qp.functional(x, deriv); F2 = qp.functional_gates(x, deriv)
            v1, v2 = F1.value(theta)[0], F2.value(theta)[0]
            print(f"{f'n={n}, L={L} (G={qp._scale_and_degree(0)[1]})':>22}  {name:>12}  {v1:>13.6f}  {v2:>13.6f}  {fd[deriv]:>13.6f}"
                  f"  {len(F1.points):>5} (general) {len(F2.points):>5} (gates)")
            assert abs(v1 - v2) < 1e-9 and abs(v1 - fd[deriv]) < 1e-4
    # two variables: x on qubits 0,1 and t on qubit 2
    qnn = QNN(3, 2); theta = qnn.random_parameters(rng)
    qp = QPINN(qnn, [[np.pi, 0], [np.pi, 0], [0, np.pi]])
    X = np.array([[0.3, 0.1], [0.7, 0.25]])
    u = lambda xv, tv: qnn.forward(qp.angles([[xv, tv]]), theta)[0]
    h = 1e-4
    for deriv, fd in [((1,), [(u(x, t + h) - u(x, t - h)) / (2 * h) for x, t in X]),
                      ((0, 0), [(u(x + h, t) - 2 * u(x, t) + u(x - h, t)) / h ** 2 for x, t in X]),
                      ((0, 1), [(u(x + h, t + h) - u(x + h, t - h) - u(x - h, t + h) + u(x - h, t - h)) / (4 * h * h) for x, t in X])]:
        v = qp.functional(X, deriv).value(theta)
        vg = qp.functional_gates(X, deriv).value(theta)
        assert np.allclose(v, fd, atol=1e-3) and np.allclose(v, vg, atol=1e-9), (deriv, v, fd)
    print("two-variable derivatives (x on two qubits, t on one) verified")
    # the gradient of a residual loss against finite differences
    from optimize import numerical_gradient
    xs = np.linspace(0.1, 0.9, 5)[:, None]
    qp = QPINN(QNN(2, 2), np.full((2, 1), np.pi)); theta = qp.qnn.random_parameters(rng)
    R = qp.functional(xs, (0, 0)) + np.pi ** 2 * np.sin(np.pi * xs[:, 0])
    lg = residual_loss(R)
    g = lg(theta)[1]; gn = numerical_gradient(lambda t: lg(t)[0], theta)
    assert np.max(np.abs(g - gn)) < 1e-6
    print(f"gradient of the residual loss: max |shift rule - finite difference| = {np.max(np.abs(g - gn)):.1e}")
