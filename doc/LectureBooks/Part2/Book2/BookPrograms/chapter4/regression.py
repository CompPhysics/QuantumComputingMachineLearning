"""Chapter 4, Sec. 4.7.1: regression with a QNN against a classical network.

Target g(x) = x^3 - 2x^2 + x/2 + 1 on [-2, 2] with Gaussian noise (standard
deviation 0.15), 150 training and 100 test points.  Quantum models: two
qubits, L = 1, 2, 4, 6 re-uploading layers, the encoding angle pi x/4 (the
data cover half a period of the model) and the classical output a f + b
(Table 4-regression); the output of each trained model is decomposed into
its Fourier coefficients (Fig. 4-regression).  Classical reference: the
MLP 1-16-16-1 of common/neuralnet.py.  Runtime about 20 s.
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]      # BookPrograms/
sys.path[:0] = [str(p) for p in root.iterdir() if p.is_dir()]

import numpy as np

from training import QNNRegressor
from neuralnet import MLP


def target(x):
    return x ** 3 - 2 * x ** 2 + 0.5 * x + 1


def data(n_train=150, n_test=100, noise=0.15, rng=42):
    rng = np.random.default_rng(rng)
    x = rng.uniform(-2, 2, n_train + n_test)
    y = target(x) + rng.normal(0, noise, len(x))
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    import bookplot

    x_tr, y_tr, x_te, y_te = data()
    y_scale = y_tr.std(); y_mean = y_tr.mean()            # standardised targets
    enc = lambda x: np.pi * (x / 4.0)[:, None]             # x in [-2,2] -> angle in [-pi/2, pi/2]
    xs = np.linspace(-2, 2, 400)
    print("Table 4-regression: mean squared error on the test set (original units)")
    print(f"{'model':>22}  {'parameters':>10}  {'train MSE':>10}  {'test MSE':>9}  {'time':>6}")
    fits, spectra = {}, {}
    for L in (1, 2, 4, 6):
        t0 = time.time()
        model = QNNRegressor(2, L).fit(enc(x_tr), (y_tr - y_mean) / y_scale, steps=400, lr=0.05, rng=1)
        pred = lambda x: y_scale * model.predict(enc(x)) + y_mean
        mse_tr = np.mean((pred(x_tr) - y_tr) ** 2); mse_te = np.mean((pred(x_te) - y_te) ** 2)
        print(f"{'QNN, n=2, L=' + str(L):>22}  {model.n_params:>10}  {mse_tr:>10.4f}  {mse_te:>9.4f}  {time.time() - t0:>5.1f}s")
        fits[f"QNN, $L={L}$"] = pred(xs)
        k, c = model.qnn.fourier_coefficients(model.params[:model.qnn.n_params])
        spectra[L] = (k, np.abs(c) * abs(model.params[-2]) * y_scale)
    t0 = time.time()
    net = MLP((1, 16, 16, 1), rng=0)
    net.fit((x_tr / 2)[:, None], (y_tr - y_mean) / y_scale, steps=2000, lr=0.01)
    pred_net = lambda x: y_scale * net.forward((x / 2)[:, None])[:, 0] + y_mean
    mse_tr = np.mean((pred_net(x_tr) - y_tr) ** 2); mse_te = np.mean((pred_net(x_te) - y_te) ** 2)
    print(f"{'MLP 1-16-16-1':>22}  {net.n_params:>10}  {mse_tr:>10.4f}  {mse_te:>9.4f}  {time.time() - t0:>5.1f}s")
    print(f"{'noise floor':>22}  {'':>10}  {'':>10}  {0.15 ** 2:>9.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    axes[0].plot(x_tr, y_tr, ".", color="0.6", ms=4, label="training data")
    axes[0].plot(xs, target(xs), "k-", lw=1, label="$g(x)$")
    for name, f in fits.items():
        axes[0].plot(xs, f, "--", label=name)
    axes[0].plot(xs, pred_net(xs), ":", label="MLP")
    axes[0].set_xlabel("$x$"); axes[0].set_ylabel("$y$"); axes[0].legend(fontsize=7, loc="lower right")
    for L, (k, c) in spectra.items():
        axes[1].plot(k[k >= 0], c[k >= 0], "o-", ms=4, label=f"$L={L}$")
    axes[1].set_xlabel("frequency $k$ (period of the model: $x \\in [-4, 4]$)"); axes[1].set_ylabel("$a\\,|c_k|$ (original units)")
    axes[1].set_yscale("log"); axes[1].set_ylim(1e-3, 30); axes[1].legend(fontsize=8)
    axes[1].set_title("Fourier spectrum of the trained models", fontsize=10)
    fig.tight_layout()
    bookplot.savefig(fig, 4, "regression")
    print("figure regression written")
