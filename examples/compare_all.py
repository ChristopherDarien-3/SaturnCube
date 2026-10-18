"""Run every vector-family model through the same experiment loop."""
import numpy as np
import saturncube as sc

rng = np.random.default_rng(0)
X = rng.normal(size=(1000, 8)).astype(np.float32)
y = (X[:, 0] * X[:, 1] > 0).astype(int) + (X[:, 2] > 0.5).astype(int)
Xtr, Xte, ytr, yte = X[:750], X[750:], y[:750], y[750:]

print(f"{'model':<10} {'params':>8} {'accuracy':>9}")
for name in sc.available("vector"):
    m = sc.create(name, seed=0).fit(Xtr, ytr)
    print(f"{name:<10} {m.n_params:>8} {m.evaluate(Xte, yte)['accuracy']:>9.3f}")
