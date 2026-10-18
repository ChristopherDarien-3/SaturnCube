"""Same MLP, same data, different optimizers and learning rules."""
import numpy as np
import saturncube as sc

rng = np.random.default_rng(0)
X = rng.normal(size=(1000, 8)).astype(np.float32)
y = (X[:, 0] * X[:, 1] > 0).astype(int) + (X[:, 2] > 0.5).astype(int)
Xtr, Xte, ytr, yte = X[:750], X[750:], y[:750], y[750:]

print(f"{'setup':<16} {'accuracy':>9}")
for opt in sc.OPTIMIZERS:
    kw = {"batch_size": 750, "epochs": 30} if opt == "lbfgs" else {}
    m = sc.MLP(optimizer=opt, seed=0, **kw).fit(Xtr, ytr)
    print(f"{'mlp+' + opt:<16} {m.evaluate(Xte, yte)['accuracy']:>9.3f}")
for name in ["fa", "dfa"]:
    m = sc.create(name, seed=0).fit(Xtr, ytr)
    print(f"{name:<16} {m.evaluate(Xte, yte)['accuracy']:>9.3f}")
