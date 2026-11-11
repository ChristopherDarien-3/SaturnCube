"""Follow-up behind docs/TUNING.md: three regression sets and the split-vs-initialisation spread on a hard problem.

    python scripts/tune_followup.py                  # writes ./tune_followup.json; needs scikit-learn
"""
import json, sys, warnings
import numpy as np
import saturncube as sc
from saturncube.evaluation import _make_split
from sklearn.datasets import load_diabetes, make_friedman1, make_regression, make_classification
warnings.filterwarnings("ignore")
out = {}
# ---- A: more regression sets
REG = {"diabetes": load_diabetes(return_X_y=True),
       "friedman1_n500": make_friedman1(n_samples=500, noise=1.0, random_state=0),
       "linear_noisy_n300": make_regression(n_samples=300, n_features=20, n_informative=8, noise=30, random_state=0)}
EP = {"mlp": 60, "kan": 60, "rbf": 150, "random_nn": 150}
specs = {}
for m in EP:
    specs[f"{m}|default"] = (m, {}); specs[f"{m}|lr=0.001"] = (m, {"lr": 1e-3}); specs[f"{m}|lr=0.003"] = (m, {"lr": 3e-3})
    specs[f"{m}|ES(p10)"] = (m, {"epochs": 2 * EP[m], "patience": 10})
    specs[f"{m}|ES(p10,vs0.2)"] = (m, {"epochs": 2 * EP[m], "patience": 10, "val_split": 0.2})
for name, (X, y) in REG.items():
    res = sc.benchmark(specs, np.asarray(X, dtype=np.float32), y, seeds=5, show=False)
    out[f"A:{name}"] = {k: [float(v) for v in res.scores[k]["r2"]] for k in res.labels}
    print("A done", name, flush=True)
# ---- B: where does the seed spread come from? (hard4)
X, y = make_classification(n_samples=800, n_features=20, n_informative=8, n_classes=4, class_sep=0.6, flip_y=0.1, random_state=0)
X = X.astype(np.float32)
fixed_split = _make_split(None, y, "classification", 0.25, 0.15, 0)
tr, va, te = fixed_split
B = {}
for m in EP:
    init_only, split_only = [], []
    for s in range(8):  # vary initialisation, one fixed split
        mod = sc.create(m, seed=s).fit(X[tr], y[tr]); init_only.append(float(np.mean(mod.predict(X[te]) == y[te])))
    for s in range(8):  # vary the split, one fixed initialisation seed
        a, b, c = _make_split(None, y, "classification", 0.25, 0.15, s)
        np_seed = 0
        mod = sc.create(m, seed=np_seed).fit(X[a], y[a]); split_only.append(float(np.mean(mod.predict(X[c]) == y[c])))
    B[m] = {"init_only": init_only, "split_only": split_only}
    print("B done", m, flush=True)
out["B"] = B
json.dump(out, open("tune_followup.json", "w"))
