"""Grid behind docs/TUNING.md: 4 models x learning rates / epochs / early stopping x 6 datasets x 5 seeds.

    python scripts/tune_defaults.py [dataset ...]     # appends to ./tune_results.jsonl; needs scikit-learn
"""
import json, sys, time, warnings
import numpy as np
import saturncube as sc
from sklearn.datasets import load_wine, load_breast_cancer, load_digits, load_diabetes, make_classification

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
Xs = rng.normal(size=(1000, 8)).astype(np.float32)
ys = (Xs[:, 0] * Xs[:, 1] > 0).astype(int) + (Xs[:, 2] > 0.5).astype(int)
Xd, yd = load_digits(return_X_y=True); o = np.random.default_rng(1).permutation(len(yd))[:1000]
DATA = {
    "synth3": (Xs, ys),
    "wine": load_wine(return_X_y=True),
    "breast_cancer": load_breast_cancer(return_X_y=True),
    "hard4": make_classification(n_samples=800, n_features=20, n_informative=8, n_classes=4, class_sep=0.6,
                                 flip_y=0.1, random_state=0),
    "digits1000": (Xd[o], yd[o]),
    "diabetes(reg)": load_diabetes(return_X_y=True),
}
MODELS = ["mlp", "kan", "rbf", "random_nn"]
DEFAULT_EPOCHS = {"mlp": 60, "kan": 60, "rbf": 150, "random_nn": 150}
LRS = [1e-3, 3e-3, 1e-2, 3e-2]

def specs():
    out = {}
    for m in MODELS:
        out[f"{m}|default"] = (m, {})
        out[f"{m}|epochs80"] = (m, {"epochs": 80})
        for lr in LRS:
            out[f"{m}|lr={lr:g}"] = (m, {"lr": lr})
        out[f"{m}|ES(p10)"] = (m, {"epochs": 2 * DEFAULT_EPOCHS[m], "patience": 10})
    return out

only = sys.argv[1:] or list(DATA)
for name in only:
    X, y = DATA[name]
    t = time.time()
    res = sc.benchmark(specs(), np.asarray(X, dtype=np.float32), y, seeds=5, show=False, baselines=True)
    row = {"dataset": name, "task": res.task, "metric": res.metric, "seconds": time.time() - t, "scores": {}}
    for label in res.labels:
        row["scores"][label] = {"vals": [None if not np.isfinite(v) else float(v) for v in res.scores[label][res.metric]],
                                "fit_s": float(np.nanmean(res.fit_seconds[label])), "errors": [e[1] for e in res.errors[label]]}
    with open("tune_results.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(name, "done in %.0fs" % row["seconds"], flush=True)
