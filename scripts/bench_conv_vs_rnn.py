"""Synthetic pattern tasks behind docs/CONV_SEQUENCE.md: cnn1d and tcn against gru and lstm, 5 seeds each.

    python scripts/bench_conv_vs_rnn.py [A] [B] [C] [D] [E]  # writes ./conv_vs_rnn.json and prints the tables (all tasks by default)

Task A (local motif): noise windows (n, T, 2); positives hold a 5-step motif at a random position, and a quarter of the
negatives hold the time-reversed motif as a distractor. Task B (XOR of the ends): the label is the XOR of the sign of
channel 0 at the first and at the last step; in our run no model learned it, so it says nothing about long-range ability.
Task C (memory, window of 96, 3 seeds): the label is the sign of channel 0 at the first step. Task D is the same with
the middle step, to check whether a good score on C comes from remembering or from the window's edge. Task E is task A
with 300 rows instead of 1200.
"""
import json
import warnings

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # run from a clone without installing

import numpy as np

import saturncube as sc

warnings.filterwarnings("ignore")
MOTIF = np.array([[1, 0], [2, 1], [3, 2], [2, 3], [1, 2]], dtype=np.float32)


def task_a(n=1200, T=48, amp=1.0, seed=0):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, T, 2)).astype(np.float32)
    y = r.integers(0, 2, n)
    for i in range(n):
        p = r.integers(0, T - 5)
        if y[i] == 1:
            X[i, p:p + 5] += amp * MOTIF
        elif r.random() < 0.25:
            X[i, p:p + 5] += amp * MOTIF[::-1]
    return X, y


def task_b(n=1200, T=48, seed=1):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, T, 2)).astype(np.float32)
    return X, ((X[:, 0, 0] > 0) ^ (X[:, -1, 0] > 0)).astype(int)


def task_c(n=1200, T=96, seed=2):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, T, 2)).astype(np.float32)
    return X, (X[:, 0, 0] > 0).astype(int)


def task_d(n=1200, T=96, seed=3):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, T, 2)).astype(np.float32)
    return X, (X[:, T // 2, 0] > 0).astype(int)


def task_e(n=300, T=48, seed=4):
    return task_a(n=n, T=T, seed=seed)


SPECS_A = {"cnn1d (max pool)": ("cnn1d", {"readout": "max"}), "cnn1d (mean pool, the default)": ("cnn1d", {}),
           "tcn (max pool)": ("tcn", {"readout": "max"}), "tcn (last step, the default)": ("tcn", {}),
           "gru": ("gru", {}), "lstm": ("lstm", {})}
SPECS_B = dict(SPECS_A, **{"tcn 5 layers (last step)": ("tcn", {"n_layers": 5})})
SPECS_D = {"cnn1d (max pool)": ("cnn1d", {"readout": "max"}), "tcn (last step, field 61)": ("tcn", {}),
           "gru": ("gru", {})}
SPECS_C = {"cnn1d (max pool)": ("cnn1d", {"readout": "max"}), "tcn (last step, field 61)": ("tcn", {}),
           "tcn 5 layers (last step, field 125)": ("tcn", {"n_layers": 5}), "gru": ("gru", {}), "lstm": ("lstm", {})}

TASKS = {"A": ("A: local motif at a random position", task_a, SPECS_A, 5),
         "B": ("B: long-range XOR of the first and last step", task_b, SPECS_B, 5),
         "C": ("C: remember the sign of the first step (window of 96)", task_c, SPECS_C, 3),
         "D": ("D: the sign of the middle step (window of 96): a position check on task C", task_d, SPECS_D, 3),
         "E": ("E: task A with only 300 rows", task_e, SPECS_A, 5)}

if __name__ == "__main__":
    import os
    import sys as _sys
    chosen = [a.upper() for a in _sys.argv[1:]] or list(TASKS)
    out = json.load(open("conv_vs_rnn.json")) if os.path.exists("conv_vs_rnn.json") else {}
    for key in chosen:
        name, make, specs, n_seeds = TASKS[key]
        X, y = make()
        res = sc.benchmark(specs, X, y, seeds=n_seeds, epochs=40, show=False)
        print("\n== task", name)
        print(res)
        out[key] = {"name": name, "rows": res.to_rows(),
                    "scores": {k: {m: v.tolist() for m, v in d.items()} for k, d in res.scores.items()}}
        json.dump(out, open("conv_vs_rnn.json", "w"), default=float)   # saved after every task
