"""Fair model comparison: group-disjoint splits, several seeds, automatic baselines, a printed table.

    result = sc.benchmark(["mlp", "random_forest", "kan"], X, y, groups=source_ids, seeds=5)

Every model sees exactly the same train / validation / test rows for a given seed. Test rows are never
passed to fit(). When ``groups`` is given, whole groups (images, patients, theorem families, ...) go to
one side of the split, so rows that share a source cannot leak between train and test.
"""
from __future__ import annotations

import difflib
import inspect
import time as time_
import warnings

import numpy as np

from .base import REGISTRY, BaseModel, warmup
from .calibration import (_reliability_rows, brier_score, expected_calibration_error, format_reliability, log_loss,
                          reliability_table)
from .data import split_indices

CLASSIFICATION_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "log_loss", "brier", "ece")
REGRESSION_METRICS = ("r2", "rmse", "mae")
_LOWER_IS_BETTER = {"rmse", "mae", "log_loss", "brier", "ece"}


# ----------------------------------------------------------------------------- input checks ---
def _check_inputs(X, y, groups, task):
    X = np.asarray(X)
    y = np.asarray(y)
    if X.dtype.kind not in "fiub":
        raise ValueError("X must be a numeric array (got dtype %s). For raw tables use "
                         "sc.load_dataset(...).prepare(...) first." % X.dtype)
    if X.dtype.kind == "f" and not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values; clean or impute them first "
                         "(sc.load_dataset(...).prepare(...) does this).")
    if X.ndim < 2:
        raise ValueError("X must have shape (n_rows, ...features); got %s" % (X.shape,))
    if y.ndim != 1:
        raise ValueError("y must be 1-D (one target per row); got shape %s" % (y.shape,))
    if len(X) != len(y):
        raise ValueError("X has %d rows but y has %d" % (len(X), len(y)))
    if len(y) < 10:
        raise ValueError("need at least 10 rows to benchmark; got %d" % len(y))
    if y.dtype.kind == "f" and not np.isfinite(y).all():
        raise ValueError("y contains NaN or infinite values")
    if y.dtype == object:
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in y):
            raise ValueError("y contains missing values")
        y = y.astype(str)
    if task not in ("auto", "classification", "regression"):
        raise ValueError("task must be 'auto', 'classification' or 'regression'")
    if task == "auto":
        task = "regression" if y.dtype.kind == "f" else "classification"
    if task == "classification" and len(np.unique(y)) < 2:
        raise ValueError("classification needs at least two classes in y")
    if groups is not None:
        groups = np.asarray(groups)
        if groups.ndim != 1 or len(groups) != len(y):
            raise ValueError("groups must be 1-D with one entry per row (%d); got shape %s"
                             % (len(y), groups.shape))
        if groups.dtype == object:
            groups = groups.astype(str)
    return X, y, groups, task


def _seed_list(seeds):
    if isinstance(seeds, (int, np.integer)):
        if seeds < 1:
            raise ValueError("seeds must be at least 1")
        return list(range(int(seeds)))
    out = [int(s) for s in seeds]
    if not out:
        raise ValueError("seeds is empty")
    return out


# ------------------------------------------------------------------------------------ splits ---
def _group_split(groups, y, task, test_size, val_size, seed, max_tries=200):
    """Group-disjoint (train, val, test) row indices. Classification keeps every class in the training groups."""
    uniq, inv = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inv)
    n, g = len(inv), len(uniq)
    need = 3 if val_size > 0 else 2
    if g < need:
        raise ValueError("%d distinct groups are too few for a train/%stest split; need at least %d"
                         % (g, "val/" if val_size > 0 else "", need))
    n_test = max(1, round(n * test_size))
    n_val = round(n * val_size) if val_size > 0 else 0
    classes = set(np.unique(y).tolist()) if task == "classification" else None
    for attempt in range(max_tries):
        order = np.random.default_rng([seed, attempt]).permutation(g)
        i, got, test_g = 0, 0, []
        while got < n_test and i < g - (need - 1):
            test_g.append(order[i]); got += sizes[order[i]]; i += 1
        val_g, got = [], 0
        if val_size > 0:
            while got < max(1, n_val) and i < g - 1:
                val_g.append(order[i]); got += sizes[order[i]]; i += 1
        train_g = order[i:]
        tr = np.flatnonzero(np.isin(inv, train_g))
        va = np.flatnonzero(np.isin(inv, val_g))
        te = np.flatnonzero(np.isin(inv, test_g))
        if classes is not None and set(np.unique(y[tr]).tolist()) != classes:
            continue
        return tr, va, te
    raise ValueError("could not find a group-disjoint split in which every class appears in the training "
                     "groups; some classes live in too few groups for this test_size")


def _make_split(groups, y, task, test_size, val_size, seed):
    if groups is not None:
        tr, va, te = _group_split(groups, y, task, test_size, val_size, seed)
        if (set(groups[tr]) & set(groups[te])) or (set(groups[tr]) & set(groups[va])) \
                or (set(groups[va]) & set(groups[te])):
            raise RuntimeError("internal error: a group landed on both sides of the split")
        return tr, va, te
    tr, va, te = split_indices(len(y), test_size, val_size, seed=seed,
                               y=y if task == "classification" else None)
    return np.asarray(tr, dtype=int), np.asarray(va, dtype=int), np.asarray(te, dtype=int)


def _rows_of(value, n, name):
    if isinstance(value, (float, np.floating)) and 0 < value < 1:
        return max(1, int(round(n * value)))
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool) and value >= 0:
        return int(value)
    if value == 0:
        return 0
    raise ValueError("%s must be a fraction in (0, 1) of the rows or a whole number of rows" % name)


def walk_forward_splits(n, n_folds=5, test_size=0.1, val_size=0.1, purge=0, embargo=0, window=None, min_train=10):
    """Walk-forward (forward-chaining) folds over n time-ordered rows: a list of dicts with row positions
    ``train``, ``val`` and ``test``. The test blocks are consecutive and cover the last ``n_folds * test_size`` of
    the series; each fold trains on the past only:

        |<- train ->| purge |<- val ->| purge |<- test ->|   (then the next fold's blocks slide forward)

    test_size, val_size  a fraction of n, or a number of rows (val_size 0 gives no validation block).
    purge     rows dropped before the validation block and before the test block. If a row's features look back
              ``window`` rows and its label looks ahead ``horizon`` rows, use purge >= window + horizon - 1, otherwise
              the last training rows share bars with the first validation or test rows.
    embargo   rows dropped from every later fold's train and validation sets, directly after each earlier test block
              (a conservative extra on top of purge; 0 disables it).
    window    None: expanding training window. An int: only the last ``window`` rows before the purge gap (rolling).
    """
    if n_folds < 1:
        raise ValueError("n_folds must be at least 1")
    if purge < 0 or embargo < 0 or (window is not None and window < 1):
        raise ValueError("purge and embargo must be >= 0 and window None or >= 1")
    length, val_rows = _rows_of(test_size, n, "test_size"), _rows_of(val_size, n, "val_size")
    ends = [n - (n_folds - j) * length + length for j in range(n_folds)]  # end of each fold's test block
    folds = []
    for k in range(n_folds):
        t0, t1 = ends[k] - length, ends[k]
        v1 = t0 - purge
        v0 = v1 - val_rows
        train_end = (v0 - purge) if val_rows else (t0 - purge)
        train_start = 0 if window is None else max(0, train_end - window)
        if train_end - train_start < min_train:
            raise ValueError("fold %d would train on only %d rows (need %d): %d folds of %d test rows, validation %d "
                             "rows and purge %d leave too little history; use fewer folds, a smaller test_size or "
                             "less purge" % (k, max(0, train_end - train_start), min_train, n_folds, length, val_rows, purge))
        train, val = np.arange(train_start, train_end), (np.arange(v0, v1) if val_rows else np.arange(0))
        if embargo and k:
            banned = np.concatenate([np.arange(ends[j], ends[j] + embargo) for j in range(k)])
            train, val = np.setdiff1d(train, banned), np.setdiff1d(val, banned)
        folds.append({"train": train, "val": val, "test": np.arange(t0, t1)})
    return folds


# ---------------------------------------------------------------------------------- metrics ---
def _classification_metrics(y, p):
    acc = float(np.mean(p == y))
    present = np.unique(y)
    recalls = [float(np.mean(p[y == c] == c)) for c in present]
    f1s = []
    for c in np.union1d(present, np.unique(p)):
        tp = int(np.sum((p == c) & (y == c)))
        fp = int(np.sum((p == c) & (y != c)))
        fn = int(np.sum((p != c) & (y == c)))
        f1s.append(2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 0.0)
    return {"accuracy": acc, "balanced_accuracy": float(np.mean(recalls)), "macro_f1": float(np.mean(f1s))}


def _proba_metrics(y, proba, classes):
    return {"log_loss": log_loss(y, proba, classes), "brier": brier_score(y, proba, classes),
            "ece": expected_calibration_error(y, proba, classes=classes)}


def _regression_metrics(y, p):
    y, p = y.astype(float), np.asarray(p, dtype=float).reshape(len(y))
    err = p - y
    sst = float(np.sum((y - y.mean()) ** 2))
    return {"r2": float(1 - np.sum(err ** 2) / sst) if sst > 1e-12 else float("nan"),
            "rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err)))}


# ------------------------------------------------------------------------------- baselines ---
class _Majority:
    """Classification: predicts the most frequent training class and gives the training class frequencies as its
    probabilities (the class-prior baseline). Regression: predicts the training mean."""

    def __init__(self, task):
        self.task = task

    def fit(self, X, y):
        if self.task == "regression":
            self.value_ = float(np.mean(y))
        else:
            vals, counts = np.unique(y, return_counts=True)
            self.classes_, self.prior_ = vals, counts / counts.sum()
            self.value_ = vals[int(np.argmax(counts))]
        return self

    def predict(self, X):
        return np.full(len(X), self.value_)

    def predict_proba(self, X):
        return np.tile(self.prior_, (len(X), 1))


class _Flat:
    """Wraps a scikit-learn estimator so it accepts arrays of any rank (flattened to 2-D)."""

    def __init__(self, est):
        self.est = est

    def fit(self, X, y):
        self.est.fit(X.reshape(len(X), -1), y)
        return self

    def predict(self, X):
        return self.est.predict(X.reshape(len(X), -1))

    def predict_proba(self, X):
        return self.est.predict_proba(X.reshape(len(X), -1))

    @property
    def classes_(self):
        return self.est.classes_


def _sklearn_baseline(task):
    try:
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None
    est = LogisticRegression(max_iter=2000) if task == "classification" else Ridge(alpha=1.0)
    return lambda seed: _Flat(make_pipeline(StandardScaler(), est))


# ----------------------------------------------------------------------------- model specs ---
def _accepts(callable_, name):
    target = callable_.__init__ if isinstance(callable_, type) else callable_
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _instantiate(cls, kwargs, seed, task):
    kw = dict(kwargs)
    if "seed" not in kw and _accepts(cls, "seed"):
        kw["seed"] = seed
    if task != "auto" and "task" not in kw and _accepts(cls, "task"):
        kw["task"] = task
    return cls(**kw)


def _normalise_models(models, task_arg):
    if isinstance(models, dict):
        items = list(models.items())
    elif isinstance(models, (str, type, BaseModel)) or callable(models) \
            or (isinstance(models, tuple) and len(models) == 2 and isinstance(models[0], str)
                and isinstance(models[1], dict)):
        items = [(None, models)]
    else:
        items = [(None, m) for m in models]
    if not items:
        raise ValueError("models is empty")
    out, used = [], {}
    for label, spec in items:
        if isinstance(spec, BaseModel) or (hasattr(spec, "fit") and hasattr(spec, "predict")
                                           and not isinstance(spec, type)):
            raise TypeError("pass a registry name, (name, kwargs), a class, or a function seed -> fresh model; "
                            "a fitted or shared instance cannot be re-created for every seed")
        kwargs = {}
        if isinstance(spec, tuple):
            if len(spec) != 2 or not isinstance(spec[0], str) or not isinstance(spec[1], dict):
                raise TypeError("a tuple model spec must be (name, kwargs_dict)")
            spec, kwargs = spec
        if isinstance(spec, str):
            key = spec.lower()
            if key not in REGISTRY:
                close = difflib.get_close_matches(key, list(REGISTRY), n=3)
                raise ValueError("unknown model %r%s; see sc.available()" % (
                    spec, " (did you mean %s?)" % ", ".join(map(repr, close)) if close else ""))
            cls = REGISTRY[key]
            factory = (lambda seed, cls=cls, kw=kwargs: _instantiate(cls, kw, seed, task_arg))
            label = label or spec
        elif isinstance(spec, type):
            factory = (lambda seed, cls=spec, kw=kwargs: _instantiate(cls, kw, seed, task_arg))
            label = label or spec.__name__
        elif callable(spec):
            factory = spec
            label = label or getattr(spec, "__name__", "model")
        else:
            raise TypeError("cannot interpret model spec %r" % (spec,))
        used[label] = used.get(label, 0) + 1
        if used[label] > 1:
            label = "%s#%d" % (label, used[label])
        out.append((label, factory))
    return out


def _fit(model, Xtr, ytr, Xva, yva, epochs):
    params = inspect.signature(model.fit).parameters
    kw = {}
    if epochs is not None and "epochs" in params:
        kw["epochs"] = epochs
    if Xva is not None and len(Xva) and "val_data" in params:
        kw["val_data"] = (Xva, yva)
    model.fit(Xtr, ytr, **kw)


# ------------------------------------------------------------------------------------ result ---
def _std(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.std(a, ddof=1)) if len(a) > 1 else float("nan")


def _mean(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if len(a) else float("nan")


def _fmt(mean, std, metric):
    if not np.isfinite(mean):
        return "n/a"
    spec = ".4g" if metric in _LOWER_IS_BETTER else ".3f"
    s = format(mean, spec)
    return s if not np.isfinite(std) else "%s ± %s" % (s, format(std, spec))


class BenchmarkResult:
    """What sc.benchmark returns. print(result) shows the table; the raw numbers are kept per run.

    A run is one model fitted on one split with one seed. Runs are grouped into units: a seed for random and grouped
    splits, a fold for walk-forward (the seeds inside a fold are averaged first). The table's mean ± std is over units.

    Attributes: task, metric, seeds, scores {label: {metric: array per run, NaN where a run failed or the model has no
    predict_proba}}, fit_seconds {label: array}, errors {label: [(seed, message)]}, baselines (labels), splits
    {seed (or fold index for walk-forward): {"train", "val", "test"} row indices}, grouped (bool), notes (list of str),
    run_units, walk (walk-forward settings or None), predictions {label: [{"y", "proba", "classes"} per run]}.
    Methods: to_rows(), to_dataframe(), fold_table() (walk-forward), reliability(label).
    """

    def __init__(self, task, metric, seeds, labels, baselines, scores, fit_seconds, errors, splits,
                 grouped, n_groups, shape, notes, majority_share=None, *, run_units=None, walk=None, predictions=None):
        self.task, self.metric, self.seeds = task, metric, seeds
        self.labels, self.baselines = labels, baselines
        self.scores, self.fit_seconds, self.errors = scores, fit_seconds, errors
        self.splits, self.grouped, self.n_groups = splits, grouped, n_groups
        self.shape, self.notes, self.majority_share = shape, notes, majority_share
        n_runs = len(next(iter(fit_seconds.values()))) if fit_seconds else 0
        self.run_units = np.arange(n_runs) if run_units is None else np.asarray(run_units)
        self.walk = walk
        self.predictions = predictions or {}

    @property
    def n_runs(self):
        return len(self.run_units)

    @property
    def units(self):
        return list(dict.fromkeys(self.run_units.tolist()))

    @property
    def secondary(self):
        if self.task == "classification":
            return "balanced_accuracy" if self.metric != "balanced_accuracy" else "accuracy"
        return "rmse" if self.metric != "rmse" else "r2"

    def _unit_values(self, label, metric):
        """One value per unit: the mean over the runs of that unit (a seed, or the seeds of one fold)."""
        arr = self.scores[label][metric]
        return np.array([_mean(arr[self.run_units == u]) for u in self.units])

    def _delta(self, label, ref):
        sign = -1.0 if self.metric in _LOWER_IS_BETTER else 1.0
        d = sign * (self._unit_values(label, self.metric) - self._unit_values(ref, self.metric))
        return d[np.isfinite(d)]

    def _best_baseline(self):
        cands = [b for b in self.baselines if np.isfinite(_mean(self._unit_values(b, self.metric)))]
        if not cands:
            return None
        sign = -1.0 if self.metric in _LOWER_IS_BETTER else 1.0
        return max(cands, key=lambda b: sign * _mean(self._unit_values(b, self.metric)))

    def to_rows(self):
        """One dict per model, best first (failed models last). Means and stds are over units."""
        ref = self._best_baseline()
        sign = -1.0 if self.metric in _LOWER_IS_BETTER else 1.0
        rows = []
        for label in self.labels:
            row = {"model": label, "baseline": label in self.baselines,
                   "n_ok": int(np.isfinite(self.scores[label][self.metric]).sum()), "n_runs": self.n_runs,
                   "n_seeds": len(self.seeds), "fit_seconds": _mean(self.fit_seconds[label])}
            for m in self.scores[label]:
                uv = self._unit_values(label, m)
                row[m + "_mean"], row[m + "_std"] = _mean(uv), _std(uv)
            row["delta_mean"] = row["delta_std"] = float("nan")
            if ref is not None and label != ref and label not in self.baselines:
                d = self._delta(label, ref)
                if len(d):
                    row["delta_mean"], row["delta_std"] = float(np.mean(d)), _std(d)
            rows.append(row)
        rows.sort(key=lambda r: (not np.isfinite(r[self.metric + "_mean"]),
                                 -sign * r[self.metric + "_mean"] if np.isfinite(r[self.metric + "_mean"]) else 0))
        return rows

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.to_rows())

    def fold_table(self, metric=None):
        """Walk-forward only: one row per model, one column per fold (the mean over that fold's seeds), then the
        mean ± std across folds, which is the spread across periods."""
        if self.walk is None:
            raise ValueError("fold_table() is for benchmark(..., split='walk_forward')")
        m = metric or self.metric
        if m not in next(iter(self.scores.values())):
            raise ValueError("unknown metric %r" % m)
        units = self.units
        order = [r["model"] for r in self.to_rows()]
        head = ["model"] + ["fold %d" % u for u in units] + [m + " mean ± std"]
        body = []
        for label in order:
            uv = self._unit_values(label, m)
            body.append([label + (" (baseline)" if label in self.baselines else "")]
                        + [("%.4g" % v) if np.isfinite(v) else "n/a" for v in uv] + [_fmt(_mean(uv), _std(uv), m)])
        widths = [max(len(c[i]) for c in [head] + body) for i in range(len(head))]
        line = lambda cells: "  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(cells, widths)))
        return "\n".join([line(head), "  ".join("-" * w for w in widths)] + [line(c) for c in body])

    def reliability(self, label, bins=10):
        """Top-label reliability table for one model, pooled over all runs (list of dicts; see
        sc.format_reliability). Needs a classifier with predict_proba."""
        runs = self.predictions.get(label)
        if not runs:
            raise ValueError("no stored probabilities for %r (regression, no predict_proba, keep_predictions=False, "
                             "or every run failed)" % label)
        conf, ok = [], []
        for r in runs:
            p = np.asarray(r["proba"], dtype=float)
            idx = np.searchsorted(np.asarray(r["classes"]), np.asarray(r["y"]))
            conf.append(p.max(axis=1))
            ok.append(p.argmax(axis=1) == idx)
        return _reliability_rows(np.concatenate(conf), np.concatenate(ok), bins)

    def table(self):
        rows, ref = self.to_rows(), self._best_baseline()
        m1, m2 = self.metric, self.secondary
        extra = ["log_loss"] if (self.task == "classification" and m1 != "log_loss" and m2 != "log_loss"
                                 and any(np.isfinite(r["log_loss_mean"]) for r in rows)) else []
        show_delta = ref is not None
        head = ["model", m1, m2] + extra + (["vs best baseline"] if show_delta else []) + ["fit s", "ok"]
        body, any_noise = [], False
        for r in rows:
            name = r["model"] + (" (baseline)" if r["baseline"] else "")
            cells = [name] + [_fmt(r[m + "_mean"], r[m + "_std"], m) for m in [m1, m2] + extra]
            if show_delta:
                if np.isfinite(r["delta_mean"]):
                    txt = ("%+.3f" if m1 not in _LOWER_IS_BETTER else "%+.4g") % r["delta_mean"]
                    if np.isfinite(r["delta_std"]):
                        txt += " ± " + format(r["delta_std"], ".3f" if m1 not in _LOWER_IS_BETTER else ".4g")
                        if abs(r["delta_mean"]) <= r["delta_std"]:
                            txt += " ~"
                            any_noise = True
                    cells.append(txt)
                else:
                    cells.append("-")
            cells += ["%.2f" % r["fit_seconds"] if np.isfinite(r["fit_seconds"]) else "-",
                      "%d/%d" % (r["n_ok"], r["n_runs"])]
            body.append(cells)
        widths = [max(len(c[i]) for c in [head] + body) for i in range(len(head))]
        line = lambda cells: "  ".join(c.ljust(w) if i == 0 else c.rjust(w)
                                      for i, (c, w) in enumerate(zip(cells, widths)))
        n_tr = np.mean([len(s["train"]) for s in self.splits.values()])
        n_va = np.mean([len(s["val"]) for s in self.splits.values()])
        n_te = np.mean([len(s["test"]) for s in self.splits.values()])
        sizes = "train %d / val %d / test %d rows" % (round(n_tr), round(n_va), round(n_te))
        if self.walk:
            w = self.walk
            out = ["SaturnCube benchmark: %s, %d rows, walk-forward %d folds x %d seed%s" % (
                self.task, self.shape[0], w["n_folds"], len(self.seeds), "" if len(self.seeds) == 1 else "s")]
            out.append("split: %s window, purge %d, embargo %d; per fold about %s; each fold trains only on the past"
                       % ("expanding" if w["window"] is None else "rolling (%d rows)" % w["window"], w["purge"],
                          w["embargo"], sizes))
        else:
            out = ["SaturnCube benchmark: %s, %d rows, %d seed%s" % (
                self.task, self.shape[0], len(self.seeds), "" if len(self.seeds) == 1 else "s")]
            if self.grouped:
                out.append("split: group-disjoint over %d groups; per seed about %s" % (self.n_groups, sizes))
            else:
                out.append("split: random rows, NOT grouped (%s). If rows share a source (patches of one image, "
                           "repeated measurements) pass groups= or scores may be inflated; for time series use "
                           "split='walk_forward'." % sizes)
        if self.majority_share is not None:
            out.append("largest class = %.1f%% of all rows" % (100 * self.majority_share))
        out += ["", line(head), "  ".join("-" * w for w in widths)] + [line(c) for c in body]
        if self.walk:
            foot = ["std is over folds (each fold's value is the mean over its seeds): the spread across periods; "
                    "result.fold_table() shows every fold"]
        else:
            foot = ["std is over seeds (each seed changes the split and the model initialisation)"]
        if any_noise:
            foot.append("~ = difference within noise (|mean| <= std of the paired difference)")
        if self.task == "classification" and self.predictions:
            foot.append("result.reliability(model) gives the calibration table; log_loss, brier and ece need predict_proba")
        foot += self.notes
        for label, errs in self.errors.items():
            if errs:
                foot.append("%s failed on %d/%d runs, e.g. %s" % (label, len(errs), self.n_runs, errs[0][1]))
        return "\n".join(out + [""] + foot)

    def __str__(self):
        return self.table()

    def __repr__(self):
        return "<BenchmarkResult %s, %d models, %d runs>" % (self.task, len(self.labels), self.n_runs)


# ------------------------------------------------------------------------------------- main ---
def benchmark(models, X, y, groups=None, seeds=5, baselines=True, *, task="auto", metric=None,
              test_size=None, val_size=None, epochs=None, scale_target=True, show=True,
              split="auto", n_folds=5, purge=0, embargo=0, window=None, time=None, keep_predictions=True):
    """Compare models fairly and print a table.

    models     registry names ("mlp"), (name, kwargs) tuples, model classes, functions seed -> fresh model, or a
               dict {label: any of those}. A fresh model is built for every run and receives seed=<seed>.
    X, y       numeric features (n, ...) and a 1-D target. Integer / bool / string y is classification,
               float y is regression (override with task=).
    groups     one group id per row. Whole groups go to train, validation or test, so rows from the same source
               never straddle the split. None means random rows, which the table flags as possibly leaky.
    seeds      an int (seeds 0..n-1) or a list of ints. Each seed changes the model init (and, for random and grouped
               splits, the split).
    baselines  adds a class-prior (classification: predicts the most frequent class and gives the training class
               frequencies as probabilities) or train-mean (regression) baseline and a standardised scikit-learn
               linear model (logistic / ridge). scikit-learn is optional: if it is missing the second baseline is
               skipped and the table says so (pip install saturncube[bench]).
    metric     ranking metric. Classification: accuracy, balanced_accuracy, macro_f1, or the probability metrics
               log_loss, brier, ece (lower is better; models without predict_proba score n/a). Regression: r2,
               rmse, mae.
    split      "auto" (grouped if groups is given, else random), "random", "grouped", or "walk_forward" for time
               series: n_folds forward-chaining folds over the rows in order (or ordered by time=), each trained on
               the past only. purge drops that many rows before the validation and test blocks (use at least
               window + horizon - 1 for windowed features and forward labels); embargo drops rows after earlier test
               blocks from later training sets; window=None grows the training set, window=k keeps the last k rows.
               The table's std is then over folds, and result.fold_table() lists every fold.
    test_size, val_size   row fractions per split (defaults 0.25 / 0.15; walk-forward: per fold, 0.1 / 0.1). Groups
               are added until the fraction is reached, so actual sizes vary. The validation part is passed as
               val_data to models whose fit() accepts it; every model and baseline trains on the same training rows.
    epochs     overrides fit(epochs=...) for models that accept it.
    scale_target   regression only: models train on a standardised target and are scored on the original scale.
    keep_predictions   keep each classifier's test probabilities in result.predictions (used by result.reliability()).
    show       print the table (the result is returned either way).

    Returns a BenchmarkResult. A model that fails on a run is recorded, not fatal; a misspelled model name or
    invalid data raises immediately.
    """
    X, y, groups, task = _check_inputs(X, y, groups, task)
    if split not in ("auto", "random", "grouped", "walk_forward"):
        raise ValueError("split must be 'auto', 'random', 'grouped' or 'walk_forward'")
    if split == "auto":
        split = "grouped" if groups is not None else "random"
    wf = split == "walk_forward"
    if split == "grouped" and groups is None:
        raise ValueError("split='grouped' needs groups=")
    if split == "random" and groups is not None:
        raise ValueError("split='random' ignores groups; use split='grouped' (or leave split on 'auto')")
    if wf and groups is not None:
        raise ValueError("walk-forward folds are ordered in time and cannot be combined with groups=")
    if not wf and (purge or embargo or window is not None or time is not None):
        raise ValueError("purge, embargo, window and time are only used with split='walk_forward'")
    order = np.arange(len(y))
    if time is not None:
        time = np.asarray(time)
        if len(time) != len(y):
            raise ValueError("time must have one entry per row (%d); got %d" % (len(y), len(time)))
        order = np.argsort(time, kind="stable")
        X, y = X[order], y[order]
    test_size = (0.1 if wf else 0.25) if test_size is None else test_size
    val_size = (0.1 if wf else 0.15) if val_size is None else val_size
    valid = CLASSIFICATION_METRICS if task == "classification" else REGRESSION_METRICS
    metric = metric or valid[0]
    if metric not in valid:
        raise ValueError("metric %r is not valid for %s; choose from %s" % (metric, task, valid))
    if not wf and not (0 < test_size < 1 and 0 <= val_size < 1 and test_size + val_size < 1):
        raise ValueError("need 0 < test_size < 1, 0 <= val_size < 1 and test_size + val_size < 1")
    seed_list = _seed_list(seeds)
    entries = [(label, factory, False) for label, factory in _normalise_models(models, task)]
    notes = []
    if baselines:
        entries.append(("class prior" if task == "classification" else "train mean",
                        lambda seed: _Majority(task), True))
        sk = _sklearn_baseline(task)
        if sk is None:
            notes.append("scaled linear baseline skipped: scikit-learn is not installed "
                         "(pip install saturncube[bench])")
        else:
            entries.append(("scaled logistic regression" if task == "classification" else "scaled ridge regression",
                            sk, True))
    names = [e[0] for e in entries]
    if len(set(names)) != len(names):  # a user model label equal to a baseline label
        entries = [(l if not b else l + " [baseline]", f, b) for l, f, b in entries]
        names = [e[0] for e in entries]

    # ---- the runs: (unit, split key, seed, (train, val, test)); seeds of one fold share a unit
    splits, jobs, walk = {}, [], None
    if wf:
        folds = walk_forward_splits(len(y), n_folds, test_size, val_size, purge, embargo, window)
        all_classes = set(np.unique(y).tolist()) if task == "classification" else None
        for fi, f in enumerate(folds):
            if all_classes is not None and set(np.unique(y[f["train"]]).tolist()) != all_classes:
                raise ValueError("fold %d: the training window is missing class(es) %s; use a longer first window "
                                 "(fewer folds or a smaller test_size)" % (fi, sorted(all_classes - set(np.unique(y[f["train"]]).tolist()))))
            splits[fi] = {k: order[v] for k, v in f.items()}
            for seed in seed_list:
                jobs.append((fi, fi, seed, (f["train"], f["val"], f["test"])))
        walk = {"n_folds": len(folds), "purge": purge, "embargo": embargo, "window": window}
    else:
        for si, seed in enumerate(seed_list):
            tr, va, te = _make_split(groups, y, task, test_size, val_size, seed)
            splits[seed] = {"train": order[tr], "val": order[va], "test": order[te]}
            jobs.append((si, seed, seed, (tr, va, te)))
    n_runs = len(jobs)

    warmup()  # PyTorch's one-time start-up cost would otherwise be billed to the first model's 'fit s'
    keys = CLASSIFICATION_METRICS if task == "classification" else REGRESSION_METRICS
    scores = {l: {k: np.full(n_runs, np.nan) for k in keys} for l in names}
    seconds = {l: np.full(n_runs, np.nan) for l in names}
    errors = {l: [] for l in names}
    predictions = {l: [] for l in names}
    for ri, (unit, _, seed, (tr, va, te)) in enumerate(jobs):
        ytr, yva, yte = y[tr], y[va], y[te]
        for label, factory, is_baseline in entries:
            scaled = task == "regression" and scale_target and not is_baseline
            mu, sd = (float(ytr.mean()), float(ytr.std()) or 1.0) if scaled else (0.0, 1.0)
            try:
                t0 = time_.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = factory(seed)
                    if is_baseline:
                        model.fit(X[tr], ytr)
                    else:
                        _fit(model, X[tr], (ytr - mu) / sd if scaled else ytr, X[va],
                             (yva - mu) / sd if scaled else yva, epochs)
                seconds[label][ri] = time_.perf_counter() - t0
                pred = np.asarray(model.predict(X[te]))
                if task == "regression":
                    pred = pred.reshape(len(te)) * sd + mu
                    got = _regression_metrics(yte, pred)
                else:
                    if pred.shape != yte.shape:
                        raise ValueError("predict returned shape %s, expected %s" % (pred.shape, yte.shape))
                    got = _classification_metrics(yte, pred)
                    proba = None
                    if hasattr(model, "predict_proba"):
                        try:  # a model without usable probabilities just scores n/a on the probability metrics
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                proba = np.asarray(model.predict_proba(X[te]), dtype=float)
                            classes = getattr(model, "classes_", None)
                            classes = np.unique(ytr) if classes is None else np.asarray(classes)
                            if proba.shape != (len(te), len(classes)) or not np.isfinite(proba).all():
                                proba = None
                            else:
                                got.update(_proba_metrics(yte, proba, classes))
                        except Exception:
                            proba = None
                    if proba is not None and keep_predictions:
                        predictions[label].append({"y": yte, "proba": proba.astype(np.float32), "classes": classes})
                for k, v in got.items():
                    scores[label][k][ri] = v
            except Exception as exc:  # one broken model must not hide the others
                msg = "%s: %s" % (type(exc).__name__, str(exc).splitlines()[0][:120] if str(exc) else "")
                errors[label].append((seed, ("fold %d: " % unit if wf else "") + msg))
    share = None
    if task == "classification":
        _, counts = np.unique(y, return_counts=True)
        share = float(counts.max() / len(y))
    result = BenchmarkResult(task, metric, seed_list, names, [e[0] for e in entries if e[2]], scores, seconds,
                             errors, splits, groups is not None,
                             0 if groups is None else len(np.unique(groups)), X.shape, notes, share,
                             run_units=[j[0] for j in jobs], walk=walk, predictions=predictions)
    if show:
        print(result)
    return result
