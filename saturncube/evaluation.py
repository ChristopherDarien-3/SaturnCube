"""Fair model comparison: group-disjoint splits, several seeds, automatic baselines, a printed table.

    result = sc.benchmark(["mlp", "random_forest", "kan"], X, y, groups=source_ids, seeds=5)

Every model sees exactly the same train / validation / test rows for a given seed. Test rows are never
passed to fit(). When ``groups`` is given, whole groups (images, patients, theorem families, ...) go to
one side of the split, so rows that share a source cannot leak between train and test.
"""
from __future__ import annotations

import difflib
import inspect
import time
import warnings

import numpy as np

from .base import REGISTRY, BaseModel, warmup
from .data import split_indices

CLASSIFICATION_METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
REGRESSION_METRICS = ("r2", "rmse", "mae")
_LOWER_IS_BETTER = {"rmse", "mae"}


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


def _regression_metrics(y, p):
    y, p = y.astype(float), np.asarray(p, dtype=float).reshape(len(y))
    err = p - y
    sst = float(np.sum((y - y.mean()) ** 2))
    return {"r2": float(1 - np.sum(err ** 2) / sst) if sst > 1e-12 else float("nan"),
            "rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err)))}


# ------------------------------------------------------------------------------- baselines ---
class _Majority:
    """Predicts the most frequent training class (classification) or the training mean (regression)."""

    def __init__(self, task):
        self.task = task

    def fit(self, X, y):
        if self.task == "regression":
            self.value_ = float(np.mean(y))
        else:
            vals, counts = np.unique(y, return_counts=True)
            self.value_ = vals[int(np.argmax(counts))]
        return self

    def predict(self, X):
        return np.full(len(X), self.value_)


class _Flat:
    """Wraps a scikit-learn estimator so it accepts arrays of any rank (flattened to 2-D)."""

    def __init__(self, est):
        self.est = est

    def fit(self, X, y):
        self.est.fit(X.reshape(len(X), -1), y)
        return self

    def predict(self, X):
        return self.est.predict(X.reshape(len(X), -1))


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
    """What sc.benchmark returns. print(result) shows the table; the raw numbers are kept per seed.

    Attributes: task, metric, seeds, scores {label: {metric: array per seed, NaN where a run failed}},
    fit_seconds {label: array}, errors {label: [(seed, message)]}, baselines (labels), splits
    {seed: {"train", "val", "test"} row indices}, grouped (bool), notes (list of str).
    """

    def __init__(self, task, metric, seeds, labels, baselines, scores, fit_seconds, errors, splits,
                 grouped, n_groups, shape, notes, majority_share=None):
        self.task, self.metric, self.seeds = task, metric, seeds
        self.labels, self.baselines = labels, baselines
        self.scores, self.fit_seconds, self.errors = scores, fit_seconds, errors
        self.splits, self.grouped, self.n_groups = splits, grouped, n_groups
        self.shape, self.notes, self.majority_share = shape, notes, majority_share

    @property
    def secondary(self):
        if self.task == "classification":
            return "balanced_accuracy" if self.metric != "balanced_accuracy" else "accuracy"
        return "rmse" if self.metric != "rmse" else "r2"

    def _delta(self, label, ref):
        sign = -1.0 if self.metric in _LOWER_IS_BETTER else 1.0
        d = sign * (self.scores[label][self.metric] - self.scores[ref][self.metric])
        return d[np.isfinite(d)]

    def _best_baseline(self):
        cands = [b for b in self.baselines if np.isfinite(_mean(self.scores[b][self.metric]))]
        if not cands:
            return None
        sign = -1.0 if self.metric in _LOWER_IS_BETTER else 1.0
        return max(cands, key=lambda b: sign * _mean(self.scores[b][self.metric]))

    def to_rows(self):
        """One dict per model, best first (failed models last)."""
        ref = self._best_baseline()
        sign = -1.0 if self.metric in _LOWER_IS_BETTER else 1.0
        rows = []
        for label in self.labels:
            row = {"model": label, "baseline": label in self.baselines,
                   "n_ok": int(np.isfinite(self.scores[label][self.metric]).sum()), "n_seeds": len(self.seeds),
                   "fit_seconds": _mean(self.fit_seconds[label])}
            for m in self.scores[label]:
                row[m + "_mean"], row[m + "_std"] = _mean(self.scores[label][m]), _std(self.scores[label][m])
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

    def table(self):
        rows, ref = self.to_rows(), self._best_baseline()
        m1, m2 = self.metric, self.secondary
        show_delta = ref is not None
        head = ["model", m1, m2] + (["vs best baseline"] if show_delta else []) + ["fit s", "ok"]
        body, any_noise = [], False
        for r in rows:
            name = r["model"] + (" (baseline)" if r["baseline"] else "")
            cells = [name, _fmt(r[m1 + "_mean"], r[m1 + "_std"], m1), _fmt(r[m2 + "_mean"], r[m2 + "_std"], m2)]
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
                      "%d/%d" % (r["n_ok"], r["n_seeds"])]
            body.append(cells)
        widths = [max(len(c[i]) for c in [head] + body) for i in range(len(head))]
        line = lambda cells: "  ".join(c.ljust(w) if i == 0 else c.rjust(w)
                                      for i, (c, w) in enumerate(zip(cells, widths)))
        n_tr = np.mean([len(s["train"]) for s in self.splits.values()])
        n_va = np.mean([len(s["val"]) for s in self.splits.values()])
        n_te = np.mean([len(s["test"]) for s in self.splits.values()])
        out = ["SaturnCube benchmark: %s, %d rows, %d seed%s" % (
            self.task, self.shape[0], len(self.seeds), "" if len(self.seeds) == 1 else "s")]
        if self.grouped:
            out.append("split: group-disjoint over %d groups; per seed about train %d / val %d / test %d rows"
                       % (self.n_groups, round(n_tr), round(n_va), round(n_te)))
        else:
            out.append("split: random rows, NOT grouped (train %d / val %d / test %d). If rows share a source "
                       "(patches of one image, repeated measurements) pass groups= or scores may be inflated."
                       % (round(n_tr), round(n_va), round(n_te)))
        if self.majority_share is not None:
            out.append("largest class = %.1f%% of all rows" % (100 * self.majority_share))
        out += ["", line(head), "  ".join("-" * w for w in widths)] + [line(c) for c in body]
        foot = ["std is over seeds (each seed changes the split and the model initialisation)"]
        if any_noise:
            foot.append("~ = difference within seed noise (|mean| <= std of the paired difference)")
        foot += self.notes
        for label, errs in self.errors.items():
            if errs:
                foot.append("%s failed on %d/%d seeds, e.g. %s" % (label, len(errs), len(self.seeds), errs[0][1]))
        return "\n".join(out + [""] + foot)

    def __str__(self):
        return self.table()

    def __repr__(self):
        return "<BenchmarkResult %s, %d models, %d seeds>" % (self.task, len(self.labels), len(self.seeds))


# ------------------------------------------------------------------------------------- main ---
def benchmark(models, X, y, groups=None, seeds=5, baselines=True, *, task="auto", metric=None,
              test_size=0.25, val_size=0.15, epochs=None, scale_target=True, show=True):
    """Compare models fairly and print a table.

    models     registry names ("mlp"), (name, kwargs) tuples, model classes, functions seed -> fresh model, or a
               dict {label: any of those}. A fresh model is built for every seed and receives seed=<seed>.
    X, y       numeric features (n, ...) and a 1-D target. Integer / bool / string y is classification,
               float y is regression (override with task=).
    groups     one group id per row. Whole groups go to train, validation or test, so rows from the same source
               never straddle the split. None means random rows, which the table flags as possibly leaky.
    seeds      an int (seeds 0..n-1) or a list of ints. Each seed changes the split and the model init.
    baselines  adds a majority-class (classification) or train-mean (regression) guess and a standardised
               scikit-learn linear model (logistic / ridge). scikit-learn is optional: if it is missing the
               second baseline is skipped and the table says so (pip install saturncube[bench]).
    metric     ranking metric: accuracy / balanced_accuracy / macro_f1, or r2 / rmse / mae.
    test_size, val_size   row fractions (groups are added until the fraction is reached, so actual sizes vary).
               The validation part is passed as val_data to models whose fit() accepts it; every model and
               baseline trains on the same training rows only.
    epochs     overrides fit(epochs=...) for models that accept it.
    scale_target   regression only: models train on a standardised target and are scored on the original scale
               (models do not standardise targets themselves). Baselines always use the raw target.
    show       print the table (the result is returned either way).

    Returns a BenchmarkResult. A model that fails on a seed is recorded, not fatal; a misspelled model name or
    invalid data raises immediately.
    """
    X, y, groups, task = _check_inputs(X, y, groups, task)
    valid = CLASSIFICATION_METRICS if task == "classification" else REGRESSION_METRICS
    metric = metric or valid[0]
    if metric not in valid:
        raise ValueError("metric %r is not valid for %s; choose from %s" % (metric, task, valid))
    if not (0 < test_size < 1 and 0 <= val_size < 1 and test_size + val_size < 1):
        raise ValueError("need 0 < test_size < 1, 0 <= val_size < 1 and test_size + val_size < 1")
    seed_list = _seed_list(seeds)
    entries = [(label, factory, False) for label, factory in _normalise_models(models, task)]
    notes = []
    if baselines:
        entries.append(("majority class" if task == "classification" else "train mean",
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

    warmup()  # PyTorch's one-time start-up cost would otherwise be billed to the first model's 'fit s'
    keys = CLASSIFICATION_METRICS if task == "classification" else REGRESSION_METRICS
    scores = {l: {k: np.full(len(seed_list), np.nan) for k in keys} for l in names}
    seconds = {l: np.full(len(seed_list), np.nan) for l in names}
    errors = {l: [] for l in names}
    splits = {}
    for si, seed in enumerate(seed_list):
        tr, va, te = _make_split(groups, y, task, test_size, val_size, seed)
        splits[seed] = {"train": tr, "val": va, "test": te}
        ytr, yva, yte = y[tr], y[va], y[te]
        for label, factory, is_baseline in entries:
            scaled = task == "regression" and scale_target and not is_baseline
            mu, sd = (float(ytr.mean()), float(ytr.std()) or 1.0) if scaled else (0.0, 1.0)
            try:
                t0 = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = factory(seed)
                    if is_baseline:
                        model.fit(X[tr], ytr)
                    else:
                        _fit(model, X[tr], (ytr - mu) / sd if scaled else ytr, X[va],
                             (yva - mu) / sd if scaled else yva, epochs)
                seconds[label][si] = time.perf_counter() - t0
                pred = np.asarray(model.predict(X[te]))
                if task == "regression":
                    pred = pred.reshape(len(te)) * sd + mu
                    got = _regression_metrics(yte, pred)
                else:
                    if pred.shape != yte.shape:
                        raise ValueError("predict returned shape %s, expected %s" % (pred.shape, yte.shape))
                    got = _classification_metrics(yte, pred)
                for k, v in got.items():
                    scores[label][k][si] = v
            except Exception as exc:  # one broken model must not hide the others
                errors[label].append((seed, "%s: %s" % (type(exc).__name__, str(exc).splitlines()[0][:120]
                                                         if str(exc) else "")))
    share = None
    if task == "classification":
        _, counts = np.unique(y, return_counts=True)
        share = float(counts.max() / len(y))
    result = BenchmarkResult(task, metric, seed_list, names, [e[0] for e in entries if e[2]], scores, seconds,
                             errors, splits, groups is not None,
                             0 if groups is None else len(np.unique(groups)), X.shape, notes, share)
    if show:
        print(result)
    return result

