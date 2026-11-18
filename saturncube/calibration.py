"""Probability quality: metrics, reliability table, prior-shift correction, temperature scaling.

    sc.log_loss(y, proba); sc.brier_score(y, proba); sc.expected_calibration_error(y, proba)
    print(sc.format_reliability(sc.reliability_table(y, proba)))

    cal = sc.Calibrated("mlp", epochs=40).fit(X_train, y_train, val_data=(X_val, y_val))
    cal.predict_proba(X_test)        # prior-shift corrected and temperature scaled, both fitted on the validation rows

``proba`` has one column per class in the order of ``model.classes_`` (sorted labels). ``y`` holds labels; pass
``classes=`` when they are not already the integers 0..k-1.
"""
from __future__ import annotations

import inspect
import json

import numpy as np

from .base import register, resolve_model


def _prepare(y, proba, classes):
    proba = np.asarray(proba, dtype=float)
    y = np.asarray(y)
    if proba.ndim != 2:
        raise ValueError("proba must have shape (n_rows, n_classes); got %s" % (proba.shape,))
    if len(y) != len(proba):
        raise ValueError("%d labels but %d probability rows" % (len(y), len(proba)))
    if not np.isfinite(proba).all():
        raise ValueError("proba contains NaN or infinite values")
    if classes is None:
        if y.dtype.kind not in "iu" or (len(y) and (y.min() < 0 or y.max() >= proba.shape[1])):
            raise ValueError("pass classes= (the label of each probability column) unless y holds the integers 0..k-1")
        idx = y.astype(int)
    else:
        classes = np.asarray(classes)
        if len(classes) != proba.shape[1]:
            raise ValueError("classes has %d entries but proba has %d columns" % (len(classes), proba.shape[1]))
        idx = np.searchsorted(classes, y)
        idx = np.clip(idx, 0, len(classes) - 1)
        if not np.all(classes[idx] == y):
            raise ValueError("y contains labels that are not in classes")
    return idx, proba


def log_loss(y, proba, classes=None, eps=1e-15):
    """Mean negative log probability given to the true class (natural log). Lower is better; a model that knows
    nothing about the rows but the class frequencies scores the entropy of those frequencies."""
    idx, p = _prepare(y, proba, classes)
    return float(-np.log(np.clip(p[np.arange(len(idx)), idx], eps, 1.0)).mean())


def brier_score(y, proba, classes=None):
    """Mean over rows of the squared distance between the probability vector and the one-hot label (0 = perfect,
    2 = worst). Lower is better."""
    idx, p = _prepare(y, proba, classes)
    return float(((p - np.eye(p.shape[1])[idx]) ** 2).sum(axis=1).mean())


def _reliability_rows(conf, ok, bins):
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for k in range(bins):
        m = (conf >= edges[k]) & ((conf < edges[k + 1]) if k < bins - 1 else (conf <= 1.0))
        if m.any():
            rows.append({"low": float(edges[k]), "high": float(edges[k + 1]), "count": int(m.sum()),
                         "confidence": float(conf[m].mean()), "accuracy": float(ok[m].mean()),
                         "gap": float(ok[m].mean() - conf[m].mean())})
    return rows


def reliability_table(y, proba, bins=10, classes=None):
    """Top-label reliability: rows are equal-width confidence bins; each says how many predictions fell in it,
    their mean confidence, and how often the predicted class was right. A calibrated model has accuracy close to
    confidence in every bin. Returns a list of dicts (empty bins are omitted)."""
    idx, p = _prepare(y, proba, classes)
    return _reliability_rows(p.max(axis=1), p.argmax(axis=1) == idx, bins)


def expected_calibration_error(y, proba, bins=10, classes=None):
    """Weighted average over confidence bins of |accuracy - confidence| (top-label, equal-width bins).
    With few rows the estimate is biased upwards; compare models on the same number of rows."""
    rows = reliability_table(y, proba, bins, classes)
    n = sum(r["count"] for r in rows)
    return float(sum(r["count"] / n * abs(r["gap"]) for r in rows)) if n else float("nan")


def format_reliability(table):
    """The reliability table as aligned text."""
    lines = ["confidence bin      n   mean conf  accuracy    gap"]
    for r in table:
        lines.append("%.1f - %.1f  %9d   %9.3f  %8.3f  %+6.3f" % (r["low"], r["high"], r["count"], r["confidence"],
                                                                  r["accuracy"], r["gap"]))
    return "\n".join(lines)


class PriorShift:
    """Corrects probabilities for a change in class frequencies between training and deployment:
    p'(c|x) is proportional to p(c|x) * target_prior[c] / source_prior[c].

    fit(y_source, y_target) estimates both priors from labels: y_source are the labels the model was trained on,
    y_target the labels of a recent validation period. ``smoothing`` adds that many pseudo-counts per class so a class
    that is missing in one period does not zero out its probability. Give source_prior / target_prior to set them
    directly (one value per class, in the order of the probability columns)."""

    def __init__(self, source_prior=None, target_prior=None, smoothing=1.0):
        self.source_prior_ = None if source_prior is None else self._norm(source_prior)
        self.target_prior_ = None if target_prior is None else self._norm(target_prior)
        self.smoothing = smoothing

    @staticmethod
    def _norm(v):
        v = np.asarray(v, dtype=float)
        if v.ndim != 1 or (v < 0).any() or v.sum() <= 0:
            raise ValueError("a prior must be a 1-D vector of non-negative numbers that is not all zero")
        return v / v.sum()

    def _prior_from(self, y, classes):
        counts = np.array([(np.asarray(y) == c).sum() for c in classes], dtype=float) + self.smoothing
        return counts / counts.sum()

    def fit(self, y_source, y_target, classes=None):
        classes = np.unique(np.concatenate([np.asarray(y_source), np.asarray(y_target)])) if classes is None else np.asarray(classes)
        self.classes_ = classes
        self.source_prior_ = self._prior_from(y_source, classes)
        self.target_prior_ = self._prior_from(y_target, classes)
        return self

    @property
    def ratio_(self):
        if self.source_prior_ is None or self.target_prior_ is None:
            raise RuntimeError("call fit() or give both priors first")
        return self.target_prior_ / np.maximum(self.source_prior_, 1e-12)

    def transform(self, proba):
        p = np.asarray(proba, dtype=float)
        if p.ndim != 2 or p.shape[1] != len(self.ratio_):
            raise ValueError("proba must have %d columns" % len(self.ratio_))
        q = p * self.ratio_
        return q / np.maximum(q.sum(axis=1, keepdims=True), 1e-300)

    __call__ = transform


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class TemperatureScaler:
    """Temperature scaling (Guo et al. 2017) applied to probabilities: p' is the softmax of log(p) / T. T > 1 softens
    over-confident probabilities, T < 1 sharpens under-confident ones; the predicted class never changes. T is fitted
    by minimising log loss on rows the model was not trained on (a 1-D convex problem, solved by golden-section
    search)."""

    def __init__(self, bounds=(0.05, 20.0)):
        self.bounds, self.temperature_ = bounds, 1.0

    def fit(self, proba, y, classes=None):
        idx, p = _prepare(y, proba, classes)
        logp = np.log(np.clip(p, 1e-12, 1.0))

        def nll(beta):  # beta = 1 / T; the loss is convex in beta
            q = _softmax(logp * beta)
            return -np.log(np.clip(q[np.arange(len(idx)), idx], 1e-15, 1.0)).mean()

        lo, hi = 1.0 / self.bounds[1], 1.0 / self.bounds[0]
        g = (np.sqrt(5) - 1) / 2
        a, b = lo, hi
        c, d = b - g * (b - a), a + g * (b - a)
        fc, fd = nll(c), nll(d)
        for _ in range(80):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - g * (b - a)
                fc = nll(c)
            else:
                a, c, fc = c, d, fd
                d = a + g * (b - a)
                fd = nll(d)
        self.temperature_ = float(1.0 / ((a + b) / 2))
        return self

    def transform(self, proba):
        p = np.asarray(proba, dtype=float)
        return _softmax(np.log(np.clip(p, 1e-12, 1.0)) / self.temperature_)

    __call__ = transform


def _stratified_holdout(y, frac, seed):
    rng = np.random.default_rng(seed)
    val = []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        k = min(int(round(len(idx) * frac)), len(idx) - 1)
        val.append(idx[:k])
    val = np.sort(np.concatenate(val))
    return np.setdiff1d(np.arange(len(y)), val), val


@register("calibrated", family="wrapper")
class Calibrated:
    """Wraps a classifier and corrects its probabilities on validation rows: a prior-shift correction (the validation
    class frequencies against the training ones) and temperature scaling, both fitted after the model is trained and
    never on the training rows.

    base          registry name or class of the model to wrap (a SeedEnsemble works too); model_kwargs go to its
                  constructor (seed= included).
    prior         apply PriorShift; temperature: apply TemperatureScaler. Both default on.
    val_split     used only when fit() gets no val_data: that stratified share of the rows is held out.
    fit(X, y, val_data=(Xv, yv)) trains on X, y (the model also gets val_data for early stopping if its fit() takes
    it) and fits the corrections on val_data, which should come from the period just before deployment. If the model
    was trained with class_weight, the weighting is taken into account when estimating its effective training prior."""

    def __init__(self, base="mlp", *, prior=True, temperature=True, val_split=0.2, **model_kwargs):
        if not 0 < val_split < 0.5:
            raise ValueError("val_split must be in (0, 0.5)")
        self._cls, self.prior, self.temperature, self.val_split = resolve_model(base), prior, temperature, val_split
        self._kwargs = model_kwargs
        self.model = self.prior_shift_ = self.scaler_ = None

    def fit(self, X, y, val_data=None, **fit_kwargs):
        y = np.asarray(y)
        X = np.asarray(X)
        if y.dtype.kind == "f":
            raise ValueError("Calibrated is for classification (integer, bool or string labels)")
        if val_data is None:
            tr, va = _stratified_holdout(y, self.val_split, self._kwargs.get("seed") or 0)
            if len(va) == 0:
                raise ValueError("too few rows to hold out a validation set; pass val_data=")
            Xv, yv, X, y = X[va], y[va], X[tr], y[tr]
        else:
            Xv, yv = np.asarray(val_data[0]), np.asarray(val_data[1])
        self.model = self._cls(**self._kwargs)
        if "val_data" in inspect.signature(self.model.fit).parameters:
            fit_kwargs.setdefault("val_data", (Xv, yv))
        self.model.fit(X, y, **fit_kwargs)
        classes = np.asarray(self.model.classes_)
        pv = np.asarray(self.model.predict_proba(Xv), dtype=float)
        if self.prior:
            ps = PriorShift().fit(y, yv, classes=classes)
            cw = getattr(self.model, "class_weight_", None)
            if cw is not None:  # a weighted loss trains the model for a prior proportional to prior * weight
                eff = ps.source_prior_ * np.asarray(cw)
                ps.source_prior_ = eff / eff.sum()
            self.prior_shift_ = ps
            pv = ps.transform(pv)
        if self.temperature:
            self.scaler_ = TemperatureScaler().fit(pv, yv, classes=classes)
        return self

    def _need_fit(self):
        if self.model is None:
            raise RuntimeError("call fit() first")

    @property
    def classes_(self):
        self._need_fit()
        return self.model.classes_

    def predict_proba(self, X):
        self._need_fit()
        p = np.asarray(self.model.predict_proba(X), dtype=float)
        if self.prior_shift_ is not None:
            p = self.prior_shift_.transform(p)
        if self.scaler_ is not None:
            p = self.scaler_.transform(p)
        return p

    def predict(self, X):
        return np.asarray(self.classes_)[self.predict_proba(X).argmax(axis=1)]

    def evaluate(self, X, y):
        p, classes = self.predict_proba(X), np.asarray(self.classes_)
        y = np.asarray(y)
        return {"accuracy": float(np.mean(classes[p.argmax(1)] == y)), "log_loss": log_loss(y, p, classes),
                "brier": brier_score(y, p, classes), "ece": expected_calibration_error(y, p, classes=classes)}

    def save(self, path):
        """Model to ``path`` (the model must have save()), corrections to ``path + '.calib.json'``."""
        self._need_fit()
        name = getattr(self._cls, "name", None)
        if not hasattr(self.model, "save") or name is None:
            raise NotImplementedError("only registered models with save() can be stored")
        self.model.save(path)
        meta = {"model": name, "val_split": self.val_split,
                "ratio": None if self.prior_shift_ is None else self.prior_shift_.ratio_.tolist(),
                "temperature": None if self.scaler_ is None else self.scaler_.temperature_}
        with open(str(path) + ".calib.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path):
        with open(str(path) + ".calib.json", encoding="utf-8") as f:
            meta = json.load(f)
        mcls = resolve_model(meta["model"])
        obj = cls(meta["model"], prior=meta["ratio"] is not None, temperature=meta["temperature"] is not None,
                  val_split=meta["val_split"])
        obj.model = mcls.load(path)
        if meta["ratio"] is not None:
            ratio = np.asarray(meta["ratio"])
            obj.prior_shift_ = PriorShift(source_prior=np.ones(len(ratio)), target_prior=ratio)  # ratio_ = target/source
        if meta["temperature"] is not None:
            obj.scaler_ = TemperatureScaler()
            obj.scaler_.temperature_ = float(meta["temperature"])
        return obj
