"""Seed ensembles: n copies of one model that differ only in their random seed, averaged.

    ens = sc.SeedEnsemble("mlp", n=5, seed=0, epochs=40).fit(X_train, y_train, val_data=(X_val, y_val))
    ens.predict_proba(X_test)        # mean of the five members' probabilities
    ens.disagreement(X_test)         # per row: how much the members disagree (a cheap uncertainty signal)

A single run of a small network is a noisy draw: different seeds give different test scores and often different
predictions. Averaging probabilities removes much of that noise. For classifiers the log loss of the averaged
probabilities can never be worse than the average of the members' log losses (Jensen's inequality).
"""
from __future__ import annotations

import inspect
import json

import numpy as np

from .base import register, resolve_model
from .calibration import brier_score, expected_calibration_error, log_loss


def _member_seeds(seed, n):
    return [int(v) % (2 ** 31 - 1) for v in np.random.SeedSequence(seed).generate_state(n)]


@register("seed_ensemble", family="wrapper")
class SeedEnsemble:
    """Trains ``n`` copies of a model that differ only in their random seed and averages their predictions.

    model         registry name or class; model_kwargs go to its constructor. The members' seeds come from the
                  ensemble's own ``seed``, so do not pass one to the model.
    n             number of members. seed: the ensemble's base seed; member seeds are derived from it with numpy's
                  SeedSequence, so they are independent and the whole ensemble is reproducible.
    average       "proba" (default): arithmetic mean of the members' probabilities. "geometric": normalised mean of
                  log-probabilities, which rewards agreement. Regression always averages the predictions.
    fit(X, y, val_data=...) passes val_data (and epochs) on to members whose fit() accepts them, so patience and
    early stopping work per member. After fit: ``members_`` and ``member_seeds_``.
    """

    def __init__(self, model="mlp", *, n=5, seed=0, average="proba", **model_kwargs):
        if not (isinstance(n, (int, np.integer)) and not isinstance(n, bool) and n >= 1):
            raise ValueError("n must be a positive integer")
        if average not in ("proba", "geometric"):
            raise ValueError("average must be 'proba' or 'geometric'")
        cls = resolve_model(model)
        params = inspect.signature(cls.__init__).parameters
        if "seed" not in params and not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            raise ValueError("%s takes no seed, so every member would be identical; a seed ensemble needs a model "
                             "whose training depends on its seed" % cls.__name__)
        self._cls, self.n, self.seed, self.average, self._kwargs = cls, int(n), seed, average, model_kwargs
        self.members_, self.member_seeds_ = [], _member_seeds(seed, self.n)

    # ---------------------------------------------------------------- training
    def fit(self, X, y, epochs=None, val_data=None, **fit_kwargs):
        members = []
        for s in self.member_seeds_:
            m = self._cls(seed=s, **self._kwargs)
            params = inspect.signature(m.fit).parameters
            kw = dict(fit_kwargs)
            if epochs is not None and "epochs" in params:
                kw["epochs"] = epochs
            if val_data is not None and "val_data" in params:
                kw["val_data"] = val_data
            m.fit(X, y, **kw)
            members.append(m)
        self.members_ = members
        return self

    def _need_fit(self):
        if not self.members_:
            raise RuntimeError("call fit() first")

    @property
    def task(self):
        self._need_fit()
        return self.members_[0].task

    @property
    def classes_(self):
        self._need_fit()
        return self.members_[0].classes_

    # -------------------------------------------------------------- prediction
    def member_proba(self, X):
        """Every member's class probabilities: array (n_members, n_rows, n_classes)."""
        self._need_fit()
        if self.task != "classification":
            raise AttributeError("predict_proba is for classification; use member_predict for regression")
        return np.stack([np.asarray(m.predict_proba(X), dtype=float) for m in self.members_])

    def member_predict(self, X):
        """Every member's prediction: array (n_members, n_rows[, outputs]); labels for classifiers."""
        self._need_fit()
        return np.stack([np.asarray(m.predict(X)) for m in self.members_])

    def predict_proba(self, X):
        P = self.member_proba(X)
        if self.average == "geometric":
            z = np.log(np.clip(P, 1e-12, 1.0)).mean(axis=0)
            z = np.exp(z - z.max(axis=1, keepdims=True))
            return z / z.sum(axis=1, keepdims=True)
        return P.mean(axis=0)

    def predict(self, X):
        self._need_fit()
        if self.task == "classification":
            return np.asarray(self.classes_)[self.predict_proba(X).argmax(axis=1)]
        return self.member_predict(X).mean(axis=0)

    def disagreement(self, X):
        """One number per row. Classification: the root-mean-square, over classes, of the standard deviation across
        members of that class's probability (0 when all members agree exactly). Regression: the standard deviation
        across members of the prediction (averaged over outputs if there are several). A high value marks rows where
        the answer depends on the seed."""
        self._need_fit()
        if self.task == "classification":
            return np.sqrt(self.member_proba(X).var(axis=0).mean(axis=1))
        sd = self.member_predict(X).astype(float).std(axis=0)
        return sd if sd.ndim == 1 else sd.mean(axis=tuple(range(1, sd.ndim)))

    def evaluate(self, X, y):
        y = np.asarray(y)
        if self.task == "classification":
            p, classes = self.predict_proba(X), np.asarray(self.classes_)
            return {"accuracy": float(np.mean(classes[p.argmax(1)] == y)), "log_loss": log_loss(y, p, classes),
                    "brier": brier_score(y, p, classes), "ece": expected_calibration_error(y, p, classes=classes)}
        err = np.asarray(self.predict(X), dtype=float).reshape(len(y), -1) - y.astype(float).reshape(len(y), -1)
        sst = ((y.astype(float).reshape(len(y), -1) - y.astype(float).reshape(len(y), -1).mean(0)) ** 2).sum()
        return {"rmse": float(np.sqrt((err ** 2).mean())), "mae": float(np.abs(err).mean()),
                "r2": float(1 - (err ** 2).sum() / sst) if sst > 1e-12 else float("nan")}

    # -------------------------------------------------------------- persistence
    def save(self, path):
        """Members to ``path + '.member<i>'`` (members need save()) and the settings to ``path + '.ensemble.json'``."""
        self._need_fit()
        name = getattr(self._cls, "name", None)
        if name is None or not all(hasattr(m, "save") for m in self.members_):
            raise NotImplementedError("only registered models with save() can be stored")
        for i, m in enumerate(self.members_):
            m.save("%s.member%d" % (path, i))
        with open(str(path) + ".ensemble.json", "w", encoding="utf-8") as f:
            json.dump({"model": name, "n": self.n, "seed": self.seed, "average": self.average}, f)

    @classmethod
    def load(cls, path):
        with open(str(path) + ".ensemble.json", encoding="utf-8") as f:
            meta = json.load(f)
        obj = cls(meta["model"], n=meta["n"], seed=meta["seed"], average=meta["average"])
        obj.members_ = [obj._cls.load("%s.member%d" % (path, i)) for i in range(meta["n"])]
        return obj

    def __repr__(self):
        return "SeedEnsemble(%s, n=%d, fitted=%s)" % (self._cls.__name__, self.n, bool(self.members_))
