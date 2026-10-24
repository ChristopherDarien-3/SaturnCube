"""Classical machine learning with the same fit / predict / evaluate interface:
decision tree + random forest, k-means and Gaussian naive Bayes (pure NumPy)."""
from __future__ import annotations

import numpy as np

from .base import register


class _Classical:
    family, module = "classical", None

    @staticmethod
    def _task(y, task):
        return task if task != "auto" else ("classification" if np.asarray(y).dtype.kind in "biuOUS" else "regression")

    def evaluate(self, X, y):
        y, pred = np.asarray(y), self.predict(X)
        if self.task == "classification":
            return {"accuracy": float((pred == y).mean())}
        mse = float(((pred - y) ** 2).mean())
        return {"mse": mse, "rmse": mse ** 0.5, "r2": 1 - mse / max(float(y.var()), 1e-12)}


class _Node:
    __slots__ = ("feature", "thr", "left", "right", "value")

    def __init__(self, value=None):
        self.value, self.feature, self.thr, self.left, self.right = value, None, None, None, None


@register("decision_tree", family="classical")
class DecisionTree(_Classical):
    """CART tree: Gini impurity (classification) or squared error (regression), exhaustive best splits."""

    def __init__(self, max_depth=8, min_samples_leaf=1, max_features=None, task="auto", seed=None):
        self.max_depth, self.min_leaf, self.max_features, self.task = max_depth, min_samples_leaf, max_features, task
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y, classes=None):
        X, y = np.asarray(X, float), np.asarray(y)
        self.task = self._task(y, self.task)
        if self.task == "classification":
            self.classes_ = np.unique(y) if classes is None else classes
            Y = (y[:, None] == self.classes_[None, :]).astype(float)
        else:
            Y = y.astype(float)[:, None]
        self.root = self._build(X, Y, 0)
        return self

    def _leaf(self, Y):
        return _Node(Y.mean(0))

    def _build(self, X, Y, depth):
        n = len(X)
        pure = (self.task == "classification" and Y.sum(0).max() == n) or (self.task == "regression" and Y.var() == 0)
        if depth >= self.max_depth or n < 2 * self.min_leaf or pure:
            return self._leaf(Y)
        d = X.shape[1]
        feats = np.arange(d) if self.max_features is None else self.rng.choice(d, min(self.max_features, d), replace=False)
        best = (np.inf, None, None)
        nl = np.arange(1, n)
        for f in feats:
            order = np.argsort(X[:, f], kind="stable")
            xs, Ys = X[order, f], Y[order]
            valid = (xs[1:] != xs[:-1]) & (nl >= self.min_leaf) & (n - nl >= self.min_leaf)
            if not valid.any():
                continue
            cs = np.cumsum(Ys, 0)
            if self.task == "classification":
                left, right = cs[:-1], cs[-1] - cs[:-1]
                score = (nl * (1 - ((left / nl[:, None]) ** 2).sum(1)) + (n - nl) * (1 - ((right / (n - nl)[:, None]) ** 2).sum(1))) / n
            else:
                sl, cq = cs[:-1, 0], np.cumsum(Ys[:, 0] ** 2)
                score = ((cq[:-1] - sl ** 2 / nl) + ((cq[-1] - cq[:-1]) - (cs[-1, 0] - sl) ** 2 / (n - nl))) / n
            score = np.where(valid, score, np.inf)
            i = int(np.argmin(score))
            if score[i] < best[0]:
                best = (score[i], f, (xs[i] + xs[i + 1]) / 2)
        if best[1] is None:
            return self._leaf(Y)
        node = _Node()
        node.feature, node.thr = best[1], best[2]
        m = X[:, node.feature] <= node.thr
        node.left, node.right = self._build(X[m], Y[m], depth + 1), self._build(X[~m], Y[~m], depth + 1)
        return node

    def _raw(self, X):
        X = np.asarray(X, float)
        out = np.zeros((len(X), len(self.root_value())))

        def go(node, idx):
            if node.value is not None:
                out[idx] = node.value
                return
            m = X[idx, node.feature] <= node.thr
            go(node.left, idx[m]); go(node.right, idx[~m])
        go(self.root, np.arange(len(X)))
        return out

    def root_value(self):
        n = self.root
        while n.value is None:
            n = n.left
        return n.value

    def predict_proba(self, X):
        return self._raw(X)

    def predict(self, X):
        raw = self._raw(X)
        return self.classes_[raw.argmax(1)] if self.task == "classification" else raw[:, 0]


@register("random_forest", family="classical")
class RandomForest(_Classical):
    """Bagged decision trees with random feature subsets (sqrt(d) for classification, d/3 for regression)."""

    def __init__(self, n_trees=50, max_depth=10, min_samples_leaf=1, max_features=None, task="auto", seed=None):
        self.n_trees, self.max_depth, self.min_leaf, self.max_features, self.task = n_trees, max_depth, min_samples_leaf, max_features, task
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.task = self._task(y, self.task)
        d = X.shape[1]
        mf = self.max_features or (max(1, int(np.sqrt(d))) if self.task == "classification" else max(1, d // 3))
        self.classes_ = np.unique(y) if self.task == "classification" else None
        self.trees = []
        for _ in range(self.n_trees):
            idx = self.rng.integers(len(X), size=len(X))
            t = DecisionTree(self.max_depth, self.min_leaf, mf, self.task, seed=int(self.rng.integers(1 << 30)))
            self.trees.append(t.fit(X[idx], y[idx], classes=self.classes_))
        return self

    def predict_proba(self, X):
        return np.mean([t.predict_proba(X) for t in self.trees], axis=0)

    def predict(self, X):
        p = self.predict_proba(X)
        return self.classes_[p.argmax(1)] if self.task == "classification" else p[:, 0]


@register("kmeans", family="classical")
class KMeans:
    """k-means with k-means++ seeding and several restarts."""
    family, module = "classical", None

    def __init__(self, n_clusters=3, n_init=5, max_iter=100, seed=None):
        self.k, self.n_init, self.max_iter, self.rng = n_clusters, n_init, max_iter, np.random.default_rng(seed)

    def fit(self, X, y=None):
        X, best = np.asarray(X, float), None
        for _ in range(self.n_init):
            c = X[[self.rng.integers(len(X))]]
            for _ in range(self.k - 1):
                d2 = ((X[:, None] - c[None]) ** 2).sum(-1).min(1)
                c = np.vstack([c, X[self.rng.choice(len(X), p=d2 / d2.sum()) if d2.sum() > 0 else self.rng.integers(len(X))]])
            for _ in range(self.max_iter):
                lab = ((X[:, None] - c[None]) ** 2).sum(-1).argmin(1)
                new = np.array([X[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(self.k)])
                if np.allclose(new, c):
                    break
                c = new
            inertia = float(((X - c[lab]) ** 2).sum())
            if best is None or inertia < best[0]:
                best = (inertia, c, lab)
        self.inertia_, self.cluster_centers_, self.labels_ = best
        return self

    def predict(self, X):
        return ((np.asarray(X, float)[:, None] - self.cluster_centers_[None]) ** 2).sum(-1).argmin(1)

    def transform(self, X):
        return np.sqrt(((np.asarray(X, float)[:, None] - self.cluster_centers_[None]) ** 2).sum(-1))

    def evaluate(self, X, y=None):
        return {"inertia": float(((np.asarray(X) - self.cluster_centers_[self.predict(X)]) ** 2).sum())}


@register("naive_bayes", family="classical")
class NaiveBayes(_Classical):
    """Gaussian naive Bayes."""

    def __init__(self, var_smoothing=1e-9):
        self.task, self.eps = "classification", var_smoothing

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.classes_ = np.unique(y)
        self.mu = np.array([X[y == c].mean(0) for c in self.classes_])
        self.var = np.array([X[y == c].var(0) for c in self.classes_]) + self.eps * X.var(0).max() + 1e-12
        self.prior = np.array([(y == c).mean() for c in self.classes_])
        return self

    def _log_joint(self, X):
        X = np.asarray(X, float)[:, None, :]
        ll = -0.5 * (np.log(2 * np.pi * self.var)[None] + (X - self.mu[None]) ** 2 / self.var[None]).sum(-1)
        return ll + np.log(self.prior)[None]

    def predict_proba(self, X):
        j = self._log_joint(X)
        p = np.exp(j - j.max(1, keepdims=True))
        return p / p.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[self._log_joint(X).argmax(1)]
