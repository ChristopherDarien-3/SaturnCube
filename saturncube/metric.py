"""Metric learning and few-shot learning (A4, B6).

Pair / triplet models take X of shape (n, 2, d) / (n, 3, d). Episodic models (prototypical,
relation, matching) train on labelled data but can classify *new* classes from a few support
examples: model.predict(X, support=(X_support, y_support)).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import layers as L
from .base import BaseModel, register
from .optim import make_optimizer


class _Embedder(BaseModel):
    family = "metric"

    def __init__(self, latent_dim=8, hidden=(64,), activation="relu", **kw):
        super().__init__(**kw)
        self.latent_dim, self.hidden, self.activation = latent_dim, tuple(hidden), activation

    def _make_encoder(self, d_in):
        return L.MLPNet(d_in, self.latent_dim, self.hidden, self.activation)

    def build(self, d_in, d_out):
        return self._make_encoder(d_in)

    def _init(self, X, y=None):  # pair / triplet models (no class labels)
        self.task, self.input_dim, self.output_dim = "metric", X.shape[-1], self.latent_dim
        if self.normalize:
            axes = tuple(range(X.ndim - 1))
            self._mu, self._sd = X.mean(axes), X.std(axes)
            self._sd[self._sd < 1e-8] = 1.0
        self._module = self.build(self.input_dim, self.latent_dim).to(self.device)

    def transform(self, X):
        """Embed items: X is (n, d) or (n, k, d)."""
        self._check_fitted()
        return self._outputs(self._norm(self._as_x(X))).cpu().numpy()


@register("siamese", family="metric")
class Siamese(_Embedder):
    """Siamese network (shared encoder) trained with the contrastive loss on pairs.
    X: (n, 2, d); y: 1 = similar, 0 = dissimilar."""
    def __init__(self, margin=1.0, **kw):
        super().__init__(**kw)
        self.margin = margin

    def _encode(self, y):
        return torch.as_tensor(np.asarray(y), dtype=torch.float32, device=self.device)

    def _loss(self, out, y):
        d = (out[:, 0] - out[:, 1]).norm(dim=-1)
        return (y * d ** 2 + (1 - y) * F.relu(self.margin - d) ** 2).mean()

    def fit(self, X, y, epochs=None, val_data=None):
        return super().fit(X, y, epochs=epochs)

    def distance(self, X):
        Z = self.transform(X)
        return np.linalg.norm(Z[:, 0] - Z[:, 1], axis=-1)

    def predict(self, X):
        """1 if the pair is judged similar (distance below margin / 2)."""
        return (self.distance(X) < self.margin / 2).astype(int)

    def evaluate(self, X, y):
        return {"accuracy": float((self.predict(X) == np.asarray(y)).mean())}


@register("triplet", family="metric")
class TripletNetwork(_Embedder):
    """Triplet network: X is (n, 3, d) = (anchor, positive, negative)."""
    def __init__(self, margin=1.0, **kw):
        super().__init__(**kw)
        self.margin = margin

    def _encode(self, y):
        return torch.zeros(len(y), device=self.device)

    def _loss(self, out, y):
        a, p, n = out[:, 0], out[:, 1], out[:, 2]
        return F.relu((a - p).norm(dim=-1) - (a - n).norm(dim=-1) + self.margin).mean()

    def fit(self, X, y=None, epochs=None, val_data=None):
        return super().fit(X, np.zeros(len(X)), epochs=epochs)

    def evaluate(self, X, y=None):
        Z = self.transform(X)
        ok = np.linalg.norm(Z[:, 0] - Z[:, 1], axis=-1) < np.linalg.norm(Z[:, 0] - Z[:, 2], axis=-1)
        return {"triplet_accuracy": float(ok.mean())}


class TwinTowerNet(nn.Module):
    def __init__(self, d_a, d_b, latent, hidden, activation):
        super().__init__()
        self.a, self.b = L.MLPNet(d_a, latent, hidden, activation), L.MLPNet(d_b, latent, hidden, activation)

    def forward(self, x):  # (n, 2, d): item from tower A and its partner from tower B
        return torch.stack([self.a(x[:, 0]), self.b(x[:, 1])], 1)


@register("twin_tower", family="metric")
class TwinTower(_Embedder):
    """Two separate encoders (e.g. query and item) trained so that matching pairs score
    highest among the batch (in-batch softmax). X: (n, 2, d) of matching pairs; both
    towers read the same feature width."""
    def __init__(self, temperature=0.2, **kw):
        super().__init__(**kw)
        self.temperature = temperature

    def build(self, d_in, d_out):
        return TwinTowerNet(d_in, d_in, self.latent_dim, self.hidden, self.activation)

    def _encode(self, y):
        return torch.zeros(len(y), device=self.device)

    def _loss(self, out, y):
        a, b = F.normalize(out[:, 0], dim=-1), F.normalize(out[:, 1], dim=-1)
        logits = a @ b.T / self.temperature
        t = torch.arange(len(a), device=a.device)
        return (F.cross_entropy(logits, t) + F.cross_entropy(logits.T, t)) / 2

    def fit(self, X, y=None, epochs=None, val_data=None):
        return super().fit(X, np.zeros(len(X)), epochs=epochs)

    def transform(self, X, tower=0):
        self._check_fitted()
        Xt = self._norm(self._as_x(X))
        self._module.eval()
        with torch.no_grad():
            return (self._module.a if tower == 0 else self._module.b)(Xt).cpu().numpy()

    def retrieve(self, queries, items, k=1):
        """Indices of the k best items for each query (cosine similarity)."""
        q, i = self.transform(queries, 0), self.transform(items, 1)
        q, i = q / np.linalg.norm(q, axis=1, keepdims=True), i / np.linalg.norm(i, axis=1, keepdims=True)
        return np.argsort(-(q @ i.T), axis=1)[:, :k]


@register("contrastive_embedding", family="metric")
class ContrastiveEmbedding(_Embedder):
    """Supervised contrastive embedding (SupCon): samples of the same class are pulled
    together, others pushed apart. fit(X, y); transform(X); predict by nearest class centroid."""
    def __init__(self, temperature=0.2, **kw):
        super().__init__(**kw)
        self.temperature, self.centroids_ = temperature, None

    def _init(self, X, y=None):
        BaseModel._init(self, X, y)

    def _loss(self, out, y):
        z = F.normalize(out, dim=-1)
        sim = z @ z.T / self.temperature
        eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(eye, float("-inf"))
        pos = (y[:, None] == y[None, :]) & ~eye
        logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        has = pos.any(1)
        return -((logp * pos).sum(1)[has] / pos.sum(1)[has]).mean() if has.any() else sim.sum() * 0

    def _fit(self, Xt, yt, epochs, val):
        super()._fit(Xt, yt, epochs, val)
        Z = self._outputs(Xt)
        self.centroids_ = torch.stack([Z[yt == c].mean(0) for c in range(len(self.classes_))])

    def predict(self, X):
        Z = self._outputs(self._norm(self._as_x(X)))
        return self.classes_[torch.cdist(Z, self.centroids_).argmin(1).cpu().numpy()]

    def evaluate(self, X, y):
        return {"accuracy": float((self.predict(X) == np.asarray(y)).mean())}


# ------------------------------------------------------------------ episodic models ---
class EpisodicNet(nn.Module):
    kind = "prototypical"

    def __init__(self, d_in, latent, hidden, activation, kind):
        super().__init__()
        self.enc, self.kind = L.MLPNet(d_in, latent, hidden, activation), kind
        if kind == "relation":
            self.rel = nn.Sequential(nn.Linear(2 * latent, latent), nn.ReLU(), nn.Linear(latent, 1))
        if kind == "matching":
            self.log_tau = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.enc(x)

    def scores(self, zq, zs, ys, n_way):
        """Logits (n_query, n_way) of each query against the support set."""
        onehot = F.one_hot(ys, n_way).float()
        if self.kind == "matching":
            att = F.softmax(F.normalize(zq, dim=-1) @ F.normalize(zs, dim=-1).T / (0.1 * self.log_tau.exp()), dim=1)
            return (att @ onehot + 1e-6).log()
        protos = (onehot.T @ zs) / onehot.sum(0).clamp(min=1)[:, None]
        if self.kind == "relation":
            pairs = torch.cat([zq.unsqueeze(1).expand(-1, n_way, -1), protos.unsqueeze(0).expand(len(zq), -1, -1)], -1)
            return self.rel(pairs).squeeze(-1)
        return -torch.cdist(zq, protos) ** 2


class _Episodic(BaseModel):
    family, kind = "metric", "prototypical"

    def __init__(self, latent_dim=16, hidden=(64,), activation="relu", n_way=None, k_shot=5, n_query=5,
                 episodes_per_epoch=20, support_per_class=20, **kw):
        super().__init__(**kw)
        self.latent_dim, self.hidden, self.activation = latent_dim, tuple(hidden), activation
        self.n_way, self.k_shot, self.n_query = n_way, k_shot, n_query
        self.episodes, self.support_per_class, self._support = episodes_per_epoch, support_per_class, None

    def build(self, d_in, d_out):
        return EpisodicNet(d_in, self.latent_dim, self.hidden, self.activation, self.kind)

    def _fit(self, Xt, yt, epochs, val):
        m = self._module
        opt = make_optimizer(self.optimizer, m.parameters(), self.lr, self.weight_decay, module=m, **self.optimizer_kwargs)
        by_class = [torch.nonzero(yt == c).flatten() for c in range(len(self.classes_))]
        usable = [c for c, idx in enumerate(by_class) if len(idx) >= 2]
        for _ in range(epochs):
            m.train()
            total = 0.0
            for _ in range(self.episodes):
                way = min(self.n_way or len(usable), len(usable))
                chosen = np.random.choice(usable, way, replace=False)
                xs, ys, xq, yq = [], [], [], []
                for j, c in enumerate(chosen):
                    idx = by_class[c][torch.randperm(len(by_class[c]))]
                    k = min(self.k_shot, len(idx) - 1)
                    xs.append(Xt[idx[:k]]); ys += [j] * k
                    q = idx[k:k + self.n_query]
                    xq.append(Xt[q]); yq += [j] * len(q)
                xs, xq = torch.cat(xs), torch.cat(xq)
                ys, yq = torch.tensor(ys, device=Xt.device), torch.tensor(yq, device=Xt.device)
                z = m(torch.cat([xs, xq]))
                loss = F.cross_entropy(m.scores(z[len(xs):], z[:len(xs)], ys, way), yq)
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item()
            self.history_["loss"].append(total / self.episodes)
        keep = torch.cat([idx[: self.support_per_class] for idx in by_class if len(idx)])
        self._support = (Xt[keep], yt[keep])

    def _logits(self, X, support):
        self._check_fitted()
        m = self._module
        m.eval()
        if support is None:
            xs, ys, classes = self._support[0], self._support[1], self.classes_
        else:
            Xs, ys_raw = np.asarray(support[0]), np.asarray(support[1])
            classes = np.unique(ys_raw)
            xs, ys = self._norm(self._as_x(Xs)), torch.as_tensor(np.searchsorted(classes, ys_raw)).to(self.device)
        with torch.no_grad():
            return m.scores(m(self._norm(self._as_x(X))), m(xs), ys, len(classes)), classes

    def predict_proba(self, X, support=None):
        return F.softmax(self._logits(X, support)[0], dim=-1).cpu().numpy()

    def predict(self, X, support=None):
        logits, classes = self._logits(X, support)
        return classes[logits.argmax(-1).cpu().numpy()]

    def evaluate(self, X, y, support=None):
        return {"accuracy": float((self.predict(X, support) == np.asarray(y)).mean())}

    def transform(self, X):
        self._check_fitted()
        return self._outputs(self._norm(self._as_x(X))).cpu().numpy()


@register("prototypical", family="metric")
class PrototypicalNetwork(_Episodic):
    """Classify by distance to class prototypes (mean embeddings); few-shot and one-shot
    (k_shot=1) learning of new classes via predict(X, support=...)."""
    kind = "prototypical"


@register("relation", family="metric")
class RelationNetwork(_Episodic):
    """A learned relation module scores each (query, prototype) pair."""
    kind = "relation"


@register("matching", family="metric")
class MatchingNetwork(_Episodic):
    """Attention over the support set (cosine similarity) votes for each class."""
    kind = "matching"
