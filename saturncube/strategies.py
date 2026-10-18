"""Plug-in strategies for continual learning, replay, distillation and active learning (B6)."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F


class Strategy:
    """Hooks called by the trainer. Pass instances via BaseModel(strategies=[...])."""

    def before_batch(self, xb, yb):
        return xb, yb

    def penalty(self, model, m, xb):
        return 0.0

    def after_task(self, model, Xt, yt):
        pass


def per_sample_loss(model, out, y):
    if model.task == "classification":
        return F.cross_entropy(out, y, reduction="none")
    return F.mse_loss(out, y, reduction="none").mean(1)


class EWC(Strategy):
    """Elastic Weight Consolidation: quadratic penalty on weights that were important
    (large diagonal Fisher information) for earlier tasks."""

    def __init__(self, lam=100.0, n_samples=500):
        self.lam, self.n, self.anchors = lam, n_samples, []

    def after_task(self, model, Xt, yt):
        m = model.module
        m.eval()
        fisher = {n: torch.zeros_like(p) for n, p in m.named_parameters()}
        idx = torch.randperm(len(Xt))[: self.n]
        for i in idx:
            m.zero_grad()
            model._loss(m(Xt[i:i + 1]), yt[i:i + 1]).backward()  # empirical Fisher
            for n, p in m.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2 / len(idx)
        m.zero_grad()
        self.anchors.append((fisher, {n: p.detach().clone() for n, p in m.named_parameters()}))

    def penalty(self, model, m, xb):
        total = 0.0
        for fisher, theta in self.anchors:
            for n, p in m.named_parameters():
                total = total + (fisher[n] * (p - theta[n]) ** 2).sum()
        return 0.5 * self.lam * total


class LwF(Strategy):
    """Learning without Forgetting: keep the new model's outputs on new data close to a
    frozen snapshot of the previous model (needs a fixed output head: pass classes=)."""

    def __init__(self, alpha=1.0, T=2.0):
        self.alpha, self.T, self.old, self.seen = alpha, T, None, None

    def after_task(self, model, Xt, yt):
        # distill only over classes the old model was actually trained on
        new = torch.unique(yt)
        self.seen = new if self.seen is None else torch.unique(torch.cat([self.seen, new]))
        self.old = copy.deepcopy(model.module).eval()
        for p in self.old.parameters():
            p.requires_grad_(False)

    def penalty(self, model, m, xb):
        if self.old is None:
            return 0.0
        with torch.no_grad():
            target = F.softmax(self.old(xb)[:, self.seen] / self.T, dim=1)
        logp = F.log_softmax(m(xb)[:, self.seen] / self.T, dim=1)
        return self.alpha * self.T ** 2 * F.kl_div(logp, target, reduction="batchmean")


class Replay(Strategy):
    """Experience replay: stored samples from earlier tasks are mixed into every batch.
    prioritized=True samples in proportion to (per-sample loss at storage time)**alpha
    (a simplified, task-level version of prioritized replay)."""

    def __init__(self, capacity=500, replay_size=None, prioritized=False, alpha=0.6):
        self.capacity, self.replay_size = capacity, replay_size
        self.prioritized, self.alpha = prioritized, alpha
        self.X = self.Y = self.prio = None

    def after_task(self, model, Xt, yt):
        X = Xt if self.X is None else torch.cat([self.X, Xt])
        Y = yt if self.Y is None else torch.cat([self.Y, yt])
        if len(X) > self.capacity:
            keep = torch.randperm(len(X), device=X.device)[: self.capacity]
            X, Y = X[keep], Y[keep]
        self.X, self.Y = X, Y
        with torch.no_grad():
            loss = per_sample_loss(model, model._outputs(X), Y)
        self.prio = (loss + 1e-3) ** self.alpha

    def before_batch(self, xb, yb):
        if self.X is None:
            return xb, yb
        k = self.replay_size or len(xb)
        if self.prioritized:
            idx = torch.multinomial(self.prio, k, replacement=True)
        else:
            idx = torch.randint(len(self.X), (k,), device=self.X.device)
        return torch.cat([xb, self.X[idx]]), torch.cat([yb, self.Y[idx]])


class Distillation(Strategy):
    """Adds alpha * T^2 * KL(teacher || student) (classification) or alpha * MSE (regression)."""

    def __init__(self, teacher, T=2.0, alpha=1.0):
        self.teacher, self.T, self.alpha = teacher, T, alpha

    def _teacher_input(self, model, xb):
        t = self.teacher
        raw = xb
        if model._mu is not None:
            raw = xb * torch.as_tensor(model._sd, device=xb.device) + torch.as_tensor(model._mu, device=xb.device)
        if t._mu is not None:
            raw = (raw - torch.as_tensor(t._mu, device=xb.device)) / torch.as_tensor(t._sd, device=xb.device)
        return raw

    def penalty(self, model, m, xb):
        t = self.teacher
        t.module.eval()
        with torch.no_grad():
            tout = t.module(self._teacher_input(model, xb))
        out = m(xb)
        if model.task == "classification":
            target = F.softmax(tout / self.T, dim=1)
            return self.alpha * self.T ** 2 * F.kl_div(F.log_softmax(out / self.T, dim=1), target,
                                                       reduction="batchmean")
        return self.alpha * F.mse_loss(out, tout)


def distill(teacher, student, X, y=None, T=2.0, alpha=1.0, epochs=None):
    """Train `student` to imitate `teacher` on X (labels default to the teacher's predictions)."""
    teacher._check_fitted()
    if teacher.task == "classification":
        student.task, student._declared_classes = "classification", teacher.classes_
    labels = teacher.predict(X) if y is None else y
    student.strategies.append(Distillation(teacher, T, alpha))
    try:
        student.fit(X, labels, epochs=epochs)
    finally:
        student.strategies.pop()
    return student


def query(model, X_pool, n=10, strategy="entropy", seed=None):
    """Active learning: indices of the n pool samples to label next."""
    if strategy == "random":
        return np.random.default_rng(seed).choice(len(X_pool), size=n, replace=False)
    p = model.predict_proba(X_pool)
    if strategy == "entropy":
        score = -(p * np.log(p + 1e-12)).sum(1)
    elif strategy == "margin":
        top2 = np.sort(p, axis=1)[:, -2:]
        score = -(top2[:, 1] - top2[:, 0])
    elif strategy == "least_confident":
        score = -p.max(1)
    else:
        raise ValueError("strategy must be 'entropy', 'margin', 'least_confident' or 'random'")
    return np.argsort(-score)[:n]
