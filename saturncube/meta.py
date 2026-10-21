"""Meta-learning, transfer, curriculum / self-paced learning and generative replay (B6)."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch.func import functional_call

from .strategies import Strategy, per_sample_loss


# ----------------------------------------------------------------------- task streams ---
@dataclass
class Task:
    support_x: np.ndarray
    support_y: np.ndarray
    query_x: np.ndarray
    query_y: np.ndarray


class SineTasks:
    """Regression tasks y = A sin(x + phase) with random amplitude and phase."""
    input_dim, output_dim, task = 1, 1, "regression"

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def sample(self, k_support=10, k_query=10):
        a, ph = self.rng.uniform(0.5, 3.0), self.rng.uniform(0, np.pi)
        x = self.rng.uniform(-5, 5, size=(k_support + k_query, 1)).astype(np.float32)
        y = (a * np.sin(x + ph)).astype(np.float32)
        return Task(x[:k_support], y[:k_support], x[k_support:], y[k_support:])


class GaussianTasks:
    """n_way classification tasks: random class centres, labels shuffled per task."""
    task = "classification"

    def __init__(self, d=8, n_way=3, spread=3.0, seed=None):
        self.input_dim, self.output_dim, self.spread = d, n_way, spread
        self.rng = np.random.default_rng(seed)

    def sample(self, k_support=5, k_query=15):
        d, n = self.input_dim, self.output_dim
        centers = self.rng.normal(size=(n, d)) * self.spread

        def draw(k):
            y = np.repeat(np.arange(n), k)
            return (centers[y] + self.rng.normal(size=(n * k, d))).astype(np.float32), y
        (xs, ys), (xq, yq) = draw(k_support), draw(k_query)
        return Task(xs, ys, xq, yq)


class _MetaBase:
    def __init__(self, model):
        if model.module is None:
            raise ValueError("call model.prepare(input_dim, output_dim, task) first")
        self.model, self.history = model, []

    def _xy(self, X, y):
        m = self.model
        X = torch.as_tensor(np.asarray(X, dtype=np.float32), device=m.device)
        y = np.asarray(y)
        y = torch.as_tensor(y).long() if m.task == "classification" else torch.as_tensor(y.astype(np.float32).reshape(len(y), -1))
        return X, y.to(m.device)

    def adapt(self, X, y, steps=None, lr=None):
        """Return a copy of the model fine-tuned on a handful of examples."""
        m = copy.deepcopy(self.model)
        X, y = self._xy(X, y)
        opt = torch.optim.SGD(m.module.parameters(), lr=lr or self.inner_lr)
        for _ in range(steps or max(self.inner_steps, 1)):
            opt.zero_grad(); m._loss(m.module(X), y).backward(); opt.step()
        return m

    def evaluate_tasks(self, tasks, k_support=10, k_query=10, steps=None, n=50):
        """Mean query loss after adapting to each task's support set."""
        out = []
        for _ in range(n):
            t = tasks.sample(k_support, k_query)
            a = self.adapt(t.support_x, t.support_y, steps)
            Xq, yq = self._xy(t.query_x, t.query_y)
            with torch.no_grad():
                out.append(a._loss(a.module(Xq), yq).item())
        return float(np.mean(out))


class MAML(_MetaBase):
    """Model-Agnostic Meta-Learning (Finn et al., 2017). first_order=True drops the
    second-order terms (FOMAML)."""

    def __init__(self, model, inner_lr=0.02, inner_steps=3, meta_lr=1e-3, first_order=False):
        super().__init__(model)
        self.inner_lr, self.inner_steps, self.meta_lr, self.first_order = inner_lr, inner_steps, meta_lr, first_order

    def meta_train(self, tasks, iterations=500, meta_batch=8, k_support=10, k_query=10, verbose=False):
        m = self.model
        opt = torch.optim.Adam(m.module.parameters(), lr=self.meta_lr)
        for it in range(iterations):
            opt.zero_grad()
            total = 0.0
            for _ in range(meta_batch):
                t = tasks.sample(k_support, k_query)
                (xs, ys), (xq, yq) = self._xy(t.support_x, t.support_y), self._xy(t.query_x, t.query_y)
                params = dict(m.module.named_parameters())
                for _ in range(self.inner_steps):
                    loss = m._loss(functional_call(m.module, params, (xs,)), ys)
                    grads = torch.autograd.grad(loss, list(params.values()), create_graph=not self.first_order)
                    params = {k: p - self.inner_lr * g for (k, p), g in zip(params.items(), grads)}
                qloss = m._loss(functional_call(m.module, params, (xq,)), yq) / meta_batch
                qloss.backward()
                total += qloss.item()
            opt.step()
            self.history.append(total)
            if verbose and it % max(1, iterations // 10) == 0:
                print(f"iter {it} meta-loss {total:.4f}")
        return self


class Reptile(_MetaBase):
    """Reptile (Nichol et al., 2018): run SGD on a task, then move the initial weights a
    fraction of the way toward the adapted weights."""

    def __init__(self, model, inner_lr=0.02, inner_steps=5, meta_lr=0.1):
        super().__init__(model)
        self.inner_lr, self.inner_steps, self.meta_lr = inner_lr, inner_steps, meta_lr

    def meta_train(self, tasks, iterations=500, k_support=10, k_query=10, verbose=False):
        m = self.model.module
        for it in range(iterations):
            t = tasks.sample(k_support, k_query)
            X, y = self._xy(np.vstack([t.support_x, t.query_x]), np.concatenate([t.support_y, t.query_y]))
            start = [p.detach().clone() for p in m.parameters()]
            opt = torch.optim.SGD(m.parameters(), lr=self.inner_lr)
            for _ in range(self.inner_steps):
                perm = torch.randperm(len(X))
                for idx in perm.split(10):
                    opt.zero_grad(); self.model._loss(m(X[idx]), y[idx]).backward(); opt.step()
            lr = self.meta_lr * (1 - it / iterations)  # linear decay, as in the paper
            with torch.no_grad():
                for p, s in zip(m.parameters(), start):
                    p.copy_(s + lr * (p - s))
            self.history.append(self.model._loss(m(X), y).item())
        return self


# -------------------------------------------------------------------- transfer learning ---
def transfer(source, target, input_dim=None, output_dim=None, task=None, classes=None, freeze=True):
    """Initialise `target` from a fitted `source` model: every parameter with the same name
    and shape is copied (typically everything except a new output head) and, if freeze=True,
    frozen. Then call target.fit(X, y) on the small new dataset."""
    source._check_fitted()
    task = task or source.task
    target.prepare(input_dim or source.input_dim, output_dim or source.output_dim, task, classes)
    src = source.module.state_dict()
    copied = []
    for name, p in target.module.named_parameters():
        if name in src and src[name].shape == p.shape:
            p.data.copy_(src[name])
            if freeze:
                p.requires_grad_(False)
            copied.append(name)
    if input_dim in (None, source.input_dim):  # keep the source's input standardisation
        target.normalize, target._mu, target._sd = source.normalize, source._mu, source._sd
    target.transferred_ = copied
    return target


# ------------------------------------------------------- curriculum / self-paced learning ---
def fit_curriculum(model, X, y, difficulty=None, stages=4, epochs_per_stage=None):
    """Train on progressively larger, easiest-first subsets. `difficulty` is an array or a
    callable (X, y) -> array; the default scores samples by the loss of a briefly trained
    pilot copy of the model."""
    X, y, n = np.asarray(X), np.asarray(y), len(X)
    if callable(difficulty):
        difficulty = difficulty(X, y)
    if difficulty is None:
        pilot = copy.deepcopy(model)
        pilot.fit(X, y, epochs=5)
        Xt, yt = pilot._norm(pilot._as_x(X)), pilot._encode(y)
        difficulty = per_sample_loss(pilot, pilot._outputs(Xt), yt).cpu().numpy()
    order = np.argsort(np.asarray(difficulty))
    model.curriculum_sizes_ = []
    for s in range(1, stages + 1):
        idx = order[: max(2, n * s // stages)]
        model.curriculum_sizes_.append(len(idx))
        model.fit(X[idx], y[idx], epochs=epochs_per_stage)
    return model


def fit_self_paced(model, X, y, rounds=6, growth=1.5, warmup_epochs=5, epochs_per_round=10, start_quantile=0.5):
    """Self-paced learning (Kumar et al., 2010): each round trains only on samples whose
    current loss is below a threshold lambda, which grows so harder samples join later."""
    X, y = np.asarray(X), np.asarray(y)
    model.fit(X, y, epochs=warmup_epochs)
    losses = lambda: per_sample_loss(model, model._outputs(model._norm(model._as_x(X))), model._encode(y)).cpu().numpy()
    loss = losses()
    lam = float(np.quantile(loss, start_quantile))
    model.selected_fraction_ = []
    for _ in range(rounds):
        sel = np.flatnonzero(loss <= lam)
        if len(sel) < 2:
            sel = np.argsort(loss)[:2]
        model.selected_fraction_.append(len(sel) / len(X))
        model.fit(X[sel], y[sel], epochs=epochs_per_round)
        loss, lam = losses(), lam * growth
    return model


# ------------------------------------------------------------------- generative replay ---
class GenerativeReplay(Strategy):
    """After each task a small VAE learns that task's inputs. Later tasks mix in generated
    samples, labelled by the model snapshot taken right after the VAE's own task, so no old
    data needs to be stored."""

    def __init__(self, latent_dim=4, hidden=(32,), epochs=80, n_replay=None):
        self.latent_dim, self.hidden, self.epochs, self.n_replay = latent_dim, hidden, epochs, n_replay
        self.memories = []  # (generator, labelling snapshot)

    def after_task(self, model, Xt, yt):
        from .unsupervised import VAE
        gen = VAE(latent_dim=self.latent_dim, hidden=self.hidden, epochs=self.epochs, normalize=False,
                  device=str(model.device)).fit(Xt.cpu().numpy())
        snap = copy.deepcopy(model.module).eval()
        self.memories.append((gen, snap))

    def before_batch(self, xb, yb):
        if not self.memories:
            return xb, yb
        k = self.n_replay or len(xb)
        gen, snap = self.memories[np.random.randint(len(self.memories))]
        xg = torch.as_tensor(gen.sample(k), dtype=torch.float32, device=xb.device)
        with torch.no_grad():
            yg = snap(xg).argmax(-1)
        return torch.cat([xb, xg]), torch.cat([yb, yg])
