"""Shared base class, trainer and registry for saturncube models."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .optim import make_optimizer

REGISTRY: dict = {}


def register(name, family="vector"):
    def deco(cls):
        cls.name, cls.family = name, family
        REGISTRY[name] = cls
        return cls
    return deco


def available(family=None):
    """Names of registered models, optionally filtered by family."""
    return sorted(n for n, c in REGISTRY.items() if family in (None, c.family))


def create(name, *args, **kwargs):
    """Build a model by registry name: create("kan", hidden=(16,))."""
    key = name.lower()
    if key not in REGISTRY:
        raise ValueError(f"unknown model {name!r}; available: {available()}")
    return REGISTRY[key](*args, **kwargs)


class BaseModel:
    """One interface for every model: fit / predict / evaluate / save / module.

    Subclasses implement build(d_in, d_out) -> nn.Module. Models that are not
    trained by gradient descent (e.g. ELM) override _fit().
    """

    name = "base"
    family = "vector"

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        obj._init_args = (args, kwargs)  # lets load() rebuild the same object
        return obj

    def __init__(self, input_dim=None, output_dim=None, task="auto", lr=None,
                 epochs=60, batch_size=64, weight_decay=0.0, normalize=True,
                 patience=None, device=None, seed=None, verbose=False,
                 optimizer="adam", optimizer_kwargs=None, classes=None, strategies=None):
        if task not in ("auto", "classification", "regression"):
            raise ValueError("task must be 'auto', 'classification' or 'regression'")
        self.input_dim, self.output_dim, self.task = input_dim, output_dim, task
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.weight_decay, self.normalize = weight_decay, normalize
        self.patience, self.verbose = patience, verbose
        self.optimizer, self.optimizer_kwargs = optimizer, dict(optimizer_kwargs or {})
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self._declared_classes = None if classes is None else np.unique(classes)
        self.strategies = list(strategies or [])  # continual-learning / distillation plug-ins
        self._module = None
        self.classes_ = None
        self._mu = self._sd = None
        self.history_ = {"loss": [], "val_loss": []}

    # ---- to implement ------------------------------------------------------
    def build(self, d_in, d_out):
        raise NotImplementedError

    # ---- escape hatch --------------------------------------------------------
    @property
    def module(self):
        """The raw torch.nn.Module (available after fit or load)."""
        return self._module

    @property
    def n_params(self):
        return 0 if self._module is None else sum(p.numel() for p in self._module.parameters())

    # ---- data handling -------------------------------------------------------
    @staticmethod
    def _as_x(X):
        X = np.asarray(X, dtype=np.float32)
        return X.reshape(-1, 1) if X.ndim == 1 else X

    def _init(self, X, y):
        y = np.asarray(y)
        if self.task == "auto":
            # integer / bool / string labels -> classification, floats -> regression
            self.task = "classification" if y.dtype.kind in "biuOUS" else "regression"
        d_in = X.shape[-1]
        if self.input_dim not in (None, d_in):
            raise ValueError(f"input_dim={self.input_dim} but X has {d_in} features")
        if self.task == "classification":
            self.classes_ = self._declared_classes if self._declared_classes is not None else np.unique(y)
            d_out = len(self.classes_)
        else:
            d_out = 1 if y.ndim == 1 else y.shape[-1]
        if self.output_dim not in (None, d_out):
            raise ValueError(f"output_dim={self.output_dim} but data implies {d_out}")
        self.input_dim, self.output_dim = d_in, d_out
        if self.normalize:
            axes = tuple(range(X.ndim - 1))  # per-feature statistics (also for 3-D sequences)
            self._mu, self._sd = X.mean(axes), X.std(axes)
            self._sd[self._sd < 1e-8] = 1.0
        self._module = self.build(d_in, d_out).to(self.device)
        if hasattr(self._module, "init_from_data"):
            self._module.init_from_data(self._norm(X))

    def _norm(self, X):
        if self.normalize and self._mu is not None:
            X = (X - self._mu) / self._sd
        return torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(self.device)

    def _encode(self, y):
        y = np.asarray(y)
        if self.task == "classification":
            idx = np.clip(np.searchsorted(self.classes_, y), 0, len(self.classes_) - 1)
            if not np.all(self.classes_[idx] == y):
                raise ValueError("y contains labels not seen when the model was first fitted")
            return torch.from_numpy(idx).long().to(self.device)
        y = y.astype(np.float32)
        return torch.from_numpy(y.reshape(len(y), -1) if y.ndim <= 2 else y).to(self.device)

    def _loss(self, out, y):
        if self.task == "classification":
            if out.dim() == 3:  # per-step outputs of sequence models
                out, y = out.reshape(-1, out.shape[-1]), y.reshape(-1)
            return F.cross_entropy(out, y)
        return F.mse_loss(out, y)

    # ---- training ------------------------------------------------------------
    def fit(self, X, y, epochs=None, val_data=None):
        X = self._as_x(X)
        if self._module is None:
            self._init(X, y)
        Xt, yt = self._norm(X), self._encode(y)
        val = None
        if val_data is not None:
            val = (self._norm(self._as_x(val_data[0])), self._encode(val_data[1]))
        self._fit(Xt, yt, epochs or self.epochs, val)
        return self

    def prepare(self, input_dim, output_dim, task="regression", classes=None):
        """Build the network without data (needed by meta-learners and transfer learning)."""
        self.task, self.input_dim, self.output_dim, self.normalize = task, input_dim, output_dim, False
        if task == "classification":
            self.classes_ = np.arange(output_dim) if classes is None else np.asarray(classes)
        self._module = self.build(input_dim, output_dim).to(self.device)
        return self

    def _train_batch(self, m, opt, xb, yb):
        """One optimisation step; returns the batch loss as a float.
        Override for learning rules that do not use ordinary backpropagation."""
        for s in self.strategies:  # e.g. replay adds stored samples to the batch
            xb, yb = s.before_batch(xb, yb)
        if getattr(opt, "needs_per_sample", False):  # natural gradient
            return opt.step_per_sample(self, m, xb, yb)

        def closure():
            opt.zero_grad()
            loss = self._loss(m(xb), yb)
            aux = getattr(m, "aux_loss", None)  # e.g. MoE load balancing
            if aux is not None:
                loss = loss + aux
            for s in self.strategies:  # e.g. EWC / LwF / distillation penalties
                loss = loss + s.penalty(self, m, xb)
            loss.backward()
            return loss
        if isinstance(opt, torch.optim.LBFGS) or getattr(opt, "uses_closure", False):
            return opt.step(closure).item()
        loss = closure()
        opt.step()
        return loss.item()

    def partial_fit(self, X, y):
        """One epoch on new data (online / incremental learning)."""
        return self.fit(X, y, epochs=1)

    def learn_task(self, X, y, epochs=None):
        """Train on one task, then let strategies consolidate (EWC Fisher, LwF snapshot, replay buffer)."""
        self.fit(X, y, epochs=epochs)
        Xt, yt = self._norm(self._as_x(X)), self._encode(y)
        for s in self.strategies:
            s.after_task(self, Xt, yt)
        return self

    def _fit(self, Xt, yt, epochs, val):
        m = self._module
        opt = make_optimizer(self.optimizer, m.parameters(), self.lr, self.weight_decay,
                             module=m, **self.optimizer_kwargs)
        n, best, wait, best_state = len(Xt), float("inf"), 0, None
        for ep in range(epochs):
            m.train()
            perm = torch.randperm(n, device=Xt.device)
            total = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                if len(idx) < 2 and n > 1:  # BatchNorm needs >1 sample
                    continue
                loss = self._train_batch(m, opt, Xt[idx], yt[idx])
                total += loss * len(idx)
            self.history_["loss"].append(total / n)
            if val is not None:
                with torch.no_grad():
                    v = self._loss(self._outputs(val[0]), val[1]).item()
                self.history_["val_loss"].append(v)
                if self.patience is not None:
                    if v < best - 1e-9:
                        best, wait = v, 0
                        best_state = {k: t.clone() for k, t in m.state_dict().items()}
                    else:
                        wait += 1
                        if wait >= self.patience:
                            break
            if self.verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
                msg = f"epoch {ep + 1}/{epochs} loss={self.history_['loss'][-1]:.4f}"
                if val is not None:
                    msg += f" val_loss={self.history_['val_loss'][-1]:.4f}"
                print(msg)
        if best_state is not None:
            m.load_state_dict(best_state)

    # ---- inference -----------------------------------------------------------
    def _outputs(self, Xt):
        self._module.eval()
        with torch.no_grad():
            return torch.cat([self._module(Xt[i:i + 4096]) for i in range(0, len(Xt), 4096)])

    def _check_fitted(self):
        if self._module is None:
            raise RuntimeError("model is not fitted yet; call fit() or load() first")

    def predict_proba(self, X):
        self._check_fitted()
        if self.task != "classification":
            raise ValueError("predict_proba is only defined for classification")
        return F.softmax(self._outputs(self._norm(self._as_x(X))), dim=-1).cpu().numpy()

    def predict(self, X):
        self._check_fitted()
        out = self._outputs(self._norm(self._as_x(X)))
        if self.task == "classification":
            return self.classes_[out.argmax(-1).cpu().numpy()]
        out = out.cpu().numpy()
        return out[..., 0] if out.shape[-1] == 1 else out

    def evaluate(self, X, y):
        self._check_fitted()
        Xt, yt = self._norm(self._as_x(X)), self._encode(y)
        out = self._outputs(Xt)
        loss = self._loss(out, yt).item()
        if self.task == "classification":
            return {"loss": loss, "accuracy": (out.argmax(-1) == yt).float().mean().item()}
        mse = F.mse_loss(out, yt).item()
        var = yt.reshape(-1, yt.shape[-1]).var(0, unbiased=False).mean().item()
        return {"loss": loss, "mse": mse, "rmse": mse ** 0.5, "r2": 1 - mse / max(var, 1e-12)}

    # ---- persistence ---------------------------------------------------------
    def save(self, path):
        self._check_fitted()
        args, kwargs = self._init_args
        kwargs = {k: v for k, v in kwargs.items() if k != "strategies"}
        torch.save({
            "cls": type(self).__name__, "init_args": (args, kwargs),
            "state": self._module.state_dict(), "task": self.task,
            "dims": (self.input_dim, self.output_dim), "classes": self.classes_,
            "norm": (self._mu, self._sd),
        }, path)

    @classmethod
    def load(cls, path, device=None):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        args, kwargs = ck["init_args"]
        if device is not None:
            kwargs = {**kwargs, "device": device}
        obj = cls(*args, **kwargs)
        obj.task, obj.classes_ = ck["task"], ck["classes"]
        obj.input_dim, obj.output_dim = ck["dims"]
        obj._mu, obj._sd = ck["norm"]
        obj._module = obj.build(*ck["dims"]).to(obj.device)
        obj._module.load_state_dict(ck["state"])
        return obj

    def __repr__(self):
        return (f"{type(self).__name__}(task={self.task}, input_dim={self.input_dim}, "
                f"output_dim={self.output_dim}, params={self.n_params})")
