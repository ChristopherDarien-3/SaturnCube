"""Optimizer factory: torch.optim plus Lion and LAMB."""
from __future__ import annotations

import torch
from torch.optim import Optimizer


class Lion(Optimizer):
    """Lion (Chen et al., 2023): sign of an interpolated momentum, decoupled weight decay."""

    def __init__(self, params, lr=3e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                m = self.state[p].setdefault("exp_avg", torch.zeros_like(p))
                p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(torch.sign(m * b1 + p.grad * (1 - b1)), alpha=-g["lr"])
                m.mul_(b2).add_(p.grad, alpha=1 - b2)
        return loss


class LAMB(Optimizer):
    """LAMB (You et al., 2020): Adam update rescaled per tensor by ||w|| / ||update||."""

    def __init__(self, params, lr=3e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["t"], st["m"], st["v"] = 0, torch.zeros_like(p), torch.zeros_like(p)
                st["t"] += 1
                st["m"].mul_(b1).add_(p.grad, alpha=1 - b1)
                st["v"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                m_hat = st["m"] / (1 - b1 ** st["t"])
                v_hat = st["v"] / (1 - b2 ** st["t"])
                update = m_hat / (v_hat.sqrt() + g["eps"]) + g["weight_decay"] * p
                w, u = p.norm(), update.norm()
                trust = (w / u) if (w > 0 and u > 0) else 1.0
                p.add_(update, alpha=-g["lr"] * float(trust))
        return loss


# sensible starting learning rates per optimizer, used when lr=None
DEFAULT_LR = {"sgd": 0.05, "momentum": 0.02, "nesterov": 0.02, "adagrad": 0.05, "rmsprop": 1e-3,
              "adadelta": 1.0, "adam": 3e-3, "adamw": 3e-3, "adamax": 3e-3, "amsgrad": 3e-3,
              "radam": 3e-3, "lamb": 3e-3, "lion": 3e-4, "adafactor": 1e-2, "lbfgs": 1.0}
OPTIMIZERS = sorted(DEFAULT_LR)


def make_optimizer(name, params, lr=None, weight_decay=0.0, **kw):
    n = name.lower()
    if n not in DEFAULT_LR:
        raise ValueError(f"unknown optimizer {name!r}; available: {OPTIMIZERS}")
    lr = lr if lr is not None else DEFAULT_LR[n]
    O, wd = torch.optim, weight_decay
    if n == "sgd":
        return O.SGD(params, lr=lr, weight_decay=wd, **kw)
    if n in ("momentum", "nesterov"):
        return O.SGD(params, lr=lr, weight_decay=wd, **{"momentum": 0.9, "nesterov": n == "nesterov", **kw})
    if n == "adagrad":
        return O.Adagrad(params, lr=lr, weight_decay=wd, **kw)
    if n == "rmsprop":
        return O.RMSprop(params, lr=lr, weight_decay=wd, **kw)
    if n == "adadelta":
        return O.Adadelta(params, lr=lr, weight_decay=wd, **kw)
    if n == "adam":
        return O.Adam(params, lr=lr, weight_decay=wd, **kw)
    if n == "amsgrad":
        return O.Adam(params, lr=lr, weight_decay=wd, **{"amsgrad": True, **kw})
    if n == "adamw":
        return O.AdamW(params, lr=lr, weight_decay=wd or 0.01, **kw)
    if n == "adamax":
        return O.Adamax(params, lr=lr, weight_decay=wd, **kw)
    if n == "radam":
        return O.RAdam(params, lr=lr, weight_decay=wd, **kw)
    if n == "adafactor":
        return O.Adafactor(params, lr=lr, weight_decay=wd, **kw)
    if n == "lamb":
        return LAMB(params, lr=lr, weight_decay=wd, **kw)
    if n == "lion":
        return Lion(params, lr=lr, weight_decay=wd, **kw)
    return O.LBFGS(params, lr=lr, **kw)  # lbfgs: weight_decay is not supported
