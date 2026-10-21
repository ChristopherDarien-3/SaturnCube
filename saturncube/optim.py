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


class ConjugateGradient(Optimizer):
    """Nonlinear conjugate gradient (Polak-Ribiere+) with backtracking line search.
    Full-batch method: use batch_size=len(X)."""
    uses_closure = True

    def __init__(self, params, lr=1.0, c1=1e-4, shrink=0.5, max_ls=25):
        super().__init__(params, dict(lr=lr, c1=c1, shrink=shrink, max_ls=max_ls))

    def step(self, closure):
        g0 = self.param_groups[0]
        params = [p for g in self.param_groups for p in g["params"]]
        st = self.state.setdefault("cg", {})
        with torch.enable_grad():
            loss = closure()
        grad = torch.cat([p.grad.reshape(-1) for p in params])
        d = -grad
        if "d" in st:
            beta = max(0.0, float(grad @ (grad - st["g"]) / (st["g"] @ st["g"] + 1e-12)))
            d = -grad + beta * st["d"]
            if grad @ d >= 0:
                d = -grad
        x0 = torch.nn.utils.parameters_to_vector(params).detach().clone()
        f0, slope, t = float(loss.detach()), float(grad @ d), min(g0["lr"], st.get("t", g0["lr"]) * 2)
        accepted = False
        for _ in range(g0["max_ls"]):
            torch.nn.utils.vector_to_parameters(x0 + t * d, params)
            with torch.enable_grad():
                f = float(closure())
            if f <= f0 + g0["c1"] * t * slope:
                accepted = True
                break
            t *= g0["shrink"]
        if not accepted:
            torch.nn.utils.vector_to_parameters(x0, params)
            f, d = f0, -grad
        st.update(d=d.detach(), g=grad.detach(), t=t)
        return torch.tensor(f)


class CoordinateDescent(Optimizer):
    """Block coordinate descent: each step moves only a subset of the coordinates
    (rule: 'random', 'cyclic' or 'greedy' = largest gradients, Gauss-Southwell)."""

    def __init__(self, params, lr=0.1, block_frac=0.25, rule="random"):
        if rule not in ("random", "cyclic", "greedy"):
            raise ValueError("rule must be 'random', 'cyclic' or 'greedy'")
        super().__init__(params, dict(lr=lr, block_frac=block_frac, rule=rule))
        self.t = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            frac, nb = g["block_frac"], max(1, round(1 / g["block_frac"]))
            for p in g["params"]:
                if p.grad is None:
                    continue
                if g["rule"] == "random":
                    mask = torch.rand_like(p) < frac
                elif g["rule"] == "cyclic":
                    mask = (torch.arange(p.numel(), device=p.device).view_as(p) % nb) == (self.t % nb)
                else:
                    k = max(1, int(frac * p.numel()))
                    thresh = p.grad.abs().flatten().topk(k).values[-1]
                    mask = p.grad.abs() >= thresh
                p.sub_(g["lr"] * p.grad * mask)
        self.t += 1
        return loss


class Shampoo(Optimizer):
    """Shampoo: Kronecker-factored preconditioning L^{-1/4} G R^{-1/4} of each matrix
    gradient, grafted onto the SGD step norm. Vectors use an Adagrad-style update."""

    def __init__(self, params, lr=0.05, eps=1e-4, update_freq=5):
        super().__init__(params, dict(lr=lr, eps=eps, update_freq=update_freq))

    @staticmethod
    def _inv_root(M, eps, power=4):
        w, V = torch.linalg.eigh(M)
        return (V * (w.clamp(min=eps) ** (-1.0 / power))) @ V.T

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if p.ndim < 2:
                    st["acc"] = st.get("acc", torch.zeros_like(p)) + p.grad ** 2
                    p.sub_(g["lr"] * p.grad / (st["acc"].sqrt() + g["eps"]))
                    continue
                G = p.grad.reshape(p.shape[0], -1)
                if not st:
                    st.update(t=0, L=g["eps"] * torch.eye(G.shape[0], device=p.device),
                              R=g["eps"] * torch.eye(G.shape[1], device=p.device))
                st["L"] += G @ G.T
                st["R"] += G.T @ G
                if st["t"] % g["update_freq"] == 0:
                    st["Li"], st["Ri"] = self._inv_root(st["L"], g["eps"]), self._inv_root(st["R"], g["eps"])
                st["t"] += 1
                P = st["Li"] @ G @ st["Ri"]
                P *= G.norm() / (P.norm() + 1e-12)  # graft onto the SGD step size
                p.sub_(g["lr"] * P.reshape(p.shape))
        return loss


class KFAC(Optimizer):
    """K-FAC for nn.Linear layers: gradient preconditioned by G^-1 (grad) A^-1, with
    A = E[a a^T] (layer inputs) and G = E[g g^T] (output-gradients); other params use SGD."""

    def __init__(self, params, module, lr=0.05, damping=1e-2, stat_decay=0.95, update_freq=5):
        super().__init__(params, dict(lr=lr))
        self.damping, self.decay, self.freq, self.k = damping, stat_decay, update_freq, 0
        self.layers = [m for m in module.modules() if type(m) is torch.nn.Linear]
        self.stats = {m: {} for m in self.layers}
        for m in self.layers:
            m.register_forward_pre_hook(self._save_input)
            m.register_full_backward_hook(self._save_grad)

    def _ema(self, st, key, new):
        st[key] = new if key not in st else self.decay * st[key] + (1 - self.decay) * new

    def _save_input(self, m, inp):
        if torch.is_grad_enabled() and m.training:
            a = inp[0].detach().reshape(-1, inp[0].shape[-1])
            a = torch.cat([a, torch.ones(len(a), 1, device=a.device)], 1)
            self._ema(self.stats[m], "A", a.T @ a / len(a))

    def _save_grad(self, m, grad_in, grad_out):
        g = grad_out[0].detach().reshape(-1, grad_out[0].shape[-1])
        self._ema(self.stats[m], "G", g.T @ g * len(g))  # undo the 1/B of the mean loss

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        lr, sq = self.param_groups[0]["lr"], self.damping ** 0.5
        handled = set()
        for m in self.layers:
            st = self.stats[m]
            if "A" not in st or "G" not in st or m.weight.grad is None:
                continue
            if self.k % self.freq == 0 or "Ai" not in st:
                eye_a, eye_g = torch.eye(len(st["A"]), device=m.weight.device), torch.eye(len(st["G"]), device=m.weight.device)
                st["Ai"], st["Gi"] = torch.linalg.inv(st["A"] + sq * eye_a), torch.linalg.inv(st["G"] + sq * eye_g)
            grad = torch.cat([m.weight.grad, m.bias.grad[:, None]], 1) if m.bias is not None else m.weight.grad
            pre = st["Gi"] @ grad @ st["Ai"]
            m.weight.sub_(lr * pre[:, : m.weight.shape[1]])
            if m.bias is not None:
                m.bias.sub_(lr * pre[:, -1])
            handled.update([id(m.weight), id(m.bias)])
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is not None and id(p) not in handled:
                    p.sub_(lr * p.grad)
        self.k += 1
        return loss


class NaturalGradient(Optimizer):
    """Natural gradient with a damped empirical Fisher matrix, solved exactly in the
    batch-sized dual space (Woodbury). Needs per-sample gradients, so it is a good fit for
    small networks without batch norm or dropout."""
    needs_per_sample = True

    def __init__(self, params, module, lr=0.05, damping=0.1):
        super().__init__(params, dict(lr=lr, damping=damping))

    def step_per_sample(self, model, m, xb, yb):
        from torch.func import functional_call, grad, vmap
        lr, lam = self.param_groups[0]["lr"], self.param_groups[0]["damping"]
        params = {n: p.detach() for n, p in m.named_parameters() if p.requires_grad}

        def loss_i(ps, x, y):
            return model._loss(functional_call(m, ps, (x.unsqueeze(0),)), y.unsqueeze(0))

        per = vmap(grad(loss_i), in_dims=(None, 0, 0), randomness="different")(params, xb, yb)
        G = torch.cat([per[n].reshape(len(xb), -1) for n in params], dim=1)  # (B, P)
        g, B = G.mean(0), len(xb)
        K = lam * B * torch.eye(B, device=G.device) + G @ G.T
        delta = (g - G.T @ torch.linalg.solve(K, G @ g)) / lam  # (lam I + G^T G / B)^-1 g
        with torch.no_grad():
            i = 0
            for p in (p for p in m.parameters() if p.requires_grad):
                p.sub_(lr * delta[i:i + p.numel()].view_as(p))
                i += p.numel()
            return model._loss(m(xb), yb).item()


# sensible starting learning rates per optimizer, used when lr=None
DEFAULT_LR = {"sgd": 0.05, "momentum": 0.02, "nesterov": 0.02, "adagrad": 0.05, "rmsprop": 1e-3,
              "adadelta": 1.0, "adam": 3e-3, "adamw": 3e-3, "adamax": 3e-3, "amsgrad": 3e-3,
              "radam": 3e-3, "lamb": 3e-3, "lion": 3e-4, "adafactor": 1e-2, "lbfgs": 1.0,
              "cg": 1.0, "coordinate_descent": 0.1, "shampoo": 0.05, "kfac": 0.05, "natural_gradient": 0.05}
OPTIMIZERS = sorted(DEFAULT_LR)


def make_optimizer(name, params, lr=None, weight_decay=0.0, module=None, **kw):
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
    if n == "cg":
        return ConjugateGradient(params, lr=lr, **kw)
    if n == "coordinate_descent":
        return CoordinateDescent(params, lr=lr, **kw)
    if n == "shampoo":
        return Shampoo(params, lr=lr, **kw)
    if n in ("kfac", "natural_gradient"):
        if module is None:
            raise ValueError(f"{n} needs the module; use it through a SaturnCube model")
        return (KFAC if n == "kfac" else NaturalGradient)(params, module, lr=lr, **kw)
    return O.LBFGS(params, lr=lr, **kw)  # lbfgs: weight_decay is not supported
