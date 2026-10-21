"""Learning rules that replace ordinary backpropagation (vector family)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import layers as L
from .base import BaseModel, register


class _LinearFAFn(torch.autograd.Function):
    """Linear layer whose backward pass sends the error through a fixed random matrix B
    instead of W^T (Lillicrap et al., 2016)."""

    @staticmethod
    def forward(ctx, x, W, b, B):
        ctx.save_for_backward(x, B)
        return F.linear(x, W, b)

    @staticmethod
    def backward(ctx, g):
        x, B = ctx.saved_tensors
        return g @ B, g.T @ x, g.sum(0), None


class FALinear(nn.Linear):
    def __init__(self, d_in, d_out):
        super().__init__(d_in, d_out)
        self.register_buffer("B", torch.randn(d_out, d_in) / math.sqrt(d_in))

    def forward(self, x):
        return _LinearFAFn.apply(x, self.weight, self.bias, self.B)


class FANet(nn.Sequential):
    def __init__(self, d_in, d_out, hidden=(128, 64), activation="tanh"):
        layers, d = [], d_in
        for h in hidden:
            layers += [FALinear(d, h), L.act(activation)]
            d = h
        layers.append(FALinear(d, d_out))
        super().__init__(*layers)


class DFANet(nn.Module):
    """Hidden layers learn from the output error projected straight to each layer by a
    fixed random matrix (Nokland, 2016); no error flows layer to layer."""

    def __init__(self, d_in, d_out, hidden=(128, 64), activation="tanh"):
        super().__init__()
        dims = [d_in, *hidden]
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.act, self.readout = L.act(activation), nn.Linear(dims[-1], d_out)
        for i, h in enumerate(hidden):
            self.register_buffer(f"B{i}", torch.randn(d_out, h) / math.sqrt(d_out))

    @property
    def feedback(self):
        return [getattr(self, f"B{i}") for i in range(len(self.layers))]

    def forward_with_hidden(self, x):
        hs, h = [], x
        for layer in self.layers:
            h = self.act(layer(h))
            hs.append(h)
            h = h.detach()  # block the gradient path between layers
        return self.readout(h), hs

    def forward(self, x):
        return self.forward_with_hidden(x)[0]


@register("fa")
class FeedbackAlignment(BaseModel):
    """MLP trained with Feedback Alignment (random fixed backward weights)."""
    def __init__(self, hidden=(128, 64), activation="tanh", **kw):
        super().__init__(**kw)
        self.hidden, self.activation = hidden, activation

    def build(self, d_in, d_out):
        return FANet(d_in, d_out, self.hidden, self.activation)


@register("dfa")
class DirectFeedbackAlignment(BaseModel):
    """MLP trained with Direct Feedback Alignment (output error sent to every hidden layer)."""
    def __init__(self, hidden=(128, 64), activation="tanh", **kw):
        super().__init__(**kw)
        self.hidden, self.activation = hidden, activation

    def build(self, d_in, d_out):
        return DFANet(d_in, d_out, self.hidden, self.activation)

    def _train_batch(self, m, opt, xb, yb):
        opt.zero_grad()
        out, hs = m.forward_with_hidden(xb)
        loss = self._loss(out, yb)
        (e,) = torch.autograd.grad(loss, out, retain_graph=True)  # dL/d(output)
        loss.backward()                                           # true gradient for the readout
        for h, B in zip(hs, m.feedback):
            h.backward(e @ B)                                     # random projection of the error
        opt.step()
        return loss.item()


# ===================================================== more learning rules (B1) ===
from torch.func import functional_call, jvp  # noqa: E402


@register("forward_gradient")
class ForwardGradient(BaseModel):
    """Forward-mode differentiation: the gradient is estimated from a single forward pass
    with a random tangent, g = (dL . v) v (Baydin et al., 2022). No backward pass at all.
    The estimate is unbiased but noisy, so small networks and several tangents work best."""
    def __init__(self, hidden=(32,), activation="tanh", n_tangents=4, **kw):
        super().__init__(**kw)
        self.hidden, self.activation, self.n_tangents = hidden, activation, n_tangents

    def build(self, d_in, d_out):
        return L.MLPNet(d_in, d_out, self.hidden, self.activation)

    def _train_batch(self, m, opt, xb, yb):
        params = {n: p.detach() for n, p in m.named_parameters()}
        grads = {n: torch.zeros_like(p) for n, p in params.items()}

        def f(ps):
            return self._loss(functional_call(m, ps, (xb,)), yb)

        for _ in range(self.n_tangents):
            v = {n: torch.randn_like(p) for n, p in params.items()}
            loss, dl = jvp(f, (params,), (v,))
            for n in grads:
                grads[n] += dl * v[n] / self.n_tangents
        opt.zero_grad()
        for n, p in m.named_parameters():
            p.grad = grads[n]
        opt.step()
        return loss.item()


class LocalNet(nn.Module):
    def __init__(self, d_in, d_out, hidden, activation):
        super().__init__()
        dims = [d_in, *hidden]
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(a, b), L.act(activation))
                                     for a, b in zip(dims[:-1], dims[1:])])
        self.heads = nn.ModuleList([nn.Linear(h, d_out) for h in hidden])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.heads[-1](x)

    def forward_all(self, x):
        outs = []
        for b, h in zip(self.blocks, self.heads):
            x = b(x.detach())  # no gradient flows between layers
            outs.append(h(x))
        return outs


@register("local_error")
class LocalErrorLearning(BaseModel):
    """Local-gradient learning: every hidden layer has its own small classifier and is
    trained on its own loss only; nothing is backpropagated between layers."""
    def __init__(self, hidden=(64, 32), activation="relu", **kw):
        super().__init__(**kw)
        self.hidden, self.activation = hidden, activation

    def build(self, d_in, d_out):
        return LocalNet(d_in, d_out, self.hidden, self.activation)

    def _train_batch(self, m, opt, xb, yb):
        opt.zero_grad()
        outs = m.forward_all(xb)
        sum(self._loss(o, yb) for o in outs).backward()
        opt.step()
        return self._loss(outs[-1], yb).item()


class SGNet(nn.Module):
    def __init__(self, d_in, d_out, hidden, activation):
        super().__init__()
        dims = [d_in, *hidden]
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(a, b), L.act(activation))
                                     for a, b in zip(dims[:-1], dims[1:])])
        self.head = nn.Linear(hidden[-1], d_out)
        self.sg = nn.ModuleList()  # synthetic-gradient predictors, zero-initialised
        for h in hidden:
            net = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
            self.sg.append(net)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.head(x)


@register("synthetic_gradients")
class SyntheticGradients(BaseModel):
    """Decoupled training (Jaderberg et al., 2017): each layer is updated immediately using a
    gradient *predicted* from its activations; predictors learn from the true gradients
    that arrive from the layer above."""
    def __init__(self, hidden=(64, 32), activation="relu", **kw):
        super().__init__(**kw)
        self.hidden, self.activation = hidden, activation

    def build(self, d_in, d_out):
        return SGNet(d_in, d_out, self.hidden, self.activation)

    def _train_batch(self, m, opt, xb, yb):
        opt.zero_grad()
        B, inp, sg_losses, prev = len(xb), xb, [], None
        for i, block in enumerate(m.blocks):
            x_in = inp.detach().requires_grad_(i > 0)
            h = block(x_in)
            pred = m.sg[i](h.detach())
            h.backward(pred.detach() / B)  # update this block with the predicted gradient
            if i > 0:
                sg_losses.append(F.mse_loss(prev, x_in.grad * B))
            prev, inp = pred, h.detach()
        x_in = inp.requires_grad_(True)
        loss = self._loss(m.head(x_in), yb)
        loss.backward()
        sg_losses.append(F.mse_loss(prev, x_in.grad * B))
        sum(sg_losses).backward()  # trains the predictors only (targets are detached)
        opt.step()
        return loss.item()


class EnergyNet(nn.Module):
    """Layered Hopfield-style energy network. Hidden/output states relax to a minimum of
    E = sum 1/2|s|^2 - sum rho(s_l)^T (W_l rho(s_{l-1}) + b_l), with rho = clamp(., 0, 1)."""

    def __init__(self, d_in, d_out, hidden, T_free=30, T_nudge=12, dt=0.4):
        super().__init__()
        dims = [d_in, *hidden, d_out]
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.T_free, self.T_nudge, self.dt = T_free, T_nudge, dt

    def energy(self, states, x):
        rho, E, prev = (lambda s: s.clamp(0, 1)), 0.0, x
        for s, layer in zip(states, self.layers):
            E = E + 0.5 * (s ** 2).sum() - (rho(s) * layer(prev)).sum()
            prev = rho(s)
        return E

    def relax(self, x, steps, target=None, beta=None, states=None, clamp_output=False):
        if states is None:  # feedforward initialisation
            states, h = [], x
            for layer in self.layers:
                h = layer(h.detach().clamp(0, 1) if len(states) else h).detach()
                states.append(h.clone())
        states = [s.detach().clone() for s in states]
        if clamp_output:
            states[-1] = target.clone()
        for _ in range(steps):
            with torch.enable_grad():
                ss = [s.requires_grad_(True) for s in states]
                E = self.energy(ss, x)
                if beta is not None and not clamp_output:
                    E = E + beta * 0.5 * ((ss[-1] - target) ** 2).sum()
                gs = torch.autograd.grad(E, ss)
            states = [(s - self.dt * g).clamp(0, 1).detach() for s, g in zip(states, gs)]
            if clamp_output:
                states[-1] = target.clone()
        return states

    def forward(self, x):
        return self.relax(x, self.T_free)[-1]


@register("equilibrium_propagation")
class EquilibriumPropagation(BaseModel):
    """Equilibrium Propagation (Scellier & Bengio, 2017): a free phase, then a weakly
    nudged phase; the weight update is the contrast of the two equilibria divided by beta.
    Set mode='chl' for Contrastive Hebbian Learning (output fully clamped, no division)."""
    def __init__(self, hidden=(32,), beta=0.5, mode="ep", T_free=30, T_nudge=12, **kw):
        if mode not in ("ep", "chl"):
            raise ValueError("mode must be 'ep' or 'chl'")
        super().__init__(**kw)
        self.hidden, self.beta, self.mode, self.T_free, self.T_nudge = hidden, beta, mode, T_free, T_nudge

    def build(self, d_in, d_out):
        return EnergyNet(d_in, d_out, self.hidden, self.T_free, self.T_nudge)

    def _target(self, yb, d_out):
        return F.one_hot(yb, d_out).float() if self.task == "classification" else yb

    def _loss(self, out, y):
        t = F.one_hot(y, out.shape[-1]).float() if self.task == "classification" else y
        return F.mse_loss(out, t)

    def _train_batch(self, m, opt, xb, yb):
        t = self._target(yb, m.layers[-1].out_features)
        free = m.relax(xb, m.T_free)
        if self.mode == "ep":
            other = m.relax(xb, m.T_nudge, target=t, beta=self.beta, states=free)
        else:
            other = m.relax(xb, m.T_free, target=t, states=free, clamp_output=True)
        params = list(m.parameters())
        g_free = torch.autograd.grad(m.energy([s.requires_grad_(False) for s in free], xb), params)
        g_other = torch.autograd.grad(m.energy([s.requires_grad_(False) for s in other], xb), params)
        scale = (1.0 / self.beta if self.mode == "ep" else 1.0) / len(xb)
        opt.zero_grad()
        for p, a, b in zip(params, g_other, g_free):
            p.grad = (a - b) * scale
        opt.step()
        return self._loss(free[-1], yb).item()


@register("contrastive_hebbian")
class ContrastiveHebbian(EquilibriumPropagation):
    """Contrastive Hebbian Learning: clamped-phase minus free-phase Hebbian statistics."""
    def __init__(self, **kw):
        super().__init__(mode="chl", **kw)


class PCNet(nn.Module):
    def __init__(self, d_in, d_out, hidden, activation):
        super().__init__()
        dims = [d_in, *hidden, d_out]
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.act = L.act(activation)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x if i == 0 else self.act(x))
        return x


@register("predictive_coding")
class PredictiveCoding(BaseModel):
    """Predictive-coding network (hierarchical when it has several hidden layers). Each layer
    predicts the one below it; the output is clamped to the target, hidden states relax to
    reduce local prediction errors, and weights update from local error x activity products."""
    def __init__(self, hidden=(32,), activation="tanh", inference_steps=25, inference_lr=0.15, **kw):
        super().__init__(**kw)
        self.hidden, self.activation = hidden, activation
        self.steps, self.eta = inference_steps, inference_lr

    def build(self, d_in, d_out):
        return PCNet(d_in, d_out, self.hidden, self.activation)

    def _train_batch(self, m, opt, xb, yb):
        f = m.act
        df = (lambda x: 1 - torch.tanh(x) ** 2) if self.activation == "tanh" else None
        layers, B = list(m.layers), len(xb)
        with torch.no_grad():
            target = F.one_hot(yb, layers[-1].out_features).float() if self.task == "classification" else yb
            acts, mus, x = [xb], [], []
            for i, layer in enumerate(layers):  # feedforward initialisation
                mu = layer(acts[-1]); mus.append(mu); x.append(mu.clone()); acts.append(f(mu))
            out_loss = self._loss(mus[-1], yb).item()
            x[-1] = target
            for _ in range(self.steps):
                mus = [layers[0](xb)] + [layers[i](f(x[i - 1])) for i in range(1, len(layers))]
                eps = [x[i] - mus[i] for i in range(len(layers))]
                for i in range(len(layers) - 1):  # hidden states only
                    back = eps[i + 1] @ layers[i + 1].weight
                    d = df(x[i]) if df else (x[i] > 0).float()
                    x[i] = x[i] + self.eta * (-eps[i] + d * back)
            mus = [layers[0](xb)] + [layers[i](f(x[i - 1])) for i in range(1, len(layers))]
            eps = [x[i] - mus[i] for i in range(len(layers))]
            inputs = [xb] + [f(s) for s in x[:-1]]
            opt.zero_grad()
            for layer, e, a in zip(layers, eps, inputs):
                layer.weight.grad = -(e.T @ a) / B
                layer.bias.grad = -e.mean(0)
            opt.step()
        return out_loss
