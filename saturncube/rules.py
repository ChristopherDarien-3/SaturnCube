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
