"""Spiking and biologically inspired models: a leaky integrate-and-fire spiking network trained with
surrogate gradients, and a Self-Organizing Map."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel, register
from .unsupervised import Unsupervised


# ------------------------------------------------------------------- spiking network ---
class _Spike(torch.autograd.Function):
    """Heaviside spike in the forward pass; the backward pass uses a smooth surrogate derivative
    (fast sigmoid) so the network can be trained with ordinary backpropagation through time."""

    @staticmethod
    def forward(ctx, x, slope):
        ctx.save_for_backward(x)
        ctx.slope = slope
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad / (1 + ctx.slope * x.abs()) ** 2, None


class LIFLayer(nn.Module):
    """Leaky integrate-and-fire neurons: mem <- beta * mem + W x; spike when mem crosses the threshold;
    soft reset (the threshold is subtracted after a spike). Input and output are (T, B, features)."""

    def __init__(self, d_in, d_out, beta=0.9, threshold=1.0, slope=10.0):
        super().__init__()
        self.fc, self.beta, self.threshold, self.slope = nn.Linear(d_in, d_out), beta, threshold, slope

    def forward(self, x_seq):
        mem = torch.zeros(x_seq.shape[1], self.fc.out_features, device=x_seq.device)
        out = []
        for x in x_seq:
            mem = self.beta * mem + self.fc(x)
            spk = _Spike.apply(mem - self.threshold, self.slope)
            mem = mem - spk.detach() * self.threshold
            out.append(spk)
        return torch.stack(out)


class SNNNet(nn.Module):
    def __init__(self, d_in, d_out, hidden, n_steps, beta, threshold, slope, encoding):
        super().__init__()
        dims = [d_in, *hidden]
        self.layers = nn.ModuleList([LIFLayer(a, b, beta, threshold, slope) for a, b in zip(dims[:-1], dims[1:])])
        self.readout = nn.Linear(dims[-1], d_out)
        self.n_steps, self.beta, self.encoding = n_steps, beta, encoding

    def encode(self, x):
        if self.encoding == "rate":   # Poisson-style spike trains with probability sigmoid(x)
            return (torch.rand(self.n_steps, *x.shape, device=x.device) < torch.sigmoid(x)).float()
        return x.unsqueeze(0).expand(self.n_steps, *x.shape)   # direct: the real value is injected as current

    def forward(self, x, return_rates=False):
        h, rates = self.encode(x), []
        for layer in self.layers:
            h = layer(h)
            rates.append(h.mean().item() if return_rates else None)
        mem, acc = torch.zeros(x.shape[0], self.readout.out_features, device=x.device), 0.0
        for cur in self.readout(h):      # non-spiking leaky readout, averaged over time
            mem = self.beta * mem + cur
            acc = acc + mem
        out = acc / self.n_steps
        return (out, rates) if return_rates else out


@register("snn", family="bio")
class SpikingNetwork(BaseModel):
    """Spiking neural network of leaky integrate-and-fire neurons, trained end to end with surrogate
    gradients. Each input is presented for `n_steps` time steps; the prediction is read from the
    time-averaged membrane potential of a non-spiking output layer.

    encoding='direct' injects the (standardised) input as a constant current; 'rate' turns it into random
    spike trains. firing_rates(X) reports the mean spike rate of every hidden layer."""
    family = "bio"

    def __init__(self, hidden=(64,), n_steps=16, beta=0.9, threshold=1.0, slope=10.0, encoding="direct", **kw):
        if encoding not in ("direct", "rate"):
            raise ValueError("encoding must be 'direct' or 'rate'")
        super().__init__(**kw)
        self.hidden, self.n_steps, self.beta = tuple(hidden), n_steps, beta
        self.threshold, self.slope, self.encoding = threshold, slope, encoding

    def build(self, d_in, d_out):
        return SNNNet(d_in, d_out, self.hidden, self.n_steps, self.beta, self.threshold, self.slope, self.encoding)

    def firing_rates(self, X):
        """Mean firing rate (spikes per neuron per step, 0..1) of each hidden layer."""
        self._check_fitted()
        self._module.eval()
        with torch.no_grad():
            return self._module(self._norm(self._as_x(X)), return_rates=True)[1]


# ------------------------------------------------------------ self-organizing map ---
class SOMNet(nn.Module):
    def __init__(self, d_in, rows, cols):
        super().__init__()
        self.rows, self.cols = rows, cols
        self.register_buffer("W", torch.randn(rows * cols, d_in) * 0.1)
        ij = torch.stack(torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij"), -1).reshape(-1, 2).float()
        self.register_buffer("coords", ij)

    @torch.no_grad()
    def init_from_data(self, X):
        idx = torch.randperm(len(X), device=X.device)[: len(self.W)]
        if len(idx) < len(self.W):
            idx = torch.randint(len(X), (len(self.W),), device=X.device)
        self.W.copy_(X[idx])

    def bmu(self, x):
        return torch.cdist(x, self.W).argmin(1)

    def forward(self, x):                     # grid position (row, col) of each sample's best-matching unit
        return self.coords[self.bmu(x)]

    def loss(self, x):                        # quantisation error
        return torch.cdist(x, self.W).min(1).values.mean()

    @torch.no_grad()
    def update(self, x, lr, sigma):
        d2 = ((self.coords[self.bmu(x)][:, None, :] - self.coords[None, :, :]) ** 2).sum(-1)   # grid distances
        h = torch.exp(-d2 / (2 * sigma ** 2))                                                  # neighbourhood weights
        den = h.sum(0, keepdim=True).T
        target = (h.T @ x) / den.clamp(min=1e-8)
        self.W += lr * (target - self.W) * (den > 1e-8)


@register("som", family="bio")
class SelfOrganizingMap(Unsupervised):
    """Self-Organizing Map (Kohonen): a grid of units whose weight vectors move toward the data so that
    nearby units respond to similar inputs, giving a topology-preserving 2-D map without labels.

        som = SelfOrganizingMap(grid=(10, 10)).fit(X)
        som.transform(X)               # (row, col) of each sample's best-matching unit
        som.u_matrix()                 # distances between neighbouring units (cluster borders are ridges)
        som.label_units(X, y); som.classify(X_new)

    The neighbourhood radius shrinks from `sigma` to `sigma_end` and the learning rate from `lr` to
    `lr_end` over the course of fit()."""
    family = "bio"

    def __init__(self, grid=(8, 8), sigma=None, sigma_end=0.5, lr_end=0.02, **kw):
        kw.setdefault("lr", 0.5)
        kw.setdefault("epochs", 50)
        super().__init__(latent_dim=2, **kw)
        self.grid, self.sigma_end, self.lr_end = tuple(grid), sigma_end, lr_end
        self.sigma = sigma if sigma is not None else max(grid) / 2
        self.unit_labels_ = None

    def build(self, d_in, d_out):
        return SOMNet(d_in, *self.grid)

    def _fit(self, Xt, yt, epochs, val):
        m, n = self._module, len(Xt)
        batches = max(1, -(-n // self.batch_size))
        total, step = epochs * batches, 0
        for _ in range(epochs):
            perm = torch.randperm(n, device=Xt.device)
            for i in range(0, n, self.batch_size):
                p = step / max(total - 1, 1)
                m.update(Xt[perm[i:i + self.batch_size]],
                         self.lr * (self.lr_end / self.lr) ** p,
                         self.sigma * (self.sigma_end / self.sigma) ** p)
                step += 1
            self.history_["loss"].append(float(m.loss(Xt)))

    # ---- analysis helpers --------------------------------------------------------
    def bmu_index(self, X):
        self._check_fitted()
        return self._module.bmu(self._norm(self._as_x(X))).cpu().numpy()

    def quantization_error(self, X):
        """Mean distance from each sample to its best-matching unit (standardised units)."""
        self._check_fitted()
        return float(self._module.loss(self._norm(self._as_x(X))))

    def topographic_error(self, X):
        """Fraction of samples whose two best units are not neighbours on the grid (lower = better ordering)."""
        self._check_fitted()
        m = self._module
        top2 = torch.cdist(self._norm(self._as_x(X)), m.W).topk(2, largest=False).indices
        gap = (m.coords[top2[:, 0]] - m.coords[top2[:, 1]]).pow(2).sum(1).sqrt()
        return float((gap > 1.5).float().mean())

    def unit_weights(self):
        """Weight vectors in the original feature scale, shape (rows, cols, features)."""
        self._check_fitted()
        W = self._module.W.cpu().numpy()
        W = W * self._sd + self._mu if self.normalize else W
        return W.reshape(*self.grid, -1)

    def u_matrix(self):
        """For each unit, the mean distance between its weights and those of its grid neighbours."""
        self._check_fitted()
        rows, cols = self.grid
        W = self._module.W.cpu().numpy().reshape(rows, cols, -1)
        out = np.zeros((rows, cols))
        for r in range(rows):
            for c in range(cols):
                nb = [W[r + dr, c + dc] for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                      if 0 <= r + dr < rows and 0 <= c + dc < cols]
                out[r, c] = np.mean([np.linalg.norm(W[r, c] - w) for w in nb])
        return out

    def hit_map(self, X):
        """How many samples each unit wins, shape (rows, cols)."""
        return np.bincount(self.bmu_index(X), minlength=self.grid[0] * self.grid[1]).reshape(self.grid)

    def label_units(self, X, y):
        """Give every unit the majority class of the samples it wins; empty units take the label of the
        nearest labelled unit in weight space. Enables classify()."""
        y, idx = np.asarray(y), self.bmu_index(X)
        classes = np.unique(y)
        labels = np.full(self.grid[0] * self.grid[1], -1)
        for u in np.unique(idx):
            vals, counts = np.unique(y[idx == u], return_counts=True)
            labels[u] = vals[counts.argmax()]
        W = self._module.W.cpu().numpy()
        known = np.flatnonzero(labels >= 0)
        for u in np.flatnonzero(labels < 0):
            labels[u] = labels[known[np.linalg.norm(W[known] - W[u], axis=1).argmin()]]
        self.unit_labels_, self.classes_ = labels, classes
        return self

    def classify(self, X):
        if self.unit_labels_ is None:
            raise RuntimeError("call label_units(X, y) first")
        return self.unit_labels_[self.bmu_index(X)]

    def evaluate(self, X, y=None):
        return {"loss": self.quantization_error(X), "quantization_error": self.quantization_error(X),
                "topographic_error": self.topographic_error(X)}
