"""Plain torch.nn modules used by the vector-family models."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTS = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh,
        "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}


def act(name):
    if name not in ACTS:
        raise ValueError(f"unknown activation {name!r}; choose from {sorted(ACTS)}")
    return ACTS[name]()


# --------------------------------------------------------------------- MLP ---
class MLPNet(nn.Sequential):
    def __init__(self, d_in, d_out, hidden=(128, 64), activation="relu",
                 dropout=0.0, batch_norm=False):
        layers, d = [], d_in
        for h in hidden:
            layers.append(nn.Linear(d, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, d_out))
        super().__init__(*layers)


# ------------------------------------------------------------------ ResNet ---
class ResBlock(nn.Module):
    def __init__(self, width, activation, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1, self.fc2 = nn.Linear(width, width), nn.Linear(width, width)
        self.act, self.drop = act(activation), nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.fc2(self.act(self.fc1(self.norm(x)))))


class ResidualNet(nn.Module):
    def __init__(self, d_in, d_out, width=128, depth=4, activation="relu", dropout=0.0):
        super().__init__()
        self.proj = nn.Linear(d_in, width)
        self.blocks = nn.Sequential(*[ResBlock(width, activation, dropout) for _ in range(depth)])
        self.norm, self.head = nn.LayerNorm(width), nn.Linear(width, d_out)

    def forward(self, x):
        return self.head(self.norm(self.blocks(self.proj(x))))


# ---------------------------------------------------------------- DenseNet ---
class DenseNet(nn.Module):
    """Each layer sees the concatenation of the input and all earlier layers."""

    def __init__(self, d_in, d_out, growth=32, depth=4, init_width=64, activation="relu"):
        super().__init__()
        self.proj = nn.Linear(d_in, init_width)
        self.norms = nn.ModuleList([nn.LayerNorm(init_width + i * growth) for i in range(depth)])
        self.layers = nn.ModuleList([nn.Linear(init_width + i * growth, growth) for i in range(depth)])
        final = init_width + depth * growth
        self.out_norm, self.head = nn.LayerNorm(final), nn.Linear(final, d_out)
        self.act = act(activation)

    def forward(self, x):
        feats = self.proj(x)
        for norm, layer in zip(self.norms, self.layers):
            feats = torch.cat([feats, self.act(layer(norm(feats)))], dim=1)
        return self.head(self.out_norm(feats))


# ----------------------------------------------------------------- Highway ---
class HighwayLayer(nn.Module):
    def __init__(self, width, activation, carry_bias=-1.0):
        super().__init__()
        self.h, self.t = nn.Linear(width, width), nn.Linear(width, width)
        nn.init.constant_(self.t.bias, carry_bias)  # start mostly carrying the input
        self.act = act(activation)

    def forward(self, x):
        t = torch.sigmoid(self.t(x))
        return t * self.act(self.h(x)) + (1 - t) * x


class HighwayNet(nn.Module):
    def __init__(self, d_in, d_out, width=128, depth=4, activation="relu"):
        super().__init__()
        self.proj = nn.Linear(d_in, width)
        self.layers = nn.Sequential(*[HighwayLayer(width, activation) for _ in range(depth)])
        self.head = nn.Linear(width, d_out)

    def forward(self, x):
        return self.head(self.layers(self.proj(x)))


# -------------------------------------------------- Squeeze-and-Excitation ---
class SEBlock(nn.Module):
    """Residual block whose output features are re-weighted by a learned gate."""

    def __init__(self, width, reduction, activation, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1, self.fc2 = nn.Linear(width, width), nn.Linear(width, width)
        r = max(1, width // reduction)
        self.squeeze, self.excite = nn.Linear(width, r), nn.Linear(r, width)
        self.act, self.drop = act(activation), nn.Dropout(dropout)

    def forward(self, x):
        r = self.fc2(self.act(self.fc1(self.norm(x))))
        s = torch.sigmoid(self.excite(self.act(self.squeeze(r))))  # per-feature gate
        return x + self.drop(r * s)


class SENet(nn.Module):
    def __init__(self, d_in, d_out, width=128, depth=4, reduction=4, activation="relu", dropout=0.0):
        super().__init__()
        self.proj = nn.Linear(d_in, width)
        self.blocks = nn.Sequential(*[SEBlock(width, reduction, activation, dropout) for _ in range(depth)])
        self.norm, self.head = nn.LayerNorm(width), nn.Linear(width, d_out)

    def forward(self, x):
        return self.head(self.norm(self.blocks(self.proj(x))))


# -------------------------------------------------------- Mixture of Experts ---
class MoENet(nn.Module):
    """Softmax-routed experts. top_k=None -> dense mixture; noisy=True -> Shazeer-style
    noisy top-k gating with a load-balancing (importance CV^2) auxiliary loss.
    All experts are evaluated and masked (simple and correct, not sparse-fast)."""

    def __init__(self, d_in, d_out, n_experts=4, expert_hidden=(64,), top_k=None,
                 noisy=False, aux_coef=0.0, activation="relu"):
        super().__init__()
        self.experts = nn.ModuleList(
            [MLPNet(d_in, d_out, expert_hidden, activation) for _ in range(n_experts)])
        self.gate = nn.Linear(d_in, n_experts)
        self.noise = nn.Linear(d_in, n_experts) if noisy else None
        self.top_k = min(top_k, n_experts) if top_k else None
        self.aux_coef, self.aux_loss = aux_coef, None

    def gates(self, x):
        logits = self.gate(x)
        if self.noise is not None and self.training:
            logits = logits + torch.randn_like(logits) * F.softplus(self.noise(x))
        if self.top_k is not None:
            top, idx = logits.topk(self.top_k, dim=1)
            logits = torch.full_like(logits, float("-inf")).scatter(1, idx, top)
        return F.softmax(logits, dim=1)

    def forward(self, x):
        g = self.gates(x)  # (B, E)
        if self.aux_coef > 0 and self.training:
            imp = g.sum(0)
            self.aux_loss = self.aux_coef * imp.var(unbiased=False) / (imp.mean() ** 2 + 1e-9)
        else:
            self.aux_loss = None
        outs = torch.stack([e(x) for e in self.experts], dim=1)  # (B, E, d_out)
        return (g.unsqueeze(-1) * outs).sum(1)


# -------------------------------------------------------------- Hypernetwork ---
class HyperNet(nn.Module):
    """A small network generates, for every sample, the weights of a target layer."""

    def __init__(self, d_in, d_out, target_hidden=16, hyper_hidden=64, width=64, activation="relu"):
        super().__init__()
        self.d_in, self.th = d_in, target_hidden
        self.hyper = nn.Sequential(
            nn.Linear(d_in, hyper_hidden), act(activation),
            nn.Linear(hyper_hidden, target_hidden * d_in + target_hidden))
        self.head = nn.Sequential(act(activation), nn.Linear(target_hidden, width),
                                  act(activation), nn.Linear(width, d_out))

    def forward(self, x):
        p = self.hyper(x)
        W = p[:, :self.th * self.d_in].view(-1, self.th, self.d_in) / math.sqrt(self.d_in)
        b = p[:, self.th * self.d_in:]
        h = torch.bmm(W, x.unsqueeze(-1)).squeeze(-1) + b  # per-sample generated layer
        return self.head(h)


# ------------------------------------------- Implicit net / Deep Equilibrium ---
def bounded(W, kappa):
    """Rescale W so its infinity-norm is <= kappa (<1) -> fixed point is unique and
    plain iteration converges for any 1-Lipschitz, elementwise activation."""
    row = W.abs().sum(1).max()
    return W * (kappa / torch.clamp(row, min=kappa))


class ImplicitNet(nn.Module):
    """Implicit deep learning (El Ghaoui et al.): z = relu(A z + B x), y = C z + D x.
    The equilibrium is found by unrolled iteration and differentiated through."""

    def __init__(self, d_in, d_out, state_dim=64, iterations=30, kappa=0.9):
        super().__init__()
        self.W = nn.Parameter(torch.randn(state_dim, state_dim) / math.sqrt(state_dim))
        self.B, self.C = nn.Linear(d_in, state_dim), nn.Linear(state_dim, d_out)
        self.D = nn.Linear(d_in, d_out, bias=False)
        self.iterations, self.kappa = iterations, kappa

    def forward(self, x):
        A, bx = bounded(self.W, self.kappa), self.B(x)
        z = torch.zeros_like(bx)
        for _ in range(self.iterations):
            z = F.relu(z @ A.T + bx)
        return self.C(z) + self.D(x)


class DEQNet(nn.Module):
    """Deep Equilibrium Model (Bai et al.): z* = tanh(A z* + U x). Forward solves for
    z* without storing the graph; backward solves the adjoint fixed point with
    implicit differentiation, so memory does not grow with solver iterations."""

    def __init__(self, d_in, d_out, state_dim=64, max_iter=50, tol=1e-4, kappa=0.9):
        super().__init__()
        self.W = nn.Parameter(torch.randn(state_dim, state_dim) / math.sqrt(state_dim))
        self.inject, self.head = nn.Linear(d_in, state_dim), nn.Linear(state_dim, d_out)
        self.max_iter, self.tol, self.kappa = max_iter, tol, kappa

    def forward(self, x):
        A, xi = bounded(self.W, self.kappa), self.inject(x)
        f = lambda z: torch.tanh(z @ A.T + xi)
        with torch.no_grad():
            z = torch.zeros_like(xi)
            for _ in range(self.max_iter):
                z_new = f(z)
                done = (z_new - z).abs().max() < self.tol
                z = z_new
                if done:
                    break
        z = f(z)  # one differentiable step at the equilibrium
        if self.training and z.requires_grad:
            z0 = z.detach().requires_grad_()
            f0 = f(z0)

            def hook(grad):  # solve g = J^T g + grad  (implicit-function theorem)
                g = grad
                for _ in range(self.max_iter):
                    g_new = torch.autograd.grad(f0, z0, g, retain_graph=True)[0] + grad
                    done = (g_new - g).abs().max() < self.tol
                    g = g_new
                    if done:
                        break
                return g

            z.register_hook(hook)
        return self.head(z)


# --------------------------------------------------------------------- KAN ---
class KANLinear(nn.Module):
    """Every edge has a learnable 1-D function: SiLU base term + B-spline term."""

    def __init__(self, d_in, d_out, grid_size=5, spline_order=3, grid_range=(-3.0, 3.0)):
        super().__init__()
        self.k, self.g = spline_order, grid_size
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]
        self.register_buffer("grid", grid.expand(d_in, -1).contiguous())
        self.base_weight = nn.Parameter(torch.empty(d_out, d_in))
        self.spline_weight = nn.Parameter(torch.empty(d_out, d_in, grid_size + spline_order))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, std=0.1 / math.sqrt(d_in))

    def b_splines(self, x):  # (B, in) -> (B, in, grid+order), Cox-de Boor recursion
        grid, x = self.grid, x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.k + 1):
            bases = ((x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * bases[:, :, :-1]
                     + (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k]) * bases[:, :, 1:])
        return bases

    def forward(self, x):
        base = F.linear(F.silu(x), self.base_weight)
        spline = F.linear(self.b_splines(x).flatten(1), self.spline_weight.flatten(1))
        return base + spline


class KANNet(nn.Sequential):
    def __init__(self, d_in, d_out, hidden=(16,), grid_size=5, spline_order=3, grid_range=(-3.0, 3.0)):
        dims = [d_in, *hidden, d_out]
        super().__init__(*[KANLinear(a, b, grid_size, spline_order, grid_range)
                           for a, b in zip(dims[:-1], dims[1:])])


# --------------------------------------------------------------------- RBF ---
class RBFNet(nn.Module):
    def __init__(self, d_in, d_out, n_centers=64):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(n_centers, d_in))
        self.log_beta = nn.Parameter(torch.zeros(1))
        self.out = nn.Linear(n_centers, d_out)

    @torch.no_grad()
    def init_from_data(self, X):  # centers = random training points; width = median spacing
        n = min(len(self.centers), len(X))
        self.centers[:n] = X[torch.randperm(len(X), device=X.device)[:n]]
        d2 = torch.cdist(self.centers, self.centers) ** 2
        med = d2[d2 > 0].median() if (d2 > 0).any() else torch.tensor(1.0)
        self.log_beta.fill_(-math.log(float(med)))

    def forward(self, x):
        d2 = torch.cdist(x, self.centers) ** 2
        return self.out(torch.exp(-self.log_beta.exp() * d2))


# --------------------------------------------------- Extreme Learning Machine ---
class ELMNet(nn.Module):
    """Frozen random hidden layer + linear readout. The readout is solved in closed form."""

    def __init__(self, d_in, d_out, n_hidden=512, activation="tanh"):
        super().__init__()
        self.hidden, self.act = nn.Linear(d_in, n_hidden), act(activation)
        self.out = nn.Linear(n_hidden, d_out)
        for p in self.hidden.parameters():
            p.requires_grad_(False)
        for p in self.out.parameters():
            p.requires_grad_(False)

    def features(self, x):
        return self.act(self.hidden(x))

    def forward(self, x):
        return self.out(self.features(x))


# --------------------------------------------------- Random Neural Network ---
class RNNLayer(nn.Module):
    """Gelenbe random-neural-network layer. Neuron i has steady-state firing
    probability q_i = min(1, lambda+_i / (r_i + lambda-_i)), where excitatory and
    inhibitory arrival rates are positive weighted sums of the previous layer's q."""

    def __init__(self, d_in, d_out):
        super().__init__()
        mean = -math.log(d_in)  # softplus(mean) ~ 1/d_in keeps rates ~ O(1) at init
        self.wp = nn.Parameter(torch.randn(d_out, d_in) * 0.5 + mean)
        self.wm = nn.Parameter(torch.randn(d_out, d_in) * 0.5 + mean)
        self.r = nn.Parameter(torch.zeros(d_out))
        self.ext = nn.Parameter(torch.zeros(2, d_out))  # constant external arrivals

    def forward(self, q):
        lam_p = q @ F.softplus(self.wp).T + F.softplus(self.ext[0])
        lam_m = q @ F.softplus(self.wm).T + F.softplus(self.ext[1])
        return torch.clamp(lam_p / (F.softplus(self.r) + 1e-3 + lam_m), max=1.0)


class RandomNeuralNet(nn.Module):
    def __init__(self, d_in, d_out, hidden=(64, 64)):
        super().__init__()
        self.rates = nn.Linear(d_in, 2 * hidden[0])  # input -> external (+, -) arrival rates
        self.r0 = nn.Parameter(torch.zeros(hidden[0]))
        dims = list(hidden)
        self.layers = nn.ModuleList([RNNLayer(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.readout = nn.Linear(dims[-1], d_out)

    def forward(self, x):
        lam_p, lam_m = F.softplus(self.rates(x)).chunk(2, dim=1)
        q = torch.clamp(lam_p / (F.softplus(self.r0) + 1e-3 + lam_m), max=1.0)
        for layer in self.layers:
            q = layer(q)
        return self.readout(q)
