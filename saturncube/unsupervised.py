"""Unsupervised / self-supervised family (B4) and Hebbian-style learning rules (B3).

fit(X) takes no labels; transform(X) returns the learned representation.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import layers as L
from .base import BaseModel, register
from .optim import make_optimizer


# ----------------------------------------------------------------- base class ---
class Unsupervised(BaseModel):
    family = "unsupervised"

    def __init__(self, latent_dim=8, hidden=(64,), activation="relu", **kw):
        super().__init__(**kw)
        self.latent_dim, self.hidden, self.activation = latent_dim, tuple(hidden), activation

    def _init(self, X, y=None):
        self.task = "unsupervised"
        d_in = X.shape[-1]
        if self.input_dim not in (None, d_in):
            raise ValueError(f"input_dim={self.input_dim} but X has {d_in} features")
        self.input_dim, self.output_dim = d_in, self.latent_dim
        if self.normalize:
            axes = tuple(range(X.ndim - 1))
            self._mu, self._sd = X.mean(axes), X.std(axes)
            self._sd[self._sd < 1e-8] = 1.0
        self._module = self.build(d_in, self.latent_dim).to(self.device)
        if hasattr(self._module, "init_from_data"):
            self._module.init_from_data(self._norm(X))

    def _encode(self, y):
        return torch.zeros(len(y), 1, device=self.device)

    def fit(self, X, y=None, epochs=None, val_data=None):
        return super().fit(X, np.zeros(len(X)) if y is None else y, epochs=epochs)

    def _train_batch(self, m, opt, xb, yb):
        def closure():
            opt.zero_grad()
            loss = m.loss(xb)
            loss.backward()
            return loss
        if isinstance(opt, torch.optim.LBFGS):
            loss = opt.step(closure)
        else:
            loss = closure()
            opt.step()
        if hasattr(m, "after_step"):
            m.after_step()
        return loss.item()

    def transform(self, X):
        self._check_fitted()
        return self._outputs(self._norm(self._as_x(X))).cpu().numpy()

    predict = transform

    def _denorm(self, Z):
        Z = Z.cpu().numpy()
        return Z * self._sd + self._mu if self.normalize else Z

    def evaluate(self, X, y=None):
        self._check_fitted()
        Xt = self._norm(self._as_x(X))
        self._module.eval()
        with torch.no_grad():
            return {"loss": float(self._module.loss(Xt))}


# ------------------------------------------------------------ autoencoders ---
class AENet(nn.Module):
    def __init__(self, d_in, latent, hidden=(64,), noise_std=0.0, activation="relu"):
        super().__init__()
        self.enc = L.MLPNet(d_in, latent, hidden, activation)
        self.dec = L.MLPNet(latent, d_in, tuple(reversed(hidden)), activation)
        self.noise_std = noise_std

    def forward(self, x):
        return self.enc(x)

    def reconstruct(self, x):
        return self.dec(self.enc(x))

    def loss(self, x):
        xin = x + self.noise_std * torch.randn_like(x) if (self.training and self.noise_std > 0) else x
        return F.mse_loss(self.dec(self.enc(xin)), x)


class VAENet(nn.Module):
    def __init__(self, d_in, latent, hidden=(64,), beta=1.0, activation="relu"):
        super().__init__()
        self.enc = L.MLPNet(d_in, 2 * latent, hidden, activation)
        self.dec = L.MLPNet(latent, d_in, tuple(reversed(hidden)), activation)
        self.latent, self.beta = latent, beta

    def forward(self, x):  # the posterior mean is the embedding
        return self.enc(x)[:, : self.latent]

    def reconstruct(self, x):
        return self.dec(self.forward(x))

    def loss(self, x):
        mu, logvar = self.enc(x).chunk(2, dim=1)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        recon = 0.5 * ((self.dec(z) - x) ** 2).sum(1).mean()
        kl = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)).mean()
        return recon + self.beta * kl


class MaskedNet(nn.Module):
    """Masked prediction: hide random features, predict them from the rest."""

    def __init__(self, d_in, latent, hidden=(64,), mask_ratio=0.3, activation="relu"):
        super().__init__()
        self.enc = L.MLPNet(2 * d_in, latent, hidden, activation)
        self.dec = L.MLPNet(latent, d_in, tuple(reversed(hidden)), activation)
        self.mask_ratio = mask_ratio

    def forward(self, x):
        return self.enc(torch.cat([x, torch.zeros_like(x)], dim=1))

    def reconstruct(self, x):
        return self.dec(self.forward(x))

    def loss(self, x):
        mask = (torch.rand_like(x) < self.mask_ratio).float()
        pred = self.dec(self.enc(torch.cat([x * (1 - mask), mask], dim=1)))
        return (((pred - x) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)


# ------------------------------------------------- contrastive / bootstrap ---
def nt_xent(z1, z2, temperature):
    """InfoNCE / NT-Xent: each view's positive is the other view of the same sample."""
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    n = len(z1)
    sim = z @ z.T / temperature
    sim = sim.masked_fill(torch.eye(2 * n, dtype=torch.bool, device=z.device), float("-inf"))
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


def augment(x, noise_std, drop_p):
    x = x + noise_std * torch.randn_like(x)
    if drop_p > 0:
        x = x * (torch.rand_like(x) > drop_p).float()
    return x


class ContrastiveNet(nn.Module):
    def __init__(self, d_in, latent, hidden=(64,), proj_dim=16, temperature=0.5,
                 noise_std=0.3, drop_p=0.2, activation="relu"):
        super().__init__()
        self.enc = L.MLPNet(d_in, latent, hidden, activation)
        self.proj = L.MLPNet(latent, proj_dim, (latent,), activation)
        self.t, self.noise_std, self.drop_p = temperature, noise_std, drop_p

    def forward(self, x):
        return self.enc(x)

    def loss(self, x):
        v1, v2 = augment(x, self.noise_std, self.drop_p), augment(x, self.noise_std, self.drop_p)
        return nt_xent(self.proj(self.enc(v1)), self.proj(self.enc(v2)), self.t)


class BYOLNet(nn.Module):
    """Bootstrap Your Own Latent: an online network predicts a slowly moving (EMA) target
    network's projection of another view. No negatives."""

    def __init__(self, d_in, latent, hidden=(64,), proj_dim=16, tau=0.99,
                 noise_std=0.3, drop_p=0.2, activation="relu"):
        super().__init__()
        self.enc = L.MLPNet(d_in, latent, hidden, activation)
        self.proj = L.MLPNet(latent, proj_dim, (latent,), activation)
        self.pred = L.MLPNet(proj_dim, proj_dim, (proj_dim,), activation)
        self.t_enc, self.t_proj = copy.deepcopy(self.enc), copy.deepcopy(self.proj)
        for p in [*self.t_enc.parameters(), *self.t_proj.parameters()]:
            p.requires_grad_(False)
        self.tau, self.noise_std, self.drop_p = tau, noise_std, drop_p

    def forward(self, x):
        return self.enc(x)

    def _branch(self, a, b):
        p = F.normalize(self.pred(self.proj(self.enc(a))), dim=1)
        with torch.no_grad():
            t = F.normalize(self.t_proj(self.t_enc(b)), dim=1)
        return (2 - 2 * (p * t).sum(1)).mean()

    def loss(self, x):
        v1, v2 = augment(x, self.noise_std, self.drop_p), augment(x, self.noise_std, self.drop_p)
        return self._branch(v1, v2) + self._branch(v2, v1)

    @torch.no_grad()
    def after_step(self):
        for o, t in [(self.enc, self.t_enc), (self.proj, self.t_proj)]:
            for po, pt in zip(o.parameters(), t.parameters()):
                pt.mul_(self.tau).add_(po.detach(), alpha=1 - self.tau)


# ------------------------------------------------------------ deep clustering ---
def kmeans(z, k, iters=25):
    idx = [torch.randint(len(z), (1,)).item()]
    for _ in range(k - 1):  # k-means++ seeding
        d2 = torch.cdist(z, z[idx]).min(1).values ** 2
        idx.append(torch.multinomial(d2 + 1e-12, 1).item())
    c = z[idx].clone()
    for _ in range(iters):
        a = torch.cdist(z, c).argmin(1)
        for j in range(k):
            if (a == j).any():
                c[j] = z[a == j].mean(0)
    return c


class DECNet(AENet):
    """Deep Embedded Clustering: autoencoder pretraining, k-means on the latent space,
    then refinement of encoder and centers with a sharpened-target KL objective."""

    def __init__(self, d_in, latent, hidden, n_clusters, activation="relu"):
        super().__init__(d_in, latent, hidden, 0.0, activation)
        self.centers = nn.Parameter(torch.zeros(n_clusters, latent))
        self.register_buffer("ready", torch.tensor(0))

    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.cdist(z, self.centers) ** 2)
        return q / q.sum(1, keepdim=True)


@register("deep_clustering", family="unsupervised")
class DeepClustering(Unsupervised):
    def __init__(self, n_clusters=3, latent_dim=4, hidden=(32,), pretrain_epochs=40, **kw):
        super().__init__(latent_dim=latent_dim, hidden=hidden, **kw)
        self.n_clusters, self.pretrain_epochs = n_clusters, pretrain_epochs

    def build(self, d_in, d_out):
        return DECNet(d_in, d_out, self.hidden, self.n_clusters, self.activation)

    def _fit(self, Xt, yt, epochs, val):
        m = self._module
        if int(m.ready) == 0:
            super()._fit(Xt, yt, self.pretrain_epochs, None)  # stage 1: reconstruction
            with torch.no_grad():
                m.centers.copy_(kmeans(m.enc(Xt), self.n_clusters))
            m.ready.fill_(1)
        params = [m.centers, *m.enc.parameters()]
        opt = make_optimizer(self.optimizer, params, (self.lr or 3e-3), 0.0, **self.optimizer_kwargs)
        m.train()
        for _ in range(epochs):  # stage 2: full-batch DEC refinement
            q = m.soft_assign(m.enc(Xt))
            p = (q ** 2 / q.sum(0)).detach()
            p = p / p.sum(1, keepdim=True)
            loss = F.kl_div(q.clamp(min=1e-8).log(), p, reduction="batchmean")
            opt.zero_grad()
            loss.backward()
            opt.step()
            self.history_["loss"].append(loss.item())

    def predict(self, X):
        self._check_fitted()
        with torch.no_grad():
            self._module.eval()
            z = self._module.enc(self._norm(self._as_x(X)))
            return self._module.soft_assign(z).argmax(1).cpu().numpy()

    def evaluate(self, X, y=None):
        self._check_fitted()
        Xt = self._norm(self._as_x(X))
        with torch.no_grad():
            self._module.eval()
            q = self._module.soft_assign(self._module.enc(Xt))
            p = (q ** 2 / q.sum(0))
            p = p / p.sum(1, keepdim=True)
            return {"loss": float(F.kl_div(q.clamp(min=1e-8).log(), p, reduction="batchmean"))}


# ------------------------------------------------------------ model wrappers ---
@register("autoencoder", family="unsupervised")
class Autoencoder(Unsupervised):
    def __init__(self, noise_std=0.0, **kw):
        super().__init__(**kw)
        self.noise_std = noise_std

    def build(self, d_in, d_out):
        return AENet(d_in, d_out, self.hidden, self.noise_std, self.activation)

    def reconstruct(self, X):
        self._check_fitted()
        self._module.eval()
        with torch.no_grad():
            return self._denorm(self._module.reconstruct(self._norm(self._as_x(X))))


@register("denoising_ae", family="unsupervised")
class DenoisingAutoencoder(Autoencoder):
    def __init__(self, noise_std=0.3, **kw):
        super().__init__(noise_std=noise_std, **kw)


@register("vae", family="unsupervised")
class VAE(Autoencoder):
    def __init__(self, beta=1.0, **kw):
        super().__init__(**kw)
        self.beta = beta

    def build(self, d_in, d_out):
        return VAENet(d_in, d_out, self.hidden, self.beta, self.activation)

    def sample(self, n=16):
        self._check_fitted()
        self._module.eval()
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim, device=self.device)
            return self._denorm(self._module.dec(z))


@register("masked", family="unsupervised")
class MaskedPrediction(Autoencoder):
    def __init__(self, mask_ratio=0.3, **kw):
        super().__init__(**kw)
        self.mask_ratio = mask_ratio

    def build(self, d_in, d_out):
        return MaskedNet(d_in, d_out, self.hidden, self.mask_ratio, self.activation)


@register("contrastive", family="unsupervised")
class Contrastive(Unsupervised):
    """SimCLR-style contrastive learning with the InfoNCE / NT-Xent objective."""
    def __init__(self, proj_dim=16, temperature=0.5, noise_std=0.3, drop_p=0.2, **kw):
        super().__init__(**kw)
        self.proj_dim, self.temperature = proj_dim, temperature
        self.noise_std, self.drop_p = noise_std, drop_p

    def build(self, d_in, d_out):
        return ContrastiveNet(d_in, d_out, self.hidden, self.proj_dim, self.temperature,
                              self.noise_std, self.drop_p, self.activation)


@register("byol", family="unsupervised")
class BYOL(Unsupervised):
    """BYOL-style bootstrap latent learning (no negative pairs)."""
    def __init__(self, proj_dim=16, tau=0.99, noise_std=0.3, drop_p=0.2, **kw):
        super().__init__(**kw)
        self.proj_dim, self.tau, self.noise_std, self.drop_p = proj_dim, tau, noise_std, drop_p

    def build(self, d_in, d_out):
        return BYOLNet(d_in, d_out, self.hidden, self.proj_dim, self.tau,
                       self.noise_std, self.drop_p, self.activation)


# ------------------------------------------------------- Hebbian-style rules ---
class HebbNet(nn.Module):
    """Weights are changed by local rules, not by gradients. Rules: hebbian (normalized),
    oja (subspace rule), sanger (generalized Hebbian, ordered components), bcm
    (sliding threshold), competitive (winner-take-all)."""

    RULES = ("hebbian", "oja", "sanger", "bcm", "competitive")

    def __init__(self, d_in, k, rule):
        super().__init__()
        if rule not in self.RULES:
            raise ValueError(f"rule must be one of {self.RULES}")
        self.rule = rule
        W = torch.randn(k, d_in)
        self.register_buffer("W", W / W.norm(dim=1, keepdim=True))
        self.register_buffer("theta", torch.ones(k))

    @torch.no_grad()
    def init_from_data(self, X):
        if self.rule == "competitive":
            self.W.copy_(X[torch.randperm(len(X), device=X.device)[: len(self.W)]])

    def _winner(self, x):
        return torch.cdist(x, self.W).argmin(1)

    def forward(self, x):
        if self.rule == "competitive":
            return F.one_hot(self._winner(x), len(self.W)).float()
        return x @ self.W.T

    def recon(self, x):
        return self.W[self._winner(x)] if self.rule == "competitive" else (x @ self.W.T) @ self.W

    def loss(self, x):
        return F.mse_loss(self.recon(x), x)

    @torch.no_grad()
    def update(self, x, lr):
        W, n = self.W, len(x)
        if self.rule == "competitive":
            win = self._winner(x)
            delta, cnt = torch.zeros_like(W), torch.zeros(len(W), device=x.device)
            delta.index_add_(0, win, x - W[win])
            cnt.index_add_(0, win, torch.ones(n, device=x.device))
            W += lr * delta / cnt.clamp(min=1)[:, None]
            return
        y = x @ W.T
        yx, yy = (y.T @ x) / n, (y.T @ y) / n
        if self.rule == "hebbian":
            W += lr * yx
            W /= W.norm(dim=1, keepdim=True) + 1e-8
        elif self.rule == "oja":
            W += lr * (yx - yy @ W)
        elif self.rule == "sanger":
            W += lr * (yx - torch.tril(yy) @ W)
        else:  # bcm
            self.theta.mul_(0.9).add_(0.1 * (y ** 2).mean(0))
            W += lr * ((y * (y - self.theta)).T @ x) / n
            W *= (5.0 / (W.norm(dim=1, keepdim=True) + 1e-8)).clamp(max=1.0)
        W.nan_to_num_(0.0)


@register("hebbian", family="unsupervised")
class HebbianLearner(Unsupervised):
    """Unsupervised feature learner trained with a local learning rule."""
    def __init__(self, rule="oja", n_components=4, **kw):
        kw.setdefault("lr", 0.02)
        super().__init__(latent_dim=n_components, **kw)
        self.rule = rule

    def build(self, d_in, d_out):
        return HebbNet(d_in, d_out, self.rule)

    def _fit(self, Xt, yt, epochs, val):
        m, n = self._module, len(Xt)
        for _ in range(epochs):
            perm = torch.randperm(n, device=Xt.device)
            for i in range(0, n, self.batch_size):
                m.update(Xt[perm[i:i + self.batch_size]], self.lr)
            self.history_["loss"].append(float(m.loss(Xt)))


# ================================================ predictive / energy-based learning ===
class TemporalNet(nn.Module):
    """Contrastive predictive coding for sequences: the embedding of the past must identify
    (among the other sequences in the batch) the embedding of that sequence's future."""

    def __init__(self, d_in, latent, hidden=32, horizon=4, temperature=0.2):
        super().__init__()
        self.past, self.future = nn.GRU(d_in, hidden, batch_first=True), nn.GRU(d_in, hidden, batch_first=True)
        self.past_out, self.future_out = nn.Linear(hidden, latent), nn.Linear(hidden, latent)
        self.pred, self.horizon, self.t = nn.Linear(latent, latent), horizon, temperature

    def encode(self, x):
        return self.past_out(self.past(x)[1][-1])

    forward = encode

    def loss(self, x):
        k = self.horizon
        zc = F.normalize(self.pred(self.encode(x[:, :-k])), dim=-1)
        zf = F.normalize(self.future_out(self.future(x[:, -k:])[1][-1]), dim=-1)
        logits = zc @ zf.T / self.t
        return F.cross_entropy(logits, torch.arange(len(x), device=x.device))


@register("temporal_contrastive", family="unsupervised")
class TemporalContrastive(Unsupervised):
    """Temporal contrastive / predictive representation learning on sequences (n, T, d)."""
    def __init__(self, horizon=4, temperature=0.2, **kw):
        super().__init__(**kw)
        self.horizon, self.temperature = horizon, temperature

    def build(self, d_in, d_out):
        return TemporalNet(d_in, d_out, self.hidden[0], self.horizon, self.temperature)


class JEPANet(nn.Module):
    """Latent prediction (JEPA-style): predict the *embedding* of the full input, produced by
    a slowly moving target encoder, from an embedding of a masked view; nothing is reconstructed."""

    def __init__(self, d_in, latent, hidden=(64,), mask_ratio=0.4, tau=0.99, activation="relu"):
        super().__init__()
        self.enc = L.MLPNet(2 * d_in, latent, hidden, activation)
        self.pred = L.MLPNet(latent, latent, (latent,), activation)
        self.target = copy.deepcopy(self.enc)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.mask_ratio, self.tau = mask_ratio, tau

    def forward(self, x):
        return self.enc(torch.cat([x, torch.zeros_like(x)], dim=1))

    def loss(self, x):
        mask = (torch.rand_like(x) < self.mask_ratio).float()
        p = F.normalize(self.pred(self.enc(torch.cat([x * (1 - mask), mask], dim=1))), dim=1)
        with torch.no_grad():
            t = F.normalize(self.target(torch.cat([x, torch.zeros_like(x)], dim=1)), dim=1)
        return (2 - 2 * (p * t).sum(1)).mean()

    @torch.no_grad()
    def after_step(self):
        for po, pt in zip(self.enc.parameters(), self.target.parameters()):
            pt.mul_(self.tau).add_(po.detach(), alpha=1 - self.tau)


@register("latent_prediction", family="unsupervised")
class LatentPrediction(Unsupervised):
    def __init__(self, mask_ratio=0.4, tau=0.99, **kw):
        super().__init__(**kw)
        self.mask_ratio, self.tau = mask_ratio, tau

    def build(self, d_in, d_out):
        return JEPANet(d_in, d_out, self.hidden, self.mask_ratio, self.tau, self.activation)


class EBMNet(nn.Module):
    """Scalar energy E(x). Trained by contrastive divergence: data energy is pushed down and
    the energy of Langevin-sampled negatives (persistent buffer) is pushed up."""

    def __init__(self, d_in, hidden=(64, 64), steps=20, step_size=0.1, noise=0.02, buffer_size=512, reg=0.1):
        super().__init__()
        self.f = L.MLPNet(d_in, 1, hidden, "silu")
        self.register_buffer("buffer", torch.randn(buffer_size, d_in))
        self.steps, self.step_size, self.noise, self.reg = steps, step_size, noise, reg

    def forward(self, x):
        return self.f(x).squeeze(-1)

    def langevin(self, x, steps=None):
        x = x.detach().clone()
        for _ in range(steps or self.steps):
            x.requires_grad_(True)
            with torch.enable_grad():
                g, = torch.autograd.grad(self(x).sum(), x)
            x = (x - self.step_size * g + self.noise * torch.randn_like(x)).clamp(-6, 6).detach()
        return x

    def loss(self, x):
        idx = torch.randint(len(self.buffer), (len(x),), device=x.device)
        start = self.buffer[idx]
        fresh = torch.rand(len(x), device=x.device) < 0.05
        start = torch.where(fresh[:, None], torch.randn_like(start), start)
        was_training = self.training
        self.eval()
        neg = self.langevin(start)
        self.train(was_training)
        self.buffer[idx] = neg
        e_pos, e_neg = self(x), self(neg)
        return e_pos.mean() - e_neg.mean() + self.reg * (e_pos ** 2 + e_neg ** 2).mean()


@register("energy_based", family="unsupervised")
class EnergyBasedModel(Unsupervised):
    """Energy-based model of the data distribution. transform(X) returns the energy of each
    sample (low = likely); sample(n) draws samples by Langevin dynamics."""
    def __init__(self, hidden=(64, 64), langevin_steps=20, step_size=0.1, noise=0.02, **kw):
        super().__init__(hidden=hidden, **kw)
        self.langevin_steps, self.step_size, self.noise = langevin_steps, step_size, noise

    def build(self, d_in, d_out):
        return EBMNet(d_in, self.hidden, self.langevin_steps, self.step_size, self.noise)

    def energy(self, X):
        return self.transform(X)

    def sample(self, n=64, steps=100):
        self._check_fitted()
        self._module.eval()
        x0 = torch.randn(n, self.input_dim, device=self.device)
        return self._denorm(self._module.langevin(x0, steps))
