"""Memory and associative networks (A2 Hopfield, A4 memory models)."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import layers as L
from .base import BaseModel, register
from .sequence import SequenceModel


# ------------------------------------------------------- classical associative memory ---
@register("hopfield", family="memory")
class HopfieldNetwork:
    """Classical Hopfield network: binary (+1/-1) patterns stored with the Hebbian rule and
    recalled by iterating sign(W x). Reliable capacity is about 0.14 * n_units."""

    def __init__(self, seed=None):
        self.W, self.rng = None, np.random.default_rng(seed)

    def fit(self, patterns):
        P = np.sign(np.asarray(patterns, dtype=float))
        P[P == 0] = 1
        self.W = P.T @ P / P.shape[1]
        np.fill_diagonal(self.W, 0.0)
        return self

    def energy(self, x):
        x = np.asarray(x, dtype=float)
        return -0.5 * np.einsum("...i,ij,...j->...", x, self.W, x)

    def recall(self, x, steps=30, asynchronous=False):
        x = np.sign(np.asarray(x, dtype=float))
        x[x == 0] = 1
        single = x.ndim == 1
        x = np.atleast_2d(x).copy()
        for _ in range(steps):
            prev = x.copy()
            if asynchronous:
                for i in self.rng.permutation(x.shape[1]):
                    x[:, i] = np.where(x @ self.W[i] >= 0, 1.0, -1.0)
            else:
                x = np.where(x @ self.W >= 0, 1.0, -1.0)
            if (x == prev).all():
                break
        return x[0] if single else x

    predict = recall


@register("modern_hopfield", family="memory")
class ModernHopfield:
    """Modern (dense-associative) Hopfield network: retrieval is softmax attention over the
    stored patterns, xi <- X^T softmax(beta X xi). Capacity grows exponentially in dimension."""

    def __init__(self, beta=16.0):
        self.beta, self.X = beta, None

    def fit(self, patterns):
        self.X = np.asarray(patterns, dtype=float)
        self._U = self.X / (np.linalg.norm(self.X, axis=1, keepdims=True) + 1e-12)
        return self

    def recall(self, query, steps=1):
        q = np.asarray(query, dtype=float)
        single = q.ndim == 1
        q = np.atleast_2d(q)
        for _ in range(steps):
            u = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
            a = self.beta * u @ self._U.T
            a = np.exp(a - a.max(1, keepdims=True))
            q = (a / a.sum(1, keepdims=True)) @ self.X
        return q[0] if single else q

    predict = recall


@register("associative_memory", family="memory")
class AssociativeMemory:
    """Hetero-associative memory (key -> value) with Hebbian outer-product storage, or
    pseudo-inverse storage (rule='pinv') which is exact for linearly independent keys."""

    def __init__(self, rule="hebb"):
        if rule not in ("hebb", "pinv"):
            raise ValueError("rule must be 'hebb' or 'pinv'")
        self.rule, self.W = rule, None

    def fit(self, keys, values):
        K, V = np.asarray(keys, dtype=float), np.asarray(values, dtype=float)
        self.W = V.T @ K if self.rule == "hebb" else V.T @ np.linalg.pinv(K.T)
        return self

    def recall(self, keys):
        return np.asarray(keys, dtype=float) @ self.W.T

    predict = recall


@register("content_addressable_memory", family="memory")
class ContentAddressableMemory:
    """Key-value store read by cosine similarity: top-k entries, softmax-weighted."""

    def __init__(self, temperature=0.05):
        self.t, self.keys, self.values = temperature, None, None

    def write(self, keys, values):
        keys, values = np.atleast_2d(keys), np.atleast_2d(values)
        self.keys = keys if self.keys is None else np.vstack([self.keys, keys])
        self.values = values if self.values is None else np.vstack([self.values, values])
        return self

    def read(self, query, k=1):
        q = np.atleast_2d(query)
        K = self.keys / (np.linalg.norm(self.keys, axis=1, keepdims=True) + 1e-12)
        sims = (q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)) @ K.T
        out = []
        for s in sims:
            idx = np.argsort(-s)[:k]
            w = np.exp((s[idx] - s[idx].max()) / self.t)
            out.append((w / w.sum()) @ self.values[idx])
        return np.array(out)


# ------------------------------------------------------------- product-key memory ---
class ProductKeyMemory(nn.Module):
    """Product-key memory (Lample et al., 2019): n_keys**2 memory slots are addressed with
    two sets of n_keys sub-keys, so a query is matched in O(n_keys) instead of O(n_keys**2)."""

    def __init__(self, d, n_keys=16, topk=4, d_value=None):
        super().__init__()
        if d % 2:
            raise ValueError("d must be even")
        self.half, self.n, self.k = d // 2, n_keys, topk
        self.query = nn.Linear(d, d)
        self.K1, self.K2 = nn.Parameter(torch.randn(n_keys, d // 2) / 4), nn.Parameter(torch.randn(n_keys, d // 2) / 4)
        self.values = nn.Embedding(n_keys ** 2, d_value or d)
        nn.init.normal_(self.values.weight, std=0.1)

    def forward(self, x):
        q = self.query(x)
        t1, i1 = (q[:, :self.half] @ self.K1.T).topk(self.k, dim=1)
        t2, i2 = (q[:, self.half:] @ self.K2.T).topk(self.k, dim=1)
        scores = (t1[:, :, None] + t2[:, None, :]).flatten(1)
        idx = (i1[:, :, None] * self.n + i2[:, None, :]).flatten(1)
        top, pos = scores.topk(self.k, dim=1)
        w = F.softmax(top, dim=1)
        return (w.unsqueeze(-1) * self.values(idx.gather(1, pos))).sum(1)


class PKMNet(nn.Module):
    def __init__(self, d_in, d_out, width, n_keys, topk):
        super().__init__()
        self.inp, self.mem = nn.Linear(d_in, width), ProductKeyMemory(width, n_keys, topk)
        self.norm, self.head = nn.LayerNorm(width), nn.Linear(width, d_out)

    def forward(self, x):
        h = F.gelu(self.inp(x))
        return self.head(self.norm(h + self.mem(h)))


@register("product_key_memory", family="vector")
class ProductKeyMemoryNet(BaseModel):
    """MLP with a large sparse product-key memory layer between input and head."""
    def __init__(self, width=32, n_keys=16, topk=4, **kw):
        super().__init__(**kw)
        self.width, self.n_keys, self.topk = width, n_keys, topk

    def build(self, d_in, d_out):
        return PKMNet(d_in, d_out, self.width, self.n_keys, self.topk)


# ---------------------------------------------------------------- memory networks ---
class MemN2N(nn.Module):
    """End-to-end memory network. Input (B, S+1, d): S memory slots followed by the query."""

    def __init__(self, d_in, hidden, d_out, hops):
        super().__init__()
        self.A, self.C, self.B = nn.Linear(d_in, hidden), nn.Linear(d_in, hidden), nn.Linear(d_in, hidden)
        self.W, self.hops = nn.Linear(hidden, d_out), hops

    def forward(self, x):
        m, c, u = self.A(x[:, :-1]), self.C(x[:, :-1]), self.B(x[:, -1])
        for _ in range(self.hops):
            p = F.softmax((m * u.unsqueeze(1)).sum(-1), dim=1)
            u = u + (p.unsqueeze(-1) * c).sum(1)
        return self.W(u)


class KVMemN(nn.Module):
    """Key-value memory network: each slot is [key | value]; addressing uses keys, reading
    uses values, and the query is refined by a separate matrix on every hop."""

    def __init__(self, d_in, d_key, hidden, d_out, hops):
        super().__init__()
        self.dk = d_key
        self.Ak, self.Cv = nn.Linear(d_key, hidden), nn.Linear(d_in - d_key, hidden)
        self.Q = nn.Linear(d_key, hidden)
        self.R = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(hops)])
        self.W = nn.Linear(hidden, d_out)

    def forward(self, x):
        k, v = self.Ak(x[:, :-1, :self.dk]), self.Cv(x[:, :-1, self.dk:])
        q = self.Q(x[:, -1, :self.dk])
        for R in self.R:
            p = F.softmax((k * q.unsqueeze(1)).sum(-1), dim=1)
            q = R(q + (p.unsqueeze(-1) * v).sum(1))
        return self.W(q)


@register("memory_network", family="memory")
class MemoryNetwork(SequenceModel):
    """X is (n, S+1, d): S memory slots, then the query as the last row."""
    def __init__(self, hidden=32, hops=2, **kw):
        super().__init__(hidden=hidden, **kw)
        self.hops = hops

    def build(self, d_in, d_out):
        return MemN2N(d_in, self.hidden, d_out, self.hops)


@register("kv_memory_network", family="memory")
class KeyValueMemoryNetwork(SequenceModel):
    """Like MemoryNetwork, but each row is [key (d_key dims) | value]; the query row uses
    its first d_key dims."""
    def __init__(self, d_key, hidden=32, hops=2, **kw):
        super().__init__(hidden=hidden, **kw)
        self.d_key, self.hops = d_key, hops

    def build(self, d_in, d_out):
        return KVMemN(d_in, self.d_key, self.hidden, d_out, self.hops)


# ---------------------------------------------------- NTM and DNC (per-step models) ---
def cosine_sim(mem, key):  # (B, N, M), (B, M) -> (B, N)
    return F.cosine_similarity(mem, key.unsqueeze(1), dim=-1, eps=1e-6)


class NTMNet(nn.Module):
    """Neural Turing Machine: LSTM controller, content + location addressing
    (interpolation, 3-way circular shift, sharpening), erase/add writes."""

    def __init__(self, d_in, d_out, hidden=64, n_slots=16, slot_size=8, readout="all"):
        super().__init__()
        self.N, self.M, self.readout = n_slots, slot_size, readout
        self.ctrl = nn.LSTMCell(d_in + slot_size, hidden)
        head = slot_size + 1 + 1 + 3 + 1
        self.read_head, self.write_head = nn.Linear(hidden, head), nn.Linear(hidden, head + 2 * slot_size)
        self.out = nn.Linear(hidden + slot_size, d_out)
        self.mem0 = nn.Parameter(torch.randn(n_slots, slot_size) * 0.1)
        self.hidden = hidden

    def address(self, mem, w_prev, p):
        M = self.M
        key, beta, g, shift, gamma = p[:, :M], p[:, M], p[:, M + 1], p[:, M + 2:M + 5], p[:, M + 5]
        w = F.softmax(F.softplus(beta).unsqueeze(1) * cosine_sim(mem, key), dim=1)
        g = torch.sigmoid(g).unsqueeze(1)
        w = g * w + (1 - g) * w_prev
        s = F.softmax(shift, dim=1)
        w = s[:, :1] * w.roll(-1, 1) + s[:, 1:2] * w + s[:, 2:] * w.roll(1, 1)
        w = w ** (1 + F.softplus(gamma)).unsqueeze(1)
        return w / (w.sum(1, keepdim=True) + 1e-8)

    def forward(self, x):
        B, T, _ = x.shape
        mem = self.mem0.expand(B, -1, -1)
        w_r = w_w = F.one_hot(torch.zeros(B, dtype=torch.long, device=x.device), self.N).float()
        h = c = torch.zeros(B, self.hidden, device=x.device)
        r, ys = torch.zeros(B, self.M, device=x.device), []
        for t in range(T):
            h, c = self.ctrl(torch.cat([x[:, t], r], -1), (h, c))
            wp = self.write_head(h)
            w_w = self.address(mem, w_w, wp[:, :self.M + 6])
            erase, add = torch.sigmoid(wp[:, self.M + 6:2 * self.M + 6]), torch.tanh(wp[:, 2 * self.M + 6:])
            mem = mem * (1 - w_w.unsqueeze(2) * erase.unsqueeze(1)) + w_w.unsqueeze(2) * add.unsqueeze(1)
            w_r = self.address(mem, w_r, self.read_head(h))
            r = (w_r.unsqueeze(2) * mem).sum(1)
            ys.append(self.out(torch.cat([h, r], -1)))
        ys = torch.stack(ys, 1)
        return ys if self.readout == "all" else (ys[:, -1] if self.readout == "last" else ys.mean(1))


class DNCNet(nn.Module):
    """Differentiable Neural Computer: usage-based allocation, a temporal link matrix that
    records write order, and read heads that mix content, forward and backward addressing."""

    def __init__(self, d_in, d_out, hidden=64, n_slots=16, slot_size=8, n_reads=2, readout="all"):
        super().__init__()
        self.N, self.W, self.R, self.readout, self.hidden = n_slots, slot_size, n_reads, readout, hidden
        self.ctrl = nn.LSTMCell(d_in + n_reads * slot_size, hidden)
        W, R = slot_size, n_reads
        self.iface = nn.Linear(hidden, R * W + R + W + 1 + W + W + R + 1 + 1 + 3 * R)
        self.out = nn.Linear(hidden + R * W, d_out)

    def forward(self, x):
        B, T, _ = x.shape
        N, W, R, dev = self.N, self.W, self.R, x.device
        mem, usage = torch.zeros(B, N, W, device=dev), torch.zeros(B, N, device=dev)
        link, prec = torch.zeros(B, N, N, device=dev), torch.zeros(B, N, device=dev)
        w_w, w_r = torch.zeros(B, N, device=dev), torch.zeros(B, R, N, device=dev)
        h = c = torch.zeros(B, self.hidden, device=dev)
        reads, ys = torch.zeros(B, R * W, device=dev), []
        for t in range(T):
            h, c = self.ctrl(torch.cat([x[:, t], reads], -1), (h, c))
            i = self.iface(h)
            o = 0

            def take(n):
                nonlocal o
                v = i[:, o:o + n]; o += n
                return v
            rk, rs = take(R * W).view(B, R, W), F.softplus(take(R))
            wk, ws = take(W), F.softplus(take(1))
            erase, wv = torch.sigmoid(take(W)), take(W)
            free, ga, gw = torch.sigmoid(take(R)), torch.sigmoid(take(1)), torch.sigmoid(take(1))
            modes = F.softmax(take(3 * R).view(B, R, 3), dim=-1)
            psi = torch.prod(1 - free.unsqueeze(-1) * w_r, dim=1)                # retention
            usage = (usage + w_w - usage * w_w) * psi
            su, phi = usage.sort(dim=1)                                         # allocation weighting
            excl = torch.cumprod(torch.cat([torch.ones(B, 1, device=dev), su[:, :-1]], 1), dim=1)
            alloc = torch.zeros_like(usage).scatter(1, phi, (1 - su) * excl)
            cw = F.softmax(ws * cosine_sim(mem, wk), dim=1)
            w_w = gw * (ga * alloc + (1 - ga) * cw)
            mem = mem * (1 - w_w.unsqueeze(2) * erase.unsqueeze(1)) + w_w.unsqueeze(2) * wv.unsqueeze(1)
            link = (1 - w_w.unsqueeze(2) - w_w.unsqueeze(1)) * link + w_w.unsqueeze(2) * prec.unsqueeze(1)
            link = link * (1 - torch.eye(N, device=dev))
            prec = (1 - w_w.sum(1, keepdim=True)) * prec + w_w
            fwd, bwd = torch.einsum("bij,brj->bri", link, w_r), torch.einsum("bji,brj->bri", link, w_r)
            cr = F.softmax(rs.unsqueeze(-1) * torch.stack([cosine_sim(mem, rk[:, r]) for r in range(R)], 1), dim=-1)
            w_r = modes[..., 0:1] * bwd + modes[..., 1:2] * cr + modes[..., 2:3] * fwd
            reads = torch.einsum("brn,bnw->brw", w_r, mem).reshape(B, R * W)
            ys.append(self.out(torch.cat([h, reads], -1)))
        ys = torch.stack(ys, 1)
        return ys if self.readout == "all" else (ys[:, -1] if self.readout == "last" else ys.mean(1))


@register("ntm", family="memory")
class NeuralTuringMachine(SequenceModel):
    def __init__(self, hidden=64, n_slots=16, slot_size=8, **kw):
        super().__init__(hidden=hidden, **kw)
        self.n_slots, self.slot_size = n_slots, slot_size

    def build(self, d_in, d_out):
        return NTMNet(d_in, d_out, self.hidden, self.n_slots, self.slot_size, self.readout)


@register("dnc", family="memory")
class DifferentiableNeuralComputer(SequenceModel):
    def __init__(self, hidden=64, n_slots=16, slot_size=8, n_reads=2, **kw):
        super().__init__(hidden=hidden, **kw)
        self.n_slots, self.slot_size, self.n_reads = n_slots, slot_size, n_reads

    def build(self, d_in, d_out):
        return DNCNet(d_in, d_out, self.hidden, self.n_slots, self.slot_size, self.n_reads, self.readout)
