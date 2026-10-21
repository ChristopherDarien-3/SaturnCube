"""Attention and Transformer family (A3)."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import register
from .sequence import Core, MambaBlock, SequenceModel, SeqNet, _CoreModel


def sinusoid(T, d, device):
    pos = torch.arange(T, device=device).float()[:, None]
    div = torch.exp(-math.log(10000.0) * torch.arange(0, d, 2, device=device).float() / d)
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)[:, : d // 2]
    return pe


def make_mask(T, causal=False, window=None, stride=None, device=None):
    """Boolean (T, T) mask, True = may attend. window -> local band; stride -> every
    stride-th earlier/later position as well (sparse, Child et al.)."""
    i = torch.arange(T, device=device)[:, None]
    j = torch.arange(T, device=device)[None, :]
    allowed = torch.ones(T, T, dtype=torch.bool, device=device)
    if window is not None or stride is not None:
        allowed = (i - j).abs() <= window if window is not None else torch.zeros_like(allowed)
        if stride:
            allowed = allowed | ((i - j) % stride == 0)
    return allowed & (j <= i) if causal else allowed


class MultiHeadAttention(nn.Module):
    """Multi-head (n_kv_heads=n_heads), multi-query (1) or grouped-query (1 < g < heads)
    attention. Pass `context` for cross-attention."""

    def __init__(self, d, n_heads=4, n_kv_heads=None):
        super().__init__()
        kv = n_kv_heads or n_heads
        if d % n_heads or n_heads % kv:
            raise ValueError("d must divide by n_heads and n_heads by n_kv_heads")
        self.h, self.kv, self.dh = n_heads, kv, d // n_heads
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, kv * self.dh), nn.Linear(d, kv * self.dh)
        self.o = nn.Linear(d, d)

    def forward(self, x, context=None, mask=None):
        ctx = x if context is None else context
        B, Tq, _ = x.shape
        Tk = ctx.shape[1]
        q = self.q(x).view(B, Tq, self.h, self.dh).transpose(1, 2)
        k = self.k(ctx).view(B, Tk, self.kv, self.dh).transpose(1, 2)
        v = self.v(ctx).view(B, Tk, self.kv, self.dh).transpose(1, 2)
        if self.kv != self.h:
            k, v = k.repeat_interleave(self.h // self.kv, 1), v.repeat_interleave(self.h // self.kv, 1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(y.transpose(1, 2).reshape(B, Tq, -1))


class LinearAttention(nn.Module):
    """Linear attention (Katharopoulos et al.): softmax replaced by a positive feature map,
    so cost is linear in sequence length. causal=True uses running sums."""

    def __init__(self, d, n_heads=4, causal=False):
        super().__init__()
        self.h, self.dh, self.causal = n_heads, d // n_heads, causal
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))

    def forward(self, x, context=None, mask=None):
        B, T, _ = x.shape
        split = lambda t: t.view(B, T, self.h, self.dh).transpose(1, 2)
        q, k, v = (split(f(x)) for f in (self.q, self.k, self.v))
        q, k = F.elu(q) + 1, F.elu(k) + 1
        if self.causal:
            S = torch.cumsum(torch.einsum("bhtd,bhte->bhtde", k, v), dim=2)
            z = torch.cumsum(k, dim=2)
            y = torch.einsum("bhtd,bhtde->bhte", q, S) / (torch.einsum("bhtd,bhtd->bht", q, z)[..., None] + 1e-6)
        else:
            y = torch.einsum("bhtd,bhde->bhte", q, torch.einsum("bhtd,bhte->bhde", k, v))
            y = y / (torch.einsum("bhtd,bhd->bht", q, k.sum(2))[..., None] + 1e-6)
        return self.o(y.transpose(1, 2).reshape(B, T, -1))


class TransformerLayer(nn.Module):
    """Pre-norm layer: self-attention (full / local / sparse / linear), optional
    cross-attention to `context`, feed-forward."""

    def __init__(self, d, n_heads=4, n_kv_heads=None, ff_mult=2, attention="full", window=None,
                 stride=None, causal=False, cross=False, dropout=0.0):
        super().__init__()
        if attention == "local" and window is None:
            raise ValueError("attention='local' needs window=")
        if attention == "sparse" and (window is None or stride is None):
            raise ValueError("attention='sparse' needs window= and stride=")
        self.attention, self.causal = attention, causal
        self.window = window if attention in ("local", "sparse") else None
        self.stride = stride if attention == "sparse" else None
        self.attn = (LinearAttention(d, n_heads, causal) if attention == "linear"
                     else MultiHeadAttention(d, n_heads, n_kv_heads))
        self.cross = MultiHeadAttention(d, n_heads, n_kv_heads) if cross else None
        self.ln1, self.ln2, self.lnc = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Linear(ff_mult * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None):
        if mask is None and self.attention != "linear" and (self.causal or self.window or self.stride):
            mask = make_mask(x.shape[1], self.causal, self.window, self.stride, x.device)
        x = x + self.drop(self.attn(self.ln1(x), mask=mask))
        if self.cross is not None and context is not None:
            x = x + self.drop(self.cross(self.lnc(x), context=context))
        return x + self.drop(self.ff(self.ln2(x)))


# ----------------------------------------------------------------------- cores ---
class TransformerCore(Core):
    def __init__(self, d, n_layers=2, n_heads=4, n_kv_heads=None, attention="full", window=None,
                 stride=None, causal=False, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([TransformerLayer(d, n_heads, n_kv_heads, 2, attention, window,
                                                      stride, causal, False, dropout) for _ in range(n_layers)])
        self.ln, self.out_dim = nn.LayerNorm(d), d

    def forward(self, x):
        x = x + sinusoid(x.shape[1], x.shape[2], x.device)
        for layer in self.layers:
            x = layer(x)
        return self.ln(x)


class UniversalCore(Core):
    """Universal Transformer: one layer applied repeatedly (shared weights, depth recurrence)
    with a per-step embedding. Fixed number of steps (no adaptive halting)."""

    def __init__(self, d, n_steps=4, n_heads=4, causal=False):
        super().__init__()
        self.layer = TransformerLayer(d, n_heads, causal=causal)
        self.step_emb = nn.Parameter(torch.randn(n_steps, d) * 0.1)
        self.ln, self.out_dim = nn.LayerNorm(d), d

    def forward(self, x):
        x = x + sinusoid(x.shape[1], x.shape[2], x.device)
        for e in self.step_emb:
            x = self.layer(x + e)
        return self.ln(x)


class MemoryCore(Core):
    """Memory-augmented Transformer: learned memory slots are prepended and take part in
    attention in every layer, giving the model a persistent scratchpad."""

    def __init__(self, d, n_mem=8, n_layers=2, n_heads=4, causal=False):
        super().__init__()
        self.mem = nn.Parameter(torch.randn(n_mem, d) * 0.1)
        self.layers = nn.ModuleList([TransformerLayer(d, n_heads) for _ in range(n_layers)])
        self.ln, self.out_dim, self.n_mem, self.causal = nn.LayerNorm(d), d, n_mem, causal

    def forward(self, x):
        B, T, d = x.shape
        n = self.n_mem
        x = torch.cat([self.mem.expand(B, -1, -1), x + sinusoid(T, d, x.device)], 1)
        mask = None
        if self.causal:
            mask = torch.zeros(n + T, n + T, dtype=torch.bool, device=x.device)
            mask[:, :n] = True
            mask[n:, n:] = make_mask(T, True, device=x.device)
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.ln(x[:, n:])


class PerceiverCore(Core):
    """Perceiver: a small latent array cross-attends to the (possibly long) input, then
    processes it with self-attention. Cost does not grow quadratically with input length."""
    no_all = True

    def __init__(self, d, n_latents=8, n_layers=2, n_heads=4):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(n_latents, d) * 0.1)
        self.cross, self.ln_l, self.ln_i = MultiHeadAttention(d, n_heads), nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff, self.ln_f = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)), nn.LayerNorm(d)
        self.layers = nn.ModuleList([TransformerLayer(d, n_heads) for _ in range(n_layers)])
        self.out_dim = d

    def forward(self, x):
        x = self.ln_i(x + sinusoid(x.shape[1], x.shape[2], x.device))
        lat = self.latents.expand(x.shape[0], -1, -1)
        lat = lat + self.cross(self.ln_l(lat), context=x)
        lat = lat + self.ff(self.ln_f(lat))
        for layer in self.layers:
            lat = layer(lat)
        return lat

    def pool(self, y, mode):
        return y.mean(1)


class HybridCore(Core):
    """State-space / attention hybrid: Mamba-style blocks alternate with attention layers."""

    def __init__(self, d, n_layers=2, n_heads=4, state_dim=8, causal=False):
        super().__init__()
        self.ssm = nn.ModuleList([MambaBlock(d, state_dim) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.attn = nn.ModuleList([TransformerLayer(d, n_heads, causal=causal) for _ in range(n_layers)])
        self.out_dim = d

    def forward(self, x):
        for s, n, a in zip(self.ssm, self.norms, self.attn):
            x = a(x + s(n(x)))
        return x


class RecurrentAttentionCore(Core):
    """Recurrent attention: a GRU controller repeatedly queries the whole input
    ('glimpses'), reading a context vector each step."""
    no_all = True

    def __init__(self, d, n_glimpses=4, n_heads=4):
        super().__init__()
        self.attn, self.cell = MultiHeadAttention(d, n_heads), nn.GRUCell(d, d)
        self.n_glimpses, self.out_dim, self.d = n_glimpses, d, d

    def forward(self, x):
        x = x + sinusoid(x.shape[1], x.shape[2], x.device)
        h, hs = torch.zeros(x.shape[0], self.d, device=x.device), []
        for _ in range(self.n_glimpses):
            h = self.cell(self.attn(h.unsqueeze(1), context=x).squeeze(1), h)
            hs.append(h)
        return torch.stack(hs, 1)


# ---------------------------------------------------------------------- models ---
class _AttnModel(_CoreModel):
    attention, n_kv_heads_default, causal_default = "full", None, False

    def __init__(self, hidden=32, n_layers=2, n_heads=4, n_kv_heads=None, window=None, stride=None,
                 causal=None, dropout=0.0, **kw):
        kw.setdefault("readout", "mean")
        super().__init__(hidden=hidden, **kw)
        self.n_layers, self.n_heads, self.dropout = n_layers, n_heads, dropout
        self.n_kv_heads = n_kv_heads or self.n_kv_heads_default
        self.window, self.stride = window, stride
        self.causal = self.causal_default if causal is None else causal

    def make_core(self):
        return TransformerCore(self.hidden, self.n_layers, self.n_heads, self.n_kv_heads, self.attention,
                               self.window, self.stride, self.causal, self.dropout)


@register("transformer_encoder", family="sequence")
class TransformerEncoder(_AttnModel):
    pass


@register("transformer_decoder", family="sequence")
class TransformerDecoder(_AttnModel):
    """Causal Transformer; use fit(tokens) with vocab_size for language modelling."""
    causal_default = True


@register("linear_attention", family="sequence")
class LinearAttentionTransformer(_AttnModel):
    attention = "linear"


@register("local_attention", family="sequence")
class LocalAttentionTransformer(_AttnModel):
    attention = "local"

    def __init__(self, window=3, **kw):
        super().__init__(window=window, **kw)


@register("sparse_attention", family="sequence")
class SparseAttentionTransformer(_AttnModel):
    attention = "sparse"

    def __init__(self, window=2, stride=4, **kw):
        super().__init__(window=window, stride=stride, **kw)


@register("multi_query", family="sequence")
class MultiQueryTransformer(_AttnModel):
    n_kv_heads_default = 1


@register("grouped_query", family="sequence")
class GroupedQueryTransformer(_AttnModel):
    n_kv_heads_default = 2


class _OwnCore(_CoreModel):
    def __init__(self, hidden=32, n_layers=2, n_heads=4, causal=False, **kw):
        kw.setdefault("readout", "mean")
        super().__init__(hidden=hidden, **kw)
        self.n_layers, self.n_heads, self.causal = n_layers, n_heads, causal


@register("universal_transformer", family="sequence")
class UniversalTransformer(_OwnCore):
    def __init__(self, n_steps=4, **kw):
        super().__init__(**kw)
        self.n_steps = n_steps

    def make_core(self):
        return UniversalCore(self.hidden, self.n_steps, self.n_heads, self.causal)


@register("memory_transformer", family="sequence")
class MemoryAugmentedTransformer(_OwnCore):
    def __init__(self, n_memory=8, **kw):
        super().__init__(**kw)
        self.n_memory = n_memory

    def make_core(self):
        return MemoryCore(self.hidden, self.n_memory, self.n_layers, self.n_heads, self.causal)


@register("perceiver", family="sequence")
class Perceiver(_OwnCore):
    def __init__(self, n_latents=8, **kw):
        super().__init__(**kw)
        self.n_latents = n_latents

    def make_core(self):
        return PerceiverCore(self.hidden, self.n_latents, self.n_layers, self.n_heads)


@register("ssm_attention_hybrid", family="sequence")
class SSMAttentionHybrid(_OwnCore):
    def make_core(self):
        return HybridCore(self.hidden, self.n_layers, self.n_heads, causal=self.causal)


@register("recurrent_attention", family="sequence")
class RecurrentAttention(_OwnCore):
    def __init__(self, n_glimpses=4, **kw):
        super().__init__(**kw)
        self.n_glimpses = n_glimpses

    def make_core(self):
        return RecurrentAttentionCore(self.hidden, self.n_glimpses, self.n_heads)


# ------------------------------------------------------------ retrieval-augmented ---
class RetrievalNet(nn.Module):
    """The input sequence (mean-pooled) is a query into an external key/value knowledge
    base; the top-k entries are retrieved by cosine similarity and read by cross-attention."""

    def __init__(self, d_in, hidden, d_out, keys, values, k, n_heads):
        super().__init__()
        self.register_buffer("keys", F.normalize(torch.as_tensor(keys, dtype=torch.float32), dim=1))
        self.register_buffer("values", torch.as_tensor(values, dtype=torch.float32))
        self.q_embed = nn.Linear(d_in, hidden)
        self.v_embed = nn.Linear(self.values.shape[1], hidden)
        self.read = MultiHeadAttention(hidden, n_heads)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, d_out))
        self.k = min(k, len(self.keys))

    def forward(self, x):
        q = x.mean(1)
        idx = (F.normalize(q, dim=1) @ self.keys.T).topk(self.k, dim=1).indices  # (B, k)
        retrieved = self.v_embed(self.values[idx])                              # (B, k, hidden)
        qe = self.q_embed(q).unsqueeze(1)
        return self.head(torch.cat([qe, self.read(qe, context=retrieved)], -1).squeeze(1))


@register("retrieval_augmented", family="sequence")
class RetrievalAugmented(SequenceModel):
    """Retrieval-augmented model. `knowledge=(keys, values)`: keys (M, d_in) live in the same
    space as the (unnormalised) input features, values (M, d_v) are what the model may read."""
    def __init__(self, knowledge, hidden=32, k=3, n_heads=4, **kw):
        kw.setdefault("normalize", False)
        super().__init__(hidden=hidden, **kw)
        self.knowledge, self.k, self.n_heads = knowledge, k, n_heads

    def build(self, d_in, d_out):
        keys, values = self.knowledge
        return RetrievalNet(d_in, self.hidden, d_out, keys, values, self.k, self.n_heads)


# --------------------------------------------------------- encoder-decoder model ---
class Seq2SeqNet(nn.Module):
    def __init__(self, vocab, d=32, n_layers=2, n_heads=4):
        super().__init__()
        self.src_emb, self.tgt_emb = nn.Embedding(vocab, d), nn.Embedding(vocab + 1, d)
        self.enc = TransformerCore(d, n_layers, n_heads)
        self.dec = nn.ModuleList([TransformerLayer(d, n_heads, causal=True, cross=True) for _ in range(n_layers)])
        self.ln, self.head, self.bos = nn.LayerNorm(d), nn.Linear(d, vocab), vocab

    def forward(self, src, tgt_in):
        mem = self.enc(self.src_emb(src))
        x = self.tgt_emb(tgt_in)
        x = x + sinusoid(x.shape[1], x.shape[2], x.device)
        for layer in self.dec:
            x = layer(x, context=mem)
        return self.head(self.ln(x))

    @torch.no_grad()
    def decode(self, src, length):
        tgt = torch.full((len(src), 1), self.bos, device=src.device)
        for _ in range(length):
            tgt = torch.cat([tgt, self(src, tgt)[:, -1].argmax(-1, keepdim=True)], 1)
        return tgt[:, 1:]


@register("encoder_decoder", family="sequence")
class EncoderDecoderTransformer(SequenceModel):
    """Sequence-to-sequence Transformer over integer tokens: fit(src, tgt), predict(src)."""
    def __init__(self, vocab_size, hidden=32, n_layers=2, n_heads=4, **kw):
        super().__init__(hidden=hidden, vocab_size=vocab_size, **kw)
        self.n_layers, self.n_heads, self.tgt_len = n_layers, n_heads, None

    def build(self, d_in, d_out):
        return Seq2SeqNet(self.vocab_size, self.hidden, self.n_layers, self.n_heads)

    def fit(self, X, y, epochs=None, val_data=None):
        y = np.asarray(y)
        self.tgt_len, self.task = y.shape[1], "classification"
        self._declared_classes = np.arange(self.vocab_size)
        return super().fit(X, y, epochs=epochs)

    def _train_batch(self, m, opt, xb, yb):
        tgt_in = torch.cat([torch.full((len(yb), 1), m.bos, device=yb.device), yb[:, :-1]], 1)
        opt.zero_grad()
        loss = self._loss(m(xb, tgt_in), yb)
        loss.backward()
        opt.step()
        return loss.item()

    def predict(self, X):
        self._check_fitted()
        self._module.eval()
        return self._module.decode(self._norm(self._as_x(X)), self.tgt_len).cpu().numpy()

    def evaluate(self, X, y):
        pred, y = self.predict(X), np.asarray(y)
        return {"token_accuracy": float((pred == y).mean()), "sequence_accuracy": float((pred == y).all(1).mean())}
