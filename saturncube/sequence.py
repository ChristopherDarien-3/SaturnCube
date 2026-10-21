"""Sequence family: recurrent, reservoir and state-space models (A2).

Input is (n, T, d) floats, or (n, T) integer tokens when vocab_size is given. Targets:
  * y per sequence (readout='last' or 'mean')  -> sequence classification / regression
  * y per step     (readout='all')             -> tagging
  * y=None with vocab_size                     -> next-token prediction (language modelling)
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel, register


class SequenceModel(BaseModel):
    family = "sequence"

    def __init__(self, hidden=32, readout="last", vocab_size=None, tbptt=None, **kw):
        if readout not in ("last", "mean", "all"):
            raise ValueError("readout must be 'last', 'mean' or 'all'")
        if vocab_size:
            kw["normalize"] = False
        super().__init__(**kw)
        self.hidden, self.readout, self.vocab_size, self.tbptt = hidden, readout, vocab_size, tbptt

    # ---- data ------------------------------------------------------------------
    def _as_x(self, X):
        if self.vocab_size:
            return np.asarray(X, dtype=np.int64)
        X = np.asarray(X, dtype=np.float32)
        return X[:, :, None] if X.ndim == 2 else X

    def _norm(self, X):
        if self.vocab_size:
            return torch.as_tensor(X).long().to(self.device)
        return super()._norm(X)

    def fit(self, X, y=None, epochs=None, val_data=None):
        X = self._as_x(X)
        if y is None:  # next-token prediction
            if not self.vocab_size:
                raise ValueError("y is required unless vocab_size is set (next-token prediction)")
            X, y = X[:, :-1], X[:, 1:]
            self.readout, self.task = "all", "classification"
            if self._declared_classes is None:
                self._declared_classes = np.arange(self.vocab_size)
        return super().fit(X, y, epochs=epochs, val_data=val_data)

    def evaluate(self, X, y=None):
        X = self._as_x(X)
        if y is None:
            X, y = X[:, :-1], X[:, 1:]
        return super().evaluate(X, y)

    def generate(self, prompt, n_new=10):
        """Greedy continuation of integer token sequences (language-model mode)."""
        self._check_fitted()
        toks = self._as_x(prompt)
        for _ in range(n_new):
            out = self._outputs(self._norm(toks))[:, -1].argmax(-1).cpu().numpy()
            toks = np.concatenate([toks, self.classes_[out][:, None]], axis=1)
        return toks

    # ---- training --------------------------------------------------------------
    def _train_batch(self, m, opt, xb, yb):
        if not (self.tbptt and self.readout == "all" and getattr(getattr(m, "core", None), "recurrent", False)):
            return super()._train_batch(m, opt, xb, yb)
        # truncated BPTT: gradients stop every `tbptt` steps, the hidden state carries on
        state, total, n = None, 0.0, 0
        for i in range(0, xb.shape[1], self.tbptt):
            out, state = m.forward_state(xb[:, i:i + self.tbptt], state)
            state = tuple(s.detach() for s in state) if isinstance(state, tuple) else state.detach()
            opt.zero_grad()
            loss = self._loss(out, yb[:, i:i + self.tbptt])
            loss.backward()
            opt.step()
            total, n = total + loss.item(), n + 1
        return total / n


# ----------------------------------------------------------------------- cores ---
class Core(nn.Module):
    recurrent = False
    no_all = False  # True if the core cannot produce one output per step

    def pool(self, y, mode):
        return y[:, -1] if mode == "last" else y.mean(1)


class SeqNet(nn.Module):
    def __init__(self, core, d_in, hidden, d_out, readout, vocab_size=None):
        super().__init__()
        if readout == "all" and core.no_all:
            raise ValueError(f"{type(core).__name__} cannot give per-step outputs; use readout='last' or 'mean'")
        self.embed = nn.Embedding(vocab_size, hidden) if vocab_size else nn.Linear(d_in, hidden)
        self.core, self.readout = core, readout
        self.head = nn.Linear(core.out_dim, d_out)

    def forward(self, x):
        y = self.core(self.embed(x))
        return self.head(y if self.readout == "all" else self.core.pool(y, self.readout))

    def forward_state(self, x, state=None):
        y, state = self.core(self.embed(x), state, True)
        return self.head(y), state


class RNNCore(Core):
    recurrent = True

    def __init__(self, kind, hidden, layers=1, bidirectional=False):
        super().__init__()
        cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.rnn = cls(hidden, hidden, layers, batch_first=True, bidirectional=bidirectional)
        self.hidden, self.bi, self.out_dim = hidden, bidirectional, hidden * (2 if bidirectional else 1)

    def forward(self, h, state=None, return_state=False):
        y, state = self.rnn(h, state)
        return (y, state) if return_state else y

    def pool(self, y, mode):
        if mode == "last" and self.bi:
            return torch.cat([y[:, -1, :self.hidden], y[:, 0, self.hidden:]], dim=-1)
        return super().pool(y, mode)


class CTRNNCore(Core):
    """Continuous-time RNN: tau dh/dt = -h + W tanh(h) + x, integrated with Euler steps."""
    recurrent = True

    def __init__(self, hidden, dt=0.2, substeps=2):
        super().__init__()
        self.W = nn.Parameter(torch.randn(hidden, hidden) / math.sqrt(hidden))
        self.log_tau = nn.Parameter(torch.zeros(hidden))
        self.dt, self.substeps, self.out_dim, self.hidden = dt, substeps, hidden, hidden

    def forward(self, x, state=None, return_state=False):
        h = torch.zeros(x.shape[0], self.hidden, device=x.device) if state is None else state
        tau, ys = 1.0 + F.softplus(self.log_tau), []
        for t in range(x.shape[1]):
            for _ in range(self.substeps):
                h = h + (self.dt / self.substeps) / tau * (-h + torch.tanh(h) @ self.W.T + x[:, t])
            ys.append(h)
        y = torch.stack(ys, 1)
        return (y, h) if return_state else y


class NeuralSSMCore(Core):
    """Latent linear dynamics h_t = A h_{t-1} + B x_t (A rescaled to be stable) with a
    nonlinear emission network y_t = MLP([h_t, x_t])."""

    def __init__(self, hidden, state_dim=32):
        super().__init__()
        self.W = nn.Parameter(torch.randn(state_dim, state_dim) / math.sqrt(state_dim))
        self.rho = nn.Parameter(torch.tensor(2.0))
        self.B = nn.Linear(hidden, state_dim)
        self.emit = nn.Sequential(nn.Linear(state_dim + hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.out_dim, self.state_dim = hidden, state_dim

    def forward(self, x):
        A = torch.sigmoid(self.rho) * self.W / (torch.linalg.matrix_norm(self.W, 2) + 1e-6)
        h, ys = torch.zeros(x.shape[0], self.state_dim, device=x.device), []
        for t in range(x.shape[1]):
            h = h @ A.T + self.B(x[:, t])
            ys.append(self.emit(torch.cat([h, x[:, t]], -1)))
        return torch.stack(ys, 1)


class S4DLayer(nn.Module):
    """Diagonal structured state-space layer (S4D-Lin initialisation): the SSM is unrolled
    into a long convolution kernel and applied with an FFT."""

    def __init__(self, d, state_dim=16):
        super().__init__()
        self.log_dt = nn.Parameter(torch.rand(d) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        self.log_neg_re = nn.Parameter(torch.full((d, state_dim), math.log(0.5)))
        self.im = nn.Parameter(math.pi * torch.arange(state_dim).float().repeat(d, 1))
        self.C = nn.Parameter(torch.randn(d, state_dim, 2) * 0.5)
        self.D = nn.Parameter(torch.ones(d))
        self.mix = nn.Linear(d, 2 * d)

    def kernel(self, T):
        dt = self.log_dt.exp()[:, None]
        A = torch.complex(-self.log_neg_re.exp(), self.im)
        dtA = A * dt
        coef = torch.view_as_complex(self.C) * (torch.exp(dtA) - 1) / A
        powers = torch.exp(dtA[..., None] * torch.arange(T, device=dtA.device))
        return 2 * torch.einsum("hn,hnt->ht", coef, powers).real

    def forward(self, u):  # (B, T, d)
        T = u.shape[1]
        x = u.transpose(1, 2)
        y = torch.fft.irfft(torch.fft.rfft(x, n=2 * T) * torch.fft.rfft(self.kernel(T), n=2 * T), n=2 * T)[..., :T]
        y = F.gelu(y + self.D[:, None] * x).transpose(1, 2)
        return F.glu(self.mix(y), dim=-1)


class S4Core(Core):
    def __init__(self, hidden, n_layers=2, state_dim=16):
        super().__init__()
        self.layers = nn.ModuleList([S4DLayer(hidden, state_dim) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.out_dim = hidden

    def forward(self, x):
        for layer, norm in zip(self.layers, self.norms):
            x = norm(x + layer(x))
        return x


class MambaBlock(nn.Module):
    """Selective state-space block: step size, B and C depend on the input (Mamba-style).
    Uses a sequential scan, which is simple but not hardware-optimised."""

    def __init__(self, d, state_dim=8, expand=2, d_conv=4):
        super().__init__()
        di, self.di, self.N = expand * d, expand * d, state_dim
        self.in_proj = nn.Linear(d, 2 * di)
        self.conv = nn.Conv1d(di, di, d_conv, groups=di, padding=d_conv - 1)
        self.dt_rank = max(1, d // 8)
        self.x_proj = nn.Linear(di, self.dt_rank + 2 * state_dim, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, di)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, state_dim + 1).float()).repeat(di, 1))
        self.D = nn.Parameter(torch.ones(di))
        self.out_proj = nn.Linear(di, d)

    def forward(self, x):
        B, T, _ = x.shape
        xi, z = self.in_proj(x).chunk(2, dim=-1)
        xi = F.silu(self.conv(xi.transpose(1, 2))[..., :T].transpose(1, 2))
        dt_raw, Bm, Cm = self.x_proj(xi).split([self.dt_rank, self.N, self.N], dim=-1)
        dt, A = F.softplus(self.dt_proj(dt_raw)), -self.A_log.exp()
        h, ys = torch.zeros(B, self.di, self.N, device=x.device), []
        for t in range(T):
            h = torch.exp(dt[:, t, :, None] * A) * h + dt[:, t, :, None] * Bm[:, t, None, :] * xi[:, t, :, None]
            ys.append((h * Cm[:, t, None, :]).sum(-1) + self.D * xi[:, t])
        return self.out_proj(torch.stack(ys, 1) * F.silu(z))


class MambaCore(Core):
    def __init__(self, hidden, n_layers=2, state_dim=8):
        super().__init__()
        self.blocks = nn.ModuleList([MambaBlock(hidden, state_dim) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.out_dim = hidden

    def forward(self, x):
        for b, n in zip(self.blocks, self.norms):
            x = x + b(n(x))
        return x


class PlasticCore(Core):
    """Differentiable plasticity (Miconi et al., 2018): every connection has a fixed weight
    plus alpha * Hebb, where the Hebbian trace is updated during the sequence and alpha is
    learned by backpropagation."""

    def __init__(self, hidden):
        super().__init__()
        self.W = nn.Parameter(torch.randn(hidden, hidden) / math.sqrt(hidden))
        self.alpha = nn.Parameter(torch.randn(hidden, hidden) * 0.1)
        self.eta = nn.Parameter(torch.tensor(-1.0))
        self.out_dim, self.hidden = hidden, hidden

    def forward(self, x):
        B, T, H = x.shape
        h, hebb, eta, ys = torch.zeros(B, H, device=x.device), torch.zeros(B, H, H, device=x.device), torch.sigmoid(self.eta), []
        for t in range(T):
            new = torch.tanh(x[:, t] + h @ self.W + torch.bmm(h.unsqueeze(1), self.alpha * hebb).squeeze(1))
            hebb = (1 - eta) * hebb + eta * h.unsqueeze(2) * new.unsqueeze(1)
            h = new
            ys.append(h)
        return torch.stack(ys, 1)


# -------------------------------------------------------------- model wrappers ---
class _CoreModel(SequenceModel):
    def make_core(self):
        raise NotImplementedError

    def build(self, d_in, d_out):
        return SeqNet(self.make_core(), d_in, self.hidden, d_out, self.readout, self.vocab_size)


class _Recurrent(_CoreModel):
    kind, bi = "rnn", False

    def __init__(self, hidden=32, layers=1, **kw):
        super().__init__(hidden=hidden, **kw)
        self.layers = layers

    def make_core(self):
        return RNNCore(self.kind, self.hidden, self.layers, self.bi)


@register("rnn", family="sequence")
class SimpleRNN(_Recurrent):
    kind = "rnn"


@register("elman", family="sequence")
class ElmanNetwork(_Recurrent):
    """Elman network: the classic simple recurrent net (hidden state fed back as context)."""
    kind = "rnn"


@register("lstm", family="sequence")
class LSTM(_Recurrent):
    kind = "lstm"


@register("gru", family="sequence")
class GRU(_Recurrent):
    kind = "gru"


@register("birnn", family="sequence")
class BidirectionalRNN(_Recurrent):
    kind, bi = "rnn", True


@register("bilstm", family="sequence")
class BidirectionalLSTM(_Recurrent):
    kind, bi = "lstm", True


@register("bigru", family="sequence")
class BidirectionalGRU(_Recurrent):
    kind, bi = "gru", True


@register("ctrnn", family="sequence")
class ContinuousTimeRNN(_CoreModel):
    def __init__(self, hidden=32, dt=0.2, substeps=2, **kw):
        super().__init__(hidden=hidden, **kw)
        self.dt, self.substeps = dt, substeps

    def make_core(self):
        return CTRNNCore(self.hidden, self.dt, self.substeps)


@register("neural_ssm", family="sequence")
class NeuralStateSpace(_CoreModel):
    def __init__(self, hidden=32, state_dim=32, **kw):
        super().__init__(hidden=hidden, **kw)
        self.state_dim = state_dim

    def make_core(self):
        return NeuralSSMCore(self.hidden, self.state_dim)


@register("s4", family="sequence")
class S4(_CoreModel):
    """Structured state-space sequence model (diagonal S4D variant, FFT convolution)."""
    def __init__(self, hidden=32, n_layers=2, state_dim=16, **kw):
        super().__init__(hidden=hidden, **kw)
        self.n_layers, self.state_dim = n_layers, state_dim

    def make_core(self):
        return S4Core(self.hidden, self.n_layers, self.state_dim)


@register("mamba", family="sequence")
class Mamba(_CoreModel):
    """Mamba-style selective state-space model."""
    def __init__(self, hidden=32, n_layers=2, state_dim=8, **kw):
        super().__init__(hidden=hidden, **kw)
        self.n_layers, self.state_dim = n_layers, state_dim

    def make_core(self):
        return MambaCore(self.hidden, self.n_layers, self.state_dim)


@register("plastic_rnn", family="sequence")
class PlasticRNN(_CoreModel):
    """Differentiable plasticity: an RNN with learned Hebbian fast weights."""
    def make_core(self):
        return PlasticCore(self.hidden)


# ---------------------------------------------------------------- Jordan network ---
class JordanNet(nn.Module):
    """Jordan network: the previous *output* (not the hidden state) is fed back."""

    def __init__(self, d_in, hidden, d_out, readout, vocab_size=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden) if vocab_size else nn.Linear(d_in, hidden)
        self.U, self.V = nn.Linear(d_out, hidden, bias=False), nn.Linear(hidden, d_out)
        self.readout, self.d_out = readout, d_out

    def forward(self, x):
        e, y, ys = self.embed(x), None, []
        y = torch.zeros(e.shape[0], self.d_out, device=e.device)
        for t in range(e.shape[1]):
            y = self.V(torch.tanh(e[:, t] + self.U(y)))
            ys.append(y)
        ys = torch.stack(ys, 1)
        return ys if self.readout == "all" else (ys[:, -1] if self.readout == "last" else ys.mean(1))


@register("jordan", family="sequence")
class JordanNetwork(SequenceModel):
    def build(self, d_in, d_out):
        return JordanNet(d_in, self.hidden, d_out, self.readout, self.vocab_size)


# --------------------------------------------------------- reservoir computing ---
def ridge_readout(layer, feats, targets, ridge):
    """Closed-form ridge regression for a linear readout (with bias)."""
    with torch.no_grad():
        H = torch.cat([feats.double(), torch.ones(len(feats), 1, dtype=torch.double, device=feats.device)], 1)
        A = H.T @ H + ridge * torch.eye(H.shape[1], dtype=torch.double, device=H.device)
        beta = torch.linalg.solve(A, H.T @ targets.double()).float()
        layer.weight.copy_(beta[:-1].T)
        layer.bias.copy_(beta[-1])


class ESNNet(nn.Module):
    def __init__(self, d_in, n_res, d_out, spectral_radius=0.9, leak=1.0, input_scale=1.0,
                 density=0.1, readout="last", vocab_size=None):
        super().__init__()
        W = torch.randn(n_res, n_res) * (torch.rand(n_res, n_res) < density)
        W = W * spectral_radius / (torch.linalg.eigvals(W).abs().max() + 1e-9)
        self.register_buffer("W", W)
        if vocab_size:
            self.inp, d_eff = nn.Embedding(vocab_size, 16), 16
            self.inp.weight.requires_grad_(False)
        else:
            self.inp, d_eff = nn.Identity(), d_in
        self.register_buffer("W_in", (torch.rand(n_res, d_eff) * 2 - 1) * input_scale)
        self.out = nn.Linear(n_res, d_out)
        for p in self.out.parameters():
            p.requires_grad_(False)
        self.leak, self.readout, self.n_res = leak, readout, n_res

    def reservoir(self, x):
        h, states = torch.zeros(x.shape[0], self.n_res, device=x.device), []
        for t in range(x.shape[1]):
            h = (1 - self.leak) * h + self.leak * torch.tanh(x[:, t] @ self.W_in.T + h @ self.W.T)
            states.append(h)
        return torch.stack(states, 1)

    @torch.no_grad()
    def features(self, x):
        S = self.reservoir(self.inp(x))
        return S if self.readout == "all" else (S[:, -1] if self.readout == "last" else S.mean(1))

    def forward(self, x):
        return self.out(self.features(x))


class LSMNet(ESNNet):
    """Liquid state machine: the reservoir is a recurrent network of leaky integrate-and-fire
    neurons; the readout sees low-pass-filtered spike trains."""

    def __init__(self, *a, decay=0.8, threshold=1.0, trace_decay=0.8, **kw):
        super().__init__(*a, **kw)
        self.decay, self.threshold, self.trace_decay = decay, threshold, trace_decay

    def reservoir(self, x):
        B = x.shape[0]
        v, spk, r = (torch.zeros(B, self.n_res, device=x.device) for _ in range(3))
        out = []
        for t in range(x.shape[1]):
            v = self.decay * v + x[:, t] @ self.W_in.T + spk @ self.W.T
            spk = (v > self.threshold).float()
            v = v * (1 - spk)
            r = self.trace_decay * r + spk
            out.append(r)
        return torch.stack(out, 1)


@register("esn", family="sequence")
class EchoStateNetwork(SequenceModel):
    """Echo state network: fixed random recurrent reservoir, ridge-regression readout."""
    net = ESNNet

    def __init__(self, n_reservoir=200, spectral_radius=0.9, leak=1.0, input_scale=1.0,
                 density=0.1, ridge=1e-3, **kw):
        super().__init__(**kw)
        self.n_res, self.sr, self.leak, self.input_scale = n_reservoir, spectral_radius, leak, input_scale
        self.density, self.ridge = density, ridge

    def build(self, d_in, d_out):
        return self.net(d_in, self.n_res, d_out, self.sr, self.leak, self.input_scale,
                        self.density, self.readout, self.vocab_size)

    def _loss(self, out, y):
        t = F.one_hot(y, out.shape[-1]).float() if self.task == "classification" else y
        return F.mse_loss(out, t)

    def _fit(self, Xt, yt, epochs, val):
        m = self._module
        feats = torch.cat([m.features(Xt[i:i + 256]) for i in range(0, len(Xt), 256)])
        feats = feats.reshape(-1, feats.shape[-1])
        if self.task == "classification":
            tgt = F.one_hot(yt.reshape(-1), m.out.out_features)
        else:
            tgt = yt.reshape(-1, yt.shape[-1])
        ridge_readout(m.out, feats, tgt, self.ridge)
        self.history_["loss"].append(self._loss(self._outputs(Xt), yt).item())


@register("lsm", family="sequence")
class LiquidStateMachine(EchoStateNetwork):
    """Liquid state machine: spiking (LIF) reservoir + ridge readout."""
    net = LSMNet

    def __init__(self, n_reservoir=400, spectral_radius=1.0, input_scale=3.0, ridge=1.0, **kw):
        super().__init__(n_reservoir=n_reservoir, spectral_radius=spectral_radius,
                         input_scale=input_scale, ridge=ridge, **kw)


# ------------------------------------------------------------------------ RTRL ---
class RTRLNet(nn.Module):
    """Plain tanh RNN whose parameters can also be trained with exact Real-Time Recurrent
    Learning (forward-propagated sensitivities dh/dtheta)."""

    def __init__(self, d_in, hidden, d_out):
        super().__init__()
        self.Wx = nn.Parameter(torch.randn(hidden, d_in) / math.sqrt(d_in))
        self.Wh = nn.Parameter(torch.randn(hidden, hidden) / math.sqrt(hidden))
        self.b = nn.Parameter(torch.zeros(hidden))
        self.readout = nn.Linear(hidden, d_out)
        self.H, self.d = hidden, d_in

    def final_state(self, x):
        h = torch.zeros(x.shape[0], self.H, device=x.device)
        for t in range(x.shape[1]):
            h = torch.tanh(x[:, t] @ self.Wx.T + h @ self.Wh.T + self.b)
        return h

    def forward(self, x):
        return self.readout(self.final_state(x))

    def rtrl_grads(self, x, loss_fn):
        """Set .grad of every parameter from forward-mode sensitivities; returns the loss."""
        B, H, d = x.shape[0], self.H, self.d
        eye = torch.eye(H, device=x.device)
        P = torch.zeros(B, H, H * H + H * d + H, device=x.device)  # dh_t / d[Wh, Wx, b]
        h = torch.zeros(B, H, device=x.device)
        with torch.no_grad():
            for t in range(x.shape[1]):
                h_new = torch.tanh(x[:, t] @ self.Wx.T + h @ self.Wh.T + self.b)
                imm = torch.cat([torch.einsum("ik,bj->bikj", eye, h).reshape(B, H, H * H),
                                 torch.einsum("ik,bj->bikj", eye, x[:, t]).reshape(B, H, H * d),
                                 eye.expand(B, H, H)], dim=2)
                P = (1 - h_new ** 2)[:, :, None] * (torch.einsum("ik,bkp->bip", self.Wh, P) + imm)
                h = h_new
        h = h.detach().requires_grad_(True)
        loss = loss_fn(self.readout(h))
        loss.backward()  # readout gradients and dL/dh_T
        g = torch.einsum("bi,bip->p", h.grad, P)
        self.Wh.grad = g[:H * H].view(H, H)
        self.Wx.grad = g[H * H:H * H + H * d].view(H, d)
        self.b.grad = g[H * H + H * d:]
        return loss


@register("rtrl_rnn", family="sequence")
class RTRLRNN(SequenceModel):
    """Simple RNN trained with exact RTRL (no backward pass through time). Cost grows as
    hidden**4, so keep `hidden` small (<= 32)."""
    def __init__(self, hidden=8, **kw):
        super().__init__(hidden=hidden, **kw)

    def build(self, d_in, d_out):
        return RTRLNet(d_in, self.hidden, d_out)

    def _train_batch(self, m, opt, xb, yb):
        opt.zero_grad()
        loss = m.rtrl_grads(xb, lambda out: self._loss(out, yb))
        opt.step()
        return loss.item()
