"""Vector/tabular family: input is a (n_samples, n_features) array."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import layers as L
from .base import BaseModel, register


@register("mlp")
class MLP(BaseModel):
    """Multilayer perceptron."""
    def __init__(self, hidden=(128, 64), activation="relu", dropout=0.0, batch_norm=False, **kw):
        super().__init__(**kw)
        self.hidden, self.activation = hidden, activation
        self.dropout, self.batch_norm = dropout, batch_norm

    def build(self, d_in, d_out):
        return L.MLPNet(d_in, d_out, self.hidden, self.activation, self.dropout, self.batch_norm)


@register("deep_ffn")
class DeepFeedforward(MLP):
    """Deep feedforward net: an MLP with deeper defaults, batch norm and dropout."""
    def __init__(self, hidden=(256,) * 6, activation="relu", dropout=0.1, batch_norm=True, **kw):
        super().__init__(hidden, activation, dropout, batch_norm, **kw)


@register("resnet")
class ResNet(BaseModel):
    """Residual MLP with pre-norm blocks."""
    def __init__(self, width=128, depth=4, activation="relu", dropout=0.0, **kw):
        super().__init__(**kw)
        self.width, self.depth, self.activation, self.dropout = width, depth, activation, dropout

    def build(self, d_in, d_out):
        return L.ResidualNet(d_in, d_out, self.width, self.depth, self.activation, self.dropout)


@register("densenet")
class DenseNet(BaseModel):
    """Densely connected MLP: every layer receives all earlier features."""
    def __init__(self, growth=32, depth=4, init_width=64, activation="relu", **kw):
        super().__init__(**kw)
        self.growth, self.depth, self.init_width, self.activation = growth, depth, init_width, activation

    def build(self, d_in, d_out):
        return L.DenseNet(d_in, d_out, self.growth, self.depth, self.init_width, self.activation)


@register("highway")
class Highway(BaseModel):
    """Highway network: learned gates mix transformed and carried features."""
    def __init__(self, width=128, depth=4, activation="relu", **kw):
        super().__init__(**kw)
        self.width, self.depth, self.activation = width, depth, activation

    def build(self, d_in, d_out):
        return L.HighwayNet(d_in, d_out, self.width, self.depth, self.activation)


@register("senet")
class SqueezeExcitation(BaseModel):
    """Residual MLP whose blocks re-weight features with a squeeze-and-excitation gate."""
    def __init__(self, width=128, depth=4, reduction=4, activation="relu", dropout=0.0, **kw):
        super().__init__(**kw)
        self.width, self.depth, self.reduction = width, depth, reduction
        self.activation, self.dropout = activation, dropout

    def build(self, d_in, d_out):
        return L.SENet(d_in, d_out, self.width, self.depth, self.reduction, self.activation, self.dropout)


@register("moe")
class MixtureOfExperts(BaseModel):
    """Mixture of experts with a softmax router (dense by default, optional top_k)."""
    def __init__(self, n_experts=4, expert_hidden=(64,), top_k=None, activation="relu", **kw):
        super().__init__(**kw)
        self.n_experts, self.expert_hidden = n_experts, expert_hidden
        self.top_k, self.activation = top_k, activation

    def build(self, d_in, d_out):
        return L.MoENet(d_in, d_out, self.n_experts, self.expert_hidden, self.top_k,
                        noisy=False, aux_coef=0.0, activation=self.activation)


@register("gated_moe")
class GatedMixtureOfExperts(BaseModel):
    """Sparsely-gated MoE: noisy top-k routing plus a load-balancing loss."""
    def __init__(self, n_experts=8, expert_hidden=(64,), top_k=2, aux_coef=0.01,
                 activation="relu", **kw):
        super().__init__(**kw)
        self.n_experts, self.expert_hidden, self.top_k = n_experts, expert_hidden, top_k
        self.aux_coef, self.activation = aux_coef, activation

    def build(self, d_in, d_out):
        return L.MoENet(d_in, d_out, self.n_experts, self.expert_hidden, self.top_k,
                        noisy=True, aux_coef=self.aux_coef, activation=self.activation)


@register("hypernet")
class Hypernetwork(BaseModel):
    """A hypernetwork generates a per-sample layer; an ordinary head reads it out."""
    def __init__(self, target_hidden=16, hyper_hidden=64, width=64, activation="relu", **kw):
        super().__init__(**kw)
        self.target_hidden, self.hyper_hidden = target_hidden, hyper_hidden
        self.width, self.activation = width, activation

    def build(self, d_in, d_out):
        return L.HyperNet(d_in, d_out, self.target_hidden, self.hyper_hidden, self.width, self.activation)


@register("implicit")
class ImplicitNetwork(BaseModel):
    """Implicit network: output defined by an equilibrium z = relu(Az + Bx)."""
    def __init__(self, state_dim=64, iterations=30, kappa=0.9, **kw):
        super().__init__(**kw)
        self.state_dim, self.iterations, self.kappa = state_dim, iterations, kappa

    def build(self, d_in, d_out):
        return L.ImplicitNet(d_in, d_out, self.state_dim, self.iterations, self.kappa)


@register("deq")
class DeepEquilibrium(BaseModel):
    """Deep Equilibrium Model with implicit-function-theorem backward pass."""
    def __init__(self, state_dim=64, max_iter=50, tol=1e-4, kappa=0.9, **kw):
        super().__init__(**kw)
        self.state_dim, self.max_iter, self.tol, self.kappa = state_dim, max_iter, tol, kappa

    def build(self, d_in, d_out):
        return L.DEQNet(d_in, d_out, self.state_dim, self.max_iter, self.tol, self.kappa)


@register("kan")
class KAN(BaseModel):
    """Kolmogorov-Arnold Network with B-spline edge functions."""
    def __init__(self, hidden=(16,), grid_size=5, spline_order=3, grid_range=(-3.0, 3.0), **kw):
        super().__init__(**kw)
        self.hidden, self.grid_size = hidden, grid_size
        self.spline_order, self.grid_range = spline_order, grid_range

    def build(self, d_in, d_out):
        return L.KANNet(d_in, d_out, self.hidden, self.grid_size, self.spline_order, self.grid_range)


@register("rbf")
class RBF(BaseModel):
    """Radial basis function network (centers/widths initialised from data, then trained)."""
    def __init__(self, n_centers=128, **kw):
        kw.setdefault("lr", 1e-2)
        kw.setdefault("epochs", 150)
        super().__init__(**kw)
        self.n_centers = n_centers

    def build(self, d_in, d_out):
        return L.RBFNet(d_in, d_out, self.n_centers)


@register("elm")
class ExtremeLearningMachine(BaseModel):
    """Random frozen hidden layer; readout solved by ridge regression (no epochs)."""
    def __init__(self, n_hidden=512, activation="tanh", ridge=1e-3, **kw):
        super().__init__(**kw)
        self.n_hidden, self.activation, self.ridge = n_hidden, activation, ridge

    def build(self, d_in, d_out):
        return L.ELMNet(d_in, d_out, self.n_hidden, self.activation)

    def _loss(self, out, y):  # ELM regresses one-hot targets in closed form
        if self.task == "classification":
            y = F.one_hot(y, out.shape[1]).float()
        return F.mse_loss(out, y)

    def _fit(self, Xt, yt, epochs, val):
        m = self._module
        T = F.one_hot(yt, m.out.out_features).float() if self.task == "classification" else yt
        with torch.no_grad():
            H = m.features(Xt).double()
            H = torch.cat([H, torch.ones(len(H), 1, dtype=H.dtype, device=H.device)], dim=1)
            A = H.T @ H + self.ridge * torch.eye(H.shape[1], dtype=H.dtype, device=H.device)
            beta = torch.linalg.solve(A, H.T @ T.double()).float()
            m.out.weight.copy_(beta[:-1].T)
            m.out.bias.copy_(beta[-1])
        self.history_["loss"].append(self._loss(self._outputs(Xt), yt).item())


@register("random_nn")
class RandomNeuralNetwork(BaseModel):
    """Gelenbe random neural network (spiking-queue neurons, positive weights)
    with a linear readout on the firing probabilities."""
    def __init__(self, hidden=(64, 64), **kw):
        kw.setdefault("lr", 1e-2)  # bounded rates give small gradients; needs a larger step
        kw.setdefault("epochs", 150)
        super().__init__(**kw)
        self.hidden = hidden

    def build(self, d_in, d_out):
        return L.RandomNeuralNet(d_in, d_out, self.hidden)
