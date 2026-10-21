import numpy as np
import pytest

import saturncube as sc
from test_v03 import blobs

X, y = blobs(n=600, k=3, d=6, sep=2.5, seed=0)
Xtr, ytr, Xte, yte = X[:450], y[:450], X[450:], y[450:]

CASES = [("forward_gradient", dict(epochs=60, lr=1e-2), 0.9),
         ("local_error", {}, 0.9),
         ("synthetic_gradients", {}, 0.9),
         ("equilibrium_propagation", dict(epochs=40, lr=5e-3), 0.9),
         ("contrastive_hebbian", dict(epochs=40, lr=5e-3), 0.6),
         ("predictive_coding", dict(epochs=40), 0.9)]


@pytest.mark.parametrize("name,kw,min_acc", CASES)
def test_learning_rules_learn(name, kw, min_acc):
    m = sc.create(name, seed=0, **kw).fit(Xtr, ytr)
    assert m.evaluate(Xte, yte)["accuracy"] > min_acc, name


def test_forward_gradient_is_unbiased_in_direction():
    """The mean of many forward-gradient estimates should align with the true gradient."""
    import torch
    from torch.func import functional_call, jvp
    m = sc.ForwardGradient(hidden=(8,), seed=0).fit(Xtr, ytr, epochs=1)
    net, xb, yb = m.module, m._norm(m._as_x(Xtr[:64])), m._encode(ytr[:64])
    params = {n: p.detach() for n, p in net.named_parameters()}
    f = lambda ps: m._loss(functional_call(net, ps, (xb,)), yb)
    true = torch.autograd.grad(m._loss(net(xb), yb), list(net.parameters()))
    true = torch.cat([g.flatten() for g in true])
    est = torch.zeros_like(true)
    for _ in range(3000):
        v = {n: torch.randn_like(p) for n, p in params.items()}
        _, dl = jvp(f, (params,), (v,))
        est += torch.cat([(dl * v[n]).flatten() for n in params]) / 3000
    cos = torch.nn.functional.cosine_similarity(est, true, dim=0)
    assert cos > 0.7, float(cos)


def test_extra_optimizers_listed():
    assert {"cg", "coordinate_descent", "shampoo", "kfac", "natural_gradient"} <= set(sc.OPTIMIZERS)
