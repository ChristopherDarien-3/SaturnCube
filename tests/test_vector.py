import numpy as np
import pytest

import saturncube as sc

rng = np.random.default_rng(0)
X = rng.normal(size=(600, 8)).astype(np.float32)
y_cls = ((X[:, 0] * X[:, 1] > 0).astype(int) + (X[:, 2] > 0.5).astype(int))  # 3 nonlinear classes
y_reg = (np.sin(X[:, 0]) + X[:, 1] * X[:, 2]).astype(np.float32)
Xtr, Xte, ctr, cte, rtr, rte = X[:450], X[450:], y_cls[:450], y_cls[450:], y_reg[:450], y_reg[450:]
ALL_VECTOR = sc.available("vector")
# alternative learning rules are tested separately (tests/test_rules.py) on an easier task: with default
# settings they are not reliable on this XOR-like problem
EXPERIMENTAL = {"contrastive_hebbian", "forward_gradient", "predictive_coding", "synthetic_gradients",
                "equilibrium_propagation", "local_error"}
NAMES = [n for n in ALL_VECTOR if n not in EXPERIMENTAL]
# local / random-feature models are weaker on this XOR-like 8-d task; chance is ~0.4
MIN_ACC = {"elm": 0.55, "rbf": 0.6, "fa": 0.55, "dfa": 0.55}


def test_registry_contents():
    assert len(ALL_VECTOR) >= 17 and {"fa", "dfa", "product_key_memory", "forward_gradient"} <= set(ALL_VECTOR)


@pytest.mark.parametrize("name", NAMES)
def test_classification(name):
    m = sc.create(name, seed=0)
    m.fit(Xtr, ctr)
    r = m.evaluate(Xte, cte)
    assert m.predict(Xte).shape == (150,)
    assert np.allclose(m.predict_proba(Xte).sum(1), 1, atol=1e-4)
    assert r["accuracy"] > MIN_ACC.get(name, 0.7), (name, r)  # chance is ~0.4


@pytest.mark.parametrize("name", NAMES)
def test_regression(name):
    m = sc.create(name, seed=0)
    m.fit(Xtr, rtr)
    r = m.evaluate(Xte, rte)
    assert m.predict(Xte).shape == (150,)
    assert r["r2"] > 0.2, (name, r)


@pytest.mark.parametrize("name", NAMES)
def test_save_load(name, tmp_path):
    m = sc.create(name, epochs=3, seed=0).fit(Xtr, ctr)
    m.save(tmp_path / "m.pt")
    m2 = type(m).load(tmp_path / "m.pt")
    assert np.allclose(m.predict_proba(Xte), m2.predict_proba(Xte), atol=1e-5)


def test_escape_hatch_and_explicit_dims():
    m = sc.MLP(input_dim=8, output_dim=3, hidden=(16,)).fit(Xtr, ctr, epochs=1)
    assert hasattr(m.module, "parameters")


# ---- v0.2: optimizers and alternative learning rules ---------------------------
from saturncube.optim import OPTIMIZERS


@pytest.mark.parametrize("opt", OPTIMIZERS)
def test_optimizers_train_an_mlp(opt):
    kw = {"batch_size": 450, "epochs": 30} if opt in ("lbfgs", "cg") else {}
    m = sc.MLP(optimizer=opt, seed=0, **kw).fit(Xtr, ctr)
    assert m.history_["loss"][-1] < m.history_["loss"][0]
    assert m.evaluate(Xte, cte)["accuracy"] > 0.6, opt


def test_unknown_optimizer():
    with pytest.raises(ValueError):
        sc.MLP(optimizer="nope").fit(Xtr, ctr, epochs=1)
