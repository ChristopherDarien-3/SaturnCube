import numpy as np
import pytest

import saturncube as sc

rng = np.random.default_rng(0)
X = rng.normal(size=(600, 8)).astype(np.float32)
y_cls = ((X[:, 0] * X[:, 1] > 0).astype(int) + (X[:, 2] > 0.5).astype(int))  # 3 nonlinear classes
y_reg = (np.sin(X[:, 0]) + X[:, 1] * X[:, 2]).astype(np.float32)
Xtr, Xte, ctr, cte, rtr, rte = X[:450], X[450:], y_cls[:450], y_cls[450:], y_reg[:450], y_reg[450:]
NAMES = sc.available("vector")
# local / random-feature models are weaker on this XOR-like 8-d task; chance is ~0.4
MIN_ACC = {"elm": 0.55, "rbf": 0.6}


def test_all_fifteen_registered():
    assert len(NAMES) == 15


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
