import numpy as np
import pytest

import saturncube as sc


class Stub:
    """A deterministic fake classifier: its probabilities depend only on its seed."""
    task = "classification"
    base = np.array([0.6, 0.3, 0.1])

    def __init__(self, seed=None, **kw):
        self.seed, self.fit_kwargs = seed, None

    def fit(self, X, y, epochs=None, val_data=None):
        self.classes_ = np.unique(y)
        self.fit_kwargs = {"epochs": epochs, "val_data": val_data}
        return self

    def vector(self):
        return np.roll(self.base, self.seed % 3)

    def predict_proba(self, X):
        return np.tile(self.vector(), (len(X), 1))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class StubPlain(Stub):
    def fit(self, X, y):                      # a fit() with no epochs / val_data parameters
        self.classes_ = np.unique(y)
        return self


class RegStub:
    task = "regression"

    def __init__(self, seed=None, **kw):
        self.seed = seed

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.asarray(X)[:, 0] + (self.seed % 5)


def data(n=60, seed=0):
    r = np.random.default_rng(seed)
    return r.normal(size=(n, 3)), np.arange(n) % 3


# ------------------------------------------------------------------------------ exact formulas ---
def test_arithmetic_average_and_member_arrays():
    X, y = data()
    ens = sc.SeedEnsemble(Stub, n=6, seed=1).fit(X, y)
    vecs = np.stack([np.roll(Stub.base, s % 3) for s in ens.member_seeds_])
    P = ens.member_proba(X)
    assert P.shape == (6, 60, 3) and np.allclose(P[:, 0, :], vecs)
    assert np.allclose(ens.predict_proba(X), np.tile(vecs.mean(0), (60, 1))) and np.allclose(ens.predict_proba(X).sum(1), 1)
    assert np.array_equal(ens.predict(X), np.full(60, ens.classes_[vecs.mean(0).argmax()]))
    assert ens.member_predict(X).shape == (6, 60) and list(ens.classes_) == [0, 1, 2]


def test_geometric_average_is_normalised_and_differs():
    X, y = data()
    arith = sc.SeedEnsemble(Stub, n=6, seed=1).fit(X, y)
    geo = sc.SeedEnsemble(Stub, n=6, seed=1, average="geometric").fit(X, y)
    vecs = np.stack([np.roll(Stub.base, s % 3) for s in arith.member_seeds_])
    expected = np.exp(np.log(vecs).mean(0))
    expected /= expected.sum()
    assert np.allclose(geo.predict_proba(X)[0], expected) and np.allclose(geo.predict_proba(X).sum(1), 1)
    assert not np.allclose(geo.predict_proba(X), arith.predict_proba(X))


def test_disagreement_formula_and_zero_when_members_agree():
    X, y = data()
    ens = sc.SeedEnsemble(Stub, n=6, seed=1).fit(X, y)
    vecs = np.stack([np.roll(Stub.base, s % 3) for s in ens.member_seeds_])
    expected = np.sqrt(vecs.var(axis=0).mean())                               # RMS over classes of the member spread
    d = ens.disagreement(X)
    assert d.shape == (60,) and np.allclose(d, expected) and expected > 0
    assert np.allclose(sc.SeedEnsemble(Stub, n=1).fit(X, y).disagreement(X), 0)

    class Same(Stub):
        def vector(self):
            return self.base
    assert np.allclose(sc.SeedEnsemble(Same, n=4).fit(X, y).disagreement(X), 0)


def test_regression_mean_disagreement_and_evaluate():
    X = np.random.default_rng(0).normal(size=(40, 2))
    y = X[:, 0] + 2.0
    ens = sc.SeedEnsemble(RegStub, n=5, seed=3).fit(X, y)
    offsets = np.array([s % 5 for s in ens.member_seeds_], dtype=float)
    assert np.allclose(ens.predict(X), X[:, 0] + offsets.mean())
    assert np.allclose(ens.disagreement(X), offsets.std())                      # std across members, constant here
    ev = ens.evaluate(X, y)
    assert set(ev) == {"rmse", "mae", "r2"} and ev["rmse"] == pytest.approx(abs(offsets.mean() - 2.0))
    with pytest.raises(AttributeError, match="classification"):
        ens.predict_proba(X)


def test_val_data_and_epochs_reach_only_the_members_that_accept_them():
    X, y = data()
    val = (X[:10], y[:10])
    ens = sc.SeedEnsemble(Stub, n=3).fit(X, y, epochs=7, val_data=val)
    assert all(m.fit_kwargs["epochs"] == 7 and m.fit_kwargs["val_data"] is val for m in ens.members_)
    sc.SeedEnsemble(StubPlain, n=2).fit(X, y, epochs=7, val_data=val)           # fit(X, y) only: nothing forwarded, no error


def test_member_seeds_are_independent_reproducible_and_depend_on_the_base_seed():
    a, b = sc.SeedEnsemble(Stub, n=8, seed=0), sc.SeedEnsemble(Stub, n=8, seed=0)
    c = sc.SeedEnsemble(Stub, n=8, seed=1)
    assert a.member_seeds_ == b.member_seeds_ and a.member_seeds_ != c.member_seeds_
    assert len(set(a.member_seeds_)) == 8 and all(0 <= s < 2 ** 31 - 1 for s in a.member_seeds_)


def test_constructor_and_state_validation():
    with pytest.raises(ValueError, match="positive integer"):
        sc.SeedEnsemble("mlp", n=0)
    with pytest.raises(ValueError, match="average"):
        sc.SeedEnsemble("mlp", average="median")
    with pytest.raises(ValueError, match="takes no seed"):
        sc.SeedEnsemble("naive_bayes")                                          # deterministic: members would be identical
    with pytest.raises(TypeError, match="registry name or a class"):
        sc.SeedEnsemble(sc.create("mlp"))
    with pytest.raises(ValueError, match="did you mean 'mlp'"):
        sc.SeedEnsemble("mlpp")
    with pytest.raises(RuntimeError, match="fit"):
        sc.SeedEnsemble("mlp").predict(np.zeros((2, 2)))


# ------------------------------------------------------------------------------- real models ---
def signal(n=700, seed=0):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, 4))
    return X, ((X[:, 0] + X[:, 1] * X[:, 2] + 0.8 * r.normal(size=n)) > 0).astype(int)


def test_averaged_probabilities_never_lose_to_the_average_member_in_log_loss():
    for k in range(3):                                                           # Jensen's inequality, on real networks
        X, y = signal(seed=k)
        ens = sc.SeedEnsemble("mlp", n=4, seed=k, epochs=25).fit(X[:450], y[:450])
        members = [sc.log_loss(y[450:], p, ens.classes_) for p in ens.member_proba(X[450:])]
        assert sc.log_loss(y[450:], ens.predict_proba(X[450:]), ens.classes_) <= np.mean(members) + 1e-12
        assert len(set(np.round(members, 6))) > 1                                # the members really differ


def test_reproducible_and_one_member_equals_the_plain_model():
    X, y = signal(300)
    a = sc.SeedEnsemble("mlp", n=3, seed=5, epochs=10).fit(X, y).predict_proba(X)
    b = sc.SeedEnsemble("mlp", n=3, seed=5, epochs=10).fit(X, y).predict_proba(X)
    c = sc.SeedEnsemble("mlp", n=3, seed=6, epochs=10).fit(X, y).predict_proba(X)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    one = sc.SeedEnsemble("mlp", n=1, seed=5, epochs=10)
    solo = sc.create("mlp", seed=one.member_seeds_[0], epochs=10).fit(X, y)
    assert np.allclose(one.fit(X, y).predict_proba(X), solo.predict_proba(X))


def test_early_stopping_runs_per_member_with_val_data():
    X, y = signal(500)
    ens = sc.SeedEnsemble("mlp", n=2, seed=0, hidden=(128, 128), epochs=200, patience=4).fit(X[:300], y[:300], val_data=(X[300:], y[300:]))
    assert all(m.n_val_ == 200 and len(m.history_["loss"]) < 200 for m in ens.members_)


def test_save_load_roundtrip_and_models_without_save(tmp_path):
    X, y = signal(300)
    ens = sc.SeedEnsemble("mlp", n=3, seed=2, epochs=8).fit(X, y)
    p = str(tmp_path / "ens.pt")
    ens.save(p)
    back = sc.SeedEnsemble.load(p)
    assert np.allclose(back.predict_proba(X), ens.predict_proba(X), atol=1e-5) and back.n == 3
    with pytest.raises(NotImplementedError, match="save"):
        sc.SeedEnsemble("random_forest", n=2).fit(X, y).save(str(tmp_path / "rf.pt"))


def test_regression_with_real_models():
    r = np.random.default_rng(0)
    X = r.normal(size=(400, 3))
    y = 3 * X[:, 0] + 0.2 * r.normal(size=400)
    ens = sc.SeedEnsemble("mlp", n=3, seed=0, epochs=40).fit(X[:300], (y[:300] - y[:300].mean()) / y[:300].std())
    assert ens.task == "regression" and ens.predict(X[300:]).shape == (100,) and (ens.disagreement(X[300:]) >= 0).all()


# ---------------------------------------------------------------------------------- the harness ---
def test_ensemble_is_a_drop_in_model_for_the_benchmark_and_follows_the_run_seed():
    X, y = signal(500)
    res = sc.benchmark({"ens": ("seed_ensemble", {"model": "mlp", "n": 3, "epochs": 15}), "mlp": ("mlp", {"epochs": 15})},
                       X, y, seeds=2, show=False)
    assert not res.errors["ens"] and np.isfinite(res.scores["ens"]["log_loss"]).all()
    a = sc.create("seed_ensemble", model="mlp", n=3, seed=0).member_seeds_
    b = sc.create("seed_ensemble", model="mlp", n=3, seed=1).member_seeds_
    assert a != b and "seed_ensemble" in sc.available("wrapper") and "calibrated" in sc.available("wrapper")


def test_calibrated_can_wrap_a_seed_ensemble():
    r = np.random.default_rng(0)
    X = r.normal(size=(900, 3))
    y = np.where(X[:, 0] + 0.5 * r.normal(size=900) > 0, 1, 0)
    cal = sc.Calibrated("seed_ensemble", model="mlp", n=2, epochs=10).fit(X[:600], y[:600], val_data=(X[600:800], y[600:800]))
    assert np.allclose(cal.predict_proba(X[800:]).sum(1), 1) and set(cal.evaluate(X[800:], y[800:])) >= {"log_loss", "ece"}
