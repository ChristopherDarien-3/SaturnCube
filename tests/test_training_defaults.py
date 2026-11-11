import numpy as np
import pytest
import torch

import saturncube as sc
import saturncube.evaluation as ev


def noisy(n=400, d=10, seed=0):
    """Labels are mostly noise, so a big network overfits and validation loss turns up quickly."""
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, d))
    return X, ((X[:, 0] + 0.3 * r.normal(size=n)) > 0).astype(int)


def test_patience_without_val_data_carves_a_split_and_stops_early():
    X, y = noisy()
    m = sc.create("mlp", hidden=(128, 128), epochs=200, patience=5, seed=0).fit(X, y)
    assert m.n_val_ == 40                                     # 10% of 400, held out automatically
    assert len(m.history_["loss"]) < 200                      # early stopping really ran
    assert len(m.history_["val_loss"]) == len(m.history_["loss"])
    assert m.stopped_epoch_ - m.best_epoch_ == 5              # stops exactly `patience` epochs after the best one


def test_default_trains_all_rows_for_all_epochs():
    X, y = noisy(200)
    m = sc.create("mlp", epochs=12, seed=0).fit(X, y)
    assert m.n_val_ == 0 and len(m.history_["loss"]) == 12 and m.history_["val_loss"] == []
    assert m.best_epoch_ is None and m.stopped_epoch_ == 11


def test_explicit_val_data_wins_and_best_weights_are_restored():
    X, y = noisy(300)
    Xv, yv = noisy(80, seed=1)
    m = sc.create("mlp", hidden=(128, 128), epochs=200, patience=4, seed=0).fit(X, y, val_data=(Xv, yv))
    assert m.n_val_ == 80                                     # the caller's validation set, nothing carved off
    now = m._loss(m._outputs(m._norm(m._as_x(Xv))), m._encode(yv)).item()
    assert now == pytest.approx(min(m.history_["val_loss"]), rel=1e-5)   # weights from the best epoch


def test_val_split_zero_warns_that_patience_does_nothing():
    X, y = noisy(100)
    with pytest.warns(UserWarning, match="patience has no effect"):
        m = sc.create("mlp", epochs=6, patience=2, val_split=0, seed=0).fit(X, y)
    assert m.n_val_ == 0 and len(m.history_["loss"]) == 6


def test_too_few_rows_warns_instead_of_silently_ignoring_patience():
    X, y = noisy(12)
    with pytest.warns(UserWarning, match="too few"):
        m = sc.create("mlp", epochs=4, patience=2, seed=0).fit(X, y)
    assert m.n_val_ == 0


def test_val_split_is_validated():
    with pytest.raises(ValueError, match="val_split"):
        sc.create("mlp", val_split=0.7)
    with pytest.raises(ValueError, match="val_split"):
        sc.create("mlp", val_split=-0.1)


def test_holdout_is_stratified_disjoint_and_keeps_every_class_in_training():
    m = sc.create("mlp", patience=3, val_split=0.2, seed=0)
    m.task = "classification"
    y = torch.tensor([0] * 90 + [1] * 8 + [2] * 2)
    X = torch.zeros(100, 3)
    X[:, 0] = torch.arange(100)  # row ids
    Xt, yt, (Xv, yv) = m._holdout(X, y)
    assert set(yt.tolist()) == {0, 1, 2}                      # the rare class 2 stays in training
    assert (yv == 0).sum() == 18 and (yv == 1).sum() == 2     # proportional to class size
    assert len(Xt) + len(Xv) == 100 and not set(Xt[:, 0].tolist()) & set(Xv[:, 0].tolist())


def test_holdout_follows_the_seed():
    def val_ids(seed):
        m = sc.create("mlp", patience=3, seed=seed)
        m.task = "regression"
        X = torch.arange(200.0).reshape(-1, 1)
        return sorted(m._holdout(X, torch.zeros(200))[2][0].flatten().tolist())
    assert val_ids(3) == val_ids(3) and val_ids(3) != val_ids(4)


def test_partial_fit_never_holds_rows_out():
    X, y = noisy(200)
    m = sc.create("mlp", patience=2, seed=0)
    m.partial_fit(X, y)
    assert m.n_val_ == 0 and len(m.history_["loss"]) == 1
    m.fit(X, y, epochs=3)  # and a normal fit afterwards still carves one
    assert m.n_val_ == 20


def test_regression_and_sequence_models_get_the_split_too():
    r = np.random.default_rng(0)
    X = r.normal(size=(200, 4))
    reg = sc.create("mlp", epochs=30, patience=3, seed=0).fit(X, X[:, 0] + 0.5 * r.normal(size=200))
    assert reg.n_val_ == 20
    toks = r.integers(1, 20, size=(200, 8))
    seq = sc.create("lstm", vocab_size=20, hidden=8, epochs=10, patience=2, seed=0).fit(toks, (toks == 5).any(1).astype(int))
    assert seq.n_val_ == 20


def test_closed_form_models_accept_patience_without_complaint(recwarn):
    X, y = noisy(100)
    sc.create("elm", patience=3, seed=0).fit(X, y)
    assert not [w for w in recwarn if "patience" in str(w.message)]


def test_val_split_survives_save_and_load(tmp_path):
    X, y = noisy(100)
    m = sc.create("mlp", epochs=3, patience=2, val_split=0.2, seed=0).fit(X, y)
    p = str(tmp_path / "m.pt")
    m.save(p)
    m2 = type(m).load(p)
    assert m2.val_split == 0.2 and m2.patience == 2


def test_warmup_is_idempotent_and_benchmark_calls_it(monkeypatch):
    import saturncube.base as base
    monkeypatch.setattr(base, "_WARMED", False)
    first = sc.warmup()
    assert isinstance(first, float) and first >= 0 and sc.warmup() == 0.0
    calls = []
    monkeypatch.setattr(ev, "warmup", lambda: calls.append(1))
    X, y = noisy(60)
    sc.benchmark(["naive_bayes"], X, y, seeds=1, baselines=False, show=False)
    assert calls == [1]


# ------------------------------------------------------------- patience="auto": 10 for regression ---
def noisy_regression(n=400, d=10, seed=0):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, d))
    return X, X[:, 0] + 1.5 * r.normal(size=n)  # mostly noise: a big network overfits quickly


def test_regression_stops_early_by_default_but_classification_does_not():
    X, y = noisy_regression()
    reg = sc.create("mlp", hidden=(128, 128), epochs=200, seed=0).fit(X, y)
    assert reg.patience == "auto" and reg.n_val_ == 40
    assert len(reg.history_["loss"]) < 200 and reg.stopped_epoch_ - reg.best_epoch_ == 10
    Xc, yc = noisy()
    clf = sc.create("mlp", epochs=15, seed=0).fit(Xc, yc)
    assert clf.n_val_ == 0 and len(clf.history_["loss"]) == 15


def test_patience_none_turns_the_regression_default_off():
    X, y = noisy_regression(200)
    m = sc.create("mlp", epochs=12, patience=None, seed=0).fit(X, y)
    assert m.n_val_ == 0 and len(m.history_["loss"]) == 12


def test_auto_stays_off_when_the_holdout_would_be_too_small_and_is_then_identical_to_off(recwarn):
    X, y = noisy_regression(150)  # 10% = 15 rows < 20: too noisy to stop on
    a = sc.create("mlp", epochs=10, seed=3).fit(X, y)
    b = sc.create("mlp", epochs=10, patience=None, seed=3).fit(X, y)
    assert a.n_val_ == 0 and len(a.history_["loss"]) == 10
    assert np.array_equal(a.predict(X), b.predict(X))  # no random draw was consumed by the skipped holdout
    assert not [w for w in recwarn if "patience" in str(w.message)]


def test_explicit_patience_is_honoured_even_on_small_data():
    X, y = noisy_regression(150)
    m = sc.create("mlp", hidden=(128, 128), epochs=100, patience=5, seed=0).fit(X, y)
    assert m.n_val_ == 15


def test_auto_never_warns_even_without_a_usable_holdout(recwarn):
    X, y = noisy_regression(300)
    sc.create("mlp", epochs=3, val_split=0, seed=0).fit(X, y)        # holdout switched off
    sc.create("mlp", epochs=3, seed=0).fit(X[:12], y[:12])           # far too few rows
    assert not [w for w in recwarn if "patience" in str(w.message)]


def test_patience_is_validated_and_survives_save_load(tmp_path):
    for bad in ("yes", 0, -3, True, 2.5):
        with pytest.raises(ValueError, match="patience"):
            sc.create("mlp", patience=bad)
    X, y = noisy_regression(100)
    m = sc.create("mlp", epochs=3, seed=0).fit(X, y)
    p = str(tmp_path / "m.pt")
    m.save(p)
    assert type(m).load(p).patience == "auto"


def test_partial_fit_on_regression_never_holds_rows_out():
    X, y = noisy_regression(400)
    m = sc.create("mlp", seed=0)
    m.partial_fit(X, y)
    assert m.n_val_ == 0 and len(m.history_["loss"]) == 1


def test_benchmark_regression_uses_its_own_validation_set_for_the_default():
    X, y = noisy_regression(400)
    seen = []

    class Spy(sc.create("mlp").__class__):
        def _fit(self, Xt, yt, epochs, val):
            seen.append(None if val is None else len(val[0]))
            return super()._fit(Xt, yt, epochs, val)

    sc.benchmark({"spy": lambda s: Spy(epochs=3, seed=s)}, X, y, seeds=1, baselines=False, show=False)
    assert seen[0] is not None and seen[0] > 20   # the harness validation rows, nothing extra carved from train
