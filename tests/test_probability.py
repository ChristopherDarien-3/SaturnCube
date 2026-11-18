import warnings

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import saturncube as sc


def gaussian_classes(n, prior, seed, sep=1.6, d=3):
    """Three Gaussian classes with the given class frequencies."""
    r = np.random.default_rng(seed)
    y = r.choice(len(prior), size=n, p=prior)
    centers = np.eye(len(prior), d) * sep
    return centers[y] + r.normal(size=(n, d)), y


# ------------------------------------------------------------------------------------ metrics ---
def test_log_loss_and_brier_match_closed_forms():
    y = np.array([0, 1, 2, 3] * 25)
    uniform = np.full((100, 4), 0.25)
    assert sc.log_loss(y, uniform) == pytest.approx(np.log(4))
    assert sc.brier_score(y, uniform) == pytest.approx(3 / 4)              # (k-1)/k for a uniform guess
    perfect = np.eye(4)[y]
    assert sc.log_loss(y, perfect) == pytest.approx(0, abs=1e-12) and sc.brier_score(y, perfect) == 0
    wrong = np.eye(4)[(y + 1) % 4]
    assert sc.brier_score(y, wrong) == pytest.approx(2.0)                   # the worst possible value
    assert sc.log_loss(y, wrong) > 30                                        # clipped, finite
    assert np.isfinite(sc.log_loss(y, wrong))


def test_string_labels_need_classes_and_are_mapped_in_column_order():
    y = np.array(["b", "a", "c", "a"])
    p = np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.1, 0.1, 0.8], [0.6, 0.3, 0.1]])  # columns a, b, c
    assert sc.log_loss(y, p, classes=["a", "b", "c"]) == pytest.approx(-np.mean(np.log([0.8, 0.7, 0.8, 0.6])))
    with pytest.raises(ValueError, match="pass classes"):
        sc.log_loss(y, p)


def test_metric_inputs_are_validated():
    p = np.full((4, 3), 1 / 3)
    with pytest.raises(ValueError, match="4 labels but 5"):
        sc.log_loss(np.zeros(4, dtype=int), np.full((5, 3), 1 / 3))
    with pytest.raises(ValueError, match="NaN"):
        sc.brier_score(np.zeros(4, dtype=int), np.where(np.eye(4, 3) > 0, np.nan, p))
    with pytest.raises(ValueError, match="not in classes"):
        sc.log_loss(np.array([0, 1, 2, 9]), p, classes=[0, 1, 2])
    with pytest.raises(ValueError, match="0..k-1"):
        sc.log_loss(np.array([0, 1, 2, 5]), p)
    with pytest.raises(ValueError, match="shape"):
        sc.log_loss(np.zeros(3, dtype=int), np.zeros(3))


def test_ece_is_small_when_calibrated_and_large_when_overconfident():
    r = np.random.default_rng(0)
    n = 40000
    logits = r.normal(size=(n, 3)) * 1.2
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([r.choice(3, p=row) for row in p])                          # labels drawn from the probabilities
    over = np.exp(logits * 3) / np.exp(logits * 3).sum(1, keepdims=True)
    assert sc.expected_calibration_error(y, p) < 0.02
    assert sc.expected_calibration_error(y, over) > 0.15


def test_reliability_table_counts_bins_and_gaps():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1])
    p = np.array([[.9, .1]] * 4 + [[.25, .75]] * 4 + [[.55, .45], [.55, .45]])
    rows = sc.reliability_table(y, p, bins=5)
    assert sum(r["count"] for r in rows) == 10 and [r["low"] for r in rows] == sorted(r["low"] for r in rows)
    top = rows[-1]
    assert top["count"] == 4 and top["confidence"] == pytest.approx(0.9) and top["accuracy"] == 1.0
    assert top["gap"] == pytest.approx(top["accuracy"] - top["confidence"])
    assert sc.expected_calibration_error(y, p, bins=5) == pytest.approx(
        sum(r["count"] * abs(r["gap"]) for r in rows) / 10)
    text = sc.format_reliability(rows)
    assert "confidence bin" in text and len(text.splitlines()) == len(rows) + 1


# ----------------------------------------------------------------------------------- priors ---
def test_prior_shift_is_the_exact_bayes_correction():
    pi0, pi1 = np.array([.6, .3, .1]), np.array([.2, .3, .5])
    X, y = gaussian_classes(20000, pi1, seed=1)                              # deployment data has prior pi1
    mu = np.eye(3) * 1.6
    like = np.exp(-0.5 * ((X[:, None, :] - mu[None]) ** 2).sum(-1))
    post0 = like * pi0 / (like * pi0).sum(1, keepdims=True)                  # what a model trained under pi0 knows
    post1 = like * pi1 / (like * pi1).sum(1, keepdims=True)                  # the right answer under pi1
    fixed = sc.PriorShift(source_prior=pi0, target_prior=pi1).transform(post0)
    assert np.allclose(fixed, post1, atol=1e-9) and np.allclose(fixed.sum(1), 1)
    assert sc.log_loss(y, fixed) < sc.log_loss(y, post0) - 0.05


def test_prior_shift_estimates_priors_from_labels_with_smoothing():
    r = np.random.default_rng(0)
    y_src, y_tgt = r.choice(3, 50000, p=[.6, .3, .1]), r.choice(3, 50000, p=[.2, .3, .5])
    ps = sc.PriorShift().fit(y_src, y_tgt)
    assert np.allclose(ps.source_prior_, [.6, .3, .1], atol=0.01) and np.allclose(ps.target_prior_, [.2, .3, .5], atol=0.01)
    assert np.allclose(ps.ratio_, ps.target_prior_ / ps.source_prior_)
    same = sc.PriorShift().fit(y_src, y_src)
    p = r.dirichlet(np.ones(3), size=20)
    assert np.allclose(same.transform(p), p, atol=1e-9)                      # no shift, no change
    missing = sc.PriorShift().fit(np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0]), classes=[0, 1, 2])
    assert np.isfinite(missing.ratio_).all() and (missing.target_prior_ > 0).all()   # smoothing: no zero priors


def test_prior_shift_validates_input():
    with pytest.raises(ValueError, match="non-negative"):
        sc.PriorShift(source_prior=[-1, 2], target_prior=[1, 1])
    with pytest.raises(RuntimeError, match="fit"):
        sc.PriorShift().ratio_
    with pytest.raises(ValueError, match="columns"):
        sc.PriorShift(source_prior=[.5, .5], target_prior=[.2, .8]).transform(np.full((3, 3), 1 / 3))


def test_temperature_scaler_recovers_the_planted_temperature():
    r = np.random.default_rng(0)
    n = 6000
    y = r.integers(0, 3, n)
    logits = r.normal(size=(n, 3))
    logits[np.arange(n), y] += 1.0
    base = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    for planted in (3.0, 0.5):                                               # over- and under-confident
        z = logits * planted
        p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        ts = sc.TemperatureScaler().fit(p[:3000], y[:3000])
        assert ts.temperature_ == pytest.approx(planted, rel=0.15)
        q = ts.transform(p[3000:])
        assert np.array_equal(q.argmax(1), p[3000:].argmax(1))               # the predicted class never changes
        assert sc.log_loss(y[3000:], q) < sc.log_loss(y[3000:], p[3000:])
        assert sc.expected_calibration_error(y[3000:], q) < sc.expected_calibration_error(y[3000:], p[3000:])
    assert sc.TemperatureScaler().fit(base, y).temperature_ == pytest.approx(1.0, rel=0.25)  # already calibrated


# ------------------------------------------------------------------------------- class weights ---
def noisy_imbalanced(n=800, seed=0):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, 4))
    y = (X[:, 0] + 0.7 * r.normal(size=n) > 1.3).astype(int)               # about 15% positives
    return X, y


def test_balanced_weights_are_n_over_k_counts_and_enter_the_loss():
    X, y = noisy_imbalanced()
    m = sc.create("mlp", epochs=2, seed=0, class_weight="balanced").fit(X, y)
    counts = np.bincount(y)
    assert np.allclose(m.class_weight_, len(y) / (2 * counts))
    out, tgt = torch.randn(32, 2), torch.randint(0, 2, (32,))
    assert m._loss(out, tgt).item() == pytest.approx(F.cross_entropy(out, tgt, weight=m._cw).item())
    assert m._loss(out, tgt).item() != pytest.approx(F.cross_entropy(out, tgt).item())


def test_class_weights_trade_precision_for_minority_recall():
    X, y = noisy_imbalanced()
    plain = sc.create("mlp", epochs=40, seed=0).fit(X, y)
    heavy = sc.create("mlp", epochs=40, seed=0, class_weight="balanced").fit(X, y)
    recall = lambda m: float((m.predict(X)[y == 1] == 1).mean())
    assert recall(heavy) > recall(plain) + 0.1


def test_class_weight_forms_and_errors():
    X, y = noisy_imbalanced(200)
    d = sc.create("mlp", epochs=1, seed=0, class_weight={1: 5.0}).fit(X, y)
    assert d.class_weight_.tolist() == [1.0, 5.0]
    s = sc.create("mlp", epochs=1, seed=0, class_weight=[2.0, 1.0]).fit(X, y)
    assert s.class_weight_.tolist() == [2.0, 1.0]
    with pytest.raises(ValueError, match="not among the classes"):
        sc.create("mlp", class_weight={7: 1.0}).fit(X, y)
    with pytest.raises(ValueError, match="one weight per class"):
        sc.create("mlp", class_weight=[1.0, 2.0, 3.0]).fit(X, y)
    with pytest.raises(ValueError, match="non-negative"):
        sc.create("mlp", class_weight=[-1.0, 1.0]).fit(X, y)
    with pytest.raises(ValueError, match="class_weight"):
        sc.create("mlp", class_weight="heavy")
    with pytest.warns(UserWarning, match="only applies to classification"):
        sc.create("mlp", epochs=1, class_weight="balanced").fit(X, X[:, 0])


def test_models_with_their_own_loss_warn_and_weights_survive_save_load(tmp_path):
    X, y = noisy_imbalanced(200)
    with pytest.warns(UserWarning, match="own loss"):
        sc.create("elm", class_weight="balanced", seed=0).fit(X, y)
    m = sc.create("mlp", epochs=2, seed=0, class_weight="balanced").fit(X, y)
    p = str(tmp_path / "m.pt")
    m.save(p)
    m2 = type(m).load(p)
    assert np.allclose(m2.class_weight_, m.class_weight_) and m2._cw is not None


# ------------------------------------------------------------------------------- Calibrated ---
def shifted_problem():
    Xtr, ytr = gaussian_classes(3000, [.6, .3, .1], seed=0)
    Xva, yva = gaussian_classes(1200, [.2, .3, .5], seed=1)
    Xte, yte = gaussian_classes(3000, [.2, .3, .5], seed=2)
    return Xtr, ytr, Xva, yva, Xte, yte


def test_calibrated_beats_the_raw_model_under_prior_shift():
    Xtr, ytr, Xva, yva, Xte, yte = shifted_problem()
    raw = sc.create("mlp", epochs=40, seed=0).fit(Xtr, ytr)
    cal = sc.Calibrated("mlp", epochs=40, seed=0).fit(Xtr, ytr, val_data=(Xva, yva))
    ll_raw = sc.log_loss(yte, raw.predict_proba(Xte), raw.classes_)
    ll_cal = cal.evaluate(Xte, yte)["log_loss"]
    assert ll_cal < ll_raw - 0.03
    p = cal.predict_proba(Xte)
    assert p.shape == (3000, 3) and np.allclose(p.sum(1), 1, atol=1e-6)
    assert set(cal.evaluate(Xte, yte)) == {"accuracy", "log_loss", "brier", "ece"}
    assert cal.predict(Xte).shape == (3000,) and list(cal.classes_) == [0, 1, 2]
    assert cal.prior_shift_.target_prior_[2] > cal.prior_shift_.source_prior_[2]   # the shift it found: more class 2


def test_calibrated_switches_and_validation():
    Xtr, ytr, Xva, yva, Xte, _ = shifted_problem()
    off = sc.Calibrated("mlp", prior=False, temperature=False, epochs=5, seed=0).fit(Xtr, ytr, val_data=(Xva, yva))
    assert off.prior_shift_ is None and off.scaler_ is None
    assert np.allclose(off.predict_proba(Xte), off.model.predict_proba(Xte))
    held = sc.Calibrated("mlp", epochs=5, seed=0, val_split=0.25).fit(Xtr, ytr)       # no val_data: holds rows out
    assert held.scaler_ is not None
    with pytest.raises(ValueError, match="classification"):
        sc.Calibrated("mlp").fit(Xtr, Xtr[:, 0])
    with pytest.raises(RuntimeError, match="fit"):
        sc.Calibrated("mlp").predict_proba(Xte)
    with pytest.raises(TypeError, match="registry name or a class"):
        sc.Calibrated(sc.create("mlp"))


def test_calibrated_accounts_for_class_weights_in_the_effective_training_prior():
    Xtr, ytr, Xva, yva, _, _ = shifted_problem()
    cal = sc.Calibrated("mlp", epochs=5, seed=0, class_weight=[1.0, 1.0, 4.0]).fit(Xtr, ytr, val_data=(Xva, yva))
    prior = np.bincount(ytr) + 1.0
    prior = prior / prior.sum()
    eff = prior * np.array([1.0, 1.0, 4.0])
    assert np.allclose(cal.prior_shift_.source_prior_, eff / eff.sum())


def test_calibrated_save_load_roundtrip(tmp_path):
    Xtr, ytr, Xva, yva, Xte, _ = shifted_problem()
    cal = sc.Calibrated("mlp", epochs=5, seed=0).fit(Xtr, ytr, val_data=(Xva, yva))
    p = str(tmp_path / "cal.pt")
    cal.save(p)
    back = sc.Calibrated.load(p)
    assert np.allclose(back.predict_proba(Xte), cal.predict_proba(Xte), atol=1e-5)


# -------------------------------------------------------------------- benchmark integration ---
def test_benchmark_reports_probability_metrics_and_a_proper_prior_baseline():
    Xtr, ytr = gaussian_classes(900, [.5, .3, .2], seed=3)
    res = sc.benchmark([("mlp", {"epochs": 20})], Xtr, ytr, seeds=2, show=False)
    for label in ("mlp", "class prior", "scaled logistic regression"):
        for m in ("log_loss", "brier", "ece"):
            assert np.isfinite(res.scores[label][m]).all(), (label, m)
    # the class-prior baseline's probabilities are the training frequencies, so its log loss is near the label entropy
    entropy = -(np.array([.5, .3, .2]) * np.log([.5, .3, .2])).sum()
    assert res.scores["class prior"]["log_loss"].mean() == pytest.approx(entropy, abs=0.08)
    assert res.scores["mlp"]["log_loss"].mean() < res.scores["class prior"]["log_loss"].mean()
    assert "log_loss" in str(res) and "result.reliability(" in str(res)


def test_benchmark_prior_baseline_probabilities_match_a_manual_computation():
    r = np.random.default_rng(0)
    y = r.choice(3, 500, p=[.5, .3, .2])
    X = r.normal(size=(500, 3))
    res = sc.benchmark(["naive_bayes"], X, y, seeds=3, show=False)
    for i, seed in enumerate(res.seeds):
        s = res.splits[seed]
        prior = np.bincount(y[s["train"]], minlength=3) / len(s["train"])
        manual = -np.mean(np.log(prior[y[s["test"]]]))
        assert res.scores["class prior"]["log_loss"][i] == pytest.approx(manual)


def test_log_loss_ranking_is_lower_is_better_and_models_without_proba_score_na():
    r = np.random.default_rng(4)
    Xtr = r.normal(size=(800, 2))
    ytr = (Xtr[:, 0] * Xtr[:, 1] > 0).astype(int)   # XOR: a linear baseline cannot solve it, an MLP can

    class Hard:                                   # a classifier with predict only
        def fit(self, X, y):
            self.v = np.bincount(y).argmax()
            return self

        def predict(self, X):
            return np.full(len(X), self.v)

    res = sc.benchmark({"hard": lambda s: Hard(), "mlp": ("mlp", {"epochs": 60})}, Xtr, ytr, seeds=2,
                       metric="log_loss", show=False)
    assert np.isnan(res.scores["hard"]["log_loss"]).all() and np.isfinite(res.scores["hard"]["accuracy"]).all()
    rows = res.to_rows()
    finite = [x["log_loss_mean"] for x in rows if np.isfinite(x["log_loss_mean"])]
    assert finite == sorted(finite) and rows[-1]["model"] == "hard"
    mlp = next(x for x in rows if x["model"] == "mlp")
    assert mlp["delta_mean"] > 0                   # positive always means better than the best baseline


def test_reliability_pools_runs_and_needs_stored_probabilities():
    Xtr, ytr = gaussian_classes(800, [.4, .3, .3], seed=5)
    res = sc.benchmark([("mlp", {"epochs": 15})], Xtr, ytr, seeds=3, show=False)
    rows = res.reliability("mlp", bins=5)
    n_test = sum(len(res.splits[s]["test"]) for s in res.seeds)
    assert sum(r["count"] for r in rows) == n_test
    assert "confidence bin" in sc.format_reliability(rows)
    off = sc.benchmark([("mlp", {"epochs": 5})], Xtr, ytr, seeds=1, keep_predictions=False, show=False)
    with pytest.raises(ValueError, match="no stored probabilities"):
        off.reliability("mlp")
    reg = sc.benchmark(["naive_bayes"], Xtr, Xtr[:, 0], seeds=1, baselines=False, show=False)
    assert reg.predictions["naive_bayes"] == []


def test_calibrated_runs_inside_the_benchmark_with_the_harness_validation_rows():
    Xtr, ytr = gaussian_classes(900, [.6, .3, .1], seed=6)
    res = sc.benchmark({"cal": ("calibrated", {"base": "mlp", "epochs": 15}), "raw": ("mlp", {"epochs": 15})},
                       Xtr, ytr, seeds=2, show=False)
    assert not res.errors["cal"] and np.isfinite(res.scores["cal"]["log_loss"]).all()
