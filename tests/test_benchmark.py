import numpy as np
import pytest

import saturncube as sc
import saturncube.evaluation as ev


class Spy:
    """Minimal model that records what fit() was given and predicts a constant (in model space)."""

    def __init__(self, log=None, value=None):
        self.log = log if log is not None else []
        self.value = value

    def fit(self, X, y, epochs=None, val_data=None):
        self.log.append({"X": np.array(X), "y": np.array(y), "epochs": epochs,
                         "val": None if val_data is None else (np.array(val_data[0]), np.array(val_data[1]))})
        self.fallback = y[0] if self.value is None else self.value
        return self

    def predict(self, X):
        return np.full(len(X), self.fallback)


def with_row_ids(n, d=3, seed=0):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, d))
    X[:, 0] = np.arange(n)  # column 0 identifies the row
    return X


def clustered(G=40, per=25, d=6, seed=0):
    """Each group has its own random centre; the label is unrelated to the centre (pure memorisation bait)."""
    r = np.random.default_rng(seed)
    centers = r.normal(size=(G, d)) * 3
    X = np.vstack([centers[g] + 0.1 * r.normal(size=(per, d)) for g in range(G)])
    return X, np.repeat(np.arange(G) % 2, per), np.repeat(np.arange(G), per)


# ------------------------------------------------------------------------------- splits ---
def test_group_splits_are_disjoint_and_cover_all_rows():
    r = np.random.default_rng(1)
    sizes = r.integers(3, 30, size=30)
    groups = np.repeat(np.arange(30), sizes)
    n = len(groups)
    y = (groups % 2).astype(int)
    X = with_row_ids(n)
    res = sc.benchmark({"spy": lambda seed: Spy()}, X, y, groups=groups, seeds=6, show=False)
    for seed, s in res.splits.items():
        tr, va, te = s["train"], s["val"], s["test"]
        assert len(set(tr) | set(va) | set(te)) == n and len(tr) + len(va) + len(te) == n
        gs = [set(groups[i]) for i in (tr, va, te)]
        assert not (gs[0] & gs[1]) and not (gs[0] & gs[2]) and not (gs[1] & gs[2])
        assert all(len(g) for g in gs)


def test_models_never_see_test_rows_and_validation_is_separate():
    X, y, groups = clustered(G=20, per=10)
    X[:, 0] = np.arange(len(X))
    log = []
    res = sc.benchmark({"spy": lambda seed: Spy(log)}, X, y, groups=groups, seeds=3, show=False)
    assert len(log) == 3
    for seed, call in zip(res.seeds, log):
        s = res.splits[seed]
        train_ids, val_ids = set(call["X"][:, 0].astype(int)), set(call["val"][0][:, 0].astype(int))
        assert train_ids == set(s["train"]) and val_ids == set(s["val"])
        assert not (train_ids & set(s["test"])) and not (val_ids & set(s["test"])) and not (train_ids & val_ids)


def test_all_models_share_identical_splits_per_seed():
    X, y, groups = clustered(G=20, per=10)
    X[:, 0] = np.arange(len(X))
    la, lb = [], []
    sc.benchmark({"a": lambda s: Spy(la), "b": lambda s: Spy(lb)}, X, y, groups=groups, seeds=3, show=False)
    for ca, cb in zip(la, lb):
        assert np.array_equal(ca["X"], cb["X"])


def test_splits_are_reproducible_and_change_with_seed():
    X, y, groups = clustered(G=20, per=10)
    a = sc.benchmark([lambda s: Spy()], X, y, groups=groups, seeds=[0, 1], baselines=False, show=False)
    b = sc.benchmark([lambda s: Spy()], X, y, groups=groups, seeds=[0, 1], baselines=False, show=False)
    assert np.array_equal(a.splits[0]["test"], b.splits[0]["test"])
    assert not np.array_equal(a.splits[0]["test"], a.splits[1]["test"])


def test_every_class_stays_in_training_groups():
    # class "c" lives in exactly one group, so only some shuffles keep it in train
    labels = np.array(["a", "b"] * 5 + ["c"])
    groups = np.repeat(np.arange(11), 8)
    y = np.repeat(labels, 8)
    X = with_row_ids(len(y))
    res = sc.benchmark({"spy": lambda seed: Spy()}, X, y, groups=groups, seeds=8, show=False)
    for s in res.splits.values():
        assert set(y[s["train"]]) == {"a", "b", "c"}


def test_ungrouped_split_is_stratified():
    y = np.array([0] * 180 + [1] * 20)
    X = with_row_ids(200)
    res = sc.benchmark({"spy": lambda seed: Spy()}, X, y, seeds=5, show=False)
    assert all((y[s["test"]] == 1).sum() >= 3 for s in res.splits.values())


# ---------------------------------------------------------------------------- leakage lesson ---
def test_random_split_inflates_scores_and_groups_expose_it():
    X, y, groups = clustered()
    grouped = sc.benchmark(["random_forest"], X, y, groups=groups, seeds=3, baselines=False, show=False)
    random_rows = sc.benchmark(["random_forest"], X, y, seeds=3, baselines=False, show=False)
    g = grouped.scores["random_forest"]["accuracy"].mean()
    r = random_rows.scores["random_forest"]["accuracy"].mean()
    assert r > 0.95, r      # memorises the clusters
    assert g < 0.8, g       # labels are unrelated to anything a new group can reveal
    assert "NOT grouped" in str(random_rows) and "group-disjoint" in str(grouped)


# ---------------------------------------------------------------------------------- baselines ---
def test_majority_baseline_equals_train_majority_share_on_test():
    r = np.random.default_rng(2)
    y = (r.random(400) > 0.7).astype(int)
    X = r.normal(size=(400, 4))
    res = sc.benchmark({"spy": lambda s: Spy()}, X, y, seeds=4, show=False)
    for i, seed in enumerate(res.seeds):
        s = res.splits[seed]
        major = np.bincount(y[s["train"]]).argmax()
        assert res.scores["majority class"]["accuracy"][i] == pytest.approx(np.mean(y[s["test"]] == major))
    assert res.baselines == ["majority class", "scaled logistic regression"]


def test_baselines_false_adds_nothing():
    X, y, groups = clustered(G=12, per=10)
    res = sc.benchmark({"spy": lambda s: Spy()}, X, y, groups=groups, seeds=2, baselines=False, show=False)
    assert res.labels == ["spy"] and res.baselines == []
    assert "vs best baseline" not in str(res)


def test_sklearn_baseline_is_scaled():
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression
    r = np.random.default_rng(3)
    n = 600
    signal = r.normal(size=n)
    X = np.column_stack([signal * 1e-4, r.normal(size=n) * 1e4])  # informative feature is tiny, noise is huge
    y = (signal > 0).astype(int)
    res = sc.benchmark({"spy": lambda s: Spy()}, X, y, seeds=3, show=False)
    scaled = res.scores["scaled logistic regression"]["accuracy"].mean()
    unscaled = LogisticRegression(max_iter=2000).fit(X[:450], y[:450]).score(X[450:], y[450:])
    assert scaled > 0.95
    assert unscaled < 0.9  # the data really does punish an unscaled baseline


def test_missing_sklearn_is_reported_not_fatal(monkeypatch):
    monkeypatch.setattr(ev, "_sklearn_baseline", lambda task: None)
    X, y, groups = clustered(G=12, per=10)
    res = sc.benchmark({"spy": lambda s: Spy()}, X, y, groups=groups, seeds=2, show=False)
    assert res.baselines == ["majority class"]
    assert "scikit-learn is not installed" in str(res)


# ---------------------------------------------------------------------------------- regression ---
def test_regression_scores_on_original_scale_with_scaled_target():
    r = np.random.default_rng(4)
    X = r.normal(size=(500, 4))
    y = 1000 * (X @ np.array([1.0, -2.0, 0.5, 0.0])) + 5000 + 10 * r.normal(size=500)
    log = []
    # predicting 0 in the model's (standardised) space must equal the train mean on the original scale
    res = sc.benchmark({"zero": lambda s: Spy(log, value=0.0)}, X, y, seeds=3, show=False)
    assert res.task == "regression"
    for call in log:
        assert abs(call["y"].mean()) < 1e-9 and call["y"].std() == pytest.approx(1.0)
    assert res.scores["zero"]["rmse"] == pytest.approx(res.scores["train mean"]["rmse"])
    assert res.scores["scaled ridge regression"]["r2"].mean() > 0.99


def test_scale_target_false_passes_raw_target():
    r = np.random.default_rng(5)
    X, y = r.normal(size=(200, 3)), 500 + 100 * r.normal(size=200)
    log = []
    sc.benchmark({"s": lambda s: Spy(log)}, X, y, seeds=1, scale_target=False, baselines=False, show=False)
    assert log[0]["y"].mean() > 400


def test_real_model_regression_beats_mean_baseline():
    r = np.random.default_rng(6)
    X = r.normal(size=(300, 3))
    y = 2000 * X[:, 0] - 1000 * X[:, 1] + 7000
    res = sc.benchmark([("mlp", {"epochs": 80})], X, y, seeds=2, show=False)
    assert res.scores["mlp"]["r2"].mean() > 0.8
    assert res.scores["train mean"]["r2"].mean() < 0.05


def test_lower_is_better_metric_sorts_and_signs_delta_correctly():
    r = np.random.default_rng(7)
    X = r.normal(size=(300, 2))
    y = 3 * X[:, 0]

    class Oracle:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return 3 * X[:, 0]

    res = sc.benchmark({"oracle": lambda s: Oracle(), "zero": lambda s: Spy(value=0.0)}, X, y, seeds=3,
                       metric="rmse", scale_target=False, show=False)
    rows = res.to_rows()
    assert rows[0]["model"] == "oracle" and rows[0]["rmse_mean"] == pytest.approx(0, abs=1e-12)
    means = [x["rmse_mean"] for x in rows]
    assert means == sorted(means)
    assert rows[0]["delta_mean"] > 0  # positive always means better than the best baseline


# ------------------------------------------------------------------------- robustness / API ---
def test_failing_model_is_recorded_and_others_still_run():
    X, y, groups = clustered(G=12, per=10)

    class Broken:
        def fit(self, X, y):
            raise RuntimeError("boom")

        def predict(self, X):
            return None

    res = sc.benchmark({"broken": lambda s: Broken(), "naive_bayes": "naive_bayes"}, X, y, groups=groups,
                       seeds=3, show=False)
    assert len(res.errors["broken"]) == 3 and "boom" in res.errors["broken"][0][1]
    assert np.isfinite(res.scores["naive_bayes"]["accuracy"]).all()
    assert "broken failed on 3/3 seeds" in str(res)
    assert res.to_rows()[-1]["model"] == "broken"  # failed models sort last


def test_misspelled_model_name_raises_with_suggestion():
    X, y, _ = clustered(G=12, per=10)
    with pytest.raises(ValueError, match="did you mean 'random_forest'"):
        sc.benchmark(["random_forrest"], X, y, show=False)


def test_invalid_inputs_raise_clear_errors():
    X, y, groups = clustered(G=12, per=10)
    bad = X.copy(); bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        sc.benchmark(["naive_bayes"], bad, y, show=False)
    with pytest.raises(ValueError, match="groups must be 1-D"):
        sc.benchmark(["naive_bayes"], X, y, groups=groups[:-1], show=False)
    with pytest.raises(ValueError, match="too few"):
        sc.benchmark(["naive_bayes"], X, y, groups=np.repeat([0, 1], len(y) // 2), show=False)
    with pytest.raises(ValueError, match="metric"):
        sc.benchmark(["naive_bayes"], X, y, metric="r2", show=False)
    with pytest.raises(ValueError, match="numeric"):
        sc.benchmark(["naive_bayes"], X.astype(str), y, show=False)
    with pytest.raises(TypeError, match="fresh model"):
        sc.benchmark([sc.create("naive_bayes")], X, y, show=False)


def test_seeds_are_passed_to_models_and_lists_are_accepted():
    X, y, groups = clustered(G=12, per=10)
    got = []

    def make(seed):
        got.append(seed)
        return Spy()

    res = sc.benchmark({"m": make}, X, y, groups=groups, seeds=[3, 7], baselines=False, show=False)
    assert got == [3, 7] and res.seeds == [3, 7]
    assert len(sc.benchmark({"m": make}, X, y, groups=groups, seeds=5, baselines=False, show=False).seeds) == 5


def test_epochs_forwarded_only_to_models_that_accept_it():
    X, y, groups = clustered(G=12, per=10)

    class NoEpochs:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.full(len(X), y_first)

    y_first = y[0]
    log = []
    res = sc.benchmark({"spy": lambda s: Spy(log), "plain": lambda s: NoEpochs()}, X, y, groups=groups,
                       seeds=2, epochs=5, show=False)
    assert all(c["epochs"] == 5 for c in log) and not res.errors["plain"]


def test_task_override_and_float_labels():
    r = np.random.default_rng(8)
    X = r.normal(size=(200, 3))
    y = (X[:, 0] > 0).astype(float)
    res = sc.benchmark({"spy": lambda s: Spy()}, X, y, seeds=2, task="classification", show=False)
    assert res.task == "classification" and "accuracy" in res.scores["spy"]
    assert 0.3 < res.scores["majority class"]["accuracy"].mean() < 0.7


def test_table_output_and_show_flag(capsys):
    X, y, groups = clustered(G=12, per=10)
    sc.benchmark(["naive_bayes"], X, y, groups=groups, seeds=3, show=False)
    assert capsys.readouterr().out == ""
    res = sc.benchmark(["naive_bayes"], X, y, groups=groups, seeds=3)
    out = capsys.readouterr().out
    assert out.strip() == str(res).strip() and "±" in out and "(baseline)" in out and "naive_bayes" in out


def test_single_seed_has_no_std_and_dataframe_export():
    X, y, groups = clustered(G=12, per=10)
    res = sc.benchmark(["naive_bayes"], X, y, groups=groups, seeds=1, show=False)
    assert "±" not in str(res)
    pd = pytest.importorskip("pandas")
    df = res.to_dataframe()
    assert isinstance(df, pd.DataFrame) and {"model", "accuracy_mean", "baseline"} <= set(df.columns)
