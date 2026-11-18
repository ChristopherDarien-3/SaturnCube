import numpy as np
import pytest

import saturncube as sc
from saturncube.evaluation import walk_forward_splits


class Spy:
    """Records which rows fit() saw (column 0 of X is the row's position in time) and predicts the first label."""

    def __init__(self, log):
        self.log = log

    def fit(self, X, y, epochs=None, val_data=None):
        self.log.append({"train": X[:, 0].astype(int).copy(), "val": None if val_data is None else val_data[0][:, 0].astype(int).copy()})
        self.first = y[0]
        return self

    def predict(self, X):
        return np.full(len(X), self.first)


def time_ordered(n=600, d=3, seed=0, classes=3):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, d))
    X[:, 0] = np.arange(n)
    return X, (np.arange(n) // 7 % classes).astype(int)


# --------------------------------------------------------------------------- the split generator ---
def test_folds_are_chronological_purged_and_cover_the_end_of_the_series():
    folds = walk_forward_splits(1000, n_folds=3, test_size=0.1, val_size=0.1, purge=39)
    assert [f["test"][0] for f in folds] == [700, 800, 900] and folds[-1]["test"][-1] == 999
    for f in folds:
        tr, va, te = f["train"], f["val"], f["test"]
        assert tr.max() < va.min() < va.max() < te.min()                      # past -> validation -> test
        assert va.min() - tr.max() - 1 == 39 and te.min() - va.max() - 1 == 39   # exactly `purge` rows dropped
        assert len(te) == 100 and len(va) == 100 and tr.min() == 0             # expanding window


def test_purge_of_window_plus_horizon_minus_one_removes_all_overlap():
    W, H = 32, 8                                                              # features look back W rows, labels ahead H
    for f in walk_forward_splits(2000, n_folds=4, test_size=0.1, val_size=0.1, purge=W + H - 1):
        for earlier, later in ((f["train"], f["val"]), (f["val"], f["test"]), (f["train"], f["test"])):
            # a row i reads bars [i-W+1, i+H]; two rows share a bar only if they are closer than W+H
            assert later.min() - earlier.max() >= W + H
    f = walk_forward_splits(2000, n_folds=4, test_size=0.1, val_size=0.1, purge=0)[0]
    assert f["test"].min() - f["val"].max() < W + H                           # without purge they do overlap


def test_rolling_window_embargo_and_row_counts():
    rolling = walk_forward_splits(1000, n_folds=3, test_size=0.1, val_size=0.1, purge=39, window=300)
    assert [len(f["train"]) for f in rolling] == [300, 300, 300] and rolling[2]["train"].max() == 721
    plain = walk_forward_splits(1000, n_folds=4, test_size=0.1, val_size=0.02, purge=5)
    emb = walk_forward_splits(1000, n_folds=4, test_size=0.1, val_size=0.02, purge=5, embargo=7)
    for k, (a, b) in enumerate(zip(plain, emb)):
        removed = np.setdiff1d(a["train"], b["train"])
        expected = np.concatenate([np.arange(plain[j]["test"][-1] + 1, plain[j]["test"][-1] + 1 + 7) for j in range(k)]) if k else np.array([], dtype=int)
        expected = np.intersect1d(expected, a["train"])
        assert np.array_equal(removed, expected)                              # exactly the rows right after earlier tests
    assert len(np.setdiff1d(plain[3]["train"], emb[3]["train"])) > 0          # and it does remove something
    assert emb[0]["train"].tolist() == plain[0]["train"].tolist()             # fold 0 has no earlier test block
    assert len(walk_forward_splits(500, n_folds=2, test_size=50, val_size=0)[0]["val"]) == 0   # row counts, no validation


def test_split_generator_rejects_impossible_layouts():
    with pytest.raises(ValueError, match="too little history"):
        walk_forward_splits(100, n_folds=5, test_size=0.2)
    with pytest.raises(ValueError, match="n_folds"):
        walk_forward_splits(100, n_folds=0)
    with pytest.raises(ValueError, match="purge"):
        walk_forward_splits(100, purge=-1)
    with pytest.raises(ValueError, match="fraction"):
        walk_forward_splits(100, test_size=1.5)


# ------------------------------------------------------------------------------------- benchmark ---
def test_models_only_ever_train_on_the_past():
    X, y = time_ordered()
    log = []
    res = sc.benchmark({"spy": lambda s: Spy(log)}, X, y, split="walk_forward", n_folds=4, purge=10, seeds=2,
                       baselines=False, show=False)
    assert len(log) == 8                                                      # 4 folds x 2 seeds
    for run, call in enumerate(log):
        fold = run // 2
        te = res.splits[fold]["test"]
        assert call["train"].max() < call["val"].min() < te.min()
        assert set(call["train"]) == set(res.splits[fold]["train"]) and set(call["val"]) == set(res.splits[fold]["val"])
    tests = [res.splits[k]["test"] for k in range(4)]
    assert all(tests[k].max() + 1 == tests[k + 1].min() for k in range(3))     # consecutive blocks


def test_units_are_folds_and_seeds_are_averaged_inside_a_fold():
    X, y = time_ordered(900)
    res = sc.benchmark([("random_forest", {"n_trees": 5})], X, y, split="walk_forward", n_folds=3, seeds=2, show=False)
    assert res.n_runs == 6 and res.units == [0, 1, 2] and res.walk["n_folds"] == 3
    runs = res.scores["random_forest"]["accuracy"]
    per_fold = [runs[:2].mean(), runs[2:4].mean(), runs[4:].mean()]
    row = next(r for r in res.to_rows() if r["model"] == "random_forest")
    assert row["accuracy_mean"] == pytest.approx(np.mean(per_fold))
    assert row["accuracy_std"] == pytest.approx(np.std(per_fold, ddof=1))     # spread across folds, not across runs
    assert row["n_ok"] == 6 and row["n_runs"] == 6


def test_fold_table_and_header():
    X, y = time_ordered(900)
    res = sc.benchmark(["naive_bayes"], X, y, split="walk_forward", n_folds=3, seeds=1, purge=5, window=400, show=False)
    t = res.fold_table()
    assert "fold 0" in t and "fold 2" in t and "naive_bayes" in t and "class prior (baseline)" in t and "±" in t
    text = str(res)
    assert "walk-forward 3 folds x 1 seed" in text and "rolling (400 rows)" in text and "purge 5" in text
    assert "std is over folds" in text
    plain = sc.benchmark(["naive_bayes"], X, y, seeds=1, show=False)
    with pytest.raises(ValueError, match="walk_forward"):
        plain.fold_table()


def test_walk_forward_exposes_what_random_splits_hide():
    n = 1200
    t = np.arange(n, dtype=float)
    X = t.reshape(-1, 1)
    y = (np.arange(n) // 100 % 2).astype(int)                                 # long regimes that alternate
    rand = sc.benchmark(["random_forest"], X, y, seeds=2, baselines=False, show=False)
    walk = sc.benchmark(["random_forest"], X, y, split="walk_forward", n_folds=4, seeds=2, baselines=False, show=False)
    assert rand.scores["random_forest"]["accuracy"].mean() > 0.9              # memorises neighbouring rows
    assert walk.scores["random_forest"]["accuracy"].mean() < 0.75             # the future regime was never seen


def test_time_argument_orders_shuffled_rows_and_maps_back_to_original_indices():
    X, y = time_ordered(500)
    stamp = X[:, 0].copy()
    perm = np.random.default_rng(1).permutation(500)
    Xs, ys, ts = X[perm], y[perm], stamp[perm]
    res = sc.benchmark(["naive_bayes"], Xs, ys, split="walk_forward", n_folds=3, seeds=1, purge=4, time=ts,
                       baselines=False, show=False)
    for f in range(3):
        s = res.splits[f]
        assert ts[s["train"]].max() < ts[s["val"]].min() < ts[s["test"]].min()   # chronological in the ORIGINAL arrays
        assert len(set(s["train"]) & set(s["test"])) == 0


def test_regression_walk_forward_runs():
    r = np.random.default_rng(0)
    X = r.normal(size=(600, 3))
    y = X[:, 0] * 2 + 0.1 * r.normal(size=600)
    res = sc.benchmark([("mlp", {"epochs": 20})], X, y, split="walk_forward", n_folds=3, seeds=1, show=False)
    assert res.task == "regression" and np.isfinite(res.scores["mlp"]["r2"]).all() and len(res.units) == 3


def test_walk_forward_argument_errors():
    X, y = time_ordered(300)
    with pytest.raises(ValueError, match="cannot be combined with groups"):
        sc.benchmark(["naive_bayes"], X, y, groups=np.arange(300) // 10, split="walk_forward", show=False)
    with pytest.raises(ValueError, match="only used with split='walk_forward'"):
        sc.benchmark(["naive_bayes"], X, y, purge=5, show=False)
    with pytest.raises(ValueError, match="needs groups"):
        sc.benchmark(["naive_bayes"], X, y, split="grouped", show=False)
    with pytest.raises(ValueError, match="ignores groups"):
        sc.benchmark(["naive_bayes"], X, y, groups=np.arange(300) // 10, split="random", show=False)
    with pytest.raises(ValueError, match="split must be"):
        sc.benchmark(["naive_bayes"], X, y, split="forward", show=False)
    with pytest.raises(ValueError, match="time must have one entry"):
        sc.benchmark(["naive_bayes"], X, y, split="walk_forward", time=np.arange(5), show=False)


def test_a_class_missing_from_the_first_training_window_is_reported_up_front():
    X = np.arange(600, dtype=float).reshape(-1, 1)
    y = np.where(np.arange(600) < 450, 0, 1)                                  # class 1 only appears late
    with pytest.raises(ValueError, match="missing class"):
        sc.benchmark(["naive_bayes"], X, y, split="walk_forward", n_folds=4, show=False)


def test_failures_in_a_fold_are_labelled_with_the_fold():
    X, y = time_ordered(400)

    class Broken:
        def fit(self, X, y):
            raise RuntimeError("boom")

        def predict(self, X):
            return None

    res = sc.benchmark({"broken": lambda s: Broken()}, X, y, split="walk_forward", n_folds=2, seeds=1,
                       baselines=False, show=False)
    assert len(res.errors["broken"]) == 2 and res.errors["broken"][0][1].startswith("fold 0:")
    assert "broken failed on 2/2 runs" in str(res)
