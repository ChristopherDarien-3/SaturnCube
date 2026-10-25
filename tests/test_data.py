import csv

import numpy as np
import pytest

import saturncube as sc


@pytest.fixture
def messy_csv(tmp_path):
    r = np.random.default_rng(0)
    rows = [["id", "Age", "Income", "City", "Joined", "Active", "Notes", "Const", "Empty", "Purchased"]]
    cities = ["Paris", "paris ", "LONDON", "London", "Tokyo", "Berlin", "  tokyo", ""]
    for i in range(300):
        age, inc, city = int(r.integers(18, 70)), int(r.integers(20, 120)) * 1000, cities[int(r.integers(len(cities)))]
        buy = "Yes" if (inc > 60000 and age < 55) or city.strip().lower() == "tokyo" else "No"
        rows.append([i, age if r.random() > 0.1 else ["N/A", "?", f" {age} "][int(r.integers(3))],
                     f"${inc:,}" if r.random() > 0.08 else "", city, f"20{int(r.integers(10, 24))}-{int(r.integers(1, 13)):02d}-{int(r.integers(1, 29)):02d}",
                     ["Yes", "no", "Y", "N"][int(r.integers(4))], "note %d " % i + "x" * 40, "same", "", buy])
    path = tmp_path / "messy.csv"
    with open(path, "w", newline="") as f:
        csv.writer(f, delimiter=";").writerows(rows)
    return str(path)


def test_profile_detects_kinds_and_drops_junk(messy_csv):
    d = sc.load_dataset(messy_csv, target="Purchased")
    assert d.n_rows == 300 and d.names[0] == "id"                        # delimiter ';' was sniffed
    kinds = {c["column"]: c["kind"] for c in d.report().columns}
    assert kinds == {"id": "id", "Age": "numeric", "Income": "numeric", "City": "categorical", "Joined": "datetime",
                     "Active": "boolean", "Notes": "text", "Const": "constant", "Empty": "empty", "Purchased": "target"}
    plans = {c["column"]: c["plan"] for c in d.report().columns}
    assert all(plans[c].startswith("DROPPED") for c in ("id", "Notes", "Const", "Empty"))
    assert "spelling variants merged" in str(d.report())


def test_prepare_is_clean_split_is_stratified_and_trainable(messy_csv):
    p = sc.load_dataset(messy_csv, target="Purchased").prepare(test_size=0.2, val_size=0.1, seed=0)
    for X in (p.X_train, p.X_val, p.X_test):
        assert X.dtype == np.float32 and np.isfinite(X).all() and X.shape[1] == len(p.feature_names)
    assert (len(p.X_train), len(p.X_val), len(p.X_test)) == (210, 30, 60)
    assert list(p.classes_) == ["No", "Yes"]
    assert abs(p.y_train.mean() - p.y_test.mean()) < 0.06                 # class balance preserved by stratification
    m = sc.MLP(epochs=40, seed=0).fit(p.X_train, p.y_train)
    assert m.evaluate(p.X_test, p.y_test)["accuracy"] > 0.85
    assert set(p.decode_target(m.predict(p.X_test))) <= {"No", "Yes"}


def test_statistics_come_from_training_rows_only():
    r = np.random.default_rng(0)
    cols = {"x": list(r.normal(10, 1, 200)), "y": [0, 1] * 100}
    cols["x"][0] = 1e6                                                   # one absurd outlier
    d = sc.load_dataset(cols, target="y")
    p = d.prepare(test_size=0.3, seed=1, stratify=False, scale="standard")
    spec = p.pipeline.preprocessor.numeric[0]
    split = sc.split_indices(200, 0.3, 0.0, 1)
    train_x = np.array(cols["x"])[split[0]]
    assert np.isclose(spec["fill"], np.median(train_x)) and np.isclose(spec["stats"]["mean"], train_x.mean())
    assert abs(p.X_train.mean()) < 1e-5                                   # train is standardised by its own statistics


def test_unseen_categories_and_missing_columns_are_handled(messy_csv):
    p = sc.load_dataset(messy_csv, target="Purchased").prepare(seed=0)
    new = {"Age": ["30", "N/A"], "Income": ["$90,000", ""], "City": ["Atlantis", None], "Joined": ["2020-05-17", "2021-01-02"], "Active": ["yes", "n"]}
    X = p.transform(new)
    assert X.shape == (2, p.n_features) and np.isfinite(X).all()
    with pytest.raises(ValueError, match="missing column"):
        p.transform({"Age": ["30"]})


def test_pipeline_save_and_load_give_identical_features(messy_csv, tmp_path):
    p = sc.load_dataset(messy_csv, target="Purchased").prepare(seed=0)
    p.save(tmp_path / "pipe.json")
    pipe = sc.load_pipeline(tmp_path / "pipe.json")
    new = {"Age": ["41"], "Income": ["$55,000"], "City": ["Tokyo"], "Joined": ["2019-03-04"], "Active": ["Y"]}
    assert np.allclose(pipe.transform(new), p.transform(new))
    assert list(pipe.decode_target(np.array([0, 1]))) == ["No", "Yes"]


def test_dataframe_arrays_and_records_inputs():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0] * 10, "b": ["u", "v", None, "u"] * 10, "t": [0, 1, 0, 1] * 10})
    assert sc.load_dataset(df, target="t").prepare(seed=0).n_features >= 3
    assert sc.load_dataset(np.random.default_rng(0).normal(size=(40, 3)), names=["p", "q", "r"], target="r").prepare().task == "regression"
    rec = [{"x": i, "y": i % 2, "z": "a" if i % 3 else "b"} for i in range(60)]
    assert sc.load_dataset(rec, target="y").prepare().task == "classification"


def test_task_inference_and_overrides():
    r = np.random.default_rng(0)
    base = {"f": list(r.normal(size=100))}
    assert sc.load_dataset({**base, "t": list(r.normal(size=100))}, target="t").prepare().task == "regression"
    assert sc.load_dataset({**base, "t": list(r.integers(0, 3, 100))}, target="t").prepare().task == "classification"
    assert sc.load_dataset({**base, "t": list(r.integers(0, 3, 100))}, target="t").prepare(task="regression").task == "regression"
    d = sc.load_dataset({**base, "hi": [None] * 95 + [1.5, 7.2, 3.3, 9.9, 4.1], "t": [0, 1] * 50}, target="t")
    assert "hi" in str(d.report()) and "DROPPED: 95% missing" in str(d.report())
    x = list(r.normal(size=100))
    assert "possible target leakage" in str(sc.load_dataset({"x": x, "t": [2 * v + 1 for v in x]}, target="t").report())
    assert "identifier" in str(sc.load_dataset({"x": list(range(100)), "t": [0, 1] * 50}, target="t").report())


def test_chronological_split_and_scale_target():
    d = sc.load_dataset({"x": [i * 1.5 for i in range(100)], "y": [float(i) * 3 for i in range(100)]}, target="y")
    p = d.prepare(test_size=0.2, split="chronological", scale_target=True)
    assert p.X_test[:, 0].min() > p.X_train[:, 0].max()                    # test comes strictly after train
    assert abs(p.y_train.mean()) < 1e-6 and np.allclose(p.decode_target(p.y_test), np.arange(80, 100) * 3.0)


def test_sequences_align_windows_and_never_leak_forward():
    ts = {"t": list(range(60, 0, -1)), "v": [float(i) for i in range(60, 0, -1)]}   # rows stored newest-first
    p = sc.load_dataset(ts, target="v").prepare_sequences(window=5, horizon=2, time_column="t", test_size=0.2, scale="none" if False else "standard")
    assert p.X_train.shape[1:] == (5, p.n_features) and p.kind == "sequence"
    assert p.y_test.min() > p.y_train.max()                                # later targets only in later splits
    raw = sc.load_dataset(ts, target="v").prepare_sequences(window=5, horizon=2, time_column="t", scale=None, test_size=0.2)
    v_col = raw.feature_names.index("v")
    assert np.allclose(raw.y_train, raw.X_train[:, -1, v_col] + 2)         # target = last observed value + horizon
    gru = sc.GRU(epochs=30, lr=1e-2, seed=0).fit(p.X_train, p.y_train, )
    assert gru.predict(p.X_test).shape == p.y_test.shape


def test_images_from_flat_pixel_columns():
    r = np.random.default_rng(0)
    n, s = 300, 8
    y = r.integers(2, size=n)
    imgs = r.normal(0, 20, size=(n, s, s)).clip(0, 255)
    imgs[y == 1, :, :4] += 120                                             # class 1: bright left half
    cols = {f"p{i}": list(imgs.reshape(n, -1)[:, i]) for i in range(s * s)}
    d = sc.load_dataset({**cols, "label": list(y)}, target="label")
    p = d.prepare_images(seed=0)
    assert p.X_train.shape[1:] == (1, 8, 8) and 0 <= p.X_train.min() and p.X_train.max() <= 1.0
    assert sc.CNN(width=8, epochs=15, lr=3e-3, seed=0).fit(p.X_train, p.y_train).evaluate(p.X_test, p.y_test)["accuracy"] > 0.9
    with pytest.raises(ValueError, match="square"):
        sc.load_dataset({c: [float(i) for i in range(10)] for c in "abcde"} | {"label": [0, 1] * 5}, target="label").prepare_images()


def test_auto_model_end_to_end_and_family_routing(messy_csv):
    res = sc.auto_model(messy_csv, target="Purchased", model="mlp", hidden=(32,), fit={"epochs": 40})
    assert res.metrics["test"]["accuracy"] > 0.85
    pred = res.predict({"Age": ["30"], "Income": ["$99,000"], "City": ["Tokyo"], "Joined": ["2020-01-01"], "Active": ["Y"]})
    assert pred[0] in ("No", "Yes")
    assert sc.auto_model(messy_csv, target="Purchased", model="random_forest").metrics["test"]["accuracy"] > 0.85
    with pytest.raises(ValueError, match="window"):
        sc.auto_model(messy_csv, target="Purchased", model="gru")
    with pytest.raises(ValueError, match="cannot build"):
        sc.auto_model(messy_csv, target="Purchased", model="gcn")


def test_models_explain_bad_input_and_checks_catch_shapes():
    with pytest.raises(ValueError, match="load_dataset"):
        sc.MLP().fit([["a", 1], ["b", 2]], [0, 1])
    with pytest.raises(ValueError, match="NaN"):
        sc.MLP().fit([[1.0, np.nan], [2.0, 3.0]], [0, 1])
    with pytest.raises(ValueError, match="shape"):
        sc.check_arrays(np.zeros((4, 3)), expect="image")
    with pytest.raises(ValueError, match="rows"):
        sc.check_arrays(np.zeros((4, 3)), np.zeros(5))
    assert sc.split_indices(10, 0.2, 0.2)[2].size == 2
    with pytest.raises(ValueError):
        sc.split_indices(10, 0.6, 0.5)
