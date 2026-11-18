import numpy as np
import pytest
import torch

import saturncube as sc

MOTIF = np.array([[1, 0], [2, 1], [3, 2], [2, 3], [1, 2]], dtype=np.float32) * 1.5


def motif_task(n=500, T=40, seed=0):
    """Noise sequences (n, T, 2); the positive class contains a fixed 5-step motif at a random position."""
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, T, 2)).astype(np.float32)
    y = r.integers(0, 2, n)
    for i in np.flatnonzero(y == 1):
        p = r.integers(0, T - 5)
        X[i, p:p + 5] += MOTIF
    return X, y


def raw_module(name, T_dim=2, **kw):
    """An untrained network built without data, switched to eval mode, for exact property checks."""
    m = sc.create(name, seed=0, **kw).prepare(T_dim, 3)
    m.module.eval()
    return m


def run(m, x):
    with torch.no_grad():
        return m.module(torch.as_tensor(x, dtype=torch.float32)).numpy()


# --------------------------------------------------------------------------------- properties ---
@pytest.mark.parametrize("readout", ["mean", "max"])
def test_cnn1d_pooled_output_does_not_depend_on_where_the_pattern_is(readout):
    m = raw_module("cnn1d", hidden=16, n_layers=3, kernel_size=5, readout=readout)
    T = 64
    base = np.zeros((1, T, 2), dtype=np.float32)
    early, late = base.copy(), base.copy()
    early[0, 20:25], late[0, 38:43] = MOTIF, MOTIF
    assert np.allclose(run(m, early), run(m, late), atol=1e-5)               # shifted pattern, identical output
    assert not np.allclose(run(m, early), run(m, base), atol=1e-3)           # and the pattern does matter


def test_tcn_is_causal_and_cnn1d_is_not():
    T, t0 = 64, 30
    x = np.random.default_rng(0).normal(size=(2, T, 2)).astype(np.float32)
    x2 = x.copy()
    x2[:, t0:] += 5.0                                                         # change the future from step t0 on
    tcn = raw_module("tcn", hidden=16, n_layers=3, readout="all")
    assert np.allclose(run(tcn, x)[:, :t0], run(tcn, x2)[:, :t0], atol=1e-6)  # steps before t0 cannot have moved
    assert not np.allclose(run(tcn, x)[:, t0:], run(tcn, x2)[:, t0:], atol=1e-3)
    cnn = raw_module("cnn1d", hidden=16, n_layers=3, kernel_size=5, readout="all")
    assert not np.allclose(run(cnn, x)[:, t0 - 6:t0], run(cnn, x2)[:, t0 - 6:t0], atol=1e-3)   # it sees ahead


def test_tcn_receptive_field_matches_its_dilation_arithmetic():
    m = raw_module("tcn", hidden=16, n_layers=3, kernel_size=3, readout="last")
    rf = m.receptive_field
    assert rf == 1 + 2 * 2 * (1 + 2 + 4) == 29 and m.dilations == [1, 2, 4]
    T = 64
    x = np.random.default_rng(1).normal(size=(3, T, 2)).astype(np.float32)
    ref = run(m, x)
    for t, should_change in ((T - 1 - rf, False), (T - rf, True)):          # one step outside / exactly at the edge
        y = x.copy()
        y[:, t] += 10.0
        assert np.allclose(run(m, y), ref, atol=1e-6) is (not should_change), t


def test_cnn1d_receptive_field_is_two_sided():
    m = raw_module("cnn1d", hidden=16, n_layers=3, kernel_size=5, readout="all")
    assert m.receptive_field == 1 + 3 * 4 == 13
    x = np.random.default_rng(2).normal(size=(2, 64, 2)).astype(np.float32)
    ref, step = run(m, x), 30
    for t, should_change in ((step + 7, False), (step + 6, True), (step - 7, False), (step - 6, True)):
        y = x.copy()
        y[:, t] += 10.0
        assert np.allclose(run(m, y)[:, step], ref[:, step], atol=1e-6) is (not should_change), t


# ------------------------------------------------------------------------------------ learning ---
@pytest.mark.parametrize("name", ["cnn1d", "tcn"])
def test_finds_a_motif_at_a_random_position(name):
    X, y = motif_task()
    m = sc.create(name, epochs=40, seed=0, readout="max").fit(X[:400], y[:400])
    assert m.predict(X[400:]).shape == (100,) and (m.predict(X[400:]) == y[400:]).mean() > 0.9
    p = m.predict_proba(X[400:])
    assert p.shape == (100, 2) and np.allclose(p.sum(1), 1, atol=1e-5)


@pytest.mark.parametrize("name", ["cnn1d", "tcn"])
def test_token_input_and_per_step_tagging(name):
    r = np.random.default_rng(0)
    toks = r.integers(1, 20, size=(400, 20))
    has7 = (toks == 7).any(1).astype(int)
    m = sc.create(name, vocab_size=20, epochs=40, seed=0, readout="max").fit(toks[:300], has7[:300])
    assert (m.predict(toks[300:]) == has7[300:]).mean() > 0.9
    X = r.normal(size=(300, 16, 1)).astype(np.float32)
    tag = (X[:, :, 0] > 0).astype(int)                                        # a label for every step
    t = sc.create(name, epochs=30, seed=0, readout="all").fit(X[:200], tag[:200])
    pred = t.predict(X[200:])
    assert pred.shape == (100, 16) and (pred == tag[200:]).mean() > 0.9


def test_regression_target():
    r = np.random.default_rng(0)
    X = r.normal(size=(400, 24, 2)).astype(np.float32)
    y = (X[:, :, 0].mean(1) * 5).astype(np.float32)
    for name in ("cnn1d", "tcn"):
        m = sc.create(name, epochs=40, seed=0, readout="mean").fit(X[:300], (y[:300] - y[:300].mean()) / y[:300].std())
        assert m.task == "regression" and m.predict(X[300:]).shape == (100,)


# --------------------------------------------------------------------------------- LM mode ---
def test_tcn_does_next_token_prediction_and_cnn1d_refuses():
    V, T = 10, 14
    starts = np.random.default_rng(0).integers(0, V, size=300)
    seqs = (starts[:, None] + np.arange(T)[None, :]) % V                       # 3,4,5,... wrapping around
    m = sc.TCN(vocab_size=V, hidden=24, n_layers=3, epochs=60, seed=0).fit(seqs[:250])
    assert m.evaluate(seqs[250:])["accuracy"] > 0.95
    assert m.generate(np.array([[0, 1, 2, 3]]), n_new=3).tolist() == [[0, 1, 2, 3, 4, 5, 6]]
    cnn = sc.CNN1D(vocab_size=V, epochs=1, seed=0)
    with pytest.raises(ValueError, match="leak the answer"):
        cnn.fit(seqs)
    with pytest.raises(ValueError, match="not causal"):
        cnn.generate(seqs[:2, :4])


# ---------------------------------------------------------------------------------- plumbing ---
def test_readout_options_and_parameter_validation():
    with pytest.raises(ValueError, match="readout must be 'last', 'mean' or 'all'"):
        sc.create("gru", readout="max")                                      # existing models are unchanged
    with pytest.raises(ValueError, match="'max'"):
        sc.create("cnn1d", readout="median")
    for kw in ({"n_layers": 0}, {"kernel_size": 0}):
        with pytest.raises(ValueError, match="at least"):
            sc.create("cnn1d", **kw)
    with pytest.raises(ValueError, match="at least"):
        sc.create("tcn", kernel_size=1)
    assert sc.create("cnn1d").readout == "mean" and sc.create("tcn").readout == "last"


@pytest.mark.parametrize("name", ["cnn1d", "tcn"])
def test_save_load_roundtrip(tmp_path, name):
    X, y = motif_task(200)
    m = sc.create(name, epochs=3, seed=0, readout="max").fit(X, y)
    p = str(tmp_path / "m.pt")
    m.save(p)
    assert np.allclose(type(m).load(p).predict_proba(X), m.predict_proba(X), atol=1e-5)


def test_registered_in_the_sequence_family_and_works_in_the_benchmark():
    assert {"cnn1d", "tcn"} <= set(sc.available("sequence"))
    X, y = motif_task(400)
    res = sc.benchmark({"cnn1d": ("cnn1d", {"readout": "max", "epochs": 15}), "tcn": ("tcn", {"readout": "max", "epochs": 15})},
                       X, y, seeds=1, show=False)
    assert not res.errors["cnn1d"] and not res.errors["tcn"] and np.isfinite(res.scores["tcn"]["log_loss"]).all()


def test_pooling_modes_do_what_they_say():
    from saturncube.sequence import ConvCore
    core = ConvCore(4, 3, [1], causal=True, n_convs=1)
    y = torch.tensor([[[1.0, 5.0], [3.0, 2.0], [2.0, 9.0]]])                  # (1, T=3, 2)
    assert core.pool(y, "max").tolist() == [[3.0, 9.0]]
    assert torch.allclose(core.pool(y, "mean"), torch.tensor([[2.0, 16.0 / 3]]))
    assert core.pool(y, "last").tolist() == [[2.0, 9.0]]
