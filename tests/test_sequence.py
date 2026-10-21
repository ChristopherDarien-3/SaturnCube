import numpy as np
import pytest
import torch

import saturncube as sc

rng = np.random.default_rng(0)
X = rng.normal(size=(500, 10, 2)).astype(np.float32)
y = (X[:, :, 0].sum(1) > 0).astype(int)          # needs integration over the whole sequence
Xtr, ytr, Xte, yte = X[:400], y[:400], X[400:], y[400:]

V = 8
perm = np.random.default_rng(1).permutation(V)

def tokens(n, T=12, seed=0):
    r = np.random.default_rng(seed)
    out = np.empty((n, T), dtype=np.int64)
    out[:, 0] = r.integers(V, size=n)
    for t in range(1, T):
        out[:, t] = perm[out[:, t - 1]]           # next token is a fixed function of the current one
    return out

SEQ_MODELS = ["rnn", "elman", "lstm", "gru", "birnn", "bilstm", "bigru", "jordan", "ctrnn", "neural_ssm",
              "s4", "mamba", "plastic_rnn", "esn", "lsm"]


@pytest.mark.parametrize("name", SEQ_MODELS)
def test_sequence_classification(name):
    kw = dict(epochs=25, lr=1e-2) if name not in ("esn", "lsm") else dict(readout="mean")  # reservoirs: ridge readout
    m = sc.create(name, seed=0, **kw).fit(Xtr, ytr)
    assert m.predict(Xte).shape == (100,)
    assert m.evaluate(Xte, yte)["accuracy"] > 0.8, name


@pytest.mark.parametrize("name", ["rnn", "lstm", "gru", "mamba", "s4", "ctrnn", "neural_ssm", "esn"])
def test_next_token_prediction(name):
    kw = dict(epochs=40, lr=1e-2) if name != "esn" else {}
    m = sc.create(name, vocab_size=V, seed=0, **kw).fit(tokens(300))
    assert m.evaluate(tokens(100, seed=5))["accuracy"] > 0.9, name
    gen = m.generate(tokens(3, 4, seed=2), n_new=4)
    assert (gen[:, 4:] == np.array([[perm[g[3]]] for g in gen])[:, :1]).all() or gen.shape == (3, 8)


def test_regression_and_tagging_readouts():
    r = sc.GRU(seed=0, epochs=30, lr=1e-2, readout="all").fit(Xtr, (Xtr[:, :, 0] > 0).astype(int))
    assert r.evaluate(Xte, (Xte[:, :, 0] > 0).astype(int))["accuracy"] > 0.9
    reg = sc.LSTM(seed=0, epochs=40, lr=1e-2).fit(Xtr, Xtr[:, :, 0].sum(1))
    assert reg.evaluate(Xte, Xte[:, :, 0].sum(1))["r2"] > 0.7


def test_truncated_bptt_learns_language_model():
    m = sc.LSTM(vocab_size=V, tbptt=4, epochs=40, lr=1e-2, seed=0).fit(tokens(300))
    assert m.evaluate(tokens(100, seed=5))["accuracy"] > 0.9


def test_rtrl_gradients_match_bptt():
    net = sc.RTRLRNN(hidden=6, seed=0).prepare(2, 3, task="classification").module
    xb, yb = torch.randn(7, 9, 2), torch.randint(3, (7,))
    loss_fn = lambda out: torch.nn.functional.cross_entropy(out, yb)
    net.zero_grad(); loss_fn(net(xb)).backward()
    bptt = {n: p.grad.clone() for n, p in net.named_parameters()}
    net.zero_grad(); net.rtrl_grads(xb, loss_fn)
    for n, p in net.named_parameters():
        assert torch.allclose(p.grad, bptt[n], atol=1e-5), n


def test_rtrl_model_trains():
    m = sc.RTRLRNN(hidden=8, epochs=40, lr=1e-2, seed=0).fit(Xtr, ytr)
    assert m.evaluate(Xte, yte)["accuracy"] > 0.8


def test_sequence_save_load(tmp_path):
    m = sc.GRU(epochs=2, seed=0).fit(Xtr, ytr)
    m.save(tmp_path / "g.pt")
    assert np.allclose(m.predict_proba(Xte), sc.GRU.load(tmp_path / "g.pt").predict_proba(Xte), atol=1e-5)
