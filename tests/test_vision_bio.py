import numpy as np
import pytest
import torch

import saturncube as sc
from test_v03 import blobs


# ------------------------------------------------------------------------------- CNN ---
def shapes(n, seed, size=16):
    """Plus (9 px) vs filled square (9 px) at a random position: only shape, not pixel count, differs."""
    r = np.random.default_rng(seed)
    X = r.normal(0, 0.25, size=(n, 1, size, size)).astype(np.float32)
    y = r.integers(2, size=n)
    for i in range(n):
        a, b = r.integers(2, size - 2, size=2)
        if y[i] == 0:
            for da, db in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (-2, 0), (0, 2), (0, -2)):
                if 0 <= a + da < size and 0 <= b + db < size:
                    X[i, 0, a + da, b + db] += 1.5
        else:
            X[i, 0, a - 1:a + 2, b - 1:b + 2] += 1.5
    return X, y


def test_cnn_uses_translation_invariance_that_an_mlp_cannot_learn_from_few_samples():
    X, y = shapes(900, 0)
    cnn = sc.CNN(epochs=25, lr=3e-3, seed=0).fit(X[:600], y[:600])
    mlp = sc.MLP(hidden=(128, 64), epochs=40, seed=0).fit(X[:600].reshape(600, -1), y[:600])
    a_cnn = cnn.evaluate(X[600:], y[600:])["accuracy"]
    a_mlp = mlp.evaluate(X[600:].reshape(300, -1), y[600:])["accuracy"]
    assert a_cnn > 0.95 and a_cnn > a_mlp + 0.15, (a_cnn, a_mlp)
    assert cnn.predict_proba(X[600:]).shape == (300, 2)


def test_cnn_input_formats_regression_and_save_load(tmp_path):
    r = np.random.default_rng(0)
    level = r.uniform(-1, 1, size=200)
    X = r.normal(0, 0.3, size=(200, 3, 12, 12)).astype(np.float32)         # 3-channel images, NCHW
    X[:, 0] += level[:, None, None]
    y = level * 3                                                          # regression target: brightness of channel 0
    m = sc.CNN(width=8, epochs=20, lr=5e-3, seed=0).fit(X[:150], y[:150])
    assert m.evaluate(X[150:], y[150:])["r2"] > 0.85
    assert m.predict(X[150:]).shape == (50,)
    m.save(tmp_path / "c.pt")
    assert np.allclose(m.predict(X[150:]), sc.CNN.load(tmp_path / "c.pt").predict(X[150:]), atol=1e-4)
    gray = sc.CNN(width=8, epochs=1, seed=0).fit(X[:20, 0], (y[:20] > 0).astype(int))        # (n, H, W) grayscale
    assert gray.input_dim == 1
    last = sc.CNN(channels_last=True)._as_x(np.zeros((2, 12, 12, 3)))                         # NHWC input
    assert last.shape == (2, 12, 12, 3) and sc.CNN()._as_x(np.zeros((2, 3, 12, 12))).shape == (2, 12, 12, 3)
    with pytest.raises(ValueError):
        sc.CNN()._as_x(np.zeros((4, 5)))


# ------------------------------------------------------------------------------- SNN ---
def test_snn_classifies_with_both_encodings_and_reports_firing_rates():
    X, y = blobs(n=600, k=3, d=6, sep=2.5)
    for enc, floor in (("direct", 0.9), ("rate", 0.85)):
        m = sc.SpikingNetwork(encoding=enc, epochs=40, lr=1e-2, seed=0).fit(X[:450], y[:450])
        assert m.evaluate(X[450:], y[450:])["accuracy"] > floor, enc
        rates = m.firing_rates(X)
        assert len(rates) == 1 and 0.0 < rates[0] < 1.0                    # neurons spike, but sparsely
    with pytest.raises(ValueError):
        sc.SpikingNetwork(encoding="nope")


def test_snn_learns_a_nonlinear_rule():
    r = np.random.default_rng(0)
    X = r.normal(size=(800, 4)).astype(np.float32)
    y = ((X[:, 0] * X[:, 1]) > 0).astype(int)                              # XOR-like: a linear model cannot do this
    m = sc.SpikingNetwork(hidden=(64,), epochs=60, lr=1e-2, seed=0).fit(X[:600], y[:600])
    assert m.evaluate(X[600:], y[600:])["accuracy"] > 0.85


def test_lif_layer_emits_binary_spikes_and_passes_gradients():
    torch.manual_seed(0)
    layer = sc.LIFLayer(4, 6, beta=0.9, threshold=1.0)
    x = (torch.randn(1, 5, 4) * 2).expand(12, 5, 4).contiguous().requires_grad_(True)
    spikes = layer(x)
    assert set(spikes.unique().tolist()) <= {0.0, 1.0} and 0 < spikes.mean() < 1
    spikes.sum().backward()                                                # surrogate derivative lets gradients through
    assert torch.isfinite(layer.fc.weight.grad).all() and layer.fc.weight.grad.abs().sum() > 0


def test_snn_save_load(tmp_path):
    X, y = blobs(n=200, k=3, d=6, sep=2.5)
    m = sc.SpikingNetwork(epochs=3, seed=0).fit(X, y)
    m.save(tmp_path / "s.pt")
    assert np.allclose(m.predict_proba(X), sc.SpikingNetwork.load(tmp_path / "s.pt").predict_proba(X), atol=1e-5)


# ------------------------------------------------------------------------------- SOM ---
def test_som_unfolds_into_an_ordered_map_of_uniform_data():
    from scipy.stats import spearmanr
    U = np.random.default_rng(0).random((1500, 2)).astype(np.float32)
    som = sc.SelfOrganizingMap(grid=(8, 8), seed=0).fit(U)
    W = som.unit_weights()
    rows, cols = np.repeat(np.arange(8), 8), np.tile(np.arange(8), 8)
    order = max(abs(spearmanr(W[..., d].ravel(), g)[0]) for d in (0, 1) for g in (rows, cols))
    assert order > 0.95                                                    # the grid axes line up with the data axes
    assert som.topographic_error(U) < 0.05 and som.quantization_error(U) < 0.3
    assert W.min() > -0.1 and W.max() < 1.1                                # units stay inside the data cloud


def test_som_clusters_labels_and_diagnostics():
    X, y = blobs(n=600, k=3, d=6, sep=3.0)
    som = sc.SelfOrganizingMap(grid=(6, 6), seed=0).fit(X[:450])
    coords = som.transform(X[450:])
    assert coords.shape == (150, 2) and coords.min() >= 0 and coords.max() <= 5
    assert som.hit_map(X[:450]).sum() == 450 and som.u_matrix().shape == (6, 6)
    with pytest.raises(RuntimeError):
        som.classify(X)
    som.label_units(X[:450], y[:450])
    assert (som.classify(X[450:]) == y[450:]).mean() > 0.9
    ev = som.evaluate(X[450:])
    assert ev["quantization_error"] > 0 and 0 <= ev["topographic_error"] < 0.3
    assert som.history_["loss"][-1] < som.history_["loss"][0]              # quantisation error falls during training
