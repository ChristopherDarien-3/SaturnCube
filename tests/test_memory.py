import numpy as np
import pytest
import torch

import saturncube as sc
from test_v03 import blobs


def test_hopfield_recalls_corrupted_patterns():
    r = np.random.default_rng(0)
    P = r.choice([-1.0, 1.0], size=(5, 100))
    net = sc.HopfieldNetwork(seed=0).fit(P)
    noisy = P.copy()
    for row in noisy:
        row[r.choice(100, 15, replace=False)] *= -1
    for asyn in (False, True):
        assert (net.recall(noisy, asynchronous=asyn) == P).all()
    assert net.energy(P[0]) < net.energy(noisy[0])           # stored pattern sits lower in the energy landscape


def test_modern_hopfield_exceeds_classical_capacity():
    r = np.random.default_rng(0)
    P = r.normal(size=(200, 32))                              # 200 patterns in 32 dims: far beyond 0.14 * 32
    out = sc.ModernHopfield(beta=30).fit(P).recall(P + 0.1 * r.normal(size=P.shape))
    assert (np.argmax(out @ P.T, 1) == np.arange(200)).mean() > 0.95
    classical = sc.HopfieldNetwork().fit(np.sign(P)).recall(np.sign(P))
    assert (classical == np.sign(P)).all(1).mean() < 0.5


def test_associative_memory_rules():
    r = np.random.default_rng(0)
    K, V = r.normal(size=(6, 20)), r.normal(size=(6, 4))
    assert np.allclose(sc.AssociativeMemory("pinv").fit(K, V).recall(K), V, atol=1e-6)
    Q, _ = np.linalg.qr(r.normal(size=(20, 6)))               # orthonormal keys: Hebbian storage is exact too
    assert np.allclose(sc.AssociativeMemory("hebb").fit(Q.T, V).recall(Q.T), V, atol=1e-6)


def test_content_addressable_memory():
    r = np.random.default_rng(0)
    K, V = r.normal(size=(20, 16)), np.arange(20)[:, None].astype(float)
    cam = sc.ContentAddressableMemory().write(K, V)
    assert (cam.read(K + 0.05 * r.normal(size=K.shape))[:, 0].round() == np.arange(20)).all()


def test_product_key_memory_layer_and_model():
    pkm = sc.ProductKeyMemory(16, n_keys=8, topk=3)
    assert pkm(torch.randn(5, 16)).shape == (5, 16) and pkm.values.num_embeddings == 64
    X, y = blobs(n=400, k=3, d=6, sep=2.5)
    m = sc.ProductKeyMemoryNet(epochs=40, seed=0).fit(X[:300], y[:300])
    assert m.evaluate(X[300:], y[300:])["accuracy"] > 0.9


def lookup_task(n=800, S=6, K=10, C=4, seed=0):
    r = np.random.default_rng(seed)
    X = np.zeros((n, S + 1, K + C), dtype=np.float32); y = np.zeros(n, dtype=int)
    for i in range(n):
        keys = r.choice(K, S, replace=False); cls = r.integers(C, size=S)
        X[i, np.arange(S), keys] = 1; X[i, np.arange(S), K + cls] = 1
        j = r.integers(S); X[i, S, keys[j]] = 1; y[i] = cls[j]       # query = a key; answer = its value
    return X, y


def test_memory_networks_do_lookup():
    X, y = lookup_task()
    for m in (sc.MemoryNetwork(hops=2, epochs=60, lr=1e-2, seed=0, normalize=False),
              sc.KeyValueMemoryNetwork(d_key=10, hops=2, epochs=60, lr=1e-2, seed=0, normalize=False)):
        m.fit(X[:600], y[:600])
        assert m.evaluate(X[600:], y[600:])["accuracy"] > 0.9, type(m).__name__


def copy_task(n=400, L=2, bits=3, seed=0):
    r = np.random.default_rng(seed)
    data = r.integers(2, size=(n, L, bits)).astype(np.float32)
    X = np.zeros((n, 2 * L + 1, bits + 1), dtype=np.float32); Y = np.zeros((n, 2 * L + 1, bits), dtype=np.float32)
    X[:, :L, :bits] = data; X[:, L, bits] = 1; Y[:, L + 1:] = data
    return X, Y


@pytest.mark.parametrize("name", ["ntm", "dnc"])
def test_memory_machines_learn_copy_task(name):
    X, Y = copy_task()
    baseline = float((Y ** 2).mean())                          # predicting zeros
    m = sc.create(name, hidden=48, n_slots=8, slot_size=6, epochs=60, lr=1e-2, normalize=False, readout="all", seed=0).fit(X, Y)
    assert m.evaluate(X, Y)["mse"] < 0.4 * baseline, (name, m.evaluate(X, Y)["mse"], baseline)
