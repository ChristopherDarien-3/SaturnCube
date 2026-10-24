import numpy as np
import pytest
import torch

import saturncube as sc
from test_v03 import blobs, tasks

pytestmark = pytest.mark.slow


# ------------------------------------------------------------------ metric learning ---
def pairs(X, y, n, seed=0):
    r = np.random.default_rng(seed)
    i, j = r.integers(len(X), size=n), r.integers(len(X), size=n)
    return np.stack([X[i], X[j]], 1), (y[i] == y[j]).astype(float)


def test_siamese_learns_similarity():
    X, y = blobs(n=600, k=4, d=6, sep=3.0)
    P, s = pairs(X, y, 1500)
    Pt, st = pairs(X, y, 300, seed=1)
    m = sc.Siamese(epochs=30, seed=0).fit(P, s)
    assert m.evaluate(Pt, st)["accuracy"] > 0.85


def test_triplet_network_orders_distances():
    X, y = blobs(n=600, k=4, d=6, sep=3.0)
    r = np.random.default_rng(0)
    def trip(n):
        a = r.integers(len(X), size=n)
        pos = np.array([r.choice(np.flatnonzero(y == y[i])) for i in a])
        neg = np.array([r.choice(np.flatnonzero(y != y[i])) for i in a])
        return np.stack([X[a], X[pos], X[neg]], 1)
    m = sc.TripletNetwork(epochs=30, seed=0).fit(trip(1200))
    assert m.evaluate(trip(300))["triplet_accuracy"] > 0.9


def test_twin_tower_retrieval():
    r = np.random.default_rng(0)
    A = r.normal(size=(600, 8)).astype(np.float32)
    M = r.normal(size=(8, 8))
    B = (A @ M + 0.1 * r.normal(size=A.shape)).astype(np.float32)   # partner lives in a different feature space
    m = sc.TwinTower(latent_dim=8, epochs=60, lr=5e-3, seed=0).fit(np.stack([A[:500], B[:500]], 1))
    top1 = m.retrieve(A[500:], B[500:], k=1)[:, 0]
    assert (top1 == np.arange(100)).mean() > 0.8


def test_contrastive_embedding_classifies_by_centroid():
    X, y = blobs(n=600, k=4, d=6, sep=3.0)
    m = sc.ContrastiveEmbedding(epochs=30, batch_size=128, seed=0).fit(X[:450], y[:450])
    assert m.evaluate(X[450:], y[450:])["accuracy"] > 0.9


# ------------------------------------------------------------------------- few-shot ---
@pytest.mark.parametrize("name", ["prototypical", "relation", "matching"])
def test_few_shot_on_unseen_classes(name):
    r = np.random.default_rng(0)
    C = r.normal(size=(12, 8)) * 4
    draw = lambda cls, n: ((lambda y: ((C[y] + r.normal(size=(n, 8))).astype(np.float32), y))(r.choice(cls, n)))
    Xtr, ytr = draw(range(8), 800)
    m = sc.create(name, epochs=15, seed=0).fit(Xtr, ytr)
    (Xs, ys), (Xq, yq) = draw(range(8, 12), 20), draw(range(8, 12), 200)
    assert m.evaluate(Xq, yq, support=(Xs, ys))["accuracy"] > 0.85          # classes never seen in training
    one = np.array([np.flatnonzero(ys == c)[0] for c in range(8, 12)])      # one-shot: a single example per class
    assert m.evaluate(Xq, yq, support=(Xs[one], ys[one]))["accuracy"] > 0.7


# ---------------------------------------------------------------------- meta-learning ---
def test_maml_adapts_faster_than_scratch():
    scratch = sc.MAML(sc.MLP(hidden=(40, 40), seed=0).prepare(1, 1, "regression"))
    base = scratch.evaluate_tasks(sc.SineTasks(seed=99), steps=5, n=60)
    meta = sc.MAML(sc.MLP(hidden=(40, 40), seed=0).prepare(1, 1, "regression"), inner_lr=0.02, inner_steps=3)
    meta.meta_train(sc.SineTasks(seed=1), iterations=800, meta_batch=8)
    assert meta.evaluate_tasks(sc.SineTasks(seed=99), steps=5, n=60) < 0.3 * base


def test_first_order_maml_and_reptile_also_help():
    base = sc.MAML(sc.MLP(hidden=(40, 40), seed=0).prepare(1, 1, "regression")).evaluate_tasks(
        sc.SineTasks(seed=99), steps=5, n=60)
    fo = sc.MAML(sc.MLP(hidden=(40, 40), seed=0).prepare(1, 1, "regression"), first_order=True)
    fo.meta_train(sc.SineTasks(seed=1), iterations=800, meta_batch=8)
    rp = sc.Reptile(sc.MLP(hidden=(40, 40), seed=0).prepare(1, 1, "regression"))
    rp.meta_train(sc.SineTasks(seed=1), iterations=1500)
    for meta in (fo, rp):
        assert meta.evaluate_tasks(sc.SineTasks(seed=99), steps=5, n=60) < 0.6 * base, type(meta).__name__


def test_maml_classification_tasks():
    tasks_ = sc.GaussianTasks(d=8, n_way=3, seed=0)
    meta = sc.MAML(sc.MLP(hidden=(32,), seed=0).prepare(8, 3, "classification"), inner_lr=0.05, inner_steps=2)
    meta.meta_train(tasks_, iterations=300, meta_batch=8)
    t = sc.GaussianTasks(d=8, n_way=3, seed=5).sample(5, 30)
    adapted = meta.adapt(t.support_x, t.support_y, steps=3)
    assert adapted.evaluate(t.query_x, t.query_y)["accuracy"] > 0.7


# ---------------------------------------------------------------------------- transfer ---
def test_transfer_copies_and_freezes():
    r = np.random.default_rng(0); C = r.normal(size=(4, 6)) * 3
    def draw(n):
        y = r.integers(4, size=n); return (C[y] + r.normal(size=(n, 6))).astype(np.float32), y
    Xa, ya = draw(600)
    src = sc.MLP(hidden=(32, 32), epochs=40, seed=0).fit(Xa, ya)
    Xb, yb = draw(16); Xt, yt = draw(400)
    pair = lambda y: (y % 2 == 0).astype(int)                      # a new task built on the same features
    tgt = sc.transfer(src, sc.MLP(hidden=(32, 32), epochs=40, seed=0), output_dim=2,
                      task="classification", classes=[0, 1])
    before = {n: p.clone() for n, p in tgt.module.named_parameters()}
    tgt.fit(Xb, pair(yb))
    assert set(tgt.transferred_) == {"0.weight", "0.bias", "2.weight", "2.bias"}
    for n in tgt.transferred_:                                     # frozen layers must not move
        assert torch.equal(before[n], dict(tgt.module.named_parameters())[n])
    assert not torch.equal(before["4.weight"], tgt.module[4].weight)  # the new head did train
    assert tgt.evaluate(Xt, pair(yt))["accuracy"] > 0.8


# ----------------------------------------------------------------- curriculum / paced ---
def noisy_blobs():
    X, y = blobs(n=800, k=3, d=6, sep=2.0)
    ytr = y[:600].copy()
    r = np.random.default_rng(1)
    flip = r.random(600) < 0.25
    ytr[flip] = r.integers(3, size=flip.sum())
    return X[:600], ytr, X[600:], y[600:]


def test_curriculum_trains_on_growing_subsets():
    Xtr, ytr, Xte, yte = noisy_blobs()
    m = sc.fit_curriculum(sc.MLP(hidden=(32,), epochs=15, seed=0), Xtr, ytr, stages=4)
    assert m.curriculum_sizes_ == [150, 300, 450, 600]
    assert m.evaluate(Xte, yte)["accuracy"] > 0.9


def test_self_paced_leaves_out_hard_noisy_samples():
    Xtr, ytr, Xte, yte = noisy_blobs()
    m = sc.fit_self_paced(sc.MLP(hidden=(32,), seed=0), Xtr, ytr)
    frac = m.selected_fraction_
    assert frac[0] < frac[-1] and frac[-1] < 0.95                   # mislabelled samples stay excluded
    assert m.evaluate(Xte, yte)["accuracy"] > 0.9


def test_generative_replay_reduces_forgetting():
    (X1, y1), (X2, y2), _, _ = tasks()
    def run(strategies):
        m = sc.MLP(hidden=(32,), epochs=80, lr=1e-2, classes=[0, 1, 2, 3], strategies=strategies, seed=0)
        m.learn_task(X1, y1).learn_task(X2, y2)
        return m.evaluate(X1, y1)["accuracy"], m.evaluate(X2, y2)["accuracy"]
    naive_old, _ = run([])
    gen_old, gen_new = run([sc.GenerativeReplay()])
    assert gen_old > naive_old + 0.25 and gen_new > 0.8


# --------------------------------------------------------- predictive / energy-based ---
def regimes(n, seed, T=24):
    r = np.random.default_rng(seed); t = np.arange(T)
    y = r.integers(2, size=n)
    w, ph = np.where(y == 0, 0.3, 1.0)[:, None], r.uniform(0, 6.28, size=(n, 1))
    return (np.sin(w * t + ph) + 0.1 * r.normal(size=(n, T)))[:, :, None].astype(np.float32), y


def test_temporal_contrastive_learns_dynamics():
    X, y = regimes(600, 0); Xt, yt = regimes(200, 1)
    m = sc.TemporalContrastive(latent_dim=8, hidden=(32,), horizon=6, epochs=60, lr=5e-3, seed=0).fit(X)
    clf = sc.MLP(hidden=(16,), epochs=40, seed=0).fit(m.transform(X), y)
    assert clf.evaluate(m.transform(Xt), yt)["accuracy"] > 0.9


def test_latent_prediction_does_not_collapse():
    X = np.random.default_rng(0).normal(size=(500, 6)).astype(np.float32)
    X[:, 3:] = X[:, :3] * 0.8 + 0.2 * X[:, 3:]
    m = sc.LatentPrediction(latent_dim=4, epochs=50, seed=0).fit(X)
    assert m.history_["loss"][-1] < 0.3 * m.history_["loss"][0]
    assert m.transform(X).std(0).min() > 0.1


def test_energy_based_model_samples_the_data_distribution():
    r = np.random.default_rng(0)
    X = (r.normal(size=(800, 2)) * 0.7 + np.array([2.0, -1.0])).astype(np.float32)
    m = sc.EnergyBasedModel(epochs=40, lr=1e-3, seed=0).fit(X)
    s = m.sample(200)
    assert np.abs(s.mean(0) - X.mean(0)).max() < 0.5
    assert 0.6 < (s.std(0) / X.std(0)).min() and (s.std(0) / X.std(0)).max() < 1.6
