import numpy as np
import pytest

import saturncube as sc

rng = np.random.default_rng(0)


def blobs(n=300, k=3, d=6, sep=5.0, seed=0):
    r = np.random.default_rng(seed)
    centers = r.normal(size=(k, d)) * sep
    y = r.integers(k, size=n)
    return (centers[y] + r.normal(size=(n, d))).astype(np.float32), y


# ------------------------------------------------------------- unsupervised ---
@pytest.mark.parametrize("name", ["autoencoder", "denoising_ae", "vae", "masked", "contrastive", "byol"])
def test_unsupervised_trains_and_embeds(name):
    X, _ = blobs()
    m = sc.create(name, latent_dim=4, epochs=40, seed=0)
    m.fit(X)
    h = m.history_["loss"]
    assert np.isfinite(h).all() and np.mean(h[-5:]) < np.mean(h[:5]), name
    Z = m.transform(X)
    assert Z.shape == (len(X), 4) and np.isfinite(Z).all()
    assert np.isfinite(m.evaluate(X)["loss"])


def test_reconstruction_and_sampling():
    X, _ = blobs()
    ae = sc.Autoencoder(latent_dim=6, epochs=80, seed=0).fit(X)
    assert np.mean((ae.reconstruct(X) - X) ** 2) < 0.5 * X.var()
    vae = sc.VAE(latent_dim=4, epochs=20, seed=0).fit(X)
    assert vae.sample(5).shape == (5, X.shape[1])


def test_deep_clustering_recovers_blobs():
    X, y = blobs(n=300, k=3, sep=6.0)
    m = sc.DeepClustering(n_clusters=3, epochs=20, seed=0).fit(X)
    pred = m.predict(X)
    purity = sum(np.bincount(y[pred == c]).max() for c in np.unique(pred)) / len(y)
    assert purity > 0.8, purity


def test_unsupervised_save_load(tmp_path):
    X, _ = blobs()
    m = sc.BYOL(epochs=3, seed=0).fit(X)
    m.save(tmp_path / "b.pt")
    m2 = sc.BYOL.load(tmp_path / "b.pt")
    assert np.allclose(m.transform(X), m2.transform(X), atol=1e-5)


# --------------------------------------------------------------- hebbian ---
def correlated(n=1000, d=6, seed=0):
    r = np.random.default_rng(seed)
    v = r.uniform(0.5, 1.0, size=d)
    return (r.normal(size=(n, 1)) * v + 0.3 * r.normal(size=(n, d))).astype(np.float32)


def test_sanger_finds_top_principal_component():
    X = correlated()
    m = sc.HebbianLearner(rule="sanger", n_components=2, epochs=100, lr=0.05, batch_size=32, seed=0).fit(X)
    Z = (X - X.mean(0)) / X.std(0)
    pc1 = np.linalg.eigh(np.cov(Z.T))[1][:, -1]
    w = m.module.W[0].cpu().numpy()
    assert abs(w @ pc1) / np.linalg.norm(w) > 0.95


@pytest.mark.parametrize("rule", ["hebbian", "oja", "bcm", "competitive"])
def test_hebbian_rules_stay_finite(rule):
    X, _ = blobs()
    m = sc.HebbianLearner(rule=rule, n_components=3, epochs=10, seed=0).fit(X)
    assert np.isfinite(m.module.W.cpu().numpy()).all()
    assert m.transform(X).shape == (len(X), 3)


def test_competitive_learning_quantizes_blobs():
    X, _ = blobs(sep=6.0)
    m = sc.HebbianLearner(rule="competitive", n_components=3, epochs=20, lr=0.1, seed=0).fit(X)
    assert m.history_["loss"][-1] < 0.5 * X.var()


# ----------------------------------------------------- continual / replay etc ---
def tasks(seed=0):
    X, y = blobs(n=1200, k=4, d=6, sep=2.0, seed=seed)
    a, b = y < 2, y >= 2
    return (X[a], y[a]), (X[b], y[b]), X, y


def run(strats):
    (X1, y1), (X2, y2), X, y = tasks()
    m = sc.MLP(hidden=(32,), epochs=80, lr=1e-2, classes=[0, 1, 2, 3], strategies=strats, seed=0)
    m.learn_task(X1, y1)
    m.learn_task(X2, y2)
    return m.evaluate(X1, y1)["accuracy"], m.evaluate(X2, y2)["accuracy"]


def test_replay_reduces_forgetting():
    naive_old, naive_new = run([])
    replay_old, replay_new = run([sc.Replay(capacity=200)])
    assert replay_old > naive_old + 0.2, (naive_old, replay_old)
    assert replay_new > 0.7


@pytest.mark.parametrize("make", [lambda: sc.EWC(lam=2000), lambda: sc.Replay(capacity=200, prioritized=True)])
def test_other_strategies_reduce_forgetting_and_keep_learning(make):
    naive_old, _ = run([])
    old, new = run([make()])
    assert old > naive_old + 0.1 and new > 0.6, (naive_old, old, new)


def test_lwf_learns_new_task_without_breaking():
    # LwF is known to help little in class-incremental setups (new data has no old-class
    # examples), so only check that it trains stably and still learns the new task.
    old, new = run([sc.LwF(alpha=2.0)])
    assert np.isfinite([old, new]).all() and new > 0.6


def test_distillation_student_tracks_teacher():
    X, y = blobs(n=600, k=3, sep=2.0)
    teacher = sc.MLP(hidden=(64, 64), epochs=60, seed=0).fit(X, y)
    student = sc.distill(teacher, sc.MLP(hidden=(8,), epochs=60, seed=1), X)
    agree = (student.predict(X) == teacher.predict(X)).mean()
    assert agree > 0.9, agree


def test_active_learning_query_and_partial_fit():
    X, y = blobs(n=400, k=3, sep=1.5)
    m = sc.MLP(hidden=(16,), epochs=5, seed=0).fit(X[:30], y[:30])
    for strat in ["entropy", "margin", "least_confident", "random"]:
        idx = sc.query(m, X[30:], n=10, strategy=strat, seed=0)
        assert len(set(idx)) == 10
    n0 = len(m.history_["loss"])
    m.partial_fit(X[30:60], y[30:60])
    assert len(m.history_["loss"]) == n0 + 1


# ---------------------------------------------------------------------- RL ---
@pytest.mark.parametrize("cls", [sc.QLearning, sc.SARSA, sc.ExpectedSARSA, sc.MonteCarloControl])
def test_tabular_agents_solve_gridworld(cls):
    env = sc.GridWorld(size=4)
    agent = cls(alpha=0.2, epsilon=0.5, epsilon_decay=0.99, seed=0)
    agent.train(env, episodes=800)
    assert agent.evaluate(env) > 0.8  # optimal return is 0.94


def test_reward_shaping_preserves_solvability():
    env = sc.PotentialShaping(sc.GridWorld(size=4), lambda s: -((3 - s // 4) + (3 - s % 4)) * 0.1)
    agent = sc.QLearning(alpha=0.2, epsilon=0.5, epsilon_decay=0.99, seed=0)
    agent.train(env, episodes=500)
    assert agent.evaluate(sc.GridWorld(size=4)) > 0.8


@pytest.mark.parametrize("make", [lambda: sc.UCB1(4, seed=0), lambda: sc.ThompsonBandit(4, seed=0),
                                   lambda: sc.EpsilonGreedyBandit(4, 0.1, seed=0)])
def test_bandits_find_best_arm(make):
    env = sc.BernoulliBandit([0.1, 0.2, 0.8, 0.3], seed=0)
    b = make()
    r = sc.run_bandit(b, env, steps=2000)
    assert b.counts.argmax() == 2 and r[-500:].mean() > 0.6


def test_linucb_learns_context_dependent_best_arm():
    r = np.random.default_rng(0)
    b, total, n = sc.LinUCB(n_arms=2, n_features=2, alpha=0.3, seed=0), 0.0, 1500
    for _ in range(n):
        x = r.normal(size=2)
        arm = b.select(x)
        reward = float((arm == 0) == (x[0] > 0))  # best arm depends on the sign of x[0]
        b.update(arm, reward, x)
        total += reward
    assert total / n > 0.7
