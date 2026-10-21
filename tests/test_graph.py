import numpy as np
import pytest

import saturncube as sc

r = np.random.default_rng(0)


def community_graph(N=200):
    c = r.integers(2, size=N)
    A = (r.random((N, N)) < np.where(c[:, None] == c[None, :], 0.15, 0.01)).astype(np.float32)
    A = np.triu(A, 1); A = A + A.T
    x = (c[:, None] * 1.0 + r.normal(size=(N, 4)) * 2).astype(np.float32)   # weak node features, informative edges
    return sc.Graph(x, A), c, r.random(N) < 0.3


@pytest.mark.parametrize("name", ["gcn", "gat", "graphsage", "gin", "mpnn", "graph_transformer", "ggnn",
                                  "graph_net_recurrent", "hypergraph"])
def test_node_classification(name):
    g, c, mask = community_graph()
    m = sc.create(name, epochs=80, lr=1e-2, seed=0).fit(g, c, train_mask=mask)
    assert m.evaluate(g, c, mask=~mask)["accuracy"] > 0.85, name


@pytest.mark.parametrize("name", ["gin", "mpnn", "ggnn", "graph_net_recurrent"])
def test_graph_classification_by_density(name):
    gs, ys = [], []
    for i in range(150):
        n = r.integers(8, 13)
        a = (r.random((n, n)) < (0.5 if i % 2 else 0.15)).astype(np.float32)
        a = np.triu(a, 1); a = a + a.T
        gs.append(sc.Graph(np.ones((n, 1), np.float32), a)); ys.append(i % 2)   # only structure carries the label
    m = sc.create(name, epochs=80, lr=1e-2, seed=0).fit(gs[:100], ys[:100])
    assert m.evaluate(gs[100:], ys[100:])["accuracy"] > 0.85, name


def test_edge_types_matter():
    N = 300
    src, typ = r.integers(N, size=N), r.integers(2, size=N)
    x = np.concatenate([np.ones((N, 1)), r.normal(size=(N, 3))], 1).astype(np.float32)
    g = sc.Graph.from_edges(x, np.stack([src, np.arange(N)], 1), undirected=False, edge_types=typ, n_relations=2)
    mk = r.random(N) < 0.5                                                   # label = type of the incoming edge
    acc = {}
    for name in ("gcn", "rgcn", "program_graph"):
        m = sc.create(name, epochs=120, lr=1e-2, seed=0).fit(g, typ, train_mask=mk)
        acc[name] = m.evaluate(g, typ, mask=~mk)["accuracy"]
    assert acc["rgcn"] > 0.9 and acc["program_graph"] > 0.9 and acc["gcn"] < 0.7, acc


def random_tree():
    n = r.integers(3, 8)
    ch = [[] for _ in range(n)]
    for i in range(1, n):
        ch[r.integers(i)].append(i)
    return sc.Tree(r.normal(size=(n, 2)).astype(np.float32), ch)


@pytest.mark.parametrize("name", ["tree_lstm", "tree_nn"])
def test_tree_models(name):
    trees = [random_tree() for _ in range(300)]
    y = np.array([int(t.x[:, 0].sum() > 0) for t in trees])
    m = sc.create(name, epochs=30, lr=1e-2, seed=0).fit(trees[:240], y[:240])
    assert m.evaluate(trees[240:], y[240:])["accuracy"] > 0.85


def test_pointer_network_learns_to_sort():
    X = r.random((1500, 5, 1)).astype(np.float32)
    Y = np.argsort(X[:, :, 0], 1)
    m = sc.PointerNetwork(epochs=40, lr=3e-3, seed=0, normalize=False).fit(X[:1200], Y[:1200])
    assert m.evaluate(X[1200:], Y[1200:])["accuracy"] > 0.85


def test_pointer_generator_copies():
    S = r.integers(10, size=(1000, 5))
    m = sc.PointerGenerator(vocab_size=10, epochs=40, lr=3e-3, seed=0).fit(S[:800], S[:800, ::-1].copy())
    assert m.evaluate(S[800:], S[800:, ::-1])["token_accuracy"] > 0.9
