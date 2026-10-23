import numpy as np
import pytest

import saturncube as sc
from test_v03 import blobs


# ------------------------------------------------------------------ wheel H: generation ---
@pytest.mark.parametrize("language", ["python", "c"])
def test_grammar_generation_is_always_valid_and_runnable(language):
    grammar, run = (sc.PY_GRAMMAR, sc.run_py) if language == "python" else (sc.C_GRAMMAR, sc.run_c)
    check = sc.check_python if language == "python" else sc.parse
    gen, rng = sc.GrammarGenerator(grammar), np.random.default_rng(0)
    texts = set()
    for _ in range(150):
        src = gen.generate(rng, max_depth=6)
        check(src)                                             # syntactically valid by construction
        texts.add(src)
        for x, y in [(2, 3), (-1, 5), (0, 0)]:
            run(src, "f", x, y, max_steps=50_000)              # and runs without error
    assert len(texts) > 50                                     # varied programs, not one repeated output
    shallow = gen.generate(rng, max_depth=1)
    assert len(shallow) < 120                                  # the depth limit forces short derivations


def test_guided_generation_follows_the_chooser():
    gen = sc.GrammarGenerator(sc.C_GRAMMAR)
    always_atom = lambda nt, prods, depth: 0                   # a stand-in for a learned policy: first production always
    src = gen.generate(max_depth=5, chooser=always_atom)
    assert src == gen.generate(max_depth=5, chooser=always_atom) and sc.run_c(src, "f", 1, 2) is not None
    py = sc.GrammarGenerator(sc.PY_GRAMMAR).generate(max_depth=5, chooser=always_atom)
    assert sc.run_py(py, "f", 1, 2) is not None


def test_template_sketch_synthesis():
    t = sc.TemplateFiller("loop_sum")
    assert t.slots == ["init", "term"]
    examples = [((n,), sum(i * i for i in range(n))) for n in range(0, 7)]
    src = t.sketch_synthesize({"init": ["0", "1"], "term": ["i", "1", "i * i", "2 * i + 1"]}, examples)
    assert src is not None and "acc + i * i" in src
    assert t.sketch_synthesize({"init": ["5"], "term": ["i"]}, examples) is None
    with pytest.raises(sc.ParseError):
        t.fill(init="0", term="i +")                          # invalid fragment is rejected
    clamp = sc.TemplateFiller("clamp").fill(lo="0", hi="10")
    assert [sc.run_py(clamp, "f", v) for v in (-5, 4, 99)] == [0, 4, 10]
    c_clamp = sc.TemplateFiller("clamp", language="c").fill(lo="0", hi="10")                # optional C variant
    assert [sc.run_c(c_clamp, "f", v) for v in (-5, 4, 99)] == [0, 4, 10]


def test_bpe_round_trips_and_compresses_code():
    corpus = ["for (int i = 0; i < n; i++) { sum += arr[i]; }", "while (i < n) { i++; }", "int main(void) { return 0; }"] * 20
    tok = sc.BPETokenizer(num_merges=60).train(corpus)
    for text in corpus[:3] + ["héllo wörld ☃ unseen_identifier(42)", ""]:
        assert tok.decode(tok.encode(text)) == text           # exact round trip, even for unseen bytes
    text = corpus[0]
    assert len(tok.encode(text)) < 0.9 * len(sc.BPETokenizer(num_merges=0).encode(text))


# ----------------------------------------------------------------------- wheel I: design ---
def test_random_architectures_are_valid_reproducible_and_budgeted():
    for conn in ("erdos_renyi", "small_world", "sparse"):
        g = sc.RandomArchitectureGenerator(connectivity=conn, budget=9)
        specs = [g.generate(s) for s in range(30)]
        for s in specs:
            assert sc.validate(s, budget=9, input_type="vector") and s.cost() <= 9
        assert g.generate(7).config_hash() == g.generate(7).config_hash()           # same seed, same architecture
        assert len({s.config_hash() for s in specs}) > 20                            # different seeds differ
    spec = sc.RandomArchitectureGenerator().generate(3)
    assert sc.ArchitectureSpec.from_dict(spec.to_dict()).config_hash() == spec.config_hash()


def test_ablation_and_execution():
    spec = sc.RandomArchitectureGenerator(min_modules=5, max_modules=6, p=0.5, budget=14).generate(11)
    mid = spec.topological_order()[1]
    smaller = sc.ablate_module(spec, mid)
    assert mid not in smaller.modules and smaller.parent == spec.config_hash() and smaller.config_hash() != spec.config_hash()
    sc.validate(smaller)                                                              # still a valid graph
    chain = sc.ArchitectureSpec(0, {"a": {"kind": "mlp", "params": {}}, "b": {"kind": "mlp", "params": {}}}, [("a", "b")])
    out = sc.execute(chain, {"mlp": lambda x, p: x + 1}, 10)
    assert out == {"b": 12}
    bad = sc.ArchitectureSpec(0, {"a": {"kind": "mlp", "params": {}}, "b": {"kind": "gnn", "params": {}}}, [("a", "b")])
    with pytest.raises(ValueError):
        sc.validate(bad)                                                              # vector -> graph has no adapter


def test_latin_hypercube_is_stratified_and_factorial_is_complete():
    n = 20
    X = sc.latin_hypercube(n, [(0, 1), (10, 20), (-5, 5)], seed=0)
    for j, (lo, hi) in enumerate([(0, 1), (10, 20), (-5, 5)]):
        strata = np.floor((X[:, j] - lo) / (hi - lo) * n).astype(int)
        assert sorted(strata) == list(range(n))                                       # exactly one sample per stratum
    assert len(sc.full_factorial({"lr": [1, 2, 3], "opt": ["a", "b"]})) == 6


def test_statistics_detect_real_differences_and_not_noise():
    rng = np.random.default_rng(0)
    a, b, c = rng.normal(0, 1, 40), rng.normal(1.0, 1, 40), rng.normal(0, 1, 40)
    assert sc.permutation_test(a, b, seed=1) < 0.01 and sc.permutation_test(a, c, seed=1) > 0.05
    lo, hi = sc.bootstrap_ci(a, seed=1)
    assert lo < a.mean() < hi and hi - lo < 1.0
    assert sc.holm_correction([0.001, 0.04, 0.03, 0.2]) == [True, False, False, False]    # 0.04/0.03 fail after correction


# ------------------------------------------------------------------------ wheel J: classical ---
def test_random_forest_and_naive_bayes_classify():
    X, y = blobs(n=600, k=3, d=6, sep=2.5)
    for m in (sc.RandomForest(n_trees=30, seed=0), sc.NaiveBayes(), sc.DecisionTree(seed=0)):
        assert m.fit(X[:450], y[:450]).evaluate(X[450:], y[450:])["accuracy"] > 0.9, type(m).__name__
    p = sc.RandomForest(n_trees=20, seed=0).fit(X[:450], y[:450]).predict_proba(X[450:])
    assert np.allclose(p.sum(1), 1)


def test_random_forest_handles_xor_and_regression():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(800, 4)); y = ((X[:, 0] * X[:, 1]) > 0).astype(int)        # XOR-like: naive Bayes cannot represent it
    rf = sc.RandomForest(n_trees=40, seed=0).fit(X[:600], y[:600])
    assert rf.evaluate(X[600:], y[600:])["accuracy"] > 0.85 > sc.NaiveBayes().fit(X[:600], y[:600]).evaluate(X[600:], y[600:])["accuracy"]
    Xr = rng.uniform(-3, 3, size=(600, 3)); yr = np.sin(Xr[:, 0]) + Xr[:, 1] ** 2
    assert sc.RandomForest(n_trees=40, seed=0).fit(Xr[:450], yr[:450]).evaluate(Xr[450:], yr[450:])["r2"] > 0.85


def test_kmeans_recovers_clusters():
    X, y = blobs(n=450, k=3, d=4, sep=5.0)
    km = sc.KMeans(3, seed=0).fit(X)
    pred = km.predict(X)
    assert sum(np.bincount(y[pred == c]).max() for c in range(3)) / len(y) > 0.97
    assert km.transform(X).shape == (450, 3)


def test_classical_models_share_the_experiment_loop_with_neural_ones():
    X, y = blobs(n=600, k=3, d=6, sep=2.5)
    for name in ("mlp", "random_forest", "naive_bayes"):
        m = sc.create(name, seed=0) if name != "naive_bayes" else sc.create(name)
        assert m.fit(X[:450], y[:450]).evaluate(X[450:], y[450:])["accuracy"] > 0.9, name


@pytest.mark.parametrize("module", ["python_pipeline", "c_pipeline"])
def test_end_to_end_examples_run(module):
    import importlib, pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "examples"))
    src = importlib.import_module(module).main()
    run = sc.run_py if module == "python_pipeline" else sc.run_c
    assert src is not None and all(run(src, "f", a, b) == max(a, b) for a in range(-4, 5) for b in range(-4, 5))
