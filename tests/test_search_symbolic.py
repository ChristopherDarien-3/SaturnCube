import itertools

import numpy as np
import pytest

import saturncube as sc


# -------------------------------------------------------------------------- search ---
def test_genetic_algorithm_float_and_discrete():
    ga = sc.GeneticAlgorithm(lambda g: -sum(x * x for x in g), [("float", -5, 5)] * 4, generations=60, seed=0)
    best, fit = ga.run()
    assert fit > -0.05 and ga.history_[-1] >= ga.history_[0]
    onemax = sc.GeneticAlgorithm(lambda g: sum(g), [("int", 0, 1)] * 40, generations=80, seed=0)
    assert onemax.run()[1] >= 38
    pick = sc.GeneticAlgorithm(lambda g: {"a": 1, "b": 5, "c": 2}[g[0]] + g[1], [("choice", ["a", "b", "c"]), ("int", 0, 9)], seed=0)
    assert pick.run()[0] == ["b", 9]


def test_successive_halving_saves_budget_and_finds_good_config():
    def evaluate(cfg, budget):                      # noisy score that sharpens with budget
        r = np.random.default_rng(int(cfg * 1e6) + budget)
        return -abs(cfg - 0.3) + r.normal() * 0.2 / budget
    sampler = lambda rng: float(rng.random())
    sh = sc.SuccessiveHalving(sampler, evaluate, n_configs=27, min_budget=1, eta=3, seed=0)
    best, _ = sh.run()
    rs = sc.RandomSearch(sampler, evaluate, n_trials=27, budget=27, seed=0)
    rs.run()
    assert sh.total_budget_ < 0.2 * rs.total_budget_                  # far cheaper than evaluating everything at full budget
    assert abs(best - 0.3) < 0.25 and sh.rounds_[0] == (27, 1) and sh.rounds_[-1][0] == 1


def test_simulated_annealing_solves_small_tsp():
    rng = np.random.default_rng(1)
    pts = rng.random((8, 2))
    length = lambda t: sum(np.linalg.norm(pts[t[i]] - pts[t[(i + 1) % 8]]) for i in range(8))
    def swap(t, r):
        i, j = r.choice(8, 2, replace=False); t = list(t); t[i], t[j] = t[j], t[i]; return tuple(t)
    best, e = sc.SimulatedAnnealing(length, swap, tuple(range(8)), t0=1.0, cooling=0.998, steps=6000, seed=0).run()
    optimum = min(length((0,) + p) for p in itertools.permutations(range(1, 8)))
    assert e <= 1.05 * optimum


# ------------------------------------------------------------ restricted evaluator ---
def test_run_py_semantics_and_restrictions():
    fib = "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n"
    assert sc.run_py(fib, "fib", 12) == 144
    loop = "def f(n):\n    acc = 0\n    for i in range(n):\n        acc += i * i\n    return acc\n"
    assert sc.run_py(loop, "f", 6) == 55 and sc.run_py("def f(a, b):\n    return a // b, a % b\n", "f", 7, 2) == (3, 1)
    with pytest.raises(sc.StepLimitExceeded):
        sc.run_py("def f(x):\n    while True:\n        x = x + 1\n    return x\n", "f", 0, max_steps=3000)
    with pytest.raises(sc.ExecutionError):
        sc.run_py("def f(x):\n    return 1 // x\n", "f", 0)
    for bad in ["import os\ndef f(x):\n    return x\n", "def f(x):\n    return x.real\n", "def f(x):\n    return x ** 9\n",
                "def f(x):\n    return open('a')\n", "def f(x):\n    return [i for i in range(3)]\n",
                "def f(x):\n    return x.__class__\n", "def f(x:\n    return x\n"]:
        with pytest.raises(sc.ParseError):
            sc.check_python(bad)


# ------------------------------------------------------------ enumerative synthesis ---
def synth(fn, n_vars=2, **kw):
    ex = [(v, fn(*v)) for v in itertools.product(range(-3, 4), repeat=n_vars)]
    s = sc.EnumerativeSynthesizer(["x", "y"][:n_vars], **kw)
    return s, s.synthesize(ex), ex


@pytest.mark.parametrize("name,fn,nv", [("sum_sq", lambda x, y: x * x + y * y, 2), ("max", max, 2),
                                         ("abs", abs, 1), ("poly", lambda x: x * x * x - x, 1)])
def test_enumerative_synthesis_finds_programs(name, fn, nv):
    s, expr, ex = synth(fn, nv, max_size=9)
    assert expr is not None, name
    src = s.to_function(expr)
    for args, out in ex:                                           # verify the generated code by running it
        assert sc.run_py(src, "f", *args) == out


def test_synthesis_returns_none_when_budget_too_small_and_validates_ops():
    _, expr, _ = synth(lambda x, y: x * x + y * y + 7 * x * y, 2, max_size=3)
    assert expr is None
    with pytest.raises(ValueError):
        sc.EnumerativeSynthesizer(["x"], ops=("+", "**"))


# ----------------------------------------------------------------------------- Datalog ---
def test_datalog_transitive_closure():
    d = sc.Datalog()
    for a, b in [("a", "b"), ("b", "c"), ("c", "d")]:
        d.add_fact("parent", a, b)
    d.add_rule(("anc", "X", "Y"), ("parent", "X", "Y"))
    d.add_rule(("anc", "X", "Z"), ("parent", "X", "Y"), ("anc", "Y", "Z"))
    assert d.run() == 6                                               # 3 + 2 + 1 ancestor facts
    assert {b["Y"] for b in d.query("anc", "a", "Y")} == {"b", "c", "d"}


def test_python_facts_feed_datalog():
    src = "def c(x):\n    return x\ndef b(x):\n    return c(x) + 1\ndef a(x):\n    t = b(x)\n    return t\n"
    g = sc.Datalog()
    g.facts |= sc.python_facts(src)
    for head, body in sc.REACHABILITY_RULES:
        g.add_rule(head, *body)
    g.run()
    assert {b["Y"] for b in g.query("reaches", "a", "Y")} == {"b", "c"} and ("assigns", "a", "t") in g.facts
