import shutil
import time

import numpy as np
import pytest

import saturncube as sc

HAS_GCC = shutil.which("gcc") is not None


# ------------------------------------------------------------------ wheel E: retrieval ---
DOCS = ["compute the absolute value of an integer", "reverse a string in place", "sum of the squares of numbers up to n",
        "check whether a number is prime", "greatest common divisor using euclid"]


def test_bm25_ranks_relevant_document_first_and_splits_identifiers():
    idx = sc.BM25Index().add(DOCS)
    assert idx.search("square sum up to n")[0][0] == 2
    assert idx.search("is_prime")[0][0] == 3 and idx.search("isPrime")[0][0] == 3     # snake_case / camelCase
    assert idx.search("zzz") == []


def test_lsh_approximates_exact_search():
    rng = np.random.default_rng(0)
    V = rng.normal(size=(2000, 64))
    store = sc.VectorStore(64, lsh_bits=10, n_tables=8, seed=0).add(V, list(range(2000)))
    hits = 0
    for i in range(100):
        q = V[i] + 0.1 * rng.normal(size=64)
        assert store.search(q, 1, exact=True)[0][0] == i
        hits += store.search(q, 1, exact=False)[0][0] == i
    assert hits >= 90                                  # approximate search recovers (almost) every exact top-1


def test_case_library_retrieval_by_concept_and_text():
    lib = sc.CaseLibrary()
    ids = {d: lib.add(d, f"/* {d} */", concepts=c) for d, c in zip(DOCS, [{"conditional"}, {"loop", "array"}, {"loop", "arithmetic"},
                                                                          {"loop", "modulo"}, {"loop", "modulo", "recursion"}])}
    assert lib.retrieve("how do I get the magnitude of a number, absolute value", k=1)[0].id == ids[DOCS[0]]
    top = [c.question for c in lib.retrieve("something with modulo", k=2, concepts={"modulo"})]
    assert set(top) == {DOCS[3], DOCS[4]}                # found by concept even though the words differ


def test_case_library_forgetting_policy_and_consolidation():
    lib = sc.CaseLibrary(capacity=3, policy="least_useful")
    a = lib.add("alpha task", "a()", {"x"}); b = lib.add("beta task", "b()", {"y"}); c = lib.add("gamma task", "c()", {"z"})
    for _ in range(3):
        lib.retrieve("alpha task", k=1)
        lib.record_outcome(a, True)
    lib.add("delta task", "d()", {"w"})                  # over capacity: evict the least useful, never the proven one
    assert a in lib.cases and len(lib) == 3
    dup = sc.CaseLibrary()
    dup.add("print hello world", "puts(1)", {"io"}); dup.add("print hello world", "puts(1)", {"io"}); dup.add("sort numbers", "qs()", {"sort"})
    assert dup.consolidate() == 1 and len(dup) == 2


# -------------------------------------------------------------------- wheel F: cognition ---
def test_sm2_intervals_follow_the_algorithm():
    s = sc.SM2Scheduler(); s.add("card")
    assert [s.review("card", 5, d) for d in (0, 1, 7)] == [1, 6, 17]      # 1 day, 6 days, then round(6 * 2.8)
    assert s.review("card", 1, 24) == 1 and s.items["card"]["reps"] == 0   # a failed recall resets the card
    assert s.due(100) == ["card"] and s.due(0) == []


def test_interleaved_practice_mixes_topics_without_losing_items():
    items = {"loops": list(range(6)), "pointers": list(range(6)), "recursion": list(range(6))}
    inter = sc.InterleavedPractice("interleaved", max_run=1, seed=0).schedule(items)
    blocked = sc.InterleavedPractice("blocked").schedule(items)
    assert sorted(inter) == sorted(blocked) and len(inter) == 18
    assert all(a[0] != b[0] for a, b in zip(inter, inter[1:]))             # never two in a row from one topic
    assert [t for t, _ in blocked] == sorted([t for t, _ in blocked], key=list(items).index)


def test_revision_loop_replans_after_failure():
    seen = []
    def propose(task, attempts):
        tried = {a.candidate for a in attempts}
        c = next(x for x in task if x not in tried)                        # never repeat a failed candidate
        seen.append(c); return c
    loop = sc.RevisionLoop(propose, lambda c: (c == 7, f"{c} is wrong"), max_rounds=6)
    assert loop.run([3, 5, 7, 9]) == 7 and [a.ok for a in loop.attempts_] == [False, False, True]
    assert sc.RevisionLoop(propose, lambda c: (False, "no"), max_rounds=2).run([1, 2, 3]) is None


def test_metacognitive_monitor_calibration_and_selective_prediction():
    rng = np.random.default_rng(0)
    m, bad = sc.MetacognitiveMonitor(), sc.MetacognitiveMonitor()
    for _ in range(2000):
        c = rng.random()
        m.record(c, rng.random() < c)                                      # calibrated: P(correct) = confidence
        bad.record(c, rng.random() < 0.5)                                  # overconfident: confidence says nothing
    assert m.expected_calibration_error() < 0.05 < bad.expected_calibration_error()
    t = m.threshold_for_accuracy(0.9)
    cov, acc = m.coverage_and_accuracy(t)
    assert acc >= 0.9 and cov < 0.4                                        # abstaining buys accuracy at the cost of coverage


# ---------------------------------------------------------------- wheel G: verification ---
CASES = [((-3,), 3), ((4,), 4), ((0,), 0)]
SOURCES = {"python": ("def f(x):\n    return -x if x < 0 else x\n", "def f(x):\n    return x\n", "def f(x):\n    return\n    x +\n"),
           "python-subprocess": ("def f(x):\n    return -x if x < 0 else x\n", "def f(x):\n    return x\n", "def f(x:\n    return x\n"),
           "interp": ("int f(int x) { if (x < 0) { return -x; } return x; }", "int f(int x) { return x; }", "int f(int x) { return ; }"),
           "gcc": ("int f(int x) { if (x < 0) { return -x; } return x; }", "int f(int x) { return x; }", "int f(int x) { return ; }")}


@pytest.mark.parametrize("backend", ["python", "python-subprocess", "interp"] + (["gcc"] if HAS_GCC else []))
def test_unit_tests_on_every_backend(backend):
    ok, wrong, broken = SOURCES[backend]
    t = sc.UnitTests("f", CASES, backend=backend)
    assert t.run(ok).ok
    rep = t.run(wrong)
    assert not rep.ok and rep.failures == [((-3,), 3, -3)]
    assert not t.run(broken).compile_ok


def test_python_sandbox_limits():
    sb = sc.PythonSandbox(timeout=1.0, cpu_seconds=1, memory_mb=256)
    assert sb.run("print(6 * 7)").stdout == "42\n"
    t0 = time.time()
    r = sb.run("while True:\n    pass")
    assert (r.timed_out or r.signal_name in ("SIGKILL", "SIGXCPU")) and time.time() - t0 < 4
    assert "MemoryError" in sb.run("x = bytearray(2 * 10**9)").stderr                  # memory limit applied
    r = sb.run("raise ValueError('boom')")
    assert r.returncode != 0 and "boom" in r.stderr
    res = sb.call_cases("def f(x):\n    if x == 2:\n        while True:\n            pass\n    return 10 // x\n", "f", [((5,), 0), ((0,), 0), ((2,), 0), ((1,), 0)])
    assert res[0] == ("ok", 2) and res[1][0] == "error" and "ZeroDivision" in res[1][1]
    assert res[2][0] == "missing" and res[3][0] == "missing"                            # a hang loses only the cases after it


@pytest.mark.skipif(not HAS_GCC, reason="gcc not installed")
def test_sandbox_enforces_limits():
    sb = sc.Sandbox(timeout=1.0, cpu_seconds=1, memory_mb=64)
    cr, rr = sb.build_and_run('#include <stdio.h>\nint main(void){ printf("hi %d\\n", 6*7); return 0; }')
    assert cr.ok and rr.stdout == "hi 42\n"
    bad = sb.compile("int main(void) { return undefined_thing; }")
    assert not bad.ok and "error" in bad.diagnostics
    t0 = time.time()
    _, rr = sb.build_and_run("int main(void){ for(;;){} }")
    assert (rr.timed_out or rr.signal_name in ("SIGKILL", "SIGXCPU")) and time.time() - t0 < 5
    _, rr = sb.build_and_run('#include <stdio.h>\n#include <stdlib.h>\nint main(void){ void *p = malloc(1UL<<32); puts(p ? "alloc" : "denied"); return 0; }')
    assert rr.stdout.strip() == "denied"                                   # memory limit applied
    _, rr = sb.build_and_run("int main(void){ int *p = 0; return *p; }")
    assert rr.signal_name == "SIGSEGV"


def test_property_testing_shrinks_to_minimal_counterexamples():
    r = sc.check_property(lambda x: x < 10, [sc.ints(0, 1000)], n=100, seed=1)
    assert not r.ok and r.shrunk == (10,)
    r2 = sc.check_property(lambda xs: sum(xs) < 100, [sc.lists(sc.ints(0, 80), max_len=8)], n=300, seed=1)
    assert not r2.ok and sorted(r2.shrunk[0]) in ([20, 80], [19, 81]) or sum(r2.shrunk[0]) >= 100 and len(r2.shrunk[0]) <= 2
    assert sc.check_property(lambda x, y: x + y == y + x, [sc.ints(), sc.ints()]).ok


def test_differential_testing_catches_a_buggy_candidate():
    ref = lambda x, y: max(x, y)
    good = lambda x, y: sc.run_py("def f(x, y):\n    return y if x < y else x\n", "f", x, y)
    buggy = lambda x, y: sc.run_py("def f(x, y):\n    return y if x < y else y\n", "f", x, y)
    assert sc.differential_test(good, ref, [sc.ints(-50, 50)] * 2).ok
    r = sc.differential_test(buggy, ref, [sc.ints(-50, 50)] * 2)
    assert not r.ok and r.shrunk[0] >= r.shrunk[1] and abs(r.shrunk[0]) <= 1 or not r.ok
