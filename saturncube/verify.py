"""Verification: unit testing of generated functions (restricted in-process evaluator or a resource-limited
child process) and property-based testing with shrinking (test-case minimization).

SECURITY NOTE: the Python backends limit steps, CPU, memory and time but are NOT a security boundary (no network or
filesystem isolation). Run untrusted code in a container or VM."""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .base import register
from .pycode import ExecutionError, PythonSandbox, check_python, run_py
from .symbolic import ParseError, StepLimitExceeded


@dataclass
class TestReport:
    passed: int
    total: int
    failures: List[tuple] = field(default_factory=list)   # (args, expected, got-or-error)
    compile_ok: bool = True
    message: str = ""

    @property
    def ok(self):
        return self.compile_ok and self.passed == self.total


@register("unit_tests", family="verification")
class UnitTests:
    """Unit tests for one function. cases: [(args_tuple, expected)]. Backends: 'python' (default) restricted
    in-process evaluator for small generated functions; 'python-subprocess' runs the source in a
    resource-limited child process (any Python is allowed there)."""

    def __init__(self, function, cases, backend="python", sandbox=None):
        if backend not in ("python", "python-subprocess"):
            raise ValueError("backend must be 'python' or 'python-subprocess'")
        self.fn, self.cases, self.backend = function, list(cases), backend
        self.sandbox = sandbox or (PythonSandbox() if backend == "python-subprocess" else None)

    def run(self, source):
        fails = []
        if self.backend == "python":
            try:
                check_python(source)
            except ParseError as e:
                return TestReport(0, len(self.cases), compile_ok=False, message=str(e))
            for args, exp in self.cases:
                try:
                    got = run_py(source, self.fn, *args)
                except (ExecutionError, StepLimitExceeded) as e:
                    got = f"error: {e}"
                if got != exp:
                    fails.append((args, exp, got))
            return TestReport(len(self.cases) - len(fails), len(self.cases), fails)
        try:
            ast.parse(source)
        except SyntaxError as e:
            return TestReport(0, len(self.cases), compile_ok=False, message=str(e))
        for (args, exp), (kind, val) in zip(self.cases, self.sandbox.call_cases(source, self.fn, self.cases)):
            got = val if kind == "ok" else f"{kind}: {val}"
            if got != exp:
                fails.append((args, exp, got))
        return TestReport(len(self.cases) - len(fails), len(self.cases), fails)


# -------------------------------------------------------------- property-based testing ---
class IntGen:
    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi

    def sample(self, rng):
        return int(rng.integers(self.lo, self.hi + 1))

    def shrink(self, v):
        cands = [0, v // 2, v - 1 if v > 0 else v + 1, abs(v)]
        return [c for c in dict.fromkeys(cands) if c != v and self.lo <= c <= self.hi and abs(c) <= abs(v)]


class ListGen:
    def __init__(self, elem, max_len=8):
        self.elem, self.max_len = elem, max_len

    def sample(self, rng):
        return [self.elem.sample(rng) for _ in range(int(rng.integers(0, self.max_len + 1)))]

    def shrink(self, v):
        out = [v[:i] + v[i + 1:] for i in range(len(v))]
        out += [v[:i] + [s] + v[i + 1:] for i in range(len(v)) for s in self.elem.shrink(v[i])]
        return out


def ints(lo=-100, hi=100):
    return IntGen(lo, hi)


def lists(elem, max_len=8):
    return ListGen(elem, max_len)


@dataclass
class PropertyResult:
    ok: bool
    tests_run: int
    counterexample: Optional[tuple] = None
    shrunk: Optional[tuple] = None


def check_property(prop, gens, n=200, seed=0, max_shrinks=2000):
    """Run prop(*args) on n random inputs. On failure, shrink the counterexample to a minimal one."""
    rng = np.random.default_rng(seed)

    def fails(args):
        try:
            return not prop(*args)
        except Exception:
            return True

    for i in range(n):
        args = tuple(g.sample(rng) for g in gens)
        if fails(args):
            orig, steps, improved = args, 0, True
            while improved and steps < max_shrinks:
                improved = False
                for k, g in enumerate(gens):
                    for cand in g.shrink(args[k]):
                        steps += 1
                        trial = args[:k] + (cand,) + args[k + 1:]
                        if fails(trial):
                            args, improved = trial, True
                            break
                    if improved:
                        break
            return PropertyResult(False, i + 1, orig, args)
    return PropertyResult(True, n)


def differential_test(candidate, reference, gens, n=300, seed=0):
    """Property: the candidate agrees with a trusted reference on random inputs."""
    return check_property(lambda *a: candidate(*a) == reference(*a), gens, n, seed)
