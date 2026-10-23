"""Wheel G: verification. Sandboxed compile-and-run for real C, unit testing (against the safe
interpreter or gcc), and property-based testing with shrinking (test-case minimization).

SECURITY NOTE: the sandbox applies CPU, memory, file-size and wall-clock limits, runs in a throw-away
directory with an empty environment, and kills the whole process group on timeout. It is NOT a
security boundary (no network or filesystem isolation). Run untrusted code in a container or VM."""
from __future__ import annotations

import os
import resource
import shutil
import signal
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .base import register
from .pycode import ExecutionError, PythonSandbox, check_python, run_py
from .symbolic import CRuntimeError, ParseError, StepLimitExceeded, run_c, parse


@dataclass
class CompileResult:
    ok: bool
    diagnostics: str
    warnings: int = 0


@dataclass
class RunResult:
    returncode: Optional[int]
    stdout: str
    stderr: str
    timed_out: bool = False
    signal_name: Optional[str] = None


@register("sandbox", family="verification")
class Sandbox:
    """Compile C with gcc and run the result under resource limits."""

    def __init__(self, timeout=2.0, cpu_seconds=2, memory_mb=256, compiler="gcc", flags=("-O0", "-Wall", "-Wextra", "-Werror=return-type", "-Werror=implicit-function-declaration")):
        if shutil.which(compiler) is None:
            raise RuntimeError(f"{compiler} not found")
        self.timeout, self.cpu, self.mem, self.compiler, self.flags = timeout, cpu_seconds, memory_mb, compiler, tuple(flags)

    def _limits(self):
        def apply():
            resource.setrlimit(resource.RLIMIT_CPU, (self.cpu, self.cpu))
            resource.setrlimit(resource.RLIMIT_AS, (self.mem << 20, self.mem << 20))
            resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        return apply

    def _compile(self, tmp, source):
        path = os.path.join(tmp, "prog.c")
        with open(path, "w") as f:
            f.write(source)
        p = subprocess.run([self.compiler, *self.flags, "-o", os.path.join(tmp, "prog"), path], capture_output=True, text=True,
                           timeout=30, cwd=tmp, env={"PATH": "/usr/bin:/bin", "LANG": "C"})
        return CompileResult(p.returncode == 0, p.stderr.strip(), p.stderr.count("warning:"))

    def compile(self, source):
        """Compile only (syntax/type check); the binary is discarded."""
        with tempfile.TemporaryDirectory() as tmp:
            return self._compile(tmp, source)

    def build_and_run(self, source, stdin="", args=()):
        with tempfile.TemporaryDirectory() as tmp:
            cr = self._compile(tmp, source)
            if not cr.ok:
                return cr, None
            proc = subprocess.Popen([os.path.join(tmp, "prog"), *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True, cwd=tmp, env={}, preexec_fn=self._limits(),
                                    start_new_session=True)
            try:
                out, err = proc.communicate(stdin, timeout=self.timeout)
                timed = False
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                out, err = proc.communicate()
                timed = True
            sig = signal.Signals(-proc.returncode).name if proc.returncode is not None and proc.returncode < 0 else None
            return cr, RunResult(proc.returncode, out, err, timed, sig)


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
    """Unit tests for one function. cases: [(args_tuple, expected)]. Backends:
    'python' (default) restricted in-process evaluator for small generated functions; 'python-subprocess'
    runs the Python source in a resource-limited child process (any Python allowed);
    'interp' / 'gcc' are the optional C backends (mini-C interpreter / real gcc in the sandbox)."""

    def __init__(self, function, cases, backend="python", sandbox=None):
        if backend not in ("python", "python-subprocess", "interp", "gcc"):
            raise ValueError("backend must be 'python', 'python-subprocess', 'interp' or 'gcc'")
        self.fn, self.cases, self.backend = function, list(cases), backend
        self.sandbox = sandbox or (Sandbox() if backend == "gcc" else PythonSandbox() if backend == "python-subprocess" else None)

    def run(self, source):
        if self.backend == "python":
            try:
                check_python(source)
            except ParseError as e:
                return TestReport(0, len(self.cases), compile_ok=False, message=str(e))
            fails = []
            for args, exp in self.cases:
                try:
                    got = run_py(source, self.fn, *args)
                except (ExecutionError, StepLimitExceeded) as e:
                    got = f"error: {e}"
                if got != exp:
                    fails.append((args, exp, got))
            return TestReport(len(self.cases) - len(fails), len(self.cases), fails)
        if self.backend == "python-subprocess":
            import ast
            try:
                ast.parse(source)
            except SyntaxError as e:
                return TestReport(0, len(self.cases), compile_ok=False, message=str(e))
            fails = []
            for (args, exp), (kind, val) in zip(self.cases, self.sandbox.call_cases(source, self.fn, self.cases)):
                got = val if kind == "ok" else f"{kind}: {val}"
                if got != exp:
                    fails.append((args, exp, got))
            return TestReport(len(self.cases) - len(fails), len(self.cases), fails)
        if self.backend == "interp":
            try:
                parse(source)
            except ParseError as e:
                return TestReport(0, len(self.cases), compile_ok=False, message=str(e))
            fails = []
            for args, exp in self.cases:
                try:
                    got = run_c(source, self.fn, *args)
                except (CRuntimeError, StepLimitExceeded) as e:
                    got = f"error: {e}"
                if got != exp:
                    fails.append((args, exp, got))
            return TestReport(len(self.cases) - len(fails), len(self.cases), fails)
        calls = "\n".join(f'    printf("%d\\n", {self.fn}({", ".join(map(str, a))}));' for a, _ in self.cases)
        cr, rr = self.sandbox.build_and_run(f"#include <stdio.h>\n{source}\nint main(void) {{\n{calls}\n    return 0;\n}}\n")
        if not cr.ok:
            return TestReport(0, len(self.cases), compile_ok=False, message=cr.diagnostics)
        lines = rr.stdout.split()
        fails = []
        for i, (args, exp) in enumerate(self.cases):
            got = int(lines[i]) if i < len(lines) else (f"crashed ({rr.signal_name})" if rr.signal_name else "timeout" if rr.timed_out else "no output")
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
