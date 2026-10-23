"""Python-native program tools (the Python counterpart of the optional C tools in symbolic.py / verify.py).

* check_python / run_py: a restricted, step-limited evaluator for small generated Python functions,
* PythonSandbox: runs arbitrary Python in a separate process with CPU / memory / time limits,
* python_facts: Datalog facts (calls, assignments, reads) extracted with the standard `ast` module.

SECURITY NOTE: run_py is a convenience for checking *small generated arithmetic-style functions* quickly;
it is NOT a security boundary. PythonSandbox adds process-level limits but no network or filesystem
isolation. Run untrusted code in a container or VM."""
from __future__ import annotations

import ast
import json
import os
import resource
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional

from .symbolic import ParseError, StepLimitExceeded


class ExecutionError(Exception):
    pass


_ALLOWED = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.If, ast.For, ast.While, ast.Assign,
            ast.AugAssign, ast.Expr, ast.Pass, ast.Break, ast.Continue, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
            ast.IfExp, ast.Call, ast.Name, ast.Load, ast.Store, ast.Constant, ast.Tuple, ast.List, ast.Subscript,
            ast.Slice, ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or,
            ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.In, ast.NotIn)
_MAX_RANGE, _MAX_CONST = 1_000_000, 2 ** 63


def _safe_range(*a):
    r = range(*a)
    if len(r) > _MAX_RANGE:
        raise ExecutionError("range too long")
    return r


def _safe_sum(it):
    it = list(it) if not isinstance(it, range) else it
    if len(it) > _MAX_RANGE:
        raise ExecutionError("sequence too long")
    return sum(it)


_SAFE_BUILTINS = {"range": _safe_range, "abs": abs, "min": min, "max": max, "len": len, "sum": _safe_sum,
                  "int": int, "bool": bool, "sorted": sorted, "list": list}


def check_python(source):
    """Parse Python source and reject anything outside a small, side-effect-free subset (no imports,
    attributes, comprehensions, classes, lambdas, exponentiation, dunder names ...). Returns the AST."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise ParseError(str(e)) from None
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ParseError(f"{type(node).__name__} is not allowed")
        if isinstance(node, ast.Name) and "__" in node.id:
            raise ParseError("dunder names are not allowed")
        if isinstance(node, ast.Constant) and not (isinstance(node.value, (int, bool)) and abs(int(node.value)) < _MAX_CONST):
            raise ParseError("only small integer and boolean constants are allowed")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and (node.func.id in _SAFE_BUILTINS or node.func.id in defined)):
            raise ParseError("only calls to whitelisted builtins and functions defined in the source are allowed")
    return tree


def run_py(source, function, *args, max_steps=200_000):
    """Run `function(*args)` from `source` in the restricted evaluator with a step budget."""
    tree = check_python(source)
    code, counter = compile(tree, "<restricted>", "exec"), [0]

    def local(frame, event, arg):
        if event == "line":
            counter[0] += 1
            if counter[0] > max_steps:
                raise StepLimitExceeded("step limit exceeded")
        return local

    glob = {"__builtins__": dict(_SAFE_BUILTINS)}
    exec(code, glob)
    if function not in glob:
        raise ExecutionError(f"unknown function {function}")
    old = sys.gettrace()
    sys.settrace(lambda frame, event, arg: local if frame.f_code.co_filename == "<restricted>" else None)
    try:
        return glob[function](*args)
    except (StepLimitExceeded, ExecutionError):
        raise
    except RecursionError:
        raise ExecutionError("recursion too deep") from None
    except Exception as e:
        raise ExecutionError(f"{type(e).__name__}: {e}") from None
    finally:
        sys.settrace(old)


# ----------------------------------------------------------------- process sandbox ---
@dataclass
class PyRunResult:
    returncode: Optional[int]
    stdout: str
    stderr: str
    timed_out: bool = False
    signal_name: Optional[str] = None


class PythonSandbox:
    """Run Python source in a fresh interpreter (isolated mode, empty environment, temporary directory)
    with CPU, memory, output-size and wall-clock limits; the whole process group is killed on timeout."""

    def __init__(self, timeout=3.0, cpu_seconds=3, memory_mb=512):
        self.timeout, self.cpu, self.mem = timeout, cpu_seconds, memory_mb

    def _limits(self):
        def apply():
            resource.setrlimit(resource.RLIMIT_CPU, (self.cpu, self.cpu))
            resource.setrlimit(resource.RLIMIT_AS, (self.mem << 20, self.mem << 20))
            resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        return apply

    def run(self, source, stdin=""):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "prog.py")
            with open(path, "w") as f:
                f.write(source)
            proc = subprocess.Popen([sys.executable, "-I", path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
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
            return PyRunResult(proc.returncode, out, err, timed, sig)

    def call_cases(self, source, function, cases):
        """Call `function` on each argument tuple, one result line per case (so a hang or crash only
        loses the cases after it). Returns a list of ('ok', value) / ('error', message) / ('missing', None)."""
        harness = (f"\n\nimport json as _j, sys as _s\nfor _a in _j.loads(_s.stdin.read()):\n    try:\n"
                   f"        _r = {{'ok': {function}(*_a)}}\n    except Exception as _e:\n"
                   f"        _r = {{'error': type(_e).__name__ + ': ' + str(_e)}}\n    print(_j.dumps(_r), flush=True)\n")
        res = self.run(source + harness, json.dumps([list(a) for a, *_ in cases]))
        lines, out = res.stdout.splitlines(), []
        for i in range(len(cases)):
            if i < len(lines):
                d = json.loads(lines[i])
                out.append(("ok", d["ok"]) if "ok" in d else ("error", d["error"]))
            else:
                out.append(("missing", "timeout" if res.timed_out else f"crashed ({res.signal_name})" if res.signal_name else res.stderr.strip()[-120:] or "no output"))
        return out


# --------------------------------------------------------------------- code facts ---
def python_facts(source):
    """Datalog facts from Python source: func(f), param(f, x), call(f, g), assigns(f, x), reads(f, x)."""
    facts = set()
    for fn in (n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)):
        facts.add(("func", fn.name))
        facts |= {("param", fn.name, a.arg) for a in fn.args.args}
        for n in ast.walk(fn):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                facts.add(("call", fn.name, n.func.id))
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                facts.add(("assigns", fn.name, n.id))
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                facts.add(("reads", fn.name, n.id))
    return facts
