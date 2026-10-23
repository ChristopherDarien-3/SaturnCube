"""Wheel D: symbolic reasoning and program synthesis.

* a recursive-descent parser, AST and safe step-limited interpreter for a small C subset
  (int only: functions, if/else, while, for, calls, 32-bit wraparound arithmetic),
* enumerative program synthesis from input/output examples (bottom-up, with observational equivalence),
* a Datalog-style forward-chaining rule engine (with helpers to extract facts from a program).
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Optional, Tuple

from .base import register


# ----------------------------------------------------------------------------- AST ---
@dataclass(frozen=True)
class Num: v: int
@dataclass(frozen=True)
class Var: name: str
@dataclass(frozen=True)
class BinOp: op: str; l: object; r: object
@dataclass(frozen=True)
class UnOp: op: str; e: object
@dataclass(frozen=True)
class Cond: c: object; a: object; b: object
@dataclass(frozen=True)
class Call: name: str; args: tuple
@dataclass(frozen=True)
class Decl: name: str; init: Optional[object]
@dataclass(frozen=True)
class Assign: name: str; op: str; e: object
@dataclass(frozen=True)
class If: c: object; then: object; other: Optional[object]
@dataclass(frozen=True)
class While: c: object; body: object
@dataclass(frozen=True)
class For: init: object; c: object; step: object; body: object
@dataclass(frozen=True)
class Return: e: object
@dataclass(frozen=True)
class ExprStmt: e: object
@dataclass(frozen=True)
class Block: stmts: tuple
@dataclass(frozen=True)
class Function: name: str; params: tuple; body: Block
@dataclass(frozen=True)
class Program: functions: tuple


class ParseError(Exception):
    pass


class StepLimitExceeded(Exception):
    pass


class CRuntimeError(Exception):
    pass


# ------------------------------------------------------------------------- lexer/parser ---
_TOKEN = re.compile(r"\s*(?:(\d+)|([A-Za-z_]\w*)|(<=|>=|==|!=|&&|\|\||\+=|-=|\*=|\+\+|--|[-+*/%<>=!(){};,?:]))")
_KEYWORDS = {"int", "if", "else", "while", "for", "return"}


def tokenize(src):
    src = re.sub(r"//[^\n]*|/\*.*?\*/", " ", src, flags=re.S)
    pos, out = 0, []
    while pos < len(src):
        if src[pos:].strip() == "":
            break
        m = _TOKEN.match(src, pos)
        if not m:
            raise ParseError(f"unexpected character {src[pos:].lstrip()[:1]!r}")
        num, ident, op = m.groups()
        out.append(("num", int(num)) if num else ("kw" if ident in _KEYWORDS else "id", ident) if ident else ("op", op))
        pos = m.end()
    return out + [("eof", None)]


_BIN_PREC = [["||"], ["&&"], ["==", "!="], ["<", ">", "<=", ">="], ["+", "-"], ["*", "/", "%"]]


class _Parser:
    def __init__(self, tokens):
        self.t, self.i = tokens, 0

    def peek(self, k=0):
        return self.t[self.i + k]

    def take(self):
        tok = self.t[self.i]
        self.i += 1
        return tok

    def expect(self, kind, val=None):
        tok = self.take()
        if tok[0] != kind or (val is not None and tok[1] != val):
            raise ParseError(f"expected {val or kind}, found {tok[1]!r}")
        return tok[1]

    def at(self, kind, val):
        return self.peek() == (kind, val)

    # expressions (precedence climbing)
    def expr(self):
        c = self.binary(0)
        if self.at("op", "?"):
            self.take(); a = self.expr(); self.expect("op", ":"); b = self.expr()
            return Cond(c, a, b)
        return c

    def binary(self, level):
        if level == len(_BIN_PREC):
            return self.unary()
        left = self.binary(level + 1)
        while self.peek()[0] == "op" and self.peek()[1] in _BIN_PREC[level]:
            op = self.take()[1]
            left = BinOp(op, left, self.binary(level + 1))
        return left

    def unary(self):
        if self.peek()[0] == "op" and self.peek()[1] in ("-", "!"):
            op = self.take()[1]
            return UnOp(op, self.unary())
        return self.primary()

    def primary(self):
        kind, val = self.take()
        if kind == "num":
            return Num(val)
        if kind == "id":
            if self.at("op", "("):
                self.take(); args = []
                while not self.at("op", ")"):
                    args.append(self.expr())
                    if self.at("op", ","):
                        self.take()
                self.expect("op", ")")
                return Call(val, tuple(args))
            return Var(val)
        if (kind, val) == ("op", "("):
            e = self.expr(); self.expect("op", ")")
            return e
        raise ParseError(f"unexpected token {val!r}")

    # statements
    def simple_stmt(self):
        if self.at("kw", "int"):
            self.take(); name = self.expect("id")
            init = None
            if self.at("op", "="):
                self.take(); init = self.expr()
            return Decl(name, init)
        if self.peek()[0] == "id" and self.peek(1)[0] == "op" and self.peek(1)[1] in ("=", "+=", "-=", "*=", "++", "--"):
            name, op = self.take()[1], self.take()[1]
            if op in ("++", "--"):
                return Assign(name, "+=" if op == "++" else "-=", Num(1))
            return Assign(name, op, self.expr())
        return ExprStmt(self.expr())

    def block(self):
        self.expect("op", "{"); stmts = []
        while not self.at("op", "}"):
            stmts.append(self.stmt())
        self.take()
        return Block(tuple(stmts))

    def body(self):
        return self.block() if self.at("op", "{") else Block((self.stmt(),))

    def stmt(self):
        if self.at("kw", "return"):
            self.take(); e = self.expr(); self.expect("op", ";")
            return Return(e)
        if self.at("kw", "if"):
            self.take(); self.expect("op", "("); c = self.expr(); self.expect("op", ")")
            then, other = self.body(), None
            if self.at("kw", "else"):
                self.take(); other = self.body()
            return If(c, then, other)
        if self.at("kw", "while"):
            self.take(); self.expect("op", "("); c = self.expr(); self.expect("op", ")")
            return While(c, self.body())
        if self.at("kw", "for"):
            self.take(); self.expect("op", "(")
            init = None if self.at("op", ";") else self.simple_stmt(); self.expect("op", ";")
            c = None if self.at("op", ";") else self.expr(); self.expect("op", ";")
            step = None if self.at("op", ")") else self.simple_stmt(); self.expect("op", ")")
            return For(init, c, step, self.body())
        if self.at("op", "{"):
            return self.block()
        s = self.simple_stmt(); self.expect("op", ";")
        return s

    def function(self):
        self.expect("kw", "int"); name = self.expect("id"); self.expect("op", "(")
        params = []
        while not self.at("op", ")"):
            self.expect("kw", "int"); params.append(self.expect("id"))
            if self.at("op", ","):
                self.take()
        self.expect("op", ")")
        return Function(name, tuple(params), self.block())


def parse(src):
    """Parse a program (one or more `int name(int a, ...) { ... }` functions) into an AST."""
    p = _Parser(tokenize(src))
    fns = []
    while p.peek()[0] != "eof":
        fns.append(p.function())
    return Program(tuple(fns))


def parse_expr(src):
    p = _Parser(tokenize(src))
    e = p.expr()
    p.expect("eof")
    return e


# ------------------------------------------------------------------------ AST utilities ---
def walk(node):
    """Depth-first traversal yielding every AST node."""
    yield node
    if isinstance(node, tuple):
        for x in node:
            yield from walk(x)
    elif hasattr(node, "__dataclass_fields__"):
        for f in node.__dataclass_fields__:
            v = getattr(node, f)
            if isinstance(v, tuple) or hasattr(v, "__dataclass_fields__"):
                yield from walk(v)


def tree_size(node):
    return sum(1 for n in walk(node) if hasattr(n, "__dataclass_fields__"))


_PREC = {"||": 1, "&&": 2, "==": 3, "!=": 3, "<": 4, ">": 4, "<=": 4, ">=": 4, "+": 5, "-": 5, "*": 6, "/": 6, "%": 6}


def to_c(node, indent=0):
    """Pretty-print an expression, statement or function as C source."""
    pad = "    " * indent
    if isinstance(node, Num):
        return str(node.v)
    if isinstance(node, Var):
        return node.name
    if isinstance(node, BinOp):
        def side(x, right):
            s = to_c(x)
            if isinstance(x, Cond) or (isinstance(x, BinOp) and (_PREC[x.op] < _PREC[node.op] or (right and _PREC[x.op] == _PREC[node.op]))):
                return f"({s})"
            return s
        return f"{side(node.l, False)} {node.op} {side(node.r, True)}"
    if isinstance(node, UnOp):
        s = to_c(node.e)
        return f"{node.op}({s})" if isinstance(node.e, (BinOp, Cond)) else f"{node.op}{s}"
    if isinstance(node, Cond):
        return f"{to_c(node.c)} ? {to_c(node.a)} : {to_c(node.b)}"
    if isinstance(node, Call):
        return f"{node.name}({', '.join(to_c(a) for a in node.args)})"
    if isinstance(node, Decl):
        return f"{pad}int {node.name}" + (f" = {to_c(node.init)}" if node.init else "") + ";"
    if isinstance(node, Assign):
        return f"{pad}{node.name} {node.op} {to_c(node.e)};"
    if isinstance(node, Return):
        return f"{pad}return {to_c(node.e)};"
    if isinstance(node, ExprStmt):
        return f"{pad}{to_c(node.e)};"
    if isinstance(node, Block):
        return "\n".join(to_c(s, indent) for s in node.stmts)
    if isinstance(node, If):
        s = f"{pad}if ({to_c(node.c)}) {{\n{to_c(node.then, indent + 1)}\n{pad}}}"
        return s + (f" else {{\n{to_c(node.other, indent + 1)}\n{pad}}}" if node.other else "")
    if isinstance(node, While):
        return f"{pad}while ({to_c(node.c)}) {{\n{to_c(node.body, indent + 1)}\n{pad}}}"
    if isinstance(node, For):
        strip = lambda s: to_c(s).strip().rstrip(";") if s else ""
        return f"{pad}for ({strip(node.init)}; {to_c(node.c) if node.c else ''}; {strip(node.step)}) {{\n{to_c(node.body, indent + 1)}\n{pad}}}"
    if isinstance(node, Function):
        return f"int {node.name}({', '.join('int ' + p for p in node.params)}) {{\n{to_c(node.body, 1)}\n}}"
    if isinstance(node, Program):
        return "\n\n".join(to_c(f) for f in node.functions)
    raise TypeError(f"cannot print {type(node).__name__}")


# -------------------------------------------------------------------------- interpreter ---
def wrap32(v):
    return ((v + 2 ** 31) % 2 ** 32) - 2 ** 31


def _div(a, b):
    if b == 0:
        raise CRuntimeError("division by zero")
    q = abs(a) // abs(b)
    return wrap32(q if (a >= 0) == (b >= 0) else -q)


_OPS = {"+": lambda a, b: wrap32(a + b), "-": lambda a, b: wrap32(a - b), "*": lambda a, b: wrap32(a * b),
        "/": _div, "%": lambda a, b: wrap32(a - _div(a, b) * b),
        "<": lambda a, b: int(a < b), ">": lambda a, b: int(a > b), "<=": lambda a, b: int(a <= b),
        ">=": lambda a, b: int(a >= b), "==": lambda a, b: int(a == b), "!=": lambda a, b: int(a != b)}


class _ReturnSignal(Exception):
    def __init__(self, v):
        self.v = v


class Interpreter:
    """Tree-walking interpreter with 32-bit wraparound, a step budget and a call-depth limit,
    so untrusted or randomly generated programs cannot hang the host."""

    def __init__(self, program, max_steps=200_000, max_depth=100):
        self.fns = {f.name: f for f in program.functions}
        self.max_steps, self.max_depth, self.steps = max_steps, max_depth, 0

    def _tick(self):
        self.steps += 1
        if self.steps > self.max_steps:
            raise StepLimitExceeded("step limit exceeded")

    def call(self, name, args, depth=0):
        if name not in self.fns:
            raise CRuntimeError(f"unknown function {name}")
        fn = self.fns[name]
        if len(args) != len(fn.params):
            raise CRuntimeError(f"{name} expects {len(fn.params)} arguments")
        if depth > self.max_depth:
            raise CRuntimeError("call depth exceeded")
        env = {p: wrap32(int(a)) for p, a in zip(fn.params, args)}
        try:
            self.exec(fn.body, env, depth)
        except _ReturnSignal as r:
            return r.v
        raise CRuntimeError(f"{name} ended without returning a value")

    def eval(self, e, env, depth):
        self._tick()
        if isinstance(e, Num):
            return wrap32(e.v)
        if isinstance(e, Var):
            if e.name not in env:
                raise CRuntimeError(f"undefined variable {e.name}")
            return env[e.name]
        if isinstance(e, BinOp):
            if e.op == "&&":
                return int(bool(self.eval(e.l, env, depth)) and bool(self.eval(e.r, env, depth)))
            if e.op == "||":
                return int(bool(self.eval(e.l, env, depth)) or bool(self.eval(e.r, env, depth)))
            return _OPS[e.op](self.eval(e.l, env, depth), self.eval(e.r, env, depth))
        if isinstance(e, UnOp):
            v = self.eval(e.e, env, depth)
            return wrap32(-v) if e.op == "-" else int(not v)
        if isinstance(e, Cond):
            return self.eval(e.a if self.eval(e.c, env, depth) else e.b, env, depth)
        if isinstance(e, Call):
            return self.call(e.name, [self.eval(a, env, depth) for a in e.args], depth + 1)
        raise CRuntimeError(f"cannot evaluate {type(e).__name__}")

    def exec(self, s, env, depth):
        self._tick()
        if isinstance(s, Block):
            for x in s.stmts:
                self.exec(x, env, depth)
        elif isinstance(s, Decl):
            env[s.name] = self.eval(s.init, env, depth) if s.init else 0
        elif isinstance(s, Assign):
            if s.name not in env:
                raise CRuntimeError(f"assignment to undeclared variable {s.name}")
            v = self.eval(s.e, env, depth)
            env[s.name] = v if s.op == "=" else _OPS[s.op[0]](env[s.name], v)
        elif isinstance(s, If):
            if self.eval(s.c, env, depth):
                self.exec(s.then, env, depth)
            elif s.other:
                self.exec(s.other, env, depth)
        elif isinstance(s, While):
            while self.eval(s.c, env, depth):
                self.exec(s.body, env, depth)
        elif isinstance(s, For):
            if s.init:
                self.exec(s.init, env, depth)
            while s.c is None or self.eval(s.c, env, depth):
                self.exec(s.body, env, depth)
                if s.step:
                    self.exec(s.step, env, depth)
        elif isinstance(s, Return):
            raise _ReturnSignal(self.eval(s.e, env, depth))
        elif isinstance(s, ExprStmt):
            self.eval(s.e, env, depth)
        else:
            raise CRuntimeError(f"cannot execute {type(s).__name__}")


def run_c(source, function, *args, max_steps=200_000):
    """Parse `source` and call `function(*args)` in the safe interpreter."""
    return Interpreter(parse(source), max_steps).call(function, args)


# ------------------------------------------------------------------ enumerative synthesis ---
_BIN_FUNCS = dict(_OPS)
_BIN_FUNCS["&&"] = lambda a, b: int(bool(a) and bool(b))
_BIN_FUNCS["||"] = lambda a, b: int(bool(a) or bool(b))
_PY_FUNCS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b,
             "<": lambda a, b: int(a < b), ">": lambda a, b: int(a > b), "<=": lambda a, b: int(a <= b),
             ">=": lambda a, b: int(a >= b), "==": lambda a, b: int(a == b), "!=": lambda a, b: int(a != b),
             "&&": lambda a, b: int(bool(a) and bool(b)), "||": lambda a, b: int(bool(a) or bool(b))}
_COMMUTATIVE = {"+", "*", "==", "!=", "&&", "||"}


def to_python(node):
    """Print an expression AST (Num, Var, BinOp, Cond, UnOp) as a Python expression."""
    if isinstance(node, Num):
        return str(node.v)
    if isinstance(node, Var):
        return node.name
    if isinstance(node, BinOp):
        op = {"&&": "and", "||": "or"}.get(node.op, node.op)
        def side(x, right):
            s = to_python(x)
            if isinstance(x, Cond) or (isinstance(x, BinOp) and (_PREC[x.op] < _PREC[node.op] or (right and _PREC[x.op] == _PREC[node.op]))):
                return f"({s})"
            return s
        text = f"{side(node.l, False)} {op} {side(node.r, True)}"
        return f"int({text})" if node.op in ("&&", "||") else text   # keep boolean connectives integer-valued
    if isinstance(node, UnOp):
        return f"{'-' if node.op == '-' else 'not '}({to_python(node.e)})" if isinstance(node.e, (BinOp, Cond)) else f"{'-' if node.op == '-' else 'not '}{to_python(node.e)}"
    if isinstance(node, Cond):
        return f"({to_python(node.a)} if {to_python(node.c)} else {to_python(node.b)})"
    raise TypeError(f"cannot print {type(node).__name__}")


@register("enumerative_synthesis", family="symbolic")
class EnumerativeSynthesizer:
    """Bottom-up enumerative synthesis of an integer expression from input/output examples.
    Expressions are built by increasing size; two expressions with identical outputs on the
    examples are treated as equivalent (only the smallest is kept), which keeps the search small."""

    def __init__(self, variables, constants=(0, 1, 2), ops=("+", "-", "*", "<", "=="), use_ite=True, max_size=9,
                 language="python"):
        if language not in ("python", "c"):
            raise ValueError("language must be 'python' or 'c'")
        self.vars, self.consts, self.ops, self.use_ite, self.max_size = tuple(variables), tuple(constants), tuple(ops), use_ite, max_size
        self.language, self.explored_ = language, 0
        self._funcs = _PY_FUNCS if language == "python" else _BIN_FUNCS     # C: 32-bit wraparound semantics
        self._norm = (lambda v: v) if language == "python" else wrap32

    def synthesize(self, examples):
        """examples: list of (inputs, output); inputs is a tuple in the order of `variables`.
        Returns the smallest matching expression (AST) or None."""
        inputs = [tuple(e[0]) if not isinstance(e[0], int) else (e[0],) for e in examples]
        target = tuple(self._norm(e[1]) for e in examples)
        seen, by_size, self.explored_ = {}, {}, 0
        for s in range(1, self.max_size + 1):
            bucket = []

            def add(expr, out):
                self.explored_ += 1
                if out not in seen:
                    seen[out] = expr
                    bucket.append((expr, out))

            if s == 1:
                for i, v in enumerate(self.vars):
                    add(Var(v), tuple(inp[i] for inp in inputs))
                for c in self.consts:
                    add(Num(c), (self._norm(c),) * len(inputs))
            else:
                for op in self.ops:
                    f = self._funcs[op]
                    for sl in range(1, s - 1):
                        sr = s - 1 - sl
                        if op in _COMMUTATIVE and sl > sr:
                            continue
                        for i, (el, ol) in enumerate(by_size.get(sl, [])):
                            for j, (er, orr) in enumerate(by_size.get(sr, [])):
                                if op in _COMMUTATIVE and sl == sr and j < i:
                                    continue
                                try:
                                    out = tuple(f(a, b) for a, b in zip(ol, orr))
                                except CRuntimeError:
                                    continue
                                add(BinOp(op, el, er), out)
                if self.use_ite:
                    for sc in range(1, s - 2):
                        conds = [(e, o) for e, o in by_size.get(sc, []) if all(v in (0, 1) for v in o)]
                        for st in range(1, s - sc - 1):
                            se = s - 1 - sc - st
                            for ec, oc in conds:
                                for et, ot in by_size.get(st, []):
                                    for ee, oe in by_size.get(se, []):
                                        add(Cond(ec, et, ee), tuple(t if c else e for c, t, e in zip(oc, ot, oe)))
            by_size[s] = bucket
            for expr, out in bucket:
                if out == target:
                    return expr
        return None

    def to_function(self, expr, name="f"):
        """Source code for the synthesized expression in the synthesizer's language."""
        if self.language == "python":
            return f"def {name}({', '.join(self.vars)}):\n    return {to_python(expr)}\n"
        return f"int {name}({', '.join('int ' + v for v in self.vars)}) {{\n    return {to_c(expr)};\n}}"


# --------------------------------------------------------------------------- Datalog ---
def _is_var(t):
    return isinstance(t, str) and t[:1].isupper()


@register("datalog", family="symbolic")
class Datalog:
    """Bottom-up (forward-chaining) evaluation of Horn rules to a fixed point. Facts and atoms are
    tuples (predicate, arg, ...). In rules, arguments starting with an upper-case letter are variables."""

    def __init__(self):
        self.facts, self.rules = set(), []

    def add_fact(self, *fact):
        self.facts.add(tuple(fact))
        return self

    def add_rule(self, head, *body):
        self.rules.append((tuple(head), [tuple(b) for b in body]))
        return self

    @staticmethod
    def _match(atom, fact, subst):
        if atom[0] != fact[0] or len(atom) != len(fact):
            return None
        s = dict(subst)
        for a, f in zip(atom[1:], fact[1:]):
            if _is_var(a):
                if s.setdefault(a, f) != f:
                    return None
            elif a != f:
                return None
        return s

    def _solve(self, body, subst):
        if not body:
            yield subst
            return
        for fact in list(self.facts):
            s = self._match(body[0], fact, subst)
            if s is not None:
                yield from self._solve(body[1:], s)

    def run(self):
        """Apply all rules until no new facts appear. Returns the number of derived facts."""
        start = len(self.facts)
        while True:
            new = set()
            for head, body in self.rules:
                for s in self._solve(body, {}):
                    fact = tuple([head[0]] + [s.get(a, a) if _is_var(a) else a for a in head[1:]])
                    if fact not in self.facts:
                        new.add(fact)
            if not new:
                return len(self.facts) - start
            self.facts |= new

    def query(self, *pattern):
        """Bindings (dicts) for every fact matching the pattern."""
        return [s for s in (self._match(tuple(pattern), f, {}) for f in self.facts) if s is not None]


def program_facts(program):
    """Facts describing a parsed program: func(f), param(f,x), call(f,g), assigns(f,x), reads(f,x)."""
    facts = set()
    for fn in program.functions:
        facts.add(("func", fn.name))
        facts |= {("param", fn.name, p) for p in fn.params}
        for n in walk(fn.body):
            if isinstance(n, Call):
                facts.add(("call", fn.name, n.name))
            elif isinstance(n, (Assign, Decl)) and (not isinstance(n, Decl) or n.init):
                facts.add(("assigns", fn.name, n.name))
            elif isinstance(n, Var):
                facts.add(("reads", fn.name, n.name))
    return facts


REACHABILITY_RULES = [(("reaches", "X", "Y"), [("call", "X", "Y")]),
                      (("reaches", "X", "Z"), [("call", "X", "Y"), ("reaches", "Y", "Z")])]
