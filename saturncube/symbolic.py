"""Program synthesis and symbolic reasoning: a small expression AST, enumerative synthesis of programs
from input/output examples, and a Datalog-style forward-chaining rule engine."""
from __future__ import annotations

from dataclasses import dataclass

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


class ParseError(Exception):
    """Source code is invalid, or uses constructs outside the allowed subset."""


class StepLimitExceeded(Exception):
    """A program ran longer than its step budget."""


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


_PREC = {"||": 1, "&&": 2, "==": 3, "!=": 3, "<": 4, ">": 4, "<=": 4, ">=": 4, "+": 5, "-": 5, "*": 6}


def to_python(node):
    """Print an expression AST as a Python expression."""
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
        inner = f"({to_python(node.e)})" if isinstance(node.e, (BinOp, Cond)) else to_python(node.e)
        return f"-{inner}" if node.op == "-" else f"not {inner}"
    if isinstance(node, Cond):
        return f"({to_python(node.a)} if {to_python(node.c)} else {to_python(node.b)})"
    raise TypeError(f"cannot print {type(node).__name__}")


# ------------------------------------------------------------------ enumerative synthesis ---
_FUNCS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b,
          "<": lambda a, b: int(a < b), ">": lambda a, b: int(a > b), "<=": lambda a, b: int(a <= b),
          ">=": lambda a, b: int(a >= b), "==": lambda a, b: int(a == b), "!=": lambda a, b: int(a != b),
          "&&": lambda a, b: int(bool(a) and bool(b)), "||": lambda a, b: int(bool(a) or bool(b))}
_COMMUTATIVE = {"+", "*", "==", "!=", "&&", "||"}


@register("enumerative_synthesis", family="symbolic")
class EnumerativeSynthesizer:
    """Bottom-up enumerative synthesis of an integer expression from input/output examples.
    Expressions are built by increasing size; two expressions with identical outputs on the
    examples are treated as equivalent (only the smallest is kept), which keeps the search small."""

    def __init__(self, variables, constants=(0, 1, 2), ops=("+", "-", "*", "<", "=="), use_ite=True, max_size=9):
        unknown = set(ops) - set(_FUNCS)
        if unknown:
            raise ValueError(f"unknown operators: {sorted(unknown)}")
        self.vars, self.consts, self.ops, self.use_ite, self.max_size = tuple(variables), tuple(constants), tuple(ops), use_ite, max_size
        self.explored_ = 0

    def synthesize(self, examples):
        """examples: list of (inputs, output); inputs is a tuple in the order of `variables`.
        Returns the smallest matching expression (AST) or None."""
        inputs = [tuple(e[0]) if not isinstance(e[0], int) else (e[0],) for e in examples]
        target = tuple(e[1] for e in examples)
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
                    add(Num(c), (c,) * len(inputs))
            else:
                for op in self.ops:
                    f = _FUNCS[op]
                    for sl in range(1, s - 1):
                        sr = s - 1 - sl
                        if op in _COMMUTATIVE and sl > sr:
                            continue
                        for i, (el, ol) in enumerate(by_size.get(sl, [])):
                            for j, (er, orr) in enumerate(by_size.get(sr, [])):
                                if op in _COMMUTATIVE and sl == sr and j < i:
                                    continue
                                add(BinOp(op, el, er), tuple(f(a, b) for a, b in zip(ol, orr)))
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
        """Python source for the synthesized expression."""
        return f"def {name}({', '.join(self.vars)}):\n    return {to_python(expr)}\n"


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


REACHABILITY_RULES = [(("reaches", "X", "Y"), [("call", "X", "Y")]),
                      (("reaches", "X", "Z"), [("call", "X", "Y"), ("reaches", "Y", "Z")])]
