"""Code representation and generation. Grammar-based generation (random or guided),
template / sketch filling, and a byte-level BPE tokenizer for token-level models."""
from __future__ import annotations

import itertools
import re
from collections import Counter

import numpy as np

from .base import register
from .pycode import ExecutionError, check_python, run_py
from .symbolic import ParseError, StepLimitExceeded

_NT = re.compile(r"<\w+>")


@register("grammar_generator", family="codegen")
class GrammarGenerator:
    """Generate programs by choosing grammar productions. A grammar maps '<nonterminal>' to productions;
    a production is a list of pieces (nonterminals like '<expr>' or literal text), optionally given as
    (weight, pieces). Because only valid productions are ever chosen, the output is syntactically valid
    by construction. Pass `chooser(nonterminal, productions, depth) -> index` to steer generation
    (neural-guided or search-guided generation)."""

    def __init__(self, grammar, start="<func>"):
        self.start, self.g = start, {}
        for nt, prods in grammar.items():
            norm = []
            for p in prods:
                w, pieces = p if isinstance(p, tuple) else (1, p)
                norm.append((w, [pieces] if isinstance(pieces, str) else list(pieces)))
            self.g[nt] = norm
        self.min_depth = {nt: float("inf") for nt in self.g}
        for _ in range(len(self.g) + 1):  # shortest derivation depth of each nonterminal
            for nt, prods in self.g.items():
                for _, pieces in prods:
                    d = 1 + max([self.min_depth[s] for s in pieces if _NT.fullmatch(s)] + [0])
                    self.min_depth[nt] = min(self.min_depth[nt], d)

    def _pd(self, pieces):
        return 1 + max([self.min_depth[s] for s in pieces if _NT.fullmatch(s)] + [0])

    def generate(self, rng=None, max_depth=6, chooser=None):
        rng = rng or np.random.default_rng()

        def expand(sym, depth):
            if not _NT.fullmatch(sym):
                return sym
            prods = self.g[sym]
            if depth >= max_depth:  # force the shortest derivations so generation terminates
                best = min(self._pd(p) for _, p in prods)
                prods = [(w, p) for w, p in prods if self._pd(p) == best]
            if chooser is not None:
                idx = chooser(sym, [p for _, p in prods], depth)
            else:
                w = np.array([x[0] for x in prods], dtype=float)
                idx = int(rng.choice(len(prods), p=w / w.sum()))
            return "".join(expand(s, depth + 1) for s in prods[idx][1])

        return expand(self.start, 0)


PY_GRAMMAR = {
    "<func>": [["def f(x, y):\n    t = ", "<init>", "\n", "<stmts>", "    return ", "<expr>", "\n"]],
    "<init>": [(2, ["<iatom>"]), ["(", "<iatom>", " + ", "<iatom>", ")"], ["(", "<iatom>", " * ", "<iatom>", ")"]],
    "<iatom>": [["x"], ["y"], ["0"], ["1"], ["2"], ["3"]],
    "<stmts>": [(1, [""]), (2, ["<stmt>", "<stmts>"])],
    "<stmt>": [["    t = ", "<expr>", "\n"],
               ["    if ", "<cond>", ":\n        t = ", "<expr>", "\n"],
               ["    if ", "<cond>", ":\n        t = ", "<expr>", "\n    else:\n        t = ", "<expr>", "\n"],
               ["    for i in range(", "<small>", "):\n        t = t + ", "<expr>", "\n"]],
    "<cond>": [["<expr>", " < ", "<expr>"], ["<expr>", " == ", "<expr>"], ["(", "<cond>", " and ", "<cond>", ")"]],
    "<expr>": [(3, ["<atom>"]), (2, ["(", "<expr>", " + ", "<expr>", ")"]), (1, ["(", "<expr>", " - ", "<expr>", ")"]),
               (1, ["(", "<expr>", " * ", "<expr>", ")"]), (1, ["(", "<expr>", " if ", "<cond>", " else ", "<expr>", ")"])],
    "<atom>": [["x"], ["y"], ["t"], ["0"], ["1"], ["2"], ["3"]],
    "<small>": [["2"], ["3"], ["4"]],
}


# ------------------------------------------------------------------ templates / sketches ---
TEMPLATES = {
    "loop_sum": "def f(n):\n    acc = {{init}}\n    for i in range(n):\n        acc = acc + {{term}}\n    return acc\n",
    "clamp": "def f(x):\n    if x < {{lo}}:\n        return {{lo}}\n    if x > {{hi}}:\n        return {{hi}}\n    return x\n",
    "select": "def f(x, y):\n    if {{cond}}:\n        return {{then}}\n    return {{other}}\n",
}
_SLOT = re.compile(r"\{\{(\w+)\}\}")


@register("template_filler", family="codegen")
class TemplateFiller:
    """Slot-based code construction: a template with {{slots}} is completed from candidate fragments.
    Every filled program is parsed, so structurally invalid fragments are rejected."""

    def __init__(self, template):
        self.template = TEMPLATES.get(template, template)
        self.slots = list(dict.fromkeys(_SLOT.findall(self.template)))

    def fill(self, **values):
        missing = set(self.slots) - set(values)
        if missing:
            raise KeyError(f"missing slots: {sorted(missing)}")
        src = _SLOT.sub(lambda m: str(values[m.group(1)]), self.template)
        check_python(src)
        return src

    def enumerate(self, candidates):
        """All valid completions from candidates: {slot: [fragment, ...]}."""
        for combo in itertools.product(*(candidates[s] for s in self.slots)):
            try:
                yield self.fill(**dict(zip(self.slots, combo)))
            except ParseError:
                continue

    def sketch_synthesize(self, candidates, examples, function="f", limit=100_000):
        """First completion that reproduces every (args, output) example (program sketching)."""
        for n, src in enumerate(self.enumerate(candidates)):
            if n >= limit:
                break
            try:
                if all(run_py(src, function, *a) == out for a, out in examples):
                    return src
            except (ExecutionError, StepLimitExceeded):
                continue
        return None


# ------------------------------------------------------------------------------- BPE ---
_PRE = re.compile(r"[A-Za-z_]\w*|\d+|\s+|[^\w\s]|\w+")


@register("bpe_tokenizer", family="codegen")
class BPETokenizer:
    """Byte-level byte-pair-encoding tokenizer with code-aware pre-splitting (identifiers, numbers,
    whitespace, punctuation). Any text round-trips exactly because every byte is in the base vocabulary."""

    def __init__(self, num_merges=200):
        self.num_merges, self.merges, self.ranks = num_merges, [], {}
        self.vocab = {i: bytes([i]) for i in range(256)}

    def train(self, corpus):
        words = Counter()
        for text in corpus:
            for w in _PRE.findall(text):
                words[tuple(w.encode())] += 1
        for _ in range(self.num_merges):
            pairs = Counter()
            for w, c in words.items():
                for a, b in zip(w, w[1:]):
                    pairs[(a, b)] += c
            if not pairs:
                break
            (a, b), cnt = pairs.most_common(1)[0]
            if cnt < 2:
                break
            new = 256 + len(self.merges)
            self.merges.append((a, b)); self.ranks[(a, b)] = new; self.vocab[new] = self.vocab[a] + self.vocab[b]
            merged = Counter()
            for w, c in words.items():
                out, i = [], 0
                while i < len(w):
                    if i + 1 < len(w) and (w[i], w[i + 1]) == (a, b):
                        out.append(new); i += 2
                    else:
                        out.append(w[i]); i += 1
                merged[tuple(out)] += c
            words = merged
        return self

    def _encode_word(self, w):
        ids = list(w.encode())
        while len(ids) > 1:
            cands = [(self.ranks[p], i) for i, p in enumerate(zip(ids, ids[1:])) if p in self.ranks]
            if not cands:
                break
            r, i = min(cands)
            ids[i:i + 2] = [r]
        return ids

    def encode(self, text):
        return [t for w in _PRE.findall(text) for t in self._encode_word(w)]

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
