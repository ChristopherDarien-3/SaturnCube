"""Randomization and experimental design. Seed-reproducible random architecture graphs with
type constraints and a cost budget (plus ablation), Latin hypercube / factorial designs, and the
statistics needed to compare results (bootstrap CI, permutation test, Holm correction)."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
from dataclasses import dataclass, field

import numpy as np

from .base import register

# kind -> (input type, output type, cost). Edit or extend to match the modules you actually have.
MODULE_KINDS = {
    "mlp": ("vector", "vector", 1), "resnet": ("vector", "vector", 2), "hopfield": ("vector", "vector", 1),
    "case_memory": ("vector", "vector", 1), "lstm": ("sequence", "vector", 2), "transformer": ("sequence", "sequence", 3),
    "retriever": ("vector", "sequence", 2), "gnn": ("graph", "vector", 2),
}
ADAPTERS = {("sequence", "vector"): "mean_pool", ("vector", "sequence"): "repeat"}   # allowed type conversions


@dataclass
class ArchitectureSpec:
    """An architecture as a graph: modules are nodes, data flows along edges."""
    seed: int
    modules: dict                              # id -> {"kind": str, "params": dict}
    edges: list = field(default_factory=list)  # [(src_id, dst_id)]
    parent: str = None                         # hash of the spec this was ablated from

    def to_dict(self):
        return {"seed": self.seed, "modules": self.modules, "edges": [list(e) for e in self.edges], "parent": self.parent}

    @staticmethod
    def from_dict(d):
        return ArchitectureSpec(d["seed"], d["modules"], [tuple(e) for e in d["edges"]], d.get("parent"))

    def config_hash(self):
        core = {"modules": self.modules, "edges": sorted(map(list, self.edges))}
        return hashlib.sha1(json.dumps(core, sort_keys=True).encode()).hexdigest()[:12]

    def cost(self, kinds=MODULE_KINDS):
        return sum(kinds[m["kind"]][2] for m in self.modules.values())

    def topological_order(self):
        indeg = {m: 0 for m in self.modules}
        for _, d in self.edges:
            indeg[d] += 1
        ready, order = sorted(m for m, k in indeg.items() if k == 0), []
        while ready:
            n = ready.pop(0); order.append(n)
            for s, d in self.edges:
                if s == n:
                    indeg[d] -= 1
                    if indeg[d] == 0:
                        ready.append(d)
        if len(order) != len(self.modules):
            raise ValueError("architecture has a cycle")
        return order


def edge_ok(spec_modules, src, dst, kinds=MODULE_KINDS):
    out_t, in_t = kinds[spec_modules[src]["kind"]][1], kinds[spec_modules[dst]["kind"]][0]
    return out_t == in_t or (out_t, in_t) in ADAPTERS


def validate(spec, budget=None, kinds=MODULE_KINDS, input_type=None):
    """Raise ValueError unless the spec is acyclic, type-compatible, connected and within budget."""
    spec.topological_order()
    for s, d in spec.edges:
        if s not in spec.modules or d not in spec.modules:
            raise ValueError(f"edge {s}->{d} refers to a missing module")
        if not edge_ok(spec.modules, s, d, kinds):
            raise ValueError(f"incompatible edge {s}({spec.modules[s]['kind']}) -> {d}({spec.modules[d]['kind']})")
    if budget is not None and spec.cost(kinds) > budget:
        raise ValueError(f"cost {spec.cost(kinds)} exceeds budget {budget}")
    has_in = {d for _, d in spec.edges}
    sources = [m for m in spec.modules if m not in has_in]
    if input_type and any(kinds[spec.modules[m]["kind"]][0] != input_type for m in sources):
        raise ValueError("a source module does not accept the input type")
    touched = {x for e in spec.edges for x in e}
    if len(spec.modules) > 1 and any(m not in touched for m in spec.modules):
        raise ValueError("isolated module")
    return True


@register("random_architecture", family="experiment")
class RandomArchitectureGenerator:
    """Randomly assembles module graphs. connectivity: 'erdos_renyi' (each valid forward edge with
    probability p), 'small_world' (ring-like local edges with random rewiring) or 'sparse' (few edges).
    The same seed always gives the same architecture (and hash)."""

    def __init__(self, kinds=None, connectivity="erdos_renyi", p=0.4, min_modules=3, max_modules=6, budget=10,
                 input_type="vector", rewire=0.2):
        if connectivity not in ("erdos_renyi", "small_world", "sparse"):
            raise ValueError("connectivity must be 'erdos_renyi', 'small_world' or 'sparse'")
        self.kinds, self.conn, self.p, self.budget = kinds or MODULE_KINDS, connectivity, p, budget
        self.lo, self.hi, self.input_type, self.rewire = min_modules, max_modules, input_type, rewire

    def _module(self, rng, kind):
        return {"kind": kind, "params": {"hidden": int(rng.choice([16, 32, 64, 128])), "activation": str(rng.choice(["relu", "gelu", "tanh"])),
                                         "seed": int(rng.integers(1 << 30))}}

    def generate(self, seed):
        for attempt in range(200):
            rng = np.random.default_rng([seed, attempt])
            spec = self._try(rng, seed)
            if spec is not None:
                return spec
        raise RuntimeError("could not generate a valid architecture; relax the budget or constraints")

    def _try(self, rng, seed):
        n, kinds, cost = int(rng.integers(self.lo, self.hi + 1)), list(self.kinds), 0
        mods = {}
        for i in range(n):
            ok = [k for k in kinds if cost + self.kinds[k][2] <= self.budget and (i > 0 or self.kinds[k][0] == self.input_type)]
            if not ok:
                return None
            k = str(rng.choice(ok)); cost += self.kinds[k][2]
            mods[f"m{i}"] = self._module(rng, k)
        ids, edges = list(mods), set()
        cand = [(a, b) for ia, a in enumerate(ids) for b in ids[ia + 1:] if edge_ok(mods, a, b, self.kinds)]
        if self.conn == "erdos_renyi":
            edges = {e for e in cand if rng.random() < self.p}
        elif self.conn == "small_world":
            for ia, a in enumerate(ids):
                for b in ids[ia + 1: ia + 3]:
                    if (a, b) in cand:
                        edges.add((a, b) if rng.random() > self.rewire else cand[int(rng.integers(len(cand)))])
        else:
            for a in ids:
                outs = [e for e in cand if e[0] == a]
                if outs and rng.random() < 0.7:
                    edges.add(outs[int(rng.integers(len(outs)))])
        for b in ids[1:]:  # every module needs an input; the first module is the entry point
            if not any(d == b for _, d in edges):
                preds = [e for e in cand if e[1] == b]
                if not preds:
                    return None
                edges.add(preds[int(rng.integers(len(preds)))])
        spec = ArchitectureSpec(seed, mods, sorted(edges))
        try:
            validate(spec, self.budget, self.kinds, self.input_type)
        except ValueError:
            return None
        return spec


def ablate_module(spec, module_id, kinds=MODULE_KINDS):
    """Remove a module, reconnecting its predecessors to its successors where types allow."""
    mods = {k: v for k, v in spec.modules.items() if k != module_id}
    preds = [s for s, d in spec.edges if d == module_id]
    succs = [d for s, d in spec.edges if s == module_id]
    edges = {(s, d) for s, d in spec.edges if module_id not in (s, d)}
    edges |= {(p, s) for p in preds for s in succs if edge_ok(mods, p, s, kinds)}
    return ArchitectureSpec(spec.seed, mods, sorted(edges), parent=spec.config_hash())


def ablate_edge(spec, edge):
    """Remove one connection (the destination must keep another input)."""
    if sum(1 for _, d in spec.edges if d == edge[1]) < 2:
        raise ValueError("would leave the destination module without an input")
    return ArchitectureSpec(spec.seed, copy.deepcopy(spec.modules), [e for e in spec.edges if e != tuple(edge)], parent=spec.config_hash())


def execute(spec, runners, x):
    """Run an architecture. runners[kind](inputs, params) -> output; sources receive x, other modules
    receive the output of their single predecessor or a list of outputs. Returns {sink_id: output}."""
    out = {}
    for m in spec.topological_order():
        preds = [s for s, d in spec.edges if d == m]
        inp = x if not preds else out[preds[0]] if len(preds) == 1 else [out[p] for p in preds]
        out[m] = runners[spec.modules[m]["kind"]](inp, spec.modules[m]["params"])
    srcs = {s for s, _ in spec.edges}
    return {m: v for m, v in out.items() if m not in srcs}


# ----------------------------------------------------------------- designs and statistics ---
def latin_hypercube(n, ranges, seed=None):
    """n samples in len(ranges) dimensions; every dimension has exactly one sample in each of n equal strata."""
    rng = np.random.default_rng(seed)
    out = np.empty((n, len(ranges)))
    for j, (lo, hi) in enumerate(ranges):
        out[:, j] = lo + (rng.permutation(n) + rng.random(n)) / n * (hi - lo)
    return out


def full_factorial(factors):
    """Every combination of factor levels: {name: [levels]} -> list of {name: level}."""
    names = list(factors)
    return [dict(zip(names, combo)) for combo in itertools.product(*(factors[n] for n in names))]


def bootstrap_ci(values, stat=np.mean, n_boot=2000, alpha=0.05, seed=None):
    rng, v = np.random.default_rng(seed), np.asarray(values)
    s = [stat(rng.choice(v, len(v))) for _ in range(n_boot)]
    return float(np.quantile(s, alpha / 2)), float(np.quantile(s, 1 - alpha / 2))


def permutation_test(a, b, n_perm=5000, seed=None):
    """Two-sided permutation test for a difference in means; returns the p-value."""
    rng, a, b = np.random.default_rng(seed), np.asarray(a, float), np.asarray(b, float)
    obs, pool, k, count = abs(a.mean() - b.mean()), np.concatenate([a, b]), len(a), 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        count += abs(pool[:k].mean() - pool[k:].mean()) >= obs
    return (count + 1) / (n_perm + 1)


def holm_correction(pvalues, alpha=0.05):
    """Holm-Bonferroni: which hypotheses can be rejected while controlling the family-wise error rate."""
    order, reject = np.argsort(pvalues), [False] * len(pvalues)
    for rank, i in enumerate(order):
        if pvalues[i] <= alpha / (len(pvalues) - rank):
            reject[i] = True
        else:
            break
    return reject
