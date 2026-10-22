"""Wheel E: memory and retrieval. BM25 keyword search, a cosine-similarity vector store with an
optional locality-sensitive-hashing index, and a case library (episodic memory / case-based
reasoning) with a forgetting policy and consolidation."""
from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .base import register

_WORD = re.compile(r"[A-Za-z]+|\d+")


def tokenize_text(text):
    """Lower-cased words; camelCase and snake_case identifiers are split into their parts."""
    return [w.lower() for w in _WORD.findall(re.sub(r"([a-z])([A-Z])", r"\1 \2", text))]


@register("bm25", family="memory")
class BM25Index:
    """Okapi BM25 ranking over a growing list of documents."""

    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b, self.docs, self.df = k1, b, [], Counter()

    def add(self, docs):
        for d in ([docs] if isinstance(docs, str) else docs):
            toks = tokenize_text(d)
            self.docs.append(Counter(toks))
            self.df.update(set(toks))
        self.avgdl = sum(sum(c.values()) for c in self.docs) / max(len(self.docs), 1)
        return self

    def search(self, query, k=5):
        N, q, out = len(self.docs), tokenize_text(query), []
        for i, tf in enumerate(self.docs):
            dl, score = sum(tf.values()), 0.0
            for t in set(q):
                if t in tf:
                    idf = math.log(1 + (N - self.df[t] + 0.5) / (self.df[t] + 0.5))
                    score += idf * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            if score > 0:
                out.append((i, score))
        return sorted(out, key=lambda x: -x[1])[:k]


class HashingEmbedder:
    """Text -> unit vector by feature hashing of word unigrams and bigrams (no training needed)."""

    def __init__(self, dim=256, bigrams=True):
        self.dim, self.bigrams = dim, bigrams

    def embed(self, texts):
        texts = [texts] if isinstance(texts, str) else texts
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, text in enumerate(texts):
            toks = tokenize_text(text)
            feats = toks + ([f"{a}_{b}" for a, b in zip(toks, toks[1:])] if self.bigrams else [])
            for f, c in Counter(feats).items():
                h = zlib.crc32(f.encode())
                out[r, h % self.dim] += (1 if zlib.crc32(f.encode() + b"#") & 1 else -1) * (1 + math.log(c))
            n = np.linalg.norm(out[r])
            out[r] /= n if n else 1.0
        return out


@register("vector_store", family="memory")
class VectorStore:
    """Cosine-similarity k-NN store. With lsh_bits > 0 it also keeps random-hyperplane LSH tables for
    approximate search (search(..., exact=False)); exact search is always available."""

    def __init__(self, dim, lsh_bits=0, n_tables=6, seed=0):
        self.dim, self.lsh_bits, self.n_tables = dim, lsh_bits, n_tables
        self.V, self.payloads = np.zeros((0, dim), dtype=np.float32), []
        r = np.random.default_rng(seed)
        self.planes = r.normal(size=(n_tables, lsh_bits, dim)).astype(np.float32) if lsh_bits else None
        self.tables = [dict() for _ in range(n_tables)] if lsh_bits else None

    def _keys(self, v):
        return [tuple((self.planes[t] @ v > 0).astype(int)) for t in range(self.n_tables)]

    def add(self, vectors, payloads=None):
        vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
        vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        start = len(self.V)
        self.V = np.vstack([self.V, vectors])
        self.payloads += list(payloads) if payloads is not None else [None] * len(vectors)
        if self.tables is not None:
            for i, v in enumerate(vectors):
                for t, key in enumerate(self._keys(v)):
                    self.tables[t].setdefault(key, []).append(start + i)
        return self

    def search(self, query, k=5, exact=True):
        q = np.asarray(query, dtype=np.float32)
        q = q / max(np.linalg.norm(q), 1e-12)
        idx = np.arange(len(self.V))
        if not exact and self.tables is not None:
            cand = sorted({i for t, key in enumerate(self._keys(q)) for i in self.tables[t].get(key, [])})
            if len(cand) >= k:
                idx = np.array(cand)
        sims = self.V[idx] @ q
        top = np.argsort(-sims)[:k]
        return [(int(idx[i]), float(sims[i])) for i in top]


@dataclass
class Case:
    id: int
    question: str
    code: str
    concepts: frozenset
    compile_ok: bool = True
    tests_passed: Optional[bool] = None
    uses: int = 0
    successes: int = 0
    created: int = 0
    last_used: int = 0


@register("case_library", family="memory")
class CaseLibrary:
    """Episodic memory of solved problems. Each case stores the question, concepts, source code and
    the compile/test outcome. retrieve() fuses keyword (BM25) and vector rankings (reciprocal rank fusion)
    with concept overlap, so a solution can be found by what a problem *needs*, not only by similar words.
    When over capacity, a forgetting policy ('least_useful', 'lru' or 'fifo') evicts cases; consolidate()
    merges near-duplicates."""

    def __init__(self, capacity=None, policy="least_useful", dim=256):
        if policy not in ("least_useful", "lru", "fifo"):
            raise ValueError("policy must be 'least_useful', 'lru' or 'fifo'")
        self.capacity, self.policy, self.emb = capacity, policy, HashingEmbedder(dim)
        self.cases, self.clock, self._next, self._dirty = {}, 0, 0, True

    @staticmethod
    def _text(c):
        return f"{c.question} {' '.join(sorted(c.concepts))} {c.code}"

    def add(self, question, code, concepts=(), compile_ok=True, tests_passed=None):
        self.clock += 1
        c = Case(self._next, question, code, frozenset(concepts), compile_ok, tests_passed, created=self.clock, last_used=self.clock)
        self.cases[c.id] = c
        self._next += 1
        self._dirty = True
        if self.capacity and len(self.cases) > self.capacity:
            self._evict()
        return c.id

    def _evict(self):
        key = {"least_useful": lambda c: ((c.successes + 1) / (c.uses + 2), c.last_used), "lru": lambda c: c.last_used,
               "fifo": lambda c: c.created}[self.policy]
        victim = min(self.cases.values(), key=key)
        del self.cases[victim.id]
        self._dirty = True

    def _index(self):
        if self._dirty:
            self.ids = list(self.cases)
            texts = [self._text(self.cases[i]) for i in self.ids]
            self.bm25 = BM25Index().add(texts) if texts else None
            self.vs = VectorStore(self.emb.dim).add(self.emb.embed(texts), self.ids) if texts else None
            self._dirty = False

    def retrieve(self, query, k=3, concepts=()):
        self._index()
        if not self.ids:
            return []
        scores, qc = Counter(), set(concepts)
        for rank, (i, _) in enumerate(self.bm25.search(query, k=len(self.ids))):
            scores[self.ids[i]] += 1 / (60 + rank)
        for rank, (i, s) in enumerate(self.vs.search(self.emb.embed(query)[0], k=len(self.ids))):
            if s > 0:
                scores[self.ids[i]] += 1 / (60 + rank)
        for cid in self.ids:
            if qc:
                scores[cid] += 0.03 * len(qc & self.cases[cid].concepts) / len(qc | self.cases[cid].concepts)
        top = [self.cases[i] for i, s in scores.most_common(k) if s > 0]
        self.clock += 1
        for c in top:
            c.uses += 1
            c.last_used = self.clock
        return top

    def record_outcome(self, case_id, success):
        if case_id in self.cases and success:
            self.cases[case_id].successes += 1

    def consolidate(self, threshold=0.95):
        """Merge cases with near-identical text and the same concepts; returns how many were merged."""
        ids = list(self.cases)
        if len(ids) < 2:
            return 0
        E = self.emb.embed([self._text(self.cases[i]) for i in ids])
        sims, merged = E @ E.T, set()
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                if ids[a] in merged or ids[b] in merged:
                    continue
                if sims[a, b] >= threshold and self.cases[ids[a]].concepts == self.cases[ids[b]].concepts:
                    keep, drop = sorted((self.cases[ids[a]], self.cases[ids[b]]), key=lambda c: -c.successes)
                    keep.uses += drop.uses; keep.successes += drop.successes
                    del self.cases[drop.id]; merged.add(drop.id)
        self._dirty = bool(merged) or self._dirty
        return len(merged)

    def __len__(self):
        return len(self.cases)
