"""Wheel F: problem-solving strategies inspired by human learning. Spaced repetition (SM-2),
interleaved practice, and a predict-act-observe-revise loop with confidence monitoring."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np

from .base import register


@register("spaced_repetition", family="cognition")
class SM2Scheduler:
    """SuperMemo-2 spaced repetition. review(item, quality 0-5, day) schedules the next review;
    quality < 3 resets the item. due(day) lists items to practise."""

    def __init__(self):
        self.items = {}

    def add(self, item, day=0):
        self.items[item] = dict(ease=2.5, interval=0, reps=0, due=day)

    def review(self, item, quality, day):
        s = self.items[item]
        if quality < 3:
            s["reps"], s["interval"] = 0, 1
        else:
            s["reps"] += 1
            s["ease"] = max(1.3, s["ease"] + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
            s["interval"] = 1 if s["reps"] == 1 else 6 if s["reps"] == 2 else round(s["interval"] * s["ease"])
        s["due"] = day + s["interval"]
        return s["interval"]

    def due(self, day):
        return sorted(i for i, s in self.items.items() if s["due"] <= day)


@register("interleaved_practice", family="cognition")
class InterleavedPractice:
    """Orders practice items from several topics. 'blocked' finishes one topic at a time; 'interleaved'
    mixes topics at random while avoiding runs longer than max_run (when other topics remain)."""

    def __init__(self, mode="interleaved", max_run=1, seed=None):
        if mode not in ("blocked", "interleaved"):
            raise ValueError("mode must be 'blocked' or 'interleaved'")
        self.mode, self.max_run, self.rng = mode, max_run, np.random.default_rng(seed)

    def schedule(self, items_by_topic):
        pools = {t: list(v) for t, v in items_by_topic.items()}
        if self.mode == "blocked":
            return [(t, x) for t, v in pools.items() for x in v]
        out, last, run = [], None, 0
        while any(pools.values()):
            options = [t for t, v in pools.items() if v and not (t == last and run >= self.max_run)] or [t for t, v in pools.items() if v]
            w = np.array([len(pools[t]) for t in options], dtype=float)
            t = options[int(self.rng.choice(len(options), p=w / w.sum()))]
            out.append((t, pools[t].pop(0)))
            run = run + 1 if t == last else 1
            last = t
        return out


@dataclass
class Attempt:
    round: int
    candidate: object
    ok: bool
    feedback: object


@register("revision_loop", family="cognition")
class RevisionLoop:
    """Predict -> act -> observe -> revise. propose(task, attempts) returns a candidate given the earlier
    attempts and their feedback (so it can re-plan instead of repeating itself); verify(candidate) returns
    (ok, feedback). Stops at the first verified candidate or after max_rounds."""

    def __init__(self, propose: Callable, verify: Callable, max_rounds=5):
        self.propose, self.verify, self.max_rounds = propose, verify, max_rounds

    def run(self, task):
        attempts: List[Attempt] = []
        for r in range(1, self.max_rounds + 1):
            cand = self.propose(task, attempts)
            ok, fb = self.verify(cand)
            attempts.append(Attempt(r, cand, ok, fb))
            if ok:
                break
        self.attempts_ = attempts
        return attempts[-1].candidate if attempts[-1].ok else None


@register("metacognitive_monitor", family="cognition")
class MetacognitiveMonitor:
    """Tracks (confidence, correct) pairs to measure calibration and to choose an abstention threshold
    (selective prediction): answer only when confidence is high enough to reach a target accuracy."""

    def __init__(self):
        self.conf, self.correct = [], []

    def record(self, confidence, correct):
        self.conf.append(float(confidence)); self.correct.append(bool(correct))

    def expected_calibration_error(self, bins=10):
        c, y = np.array(self.conf), np.array(self.correct, dtype=float)
        edges, ece = np.linspace(0, 1, bins + 1), 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (c >= lo) & ((c < hi) | (hi == 1.0))
            if m.any():
                ece += m.mean() * abs(y[m].mean() - c[m].mean())
        return float(ece)

    def threshold_for_accuracy(self, target):
        """Smallest confidence threshold whose answered subset reaches `target` accuracy (None if impossible)."""
        c, y = np.array(self.conf), np.array(self.correct, dtype=float)
        for t in sorted(set(c)):
            if y[c >= t].mean() >= target:
                return float(t)
        return None

    def coverage_and_accuracy(self, threshold):
        c, y = np.array(self.conf), np.array(self.correct, dtype=float)
        m = c >= threshold
        return float(m.mean()), float(y[m].mean()) if m.any() else float("nan")
