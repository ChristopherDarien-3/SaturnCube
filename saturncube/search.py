"""Wheel C: search and optimization. Genetic algorithm, successive halving (with a random-search
baseline) and simulated annealing. All are gradient-free and work on any evaluable object."""
from __future__ import annotations

import math

import numpy as np

from .base import register


@register("genetic_algorithm", family="search")
class GeneticAlgorithm:
    """Genetic algorithm over a list of typed genes.

    gene_space entries: ("float", lo, hi), ("int", lo, hi) or ("choice", [options...]).
    fitness(genome_list) -> float. Tournament selection, uniform crossover, per-gene mutation, elitism."""

    def __init__(self, fitness, gene_space, pop_size=60, generations=100, crossover_rate=0.8,
                 mutation_rate=None, elite=2, tournament=3, maximize=True, seed=None):
        for g in gene_space:
            if g[0] not in ("float", "int", "choice"):
                raise ValueError(f"unknown gene type {g[0]!r}")
        self.fitness, self.space, self.pop_size, self.generations = fitness, list(gene_space), pop_size, generations
        self.cx, self.mut = crossover_rate, mutation_rate if mutation_rate is not None else 1.0 / len(gene_space)
        self.elite, self.tournament, self.sign = elite, tournament, 1.0 if maximize else -1.0
        self.rng, self.history_, self._cache = np.random.default_rng(seed), [], {}

    def _random_gene(self, g):
        if g[0] == "float":
            return float(self.rng.uniform(g[1], g[2]))
        if g[0] == "int":
            return int(self.rng.integers(g[1], g[2] + 1))
        return g[1][int(self.rng.integers(len(g[1])))]

    def _mutate_gene(self, g, v):
        if g[0] == "float":
            return float(np.clip(v + self.rng.normal(0, 0.1 * (g[2] - g[1])), g[1], g[2]))
        return self._random_gene(g)

    def _score(self, genome):
        key = tuple(genome)
        if key not in self._cache:
            self._cache[key] = self.sign * float(self.fitness(list(genome)))
        return self._cache[key]

    def _select(self, pop, scores):
        idx = self.rng.integers(len(pop), size=self.tournament)
        return pop[int(idx[np.argmax([scores[i] for i in idx])])]

    def run(self):
        pop = [[self._random_gene(g) for g in self.space] for _ in range(self.pop_size)]
        for _ in range(self.generations):
            scores = [self._score(p) for p in pop]
            order = np.argsort(scores)[::-1]
            self.history_.append(self.sign * scores[order[0]])
            nxt = [list(pop[i]) for i in order[: self.elite]]
            while len(nxt) < self.pop_size:
                a, b = self._select(pop, scores), self._select(pop, scores)
                child = [x if self.rng.random() < 0.5 else y for x, y in zip(a, b)] if self.rng.random() < self.cx else list(a)
                child = [self._mutate_gene(g, v) if self.rng.random() < self.mut else v for g, v in zip(self.space, child)]
                nxt.append(child)
            pop = nxt
        scores = [self._score(p) for p in pop]
        best = int(np.argmax(scores))
        self.best_, self.best_fitness_ = pop[best], self.sign * scores[best]
        return self.best_, self.best_fitness_


@register("successive_halving", family="search")
class SuccessiveHalving:
    """Successive halving: evaluate many configurations cheaply, keep the best 1/eta, give the
    survivors eta times more budget, repeat. sampler(rng) -> config; evaluate(config, budget) -> score
    (higher is better), where budget could be epochs, data size, or search depth."""

    def __init__(self, sampler, evaluate, n_configs=27, min_budget=1, eta=3, seed=None):
        self.sampler, self.evaluate, self.n, self.min_budget, self.eta = sampler, evaluate, n_configs, min_budget, eta
        self.rng, self.rounds_, self.total_budget_ = np.random.default_rng(seed), [], 0

    def run(self):
        configs, budget = [self.sampler(self.rng) for _ in range(self.n)], self.min_budget
        while True:
            scores = [self.evaluate(c, budget) for c in configs]
            self.total_budget_ += len(configs) * budget
            self.rounds_.append((len(configs), budget))
            order = np.argsort(scores)[::-1]
            if len(configs) == 1:
                self.best_, self.best_score_ = configs[0], scores[0]
                return self.best_, self.best_score_
            keep = max(1, len(configs) // self.eta)
            configs, budget = [configs[i] for i in order[:keep]], budget * self.eta


class RandomSearch:
    """Baseline: n_trials random configurations, each evaluated at the full budget."""

    def __init__(self, sampler, evaluate, n_trials=27, budget=1, seed=None):
        self.sampler, self.evaluate, self.n, self.budget = sampler, evaluate, n_trials, budget
        self.rng, self.total_budget_ = np.random.default_rng(seed), 0

    def run(self):
        cs = [self.sampler(self.rng) for _ in range(self.n)]
        scores = [self.evaluate(c, self.budget) for c in cs]
        self.total_budget_ = self.n * self.budget
        i = int(np.argmax(scores))
        return cs[i], scores[i]


@register("simulated_annealing", family="search")
class SimulatedAnnealing:
    """Simulated annealing (minimises energy). neighbor(state, rng) -> new state."""

    def __init__(self, energy, neighbor, initial, t0=1.0, cooling=0.995, steps=5000, seed=None):
        self.energy, self.neighbor, self.state, self.t0 = energy, neighbor, initial, t0
        self.cooling, self.steps, self.rng, self.history_ = cooling, steps, np.random.default_rng(seed), []

    def run(self):
        cur, e, T = self.state, self.energy(self.state), self.t0
        best, best_e = cur, e
        for _ in range(self.steps):
            cand = self.neighbor(cur, self.rng)
            ce = self.energy(cand)
            if ce <= e or self.rng.random() < math.exp(-(ce - e) / max(T, 1e-12)):
                cur, e = cand, ce
                if e < best_e:
                    best, best_e = cur, e
            T *= self.cooling
            self.history_.append(best_e)
        self.best_, self.best_energy_ = best, best_e
        return best, best_e
