"""
genetic_optimizer.py
---------------------
GA baseline: a chromosome is a fixed-length sequence of action indices
(same action menu as the RL agent, environment.py::ACTION_MENU). Fitness
is the negative NISQ cost of the circuit after applying the sequence of
passes in order (a "stop" action truncates the sequence early).

This gives an apples-to-apples comparison against the RL agent: both
search over the same action space, but the GA is a population-based,
gradient-free search rather than a learned per-state policy.
"""

from __future__ import annotations
import numpy as np
from qiskit import QuantumCircuit
from environment import ACTION_MENU, N_ACTIONS, STOP_ACTION, circuit_cost


def apply_sequence(circuit: QuantumCircuit, sequence) -> QuantumCircuit:
    circ = circuit.copy()
    for action in sequence:
        if action == STOP_ACTION:
            break
        _, factory = ACTION_MENU[action]
        pm = factory()
        try:
            circ = pm.run(circ)
        except Exception:
            continue  # pass not applicable, skip
    return circ


def fitness(circuit: QuantumCircuit, sequence) -> float:
    optimized = apply_sequence(circuit, sequence)
    return -circuit_cost(optimized)


class GeneticOptimizer:
    def __init__(self, chromosome_len: int = 10, pop_size: int = 40,
                 mutation_rate: float = 0.15, crossover_rate: float = 0.7,
                 elite_frac: float = 0.1, seed: int = 0):
        self.chromosome_len = chromosome_len
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.n_elite = max(1, int(elite_frac * pop_size))
        self.rng = np.random.default_rng(seed)

    def _random_chromosome(self):
        return self.rng.integers(0, N_ACTIONS, size=self.chromosome_len)

    def _tournament_select(self, pop, fitnesses, k=3):
        idx = self.rng.choice(len(pop), size=k, replace=False)
        best = idx[np.argmax(fitnesses[idx])]
        return pop[best]

    def _crossover(self, a, b):
        if self.rng.random() > self.crossover_rate:
            return a.copy()
        point = self.rng.integers(1, self.chromosome_len)
        return np.concatenate([a[:point], b[point:]])

    def _mutate(self, chrom):
        chrom = chrom.copy()
        for i in range(len(chrom)):
            if self.rng.random() < self.mutation_rate:
                chrom[i] = self.rng.integers(0, N_ACTIONS)
        return chrom

    def run(self, circuit: QuantumCircuit, generations: int = 40, verbose=True):
        pop = [self._random_chromosome() for _ in range(self.pop_size)]
        best_chrom, best_fit = None, -np.inf
        history = []

        for gen in range(generations):
            fitnesses = np.array([fitness(circuit, c) for c in pop])
            gen_best_idx = np.argmax(fitnesses)
            if fitnesses[gen_best_idx] > best_fit:
                best_fit = fitnesses[gen_best_idx]
                best_chrom = pop[gen_best_idx].copy()
            history.append(best_fit)

            order = np.argsort(-fitnesses)
            new_pop = [pop[i].copy() for i in order[:self.n_elite]]  # elitism
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(pop, fitnesses)
                p2 = self._tournament_select(pop, fitnesses)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                new_pop.append(child)
            pop = new_pop

            if verbose and (gen + 1) % max(1, generations // 5) == 0:
                print(f"  [GA] generation {gen+1}/{generations}  best fitness={best_fit:.3f}")

        optimized_circuit = apply_sequence(circuit, best_chrom)
        return optimized_circuit, best_chrom, history
