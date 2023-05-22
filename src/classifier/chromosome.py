from __future__ import annotations
import numpy as np
import random
from typing import Tuple

class Chromosome(object):
    def __init__(self, chromosome: np.ndarray = None, n: int = None) -> None:
        self.chromosome = chromosome if chromosome is not None else np.random.randint(2, size=n)
        self.model = None
    def mutate(self, p: float = 0.05) -> Chromosome:
        mutation_flags = np.random.choice([1, 0], size=self.chromosome.shape[0], p=[p, 1 - p])
        out = []
        for x, y in zip(self.chromosome, mutation_flags):
            if y == 1:
                out.append(not x)
            else:
                out.append(x)
        return Chromosome(chromosome=np.array(out))
    def cross(self, other: Chromosome, p: float) -> Tuple[Chromosome, Chromosome]:
        if np.random.rand() >= p:
            return self, other

        crossover_index = random.randint(1, self.chromosome.shape[0] - 2)
        a = Chromosome(chromosome=np.concatenate((self.chromosome[:crossover_index], other.chromosome[crossover_index:])))
        b = Chromosome(chromosome=np.concatenate((other.chromosome[:crossover_index], self.chromosome[crossover_index:])))
        return a, b
    def evaluate(self, sensitivity: float, latency: float, specifity: float) -> int:
        self.sensitivity = sensitivity
        self.sensitivity_margin = sensitivity * 0.05
        self.latency = latency
        self.latency_margin = latency * 0.05
        self.specifity = specifity
        self.specifity_margin = specifity * 0.05
    def is_dominating(self, chromosome: Chromosome) -> bool:
        if np.abs(self.sensitivity - chromosome.sensitivity) >= self.sensitivity_margin:
            if self.sensitivity > chromosome.sensitivity:
                return True
            if np.abs(self.latency - chromosome.latency) >= self.latency_margin:
                if self.latency < chromosome.latency:
                    return True
                if np.abs(self.specifity - chromosome.specifity) >= self.specifity_margin:
                    if self.specifity < chromosome.specifity:
                        return True
        return False