"""Core random-search baseline logic."""

from dataclasses import dataclass

import networkx as nx
import numpy as np

from ariel.ec.genotypes.tree.operators import random_tree
from ariel.ec.genotypes.tree.tree_genome import TreeGenome

from constants import DIVERSITY_SAMPLE_SIZE
from ea import fitness_function
from metrics import mean_pairwise_tree_distance


@dataclass(frozen=True)
class EvaluatedGenome:
    genome: TreeGenome
    fitness: float


@dataclass(frozen=True)
class GenerationStats:
    generation: int
    evals: int
    best_so_far: float
    mean: float
    worst: float
    diversity: float


def diversity(genomes: list[TreeGenome]) -> float:
    sample = genomes[:DIVERSITY_SAMPLE_SIZE]
    bodies = [genome.to_networkx() for genome in sample]
    return mean_pairwise_tree_distance(bodies)


def evaluate_batch(
    pop_size: int,
    max_modules: int,
    targets: list[nx.DiGraph],
) -> list[EvaluatedGenome]:
    genomes = [random_tree(max_modules - 1) for _ in range(pop_size)]
    return [
        EvaluatedGenome(
            genome,
            fitness_function(genome.to_networkx(), targets),
        )
        for genome in genomes
    ]


def best_in(batch: list[EvaluatedGenome]) -> EvaluatedGenome:
    return min(batch, key=lambda item: item.fitness)


def summarize_generation(
    generation: int,
    pop_size: int,
    batch: list[EvaluatedGenome],
    best_so_far: float,
) -> GenerationStats:
    fitnesses = [item.fitness for item in batch]
    genomes = [item.genome for item in batch]
    return GenerationStats(
        generation=generation,
        evals=pop_size * (generation + 1),
        best_so_far=best_so_far,
        mean=float(np.mean(fitnesses)),
        worst=max(fitnesses),
        diversity=diversity(genomes),
    )
