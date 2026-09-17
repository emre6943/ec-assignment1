"""EC Assignment 1 - the EA (block A).

Evolves ARIEL tree genomes towards the five target bodies in ``target_bodies/``
using the course fitness: mean tree edit distance to the targets plus one
standard deviation, lower is better.

The single experimental variable is ``--parents``: how many parents the
face-aligned recombination in ``recombine.py`` draws from. Every other setting
is fixed by the CLI defaults, which match the team task list.

Two conventions that the logs and the report depend on:

* Generation 0 is the evaluated initial population, so ``--gens G`` writes
  G + 1 rows and ``evals == pop * (gen + 1)`` holds on every row.
* A module is a node of the tree, core included, so ``--max-modules 20`` means
  at most 20 nodes.

``run(args)`` is importable for tests; ``main()`` only parses the CLI.
"""

# No ``from __future__ import annotations`` here: ``EAOperation`` checks the
# runtime annotation of its first parameter against ``Population``, and
# stringified annotations fail that check.

import argparse
import copy
import csv
import json
import random
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from rich.console import Console

from ariel.body_phenotypes.robogen_lite.decoders._blueprint import (
    load_graph_from_json,
)
from ariel.ec import EA, EAOperation, Individual, Population
from ariel.ec.genotypes.tree.operators import (
    _prune_invalid_edges,
    mutate_hoist,
    mutate_replace_node,
    mutate_shrink,
    mutate_subtree_replacement,
    random_tree,
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.validation import validate_genome_dict

import recombine as recombine_mod
from recombine import module_count, recombine
from tree_edit_distance import (
    mean_plus_std_tree_edit_distance,
    tree_edit_distance,
)

HERE = Path(__file__).parent
TARGET_DIR = HERE / "target_bodies"

MUTATION_PROBABILITIES: tuple[tuple[str, float], ...] = (
    ("point", 0.4),
    ("subtree", 0.4),
    ("shrink", 0.1),
    ("hoist", 0.1),
)
MAX_SHRINK_ATTEMPTS = 20
DIVERSITY_SAMPLE_SIZE = 20
DIVERSITY_SEED_OFFSET = 1_000_003

LOG_COLUMNS = (
    "gen", "evals", "best", "mean", "worst", "diversity",
    "fallbacks", "cap_fallbacks",
)

console = Console()


def load_targets(target_dir: Path = TARGET_DIR) -> list[nx.DiGraph]:
    paths = sorted(target_dir.glob("*.json"))
    if not paths:
        msg = f"no target bodies found in {target_dir}"
        raise FileNotFoundError(msg)
    return [load_graph_from_json(path) for path in paths]


def fitness_function(body: nx.DiGraph, targets: list[nx.DiGraph]) -> float:
    """Score one body against the whole target set. Lower is better."""
    return mean_plus_std_tree_edit_distance(body, targets)


def genome_of(individual: Individual) -> TreeGenome:
    """Return a genome sharing nothing with the individual's stored dict.

    ``TreeGenome.from_dict`` keeps a reference to the ``edges`` list it is
    given, so without the deep copy a mutation on a child would rewrite its
    parent's genotype in place.
    """
    return TreeGenome.from_dict(copy.deepcopy(individual.genotype))


def new_individual(genome: TreeGenome) -> Individual:
    individual = Individual()
    individual.genotype = genome.to_dict()
    individual.tags["ps"] = False
    individual.tags["ps_wins"] = 0
    return individual


def is_valid(genome: TreeGenome) -> bool:
    try:
        validate_genome_dict(genome.to_dict())
    except ValueError:
        return False
    return True


class Experiment:
    """One EA run: the configuration plus the operations as bound methods."""

    def __init__(self, args: argparse.Namespace, out: Path) -> None:
        self.args = args
        self.out = out
        self.rng = random.Random(args.seed)
        # A separate stream, so switching the diagnostic on or off cannot
        # change the evolution.
        self.diversity_rng = random.Random(args.seed + DIVERSITY_SEED_OFFSET)
        self.targets = load_targets()
        self.generation = 0
        self.evals = 0
        self.cap_fallbacks = 0
        self.log_path = out / "log.csv"
        self.started_at = time.time()

        with self.log_path.open("w", newline="") as handle:
            csv.writer(handle).writerow(LOG_COLUMNS)

    def initial_population(self) -> Population:
        population = Population.empty()
        for _ in range(self.args.pop):
            # random_tree(n) adds up to n non-core modules, so n = cap - 1
            # keeps the whole tree, core included, at or under the cap.
            genome = random_tree(self.args.max_modules - 1)
            population.append(new_individual(genome))
        return population

    def tournament(self, candidates: list[Individual]) -> Individual:
        """Run one tournament. ``min`` wins because lower fitness is better."""
        size = min(self.args.tournament, len(candidates))
        contestants = self.rng.sample(candidates, size)
        return min(contestants, key=lambda individual: individual.fitness)

    def parent_selection(self, population: Population) -> Population:
        """Run ``pop`` tournaments and tag the winners on the individuals.

        ARIEL operations must take and return a ``Population``, so the winners
        are recorded as tags rather than returned: ``ps`` marks a winner and
        ``ps_wins`` counts how many tournaments it took. ``mating_pool`` turns
        those counts back into a list, which is where selection pressure
        actually comes from.
        """
        alive = population.alive.to_list()
        for individual in alive:
            individual.tags["ps"] = False
            individual.tags["ps_wins"] = 0
        for _ in range(self.args.pop):
            winner = self.tournament(alive)
            winner.tags["ps"] = True
            winner.tags["ps_wins"] = int(winner.tags.get("ps_wins", 0)) + 1
        return population

    def mating_pool(self, population: Population) -> list[Individual]:
        """List every tournament winner once per win it scored."""
        pool: list[Individual] = []
        for individual in population.alive:
            if individual.tags.get("ps"):
                wins = int(individual.tags.get("ps_wins", 1))
                pool.extend([individual] * wins)
        if not pool:
            console.log("[yellow]no tagged parents - using whole population[/yellow]")
            pool = population.alive.to_list()
        return pool

    def pick_mutation(self) -> str:
        """Roulette over ``MUTATION_PROBABILITIES`` using cumulative bounds."""
        roll = self.rng.random()
        cumulative = 0.0
        for name, probability in MUTATION_PROBABILITIES:
            cumulative += probability
            if roll < cumulative:
                return name
        return MUTATION_PROBABILITIES[-1][0]

    def mutate_once(self, genome: TreeGenome) -> None:
        mutation = self.pick_mutation()
        if mutation == "point":
            mutate_replace_node(genome)
        elif mutation == "subtree":
            mutate_subtree_replacement(genome, max_modules=self.args.max_modules)
        elif mutation == "shrink":
            mutate_shrink(genome)
        else:
            mutate_hoist(genome)

    def enforce_cap(self, genome: TreeGenome) -> bool:
        """Shrink until under the module cap. False if it gave up."""
        for _ in range(MAX_SHRINK_ATTEMPTS):
            if module_count(genome) <= self.args.max_modules:
                return True
            mutate_shrink(genome)
        return module_count(genome) <= self.args.max_modules

    def draw_parents(self, pool: list[Individual], count: int) -> list[Individual]:
        """Draw ``count`` distinct individuals from the win-weighted pool.

        A plain ``sample`` would return the same individual twice, quietly
        turning a k-parent recombination into a (k-1)-parent one. Each draw is
        still weighted by wins, but a chosen individual leaves the pool.
        Repeats are only allowed when fewer than ``count`` winners exist.
        """
        remaining = list(pool)
        chosen: list[Individual] = []
        while remaining and len(chosen) < count:
            pick = self.rng.choice(remaining)
            chosen.append(pick)
            remaining = [ind for ind in remaining if ind is not pick]
        if len(chosen) < count:
            chosen.extend(self.rng.choices(chosen, k=count - len(chosen)))
        return chosen

    def make_child(self, pool: list[Individual]) -> TreeGenome:
        """Build one child: recombine or clone, then always mutate exactly once.

        The two ``return copy.deepcopy(parents[0])`` branches are safety nets,
        not normal paths: a child that cannot be shrunk under the module cap,
        or that still fails validation after pruning, is discarded and replaced
        by an unmodified parent so nothing invalid can enter the population.
        Both read 0 in practice and are reported in the log.
        """
        if self.rng.random() < self.args.pxo:
            chosen = self.draw_parents(pool, self.args.parents)
            parents = [genome_of(individual) for individual in chosen]
            child = recombine(parents, self.rng)
        else:
            parents = [genome_of(self.rng.choice(pool))]
            child = copy.deepcopy(parents[0])

        self.mutate_once(child)

        if not self.enforce_cap(child):
            self.cap_fallbacks += 1
            return copy.deepcopy(parents[0])

        _prune_invalid_edges(child)
        if not is_valid(child) or module_count(child) > self.args.max_modules:
            return copy.deepcopy(parents[0])
        return child

    def reproduction(self, population: Population) -> Population:
        """Append ``pop`` children, so the population briefly holds 2 x pop."""
        pool = self.mating_pool(population)
        children = [
            new_individual(self.make_child(pool))
            for _ in range(self.args.pop)
        ]
        population.extend(children)
        return population

    def evaluate(self, population: Population) -> Population:
        """Evaluate whatever still needs it.

        ARIEL clears ``requires_eval`` inside the ``fitness`` setter, so nobody
        is scored twice and the run lands on exactly pop x (gens + 1) evals.
        """
        for individual in population:
            if individual.alive and individual.requires_eval:
                body = genome_of(individual).to_networkx()
                individual.fitness = fitness_function(body, self.targets)
                self.evals += 1
        return population

    def survivor_selection(self, population: Population) -> Population:
        """mu + lambda: keep the best ``pop`` of parents and children together.

        Losers are flagged rather than removed - ARIEL keeps them in the list
        for the database log, and ``population.alive`` filters them everywhere
        else. Identity is compared with ``id`` because individuals are not
        hashable by value.
        """
        ranked = population.alive.sort(sort="min", attribute="fitness_")
        survivors = {id(individual) for individual in ranked[: self.args.pop]}
        for individual in population:
            if id(individual) not in survivors:
                individual.alive = False
        return population

    def diversity(self, alive: list[Individual]) -> float:
        """Mean pairwise tree edit distance over a sample of the survivors."""
        size = min(DIVERSITY_SAMPLE_SIZE, len(alive))
        sample = self.diversity_rng.sample(alive, size)
        bodies = [genome_of(individual).to_networkx() for individual in sample]
        distances = [
            tree_edit_distance(one, other)
            for index, one in enumerate(bodies)
            for other in bodies[index + 1 :]
        ]
        return float(np.mean(distances)) if distances else 0.0

    def log(self, population: Population) -> Population:
        """Append one CSV row and print one line for the current generation."""
        alive = population.alive.to_list()
        fitnesses = [individual.fitness for individual in alive]
        best = min(fitnesses)
        mean = float(np.mean(fitnesses))
        worst = max(fitnesses)
        diversity = self.diversity(alive)
        fallbacks = recombine_mod.FALLBACKS

        with self.log_path.open("a", newline="") as handle:
            csv.writer(handle).writerow([
                self.generation, self.evals,
                f"{best:.4f}", f"{mean:.4f}", f"{worst:.4f}", f"{diversity:.4f}",
                fallbacks, self.cap_fallbacks,
            ])
        console.print(
            f"gen {self.generation:>3} ev {self.evals:>5} "
            f"best {best:6.2f} mean {mean:6.2f} worst {worst:6.2f} "
            f"div {diversity:5.2f} fb {fallbacks} cap {self.cap_fallbacks} "
            f"{time.time() - self.started_at:.0f}s",
            highlight=False,
        )
        self.generation += 1
        return population

    def build_ea(self, population: Population) -> EA:
        # Reproducibility guard. ARIEL warns through Rich when it deletes an
        # existing database.db, Rich turns the path into a link, and
        # rich.style.Style draws random.randint() per linked style - which
        # advances the global random module the tree operators use. Delete the
        # file ourselves so the warning never fires, and restore the global
        # state around the constructor in case anything else renders a path.
        database = self.out / "database.db"
        database.unlink(missing_ok=True)
        rng_state = random.getstate()
        ea = EA(
            population,
            operations=[
                EAOperation(self.parent_selection),
                EAOperation(self.reproduction),
                EAOperation(self.evaluate),
                EAOperation(self.survivor_selection),
                EAOperation(self.log),
            ],
            num_steps=self.args.gens,
            is_maximisation=False,
            db_file_path=database,
            db_handling="delete",
            quiet=True,
        )
        random.setstate(rng_state)
        return ea

    def evolve(self) -> Individual:
        """Evaluate and log generation 0, then hand the population to ARIEL.

        Logging before ``ea.run()`` is what makes generation 0 the evaluated
        initial population, so ``--gens G`` produces G + 1 rows.
        """
        population = self.log(self.evaluate(self.initial_population()))
        ea = self.build_ea(population)
        ea.run()
        return ea.get_solution("best", only_alive=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EC A1 - multi-parent tree EA")
    parser.add_argument(
        "--parents", type=int, default=2, help="parents per recombination (k)",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pop", type=int, default=50)
    parser.add_argument("--gens", type=int, default=100)
    parser.add_argument("--max-modules", dest="max_modules", type=int, default=20)
    parser.add_argument("--tournament", type=int, default=3)
    parser.add_argument(
        "--pxo", type=float, default=0.7, help="recombination probability",
    )
    parser.add_argument(
        "--out", type=str, default=None, help="default results/k{K}/seed{S}",
    )
    return parser


def resolve_out(args: argparse.Namespace) -> Path:
    if args.out is None:
        return HERE / "results" / f"k{args.parents}" / f"seed{args.seed}"
    return Path(args.out)


def seed_everything(seed: int) -> None:
    random.seed(seed)  # ARIEL's tree operators use the global random module
    np.random.seed(seed)
    torch.manual_seed(seed)


def run(args: argparse.Namespace) -> Path:
    out = resolve_out(args)
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    recombine_mod.reset_fallbacks()

    console.rule(
        f"[bold]EA  k={args.parents}  seed={args.seed}  "
        f"pop={args.pop}  gens={args.gens}",
    )
    experiment = Experiment(args, out)
    best = experiment.evolve()

    best_genome = genome_of(best)
    with (out / "best.json").open("w", encoding="utf-8") as handle:
        json.dump(best_genome.to_dict(), handle, indent=2)
    with (out / "config.json").open("w", encoding="utf-8") as handle:
        json.dump({**vars(args), "out": str(out)}, handle, indent=2)

    console.print(
        f"done: best {best.fitness:.4f} ({module_count(best_genome)} modules), "
        f"{experiment.evals} evals, {recombine_mod.FALLBACKS} fallbacks, "
        f"{experiment.cap_fallbacks} cap fallbacks -> {out}",
        highlight=False,
    )
    return out


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
