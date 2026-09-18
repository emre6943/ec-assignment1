import argparse
import copy
import csv
import math
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
from constants import (
    DEFAULT_GENS,
    DEFAULT_MAX_MODULES,
    DEFAULT_POP,
    DEFAULT_SEED,
    DIVERSITY_SEED_OFFSET,
    DIVERSITY_SAMPLE_SIZE,
    LOG_COLUMNS,
    MAX_SHRINK_ATTEMPTS,
    MUTATION_PROBABILITIES,
    RESULTS_DIR,
    TARGET_DIR,
    ea_result_dir,
)
from metrics import mean_pairwise_tree_distance
from result_files import save_genome_outputs
from recombine import module_count, recombine
from tree_edit_distance import mean_plus_std_tree_edit_distance

console= Console()
def load_targets(target_dir: Path = TARGET_DIR) -> list[nx.DiGraph]:
    paths=sorted(target_dir.glob("*.json"))
    if not paths:
        msg=f"no target bodies found in {target_dir}"
        raise FileNotFoundError(msg)
    return [load_graph_from_json(path) for path in paths]
def fitness_function(body: nx.DiGraph, targets: list[nx.DiGraph]) -> float:
    return mean_plus_std_tree_edit_distance(body, targets)

def genome_of(individual: Individual) -> TreeGenome:
    return TreeGenome.from_dict(copy.deepcopy(individual.genotype))




def new_individual(genome: TreeGenome) -> Individual:
    individual =Individual()
    individual.genotype =genome.to_dict()
    individual.tags["ps"]= False
    individual.tags["ps_wins"] =0
    return individual

def is_valid(genome: TreeGenome) -> bool:
    try:
        validate_genome_dict(genome.to_dict())
    except ValueError:
        return False
    return True



class Experiment:
    def __init__(self, args: argparse.Namespace, out: Path) -> None:
        self.args =args
        self.out= out
        self.rng= random.Random(args.seed)
        self.diversity_rng =random.Random(args.seed + DIVERSITY_SEED_OFFSET)
        self.targets=load_targets()
        self.generation =0
        self.evals=0
        self.cap_fallbacks= 0
        self.forced_mutations= 0
        self.log_path =out / "log.csv"
        self.started_at= time.time()


        with self.log_path.open("w", newline="") as handle:
            csv.writer(handle).writerow(LOG_COLUMNS)

    def initial_population(self) -> Population:
        population =Population.empty()
        for _ in range(self.args.pop):
            genome =random_tree(self.args.max_modules -1)
            population.append(new_individual(genome))
        return population
    def tournament(self, candidates: list[Individual]) -> Individual:
        size = min(self.args.tournament, len(candidates))
        contestants = self.rng.sample(candidates, size)
        return min(contestants, key=lambda individual: individual.fitness)

    def award_slot(self, individual: Individual) -> None:
        individual.tags["ps"]= True
        individual.tags["ps_wins"] =int(individual.tags["ps_wins"]) + 1

    def parent_selection(self, population: Population) -> Population:
        alive =population.alive.to_list()
        for individual in alive:
            individual.tags["ps"] =False
            individual.tags["ps_wins"] = 0

        # ceil, not int: a rate above 0 must buy at least one random slot
        num_random= math.ceil(self.args.pop * self.args.immigrants)
        num_tournaments =self.args.pop - num_random

        for _ in range(num_tournaments):
            self.award_slot(self.tournament(alive))
        for _ in range(num_random):
            self.award_slot(self.rng.choice(alive))
        return population

    def mating_pool(self, population: Population) -> list[Individual]:
        pool: list[Individual] =[]
        for individual in population.alive:
            if individual.tags.get("ps"):
                wins= int(individual.tags.get("ps_wins", 1))
                pool.extend([individual] *wins)
        if not pool:
            console.log("[yellow]no tagged parents -using whole population[/yellow]")
            pool= population.alive.to_list()
        return pool

    def pick_mutation(self) -> str:
        roll =self.rng.random()
        cumulative=0.0
        for name, probability in MUTATION_PROBABILITIES:
            cumulative += probability
            if roll <cumulative:
                return name
        return MUTATION_PROBABILITIES[-1][0]



    def mutate_once(self, genome: TreeGenome) -> None:
        mutation= self.pick_mutation()
        if mutation== "point":
            mutate_replace_node(genome)
        elif mutation =="subtree":
            mutate_subtree_replacement(genome, max_modules=self.args.max_modules)
        elif mutation =="shrink":
            mutate_shrink(genome)
        else:
            mutate_hoist(genome)

    def enforce_cap(self, genome: TreeGenome) -> bool:
        for _ in range(MAX_SHRINK_ATTEMPTS):
            if module_count(genome)<= self.args.max_modules:
                return True
            mutate_shrink(genome)
        return module_count(genome)<= self.args.max_modules

    def draw_parents(self, pool: list[Individual], count: int) -> list[Individual]:
        remaining =list(pool)
        chosen: list[Individual]= []
        while remaining and len(chosen) < count:
            pick = self.rng.choice(remaining)
            chosen.append(pick)

            # Drop EVERY slot belonging to that individual
            still_available = []
            for candidate in remaining:
                if candidate is not pick:
                    still_available.append(candidate)
            remaining = still_available

        # Only reachable when there were fewer distinct winners than parents
        if len(chosen) < count:
            shortfall = count - len(chosen)
            chosen.extend(self.rng.choices(chosen, k=shortfall))
        return chosen

    def make_child(self, pool: list[Individual]) -> TreeGenome:
        """Build one child: recombine or clone, then usually mutate it once.

        The two ``return copy.deepcopy(parents[0])`` branches are safety nets,
        not normal paths: a child that cannot be shrunk under the module cap,
        or that still fails validation after pruning, is discarded and replaced
        by an unmodified parent so nothing invalid can enter the population.
        Both read 0 in practice and are reported in the log.
        """
        recombined = self.rng.random() < self.args.pxo # probability of recombination
        if recombined:
            chosen = self.draw_parents(pool, self.args.parents)
            parents = []
            for individual in chosen:
                parents.append(genome_of(individual))
            child = recombine(parents, self.rng)
        else:
            parents = [genome_of(self.rng.choice(pool))]
            child = copy.deepcopy(parents[0])

        mutated = self.rng.random() < self.args.pmut
        if mutated:
            self.mutate_once(child)
        elif not recombined:
            # if the child is a clone then force mutation to avoid exact copies of the parent
            self.mutate_once(child)
            self.forced_mutations += 1

        # fallbacks
        if not self.enforce_cap(child):
            self.cap_fallbacks += 1
            return copy.deepcopy(parents[0])

        _prune_invalid_edges(child)
        if not is_valid(child) or module_count(child) > self.args.max_modules:
            return copy.deepcopy(parents[0])
        return child

    def reproduction(self, population: Population) -> Population:
        """Append ``pop`` children to the population.

        The population holds 2 x pop ALIVE individuals until
        ``survivor_selection`` culls it back
        """
        pool = self.mating_pool(population)

        children = []
        for _ in range(self.args.pop):
            child_genome = self.make_child(pool)
            children.append(new_individual(child_genome))

        population.extend(children)
        return population

    def evaluate(self, population: Population) -> Population:
        """Evaluate whatever still needs it."""
        for individual in population:
            if individual.alive and individual.requires_eval:
                body = genome_of(individual).to_networkx()
                individual.fitness = fitness_function(body, self.targets)
                self.evals += 1
        return population

    def survivor_selection(self, population: Population) -> Population:
        """mu + lambda: keep the best ``pop`` of parents and children together."""
        # Parents and children together, sorted best (lowest) fitness first.
        ranked = population.alive.sort(sort="min", attribute="fitness_")
        best = ranked[: self.args.pop]

        # id() because Individual is not hashable by value, so we cannot put
        # the objects themselves in a set.
        survivor_ids = set()
        for individual in best:
            survivor_ids.add(id(individual))

        for individual in population:
            if id(individual) not in survivor_ids:
                individual.alive = False
        return population

    def diversity(self, alive: list[Individual]) -> float:
        """Mean pairwise tree edit distance over a sample of the survivors."""
        size = DIVERSITY_SAMPLE_SIZE
        if size > len(alive):
            size = len(alive)
        sample = self.diversity_rng.sample(alive, size)

        bodies = []
        for individual in sample:
            bodies.append(genome_of(individual).to_networkx())
        return mean_pairwise_tree_distance(bodies)

    def log(self, population: Population) -> Population:
        """Append one CSV row and print one line for the current generation."""
        alive = population.alive.to_list()

        fitnesses = []
        for individual in alive:
            fitnesses.append(individual.fitness)

        best = min(fitnesses)
        mean = float(np.mean(fitnesses))
        worst = max(fitnesses)
        diversity = self.diversity(alive)
        fallbacks = recombine_mod.FALLBACKS

        with self.log_path.open("a", newline="") as handle:
            csv.writer(handle).writerow([
                self.generation, self.evals,
                f"{best:.4f}", f"{mean:.4f}", f"{worst:.4f}", f"{diversity:.4f}",
                fallbacks, self.cap_fallbacks, self.forced_mutations,
            ])
        console.print(
            f"gen {self.generation:>3} ev {self.evals:>5} "
            f"best {best:6.2f} mean {mean:6.2f} worst {worst:6.2f} "
            f"div {diversity:5.2f} fb {fallbacks} cap {self.cap_fallbacks} "
            f"fm {self.forced_mutations} "
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
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--pop", type=int, default=DEFAULT_POP)
    parser.add_argument("--gens", type=int, default=DEFAULT_GENS)
    parser.add_argument(
        "--max-modules", dest="max_modules", type=int, default=DEFAULT_MAX_MODULES,
    )
    parser.add_argument("--tournament", type=int, default=3)
    parser.add_argument(
        "--immigrants",
        type=float,
        default=0.01,
        help="fraction of parent slots drawn at random, ignoring fitness "
             "(rounded up, so any value > 0 gives at least one slot)",
    )
    parser.add_argument(
        "--pxo", type=float, default=0.8, help="recombination probability",
    )
    parser.add_argument(
        "--pmut",
        type=float,
        default=0.8,
        help="mutation probability; a cloned child that skips it is mutated "
             "anyway, so no child is ever an exact copy of its parent",
    )
    parser.add_argument(
        "--out", type=str, default=None, help="default results/k{K}/seed{S}",
    )
    return parser


def resolve_out(args: argparse.Namespace) -> Path:
    if args.out is None:
        return ea_result_dir(RESULTS_DIR, args.parents, args.seed)
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
    save_genome_outputs(out, args, best_genome)

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
