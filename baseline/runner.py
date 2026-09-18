"""Random-search baseline runner."""

import argparse
import csv
import time
from pathlib import Path

from rich.console import Console

from baseline.core import best_in, evaluate_batch, summarize_generation
from baseline.core import EvaluatedGenome, GenerationStats
from constants import (
    DEFAULT_GENS,
    DEFAULT_MAX_MODULES,
    DEFAULT_POP,
    DEFAULT_SEED,
    LOG_COLUMNS,
    RESULTS_DIR,
    baseline_result_dir,
)
from ea import load_targets, seed_everything
from recombine import module_count
from result_files import save_genome_outputs

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EC A1 - random-search baseline",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--pop", type=int, default=DEFAULT_POP)
    parser.add_argument("--gens", type=int, default=DEFAULT_GENS)
    parser.add_argument(
        "--max-modules",
        dest="max_modules",
        type=int,
        default=DEFAULT_MAX_MODULES,
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="default results/random/seed{S}",
    )
    return parser


def resolve_out(args: argparse.Namespace) -> Path:
    if args.out:
        return Path(args.out)
    return baseline_result_dir(RESULTS_DIR, args.seed)


def write_log_row(writer: csv.writer, stats: GenerationStats) -> None:
    writer.writerow([
        stats.generation,
        stats.evals,
        f"{stats.best_so_far:.4f}",
        f"{stats.mean:.4f}",
        f"{stats.worst:.4f}",
        f"{stats.diversity:.4f}",
        0,  # fallbacks: the baseline never recombines
        0,  # cap_fallbacks: random_tree respects the cap by construction
        0,  # forced_mutations: the baseline never mutates
    ])


def print_generation(stats: GenerationStats, started_at: float) -> None:
    console.print(
        f"gen {stats.generation:>3} ev {stats.evals:>5} "
        f"best {stats.best_so_far:6.2f} mean {stats.mean:6.2f} "
        f"worst {stats.worst:6.2f} div {stats.diversity:5.2f} "
        f"{time.time() - started_at:.0f}s",
        highlight=False,
    )


def run(args: argparse.Namespace) -> Path:
    out = resolve_out(args)
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    targets = load_targets()
    best: EvaluatedGenome | None = None
    started_at = time.time()

    console.rule(
        f"[bold]Random baseline  seed={args.seed}  "
        f"pop={args.pop}  gens={args.gens}",
    )

    with (out / "log.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(LOG_COLUMNS)

        for generation in range(args.gens + 1):
            batch = evaluate_batch(args.pop, args.max_modules, targets)
            batch_best = best_in(batch)
            if best is None or batch_best.fitness < best.fitness:
                best = batch_best

            stats = summarize_generation(
                generation,
                args.pop,
                batch,
                best.fitness,
            )
            write_log_row(writer, stats)
            print_generation(stats, started_at)

    if best is None:
        msg = "random baseline evaluated no genomes"
        raise RuntimeError(msg)

    save_genome_outputs(out, args, best.genome)
    console.print(
        f"done: best {best.fitness:.4f} "
        f"({module_count(best.genome)} modules), "
        f"{args.pop * (args.gens + 1)} evals -> {out}",
        highlight=False,
    )
    return out


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
