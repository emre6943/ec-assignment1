import argparse
from pathlib import Path

import baseline
import ea
from constants import DEFAULT_GENS, DEFAULT_MAX_MODULES, DEFAULT_SEEDS
from constants import DEFAULT_VARIANTS, baseline_result_dir, ea_result_dir
from constants import DEFAULT_POP, RESULTS_DIR
from evaluation import plot, statistics

def build_parser() -> argparse.ArgumentParser:
    parser =argparse.ArgumentParser(
        description="Run all assignment experiments",
    )
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--variants",
        type=int,
        nargs="+",
        default=DEFAULT_VARIANTS,
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--pop", type=int, default=DEFAULT_POP)
    parser.add_argument("--gens", type=int, default=DEFAULT_GENS)
    parser.add_argument("--max-modules", type=int, default=DEFAULT_MAX_MODULES)
    parser.add_argument("--force", action="store_true")
    return parser



def should_run(out: Path, force: bool) -> bool:
    if force:
        return True
    if (out / "log.csv").exists():
        print(f"skip existing {out}")
        return False
    return True



def experiment_args(args: argparse.Namespace, seed: int, out: Path) -> list[str]:
    return [
        "--seed", str(seed),
        "--pop", str(args.pop),
        "--gens", str(args.gens),
        "--max-modules", str(args.max_modules),
        "--out", str(out),
    ]


def run_ea(args: argparse.Namespace, parents: int, seed: int) -> None:
    out= ea_result_dir(args.results, parents, seed)
    if not should_run(out, args.force):
        return
    ea_args =ea.build_parser().parse_args(
        ["--parents", str(parents), *experiment_args(args, seed, out)],
    )
    ea.run(ea_args)

def run_baseline(args: argparse.Namespace, seed: int) -> None:
    out = baseline_result_dir(args.results, seed)
    if not should_run(out, args.force):
        return
    baseline_args = baseline.build_parser().parse_args(
        experiment_args(args, seed, out),
    )
    baseline.run(baseline_args)

def run_evaluation(results: Path) -> None:
    statistics.run(argparse.Namespace(
        results=results,
        out=results / "summary.csv",
    ))
    plot.run(argparse.Namespace(
        results=results,
        out=results / "plots",
    ))


def run(args: argparse.Namespace) -> None:
    for parents in args.variants:
        for seed in args.seeds:
            run_ea(args, parents, seed)
    for seed in args.seeds:
        run_baseline(args, seed)
    run_evaluation(args.results)



def main() -> None:
    run(build_parser().parse_args())



if __name__ == "__main__":
    main()
