"""Plot fitness convergence over evaluations."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from constants import RESULTS_DIR
from evaluation.common import (
    RunLog,
    discover_logs,
    group_by_variant,
    mean,
    sample_std,
    variant_sort_key,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot assignment results")
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "plots")
    return parser


def aligned_series(
    runs: list[RunLog],
    column: str,
) -> tuple[list[int], list[float], list[float]]:
    evals = sorted(set.intersection(*[
        {int(row["evals"]) for row in run.rows}
        for run in runs
    ]))
    means: list[float] = []
    stds: list[float] = []
    for eval_count in evals:
        values = [
            row[column]
            for run in runs
            for row in run.rows
            if int(row["evals"]) == eval_count
        ]
        means.append(mean(values))
        stds.append(sample_std(values))
    return evals, means, stds


def plot_fitness(
    grouped: dict[str, list[RunLog]],
    out: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    for variant in sorted(grouped, key=variant_sort_key):
        evals, means, stds = aligned_series(grouped[variant], "best")
        lower = [avg - std for avg, std in zip(means, stds)]
        upper = [avg + std for avg, std in zip(means, stds)]
        label = f"{variant} (n={len(grouped[variant])})"
        axis.plot(evals, means, label=label)
        axis.fill_between(evals, lower, upper, alpha=0.18)
    axis.set_xlabel("Evaluations")
    axis.set_ylabel("Best fitness")
    axis.set_title("Fitness over evaluations")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    logs = discover_logs(args.results)
    if not logs:
        raise SystemExit(f"no logs found under {args.results}")
    args.out.mkdir(parents=True, exist_ok=True)
    grouped = group_by_variant(logs)
    plot_fitness(grouped, args.out / "fitness.png")
    print(f"wrote {args.out / 'fitness.png'}")
    return args.out


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
