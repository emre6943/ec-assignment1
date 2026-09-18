"""Shared result-loading code for plots and statistics."""
import csv
import re
from dataclasses import dataclass
from pathlib import Path

from constants import RESULTS_DIR

SEED_DIR = re.compile(r"seed\d+$")


@dataclass(frozen=True)
class RunLog:
    variant: str
    seed: str
    path: Path
    rows: list[dict[str, float]]

    @property
    def final_best(self) -> float:
        return self.rows[-1]["best"]



def variant_sort_key(variant: str) -> tuple[int, int | str]:
    if variant.startswith("k") and variant[1:].isdigit():
        return 0, int(variant[1:])
    if variant == "random":
        return 1, variant
    return 2, variant


def read_log(path: Path, variant: str, seed: str) -> RunLog:
    with path.open(newline="") as handle:
        rows= [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    if not rows:
        raise ValueError(f"empty log: {path}")
    return RunLog(variant=variant, seed=seed, path=path, rows=rows)




def discover_logs(results_dir: Path= RESULTS_DIR) -> list[RunLog]:
    logs: list[RunLog]=[]
    for path in sorted(results_dir.glob("*/*/log.csv")):
        variant=path.parent.parent.name
        seed =path.parent.name
        if SEED_DIR.fullmatch(seed):
            logs.append(read_log(path, variant, seed))
    return sorted(
        logs,
        key=lambda run: (variant_sort_key(run.variant), run.seed),
    )

def group_by_variant(logs: list[RunLog]) -> dict[str, list[RunLog]]:
    grouped: dict[str, list[RunLog]] = {}
    for run in logs:
        grouped.setdefault(run.variant, []).append(run)
    return grouped



def mean(values: list[float]) -> float:
    return sum(values) / len(values)

def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return variance ** 0.5
