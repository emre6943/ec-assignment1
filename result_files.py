"""Shared output file helpers."""

import argparse
import json
from pathlib import Path

from ariel.ec.genotypes.tree.tree_genome import TreeGenome




def save_genome_outputs(
    out: Path,
    args: argparse.Namespace,
    genome: TreeGenome,
) -> None:
    with (out / "best.json").open("w", encoding="utf-8") as handle:
        json.dump(genome.to_dict(), handle, indent=2)
    with (out / "config.json").open("w", encoding="utf-8") as handle:
        json.dump({**vars(args), "out": str(out)}, handle, indent=2)
