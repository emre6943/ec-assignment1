"""Smoke test: the EA runs end to end and writes its outputs."""

import csv
import json

from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.validation import validate_genome_dict

from ea import build_parser, run
from recombine import module_count


def test_ea_smoke(tmp_path):
    args = build_parser().parse_args([
        "--pop", "6", "--gens", "3", "--parents", "3", "--seed", "1",
        "--out", str(tmp_path),
    ])
    out = run(args)
    assert out == tmp_path

    with (out / "log.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4  # generation 0 (initial population) + 3 EA steps
    assert [r["gen"] for r in rows] == ["0", "1", "2", "3"]
    assert set(rows[0]) == {
        "gen", "evals", "best", "mean", "worst", "diversity",
        "fallbacks", "cap_fallbacks",
    }
    bests = [float(r["best"]) for r in rows]
    assert bests == sorted(bests, reverse=True)  # mu+lambda: best never gets worse
    assert [int(r["evals"]) for r in rows] == [6, 12, 18, 24]  # pop * (gen + 1)

    assert (out / "best.json").exists()
    genome = TreeGenome.from_dict(json.loads((out / "best.json").read_text()))
    validate_genome_dict(genome.to_dict())
    assert module_count(genome) <= args.max_modules

    config = json.loads((out / "config.json").read_text())
    assert config["parents"] == 3 and config["pop"] == 6 and config["gens"] == 3

    assert (out / "database.db").exists()


def test_same_seed_reproduces_into_reused_and_fresh_dirs(tmp_path):
    """Regression: a pre-existing database.db in --out used to change the run.

    ARIEL logs a Rich warning when it deletes an existing database, Rich links
    the path, and rich.style.Style draws random.randint() for linked styles,
    advancing the global random module the tree operators use.
    """
    argv = ["--pop", "6", "--gens", "3", "--parents", "4", "--seed", "2"]

    fresh = tmp_path / "fresh"
    run(build_parser().parse_args([*argv, "--out", str(fresh)]))
    reference = (fresh / "log.csv").read_text()
    best_reference = (fresh / "best.json").read_text()

    # second run into the same directory, database.db and log.csv still there
    run(build_parser().parse_args([*argv, "--out", str(fresh)]))
    assert (fresh / "log.csv").read_text() == reference
    assert (fresh / "best.json").read_text() == best_reference

    # and into a brand-new directory
    other = tmp_path / "other"
    run(build_parser().parse_args([*argv, "--out", str(other)]))
    assert (other / "log.csv").read_text() == reference
