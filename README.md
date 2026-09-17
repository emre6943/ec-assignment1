# Evolutionary Computing 2026 - Assignment 1

Body evolution assignment for X_400111 at the VU. Team of 4, worth 10 points.
Deadline Monday 21 September 2026, 09:00 CEST.

## The task

Evolve a robot body that is structurally as close as possible to five given
target bodies at the same time.

Fitness is the mean tree edit distance from the candidate body to each of the
five targets, plus one standard deviation over those five distances. Lower is
better. The standard deviation term means a body that matches some targets and
misses others scores worse than one that is evenly mediocre, so matching the
average module count is not enough.

The distance itself is a weighted Zhang-Shasha tree edit distance over the body
graphs, with costs of 1.0 for a deletion, insertion or type mismatch and 0.5 for
a rotation mismatch. The targets have 7, 11, 15, 19 and 25 modules and their own
mean pairwise distance is 15.75, which is roughly the floor any single body can
reach against the whole set.

## Files

    A1_template_2026.py    course template: loads the targets, samples one
                           random body, scores it, renders a frame
    tree_edit_distance.py  the distance and the fitness function
    target_bodies/         the five target graphs as networkx DiGraph JSON

All three are the unmodified course copies from the ARIEL fork at
github.com/AndrzejSzczepura/EvolutionaryComputing2026. Our own work goes in new
files alongside them.

## Setup

This repo holds only our own code. It has no environment of its own - every
command runs against the ARIEL clone, which must sit **next to** this directory
and must be **named `ariel`**:

    EvolutionaryComputing/
    |-- ariel/         <- the course fork, cloned and renamed (see below)
    `-- assignment1/   <- this repo

From scratch:

    # 1. install uv, if you do not have it
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # 2. make the parent directory and clone both repos side by side
    mkdir -p EvolutionaryComputing && cd EvolutionaryComputing
    git clone https://github.com/AndrzejSzczepura/EvolutionaryComputing2026.git ariel
    git clone https://github.com/emre6943/ec-assignment1.git assignment1

    # 3. build the environment (Python 3.12 + MuJoCo 3.8.0, a few minutes)
    cd ariel && uv sync && cd ../assignment1

Note the trailing `ariel` on the first clone. Without it git creates
`EvolutionaryComputing2026/`, and every command below - which all say
`--project ../ariel` - fails with "No such file or directory". Renaming the
folder afterwards is equally fine.

Check it worked:

    uv run --project ../ariel python A1_template_2026.py   # course template
    uv run --project ../ariel python -m pytest tests -q    # our tests, 10 passing

Both commands run from `assignment1/`, never from `ariel/`. There is no
`uv sync` to run here and no virtualenv to activate - `--project ../ariel`
points uv at the environment each time, so it has to be on every command.

Output frames land in `__data__/`, which is not tracked.

## Run

Our code, next to the course files:

    ea.py            the EA (block A) - tree genotype on ariel.ec, fitness from
                     tree_edit_distance.py, one CLI flag per experiment setting
    recombine.py     face-aligned multi-parent recombination (block B owns it;
                     the signature is fixed, the internals may change)
    tests/           pytest: 8 operator unit tests + 2 EA tests

Both are complete and tested. Still missing: `baseline.py`, `plot.py`,
`stats.py` and `run_all.sh` (block C), and the report (block D).

One EA run, all defaults (k=2, seed 1, pop 50, 100 generations, cap 20 modules,
tournament 3, p_xo 0.7):

    uv run --project ../ariel python ea.py

The experiment variable is `--parents`; `--seed` picks the repeat. Everything
lands in `results/k<K>/seed<S>/` unless `--out` says otherwise:

    uv run --project ../ariel python ea.py --parents 4 --seed 3
    uv run --project ../ariel python ea.py --pop 10 --gens 5 --out /tmp/smoke   # ~1 s

Flags: `--parents K` (2), `--seed S` (1), `--pop` (50), `--gens` (100),
`--max-modules` (20, core included), `--tournament` (3), `--pxo` (0.7),
`--out DIR`. `random`, `numpy` and `torch` are all seeded from `--seed`; the
same seed reproduces the same `log.csv`.

Each run writes:

    log.csv       one row per generation, starting at gen 0 (the evaluated
                  initial population), so `--gens G` gives G + 1 rows and
                  evals == pop * (gen + 1) on every row. Columns: gen, evals,
                  best, mean, worst, diversity (mean pairwise tree edit
                  distance over a random sample of at most 20 survivors, drawn
                  from its own RNG stream so the diagnostic never perturbs the
                  evolution), fallbacks (cumulative count of recombinations
                  that had to fall back to parent 0), cap_fallbacks (cumulative
                  count of children replaced by a parent clone because 20
                  shrink mutations could not get them under the module cap)
    best.json     the best genome of the run as a TreeGenome dict
    config.json   the CLI arguments the run was started with
    database.db   ariel.ec's SQLite log of every individual, every generation

`results/` is gitignored. A full run (50 x 100 = 5 000 evaluations + the
initial 50, so 5 050 in total) takes about 11 s on a laptop, so the whole
experiment - 3 variants x 5 seeds, plus the baseline - is a few minutes rather
than an overnight job.

The output directory is built from `--parents` and `--seed` only. `--pop`,
`--gens` and `--pxo` do not appear in it, so two runs differing only in those
would overwrite each other; pass `--out` explicitly when sweeping anything
other than k and the seed.

Reproducibility: the same flags give a byte-identical `log.csv` and
`best.json` across separate processes and across different `PYTHONHASHSEED`
values, and whether `--out` is a fresh directory or one that already holds a
previous run. There are three seeded streams: the global `random` module (which
is what ARIEL's own tree operators draw from), the EA's own `random.Random`
(tournaments, the crossover coin flip, parent draws, mutation choice, and the
`rng` handed to `recombine`), and a separate offset stream used only by the
diversity diagnostic, so switching that diagnostic on or off cannot change the
evolution. `recombine.py` has no generator of its own at all.

`ea.py` deletes any old `database.db` itself before handing the path to
`ariel.ec.EA`. That matters because ARIEL's own "deleting existing
database" warning is rendered by Rich, Rich turns file paths into links, and
`rich.style.Style` draws `random.randint()` for every linked style, which
advances the global `random` module ARIEL's tree operators depend on. One
hidden draw was enough to make a run into a reused directory diverge from a
run into a fresh one. Do not print paths through Rich while a run is in
progress for the same reason.

Parents are drawn from the tournament winners weighted by the number of
tournaments each won, but an individual is never used twice in one
recombination unless fewer than k distinct winners exist, so "k parents"
really means k different genomes.

The operator's core faces come from ARIEL's own
`ALLOWED_FACES[ModuleType.CORE]` rather than a hardcoded list, so if the course
ever enables the TOP and BOTTOM faces the operator picks them up instead of
silently discarding those subtrees.

Tests:

    uv run --project ../ariel python -m pytest tests -q

## Constraints

Genome is either `ariel.ec.genotypes.nde` or `ariel.ec.genotypes.tree`. CPPN
exists in ARIEL but is not allowed for this assignment. Nothing under
`ariel/src/ariel` may be edited. ARIEL's built-in operators can be used as
building blocks but the EA design has to be ours.

## What we hand in

One zip named after the group number, containing a folder of the same name with
the report as `groupnumber.pdf` and the code.

The report is at most three pages excluding the cover page and bibliography, in
the GECCO19 template. The cover page carries the course name, the assignment
number and name, the team number, and every member's name and student ID. It
needs an introduction stating a specific research question or hypothesis, a
methods section covering the algorithm, its parameters, the experimental setup,
the fitness function and the evaluation budget, then results and discussion,
conclusions and a bibliography.

The experiments behind it are two EA variants differing in one aspect, a random
search baseline at the same evaluation budget, at least five independent repeats
of each, and a plot of fitness over generations with mean and spread across
runs. The grade is on the report; the code is checked for correctness and for
agreeing with what the report claims.
