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

Python 3.12 and MuJoCo 3.8.0, managed with uv from the ARIEL clone that sits
next to this directory:

    cd ../ariel
    uv sync

Run from here against that environment:

    uv run --project ../ariel python A1_template_2026.py

Output frames land in `__data__/`, which is not tracked.

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
