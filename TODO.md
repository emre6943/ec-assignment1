# EC Assignment 1 — Team task list

**Deadline: Mon 2026-09-21, 09:00.** Late = −0,5 pt/day.

**Research question.** How does the number of parents involved in recombination affect the
convergence and final fitness of evolved robot morphologies?

**Hypothesis.** Increasing the number of parents increases genetic diversity and exploration, but
too many parents disrupts useful genetic structure and therefore reduces convergence performance.

**Design.** Tree genotype, max 20 modules. One EA, one operator (face-aligned multi-parent
recombination), one variable: number of parents k ∈ {2, 4, 8}. k=2 is the control. Everything
else fixed. Plus a random-search baseline at the same evaluation budget.

Fixed settings (do not change between variants) — these are the `ea.py` CLI defaults,
so a variant run only needs `--parents k --seed s`:
- population 50, 100 generations → **5 050** evaluations per run (gen 0 = the initial 50 + 100 × 50)
- tournament parent selection, k_tournament = 3 (`--tournament 3`)
- **`--immigrants 0.01`** — ⌈0.01 × 50⌉ = **1** of the 50 parent slots per generation is filled by
  an individual drawn uniformly at random, fitness ignored, so weak-but-different bodies still
  breed. Note the ceiling: plain flooring would give 0 slots and silently disable it.
- **crossover probability 0.8** (`--pxo 0.8`) — ~40 of 50 children are recombined from k parents,
  the other ~10 are clones of a single parent
- **mutation probability 0.8** (`--pmut 0.8`) — exactly one mutation
  (point 0.4 / subtree 0.4 / shrink 0.1 / hoist 0.1). A child that was *cloned* and then skipped
  mutation is mutated anyway, so no child is ever an exact duplicate of its parent; those are
  counted in the `forced_mutations` log column
- survivor selection μ+λ (keep best 50 of parents + children)
- module cap 20, enforced after variation (`--max-modules 20`)
- 5 seeds per variant: 1, 2, 3, 4, 5
- fitness = mean tree edit distance to the 5 targets + 1 std (lower is better), from the template

⚠️ **Seed-to-seed spread swamps parameter differences.** Measured on k=4, three seeds: best
fitness varied by up to 0.88 between seeds while `--pmut 0.8` vs `1.0` was indistinguishable.
Never conclude anything from a single run — this is why the assignment demands 5 seeds with
mean and spread.

---

## A. Code — the EA  — ✅ DONE 2026-09-15, reviewed 2026-09-17, 10 tests green

- [x] `ea.py` built on `ariel.ec` (`EA`, `EAOperation`, `Individual`, `Population`), starting
      from `ariel/examples/c_genotypes/1_body_evolution_tree.py`, fitness swapped for the
      template's `fitness_function`
- [x] CLI flags: `--parents k`, `--seed s`, `--pop`, `--gens`, `--out`
      (plus `--max-modules`, `--tournament`, `--pxo`, `--pmut`, `--immigrants`)
- [x] seed `random`, `numpy` (and `torch`, harmless) from `--seed`
- [x] tournament parent selection (k_tournament = 3), tag parents, plus 1 random-immigrant
      parent slot per generation (`--immigrants`)
- [x] reproduction: with p = 0.8 pick k parents, call multi-parent recombination; else clone one
      parent. Then one mutation with p = 0.8 (forced on unrecombined clones). Enforce ≤ 20
      modules (shrink/prune until under the cap)
- [x] μ+λ survivor selection
- [x] per-generation log to CSV as well as the ariel SQLite DB (CSV is what the plot script
      reads; DB is the backup). Columns: `gen, evals, best, mean, worst, diversity, fallbacks,
      cap_fallbacks, forced_mutations`
- [x] save best genome of the run as JSON
- [x] smoke test: `--pop 10 --gens 5` runs end to end in under a minute (takes ~1 s)

## B. Code — the multi-parent operator  — ✅ IMPLEMENTED in `recombine.py`

The reference implementation written alongside block A already satisfies every item below,
with 8 unit tests. **Block B's remaining job is to review it and extend it if you want a
different scheme** — the signature `recombine(parents, rng) -> TreeGenome` must stay.

- [x] `recombine(parents: list[TreeGenome], rng) -> TreeGenome`
- [x] child = copy of parent 0's core node
- [x] for each core face: pick a donor uniformly from the k parents, copy the whole subtree
      hanging on that face (if any) with fresh node ids, attach on the same face.
      `CORE_FACES` is derived from ariel's `ALLOWED_FACES[ModuleType.CORE]`, not hardcoded,
      so it cannot drift if the course ever enables TOP/BOTTOM (there is a `[ ]` in ariel's
      config saying they intend to)
- [x] result must pass `validate_genome_dict`; if not, fall back to a copy of parent 0 and count
      the fallback (reported as the `fallbacks` column)
- [x] with k = 2 this is the control — ariel's `crossover_subtree` is never imported
- [x] unit test: 3 hand-built parents → child's faces come from the expected donors, ids unique,
      module count ≤ sum of donors
- [x] diagnostic: mean pairwise tree edit distance of the population, logged per generation
      as the `diversity` column (lives in `ea.py`, not here)

Measured behaviour with k = 3 over 500 seeds: 45 % of children draw from all three parents,
51 % from two, 4 % from one, 0 fallbacks. That is what uniform per-face donor choice should give.

Ideas if block B wants to go further (none are required): mix at nodes deeper than the core,
or a majority-vote / diagonal scheme across all k parents instead of an independent per-face pick.

## C. Code — baseline + plots  — ✅ DONE 2026-09-18, all 20 runs executed

Landed in commit `1b00508` under different filenames than planned: a `baseline/` package
instead of `baseline.py`, an `evaluation/` package instead of `plot.py` + `stats.py`, and
`run_all.py` instead of `run_all.sh`.

- [x] `baseline/` — random search, `random_tree(19)` so the module count matches the EA's cap,
      best-so-far logged on the same `evals` grid as the EA
- [x] reads `results/*/seed*/log.csv` by column name, 101 rows per run (gen 0..100)
- [x] `evaluation/plot.py` — one line per variant (k=2, 4, 8, random), mean over seeds with
      shaded ±1 std, x = evaluations, y = best fitness → `results/plots/fitness.png`
- [x] `evaluation/statistics.py` — final best fitness per run, mean ± std per variant
      → `results/summary.csv`
- [x] `run_all.py` — 3 variants × 5 seeds + 5 baseline seeds, then the evaluation, in one
      command. Skips runs whose `log.csv` exists; `--force` recomputes
- [ ] **Still missing: the Mann–Whitney U test** between k=2 and each other k. `statistics.py`
      reports mean ± std only. Given the result below it would not change the conclusion, but
      the report should either include it or say plainly that n=5 does not support it
- [ ] **Still missing: the diversity plot.** The column is logged, nothing plots it
- [ ] **Worth adding: an evals-to-threshold plot.** That is where the actual k effect lives
      (see the results below) and the fitness plot does not show it clearly

Three things `ea.py` already decided for you:
- **Align the baseline on the `evals` column, not on `gen`.** The EA logs gen 0 at 50 evals, so
  EA row `gen == i` sits at 50·(i+1) evaluations. Matching on `evals` makes drift impossible.
- **The output directory is built from `--parents` and `--seed` only.** `--pop`, `--gens` and
  `--pxo` do not appear in the path, so two runs that differ only in those would overwrite each
  other. `run_all.sh` only varies k and seed, so it is safe — but pass `--out` explicitly if you
  ever sweep anything else.
- **A full run takes ~11 s**, so all 20 runs are about 4 minutes, not an overnight job. Launching
  them is not the bottleneck; deciding the settings beforehand is.

## D. Report

- [ ] GECCO19 template set up (LaTeX, Overleaf shared with the team), cover page: course, assignment
      number + name, team number/name, member names + student IDs, date
- [ ] Introduction: the research question and hypothesis above, verbatim, plus 3–4 sentences of
      motivation citing Eiben's multi-parent recombination work (Eiben & Smith ch. 4; Eiben,
      Raué & Ruttkay 1994 "Genetic algorithms with multi-parent recombination")
- [ ] Methods: encoding, operator (a small figure of the face-aligned recombination), all fixed
      settings above, fitness definition, evaluation budget, seeds, baseline. Reproducible from
      the text alone
- [ ] Results: convergence figure, final-fitness table, diversity figure, statistical test,
      discussion against the hypothesis (which half held, which did not)
- [ ] Conclusions: 1 paragraph, plus limitations (5 seeds, one target set, one encoding)
- [ ] Bibliography
- [ ] Contribution list per member at the end (required — free-riding is policed)
- [ ] GenAI disclosure line if anything was used
- [ ] ≤ 3 pages excluding cover + bibliography

## E. Hand-in  (owner: ______)

- [ ] folder `<groupnumber>/` containing `<groupnumber>.pdf`, `ea.py`, `recombine.py`,
      `baseline.py`, `plot.py`, `stats.py`, `run_all.sh`, `results/`, `README.md` with run commands
- [ ] zip it as `<groupnumber>.zip`
- [ ] one person submits on Canvas before Mon 09:00; someone else confirms it shows as submitted

---

## THE RESULTS (2026-09-18, all 20 runs, `results/summary.csv`)

Full experiment: 3 variants × 5 seeds + 5 baseline seeds, 5 050 evaluations each, 4m17s total.

### Final best fitness — the EA crushes the baseline, k makes no difference

| variant | n | mean final best | std |
|---|---|---|---|
| k=2 | 5 | **12.378** | 0.421 |
| k=4 | 5 | 12.606 | 0.505 |
| k=8 | 5 | 12.518 | 0.646 |
| random | 5 | 16.439 | 0.163 |

- **The baseline comparison is unambiguous.** Every EA seed beats every random seed by a wide
  margin (~3.9 distance units). Random flatlines at ~16.5 by 1 000 evaluations; the EA is already
  below 13 there. This is the easy half of the report.
- **The k comparison is a null result on final fitness.** The three means span 0.23 while the
  within-variant std is 0.42–0.65 — seed spread is 2–3× the effect. Per-seed finals overlap
  completely (k=4's best seed, 11.77, beats every k=2 seed). Do not claim a winner.

### Convergence speed — this is where k actually shows up

Evaluations needed to first reach best ≤ 13.0:

| variant | mean | std | per-seed |
|---|---|---|---|
| k=2 | 1 850 | 1 305 | 1250, 1100, 3950, 700, 2250 |
| k=4 | 1 313 | 779 | 2400, 550, 1200, 1100 |
| k=8 | **688** | **48** | 750, 650, 650, 700 |

k=8 converges **2.7× faster than k=2 with 27× less variance** — 650–750 evaluations on every
single seed, while one k=2 seed took 3 950. **This is the headline result, not the fitness
table.** More parents does not raise the ceiling; it makes every recombination reliably
productive instead of occasionally a clone, so the early search is faster and far more
consistent.

### Why the ceiling does not move — the 4-face cap

`CORE_FACES` has 4 entries, so **at most 4 parents can ever contribute to one child**, whatever
k is. Measured over 3 000 recombinations:

| k | distinct parents picked | actually donated | P(child is a clone of one parent) = 1/k³ | P(a drawn parent contributes nothing) |
|---|---|---|---|---|
| 2 | 1.88 | 1.72 | 12.5 % | 6.2 % |
| 4 | 2.74 | 2.30 | 1.6 % | 31.6 % |
| 8 | 3.30 | 2.66 | 0.2 % | 58.6 % |

At k=8, 58.6 % of the parents we carefully draw as distinct contribute nothing at all. k=4 and
k=8 are very nearly the same operator, which is exactly why their final fitness is
indistinguishable.

### ⚠️ The hypothesis needs rewording before Methods is written

> "too many parents disrupts useful genetic structure and therefore reduces convergence"

**This operator is structurally non-disruptive.** Every face receives one whole, coherent
subtree; it never splices two parents' branches together or breaks a limb mid-way. The
disruption mechanism the hypothesis predicts largely *cannot occur here*. The data agrees: no
degradation at k=8, and it converges fastest. Write the hypothesis knowing this, and report the
refutation with its mechanism — that scores better than a vague confirmation.

### Other measured facts for Methods

- **Diversity collapses to ~0.1–0.2 for every variant** well before the budget ends, so roughly
  3 500 of the 5 050 evaluations are spent after the search has stopped moving. This is why
  final fitness cannot separate the variants, and it is worth one honest sentence in Discussion.
  (The 2026-09-17 worry that this made the k comparison untestable was half right: it killed the
  *final fitness* comparison, and the effect surfaced in convergence speed instead.)
- **`fallbacks` = 0 across all 15 EA runs.** The operator's safety net never fired once. Same for
  `cap_fallbacks`. Reportable.
- `forced_mutations` ≈ 190–210 per run, i.e. ~4 % of children were clones that also lost the
  mutation coin flip and were mutated anyway.
- Runs are **deterministic across processes** — same flags give byte-identical `log.csv` and
  `best.json` under four different `PYTHONHASHSEED` values.
- ⚠️ **`results/` is gitignored**, so these numbers exist only on Emre's machine. Block E's
  hand-in list requires `results/` in the zip — either un-ignore it before submitting or have
  whoever zips it re-run `run_all.py` (4 minutes).

## Timeline

| Day | What |
|---|---|
| Tue 16 | ✅ A written. B's operator written too (reference impl). ⬜ Roles for C/D/E still unassigned, Overleaf not started |
| Wed 17 | ✅ A + B reviewed, 10 tests green, full run verified. ⬜ C not started — **this is the critical path** |
| Thu 18 | ✅ C landed and all 20 runs executed; results + plot below. ⬜ D still unstarted — **now the critical path** |
| Fri 19 | D writes Intro + Methods + Results against the numbers below. C adds the evals-to-threshold plot and the U test |
| Sat 20 | Full draft, everyone reads it. Fix, trim to 3 pages, contributions |
| Sun 21 | Final PDF, zip, submit. Do not leave it for Monday morning |

## Rules everyone must know

- Never edit anything under `ariel/src/ariel`. It is fraud and fails the assignment.
- No CPPN.
- Change one thing at a time. k is the only variable.
- Keep every run's output; never overwrite a results file.
