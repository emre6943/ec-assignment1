# EC Assignment 1 — Team task list

**Deadline: Mon 2026-09-21, 09:00.** Late = −0,5 pt/day.

**Research question.** How does the number of parents involved in recombination affect the
convergence and final fitness of evolved robot morphologies?

**Hypothesis.** Increasing the number of parents increases genetic diversity and exploration, but
too many parents disrupts useful genetic structure and therefore reduces convergence performance.

**Design.** Tree genotype, max 20 modules. One EA, one operator (face-aligned multi-parent
recombination), one variable: number of parents k ∈ {2, 4, 8}. k=2 is the control. Everything
else fixed. Plus a random-search baseline at the same evaluation budget.

Fixed settings (do not change between variants):
- population 50, 100 generations → **5 050** evaluations per run (gen 0 = the initial 50 + 100 × 50)
- tournament parent selection, k_tournament = 3
- crossover probability 0.7, then exactly one mutation (point / subtree / shrink / hoist)
- survivor selection μ+λ (keep best 50 of parents + children)
- 5 seeds per variant: 1, 2, 3, 4, 5
- fitness = mean tree edit distance to the 5 targets + 1 std (lower is better), from the template

---

## A. Code — the EA  — ✅ DONE 2026-09-15, reviewed 2026-09-17, 10 tests green

- [x] `ea.py` built on `ariel.ec` (`EA`, `EAOperation`, `Individual`, `Population`), starting
      from `ariel/examples/c_genotypes/1_body_evolution_tree.py`, fitness swapped for the
      template's `fitness_function`
- [x] CLI flags: `--parents k`, `--seed s`, `--pop`, `--gens`, `--out`
      (plus `--max-modules`, `--tournament`, `--pxo`)
- [x] seed `random`, `numpy` (and `torch`, harmless) from `--seed`
- [x] tournament parent selection (k_tournament = 3), tag parents
- [x] reproduction: with p = 0.7 pick k parents, call multi-parent recombination; else clone one
      parent. Then one mutation. Enforce ≤ 20 modules (shrink/prune until under the cap)
- [x] μ+λ survivor selection
- [x] per-generation log of best / mean / worst fitness to CSV as well as the ariel SQLite DB
      (CSV is what the plot script reads; DB is the backup)
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

## C. Code — baseline + plots

- [ ] `baseline.py --seed s`: 5 050 random trees (`random_tree(19)` — module count includes the
      core, so 19 gives ≤ 20 nodes like the EA), track best-so-far every 50 evaluations; point i
      ↔ 50·i evals ↔ EA generation i−1 (the EA logs gen 0 as the initial population)
- [ ] read `results/k*/seed*/log.csv` by column name: `gen,evals,best,mean,worst,diversity,
      fallbacks,cap_fallbacks`, 101 rows per run (gen 0..100)
- [ ] `plot.py`: reads every CSV in `results/`, one line per variant (k=2, 4, 8, random),
      mean over seeds with shaded ±1 std, x = generation, y = fitness. Also a diversity plot if B
      logs it
- [ ] `stats.py`: final best fitness per run, table of mean ± std per variant, Mann–Whitney U
      between k=2 and each other k (5 samples each, so report the p-value but do not oversell it)
- [ ] `run_all.sh`: launches 3 variants × 5 seeds + 5 baseline seeds, outputs to
      `results/<variant>/seed<s>.csv`

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

## What we already know from `ea.py` (2026-09-17)

Measured on a full default run (k=2, seed 1, 5 050 evaluations, 11 s):

| | gen 0 | gen 10 | gen 30 | gen 60 | gen 100 |
|---|---|---|---|---|---|
| best | 17.70 | 14.03 | 13.28 | 12.87 | **12.63** |
| diversity | 18.45 | 6.74 | 1.86 | 1.52 | 1.26 |

- **The fitness floor is ~15.75** by the README's own reading (the targets' mean pairwise
  distance), and we are already at 12.63. Good — the EA works.
- ⚠️ **Diversity collapses by generation 30** and barely moves after. If all three k variants
  flatten that early, the k comparison has ~30 useful generations and 70 of noise, and the
  "more parents → more diversity" half of the hypothesis may be untestable as configured.
  **Worth a 3-seed spot check before committing to all 20 runs**, because the fix (weaker
  selection pressure, or μ,λ instead of μ+λ) is a settings change we want to make *before* the
  real runs, not after. If we change it, it changes for all variants equally — k stays the only
  variable.
- `fallbacks` and `cap_fallbacks` are **0** across 5 050 evaluations. The safety nets exist but
  never fire, so the operator never degenerates. That is a reportable number for Methods.
- Runs are **deterministic across processes** — the same flags give byte-identical `log.csv` and
  `best.json` under four different `PYTHONHASHSEED` values.

## Timeline

| Day | What |
|---|---|
| Tue 16 | ✅ A written. B's operator written too (reference impl). ⬜ Roles for C/D/E still unassigned, Overleaf not started |
| Wed 17 | ✅ A + B reviewed, 10 tests green, full run verified. ⬜ C not started — **this is the critical path** |
| Thu 18 | `run_all.sh` launched (all 20 runs). D writes Intro + Methods |
| Fri 19 | Results in. C produces figures + stats. D writes Results |
| Sat 20 | Full draft, everyone reads it. Fix, trim to 3 pages, contributions |
| Sun 21 | Final PDF, zip, submit. Do not leave it for Monday morning |

## Rules everyone must know

- Never edit anything under `ariel/src/ariel`. It is fraud and fails the assignment.
- No CPPN.
- Change one thing at a time. k is the only variable.
- Keep every run's output; never overwrite a results file.
