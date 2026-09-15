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
- population 50, 100 generations → 5 000 evaluations per run
- tournament parent selection, k_tournament = 3
- crossover probability 0.7, then exactly one mutation (point / subtree / shrink / hoist)
- survivor selection μ+λ (keep best 50 of parents + children)
- 5 seeds per variant: 1, 2, 3, 4, 5
- fitness = mean tree edit distance to the 5 targets + 1 std (lower is better), from the template

---

## A. Code — the EA  (owner: ______)

- [ ] `ea.py` built on `ariel.ec` (`EA`, `EAOperation`, `Individual`, `Population`), starting
      from `ariel/examples/c_genotypes/1_body_evolution_tree.py`, fitness swapped for the
      template's `fitness_function`
- [ ] CLI flags: `--parents k`, `--seed s`, `--pop`, `--gens`, `--out`
- [ ] seed `random`, `numpy` (and `torch`, harmless) from `--seed`
- [ ] tournament parent selection (k_tournament = 3), tag parents
- [ ] reproduction: with p = 0.7 pick k parents, call multi-parent recombination; else clone one
      parent. Then one mutation. Enforce ≤ 20 modules (shrink/prune until under the cap)
- [ ] μ+λ survivor selection
- [ ] per-generation log of best / mean / worst fitness to CSV as well as the ariel SQLite DB
      (CSV is what the plot script reads; DB is the backup)
- [ ] save best genome of the run as JSON
- [ ] smoke test: `--pop 10 --gens 5` runs end to end in under a minute

## B. Code — the multi-parent operator  (owner: ______)

- [ ] `recombine(parents: list[TreeGenome], rng) -> TreeGenome`
- [ ] child = copy of parent 0's core node
- [ ] for each core face (FRONT, BACK, LEFT, RIGHT): pick a donor uniformly from the k parents,
      copy the whole subtree hanging on that face (if any) with fresh node ids, attach on the same
      face
- [ ] result must pass `validate_genome_dict`; if not, fall back to a copy of parent 0 and count
      the fallback (report it)
- [ ] with k = 2 this is the control — do NOT use ariel's `crossover_subtree`, it would change
      two things at once
- [ ] unit test: 3 hand-built parents → child's faces come from the expected donors, ids unique,
      module count ≤ sum of donors
- [ ] optional diagnostic: log per generation the mean pairwise tree edit distance of the
      population (diversity) — this is what tests the "more diversity" half of the hypothesis

## C. Code — baseline + plots  (owner: ______)

- [ ] `baseline.py --seed s`: 5 000 random trees (`random_tree(20)`), track best-so-far every
      50 evaluations so it lines up with one generation of the EA
- [ ] `plot.py`: reads every CSV in `results/`, one line per variant (k=2, 4, 8, random),
      mean over seeds with shaded ±1 std, x = generation, y = fitness. Also a diversity plot if B
      logs it
- [ ] `stats.py`: final best fitness per run, table of mean ± std per variant, Mann–Whitney U
      between k=2 and each other k (5 samples each, so report the p-value but do not oversell it)
- [ ] `run_all.sh`: launches 3 variants × 5 seeds + 5 baseline seeds, outputs to
      `results/<variant>/seed<s>.csv`

## D. Report  (owner: ______)

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

## Timeline

| Day | What |
|---|---|
| Tue 16 | Roles assigned. A and B start. D sets up Overleaf + skeleton |
| Wed 17 | A + B merged, smoke test passes. C has baseline + plot on dummy data |
| Thu 18 | `run_all.sh` launched (all 20 runs). D writes Intro + Methods |
| Fri 19 | Results in. C produces figures + stats. D writes Results |
| Sat 20 | Full draft, everyone reads it. Fix, trim to 3 pages, contributions |
| Sun 21 | Final PDF, zip, submit. Do not leave it for Monday morning |

## Rules everyone must know

- Never edit anything under `ariel/src/ariel`. It is fraud and fails the assignment.
- No CPPN.
- Change one thing at a time. k is the only variable.
- Keep every run's output; never overwrite a results file.
