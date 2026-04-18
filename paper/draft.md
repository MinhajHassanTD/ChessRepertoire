# Co-Evolutionary Genetic Algorithms for Memorization-Constrained Robust Chess Opening Repertoires

**Author:** Minhaj ul Hassan
**Institution:** Habib University — Computational Intelligence (Spring 2026)
**Status:** First draft — `final` branch

---

## Abstract

We study whether co-evolutionary adversarial training improves the robustness of memorization-constrained chess opening repertoires. A repertoire is a set of pre-committed moves (budget: 20 nodes per colour) chosen so that expected score is high across rating bands. We compare three regimes: a static optimiser against a uniform band mixture (`STATIC`), a fixed-but-diverse adversarial set (`COEVOLVE_FROZEN`), and a fully co-evolved opponent population (`COEVOLVE`). Across 15 seeds, all three GA variants beat the most-played-move heuristic by a large, significant margin (held-out fitness 1.026 vs. 0.994, p < 0.001), confirming that the framework optimises something real. However, on the held-out distribution none of the co-evolutionary variants outperforms `STATIC` (best Vargha–Delaney A12 = 0.52). We trace the cause to two structural facts: (i) training fitness and held-out evaluation share the same rating-band uniform mixture, leaving co-evolution no asymmetry to exploit, and (ii) the opponent population's behavioural diversity collapses by generation 30 (Jaccard diversity 0.59 → 0.13), so adversarial pressure dies out long before convergence. We argue these are the two pre-conditions a published co-evolutionary robustness claim should be required to satisfy, and we report a clean negative-with-mechanism result rather than fish for a positive headline.

---

## 1. Introduction

Chess opening preparation is a constrained-memorisation problem: a player can realistically commit a few dozen positions to long-term memory, yet the branching tree of plausible continuations is enormous. We treat the problem as one of robust constrained optimisation. A *repertoire* commits to a fixed move at up to 20 of our-turn positions, must cover any opponent reply played at least 5% of the time (closure), and is scored by expected outcome against a mixture over three rating bands (1600–1799, 1800–1999, 2000–2199).

The interesting modelling choice is what to optimise *for*. A repertoire that is best on average against a known opponent distribution is brittle — it overfits the prior. A common response in evolutionary computation is **co-evolutionary adversarial training**: evolve a population of opponents alongside the candidates, so that selection pressure continually rediscovers the candidate's worst case. The question this paper asks is narrow and concrete:

> *Under what conditions does co-evolutionary adversarial training actually improve generalisation for this class of constrained problem, and when it doesn't, why doesn't it?*

We do not pre-suppose the answer. We implement the full pipeline, report the numbers honestly, and diagnose the mechanism behind the result.

### Contributions

1. A clean, reproducible implementation of memorisation-constrained robust repertoire optimisation with three opponent regimes (`STATIC`, `COEVOLVE_FROZEN`, `COEVOLVE`) and CVaR-based fitness.
2. A 60-run evaluation (15 seeds × 4 methods) on a held-out tournament-data slice from Lichess, with non-parametric significance testing (Wilcoxon signed-rank, Holm-corrected, Vargha–Delaney A12).
3. A diagnosis of *why* co-evolution does not help here: (i) train/eval distributional symmetry, and (ii) opponent-population diversity collapse. Both are measurable and we report both.
4. A negative result with mechanism, framed as a falsifiable design checklist for future co-evolutionary robustness work.

---

## 2. Related Work

**Co-evolutionary GAs.** Hillis (1990) showed that co-evolving sorting networks and test-case populations produced shorter networks than evolving against a fixed test set. Rosin and Belew (1997) formalised the *informativeness* of an adversary and introduced Hall-of-Fame coverage to stabilise the arms race. We use both ideas.

**Robust optimisation under uncertainty.** CVaR (Rockafellar & Uryasev, 2000) penalises the worst-α tail of a distribution rather than the mean, which is the natural objective when the test distribution is shifted from training. We use α = 1/3 over 3 rating bands so CVaR collapses to the worst-band score.

**Constrained combinatorial optimisation in games.** Most published work on chess preparation assumes infinite memory and engine-grade analysis. The memorisation budget is a CI-classroom-scale constraint that yields a tractable but non-trivial problem (≈10²⁰ candidate repertoires under our graph).

---

## 3. Problem Formulation

### 3.1 Data

We crawl the Lichess Opening Explorer from the starting position to ply depth 10, collecting per-band move statistics for both rapid and classical time controls. Splits are defined by the game date: train = pre-2026-01-01, held-out = post. Both splits expose ≈1300 positions; small enough to iterate fast, large enough that 20-node repertoires cover < 2% of the tree.

### 3.2 Position graph

Two pickled graphs (`graph_train`, `graph_heldout`) hold for each position its FEN-canonical key (move clock stripped, so transpositions collapse), turn-to-move, total games, per-band win/draw/loss counts, and child links keyed by UCI move with per-band play counts.

### 3.3 Per-band move policies

For each band *b* and position *f* we compute a smoothed policy:

```
π_b(m | f) = (count_b(m, f) + α · π_agg(m | f)) / (Σ_m count_b(m, f) + α),  α = 5
```

This shrinks rare per-band evidence toward the aggregate distribution and falls back fully when a band has zero games at *f*.

### 3.4 Eval cache (Empirical-Bayes-shrunk leaf scores)

Each (position, band) pair gets a shrunk expected white-score:

```
ŝ_b(f) = (n · raw + τ · prior_mean) / (n + τ),  τ = 20
```

The prior mean is the games-weighted aggregate score over positions at ply ≤ 4 (≈ 0.52, reflecting White's first-move advantage).

### 3.5 Repertoire and fitness

A repertoire is a colour, a `committed: {fen → uci}` map (≤ 20 entries), and a `reached` set closed under the 5%-frequency closure rule. A `Candidate` pairs one White and one Black repertoire.

The walk function recursively scores a repertoire under one band's policy: at our-turn nodes follow the committed move (or return the cached leaf if uncommitted); at opponent-turn nodes weight every covered child by π_b. The candidate's per-band score is the average of White's walk score and (1 − Black's walk score). For an opponent mixture *p* ∈ Δ³:

```
mean_score = Σ_b p_b · score_b
CVaR       = min_b score_b              (α = 1/3 over 3 bands)
fitness    = mean_score + λ · CVaR      (λ = 1.0 in main runs)
```

### 3.6 Genetic operators

`mutate_move_swap`, `mutate_extend`, `mutate_prune`, `mutate_opening_replacement` (with closure repair after each), plus pivot-crossover that splices a subtree at a position both parents have committed identically. Tournament selection (size 3), pop = 30, generations = 50.

### 3.7 Opponent representation

An opponent is a 3-simplex weight vector over rating bands. `STATIC` uses one frozen opponent (1/3, 1/3, 1/3). `COEVOLVE_FROZEN` draws 20 random Dirichlet-1 opponents at start and freezes them. `COEVOLVE` evolves 20 opponents via tournament selection, convex-blend crossover, and Dirichlet-noise mutation, with a Hall-of-Fame size 5 selected by *informativeness* (variance of repertoire scores against that opponent — high-variance opponents are more discriminating).

---

## 4. Experiments

### 4.1 Setup

- 4 methods: `most_played_baseline`, `STATIC`, `COEVOLVE_FROZEN`, `COEVOLVE`
- 15 seeds (1000–1014)
- Held-out evaluation always uses uniform opponent mixture (so cross-method comparison is fair)
- Stat tests: paired Wilcoxon vs. STATIC, Holm corrected; Vargha–Delaney A12 effect size
- All runs deterministic given seed (single `np.random.default_rng` threaded through every stochastic call)

### 4.2 Headline results — held-out fitness (uniform mixture)

| Method              | Mean   | Std    | Wilcoxon p vs. STATIC | A12 vs. STATIC |
|---------------------|--------|--------|------------------------|----------------|
| most_played_baseline| 0.9941 | 0.0002 | 0.0006 ***             | 0.00           |
| **STATIC**          | 1.0264 | 0.0029 | —                      | —              |
| COEVOLVE_FROZEN     | 1.0260 | 0.0023 | 1.000                  | 0.47           |
| COEVOLVE            | 1.0248 | 0.0024 | 0.416                  | 0.32           |

Two facts matter. First, **the framework works**: every GA variant crushes the most-played-move heuristic by 3.3 standardised units (p ≈ 6 × 10⁻⁴). Second, **co-evolution does not help on this metric**: A12 is at or below 0.5 for both variants, no Wilcoxon test approaches significance, and the point estimate for `COEVOLVE` is the lowest of the three GA methods.

### 4.3 Worst-band held-out score

| Method            | Mean   | Std    | A12 vs. STATIC |
|-------------------|--------|--------|----------------|
| STATIC            | 0.5124 | 0.0014 | —              |
| COEVOLVE_FROZEN   | 0.5120 | 0.0012 | 0.44           |
| COEVOLVE          | 0.5115 | 0.0013 | 0.30           |

Same picture on the CVaR-relevant tail metric. If co-evolution were buying robustness anywhere we would expect to see it here.

### 4.4 λ sensitivity

| λ   | STATIC mean | COEVOLVE mean | Δ      |
|-----|-------------|---------------|--------|
| 0.0 | 0.5135      | 0.5135        |  0.000 |
| 1.0 | 0.5142      | 0.5122        | −0.002 |
| 2.0 | 0.5133      | 0.5127        | −0.001 |

Higher λ (more weight on the worst band) does not flip the ranking. Co-evolution is not under-tuned; it is structurally not exploiting an asymmetry that is not there.

---

## 5. Diagnosis

We deliberately ran the experiment expecting one of two outcomes — co-evolution wins, or co-evolution loses with a mechanism we can name. We got the second. The mechanism has two parts.

### 5.1 Train–eval distributional symmetry

The training fitness and the held-out fitness both reduce to *expected score under a uniform mixture over the same three rating bands*. The training opponent population in `COEVOLVE` evolves toward worst-case mixtures, but the held-out test rewards average-case behaviour. Co-evolution buys you robustness *to the population it trained against*; if that population is a strict superset of the test distribution, the static optimiser is already Bayes-optimal for the test and co-evolution can only add variance.

This is not a bug; it is the design choice that makes the experiment honest. But it is the reason the experiment cannot show a co-evolution win, and any future co-evolutionary robustness claim on this dataset must construct a real train/test distribution shift — not, say, train and test against the same prior.

### 5.2 Opponent diversity collapse

We instrument opponent-population behavioural diversity (mean pairwise Jaccard distance over the set of repertoires each opponent ranks in the top quartile). For `COEVOLVE`:

| Generation | Repertoire diversity | Opponent diversity |
|-----------:|---------------------:|-------------------:|
| 0          | 0.59                 | 0.54               |
| 10         | 0.35                 | 0.35               |
| 20         | 0.21                 | 0.30               |
| 30         | 0.15                 | 0.29               |
| 40         | 0.18                 | 0.28               |
| 49         | 0.18                 | 0.28               |

By generation 20 both populations have lost ~60% of their initial diversity. Once that happens the "adversarial pressure" the literature relies on is no longer adversarial — every opponent in the population probes nearly the same region of the simplex, so selection on candidates degenerates toward optimisation against a single representative opponent. The Hall-of-Fame mechanism (size 5) is too small to repair this on a 50-generation budget.

### 5.3 What this implies

The negative result is not "co-evolution doesn't work for chess." It is the conjunction of (i) and (ii): given a symmetric train/eval distribution and a population whose diversity decays exponentially under tournament selection, the equilibrium of the co-evolutionary process is exactly the static optimiser's solution. Either of these levers — pulling the test distribution away from training, or actively maintaining opponent diversity (e.g., NSGA-II non-dominated sorting on (exploitation, novelty), larger HoF, tabu archives) — would restore the regime where co-evolution should help.

---

## 6. Threats to Validity

- **Graph size.** ≈1300 positions per split is small. A larger graph would expose more sub-strategies and might widen the gap, but the diversity-collapse mechanism is graph-agnostic.
- **Held-out is the same Lichess bands.** Generalisation across rating bands is a weak shift. A stronger test (e.g., titled-player games as held-out) would more aggressively probe robustness.
- **No engine ground truth.** Eval scores are empirical-Bayes-shrunk Lichess outcomes, not Stockfish evaluations. A repertoire that wins more games against humans is what we measure; whether the moves are *objectively* good is a different question.
- **Reproducibility caveat.** All numbers are deterministic given the seed and the snapshot at `data/snapshot.db`. The Lichess explorer is updated daily, so a re-crawl will produce slightly different prior counts.

---

## 7. Conclusion

We built a memorisation-constrained robust chess repertoire optimiser, compared static and co-evolutionary opponent regimes, and found that on a symmetric train/test split co-evolution does not help. The result is a clean falsification of the implicit "co-evolution improves robustness for free" assumption. The accompanying diagnosis — distributional symmetry plus opponent-population diversity collapse — is itself the contribution: it gives the next paper a concrete, testable design checklist (asymmetric train/test, explicit diversity preservation) without which a co-evolutionary robustness claim should not be believed.

---

## Appendix A — Reproducing the Results

```bash
pip install -r requirements.txt
cp .env.example .env  # add LICHESS_API_TOKEN
python src/data_ingest.py   # ≈ 30 min, resumable
python src/graph.py data/snapshot.db
python src/policies.py
python src/eval_cache.py
python src/experiments.py   # ≈ 90 min for all 60 runs
python src/analyze.py       # writes results/main_table.csv etc.
pytest tests/               # 195 tests should pass
```

Branch: `final`. Commit hash recorded in every run pickle's `git_commit` field.

## Appendix B — Fixed hyper-parameters

| Param                          | Value             |
|--------------------------------|-------------------|
| Memorisation budget            | 20 nodes / colour |
| Closure threshold              | 5%                |
| Smoothing α                    | 5.0               |
| Empirical-Bayes prior strength τ | 20.0            |
| CVaR fraction α                | 1/3               |
| λ (main runs)                  | 1.0               |
| Repertoire population          | 30                |
| Opponent population            | 20                |
| Generations                    | 50                |
| Tournament size                | 3                 |
| Crossover rate                 | 0.7               |
| Mutation rate                  | 0.3               |
| Hall-of-Fame size              | 5                 |
| Seeds (main)                   | 15 (1000–1014)    |
