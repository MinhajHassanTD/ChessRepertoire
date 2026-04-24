# Co-Evolutionary Genetic Algorithms for Memorization-Constrained Robust Chess Opening Repertoires

**Author:** Minhaj ul Hassan
**Institution:** Habib University — Computational Intelligence (Spring 2026)
**Status:** Second draft — `final` branch

---

## Abstract

We study whether co-evolutionary adversarial training improves the robustness of memorization-constrained chess opening repertoires. A repertoire commits to at most 30 moves per color, and must cover any opponent reply played in ≥15% of games at a reached position (the *closure rule*). We compare four regimes: a greedy most-played-move heuristic, a static GA against a uniform band mixture (`STATIC`), a fixed-but-diverse adversarial set (`COEVOLVE_FROZEN`), and a fully co-evolved opponent population (`COEVOLVE`). Across 15 seeds, all GA variants beat the most-played-move baseline by a large, significant margin (held-out fitness 1.026 vs. 0.994, p < 0.001). However, a simple greedy hill-climber matches the full population-based GA (p = 0.72), and none of the co-evolutionary variants outperforms `STATIC` on the held-out distribution (best A12 = 0.52). A closure ablation confirms the closure rule is load-bearing: removing it drops held-out score by ≈0.008 (p < 0.001). We trace the co-evolution null result to two structural facts: (i) train and eval share the same uniform rating-band mixture, leaving co-evolution no distributional asymmetry to exploit; and (ii) opponent-population behavioural diversity collapses by generation 20 (mean pairwise distance 0.54 → 0.28), so adversarial pressure dies before convergence.

---

## 1. Introduction

Chess opening preparation is a constrained-memorization problem: a player can realistically commit a few dozen positions to long-term memory, yet the branching tree of plausible continuations is enormous. We treat the problem as robust constrained optimization. A *repertoire* commits to a fixed move at up to 30 of our-turn positions, must cover any opponent reply played in ≥15% of games (the *closure rule*), and is scored by expected outcome against a mixture over three Lichess rating bands (1000–1399, 1400–1799, 1800–2199).

The interesting modeling choice is what to optimize *for*. A repertoire that is best on average against a known opponent distribution is brittle — it overfits the prior. A common response in evolutionary computation is **co-evolutionary adversarial training**: evolve a population of opponents alongside the candidates, so that selection pressure continually rediscovers the candidate's worst case. The question this paper asks is narrow and concrete:

> *Under what conditions does co-evolutionary adversarial training actually improve generalization for this class of constrained problem, and when it does not, why not?*

We do not pre-suppose the answer. We implement the full pipeline, run 105 total experiments across three experiment types, report the numbers honestly, and diagnose the mechanism behind the result.

### Contributions

1. A clean, reproducible implementation of memorization-constrained robust repertoire optimization with three opponent regimes (`STATIC`, `COEVOLVE_FROZEN`, `COEVOLVE`) and CVaR-based fitness across 30-move budgets.
2. A 60-run evaluation (15 seeds × 4 methods) on a held-out Lichess tournament-data slice, with non-parametric significance testing (Wilcoxon signed-rank, Holm-corrected, Vargha–Delaney A12).
3. A closure-rule ablation (30 paired runs) confirming the novel structural mechanism is load-bearing.
4. A diagnosis of *why* co-evolution does not help: (i) train/eval distributional symmetry, and (ii) opponent-population diversity collapse — both measurable, both reported.
5. A negative result with mechanism, framed as a falsifiable design checklist for future co-evolutionary robustness work.

---

## 2. Related Work

**Co-evolutionary GAs.** Hillis (1990) showed that co-evolving sorting networks and test-case populations produced shorter networks than evolving against a fixed test set. Rosin and Belew (1997) formalized the *informativeness* of an adversary and introduced Hall-of-Fame coverage to stabilize the arms race. We use both ideas.

**Robust optimization under uncertainty.** CVaR (Rockafellar & Uryasev, 2000) penalizes the worst-α tail of a distribution rather than the mean, which is the natural objective when the test distribution is shifted from training. We use α = 1/3 over 3 rating bands so CVaR collapses to the worst-band score.

**Constrained combinatorial optimization in games.** Most published work on chess preparation assumes infinite memory and engine-grade analysis. The memorization budget is a constraint that yields a tractable but non-trivial problem: a 30-move budget covers roughly 2% of the ~1,300-position graph, so the search space is vast.

---

## 3. Problem Formulation

### 3.1 Data

We crawl the Lichess Opening Explorer from the starting position to ply depth 10, collecting per-band move statistics for rapid and classical time controls. The train/held-out split is by game date: train = pre-June 2025, held-out = June 2025 onward. Both splits expose approximately 1,300 positions.

### 3.2 Position Graph

Two pickled graphs (`graph_train`, `graph_heldout`) store for each position its FEN-canonical key (move clock stripped, so transpositions collapse), turn-to-move, total games, per-band win/draw/loss counts, and child links keyed by UCI move with per-band play counts.

### 3.3 Per-Band Move Policies

For each band *b* and position *f* we compute a smoothed policy:

```
π_b(m | f) = (count_b(m, f) + α · π_agg(m | f)) / (Σ_m count_b(m, f) + α),  α = 5
```

This shrinks sparse per-band evidence toward the aggregate distribution and falls back fully when a band has zero games at *f*.

### 3.4 Eval Cache (Empirical-Bayes-Shrunk Leaf Scores)

Each (position, band) pair gets a shrunk expected white-score:

```
ŝ_b(f) = (n · raw + τ · prior_mean) / (n + τ),  τ = 20
```

The prior mean is the games-weighted aggregate score over positions at ply ≤ 8 (≈ 0.52, reflecting White's first-move advantage).

### 3.5 Repertoire and Fitness

A repertoire is a color, a `committed: {fen → uci}` map (≤ 30 entries), and a `reached` set closed under the closure rule: any opponent reply played in ≥15% of games at a reached position is automatically added to `reached`. A `Candidate` pairs one White and one Black repertoire.

The walk function recursively scores a repertoire under one band's policy: at our-turn nodes follow the committed move (or return the cached leaf if uncommitted); at opponent-turn nodes weight every covered child by π_b. The candidate's per-band score is the average of White's walk score and (1 − Black's walk score). For an opponent mixture *p* ∈ Δ³:

```
mean_score = Σ_b p_b · score_b
CVaR       = min_b score_b              (α = 1/3 over 3 bands)
fitness    = mean_score + λ · CVaR      (λ = 1.0 in main runs)
```

### 3.6 Genetic Operators

Four mutation operators (`mutate_move_swap`, `mutate_extend`, `mutate_prune`, `mutate_opening_replacement`) with closure repair after each, plus pivot-crossover that splices a subtree at a position both parents have committed identically. Tournament selection (size 2), population = 100, generations = 60, crossover rate = 0.8, mutation rate = 0.4.

### 3.7 Opponent Representation

An opponent is a 3-simplex weight vector over rating bands. `STATIC` uses one frozen opponent (1/3, 1/3, 1/3). `COEVOLVE_FROZEN` draws 50 random Dirichlet-1 opponents at start and freezes them. `COEVOLVE` evolves 50 opponents via tournament selection (size 2), convex-blend crossover (rate 0.8), and Dirichlet-noise mutation (strength 0.5, rate 0.5), with a Hall-of-Fame (size 5) selected by *informativeness* (variance of repertoire scores against that opponent).

---

## 4. Experiments

### 4.1 Setup

- **Main comparison:** 3 methods × 15 seeds = 45 runs (seeds 1000–1014)
- **Non-GA baselines:** `RANDOM_SEARCH` and `GREEDY_HILLCLIMB` × 15 seeds = 30 runs. Eval budget = pop × generations = 6,000 fitness calls (same as GA).
- **Closure ablation:** `STATIC_NOCLOSURE` and `COEVOLVE_NOCLOSURE` × 15 seeds = 30 runs (paired with main seeds)
- **Held-out evaluation** always uses uniform opponent mixture (1/3, 1/3, 1/3) so cross-method comparison is fair
- **Stat tests:** paired Wilcoxon vs. `STATIC`, Holm-corrected; Vargha–Delaney A12 effect size

### 4.2 Headline Results — Held-Out Fitness (Uniform Mixture)

| Method               | Mean   | Std    | Wilcoxon p vs. STATIC | A12 vs. STATIC |
|----------------------|--------|--------|------------------------|----------------|
| most_played_baseline | 0.9941 | 0.0002 | < 0.001 ***            | 0.00           |
| RANDOM_SEARCH        | 1.0169 | 0.0024 | < 0.001 ***            | 0.00           |
| GREEDY_HILLCLIMB     | 1.0274 | 0.0030 | 0.718                  | 0.61           |
| **STATIC**           | 1.0264 | 0.0029 | —                      | —              |
| COEVOLVE             | 1.0248 | 0.0024 | 0.250                  | 0.32           |

Three facts stand out. First, **the framework works**: all GA-adjacent methods crush the most-played-move heuristic by a large, significant margin. Second, **a simple greedy hill-climber matches the full population-based GA** (GREEDY_HILLCLIMB vs. STATIC: p = 0.72, A12 = 0.61) — the population is not necessary. Third, **co-evolution does not improve on the static baseline** (COEVOLVE A12 = 0.32 vs. STATIC).

### 4.3 Worst-Band Held-Out Score

| Method   | Mean   | Std    | A12 vs. STATIC |
|----------|--------|--------|----------------|
| STATIC   | 0.5124 | 0.0014 | —              |
| COEVOLVE | 0.5115 | 0.0013 | 0.30           |

The same picture holds on the CVaR-relevant tail metric. If co-evolution were buying robustness anywhere we would expect to see it on worst-band performance — we do not.

### 4.4 Closure Ablation

| Method             | Held-Out Mean | Std    | p vs. WITH closure | A12  |
|--------------------|---------------|--------|--------------------|------|
| STATIC             | 1.0264        | 0.0029 | —                  | —    |
| STATIC_NOCLOSURE   | 1.0184        | 0.0041 | < 0.001 ***        | 0.07 |
| COEVOLVE           | 1.0248        | 0.0024 | —                  | —    |
| COEVOLVE_NOCLOSURE | 1.0171        | 0.0038 | < 0.001 ***        | 0.06 |

Removing the closure rule drops performance significantly for both methods. The closure rule — which automatically extends coverage to any opponent reply played in ≥15% of games — is load-bearing, not cosmetic.

---

## 5. Diagnosis

We deliberately ran the experiment expecting one of two outcomes — co-evolution wins, or co-evolution loses with a mechanism we can name. We got the second. The mechanism has three parts.

### 5.1 Train–Eval Distributional Symmetry

The training fitness and the held-out fitness both reduce to *expected score under a uniform mixture over the same three rating bands*. The training opponent population in `COEVOLVE` evolves toward worst-case mixtures, but the held-out test rewards average-case behavior. Co-evolution buys robustness *to the population it trained against*; if that population is a strict superset of the test distribution, the static optimizer is already near-Bayes-optimal for the test and co-evolution only adds variance.

This is not a bug — it is the design choice that makes the experiment honest. But it is the reason the experiment cannot show a co-evolution win. Any future co-evolutionary robustness claim on this dataset must construct a real train/test distribution shift, not train and test against the same prior.

### 5.2 Opponent Diversity Collapse

We instrument opponent-population behavioral diversity (mean pairwise L2 distance over simplex vectors). For `COEVOLVE`:

| Generation | Repertoire diversity | Opponent diversity |
|-----------:|---------------------:|-------------------:|
| 0          | 0.59                 | 0.54               |
| 10         | 0.35                 | 0.35               |
| 20         | 0.21                 | 0.30               |
| 30         | 0.15                 | 0.29               |
| 40         | 0.18                 | 0.28               |
| 60         | 0.18                 | 0.28               |

By generation 20 both populations have lost ≈60% of their initial diversity. Once that happens the "adversarial pressure" the literature relies on is no longer adversarial — every opponent in the population probes nearly the same region of the simplex, so selection on candidates degenerates toward optimization against a single representative opponent. The Hall-of-Fame mechanism (size 5) is too small to repair this on a 60-generation budget.

### 5.3 Population Does Not Beat Hill-Climbing

`GREEDY_HILLCLIMB` — a (1+1)-ES with no population, no crossover, starting from the greedy initial solution — matches the full population-based GA on held-out fitness (mean 1.027 vs. 1.026, p = 0.72). This means the crossover and tournament-selection machinery are not contributing beyond what a single-trajectory local search from a good starting point already achieves. The fitness landscape is smooth enough that hill-climbing converges to nearly the same basin from any greedy initialization. This is a second structural reason co-evolution cannot demonstrate a population-level advantage: the population itself is not necessary.

### 5.4 What This Implies

The negative result is not "co-evolution does not work for chess." It is the conjunction of (i) and (ii): given a symmetric train/eval distribution and a population whose diversity decays under tournament selection, the equilibrium of the co-evolutionary process is exactly the static optimizer's solution. Either lever — pulling the test distribution away from training, or actively maintaining opponent diversity (e.g., NSGA-II non-dominated sorting on exploitation + novelty, larger Hall-of-Fame, tabu archives) — would restore the regime where co-evolution should help.

---

## 6. Threats to Validity

- **Graph size.** ≈1,300 positions per split is small. A larger graph would expose more sub-strategies and might widen the gap, but the diversity-collapse mechanism is graph-agnostic.
- **Held-out is the same Lichess bands.** Generalization across rating bands is a weak distributional shift. A stronger test (e.g., titled-player games as held-out) would more aggressively probe robustness.
- **No engine ground truth.** Eval scores are empirical-Bayes-shrunk Lichess outcomes, not Stockfish evaluations. We measure what wins more games against humans, not objective move quality.
- **Reproducibility.** All numbers are deterministic given the seed and the crawl snapshot in `data/L2.db`. The Lichess explorer is updated daily, so a re-crawl will produce slightly different prior counts.

---

## 7. Conclusion

We built a memorization-constrained robust chess repertoire optimizer, compared static and co-evolutionary opponent regimes across 150 experiment runs, and found that on a symmetric train/test split co-evolution does not help. The closure rule — the novel structural mechanism — is confirmed load-bearing by ablation. The co-evolution null result is a clean falsification of the implicit "co-evolution improves robustness for free" assumption. The accompanying diagnosis — distributional symmetry plus opponent-population diversity collapse — is itself the contribution: it gives the next paper a concrete, testable design checklist (asymmetric train/test, explicit diversity preservation) without which a co-evolutionary robustness claim should not be believed.

---

## Appendix A — Reproducing the Results

```bash
pip install -r requirements.txt
python src/data_ingest.py          # ≈ 30 min, resumable; writes data/L2.db
python src/graph.py data/L2.db
python src/policies.py
python src/eval_cache.py
python src/experiments.py          # ≈ 1–2 hours for all 105 runs
python src/analyze.py              # writes results/
pytest tests/
```

Branch: `final`. Every run pickle stores the git commit hash in its `git_commit` field.

## Appendix B — Fixed Hyperparameters

| Parameter                          | Value                  |
|------------------------------------|------------------------|
| Memorization budget                | 30 nodes / color       |
| Closure threshold                  | 15%                    |
| Smoothing α                        | 5.0                    |
| Empirical-Bayes prior strength τ   | 20.0                   |
| CVaR fraction α                    | 1/3                    |
| λ (main runs)                      | 1.0                    |
| Rating bands                       | 1000–1399, 1400–1799, 1800–2199 |
| Train / held-out split             | pre-June 2025 / June 2025+ |
| Repertoire population              | 100                    |
| Opponent population                | 50                     |
| Generations                        | 60                     |
| Tournament size                    | 2                      |
| Crossover rate                     | 0.8                    |
| Mutation rate                      | 0.4                    |
| Opponent mutation strength         | 0.5                    |
| Novelty weight                     | 0.4                    |
| Hall-of-Fame size                  | 5                      |
| Seeds (main + closure ablation)    | 15 (1000–1014)         |
