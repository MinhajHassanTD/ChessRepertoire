# Co-Evolutionary Genetic Algorithms for Memorization-Constrained Robust Chess Opening Repertoires

**Author:** Minhaj ul Hassan, Zainab Zahid, Hazeera Muhammad Hashim
**Institution:** Habib University — Computational Intelligence (Spring 2026)
**Status:** Third draft — `main` branch

---

## Abstract

We study whether diversity-preserving co-evolutionary adversarial training improves the robustness of memorization-constrained chess opening repertoires. A repertoire commits to at most 25 moves per color and must cover any opponent reply played in ≥15% of games at a reached position (the *closure rule*). We compare three methods across 15 independent seeds: a greedy most-played-move heuristic, a static GA trained against a uniform rating-band mixture (`STATIC`), and a co-evolved opponent population with explicit diversity preservation (`COEVOLVE`). A greedy hill-climber (`GREEDY_HILLCLIMB`) serves as a non-population baseline with an equal evaluation budget.

All GA variants significantly outperform the most-played-move heuristic (held-out mean 0.526 vs. 0.494, p < 0.001, A12 = 0.00). Unlike naive co-evolution, the diversity-preserving implementation keeps opponent behavioral diversity stable throughout 100 generations (mean pairwise L2 distance ≈ 0.52, flat from generation 0 to 100). Under this regime, `COEVOLVE` shows a positive trend over `STATIC` (mean 0.5260 vs. 0.5255, A12 = 0.60) and the full population-based GA beats the hill-climber (STATIC vs. GREEDY: A12 = 0.70). However, neither gap reaches statistical significance at α = 0.05 (COEVOLVE vs. STATIC: p = 0.49). We trace the persistent null result to train–eval distributional symmetry: both training fitness and held-out evaluation reduce to expected score under the same uniform rating-band mixture, leaving co-evolution no distributional shift to exploit. A closure-rule ablation confirms the novel structural constraint is load-bearing (removal costs ≈ 0.008 held-out score, p < 0.001).

---

## 1. Introduction

Chess opening preparation is fundamentally a constrained memorization problem. A club-level player can realistically commit a few dozen specific positions to long-term memory, yet the branching tree of plausible opponent replies within the first five moves alone runs into thousands of positions. This tension — finite memory, exponential branching — makes opening preparation an instance of constrained combinatorial optimization under uncertainty.

The standard human approach to this problem is deterministic and fragile: pick the moves that appear most often in databases of games at your level, and memorize them. This heuristic ignores two structural problems. First, a move that is most popular on average may perform poorly against specific sub-populations of opponents (e.g., players in a particular rating band who favor unusual lines). Second, it treats every opponent reply as equally likely, even rare ones that can be catastrophic if encountered unprepared.

We treat the problem as **robust constrained optimization**: build a repertoire that performs well not just in expectation but against adversarial opponent mixtures drawn from a realistic distribution. The key mechanisms are:

- A **memorization budget**: at most 25 committed moves per color.
- A **closure rule**: any opponent reply played in ≥15% of games at a reached position must be covered automatically, ensuring the repertoire cannot be blindsided by common replies.
- A **CVaR-based fitness function**: the optimization objective penalizes worst-case performance across rating bands, not just average performance.

The interesting design question is *what opponent model to train against*. A static uniform mixture is simple and unbiased but may miss adversarial sub-distributions. Co-evolutionary adversarial training — evolving a population of opponents alongside repertoire candidates — promises to discover the candidate's worst case dynamically. The literature on co-evolutionary GAs (Hillis 1990; Rosin and Belew 1997) suggests this should improve robustness.

Our central question is whether this promise holds for this specific class of problem, and if not, why not.

> *Under what conditions does diversity-preserving co-evolutionary adversarial training improve generalization for memorization-constrained chess repertoire optimization?*

We implement the full pipeline including explicit diversity-preservation mechanisms (opponent crowding, population reinit on collapse, novelty-weighted fitness), run 45 main experiments plus 30 non-GA baseline runs across 15 seeds, and report results honestly. The answer is: co-evolution shows a consistent positive trend but fails to achieve statistical significance, and we identify train–eval distributional symmetry as the structural cause.

### 1.1 Contributions

1. A clean, reproducible implementation of memorization-constrained robust repertoire optimization with explicit diversity-preserving co-evolution: opponent crowding detection, diversity-threshold reinit, Hall-of-Fame (size 10), and novelty-weighted opponent fitness.
2. A 60-run evaluation (15 seeds × 4 methods) on a held-out Lichess data slice, with non-parametric significance testing (Wilcoxon signed-rank, Holm-corrected, Vargha–Delaney A12).
3. An empirical demonstration that diversity preservation stabilizes opponent populations (L2 distance ≈ 0.52 flat vs. collapse to 0.28 in naive co-evolution), partially rescuing the co-evolutionary dynamic.
4. A diagnosis of why the null result persists despite diversity preservation: train–eval distributional symmetry eliminates the information gradient co-evolution needs.
5. A closure-rule ablation confirming the structural constraint is load-bearing.

---

## 2. Related Work and Technical Background

### 2.1 Co-Evolutionary Genetic Algorithms

Co-evolutionary GAs maintain two or more co-adapting populations whose fitness depends on interactions between them. Hillis (1990) showed that co-evolving sorting networks against test-case adversary populations produced shorter networks than evolving against a fixed test set. Rosin and Belew (1997) formalized adversary *informativeness* — the degree to which an opponent reveals performance variation across candidates — and introduced Hall-of-Fame archives to prevent loss of historical pressure. Both ideas inform our implementation.

A well-documented failure mode of co-evolutionary systems is **diversity collapse**: under tournament selection the opponent population converges toward a narrow region of the space, so the adversarial signal becomes uninformative and selection degenerates toward a single representative opponent. Techniques for maintaining diversity include fitness sharing, niching, novelty search (Lehman and Stanley 2011), and explicit crowding mechanisms. We use novelty-weighted opponent fitness (diversity bonus proportional to mean L2 distance from other opponents) plus a hard reinit trigger when mean pairwise distance falls below a threshold.

### 2.2 Robust Optimization and CVaR

Conditional Value at Risk (CVaR; Rockafellar and Uryasev 2000) penalizes the worst-α tail of a score distribution rather than its expectation. With α = 1/3 over 3 rating bands, CVaR collapses to the worst-band score — a natural robustness objective when the agent must perform across multiple opponent sub-populations. Our fitness function is:

```
fitness = mean_score + λ · CVaR        (λ = 1.0 in main runs)
```

This balances average-case and worst-case performance. λ = 0 recovers mean-only optimization; λ → ∞ recovers minimax.

### 2.3 Constrained Combinatorial Optimization in Games

Most published work on chess preparation assumes engine-grade position evaluation (e.g., Stockfish centipawn scores) and an unlimited memory budget. The memorization budget is the constraint that makes the problem tractable but non-trivial: a 25-move budget covers approximately 2–4% of the ≈1,300-position opening graph, so the search space is combinatorially large. We evaluate using empirical win-rate data from Lichess rather than engine evaluation, measuring what actually wins more games against humans in each rating band.

---

## 3. Problem Description

### 3.1 What Is a Chess Opening Repertoire?

A chess opening repertoire is a pre-committed decision tree for the opening phase of the game. For each position where it is the player's turn to move, the repertoire specifies exactly one response. For positions where it is the opponent's turn, the repertoire must cover all replies likely to be encountered — otherwise the player exits their preparation and must improvise.

Formally, a repertoire for one color is a pair:
- `committed`: a map from board positions (FEN strings) to chosen moves (UCI strings), size ≤ 25.
- `reached`: a set of positions that can be reached if the player follows the committed moves and the opponent plays any covered reply.

The `reached` set is constructed by the **closure rule**: starting from the initial position, whenever a committed move is followed by an opponent reply played in ≥15% of games, that reply position is added to `reached`. This continues recursively until no new positions are added. The closure rule prevents the pathological case where a repertoire commits a move but leaves the player unable to respond to the most common replies.

### 3.2 Why a Genetic Algorithm?

The space of valid repertoires is large, discontinuous, and non-convex. Changing one committed move can cascade: it may violate the budget constraint (closure repair adds new positions), invalidate other branches (if a shared transposition no longer holds), or open qualitatively different lines (an entirely different opening system). Gradient methods are inapplicable because the fitness landscape is defined over discrete move choices. Exact search is infeasible for the same reason gradient methods fail and because the graph has ≈1,300 nodes with branching factors up to 10+.

Genetic algorithms handle this naturally:
- **Discrete operators**: mutation swaps one committed move for another valid move, extends the repertoire with a new branch, prunes an underperforming subtree, or replaces an entire opening system. Each mutation is followed by closure repair.
- **Crossover**: pivot-crossover splices a subtree at a shared committed position between two parents, combining opening systems.
- **Population**: 50 candidates explore the space in parallel, preserving diversity in the early generations when fitness gradients are weak.

The population also enables co-evolutionary adversarial training — fitness can be computed against a population of opponents rather than a fixed one.

---

## 4. Problem Formulation

### 4.1 Data

We crawl the Lichess Opening Explorer from the starting position to ply depth 8, collecting per-band move statistics for rapid and classical time controls. The crawl threshold is 5% minimum move frequency (only follow moves played in ≥5% of games) and minimum game counts that tighten with depth (5,000 for ply ≤3, 20,000 for ply ≤6, 50,000 for ply >6). The train/held-out split is temporal: train = pre-June 2025, held-out = June 2025 onward. Both splits expose approximately 1,300 reachable positions.

### 4.2 Position Graph

Two pickled graphs store, for each position, its FEN-canonical key (move clocks stripped, so transpositions collapse), turn-to-move, total game count, per-band win/draw/loss counts, and child links keyed by UCI move with per-band play counts.

### 4.3 Per-Band Move Policies

For each rating band *b* and position *f* we compute a smoothed move policy:

```
π_b(m | f) = (count_b(m, f) + α · π_agg(m | f)) / (Σ_m count_b(m, f) + α),     α = 5
```

This Laplace-style smoothing shrinks sparse per-band evidence toward the aggregate distribution, falling back fully when a band has zero games at a position.

### 4.4 Eval Cache (Empirical-Bayes-Shrunk Leaf Scores)

Each (position, band) pair receives a shrunk expected white-score:

```
ŝ_b(f) = (n · raw + τ · prior_mean) / (n + τ),     τ = 20
```

The prior mean is the games-weighted aggregate score over positions at ply ≤8 (≈0.52, reflecting White's first-move advantage). Positions with few games are pulled toward the prior; well-sampled positions trust the raw win-rate.

### 4.5 Repertoire Fitness

The **walk function** scores a repertoire under one band's policy recursively: at our-turn nodes, follow the committed move (or return the cached leaf if uncommitted); at opponent-turn nodes, weight every covered child by π_b. The per-band score is the average of White's walk score and (1 − Black's walk score).

For an opponent mixture *p* ∈ Δ³ (the probability simplex over 3 bands):

```
mean_score  = Σ_b p_b · score_b
CVaR        = min_b score_b                   (α = 1/3 over 3 bands)
fitness     = mean_score + λ · CVaR           (λ = 1.0 in main runs)
```

Held-out evaluation always uses the uniform mixture p = (1/3, 1/3, 1/3) regardless of how training was run, so cross-method comparison is unbiased.

### 4.6 Genetic Operators

**Mutation** applies one of four operators with equal probability, followed by closure repair:
- `mutate_move_swap`: replace one committed move with a different legal move at the same position.
- `mutate_extend`: add a new committed move at an uncovered our-turn position within `reached`.
- `mutate_prune`: remove a committed move (and close descendant positions) to free budget.
- `mutate_opening_replacement`: replace the entire subtree below a shallow (ply ≤4) position with a different opening system.

**Crossover** (pivot-crossover): find a position both parents have committed identically; splice parent B's subtree below that pivot into parent A's repertoire, followed by closure repair and budget enforcement.

**Selection**: binary tournament — draw 4 candidates uniformly at random, return the fittest.

### 4.7 Opponent Representation and Co-Evolution

An opponent is a 3-simplex weight vector p ∈ Δ³ over rating bands. The three opponent regimes:

- `STATIC`: one frozen opponent (1/3, 1/3, 1/3). Used for training; held-out also uses this mixture.
- `GREEDY_HILLCLIMB`: a (1+1)-ES with no population, starting from the greedy most-played-move solution. Same evaluation budget as the GA (50 × 100 = 5,000 fitness calls).
- `COEVOLVE`: 50 opponents co-evolved via tournament selection (size 4), Dirichlet-resample crossover (concentration scale 2.0), and Dirichlet-noise mutation (strength 0.8, rate 0.5). A Hall-of-Fame of size 10 retains the most *informative* past opponents (those that maximally discriminate between repertoire candidates).

**Diversity preservation in COEVOLVE.** Opponent fitness is:

```
opponent_fitness = exploitation + ω · diversity
exploitation = −score(opponent against current population)
diversity    = mean L2 distance from this opponent to all others in the population
ω = 1.0 (novelty weight)
```

Additionally: (i) pairs of new opponents closer than L2 = 0.05 trigger crowding replacement of the later member with a fresh random opponent; (ii) if mean pairwise population L2 distance falls below 0.3, the worst 40% of opponents are replaced with fresh random opponents (diversity reinit).

---

## 5. Experiments

### 5.1 Setup

| Parameter                        | Value                           |
|----------------------------------|---------------------------------|
| Memorization budget              | 25 moves / color                |
| Closure threshold                | 15%                             |
| Smoothing α                      | 5.0                             |
| Empirical-Bayes τ                | 20                              |
| CVaR fraction                    | 1/3 (worst band of 3)           |
| λ (fitness weight)               | 1.0                             |
| Rating bands                     | 1000–1399, 1400–1799, 1800–2199 |
| Train / held-out split           | pre-June 2025 / June 2025+      |
| Repertoire population            | 50                              |
| Opponent population (COEVOLVE)   | 50                              |
| Generations                      | 100                             |
| Tournament size                  | 4                               |
| Crossover rate                   | 1.0                             |
| Mutation rate                    | 0.5                             |
| Opponent mutation strength       | 0.8                             |
| Novelty weight ω                 | 1.0                             |
| Opponent crowding threshold (L2) | 0.05                            |
| Opponent diversity threshold (L2)| 0.3                             |
| Opponent reinit fraction         | 0.40                            |
| Hall-of-Fame size                | 10                              |
| Seeds (main runs)                | 15 (1000–1014)                  |
| Total main runs                  | 3 methods × 15 seeds = 45 runs  |
| Total baseline runs              | 2 methods × 15 seeds = 30 runs  |

Statistical tests: paired Wilcoxon signed-rank (paired across seeds), Holm-corrected for multiple comparisons; Vargha–Delaney A12 effect size (A12 > 0.5 means the first method wins more often in pairwise comparison).

### 5.2 Headline Results — Held-Out Uniform-Mixture Score

| Method               | Mean   | Std    | Wilcoxon p vs. STATIC | A12 vs. STATIC |
|----------------------|--------|--------|------------------------|----------------|
| most_played_baseline | 0.4938 | ≈ 0    | < 0.001 ***            | 0.00           |
| GREEDY_HILLCLIMB     | 0.5244 | 0.0021 | 0.167                  | 0.30           |
| **STATIC**           | 0.5255 | 0.0013 | —                      | —              |
| COEVOLVE             | 0.5260 | 0.0017 | 0.489                  | 0.60           |

Three observations stand out.

**The framework works.** All GA-adjacent methods beat the most-played-move heuristic by a large, significant margin (p < 0.001, A12 = 0.00 for the baseline). Blindly following the most common database move at each position is materially suboptimal.

**The population earns its cost.** `GREEDY_HILLCLIMB` — a single-trajectory local search with the same evaluation budget — scores A12 = 0.30 against `STATIC`, meaning the full population-based GA wins in roughly 70% of paired comparisons. Unlike results from earlier experimental configurations (fewer generations, smaller HOF), the population here is doing useful exploration that a hill-climber cannot replicate.

**Co-evolution shows a positive trend but is not significant.** `COEVOLVE` scores A12 = 0.60 against `STATIC` — winning in 60% of paired comparisons — but the Wilcoxon p-value is 0.49. We cannot reject the null hypothesis that the two methods are equivalent at any conventional significance level.

### 5.3 Worst-Band Held-Out Score

| Method           | Mean   | Std    | Wilcoxon p vs. STATIC | A12 vs. STATIC |
|------------------|--------|--------|------------------------|----------------|
| most_played_baseline | 0.4898 | ≈ 0 | < 0.001 ***         | 0.00           |
| GREEDY_HILLCLIMB | 0.5194 | 0.0014 | 0.375                  | 0.38           |
| **STATIC**       | 0.5201 | 0.0020 | —                      | —              |
| COEVOLVE         | 0.5207 | 0.0017 | 0.330                  | 0.59           |

The worst-band score (the CVaR-relevant metric) tells the same story. `COEVOLVE` scores A12 = 0.59 on the tail metric, marginally ahead of `STATIC`, but the difference is again not significant (p = 0.33). If co-evolutionary robustness were activating, we would expect a larger effect here — we do not see it.

### 5.4 Diversity Dynamics in COEVOLVE

We track mean pairwise L2 distance in both populations across 100 generations (averaged over the seed at which the diagnostic table was produced):

| Generation | Repertoire diversity | Opponent diversity |
|-----------:|---------------------:|-------------------:|
| 0          | 0.85                 | 0.52               |
| 10         | 0.53                 | 0.50               |
| 20         | 0.52                 | 0.53               |
| 30         | 0.45                 | 0.52               |
| 50         | 0.45                 | 0.51               |
| 75         | 0.36                 | 0.53               |
| 100        | 0.37                 | 0.54               |

The opponent population **does not collapse**. Diversity stays in the range 0.50–0.54 across the entire run — a marked contrast to naive co-evolution without diversity preservation, where opponent diversity can fall to ≈0.28 by generation 20 and the adversarial signal becomes uninformative. The crowding replacement and reinit mechanisms are functioning as designed.

The repertoire population does lose diversity (0.85 → ≈0.37) as the GA converges, which is expected and healthy — it reflects convergence toward good solutions rather than stagnation.

### 5.5 Closure-Rule Ablation

Running `STATIC` and `COEVOLVE` without the closure rule (only the direct child of each committed move is added to `reached`, rather than the full transitive closure at ≥15% frequency):

| Method             | Held-Out Mean | Std    | p vs. WITH closure | A12  |
|--------------------|---------------|--------|--------------------|------|
| STATIC             | 1.0264        | 0.0029 | —                  | —    |
| STATIC_NOCLOSURE   | 1.0184        | 0.0041 | < 0.001 ***        | 0.07 |
| COEVOLVE           | 1.0248        | 0.0024 | —                  | —    |
| COEVOLVE_NOCLOSURE | 1.0171        | 0.0038 | < 0.001 ***        | 0.06 |

Removing the closure rule drops held-out performance significantly for both methods (≈0.008, p < 0.001). The closure rule is not cosmetic — it forces the repertoire to remain coherent against the most common opponent replies and is a structural load-bearing component of the optimization.

---

## 6. Conclusion

We built a memorization-constrained robust chess opening repertoire optimizer with three opponent regimes and explicit diversity-preserving mechanisms, evaluated across 45 main runs and 30 baseline runs over 15 seeds.

**What works.** The GA framework substantially outperforms the most-played-move heuristic — a result that holds robustly across all configurations and is the clearest practical finding. The population earns its cost in our current configuration: a greedy hill-climber with the same budget loses to the full population-based GA in roughly 70% of paired comparisons (A12 = 0.30 for GREEDY vs. STATIC). The closure rule is confirmed load-bearing by ablation.

**What diversity preservation achieves.** Explicit diversity mechanisms — novelty-weighted opponent fitness, crowding replacement, and diversity-threshold reinit — successfully prevent opponent-population collapse. Opponent diversity remains stable at ≈0.52 throughout 100 generations. This addresses the structural failure mode identified in naive co-evolutionary configurations and ensures the adversarial pressure is sustained.

**What diversity preservation does not achieve.** Despite stable opponent diversity, `COEVOLVE` does not significantly outperform `STATIC` (p = 0.49, A12 = 0.60). The positive trend is consistent across both the uniform-mixture and worst-band metrics, but it falls short of statistical significance at n = 15. The remaining bottleneck is structural: the training fitness and the held-out evaluation both reduce to expected score under the *same* uniform rating-band mixture. Co-evolution evolves opponents toward adversarial mixtures, but the held-out test rewards average-case behavior over that same prior. The static optimizer is already near-Bayes-optimal for the test distribution, and co-evolution adds variance without accessing a distributional gradient it can exploit.

The overall picture: diversity preservation partially rescues co-evolutionary dynamics, producing a consistent positive trend over the static baseline. A real train/test distribution shift — one where the opponent mixture during training systematically differs from the held-out distribution — would give co-evolution the asymmetry it needs to demonstrate a clear advantage.

---

## 7. Future Work

- **Asymmetric train/test distribution.** The most direct extension is to construct a genuine distributional shift: train against one rating-band mixture (e.g., overweighting the lowest band) and evaluate on a different mixture (e.g., the highest band). This gives co-evolution an exploitable gradient and tests whether the positive trend we observe becomes a significant win.

- **Larger graph and deeper ply.** The ≈1,300-position graph at ply ≤8 constrains the search space. A deeper crawl (ply ≤12) would expose more opening systems and sub-strategies, potentially widening the performance gap between methods and between the hill-climber and the full population.

- **Stronger diversity mechanisms.** NSGA-II non-dominated sorting on exploitation and novelty simultaneously, or a tabu archive for opponents, could maintain diversity more aggressively than the current threshold-based reinit. A larger Hall-of-Fame or quality-diversity approaches (MAP-Elites over the simplex) are also natural extensions.

- **Engine-validated ground truth.** Our evaluation is based on empirical Lichess win-rates, which measure what wins more games against human opponents at each rating band, not objective move quality. Cross-referencing with Stockfish evaluations on the committed positions would separate "exploits human tendencies" from "is objectively stronger."

- **Multi-color co-evolution.** Currently White and Black repertoires are optimized independently. A joint candidate that co-evolves White and Black simultaneously against a shared opponent population could discover synergistic opening choices across colors.

---

## 8. Acknowledgments

This project was developed as part of the Computational Intelligence (CS 462) course at Habib University, Spring 2026. Opening data is sourced from the Lichess Opening Explorer API under the Creative Commons Attribution 4.0 license.

---

## 9. References

Hillis, W. D. (1990). Co-evolving parasites improve simulated evolution as an optimization procedure. *Physica D: Nonlinear Phenomena*, 42(1–3), 228–234.

Lehman, J., and Stanley, K. O. (2011). Abandoning objectives: Evolution through the search for novelty alone. *Evolutionary Computation*, 19(2), 189–223.

Rockafellar, R. T., and Uryasev, S. (2000). Optimization of conditional value-at-risk. *Journal of Risk*, 2(3), 21–41.

Rosin, C. D., and Belew, R. K. (1997). New methods for competitive coevolution. *Evolutionary Computation*, 5(1), 1–29.

---

## Appendix A — Reproducing the Results

```bash
pip install -r requirements.txt
python src/data_ingest.py          # ≈ 30 min, resumable; writes data/snapshot.db
python src/graph.py data/snapshot.db
python src/policies.py
python src/eval_cache.py
python src/experiments.py          # ≈ 2–3 hours for all 75 runs
python src/analyze.py              # writes experiments/<config>/results/
pytest tests/
```

All runs are deterministic given the seed and the crawl snapshot in `data/snapshot.db`. Every run pickle stores the git commit hash in its `git_commit` field.
