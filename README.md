# Co-Evolutionary Genetic Algorithm for Chess Opening Repertoires

**Full title:** Co-Evolutionary Genetic Algorithm for Memorization-Constrained Robust Chess Opening Repertoires

**Core claim:** A closure-constrained chromosome representation forces opponent reply coverage during evolution, and co-evolutionary adversarial training acts as adaptive regularization. Together they produce opening repertoires that generalize across player skill levels.

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [Key Concepts](#2-key-concepts)
3. [How the Algorithm Works](#3-how-the-algorithm-works)
4. [Project Structure](#4-project-structure)
5. [Data Flow](#5-data-flow)
6. [Setup](#6-setup)
7. [Running the Pipeline](#7-running-the-pipeline)
8. [Component Reference](#8-component-reference)
9. [Fixed Parameters](#9-fixed-parameters)
10. [Interpreting Output](#10-interpreting-output)

---

## 1. What Is This Project?

A chess **opening repertoire** is a pre-planned set of moves a player memorizes before a game. When White plays `1.e4`, Black might respond `1...e5`, White then plays `2.Nf3`, and so on — these branching sequences form a tree of decisions.

**The problem:** memorizing hundreds of lines is impractical. This project answers:

> *Given a strict budget of committed moves per color, what opening repertoire performs best across players of different skill levels?*

We frame it as a **genetic algorithm** (an evolutionary optimization technique inspired by natural selection). The key novelty is twofold:

1. **Closure-constrained chromosome** — the repertoire is not just a list of our moves; it also tracks all common opponent replies that must be covered. Any opponent move played in ≥15% of real games is automatically added to the repertoire's coverage set. This mirrors how a real player thinks: "I play 1.e4, so I must also have answers to 1...e5, 1...c5, 1...e6, and 1...c6."

2. **Co-evolutionary opponent** — an adversarial population of "opponents" (represented as mixtures over player skill bands) evolves *alongside* the repertoires, constantly probing for weaknesses. This prevents the repertoire from overfitting to a fixed opponent type.

### Methods compared

| Method | Role | Evolves? |
|---|---|---|
| `most_played_baseline` | Greedy: always pick the most-played move. No optimization. | No |
| `GREEDY_HILLCLIMB` | Non-GA search: start greedy, accept any mutation that improves fitness. Same eval budget as GA. | No |
| `STATIC` | GA with a fixed uniform-mixture opponent (equal weight to all skill bands) | GA only (no opponent) |
| `COEVOLVE` | Full co-evolution: both repertoire and opponent populations evolve together | Yes |

`GREEDY_HILLCLIMB` is the critical non-GA baseline — if it matches `STATIC`, the GA population + crossover machinery adds no value beyond plain hill-climbing.

### Ablation experiments (new)

| Experiment | Purpose |
|---|---|
| `STATIC_NOCLOSURE` / `COEVOLVE_NOCLOSURE` | Does the closure rule actually help, or would a simpler unconstrained representation work equally well? |
| Closure threshold sweep | Is 15% the right threshold, or does performance change a lot across 5%–30%? |
| Budget sweep | Does performance plateau before 25 committed moves, or continue growing? |

---

## 2. Key Concepts

### FEN — a chess position fingerprint
Every unique chess position has a compact text label called a **FEN** string. This project uses only the first 4 fields so that positions reached by different move orders but identical board states are treated as the same node (transposition collapsing).

### Rating Bands — opponent skill levels
Real Lichess games are split into three skill brackets:

| Band | Approximate Elo |
|---|---|
| `1000-1399` | Beginners |
| `1400-1799` | Club players |
| `1800-2199` | Strong amateurs |

A repertoire that performs well against all three bands is **robust**. A repertoire that only works against one is exploitable.

### Closure — covering all common opponent replies
If we commit to `1.e4`, and 20% of opponents play `1...e5` while 18% play `1...c5`, both exceed the 15% threshold, so the repertoire **must** have prepared answers to both. This is the closure rule.

Without closure, a repertoire could commit to `1.e4` and then have no response to `1...c5` — a real blind spot against the Sicilian. Closure eliminates those holes automatically.

### Budget — the memorization constraint
Each repertoire may commit to at most **25 decision nodes** per color (White and Black handled independently). A committed node is a position where it is our turn and we have chosen a specific move to play.

### CVaR — worst-case robustness
CVaR (Conditional Value at Risk) measures the expected score in the **worst** rating band. With 3 bands, CVaR equals the lowest single-band score. The fitness function is:

```
fitness = mean_score + lambda × CVaR
```

With lambda = 1.0, equal weight is given to overall mean performance and worst-case robustness.

Note: fitness can exceed 1.0. With lambda = 1.0, the maximum is `mean + cvar ≤ 1 + 1 = 2.0`. Values around 1.0 mean both mean and worst-case are near 0.5 (roughly random-play level).

---

## 3. How the Algorithm Works

### Step-by-step overview

```
1. Initialize two populations:
   - R_pop: 50 candidate repertoires (chromosome = closure-constrained committed-move set)
   - O_pop: 30 opponent mixtures (chromosome = 3-number weight vector over skill bands)

2. For each generation (default: 100):
   a. EVALUATE each repertoire against all opponents → assign fitness
   b. REPRODUCE repertoires via tournament selection + crossover + mutation
   c. [COEVOLVE only] EVALUATE opponents → how hard did they challenge R_pop?
   d. [COEVOLVE only] REPRODUCE opponents
   e. [COEVOLVE only] UPDATE Hall of Fame (5 most informative past opponents)
   f. LOG stats for this generation

3. After the loop:
   - Pick best repertoire from final generation
   - Evaluate it on held-out data under a uniform opponent mixture
   - Return result dictionary
```

### Fitness computation (the `walk` function)

To score a repertoire, we simulate a game tree — **not by sampling, but by computing the exact expected value**:
- **Our turn + committed move:** follow that move, continue recursing
- **Our turn + no committed move (leaf):** return the pre-computed expected score from the eval cache
- **Opponent's turn:** take a weighted average over all opponent moves, using the band's empirical move probabilities as weights

This is a deterministic expected-value computation, not a Monte Carlo simulation. The eval cache provides pre-computed Bayesian-shrunk scores for all leaf positions.

### Mutation operators (four types)

| Operator | What it does |
|---|---|
| `mutate_move_swap` | Pick a committed node, swap its move to a different one; rebuild the subtree below |
| `mutate_extend` | Pick an uncommitted leaf, commit a move there (uses 1 budget slot) |
| `mutate_prune` | Remove a committed node, making it a leaf (frees 1 budget slot) |
| `mutate_opening_replacement` | Replace a shallow committed node (ply ≤ 4) and rebuild everything below randomly |

If a mutation would exceed the budget or violate closure, it raises `MutationFailed` and is retried (up to 5 attempts). If all 5 fail, the individual is cloned unchanged.

### Crossover

Two parent repertoires share a **pivot position** — a node where both committed the same move. The child keeps Parent A's structure from the root up to the pivot, and grafts Parent B's subtree from the pivot onward.

The subtree lives in a DAG (directed acyclic graph) because chess positions can be reached via different move orders (transpositions). The `_remove_subtree` and `_still_reachable_without` helpers handle this correctly — a position shared by two paths is only removed if it's genuinely unreachable after the graft.

---

## 4. Project Structure

```
ChessRepertoire/
├── README.md                 ← this file
├── requirements.txt          ← Python dependencies
├── .env.example              ← template for Lichess API token
├── .gitignore
│
├── data/
│   ├── snapshot.db           ← SQLite snapshot (C1 output, main database)
│   ├── graph_train.pkl       ← training position graph (C2)
│   ├── graph_heldout.pkl     ← held-out position graph (C2)
│   ├── base_policies.pkl     ← smoothed per-band move policies (C3)
│   ├── eval_cache_train.pkl  ← shrunk expected scores, train (C4)
│   └── eval_cache_heldout.pkl← shrunk expected scores, held-out (C4)
│
├── src/
│   ├── config.py             ← all tunable constants (one place to change everything)
│   ├── data_ingest.py        ← C1: Lichess API crawler
│   ├── graph.py              ← C2: position graph builder
│   ├── policies.py           ← C3: per-band move distributions
│   ├── eval_cache.py         ← C4: Bayesian position scores
│   ├── repertoire.py         ← C5: closure-constrained chromosome + genetic operators
│   ├── fitness.py            ← C6: fitness function + walk
│   ├── opponent.py           ← C7: opponent chromosome
│   ├── coevolution.py        ← C8: main GA loop (STATIC / COEVOLVE)
│   ├── experiments.py        ← C9: full experimental run matrix + dispatch
│   └── analyze.py            ← C10: results analysis + all figures
│
├── runs/                     ← per-run pickle files (gitignored)
├── results/                  ← generated figures and tables
│   └── appendix/             ← appendix figures (--appendix flag)
└── tests/                    ← pytest suite (one file per component)
```

---

## 5. Data Flow

```
Lichess API
    │  (C1 — data_ingest.py)
    ▼
data/snapshot.db  (SQLite)
    │  (C2 — graph.py)
    ▼
graph_train.pkl + graph_heldout.pkl
    │  (C3 — policies.py)          (C4 — eval_cache.py)
    ▼                                      ▼
base_policies.pkl              eval_cache_train.pkl
                                eval_cache_heldout.pkl
    │                                      │
    └──────────────┬────────────────────────┘
                   │  (C5 — repertoire.py + C6 — fitness.py)
                   ▼
         Candidate (closure-constrained white + black Repertoire pair)
         + fitness function (walk → mean_score + lambda × CVaR)
                   │  (C8 — coevolution.py)
                   ▼
         Coordinator: STATIC / COEVOLVE / *_NOCLOSURE (ablation)
                   │  (C9 — experiments.py)
                   ▼
             runs/*.pkl
                   │  (C10 — analyze.py)
                   ▼
             results/
```

---

## 6. Setup

### Prerequisites
- Python 3.10 or higher
- A free [Lichess](https://lichess.org) account (for the API token)

### Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies: `python-chess`, `numpy`, `pandas`, `scipy`, `matplotlib`, `requests`, `python-dotenv`, `pytest`

### Set up your Lichess API token

1. Go to `https://lichess.org/account/oauth/token`
2. Create a token (no special scopes needed)
3. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
4. Open `.env` and paste your token:
   ```
   LICHESS_API_TOKEN=lip_yourTokenHere
   ```

The `.env` file is gitignored and must never be committed.

---

## 7. Running the Pipeline

Run components **in order**. Each component depends on the output of the previous one.

### Step 1 — Collect data (C1)
*Skip if `data/snapshot.db` already exists and is populated.*

```bash
python src/data_ingest.py
```

Crawls Lichess Opening Explorer from the starting position up to ply depth 10. Stores all positions and move statistics in `data/snapshot.db`. **Resumable** — if interrupted, restart and it picks up where it left off.

### Step 2 — Build position graphs (C2)

```bash
python src/graph.py data/snapshot.db
```

Outputs: `data/graph_train.pkl`, `data/graph_heldout.pkl`

### Step 3 — Compute base policies (C3)

```bash
python src/policies.py
```

Outputs: `data/base_policies.pkl`

Also prints a **band separation check** — the mean total-variation distance between rating band policies. Should be ≥ 0.05 for the CVaR story to be meaningful.

### Step 4 — Build eval cache (C4)

```bash
python src/eval_cache.py
```

Outputs: `data/eval_cache_train.pkl`, `data/eval_cache_heldout.pkl`

### Step 5 — Main experiments (C9)

```bash
python src/experiments.py
```

Runs 120 total experiments:
- **Main comparison:** 4 methods × 15 seeds = 60 runs (lambda = 1.0)
- **Non-GA baselines:** 2 methods × 15 seeds = 30 runs
- **Lambda sensitivity:** 2 methods × 3 lambdas × 5 seeds = 30 runs

Resumable — skips runs whose output file already exists. Results saved to `runs/`.

### Step 5b — Ablation experiments (optional, new)

```bash
python -m src.experiments ablations
```

Runs ~70 additional experiments:
- **Closure ablation:** `STATIC_NOCLOSURE` + `COEVOLVE_NOCLOSURE` × 15 seeds = 30 runs
- **Closure threshold sweep:** STATIC × 5 thresholds × 5 seeds = 25 runs
- **Budget sweep:** STATIC × 3 budgets × 5 seeds = 15 runs

### Step 6 — Analyze results (C10)

```bash
# Core figures only (fast, for iteration):
python src/analyze.py

# Core + appendix figures (for final paper):
python src/analyze.py --appendix
```

Generates figures, tables, and statistics in `results/`. With `--appendix`, also writes supplementary figures to `results/appendix/`.

### Run tests

```bash
pytest tests/
```

---

## 8. Component Reference

### C1 — `src/data_ingest.py`

Queries the Lichess Opening Explorer API recursively from the starting position. For each position and each rating band, fetches aggregate statistics and per-band move counts.

**Key design decisions:**
- **Canonical FEN:** only the first 4 fields are kept. Transpositions collapse to the same node.
- **Resumable:** a `fetched_bands` table tracks completed API calls.
- **Rate limit:** 1.1 seconds between calls to avoid throttling.
- **Enqueue threshold:** a child position is only queued if its play count exceeds a depth-dependent minimum (10k at depth ≤ 3, 30k at depth ≤ 6, 80k at depth ≤ 10) AND aggregate frequency ≥ 10%.

---

### C2 — `src/graph.py`

Loads the SQLite snapshot into a Python dict. One graph per split (train / heldout).

**Node structure:**
```python
graph['nodes'][fen] = {
    'fen': str,
    'ply_depth': int,
    'turn': 'white' | 'black',
    'total_games': int,
    'score_raw': float,        # (white_wins + 0.5×draws) / total_games
    'children': {
        '<move_uci>': {
            'child_fen': str,
            'move_san': str,   # human-readable (e.g. "e4")
            'aggregate_count': int,
        }
    }
}
```

---

### C3 — `src/policies.py`

Computes how likely each rating band is to play each move at each position.

**Smoothing formula** (additive smoothing toward aggregate distribution):
```
policy[band][fen][move] = (band_count[move] + alpha × agg_prob[move]) / (band_total + alpha)
```
- `alpha = 5.0` — smoothing strength
- Falls back to aggregate distribution when a band has no games at a position.

---

### C4 — `src/eval_cache.py`

Pre-computes a shrunk expected score for every (position, band) pair using Empirical Bayes.

**Formula:**
```
prior_mean = weighted average score across positions at ply ≤ 8
shrunk     = (N × raw + tau × prior_mean) / (N + tau)
```
- `tau = 20` — how strongly sparse positions are pulled toward the prior
- All scores are from White's perspective. For Black: use `1 - score`.

---

### C5 — `src/repertoire.py`

Defines the closure-constrained chromosome and all evolutionary operators.

**Chromosome:**
```python
Repertoire:
    color:       'white' | 'black'
    committed:   {fen: move_uci}   # our decision nodes (max BUDGET)
    reached:     set[fen]          # committed children + all closure-forced positions
    use_closure: bool              # False = ablation mode (no auto-opponent coverage)

Candidate:
    white:             Repertoire
    black:             Repertoire
    fitness:           float | None
    band_scores_cache: dict | None  # invalidated after any structural change
```

**Closure rule (when `use_closure=True`):**
After committing a move at position `P`, the BFS expander `_expand_to_closure` traverses all opponent-turn nodes and force-adds any child with aggregate frequency ≥ `CLOSURE_THRESHOLD` to `reached`. This ensures the repertoire is always complete: no position in `reached` has an unhandled common opponent reply.

**Ablation mode (`use_closure=False`):**
Only the direct child of each committed move is added. Opponents replies are not forced. Used to measure the value of the closure constraint.

**Mutation operators:**

| Function | Description |
|---|---|
| `mutate_move_swap` | Swap a committed move; rebuild the subtree using closure expansion |
| `mutate_extend` | Commit at a leaf (uses 1 budget slot); expand closure below it |
| `mutate_prune` | Remove a committed node, making it a leaf (frees 1 budget slot) |
| `mutate_opening_replacement` | Replace a shallow node (ply ≤ 4) and rebuild randomly below it |

**Subtree removal in a DAG:** positions reachable via an alternative path (transpositions) are preserved. Handled by `_still_reachable_without()`.

---

### C6 — `src/fitness.py`

**`walk(rep, band, eval_cache, base_policies, graph)`**

Deterministic expected-value tree walk:
```
our turn  + committed move → follow the move, recurse
our turn  + no move (leaf) → return eval_cache score for this position/band
opp turn                   → weighted sum over all moves (policy weights)
                             off-book child → use eval_cache score directly
```

**`evaluate(candidate, opponent_mixture, config, ...)`**

```
band_scores[band] = 0.5 × walk(white_rep, band) + 0.5 × (1 - walk(black_rep, band))
mean_score        = sum(opponent_mixture[i] × band_scores[i])
CVaR              = min(band_scores.values())     # worst single band
fitness           = mean_score + lambda × CVaR
```

**`evaluate_heldout()`** — same logic on held-out graph + eval cache, always under a uniform opponent mixture for fair cross-mode comparison.

---

### C7 — `src/opponent.py`

The opponent chromosome is a 3-element probability vector (one weight per rating band) that sums to 1.

```python
Opponent.uniform()        # [1/3, 1/3, 1/3]
Opponent.random(rng)      # Dirichlet(1,1,1) sample
opp.mutate(rng)           # blend with Dirichlet noise
opp.crossover(other, rng) # convex combination of two parents
```

In COEVOLVE mode, opponent fitness = `exploitation + novelty`:
- **Exploitation** = how consistently this opponent hurts the repertoire population
- **Novelty** = 0.4 × mean L2 distance from this opponent to all others (encourages diverse simplex coverage)

---

### C8 — `src/coevolution.py`

The main loop. One call to `run_coevolution(mode, config, seed, ...)` executes one complete run and returns a result dict.

**RNG discipline:** a single `np.random.default_rng(seed)` is created once and passed everywhere. No other random source is used. This guarantees **bit-identical reproducibility** given the same seed.

**Config dict keys:**

| Key | Default | Description |
|---|---|---|
| `lambda_weight` | 1.0 | CVaR weight in fitness |
| `use_closure` | True | Whether closure rule is active |
| `novelty_weight` | 0.4 | Opponent diversity bonus |
| `hof_size` | 5 | Hall of Fame size (COEVOLVE only) |

**Return dict:** `mode`, `config`, `seed`, `git_commit`, `history`, `final_best_candidate`, `final_training_fitness`, `heldout_score`, `wall_time_seconds`

The `final_best_candidate` serialization:
```python
{
    'white_committed': {fen: move_uci, ...},
    'white_reached':   [fen, ...],
    'black_committed': {fen: move_uci, ...},
    'black_reached':   [fen, ...],
}
```

---

### C9 — `src/experiments.py`

Defines six experiment batches and dispatches each run to the right function.

**Dispatch logic for new ablation modes:**
- `STATIC_NOCLOSURE` / `COEVOLVE_NOCLOSURE`: strips `_NOCLOSURE` suffix, sets `use_closure=False` in the config dict, then calls `run_coevolution` with the base mode. The `mode` field in the saved pickle retains the original name so `analyze.py` can group them separately.
- `closure_threshold` key: overrides `CLOSURE_THRESHOLD` for a single run.
- `budget` key: overrides `BUDGET` for a single run.

---

### C10 — `src/analyze.py`

**Core outputs always produced:**

| Output | Description |
|---|---|
| `main_table.csv` | Mean, std, Wilcoxon p (Holm-corrected), A12 vs STATIC for all 4 methods |
| `convergence.png` | GA training curves (STATIC, COEVOLVE) with 95% CI |
| `score_distributions.png` | Box + strip plots for both held-out metrics |
| `band_breakdown.png` | Per-band scores grouped by method |
| `closure_ablation.png` | Closure ON vs OFF for STATIC and COEVOLVE |
| `closure_threshold.png` | Held-out score vs threshold (5%–30%) |
| `budget_sensitivity.png` | Held-out score vs budget (15, 25, 35) |
| `opening_diversity.csv` | First-move distribution per method across seeds |
| `compute_cost.csv` | Median wall time per method |

**Appendix outputs (with `--appendix`):**
`effect_sizes.png`, `ga_vs_nonga.png`, `white_black_breakdown.png`, `generalization.png`, `generalization_gap.png`, `repertoire_structure.png`, `lambda_sensitivity.png`, `coevolve_dynamics.png`, `seed_heatmap.png`, `sensitivity_table.csv`, `diagnostic_table.csv`, `repertoire_tree.txt`, `repertoire_grouped.txt`

---

## 9. Fixed Parameters

| Parameter | Value | Note |
|---|---|---|
| Max ply depth | 10 | 5 full moves |
| Budget per color | 25 committed nodes | Memorization constraint |
| Closure threshold | 15% | Min frequency for mandatory coverage |
| Rating bands | 1000-1399, 1400-1799, 1800-2199 | Must match crawled data |
| Time controls | rapid, classical | From Lichess |
| Smoothing alpha | 5.0 | Band policy smoothing |
| Bayes prior strength tau | 20 | Eval cache shrinkage |
| Lambda (main runs) | 1.0 | CVaR weight in fitness |
| Lambda (sensitivity) | 0.0, 1.0, 2.0 | Appendix experiment |
| Population — repertoires | 50 | |
| Population — opponents | 30 (COEVOLVE), 1 (STATIC) | |
| Generations | 100 | |
| Tournament size | 2 | |
| Crossover rate | 0.8 | |
| Mutation rate | 0.3 | |
| Hall of Fame size | 5 | COEVOLVE only |
| Seeds (main) | 15 seeds: 1000–1014 | |
| Seeds (sensitivity/ablation) | 5 seeds: 2000–2004 | |
| Train/held-out cutoff | 2025-06 | Lichess data split |

---

## 10. Interpreting Output

### Fitness > 1.0 — is that correct?

Yes. Fitness = `mean_score + lambda × CVaR`. Both `mean_score` and `CVaR` are ~0.50 (fraction of non-losing games). With lambda = 1.0 the max theoretical value is 2.0. Values around 1.0 mean roughly equal-to-random performance on both objectives simultaneously. Values above 1.05 indicate genuine improvement over random play on both.

### What the held-out metrics mean

- **`heldout_uniform_mean`** — average win rate across all 3 bands under a uniform opponent. The primary metric. Higher = better overall.
- **`heldout_worst_band`** — win rate in the single worst-performing band. A high value means no band is being crushed. Low = the repertoire has a skill-level blind spot.

### Closure ablation result interpretation

- **Closure significantly helps** → The chromosome's auto-coverage constraint is a genuine contribution. It forces the GA to cover more of the game tree, leading to more robust repertoires.
- **No significant difference** → The GA finds good coverage without being forced to. The representation is flexible enough that the closure rule is redundant. This is still a finding: simpler representations work equally well.
- **Closure hurts** → Unlikely, but would mean forced coverage wastes budget on positions that rarely arise in practice.

### COEVOLVE vs STATIC — what to expect

COEVOLVE should outperform STATIC on worst-band score because its evolving opponents actively seek out the weakest rating band. On average score, the difference may be small (both methods optimize the same objective). If COEVOLVE matches STATIC everywhere, the data supports an honest null result for the co-evolution contribution.

### Repertoire tree notation

```
1.e4
├─ 1...e5  →  2.Nf3
│   ├─ 2...Nc6  →  3.Bb5  (Ruy Lopez)
│   └─ 2...Nf6  →  3.d4   (Petrov)
└─ 1...c5  →  2.c3        (Alapin Sicilian)
```

- **Branches = opponent choices** covered by the closure rule
- **→ Move** = our committed response to that branch
- Lines without a `→` are leaves: the position is in `reached` but we have no committed move (the eval cache gives us a score directly)

---

## Notes

- **Checkpoint safety:** C1 is fully resumable. If the crawler is killed mid-run, restart it — it skips already-fetched positions.
- **Held-out evaluation always uses a uniform opponent mixture** regardless of mode, ensuring fair cross-mode comparison.
- **Training policies are reused for held-out evaluation** — held-out policies are not separately computed, because held-out data is sparse and training policies generalize well enough.
- **All scores are from White's perspective** in the eval cache. The fitness function converts Black's score with `1 - score` before combining.
- **Ablation runs produce standard pickle files** with the same schema as main runs, except the `mode` field contains the ablation method name (e.g. `STATIC_NOCLOSURE`) so `analyze.py` can route them to the right plot.
