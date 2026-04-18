# Co-Evolutionary Genetic Algorithm for Chess Opening Repertoires

**Full title:** Co-Evolutionary Genetic Algorithm for Memorization-Constrained Robust Chess Opening Repertoires

**Central claim:** Co-evolutionary adversarial training acts as adaptive regularization for memorization-constrained robust repertoire optimization. In plain language: instead of learning a chess opening against a fixed opponent, we evolve both the repertoire *and* the opponent simultaneously, which forces the repertoire to stay strong even against the worst possible rating-band distribution.

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [Key Concepts (No Chess Knowledge Required)](#2-key-concepts-no-chess-knowledge-required)
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

**The problem:** memorizing thousands of moves is impractical. This project answers the question:

> *Given a strict budget of only 20 committed moves per color, what opening repertoire is most robust across players of different skill levels?*

We frame this as a **genetic algorithm** (an evolutionary optimization technique inspired by natural selection). Candidate repertoires compete, reproduce, mutate, and survive based on a fitness score. The key twist is **co-evolution**: an adversarial population of "opponents" (represented as mixtures over rating bands) evolves *alongside* the repertoires, constantly probing for weaknesses. This prevents the repertoire from overfitting to any single opponent type.

Three modes are compared:

| Mode | Opponent | Evolves? |
|---|---|---|
| `STATIC` | Uniform mixture (equal weight to all bands) | No |
| `COEVOLVE_FROZEN` | 20 random mixtures drawn at start | No |
| `COEVOLVE` | 20 mixtures that evolve each generation | Yes |

---

## 2. Key Concepts (No Chess Knowledge Required)

### FEN — a chess position fingerprint
Every unique chess position has a compact text label called a **FEN** string, e.g.:
```
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -
```
This project uses only the first 4 fields (strips the move clock) so positions reached by different move orders but identical board states are treated as the same node.

### Rating Bands — opponent skill levels
Real games are split into three skill brackets:

| Band | Approximate Elo |
|---|---|
| `1600-1799` | Club players |
| `1800-1999` | Intermediate |
| `2000-2199` | Strong amateurs |

A repertoire that scores well against all three bands is considered **robust**.

### Closure — covering all common opponent replies
If White commits to `1.e4`, and 15% of opponents play `1...e5` while 12% play `1...c5` — both exceed the 5% threshold — then the repertoire **must** have prepared answers to both. This is the closure rule: any opponent move played >= 5% of the time must be covered.

### Budget — the memorization constraint
Each repertoire may commit to at most **20 decision nodes** per color (White and Black handled separately). A "committed node" is a position where it is our turn and we have chosen a specific move to play.

### CVaR — worst-case robustness
CVaR (Conditional Value at Risk) measures the expected score in the **worst** rating band. With 3 bands and alpha = 1/3, CVaR equals the lowest single-band score. The fitness function is:

```
fitness = mean_score + lambda x CVaR
```

With lambda = 1.0, equal weight is given to overall performance and worst-case robustness.

---

## 3. How the Algorithm Works

### Step-by-step overview

```
1. Initialize two populations:
   - R_pop: 30 candidate repertoires (chromosome = set of committed moves)
   - O_pop: opponent mixture population (chromosome = 3-number weight vector)

2. For each generation (default: 50):
   a. EVALUATE each repertoire against all opponents -> assign fitness
   b. REPRODUCE repertoires via tournament selection + crossover + mutation
   c. [COEVOLVE mode only] EVALUATE opponents -> how hard did they challenge R_pop?
   d. [COEVOLVE mode only] REPRODUCE opponents
   e. [COEVOLVE mode only] UPDATE Hall of Fame (5 most informative past opponents)
   f. LOG stats for this generation

3. After the loop:
   - Pick best repertoire from final generation
   - Evaluate it on held-out data (games unseen during training)
   - Return full result dictionary
```

### Fitness computation (the `walk` function)

To score a repertoire, we simulate a game tree:
- **Our turn + committed move:** follow that move, continue recursing
- **Our turn + no committed move (leaf):** return the pre-computed expected score from the eval cache
- **Opponent's turn:** take a weighted average over all opponent moves, weighted by the band policy (how likely that band's players are to play each move)

White score is direct; Black score is `1 - White_score` (flipped perspective).

### Mutation operators (four types)

| Operator | What it does |
|---|---|
| `mutate_move_swap` | Pick a committed node, swap its move to a different one |
| `mutate_extend` | Pick an uncommitted leaf, commit a move there |
| `mutate_prune` | Remove a committed node, making it a leaf again |
| `mutate_opening_replacement` | Replace an early committed node (ply <= 2) and rebuild below |

If a mutation would exceed the 20-node budget or violate closure, it raises `MutationFailed` and is retried (up to 5 attempts).

### Crossover

Two parent repertoires share a **pivot position** — a node where both have committed the same move. The child keeps Parent A's structure above the pivot and Parent B's subtree below it.

---

## 4. Project Structure

```
ChessRepertoire/
|-- BLUEPRINT.md              <- source of truth for all design decisions
|-- README.md                 <- this file
|-- requirements.txt          <- Python dependencies
|-- .env.example              <- template for Lichess API token
|-- .gitignore
|
|-- paper/
|   `-- draft.md              <- paper draft
|
|-- data/
|   |-- L2.db                 <- SQLite snapshot (C1 output, main database)
|   |-- graph_train.pkl       <- training position graph (C2)
|   |-- graph_heldout.pkl     <- held-out position graph (C2)
|   |-- base_policies.pkl     <- smoothed per-band policies (C3)
|   |-- eval_cache_train.pkl  <- shrunk expected scores, train (C4)
|   `-- eval_cache_heldout.pkl<- shrunk expected scores, held-out (C4)
|
|-- src/
|   |-- config.py             <- loads API token from .env
|   |-- data_ingest.py        <- C1: Lichess API crawler
|   |-- graph.py              <- C2: position graph builder
|   |-- policies.py           <- C3: per-band move distributions
|   |-- eval_cache.py         <- C4: Bayesian position scores
|   |-- repertoire.py         <- C5: chromosome + genetic operators
|   |-- fitness.py            <- C6: fitness function + walk
|   |-- opponent.py           <- C7: opponent chromosome
|   |-- coevolution.py        <- C8: main GA loop (all 3 modes)
|   |-- experiments.py        <- C9: full experimental runs
|   `-- analyze.py            <- C10: results analysis + figures
|
|-- runs/                     <- per-run pickles (gitignored)
|-- results/                  <- generated figures and tables (C10)
`-- tests/                    <- pytest suite (one file per component)
```

---

## 5. Data Flow

```
Lichess API
    |  (C1 -- data_ingest.py)
    v
data/L2.db  (SQLite)
    |  (C2 -- graph.py)
    v
graph_train.pkl + graph_heldout.pkl
    |  (C3 -- policies.py)          (C4 -- eval_cache.py)
    v                                      v
base_policies.pkl              eval_cache_train.pkl
                                eval_cache_heldout.pkl
    |                                      |
    `-------------------.-------------------'
                        |  (C5 -- repertoire.py + C6 -- fitness.py)
                        v
              Candidate (Repertoire pair)
              + fitness function
                        |  (C8 -- coevolution.py)
                        v
              Coordinator: STATIC / COEVOLVE_FROZEN / COEVOLVE
                        |
                        v
                  runs/*.pkl
                        |  (C10 -- analyze.py)
                        v
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
*Skip if `data/L2.db` already exists and is populated.*

```bash
python src/data_ingest.py
```

Crawls the Lichess Opening Explorer from the starting position, up to depth 10. Stores all positions and move statistics into `data/L2.db`. The script is **resumable** — if interrupted, restart it and it picks up where it left off.

Expected output: 10,000+ train positions, 5,000+ held-out positions.
Current database (`L2.db`): 910 positions per split (smaller dataset for faster iteration).

### Step 2 — Build position graphs (C2)

```bash
python src/graph.py data/L2.db
```

Outputs: `data/graph_train.pkl`, `data/graph_heldout.pkl`

### Step 3 — Compute base policies (C3)

```bash
python src/policies.py
```

Outputs: `data/base_policies.pkl`

Also prints a **band separation check** — the mean total-variation distance between rating band policies. Should be >= 0.05 for the CVaR story to be meaningful.

### Step 4 — Build eval cache (C4)

```bash
python src/eval_cache.py
```

Outputs: `data/eval_cache_train.pkl`, `data/eval_cache_heldout.pkl`

Prints `prior_mean` (~0.52, reflecting White's first-move advantage).

### Step 5 — Full experiments (C9)

```bash
python src/experiments.py
```

Runs 90 total experiments:
- **Main comparison:** 4 methods (`most_played_baseline`, `STATIC`, `COEVOLVE_FROZEN`, `COEVOLVE`) x 15 seeds (1000-1014) = 60 runs
- **Sensitivity:** 2 methods (`STATIC`, `COEVOLVE`) x 3 lambdas (0.0, 1.0, 2.0) x 5 seeds (2000-2004) = 30 runs

Each GA run: 50 generations, 30 repertoires. Resumable — skips runs whose output file already exists. Saves results to `runs/`.

### Step 6 — Analyze results (C10)

```bash
python src/analyze.py
```

Generates figures, tables, and statistics in `results/`.

### Run tests

```bash
pytest tests/
```

---

## 8. Component Reference

### C1 — `src/data_ingest.py`

Queries the Lichess Opening Explorer API recursively from the starting chess position. For each position and each rating band, it fetches aggregate statistics and per-band move counts.

**Key design decisions:**
- **Canonical FEN:** only the first 4 fields are kept (strips move clock). Transpositions collapse to the same node.
- **Resumable:** a `fetched_bands` table tracks which API calls completed. Restarting skips already-fetched positions.
- **Rate limit:** 1.1 seconds between calls to avoid throttling.
- **Enqueue threshold:** a child position is only queued if its play count exceeds a depth-dependent minimum (10k at depth <= 3, 30k at depth <= 6, 80k at depth <= 10) AND its aggregate frequency is >= 10%.

**Output schema (SQLite):**
```sql
positions(fen, split, ply_depth, total_games, white_wins, draws, black_wins, fetched_at)
moves(parent_fen, split, move_uci, move_san, child_fen, rating_band, play_count, ...)
```

---

### C2 — `src/graph.py`

Loads the SQLite snapshot into a Python dict. One graph per split (train / heldout).

**Node structure:**
```python
graph['nodes'][fen] = {
    'fen': str,
    'ply_depth': int,
    'total_games': int,
    'score_raw': float,      # (white_wins + 0.5*draws) / total_games
    'turn': 'white'|'black',
    'band_stats': {
        '1600-1799': {'games': int, 'wins': int, 'draws': int, 'losses': int},
        '1800-1999': { ... },
        '2000-2199': { ... },
    },
    'children': {
        '<move_uci>': {
            'child_fen': str,
            'move_san': str,       # human-readable (e.g. "e4")
            'aggregate_count': int,
            'band_counts': {'1600-1799': int, ...}
        }
    }
}
```

**Helper functions:** `build_graph()`, `save_graph()`, `load_graph()`, `children_of()`, `aggregate_move_freq()`

---

### C3 — `src/policies.py`

Computes how likely each rating band is to play each move at each position.

**Smoothing formula** (additive smoothing toward aggregate distribution):
```
policy[band][fen][move] = (band_count[move] + alpha x agg_prob[move]) / (band_total + alpha)
```
- `alpha = 5.0` — smoothing strength (from Section A of BLUEPRINT.md)
- When `band_total = 0` (no games from that band at this position): falls back to aggregate distribution
- When `band_total` is large: approaches the raw empirical band distribution

**Output:** `base_policies[band][fen][move_uci] = float` — sums to 1.0 across all moves at a position.

---

### C4 — `src/eval_cache.py`

Pre-computes a shrunk expected score for every (position, band) pair using Empirical Bayes.

**Why shrink?** Sparse positions (few games) have noisy raw scores. Shrinking toward a stable prior mean reduces variance.

**Formula:**
```
prior_mean = weighted average score across positions at ply <= 4
raw        = (wins + 0.5 x draws) / n
shrunk     = (n x raw + tau x prior_mean) / (n + tau)
```
- `tau = 20` — how strongly sparse positions are pulled toward the prior
- All scores are from White's perspective. For Black: use `1 - score`.
- `prior_mean ~= 0.52` reflects White's inherent first-move advantage in real games.

**Output:** `eval_cache['scores'][fen][band] = float`, plus `eval_cache['prior_mean']`

---

### C5 — `src/repertoire.py`

Defines the chromosome and all evolutionary operators.

**Chromosome:**
```python
Repertoire:
    color:     'white' | 'black'
    committed: {fen: move_uci}    # our decision nodes (max 20)
    reached:   set[fen]           # all positions in the subgraph
    graph:     dict               # reference to position graph

Candidate:
    white:            Repertoire
    black:            Repertoire
    fitness:          float | None
    band_scores_cache: dict | None   # invalidated after any structural change
```

**Construction:**
- `construct_initial()` — greedy: always picks the most-played move at each position
- `construct_random()` — stochastic: samples moves weighted by frequency

**Mutation operators** (each raises `MutationFailed` if constraints are violated):

| Function | Description |
|---|---|
| `mutate_move_swap` | Swap a committed move to a different one; rebuild subtree |
| `mutate_extend` | Commit at a leaf (increases budget usage by 1) |
| `mutate_prune` | Remove a committed node (frees budget by 1) |
| `mutate_opening_replacement` | Replace a shallow node (ply <= 2) and rebuild randomly |

**Subtree removal in a DAG:** When removing descendants below a node, positions reachable via an alternative path (transpositions) are preserved. This is checked via a BFS reachability test before deletion.

**`validate(rep)`** checks:
1. `len(committed) <= 20`
2. Every committed node's chosen move leads to a child in `reached`
3. Every opponent-turn node in `reached` has all moves with freq >= 5% also in `reached`

---

### C6 — `src/fitness.py`

**`walk(rep, band, eval_cache, base_policies, graph)`**

Recursively computes the expected White-perspective score for one repertoire under one band's move policy:

```
our turn  + committed move -> follow move, recurse into child
our turn  + no move (leaf) -> return eval_cache score for this position
opp turn                   -> weighted sum over all moves (policy weights)
                              off-book child -> use eval_cache score directly
```

**`evaluate(candidate, opponent_mixture, config, eval_cache, base_policies, graph)`**

```
band_scores[band] = 0.5 x walk(white_rep, band) + 0.5 x (1 - walk(black_rep, band))
mean_score        = sum(opponent_mixture[i] x band_scores[i])
CVaR              = min(band_scores.values())    # worst single band (alpha = 1/3)
fitness           = mean_score + lambda x CVaR
```

Returns: `{'mean_score', 'cvar', 'fitness', 'band_scores'}`

**`evaluate_heldout()`** — same logic but on held-out graph + eval cache, always uses uniform opponent mixture so cross-mode comparison is fair.

---

### C7 — `src/opponent.py`

The opponent chromosome is a 3-element probability vector (one weight per rating band) that must sum to 1.

```python
Opponent.uniform()        # [1/3, 1/3, 1/3]
Opponent.random(rng)      # Dirichlet(1,1,1) sample
opp.mutate(rng)           # blend with a new Dirichlet sample
opp.crossover(other, rng) # convex combination of two parent mixtures
```

In COEVOLVE mode, opponent fitness = `exploitation + novelty`:
- `exploitation = -mean(repertoire_scores)` — an opponent that consistently challenges the population is valuable
- `novelty = 0.1 x mean_distance_to_others` — encourages diverse coverage of the simplex, preventing all opponents from collapsing to the same point

---

### C8 — `src/coevolution.py`

The main loop. One call to `run_coevolution()` executes one complete run and returns a result dict.

**RNG discipline:** a single `np.random.default_rng(seed)` is created once at the top and passed to every function that needs randomness. No other random source is used anywhere. This guarantees **bit-identical reproducibility** given the same seed.

**Mode comparison:**

| Step | STATIC | COEVOLVE_FROZEN | COEVOLVE |
|---|---|---|---|
| O_pop size | 1 (uniform) | 20 (random, fixed forever) | 20 (random, evolves each gen) |
| Eval set | {uniform} | 20 random opponents | O_pop + up to 5 HoF members |
| Opponent reproduction | No | No | Yes (tournament + crossover + mutate) |
| Hall of Fame | No | No | Yes (size 5, tracks most informative opponents) |

**Return dict keys:** `mode`, `config`, `seed`, `git_commit`, `history`, `final_best_candidate`, `final_training_fitness`, `heldout_score`, `wall_time_seconds`

The `final_best_candidate` is serialized as:
```python
{
    'white_committed': {fen: move_uci, ...},
    'white_reached':   [fen, ...],
    'black_committed': {fen: move_uci, ...},
    'black_reached':   [fen, ...],
}
```

---

## 9. Fixed Parameters

| Parameter | Value |
|---|---|
| Max ply depth | 10 |
| Budget per color | 20 committed nodes |
| Closure threshold | 5% |
| Rating bands | 1600-1799, 1800-1999, 2000-2199 |
| Time controls | rapid, classical |
| Smoothing alpha | 5.0 |
| Bayes prior strength tau | 20 |
| CVaR fraction | 1/3 |
| Lambda (main runs) | 1.0 |
| Lambda (sensitivity) | 0.0, 1.0, 2.0 |
| Population size — repertoires | 30 |
| Population size — opponents | 20 (COEVOLVE/FROZEN), 1 (STATIC) |
| Generations | 50 |
| Tournament size | 3 |
| Crossover rate | 0.7 |
| Mutation rate | 0.3 |
| Hall of fame size | 5 |
| Seeds (main comparison) | 15 seeds: 1000-1014 |
| Seeds (sensitivity) | 5 seeds: 2000-2004 |
| Train/held-out cutoff | 2026-01-01 |

---

## 10. Interpreting Output

### Fitness > 1.0 — is that right?

Yes. Fitness = `mean_score + lambda x CVaR`. Both terms are ~0.50 (roughly the fraction of games not lost). With lambda = 1.0 the sum is ~1.0. A value above 1.0 means the repertoire achieves slightly better than 50% on both the mean and worst-case score simultaneously.

### Repertoire tree notation

```
1. e4           <- White commits to e4 (White's decision node)
1... e5         <- opponent reply (covered because freq >= 5%)
  2. Nf3        <- White's next committed move
  2... Nc6 *    <- * means leaf: White has no committed move here yet
```

- `*` = leaf node: position is in `reached` but has no committed move below it
- `[= transposition, already shown]` = same board position reached via a different move order; not printed twice

### What does CVaR tell you?

If one band (e.g. 1600-1799) consistently scores much lower than the others, the repertoire has a blind spot against that rating group. A high CVaR means the repertoire is genuinely robust — no band is being crushed.

### COEVOLVE vs STATIC — what to expect

COEVOLVE should outperform STATIC on training fitness because its evolving opponents actively discover weaknesses and force the repertoire to patch them. If it does *not*, the graph is likely too small or the population too homogeneous to produce diverse adversaries.

---

## Notes

- **Checkpoint safety:** C1 is fully resumable. If the API crawler is killed mid-run, just rerun it — it skips positions already fetched.
- **Database naming:** The main database is `data/L2.db`. All components from C2 onward are configured to use this path.
- **Held-out evaluation always uses a uniform opponent mixture** regardless of mode, ensuring fair cross-mode comparison.
- **Training policies are reused for held-out evaluation** — held-out policies are not separately computed, because held-out data is small and training policies generalize well enough for evaluation purposes.
- **All scores are from White's perspective** in the eval cache and walk function. The fitness function converts Black's score with `1 - score` before combining.
