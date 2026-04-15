# BLUEPRINT.md

Project: Co-Evolutionary Genetic Algorithm for Memorization-Constrained Robust Chess Opening Repertoires.

Central claim: co-evolutionary adversarial training acts as adaptive regularization for memorization-constrained robust repertoire optimization. A three-mode ablation isolates adversarial adaptation from generic population-based search.

This document is the source of truth for the implementation. Every Claude Code session should begin by reading it.

---

## Section A — Fixed parameters

| Parameter | Value |
|---|---|
| Max ply depth | 10 |
| Min games per position to keep | 100 |
| Rating bands | `1600-1799`, `1800-1999`, `2000-2199` |
| Time controls | rapid, classical |
| Train/held-out cutoff | 2026-01-01: games before are train, games from this date are held-out |
| Budget per color | 20 committed decision nodes |
| Closure threshold | opponent moves played ≥ 5% must be covered |
| Smoothing strength α (policies) | 5.0 |
| Empirical Bayes prior strength τ | 20 |
| CVaR fraction | 1/3 (with 3 bands, this is the worst single band) |
| λ (CVaR weight), main runs | 1.0 |
| Sensitivity λ values | 0.0, 1.0, 2.0 |
| Population size — repertoires | 30 |
| Population size — opponents | 20 for COEVOLVE and COEVOLVE_FROZEN; 1 for STATIC |
| Generations | 50 |
| Tournament size | 3 |
| Crossover rate (repertoires) | 0.7 |
| Mutation rate (repertoires) | 0.3 |
| Hall of fame size | 5 |
| Seeds — main comparison | 15 (seeds 1000–1014) |
| Seeds — sensitivity experiment | 5 (seeds 2000–2004) |

## Section B — Data flow

```
Lichess API → SQLite snapshot → graph_train.pkl + graph_heldout.pkl
                                     ↓
                            base_policies.pkl (from train graph)
                                     ↓
                            eval_cache_train.pkl + eval_cache_heldout.pkl
                                     ↓
                            Candidate (Repertoire pair) + fitness function
                                     ↓
                            Coordinator (3 modes) → runs/*.pkl
                                     ↓
                            Analysis → results/
```

## Section C — Repository layout

```
project/
├── BLUEPRINT.md
├── README.md
├── requirements.txt
├── .env.example                 # template for API token
├── .gitignore                   # must exclude .env
├── data/
│   ├── snapshot.db
│   ├── graph_train.pkl
│   ├── graph_heldout.pkl
│   ├── base_policies.pkl
│   ├── eval_cache_train.pkl
│   └── eval_cache_heldout.pkl
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_ingest.py           # C1
│   ├── graph.py                 # C2
│   ├── policies.py              # C3
│   ├── eval_cache.py            # C4
│   ├── repertoire.py            # C5
│   ├── fitness.py               # C6
│   ├── opponent.py              # C7
│   ├── coevolution.py           # C8
│   ├── experiments.py           # C9
│   └── analyze.py               # C10
├── tests/
│   ├── test_graph.py
│   ├── test_repertoire.py
│   ├── test_fitness.py
│   └── test_coevolution.py
├── runs/
├── results/
└── paper/
```

## Section D — API access

The Lichess Opening Explorer requires an API token for reliable access at this volume of requests. Create a Lichess account if you don't have one, generate a personal access token at `lichess.org/account/oauth/token`, and store it in a local `.env` file in the project root:

```
LICHESS_API_TOKEN = "lip_2zfqV0nEmw36wiimSvyf"
```

The `.env` file must be listed in `.gitignore` and never committed. A `.env.example` file should exist in the repo with the variable name but no value, so anyone setting up the project knows what to fill in.

In `src/config.py`, load the token from the environment using `python-dotenv`:

```python
from dotenv import load_dotenv
import os
load_dotenv()
LICHESS_API_TOKEN = os.environ["LICHESS_API_TOKEN"]
```

Every HTTP request to the Lichess API must include the header `Authorization: Bearer <token>`. Without the token, rate limits are stricter and the snapshot takes significantly longer.

Base URL: `https://explorer.lichess.ovh/lichess`.

## Section E — Components

Each component specifies: purpose, dependencies, inputs, outputs, and acceptance criteria. Build components in order (C1 → C10). Do not begin a component until the previous one passes its acceptance tests.

### C1 — Data ingest (`src/data_ingest.py`)

Purpose: recursively query the Lichess Opening Explorer API from the starting position, caching every response into a SQLite database. Must be resumable after interruption.

Depends on: nothing.

SQLite schema (the `split` field is part of the primary key on `positions` because a position can exist in both splits):

```sql
CREATE TABLE IF NOT EXISTS positions (
    fen TEXT NOT NULL,
    split TEXT NOT NULL CHECK(split IN ('train','heldout')),
    ply_depth INTEGER NOT NULL,
    total_games INTEGER NOT NULL,
    white_wins INTEGER NOT NULL,
    draws INTEGER NOT NULL,
    black_wins INTEGER NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (fen, split)
);

CREATE TABLE IF NOT EXISTS moves (
    parent_fen TEXT NOT NULL,
    split TEXT NOT NULL,
    move_uci TEXT NOT NULL,
    move_san TEXT NOT NULL,
    child_fen TEXT NOT NULL,
    rating_band TEXT NOT NULL,
    play_count INTEGER NOT NULL,
    white_wins INTEGER NOT NULL,
    draws INTEGER NOT NULL,
    black_wins INTEGER NOT NULL,
    PRIMARY KEY (parent_fen, split, move_uci, rating_band)
);

CREATE INDEX IF NOT EXISTS idx_moves_parent ON moves(parent_fen, split);
CREATE INDEX IF NOT EXISTS idx_positions_depth ON positions(ply_depth);
```

Canonical FEN: strip the halfmove clock and fullmove number. Keep only the first four space-separated fields (piece placement, active color, castling rights, en passant target). Two positions reached by different move orders but with identical first four FEN fields collapse to the same node.

API call format:
- Endpoint: `GET https://explorer.lichess.ovh/lichess`
- Headers: `Authorization: Bearer <LICHESS_API_TOKEN>`
- Query parameters per call:
  - `variant=standard`
  - `fen=<urlencoded canonical FEN>`
  - `speeds=rapid,classical`
  - `ratings=<single band>` — one of `1600`, `1800`, `2000` for band-specific queries, or the full list `1600,1800,2000` for the aggregate query
  - `since=2025-12` for the held-out split (games from 2026-01 onward)
  - `until=2025-12` for the train split (games before 2026-01)
- Rate limit: sleep 1.1 seconds between consecutive API calls. With a valid token, a single machine can sustain roughly one call per second without being throttled.

Algorithm (pseudocode):

```
init_db()
queue = [(starting_fen, depth=0)]
while queue is not empty:
    fen, depth = queue.pop()
    if position already exists in both splits with all 4 rating_band variants in moves:
        continue
    for split in ['train', 'heldout']:
        agg = call_api(fen, split, ratings='1600,1800,2000')
        insert_position(fen, split, depth, agg.white, agg.draws, agg.black)
        insert_moves(fen, split, 'aggregate', agg.moves)
        for band_tag in ['1600', '1800', '2000']:
            band_response = call_api(fen, split, ratings=band_tag)
            band_label = band_tag_to_label(band_tag)  # '1600-1799', '1800-1999', '2000-2199'
            insert_moves(fen, split, band_label, band_response.moves)
    for move in agg.moves:
        if move.play_count >= 100 and depth + 1 <= 10:
            enqueue((move.child_fen, depth + 1))
    log_progress_every_100_positions()
```

Outputs: `data/snapshot.db` populated with all positions up to depth 10 and all move statistics for both splits and all bands.

Acceptance criteria:
- Interrupting the script and restarting it picks up where it left off without making duplicate API calls.
- `SELECT COUNT(*) FROM positions WHERE split='train'` returns at least 10,000.
- `SELECT COUNT(*) FROM positions WHERE split='heldout'` returns at least 5,000.
- The starting position has `total_games` greater than 1,000,000 in both splits.
- A hand-selected deep position (e.g., the standard Italian Game position after `1.e4 e5 2.Nf3 Nc6 3.Bc4`) exists in the database in both splits.

### C2 — Position graph (`src/graph.py`)

Purpose: load the SQLite snapshot into an in-memory Python dict, one per split.

Depends on: C1.

Data structure (plain dict, easily pickleable):

```python
graph = {
    'root_fen': '<canonical starting position FEN>',
    'nodes': {
        '<fen>': {
            'fen': '<fen>',
            'ply_depth': int,
            'total_games': int,
            'score_raw': float,   # (white_wins + 0.5 * draws) / total_games
            'turn': 'white' | 'black',  # derived from FEN's active color field
            'band_stats': {
                '1600-1799': {'games': int, 'wins': int, 'draws': int, 'losses': int},
                '1800-1999': {...},
                '2000-2199': {...},
            },
            'children': {
                '<move_uci>': {
                    'child_fen': '<fen>',
                    'move_san': str,
                    'aggregate_count': int,
                    'band_counts': {
                        '1600-1799': int,
                        '1800-1999': int,
                        '2000-2199': int,
                    },
                },
                ...
            },
        },
        ...
    }
}
```

Functions:
- `build_graph(db_path: str, split: str) -> dict`: reads from SQLite and constructs the graph.
- `save_graph(graph: dict, path: str)`: pickles to disk.
- `load_graph(path: str) -> dict`: loads from disk.
- `children_of(graph: dict, fen: str) -> dict`: returns `graph['nodes'][fen]['children']`, empty dict if the position has no children.
- `aggregate_move_freq(graph: dict, fen: str) -> dict[str, float]`: returns `{move_uci: aggregate_count / sum_of_aggregate_counts}`.

Outputs: `data/graph_train.pkl` and `data/graph_heldout.pkl`.

Acceptance criteria:
- `root_fen` matches the standard starting position.
- For the starting position, the three highest-frequency moves according to `aggregate_move_freq` include `e2e4` and `d2d4`.
- For every node, every listed child's `child_fen` exists as a key in `nodes`, OR the child is a leaf position at max depth with no further children.
- Graphs for both splits are saved as separate pickle files.

### C3 — Rating-band base policies (`src/policies.py`)

Purpose: compute per-band move distributions from the training graph, with additive smoothing toward the aggregate distribution.

Depends on: C2.

Smoothing formula for a position `p`, band `b`, move `m`:

```
aggregate_total = sum of aggregate_count over all children of p
aggregate_prob[m] = children[m].aggregate_count / aggregate_total

band_total = sum of children[m'].band_counts[b] over all children m'
band_count[m] = children[m].band_counts[b]

policy[b][p][m] = (band_count[m] + α * aggregate_prob[m]) / (band_total + α)
```

with α = 5.0. When `band_total` is zero (no games from that band at that position), the formula collapses to `aggregate_prob[m]`, which is the correct fallback. When `band_total` is large, the formula approaches `band_count[m] / band_total`, which is the empirical band distribution.

Output data structure:

```python
base_policies = {
    '1600-1799': {
        '<fen>': {'<move_uci>': float, ...},   # sums to 1.0 across moves
        ...
    },
    '1800-1999': {...},
    '2000-2199': {...},
}
```

Outputs: `data/base_policies.pkl`.

Band separation check (must be run before proceeding): compute mean total-variation distance between pairs of band policies across all positions with `total_games >= 200`. The TV distance between distributions `p` and `q` over moves is `0.5 * sum(|p[m] - q[m]|)`. If the mean TV distance across the three pairs is below 0.05, the bands are too similar to support a meaningful CVaR robustness story — widen the bands or rebuild with different boundaries. Log the actual mean TV distance to stdout.

Acceptance criteria:
- Every policy distribution sums to 1.0 within 1e-9 tolerance.
- For a position with zero games in a specific band, that band's distribution equals the aggregate distribution.
- The band separation check passes (mean TV distance ≥ 0.05) and logs the actual value.

### C4 — Evaluation cache (`src/eval_cache.py`)

Purpose: precompute empirical-Bayes-shrunk expected scores for every position in each graph, for each band.

Depends on: C2.

Prior mean computation: compute once per split. The prior is the aggregate score across all positions at ply depth ≤ 4 in the graph, weighted by `total_games`:

```
numerator = sum over early positions of total_games[p] * score_raw[p]
denominator = sum over early positions of total_games[p]
prior_mean = numerator / denominator
```

This produces a stable "typical early-game expected score for White" that sparse positions are shrunk toward.

Shrinkage: for each position `p`, each band `b`:

```
stats = graph['nodes'][p]['band_stats'][b]
n = stats['games']
if n == 0:
    shrunk_score = prior_mean
else:
    raw = (stats['wins'] + 0.5 * stats['draws']) / n
    shrunk_score = (n * raw + τ * prior_mean) / (n + τ)

cache['scores'][fen][band] = shrunk_score
```

with τ = 20.

Perspective: all scores in the cache are from White's perspective (where `wins` means White won). The fitness function converts for Black as needed by computing `1 - shrunk_score`.

Output data structure:

```python
eval_cache = {
    'prior_mean': float,
    'scores': {
        '<fen>': {'1600-1799': float, '1800-1999': float, '2000-2199': float},
        ...
    }
}
```

Build caches for both the train and held-out graphs. Outputs: `data/eval_cache_train.pkl` and `data/eval_cache_heldout.pkl`.

Acceptance criteria:
- Every position in the graph has a cache entry for every band.
- A position with 0 games in a band has `shrunk_score == prior_mean` exactly.
- A position with 1000+ games in a band has `shrunk_score` within 0.02 of the raw empirical score.

### C5 — Repertoire and Candidate (`src/repertoire.py`)

Purpose: define the chromosome representation and all operators. This is the most delicate component and must pass extensive tests before downstream components depend on it.

Depends on: C2.

Classes:

```python
class Repertoire:
    color: str                             # 'white' or 'black'
    committed: dict[str, str]              # fen -> move_uci at our decision nodes
    reached: set[str]                      # all positions in the subgraph
    graph: dict                            # reference to the position graph

class Candidate:
    white: Repertoire
    black: Repertoire
    fitness: Optional[float] = None        # cached by fitness function
    band_scores_cache: Optional[dict] = None  # cached for shared sampling
```

Key definitions:
- A position `p` in `reached` is a *our-turn position* for a repertoire if `graph['nodes'][p]['turn'] == repertoire.color`.
- A our-turn position is a *committed node* if `p in committed`.
- A our-turn position in `reached` but not in `committed` is a *leaf*.
- An opponent-turn position in `reached` must have every child `m` with aggregate frequency ≥ 0.05 also present in `reached` (the closure rule).

Roots:
- The White repertoire has its root at the standard starting position (where it's White's turn), and must have a committed move there.
- The Black repertoire's `reached` also starts at the starting position, but the starting position is not a committed node (it's White's turn). Black's first committed nodes are at the positions after each frequent White first move.

Constants: budget per repertoire is 20 committed nodes; closure threshold is 0.05.

Functions:

```python
def construct_initial(graph, color, budget, rng) -> Repertoire:
    """Greedy construction: at each our-turn node, pick the move with highest
    aggregate frequency. Expand all opponent replies with aggregate frequency
    >= 0.05. Stop expanding when budget is reached."""

def construct_random(graph, color, budget, rng) -> Repertoire:
    """Like construct_initial, but at each our-turn node, pick the move via
    weighted random choice using aggregate frequencies as weights."""

def validate(rep) -> bool:
    """Check that the repertoire satisfies the closure rule, is within budget,
    and is internally consistent (every reached position's required children
    are also in reached)."""

def mutate_move_swap(rep, rng) -> Repertoire:
    """Pick a random committed node, change its committed move to a different
    move weighted by aggregate frequency. Remove all descendants below the old
    move from reached and committed. Add descendants below the new move,
    expanding to closure. Raises MutationFailed if the result violates budget."""

def mutate_extend(rep, rng) -> Repertoire:
    """Pick a random leaf (our-turn position in reached but not in committed).
    Commit a move there (weighted random choice). Expand to closure.
    Raises MutationFailed if no budget headroom or no leaves exist."""

def mutate_prune(rep, rng) -> Repertoire:
    """Pick a random committed node other than the root. Remove its committed
    move and all descendants from reached and committed. The position becomes
    a leaf. Raises MutationFailed if pruning would violate closure at the
    parent opponent node."""

def mutate_opening_replacement(rep, rng) -> Repertoire:
    """Pick a random committed node at ply_depth <= 2. Replace its committed
    move and rebuild everything below from scratch via construct_random."""

def crossover(rep_a, rep_b, rng) -> Repertoire:
    """Find the set of positions committed in both parents with the same
    committed move. If non-empty, pick one at random. Keep rep_a's structure
    from root to (and including) that position, and rep_b's subtree from that
    position onward. If the resulting repertoire exceeds budget, raise
    MutationFailed. If no shared committed moves exist, return a clone of
    rep_a."""

def mutate_candidate(candidate, rng) -> Candidate:
    """Randomly choose white or black (50/50). Randomly choose one of the four
    mutation operators (uniform). Apply it. If it raises MutationFailed, retry
    with a different operator choice, up to 5 attempts. If all 5 fail, return
    an unchanged clone."""

def crossover_candidates(cand_a, cand_b, rng) -> Candidate:
    """Crossover white parts independently from black parts. Return a new
    Candidate with the resulting pair."""
```

Subtree removal in a DAG: when removing "all descendants below a position `p`," do a breadth-first walk from `p` following only edges whose children are in `reached`, collecting descendant FENs. Before removing each collected FEN from `reached`, check whether it is also reachable from the root via a path that does not pass through `p`. If so, leave it in `reached`. This is a rare edge case at depth 10 but must be handled.

Acceptance criteria:
- `construct_initial` with 100 different seeds produces 100 valid repertoires, all passing `validate`.
- Fuzz test: apply 1000 random mutation attempts to 20 random initial repertoires. Every result either passes `validate` or the operator raised `MutationFailed`. No silently invalid outputs.
- `crossover` applied to two valid repertoires returns either a valid repertoire or a clone of one parent.
- Determinism: the same inputs and same RNG state produce bit-identical outputs across repeated calls.

### C6 — Fitness function (`src/fitness.py`)

Purpose: compute the fitness of a Candidate given an opponent mixture.

Depends on: C2, C3, C4, C5.

```python
def evaluate(
    candidate: Candidate,
    opponent_mixture: np.ndarray,   # length 3, sums to 1
    config: dict,                    # must contain 'lambda_weight' and 'alpha'
    eval_cache: dict,
    base_policies: dict,
    graph: dict,
    use_cache: bool = True,
) -> dict:
    """Returns {'mean_score', 'cvar', 'fitness', 'band_scores'}."""
```

Algorithm:

```
bands = ['1600-1799', '1800-1999', '2000-2199']

# Budget check (per color)
if len(candidate.white.committed) > 20 or len(candidate.black.committed) > 20:
    return {'mean_score': 0.0, 'cvar': 0.0, 'fitness': -float('inf'), 'band_scores': {}}

# Compute per-band scores (cached on candidate for shared sampling)
if use_cache and candidate.band_scores_cache is not None:
    band_scores = candidate.band_scores_cache
else:
    band_scores = {}
    for band in bands:
        white_ws = walk(candidate.white, band, eval_cache, base_policies, graph)
        black_ws = walk(candidate.black, band, eval_cache, base_policies, graph)
        # White repertoire's walk returns White-perspective score directly
        # Black repertoire's walk returns White-perspective score; convert for Black
        black_score_for_player = 1.0 - black_ws
        band_scores[band] = 0.5 * white_ws + 0.5 * black_score_for_player
    if use_cache:
        candidate.band_scores_cache = band_scores

# Mean weighted by opponent mixture
mean_score = sum(opponent_mixture[i] * band_scores[bands[i]] for i in range(3))

# CVaR: with 3 bands and alpha = 1/3, this is the worst single band
sorted_band_scores = sorted(band_scores.values())
cvar = sorted_band_scores[0]

fitness = mean_score + config['lambda_weight'] * cvar
return {
    'mean_score': mean_score,
    'cvar': cvar,
    'fitness': fitness,
    'band_scores': band_scores,
}
```

Walk function (computes expected White-perspective score for a repertoire under one band's policy):

```python
def walk(rep, band, eval_cache, base_policies, graph, node_fen=None):
    if node_fen is None:
        node_fen = graph['root_fen']
    node = graph['nodes'][node_fen]
    is_our_turn = (node['turn'] == rep.color)

    if is_our_turn:
        if node_fen not in rep.committed:
            # Leaf: our turn, no committed move
            return eval_cache['scores'][node_fen][band]
        move = rep.committed[node_fen]
        child_fen = node['children'][move]['child_fen']
        return walk(rep, band, eval_cache, base_policies, graph, child_fen)
    else:
        # Opponent turn: weighted sum over all opponent moves in the graph
        total = 0.0
        policy_at_node = base_policies[band].get(node_fen, {})
        for move, child_info in node['children'].items():
            p = policy_at_node.get(move, 0.0)
            if p == 0.0:
                continue
            child_fen = child_info['child_fen']
            if child_fen in rep.reached:
                total += p * walk(rep, band, eval_cache, base_policies, graph, child_fen)
            else:
                # Off-book: opponent played a move the repertoire doesn't cover
                total += p * eval_cache['scores'][child_fen][band]
        return total
```

Cache invalidation: after any mutation or crossover that produces a new Candidate, set `candidate.band_scores_cache = None`. The cache is valid only for a specific Candidate state; any structural change invalidates it. The coordinator is responsible for clearing the cache when producing new candidates.

Held-out evaluation: there is a separate function:

```python
def evaluate_heldout(candidate, eval_cache_heldout, base_policies_train, graph_heldout, config) -> float:
    """Returns the held-out fitness under the uniform opponent mixture. Uses
    the training base policies (since held-out policies aren't built) and the
    held-out eval cache and graph. For positions in the candidate that don't
    exist in graph_heldout, treat as leaves and use the off-book baseline
    from eval_cache_heldout if available, otherwise fall back to the
    held-out prior_mean."""
```

Acceptance criteria:
- Hand-verified toy test: construct a repertoire of known structure with known band scores; the walk produces the expected number.
- For a valid in-budget candidate, fitness is a finite number.
- For an over-budget candidate, fitness is `-inf`.
- Calling `evaluate` twice on the same candidate (with caching) returns bit-identical results.
- Clearing the cache and calling `evaluate` again returns the same result as the cached call.

### C7 — Opponent (`src/opponent.py`)

Purpose: chromosome type for the opponent population. Each opponent is a point on the simplex over the three rating bands.

Depends on: nothing beyond numpy.

```python
import numpy as np
from typing import Optional

class Opponent:
    def __init__(self, mixture: np.ndarray):
        assert mixture.shape == (3,)
        assert abs(mixture.sum() - 1.0) < 1e-9
        assert (mixture >= 0).all()
        self.mixture = mixture
        self.fitness: Optional[float] = None

    @classmethod
    def uniform(cls) -> 'Opponent':
        return cls(np.ones(3) / 3.0)

    @classmethod
    def random(cls, rng) -> 'Opponent':
        return cls(rng.dirichlet(np.ones(3)))

    def mutate(self, rng, strength: float = 0.3) -> 'Opponent':
        noise = rng.dirichlet(np.ones(3))
        new_mix = (1.0 - strength) * self.mixture + strength * noise
        new_mix = new_mix / new_mix.sum()
        return Opponent(new_mix)

    def crossover(self, other: 'Opponent', rng) -> 'Opponent':
        w = rng.uniform(0.0, 1.0)
        new_mix = w * self.mixture + (1.0 - w) * other.mixture
        new_mix = new_mix / new_mix.sum()
        return Opponent(new_mix)
```

Acceptance criteria: every Opponent instance (constructed by any method) has a mixture that is non-negative and sums to 1 within 1e-9.

### C8 — Co-evolution coordinator (`src/coevolution.py`)

Purpose: the main GA loop supporting all three modes via a single flag. A single call executes one complete run and returns a result dictionary.

Depends on: C5, C6, C7.

```python
def run_coevolution(
    mode: str,                           # 'STATIC' | 'COEVOLVE_FROZEN' | 'COEVOLVE'
    config: dict,                        # contains lambda_weight, alpha, pop sizes, etc.
    seed: int,
    graph_train: dict,
    graph_heldout: dict,
    base_policies_train: dict,
    eval_cache_train: dict,
    eval_cache_heldout: dict,
) -> dict:
    ...
```

RNG discipline: call `rng = np.random.default_rng(seed)` exactly once at the top of the function. Pass `rng` as an argument to every function that needs randomness. No other random number source may be used anywhere in this function's call tree.

Git commit recording: at the top of the function, call `subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()` and store in the result dict as `git_commit`.

Initialization:
- Repertoire population: create 30 Candidate objects. Half are constructed via `construct_initial` (greedy most-played-move) for both colors, the other half via `construct_random`. Each uses a fresh derivation from the main `rng`.
- Opponent population:
  - `STATIC`: `O_pop = [Opponent.uniform()]`, length 1.
  - `COEVOLVE_FROZEN`: `O_pop = [Opponent.random(rng) for _ in range(20)]`, length 20, never modified after this.
  - `COEVOLVE`: `O_pop = [Opponent.random(rng) for _ in range(20)]`, length 20, evolves each generation.
- Hall of fame: `HoF = []`, used only in `COEVOLVE` mode.
- History log: `history = []`.

Generational loop (for generation in range(50)):

Step 1 — Build the evaluation set.
- `STATIC`, `COEVOLVE_FROZEN`: `eval_set = O_pop`.
- `COEVOLVE`: `eval_set = O_pop + rng.choice(HoF, size=min(5, len(HoF)), replace=False)`.

Step 2 — Evaluate every repertoire against every opponent in the evaluation set (shared sampling). Clear each candidate's `band_scores_cache` before the first evaluation in the generation (since candidates from the previous generation's crossover/mutation need fresh computation). For each candidate, compute `fitness_i = evaluate(candidate, opp.mixture, ...)['fitness']` for each opponent in the eval set, and set `candidate.fitness = mean(fitness_i)`. Note that the `band_scores_cache` is reused across the multiple opponents within this step, since mixtures vary but band walks don't.

Step 3 — Repertoire selection and reproduction. Apply tournament selection (size 3) to produce 30 parents. For each of the 30 next-generation slots, with probability 0.7 apply `crossover_candidates` to two parents; otherwise clone a single parent. With probability 0.3 apply `mutate_candidate` to the result. Replace `R_pop` with the new generation. Clear `band_scores_cache` on every new candidate.

Step 4 (COEVOLVE only) — Opponent evaluation. For each opponent in `O_pop`, compute `rep_scores = [evaluate(rep, opp.mixture, ...)['mean_score'] for rep in R_pop]`. Compute `exploitation = -mean(rep_scores)` and `novelty = 0.1 * mean(np.linalg.norm(opp.mixture - other.mixture) for other in O_pop if other is not opp)`. Set `opp.fitness = exploitation + novelty`.

Step 5 (COEVOLVE only) — Opponent reproduction. Apply tournament selection (size 3) to produce 20 parents. For each next-generation slot, with probability 0.5 apply `crossover` to two parent Opponents; otherwise clone one parent. With probability 0.5 apply `mutate` to the result. Replace `O_pop`.

Step 6 (COEVOLVE only) — Hall of fame update. For each opponent in the new `O_pop`, compute its *informativeness*: the variance of `evaluate(rep, opp.mixture, ...)['fitness']` across all repertoires in `R_pop`. If `HoF` has fewer than 5 members, add this opponent directly. Otherwise, if the opponent's informativeness exceeds the minimum informativeness in `HoF`, replace that minimum member with this opponent.

Step 7 — Log generation statistics:

```python
history.append({
    'gen': gen,
    'best_training_fitness': max(c.fitness for c in R_pop),
    'mean_training_fitness': mean(c.fitness for c in R_pop),
    'repertoire_diversity': mean_pairwise_distance(R_pop),
    'opponent_diversity': (
        mean_pairwise_opponent_distance(O_pop) if mode == 'COEVOLVE' else None
    ),
    'hof_size': len(HoF) if mode == 'COEVOLVE' else 0,
})
```

Where `mean_pairwise_distance` over repertoires is approximated as the mean symmetric set difference between `candidate.white.committed` across all pairs (divided by union size).

After the loop — Held-out evaluation. Pick the candidate from the final `R_pop` with the highest `fitness`. Compute its held-out score via `evaluate_heldout(best, eval_cache_heldout, base_policies_train, graph_heldout, config)`. Use the uniform opponent mixture for this evaluation regardless of mode, so the comparison across modes is fair.

Return dictionary:

```python
{
    'mode': mode,
    'config': config,
    'seed': seed,
    'git_commit': git_commit,
    'history': history,
    'final_best_candidate': serialize_candidate(best),
    'final_training_fitness': best.fitness,
    'heldout_score': heldout_score,
    'wall_time_seconds': elapsed,
}
```

where `serialize_candidate` converts the Candidate to a pickle-safe dict: `{'white_committed': dict, 'white_reached': list, 'black_committed': dict, 'black_reached': list}`.

Acceptance criteria:
- All three modes run end-to-end on a pilot configuration (population size 10, 5 generations) without errors.
- Running the function twice with the same mode, config, and seed produces bit-identical result dictionaries (excluding the `wall_time_seconds` field and `git_commit` if the commit hasn't changed, both of which should match).
- Switching modes requires only changing the `mode` string argument — no other code changes needed.

### C9 — Experiment runner (`src/experiments.py`)

Purpose: define the full run matrix and execute all runs, saving one pickle file per run.

Depends on: C8.

Run matrix:

```python
MAIN_EXPERIMENTS = [
    {'method': method, 'seed': seed, 'lambda_weight': 1.0, 'alpha': 1/3}
    for method in ['most_played_baseline', 'STATIC', 'COEVOLVE_FROZEN', 'COEVOLVE']
    for seed in range(1000, 1015)
]   # 4 methods × 15 seeds = 60 runs

SENSITIVITY_EXPERIMENTS = [
    {'method': method, 'seed': seed, 'lambda_weight': lam, 'alpha': 1/3}
    for method in ['STATIC', 'COEVOLVE']
    for lam in [0.0, 1.0, 2.0]
    for seed in range(2000, 2005)
]   # 2 methods × 3 lambdas × 5 seeds = 30 runs

ALL_EXPERIMENTS = MAIN_EXPERIMENTS + SENSITIVITY_EXPERIMENTS   # 90 runs total
```

Execution:
- Before starting, check that `git status --porcelain` returns empty. If not, print an error and exit.
- Load the training and held-out graphs, policies, and eval caches once at startup (do not reload per run).
- For each run in `ALL_EXPERIMENTS`:
  - Compute the output filename: `runs/{method}_l{lambda_weight}_s{seed}.pkl`.
  - If the file already exists, skip (supports resume).
  - If `method == 'most_played_baseline'`: construct one Candidate via `construct_initial` for both colors, evaluate on held-out, save the result dict without running the GA.
  - Otherwise: call `run_coevolution(method, config, seed, ...)`.
  - Pickle the result dict to the output filename.
- Log progress after each run completes.

Acceptance criteria:
- A clean run produces all 90 pickle files in `runs/`.
- Interrupting and restarting the runner does not redo completed runs.
- The runner refuses to start if there are uncommitted git changes.

### C10 — Analysis (`src/analyze.py`)

Purpose: load all run pickles, compute statistics, and produce tables and plots for the paper.

Depends on: C9 outputs.

Outputs:
- `results/main_table.csv`: one row per method. Columns: `method`, `mean_heldout`, `std_heldout`, `median_heldout`, `wilcoxon_p_vs_STATIC` (Holm-corrected across the three non-STATIC methods), `A12_vs_STATIC`.
- `results/convergence.png`: line plot, one line per method, x-axis is generation, y-axis is best training fitness, with a 95% confidence band computed across seeds for each method.
- `results/sensitivity_table.csv`: one row per (method, lambda). Columns: `method`, `lambda_weight`, `mean_heldout`, `std_heldout`.
- `results/diagnostic_table.csv`: for COEVOLVE runs only, per-generation averages across seeds for `repertoire_diversity`, `opponent_diversity`, `hof_size`.

Statistical tests:
- Use `scipy.stats.wilcoxon` with `zero_method='wilcox'` for the paired signed-rank test. Pair by seed (methods must share seeds in the main experiment — they do, since all use seeds 1000–1014).
- Holm correction: sort p-values ascending, multiply the smallest by `k`, the next by `k-1`, etc., and clip at 1.0. With three comparisons against STATIC, `k = 3`.
- A12 effect size: `A12(X, Y) = (|{(x, y) : x > y}| + 0.5 * |{(x, y) : x == y}|) / (|X| * |Y|)`, computed across the paired held-out scores.

Acceptance criteria: re-running `analyze.py` produces bit-identical output files. Every number in the paper should be traceable to a specific cell in one of these output files.

## Section F — Using this blueprint with Claude Code

### One-time setup

1. Create the repository and commit `BLUEPRINT.md` and an empty `README.md`.
2. Create the folder skeleton from Section C, including empty `__init__.py` files and an empty `requirements.txt` containing at minimum: `python-chess`, `numpy`, `pandas`, `scipy`, `matplotlib`, `requests`, `python-dotenv`, `pytest`.
3. Create `.env.example` with the line `LICHESS_API_TOKEN=`. Create `.gitignore` containing `.env`, `__pycache__/`, `*.pkl`, `runs/`, `results/`, `data/*.db`, and `data/*.pkl`.
4. Create a local `.env` file (not committed) with a real token value obtained from Lichess.
5. Commit the setup as the initial commit.

### Per-component workflow

For each component C1 through C10, in order:

Start a fresh Claude Code session. Do not reuse a session from the previous component.

Use this prompt template, filling in the component identifier:

> Please read `BLUEPRINT.md` in the repo root. I am implementing Component [C1/C2/.../C10]: [name]. Please:
>
> 1. Read Section E's full specification for this component.
> 2. Read Section A (fixed parameters) for any constants you need.
> 3. Read Section D if the component uses the Lichess API.
> 4. Implement the component in `src/[filename].py` exactly as specified. Do not deviate from the spec. If the spec is ambiguous at any point, stop and ask me before proceeding.
> 5. Write unit tests in `tests/test_[filename].py` that verify every acceptance criterion listed in Section E for this component. Each acceptance criterion should correspond to a test function.
> 6. Run the tests and show me the output. Do not modify any file other than `src/[filename].py` and `tests/test_[filename].py`.
>
> When done, summarize: what you implemented, what each test verifies, anything in the spec that was unclear, and any deviation from the spec with justification.

After Claude Code finishes, read the generated code yourself before trusting the tests. Run the tests manually in your terminal with `pytest tests/test_<component>.py -v`. If everything passes, commit: `git add src/<component>.py tests/test_<component>.py && git commit -m "C<N>: <name>"`.

If tests fail or the code looks wrong, use this debugging prompt in the same or a new session:

> Running `[command]` produces `[paste the output]`. According to Section E of BLUEPRINT.md, I expected `[expected behavior]`. Please investigate without modifying any code yet. Explain what is happening, why it deviates from the spec, and propose a minimal fix. Wait for my approval before making changes.

### Integration pilot (after all components implemented)

Use this prompt in a fresh session:

> All ten components are implemented and committed. I need to run an end-to-end pilot to verify the pipeline. Please:
>
> 1. Create a pilot config with population size 10 and 5 generations.
> 2. Run `run_coevolution` in each of the three modes (STATIC, COEVOLVE_FROZEN, COEVOLVE) with seed 42.
> 3. Run each mode a second time with seed 42 and verify the output is bit-identical to the first run (excluding wall time). This is the reproducibility gate.
> 4. Report the final fitness, the wall-clock time, and whether reproducibility passed for each mode.
> 5. If anything fails, stop and explain what went wrong.

### Hard rules

- Never let Claude Code edit `BLUEPRINT.md`. If the blueprint seems wrong, edit it yourself, commit the change, then continue.
- Commit after every working component.
- Never skip the acceptance tests. Debug first, then move on.
- Never implement multiple components in one session.
- The reproducibility test goes after C8 completes, not at the end. If seeding is broken, everything downstream is invalid.

### Pushing back on spec deviation

If Claude Code produces code that differs from the spec, respond directly:

> The spec in Section E of BLUEPRINT.md specifies `[quote the exact line]`. Your implementation does `[describe]`. Please either align with the spec, or explain why the spec should be changed and propose a specific amendment for me to review and commit to the blueprint.

The blueprint is the source of truth. If it's genuinely wrong, fix the blueprint first, commit the fix, then update the code.

## Section G — Paper claim

> We formalize amateur chess opening preparation as a constrained robust optimization problem over closed subgraphs of the game tree, with a distribution-over-types fitness function combining expected score against rating-band empirical opponent policies with a CVaR tail-risk term. We investigate whether co-evolutionary adversarial training offers advantages over static robust optimization for this problem class. A three-mode ablation (static baseline, co-evolution with adaptive adversary, co-evolution with frozen adversary) isolates adversarial adaptation from generic population-based search. On Lichess amateur data with temporal held-out evaluation, we find [to be filled in based on experimental results]. This has implications for committed-policy optimization problems under distributional uncertainty more broadly.

Three possible framings for the bracketed clause, depending on experimental outcomes:

*Mechanism confirmed*: "co-evolutionary adversarial training produces repertoires with better held-out generalization than static robust optimization, and this advantage is attributable specifically to adversarial adaptation rather than to population diversity alone."

*Characterization*: "co-evolutionary methods and well-tuned static robust optimization achieve comparable held-out performance, suggesting that the diversity introduced by an opponent population is the key ingredient rather than the adaptation itself."

*Negative*: "well-tuned static robust optimization is sufficient for this problem class, and co-evolutionary adaptation does not provide additional benefit in our experimental setup."

All three framings are publishable. The blueprint is designed so the paper can truthfully tell whichever story the data supports.