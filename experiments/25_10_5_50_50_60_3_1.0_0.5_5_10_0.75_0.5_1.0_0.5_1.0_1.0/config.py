"""
config.py — Central configuration for the ChessRepertoire project.

Every tunable constant lives here, organized by component.
Change a value here and it takes effect everywhere automatically.

SECTIONS:
  0. SHARED CONSTANTS      — starting position, API URL
  1. DATA COLLECTION       — how the Lichess crawler behaves
  2. SCORING               — how scores are computed
  3. REPERTOIRE CHROMOSOME — move budget and closure rules
  4. GENETIC ALGORITHM     — population, selection, crossover, mutation
  5. OPPONENT EVOLUTION    — opponent chromosome behavior
  6. FITNESS FUNCTION      — how fitness is computed from scores
  7. EXPERIMENT MATRIX     — which runs to execute and with what parameters
  8. ANALYSIS              — held-out adversarial evaluation parameters
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. SHARED CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Standard chess starting position (4-field canonical FEN).
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"

# Lichess Opening Explorer API base URL.
LICHESS_BASE_URL = "https://explorer.lichess.ovh/lichess"

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA COLLECTION  (src/data_ingest.py)
# ─────────────────────────────────────────────────────────────────────────────

# Seconds to wait between consecutive Lichess API calls.
# Lichess allows ~1 req/sec; do not go below 1.0 or you risk being rate-limited.
RATE_LIMIT_SLEEP = 1.1

# Maximum ply depth to crawl. 1 ply = 1 half-move (White OR Black plays).
# Depth 10 means 5 full moves into the opening. More depth = bigger database.
MAX_PLY_DEPTH = 10

# A child position is only enqueued if its move was played in at least this
# fraction of aggregate games at the parent.
# 0.10 = only follow moves played in >=10% of games. Lower = more positions.
MIN_MOVE_FREQUENCY = 0.05

# Minimum number of aggregate games a child position must have to be enqueued,
# depending on depth. Shallower positions need fewer games (they always have
# plenty); deeper positions are rare so we require more to trust the data.
#   depth <= 3:  10,000 games
#   depth <= 6:  30,000 games
#   depth  > 6:  80,000 games
MIN_GAMES_SHALLOW = 5_000   # ply depth 1–3
MIN_GAMES_MID     = 20_000   # ply depth 4–6
MIN_GAMES_DEEP    = 50_000   # ply depth 7+
MIN_GAMES_MID_CUTOFF    = 3  # depths <= this use MIN_GAMES_SHALLOW
MIN_GAMES_DEEP_CUTOFF   = 6  # depths <= this use MIN_GAMES_MID

# The date that separates training data from held-out data.
# Games before this date go into the train split; games on/after go to heldout.
# Format: "YYYY-MM"
TRAIN_UNTIL   = "2025-06"
HELDOUT_SINCE = "2025-06"

# Game time controls to include from the Lichess database.
# Options: "bullet", "blitz", "rapid", "classical", "correspondence"
# Faster time controls have more games but noisier play quality.
API_SPEEDS = "rapid,classical"

# ─────────────────────────────────────────────────────────────────────────────
# 2. SCORING  (src/eval_cache.py)
# ─────────────────────────────────────────────────────────────────────────────

# Which ply depths to use when computing the prior mean.
# Only positions at ply_depth <= this value are used. Shallow positions have
# the most games so they give the most reliable prior estimate.
PRIOR_MAX_PLY = 10

# ─────────────────────────────────────────────────────────────────────────────
# 3. REPERTOIRE CHROMOSOME  (src/repertoire.py)
# ─────────────────────────────────────────────────────────────────────────────

# Maximum number of "committed" moves per color. This is the memorization
# budget constraint — a player can only reliably memorize this many specific
# positions.
# Increase = bigger/more complete repertoire, harder search problem.
# Decrease = simpler repertoire, easier to optimize.
BUDGET = 25

# Closure threshold: an opponent reply must be covered in the repertoire if
# it appears in >= this fraction of games at a given position.
# 0.05 = cover any move played in >=5% of games. Raise to reduce the size of
# forced coverage (fewer required moves); lower to be more comprehensive.
CLOSURE_THRESHOLD = 0.15

# How many times to retry a mutation before giving up and returning a clone.
# More retries = more likely to succeed but slower per generation.
MUTATION_RETRIES = 5

# Maximum ply depth at which a committed node is eligible for opening replacement.
# Nodes at ply_depth <= this value can have their subtree replaced wholesale.
OPENING_REPLACEMENT_MAX_PLY = 5

# Whether to apply the closure rule during construction and mutation.
# True  = standard mode: auto-cover all opponent replies >= CLOSURE_THRESHOLD.
# False = no-closure ablation: only the direct child of each committed move is added.
# Individual experiments can override this via their config dict.
USE_CLOSURE = True

# ─────────────────────────────────────────────────────────────────────────────
# 4. GENETIC ALGORITHM  (src/coevolution.py)
# ─────────────────────────────────────────────────────────────────────────────

# Number of repertoire candidates in the population.
# More = better coverage of search space, slower per generation.
POP_SIZE_REPERTOIRES = 50

# Number of opponent individuals in the population (COEVOLVE mode only).
POP_SIZE_OPPONENTS = 50

# Number of generations to run the GA.
# More = more evolution time. Total evaluations = POP_SIZE_REPERTOIRES × N_GENERATIONS.
N_GENERATIONS = 60

# Tournament selection size. A random subset of this many individuals is drawn,
# and the fittest wins. Larger = stronger selection pressure (good individuals
# dominate faster but diversity drops). Smaller = weaker selection (slower
# but more diversity preserved). Must be <= population size.
TOURNAMENT_SIZE = 3

# Probability that two selected parents undergo crossover to produce a child.
# If crossover does not trigger, the child is a clone of parent A.
# 0.0 = no crossover (pure mutation), 1.0 = always crossover.
CROSSOVER_RATE = 1.0

# Probability that a mutation is applied to a child after crossover/cloning.
# 0.0 = no mutation, 1.0 = always mutate.
MUTATION_RATE = 0.5

# Hall of Fame size (COEVOLVE mode only).
# Keeps the N most informative past opponents seen during evolution, so
# they can continue challenging the current population even after the live
# opponent population has moved on. More = broader historical pressure.
HOF_SIZE = 10

# Fraction of the initial repertoire population built with the greedy
# (construct_initial) strategy vs randomly (construct_random).
# Must sum to 1.0.  Adjust to bias initialisation toward principled or
# exploratory starting points.
INIT_GREEDY_FRACTION = 0.0
INIT_RANDOM_FRACTION = 1.0

# Mean pairwise Jaccard distance below which the repertoire population is
# considered collapsed.  When triggered, the lowest-fitness REINIT_FRACTION
# of the population is replaced with fresh random repertoires.
REPERTOIRE_DIVERSITY_THRESHOLD = 0.25
REPERTOIRE_REINIT_FRACTION     = 0.3

# ─────────────────────────────────────────────────────────────────────────────
# 5. OPPONENT EVOLUTION  (src/opponent.py, src/coevolution.py)
# ─────────────────────────────────────────────────────────────────────────────

# Mutation strength for opponent chromosomes (fraction of Dirichlet noise).
# new_mixture = (1 - strength) * old + strength * noise
# 0.0 = no change, 1.0 = completely random. Controls how fast opponents drift.
OPPONENT_MUTATION_STRENGTH = 0.75

# Weight for the novelty (diversity) term in opponent fitness.
# opponent_fitness = exploitation + NOVELTY_WEIGHT * diversity
# Exploitation = how much the opponent hurts the repertoire population.
# Diversity = mean L2 distance from this opponent to all others.
# Higher novelty weight = opponents stay spread out (avoid all converging to
# the same worst-case band). Set to 0.0 to disable diversity bonus.
NOVELTY_WEIGHT = 0.5

# Probability that two selected opponent parents undergo crossover.
# 0.0 = always clone parent A, 1.0 = always crossover.
OPPONENT_CROSSOVER_RATE = 1.0

# Probability that an opponent child is mutated after crossover/cloning.
# 0.0 = no mutation, 1.0 = always mutate.
OPPONENT_MUTATION_RATE = 0.5

# ─────────────────────────────────────────────────────────────────────────────
# 6. FITNESS FUNCTION  (src/fitness.py)
# ─────────────────────────────────────────────────────────────────────────────

# The three rating bands used in all computations.
# These must match what was crawled and stored in the database.
RATING_BANDS = ("1000-1399", "1400-1799", "1800-2199")


# The robust fitness objective:
#   fitness = mean_score + LAMBDA_WEIGHT * CVaR
#
# LAMBDA_WEIGHT controls the tradeoff between average and worst-case performance:
#   0.0  = only maximize average (ignores weak bands — can be exploitable)
#   1.0  = equal weight to mean and worst-case (balanced)
#   2.0  = worst-case dominates (very conservative, generalist repertoire)
#
# This is the main experimental variable. The sensitivity experiments test
# 0.0, 1.0, and 2.0. The main experiments use 1.0.
LAMBDA_WEIGHT = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# 7. EXPERIMENT MATRIX  (src/experiments.py)
# ─────────────────────────────────────────────────────────────────────────────

# --- Main comparison experiment ---
# Methods to compare: greedy baseline, static opponent, co-evolved.
MAIN_METHODS  = ["most_played_baseline", "STATIC", "COEVOLVE"]
MAIN_LAMBDA   = 1.0         # lambda used for all main runs
MAIN_SEEDS    = list(range(1000, 1005))   # 15 seeds → 15 independent runs per method

# --- Non-GA baselines ---
# Total evaluation budget = POP_SIZE_REPERTOIRES × N_GENERATIONS (same as the GA).
# This ensures fair comparison: baselines get the same number of fitness calls.
BASELINE_METHODS = ["RANDOM_SEARCH", "GREEDY_HILLCLIMB"]
BASELINE_SEEDS   = list(range(1000, 1005))
GA_EVAL_BUDGET   = POP_SIZE_REPERTOIRES * N_GENERATIONS  # = 6000

# --- Closure-constraint ablation ---
# The key novel contribution: does the closure rule actually help?
# Runs STATIC and COEVOLVE both WITH and WITHOUT the closure constraint,
# using the same seeds so results are paired (same random initialisation).
# 2 methods × 15 seeds = 30 new runs.
CLOSURE_ABLATION_METHODS = ["STATIC_NOCLOSURE", "COEVOLVE_NOCLOSURE"]
CLOSURE_ABLATION_SEEDS   = list(range(1000, 1005))   # same 15 seeds as main