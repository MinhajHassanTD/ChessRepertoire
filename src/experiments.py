"""
Experiment runner: define the run matrix and execute all runs, saving one
pickle per run.

Three batches:
  MAIN_EXPERIMENTS        — 4 methods × 15 seeds = 60 runs (lambda = 1.0)
  BASELINE_EXPERIMENTS    — 2 non-GA baselines × 15 seeds = 30 runs
  SENSITIVITY_EXPERIMENTS — 2 methods × 3 lambdas × 5 seeds = 30 runs

Non-GA baselines establish that GA crossover/mutation adds value beyond
greedy initialisation:
  RANDOM_SEARCH    — evaluate 1500 random candidates, return the best
  GREEDY_HILLCLIMB — start from greedy init, accept mutations that improve
                     fitness (hill-climbing without a population), 1500 steps
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from src.coevolution import run_coevolution, serialize_candidate
from src.config import (
    BUDGET,
    GA_EVAL_BUDGET as _GA_BUDGET,
    MAIN_METHODS,
    MAIN_LAMBDA,
    MAIN_SEEDS,
    BASELINE_METHODS,
    BASELINE_SEEDS,
    SENSITIVITY_METHODS,
    SENSITIVITY_LAMBDAS,
    SENSITIVITY_SEEDS,
    LAMBDA_WEIGHT,
    NOVELTY_WEIGHT,
    HOF_SIZE,
    RATING_BANDS,
)
from src.eval_cache import load_eval_cache
from src.fitness import evaluate, evaluate_heldout
from src.graph import load_graph
from src.policies import load_policies
from src.repertoire import (
    Candidate, MutationFailed, construct_initial, construct_random,
    mutate_extend, mutate_move_swap, mutate_opening_replacement, mutate_prune,
)


# ── Run matrix ────────────────────────────────────────────────────────────────

MAIN_EXPERIMENTS = [
    {'method': method, 'seed': seed, 'lambda_weight': MAIN_LAMBDA, 'alpha': 1 / 3}
    for method in MAIN_METHODS
    for seed in MAIN_SEEDS
]

BASELINE_EXPERIMENTS = [
    {'method': method, 'seed': seed, 'lambda_weight': MAIN_LAMBDA, 'alpha': 1 / 3,
     'eval_budget': _GA_BUDGET}
    for method in BASELINE_METHODS
    for seed in BASELINE_SEEDS
]

SENSITIVITY_EXPERIMENTS = [
    {'method': method, 'seed': seed, 'lambda_weight': lam, 'alpha': 1 / 3}
    for method in SENSITIVITY_METHODS
    for lam in SENSITIVITY_LAMBDAS
    for seed in SENSITIVITY_SEEDS
]

ALL_EXPERIMENTS = MAIN_EXPERIMENTS + BASELINE_EXPERIMENTS + SENSITIVITY_EXPERIMENTS  # 120 runs total


# ── Helpers ───────────────────────────────────────────────────────────────────


def run_filename(method: str, lambda_weight: float, seed: int, runs_dir: str = 'runs') -> str:
    return os.path.join(runs_dir, f"{method}_l{lambda_weight}_s{seed}.pkl")


def check_clean_git() -> None:
    """Exit with error if there are uncommitted git changes.

    Skipped when the environment variable SKIP_GIT_CHECK=1 is set.
    """
    if os.environ.get("SKIP_GIT_CHECK", "0") == "1":
        print("[git] SKIP_GIT_CHECK=1 — skipping clean-tree check.")
        return
    try:
        status = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except subprocess.CalledProcessError:
        print("ERROR: Could not run 'git status'.", file=sys.stderr)
        sys.exit(1)
    if status:
        print(
            "ERROR: Uncommitted git changes detected.\n"
            "Commit or stash changes, or set SKIP_GIT_CHECK=1 to bypass.\n"
            f"{status}",
            file=sys.stderr,
        )
        sys.exit(1)


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return 'unknown'


def run_baseline(
    run: dict,
    graph_train: dict,
    graph_heldout: dict,
    base_policies_train: dict,
    eval_cache_heldout: dict,
) -> dict:
    """Greedy most-played-move baseline (no GA), evaluated on held-out."""
    start_time = time.time()
    git_commit = _git_commit_hash()

    rng = np.random.default_rng(run['seed'])
    white_rep = construct_initial(graph_train, 'white', BUDGET, rng)
    black_rep = construct_initial(graph_train, 'black', BUDGET, rng)
    candidate = Candidate(white=white_rep, black=black_rep,
                          fitness=None, band_scores_cache=None)

    config = {'lambda_weight': run['lambda_weight'], 'alpha': run['alpha']}

    heldout_score = evaluate_heldout(
        candidate,
        eval_cache_heldout,
        base_policies_train,
        graph_heldout,
        config,
    )

    elapsed = time.time() - start_time

    return {
        'mode': 'most_played_baseline',
        'config': config,
        'seed': run['seed'],
        'git_commit': git_commit,
        'history': [],
        'final_best_candidate': serialize_candidate(candidate),
        'final_training_fitness': None,
        'heldout_score': heldout_score,
        'wall_time_seconds': elapsed,
    }


# ── Non-GA baselines ─────────────────────────────────────────────────────────


def _eval_candidate(candidate, config, eval_cache_train, base_policies_train, graph_train):
    """Evaluate candidate against uniform mixture; update fitness in place."""
    uniform = np.ones(len(RATING_BANDS)) / len(RATING_BANDS)
    result = evaluate(
        candidate, uniform, config,
        eval_cache_train, base_policies_train, graph_train,
        use_cache=True,
    )
    candidate.fitness = result['fitness']
    return candidate.fitness


def run_random_search(
    run: dict,
    graph_train: dict,
    graph_heldout: dict,
    base_policies_train: dict,
    eval_cache_train: dict,
    eval_cache_heldout: dict,
) -> dict:
    """Evaluate `eval_budget` random candidates; return the best on held-out."""
    start_time = time.time()
    git_commit = _git_commit_hash()
    rng = np.random.default_rng(run['seed'])
    config = {'lambda_weight': run['lambda_weight'], 'alpha': run['alpha']}
    budget = run.get('eval_budget', _GA_BUDGET)

    best_candidate = None
    best_fitness = -float('inf')

    for _ in range(budget):
        white_rep = construct_random(graph_train, 'white', BUDGET, rng)
        black_rep = construct_random(graph_train, 'black', BUDGET, rng)
        cand = Candidate(white=white_rep, black=black_rep,
                         fitness=None, band_scores_cache=None)
        f = _eval_candidate(cand, config, eval_cache_train, base_policies_train, graph_train)
        if f > best_fitness:
            best_fitness = f
            best_candidate = cand

    heldout_score = evaluate_heldout(
        best_candidate, eval_cache_heldout, base_policies_train, graph_heldout, config,
    )

    return {
        'mode': 'RANDOM_SEARCH',
        'config': config,
        'seed': run['seed'],
        'git_commit': git_commit,
        'history': [],
        'final_best_candidate': serialize_candidate(best_candidate),
        'final_training_fitness': best_fitness,
        'heldout_score': heldout_score,
        'wall_time_seconds': time.time() - start_time,
    }


_MUTATORS = [mutate_move_swap, mutate_extend, mutate_prune, mutate_opening_replacement]


def run_greedy_hillclimb(
    run: dict,
    graph_train: dict,
    graph_heldout: dict,
    base_policies_train: dict,
    eval_cache_train: dict,
    eval_cache_heldout: dict,
) -> dict:
    """Hill-climber: start from greedy init, accept any mutation that improves fitness.

    Uses the same eval budget as the GA so comparisons are fair. No population,
    no crossover — pure (1+1)-ES on the fitness landscape.
    """
    start_time = time.time()
    git_commit = _git_commit_hash()
    rng = np.random.default_rng(run['seed'])
    config = {'lambda_weight': run['lambda_weight'], 'alpha': run['alpha']}
    budget = run.get('eval_budget', _GA_BUDGET)

    white_rep = construct_initial(graph_train, 'white', BUDGET, rng)
    black_rep = construct_initial(graph_train, 'black', BUDGET, rng)
    current = Candidate(white=white_rep, black=black_rep,
                        fitness=None, band_scores_cache=None)
    current_fitness = _eval_candidate(
        current, config, eval_cache_train, base_policies_train, graph_train,
    )

    for _ in range(budget - 1):
        # Pick a random colour and a random mutator
        color = rng.choice(['white', 'black'])
        mutator = _MUTATORS[rng.integers(len(_MUTATORS))]
        rep = current.white if color == 'white' else current.black
        try:
            new_rep = mutator(rep, rng)
        except MutationFailed:
            continue

        if color == 'white':
            child = Candidate(white=new_rep, black=current.black.copy(),
                              fitness=None, band_scores_cache=None)
        else:
            child = Candidate(white=current.white.copy(), black=new_rep,
                              fitness=None, band_scores_cache=None)

        child_fitness = _eval_candidate(
            child, config, eval_cache_train, base_policies_train, graph_train,
        )
        if child_fitness > current_fitness:
            current = child
            current_fitness = child_fitness

    heldout_score = evaluate_heldout(
        current, eval_cache_heldout, base_policies_train, graph_heldout, config,
    )

    return {
        'mode': 'GREEDY_HILLCLIMB',
        'config': config,
        'seed': run['seed'],
        'git_commit': git_commit,
        'history': [],
        'final_best_candidate': serialize_candidate(current),
        'final_training_fitness': current_fitness,
        'heldout_score': heldout_score,
        'wall_time_seconds': time.time() - start_time,
    }


# ── Main execution ────────────────────────────────────────────────────────────


def run_all(
    data_dir: str = 'data',
    runs_dir: str = 'runs',
    experiments: list | None = None,
    skip_git_check: bool = False,
) -> None:
    if experiments is None:
        experiments = ALL_EXPERIMENTS

    if not skip_git_check:
        check_clean_git()

    print("Loading data files ...")
    graph_train = load_graph(os.path.join(data_dir, 'graph_train.pkl'))
    graph_heldout = load_graph(os.path.join(data_dir, 'graph_heldout.pkl'))
    base_policies_train = load_policies(os.path.join(data_dir, 'base_policies.pkl'))
    eval_cache_train = load_eval_cache(os.path.join(data_dir, 'eval_cache_train.pkl'))
    eval_cache_heldout = load_eval_cache(os.path.join(data_dir, 'eval_cache_heldout.pkl'))
    print("Data loaded.\n")

    Path(runs_dir).mkdir(parents=True, exist_ok=True)

    total = len(experiments)
    completed = 0

    for run in experiments:
        method = run['method']
        seed = run['seed']
        lam = run['lambda_weight']
        out_path = run_filename(method, lam, seed, runs_dir)

        if os.path.exists(out_path):
            completed += 1
            print(f"[{completed}/{total}] SKIP  {out_path}")
            continue

        print(f"[{completed + 1}/{total}] RUN   method={method}  lam={lam}  seed={seed} ...")

        if method == 'most_played_baseline':
            result = run_baseline(
                run,
                graph_train,
                graph_heldout,
                base_policies_train,
                eval_cache_heldout,
            )
        elif method == 'RANDOM_SEARCH':
            result = run_random_search(
                run,
                graph_train,
                graph_heldout,
                base_policies_train,
                eval_cache_train,
                eval_cache_heldout,
            )
        elif method == 'GREEDY_HILLCLIMB':
            result = run_greedy_hillclimb(
                run,
                graph_train,
                graph_heldout,
                base_policies_train,
                eval_cache_train,
                eval_cache_heldout,
            )
        else:
            config = {
                'lambda_weight': lam,
                'alpha': run['alpha'],
                'novelty_weight': run.get('novelty_weight', NOVELTY_WEIGHT),
                'hof_size': run.get('hof_size', HOF_SIZE),
            }
            result = run_coevolution(
                mode=method,
                config=config,
                seed=seed,
                graph_train=graph_train,
                graph_heldout=graph_heldout,
                base_policies_train=base_policies_train,
                eval_cache_train=eval_cache_train,
                eval_cache_heldout=eval_cache_heldout,
            )

        with open(out_path, 'wb') as fh:
            pickle.dump(result, fh)

        completed += 1
        print(f"          -> saved  {out_path}  (heldout={result['heldout_score']:.4f})")

    print(f"\nFinished: {completed}/{total} runs.")


if __name__ == '__main__':
    run_all()
