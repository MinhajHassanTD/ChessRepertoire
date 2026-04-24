"""
Experiment runner — defines every run in the experiment matrix and dispatches them.

Six experiment batches:

  MAIN_EXPERIMENTS           — 4 methods × 15 seeds = 60 runs  (lambda = 1.0)
  BASELINE_EXPERIMENTS       — 2 non-GA baselines × 15 seeds = 30 runs
  SENSITIVITY_EXPERIMENTS    — 2 methods × 3 lambdas × 5 seeds = 30 runs
  CLOSURE_ABLATION_EXPERIMENTS — 2 no-closure variants × 15 seeds = 30 runs
  THRESHOLD_SWEEP_EXPERIMENTS  — 5 thresholds × 5 seeds = 25 runs (STATIC only)
  BUDGET_SWEEP_EXPERIMENTS     — 3 budgets × 5 seeds = 15 runs   (STATIC only, optional)

Non-GA baselines test whether GA population + crossover adds value over
single-solution search with the same fitness-call budget:
  RANDOM_SEARCH    — sample 1500 random candidates, return the best
  GREEDY_HILLCLIMB — (1+1)-ES: start from greedy init, 1500 accept-if-better steps

Closure ablation tests the novel chromosome contribution: does forcing opponent
reply coverage (the closure rule) actually improve held-out performance, or
would a simpler unconstrained representation work equally well?
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
    CLOSURE_ABLATION_METHODS,
    CLOSURE_ABLATION_SEEDS,
    CLOSURE_THRESHOLD_VALUES,
    CLOSURE_THRESHOLD_SEEDS,
    BUDGET_VALUES,
    BUDGET_SEEDS,
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

# Closure ablation: same as main STATIC/COEVOLVE but with use_closure=False.
# Mode string is "STATIC_NOCLOSURE" or "COEVOLVE_NOCLOSURE"; dispatcher strips
# the suffix and passes use_closure=False in the config dict.
CLOSURE_ABLATION_EXPERIMENTS = [
    {'method': method, 'seed': seed, 'lambda_weight': MAIN_LAMBDA, 'alpha': 1 / 3}
    for method in CLOSURE_ABLATION_METHODS
    for seed in CLOSURE_ABLATION_SEEDS
]

# Closure threshold sweep: STATIC at five different thresholds.
# Uses a separate 'closure_threshold' key so the dispatcher can override it.
THRESHOLD_SWEEP_EXPERIMENTS = [
    {'method': 'STATIC', 'seed': seed, 'lambda_weight': MAIN_LAMBDA,
     'alpha': 1 / 3, 'closure_threshold': thresh}
    for thresh in CLOSURE_THRESHOLD_VALUES
    for seed in CLOSURE_THRESHOLD_SEEDS
]

# Budget sweep: STATIC at three different committed-move budgets (optional).
BUDGET_SWEEP_EXPERIMENTS = [
    {'method': 'STATIC', 'seed': seed, 'lambda_weight': MAIN_LAMBDA,
     'alpha': 1 / 3, 'budget': bgt}
    for bgt in BUDGET_VALUES
    for seed in BUDGET_SEEDS
]

# 120 original runs.  New ablation runs are in separate lists so they can be
# run independently without re-running everything.
ALL_EXPERIMENTS = MAIN_EXPERIMENTS + BASELINE_EXPERIMENTS + SENSITIVITY_EXPERIMENTS
ALL_ABLATION_EXPERIMENTS = CLOSURE_ABLATION_EXPERIMENTS + THRESHOLD_SWEEP_EXPERIMENTS + BUDGET_SWEEP_EXPERIMENTS


# ── Helpers ───────────────────────────────────────────────────────────────────


def run_filename(method: str, lambda_weight: float, seed: int, runs_dir: str = 'runs',
                 suffix: str = '') -> str:
    """Build the output path for one run. suffix allows ablation variants to
    use distinct filenames (e.g. '_thresh0.05' or '_budget15')."""
    return os.path.join(runs_dir, f"{method}_l{lambda_weight}_s{seed}{suffix}.pkl")


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
    eval_cache_train: dict | None = None,
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

    training_fitness = None
    if eval_cache_train is not None:
        # Keep a comparable scalar to other methods for convergence plotting.
        training_fitness = _eval_candidate(
            candidate,
            config,
            eval_cache_train,
            base_policies_train,
            graph_train,
        )

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
        'final_training_fitness': training_fitness,
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

        # Build a filename suffix for ablation variants so they don't collide with main runs.
        suffix = ''
        if 'closure_threshold' in run:
            suffix = f"_thresh{run['closure_threshold']}"
        if 'budget' in run:
            suffix = f"_budget{run['budget']}"

        out_path = run_filename(method, lam, seed, runs_dir, suffix=suffix)

        if os.path.exists(out_path):
            completed += 1
            print(f"[{completed}/{total}] SKIP  {out_path}")
            continue

        print(f"[{completed + 1}/{total}] RUN   method={method}  lam={lam}  seed={seed}{suffix} ...")

        if method == 'most_played_baseline':
            result = run_baseline(
                run,
                graph_train,
                graph_heldout,
                base_policies_train,
                eval_cache_heldout,
                eval_cache_train=eval_cache_train,
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
            # All GA modes go through run_coevolution.
            # STATIC_NOCLOSURE / COEVOLVE_NOCLOSURE: strip the suffix, disable closure.
            ga_mode = method
            use_closure = True
            if method.endswith('_NOCLOSURE'):
                ga_mode = method[: -len('_NOCLOSURE')]
                use_closure = False

            config = {
                'lambda_weight': lam,
                'alpha': run['alpha'],
                'novelty_weight': run.get('novelty_weight', NOVELTY_WEIGHT),
                'hof_size': run.get('hof_size', HOF_SIZE),
                'use_closure': use_closure,
            }
            # Allow individual runs to override closure threshold and budget.
            if 'closure_threshold' in run:
                config['closure_threshold'] = run['closure_threshold']
            if 'budget' in run:
                config['budget'] = run['budget']

            result = run_coevolution(
                mode=ga_mode,
                config=config,
                seed=seed,
                graph_train=graph_train,
                graph_heldout=graph_heldout,
                base_policies_train=base_policies_train,
                eval_cache_train=eval_cache_train,
                eval_cache_heldout=eval_cache_heldout,
            )
            # Store the original method name (including _NOCLOSURE) so analysis
            # can group these runs separately from the standard runs.
            result['mode'] = method

        with open(out_path, 'wb') as fh:
            pickle.dump(result, fh)

        completed += 1
        print(f"          -> saved  {out_path}  (heldout={result['heldout_score']:.4f})")

    print(f"\nFinished: {completed}/{total} runs.")


def run_ablations(
    data_dir: str = 'data',
    runs_dir: str = 'runs',
    skip_git_check: bool = False,
) -> None:
    """Run only the new ablation experiments (closure ablation + sweeps).

    Skips existing files so it's safe to call multiple times. Separates them
    from the main experiments to avoid re-running those 120 runs.
    """
    run_all(
        data_dir=data_dir,
        runs_dir=runs_dir,
        experiments=ALL_ABLATION_EXPERIMENTS,
        skip_git_check=skip_git_check,
    )


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'ablations':
        # python -m src.experiments ablations
        run_ablations(skip_git_check=True)
    else:
        run_all()
