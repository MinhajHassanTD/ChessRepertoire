"""
C9 — Experiment runner (src/experiments.py)
Define the full run matrix and execute all runs, saving one pickle file per run.

Depends on: C8 (coevolution).
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
from src.eval_cache import load_eval_cache
from src.fitness import evaluate_heldout
from src.graph import load_graph
from src.policies import load_policies
from src.repertoire import BUDGET, Candidate, construct_initial
from src.style_eval_cache import load_style_eval_cache
from src.style_policies import load_style_policies

# ── Run matrix ────────────────────────────────────────────────────────────────

MAIN_EXPERIMENTS = [
    {'method': method, 'seed': seed, 'lambda_weight': 1.0, 'alpha': 1 / 3}
    for method in ['most_played_baseline', 'STATIC', 'COEVOLVE_FROZEN', 'COEVOLVE']
    for seed in range(1000, 1015)
]  # 4 methods × 15 seeds = 60 runs

SENSITIVITY_EXPERIMENTS = [
    {'method': method, 'seed': seed, 'lambda_weight': lam, 'alpha': 1 / 3}
    for method in ['STATIC', 'COEVOLVE']
    for lam in [0.0, 1.0, 2.0]
    for seed in range(2000, 2005)
]  # 2 methods × 3 lambdas × 5 seeds = 30 runs

ALL_EXPERIMENTS = MAIN_EXPERIMENTS + SENSITIVITY_EXPERIMENTS  # 90 runs total

# Path B ablation: COEVOLVE with increased novelty pressure and larger HoF.
# Method label 'COEVOLVE_B' keeps files separate from the original COEVOLVE runs.
PATH_B_EXPERIMENTS = [
    {
        'method': 'COEVOLVE_B',
        'seed': seed,
        'lambda_weight': 1.0,
        'alpha': 1 / 3,
        'novelty_weight': 0.5,   # was 0.1 in original COEVOLVE
        'hof_size': 15,          # was 5 in original COEVOLVE
    }
    for seed in range(1000, 1015)
]  # 15 seeds — main comparison only

# testv1: Tier 1 only (richer opponent space — per-position perturbations).
PATH_C_EXPERIMENTS = [
    {
        'method': 'COEVOLVE_C',
        'seed': seed,
        'lambda_weight': 1.0,
        'alpha': 1 / 3,
        'use_perturbations': True,
        'max_perturbations': 30,
    }
    for seed in range(1000, 1015)
]

# testv1: Tier 1 + Tier 2 (richer opponents + NSGA-II diversity preservation).
PATH_D_EXPERIMENTS = [
    {
        'method': 'COEVOLVE_D',
        'seed': seed,
        'lambda_weight': 1.0,
        'alpha': 1 / 3,
        'use_perturbations': True,
        'max_perturbations': 30,
        'use_nsga2': True,
    }
    for seed in range(1000, 1015)
]

TESTV1_EXPERIMENTS = PATH_C_EXPERIMENTS + PATH_D_EXPERIMENTS

# Path E: STATIC opponent (no co-evolution) + NSGA-II diversity preservation.
# Isolates the effect of NSGA-II alone — if COEVOLVE_D > STATIC_DIV, then
# the adversarial evolution part (not diversity preservation alone) is
# responsible for the improvement.
PATH_E_EXPERIMENTS = [
    {
        'method': 'STATIC_DIV',
        'seed': seed,
        'lambda_weight': 1.0,
        'alpha': 1 / 3,
        'use_perturbations': False,
        'use_nsga2': True,
    }
    for seed in range(1000, 1015)
]

# testv2 — Path F: train against synthetic style-archetype opponents
# (aggressive / defensive / positional) instead of rating bands. Held-out
# evaluation stays band-uniform, so this introduces a deliberate train/eval
# distribution shift that co-evolutionary robustness should exploit.
PATH_F_EXPERIMENTS = (
    [
        {
            'method': 'STATIC_STYLE',
            'seed': seed,
            'lambda_weight': 1.0,
            'alpha': 1 / 3,
            'opponent_mode': 'styles',
            'use_perturbations': False,
            'use_nsga2': False,
        }
        for seed in range(1000, 1015)
    ]
    + [
        {
            'method': 'COEVOLVE_STYLE',
            'seed': seed,
            'lambda_weight': 1.0,
            'alpha': 1 / 3,
            'opponent_mode': 'styles',
            'use_perturbations': True,
            'use_nsga2': False,
            'max_perturbations': 30,
        }
        for seed in range(1000, 1015)
    ]
    + [
        {
            'method': 'COEVOLVE_STYLE_D',
            'seed': seed,
            'lambda_weight': 1.0,
            'alpha': 1 / 3,
            'opponent_mode': 'styles',
            'use_perturbations': True,
            'use_nsga2': True,
            'max_perturbations': 30,
        }
        for seed in range(1000, 1015)
    ]
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def run_filename(method: str, lambda_weight: float, seed: int, runs_dir: str = 'runs') -> str:
    """Return the output pickle path for a run."""
    return os.path.join(runs_dir, f"{method}_l{lambda_weight}_s{seed}.pkl")


def check_clean_git() -> None:
    """Exit with error message if there are uncommitted git changes."""
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
            "Commit or stash all changes before running experiments.\n"
            f"{status}",
            file=sys.stderr,
        )
        sys.exit(1)


def _git_commit_hash() -> str:
    """Return the current git HEAD commit hash, or 'unknown' on failure."""
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
    """
    Construct the greedy most-played-move Candidate (no GA) and evaluate on
    held-out data.  Returns a result dict compatible with run_coevolution's
    output format so downstream analysis treats all methods uniformly.
    """
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


# ── Main execution ────────────────────────────────────────────────────────────

def run_all(
    data_dir: str = 'data',
    runs_dir: str = 'runs',
    experiments: list | None = None,
) -> None:
    """
    Execute all runs in the experiment matrix.

    Loads data files once, then iterates over *experiments*, skipping any run
    whose output pickle already exists (resume support).

    Parameters
    ----------
    data_dir : directory containing the .pkl data files
    runs_dir : directory where per-run pickle results are written
    experiments : list of run dicts; defaults to ALL_EXPERIMENTS
    """
    if experiments is None:
        experiments = ALL_EXPERIMENTS

    # Refuse to run on a dirty working tree
    check_clean_git()

    # Load all shared data once ─ never reload per run
    print("Loading data files ...")
    graph_train = load_graph(os.path.join(data_dir, 'graph_train.pkl'))
    graph_heldout = load_graph(os.path.join(data_dir, 'graph_heldout.pkl'))
    base_policies_train = load_policies(os.path.join(data_dir, 'base_policies.pkl'))
    eval_cache_train = load_eval_cache(os.path.join(data_dir, 'eval_cache_train.pkl'))
    eval_cache_heldout = load_eval_cache(os.path.join(data_dir, 'eval_cache_heldout.pkl'))

    # Style artefacts loaded lazily — only if any run in the batch needs them.
    needs_style = any(
        r.get('opponent_mode') == 'styles'
        or r['method'] in ('STATIC_STYLE', 'COEVOLVE_STYLE', 'COEVOLVE_STYLE_D')
        for r in experiments
    )
    style_policies_train = None
    style_eval_cache_train = None
    if needs_style:
        style_policies_train = load_style_policies(
            os.path.join(data_dir, 'style_policies_train.pkl')
        )
        style_eval_cache_train = load_style_eval_cache(
            os.path.join(data_dir, 'style_eval_cache_train.pkl')
        )
    print("Data loaded.\n")

    # Ensure the output directory exists
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
        else:
            # COEVOLVE_B/C/D and the style variants are COEVOLVE with extra
            # config keys; STATIC_DIV / STATIC_STYLE are STATIC with extra
            # config keys. We map to a canonical mode and relabel afterward
            # so each file is uniquely named.
            if method in ('COEVOLVE_B', 'COEVOLVE_C', 'COEVOLVE_D',
                          'COEVOLVE_STYLE', 'COEVOLVE_STYLE_D'):
                coevolve_mode = 'COEVOLVE'
            elif method in ('STATIC_DIV', 'STATIC_STYLE'):
                coevolve_mode = 'STATIC'
            else:
                coevolve_mode = method
            config = {
                'lambda_weight': lam,
                'alpha': run['alpha'],
                'novelty_weight': run.get('novelty_weight', 0.1),
                'hof_size': run.get('hof_size', 5),
                'use_perturbations': run.get('use_perturbations', False),
                'use_nsga2': run.get('use_nsga2', False),
                'max_perturbations': run.get('max_perturbations', 30),
                'opponent_mode': run.get('opponent_mode', 'bands'),
            }
            result = run_coevolution(
                mode=coevolve_mode,
                config=config,
                seed=seed,
                graph_train=graph_train,
                graph_heldout=graph_heldout,
                base_policies_train=base_policies_train,
                eval_cache_train=eval_cache_train,
                eval_cache_heldout=eval_cache_heldout,
                style_policies_train=style_policies_train,
                style_eval_cache_train=style_eval_cache_train,
            )
            if method in ('COEVOLVE_B', 'COEVOLVE_C', 'COEVOLVE_D', 'STATIC_DIV',
                          'STATIC_STYLE', 'COEVOLVE_STYLE', 'COEVOLVE_STYLE_D'):
                result['mode'] = method

        with open(out_path, 'wb') as fh:
            pickle.dump(result, fh)

        completed += 1
        print(f"          -> saved  {out_path}  (heldout={result['heldout_score']:.4f})")

    print(f"\nFinished: {completed}/{total} runs.")


if __name__ == '__main__':
    run_all()
