"""
testv2 runner — execute the style-archetype experiment set.

Trains against synthetic style mixtures (aggressive / defensive / positional)
while keeping held-out evaluation band-uniform — a deliberate train/eval
distribution shift.

Usage:
    python run_testv2.py [n_seeds] [which]
        n_seeds : int (default 15)
        which   : 'all' | 'STATIC' | 'COEVOLVE' | 'COEVOLVE_D' (default 'all')
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

from src.coevolution import run_coevolution
from src.eval_cache import load_eval_cache
from src.experiments import PATH_F_EXPERIMENTS, run_filename
from src.graph import load_graph
from src.policies import load_policies
from src.style_eval_cache import load_style_eval_cache
from src.style_policies import load_style_policies


def _run_subset(experiments: list, data_dir: str = "data", runs_dir: str = "runs") -> None:
    print("Loading data files ...")
    graph_train = load_graph(os.path.join(data_dir, "graph_train.pkl"))
    graph_heldout = load_graph(os.path.join(data_dir, "graph_heldout.pkl"))
    base_policies_train = load_policies(os.path.join(data_dir, "base_policies.pkl"))
    eval_cache_train = load_eval_cache(os.path.join(data_dir, "eval_cache_train.pkl"))
    eval_cache_heldout = load_eval_cache(os.path.join(data_dir, "eval_cache_heldout.pkl"))
    style_policies_train = load_style_policies(
        os.path.join(data_dir, "style_policies_train.pkl")
    )
    style_eval_cache_train = load_style_eval_cache(
        os.path.join(data_dir, "style_eval_cache_train.pkl")
    )
    print("Data loaded.")

    Path(runs_dir).mkdir(parents=True, exist_ok=True)

    total = len(experiments)
    for i, run in enumerate(experiments, 1):
        method = run["method"]
        seed = run["seed"]
        lam = run["lambda_weight"]
        out_path = run_filename(method, lam, seed, runs_dir)

        if os.path.exists(out_path):
            print(f"[{i}/{total}] SKIP  {out_path}")
            continue

        print(f"[{i}/{total}] RUN   method={method}  lam={lam}  seed={seed} ...", flush=True)
        t0 = time.time()

        config = {
            "lambda_weight": lam,
            "alpha": run["alpha"],
            "novelty_weight": run.get("novelty_weight", 0.1),
            "hof_size": run.get("hof_size", 5),
            "use_perturbations": run.get("use_perturbations", False),
            "use_nsga2": run.get("use_nsga2", False),
            "max_perturbations": run.get("max_perturbations", 30),
            "opponent_mode": run.get("opponent_mode", "styles"),
        }

        if method == "STATIC_STYLE":
            coevolve_mode = "STATIC"
        elif method in ("COEVOLVE_STYLE", "COEVOLVE_STYLE_D"):
            coevolve_mode = "COEVOLVE"
        else:
            coevolve_mode = method

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
        result["mode"] = method  # relabel so analyze.py can find it

        with open(out_path, "wb") as fh:
            pickle.dump(result, fh)

        dt = time.time() - t0
        print(
            f"          -> saved  {out_path}  "
            f"(heldout={result['heldout_score']:.4f} "
            f"adv_co={result['heldout_adversarial_coevolved']:.4f} "
            f"time={dt:.1f}s)",
            flush=True,
        )


def _filter(method_substr: str) -> list:
    if method_substr == "all":
        return PATH_F_EXPERIMENTS
    return [r for r in PATH_F_EXPERIMENTS if method_substr in r["method"]]


def main() -> None:
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    which = sys.argv[2] if len(sys.argv) > 2 else "all"

    method_keys = {
        "all": ["STATIC_STYLE", "COEVOLVE_STYLE", "COEVOLVE_STYLE_D"],
        "STATIC": ["STATIC_STYLE"],
        "COEVOLVE": ["COEVOLVE_STYLE"],
        "COEVOLVE_D": ["COEVOLVE_STYLE_D"],
    }
    if which not in method_keys:
        print(f"Unknown 'which': {which}. Use one of: {list(method_keys)}")
        sys.exit(1)

    experiments: list = []
    for m in method_keys[which]:
        experiments += [r for r in PATH_F_EXPERIMENTS if r["method"] == m][:n_seeds]

    _run_subset(experiments)
    print("All testv2 runs complete.")


if __name__ == "__main__":
    main()
