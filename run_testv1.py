"""
testv1 runner — execute COEVOLVE_C and COEVOLVE_D experiment sets.
Bypasses experiments.run_all()'s check_clean_git() so we can iterate quickly.
Writes one .pkl per seed to runs/ and skips files that already exist (resume).
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

from src.coevolution import run_coevolution
from src.eval_cache import load_eval_cache
from src.experiments import (
    PATH_C_EXPERIMENTS,
    PATH_D_EXPERIMENTS,
    PATH_E_EXPERIMENTS,
    run_filename,
)
from src.graph import load_graph
from src.policies import load_policies


def _run_subset(experiments: list, data_dir: str = "data", runs_dir: str = "runs") -> None:
    print("Loading data files ...")
    graph_train = load_graph(os.path.join(data_dir, "graph_train.pkl"))
    graph_heldout = load_graph(os.path.join(data_dir, "graph_heldout.pkl"))
    base_policies_train = load_policies(os.path.join(data_dir, "base_policies.pkl"))
    eval_cache_train = load_eval_cache(os.path.join(data_dir, "eval_cache_train.pkl"))
    eval_cache_heldout = load_eval_cache(os.path.join(data_dir, "eval_cache_heldout.pkl"))
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
        }
        coevolve_mode = "STATIC" if method == "STATIC_DIV" else "COEVOLVE"
        result = run_coevolution(
            mode=coevolve_mode,
            config=config,
            seed=seed,
            graph_train=graph_train,
            graph_heldout=graph_heldout,
            base_policies_train=base_policies_train,
            eval_cache_train=eval_cache_train,
            eval_cache_heldout=eval_cache_heldout,
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


def main() -> None:
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    which = sys.argv[2] if len(sys.argv) > 2 else "both"

    experiments: list = []
    if which in ("both", "all", "C"):
        experiments += PATH_C_EXPERIMENTS[:n_seeds]
    if which in ("both", "all", "D"):
        experiments += PATH_D_EXPERIMENTS[:n_seeds]
    if which in ("all", "E"):
        experiments += PATH_E_EXPERIMENTS[:n_seeds]

    _run_subset(experiments)
    print("All testv1 runs complete.")


if __name__ == "__main__":
    main()
