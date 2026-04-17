"""
C8 — Co-evolution coordinator (src/coevolution.py)
Main GA loop supporting STATIC, COEVOLVE_FROZEN, and COEVOLVE modes.
A single call executes one complete run and returns a result dictionary.

Depends on: C5 (repertoire), C6 (fitness), C7 (opponent).
"""

from __future__ import annotations

import subprocess
import time
from typing import Optional

import numpy as np

from src.fitness import evaluate, evaluate_heldout
from src.opponent import Opponent
from src.repertoire import (
    BUDGET,
    Candidate,
    construct_initial,
    construct_random,
    crossover_candidates,
    mutate_candidate,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _tournament_select(population: list, rng, size: int = 3):
    """Pick *size* random individuals and return the one with the highest fitness."""
    actual_size = min(size, len(population))
    indices = rng.choice(len(population), size=actual_size, replace=False)
    best_idx = max(indices, key=lambda i: population[i].fitness
                   if population[i].fitness is not None else -float("inf"))
    return population[best_idx]


def _tournament_select_opp(population: list, rng, size: int = 3) -> Opponent:
    """Tournament selection for opponent population."""
    actual_size = min(size, len(population))
    indices = rng.choice(len(population), size=actual_size, replace=False)
    best_idx = max(indices, key=lambda i: population[i].fitness
                   if population[i].fitness is not None else -float("inf"))
    return population[best_idx]


def _mean_pairwise_distance(r_pop: list) -> float:
    """
    Mean pairwise repertoire diversity.
    Approximated as the mean Jaccard distance between white.committed key sets
    across all (i, j) pairs with i < j.
    """
    n = len(r_pop)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = set(r_pop[i].white.committed.keys())
            b = set(r_pop[j].white.committed.keys())
            union = a | b
            if not union:
                continue
            sym_diff = a.symmetric_difference(b)
            total += len(sym_diff) / len(union)
            count += 1
    return total / count if count > 0 else 0.0


def _mean_pairwise_opponent_distance(o_pop: list) -> float:
    """Mean pairwise L2 distance between opponent mixture vectors."""
    n = len(o_pop)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += float(np.linalg.norm(o_pop[i].mixture - o_pop[j].mixture))
            count += 1
    return total / count if count > 0 else 0.0


def serialize_candidate(candidate: Candidate) -> dict:
    """Convert a Candidate to a pickle-safe dict."""
    return {
        "white_committed": dict(candidate.white.committed),
        "white_reached": list(candidate.white.reached),
        "black_committed": dict(candidate.black.committed),
        "black_reached": list(candidate.black.reached),
    }


# ── Main function ──────────────────────────────────────────────────────────────


def run_coevolution(
    mode: str,
    config: dict,
    seed: int,
    graph_train: dict,
    graph_heldout: dict,
    base_policies_train: dict,
    eval_cache_train: dict,
    eval_cache_heldout: dict,
) -> dict:
    """
    Execute one complete co-evolutionary GA run.

    Parameters
    ----------
    mode : 'STATIC' | 'COEVOLVE_FROZEN' | 'COEVOLVE'
    config : dict with at minimum 'lambda_weight' and 'alpha'.
             Optional keys (with Section A defaults):
               pop_size_repertoires  (30)
               pop_size_opponents    (20)
               n_generations         (50)
               tournament_size       (3)
               crossover_rate        (0.7)
               mutation_rate         (0.3)
               hof_size              (5)

    Returns
    -------
    dict with keys: mode, config, seed, git_commit, history,
                    final_best_candidate, final_training_fitness,
                    heldout_score, wall_time_seconds.
    """
    assert mode in ("STATIC", "COEVOLVE_FROZEN", "COEVOLVE"), f"Unknown mode: {mode}"

    start_time = time.time()

    # Record git HEAD
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = "unknown"

    # Single RNG source — the only source of randomness in this call tree
    rng = np.random.default_rng(seed)

    # Config with defaults from Section A
    pop_size_r: int = int(config.get("pop_size_repertoires", 30))
    pop_size_o: int = int(config.get("pop_size_opponents", 20))
    n_generations: int = int(config.get("n_generations", 50))
    tournament_size: int = int(config.get("tournament_size", 3))
    crossover_rate: float = float(config.get("crossover_rate", 0.7))
    mutation_rate: float = float(config.get("mutation_rate", 0.3))
    hof_size: int = int(config.get("hof_size", 5))

    # ── Initialisation ─────────────────────────────────────────────────────────

    R_pop: list[Candidate] = []
    half = pop_size_r // 2

    # First half: greedy (construct_initial)
    for _ in range(half):
        w = construct_initial(graph_train, "white", BUDGET, rng)
        b = construct_initial(graph_train, "black", BUDGET, rng)
        R_pop.append(Candidate(white=w, black=b, fitness=None, band_scores_cache=None))

    # Second half: random (construct_random)
    for _ in range(pop_size_r - half):
        w = construct_random(graph_train, "white", BUDGET, rng)
        b = construct_random(graph_train, "black", BUDGET, rng)
        R_pop.append(Candidate(white=w, black=b, fitness=None, band_scores_cache=None))

    # Opponent population
    if mode == "STATIC":
        O_pop: list[Opponent] = [Opponent.uniform()]
    elif mode == "COEVOLVE_FROZEN":
        O_pop = [Opponent.random(rng) for _ in range(pop_size_o)]
    else:  # COEVOLVE
        O_pop = [Opponent.random(rng) for _ in range(pop_size_o)]

    # Hall of fame — only used in COEVOLVE mode
    HoF: list[Opponent] = []
    # Parallel dict: opp id → informativeness score (avoids mutating Opponent)
    hof_informativeness: dict[int, float] = {}

    history: list[dict] = []

    # ── Generational loop ──────────────────────────────────────────────────────

    # Keeps the last evaluated R_pop (before Step 3 replacement) so we can
    # pick the best candidate after the loop completes.
    last_evaluated_r_pop: list[Candidate] = []

    for gen in range(n_generations):

        # ── Step 1: Build evaluation set ──────────────────────────────────────
        if mode == "COEVOLVE" and HoF:
            n_hof_sample = min(hof_size, len(HoF))
            hof_idx = rng.choice(len(HoF), size=n_hof_sample, replace=False)
            hof_sample = [HoF[i] for i in hof_idx]
            eval_set: list[Opponent] = list(O_pop) + hof_sample
        else:
            eval_set = list(O_pop)

        # ── Step 2: Evaluate every repertoire against every opponent ──────────
        # Clear band_scores_cache before the first evaluation this generation.
        for cand in R_pop:
            cand.band_scores_cache = None

        for cand in R_pop:
            fitnesses_per_opp: list[float] = []
            for opp in eval_set:
                res = evaluate(
                    cand,
                    opp.mixture,
                    config,
                    eval_cache_train,
                    base_policies_train,
                    graph_train,
                    use_cache=True,   # band_scores reused across opponents this step
                )
                fitnesses_per_opp.append(res["fitness"])
            cand.fitness = float(np.mean(fitnesses_per_opp))

        # Snapshot the evaluated population for stats and final-best selection.
        # Must be done BEFORE Step 3 replaces R_pop with the unevaluated offspring.
        last_evaluated_r_pop = list(R_pop)

        # Collect fitness stats from the evaluated generation.
        valid_fitnesses = [c.fitness for c in last_evaluated_r_pop if c.fitness is not None]
        best_training_fitness = float(max(valid_fitnesses)) if valid_fitnesses else float("nan")
        mean_training_fitness = float(np.mean(valid_fitnesses)) if valid_fitnesses else float("nan")
        rep_diversity = _mean_pairwise_distance(last_evaluated_r_pop)

        # ── Step 3: Repertoire selection and reproduction ─────────────────────
        new_r_pop: list[Candidate] = []
        for _ in range(pop_size_r):
            parent_a = _tournament_select(R_pop, rng, tournament_size)

            if rng.random() < crossover_rate:
                parent_b = _tournament_select(R_pop, rng, tournament_size)
                child = crossover_candidates(parent_a, parent_b, rng)
            else:
                child = parent_a.copy()
                child.fitness = None
                child.band_scores_cache = None

            if rng.random() < mutation_rate:
                child = mutate_candidate(child, rng)

            child.band_scores_cache = None
            new_r_pop.append(child)

        R_pop = new_r_pop

        # ── Steps 4–6: COEVOLVE-only opponent evolution ───────────────────────
        if mode == "COEVOLVE":

            # Step 4: Opponent evaluation
            for opp in O_pop:
                rep_scores: list[float] = []
                for cand in R_pop:
                    res = evaluate(
                        cand,
                        opp.mixture,
                        config,
                        eval_cache_train,
                        base_policies_train,
                        graph_train,
                        use_cache=False,
                    )
                    rep_scores.append(res["mean_score"])

                exploitation = -float(np.mean(rep_scores))

                novelty_weight: float = float(config.get("novelty_weight", 0.1))
                other_mixtures = [o.mixture for o in O_pop if o is not opp]
                if other_mixtures:
                    novelty = novelty_weight * float(
                        np.mean([np.linalg.norm(opp.mixture - m) for m in other_mixtures])
                    )
                else:
                    novelty = 0.0

                opp.fitness = exploitation + novelty

            # Step 5: Opponent reproduction
            new_o_pop: list[Opponent] = []
            for _ in range(pop_size_o):
                opp_parent_a = _tournament_select_opp(O_pop, rng, tournament_size)

                if rng.random() < 0.5:
                    opp_parent_b = _tournament_select_opp(O_pop, rng, tournament_size)
                    child_opp = opp_parent_a.crossover(opp_parent_b, rng)
                else:
                    child_opp = Opponent(opp_parent_a.mixture.copy())
                    child_opp.fitness = None

                if rng.random() < 0.5:
                    child_opp = child_opp.mutate(rng)

                new_o_pop.append(child_opp)

            O_pop = new_o_pop

            # Step 6: Hall of fame update
            for opp in O_pop:
                opp_fitnesses: list[float] = []
                for cand in R_pop:
                    res = evaluate(
                        cand,
                        opp.mixture,
                        config,
                        eval_cache_train,
                        base_policies_train,
                        graph_train,
                        use_cache=False,
                    )
                    opp_fitnesses.append(res["fitness"])
                informativeness = float(np.var(opp_fitnesses))

                if len(HoF) < hof_size:
                    HoF.append(opp)
                    hof_informativeness[id(opp)] = informativeness
                else:
                    min_hof_idx = min(
                        range(len(HoF)),
                        key=lambda i: hof_informativeness.get(id(HoF[i]), 0.0),
                    )
                    if informativeness > hof_informativeness.get(id(HoF[min_hof_idx]), 0.0):
                        old = HoF[min_hof_idx]
                        hof_informativeness.pop(id(old), None)
                        HoF[min_hof_idx] = opp
                        hof_informativeness[id(opp)] = informativeness

        # ── Step 7: Log generation statistics ─────────────────────────────────
        # Fitness stats come from the pre-Step-3 evaluated pop (saved above).
        # Opponent diversity and HoF size come from the post-Step-5/6 state.
        opp_diversity: Optional[float] = (
            _mean_pairwise_opponent_distance(O_pop) if mode == "COEVOLVE" else None
        )

        history.append({
            "gen": gen,
            "best_training_fitness": best_training_fitness,
            "mean_training_fitness": mean_training_fitness,
            "repertoire_diversity": rep_diversity,
            "opponent_diversity": opp_diversity,
            "hof_size": len(HoF) if mode == "COEVOLVE" else 0,
        })

    # ── After loop: held-out evaluation ───────────────────────────────────────
    # Pick best from the LAST evaluated generation (before its offspring replaced it).

    best = max(
        last_evaluated_r_pop,
        key=lambda c: c.fitness if c.fitness is not None else -float("inf"),
    )

    heldout_score = evaluate_heldout(
        best,
        eval_cache_heldout,
        base_policies_train,
        graph_heldout,
        config,
    )

    elapsed = time.time() - start_time

    return {
        "mode": mode,
        "config": config,
        "seed": seed,
        "git_commit": git_commit,
        "history": history,
        "final_best_candidate": serialize_candidate(best),
        "final_training_fitness": best.fitness,
        "heldout_score": heldout_score,
        "wall_time_seconds": elapsed,
    }
