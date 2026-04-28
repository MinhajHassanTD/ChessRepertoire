"""
Co-evolution coordinator.

Implements a single GA loop with three modes:

    STATIC          — fixed uniform-mixture opponent, no opponent evolution.
    COEVOLVE_FROZEN — random opponent population at gen 0, frozen thereafter.
    COEVOLVE        — opponent population co-evolves against the repertoire pop.

This is the only training loop in the project; all experimental conditions
are expressed by selecting one of the three modes above.
"""

from __future__ import annotations

import subprocess
import time
from typing import Optional

import numpy as np

from src.config import (
    POP_SIZE_REPERTOIRES,
    POP_SIZE_OPPONENTS,
    N_GENERATIONS,
    TOURNAMENT_SIZE,
    CROSSOVER_RATE,
    MUTATION_RATE,
    HOF_SIZE,
    NOVELTY_WEIGHT,
    LAMBDA_WEIGHT,
    OPPONENT_CROSSOVER_RATE,
    OPPONENT_MUTATION_RATE,
    INIT_GREEDY_FRACTION,
    REPERTOIRE_DIVERSITY_THRESHOLD,
    REPERTOIRE_REINIT_FRACTION,
    USE_CLOSURE,
)
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


# ── Selection helpers ─────────────────────────────────────────────────────────


def _tournament_select(population: list, rng, size: int = 3):
    actual_size = min(size, len(population))
    indices = rng.choice(len(population), size=actual_size, replace=False)
    best_idx = max(
        indices,
        key=lambda i: population[i].fitness if population[i].fitness is not None else -float("inf"),
    )
    return population[best_idx]


def _mean_pairwise_distance(r_pop: list) -> float:
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
            total += len(a.symmetric_difference(b)) / len(union)
            count += 1
    return total / count if count > 0 else 0.0


def _mean_pairwise_opponent_distance(o_pop: list) -> float:
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
    return {
        "white_committed": dict(candidate.white.committed),
        "white_reached": list(candidate.white.reached),
        "black_committed": dict(candidate.black.committed),
        "black_reached": list(candidate.black.reached),
    }


# ── Main loop ─────────────────────────────────────────────────────────────────


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
    """Execute one GA run.

    *mode* is one of: 'STATIC', 'COEVOLVE_FROZEN', 'COEVOLVE'.

    Returns a result dict with the final best candidate, per-generation
    history, and held-out evaluation under a uniform opponent mixture.
    """
    assert mode in ("STATIC", "COEVOLVE_FROZEN", "COEVOLVE"), f"Unknown mode: {mode}"

    start_time = time.time()

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = "unknown"

    rng = np.random.default_rng(seed)

    pop_size_r: int = int(config.get("pop_size_repertoires", POP_SIZE_REPERTOIRES))
    pop_size_o: int = int(config.get("pop_size_opponents", POP_SIZE_OPPONENTS))
    n_generations: int = int(config.get("n_generations", N_GENERATIONS))
    tournament_size: int = int(config.get("tournament_size", TOURNAMENT_SIZE))
    crossover_rate: float = float(config.get("crossover_rate", CROSSOVER_RATE))
    mutation_rate: float = float(config.get("mutation_rate", MUTATION_RATE))
    hof_size: int = int(config.get("hof_size", HOF_SIZE))
    novelty_weight: float = float(config.get("novelty_weight", NOVELTY_WEIGHT))
    # use_closure=False disables the auto-coverage rule (ablation experiment).
    use_closure: bool = bool(config.get("use_closure", USE_CLOSURE))

    init_greedy_fraction: float = float(config.get("init_greedy_fraction", INIT_GREEDY_FRACTION))

    rep_diversity_threshold: float = float(
        config.get("repertoire_diversity_threshold", REPERTOIRE_DIVERSITY_THRESHOLD)
    )
    rep_reinit_fraction: float = float(
        config.get("repertoire_reinit_fraction", REPERTOIRE_REINIT_FRACTION)
    )

    # ── Initialisation ────────────────────────────────────────────────────────

    R_pop: list[Candidate] = []
    n_greedy = round(pop_size_r * init_greedy_fraction)
    for _ in range(n_greedy):
        w = construct_initial(graph_train, "white", BUDGET, rng, use_closure=use_closure)
        b = construct_initial(graph_train, "black", BUDGET, rng, use_closure=use_closure)
        R_pop.append(Candidate(white=w, black=b, fitness=None, band_scores_cache=None))
    for _ in range(pop_size_r - n_greedy):
        w = construct_random(graph_train, "white", BUDGET, rng, use_closure=use_closure)
        b = construct_random(graph_train, "black", BUDGET, rng, use_closure=use_closure)
        R_pop.append(Candidate(white=w, black=b, fitness=None, band_scores_cache=None))

    if mode == "STATIC":
        O_pop: list[Opponent] = [Opponent.uniform()]
    else:
        O_pop = [Opponent.random(rng) for _ in range(pop_size_o)]

    HoF: list[Opponent] = []
    hof_informativeness: dict[int, float] = {}

    history: list[dict] = []
    last_evaluated_r_pop: list[Candidate] = []

    def _eval_cand_vs_opp(cand: Candidate, opp: Opponent, use_cache: bool) -> dict:
        return evaluate(
            cand,
            opp.mixture,
            config,
            eval_cache_train,
            base_policies_train,
            graph_train,
            use_cache=use_cache,
        )

    # ── Generational loop ────────────────────────────────────────────────────

    for gen in range(n_generations):

        # Step 1: build evaluation set (population + sample of HoF if COEVOLVE)
        if mode == "COEVOLVE" and HoF:
            n_hof_sample = min(hof_size, len(HoF))
            hof_idx = rng.choice(len(HoF), size=n_hof_sample, replace=False)
            hof_sample = [HoF[i] for i in hof_idx]
            eval_set: list[Opponent] = list(O_pop) + hof_sample
        else:
            eval_set = list(O_pop)

        # Step 2: evaluate every repertoire against every opponent
        for cand in R_pop:
            cand.band_scores_cache = None
            cand.white_band_scores_cache = None
            cand.black_band_scores_cache = None

        for cand in R_pop:
            fitnesses_per_opp: list[float] = []
            for opp in eval_set:
                res = _eval_cand_vs_opp(cand, opp, use_cache=True)
                fitnesses_per_opp.append(res["fitness"])
            cand.fitness = float(np.mean(fitnesses_per_opp))

        last_evaluated_r_pop = list(R_pop)

        valid_fitnesses = [c.fitness for c in last_evaluated_r_pop if c.fitness is not None]
        best_training_fitness = float(max(valid_fitnesses)) if valid_fitnesses else float("nan")
        mean_training_fitness = float(np.mean(valid_fitnesses)) if valid_fitnesses else float("nan")
        rep_diversity = _mean_pairwise_distance(last_evaluated_r_pop)

        # Step 3: repertoire reproduction (tournament + crossover + mutation)
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

        # Repertoire diversity maintenance: inject random individuals when collapsed.
        if rep_diversity_threshold > 0.0 and rep_diversity < rep_diversity_threshold:
            n_rep_reinit = max(1, round(pop_size_r * rep_reinit_fraction))
            reinit_idx = rng.choice(pop_size_r, size=n_rep_reinit, replace=False)
            for idx in reinit_idx:
                w = construct_random(graph_train, "white", BUDGET, rng, use_closure=use_closure)
                b = construct_random(graph_train, "black", BUDGET, rng, use_closure=use_closure)
                new_r_pop[idx] = Candidate(white=w, black=b, fitness=None, band_scores_cache=None)

        R_pop = new_r_pop

        # Step 4: opponent evolution (COEVOLVE only)
        if mode == "COEVOLVE":
            for opp in O_pop:
                rep_scores: list[float] = []
                for cand in R_pop:
                    res = _eval_cand_vs_opp(cand, opp, use_cache=False)
                    rep_scores.append(res["mean_score"])
                exploitation = -float(np.mean(rep_scores))

                other_mixtures = [o.mixture for o in O_pop if o is not opp]
                if other_mixtures:
                    novelty = novelty_weight * float(
                        np.mean([np.linalg.norm(opp.mixture - m) for m in other_mixtures])
                    )
                else:
                    novelty = 0.0
                opp.fitness = exploitation + novelty

            # Reproduce opponents with Dirichlet-resample crossover.
            new_o_pop: list[Opponent] = []
            for _ in range(pop_size_o):
                parent_a = _tournament_select(O_pop, rng, tournament_size)
                if rng.random() < OPPONENT_CROSSOVER_RATE:
                    parent_b = _tournament_select(O_pop, rng, tournament_size)
                    child_opp = parent_a.crossover(parent_b, rng)
                else:
                    child_opp = Opponent(parent_a.mixture.copy())
                if rng.random() < OPPONENT_MUTATION_RATE:
                    child_opp = child_opp.mutate(rng)
                new_o_pop.append(child_opp)

            O_pop = new_o_pop

            # Hall of Fame: keep the most informative opponents seen so far
            for opp in O_pop:
                opp_fitnesses: list[float] = []
                for cand in R_pop:
                    res = _eval_cand_vs_opp(cand, opp, use_cache=False)
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

    # ── Final selection + held-out evaluation ────────────────────────────────

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
