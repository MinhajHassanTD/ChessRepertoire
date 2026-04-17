"""
C8 — Co-evolution coordinator (src/coevolution.py)
Main GA loop supporting STATIC, COEVOLVE_FROZEN, and COEVOLVE modes,
plus the testv1 variants COEVOLVE_C (rich opponents) and COEVOLVE_D
(rich opponents + NSGA-II on repertoires).

Depends on: C5 (repertoire), C6 (fitness), C7 (opponent).
"""

from __future__ import annotations

import subprocess
import time
from typing import Optional

import numpy as np

from src.fitness import (
    evaluate,
    evaluate_heldout,
    evaluate_heldout_adversarial,
)
from src.opponent import MAX_PERTURBATIONS, Opponent
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


# ── NSGA-II helpers ───────────────────────────────────────────────────────────


def _jaccard_distance(a_keys, b_keys) -> float:
    a = set(a_keys); b = set(b_keys)
    u = a | b
    if not u:
        return 0.0
    return len(a.symmetric_difference(b)) / len(u)


def _compute_novelty(pop: list, archive: list, k: int = 5) -> list:
    """Novelty = mean Jaccard distance to k nearest neighbors in (pop ∪ archive),
    using white.committed key sets as behavior descriptors."""
    all_descriptors = [list(c.white.committed.keys()) for c in pop]
    all_descriptors += archive
    n_pop = len(pop)
    novelties: list = []
    for i in range(n_pop):
        dists = []
        for j, other in enumerate(all_descriptors):
            if j == i:
                continue
            dists.append(_jaccard_distance(all_descriptors[i], other))
        if not dists:
            novelties.append(0.0)
            continue
        dists.sort()
        kk = min(k, len(dists))
        novelties.append(float(np.mean(dists[:kk])))
    return novelties


def _non_dominated_sort(objectives: list) -> list:
    """Standard fast non-dominated sort (Deb 2002).

    *objectives* is a list of (f1, f2) tuples where we want to MAXIMIZE both.
    Returns a list of fronts (each a list of indices into objectives).
    """
    n = len(objectives)
    S = [[] for _ in range(n)]
    nd = [0] * n
    rank = [0] * n
    fronts: list = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            # p dominates q if p is >= in all and > in at least one.
            pp = objectives[p]; qq = objectives[q]
            p_dom_q = (pp[0] >= qq[0] and pp[1] >= qq[1]
                       and (pp[0] > qq[0] or pp[1] > qq[1]))
            q_dom_p = (qq[0] >= pp[0] and qq[1] >= pp[1]
                       and (qq[0] > pp[0] or qq[1] > pp[1]))
            if p_dom_q:
                S[p].append(q)
            elif q_dom_p:
                nd[p] += 1
        if nd[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front: list = []
        for p in fronts[i]:
            for q in S[p]:
                nd[q] -= 1
                if nd[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return [f for f in fronts if f]


def _crowding_distance(front_idx: list, objectives: list) -> list:
    """Return a list of crowding distances aligned with *front_idx*."""
    m = len(front_idx)
    if m == 0:
        return []
    if m <= 2:
        return [float("inf")] * m
    dists = [0.0] * m
    for obj_i in range(2):  # two objectives
        order = sorted(range(m), key=lambda i: objectives[front_idx[i]][obj_i])
        dists[order[0]] = float("inf")
        dists[order[-1]] = float("inf")
        f_min = objectives[front_idx[order[0]]][obj_i]
        f_max = objectives[front_idx[order[-1]]][obj_i]
        denom = f_max - f_min
        if denom == 0:
            continue
        for k in range(1, m - 1):
            prev_v = objectives[front_idx[order[k - 1]]][obj_i]
            next_v = objectives[front_idx[order[k + 1]]][obj_i]
            dists[order[k]] += (next_v - prev_v) / denom
    return dists


def _nsga2_select(population: list, objectives: list, k: int) -> list:
    """Select *k* individuals from *population* using NSGA-II ranking."""
    fronts = _non_dominated_sort(objectives)
    chosen: list = []
    for front in fronts:
        if len(chosen) + len(front) <= k:
            chosen.extend(front)
        else:
            remain = k - len(chosen)
            cds = _crowding_distance(front, objectives)
            order = sorted(range(len(front)), key=lambda i: cds[i], reverse=True)
            chosen.extend([front[i] for i in order[:remain]])
            break
    return [population[i] for i in chosen]


def _hash_descriptor(keys) -> tuple:
    return tuple(sorted(keys))


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
    """Execute one co-evolutionary GA run.

    *mode* is one of: 'STATIC', 'COEVOLVE_FROZEN', 'COEVOLVE'.
    COEVOLVE_C / COEVOLVE_D method labels are mapped to 'COEVOLVE' by the caller
    (see experiments.py) and distinguished via config flags:
      * use_perturbations : bool  — Tier 1 (richer opponent space)
      * use_nsga2         : bool  — Tier 2 (NSGA-II on repertoires)
      * max_perturbations : int   — cap on opponent perturbation dict size

    Always computes `heldout_adversarial_coevolved` at the end (cheap).
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

    pop_size_r: int = int(config.get("pop_size_repertoires", 30))
    pop_size_o: int = int(config.get("pop_size_opponents", 20))
    n_generations: int = int(config.get("n_generations", 50))
    tournament_size: int = int(config.get("tournament_size", 3))
    crossover_rate: float = float(config.get("crossover_rate", 0.7))
    mutation_rate: float = float(config.get("mutation_rate", 0.3))
    hof_size: int = int(config.get("hof_size", 5))
    novelty_weight: float = float(config.get("novelty_weight", 0.1))

    use_perturbations: bool = bool(config.get("use_perturbations", False))
    use_nsga2: bool = bool(config.get("use_nsga2", False))
    max_perturbations: int = int(config.get("max_perturbations", MAX_PERTURBATIONS))

    # Tier 1 only engages when co-evolution is active.
    opp_graph_for_mut = graph_train if use_perturbations and mode == "COEVOLVE" else None

    # ── Initialisation ─────────────────────────────────────────────────────────

    R_pop: list[Candidate] = []
    half = pop_size_r // 2
    for _ in range(half):
        w = construct_initial(graph_train, "white", BUDGET, rng)
        b = construct_initial(graph_train, "black", BUDGET, rng)
        R_pop.append(Candidate(white=w, black=b, fitness=None, band_scores_cache=None))
    for _ in range(pop_size_r - half):
        w = construct_random(graph_train, "white", BUDGET, rng)
        b = construct_random(graph_train, "black", BUDGET, rng)
        R_pop.append(Candidate(white=w, black=b, fitness=None, band_scores_cache=None))

    if mode == "STATIC":
        O_pop: list[Opponent] = [Opponent.uniform()]
    else:
        O_pop = [Opponent.random(rng) for _ in range(pop_size_o)]

    HoF: list[Opponent] = []
    hof_informativeness: dict[int, float] = {}

    # Novelty archive for NSGA-II (bounded by size 200).
    novelty_archive: list = []
    NOVELTY_ARCHIVE_CAP = 200

    history: list[dict] = []
    last_evaluated_r_pop: list[Candidate] = []

    # ── Helper closure for evaluating one candidate vs one opponent ────────────

    def _eval_cand_vs_opp(cand: Candidate, opp: Opponent, use_cache: bool) -> dict:
        pert = opp.perturbations if (use_perturbations and opp.perturbations) else None
        return evaluate(
            cand,
            opp.mixture,
            config,
            eval_cache_train,
            base_policies_train,
            graph_train,
            use_cache=use_cache and not pert,
            perturbations=pert,
        )

    # ── Generational loop ──────────────────────────────────────────────────────

    for gen in range(n_generations):

        # Step 1: build evaluation set
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

        # ── Step 3: repertoire reproduction ───────────────────────────────────
        if use_nsga2 and mode == "COEVOLVE":
            # Build offspring (same size as parent pop).
            offspring: list[Candidate] = []
            for _ in range(pop_size_r):
                parent_a = _tournament_select(R_pop, rng, tournament_size)
                if rng.random() < crossover_rate:
                    parent_b = _tournament_select(R_pop, rng, tournament_size)
                    child = crossover_candidates(parent_a, parent_b, rng)
                else:
                    child = parent_a.copy()
                    child.fitness = None
                if rng.random() < mutation_rate:
                    child = mutate_candidate(child, rng)
                child.fitness = None
                child.band_scores_cache = None
                child.white_band_scores_cache = None
                child.black_band_scores_cache = None
                offspring.append(child)

            # Evaluate offspring against the same eval_set.
            for cand in offspring:
                fitnesses_per_opp: list[float] = []
                for opp in eval_set:
                    res = _eval_cand_vs_opp(cand, opp, use_cache=True)
                    fitnesses_per_opp.append(res["fitness"])
                cand.fitness = float(np.mean(fitnesses_per_opp))

            # NSGA-II select pop_size_r from (parents + offspring) by
            # (fitness, novelty) both maximised.
            combined = R_pop + offspring
            nov = _compute_novelty(combined, novelty_archive, k=5)
            objectives = [
                (c.fitness if c.fitness is not None else -float("inf"), nov[i])
                for i, c in enumerate(combined)
            ]
            R_pop = _nsga2_select(combined, objectives, pop_size_r)

            # Refresh the novelty archive with distinct descriptors from survivors.
            seen = {_hash_descriptor(d) for d in novelty_archive}
            for c in R_pop:
                h = _hash_descriptor(c.white.committed.keys())
                if h not in seen:
                    novelty_archive.append(list(c.white.committed.keys()))
                    seen.add(h)
            if len(novelty_archive) > NOVELTY_ARCHIVE_CAP:
                drop = len(novelty_archive) - NOVELTY_ARCHIVE_CAP
                novelty_archive = novelty_archive[drop:]
        else:
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

        # ── Steps 4–6: COEVOLVE opponent evolution ────────────────────────────
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

            new_o_pop: list[Opponent] = []
            for _ in range(pop_size_o):
                opp_parent_a = _tournament_select_opp(O_pop, rng, tournament_size)
                if rng.random() < 0.5:
                    opp_parent_b = _tournament_select_opp(O_pop, rng, tournament_size)
                    child_opp = opp_parent_a.crossover(
                        opp_parent_b, rng, max_perturbations=max_perturbations,
                    )
                else:
                    child_opp = Opponent(
                        opp_parent_a.mixture.copy(),
                        perturbations=dict(opp_parent_a.perturbations),
                    )
                    child_opp.fitness = None
                if rng.random() < 0.5:
                    child_opp = child_opp.mutate(
                        rng,
                        graph=opp_graph_for_mut,
                        max_perturbations=max_perturbations,
                    )
                new_o_pop.append(child_opp)
            O_pop = new_o_pop

            # HoF update
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
        hof_mean_informativeness: Optional[float] = (
            float(np.mean(list(hof_informativeness.values())))
            if (mode == "COEVOLVE" and hof_informativeness) else None
        )
        opp_mean_perturbations: Optional[float] = (
            float(np.mean([len(o.perturbations) for o in O_pop]))
            if (mode == "COEVOLVE" and use_perturbations) else None
        )

        history.append({
            "gen": gen,
            "best_training_fitness": best_training_fitness,
            "mean_training_fitness": mean_training_fitness,
            "repertoire_diversity": rep_diversity,
            "opponent_diversity": opp_diversity,
            "hof_size": len(HoF) if mode == "COEVOLVE" else 0,
            "hof_mean_informativeness": hof_mean_informativeness,
            "opp_mean_perturbations": opp_mean_perturbations,
        })

    # ── Final selection + held-out evaluation ─────────────────────────────────

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

    # Tier 3: adversarial held-out — small co-evolutionary probe on held-out graph.
    heldout_adversarial = evaluate_heldout_adversarial(
        best,
        eval_cache_heldout,
        base_policies_train,
        graph_heldout,
        config,
        seed=seed + 10_000,
        n_generations=int(config.get("adv_heldout_generations", 8)),
        pop_size=int(config.get("adv_heldout_pop", 8)),
        max_perturbations=int(config.get("adv_heldout_max_perturbations", 10)),
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
        "heldout_adversarial_coevolved": heldout_adversarial,
        "wall_time_seconds": elapsed,
    }
