"""
C6 — Fitness function (src/fitness.py)
Compute the fitness of a Candidate given an opponent mixture.

Depends on: C2 (graph), C3 (base_policies), C4 (eval_cache), C5 (repertoire).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.repertoire import Candidate

# ── Constants ─────────────────────────────────────────────────────────────────

BANDS = ["1600-1799", "1800-1999", "2000-2199"]
BUDGET = 20


# ── Walk ──────────────────────────────────────────────────────────────────────

def walk(rep, band: str, eval_cache: dict, base_policies: dict, graph: dict,
         node_fen: str = None,
         perturbations: Optional[dict] = None) -> float:
    """Compute the expected White-perspective score for a repertoire under one
    band's policy, recursively walking the position tree.

    At our-turn nodes with no committed move (leaf), return the eval_cache score.
    At our-turn nodes with a committed move, follow that move.
    At opponent-turn nodes, take a weighted sum over all moves with nonzero policy.
    Off-book child positions (not in rep.reached) use eval_cache directly.

    If *perturbations* is provided (dict[fen -> (move_uci, strength)]), the
    opponent-turn policy at perturbed FENs is blended toward a delta on the
    target move:  effective[m] = (1-s)*base[m] + s*1[m == target].
    """
    if node_fen is None:
        node_fen = graph["root_fen"]

    node = graph["nodes"].get(node_fen)
    if node is None:
        if node_fen in eval_cache["scores"]:
            return eval_cache["scores"][node_fen][band]
        return eval_cache["prior_mean"]
    is_our_turn = (node["turn"] == rep.color)

    if is_our_turn:
        if node_fen not in rep.committed:
            # Leaf: return cached score
            return eval_cache["scores"][node_fen][band]
        move = rep.committed[node_fen]
        child_fen = node["children"][move]["child_fen"]
        return walk(rep, band, eval_cache, base_policies, graph, child_fen,
                    perturbations=perturbations)
    else:
        # Opponent turn: weighted sum over all moves with nonzero policy
        policy_at_node = base_policies[band].get(node_fen, {})

        # Resolve perturbation (if any) at this FEN.
        pert = None
        if perturbations is not None:
            pert = perturbations.get(node_fen)

        # Build effective policy. Cheap path when no perturbation applies.
        if pert is None:
            effective = policy_at_node
        else:
            target_move, s = pert
            # Only apply if the perturbation targets a move that exists at this FEN.
            if target_move not in node["children"]:
                effective = policy_at_node
            else:
                effective = {}
                # Blend each base-policy entry with the delta on target_move.
                for mv in node["children"]:
                    base_p = policy_at_node.get(mv, 0.0)
                    delta = 1.0 if mv == target_move else 0.0
                    p = (1.0 - s) * base_p + s * delta
                    if p > 0.0:
                        effective[mv] = p

        total = 0.0
        for move, child_info in node["children"].items():
            p = effective.get(move, 0.0)
            if p == 0.0:
                continue
            child_fen = child_info["child_fen"]
            if child_fen in rep.reached:
                total += p * walk(rep, band, eval_cache, base_policies, graph,
                                  child_fen, perturbations=perturbations)
            else:
                # Off-book: use eval_cache score for child position
                if child_fen in eval_cache["scores"]:
                    total += p * eval_cache["scores"][child_fen][band]
                else:
                    # Child not in cache at all — use prior mean as fallback
                    total += p * eval_cache["prior_mean"]
        return total


# ── Main evaluate ─────────────────────────────────────────────────────────────

def evaluate(
    candidate: Candidate,
    opponent_mixture: np.ndarray,   # length 3, sums to 1
    config: dict,                   # must contain 'lambda_weight' and 'alpha'
    eval_cache: dict,
    base_policies: dict,
    graph: dict,
    use_cache: bool = True,
    perturbations: Optional[dict] = None,
) -> dict:
    """Compute fitness for a Candidate given an opponent mixture.

    Returns dict with keys: 'mean_score', 'cvar', 'fitness', 'band_scores'.

    When *perturbations* is non-empty, band_scores depend on the opponent and
    the per-candidate cache is bypassed (neither read nor written).
    """
    # Budget check
    if len(candidate.white.committed) > BUDGET or len(candidate.black.committed) > BUDGET:
        return {
            "mean_score": 0.0,
            "cvar": 0.0,
            "fitness": -float("inf"),
            "band_scores": {},
        }

    has_perts = bool(perturbations)
    cache_ok = use_cache and not has_perts

    # Compute per-band scores (cached on candidate for shared sampling)
    if cache_ok and candidate.band_scores_cache is not None:
        band_scores = candidate.band_scores_cache
        white_band_scores = candidate.white_band_scores_cache
        black_band_scores = candidate.black_band_scores_cache
    else:
        band_scores = {}
        white_band_scores = {}
        black_band_scores = {}
        for band in BANDS:
            white_ws = walk(candidate.white, band, eval_cache, base_policies, graph,
                            perturbations=perturbations)
            black_ws = walk(candidate.black, band, eval_cache, base_policies, graph,
                            perturbations=perturbations)
            # White walk: White-perspective score directly
            # Black walk: White-perspective score; convert for Black player
            black_score_for_player = 1.0 - black_ws
            white_band_scores[band] = white_ws
            black_band_scores[band] = black_score_for_player
            band_scores[band] = 0.5 * white_ws + 0.5 * black_score_for_player
        if cache_ok:
            candidate.band_scores_cache = band_scores
            candidate.white_band_scores_cache = white_band_scores
            candidate.black_band_scores_cache = black_band_scores

    # Mean weighted by opponent mixture
    mean_score = sum(
        opponent_mixture[i] * band_scores[BANDS[i]] for i in range(3)
    )

    # CVaR: with 3 bands and alpha=1/3, this is worst single band
    sorted_scores = sorted(band_scores.values())
    cvar = sorted_scores[0]

    fitness = mean_score + config["lambda_weight"] * cvar

    return {
        "mean_score": mean_score,
        "cvar": cvar,
        "fitness": fitness,
        "band_scores": dict(band_scores),
        "white_band_scores": dict(white_band_scores),
        "black_band_scores": dict(black_band_scores),
    }


# ── Held-out evaluation ───────────────────────────────────────────────────────

def evaluate_heldout(
    candidate: Candidate,
    eval_cache_heldout: dict,
    base_policies_train: dict,
    graph_heldout: dict,
    config: dict,
) -> float:
    """Evaluate the candidate on held-out data under a uniform opponent mixture.

    Uses training base policies (held-out policies aren't built) and the
    held-out eval cache and graph.  For positions in the candidate that don't
    exist in graph_heldout, treat as leaves and use eval_cache_heldout if
    available, otherwise fall back to the held-out prior_mean.
    """
    uniform_mixture = np.ones(3) / 3.0

    heldout_cand = _wrap_for_heldout(candidate, graph_heldout)

    result = evaluate(
        heldout_cand,
        uniform_mixture,
        config,
        eval_cache_heldout,
        base_policies_train,
        graph_heldout,
        use_cache=False,
    )
    return result["fitness"]


def evaluate_heldout_adversarial(
    candidate: Candidate,
    eval_cache_heldout: dict,
    base_policies_train: dict,
    graph_heldout: dict,
    config: dict,
    seed: int = 42,
    n_generations: int = 8,
    pop_size: int = 8,
    max_perturbations: int = 10,
) -> float:
    """Co-evolve a small adversarial opponent population on the held-out graph
    against a single fixed candidate, returning the *worst* fitness observed.

    This is a strict generalization of `evaluate_heldout`: instead of pinning
    the opponent to the uniform mixture, we let the adversary search for the
    weakest spot in the candidate's repertoire (within a small budget).

    Lower return value = candidate is more exploitable.
    """
    # Lazy import to avoid a circular dependency at module load time.
    from src.opponent import Opponent

    rng = np.random.default_rng(seed)
    heldout_cand = _wrap_for_heldout(candidate, graph_heldout)

    O_pop: list = [Opponent.random(rng) for _ in range(pop_size)]

    worst_fitness = float("inf")

    for _ in range(n_generations):
        # Evaluate every opponent against the (single) candidate
        fitnesses: list = []
        for opp in O_pop:
            res = evaluate(
                heldout_cand,
                opp.mixture,
                config,
                eval_cache_heldout,
                base_policies_train,
                graph_heldout,
                use_cache=False,
                perturbations=opp.perturbations or None,
            )
            f = res["fitness"]
            opp.fitness = -f  # adversary minimises candidate's fitness
            fitnesses.append(f)
            if f < worst_fitness:
                worst_fitness = f

        # Tournament select + mutate / crossover for next generation
        new_pop: list = []
        for _ in range(pop_size):
            i, j = rng.choice(len(O_pop), size=2, replace=False)
            parent_a = O_pop[i] if O_pop[i].fitness >= O_pop[j].fitness else O_pop[j]
            if rng.random() < 0.5:
                k, l = rng.choice(len(O_pop), size=2, replace=False)
                parent_b = O_pop[k] if O_pop[k].fitness >= O_pop[l].fitness else O_pop[l]
                child = parent_a.crossover(parent_b, rng,
                                           max_perturbations=max_perturbations)
            else:
                child = Opponent(parent_a.mixture.copy(),
                                 perturbations=dict(parent_a.perturbations))
            if rng.random() < 0.7:
                child = child.mutate(rng, graph=graph_heldout,
                                     max_perturbations=max_perturbations)
            new_pop.append(child)
        O_pop = new_pop

    return float(worst_fitness)


# ── Internal: held-out wrapping ───────────────────────────────────────────────

class _HeldoutRep:
    """Thin wrapper: same committed/reached as a Repertoire, but with an
    explicit graph reference for held-out walking."""
    __slots__ = ("color", "committed", "reached", "graph")

    def __init__(self, rep, heldout_graph: dict):
        self.color = rep.color
        self.committed = rep.committed
        self.reached = rep.reached
        self.graph = heldout_graph


class _HeldoutCandidate:
    """Stand-in for Candidate: only the fields that fitness.evaluate touches."""
    __slots__ = (
        "white", "black", "fitness",
        "band_scores_cache",
        "white_band_scores_cache",
        "black_band_scores_cache",
    )

    def __init__(self, white, black):
        self.white = white
        self.black = black
        self.fitness = None
        self.band_scores_cache = None
        self.white_band_scores_cache = None
        self.black_band_scores_cache = None


def _wrap_for_heldout(candidate: Candidate, graph_heldout: dict) -> "_HeldoutCandidate":
    return _HeldoutCandidate(
        _HeldoutRep(candidate.white, graph_heldout),
        _HeldoutRep(candidate.black, graph_heldout),
    )
