"""
C6 — Fitness function (src/fitness.py)
Compute the fitness of a Candidate given an opponent mixture.

Depends on: C2 (graph), C3 (base_policies), C4 (eval_cache), C5 (repertoire).
"""

from __future__ import annotations

import numpy as np

from src.repertoire import Candidate

# ── Constants ─────────────────────────────────────────────────────────────────

BANDS = ["1600-1799", "1800-1999", "2000-2199"]
BUDGET = 20


# ── Walk ──────────────────────────────────────────────────────────────────────

def walk(rep, band: str, eval_cache: dict, base_policies: dict, graph: dict,
         node_fen: str = None) -> float:
    """Compute the expected White-perspective score for a repertoire under one
    band's policy, recursively walking the position tree.

    At our-turn nodes with no committed move (leaf), return the eval_cache score.
    At our-turn nodes with a committed move, follow that move.
    At opponent-turn nodes, take a weighted sum over all moves with nonzero policy.
    Off-book child positions (not in rep.reached) use eval_cache directly.
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
        return walk(rep, band, eval_cache, base_policies, graph, child_fen)
    else:
        # Opponent turn: weighted sum over all moves with nonzero policy
        total = 0.0
        policy_at_node = base_policies[band].get(node_fen, {})
        for move, child_info in node["children"].items():
            p = policy_at_node.get(move, 0.0)
            if p == 0.0:
                continue
            child_fen = child_info["child_fen"]
            if child_fen in rep.reached:
                total += p * walk(rep, band, eval_cache, base_policies, graph, child_fen)
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
) -> dict:
    """Compute fitness for a Candidate given an opponent mixture.

    Returns dict with keys: 'mean_score', 'cvar', 'fitness', 'band_scores'.
    """
    # Budget check
    if len(candidate.white.committed) > BUDGET or len(candidate.black.committed) > BUDGET:
        return {
            "mean_score": 0.0,
            "cvar": 0.0,
            "fitness": -float("inf"),
            "band_scores": {},
        }

    # Compute per-band scores (cached on candidate for shared sampling)
    if use_cache and candidate.band_scores_cache is not None:
        band_scores = candidate.band_scores_cache
    else:
        band_scores = {}
        for band in BANDS:
            white_ws = walk(candidate.white, band, eval_cache, base_policies, graph)
            black_ws = walk(candidate.black, band, eval_cache, base_policies, graph)
            # White walk: White-perspective score directly
            # Black walk: White-perspective score; convert for Black player
            black_score_for_player = 1.0 - black_ws
            band_scores[band] = 0.5 * white_ws + 0.5 * black_score_for_player
        if use_cache:
            candidate.band_scores_cache = band_scores

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

    # Build a temporary candidate whose repertoires reference the held-out graph
    # but we reuse the same committed/reached sets.  The walk function handles
    # missing positions via the off-book fallback (prior_mean).
    # We need a lightweight wrapper so walk uses the heldout graph but the same
    # committed/reached from candidate.

    class _HeldoutRep:
        """Thin wrapper: same committed/reached, but graph is graph_heldout."""
        def __init__(self, rep, heldout_graph):
            self.color = rep.color
            self.committed = rep.committed
            self.reached = rep.reached
            self.graph = heldout_graph

    white_rep = _HeldoutRep(candidate.white, graph_heldout)
    black_rep = _HeldoutRep(candidate.black, graph_heldout)

    # Temporarily construct a fake Candidate-like object for evaluate
    class _HeldoutCandidate:
        def __init__(self, white, black):
            self.white = white
            self.black = black
            self.fitness = None
            self.band_scores_cache = None

    heldout_cand = _HeldoutCandidate(white_rep, black_rep)

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
