"""
Fitness function for a Candidate against an opponent mixture over rating bands.

Walks the position graph from the root, following the player's committed moves
at our-turn nodes and weighting opponent-turn children by the band's empirical
move policy. Aggregates leaf evaluations into a per-band score, then collapses
across bands using a CVaR-weighted robust objective:

    fitness = mean_score + lambda_weight * worst_band_score
"""

from __future__ import annotations

import numpy as np

from src.repertoire import Candidate

BANDS = ["1600-1799", "1800-1999", "2000-2199"]
BUDGET = 20


# ── Walk ─────────────────────────────────────────────────────────────────────


def walk(rep, band: str, eval_cache: dict, base_policies: dict, graph: dict,
         node_fen: str = None) -> float:
    """Expected White-perspective score for *rep* under one band's policy.

    Recursively walks the position tree.  At our-turn nodes follow the committed
    move (or return the cached leaf score if uncommitted).  At opponent-turn
    nodes, take a weighted sum over all children with nonzero policy.  Off-book
    children (not in rep.reached) collapse to their cached score directly.
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
            return eval_cache["scores"][node_fen][band]
        move = rep.committed[node_fen]
        child_info = node["children"].get(move)
        if child_info is None:
            # Held-out graph has the position but never observed this response.
            # Treat as a leaf and fall back to the cached score.
            return eval_cache["scores"][node_fen][band]
        return walk(rep, band, eval_cache, base_policies, graph,
                    child_info["child_fen"])

    policy_at_node = base_policies[band].get(node_fen, {})
    total = 0.0
    for move, child_info in node["children"].items():
        p = policy_at_node.get(move, 0.0)
        if p == 0.0:
            continue
        child_fen = child_info["child_fen"]
        if child_fen in rep.reached:
            total += p * walk(rep, band, eval_cache, base_policies, graph, child_fen)
        else:
            if child_fen in eval_cache["scores"]:
                total += p * eval_cache["scores"][child_fen][band]
            else:
                total += p * eval_cache["prior_mean"]
    return total


# ── Main evaluate ────────────────────────────────────────────────────────────


def evaluate(
    candidate: Candidate,
    opponent_mixture: np.ndarray,   # shape (3,), sums to 1
    config: dict,                   # must contain 'lambda_weight'
    eval_cache: dict,
    base_policies: dict,
    graph: dict,
    use_cache: bool = True,
) -> dict:
    """Compute fitness for a Candidate under one opponent mixture.

    Returns dict with keys: 'mean_score', 'cvar', 'fitness', 'band_scores',
    'white_band_scores', 'black_band_scores'.
    """
    if len(candidate.white.committed) > BUDGET or len(candidate.black.committed) > BUDGET:
        return {
            "mean_score": 0.0,
            "cvar": 0.0,
            "fitness": -float("inf"),
            "band_scores": {},
            "white_band_scores": {},
            "black_band_scores": {},
        }

    cache_hit = use_cache and candidate.band_scores_cache is not None
    if cache_hit:
        band_scores = candidate.band_scores_cache
        white_band_scores = candidate.white_band_scores_cache
        black_band_scores = candidate.black_band_scores_cache
    else:
        band_scores = {}
        white_band_scores = {}
        black_band_scores = {}
        for band in BANDS:
            white_ws = walk(candidate.white, band, eval_cache, base_policies, graph)
            black_ws = walk(candidate.black, band, eval_cache, base_policies, graph)
            black_score_for_player = 1.0 - black_ws  # convert White-perspective -> Black player
            white_band_scores[band] = white_ws
            black_band_scores[band] = black_score_for_player
            band_scores[band] = 0.5 * white_ws + 0.5 * black_score_for_player
        if use_cache:
            candidate.band_scores_cache = band_scores
            candidate.white_band_scores_cache = white_band_scores
            candidate.black_band_scores_cache = black_band_scores

    mean_score = sum(opponent_mixture[i] * band_scores[BANDS[i]] for i in range(3))

    # CVaR with alpha = 1/3 over 3 bands collapses to the single worst band.
    cvar = min(band_scores.values())

    fitness = mean_score + config["lambda_weight"] * cvar

    return {
        "mean_score": mean_score,
        "cvar": cvar,
        "fitness": fitness,
        "band_scores": dict(band_scores),
        "white_band_scores": dict(white_band_scores),
        "black_band_scores": dict(black_band_scores),
    }


# ── Held-out evaluation ──────────────────────────────────────────────────────


def evaluate_heldout(
    candidate: Candidate,
    eval_cache_heldout: dict,
    base_policies_train: dict,
    graph_heldout: dict,
    config: dict,
) -> float:
    """Evaluate the candidate on held-out data under a uniform opponent mixture.

    Uses training base policies (held-out policies aren't built) and the
    held-out eval cache and graph.  Positions absent from graph_heldout are
    treated as leaves via the held-out eval cache (or its prior_mean).
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


# ── Held-out wrapping ────────────────────────────────────────────────────────


class _HeldoutRep:
    __slots__ = ("color", "committed", "reached", "graph")

    def __init__(self, rep, heldout_graph: dict):
        self.color = rep.color
        self.committed = rep.committed
        self.reached = rep.reached
        self.graph = heldout_graph


class _HeldoutCandidate:
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
