"""Probe A — per-game trajectory sampling.

The existing held-out fitness averages per-band scores using a weighted sum
over opponent policies at each opp-turn node.  That produces a single
expected score and smooths out tail behaviour.  This module instead *samples*
trajectories one at a time: at each opponent-turn node it draws one move
from the band policy (weighted by policy probability), follows the committed
move at our-turn nodes, and terminates at leaves.  The resulting distribution
of terminal scores exposes tail performance that the averaged metric hides.

Used for Probe A of the testv2 plan.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from src.fitness import BANDS


class _FrozenRep:
    """Thin read-only stand-in for a Repertoire for walking a graph."""
    __slots__ = ("color", "committed", "reached")

    def __init__(self, color: str, committed: dict, reached):
        self.color = color
        self.committed = committed
        self.reached = set(reached)


def _sample_one_trajectory(
    rep: _FrozenRep,
    band: str,
    eval_cache: dict,
    base_policies: dict,
    graph: dict,
    rng: np.random.Generator,
) -> float:
    """Sample a single terminal score by walking one path through the graph."""
    fen = graph["root_fen"]
    while True:
        node = graph["nodes"].get(fen)
        if node is None or not node.get("children"):
            return eval_cache["scores"].get(fen, {}).get(band, eval_cache["prior_mean"])
        is_our_turn = (node["turn"] == rep.color)
        if is_our_turn:
            if fen in rep.committed:
                move = rep.committed[fen]
                child = node["children"].get(move)
                if child is None:
                    return eval_cache["scores"][fen][band]
                fen = child["child_fen"]
            else:
                return eval_cache["scores"][fen][band]
        else:
            policy = base_policies[band].get(fen, {})
            if not policy:
                return eval_cache["scores"][fen][band]
            moves = list(policy.keys())
            probs = np.array(list(policy.values()), dtype=float)
            s = probs.sum()
            if s <= 0:
                return eval_cache["scores"][fen][band]
            probs = probs / s
            move = rng.choice(moves, p=probs)
            child = node["children"].get(move)
            if child is None:
                return eval_cache["scores"][fen][band]
            child_fen = child["child_fen"]
            if child_fen in rep.reached:
                fen = child_fen
            else:
                # Off-book: use child's cached score as trajectory outcome.
                return eval_cache["scores"].get(child_fen, {}).get(
                    band, eval_cache["prior_mean"]
                )


def sample_trajectories(
    candidate_ser: dict,
    eval_cache: dict,
    base_policies: dict,
    graph: dict,
    bands: Iterable[str] = BANDS,
    n_samples_per_band: int = 400,
    seed: int = 0,
) -> dict:
    """Return per-band arrays of sampled terminal scores for the whole candidate.

    A candidate plays White with its white repertoire and Black with its black
    repertoire.  For each band we interleave samples: half-white, half-black,
    then combine into a single player-perspective distribution.

    Returns dict keyed by band -> np.ndarray of shape (n_samples_per_band,).
    """
    rng = np.random.default_rng(seed)
    white = _FrozenRep("white", candidate_ser["white_committed"], candidate_ser["white_reached"])
    black = _FrozenRep("black", candidate_ser["black_committed"], candidate_ser["black_reached"])

    out: dict = {}
    half = n_samples_per_band // 2
    rest = n_samples_per_band - half
    for band in bands:
        vals = np.empty(n_samples_per_band, dtype=float)
        for i in range(half):
            vals[i] = _sample_one_trajectory(white, band, eval_cache, base_policies, graph, rng)
        for j in range(rest):
            black_ws = _sample_one_trajectory(black, band, eval_cache, base_policies, graph, rng)
            vals[half + j] = 1.0 - black_ws
        out[band] = vals
    return out


def summarise_trajectories(traj_by_band: dict) -> dict:
    """Compute scalar summary metrics from the band-indexed trajectory dict.

    Returns:
      - per_game_mean          : mean across all samples across all bands
      - per_game_worst_decile  : 10th percentile across the pooled distribution
      - per_game_worst_band_mean: mean in the band with the lowest mean
      - per_game_std           : std across the pooled distribution
    """
    pooled = np.concatenate(list(traj_by_band.values()))
    band_means = {b: float(v.mean()) for b, v in traj_by_band.items()}
    worst_band = min(band_means, key=band_means.get)
    return {
        "per_game_mean":            float(pooled.mean()),
        "per_game_worst_decile":    float(np.quantile(pooled, 0.10)),
        "per_game_worst_band_mean": float(band_means[worst_band]),
        "per_game_std":             float(pooled.std()),
    }
