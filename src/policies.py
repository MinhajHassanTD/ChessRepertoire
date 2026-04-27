"""
C3 — Rating-band base policies
Compute per-band move distributions from the training graph,
with additive smoothing toward the aggregate distribution.
"""

import pickle
import logging
from itertools import combinations
from typing import Dict

from src.config import SMOOTHING_ALPHA as ALPHA, RATING_BANDS, BAND_SEPARATION_MIN_TV, BAND_SEPARATION_MIN_GAMES

logger = logging.getLogger(__name__)


# ── Core computation ──────────────────────────────────────────────────────────

def compute_policies(graph: dict) -> dict:
    """
    Compute smoothed per-band move distributions for every position in *graph*.

    Formula (Section E, C3):
        policy[b][p][m] = (band_count[m] + α * aggregate_prob[m]) / (band_total + α)

    When band_total == 0  →  policy[b][p][m] = aggregate_prob[m]  (fallback).

    Returns:
        base_policies = {
            '1600-1799': { '<fen>': {'<move_uci>': float, ...}, ... },
            '1800-1999': { ... },
            '2000-2199': { ... },
        }
    """
    base_policies: Dict[str, Dict[str, Dict[str, float]]] = {b: {} for b in RATING_BANDS}

    nodes = graph["nodes"]

    for fen, node in nodes.items():
        children = node.get("children", {})
        if not children:
            continue

        # ── aggregate distribution ────────────────────────────────────────────
        aggregate_total = sum(c["aggregate_count"] for c in children.values())

        if aggregate_total == 0:
            # No aggregate data at all — uniform fallback
            n_moves = len(children)
            agg_prob = {m: 1.0 / n_moves for m in children}
        else:
            agg_prob = {
                m: c["aggregate_count"] / aggregate_total
                for m, c in children.items()
            }

        # ── per-band policy (no smoothing — all positions have band_total > 0) ──
        for band in RATING_BANDS:
            band_total = sum(c["band_counts"][band] for c in children.values())

            policy: Dict[str, float] = {}
            for m, c in children.items():
                band_count = c["band_counts"][band]
                # numerator = band_count + ALPHA * agg_prob[m]  # Laplace smoothing (disabled)
                # denominator = band_total + ALPHA               # Laplace smoothing (disabled)
                policy[m] = band_count / band_total if band_total > 0 else agg_prob[m]

            base_policies[band][fen] = policy

    return base_policies


def band_separation_check(base_policies: dict, graph: dict) -> float:
    """
    Compute mean total-variation distance between every pair of band policies
    across positions with total_games >= 200.

    TV(p, q) = 0.5 * sum |p[m] - q[m]|

    Logs the mean TV distance and warns if below 0.05.
    Returns the mean TV distance across all three pairs.
    """
    nodes = graph["nodes"]
    qualifying_fens = [
        fen for fen, node in nodes.items()
        if node.get("total_games", 0) >= BAND_SEPARATION_MIN_GAMES
        and node.get("children")
    ]

    pairs = list(combinations(RATING_BANDS, 2))

    pair_tvs = {pair: [] for pair in pairs}

    for fen in qualifying_fens:
        for b1, b2 in pairs:
            p1 = base_policies[b1].get(fen)
            p2 = base_policies[b2].get(fen)
            if p1 is None or p2 is None:
                continue
            moves = set(p1) | set(p2)
            tv = 0.5 * sum(abs(p1.get(m, 0.0) - p2.get(m, 0.0)) for m in moves)
            pair_tvs[(b1, b2)].append(tv)

    pair_means = {}
    for pair, tvs in pair_tvs.items():
        pair_means[pair] = sum(tvs) / len(tvs) if tvs else 0.0

    mean_tv = sum(pair_means.values()) / len(pair_means) if pair_means else 0.0

    print(f"[C3] Band separation — mean TV distances:")
    for (b1, b2), mv in pair_means.items():
        print(f"     {b1} vs {b2}: {mv:.4f}")
    print(f"[C3] Overall mean TV distance: {mean_tv:.4f}")

    if mean_tv < BAND_SEPARATION_MIN_TV:
        logger.warning(
            "[C3] Mean TV distance %.4f is below 0.05 threshold. "
            "Bands may be too similar to support a meaningful CVaR story.",
            mean_tv,
        )
    else:
        print(f"[C3] Band separation check PASSED (mean TV = {mean_tv:.4f} >= 0.05).")

    return mean_tv


# ── I/O helpers ───────────────────────────────────────────────────────────────

def save_policies(base_policies: dict, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(base_policies, f)
    print(f"[C3] Saved base_policies to {path}")


def load_policies(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.graph import load_graph

    GRAPH_PATH = "data/graph_train.pkl"
    OUTPUT_PATH = "data/base_policies.pkl"

    print(f"[C3] Loading training graph from {GRAPH_PATH} ...")
    graph = load_graph(GRAPH_PATH)

    print("[C3] Computing base policies ...")
    base_policies = compute_policies(graph)

    mean_tv = band_separation_check(base_policies, graph)

    save_policies(base_policies, OUTPUT_PATH)
    print("[C3] Done.")
