"""
C4 — Evaluation cache (src/eval_cache.py)
Precompute empirical-Bayes-shrunk expected scores for every position in each
graph, for each rating band.
"""

import pickle

# ── Constants ─────────────────────────────────────────────────────────────────

RATING_BANDS = ("1600-1799", "1800-1999", "2000-2199")
TAU = 20  # empirical Bayes prior strength (Section A)
PRIOR_MAX_PLY = 4  # positions at ply_depth <= 4 used for prior mean


# ── Core functions ─────────────────────────────────────────────────────────────

def compute_prior_mean(graph: dict) -> float:
    """Compute the aggregate-weighted prior mean score for early positions.

    Only positions at ply_depth <= PRIOR_MAX_PLY contribute.  Score is
    White's expected score: (white_wins + 0.5 * draws) / total_games,
    already stored as score_raw on each node.
    """
    numerator = 0.0
    denominator = 0.0
    for node in graph["nodes"].values():
        if node["ply_depth"] <= PRIOR_MAX_PLY and node["total_games"] > 0:
            numerator += node["total_games"] * node["score_raw"]
            denominator += node["total_games"]
    if denominator == 0.0:
        return 0.5
    return numerator / denominator


def build_eval_cache(graph: dict) -> dict:
    """Build the eval cache for a single graph (train or heldout).

    Returns:
        {
            'prior_mean': float,
            'scores': {
                '<fen>': {'1600-1799': float, '1800-1999': float, '2000-2199': float},
                ...
            }
        }
    All scores are from White's perspective.
    """
    prior_mean = compute_prior_mean(graph)
    scores: dict = {}

    for fen, node in graph["nodes"].items():
        band_scores: dict = {}
        for band in RATING_BANDS:
            stats = node["band_stats"][band]
            n = stats["games"]
            if n == 0:
                shrunk = prior_mean
            else:
                raw = (stats["wins"] + 0.5 * stats["draws"]) / n
                shrunk = (n * raw + TAU * prior_mean) / (n + TAU)
            band_scores[band] = shrunk
        scores[fen] = band_scores

    return {"prior_mean": prior_mean, "scores": scores}


# ── Persistence ───────────────────────────────────────────────────────────────

def save_eval_cache(cache: dict, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_eval_cache(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from src.graph import load_graph

    for split, graph_path, out_path in [
        ("train",   "data/graph_train.pkl",   "data/eval_cache_train.pkl"),
        ("heldout", "data/graph_heldout.pkl", "data/eval_cache_heldout.pkl"),
    ]:
        print(f"Building eval cache for {split} …", flush=True)
        g = load_graph(graph_path)
        cache = build_eval_cache(g)
        save_eval_cache(cache, out_path)
        print(f"  prior_mean = {cache['prior_mean']:.6f}")
        print(f"  {len(cache['scores'])} positions cached → {out_path}")
