"""testv2 — per-style evaluation cache (training side).

Mirrors src/eval_cache.py but keys per-FEN scores by **style** instead of
rating band. Since the raw Lichess data has no per-style win/draw/loss
statistics (styles are synthetic, defined in src/style_policies.py), all
three styles share the same empirical-Bayes-shrunk aggregate score at each
FEN. Styles affect only the *trajectory distribution* over moves, not the
*evaluation at leaves*.

This preserves the full shape expected by src/fitness.walk (which indexes
`eval_cache["scores"][fen][key]`), so no walk-side changes are needed.
"""
from __future__ import annotations

import os
import pickle

from src.eval_cache import TAU, compute_prior_mean
from src.graph import load_graph
from src.style_policies import STYLES


def build_style_eval_cache(graph: dict) -> dict:
    """Build the style-keyed eval cache for *graph*.

    All styles share the same per-FEN score, computed as empirical-Bayes
    shrinkage of the aggregate (band-pooled) raw score toward the prior mean.
    """
    prior_mean = compute_prior_mean(graph)
    scores: dict = {}

    for fen, node in graph["nodes"].items():
        n = node["total_games"]
        if n == 0:
            shrunk = prior_mean
        else:
            raw = node["score_raw"]
            shrunk = (n * raw + TAU * prior_mean) / (n + TAU)
        scores[fen] = {style: shrunk for style in STYLES}

    return {"prior_mean": prior_mean, "scores": scores}


def save_style_eval_cache(cache: dict, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_style_eval_cache(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    data_dir = "data"
    graph_path = os.path.join(data_dir, "graph_train.pkl")
    out_path = os.path.join(data_dir, "style_eval_cache_train.pkl")

    print(f"Loading {graph_path} ...")
    graph = load_graph(graph_path)
    print("Building style eval cache ...")
    cache = build_style_eval_cache(graph)
    save_style_eval_cache(cache, out_path)
    print(f"  prior_mean = {cache['prior_mean']:.6f}")
    print(f"  {len(cache['scores'])} positions cached -> {out_path}")
