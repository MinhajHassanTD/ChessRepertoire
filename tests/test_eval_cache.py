"""
Unit tests for C4 — Evaluation cache (src/eval_cache.py).

Each test corresponds to one acceptance criterion from Section E of BLUEPRINT.md.
Tests build a minimal in-memory graph so they run without any data files.
One integration test uses L2.db if it is present.
"""

import os
import sys
import pickle
import tempfile
import pytest

# Make sure src/ is importable when run from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.eval_cache import (
    build_eval_cache,
    compute_prior_mean,
    save_eval_cache,
    load_eval_cache,
    RATING_BANDS,
    TAU,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_node(fen, ply_depth, total_games, white_wins, draws,
               band_games=None, band_wins=None, band_draws=None):
    """Helper to build a graph node dict matching the C2 schema."""
    black_wins = total_games - white_wins - draws
    score_raw = (white_wins + 0.5 * draws) / total_games if total_games > 0 else 0.5

    band_stats = {}
    for b in RATING_BANDS:
        g = (band_games or {}).get(b, 0)
        w = (band_wins  or {}).get(b, 0)
        d = (band_draws or {}).get(b, 0)
        band_stats[b] = {
            "games": g, "wins": w, "draws": d, "losses": g - w - d
        }

    return {
        "fen": fen,
        "ply_depth": ply_depth,
        "total_games": total_games,
        "score_raw": score_raw,
        "turn": "white",
        "band_stats": band_stats,
        "children": {},
    }


def _make_graph(nodes: list) -> dict:
    return {
        "root_fen": nodes[0]["fen"],
        "nodes": {n["fen"]: n for n in nodes},
    }


# One position with non-zero band stats, one with zero.
FEN_A = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
FEN_B = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -"
FEN_C = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -"


@pytest.fixture
def simple_graph():
    """Graph with three nodes: depth 0, 1, 2."""
    nodes = [
        _make_node(FEN_A, ply_depth=0, total_games=500_000,
                   white_wins=200_000, draws=150_000,
                   band_games={"1600-1799": 100_000, "1800-1999": 120_000, "2000-2199": 80_000},
                   band_wins ={"1600-1799":  38_000, "1800-1999":  45_000, "2000-2199": 32_000},
                   band_draws={"1600-1799":  30_000, "1800-1999":  36_000, "2000-2199": 24_000}),
        _make_node(FEN_B, ply_depth=1, total_games=400_000,
                   white_wins=160_000, draws=120_000,
                   band_games={"1600-1799":  80_000, "1800-1999":  90_000, "2000-2199": 70_000},
                   band_wins ={"1600-1799":  31_000, "1800-1999":  36_000, "2000-2199": 29_000},
                   band_draws={"1600-1799":  24_000, "1800-1999":  27_000, "2000-2199": 21_000}),
        # FEN_C has ZERO band games — must fall back to prior_mean
        _make_node(FEN_C, ply_depth=2, total_games=300_000,
                   white_wins=120_000, draws=90_000),
    ]
    return _make_graph(nodes)


# ── AC1: Every position in the graph has a cache entry for every band ─────────

def test_all_positions_all_bands_present(simple_graph):
    """AC1 — every position in the graph has a score for every band."""
    cache = build_eval_cache(simple_graph)
    for fen in simple_graph["nodes"]:
        assert fen in cache["scores"], f"Missing cache entry for {fen}"
        for band in RATING_BANDS:
            assert band in cache["scores"][fen], (
                f"Missing band {band} for {fen}"
            )


# ── AC2: Zero-games position → shrunk_score == prior_mean exactly ────────────

def test_zero_games_equals_prior_mean(simple_graph):
    """AC2 — a position with 0 band games gets exactly prior_mean."""
    cache = build_eval_cache(simple_graph)
    prior = cache["prior_mean"]
    # FEN_C was constructed with no band games
    for band in RATING_BANDS:
        score = cache["scores"][FEN_C][band]
        assert score == prior, (
            f"Expected prior_mean {prior} for zero-games position, got {score}"
        )


# ── AC3: 1000+ games → shrunk_score within 0.02 of raw empirical ─────────────

def test_high_games_close_to_raw(simple_graph):
    """AC3 — position with 1000+ band games has shrunk_score ≈ raw score."""
    cache = build_eval_cache(simple_graph)
    prior = cache["prior_mean"]

    # FEN_A has 100_000+ games per band — shrinkage should be negligible
    for band in RATING_BANDS:
        stats = simple_graph["nodes"][FEN_A]["band_stats"][band]
        n = stats["games"]
        raw = (stats["wins"] + 0.5 * stats["draws"]) / n
        shrunk = cache["scores"][FEN_A][band]
        assert abs(shrunk - raw) < 0.02, (
            f"Band {band}: shrunk={shrunk:.6f}, raw={raw:.6f}, diff too large"
        )


# ── Extra: prior_mean is stored in the cache ──────────────────────────────────

def test_prior_mean_stored(simple_graph):
    """prior_mean is stored and is a finite float in [0, 1]."""
    cache = build_eval_cache(simple_graph)
    pm = cache["prior_mean"]
    assert isinstance(pm, float)
    assert 0.0 <= pm <= 1.0


# ── Extra: shrinkage formula correctness ─────────────────────────────────────

def test_shrinkage_formula():
    """Verify the exact shrinkage formula: (n*raw + τ*prior) / (n + τ)."""
    # Single-node graph
    band = "1600-1799"
    n, w, d = 50, 20, 15
    raw = (w + 0.5 * d) / n  # 20 + 7.5 = 27.5 / 50 = 0.55

    # prior_mean comes from ply ≤ 4; here only one node at depth 0
    node = _make_node(
        FEN_A, ply_depth=0, total_games=n,
        white_wins=w, draws=d,
        band_games={band: n, "1800-1999": 0, "2000-2199": 0},
        band_wins ={band: w, "1800-1999": 0, "2000-2199": 0},
        band_draws={band: d, "1800-1999": 0, "2000-2199": 0},
    )
    graph = _make_graph([node])
    cache = build_eval_cache(graph)

    prior = cache["prior_mean"]
    expected = (n * raw + TAU * prior) / (n + TAU)
    assert abs(cache["scores"][FEN_A][band] - expected) < 1e-12


# ── Extra: save/load round-trip ───────────────────────────────────────────────

def test_save_load_roundtrip(simple_graph):
    """save_eval_cache / load_eval_cache preserves data exactly."""
    cache = build_eval_cache(simple_graph)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    try:
        save_eval_cache(cache, path)
        loaded = load_eval_cache(path)
        assert loaded["prior_mean"] == cache["prior_mean"]
        for fen in cache["scores"]:
            for band in RATING_BANDS:
                assert loaded["scores"][fen][band] == cache["scores"][fen][band]
    finally:
        os.unlink(path)


# ── Integration: build from L2.db if available ───────────────────────────────

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "L2.db")

@pytest.mark.skipif(not os.path.exists(DB_PATH), reason="L2.db not present")
def test_integration_all_positions_covered():
    """Integration — every graph node has a cache entry for every band."""
    from src.graph import build_graph
    for split in ("train", "heldout"):
        graph = build_graph(DB_PATH, split)
        cache = build_eval_cache(graph)
        for fen in graph["nodes"]:
            assert fen in cache["scores"]
            for band in RATING_BANDS:
                assert band in cache["scores"][fen]


@pytest.mark.skipif(not os.path.exists(DB_PATH), reason="L2.db not present")
def test_integration_prior_mean_reasonable():
    """Integration — prior_mean should be near 0.5 (White's slight advantage)."""
    from src.graph import build_graph
    graph = build_graph(DB_PATH, "train")
    cache = build_eval_cache(graph)
    pm = cache["prior_mean"]
    # Empirically, White scores roughly 0.52–0.56 from the starting position.
    assert 0.40 <= pm <= 0.65, f"prior_mean={pm} seems wrong"
