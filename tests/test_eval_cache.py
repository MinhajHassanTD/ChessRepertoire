"""Tests for C4 — Evaluation cache (src/eval_cache.py)."""

import pickle
import tempfile

import pytest

from src.config import RATING_BANDS
from src.eval_cache import (
    build_eval_cache,
    compute_prior_mean,
    load_eval_cache,
    save_eval_cache,
)


def _make_node(fen, ply_depth, total_games, white_wins, draws,
               band_games=None, band_wins=None, band_draws=None):
    score_raw = (white_wins + 0.5 * draws) / total_games if total_games > 0 else 0.5
    band_stats = {}
    for b in RATING_BANDS:
        g = (band_games or {}).get(b, 0)
        w = (band_wins or {}).get(b, 0)
        d = (band_draws or {}).get(b, 0)
        band_stats[b] = {"games": g, "wins": w, "draws": d, "losses": g - w - d}
    return {
        "fen": fen, "ply_depth": ply_depth, "total_games": total_games,
        "score_raw": score_raw, "turn": "white",
        "band_stats": band_stats, "children": {},
    }


def _make_graph(nodes):
    return {"root_fen": nodes[0]["fen"], "nodes": {n["fen"]: n for n in nodes}}


FEN_A = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
FEN_B = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -"
FEN_C = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -"


@pytest.fixture
def simple_graph():
    b1, b2, b3 = RATING_BANDS
    return _make_graph([
        _make_node(FEN_A, 0, 500_000, 200_000, 150_000,
                   band_games={b1: 100_000, b2: 120_000, b3: 80_000},
                   band_wins={b1: 38_000, b2: 45_000, b3: 32_000},
                   band_draws={b1: 30_000, b2: 36_000, b3: 24_000}),
        _make_node(FEN_B, 1, 400_000, 160_000, 120_000,
                   band_games={b1: 80_000, b2: 90_000, b3: 70_000},
                   band_wins={b1: 31_000, b2: 36_000, b3: 29_000},
                   band_draws={b1: 24_000, b2: 27_000, b3: 21_000}),
        _make_node(FEN_C, 2, 300_000, 120_000, 90_000),
    ])


def test_all_positions_all_bands_present(simple_graph):
    cache = build_eval_cache(simple_graph)
    for fen in simple_graph["nodes"]:
        assert fen in cache["scores"]
        for band in RATING_BANDS:
            assert band in cache["scores"][fen]


def test_zero_games_equals_prior_mean(simple_graph):
    cache = build_eval_cache(simple_graph)
    prior = cache["prior_mean"]
    for band in RATING_BANDS:
        assert cache["scores"][FEN_C][band] == prior


def test_raw_score_when_band_has_games(simple_graph):
    cache = build_eval_cache(simple_graph)
    for band in RATING_BANDS:
        stats = simple_graph["nodes"][FEN_A]["band_stats"][band]
        n = stats["games"]
        raw = (stats["wins"] + 0.5 * stats["draws"]) / n
        assert abs(cache["scores"][FEN_A][band] - raw) < 1e-12


def test_prior_mean_in_range(simple_graph):
    pm = compute_prior_mean(simple_graph)
    assert isinstance(pm, float)
    assert 0.0 <= pm <= 1.0


def test_prior_mean_empty_graph_defaults_to_half():
    graph = {"root_fen": "x", "nodes": {}}
    assert compute_prior_mean(graph) == 0.5


def test_save_load_roundtrip(simple_graph, tmp_path):
    cache = build_eval_cache(simple_graph)
    path = str(tmp_path / "cache.pkl")
    save_eval_cache(cache, path)
    loaded = load_eval_cache(path)
    assert loaded["prior_mean"] == cache["prior_mean"]
    assert loaded["scores"] == cache["scores"]
