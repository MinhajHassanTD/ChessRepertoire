"""Tests for C3 — Rating-band base policies (src/policies.py)."""

import pytest

from src.config import RATING_BANDS
from src.policies import (
    band_separation_check,
    compute_policies,
    load_policies,
    save_policies,
)


def _make_minimal_graph(children_spec: dict) -> dict:
    """Build a tiny single-node graph from a children spec."""
    return {
        "root_fen": "start",
        "nodes": {
            "start": {
                "fen": "start", "ply_depth": 0,
                "total_games": sum(v["aggregate_count"] for v in children_spec.values()),
                "score_raw": 0.5, "turn": "white",
                "band_stats": {b: {"games": 0, "wins": 0, "draws": 0, "losses": 0}
                               for b in RATING_BANDS},
                "children": {
                    uci: {
                        "child_fen": f"child_{uci}",
                        "move_san": uci,
                        "aggregate_count": spec["aggregate_count"],
                        "band_counts": spec["band_counts"],
                    }
                    for uci, spec in children_spec.items()
                },
            }
        },
    }


def test_distributions_sum_to_one():
    b1, b2, b3 = RATING_BANDS
    spec = {
        "e2e4": {"aggregate_count": 600, "band_counts": {b1: 100, b2: 200, b3: 300}},
        "d2d4": {"aggregate_count": 300, "band_counts": {b1: 80,  b2: 120, b3: 100}},
        "c2c4": {"aggregate_count": 100, "band_counts": {b1: 20,  b2: 30,  b3: 50}},
    }
    policies = compute_policies(_make_minimal_graph(spec))
    for band in RATING_BANDS:
        for fen, dist in policies[band].items():
            assert abs(sum(dist.values()) - 1.0) < 1e-9


def test_zero_band_falls_back_to_aggregate():
    """When a band has zero games at a position, policy equals the aggregate distribution."""
    b1, b2, b3 = RATING_BANDS
    spec = {
        "e2e4": {"aggregate_count": 600, "band_counts": {b1: 0, b2: 200, b3: 300}},
        "d2d4": {"aggregate_count": 400, "band_counts": {b1: 0, b2: 100, b3: 100}},
    }
    policies = compute_policies(_make_minimal_graph(spec))
    dist = policies[b1]["start"]
    assert abs(dist["e2e4"] - 0.6) < 1e-9
    assert abs(dist["d2d4"] - 0.4) < 1e-9


def test_band_with_games_uses_band_frequency():
    """When a band has games, its policy is the band-specific frequency."""
    b1, b2, b3 = RATING_BANDS
    spec = {
        "e2e4": {"aggregate_count": 600, "band_counts": {b1: 80, b2: 0, b3: 0}},
        "d2d4": {"aggregate_count": 400, "band_counts": {b1: 20, b2: 0, b3: 0}},
    }
    policies = compute_policies(_make_minimal_graph(spec))
    dist = policies[b1]["start"]
    assert abs(dist["e2e4"] - 0.80) < 1e-9
    assert abs(dist["d2d4"] - 0.20) < 1e-9


def test_band_separation_returns_float():
    """band_separation_check returns a float; TV distance computed correctly."""
    b1, b2, b3 = RATING_BANDS
    policies = {
        b1: {"pos1": {"e2e4": 1.0, "d2d4": 0.0}},
        b2: {"pos1": {"e2e4": 0.0, "d2d4": 1.0}},
        b3: {"pos1": {"e2e4": 0.0, "d2d4": 1.0}},
    }
    graph = {
        "root_fen": "pos1",
        "nodes": {
            "pos1": {
                "fen": "pos1", "ply_depth": 0, "total_games": 1000,
                "score_raw": 0.5, "turn": "white",
                "band_stats": {b: {"games": 0, "wins": 0, "draws": 0, "losses": 0}
                               for b in RATING_BANDS},
                "children": {
                    "e2e4": {"child_fen": "c1", "move_san": "e4",
                             "aggregate_count": 500,
                             "band_counts": {b: 100 for b in RATING_BANDS}},
                },
            }
        },
    }
    mean_tv = band_separation_check(policies, graph)
    # TV(b1,b2) = 1.0; TV(b1,b3) = 1.0; TV(b2,b3) = 0.0 -> mean = 2/3
    assert isinstance(mean_tv, float)
    assert abs(mean_tv - 2.0 / 3.0) < 1e-9


def test_save_load_roundtrip(tmp_path):
    b1, b2, b3 = RATING_BANDS
    spec = {
        "e2e4": {"aggregate_count": 600, "band_counts": {b1: 100, b2: 200, b3: 300}},
        "d2d4": {"aggregate_count": 400, "band_counts": {b1: 80,  b2: 120, b3: 200}},
    }
    policies = compute_policies(_make_minimal_graph(spec))
    path = str(tmp_path / "policies.pkl")
    save_policies(policies, path)
    loaded = load_policies(path)
    assert loaded == policies
