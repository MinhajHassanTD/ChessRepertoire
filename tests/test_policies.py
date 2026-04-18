"""
Tests for C3 — Rating-band base policies (src/policies.py).
Each test corresponds to an acceptance criterion from Section E of BLUEPRINT.md.

Acceptance criteria (from BLUEPRINT.md):
  AC1: Every policy distribution sums to 1.0 within 1e-9 tolerance.
  AC2: For a position with zero games in a specific band, that band's
       distribution equals the aggregate distribution.
  AC3: The band separation check passes (mean TV distance >= 0.05) and
       logs the actual value.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.policies import (
    ALPHA,
    RATING_BANDS,
    band_separation_check,
    compute_policies,
    load_policies,
    save_policies,
)
from src.graph import build_graph

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "snapshot.db")

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_minimal_graph(children_spec: dict) -> dict:
    """
    Build a tiny synthetic graph with a single node 'root_fen' whose children
    are given by *children_spec*:
        { move_uci: { 'aggregate_count': int, 'band_counts': {band: int, ...} } }
    """
    graph = {
        "root_fen": "start",
        "nodes": {
            "start": {
                "fen": "start",
                "ply_depth": 0,
                "total_games": sum(v["aggregate_count"] for v in children_spec.values()),
                "score_raw": 0.5,
                "turn": "white",
                "band_stats": {b: {"games": 0, "wins": 0, "draws": 0, "losses": 0} for b in RATING_BANDS},
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
    return graph


@pytest.fixture(scope="module")
def train_graph():
    """Load the real training graph from snapshot.db (used for AC3)."""
    return build_graph(DB_PATH, "train")


# ---------------------------------------------------------------------------
# AC1: Every policy distribution sums to 1.0 within 1e-9
# ---------------------------------------------------------------------------

class TestAC1_PoliciesSumToOne:
    def test_synthetic_sums_to_one(self):
        """Distributions over a synthetic graph all sum to 1.0."""
        children_spec = {
            "e2e4": {"aggregate_count": 600, "band_counts": {"1600-1799": 100, "1800-1999": 200, "2000-2199": 300}},
            "d2d4": {"aggregate_count": 300, "band_counts": {"1600-1799": 80, "1800-1999": 120, "2000-2199": 100}},
            "c2c4": {"aggregate_count": 100, "band_counts": {"1600-1799": 20, "1800-1999": 30, "2000-2199": 50}},
        }
        graph = _make_minimal_graph(children_spec)
        base_policies = compute_policies(graph)

        for band in RATING_BANDS:
            for fen, dist in base_policies[band].items():
                total = sum(dist.values())
                assert abs(total - 1.0) < 1e-9, (
                    f"Band {band}, fen {fen}: distribution sums to {total}, expected 1.0"
                )

    def test_real_graph_sums_to_one(self, train_graph):
        """All distributions in the real training graph sum to 1.0."""
        base_policies = compute_policies(train_graph)

        violations = []
        for band in RATING_BANDS:
            for fen, dist in base_policies[band].items():
                total = sum(dist.values())
                if abs(total - 1.0) >= 1e-9:
                    violations.append((band, fen, total))

        assert violations == [], (
            f"{len(violations)} distributions do not sum to 1.0: {violations[:5]}"
        )


# ---------------------------------------------------------------------------
# AC2: Zero-band-games position falls back to aggregate distribution
# ---------------------------------------------------------------------------

class TestAC2_ZeroBandFallback:
    def test_zero_band_equals_aggregate(self):
        """When a band has 0 games, its policy must equal the aggregate distribution."""
        # '1600-1799' has zero counts for all moves
        children_spec = {
            "e2e4": {"aggregate_count": 600, "band_counts": {"1600-1799": 0, "1800-1999": 200, "2000-2199": 300}},
            "d2d4": {"aggregate_count": 400, "band_counts": {"1600-1799": 0, "1800-1999": 100, "2000-2199": 100}},
        }
        graph = _make_minimal_graph(children_spec)
        base_policies = compute_policies(graph)

        # Aggregate probs: e2e4=0.6, d2d4=0.4
        agg_prob = {"e2e4": 0.6, "d2d4": 0.4}

        zero_band = "1600-1799"
        dist = base_policies[zero_band]["start"]

        for move, expected_prob in agg_prob.items():
            assert abs(dist[move] - expected_prob) < 1e-9, (
                f"Zero-band fallback failed for move {move}: "
                f"got {dist[move]}, expected {expected_prob}"
            )

    def test_zero_band_formula_derivation(self):
        """
        Verify formula: when band_total=0,
        policy[m] = (0 + α * agg_prob[m]) / (0 + α) = agg_prob[m].
        """
        children_spec = {
            "e2e4": {"aggregate_count": 700, "band_counts": {"1600-1799": 0, "1800-1999": 0, "2000-2199": 0}},
            "c2c4": {"aggregate_count": 300, "band_counts": {"1600-1799": 0, "1800-1999": 0, "2000-2199": 0}},
        }
        graph = _make_minimal_graph(children_spec)
        base_policies = compute_policies(graph)

        for band in RATING_BANDS:
            dist = base_policies[band]["start"]
            assert abs(dist["e2e4"] - 0.7) < 1e-9, f"{band}: e2e4 should be 0.7 but got {dist['e2e4']}"
            assert abs(dist["c2c4"] - 0.3) < 1e-9, f"{band}: c2c4 should be 0.3 but got {dist['c2c4']}"

    def test_smoothing_formula_correctness(self):
        """
        Verify the smoothing formula numerically for a non-zero band.
        policy[b][p][m] = (band_count[m] + α * agg_prob[m]) / (band_total + α)
        """
        alpha = ALPHA  # 5.0
        # e2e4: agg=600/1000=0.6, band=80;  d2d4: agg=400/1000=0.4, band=20
        children_spec = {
            "e2e4": {"aggregate_count": 600, "band_counts": {"1600-1799": 80, "1800-1999": 0, "2000-2199": 0}},
            "d2d4": {"aggregate_count": 400, "band_counts": {"1600-1799": 20, "1800-1999": 0, "2000-2199": 0}},
        }
        graph = _make_minimal_graph(children_spec)
        base_policies = compute_policies(graph)

        band = "1600-1799"
        dist = base_policies[band]["start"]

        band_total = 80 + 20  # 100
        expected_e2e4 = (80 + alpha * 0.6) / (band_total + alpha)
        expected_d2d4 = (20 + alpha * 0.4) / (band_total + alpha)

        assert abs(dist["e2e4"] - expected_e2e4) < 1e-9, (
            f"e2e4: got {dist['e2e4']}, expected {expected_e2e4}"
        )
        assert abs(dist["d2d4"] - expected_d2d4) < 1e-9, (
            f"d2d4: got {dist['d2d4']}, expected {expected_d2d4}"
        )


# ---------------------------------------------------------------------------
# AC3: Band separation check passes (mean TV >= 0.05) and logs the value
# ---------------------------------------------------------------------------

class TestAC3_BandSeparation:
    def test_band_separation_on_real_graph(self, train_graph, capsys):
        """
        Band separation check must pass (mean TV distance >= 0.05)
        and must print the actual value to stdout.
        """
        base_policies = compute_policies(train_graph)
        mean_tv = band_separation_check(base_policies, train_graph)

        # Check that the value was printed
        captured = capsys.readouterr()
        assert "mean TV" in captured.out.lower() or "tv distance" in captured.out.lower(), (
            "band_separation_check should print the TV distance to stdout"
        )

        assert mean_tv >= 0.05, (
            f"Mean TV distance {mean_tv:.4f} is below 0.05 threshold. "
            "Bands may be too similar — check the data."
        )

    def test_band_separation_returns_float(self, train_graph):
        """band_separation_check must return a float."""
        base_policies = compute_policies(train_graph)
        result = band_separation_check(base_policies, train_graph)
        assert isinstance(result, float), f"Expected float, got {type(result)}"

    def test_band_separation_synthetic_known_tv(self, capsys):
        """
        Verify TV distance computation on a synthetic case with known answer.
        Two bands: p1 = {e2e4: 1.0}, p2 = {d2d4: 1.0}
        TV = 0.5 * (|1-0| + |0-1|) = 1.0
        A third band identical to p2 gives pair (b2,b3) TV=0.
        mean TV across 3 pairs = (1.0 + 1.0 + 0.0) / 3 = 0.667
        """
        # We inject policies directly (bypassing compute_policies)
        # to test band_separation_check in isolation.
        base_policies = {
            "1600-1799": {"pos1": {"e2e4": 1.0}},
            "1800-1999": {"pos1": {"d2d4": 1.0}},
            "2000-2199": {"pos1": {"d2d4": 1.0}},
        }
        # Synthetic graph with pos1 having total_games >= 200 and children
        graph = {
            "root_fen": "pos1",
            "nodes": {
                "pos1": {
                    "fen": "pos1",
                    "ply_depth": 0,
                    "total_games": 500,
                    "score_raw": 0.5,
                    "turn": "white",
                    "band_stats": {b: {"games": 0, "wins": 0, "draws": 0, "losses": 0} for b in RATING_BANDS},
                    "children": {"e2e4": {"child_fen": "c1", "move_san": "e4", "aggregate_count": 300, "band_counts": {b: 100 for b in RATING_BANDS}}},
                }
            },
        }
        mean_tv = band_separation_check(base_policies, graph)
        # TV(1600,1800): |1-0| + |0-1| = 2 -> *0.5 = 1.0
        # TV(1600,2000): 1.0
        # TV(1800,2000): 0.0
        # mean = 2/3
        assert abs(mean_tv - 2.0 / 3.0) < 1e-9, f"Expected 0.6667, got {mean_tv}"


# ---------------------------------------------------------------------------
# Bonus: save/load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_load_roundtrip(self, tmp_path):
        """Policies saved to disk can be loaded back identically."""
        children_spec = {
            "e2e4": {"aggregate_count": 600, "band_counts": {"1600-1799": 100, "1800-1999": 200, "2000-2199": 300}},
            "d2d4": {"aggregate_count": 400, "band_counts": {"1600-1799": 80, "1800-1999": 120, "2000-2199": 200}},
        }
        graph = _make_minimal_graph(children_spec)
        base_policies = compute_policies(graph)

        path = str(tmp_path / "base_policies.pkl")
        save_policies(base_policies, path)
        loaded = load_policies(path)

        assert loaded.keys() == base_policies.keys()
        for band in RATING_BANDS:
            for fen in base_policies[band]:
                for move in base_policies[band][fen]:
                    assert abs(loaded[band][fen][move] - base_policies[band][fen][move]) < 1e-12
