"""
Tests for C6 — Fitness function (src/fitness.py).

Each test corresponds to an acceptance criterion from Section E of BLUEPRINT.md:
  AC1: Hand-verified toy test — known structure + known band scores → expected walk result.
  AC2: Valid in-budget candidate produces finite fitness.
  AC3: Over-budget candidate produces fitness = -inf.
  AC4: Calling evaluate twice (with caching) returns bit-identical results.
  AC5: Clearing cache and re-evaluating returns same result as cached call.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.fitness import evaluate, evaluate_heldout, walk, BANDS
from src.repertoire import Candidate, Repertoire


# ── Toy graph builder ─────────────────────────────────────────────────────────

def _make_toy_graph():
    """Build a minimal hand-crafted graph for deterministic testing.

    Structure (White plays first):
      root (white turn, ply 0)
        └─ e2e4 → pos_after_e4 (black turn, ply 1)
                     ├─ e7e5 (freq 0.6) → pos_after_e4_e5 (white turn, ply 2) [leaf for white]
                     └─ c7c5 (freq 0.4) → pos_after_e4_c5 (white turn, ply 2) [leaf for white]

    Black repertoire root: same starting position (white's turn).
    Black committed nodes: pos_after_e4 is black's turn — black commits e7e5.
    """
    root = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
    pos_after_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -"
    pos_after_e4_e5 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -"
    pos_after_e4_c5 = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -"

    def _band_stats(games, wins, draws, losses):
        return {"games": games, "wins": wins, "draws": draws, "losses": losses}

    graph = {
        "root_fen": root,
        "nodes": {
            root: {
                "fen": root,
                "ply_depth": 0,
                "total_games": 1000,
                "score_raw": 0.52,
                "turn": "white",
                "band_stats": {
                    "1600-1799": _band_stats(300, 156, 90, 54),
                    "1800-1999": _band_stats(400, 208, 120, 72),
                    "2000-2199": _band_stats(300, 156, 90, 54),
                },
                "children": {
                    "e2e4": {
                        "child_fen": pos_after_e4,
                        "move_san": "e4",
                        "aggregate_count": 700,
                        "band_counts": {
                            "1600-1799": 200,
                            "1800-1999": 280,
                            "2000-2199": 220,
                        },
                    },
                    "d2d4": {
                        "child_fen": "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq -",
                        "move_san": "d4",
                        "aggregate_count": 300,
                        "band_counts": {
                            "1600-1799": 100,
                            "1800-1999": 120,
                            "2000-2199": 80,
                        },
                    },
                },
            },
            pos_after_e4: {
                "fen": pos_after_e4,
                "ply_depth": 1,
                "total_games": 700,
                "score_raw": 0.51,
                "turn": "black",
                "band_stats": {
                    "1600-1799": _band_stats(200, 102, 60, 38),
                    "1800-1999": _band_stats(280, 143, 84, 53),
                    "2000-2199": _band_stats(220, 112, 66, 42),
                },
                "children": {
                    "e7e5": {
                        "child_fen": pos_after_e4_e5,
                        "move_san": "e5",
                        "aggregate_count": 420,
                        "band_counts": {
                            "1600-1799": 120,
                            "1800-1999": 168,
                            "2000-2199": 132,
                        },
                    },
                    "c7c5": {
                        "child_fen": pos_after_e4_c5,
                        "move_san": "c5",
                        "aggregate_count": 280,
                        "band_counts": {
                            "1600-1799": 80,
                            "1800-1999": 112,
                            "2000-2199": 88,
                        },
                    },
                },
            },
            pos_after_e4_e5: {
                "fen": pos_after_e4_e5,
                "ply_depth": 2,
                "total_games": 420,
                "score_raw": 0.50,
                "turn": "white",
                "band_stats": {
                    "1600-1799": _band_stats(120, 60, 36, 24),
                    "1800-1999": _band_stats(168, 84, 50, 34),
                    "2000-2199": _band_stats(132, 66, 40, 26),
                },
                "children": {},
            },
            pos_after_e4_c5: {
                "fen": pos_after_e4_c5,
                "ply_depth": 2,
                "total_games": 280,
                "score_raw": 0.53,
                "turn": "white",
                "band_stats": {
                    "1600-1799": _band_stats(80, 42, 24, 14),
                    "1800-1999": _band_stats(112, 59, 34, 19),
                    "2000-2199": _band_stats(88, 47, 26, 15),
                },
                "children": {},
            },
            # Dummy d4 position (not in repertoire, used as off-book leaf)
            "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq -": {
                "fen": "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq -",
                "ply_depth": 1,
                "total_games": 300,
                "score_raw": 0.52,
                "turn": "black",
                "band_stats": {
                    "1600-1799": _band_stats(100, 52, 30, 18),
                    "1800-1999": _band_stats(120, 62, 36, 22),
                    "2000-2199": _band_stats(80, 42, 24, 14),
                },
                "children": {},
            },
        },
    }
    return (graph, root, pos_after_e4, pos_after_e4_e5, pos_after_e4_c5)


def _make_toy_eval_cache(graph):
    """Build a simple eval cache from the toy graph (no Bayesian shrinkage, just raw)."""
    prior_mean = 0.52
    scores = {}
    for fen, node in graph["nodes"].items():
        band_scores = {}
        for band in BANDS:
            stats = node["band_stats"][band]
            n = stats["games"]
            if n == 0:
                s = prior_mean
            else:
                raw = (stats["wins"] + 0.5 * stats["draws"]) / n
                tau = 20
                s = (n * raw + tau * prior_mean) / (n + tau)
            band_scores[band] = s
        scores[fen] = band_scores
    return {"prior_mean": prior_mean, "scores": scores}


def _make_toy_base_policies(graph):
    """Compute base policies for the toy graph using the same smoothing as C3 (alpha=5)."""
    alpha = 5.0
    policies = {b: {} for b in BANDS}

    for fen, node in graph["nodes"].items():
        children = node["children"]
        if not children:
            continue
        agg_total = sum(c["aggregate_count"] for c in children.values())
        if agg_total == 0:
            continue
        agg_prob = {m: c["aggregate_count"] / agg_total for m, c in children.items()}

        for band in BANDS:
            band_total = sum(c["band_counts"][band] for c in children.values())
            dist = {}
            for m, c in children.items():
                bc = c["band_counts"][band]
                dist[m] = (bc + alpha * agg_prob[m]) / (band_total + alpha)
            policies[band][fen] = dist

    return policies


def _make_white_repertoire(graph):
    """White commits e2e4 at root. Reached: root + pos_after_e4 + both leaves."""
    root, pos_e4, pos_e4e5, pos_e4c5 = (
        graph["root_fen"],
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -",
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -",
    )
    return Repertoire(
        color="white",
        committed={root: "e2e4"},
        reached={root, pos_e4, pos_e4e5, pos_e4c5},
        graph=graph,
    )


def _make_black_repertoire(graph):
    """Black commits e7e5 at pos_after_e4. Reached: root + pos_after_e4 + pos_e4e5."""
    root = graph["root_fen"]
    pos_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -"
    pos_e4e5 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -"
    return Repertoire(
        color="black",
        committed={pos_e4: "e7e5"},
        reached={root, pos_e4, pos_e4e5},
        graph=graph,
    )


def _make_candidate(graph):
    return Candidate(
        white=_make_white_repertoire(graph),
        black=_make_black_repertoire(graph),
    )


# ── AC1: Hand-verified toy test ───────────────────────────────────────────────

class TestAC1HandVerifiedToy:
    """White walk with known policy should produce the expected score."""

    def test_white_walk_expected_score(self):
        """White commits e2e4. After e4, black plays e5 (p~0.6) or c5 (p~0.4).
        Walk should return a weighted sum of the two leaf cache scores."""
        graph, root, pos_e4, pos_e4e5, pos_e4c5 = _make_toy_graph()
        eval_cache = _make_toy_eval_cache(graph)
        base_policies = _make_toy_base_policies(graph)

        white_rep = _make_white_repertoire(graph)

        for band in BANDS:
            result = walk(white_rep, band, eval_cache, base_policies, graph)

            # Manual calculation: white committed e4 at root → goes to pos_e4.
            # At pos_e4 (opponent turn), policy distributes over e7e5 and c7c5.
            policy = base_policies[band][pos_e4]
            p_e5 = policy.get("e7e5", 0.0)
            p_c5 = policy.get("c7c5", 0.0)
            score_e5 = eval_cache["scores"][pos_e4e5][band]
            score_c5 = eval_cache["scores"][pos_e4c5][band]
            expected = p_e5 * score_e5 + p_c5 * score_c5

            assert abs(result - expected) < 1e-9, (
                f"Band {band}: walk={result}, expected={expected}"
            )

    def test_black_walk_expected_score(self):
        """Black committed e7e5 at pos_after_e4.
        Walk starts at root (white's turn = opponent for black rep).
        Policy at root distributes over e2e4 and d2d4.
        After e2e4 → pos_e4 (our turn, committed e7e5) → pos_e4_e5 (opponent turn, no children → 0.0).
        After d2d4 → off-book for black rep → eval_cache[d4_fen][band].
        Expected = p_e4 * 0.0 + p_d4 * eval_cache[d4_fen][band].
        """
        graph, root, pos_e4, pos_e4e5, pos_e4c5 = _make_toy_graph()
        eval_cache = _make_toy_eval_cache(graph)
        base_policies = _make_toy_base_policies(graph)

        black_rep = _make_black_repertoire(graph)

        d4_fen = "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq -"

        for band in BANDS:
            result = walk(black_rep, band, eval_cache, base_policies, graph)

            # At root (opponent turn for Black), policy distributes over e2e4 and d2d4.
            policy_root = base_policies[band][root]
            p_e4 = policy_root.get("e2e4", 0.0)
            p_d4 = policy_root.get("d2d4", 0.0)

            # After e2e4: pos_e4 in reached → recurse.
            # At pos_e4 (our turn for Black): committed e7e5 → pos_e4_e5.
            # At pos_e4_e5 (white's turn = opponent for Black): no children → returns 0.0.
            score_e4_branch = 0.0

            # After d2d4: off-book (d4_fen not in black_rep.reached) → eval_cache directly.
            score_d4_branch = eval_cache["scores"][d4_fen][band]

            expected = p_e4 * score_e4_branch + p_d4 * score_d4_branch

            assert abs(result - expected) < 1e-9, (
                f"Band {band}: walk={result}, expected={expected}"
            )

    def test_combined_candidate_band_scores(self):
        """band_scores[band] = 0.5 * white_ws + 0.5 * (1 - black_ws)."""
        graph, root, pos_e4, pos_e4e5, pos_e4c5 = _make_toy_graph()
        eval_cache = _make_toy_eval_cache(graph)
        base_policies = _make_toy_base_policies(graph)
        candidate = _make_candidate(graph)

        config = {"lambda_weight": 1.0, "alpha": 5.0}
        uniform = np.ones(3) / 3.0
        result = evaluate(candidate, uniform, config, eval_cache, base_policies, graph)

        for band in BANDS:
            white_ws = walk(candidate.white, band, eval_cache, base_policies, graph)
            black_ws = walk(candidate.black, band, eval_cache, base_policies, graph)
            expected_band = 0.5 * white_ws + 0.5 * (1.0 - black_ws)
            assert abs(result["band_scores"][band] - expected_band) < 1e-9


# ── AC2: Valid in-budget candidate has finite fitness ─────────────────────────

class TestAC2FiniteFitnessForValidCandidate:

    def test_finite_fitness(self):
        graph, *_ = _make_toy_graph()
        eval_cache = _make_toy_eval_cache(graph)
        base_policies = _make_toy_base_policies(graph)
        candidate = _make_candidate(graph)

        config = {"lambda_weight": 1.0, "alpha": 5.0}
        uniform = np.ones(3) / 3.0

        result = evaluate(candidate, uniform, config, eval_cache, base_policies, graph)

        assert math.isfinite(result["fitness"]), "Fitness should be finite for valid candidate"
        assert math.isfinite(result["mean_score"])
        assert math.isfinite(result["cvar"])
        assert set(result["band_scores"].keys()) == set(BANDS)

    def test_fitness_formula(self):
        """fitness = mean_score + lambda_weight * cvar."""
        graph, *_ = _make_toy_graph()
        eval_cache = _make_toy_eval_cache(graph)
        base_policies = _make_toy_base_policies(graph)
        candidate = _make_candidate(graph)

        for lam in [0.0, 1.0, 2.0]:
            config = {"lambda_weight": lam, "alpha": 5.0}
            uniform = np.ones(3) / 3.0
            r = evaluate(candidate, uniform, config, eval_cache, base_policies, graph,
                         use_cache=False)
            expected = r["mean_score"] + lam * r["cvar"]
            assert abs(r["fitness"] - expected) < 1e-12

    def test_cvar_is_worst_band(self):
        """CVaR with 3 bands is the minimum band score."""
        graph, *_ = _make_toy_graph()
        eval_cache = _make_toy_eval_cache(graph)
        base_policies = _make_toy_base_policies(graph)
        candidate = _make_candidate(graph)

        config = {"lambda_weight": 1.0, "alpha": 5.0}
        uniform = np.ones(3) / 3.0
        r = evaluate(candidate, uniform, config, eval_cache, base_policies, graph,
                     use_cache=False)

        assert r["cvar"] == min(r["band_scores"].values())

    def test_mean_score_weighted_by_mixture(self):
        """mean_score = sum(mixture[i] * band_scores[bands[i]])."""
        graph, *_ = _make_toy_graph()
        eval_cache = _make_toy_eval_cache(graph)
        base_policies = _make_toy_base_policies(graph)
        candidate = _make_candidate(graph)

        config = {"lambda_weight": 1.0, "alpha": 5.0}
        # Non-uniform mixture
        mixture = np.array([0.5, 0.3, 0.2])
        r = evaluate(candidate, mixture, config, eval_cache, base_policies, graph,
                     use_cache=False)

        expected_mean = sum(
            mixture[i] * r["band_scores"][BANDS[i]] for i in range(3)
        )
        assert abs(r["mean_score"] - expected_mean) < 1e-12


# ── AC3: Over-budget candidate produces fitness = -inf ────────────────────────

class TestAC3OverBudgetIsInfNeg:

    def _over_budget_candidate(self, graph):
        """Create a candidate where white has 21 committed nodes (over budget=20)."""
        root = graph["root_fen"]
        # Build a fake committed dict with 21 entries
        fake_committed = {f"fake_fen_{i} w KQkq -": f"e2e{i}" for i in range(21)}
        fake_committed[root] = "e2e4"
        white_rep = Repertoire(
            color="white",
            committed=fake_committed,
            reached=set(fake_committed.keys()),
            graph=graph,
        )
        black_rep = _make_black_repertoire(graph)
        return Candidate(white=white_rep, black=black_rep)

    def test_over_budget_white_returns_neg_inf(self):
        graph, *_ = _make_toy_graph()
        eval_cache = _make_toy_eval_cache(graph)
        base_policies = _make_toy_base_policies(graph)
        candidate = self._over_budget_candidate(graph)

        config = {"lambda_weight": 1.0, "alpha": 5.0}
        uniform = np.ones(3) / 3.0
        r = evaluate(candidate, uniform, config, eval_cache, base_policies, graph)

        assert r["fitness"] == -float("inf")
        assert r["mean_score"] == 0.0
        assert r["cvar"] == 0.0
        assert r["band_scores"] == {}

    def test_over_budget_black_returns_neg_inf(self):
        graph, *_ = _make_toy_graph()
        eval_cache = _make_toy_eval_cache(graph)
        base_policies = _make_toy_base_policies(graph)

        # Black over budget
        fake_committed = {f"fake_fen_{i} b KQkq -": f"e7e{i}" for i in range(21)}
        black_rep = Repertoire(
            color="black",
            committed=fake_committed,
            reached=set(fake_committed.keys()),
            graph=graph,
        )
        white_rep = _make_white_repertoire(graph)
        candidate = Candidate(white=white_rep, black=black_rep)

        config = {"lambda_weight": 1.0, "alpha": 5.0}
        uniform = np.ones(3) / 3.0
        r = evaluate(candidate, uniform, config, eval_cache, base_policies, graph)

        assert r["fitness"] == -float("inf")

    def test_exactly_at_budget_is_valid(self):
        """20 committed nodes should NOT trigger the over-budget penalty."""
        graph, *_ = _make_toy_graph()
        eval_cache = _make_toy_eval_cache(graph)
        base_policies = _make_toy_base_policies(graph)

        root = graph["root_fen"]
        # Exactly 20 committed nodes for white (including root with real move)
        fake_committed = {f"fake_fen_{i} w KQkq -": "e2e4" for i in range(19)}
        fake_committed[root] = "e2e4"
        assert len(fake_committed) == 20

        white_rep = Repertoire(
            color="white",
            committed=fake_committed,
            reached={root},
            graph=graph,
        )
        black_rep = _make_black_repertoire(graph)
        candidate = Candidate(white=white_rep, black=black_rep)

        config = {"lambda_weight": 1.0, "alpha": 5.0}
        uniform = np.ones(3) / 3.0
        # Should not return -inf (budget check passes); walk may fail but not budget gate
        r = evaluate(candidate, uniform, config, eval_cache, base_policies, graph)
        assert r["fitness"] != -float("inf")


# ── AC4: Calling evaluate twice with caching returns bit-identical results ────

class TestAC4CachingBitIdentical:

    def test_cached_call_is_identical(self):
        graph, *_ = _make_toy_graph()
        eval_cache = _make_toy_eval_cache(graph)
        base_policies = _make_toy_base_policies(graph)
        candidate = _make_candidate(graph)

        config = {"lambda_weight": 1.0, "alpha": 5.0}
        uniform = np.ones(3) / 3.0

        # First call — populates cache
        r1 = evaluate(candidate, uniform, config, eval_cache, base_policies, graph,
                      use_cache=True)
        # Second call — uses cache
        r2 = evaluate(candidate, uniform, config, eval_cache, base_policies, graph,
                      use_cache=True)

        assert r1["fitness"] == r2["fitness"]
        assert r1["mean_score"] == r2["mean_score"]
        assert r1["cvar"] == r2["cvar"]
        for band in BANDS:
            assert r1["band_scores"][band] == r2["band_scores"][band]

    def test_cache_is_set_on_candidate(self):
        """After first evaluate call with use_cache=True, band_scores_cache is populated."""
        graph, *_ = _make_toy_graph()
        eval_cache = _make_toy_eval_cache(graph)
        base_policies = _make_toy_base_policies(graph)
        candidate = _make_candidate(graph)

        assert candidate.band_scores_cache is None

        config = {"lambda_weight": 1.0, "alpha": 5.0}
        uniform = np.ones(3) / 3.0
        evaluate(candidate, uniform, config, eval_cache, base_policies, graph,
                 use_cache=True)

        assert candidate.band_scores_cache is not None
        assert set(candidate.band_scores_cache.keys()) == set(BANDS)

    def test_different_mixtures_use_same_band_scores(self):
        """Two calls with different mixtures but same candidate should reuse band_scores_cache."""
        graph, *_ = _make_toy_graph()
        eval_cache = _make_toy_eval_cache(graph)
        base_policies = _make_toy_base_policies(graph)
        candidate = _make_candidate(graph)

        config = {"lambda_weight": 1.0, "alpha": 5.0}
        mix1 = np.array([0.5, 0.3, 0.2])
        mix2 = np.array([0.2, 0.3, 0.5])

        r1 = evaluate(candidate, mix1, config, eval_cache, base_policies, graph,
                      use_cache=True)
        r2 = evaluate(candidate, mix2, config, eval_cache, base_policies, graph,
                      use_cache=True)

        # Band scores should be identical (same candidate structure)
        for band in BANDS:
            assert r1["band_scores"][band] == r2["band_scores"][band]

        # But mean_score and fitness differ because mixture differs
        assert r1["mean_score"] != r2["mean_score"]


# ── AC5: Clearing cache and re-evaluating returns same result ─────────────────

class TestAC5CacheClearReturnsConsistentResult:

    def test_cleared_cache_matches_cached(self):
        graph, *_ = _make_toy_graph()
        eval_cache = _make_toy_eval_cache(graph)
        base_policies = _make_toy_base_policies(graph)
        candidate = _make_candidate(graph)

        config = {"lambda_weight": 1.0, "alpha": 5.0}
        uniform = np.ones(3) / 3.0

        # Call with cache
        r_cached = evaluate(candidate, uniform, config, eval_cache, base_policies, graph,
                            use_cache=True)

        # Clear cache manually
        candidate.band_scores_cache = None

        # Re-evaluate (cache will be rebuilt)
        r_fresh = evaluate(candidate, uniform, config, eval_cache, base_policies, graph,
                           use_cache=True)

        assert r_cached["fitness"] == r_fresh["fitness"]
        assert r_cached["mean_score"] == r_fresh["mean_score"]
        assert r_cached["cvar"] == r_fresh["cvar"]
        for band in BANDS:
            assert r_cached["band_scores"][band] == r_fresh["band_scores"][band]

    def test_no_cache_matches_cached(self):
        """use_cache=False should produce the same numerical result as use_cache=True."""
        graph, *_ = _make_toy_graph()
        eval_cache = _make_toy_eval_cache(graph)
        base_policies = _make_toy_base_policies(graph)
        candidate = _make_candidate(graph)

        config = {"lambda_weight": 1.0, "alpha": 5.0}
        uniform = np.ones(3) / 3.0

        r_no_cache = evaluate(candidate, uniform, config, eval_cache, base_policies, graph,
                              use_cache=False)
        # Reset and use cache
        candidate.band_scores_cache = None
        r_with_cache = evaluate(candidate, uniform, config, eval_cache, base_policies, graph,
                                use_cache=True)

        assert r_no_cache["fitness"] == r_with_cache["fitness"]
        for band in BANDS:
            assert r_no_cache["band_scores"][band] == r_with_cache["band_scores"][band]


# ── Bonus: evaluate_heldout smoke test ────────────────────────────────────────

class TestEvaluateHeldout:

    def test_heldout_returns_finite(self):
        """evaluate_heldout should return a finite float."""
        graph, *_ = _make_toy_graph()
        eval_cache = _make_toy_eval_cache(graph)
        base_policies = _make_toy_base_policies(graph)
        candidate = _make_candidate(graph)

        config = {"lambda_weight": 1.0, "alpha": 5.0}
        # Use the same toy graph as heldout graph (same structure)
        score = evaluate_heldout(
            candidate,
            eval_cache_heldout=eval_cache,
            base_policies_train=base_policies,
            graph_heldout=graph,
            config=config,
        )
        assert math.isfinite(score)

    def test_heldout_uniform_mixture(self):
        """evaluate_heldout uses uniform mixture, so result equals evaluate under uniform."""
        graph, *_ = _make_toy_graph()
        eval_cache = _make_toy_eval_cache(graph)
        base_policies = _make_toy_base_policies(graph)
        candidate = _make_candidate(graph)

        config = {"lambda_weight": 1.0, "alpha": 5.0}
        uniform = np.ones(3) / 3.0

        heldout_score = evaluate_heldout(
            candidate,
            eval_cache_heldout=eval_cache,
            base_policies_train=base_policies,
            graph_heldout=graph,
            config=config,
        )
        train_result = evaluate(candidate, uniform, config, eval_cache, base_policies, graph,
                                use_cache=False)

        assert abs(heldout_score - train_result["fitness"]) < 1e-9
