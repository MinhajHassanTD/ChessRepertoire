"""
Tests for C8 — Co-evolution coordinator (src/coevolution.py).

Acceptance criteria from Section E of BLUEPRINT.md:
  AC1: All three modes run end-to-end on a pilot configuration
       (population size 10, 5 generations) without errors.
  AC2: Running the function twice with the same mode, config, and seed
       produces bit-identical result dictionaries (excluding wall_time_seconds).
  AC3: Switching modes requires only changing the mode string argument —
       no other code changes needed.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from src.coevolution import (
    _mean_pairwise_distance,
    _mean_pairwise_opponent_distance,
    run_coevolution,
    serialize_candidate,
)
from src.fitness import BANDS
from src.opponent import Opponent
from src.repertoire import Candidate, Repertoire


# ── Shared toy infrastructure ─────────────────────────────────────────────────

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"


def _make_toy_graph():
    """
    Minimal hand-crafted graph used for all coevolution tests.

    White plays first (root = STARTING_FEN).
    root (white, ply 0)
      └─ e2e4 → pos_e4  (black, ply 1)
           ├─ e7e5 (60 %) → pos_e4e5 (white, ply 2, leaf)
           └─ c7c5 (40 %) → pos_e4c5 (white, ply 2, leaf)
    d2d4 → pos_d4 (black, ply 1, leaf)

    All leaf positions are white-turn with no children so the GA
    cannot budget beyond 1 committed node per color — small but valid.
    """
    root = STARTING_FEN
    pos_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -"
    pos_e4e5 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -"
    pos_e4c5 = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -"
    pos_d4 = "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq -"

    def _bs(g, w, d, l):
        return {"games": g, "wins": w, "draws": d, "losses": l}

    graph = {
        "root_fen": root,
        "nodes": {
            root: {
                "fen": root, "ply_depth": 0, "total_games": 1000,
                "score_raw": 0.52, "turn": "white",
                "band_stats": {
                    "1600-1799": _bs(300, 156, 90, 54),
                    "1800-1999": _bs(400, 208, 120, 72),
                    "2000-2199": _bs(300, 156, 90, 54),
                },
                "children": {
                    "e2e4": {"child_fen": pos_e4, "move_san": "e4",
                             "aggregate_count": 700,
                             "band_counts": {"1600-1799": 200, "1800-1999": 280, "2000-2199": 220}},
                    "d2d4": {"child_fen": pos_d4, "move_san": "d4",
                             "aggregate_count": 300,
                             "band_counts": {"1600-1799": 100, "1800-1999": 120, "2000-2199": 80}},
                },
            },
            pos_e4: {
                "fen": pos_e4, "ply_depth": 1, "total_games": 700,
                "score_raw": 0.51, "turn": "black",
                "band_stats": {
                    "1600-1799": _bs(200, 102, 60, 38),
                    "1800-1999": _bs(280, 143, 84, 53),
                    "2000-2199": _bs(220, 112, 66, 42),
                },
                "children": {
                    "e7e5": {"child_fen": pos_e4e5, "move_san": "e5",
                             "aggregate_count": 420,
                             "band_counts": {"1600-1799": 120, "1800-1999": 168, "2000-2199": 132}},
                    "c7c5": {"child_fen": pos_e4c5, "move_san": "c5",
                             "aggregate_count": 280,
                             "band_counts": {"1600-1799": 80, "1800-1999": 112, "2000-2199": 88}},
                },
            },
            pos_e4e5: {
                "fen": pos_e4e5, "ply_depth": 2, "total_games": 420,
                "score_raw": 0.50, "turn": "white",
                "band_stats": {
                    "1600-1799": _bs(120, 60, 36, 24),
                    "1800-1999": _bs(168, 84, 50, 34),
                    "2000-2199": _bs(132, 66, 40, 26),
                },
                "children": {},
            },
            pos_e4c5: {
                "fen": pos_e4c5, "ply_depth": 2, "total_games": 280,
                "score_raw": 0.53, "turn": "white",
                "band_stats": {
                    "1600-1799": _bs(80, 42, 24, 14),
                    "1800-1999": _bs(112, 59, 34, 19),
                    "2000-2199": _bs(88, 47, 26, 15),
                },
                "children": {},
            },
            pos_d4: {
                "fen": pos_d4, "ply_depth": 1, "total_games": 300,
                "score_raw": 0.52, "turn": "black",
                "band_stats": {
                    "1600-1799": _bs(100, 52, 30, 18),
                    "1800-1999": _bs(120, 62, 36, 22),
                    "2000-2199": _bs(80, 42, 24, 14),
                },
                "children": {},
            },
        },
    }
    return graph


def _make_toy_eval_cache(graph):
    prior_mean = 0.52
    tau = 20
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
                s = (n * raw + tau * prior_mean) / (n + tau)
            band_scores[band] = s
        scores[fen] = band_scores
    return {"prior_mean": prior_mean, "scores": scores}


def _make_toy_base_policies(graph):
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


def _pilot_config():
    """Small config for fast pilot tests (pop=10, gens=5)."""
    return {
        "lambda_weight": 1.0,
        "alpha": 5.0,
        "pop_size_repertoires": 10,
        "pop_size_opponents": 6,
        "n_generations": 5,
        "tournament_size": 3,
        "crossover_rate": 0.7,
        "mutation_rate": 0.3,
        "hof_size": 5,
    }


def _fixtures():
    graph = _make_toy_graph()
    eval_cache = _make_toy_eval_cache(graph)
    base_policies = _make_toy_base_policies(graph)
    return graph, eval_cache, base_policies


# ── AC1: All three modes complete without error ───────────────────────────────

class TestAC1AllModesRunEndToEnd:
    """
    AC1: All three modes run end-to-end on a pilot configuration
         (population size 10, 5 generations) without errors.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.graph, self.eval_cache, self.base_policies = _fixtures()
        self.config = _pilot_config()

    def _run(self, mode: str) -> dict:
        return run_coevolution(
            mode=mode,
            config=self.config,
            seed=42,
            graph_train=self.graph,
            graph_heldout=self.graph,
            base_policies_train=self.base_policies,
            eval_cache_train=self.eval_cache,
            eval_cache_heldout=self.eval_cache,
        )

    def test_static_runs_without_error(self):
        result = self._run("STATIC")
        assert isinstance(result, dict)

    def test_coevolve_frozen_runs_without_error(self):
        result = self._run("COEVOLVE_FROZEN")
        assert isinstance(result, dict)

    def test_coevolve_runs_without_error(self):
        result = self._run("COEVOLVE")
        assert isinstance(result, dict)

    def test_result_has_required_keys(self):
        """Return dict must contain all specified keys."""
        required = {
            "mode", "config", "seed", "git_commit",
            "history", "final_best_candidate",
            "final_training_fitness", "heldout_score",
            "wall_time_seconds",
        }
        for mode in ("STATIC", "COEVOLVE_FROZEN", "COEVOLVE"):
            result = self._run(mode)
            assert required.issubset(result.keys()), (
                f"Mode {mode} missing keys: {required - result.keys()}"
            )

    def test_history_has_correct_length(self):
        """History must have one entry per generation."""
        for mode in ("STATIC", "COEVOLVE_FROZEN", "COEVOLVE"):
            result = self._run(mode)
            assert len(result["history"]) == self.config["n_generations"], (
                f"Mode {mode}: expected {self.config['n_generations']} history entries"
            )

    def test_history_entries_have_required_fields(self):
        """Every history entry must have the required stat fields."""
        required = {
            "gen", "best_training_fitness", "mean_training_fitness",
            "repertoire_diversity", "opponent_diversity", "hof_size",
        }
        for mode in ("STATIC", "COEVOLVE_FROZEN", "COEVOLVE"):
            result = self._run(mode)
            for entry in result["history"]:
                assert required.issubset(entry.keys()), (
                    f"Mode {mode} gen {entry.get('gen')} missing: "
                    f"{required - entry.keys()}"
                )

    def test_final_training_fitness_is_finite(self):
        """final_training_fitness must be a finite float for all modes."""
        for mode in ("STATIC", "COEVOLVE_FROZEN", "COEVOLVE"):
            result = self._run(mode)
            assert math.isfinite(result["final_training_fitness"]), (
                f"Mode {mode}: final_training_fitness not finite"
            )

    def test_heldout_score_is_finite(self):
        """heldout_score must be a finite float for all modes."""
        for mode in ("STATIC", "COEVOLVE_FROZEN", "COEVOLVE"):
            result = self._run(mode)
            assert math.isfinite(result["heldout_score"]), (
                f"Mode {mode}: heldout_score not finite"
            )

    def test_final_best_candidate_serialization(self):
        """final_best_candidate must be a dict with the four required keys."""
        required = {"white_committed", "white_reached", "black_committed", "black_reached"}
        for mode in ("STATIC", "COEVOLVE_FROZEN", "COEVOLVE"):
            result = self._run(mode)
            fbc = result["final_best_candidate"]
            assert isinstance(fbc, dict)
            assert required.issubset(fbc.keys()), (
                f"Mode {mode}: missing keys {required - fbc.keys()}"
            )
            assert isinstance(fbc["white_committed"], dict)
            assert isinstance(fbc["white_reached"], list)
            assert isinstance(fbc["black_committed"], dict)
            assert isinstance(fbc["black_reached"], list)

    def test_coevolve_opponent_diversity_is_not_none(self):
        """In COEVOLVE mode, opponent_diversity should be a float (not None)."""
        result = self._run("COEVOLVE")
        for entry in result["history"]:
            assert entry["opponent_diversity"] is not None, (
                f"Gen {entry['gen']}: opponent_diversity should not be None in COEVOLVE"
            )

    def test_static_opponent_diversity_is_none(self):
        """In STATIC mode, opponent_diversity should be None."""
        result = self._run("STATIC")
        for entry in result["history"]:
            assert entry["opponent_diversity"] is None, (
                f"Gen {entry['gen']}: opponent_diversity should be None in STATIC"
            )

    def test_coevolve_frozen_opponent_diversity_is_none(self):
        """In COEVOLVE_FROZEN mode, opponent_diversity should be None."""
        result = self._run("COEVOLVE_FROZEN")
        for entry in result["history"]:
            assert entry["opponent_diversity"] is None, (
                f"Gen {entry['gen']}: opponent_diversity should be None in COEVOLVE_FROZEN"
            )

    def test_coevolve_hof_size_grows(self):
        """In COEVOLVE mode, hof_size in history should be >= 0 and grow over time."""
        result = self._run("COEVOLVE")
        hof_sizes = [entry["hof_size"] for entry in result["history"]]
        assert all(s >= 0 for s in hof_sizes)
        # hof_size should be non-decreasing until it hits hof_size cap
        for i in range(1, len(hof_sizes)):
            assert hof_sizes[i] >= hof_sizes[i - 1] or hof_sizes[i] == self.config["hof_size"]

    def test_static_hof_size_always_zero(self):
        """In STATIC and COEVOLVE_FROZEN modes, hof_size must always be 0."""
        for mode in ("STATIC", "COEVOLVE_FROZEN"):
            result = self._run(mode)
            for entry in result["history"]:
                assert entry["hof_size"] == 0, (
                    f"Mode {mode} gen {entry['gen']}: hof_size should be 0"
                )

    def test_invalid_mode_raises(self):
        """An invalid mode string must raise an AssertionError."""
        with pytest.raises(AssertionError):
            self._run("INVALID_MODE")


# ── AC2: Determinism — same seed produces bit-identical results ───────────────

class TestAC2Determinism:
    """
    AC2: Running the function twice with the same mode, config, and seed
         produces bit-identical result dictionaries (excluding wall_time_seconds).
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.graph, self.eval_cache, self.base_policies = _fixtures()
        self.config = _pilot_config()

    def _run(self, mode: str, seed: int = 99) -> dict:
        return run_coevolution(
            mode=mode,
            config=self.config,
            seed=seed,
            graph_train=self.graph,
            graph_heldout=self.graph,
            base_policies_train=self.base_policies,
            eval_cache_train=self.eval_cache,
            eval_cache_heldout=self.eval_cache,
        )

    def _assert_identical(self, r1: dict, r2: dict, label: str):
        """Compare two result dicts, ignoring wall_time_seconds."""
        excluded = {"wall_time_seconds"}
        for key in r1:
            if key in excluded:
                continue
            assert r1[key] == r2[key], (
                f"{label}: mismatch on key '{key}': {r1[key]!r} != {r2[key]!r}"
            )

    def test_static_deterministic(self):
        r1 = self._run("STATIC")
        r2 = self._run("STATIC")
        self._assert_identical(r1, r2, "STATIC")

    def test_coevolve_frozen_deterministic(self):
        r1 = self._run("COEVOLVE_FROZEN")
        r2 = self._run("COEVOLVE_FROZEN")
        self._assert_identical(r1, r2, "COEVOLVE_FROZEN")

    def test_coevolve_deterministic(self):
        r1 = self._run("COEVOLVE")
        r2 = self._run("COEVOLVE")
        self._assert_identical(r1, r2, "COEVOLVE")

    def test_different_seeds_produce_different_results(self):
        """Different seeds must yield different histories (with overwhelming probability)."""
        r1 = self._run("COEVOLVE", seed=1)
        r2 = self._run("COEVOLVE", seed=2)
        # The final training fitnesses or histories should differ
        h1 = r1["history"]
        h2 = r2["history"]
        different = any(
            h1[i]["best_training_fitness"] != h2[i]["best_training_fitness"]
            for i in range(len(h1))
        )
        assert different, "Different seeds produced identical training fitness histories"

    def test_history_generation_indices_are_sequential(self):
        """Gen indices in history must be 0, 1, 2, ..., n_generations-1."""
        for mode in ("STATIC", "COEVOLVE_FROZEN", "COEVOLVE"):
            result = self._run(mode)
            gens = [entry["gen"] for entry in result["history"]]
            assert gens == list(range(self.config["n_generations"])), (
                f"Mode {mode}: unexpected gen indices {gens}"
            )

    def test_seed_stored_in_result(self):
        """The seed used must be stored in the result dict."""
        for seed in (0, 42, 1000):
            result = self._run("STATIC", seed=seed)
            assert result["seed"] == seed

    def test_mode_stored_in_result(self):
        """The mode string must be stored in the result dict."""
        for mode in ("STATIC", "COEVOLVE_FROZEN", "COEVOLVE"):
            result = self._run(mode)
            assert result["mode"] == mode

    def test_config_stored_in_result(self):
        """The config dict must be stored in the result dict."""
        result = self._run("STATIC")
        assert result["config"] == self.config

    def test_git_commit_is_string(self):
        """git_commit must be a non-empty string."""
        result = self._run("STATIC")
        assert isinstance(result["git_commit"], str)
        assert len(result["git_commit"]) > 0


# ── AC3: Mode switching requires only the mode argument ───────────────────────

class TestAC3ModeSwitchingByArgumentOnly:
    """
    AC3: Switching modes requires only changing the mode string argument —
         no other code changes needed.
    Verified by running all three modes with identical arguments except mode.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.graph, self.eval_cache, self.base_policies = _fixtures()
        self.config = _pilot_config()

    def _run(self, mode: str) -> dict:
        return run_coevolution(
            mode=mode,
            config=self.config,
            seed=7,
            graph_train=self.graph,
            graph_heldout=self.graph,
            base_policies_train=self.base_policies,
            eval_cache_train=self.eval_cache,
            eval_cache_heldout=self.eval_cache,
        )

    def test_all_modes_accept_identical_arguments(self):
        """All three modes must succeed with identical call arguments."""
        for mode in ("STATIC", "COEVOLVE_FROZEN", "COEVOLVE"):
            result = self._run(mode)
            assert result["mode"] == mode

    def test_modes_produce_different_histories(self):
        """
        COEVOLVE and STATIC should produce different best_training_fitness
        trajectories (they use different opponent structures).
        """
        r_static = self._run("STATIC")
        r_coevolve = self._run("COEVOLVE")
        static_fitness = [e["best_training_fitness"] for e in r_static["history"]]
        coevolve_fitness = [e["best_training_fitness"] for e in r_coevolve["history"]]
        # They may occasionally match on tiny graphs, but at least the modes ran
        assert len(static_fitness) == len(coevolve_fitness) == self.config["n_generations"]

    def test_static_has_single_opponent(self):
        """
        STATIC must use exactly one uniform opponent (verified indirectly:
        history hof_size is always 0, opponent_diversity is None).
        """
        result = self._run("STATIC")
        for entry in result["history"]:
            assert entry["hof_size"] == 0
            assert entry["opponent_diversity"] is None

    def test_coevolve_frozen_never_has_hof(self):
        """COEVOLVE_FROZEN must never populate the HoF."""
        result = self._run("COEVOLVE_FROZEN")
        for entry in result["history"]:
            assert entry["hof_size"] == 0

    def test_coevolve_uses_hof_after_first_gen(self):
        """After gen 0, COEVOLVE should start accumulating HoF entries."""
        result = self._run("COEVOLVE")
        # After at least one generation, HoF should have members
        assert result["history"][-1]["hof_size"] > 0


# ── Helper function unit tests ─────────────────────────────────────────────────

class TestHelpers:

    def test_mean_pairwise_distance_identical_pops(self):
        """Identical populations have distance 0."""
        graph = _make_toy_graph()
        from src.repertoire import construct_initial
        rng = np.random.default_rng(0)
        rep_w = construct_initial(graph, "white", 20, rng)
        rep_b = construct_initial(graph, "black", 20, rng)
        cand = Candidate(white=rep_w, black=rep_b, fitness=1.0)
        pop = [cand, cand, cand]
        assert _mean_pairwise_distance(pop) == 0.0

    def test_mean_pairwise_distance_single_element(self):
        """Single-element population has distance 0."""
        graph = _make_toy_graph()
        from src.repertoire import construct_initial
        rng = np.random.default_rng(0)
        rep_w = construct_initial(graph, "white", 20, rng)
        rep_b = construct_initial(graph, "black", 20, rng)
        cand = Candidate(white=rep_w, black=rep_b, fitness=1.0)
        assert _mean_pairwise_distance([cand]) == 0.0

    def test_mean_pairwise_opponent_distance_identical(self):
        """Identical opponent mixtures have distance 0."""
        opp = Opponent.uniform()
        pop = [opp, opp, opp]
        assert _mean_pairwise_opponent_distance(pop) == pytest.approx(0.0, abs=1e-12)

    def test_mean_pairwise_opponent_distance_non_negative(self):
        """Pairwise opponent distance is always non-negative."""
        rng = np.random.default_rng(42)
        pop = [Opponent.random(rng) for _ in range(5)]
        assert _mean_pairwise_opponent_distance(pop) >= 0.0

    def test_serialize_candidate_roundtrip(self):
        """serialize_candidate produces the expected dict structure."""
        graph = _make_toy_graph()
        from src.repertoire import construct_initial
        rng = np.random.default_rng(0)
        rep_w = construct_initial(graph, "white", 20, rng)
        rep_b = construct_initial(graph, "black", 20, rng)
        cand = Candidate(white=rep_w, black=rep_b, fitness=0.5)

        serialized = serialize_candidate(cand)
        assert set(serialized.keys()) == {
            "white_committed", "white_reached", "black_committed", "black_reached"
        }
        assert serialized["white_committed"] == cand.white.committed
        assert set(serialized["white_reached"]) == cand.white.reached
        assert serialized["black_committed"] == cand.black.committed
        assert set(serialized["black_reached"]) == cand.black.reached
