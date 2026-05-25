"""
Tests for C5 — Repertoire and Candidate.

Each test function maps to one acceptance criterion from the spec:

AC1: construct_initial with 100 seeds → 100 valid repertoires.
AC2: Fuzz test: 1000 random mutations on 20 random repertoires — every result
     either valid or MutationFailed; no silently invalid outputs.
AC3: crossover on two valid repertoires returns valid repertoire or clone of parent.
AC4: Determinism — same RNG state → bit-identical outputs.
"""

import copy
import os
import sys
import pytest
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.graph import STARTING_FEN, aggregate_move_freq
from src.repertoire import (
    BUDGET,
    CLOSURE_THRESHOLD,
    Candidate,
    MutationFailed,
    Repertoire,
    construct_initial,
    construct_random,
    crossover,
    crossover_candidates,
    mutate_candidate,
    mutate_extend,
    mutate_move_swap,
    mutate_opening_replacement,
    mutate_prune,
    validate,
)

# ── Minimal synthetic graph builder ──────────────────────────────────────────

def _make_graph():
    """
    Build a small but valid position graph for testing without a real database.

    Layout (White moves first):
      root (w) --e2e4--> pos_e4 (b)
                             --e7e5--> pos_e4e5 (w)   [freq 0.6]
                             --e7e6--> pos_e4e6 (w)   [freq 0.4]
               --d2d4--> pos_d4 (b)
                             --d7d5--> pos_d4d5 (w)   [freq 0.7]
                             --g8f6--> pos_d4nf6 (w)  [freq 0.3]

    All leaf positions have no children.
    """
    def node(fen, depth, total, turn, children):
        return {
            "fen": fen,
            "ply_depth": depth,
            "total_games": total,
            "score_raw": 0.5,
            "turn": turn,
            "band_stats": {
                b: {"games": 100, "wins": 30, "draws": 20, "losses": 50}
                for b in ("1000-1399", "1400-1799", "1800-2199")
            },
            "children": children,
        }

    def child(child_fen, agg_count):
        return {
            "child_fen": child_fen,
            "move_san": "?",
            "aggregate_count": agg_count,
            "band_counts": {b: agg_count for b in ("1000-1399", "1400-1799", "1800-2199")},
        }

    root = STARTING_FEN
    pos_e4    = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -"
    pos_d4    = "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq -"
    pos_e4e5  = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -"
    pos_e4e6  = "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -"
    pos_d4d5  = "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq -"
    pos_d4nf6 = "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq -"

    nodes = {
        root: node(root, 0, 10_000_000, "white", {
            "e2e4": child(pos_e4, 6000),
            "d2d4": child(pos_d4, 4000),
        }),
        pos_e4: node(pos_e4, 1, 6000, "black", {
            "e7e5": child(pos_e4e5, 3600),
            "e7e6": child(pos_e4e6, 2400),
        }),
        pos_d4: node(pos_d4, 1, 4000, "black", {
            "d7d5": child(pos_d4d5, 2800),
            "g8f6": child(pos_d4nf6, 1200),
        }),
        pos_e4e5:  node(pos_e4e5,  2, 3600, "white", {}),
        pos_e4e6:  node(pos_e4e6,  2, 2400, "white", {}),
        pos_d4d5:  node(pos_d4d5,  2, 2800, "white", {}),
        pos_d4nf6: node(pos_d4nf6, 2, 1200, "white", {}),
    }

    return {"root_fen": root, "nodes": nodes}


GRAPH = _make_graph()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rng(seed):
    return np.random.default_rng(seed)


def _make_initial(color="white", seed=42):
    return construct_initial(GRAPH, color, BUDGET, _rng(seed))


def _make_random(color="white", seed=42):
    return construct_random(GRAPH, color, BUDGET, _rng(seed))


def _make_candidate(seed=42):
    return Candidate(
        white=_make_initial("white", seed),
        black=_make_initial("black", seed + 1),
    )


# ── AC1: construct_initial with 100 seeds produces 100 valid repertoires ──────

def test_ac1_construct_initial_100_seeds_all_valid():
    """AC1: 100 different seeds → 100 valid repertoires for both colors."""
    for seed in range(100):
        rep_w = construct_initial(GRAPH, "white", BUDGET, _rng(seed))
        assert validate(rep_w), f"seed {seed} white invalid"
        rep_b = construct_initial(GRAPH, "black", BUDGET, _rng(seed))
        assert validate(rep_b), f"seed {seed} black invalid"


def test_ac1_construct_initial_within_budget():
    """AC1 supplement: committed nodes never exceed budget."""
    for seed in range(20):
        rep = construct_initial(GRAPH, "white", BUDGET, _rng(seed))
        assert len(rep.committed) <= BUDGET


def test_ac1_construct_initial_white_has_root_committed():
    """White repertoire must have a committed move at the starting position."""
    rep = _make_initial("white")
    assert STARTING_FEN in rep.committed


def test_ac1_construct_initial_starting_fen_in_reached():
    """Starting FEN must be in reached for both colors."""
    for color in ("white", "black"):
        rep = construct_initial(GRAPH, color, BUDGET, _rng(0))
        assert STARTING_FEN in rep.reached


def test_ac1_construct_random_100_seeds_all_valid():
    """AC1 variant: construct_random with 100 seeds → valid."""
    for seed in range(100):
        rep = construct_random(GRAPH, "white", BUDGET, _rng(seed))
        assert validate(rep), f"construct_random seed {seed} invalid"


# ── AC2: Fuzz test ────────────────────────────────────────────────────────────

def test_ac2_fuzz_mutations_no_silent_invalids():
    """
    AC2: Apply 1000 random mutation attempts to 20 random initial repertoires.
    Every result either passes validate or the operator raised MutationFailed.
    No silently invalid outputs.
    """
    operators = [mutate_move_swap, mutate_extend, mutate_prune, mutate_opening_replacement]
    rng = _rng(9999)
    failures = []

    for base_seed in range(20):
        rep = construct_random(GRAPH, "white", BUDGET, _rng(base_seed * 7))
        assert validate(rep), f"Base rep {base_seed} invalid before fuzzing"

        for attempt in range(50):  # 20 * 50 = 1000 attempts
            op = operators[int(rng.integers(0, 4))]
            try:
                result = op(rep, rng)
                ok = validate(result)
                if not ok:
                    failures.append(
                        f"op={op.__name__} base_seed={base_seed} attempt={attempt}: "
                        f"invalid result without raising MutationFailed"
                    )
            except MutationFailed:
                pass  # expected

    assert not failures, "\n".join(failures)


def test_ac2_fuzz_black_mutations():
    """AC2 variant: same fuzz test for black repertoires."""
    operators = [mutate_move_swap, mutate_extend, mutate_prune, mutate_opening_replacement]
    rng = _rng(12345)
    failures = []

    for base_seed in range(20):
        rep = construct_random(GRAPH, "black", BUDGET, _rng(base_seed * 13))
        assert validate(rep)

        for attempt in range(50):
            op = operators[int(rng.integers(0, 4))]
            try:
                result = op(rep, rng)
                ok = validate(result)
                if not ok:
                    failures.append(
                        f"op={op.__name__} base_seed={base_seed} attempt={attempt}"
                    )
            except MutationFailed:
                pass

    assert not failures, "\n".join(failures)


def test_ac2_mutate_candidate_no_silent_invalids():
    """AC2 supplement: mutate_candidate produces valid candidates or clones."""
    rng = _rng(777)
    for seed in range(10):
        cand = _make_candidate(seed)
        for _ in range(20):
            new_cand = mutate_candidate(cand, rng)
            assert validate(new_cand.white), "mutate_candidate produced invalid white"
            assert validate(new_cand.black), "mutate_candidate produced invalid black"


# ── AC3: crossover returns valid repertoire or clone of parent ────────────────

def test_ac3_crossover_valid_or_clone():
    """AC3: crossover on two valid repertoires returns valid or clone of rep_a."""
    rng = _rng(42)
    failures = []
    for seed_a in range(10):
        for seed_b in range(10):
            rep_a = construct_random(GRAPH, "white", BUDGET, _rng(seed_a))
            rep_b = construct_random(GRAPH, "white", BUDGET, _rng(seed_b + 100))
            try:
                result = crossover(rep_a, rep_b, rng)
                if not validate(result):
                    failures.append(
                        f"seed_a={seed_a} seed_b={seed_b}: invalid result without exception"
                    )
            except MutationFailed:
                pass  # acceptable per spec

    assert not failures, "\n".join(failures)


def test_ac3_crossover_no_shared_returns_clone():
    """AC3: if no shared committed moves exist, returns a clone of rep_a."""
    rng = _rng(0)
    # Force two repertoires with different moves
    rep_a = construct_initial(GRAPH, "white", BUDGET, _rng(0))
    # Build rep_b with forced different committed moves (all different from rep_a)
    rep_b = rep_a.copy()
    for fen in list(rep_b.committed.keys()):
        freq = aggregate_move_freq(GRAPH, fen)
        alts = [m for m in freq if m != rep_a.committed.get(fen)]
        if alts:
            rep_b.committed[fen] = alts[0]

    result = crossover(rep_a, rep_b, rng)
    # If no shared moves found, result should equal rep_a
    assert result.committed == rep_a.committed


def test_ac3_crossover_candidates_valid():
    """AC3 supplement: crossover_candidates returns valid candidate."""
    rng = _rng(55)
    for seed in range(10):
        cand_a = _make_candidate(seed)
        cand_b = _make_candidate(seed + 50)
        result = crossover_candidates(cand_a, cand_b, rng)
        assert validate(result.white), "crossover_candidates invalid white"
        assert validate(result.black), "crossover_candidates invalid black"


# ── AC4: Determinism ──────────────────────────────────────────────────────────

def test_ac4_construct_initial_deterministic():
    """AC4: same seed → bit-identical construct_initial output."""
    rep1 = construct_initial(GRAPH, "white", BUDGET, _rng(42))
    rep2 = construct_initial(GRAPH, "white", BUDGET, _rng(42))
    assert rep1.committed == rep2.committed
    assert rep1.reached == rep2.reached


def test_ac4_construct_random_deterministic():
    """AC4: same seed → bit-identical construct_random output."""
    rep1 = construct_random(GRAPH, "white", BUDGET, _rng(99))
    rep2 = construct_random(GRAPH, "white", BUDGET, _rng(99))
    assert rep1.committed == rep2.committed
    assert rep1.reached == rep2.reached


def test_ac4_mutate_move_swap_deterministic():
    """AC4: same rep + same seed → bit-identical mutate_move_swap output."""
    rep = _make_initial("white", seed=7)
    try:
        r1 = mutate_move_swap(rep, _rng(7))
        r2 = mutate_move_swap(rep, _rng(7))
        assert r1.committed == r2.committed
        assert r1.reached == r2.reached
    except MutationFailed:
        pass  # acceptable if no swap possible


def test_ac4_crossover_deterministic():
    """AC4: same pair + same seed → bit-identical crossover output."""
    rep_a = construct_random(GRAPH, "white", BUDGET, _rng(1))
    rep_b = construct_random(GRAPH, "white", BUDGET, _rng(2))
    r1 = crossover(rep_a, rep_b, _rng(99))
    r2 = crossover(rep_a, rep_b, _rng(99))
    assert r1.committed == r2.committed
    assert r1.reached == r2.reached


# ── Additional correctness checks ─────────────────────────────────────────────

def test_validate_rejects_over_budget():
    """validate returns False for a repertoire that exceeds budget."""
    rep = _make_initial("white")
    # Manually inflate committed beyond budget
    for i in range(BUDGET + 5):
        rep.committed[f"fake_fen_{i}"] = "e2e4"
    assert not validate(rep)


def test_mutate_extend_increases_committed_by_one():
    """mutate_extend adds exactly one committed node (when budget allows)."""
    rep = _make_initial("white", seed=0)
    if len(rep.committed) >= BUDGET:
        pytest.skip("Already at budget")
    initial_count = len(rep.committed)
    try:
        new_rep = mutate_extend(rep, _rng(0))
        assert len(new_rep.committed) == initial_count + 1
        assert validate(new_rep)
    except MutationFailed:
        pass


def test_mutate_prune_decreases_committed():
    """mutate_prune removes at least one committed node."""
    rep = _make_initial("white", seed=0)
    initial_count = len(rep.committed)
    if initial_count <= 1:
        pytest.skip("Nothing to prune")
    try:
        new_rep = mutate_prune(rep, _rng(0))
        assert len(new_rep.committed) < initial_count
        assert validate(new_rep)
    except MutationFailed:
        pass


def test_closure_rule_holds_after_all_mutations():
    """All four mutation ops produce closure-valid results on a white repertoire."""
    rep = _make_initial("white", seed=3)
    rng = _rng(42)
    ops = [mutate_move_swap, mutate_extend, mutate_prune, mutate_opening_replacement]
    for op in ops:
        try:
            result = op(rep.copy(), rng)
            assert validate(result), f"{op.__name__} broke closure"
        except MutationFailed:
            pass


def test_candidate_copy_is_independent():
    """Modifying a copy does not affect the original candidate."""
    cand = _make_candidate(0)
    copy_cand = cand.copy()
    copy_cand.white.committed["extra_fake"] = "e2e4"
    assert "extra_fake" not in cand.white.committed


def test_construct_initial_greedy_picks_best_move():
    """construct_initial should pick e2e4 (highest freq 0.6) at the root for white."""
    rep = construct_initial(GRAPH, "white", BUDGET, _rng(0))
    # e2e4 has aggregate_count 6000 vs d2d4 4000, so greedy picks e2e4
    assert rep.committed.get(STARTING_FEN) == "e2e4"


def test_opponent_moves_covered_by_closure():
    """
    After construct_initial for white with e2e4, both e7e5 and e7e6 responses
    (both >= 5% of e4 position's moves) must be in reached.
    """
    rep = construct_initial(GRAPH, "white", BUDGET, _rng(0))
    pos_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -"
    pos_e4e5 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -"
    pos_e4e6 = "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -"
    assert pos_e4 in rep.reached
    assert pos_e4e5 in rep.reached
    assert pos_e4e6 in rep.reached
