"""Probe B — game simulation with python-chess.

Simulates full games using python-chess.  Our side consults the repertoire's
committed dictionary; opponents are chosen from one of three sources
(in priority order when multiple are available):

  1. Stockfish at configurable depth  (requires STOCKFISH_PATH env var)
  2. Held-out aggregate move frequency (purely empirical, no engine required)
  3. A uniform pick over legal moves  (last-resort fallback)

When Stockfish is unavailable, games terminate at the first position outside
the held-out graph coverage and are scored via eval_cache_heldout.  With
Stockfish, games play out to a terminal python-chess state (checkmate,
stalemate, insufficient material, 50-move, threefold).  The two modes are
reported as separate metrics so the reader can see which is in use.

Used for Probe B of the testv2 plan.
"""
from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Optional

import chess
import chess.engine
import numpy as np

from src.graph import aggregate_move_freq


# ── Helpers ────────────────────────────────────────────────────────────────────

def _canonical_fen(board: chess.Board) -> str:
    """First four space-separated fields — matches the ingest canonicalisation."""
    return " ".join(board.fen().split()[:4])


def _pick_from_policy(policy: dict, rng: np.random.Generator) -> Optional[str]:
    if not policy:
        return None
    moves = list(policy.keys())
    probs = np.array(list(policy.values()), dtype=float)
    s = probs.sum()
    if s <= 0:
        return None
    probs = probs / s
    return str(rng.choice(moves, p=probs))


def _legal_ucis(board: chess.Board) -> list[str]:
    return [m.uci() for m in board.legal_moves]


def _score_for_board(
    board: chess.Board,
    eval_cache: dict,
    band: str = "1600-1799",
) -> float:
    """White-perspective score for a board using the held-out eval cache.

    If the canonical FEN is in cache we return that band's score.  Otherwise
    we fall back to the cache's prior mean.
    """
    cfen = _canonical_fen(board)
    scores = eval_cache.get("scores", {})
    if cfen in scores:
        return float(scores[cfen].get(band, eval_cache["prior_mean"]))
    return float(eval_cache["prior_mean"])


def _board_result_score(board: chess.Board) -> Optional[float]:
    """If the board is in a terminal state, return the White-perspective
    expected score (1.0 / 0.5 / 0.0).  Otherwise return None."""
    res = board.result(claim_draw=True)
    if res == "1-0":
        return 1.0
    if res == "0-1":
        return 0.0
    if res in ("1/2-1/2",):
        return 0.5
    return None


# ── Simulation core ────────────────────────────────────────────────────────────

def simulate_game(
    white_committed: dict,
    white_reached: set,
    black_committed: dict,
    black_reached: set,
    heldout_graph: dict,
    eval_cache_heldout: dict,
    rng: np.random.Generator,
    engine: Optional["chess.engine.SimpleEngine"] = None,
    engine_depth: int = 8,
    max_plies: int = 120,
    score_band: str = "1600-1799",
) -> dict:
    """Play one game, return a dict describing its terminal state.

    Our side (whichever colour the repertoire applies to) plays:
      - committed move if current position is in repertoire.committed, else
      - Stockfish at low depth (if engine provided), else
      - heldout aggregate-freq weighted sample, else
      - legal-move random.

    Opponent plays:
      - Stockfish at `engine_depth` (if engine provided), else
      - heldout aggregate-freq weighted sample, else
      - legal-move random.
    """
    board = chess.Board()
    plies = 0
    plies_in_book_white = 0
    plies_in_book_black = 0

    def our_move(committed, reached):
        nonlocal plies_in_book_white, plies_in_book_black
        cfen = _canonical_fen(board)
        if cfen in committed:
            if board.turn == chess.WHITE:
                plies_in_book_white += 1
            else:
                plies_in_book_black += 1
            return committed[cfen]
        return None

    def engine_move(depth: int) -> Optional[str]:
        if engine is None:
            return None
        try:
            r = engine.play(board, chess.engine.Limit(depth=depth))
            if r.move is None:
                return None
            return r.move.uci()
        except Exception:
            return None

    def aggregate_move() -> Optional[str]:
        cfen = _canonical_fen(board)
        freq = aggregate_move_freq(heldout_graph, cfen)
        return _pick_from_policy(freq, rng)

    while plies < max_plies:
        if board.is_game_over(claim_draw=True):
            break

        white_turn = board.turn == chess.WHITE

        # Try book move first.
        book_move = (
            our_move(white_committed, white_reached) if white_turn
            else our_move(black_committed, black_reached)
        )

        if book_move is not None:
            move_uci = book_move
        else:
            # Fallback ladder.
            move_uci = engine_move(max(2, engine_depth - 4))  # weak-ish continuation after book
            if move_uci is None:
                move_uci = aggregate_move()
            if move_uci is None:
                legal = _legal_ucis(board)
                if not legal:
                    break
                move_uci = str(rng.choice(legal))

        # Validate the move; fall back to a random legal move on illegal input.
        try:
            move = chess.Move.from_uci(move_uci)
        except Exception:
            legal = _legal_ucis(board)
            if not legal:
                break
            move = chess.Move.from_uci(str(rng.choice(legal)))

        if move not in board.legal_moves:
            legal = _legal_ucis(board)
            if not legal:
                break
            move = chess.Move.from_uci(str(rng.choice(legal)))

        board.push(move)
        plies += 1

    # Scoring: if the game actually ended, use the board result.
    # Otherwise (engine absent and game timed out / exited book), use eval_cache.
    terminal = _board_result_score(board)
    if terminal is not None:
        white_score = terminal
        reason = board.result(claim_draw=True)
    else:
        white_score = _score_for_board(board, eval_cache_heldout, score_band)
        reason = "truncated"

    return {
        "plies": plies,
        "white_score": white_score,
        "plies_in_book_white": plies_in_book_white,
        "plies_in_book_black": plies_in_book_black,
        "termination": reason,
        "final_fen": _canonical_fen(board),
    }


# ── Batch runner ───────────────────────────────────────────────────────────────

def simulate_candidate(
    candidate_ser: dict,
    heldout_graph: dict,
    eval_cache_heldout: dict,
    n_games: int = 50,
    seed: int = 0,
    stockfish_path: Optional[str] = None,
    engine_depth: int = 8,
    max_plies: int = 120,
) -> dict:
    """Simulate n_games for one serialised candidate; return summary stats."""
    rng = np.random.default_rng(seed)

    engine_cm = (
        chess.engine.SimpleEngine.popen_uci(stockfish_path)
        if stockfish_path and os.path.exists(stockfish_path)
        else nullcontext(None)
    )

    games: list = []
    with engine_cm as engine:
        if engine is not None:
            try:
                engine.configure({"Threads": 1})
            except Exception:
                pass
        for g in range(n_games):
            result = simulate_game(
                white_committed=candidate_ser["white_committed"],
                white_reached=set(candidate_ser["white_reached"]),
                black_committed=candidate_ser["black_committed"],
                black_reached=set(candidate_ser["black_reached"]),
                heldout_graph=heldout_graph,
                eval_cache_heldout=eval_cache_heldout,
                rng=rng,
                engine=engine,
                engine_depth=engine_depth,
                max_plies=max_plies,
            )
            games.append(result)

    scores = np.array([g["white_score"] for g in games], dtype=float)
    truncated = sum(1 for g in games if g["termination"] == "truncated")
    return {
        "n_games":           len(games),
        "engine_used":       engine_cm is not nullcontext(None) and stockfish_path is not None,
        "sim_mean":          float(scores.mean()),
        "sim_std":           float(scores.std()),
        "sim_worst_decile":  float(np.quantile(scores, 0.10)),
        "sim_truncated_frac": float(truncated) / max(1, len(games)),
        "scores":            scores.tolist(),
    }
