"""testv2 — synthetic style-archetype opponent policies.

Derives per-style move distributions from the aggregate (band-pooled) move
frequencies by reweighting each move with a style-specific multiplier based on
chess features (capture, check, castle, queen move, pawn push, minor-piece
development).

Three styles: aggressive / defensive / positional. Each style is a
deterministic function of the training graph — no re-ingestion required.

The resulting object has the same shape as `base_policies` from
src/policies.py (dict[key → dict[fen → dict[move → prob]]]), so the rest of
the fitness pipeline consumes it unchanged.
"""
from __future__ import annotations

import os
import pickle
from typing import Dict, Set

import chess

from src.graph import aggregate_move_freq, load_graph


# ── Configuration ─────────────────────────────────────────────────────────────

STYLES = ("aggressive", "defensive", "positional")

EARLY_PLY_THRESHOLD = 20   # "early queen move" cut-off in plies

# The raw aggregate distribution is strongly concentrated (one move often > 50%),
# so per-style multipliers alone can't flip argmax. We first flatten the
# aggregate with a temperature (p^BETA), then apply style multipliers, then
# renormalise. BETA < 1 flattens; BETA = 1 is the raw aggregate.
AGG_TEMPERATURE_BETA = 0.4

CENTRAL_SQUARES = frozenset({chess.D4, chess.D5, chess.E4, chess.E5})
EXT_CENTRAL_SQUARES = frozenset({
    chess.C3, chess.C4, chess.C5, chess.C6,
    chess.D3, chess.D6,
    chess.E3, chess.E6,
    chess.F3, chess.F4, chess.F5, chess.F6,
})

# Feature multipliers per style. Each tag's multiplier is applied
# multiplicatively. A move with multiple tags has its weights multiplied.
STYLE_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "aggressive": {
        "capture":      6.0,
        "check":        5.0,
        "castle":       0.7,
        "queen_early":  4.0,
        "pawn_push2":   3.0,
        "minor_dev":    0.8,
        "king_move":    0.5,
        "to_central":   2.0,
        "to_ext_central": 1.4,
        "advances":     2.0,
        "retreats":     0.4,
    },
    "defensive": {
        "capture":      0.4,
        "check":        0.7,
        "castle":       8.0,
        "queen_early":  0.3,
        "pawn_push2":   0.5,
        "minor_dev":    1.2,
        "king_move":    1.4,
        "to_central":   0.6,
        "to_ext_central": 0.8,
        "advances":     0.5,
        "retreats":     3.5,
    },
    "positional": {
        "capture":      0.7,
        "check":        0.6,
        "castle":       2.5,
        "queen_early":  0.3,
        "pawn_push2":   0.6,
        "minor_dev":    5.0,
        "king_move":    0.6,
        "to_central":   3.0,
        "to_ext_central": 1.8,
        "advances":     1.0,
        "retreats":     1.0,
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _board_from_canonical(fen: str) -> chess.Board:
    """Construct a chess.Board from a canonical FEN (first 4 fields).

    python-chess requires all six FEN fields; we fill in halfmove=0 fullmove=1
    as defaults (they don't affect move legality or our features).
    """
    parts = fen.split(" ")
    if len(parts) == 4:
        fen = fen + " 0 1"
    return chess.Board(fen)


def _move_features(board: chess.Board, move: chess.Move, ply_depth: int) -> Set[str]:
    """Return the set of feature tags that apply to *move* on *board*."""
    tags: Set[str] = set()

    if board.is_capture(move):
        tags.add("capture")
    if board.gives_check(move):
        tags.add("check")
    if board.is_castling(move):
        tags.add("castle")

    piece = board.piece_at(move.from_square)
    if piece is None:
        return tags

    if piece.piece_type == chess.QUEEN and ply_depth < EARLY_PLY_THRESHOLD:
        tags.add("queen_early")

    if piece.piece_type == chess.PAWN:
        from_rank = chess.square_rank(move.from_square)
        to_rank = chess.square_rank(move.to_square)
        if abs(to_rank - from_rank) >= 2:
            tags.add("pawn_push2")

    if piece.piece_type in (chess.KNIGHT, chess.BISHOP):
        from_rank = chess.square_rank(move.from_square)
        home_rank = 0 if piece.color == chess.WHITE else 7
        if from_rank == home_rank:
            tags.add("minor_dev")

    if piece.piece_type == chess.KING and not board.is_castling(move):
        tags.add("king_move")

    if move.to_square in CENTRAL_SQUARES:
        tags.add("to_central")
    elif move.to_square in EXT_CENTRAL_SQUARES:
        tags.add("to_ext_central")

    # Advances toward the opponent / retreats toward our own back rank,
    # measured by which way the piece moves on the rank axis.
    from_rank = chess.square_rank(move.from_square)
    to_rank = chess.square_rank(move.to_square)
    delta = to_rank - from_rank
    if piece.color == chess.WHITE:
        if delta > 0:
            tags.add("advances")
        elif delta < 0:
            tags.add("retreats")
    else:
        if delta < 0:
            tags.add("advances")
        elif delta > 0:
            tags.add("retreats")

    return tags


def _style_weight(style: str, tags: Set[str]) -> float:
    """Product of per-tag multipliers for a given style; default 1.0 per tag."""
    mult = 1.0
    table = STYLE_MULTIPLIERS[style]
    for tag in tags:
        mult *= table.get(tag, 1.0)
    return mult


# ── Public API ────────────────────────────────────────────────────────────────

def build_style_policies(graph: dict) -> dict:
    """Build the per-style policies from *graph*.

    Returns a dict shaped {style → {fen → {move_uci → prob}}}. Each per-FEN
    distribution sums to 1.0. Positions whose board cannot be parsed, or whose
    aggregate distribution is empty, are skipped (downstream code handles
    missing FENs as uniform / off-graph).
    """
    out: Dict[str, Dict[str, Dict[str, float]]] = {s: {} for s in STYLES}

    for fen, node in graph["nodes"].items():
        if not node.get("children"):
            continue
        agg = aggregate_move_freq(graph, fen)
        if not agg:
            continue

        try:
            board = _board_from_canonical(fen)
        except Exception:
            continue

        ply = node.get("ply_depth", 0)

        # Compute feature-tags once per move.
        move_tags: Dict[str, Set[str]] = {}
        for uci in agg:
            try:
                mv = chess.Move.from_uci(uci)
                if mv in board.legal_moves:
                    move_tags[uci] = _move_features(board, mv, ply)
                else:
                    move_tags[uci] = set()
            except Exception:
                move_tags[uci] = set()

        # Flatten the aggregate distribution with a temperature exponent so
        # that style multipliers can actually shift the argmax. Without this,
        # one move often dominates so heavily that no realistic multiplier
        # can flip it.
        flattened: Dict[str, float] = {}
        flat_total = 0.0
        for uci, p in agg.items():
            f = p ** AGG_TEMPERATURE_BETA
            flattened[uci] = f
            flat_total += f
        if flat_total <= 0.0:
            continue
        for uci in flattened:
            flattened[uci] /= flat_total

        for style in STYLES:
            weighted: Dict[str, float] = {}
            total = 0.0
            for uci, p in flattened.items():
                w = p * _style_weight(style, move_tags[uci])
                if w > 0.0:
                    weighted[uci] = w
                    total += w
            if total > 0.0:
                for uci in weighted:
                    weighted[uci] /= total
            out[style][fen] = weighted

    return out


# ── I/O ───────────────────────────────────────────────────────────────────────

def save_style_policies(policies: dict, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(policies, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_style_policies(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Calibration check ────────────────────────────────────────────────────────

def calibration_report(graph: dict, policies: dict) -> dict:
    """Per-style calibration metrics on positions with >= 2 legal moves.

    Returns {style → {"argmax_diff_frac": float, "mean_tv_vs_agg": float}}.

    Argmax-differs is a strict bar: many style/aggregate pairs share an
    argmax but assign very different mass to the second/third move. TV
    distance captures the full distributional shift the walk integral
    actually sees.
    """
    report: Dict[str, dict] = {}
    for style in STYLES:
        n_total = 0
        n_diff = 0
        tv_sum = 0.0
        for fen, node in graph["nodes"].items():
            if not node.get("children") or len(node["children"]) < 2:
                continue
            agg = aggregate_move_freq(graph, fen)
            if not agg:
                continue
            style_dist = policies[style].get(fen, {})
            if not style_dist:
                continue
            agg_argmax = max(agg, key=agg.get)
            style_argmax = max(style_dist, key=style_dist.get)
            n_total += 1
            if agg_argmax != style_argmax:
                n_diff += 1
            moves = set(agg) | set(style_dist)
            tv = 0.5 * sum(abs(agg.get(m, 0.0) - style_dist.get(m, 0.0))
                           for m in moves)
            tv_sum += tv
        report[style] = {
            "argmax_diff_frac": (n_diff / n_total) if n_total > 0 else 0.0,
            "mean_tv_vs_agg": (tv_sum / n_total) if n_total > 0 else 0.0,
        }
    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    data_dir = "data"
    graph_path = os.path.join(data_dir, "graph_train.pkl")
    out_path = os.path.join(data_dir, "style_policies_train.pkl")

    print(f"Loading {graph_path} ...")
    graph = load_graph(graph_path)
    print(f"Building style policies over {len(graph['nodes'])} positions ...")
    policies = build_style_policies(graph)
    save_style_policies(policies, out_path)
    print(f"Saved {out_path}")

    print("\nCalibration report:")
    print(f"  {'style':12s}  argmax_diff   mean_TV_vs_agg   flag")
    report = calibration_report(graph, policies)
    for style, m in report.items():
        ok = (m["argmax_diff_frac"] >= 0.60) or (m["mean_tv_vs_agg"] >= 0.20)
        flag = "OK" if ok else "LOW"
        print(f"  {style:12s}  {m['argmax_diff_frac']*100:5.1f}%        "
              f"{m['mean_tv_vs_agg']:.3f}            [{flag}]")


if __name__ == "__main__":
    _cli()
