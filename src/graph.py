"""
C2 — Position graph
Loads the SQLite snapshot into an in-memory Python dict (one per split),
then pickles it to / loads it from disk.
"""

import pickle
import sqlite3
from typing import Optional

# ── Constants ─────────────────────────────────────────────────────────────────

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
RATING_BANDS = ("1600-1799", "1800-1999", "2000-2199")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _turn_from_fen(fen: str) -> str:
    """Return 'white' or 'black' from the active-colour field of a canonical FEN."""
    parts = fen.split(" ")
    return "white" if parts[1] == "w" else "black"


# ── Core builder ──────────────────────────────────────────────────────────────

def build_graph(db_path: str, split: str) -> dict:
    """Read from SQLite and construct the position graph for *split*."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # ── 1. Load all positions ─────────────────────────────────────────────────
    cur.execute(
        """
        SELECT fen, ply_depth, total_games, white_wins, draws, black_wins
        FROM positions
        WHERE split = ?
        """,
        (split,),
    )
    position_rows = cur.fetchall()

    # ── 2. Load all moves (both aggregate and per-band) ───────────────────────
    cur.execute(
        """
        SELECT parent_fen, move_uci, move_san, child_fen,
               rating_band, play_count, white_wins, draws, black_wins
        FROM moves
        WHERE split = ?
        """,
        (split,),
    )
    move_rows = cur.fetchall()
    conn.close()

    # ── 3. Index moves by (parent_fen, rating_band) ───────────────────────────
    # agg_moves[parent_fen][move_uci] = {child_fen, move_san, play_count}
    agg_moves: dict = {}
    # band_moves[parent_fen][band][move_uci] = play_count
    band_moves: dict = {}
    # band_totals[parent_fen][band] = {games, wins, draws, losses}
    band_totals: dict = {}

    for row in move_rows:
        pfn = row["parent_fen"]
        uci = row["move_uci"]
        band = row["rating_band"]

        if band == "aggregate":
            agg_moves.setdefault(pfn, {})[uci] = {
                "child_fen": row["child_fen"],
                "move_san": row["move_san"],
                "play_count": row["play_count"],
            }
        elif band in RATING_BANDS:
            band_moves.setdefault(pfn, {}).setdefault(band, {})[uci] = row["play_count"]
            bt = band_totals.setdefault(pfn, {}).setdefault(
                band, {"games": 0, "wins": 0, "draws": 0, "losses": 0}
            )
            bt["games"] += row["play_count"]
            bt["wins"] += row["white_wins"]
            bt["draws"] += row["draws"]
            bt["losses"] += row["black_wins"]

    # ── 4. Build nodes ────────────────────────────────────────────────────────
    nodes: dict = {}
    for row in position_rows:
        fen = row["fen"]
        total = row["total_games"]
        white = row["white_wins"]
        draws = row["draws"]

        score_raw = (white + 0.5 * draws) / total if total > 0 else 0.5

        # band_stats — derived from summed move counts for this position
        band_stats: dict = {}
        for b in RATING_BANDS:
            if fen in band_totals and b in band_totals[fen]:
                band_stats[b] = dict(band_totals[fen][b])
            else:
                band_stats[b] = {"games": 0, "wins": 0, "draws": 0, "losses": 0}

        # children — keyed by move_uci
        children: dict = {}
        for uci, agg in agg_moves.get(fen, {}).items():
            bcs: dict = {}
            for b in RATING_BANDS:
                bcs[b] = band_moves.get(fen, {}).get(b, {}).get(uci, 0)
            children[uci] = {
                "child_fen": agg["child_fen"],
                "move_san": agg["move_san"],
                "aggregate_count": agg["play_count"],
                "band_counts": bcs,
            }

        nodes[fen] = {
            "fen": fen,
            "ply_depth": row["ply_depth"],
            "total_games": total,
            "score_raw": score_raw,
            "turn": _turn_from_fen(fen),
            "band_stats": band_stats,
            "children": children,
        }

    return {
        "root_fen": STARTING_FEN,
        "nodes": nodes,
    }


# ── Persistence ───────────────────────────────────────────────────────────────

def save_graph(graph: dict, path: str) -> None:
    """Pickle the graph to *path*."""
    with open(path, "wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_graph(path: str) -> dict:
    """Load a pickled graph from *path*."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Query helpers ─────────────────────────────────────────────────────────────

def children_of(graph: dict, fen: str) -> dict:
    """Return the children dict for *fen*, or {} if the position has no children."""
    node = graph["nodes"].get(fen)
    if node is None:
        return {}
    return node.get("children", {})


def aggregate_move_freq(graph: dict, fen: str) -> dict:
    """Return {move_uci: frequency} using aggregate counts, summing to 1.0."""
    kids = children_of(graph, fen)
    if not kids:
        return {}
    total = sum(c["aggregate_count"] for c in kids.values())
    if total == 0:
        n = len(kids)
        return {uci: 1.0 / n for uci in kids}
    return {uci: c["aggregate_count"] / total for uci, c in kids.items()}


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os

    db = sys.argv[1] if len(sys.argv) > 1 else "data/L2.db"

    for split, out in [("train", "data/graph_train.pkl"), ("heldout", "data/graph_heldout.pkl")]:
        print(f"Building {split} graph …", flush=True)
        g = build_graph(db, split)
        print(f"  {len(g['nodes'])} nodes")
        save_graph(g, out)
        print(f"  Saved -> {out}")
