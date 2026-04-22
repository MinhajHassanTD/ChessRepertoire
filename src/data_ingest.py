"""
C1 — Data ingest
Recursively queries the Lichess Opening Explorer API from the starting position,
caching every response into a SQLite database. Resumable after interruption.
"""

import sqlite3
import time
import logging
from collections import deque
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv
import os

from src.config import (
    RATE_LIMIT_SLEEP,
    MAX_PLY_DEPTH,
    MIN_MOVE_FREQUENCY,
    MIN_GAMES_SHALLOW,
    MIN_GAMES_MID,
    MIN_GAMES_DEEP,
    MIN_GAMES_MID_CUTOFF,
    MIN_GAMES_DEEP_CUTOFF,
    TRAIN_UNTIL,
    HELDOUT_SINCE,
    API_SPEEDS,
    RATING_BANDS,
    STARTING_FEN,
    LICHESS_BASE_URL,
)

load_dotenv()
LICHESS_API_TOKEN = os.environ["LICHESS_API_TOKEN"]

# ── Derived band constants (computed from RATING_BANDS in config) ─────────────
# Lichess API uses the lower bound of each band as its ratings parameter tag.
# e.g. "1000-1399" → tag "1000", label "1000-1399"
BAND_TAGS = [band.split("-")[0] for band in RATING_BANDS]
BAND_TAG_TO_LABEL = {band.split("-")[0]: band for band in RATING_BANDS}
_AGGREGATE_RATINGS_PARAM = ",".join(BAND_TAGS)

def min_games_for_depth(depth: int) -> int:
    """Depth-dependent minimum play_count required to enqueue a child position."""
    if depth <= MIN_GAMES_MID_CUTOFF:
        return MIN_GAMES_SHALLOW
    if depth <= MIN_GAMES_DEEP_CUTOFF:
        return MIN_GAMES_MID
    return MIN_GAMES_DEEP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── FEN helpers ───────────────────────────────────────────────────────────────

def canonical_fen(fen: str) -> str:
    """Return the first four space-separated fields of a FEN string."""
    return " ".join(fen.split()[:4])


# ── Database ──────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS positions (
    fen TEXT NOT NULL,
    split TEXT NOT NULL CHECK(split IN ('train','heldout')),
    ply_depth INTEGER NOT NULL,
    total_games INTEGER NOT NULL,
    white_wins INTEGER NOT NULL,
    draws INTEGER NOT NULL,
    black_wins INTEGER NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (fen, split)
);

CREATE TABLE IF NOT EXISTS moves (
    parent_fen TEXT NOT NULL,
    split TEXT NOT NULL,
    move_uci TEXT NOT NULL,
    move_san TEXT NOT NULL,
    child_fen TEXT NOT NULL,
    rating_band TEXT NOT NULL,
    play_count INTEGER NOT NULL,
    white_wins INTEGER NOT NULL,
    draws INTEGER NOT NULL,
    black_wins INTEGER NOT NULL,
    PRIMARY KEY (parent_fen, split, move_uci, rating_band)
);

-- Tracks which (fen, split, rating_band) API calls have been completed.
-- A row here means the call was made and the response was stored, even if the
-- response contained zero moves.
CREATE TABLE IF NOT EXISTS fetched_bands (
    fen TEXT NOT NULL,
    split TEXT NOT NULL,
    rating_band TEXT NOT NULL,
    PRIMARY KEY (fen, split, rating_band)
);

CREATE INDEX IF NOT EXISTS idx_moves_parent ON moves(parent_fen, split);
CREATE INDEX IF NOT EXISTS idx_positions_depth ON positions(ply_depth);
"""


def init_db(db_path: str) -> sqlite3.Connection:
    """Open (or create) the SQLite database and ensure schema is present."""
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


def _mark_band_fetched(
    conn: sqlite3.Connection, fen: str, split: str, rating_band: str
) -> None:
    """Record that the API call for (fen, split, rating_band) has completed."""
    conn.execute(
        "INSERT OR IGNORE INTO fetched_bands (fen, split, rating_band) VALUES (?, ?, ?)",
        (fen, split, rating_band),
    )


def _position_fully_processed(conn: sqlite3.Connection, fen: str) -> bool:
    """
    Return True if this FEN has already been fully processed:
      - exists in positions for both splits
      - fetched_bands contains rows for both splits × all 4 rating_band variants
        (aggregate + three band labels)
    """
    cur = conn.execute(
        "SELECT COUNT(*) FROM positions WHERE fen = ?", (fen,)
    )
    if cur.fetchone()[0] < 2:
        return False

    expected_bands = {"aggregate"} | set(BAND_TAG_TO_LABEL.values())
    expected_count = len(expected_bands)
    for split in ("train", "heldout"):
        cur = conn.execute(
            "SELECT COUNT(*) FROM fetched_bands WHERE fen = ? AND split = ?",
            (fen, split),
        )
        if cur.fetchone()[0] < expected_count:
            return False
    return True


def _insert_position(
    conn: sqlite3.Connection,
    fen: str,
    split: str,
    ply_depth: int,
    data: dict,
) -> None:
    """Insert or replace a position row."""
    from datetime import datetime, timezone

    fetched_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT OR REPLACE INTO positions
            (fen, split, ply_depth, total_games, white_wins, draws, black_wins, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            fen,
            split,
            ply_depth,
            data.get("white", 0) + data.get("draws", 0) + data.get("black", 0),
            data.get("white", 0),
            data.get("draws", 0),
            data.get("black", 0),
            fetched_at,
        ),
    )


def _insert_moves(
    conn: sqlite3.Connection,
    parent_fen: str,
    split: str,
    rating_band: str,
    moves: list,
    board_after_moves: dict,
) -> None:
    """Insert move rows for the given parent_fen / split / rating_band."""
    for move in moves:
        uci = move.get("uci", "")
        san = move.get("san", "")
        child_fen = board_after_moves.get(uci, "")
        conn.execute(
            """
            INSERT OR REPLACE INTO moves
                (parent_fen, split, move_uci, move_san, child_fen,
                 rating_band, play_count, white_wins, draws, black_wins)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                parent_fen,
                split,
                uci,
                san,
                child_fen,
                rating_band,
                move.get("white", 0) + move.get("draws", 0) + move.get("black", 0),
                move.get("white", 0),
                move.get("draws", 0),
                move.get("black", 0),
            ),
        )


# ── API ───────────────────────────────────────────────────────────────────────

def _api_call(fen: str, split: str, ratings: str) -> dict:
    """
    Call the Lichess Opening Explorer API for one (fen, split, ratings) triple.
    Returns the parsed JSON response dict (or empty structure on error).
    """
    params: dict = {
        "variant": "standard",
        "fen": fen,
        "speeds": API_SPEEDS,
        "ratings": ratings,
    }
    if split == "train":
        params["until"] = TRAIN_UNTIL
    else:  # heldout
        params["since"] = HELDOUT_SINCE

    headers = {"Authorization": f"Bearer {LICHESS_API_TOKEN}"}
    url = f"{LICHESS_BASE_URL}?{urlencode(params)}"
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        log.warning("API error for fen=%s split=%s ratings=%s: %s", fen, split, ratings, exc)
        return {"white": 0, "draws": 0, "black": 0, "moves": []}


def _compute_child_fens(fen: str, moves: list) -> dict:
    """
    Use python-chess to compute the child FEN for each move UCI.
    Returns {uci: canonical_child_fen}.
    """
    import chess

    board = chess.Board(fen + " 0 1")  # append dummy clock fields
    result = {}
    for move_data in moves:
        uci = move_data.get("uci", "")
        try:
            mv = chess.Move.from_uci(uci)
            board.push(mv)
            result[uci] = canonical_fen(board.fen())
            board.pop()
        except Exception:
            pass
    return result


# ── Main ingest ───────────────────────────────────────────────────────────────

def run_ingest(db_path: str = "data/snapshot.db") -> None:
    """
    Run the full ingest from the starting position, resuming if interrupted.
    Populates data/snapshot.db with all positions up to depth 8.
    """
    conn = init_db(db_path)
    queue: deque = deque()

    start_fen = canonical_fen(STARTING_FEN)
    queue.append((start_fen, 0))

    processed = 0
    api_calls = 0

    while queue:
        fen, depth = queue.popleft()

        if _position_fully_processed(conn, fen):
            # Still enqueue children so the resume walk continues deeper
            if depth < MAX_PLY_DEPTH:
                cur = conn.execute(
                    """
                    SELECT DISTINCT child_fen, play_count
                    FROM moves
                    WHERE parent_fen = ? AND split = 'train' AND rating_band = 'aggregate'
                    """,
                    (fen,),
                )
                total_agg = conn.execute(
                    "SELECT COALESCE(SUM(play_count),0) FROM moves WHERE parent_fen = ? AND split = 'train' AND rating_band = 'aggregate'",
                    (fen,),
                ).fetchone()[0]
                for child_fen, play_count in cur.fetchall():
                    frequency = play_count / total_agg if total_agg > 0 else 0.0
                    if play_count >= min_games_for_depth(depth) and frequency >= MIN_MOVE_FREQUENCY:
                        queue.append((child_fen, depth + 1))
            continue

        last_agg_moves: list = []

        for split in ("train", "heldout"):
            # Aggregate call
            time.sleep(RATE_LIMIT_SLEEP)
            agg = _api_call(fen, split, _AGGREGATE_RATINGS_PARAM)
            api_calls += 1

            child_fens = _compute_child_fens(fen, agg.get("moves", []))

            _insert_position(conn, fen, split, depth, agg)
            _insert_moves(conn, fen, split, "aggregate", agg.get("moves", []), child_fens)
            _mark_band_fetched(conn, fen, split, "aggregate")

            # Per-band calls
            for band_tag in BAND_TAGS:
                time.sleep(RATE_LIMIT_SLEEP)
                band_resp = _api_call(fen, split, band_tag)
                api_calls += 1
                band_label = BAND_TAG_TO_LABEL[band_tag]
                _insert_moves(
                    conn, fen, split, band_label, band_resp.get("moves", []), child_fens
                )
                _mark_band_fetched(conn, fen, split, band_label)

            if split == "train":
                last_agg_moves = agg.get("moves", [])

        conn.commit()
        processed += 1

        # Enqueue children (based on aggregate train counts)
        if depth < MAX_PLY_DEPTH:
            # Recompute child_fens from the last train aggregate response
            import chess
            try:
                board = chess.Board(fen + " 0 1")
            except Exception:
                board = None

            total_agg = sum(
                m.get("white", 0) + m.get("draws", 0) + m.get("black", 0)
                for m in last_agg_moves
            )

            for move_data in last_agg_moves:
                play_count = (
                    move_data.get("white", 0)
                    + move_data.get("draws", 0)
                    + move_data.get("black", 0)
                )
                frequency = play_count / total_agg if total_agg > 0 else 0.0
                if play_count >= min_games_for_depth(depth) and frequency >= MIN_MOVE_FREQUENCY:
                    uci = move_data.get("uci", "")
                    if board is not None:
                        try:
                            import chess as _chess
                            mv = _chess.Move.from_uci(uci)
                            board.push(mv)
                            child_fen = canonical_fen(board.fen())
                            board.pop()
                            queue.append((child_fen, depth + 1))
                        except Exception:
                            pass

        if processed % 100 == 0:
            log.info(
                "Progress: %d positions processed, %d API calls made, queue size %d",
                processed,
                api_calls,
                len(queue),
            )

    conn.close()
    log.info(
        "Ingest complete: %d positions processed, %d total API calls.",
        processed,
        api_calls,
    )


if __name__ == "__main__":
    run_ingest()
