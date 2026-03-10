from collections import deque
import chess
from collector.api import query_position
from collector.database import get_conn
from collector.config import (
    MIN_GAMES, MIN_MOVE_FREQ,
    MIN_DEPTH, MAX_DEPTH,
    FIRST_MOVES
)

# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def position_key(moves_uci: list[str]) -> str:
    """Stable string key for a position. Root = 'root'."""
    return " ".join(moves_uci) if moves_uci else "root"


def get_first_move_tag(moves_uci: list[str]) -> str:
    """
    Which opening family does this line belong to?
    Always determined by White's first move regardless of
    which color's perspective we are saving.
    """
    if not moves_uci:
        return "root"
    return FIRST_MOVES.get(moves_uci[0], "other")


def compute_scores(white_wins: int, draws: int, black_wins: int) -> dict | None:
    """
    Compute all metrics from raw counts for BOTH color perspectives.
    Returns None if total games is zero.
    """
    total = white_wins + draws + black_wins
    if total == 0:
        return None

    return {
        "white_wins":  white_wins,
        "draws":       draws,
        "black_wins":  black_wins,
        "total_games": total,

        # White perspective — higher is better for White
        "white_perf":      (white_wins + 0.5 * draws) / total,
        "white_win_rate":  white_wins / total,
        "white_draw_rate": draws / total,
        "white_loss_rate": black_wins / total,

        # Black perspective — higher is better for Black
        "black_perf":      (black_wins + 0.5 * draws) / total,
        "black_win_rate":  black_wins / total,
        "black_draw_rate": draws / total,
        "black_loss_rate": white_wins / total,
    }


def is_visited(key: str, conn) -> bool:
    row = conn.execute(
        "SELECT 1 FROM visited_positions WHERE position_key = ?",
        (key,)
    ).fetchone()
    return row is not None


def mark_visited(key: str, total_games: int, skip_reason: str | None, conn):
    conn.execute(
        """INSERT OR IGNORE INTO visited_positions
           (position_key, total_games, skip_reason)
           VALUES (?, ?, ?)""",
        (key, total_games, skip_reason)
    )


def save_line(moves_san: list[str], moves_uci: list[str],
              fen: str, scores: dict, color: str, conn):
    """
    Save one line from one color's perspective.
    UNIQUE(final_fen, color) silently rejects transpositions.
    """

    if color == "white":
        perf      = scores["white_perf"]
        win_rate  = scores["white_win_rate"]
        draw_rate = scores["white_draw_rate"]
        loss_rate = scores["white_loss_rate"]
    else:
        perf      = scores["black_perf"]
        win_rate  = scores["black_win_rate"]
        draw_rate = scores["black_draw_rate"]
        loss_rate = scores["black_loss_rate"]

    try:
        conn.execute("""
            INSERT INTO opening_lines (
                moves_san, moves_uci, final_fen, color, depth,
                first_move_tag,
                white_wins, draws, black_wins, total_games,
                performance_score, win_rate, draw_rate, loss_rate
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            " ".join(moves_san),
            " ".join(moves_uci),
            fen,
            color,
            len(moves_uci),
            get_first_move_tag(moves_uci),
            scores["white_wins"],
            scores["draws"],
            scores["black_wins"],
            scores["total_games"],
            perf,
            win_rate,
            draw_rate,
            loss_rate
        ))

    except Exception:
        # UNIQUE constraint fired — transposition, skip silently
        pass


# ─────────────────────────────────────────────────────────
# Main traversal
# ─────────────────────────────────────────────────────────

def run_traversal():
    """
    Breadth-first traversal of the Lichess opening tree.

    Single pass — saves both White and Black perspective lines
    at every qualifying position. No separate white/black runs.

    Resumable — already-visited positions are skipped.
    """

    conn = get_conn()

    # Queue holds (moves_uci, moves_san) tuples.
    # We reconstruct the board from moves on dequeue
    # rather than storing Board objects in the queue.
    queue = deque()
    queue.append(([], []))   # Root: no moves yet

    processed = 0
    saved     = 0
    skipped   = 0

    print("Starting traversal (breadth-first)...")
    print(f"  MIN_GAMES={MIN_GAMES}  MIN_MOVE_FREQ={MIN_MOVE_FREQ}"
          f"  MIN_DEPTH={MIN_DEPTH}  MAX_DEPTH={MAX_DEPTH}\n")

    while queue:

        moves_uci, moves_san = queue.popleft()
        key   = position_key(moves_uci)
        depth = len(moves_uci)

        # ── Already visited ───────────────────────────────
        if is_visited(key, conn):
            continue

        # ── Hard depth ceiling ────────────────────────────
        if depth > MAX_DEPTH:
            mark_visited(key, -1, "max_depth", conn)
            conn.commit()
            continue

        # ── Reconstruct board ──────────────────────────────
        # Done here rather than storing boards in the queue
        # to keep memory usage flat regardless of queue size
        board = chess.Board()
        for uci in moves_uci:
            board.push(chess.Move.from_uci(uci))

        # ── Query Lichess ─────────────────────────────────
        print(f"  [{processed}] depth={depth:2}  {key or 'root'}")
        data = query_position(moves_uci)
        processed += 1

        if data is None:
            mark_visited(key, -1, "api_error", conn)
            conn.commit()
            skipped += 1
            continue

        total = data["white"] + data["draws"] + data["black"]

        # ── Game count threshold ───────────────────────────
        if total < MIN_GAMES:
            mark_visited(key, total, "insufficient_games", conn)
            conn.commit()
            skipped += 1
            continue

        mark_visited(key, total, None, conn)

        # ── Save as line if deep enough ────────────────────
        # Only save positions that represent genuine preparation.
        # Positions shallower than MIN_DEPTH are too generic.
        if depth >= MIN_DEPTH:

            scores = compute_scores(
                data["white"], data["draws"], data["black"]
            )

            if scores:
                fen = board.fen()

                # Save White perspective
                save_line(moves_san, moves_uci, fen,
                          scores, "white", conn)

                # Save Black perspective
                # Same moves and board — different performance score
                save_line(moves_san, moves_uci, fen,
                          scores, "black", conn)

                saved += 2

        conn.commit()

        # ── Enqueue qualifying children ────────────────────
        if depth < MAX_DEPTH:

            for move_data in data.get("moves", []):

                child_total = (move_data["white"] +
                               move_data["draws"] +
                               move_data["black"])

                # Absolute game count threshold
                if child_total < MIN_GAMES:
                    continue

                # Relative frequency threshold
                frequency = child_total / total
                if frequency < MIN_MOVE_FREQ:
                    continue

                try:
                    move = chess.Move.from_uci(move_data["uci"])
                    san  = board.san(move)
                except Exception as e:
                    print(f"  Move error {move_data['uci']}: {e}")
                    continue

                queue.append((
                    moves_uci + [move_data["uci"]],
                    moves_san + [san]
                ))

    # ── Summary ───────────────────────────────────────────
    total_lines = conn.execute(
        "SELECT COUNT(*) FROM opening_lines"
    ).fetchone()[0]

    white_count = conn.execute(
        "SELECT COUNT(*) FROM opening_lines WHERE color='white'"
    ).fetchone()[0]

    black_count = conn.execute(
        "SELECT COUNT(*) FROM opening_lines WHERE color='black'"
    ).fetchone()[0]

    visited_count = conn.execute(
        "SELECT COUNT(*) FROM visited_positions"
    ).fetchone()[0]

    skipped_count = conn.execute(
        "SELECT COUNT(*) FROM visited_positions "
        "WHERE skip_reason IS NOT NULL"
    ).fetchone()[0]

    conn.close()

    print(f"""
{'='*45}
TRAVERSAL COMPLETE
{'='*45}
Positions queried   : {processed}
Positions skipped   : {skipped}
Total lines saved   : {total_lines}
  White lines       : {white_count}
  Black lines       : {black_count}
Positions visited   : {visited_count}
Positions pruned    : {skipped_count}
{'='*45}
    """)