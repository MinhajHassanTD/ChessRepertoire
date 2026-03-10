# collector/traversal.py

import chess
import sqlite3
from collector.api import query_position
from collector.config import MIN_GAMES, MIN_MOVE_FREQUENCY, MAX_DEPTH_MOVES


def get_first_move_tag(moves_uci: list[str], board: chess.Board) -> str:
    if not moves_uci:
        return "root"

    first = moves_uci[0]
    mapping = {
        "e2e4": "e4",
        "d2d4": "d4",
        "c2c4": "c4",
        "g1f3": "Nf3",
    }
    return mapping.get(first, "other")


def compute_performance_score(white: int, draws: int, black: int, color: str) -> float:
    total = white + draws + black
    if total == 0:
        return 0.0

    if color == "white":
        return (white + 0.5 * draws) / total
    return (black + 0.5 * draws) / total


def already_visited(color: str, position_key: str, conn) -> bool:
    row = conn.execute(
        "SELECT visited FROM collection_log WHERE color = ? AND position_uci = ?",
        (color, position_key),
    ).fetchone()
    return row is not None


def log_position(
    color: str, position_key: str, total_games: int, skipped_reason: str | None, conn
):
    conn.execute(
        """INSERT OR IGNORE INTO collection_log
           (color, position_uci, visited, total_games, skipped_reason)
           VALUES (?, ?, 1, ?, ?)""",
        (color, position_key, total_games, skipped_reason),
    )


def save_line(moves_san: list[str], moves_uci: list[str], board: chess.Board,
              color: str, api_data: dict, conn):
    total = api_data["white"] + api_data["draws"] + api_data["black"]
    fen = board.fen()

    perf = compute_performance_score(
        api_data["white"], api_data["draws"], api_data["black"], color
    )

    try:
        conn.execute(
            """
            INSERT INTO opening_lines
            (moves_san, moves_uci, final_fen, color, depth,
             first_move_tag, win_count, draw_count, loss_count,
             total_games, win_rate, draw_rate, loss_rate, performance_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                " ".join(moves_san),
                " ".join(moves_uci),
                fen,
                color,
                len(moves_uci),
                get_first_move_tag(moves_uci, board),
                api_data["white"],
                api_data["draws"],
                api_data["black"],
                total,
                api_data["white"] / total if total > 0 else 0,
                api_data["draws"] / total if total > 0 else 0,
                api_data["black"] / total if total > 0 else 0,
                perf,
            ),
        )
        return True
    except sqlite3.IntegrityError:
        print(f"  Transposition detected, skipping ({color}): {' '.join(moves_san)}")
        return False


def traverse(moves_uci: list[str], moves_san: list[str],
             board: chess.Board, color: str, depth: int, conn):
    position_key = " ".join(moves_uci) if moves_uci else "root"

    if already_visited(color, position_key, conn):
        return

    if depth >= MAX_DEPTH_MOVES:
        log_position(color, position_key, -1, "max_depth_reached", conn)
        conn.commit()
        return

    print(f"  Querying [{color}]: {position_key} (depth {depth})")
    data = query_position(moves_uci)

    if data is None:
        log_position(color, position_key, -1, "api_error", conn)
        conn.commit()
        return

    total_games = data["white"] + data["draws"] + data["black"]

    if total_games < MIN_GAMES:
        log_position(color, position_key, total_games, "insufficient_games", conn)
        conn.commit()
        return

    log_position(color, position_key, total_games, None, conn)

    if depth >= 6:
        save_line(moves_san, moves_uci, board, color, data, conn)

    conn.commit()

    for move_data in data.get("moves", []):
        move_total = move_data["white"] + move_data["draws"] + move_data["black"]
        if move_total < MIN_GAMES:
            continue

        frequency = move_total / total_games
        if frequency < MIN_MOVE_FREQUENCY:
            continue

        try:
            move = chess.Move.from_uci(move_data["uci"])
            san = board.san(move)
            board.push(move)

            traverse(
                moves_uci + [move_data["uci"]],
                moves_san + [san],
                board,
                color,
                depth + 1,
                conn,
            )

            board.pop()
        except Exception as e:
            print(f"  Move error {move_data['uci']}: {e}")
            continue