from collections import deque
import chess
from collector.api import query_position
from collector.config import (
    FIRST_MOVES,
    MAX_DEPTH,
    MAX_MOVES_PER_POSITION,
    MIN_DEPTH,
    MIN_GAMES,
    MIN_MOVE_FREQ,
)
from collector.database import get_conn


def position_key(moves_uci: list[str]) -> str:
    return " ".join(moves_uci) if moves_uci else "root"


def get_first_move_tag(moves_uci: list[str]) -> str:
    if not moves_uci:
        return "root"
    return FIRST_MOVES.get(moves_uci[0], "other")


def compute_scores(white_wins: int, draws: int, black_wins: int) -> dict | None:
    total = white_wins + draws + black_wins
    if total == 0:
        return None

    return {
        "white_wins": white_wins,
        "draws": draws,
        "black_wins": black_wins,
        "total_games": total,
        "white_perf": (white_wins + 0.5 * draws) / total,
        "white_win_rate": white_wins / total,
        "white_draw_rate": draws / total,
        "white_loss_rate": black_wins / total,
        "black_perf": (black_wins + 0.5 * draws) / total,
        "black_win_rate": black_wins / total,
        "black_draw_rate": draws / total,
        "black_loss_rate": white_wins / total,
    }


def is_visited(key: str, conn) -> bool:
    row = conn.execute(
        "SELECT 1 FROM visited_positions WHERE position_key = ?",
        (key,),
    ).fetchone()
    return row is not None


def mark_visited(key: str, total_games: int, skip_reason: str | None, conn):
    conn.execute(
        """
        INSERT OR IGNORE INTO visited_positions
        (position_key, total_games, skip_reason)
        VALUES (?, ?, ?)
        """,
        (key, total_games, skip_reason),
    )


def save_line(
    moves_san: list[str],
    moves_uci: list[str],
    fen: str,
    scores: dict,
    color: str,
    conn,
):
    if color == "white":
        perf = scores["white_perf"]
        win_rate = scores["white_win_rate"]
        draw_rate = scores["white_draw_rate"]
        loss_rate = scores["white_loss_rate"]
    else:
        perf = scores["black_perf"]
        win_rate = scores["black_win_rate"]
        draw_rate = scores["black_draw_rate"]
        loss_rate = scores["black_loss_rate"]

    try:
        conn.execute(
            """
            INSERT INTO opening_lines (
                moves_san, moves_uci, final_fen, color, depth,
                first_move_tag,
                white_wins, draws, black_wins, total_games,
                performance_score, win_rate, draw_rate, loss_rate
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
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
                loss_rate,
            ),
        )
    except Exception:
        pass


def run_traversal():
    conn = get_conn()
    queue = deque()
    queue.append(([], []))

    processed = 0
    saved = 0
    skipped = 0

    print("Starting traversal (breadth-first)...")
    print(
        f"  MIN_GAMES={MIN_GAMES}  MIN_MOVE_FREQ={MIN_MOVE_FREQ}"
        f"  MIN_DEPTH={MIN_DEPTH}  MAX_DEPTH={MAX_DEPTH}\n"
    )

    while queue:
        moves_uci, moves_san = queue.popleft()
        key = position_key(moves_uci)
        depth = len(moves_uci)

        if is_visited(key, conn):
            continue

        if depth > MAX_DEPTH:
            mark_visited(key, -1, "max_depth", conn)
            conn.commit()
            continue

        board = chess.Board()
        for uci in moves_uci:
            board.push(chess.Move.from_uci(uci))

        print(f"  [{processed}] depth={depth:2}  {key or 'root'}")
        data = query_position(moves_uci)
        processed += 1

        if data is None:
            mark_visited(key, -1, "api_error", conn)
            conn.commit()
            skipped += 1
            continue

        total = data["white"] + data["draws"] + data["black"]

        if total < MIN_GAMES:
            mark_visited(key, total, "insufficient_games", conn)
            conn.commit()
            skipped += 1
            continue

        mark_visited(key, total, None, conn)

        if depth >= MIN_DEPTH:
            scores = compute_scores(data["white"], data["draws"], data["black"])
            if scores:
                fen = board.fen()
                save_line(moves_san, moves_uci, fen, scores, "white", conn)
                save_line(moves_san, moves_uci, fen, scores, "black", conn)
                saved += 2

        conn.commit()

        if depth < MAX_DEPTH:
            qualifying_moves = [
                move_data
                for move_data in data.get("moves", [])
                if (
                    (move_data["white"] + move_data["draws"] + move_data["black"])
                    >= MIN_GAMES
                    and (
                        (move_data["white"] + move_data["draws"] + move_data["black"])
                        / total
                    )
                    >= MIN_MOVE_FREQ
                )
            ]

            qualifying_moves.sort(
                key=lambda m: m["white"] + m["draws"] + m["black"],
                reverse=True,
            )

            top_moves = qualifying_moves[:MAX_MOVES_PER_POSITION]

            for move_data in top_moves:
                try:
                    move = chess.Move.from_uci(move_data["uci"])
                    san = board.san(move)
                except Exception as exc:
                    print(f"  Move error {move_data['uci']}: {exc}")
                    continue

                queue.append((moves_uci + [move_data["uci"]], moves_san + [san]))

    total_lines = conn.execute("SELECT COUNT(*) FROM opening_lines").fetchone()[0]
    white_count = conn.execute(
        "SELECT COUNT(*) FROM opening_lines WHERE color='white'"
    ).fetchone()[0]
    black_count = conn.execute(
        "SELECT COUNT(*) FROM opening_lines WHERE color='black'"
    ).fetchone()[0]
    visited_count = conn.execute("SELECT COUNT(*) FROM visited_positions").fetchone()[0]
    skipped_count = conn.execute(
        "SELECT COUNT(*) FROM visited_positions WHERE skip_reason IS NOT NULL"
    ).fetchone()[0]

    conn.close()

    print(f"""
    {'=' * 45}
    TRAVERSAL COMPLETE
    {'=' * 45}
    Positions queried   : {processed}
    Positions skipped   : {skipped}
    Total lines saved   : {total_lines}
    White lines       : {white_count}
    Black lines       : {black_count}
    Positions visited   : {visited_count}
    Positions pruned    : {skipped_count}
    {'=' * 45}
    """)