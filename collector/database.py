import sqlite3
from pathlib import Path

DB_PATH = Path("data/repertoire.db")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _table_sql(conn, table_name: str) -> str | None:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row["sql"] if row else None


def _migrate_opening_lines_if_needed(conn):
    sql = _table_sql(conn, "opening_lines")
    if not sql:
        return
    if "UNIQUE(final_fen, color)" in sql:
        return

    conn.execute("ALTER TABLE opening_lines RENAME TO opening_lines_old")
    conn.execute(
        """
        CREATE TABLE opening_lines (
            line_id            INTEGER PRIMARY KEY AUTOINCREMENT,
            moves_san          TEXT NOT NULL,
            moves_uci          TEXT NOT NULL,
            final_fen          TEXT NOT NULL,
            color              TEXT NOT NULL,
            depth              INTEGER NOT NULL,
            first_move_tag     TEXT NOT NULL,
            win_count          INTEGER NOT NULL,
            draw_count         INTEGER NOT NULL,
            loss_count         INTEGER NOT NULL,
            total_games        INTEGER NOT NULL,
            win_rate           REAL NOT NULL,
            draw_rate          REAL NOT NULL,
            loss_rate          REAL NOT NULL,
            performance_score  REAL NOT NULL,
            UNIQUE(final_fen, color)
        )
        """
    )
    conn.execute(
        """
        INSERT INTO opening_lines (
            line_id, moves_san, moves_uci, final_fen, color, depth, first_move_tag,
            win_count, draw_count, loss_count, total_games, win_rate, draw_rate,
            loss_rate, performance_score
        )
        SELECT
            line_id, moves_san, moves_uci, final_fen, color, depth, first_move_tag,
            win_count, draw_count, loss_count, total_games, win_rate, draw_rate,
            loss_rate, performance_score
        FROM opening_lines_old
        """
    )
    conn.execute("DROP TABLE opening_lines_old")


def _migrate_collection_log_if_needed(conn):
    sql = _table_sql(conn, "collection_log")
    if not sql:
        return
    if "PRIMARY KEY (color, position_uci)" in sql:
        return

    conn.execute("ALTER TABLE collection_log RENAME TO collection_log_old")
    conn.execute(
        """
        CREATE TABLE collection_log (
            color          TEXT NOT NULL,
            position_uci   TEXT NOT NULL,
            visited        INTEGER DEFAULT 1,
            total_games    INTEGER,
            skipped_reason TEXT,
            PRIMARY KEY (color, position_uci)
        )
        """
    )
    # Existing logs were historically from a single run; keep them under white.
    conn.execute(
        """
        INSERT INTO collection_log (color, position_uci, visited, total_games, skipped_reason)
        SELECT 'white', position_uci, visited, total_games, skipped_reason
        FROM collection_log_old
        """
    )
    conn.execute("DROP TABLE collection_log_old")


def initialize_database():
    """Create tables and run schema migrations."""
    conn = get_connection()
    c = conn.cursor()

    c.executescript(
        """
        CREATE TABLE IF NOT EXISTS opening_lines (
            line_id            INTEGER PRIMARY KEY AUTOINCREMENT,
            moves_san          TEXT NOT NULL,
            moves_uci          TEXT NOT NULL,
            final_fen          TEXT NOT NULL,
            color              TEXT NOT NULL,
            depth              INTEGER NOT NULL,
            first_move_tag     TEXT NOT NULL,
            win_count          INTEGER NOT NULL,
            draw_count         INTEGER NOT NULL,
            loss_count         INTEGER NOT NULL,
            total_games        INTEGER NOT NULL,
            win_rate           REAL NOT NULL,
            draw_rate          REAL NOT NULL,
            loss_rate          REAL NOT NULL,
            performance_score  REAL NOT NULL,
            UNIQUE(final_fen, color)
        );

        CREATE TABLE IF NOT EXISTS position_responses (
            response_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            line_id         INTEGER NOT NULL,
            position_fen    TEXT NOT NULL,
            move_san        TEXT NOT NULL,
            move_uci        TEXT NOT NULL,
            frequency       REAL NOT NULL,
            win_count       INTEGER NOT NULL,
            draw_count      INTEGER NOT NULL,
            loss_count      INTEGER NOT NULL,
            FOREIGN KEY (line_id) REFERENCES opening_lines(line_id)
        );

        CREATE TABLE IF NOT EXISTS collection_log (
            color          TEXT NOT NULL,
            position_uci   TEXT NOT NULL,
            visited        INTEGER DEFAULT 1,
            total_games    INTEGER,
            skipped_reason TEXT,
            PRIMARY KEY (color, position_uci)
        );
        """
    )

    _migrate_opening_lines_if_needed(conn)
    _migrate_collection_log_if_needed(conn)

    conn.commit()
    conn.close()
    print("Database initialized.")