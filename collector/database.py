import sqlite3
from pathlib import Path

DB_PATH = Path("data/repertoire.db")

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    Path("data").mkdir(exist_ok=True)
    conn = get_conn()
    conn.executescript("""

        CREATE TABLE IF NOT EXISTS opening_lines (

            line_id           INTEGER PRIMARY KEY AUTOINCREMENT,

            -- Complete alternating move sequence (both colors)
            moves_san         TEXT NOT NULL,   -- "e4 e5 Nf3 Nc6 Bb5"
            moves_uci         TEXT NOT NULL,   -- "e2e4 e7e5 g1f3 b8c6 f1b5"

            -- Board state at end of line
            -- UNIQUE per color so same position can be both W and B line
            final_fen         TEXT NOT NULL,
            color             TEXT NOT NULL,   -- 'white' or 'black'

            depth             INTEGER NOT NULL,
            first_move_tag    TEXT NOT NULL,   -- 'e4', 'd4', 'c4', 'Nf3', 'other'

            -- Raw counts — named unambiguously
            white_wins        INTEGER NOT NULL,
            draws             INTEGER NOT NULL,
            black_wins        INTEGER NOT NULL,
            total_games       INTEGER NOT NULL,

            -- Performance from the perspective of 'color'
            -- white: (white_wins + 0.5*draws) / total
            -- black: (black_wins + 0.5*draws) / total
            performance_score REAL NOT NULL,

            -- Raw rates stored for fitness function flexibility
            win_rate          REAL NOT NULL,
            draw_rate         REAL NOT NULL,
            loss_rate         REAL NOT NULL,

            UNIQUE(final_fen, color)
        );

        -- Tracks every position the traversal has queried.
        -- Key is just the UCI move sequence — color-independent
        -- because a chess position is the same position regardless
        -- of which color's repertoire we are building.
        CREATE TABLE IF NOT EXISTS visited_positions (
            position_key   TEXT PRIMARY KEY,  -- "e2e4 e7e5 g1f3" or "root"
            total_games    INTEGER,
            skip_reason    TEXT               -- NULL = processed normally
        );

    """)
    conn.commit()
    conn.close()
    print("Database ready.")