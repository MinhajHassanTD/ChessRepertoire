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
            line_id INTEGER PRIMARY KEY AUTOINCREMENT,
            moves_san TEXT NOT NULL,
            moves_uci TEXT NOT NULL,
            final_fen TEXT NOT NULL,
            color TEXT NOT NULL,
            depth INTEGER NOT NULL,
            first_move_tag TEXT NOT NULL,
            white_wins INTEGER NOT NULL,
            draws INTEGER NOT NULL,
            black_wins INTEGER NOT NULL,
            total_games INTEGER NOT NULL,
            performance_score REAL NOT NULL,
            win_rate REAL NOT NULL,
            draw_rate REAL NOT NULL,
            loss_rate REAL NOT NULL,
            UNIQUE(final_fen, color)
        );

        CREATE TABLE IF NOT EXISTS visited_positions (
            position_key TEXT PRIMARY KEY,
            total_games INTEGER,
            skip_reason TEXT
        );
    """)
    conn.commit()
    conn.close()
    print("Database ready.")