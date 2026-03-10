# main.py

import chess
from collector.database import initialize_database, get_connection
from collector.traversal import traverse

def collect_white_lines():
    """Collect lines where we are playing White."""
    print("\n=== Collecting White lines ===")
    conn = get_connection()
    board = chess.Board()
    
    # Start traversal from the root — White moves first
    traverse([], [], board, "white", 0, conn)
    
    conn.close()
    print("White line collection complete.")

def collect_black_lines():
    """
    Collect lines where we are playing Black.
    We still start from the root but we are building Black's responses.
    The color parameter tells the save function whose perspective to score from.
    """
    print("\n=== Collecting Black lines ===")
    conn = get_connection()
    board = chess.Board()
    
    traverse([], [], board, "black", 0, conn)
    
    conn.close()
    print("Black line collection complete.")

def print_collection_summary():
    conn = get_connection()
    
    total = conn.execute(
        "SELECT COUNT(*) FROM opening_lines"
    ).fetchone()[0]
    
    white = conn.execute(
        "SELECT COUNT(*) FROM opening_lines WHERE color = 'white'"
    ).fetchone()[0]
    
    black = conn.execute(
        "SELECT COUNT(*) FROM opening_lines WHERE color = 'black'"
    ).fetchone()[0]
    
    visited = conn.execute(
        "SELECT COUNT(*) FROM collection_log"
    ).fetchone()[0]
    
    pruned = conn.execute(
        "SELECT COUNT(*) FROM collection_log WHERE skipped_reason IS NOT NULL"
    ).fetchone()[0]
    
    print(f"\n{'='*40}")
    print(f"COLLECTION SUMMARY")
    print(f"{'='*40}")
    print(f"Total lines collected : {total}")
    print(f"  White lines         : {white}")
    print(f"  Black lines         : {black}")
    print(f"Positions visited     : {visited}")
    print(f"Positions pruned      : {pruned}")
    print(f"{'='*40}")
    
    conn.close()

if __name__ == "__main__":
    initialize_database()
    collect_white_lines()
    collect_black_lines()
    print_collection_summary()