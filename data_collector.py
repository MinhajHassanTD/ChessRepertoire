import requests
import time
import json
import chess

# --- CONFIGURATION ---
TOKEN = "lip_2zfqV0nEmw36wiimSvyf"
STARTING_FEN = chess.STARTING_FEN
# Aim for 10-20 lines for a test; increase max_depth for final run
MIN_GAMES = 50000 
MAX_DEPTH = 3
BRANCHING = 3  # Number of moves to explore at each step

# Setup persistent session and headers
session = requests.Session()
session.headers.update({
    "Authorization": f"Bearer {TOKEN}",
    "User-Agent": "ChessRepertoireProject/1.0 (Course Project)",
})

# To avoid re-querying the same position (Transpositions)
visited_fens = {}

def query_position(fen):
    """Fetches data from Lichess Explorer with a token and error handling."""
    if fen in visited_fens:
        return visited_fens[fen]

    url = "https://explorer.lichess.ovh/lichess"
    params = {
        "variant": "standard",
        "speeds": "rapid,classical", 
        "ratings": "1600,1800",
        "fen": fen,
        "moves": 10  # Get enough moves for potential branching
    }
    
    try:
        time.sleep(1.1)  # Respect Lichess rate limit
        response = session.get(url, params=params, timeout=10)
        
        if response.status_code == 429:
            print("Rate limited! Sleeping for 60s...")
            time.sleep(60)
            return query_position(fen)
            
        response.raise_for_status()
        data = response.json()
        visited_fens[fen] = data
        return data
    except Exception as e:
        print(f"Error fetching {fen}: {e}")
        return None

def crawl(board, moves_uci, moves_san, depth, pool):
    """Recursively crawls the opening tree and builds the dataset."""
    fen = board.fen()
    data = query_position(fen)
    
    if not data:
        return

    total_games = data['white'] + data['draws'] + data['black']
    
    # Base Case: Reach Max Depth or run out of statistically significant games
    if depth >= MAX_DEPTH or total_games < MIN_GAMES:
        pool.append({
            "moves_uci": list(moves_uci),
            "moves_san": list(moves_san),
            "fen": fen,
            "stats": {
                "white_wins": data['white'],
                "draws": data['draws'],
                "black_wins": data['black'],
                "total_games": total_games,
                "white_win_rate": data['white'] / total_games if total_games > 0 else 0,
                "draw_rate": data['draws'] / total_games if total_games > 0 else 0,
                "black_win_rate": data['black'] / total_games if total_games > 0 else 0
            }
        })
        return
    
    # Recursive Step: Explore the top moves (Branching)
    for move_data in data['moves'][:BRANCHING]:
        move = chess.Move.from_uci(move_data['uci'])
        
        board.push(move)
        crawl(board, 
              moves_uci + [move_data['uci']], 
              moves_san + [move_data['san']], 
              depth + 1, 
              pool)
        board.pop()

# --- Execution ---
opening_pool = []
initial_board = chess.Board(STARTING_FEN)

print("Starting crawl... this may take a few minutes.")
crawl(initial_board, [], [], 0, opening_pool)

# Save the structured data
with open("opening_pool.json", "w") as f:
    json.dump(opening_pool, f, indent=2)

print(f"Found {len(opening_pool)} opening lines. Data saved to opening_pool.json")