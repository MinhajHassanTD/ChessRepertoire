LICHESS_API_BASE = "https://explorer.lichess.ovh/lichess"
LICHESS_API_TOKEN = "lip_2zfqV0nEmw36wiimSvyf"

REQUEST_DELAY = 1.2

RATING_RANGES = ["1800", "2000"]
SPEEDS = ["rapid", "classical"]
MIN_GAMES = 20000                 # Minimum games for a position to enter the pool
MIN_MOVE_FREQ = 0.15              # Only follow moves played >= X% of the time

MIN_DEPTH = 10                     # Lowest depth to start collecting positions from <-> 10 = 5 moves per side
MAX_DEPTH = 14                    # Maximum half-moves to traverse <-> 14 = 7 moves per side

MAX_MOVES_PER_POSITION = 3
FIRST_MOVES = {
    "e2e4": "e4",
    "d2d4": "d4",
    "c2c4": "c4",
    "g1f3": "Nf3"
}