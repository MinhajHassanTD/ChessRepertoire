LICHESS_API_BASE = "https://explorer.lichess.ovh/lichess"
LICHESS_API_TOKEN = "lip_2zfqV0nEmw36wiimSvyf"

REQUEST_DELAY_SECONDS = 1.2

RATING_RANGES = ["1600", "1800"]
SPEEDS = ["rapid", "classical"]
MIN_GAMES = 25000                 # Minimum games for a position to enter the pool
MIN_MOVE_FREQUENCY = 0.20         # Only follow moves played >= X% of the time
MAX_DEPTH_MOVES = 10              # X half-moves = X/2 full moves per side

REQUIRED_WHITE_FIRST_MOVES = ["e4", "d4", "c4", "Nf3"]
REQUIRED_BLACK_VS_E4 = True
REQUIRED_BLACK_VS_D4 = True
REQUIRED_BLACK_VS_C4 = True
REQUIRED_BLACK_VS_NF3 = True