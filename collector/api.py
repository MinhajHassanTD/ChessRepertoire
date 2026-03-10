import time
import requests
from collector.config import (
    LICHESS_API_BASE,
    REQUEST_DELAY_SECONDS,
    RATING_RANGES,
    SPEEDS,
    LICHESS_API_TOKEN,
)


def query_position(moves_uci: list[str]) -> dict | None:
    """
    Query Lichess Opening Explorer for a position defined by a
    sequence of UCI moves from the starting position.
    """
    params = {
        "variant": "standard",
        "speeds": ",".join(SPEEDS),
        "ratings": ",".join(RATING_RANGES),
        "moves": 15,
    }

    if moves_uci:
        params["play"] = ",".join(moves_uci)

    headers = {}
    if LICHESS_API_TOKEN:
        headers["Authorization"] = f"Bearer {LICHESS_API_TOKEN}"

    try:
        response = requests.get(
            LICHESS_API_BASE,
            params=params,
            headers=headers or None,
            timeout=10,
        )
        response.raise_for_status()
        time.sleep(REQUEST_DELAY_SECONDS)
        return response.json()
    except requests.RequestException as e:
        print(f"API error at position {moves_uci}: {e}")
        time.sleep(5)
        return None