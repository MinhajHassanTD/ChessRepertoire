import time
import requests
from collector.config import (
    LICHESS_API_BASE,
    LICHESS_API_TOKEN,
    REQUEST_DELAY,
    RATING_RANGES,
    SPEEDS,
)

def query_position(moves_uci: list[str]) -> dict | None:
    """Return Lichess Opening Explorer data for a position."""
    params = {
        "variant": "standard",
        "speeds": ",".join(SPEEDS),
        "ratings": ",".join(RATING_RANGES),
        "moves": 20,
    }

    if moves_uci:
        params["play"] = ",".join(moves_uci)

    headers = {}
    if LICHESS_API_TOKEN:
        headers["Authorization"] = f"Bearer {LICHESS_API_TOKEN}"

    try:
        resp = requests.get(
            LICHESS_API_BASE,
            params=params,
            headers=headers if headers else None,
            timeout=15,
        )

        if resp.status_code == 429:
            print("Rate limited by Lichess. Waiting 60 seconds...")
            time.sleep(60)
            return query_position(moves_uci)

        resp.raise_for_status()
        time.sleep(REQUEST_DELAY)
        return resp.json()

    except requests.exceptions.RequestException as exc:
        print(f"  Request failed: {exc}")
        time.sleep(5)
        return None