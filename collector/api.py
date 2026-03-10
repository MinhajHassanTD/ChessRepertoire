# collector/api.py  — FULL REPLACEMENT

import time
import requests
from collector.config import (
    LICHESS_API_BASE,
    REQUEST_DELAY,
    RATING_RANGES,
    SPEEDS,
    LICHESS_API_TOKEN,
)

def query_position(moves_uci: list[str]) -> dict | None:
    """
    Query Lichess Opening Explorer for a given position.

    moves_uci: UCI move sequence to this position.
               Empty list = starting position.

    Returns API response dict or None on unrecoverable failure.
    """

    params = {
        "variant": "standard",
        "speeds":  ",".join(SPEEDS),
        "ratings": ",".join(RATING_RANGES),
        "moves":   20,
    }

    if moves_uci:
        params["play"] = ",".join(moves_uci)

    # Only attach header if token is actually set
    headers = {}
    if LICHESS_API_TOKEN:
        headers["Authorization"] = f"Bearer {LICHESS_API_TOKEN}"

    try:
        resp = requests.get(
            LICHESS_API_BASE,
            params=params,
            headers=headers if headers else None,
            timeout=15
        )

        # Rate limited — wait and retry once
        if resp.status_code == 429:
            print("  Rate limited by Lichess. Waiting 60 seconds...")
            time.sleep(60)
            return query_position(moves_uci)

        resp.raise_for_status()
        time.sleep(REQUEST_DELAY)
        return resp.json()

    except requests.exceptions.RequestException as e:
        print(f"  Request failed: {e}")
        time.sleep(5)
        return None