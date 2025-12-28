# ID Client f99fd5e3-bd18-44e1-b3cc-3080281bb93c ID Secret a6f78c9b-9133-4fd6-ad71-5b54b18e99f1
#To run it :  export RTE_CLIENT_ID="..."
#export RTE_CLIENT_SECRET="..."
#before


#!/usr/bin/env python3
"""
Fetch RTE measured (REALISED) electricity consumption from the beginning of the current year
up to the latest available measured interval.

Output:
- CSV only
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
import pandas as pd

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    ZoneInfo = None  # type: ignore


TOKEN_URL = "https://digital.iservices.rte-france.com/token/oauth/"
BASE_API = "https://digital.iservices.rte-france.com/open_api/consumption/v1"
SHORT_TERM_URL = f"{BASE_API}/short_term"


@dataclass
class OAuthToken:
    access_token: str
    expires_in: int
    obtained_at: float

    @property
    def is_expired(self) -> bool:
        return time.time() >= self.obtained_at + self.expires_in - 30


def get_env_or_fail(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise SystemExit(f"Missing environment variable {name}")
    return val


def get_oauth_token(client_id: str, client_secret: str) -> OAuthToken:
    resp = requests.post(
        TOKEN_URL,
        data={"grant_type": "client_credentials"},
        auth=(client_id, client_secret),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return OAuthToken(
        access_token=data["access_token"],
        expires_in=int(data["expires_in"]),
        obtained_at=time.time(),
    )


def iso_paris(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def fetch_short_term_realised(
    token: OAuthToken,
    start_dt: datetime,
    end_dt: datetime,
    session: requests.Session,
) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {token.access_token}"}
    params = {
        "start_date": iso_paris(start_dt),
        "end_date": iso_paris(end_dt),
        "type": "REALISED",
    }
    resp = session.get(SHORT_TERM_URL, headers=headers, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def extract_values(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for block in payload.get("short_term", []):
        if block.get("type") != "REALISED":
            continue
        for v in block.get("values", []):
            rows.append(
                {
                    "start_date": v["start_date"],
                    "end_date": v["end_date"],
                    "updated_date": v["updated_date"],
                    "consumption_mw": v["value"],
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="rte_consumption_realised_ytd.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--timezone",
        default="Europe/Paris",
        help="Timezone for dates",
    )
    args = parser.parse_args()

    if ZoneInfo is None:
        raise SystemExit("Python >= 3.9 required")

    tz = ZoneInfo(args.timezone)

    client_id = get_env_or_fail("RTE_CLIENT_ID")
    client_secret = get_env_or_fail("RTE_CLIENT_SECRET")

    token = get_oauth_token(client_id, client_secret)

    start_dt = datetime(datetime.now(tz).year, 1, 1, tzinfo=tz)
    end_dt = datetime.now(tz)

    session = requests.Session()
    records: List[Dict[str, Any]] = []

    cursor = start_dt
    step = timedelta(days=30)

    while cursor < end_dt:
        chunk_end = min(cursor + step, end_dt)

        if token.is_expired:
            token = get_oauth_token(client_id, client_secret)

        payload = fetch_short_term_realised(token, cursor, chunk_end, session)
        records.extend(extract_values(payload))

        cursor = chunk_end

    df = pd.DataFrame(records)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    df["updated_date"] = pd.to_datetime(df["updated_date"])

    df = (
        df.sort_values("start_date")
        .drop_duplicates(subset=["start_date", "end_date"], keep="last")
        .reset_index(drop=True)
    )

    df.to_csv(args.out, index=False)

    print(f"Saved {len(df)} rows to {args.out}")
    print(f"Coverage: {df.start_date.min()} → {df.end_date.max()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
