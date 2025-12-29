#!/usr/bin/env python3
"""
Fetch RTE measured (REALISED) electricity consumption from the beginning of the current year
up to the latest available measured interval.

UTC POLICY:
- API calls are made in Europe/Paris (as required by RTE)
- All timestamps are converted to UTC before being written to CSV
- Output CSV contains UTC-only, timezone-aware timestamps

Output:
- CSV only (UTC)
"""

from __future__ import annotations

import os
import time
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List

import requests
import pandas as pd

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    ZoneInfo = None  # type: ignore


TOKEN_URL = "https://digital.iservices.rte-france.com/token/oauth/"
BASE_API = "https://digital.iservices.rte-france.com/open_api/consumption/v1"
SHORT_TERM_URL = f"{BASE_API}/short_term"

LOCAL_TZ = "Europe/Paris"


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


def iso_local(dt: datetime) -> str:
    """ISO format with timezone info (used for API calls)."""
    return dt.isoformat(timespec="seconds")


def fetch_short_term_realised(
    token: OAuthToken,
    start_dt: datetime,
    end_dt: datetime,
    session: requests.Session,
) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {token.access_token}"}
    params = {
        "start_date": iso_local(start_dt),
        "end_date": iso_local(end_dt),
        "type": "REALISED",
    }
    resp = session.get(SHORT_TERM_URL, headers=headers, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def extract_values(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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


def to_utc(series: pd.Series) -> pd.Series:
    """
    Robust conversion to UTC.

    RTE timestamps may contain mixed offsets (+01:00 / +02:00 depending on DST).
    Parsing with utc=True normalizes everything to a single datetime64[ns, UTC] dtype,
    avoids mixed-tz object dtype, and prevents .dt accessor errors.
    """
    return pd.to_datetime(series, errors="coerce", utc=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="/home/onyxia/work/france-grid-stress-prediction/data/interim/consommation/rte_consumption_realised_ytd_utc.csv",
        help="Output CSV file path (UTC timestamps)",
    )
    args = parser.parse_args()

    if ZoneInfo is None:
        raise SystemExit("Python >= 3.9 required")

    tz_local = ZoneInfo(LOCAL_TZ)

    client_id = get_env_or_fail("RTE_CLIENT_ID")
    client_secret = get_env_or_fail("RTE_CLIENT_SECRET")

    token = get_oauth_token(client_id, client_secret)

    # API time window (local time as required by RTE)
    start_dt = datetime(datetime.now(tz_local).year, 1, 1, tzinfo=tz_local)
    end_dt = datetime.now(tz_local)

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
    if df.empty:
        raise SystemExit("No data returned by RTE API.")

    # ---- UTC NORMALIZATION (CORE FIX) ----
    df["start_date"] = to_utc(df["start_date"])
    df["end_date"] = to_utc(df["end_date"])
    df["updated_date"] = to_utc(df["updated_date"])

    # Clean / deduplicate
    df = (
        df.sort_values("start_date")
        .drop_duplicates(subset=["start_date", "end_date"], keep="last")
        .reset_index(drop=True)
    )

    df.to_csv(args.out, index=False)

    print(f"Saved {len(df):,} rows to {args.out}")
    print(f"Coverage UTC: {df.start_date.min()} → {df.end_date.max()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
