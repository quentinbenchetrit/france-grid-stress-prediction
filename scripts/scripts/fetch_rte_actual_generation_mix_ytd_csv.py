#!/usr/bin/env python3
"""
Fetch RTE "Actual Generation" mix (15-min time scale) from the beginning of the current year
up to the latest available measured interval, and save to CSV.

This version keeps ONLY production_type in {"SOLAR", "WIND"} (post-filtering).
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests
import pandas as pd

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    ZoneInfo = None  # type: ignore


TOKEN_URL = "https://digital.iservices.rte-france.com/token/oauth/"
BASE_API = "https://digital.iservices.rte-france.com/open_api/actual_generation/v1"
MIX_15MIN_URL = f"{BASE_API}/generation_mix_15min_time_scale"

# Keep only these production types in the final CSV (as present in your working output)
KEEP_TYPES = {"SOLAR", "WIND"}


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
        raise SystemExit(
            f"Missing environment variable {name}. "
            f"Set it like: export {name}='...'"
        )
    return val


def get_oauth_token(client_id: str, client_secret: str) -> OAuthToken:
    resp = requests.post(
        TOKEN_URL,
        data={"grant_type": "client_credentials"},
        auth=(client_id, client_secret),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Token request failed ({resp.status_code}): {resp.text}")
    data = resp.json()
    return OAuthToken(
        access_token=data["access_token"],
        expires_in=int(data.get("expires_in", 0)),
        obtained_at=time.time(),
    )


def iso_tz(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError("Datetime must be timezone-aware")
    return dt.isoformat(timespec="seconds")


def fetch_generation_mix_15min(
    session: requests.Session,
    token: OAuthToken,
    start_dt: datetime,
    end_dt: datetime,
    production_type: Optional[List[str]] = None,
    production_subtype: Optional[List[str]] = None,
) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {token.access_token}"}
    params: Dict[str, Any] = {
        "start_date": iso_tz(start_dt),
        "end_date": iso_tz(end_dt),
    }

    # Keep behaviour identical to your working script:
    if production_type:
        params["production_type"] = production_type
    if production_subtype:
        params["production_subtype"] = production_subtype

    resp = session.get(MIX_15MIN_URL, headers=headers, params=params, timeout=90)

    if resp.status_code == 200:
        return resp.json()

    raise RuntimeError(f"API call failed ({resp.status_code}): {resp.text[:500]}")


def extract_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    blocks = payload.get("generation_mix_15min_time_scale", []) or []

    for block in blocks:
        ptype = block.get("production_type")
        psub = block.get("production_subtype")

        for v in block.get("values", []) or []:
            out.append(
                {
                    "production_type": ptype,
                    "production_subtype": psub,
                    "start_date": v.get("start_date"),
                    "end_date": v.get("end_date"),
                    "updated_date": v.get("updated_date"),
                    "value_mw": v.get("value"),
                }
            )
    return out


def dt_year_start(tz) -> datetime:
    now = datetime.now(tz)
    return datetime(now.year, 1, 1, 0, 0, 0, tzinfo=tz)


def adaptive_chunk_fetch(
    client_id: str,
    client_secret: str,
    start_dt: datetime,
    end_dt: datetime,
    initial_days: int = 10,
    min_days: int = 1,
    production_type: Optional[List[str]] = None,
    production_subtype: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    session = requests.Session()
    token = get_oauth_token(client_id, client_secret)

    cursor = start_dt
    window = timedelta(days=initial_days)
    records: List[Dict[str, Any]] = []

    while cursor < end_dt:
        chunk_end = min(cursor + window, end_dt)
        attempt = 0
        cur_window = window

        while True:
            attempt += 1

            if token.is_expired:
                token = get_oauth_token(client_id, client_secret)

            try:
                payload = fetch_generation_mix_15min(
                    session=session,
                    token=token,
                    start_dt=cursor,
                    end_dt=chunk_end,
                    production_type=production_type,
                    production_subtype=production_subtype,
                )
                records.extend(extract_rows(payload))
                cursor = chunk_end
                break

            except RuntimeError as e:
                msg = str(e)

                if "(413)" in msg or "Request Entity Too Large" in msg:
                    new_days = max(min_days, int(cur_window.days / 2) or min_days)
                    new_window = timedelta(days=new_days)

                    if new_window >= cur_window:
                        raise RuntimeError(
                            "Chunk still too large. Try --initial-days 3 or smaller."
                        ) from e

                    cur_window = new_window
                    chunk_end = min(cursor + cur_window, end_dt)

                    if attempt > 8:
                        raise RuntimeError(
                            f"Too many retries while shrinking chunk around {cursor}."
                        ) from e
                    continue

                if "(429)" in msg or "Too Many Requests" in msg:
                    time.sleep(2.0)
                    continue

                raise

    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch RTE Actual Generation (mix 15-min) YTD and save to CSV (SOLAR+WIND only)."
    )
    parser.add_argument(
        "--out",
        default="rte_actual_generation_mix_15min_ytd_solar_wind.csv",
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--timezone",
        default="Europe/Paris",
        help="Timezone used for date parameters and 'start of year'.",
    )
    parser.add_argument(
        "--initial-days",
        type=int,
        default=10,
        help="Initial chunk size in days (auto-shrinks if API returns 413).",
    )

    # We keep CLI params to remain close to your original script (but not needed)
    parser.add_argument("--production-type", nargs="*", default=None)
    parser.add_argument("--production-subtype", nargs="*", default=None)

    args = parser.parse_args()

    if ZoneInfo is None:
        raise SystemExit("Python >= 3.9 is required (zoneinfo).")

    tz = ZoneInfo(args.timezone)

    client_id = get_env_or_fail("RTE_CLIENT_ID")
    client_secret = get_env_or_fail("RTE_CLIENT_SECRET")

    start_dt = dt_year_start(tz)
    end_dt = datetime.now(tz) + timedelta(hours=1)

    records = adaptive_chunk_fetch(
        client_id=client_id,
        client_secret=client_secret,
        start_dt=start_dt,
        end_dt=end_dt,
        initial_days=args.initial_days,
        min_days=1,
        production_type=args.production_type,
        production_subtype=args.production_subtype,
    )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        print("No data returned.", file=sys.stderr)
        return 2

    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    df["updated_date"] = pd.to_datetime(df["updated_date"])

    # ✅ Filter ONLY SOLAR and WIND (as present in your working output)
    df = df[df["production_type"].isin(KEEP_TYPES)].copy()

    # Deduplicate & sort
    df = (
        df.sort_values(["production_type", "production_subtype", "start_date", "end_date"])
        .drop_duplicates(
            subset=["production_type", "production_subtype", "start_date", "end_date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    last_available = df["end_date"].max()

    df.to_csv(args.out, index=False)

    present = sorted(df["production_type"].dropna().unique().tolist())
    print(f"Saved {len(df):,} rows to {args.out}")
    print(f"Production types present: {present}")
    print(f"Data coverage (returned): {df['start_date'].min()} -> {last_available}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
