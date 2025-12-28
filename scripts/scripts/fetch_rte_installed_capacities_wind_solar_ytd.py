#!/usr/bin/env python3
"""
Fetch RTE "Generation Installed Capacities" (v1.2) - ALL production types,
and save to CSV.

"""

from __future__ import annotations

import os
import sys
import time
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
import pandas as pd


TOKEN_URL = "https://digital.iservices.rte-france.com/token/oauth/"
BASE_API = "https://digital.iservices.rte-france.com/open_api/generation_installed_capacities/v1"
CAP_PER_TYPE_URL = f"{BASE_API}/capacities_per_production_type"


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


def parse_api_error(resp: requests.Response) -> Tuple[Optional[str], Optional[str]]:
    try:
        j = resp.json()
        return j.get("error"), j.get("error_description")
    except Exception:
        return None, None


def fetch_capacities_all(session: requests.Session, token: OAuthToken) -> Dict[str, Any]:
    """
    Call WITHOUT start_date/end_date and WITHOUT production_type filter,
    to get the default 'current year' dataset returned by the service.
    """
    headers = {"Authorization": f"Bearer {token.access_token}"}
    resp = session.get(CAP_PER_TYPE_URL, headers=headers, timeout=90)

    if resp.status_code == 200:
        return resp.json()

    err, desc = parse_api_error(resp)
    raise RuntimeError(
        f"API call failed ({resp.status_code}) err={err!r} desc={desc!r} raw={resp.text[:800]}"
    )


def extract_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Swagger shape:
    {
      "capacities_per_production_type": [
        {
          "start_date": "...",
          "end_date": "...",
          "values": [
            {"start_date": "...", "end_date": "...", "type": "...", "value": 1234, "updated_date": "..."},
            ...
          ]
        }
      ]
    }
    """
    rows: List[Dict[str, Any]] = []
    for block in payload.get("capacities_per_production_type", []) or []:
        for v in block.get("values", []) or []:
            rows.append(
                {
                    "production_type": v.get("type"),
                    "start_date": v.get("start_date"),
                    "end_date": v.get("end_date"),
                    "updated_date": v.get("updated_date"),
                    "installed_capacity_mw": v.get("value"),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch RTE Generation Installed Capacities (ALL types) and save to CSV."
    )
    parser.add_argument(
        "--out",
        default="rte_installed_capacities_all_current_year.csv",
        help="Output CSV file path.",
    )
    args = parser.parse_args()

    client_id = get_env_or_fail("RTE_CLIENT_ID")
    client_secret = get_env_or_fail("RTE_CLIENT_SECRET")

    session = requests.Session()
    token = get_oauth_token(client_id, client_secret)

    payload = fetch_capacities_all(session, token)
    records = extract_rows(payload)

    df = pd.DataFrame.from_records(records)
    if df.empty:
        print("No installed capacity data returned.", file=sys.stderr)
        return 2

    # Parse datetimes (safe even if timezone format varies)
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    df["updated_date"] = pd.to_datetime(df["updated_date"], errors="coerce")

    df = (
        df.sort_values(["production_type", "start_date", "end_date"])
        .drop_duplicates(subset=["production_type", "start_date", "end_date"], keep="last")
        .reset_index(drop=True)
    )

    df.to_csv(args.out, index=False)

    present = sorted(df["production_type"].dropna().unique().tolist())
    print(f"Saved {len(df):,} rows to {args.out}")
    print(f"Production types present: {present}")
    print(f"Coverage: {df['start_date'].min()} -> {df['end_date'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
