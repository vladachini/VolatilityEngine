"""
Finnhub earnings-calendar provider (free API key).

Why Finnhub: yfinance gives quotes / chains / IV but no forward earnings *calendar*
across the whole market. Finnhub's /calendar/earnings endpoint lists every
US-listed company reporting in a date range, each tagged BMO / AMC / DMH — exactly
what we need to answer "what reports in the next N days, and when do I enter?".

Free tier: register at https://finnhub.io/register -> 60 API calls/minute. One call
covers a whole date window, so a daily market scan is well within the limit.

Design: the pure helpers (entry_session / entry_label / parse_calendar /
upcoming_window) are network-free and unit-tested in tests/test_finnhub_helpers.py.
Only fetch_earnings_calendar touches the network, and it uses the stdlib (urllib),
so this module adds no new dependency.

Standalone smoke test (needs a key in FINNHUB_API_KEY or --key):
    python data_provider_finnhub.py --days 7
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

FINNHUB_EARNINGS_URL = "https://finnhub.io/api/v1/calendar/earnings"


# ----------------------------------------------------------------------
# Pure helpers (no network — unit-tested in tests/test_finnhub_helpers.py)
# ----------------------------------------------------------------------
def _prev_business_day(ts: pd.Timestamp) -> pd.Timestamp:
    """Previous Mon–Fri day (holiday-agnostic, documented stand-in)."""
    d = ts - pd.Timedelta(days=1)
    while d.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        d -= pd.Timedelta(days=1)
    return d


def _next_business_day(ts: pd.Timestamp) -> pd.Timestamp:
    d = ts + pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d


def entry_session(earnings_day: Any, hour: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Map an earnings announcement to (entry_session, reaction_session).

    The calendar trade must be HELD over the announcement to harvest the IV crush,
    so you enter on the close just before the print:

      * AMC  (after close on day D): the crush prints overnight, so you enter by the
        close of D.        -> entry = D, reaction = next business day.
      * BMO  (before open on day D): the crush prints at D's open, so you must be in
        by the prior close. -> entry = previous business day, reaction = D.
      * DMH / unknown: treated like BMO (enter the prior close) to stay conservative.
    """
    d = pd.Timestamp(earnings_day).normalize()
    h = (hour or "").strip().lower()
    if h == "amc":
        return d, _next_business_day(d)
    return _prev_business_day(d), d


def entry_label(entry: Any, hour: str, today: Optional[Any] = None) -> str:
    """
    Human entry-timing label, e.g. 'Today · AMC', 'Tomorrow · BMO', 'Mon Jun 15 · BMO'.
    `today` is injectable for testing; defaults to the local date.
    """
    today_ts = (pd.Timestamp(today) if today is not None else pd.Timestamp.today()).normalize()
    entry_ts = pd.Timestamp(entry).normalize()
    tag = (hour or "").strip().upper() or "—"
    delta = (entry_ts - today_ts).days
    if delta < 0:
        when = "Passed"
    elif delta == 0:
        when = "Today"
    elif delta == 1:
        when = "Tomorrow"
    else:
        when = entry_ts.strftime("%a %b %d")
    return f"{when} · {tag}"


def parse_calendar(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize Finnhub's /calendar/earnings JSON into clean rows, soonest first."""
    rows: List[Dict[str, Any]] = []
    for e in (payload or {}).get("earningsCalendar", []) or []:
        sym = (e.get("symbol") or "").strip().upper()
        day = e.get("date")
        if not sym or not day:
            continue
        rows.append({
            "symbol": sym,
            "date": str(day),
            "hour": (e.get("hour") or "").strip().lower(),
            "eps_estimate": e.get("epsEstimate"),
            "eps_actual": e.get("epsActual"),
            "revenue_estimate": e.get("revenueEstimate"),
        })
    rows.sort(key=lambda r: (r["date"], r["symbol"]))
    return rows


def upcoming_window(days_ahead: int = 7, today: Optional[Any] = None) -> Tuple[str, str]:
    """[today, today + days_ahead] as YYYY-MM-DD strings for the calendar query."""
    t = (pd.Timestamp(today) if today is not None else pd.Timestamp.today()).normalize()
    return t.date().isoformat(), (t + pd.Timedelta(days=days_ahead)).date().isoformat()


# ----------------------------------------------------------------------
# Network fetch
# ----------------------------------------------------------------------
def resolve_api_key(explicit: Optional[str] = None) -> str:
    """Key precedence: explicit arg -> FINNHUB_API_KEY env var."""
    if explicit:
        return explicit.strip()
    return (os.environ.get("FINNHUB_API_KEY") or "").strip()


def fetch_earnings_calendar(
    from_date: Any, to_date: Any, api_key: Optional[str], timeout: float = 30.0,
) -> List[Dict[str, Any]]:
    """Fetch and normalize the earnings calendar for the inclusive [from, to] window."""
    key = resolve_api_key(api_key)
    if not key:
        raise RuntimeError(
            "No Finnhub API key — set FINNHUB_API_KEY or pass api_key "
            "(free key at https://finnhub.io/register)."
        )
    params = {"from": str(from_date), "to": str(to_date), "token": key}
    url = FINNHUB_EARNINGS_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "VolatilityEngine/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (https only)
        payload = json.loads(resp.read().decode("utf-8"))
    return parse_calendar(payload)


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import argparse

    p = argparse.ArgumentParser(description="Fetch the upcoming earnings calendar from Finnhub.")
    p.add_argument("--days", type=int, default=7, help="days ahead to scan")
    p.add_argument("--key", default=None, help="Finnhub API key (else FINNHUB_API_KEY env)")
    args = p.parse_args()

    frm, to = upcoming_window(args.days)
    cal = fetch_earnings_calendar(frm, to, args.key)
    seen = set()
    for row in cal:
        if row["symbol"] in seen:
            continue
        seen.add(row["symbol"])
        ent, _react = entry_session(row["date"], row["hour"])
        print(f"{row['symbol']:<8} reports {row['date']} {row['hour'] or '—':<4} "
              f"-> enter {entry_label(ent, row['hour'])}")
    print(f"\n{len(cal)} calendar rows, {len(seen)} unique symbols, {frm}..{to}")
