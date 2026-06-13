"""
Offline tests for the Finnhub provider's pure helpers (no network, no API key).

Run with:
    python tests/test_finnhub_helpers.py
    python -m pytest tests/ -q           # if pytest is installed
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_provider_finnhub import (  # noqa: E402
    entry_label,
    entry_session,
    parse_calendar,
    upcoming_window,
)


def test_entry_session_amc_vs_bmo():
    # 2026-06-19 is a Friday; 2026-06-22 is the following Monday.
    # AMC on Friday -> enter Friday, reaction rolls over the weekend to Monday.
    entry, react = entry_session("2026-06-19", "AMC")
    assert entry == pd.Timestamp("2026-06-19")
    assert react == pd.Timestamp("2026-06-22")
    # BMO on Monday -> must be in by the prior close, which skips back to Friday.
    entry, react = entry_session("2026-06-22", "bmo")
    assert entry == pd.Timestamp("2026-06-19")
    assert react == pd.Timestamp("2026-06-22")
    # DMH / unknown is treated like BMO (enter the prior business day).
    assert entry_session("2026-06-17", "dmh")[0] == pd.Timestamp("2026-06-16")
    assert entry_session("2026-06-17", "")[0] == pd.Timestamp("2026-06-16")


def test_entry_label():
    today = pd.Timestamp("2026-06-13")  # a Saturday
    assert entry_label("2026-06-13", "amc", today=today) == "Today · AMC"
    assert entry_label("2026-06-14", "bmo", today=today) == "Tomorrow · BMO"
    assert entry_label("2026-06-15", "bmo", today=today) == "Mon Jun 15 · BMO"
    assert entry_label("2026-06-10", "amc", today=today) == "Passed · AMC"
    assert entry_label("2026-06-15", "", today=today).endswith("· —")


def test_parse_calendar():
    payload = {"earningsCalendar": [
        {"symbol": "msft", "date": "2026-06-18", "hour": "AMC", "epsEstimate": 3.1},
        {"symbol": "AAPL", "date": "2026-06-15", "hour": "bmo", "epsEstimate": 1.5},
        {"symbol": "", "date": "2026-06-15", "hour": "bmo"},        # dropped: no symbol
        {"symbol": "NVDA", "date": None, "hour": "amc"},            # dropped: no date
    ]}
    rows = parse_calendar(payload)
    assert [r["symbol"] for r in rows] == ["AAPL", "MSFT"]          # sorted by date, upper-cased
    assert rows[0]["hour"] == "bmo" and rows[1]["hour"] == "amc"    # hour lower-cased
    assert rows[0]["eps_estimate"] == 1.5
    assert parse_calendar({}) == [] and parse_calendar({"earningsCalendar": None}) == []


def test_upcoming_window():
    frm, to = upcoming_window(7, today="2026-06-13")
    assert frm == "2026-06-13" and to == "2026-06-20"
    frm, to = upcoming_window(1, today="2026-12-31")
    assert frm == "2026-12-31" and to == "2027-01-01"              # year rollover


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok {fn.__name__}")
    print(f"\nAll {len(fns)} test groups passed.")
