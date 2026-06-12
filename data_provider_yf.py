"""
Yahoo Finance data provider (free — no account or API key required).

Pulls everything VolatilityEngine.evaluate_ticker() needs via `yfinance`:

  spot, iv_near, iv_45, iv_30, rv_30, avg_30day_volume, atm_call, atm_put,
  historical_moves (from REAL past earnings dates), drift, historical_iv_series

Data notes / honest limitations
-------------------------------
* Quotes are delayed ~15 min (fine for setup detection; not for the §6.5
  velocity exit — that needs a real-time feed such as IBKR with OPRA).
* Yahoo does not expose an implied-volatility *history*, so the 252-day series
  backing the IV percentile is a documented proxy: the trailing 30-day realized
  vol series. Current IV vs that distribution behaves like an IV rank as long
  as IV and RV track each other; swap in IBKR's OPTION_IMPLIED_VOLATILITY
  history for the true percentile.
* Earnings dates come from Yahoo's calendar — real announcement timestamps, so
  `historical_moves` are true earnings reactions (close before the print ->
  first close after), an upgrade over the "largest daily move" proxy.

Quick standalone test (requires internet access to Yahoo):
    python data_provider_yf.py AAPL
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ----------------------------------------------------------------------
# Pure helpers (no network — unit-tested in tests/test_yf_helpers.py)
# ----------------------------------------------------------------------
def realized_vol_series(closes: pd.Series, window: int = 30) -> pd.Series:
    """Annualized rolling close-to-close realized volatility."""
    rets = np.log(closes / closes.shift(1))
    return rets.rolling(window).std() * math.sqrt(TRADING_DAYS)


def _mid_or_last(row: pd.Series) -> Optional[float]:
    bid = row.get("bid") or 0.0
    ask = row.get("ask") or 0.0
    if bid > 0 and ask > 0:
        return float((bid + ask) / 2.0)
    last = row.get("lastPrice") or 0.0
    return float(last) if last > 0 else None


def atm_from_chain(
    calls: pd.DataFrame, puts: pd.DataFrame, spot: float
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    (atm_iv, atm_strike, call_price, put_price) at the strike nearest spot.
    IV is the mean of the call/put impliedVolatility when both quote, else
    whichever side exists. Prices are bid/ask mid, falling back to last.
    """
    if calls.empty and puts.empty:
        return None, None, None, None
    strikes = pd.concat([calls["strike"], puts["strike"]]).drop_duplicates()
    atm_strike = float(strikes.iloc[(strikes - spot).abs().argsort().iloc[0]])

    def _row(df: pd.DataFrame) -> Optional[pd.Series]:
        hit = df[df["strike"] == atm_strike]
        return hit.iloc[0] if len(hit) else None

    c_row, p_row = _row(calls), _row(puts)
    ivs = [float(r["impliedVolatility"]) for r in (c_row, p_row)
           if r is not None and r.get("impliedVolatility") and r["impliedVolatility"] > 0]
    atm_iv = float(np.mean(ivs)) if ivs else None
    call_px = _mid_or_last(c_row) if c_row is not None else None
    put_px = _mid_or_last(p_row) if p_row is not None else None
    return atm_iv, atm_strike, call_px, put_px


def interp_iv(days_iv: List[Tuple[int, float]], target_days: int) -> Optional[float]:
    """Linear interpolation of ATM IV across expiries to `target_days`."""
    pts = sorted((d, iv) for d, iv in days_iv if iv and iv > 0)
    if not pts:
        return None
    if len(pts) == 1 or target_days <= pts[0][0]:
        return pts[0][1]
    if target_days >= pts[-1][0]:
        return pts[-1][1]
    for (d0, v0), (d1, v1) in zip(pts, pts[1:]):
        if d0 <= target_days <= d1:
            w = (target_days - d0) / (d1 - d0) if d1 > d0 else 0.0
            return v0 + w * (v1 - v0)
    return pts[-1][1]


def earnings_moves(
    closes: pd.Series, earnings_ts: List[pd.Timestamp], n: int = 8
) -> List[float]:
    """
    Signed close-to-close reaction for each earnings print: the close before
    the reaction session -> the reaction session's close. Daily bars are
    stamped at midnight, so the announcement is first resolved to its reaction
    session: after-close prints (>= 16:00) react the NEXT session; before-open
    or unstamped (midnight) prints react the SAME session.
    """
    if closes.index.tz is not None:
        closes = closes.tz_localize(None)
    idx = closes.index
    moves: List[float] = []
    for ts in sorted(earnings_ts, reverse=True):
        ts = pd.Timestamp(ts)
        if ts.tz is not None:
            ts = ts.tz_localize(None)
        reaction_floor = ts.normalize() + pd.Timedelta(days=1) if ts.hour >= 16 else ts.normalize()
        pos = idx.searchsorted(reaction_floor)
        if pos <= 0 or pos >= len(idx):
            continue
        prev_close, react_close = float(closes.iloc[pos - 1]), float(closes.iloc[pos])
        if prev_close > 0:
            moves.append(react_close / prev_close - 1.0)
        if len(moves) >= n:
            break
    return moves


# ----------------------------------------------------------------------
# Network fetch
# ----------------------------------------------------------------------
def fetch_yf_metrics(symbol: str, n_earnings_moves: int = 8) -> Dict[str, Any]:
    """One-shot fetch: returns the same engine-ready dict shape as the IBKR
    provider (so the app can seed its inputs from either source)."""
    import yfinance as yf  # lazy: keep module importable without yfinance

    symbol = symbol.upper()
    t = yf.Ticker(symbol)

    # --- Spot, volume, realized vol from 1y daily bars ---
    hist = t.history(period="1y", auto_adjust=False)
    if hist is None or hist.empty:
        raise RuntimeError(f"No price history returned for {symbol} — check the symbol.")
    closes = hist["Close"].astype(float)
    if closes.index.tz is not None:
        closes = closes.tz_localize(None)
    spot = float(closes.iloc[-1])
    avg_30day_volume = int(hist["Volume"].tail(30).mean())
    rv_series = realized_vol_series(closes).dropna()
    rv_30 = float(rv_series.iloc[-1]) if len(rv_series) else 0.0

    # --- Option chains: near & ~45d expiries, ATM IVs + straddle ---
    expiries = list(t.options or [])
    if not expiries:
        raise RuntimeError(f"No listed options found for {symbol}.")
    today = datetime.now(timezone.utc).date()
    exp_days = [(e, (datetime.strptime(e, "%Y-%m-%d").date() - today).days)
                for e in expiries]
    exp_days = [(e, d) for e, d in exp_days if d >= 0]
    if not exp_days:
        raise RuntimeError(f"No future option expirations for {symbol}.")
    near_exp, near_days = exp_days[0]
    exp_45, days_45 = min(exp_days, key=lambda x: abs(x[1] - 45))

    chain_near = t.option_chain(near_exp)
    iv_near, atm_strike, atm_call, atm_put = atm_from_chain(
        chain_near.calls, chain_near.puts, spot
    )
    chain_45 = t.option_chain(exp_45)
    iv_45, _, _, _ = atm_from_chain(chain_45.calls, chain_45.puts, spot)

    # iv_30: interpolate ATM IV across up to 4 expiries bracketing 30d
    days_iv: List[Tuple[int, float]] = [(near_days, iv_near or 0.0), (days_45, iv_45 or 0.0)]
    for e, d in exp_days[1:5]:
        if e in (near_exp, exp_45):
            continue
        ch = t.option_chain(e)
        iv_e, _, _, _ = atm_from_chain(ch.calls, ch.puts, spot)
        if iv_e:
            days_iv.append((d, iv_e))
    iv_30 = interp_iv(days_iv, 30) or iv_45 or iv_near or 0.0

    # --- Real earnings dates -> reactions, drift, next print ---
    hist_moves: List[float] = []
    next_earnings: Optional[str] = None
    try:
        ed = t.get_earnings_dates(limit=16)
        if ed is not None and len(ed):
            now = pd.Timestamp.now(tz=ed.index.tz) if ed.index.tz else pd.Timestamp.now()
            past = [ts for ts in ed.index if ts < now]
            future = sorted(ts for ts in ed.index if ts >= now)
            if future:
                next_earnings = str(future[0].date())
            hist_moves = earnings_moves(closes, past, n=n_earnings_moves)
    except Exception:
        pass  # earnings calendar is best-effort
    if not hist_moves:  # fall back to the transparent largest-moves proxy
        daily_ret = closes.pct_change().dropna()
        largest = daily_ret.reindex(daily_ret.abs().sort_values(ascending=False).index)
        hist_moves = [float(x) for x in largest.head(n_earnings_moves)]
    drift = float(np.mean(hist_moves)) if hist_moves else 0.0

    return {
        "ticker": symbol,
        "spot": spot,
        "iv_near": iv_near or 0.0,
        "iv_45": iv_45 or 0.0,
        "iv_30": float(iv_30),
        "rv_30": rv_30,
        "volume": avg_30day_volume,
        "atm_call": atm_call or 0.0,
        "atm_put": atm_put or 0.0,
        "drift": drift,
        # IV-history proxy (see module docstring): trailing 30d RV distribution
        "historical_iv_series": [float(x) for x in rv_series.tail(TRADING_DAYS).tolist()],
        "hist_moves": hist_moves,
        "_meta": {
            "source": "yfinance (delayed)",
            "atm_strike": atm_strike,
            "near_expiry": near_exp,
            "exp_45": exp_45,
            "days_to_near": near_days,
            "days_to_45leg": days_45,
            "next_earnings": next_earnings,
            "iv_history_proxy": "30d realized-vol series (Yahoo has no IV history)",
        },
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import argparse
    import json

    p = argparse.ArgumentParser(description="Fetch engine metrics from Yahoo Finance.")
    p.add_argument("symbol")
    args = p.parse_args()
    data = fetch_yf_metrics(args.symbol)
    preview = dict(data)
    preview["historical_iv_series"] = f"<{len(data['historical_iv_series'])} values>"
    print(json.dumps(preview, indent=2, default=str))
