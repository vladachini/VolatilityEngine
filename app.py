"""
Volatility Engine — Earnings IV-Crush Trading App (Streamlit frontend, PRD v3.1).

A dark, mobile-responsive dashboard around engine.VolatilityEngine:
signal hero + gate pills, metric cards, term-structure & IV-history charts,
trade structuring with payoff diagram, Kelly/contract position plan, §6.5 exit
protocols, and live backtest + Monte Carlo validation.

Run:  streamlit run app.py
"""
import html
import os
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import (
    DRIFT_THRESHOLD,
    HISTORICAL_AVG_LOSS,
    HISTORICAL_AVG_WIN,
    HISTORICAL_WIN_RATE,
    KELLY_MULTIPLIER,
    MAX_PORTFOLIO_FRACTION,
    T_FAR_DAYS,
    T_NEAR_DAYS,
    VolatilityEngine,
    bs_call,
)
from backtest import BacktestConfig, run_backtest
from data_provider_finnhub import entry_label, entry_session, upcoming_window
from demo import DEFAULT_HIST_MOVES, DEMO, demo_iv_history

st.set_page_config(
    page_title="Volatility Engine — Earnings IV Crush",
    page_icon=":material/candlestick_chart:",
    layout="wide",
)

# ==================================================================
# Design system
# ==================================================================
ACCENT, CYAN = "#818cf8", "#22d3ee"
GREEN, AMBER, RED, MUTED = "#34d399", "#fbbf24", "#f87171", "#94a3b8"
GRID = "rgba(148,163,184,.10)"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');
:root{
  --panel:rgba(148,163,184,.055); --panel-border:rgba(148,163,184,.14);
  --panel-border-hi:rgba(148,163,184,.30);
  --muted:#94a3b8; --green:#34d399; --amber:#fbbf24; --red:#f87171;
  --indigo:#818cf8; --cyan:#22d3ee; --radius:14px;
  --ease:cubic-bezier(.22,.9,.3,1);
}
html, body, [data-testid="stAppViewContainer"], button, input, textarea{
  font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;
  -webkit-font-smoothing:antialiased;
}
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1100px 540px at 12% -8%, rgba(99,102,241,.13), transparent 60%),
    radial-gradient(900px 460px at 108% 6%, rgba(34,211,238,.07), transparent 55%),
    #0a1120;
}
[data-testid="stHeader"]{ background:transparent; }
#MainMenu, footer{ visibility:hidden; }
.block-container{ padding-top:2.0rem; padding-bottom:4.5rem; max-width:1240px; }
@media (max-width:740px){
  .block-container{ padding:1.1rem .85rem 5rem; }
}
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#0e1830 0%,#0a1120 100%);
  border-right:1px solid var(--panel-border);
}
[data-testid="stSidebar"] .block-container{ padding-top:1.4rem; }

/* ---- motion ---- */
@keyframes veFadeUp{ from{ opacity:0; transform:translateY(7px); } to{ opacity:1; transform:none; } }
@keyframes vePing{ 0%{ transform:scale(1); opacity:.55; } 75%,100%{ transform:scale(2.4); opacity:0; } }
.ve-signal,.ve-card,.ve-panel,.ve-note,.ve-warn{ animation:veFadeUp .38s var(--ease) both; }
.ve-grid .ve-card:nth-child(2){ animation-delay:.03s; }
.ve-grid .ve-card:nth-child(3){ animation-delay:.06s; }
.ve-grid .ve-card:nth-child(4){ animation-delay:.09s; }
.ve-grid .ve-card:nth-child(5){ animation-delay:.12s; }
.ve-grid .ve-card:nth-child(6){ animation-delay:.15s; }
.ve-grid .ve-card:nth-child(7){ animation-delay:.18s; }
.ve-grid .ve-card:nth-child(8){ animation-delay:.21s; }
@media (prefers-reduced-motion:reduce){
  *,*::before,*::after{ animation:none !important; transition:none !important; }
}

/* ---- sidebar brand ---- */
.ve-brand{ display:flex; align-items:center; gap:.65rem; }
.ve-brand .logo{ width:34px; height:34px; border-radius:9px; flex:none;
  display:flex; align-items:center; justify-content:center;
  background:linear-gradient(135deg,var(--indigo),var(--cyan));
  color:#0a1120; font-weight:800; font-size:.78rem; letter-spacing:.02em; }
.ve-brand .name{ font-size:1.0rem; font-weight:700; letter-spacing:-.02em; line-height:1.15; }
.ve-brand .sub{ color:var(--muted); font-weight:500; font-size:.74rem; margin-top:.1rem; }

/* ---- hero ---- */
.ve-hero .kicker{ display:flex; align-items:center; gap:.55rem; color:var(--muted);
  font-size:.72rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase;
  margin-bottom:.45rem; }
.ve-hero .kicker::before{ content:""; width:20px; height:2px; border-radius:2px;
  background:linear-gradient(90deg,var(--indigo),var(--cyan)); }
.ve-hero h1{
  font-size:clamp(1.45rem,4.5vw,2.2rem); font-weight:800; letter-spacing:-.03em; margin:0;
  background:linear-gradient(90deg,#f1f5f9,#a5b4fc 55%,#67e8f9);
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
}
.ve-hero p{ color:var(--muted); margin:.35rem 0 0; font-size:clamp(.84rem,2.6vw,.95rem); }

/* ---- section headings ---- */
.ve-sec{ display:flex; align-items:baseline; gap:.55rem; margin:1.5rem 0 .65rem;
  font-weight:700; font-size:1.02rem; letter-spacing:-.01em; color:#f1f5f9; }
.ve-sec .dot{ align-self:center; width:8px; height:8px; border-radius:2.5px; flex:none;
  background:linear-gradient(135deg,var(--indigo),var(--cyan)); }
.ve-sec .sub{ color:var(--muted); font-weight:500; font-size:.76rem; white-space:nowrap; }
.ve-sec::after{ content:""; flex:1; align-self:center; height:1px; margin-left:.35rem;
  background:linear-gradient(90deg,var(--panel-border),transparent); }

/* ---- signal hero card ---- */
.ve-signal{ border-radius:var(--radius); border:1px solid; padding:1.05rem 1.2rem;
  margin:.35rem 0 .9rem; display:flex; flex-direction:column; gap:.55rem; }
.ve-signal .sig-row{ display:flex; align-items:center; gap:.65rem; flex-wrap:wrap; }
.ve-signal .sig-dot{ width:10px; height:10px; border-radius:50%; position:relative; flex:none; }
.ve-signal .sig-dot::after{ content:""; position:absolute; inset:0; border-radius:50%;
  background:inherit; animation:vePing 2.4s var(--ease) infinite; }
.ve-signal .sig-word{ font-size:clamp(1.2rem,5vw,1.55rem); font-weight:800; letter-spacing:.02em; }
.ve-signal .sig-ticker{ font-family:'JetBrains Mono',monospace; font-weight:700; font-size:.9rem;
  padding:.18rem .6rem; border-radius:8px; background:rgba(255,255,255,.07);
  border:1px solid rgba(255,255,255,.14); }
.ve-signal .sig-badge{ font-size:.68rem; font-weight:700; letter-spacing:.09em; padding:.24rem .6rem;
  border-radius:6px; text-transform:uppercase; }
.ve-signal .sig-badge.high{ color:#fcd34d; background:rgba(251,191,36,.12);
  border:1px solid rgba(251,191,36,.45); }
.ve-signal .sig-badge.std{ color:#c7d2fe; background:rgba(129,140,248,.12);
  border:1px solid rgba(129,140,248,.4); }
.ve-signal .sig-reason{ color:var(--muted); font-size:.9rem; line-height:1.45; }
.ve-signal.recommend{ background:linear-gradient(135deg,rgba(52,211,153,.13),rgba(16,185,129,.03) 60%);
  border-color:rgba(52,211,153,.38); }
.ve-signal.recommend .sig-word{ color:var(--green); }
.ve-signal.recommend .sig-dot{ background:var(--green); }
.ve-signal.consider{ background:linear-gradient(135deg,rgba(251,191,36,.12),rgba(245,158,11,.03) 60%);
  border-color:rgba(251,191,36,.36); }
.ve-signal.consider .sig-word{ color:var(--amber); }
.ve-signal.consider .sig-dot{ background:var(--amber); }
.ve-signal.avoid{ background:linear-gradient(135deg,rgba(248,113,113,.12),rgba(239,68,68,.03) 60%);
  border-color:rgba(248,113,113,.36); }
.ve-signal.avoid .sig-word{ color:var(--red); }
.ve-signal.avoid .sig-dot{ background:var(--red); }

/* ---- gate pills ---- */
.ve-pills{ display:flex; flex-wrap:wrap; gap:.4rem; }
.ve-pill{ display:inline-flex; align-items:center; gap:.42rem; padding:.27rem .65rem;
  font-size:.76rem; font-weight:600; border-radius:7px; border:1px solid; cursor:default;
  transition:border-color .2s var(--ease), background .2s var(--ease); }
.ve-pill .dot{ width:6px; height:6px; border-radius:50%; flex:none; }
.ve-pill.pass{ color:#86efac; background:rgba(34,197,94,.08); border-color:rgba(34,197,94,.30); }
.ve-pill.pass .dot{ background:var(--green); }
.ve-pill.fail{ color:#fca5a5; background:rgba(239,68,68,.08); border-color:rgba(239,68,68,.28); }
.ve-pill.fail .dot{ background:var(--red); }
.ve-pill.adv-pass{ color:#fcd34d; background:rgba(251,191,36,.08); border-color:rgba(251,191,36,.34); }
.ve-pill.adv-pass .dot{ background:var(--amber); }
.ve-pill.adv-fail{ color:var(--muted); background:rgba(148,163,184,.05); border-color:rgba(148,163,184,.22); }
.ve-pill.adv-fail .dot{ background:transparent; box-shadow:inset 0 0 0 1.5px var(--muted); }
.ve-pill:hover{ border-color:var(--panel-border-hi); }

/* ---- metric cards ---- */
.ve-grid{ display:grid; grid-template-columns:repeat(auto-fit,minmax(152px,1fr)); gap:.7rem; }
.ve-card{ background:var(--panel); border:1px solid var(--panel-border);
  border-radius:var(--radius); padding:.85rem .95rem; min-width:0;
  transition:transform .22s var(--ease), border-color .22s var(--ease), background .22s var(--ease); }
.ve-card:hover{ transform:translateY(-2px); border-color:var(--panel-border-hi);
  background:rgba(148,163,184,.08); }
.ve-card .k{ color:var(--muted); font-size:.71rem; font-weight:600;
  text-transform:uppercase; letter-spacing:.08em; }
.ve-card .v{ font-family:'JetBrains Mono',monospace; font-size:clamp(1.02rem,3vw,1.26rem);
  font-weight:700; margin-top:.3rem; color:#f1f5f9; overflow-wrap:anywhere;
  font-variant-numeric:tabular-nums; }
.ve-card .v.good{ color:var(--green); } .ve-card .v.bad{ color:var(--red); }
.ve-card .v.warn{ color:var(--amber); } .ve-card .v.accent{ color:var(--indigo); }
.ve-card .s{ color:var(--muted); font-size:.73rem; margin-top:.24rem; line-height:1.35; }

/* ---- panels & key-value rows ---- */
.ve-panel{ background:var(--panel); border:1px solid var(--panel-border);
  border-radius:var(--radius); padding:1rem 1.1rem;
  transition:border-color .22s var(--ease); }
.ve-panel:hover{ border-color:var(--panel-border-hi); }
.ve-rows{ display:flex; flex-direction:column; gap:.55rem; }
.ve-row{ display:flex; justify-content:space-between; align-items:baseline; gap:1rem;
  border-bottom:1px dashed rgba(148,163,184,.12); padding-bottom:.5rem; }
.ve-row:last-child{ border-bottom:none; padding-bottom:0; }
.ve-row .l{ color:var(--muted); font-size:.84rem; }
.ve-row .r{ font-family:'JetBrains Mono',monospace; font-weight:700; font-size:.92rem;
  text-align:right; color:#f1f5f9; font-variant-numeric:tabular-nums; }
.ve-row .r.good{ color:var(--green); } .ve-row .r.bad{ color:var(--red); }
.ve-row .r.warn{ color:var(--amber); }

/* ---- tables ---- */
table.ve-table{ width:100%; border-collapse:separate; border-spacing:0; font-size:.88rem; }
table.ve-table th{ text-align:left; color:var(--muted); font-size:.7rem; text-transform:uppercase;
  letter-spacing:.08em; padding:.45rem .7rem; border-bottom:1px solid var(--panel-border); }
table.ve-table td{ padding:.62rem .7rem; border-bottom:1px solid rgba(148,163,184,.07);
  color:#e2e8f0; transition:background .15s var(--ease); }
table.ve-table tr:last-child td{ border-bottom:none; }
table.ve-table tbody tr:hover td{ background:rgba(148,163,184,.045); }
table.ve-table td.mono{ font-family:'JetBrains Mono',monospace; font-weight:600;
  font-variant-numeric:tabular-nums; }
.act{ font-weight:700; padding:.13rem .55rem; border-radius:6px; font-size:.76rem;
  display:inline-block; letter-spacing:.03em; }
.act.sell{ color:#fda4af; background:rgba(244,63,94,.10); border:1px solid rgba(244,63,94,.30); }
.act.buy{ color:#86efac; background:rgba(34,197,94,.10); border:1px solid rgba(34,197,94,.30); }

/* ---- callouts ---- */
.ve-warn{ border-radius:10px; padding:.8rem 1rem; margin-top:.7rem; font-size:.88rem;
  background:rgba(251,191,36,.07); border:1px solid rgba(251,191,36,.30);
  border-left:3px solid var(--amber); color:#fde68a; }
.ve-note{ border-radius:10px; padding:.8rem 1rem; font-size:.88rem; line-height:1.5;
  background:rgba(129,140,248,.06); border:1px solid rgba(129,140,248,.28);
  border-left:3px solid var(--indigo); color:#c7d2fe; }

/* ---- streamlit chrome ---- */
[data-testid="stExpander"]{ border-radius:var(--radius); overflow:hidden; }
[data-testid="stExpander"] details{ background:var(--panel); border:1px solid var(--panel-border);
  border-radius:var(--radius); transition:border-color .22s var(--ease); }
[data-testid="stExpander"] details:hover{ border-color:var(--panel-border-hi); }
[data-testid="stTabs"] button p{ font-weight:600; font-size:.9rem; }
.stButton button{ border-radius:10px; font-weight:600; letter-spacing:.01em;
  transition:transform .18s var(--ease), box-shadow .18s var(--ease),
             border-color .18s var(--ease), background .18s var(--ease); }
.stButton button:hover{ transform:translateY(-1px); box-shadow:0 6px 20px rgba(99,102,241,.22); }
.stButton button:active{ transform:translateY(0); box-shadow:none; }
.ve-foot{ color:var(--muted); font-size:.77rem; margin-top:2.2rem; text-align:center;
  border-top:1px solid var(--panel-border); padding-top:1rem; letter-spacing:.01em; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def section(title: str, sub: str = "") -> None:
    sub_html = f'<span class="sub">{sub}</span>' if sub else ""
    st.markdown(
        f'<div class="ve-sec"><span class="dot"></span>{title}{sub_html}</div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, sub: str = "", tone: str = "") -> str:
    sub_html = f'<div class="s">{sub}</div>' if sub else ""
    return f'<div class="ve-card"><div class="k">{label}</div><div class="v {tone}">{value}</div>{sub_html}</div>'


def cards_grid(cards: list) -> None:
    st.markdown(f'<div class="ve-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def rows_panel(rows: list, lead: str = "") -> str:
    body = "".join(
        f'<div class="ve-row"><span class="l">{l}</span><span class="r {tone}">{r}</span></div>'
        for l, r, tone in rows
    )
    return f'<div class="ve-panel">{lead}<div class="ve-rows">{body}</div></div>'


def ve_table(headers: list, rows_html: str) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    return (f'<div class="ve-panel" style="padding:.4rem .5rem;"><table class="ve-table">'
            f'<thead><tr>{head}</tr></thead><tbody>{rows_html}</tbody></table></div>')


def base_layout(**kw) -> dict:
    lay = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#cbd5e1", size=12),
        margin=dict(l=10, r=12, t=34, b=10),
        hoverlabel=dict(bgcolor="#101b30", bordercolor="rgba(148,163,184,.35)",
                        font=dict(family="Inter, sans-serif", color="#e2e8f0", size=12)),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, bgcolor="rgba(0,0,0,0)"),
        showlegend=False,
    )
    lay.update(kw)
    return lay


def show_chart(fig: go.Figure, height: int = 300) -> None:
    fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", theme=None,
                    config={"displayModeBar": False, "responsive": True})


# ==================================================================
# Demo seed & session helpers (fixture shared with tests via demo.py)
# ==================================================================
def _seed_demo():
    for k, v in DEMO.items():
        st.session_state[f"in_{k}"] = v
    st.session_state["iv_history"] = demo_iv_history()
    st.session_state["hist_moves"] = list(DEFAULT_HIST_MOVES)


def _get(key, default):
    return st.session_state.get(f"in_{key}", default)


def _seed_from_ibkr(data: dict):
    for k in ("ticker", "spot", "iv_near", "iv_45", "iv_30", "rv_30",
              "volume", "atm_call", "atm_put", "drift"):
        if k in data:
            st.session_state[f"in_{k}"] = data[k]
    if data.get("historical_iv_series"):
        st.session_state["iv_history"] = data["historical_iv_series"]
    if data.get("hist_moves"):
        st.session_state["hist_moves"] = data["hist_moves"]


def _default_finnhub_key() -> str:
    """Finnhub key from st.secrets, then the FINNHUB_API_KEY env var."""
    try:
        secret = st.secrets.get("FINNHUB_API_KEY", "")
    except Exception:
        secret = ""
    return secret or os.environ.get("FINNHUB_API_KEY", "")


def _evaluate_symbol(symbol: str, vol_thr: int, iv_rv_thr: float):
    """Fetch one symbol from Yahoo and run it through the engine (shared by the
    Scanner and Upcoming Earnings tabs). Returns (raw_data, engine_result)."""
    d = _yf_fetch(symbol)
    r = VolatilityEngine.evaluate_ticker(
        iv_near=d["iv_near"], iv_45=d["iv_45"], iv_30=d["iv_30"],
        rv_30=d["rv_30"], avg_30day_volume=int(d["volume"]),
        historical_iv_series=pd.Series(d["historical_iv_series"]),
        atm_call_price=d["atm_call"], atm_put_price=d["atm_put"],
        historical_moves=d["hist_moves"], spot_price=float(d["spot"]),
        vol_threshold=int(vol_thr), iv_rv_threshold=float(iv_rv_thr),
    )
    return d, r


# ==================================================================
# Sidebar — global inputs (PRD 8)
# ==================================================================
st.sidebar.markdown(
    '<div class="ve-brand"><span class="logo">VE</span>'
    '<div><div class="name">Volatility Engine</div>'
    '<div class="sub">Earnings IV-Crush · Long Calendars · PRD v3.1</div></div></div>',
    unsafe_allow_html=True,
)
st.sidebar.divider()
if "flash" in st.session_state:
    st.sidebar.success(st.session_state.pop("flash"))

source = st.sidebar.radio(
    "Data source",
    ["Manual / Demo", "Yahoo (yfinance) — free", "IBKR (ib_insync)"],
    index=0,
)


@st.cache_data(show_spinner=False, ttl=600)
def _yf_fetch(symbol: str) -> dict:
    from data_provider_yf import fetch_yf_metrics
    return fetch_yf_metrics(symbol)


if source == "Manual / Demo":
    st.sidebar.button("Load demo ticker", on_click=_seed_demo, width="stretch", type="primary")
elif source.startswith("Yahoo"):
    with st.sidebar.expander("Yahoo Finance", expanded=True):
        st.caption("Free & keyless · ~15-min delayed quotes · real earnings dates. "
                   "IV-percentile history uses a realized-vol proxy (see README).")
        yf_symbol = st.text_input("Symbol to fetch", value=_get("ticker", "AAPL"),
                                  key="yf_symbol").upper()
        if st.button("Fetch from Yahoo", width="stretch", type="primary"):
            try:
                with st.spinner(f"Fetching {yf_symbol} from Yahoo..."):
                    data = _yf_fetch(yf_symbol)
                _seed_from_ibkr(data)
                meta = data.get("_meta", {})
                ne = meta.get("next_earnings")
                st.session_state["flash"] = (
                    f"Fetched {yf_symbol}: ATM {meta.get('atm_strike')} · "
                    f"near {meta.get('near_expiry')} ({meta.get('days_to_near')}d)"
                    + (f" · next earnings {ne}" if ne else "")
                )
                st.rerun()
            except Exception as e:
                st.error(f"Yahoo fetch failed: {e}")
                st.caption("Yahoo rate-limits busy IPs — wait a minute and retry.")
else:
    with st.sidebar.expander("IBKR connection", expanded=True):
        ib_symbol = st.text_input("Symbol to fetch", value=_get("ticker", "AAPL")).upper()
        ib_host = st.text_input("Host", value="127.0.0.1")
        ib_port = st.number_input(
            "Port (TWS 7497 paper / 7496 live · GW 4002/4001)", value=7497, step=1
        )
        ib_client = st.number_input("Client ID", value=17, step=1)
        ib_mdt = st.selectbox(
            "Market data type", options=[(3, "Delayed"), (1, "Live"), (4, "Delayed-frozen")],
            format_func=lambda o: o[1], index=0,
        )[0]
        if st.button("Fetch live data", width="stretch", type="primary"):
            try:
                from data_provider import fetch_ibkr_metrics
                with st.spinner(f"Connecting to IBKR and pulling {ib_symbol}..."):
                    data = fetch_ibkr_metrics(
                        ib_symbol, host=ib_host, port=int(ib_port),
                        client_id=int(ib_client), market_data_type=int(ib_mdt),
                    )
                _seed_from_ibkr(data)
                meta = data.get("_meta", {})
                st.session_state["flash"] = (
                    f"Fetched {ib_symbol}: ATM {meta.get('atm_strike')} · "
                    f"near {meta.get('near_expiry')} ({meta.get('days_to_near')}d) · "
                    f"45-leg {meta.get('exp_45')} ({meta.get('days_to_45leg')}d)"
                )
                st.rerun()
            except Exception as e:
                st.error(f"IBKR fetch failed: {e}")
                st.caption("Is TWS/IB Gateway running with the API enabled on this port?")

with st.sidebar.expander("Earnings calendar key (Finnhub)", expanded=False):
    st.caption("Free key from finnhub.io/register — powers the Upcoming Earnings tab. "
               "Leave blank to use the FINNHUB_API_KEY env var / secret.")
    finnhub_key = st.text_input("Finnhub API key", value=_default_finnhub_key(),
                                type="password", label_visibility="collapsed")

ticker = st.sidebar.text_input("Ticker Symbol", value=_get("ticker", "AAPL")).upper()
portfolio = st.sidebar.number_input(
    "Total Portfolio Value ($)", min_value=0.0,
    value=float(_get("portfolio", 100_000.0)), step=1000.0,
)
spot = st.sidebar.number_input(
    "Current Spot Price ($)", min_value=0.0, value=float(_get("spot", 150.0)), step=0.5
)

with st.sidebar.expander("Signal thresholds", expanded=False):
    vol_threshold = st.number_input("Min 30d Avg Volume", min_value=0, value=1_000_000, step=100_000)
    iv_rv_threshold = st.number_input("Min IV/RV Ratio", min_value=0.0, value=1.2, step=0.05)
    drift_threshold = st.number_input(
        "Strike-tilt drift threshold (|mean qtr drift|)", min_value=0.0,
        value=DRIFT_THRESHOLD, step=0.005, format="%.3f",
        help="PRD 6.4 — shift the strike ±0.5×EM only when historical drift exceeds this.",
    )

st.sidebar.caption("Educational tool — not investment advice.")

# ==================================================================
# Hero
# ==================================================================
st.markdown(
    '<div class="ve-hero"><div class="kicker">Volatility Engine</div>'
    '<h1>Earnings IV-Crush Engine</h1>'
    '<p>Long Calendar Spread detection · deterministic signal routing · '
    '10% Kelly sizing under a hard 6% debit cap</p></div>',
    unsafe_allow_html=True,
)

# ==================================================================
# Market inputs (pulled from a data provider in production)
# ==================================================================
with st.expander("Options-chain & volatility inputs", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        iv_near = st.number_input("IV — near term (front week)", min_value=0.0,
                                  value=float(_get("iv_near", 0.85)), step=0.01)
        iv_45 = st.number_input("IV — 45+ days", min_value=0.0,
                                value=float(_get("iv_45", 0.55)), step=0.01)
    with c2:
        iv_30 = st.number_input("IV — 30 day", min_value=0.0,
                                value=float(_get("iv_30", 0.70)), step=0.01)
        rv_30 = st.number_input("RV — 30 day (realized)", min_value=0.0,
                                value=float(_get("rv_30", 0.45)), step=0.01)
    with c3:
        atm_call = st.number_input("ATM Call price ($)", min_value=0.0,
                                   value=float(_get("atm_call", 4.20)), step=0.05)
        atm_put = st.number_input("ATM Put price ($)", min_value=0.0,
                                  value=float(_get("atm_put", 3.90)), step=0.05)
    c4, c5 = st.columns(2)
    with c4:
        volume = st.number_input("30-day Avg Daily Volume", min_value=0,
                                 value=int(_get("volume", 4_500_000)), step=100_000)
    with c5:
        drift = st.number_input("Historical quarterly drift (signed, e.g. 0.02)",
                                value=float(_get("drift", 0.018)), step=0.005, format="%.3f")

if "iv_history" in st.session_state:
    iv_history = pd.Series(st.session_state["iv_history"])
else:
    _rng = np.random.default_rng(1)
    iv_history = pd.Series(np.clip(_rng.normal(max(iv_30 * 0.7, 0.1), 0.08, 252), 0.05, 1.5))

hist_moves = st.session_state.get("hist_moves", list(DEFAULT_HIST_MOVES))

# ==================================================================
# Evaluate (§3 + §6)
# ==================================================================
result = VolatilityEngine.evaluate_ticker(
    iv_near=iv_near, iv_45=iv_45, iv_30=iv_30, rv_30=rv_30,
    avg_30day_volume=int(volume), historical_iv_series=iv_history,
    atm_call_price=atm_call, atm_put_price=atm_put, historical_moves=hist_moves,
    spot_price=float(spot),
    vol_threshold=int(vol_threshold), iv_rv_threshold=float(iv_rv_threshold),
)
m = result["metrics"]
signal = result["signal"]

# ==================================================================
# Signal hero card + gate pills (PRD 8: green/yellow/red)
# ==================================================================
badge = ""
if result["conviction"] == "High":
    badge = '<span class="sig-badge high">High conviction</span>'
elif result["conviction"] == "Standard":
    badge = '<span class="sig-badge std">Standard</span>'

pills = []
for c in result["checks"]:
    if c["required"]:
        cls = "pass" if c["passed"] else "fail"
    else:
        cls = "adv-pass" if c["passed"] else "adv-fail"
    pills.append(
        f'<span class="ve-pill {cls}" title="{html.escape(c["detail"], quote=True)}">'
        f'<span class="dot"></span>{html.escape(c["label"])}</span>'
    )

st.markdown(
    f'<div class="ve-signal {signal.lower()}">'
    f'<div class="sig-row"><span class="sig-dot"></span>'
    f'<span class="sig-word">{signal.upper()}</span>'
    f'<span class="sig-ticker">{html.escape(ticker)}</span>{badge}</div>'
    f'<div class="sig-reason">{html.escape(result["reason"])}</div>'
    f'<div class="ve-pills">{"".join(pills)}</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ==================================================================
# Metric cards (PRD 8 metric dataframe, as a responsive grid)
# ==================================================================
section("Core & advanced metrics", "PRD §2 + §6")
slope = m["term_structure_slope"]
cards_grid([
    metric_card("Term slope", f"{slope:+.3f}",
                "IV_near − IV_45 · backwardation > 0",
                "good" if slope > 0 else "bad"),
    metric_card("IV / RV", f"{m['iv_rv_ratio']:.2f}",
                f"need > {iv_rv_threshold:.2f}",
                "good" if m["iv_rv_ratio"] > iv_rv_threshold else "bad"),
    metric_card("IV percentile", f"{m['iv_percentile']:.0f}%",
                "252-day window · need ≥ 70%",
                "good" if m["iv_percentile"] >= 70 else "bad"),
    metric_card("Expected move", f"${m['expected_move_dollars']:.2f}",
                f"{m['expected_move_pct']:.1%} of spot · 0.85 × straddle", "accent"),
    metric_card("Hist. earnings move", f"{m['historical_move_mean']:.1%}",
                "mean |move|, last 4–8 quarters"),
    metric_card("Magnitude premium", "Yes" if m["magnitude_premium_detected"] else "No",
                "EM > 1.25 × historical move",
                "warn" if m["magnitude_premium_detected"] else ""),
    metric_card("30-day ADV", f"{m['avg_30day_volume']/1e6:.1f}M",
                f"need > {vol_threshold/1e6:.1f}M shares",
                "good" if m["avg_30day_volume"] > vol_threshold else "bad"),
])

with st.expander("Metrics table view"):
    st.dataframe(pd.DataFrame({
        "Metric": [
            "Term Structure Slope (IV_near − IV_45)", "IV / RV Ratio",
            "IV Percentile (252d)", "Expected Move ($, 1σ)", "Expected Move (% of spot)",
            "Historical Move Mean (%)", "Earnings Magnitude Premium (EM > hist×1.25)",
            "30-day Avg Daily Volume",
        ],
        "Value": [
            f"{slope:+.4f}", f"{m['iv_rv_ratio']:.2f}", f"{m['iv_percentile']:.1f}%",
            f"${m['expected_move_dollars']:.2f}", f"{m['expected_move_pct']:.2%}",
            f"{m['historical_move_mean']:.2%}",
            "Yes" if m["magnitude_premium_detected"] else "No",
            f"{m['avg_30day_volume']:,}",
        ],
    }), hide_index=True, width="stretch")

# ==================================================================
# Volatility structure charts
# ==================================================================
ch1, ch2 = st.columns(2)
with ch1:
    section("IV term structure", "front-week premium vs 45d")
    fig = go.Figure()
    xs, ys = [int(T_NEAR_DAYS), 30, 45], [iv_near * 100, iv_30 * 100, iv_45 * 100]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines+markers+text",
        text=[f"{v:.0f}%" for v in ys], textposition="top center",
        textfont=dict(family="JetBrains Mono, monospace", size=11, color="#cbd5e1"),
        line=dict(color=ACCENT, width=3, shape="spline"),
        marker=dict(size=9, color=[RED if slope > 0 else MUTED, ACCENT, CYAN]),
        cliponaxis=False,
        hovertemplate="%{x}d → %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=iv_45 * 100, line_dash="dot", line_color="rgba(148,163,184,.35)")
    fig.update_layout(**base_layout(
        xaxis_title="days to expiry", yaxis_ticksuffix="%",
        xaxis_range=[2, 50], margin=dict(l=10, r=28, t=34, b=10),
        title=dict(text=("Backwardation — front IV elevated" if slope > 0
                         else "Contango / flat — no front premium"),
                   font=dict(size=12, color=GREEN if slope > 0 else MUTED), x=0),
    ))
    show_chart(fig, height=265)
with ch2:
    section("IV history vs current", "drives the 252d percentile")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=iv_history.tail(252).to_numpy() * 100, mode="lines",
        line=dict(color=CYAN, width=2), fill="tozeroy", fillcolor="rgba(34,211,238,.07)",
        hovertemplate="day %{x} → %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=iv_30 * 100, line_dash="dash", line_color=AMBER,
                  annotation_text=f"current {iv_30*100:.0f}%",
                  annotation_font_color=AMBER, annotation_position="top left")
    fig.update_layout(**base_layout(
        xaxis_title="trading days", yaxis_ticksuffix="%",
        title=dict(text=f"IV percentile: {m['iv_percentile']:.0f}%",
                   font=dict(size=12, color="#cbd5e1"), x=0),
    ))
    show_chart(fig, height=265)

# ==================================================================
# Position sizing & risk (PRD 5 — always displayed)
# ==================================================================
section("Position sizing & risk", "10% Kelly + hard 6% debit cap · PRD §5")
max_debit = VolatilityEngine.calculate_position_sizing(portfolio)
kelly = VolatilityEngine.calculate_kelly_fraction()
cards_grid([
    metric_card("Max debit (6% cap)", f"${max_debit:,.0f}",
                "hard limit — never exceed", "bad"),
    metric_card("Suggested debit", f"${portfolio * kelly['fractional_kelly']:,.0f}",
                f"applied {KELLY_MULTIPLIER:.0%} Kelly ≈ {kelly['fractional_kelly']:.2%}", "good"),
    metric_card("Full Kelly f*", f"{kelly['full_kelly']:.2%}",
                f"p−q/b · b = {kelly['win_loss_ratio_b']:.2f}"),
    metric_card("Per-trade stats",
                f"{HISTORICAL_WIN_RATE:.0%} / +{HISTORICAL_AVG_WIN:.0%} / −{HISTORICAL_AVG_LOSS:.0%}",
                "win rate · avg win · avg loss (72.5k events)"),
])
st.markdown(
    f'<div class="ve-warn">Never exceed <b>${max_debit:,.2f}</b> of total debit on this '
    f'trade ({MAX_PORTFOLIO_FRACTION:.0%} of portfolio). The applied-Kelly suggestion '
    f'(~{kelly["fractional_kelly"]:.2%}) is the smoother-equity default.</div>',
    unsafe_allow_html=True,
)

# ==================================================================
# Trade structuring (PRD 4, 6.4, 6.5) — Recommend only
# ==================================================================
section("Proposed trade — Long Calendar Spread", "PRD §4 + §6.4 strike tilt + §6.5 exits")
if signal == "Recommend":
    spread = VolatilityEngine.build_calendar_spread(
        ticker=ticker, spot_price=spot, expected_move=m["expected_move_dollars"],
        historical_drift=drift, drift_threshold=float(drift_threshold),
    )
    strike = spread["strike"]

    est_debit = VolatilityEngine.estimate_calendar_debit(spot, strike, iv_near, iv_45)
    tcol1, tcol2 = st.columns([3, 2])
    with tcol1:
        legs_rows = "".join(
            f'<tr><td>{leg["leg"]}</td>'
            f'<td><span class="act {leg["action"].lower()}">{leg["action"]}</span></td>'
            f'<td class="mono">${leg["strike"]:,.2f}</td><td>{leg["expiry"]}</td></tr>'
            for leg in spread["legs"]
        )
        st.markdown(ve_table(["Leg", "Action", "Strike", "Expiry"], legs_rows),
                    unsafe_allow_html=True)
        st.caption(f"Strike **${strike:,.2f}** · {spread['rationale']}")
        debit_in = st.number_input(
            "Calendar debit per spread ($/share — overwrite with your broker's quote)",
            min_value=0.0, value=round(est_debit, 2), step=0.05,
            help="Prefilled with a Black-Scholes estimate from your IV inputs.",
        )
        plan = VolatilityEngine.build_position_plan(portfolio, debit_in)
        cards_grid([
            metric_card("Debit / spread", f"${plan['cost_per_spread']:,.0f}",
                        f"${debit_in:.2f} × 100 multiplier"),
            metric_card("Contracts (Kelly)", f"{plan['contracts_suggested']}",
                        f"≈ ${plan['estimated_cost_suggested']:,.0f} at risk", "good"),
            metric_card("Contracts (6% max)", f"{plan['contracts_max']}",
                        f"≈ ${plan['estimated_cost_max']:,.0f} — ceiling", "warn"),
        ])
        if plan["contracts_max"] == 0:
            st.markdown('<div class="ve-note">Debit too large (or zero) for this portfolio '
                        'under the 6% cap — size below one contract.</div>',
                        unsafe_allow_html=True)

        exit_plan = VolatilityEngine.build_exit_plan(debit_in, iv_near, iv_45)
        st.markdown("")
        st.markdown(rows_panel([
            (f"Take-profit alert (+{exit_plan['take_profit_low_pct']:.0%} on debit)",
             f"${exit_plan['take_profit_low_value']*100:,.0f} / spread", "good"),
            (f"Stretch target (+{exit_plan['take_profit_high_pct']:.0%} on debit)",
             f"${exit_plan['take_profit_high_value']*100:,.0f} / spread", "good"),
            (f"Velocity exit — front IV prints ≤ this in first 5 min "
             f"({exit_plan['velocity_fraction']:.0%} of crush)",
             f"{exit_plan['velocity_iv_level']:.0%} IV", "warn"),
            ("Expected IV crush (front → back)",
             f"−{exit_plan['expected_iv_crush']:.0%} pts", ""),
            ("Max loss (debit paid)",
             f"${exit_plan['max_loss']*100:,.0f} / spread", "bad"),
        ], lead='<div class="k" style="color:var(--muted);font-size:.72rem;font-weight:600;'
                'text-transform:uppercase;letter-spacing:.07em;margin-bottom:.6rem;">'
                'Dynamic exit protocol · PRD §6.5</div>'), unsafe_allow_html=True)
    with tcol2:
        section("Payoff at front expiry", "per 1 spread, back leg at IV_45")
        S_range = np.linspace(spot * 0.82, spot * 1.18, 121)
        t_back_left = (T_FAR_DAYS - T_NEAR_DAYS) / 365.0  # back-leg tenor remaining
        vals = np.array([bs_call(s, strike, t_back_left, iv_45)
                         - bs_call(s, strike, 0.0, iv_near)
                         for s in S_range])
        pnl = (vals - debit_in) * 100.0
        pos = np.where(pnl >= 0, pnl, np.nan)
        neg = np.where(pnl < 0, pnl, np.nan)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=S_range, y=pos, mode="lines",
                                 line=dict(color=GREEN, width=2.5), fill="tozeroy",
                                 fillcolor="rgba(52,211,153,.12)",
                                 hovertemplate="$%{x:.2f} → $%{y:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=S_range, y=neg, mode="lines",
                                 line=dict(color=RED, width=2.5), fill="tozeroy",
                                 fillcolor="rgba(248,113,113,.10)",
                                 hovertemplate="$%{x:.2f} → $%{y:,.0f}<extra></extra>"))
        fig.add_vline(x=spot, line_dash="dash", line_color="rgba(148,163,184,.5)",
                      annotation_text="spot", annotation_font_color=MUTED)
        if abs(strike - spot) > 1e-9:
            fig.add_vline(x=strike, line_dash="dot", line_color=ACCENT,
                          annotation_text="strike", annotation_font_color=ACCENT,
                          annotation_position="top left")
        fig.add_hline(y=0, line_color="rgba(148,163,184,.35)")
        fig.update_layout(**base_layout(
            xaxis_title="underlying at front expiry ($)", yaxis_title="P&L / spread ($)",
            xaxis_tickprefix="$", yaxis_tickprefix="$",
        ))
        show_chart(fig, height=430)
else:
    st.markdown(
        f'<div class="ve-note">Trade legs are only generated for <b>RECOMMEND</b> signals '
        f'(current: <b>{signal.upper()}</b>). Adjust the inputs or thresholds — the failed '
        f'gates are highlighted red in the banner above.</div>',
        unsafe_allow_html=True,
    )

# ==================================================================
# Validation — backtest + Monte Carlo (PRD 5 & 7), recomputed live
# ==================================================================
section("Screening & validation",
        "earnings calendar · watchlist scan · backtest-derived edge · Monte Carlo · PRD §5 + §7")


@st.cache_data(show_spinner=False, ttl=900)
def _finnhub_cal(from_date: str, to_date: str, key: str) -> list:
    from data_provider_finnhub import fetch_earnings_calendar
    return fetch_earnings_calendar(from_date, to_date, key)


@st.cache_data(show_spinner=False)
def cached_backtest(n_events: int, trades_per_year: int, capital: float) -> dict:
    return run_backtest(cfg=BacktestConfig(
        n_events=n_events, starting_capital=capital, trades_per_year=trades_per_year,
    ))


@st.cache_data(show_spinner=False)
def cached_mc(capital: float, win_rate: float, avg_win: float, avg_loss: float,
              sizing: float, n_trades: int, n_paths: int) -> dict:
    return VolatilityEngine.run_monte_carlo(
        starting_capital=capital, win_rate=win_rate, avg_win=avg_win,
        avg_loss=avg_loss, sizing_fraction=sizing, n_trades=n_trades, n_paths=n_paths,
    )


tab_earn, tab_scan, tab_bt, tab_mc = st.tabs(
    ["Upcoming Earnings", "Scanner", "Backtest", "Monte Carlo"])

REAL_EVENT_COLS = ["iv_near", "iv_far", "iv_near_post", "iv_far_post", "realized_move"]

SIG_RANK = {"Recommend": 0, "Consider": 1, "Avoid": 2}
CONV_RANK = {"High": 0, "Standard": 1, "—": 2}

with tab_earn:
    st.caption("Pulls the upcoming **earnings calendar** from Finnhub (free key), runs each "
               "reporting name through the engine on Yahoo data, and ranks the setups — with "
               "the session you'd enter the calendar (**AMC** reports → that afternoon; "
               "**BMO** → the prior close). Sidebar thresholds apply.")
    if not finnhub_key:
        st.markdown('<div class="ve-note">Add a free Finnhub API key in the sidebar '
                    '(<b>Earnings calendar key</b>) to enable this tab — register at '
                    'finnhub.io/register.</div>', unsafe_allow_html=True)
    e1, e2, e3 = st.columns(3)
    days_ahead = e1.slider("Days ahead", 1, 14, 7)
    max_names = e2.slider("Max names to evaluate", 5, 60, 25, step=5,
                          help="Each name is a Yahoo fetch — higher is slower and more "
                               "rate-limit prone. Soonest-to-report are evaluated first.")
    min_adv_m = e3.number_input("Min ADV (M shares)", min_value=0.0, value=1.0, step=0.5)

    if st.button("Fetch upcoming earnings", type="primary", disabled=not finnhub_key):
        try:
            frm, to = upcoming_window(int(days_ahead))
            with st.spinner(f"Fetching earnings calendar {frm} → {to}…"):
                cal = _finnhub_cal(frm, to, finnhub_key)
            seen, agenda = set(), []
            for row in cal:  # one row per symbol = its soonest upcoming report
                if row["symbol"] not in seen:
                    seen.add(row["symbol"])
                    agenda.append(row)
            agenda = agenda[: int(max_names)]
            rows, errors = [], []
            prog = st.progress(0.0, text="Evaluating…")
            for i, row in enumerate(agenda):
                sym = row["symbol"]
                try:
                    d, r = _evaluate_symbol(sym, vol_threshold, iv_rv_threshold)
                    mm = r["metrics"]
                    adv_m = mm["avg_30day_volume"] / 1e6
                    if adv_m < float(min_adv_m):
                        continue
                    entry, _react = entry_session(row["date"], row["hour"])
                    rows.append({
                        "Ticker": sym,
                        "Entry": entry_label(entry, row["hour"]),
                        "_entry_iso": entry.date().isoformat(),
                        "Earnings": row["date"],
                        "Signal": r["signal"],
                        "Conviction": r["conviction"] or "—",
                        "Slope": round(mm["term_structure_slope"], 3),
                        "IV/RV": round(mm["iv_rv_ratio"], 2),
                        "IV %ile": round(mm["iv_percentile"]),
                        "EM %": round(mm["expected_move_pct"] * 100, 1),
                        "ADV (M)": round(adv_m, 1),
                    })
                except Exception as e:
                    errors.append(f"{sym}: {e}")
                prog.progress((i + 1) / len(agenda), text=f"Evaluated {sym} ({i + 1}/{len(agenda)})")
            prog.empty()
            rows.sort(key=lambda x: (SIG_RANK[x["Signal"]], CONV_RANK.get(x["Conviction"], 2),
                                     x["_entry_iso"], -x["IV %ile"]))
            st.session_state["earn_rows"] = rows
            st.session_state["earn_errors"] = errors
            if not rows and not errors:
                st.warning("No reporting names cleared the ADV filter in that window.")
        except Exception as e:
            st.error(f"Earnings calendar fetch failed: {e}")
            st.caption("Check the Finnhub key, or that the free-tier rate limit (60/min) "
                       "isn't exhausted.")

    if st.session_state.get("earn_rows"):
        show = [{k: v for k, v in r.items() if not k.startswith("_")}
                for r in st.session_state["earn_rows"]]
        st.dataframe(pd.DataFrame(show), hide_index=True, width="stretch")
        ecols = st.columns([3, 2])
        epick = ecols[0].selectbox(
            "Load a candidate into the engine",
            [r["Ticker"] for r in st.session_state["earn_rows"]], key="earn_pick",
        )
        if ecols[1].button(f"Load {epick} into inputs", width="stretch", key="earn_load"):
            _seed_from_ibkr(_yf_fetch(epick))
            st.session_state["flash"] = f"Loaded {epick} from the earnings calendar."
            st.rerun()
    for err in st.session_state.get("earn_errors", []):
        st.caption(err)

with tab_bt:
    st.caption("Prices an ATM calendar through each earnings event with Black-Scholes to "
               "**derive** the win rate / avg win / avg loss that drive Kelly sizing — "
               "instead of asserting them. Updates live.")
    bc1, bc2 = st.columns(2)
    bt_events = bc1.slider("Earnings events", 200, 5000, 2000, step=100)
    bt_per_year = bc2.slider("Trades / year (CAGR & Sharpe)", 10, 100, 50, step=5)
    with st.expander("Use real earnings events (CSV) instead of the synthetic universe"):
        st.caption("Columns: `" + "`, `".join(REAL_EVENT_COLS) + "` — IVs as decimals "
                   "(0.85 = 85%), `realized_move` as a signed fraction (0.04 = +4%).")
        events_file = st.file_uploader("Events CSV", type=["csv"], label_visibility="collapsed")

    bt, bt_source = None, f"synthetic universe · {int(bt_events):,} events"
    if events_file is not None:
        try:
            ev = pd.read_csv(events_file)
            missing = set(REAL_EVENT_COLS) - set(ev.columns)
            if missing:
                st.error(f"CSV is missing columns: {sorted(missing)} — using synthetic events.")
            else:
                bt = run_backtest(events=ev, cfg=BacktestConfig(
                    starting_capital=float(portfolio), trades_per_year=int(bt_per_year)))
                bt_source = f"real events CSV · {len(ev):,} rows (events slider ignored)"
        except Exception as e:
            st.error(f"Could not read events CSV: {e} — using synthetic events.")
    if bt is None:
        bt = cached_backtest(int(bt_events), int(bt_per_year), float(portfolio))
    st.caption(f"Source: **{bt_source}**")
    prd = bt["prd_reference"]
    ek = bt["empirical_kelly"]

    bcol, ecol = st.columns([2, 3])
    with bcol:
        stat_rows = "".join(
            f'<tr><td>{name}</td><td class="mono">{val}</td><td class="mono" '
            f'style="color:var(--muted)">{ref}</td></tr>'
            for name, val, ref in [
                ("Win rate", f"{bt['win_rate']:.1%}", f"{prd['win_rate']:.1%}"),
                ("Avg win", f"+{bt['avg_win']:.1%}", f"+{prd['avg_win']:.1%}"),
                ("Avg loss", f"−{bt['avg_loss']:.1%}", f"−{prd['avg_loss']:.1%}"),
                ("Profit factor", f"{bt['profit_factor']:.2f}", "—"),
                ("Expectancy / trade", f"{bt['expectancy']:+.2%}", "—"),
                ("CAGR", f"{bt['cagr']:.1%}", "—"),
                ("Max drawdown", f"{bt['max_drawdown']:.1%}", "—"),
                ("Sharpe (annual)", f"{bt['sharpe_annual']:.2f}", "—"),
            ]
        )
        st.markdown(ve_table(["Metric", "Backtest", "PRD ref"], stat_rows),
                    unsafe_allow_html=True)
        st.caption(f"{bt['n_trades']:,} valid trades · Empirical Kelly: full "
                   f"**{ek['full_kelly']:.1%}**, applied 10% → **{ek['fractional_kelly']:.2%}** "
                   f"(vs {MAX_PORTFOLIO_FRACTION:.0%} cap).")
    with ecol:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=bt["equity_curve"], mode="lines", line=dict(color=ACCENT, width=2.5),
            hovertemplate="trade %{x} → $%{y:,.0f}<extra></extra>",
        ))
        fig.add_hline(y=bt["equity_curve"][0], line_dash="dash",
                      line_color="rgba(148,163,184,.4)")
        fig.update_layout(**base_layout(
            xaxis_title="trade #", yaxis_tickprefix="$", yaxis_type="log",
            title=dict(text=f"Equity (log) — 6% debit sizing · CAGR {bt['cagr']:.1%}",
                       font=dict(size=12, color="#cbd5e1"), x=0),
        ))
        show_chart(fig, height=250)

        wins_pnl = bt["pnl_pct"][bt["pnl_pct"] > 0] * 100
        loss_pnl = bt["pnl_pct"][bt["pnl_pct"] <= 0] * 100
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=loss_pnl, nbinsx=40, marker_color=RED, opacity=.85,
                                   hovertemplate="%{x:.0f}%: %{y}<extra>losses</extra>"))
        fig.add_trace(go.Histogram(x=wins_pnl, nbinsx=40, marker_color=GREEN, opacity=.85,
                                   hovertemplate="%{x:.0f}%: %{y}<extra>wins</extra>"))
        fig.add_vline(x=0, line_color="rgba(148,163,184,.4)")
        fig.update_layout(**base_layout(
            barmode="overlay", xaxis_title="P&L on debit (%)", xaxis_ticksuffix="%",
            title=dict(text="Per-trade P&L distribution",
                       font=dict(size=12, color="#cbd5e1"), x=0),
        ))
        show_chart(fig, height=240)

with tab_mc:
    st.caption("Each trade risks a fixed fraction of *current* equity, so capital "
               "**compounds geometrically** (exponential growth). The fan shows the "
               "5th / 50th / 95th percentile of equity across all paths; Risk of Ruin = "
               "probability of a ≥50% peak-to-trough drawdown. Updates live.")
    mcc1, mcc2, mcc3 = st.columns(3)
    n_trades = mcc1.slider("Trades per path", 50, 1000, 500, step=50)
    n_paths = mcc2.slider("Parallel paths", 100, 5000, 1000, step=100)
    sizing_options = {
        f"{MAX_PORTFOLIO_FRACTION:.0%} cap (PRD)": MAX_PORTFOLIO_FRACTION,
        f"Applied Kelly ({kelly['fractional_kelly']:.2%})": kelly["fractional_kelly"],
    }
    sizing_choice = mcc3.radio("Sizing per trade", list(sizing_options), horizontal=True)
    sizing = sizing_options[sizing_choice]
    opt1, opt2 = st.columns(2)
    use_bt = opt1.checkbox("Use backtest-derived win/loss stats (else PRD constants)", value=True)
    y_scale = opt2.radio(
        "Equity axis", ["Log", "Linear"], horizontal=True,
        help="Compounding is exponential: a straight line on a log axis (equal % steps), "
             "and an upward-curving sweep on a linear axis.",
    )
    is_log = y_scale == "Log"
    if use_bt:
        stats = dict(win_rate=bt["win_rate"], avg_win=bt["avg_win"], avg_loss=bt["avg_loss"])
    else:
        stats = dict(win_rate=prd["win_rate"], avg_win=prd["avg_win"], avg_loss=prd["avg_loss"])

    mc = cached_mc(float(portfolio), stats["win_rate"], stats["avg_win"],
                   stats["avg_loss"], float(sizing), int(n_trades), int(n_paths))

    fig = go.Figure()
    xs = mc["trade_index"]
    fig.add_trace(go.Scatter(
        x=np.concatenate([xs, xs[::-1]]),
        y=np.concatenate([mc["p95"], mc["p5"][::-1]]),
        fill="toself", fillcolor="rgba(129,140,248,.14)",
        line=dict(color="rgba(129,140,248,.35)", width=1),
        hoverinfo="skip", name="5th–95th pct",
    ))
    fig.add_trace(go.Scatter(x=xs, y=mc["p50"], mode="lines",
                             line=dict(color=CYAN, width=2.5), name="median",
                             hovertemplate="trade %{x} → $%{y:,.0f}<extra>median</extra>"))
    fig.add_hline(y=float(portfolio), line_dash="dash", line_color="rgba(148,163,184,.45)",
                  annotation_text="start", annotation_font_color=MUTED)
    fig.update_layout(**base_layout(
        xaxis_title="trade #", yaxis_type="log" if is_log else "linear",
        yaxis_tickprefix="$", showlegend=True,
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right",
                    bgcolor="rgba(0,0,0,0)"),
        title=dict(text=f"Equity fan ({y_scale.lower()}) — {sizing:.2%} sizing · "
                        f"p {stats['win_rate']:.0%} · +{stats['avg_win']:.0%} / "
                        f"−{stats['avg_loss']:.0%}",
                   font=dict(size=12, color="#cbd5e1"), x=0),
    ))
    show_chart(fig, height=360)

    ror = mc["risk_of_ruin"]
    cards_grid([
        metric_card("Median final equity", f"${mc['median_final_equity']:,.0f}",
                    f"start ${portfolio:,.0f} · {mc['n_trades']} trades", "accent"),
        metric_card("Risk of ruin", f"{ror:.2%}",
                    "P(≥50% drawdown) · target < 5%",
                    "good" if ror < 0.05 else "bad"),
        metric_card("P(finish below start)", f"{mc['prob_below_start']:.2%}",
                    f"across {mc['n_paths']:,} paths",
                    "good" if mc["prob_below_start"] < 0.05 else "warn"),
    ])

with tab_scan:
    st.caption("Scan a watchlist via **Yahoo Finance** (free, ~15-min delayed) and rank by "
               "signal → conviction → IV percentile. Uses the sidebar thresholds.")
    watch = st.text_area(
        "Tickers (comma / space separated)",
        value="AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, AMD",
        height=70,
    )
    if st.button("Scan watchlist", type="primary"):
        syms = sorted({s.strip().upper() for s in re.split(r"[,\s]+", watch) if s.strip()})
        rows, errors = [], []
        prog = st.progress(0.0, text="Scanning…")
        for i, sym in enumerate(syms):
            try:
                d, r = _evaluate_symbol(sym, vol_threshold, iv_rv_threshold)
                mm = r["metrics"]
                rows.append({
                    "Ticker": sym,
                    "Signal": r["signal"],
                    "Conviction": r["conviction"] or "—",
                    "Slope": round(mm["term_structure_slope"], 3),
                    "IV/RV": round(mm["iv_rv_ratio"], 2),
                    "IV %ile": round(mm["iv_percentile"]),
                    "EM %": round(mm["expected_move_pct"] * 100, 1),
                    "ADV (M)": round(mm["avg_30day_volume"] / 1e6, 1),
                    "Next earnings": (d.get("_meta") or {}).get("next_earnings") or "—",
                })
            except Exception as e:
                errors.append(f"{sym}: {e}")
            prog.progress((i + 1) / len(syms), text=f"Scanned {sym} ({i + 1}/{len(syms)})")
        prog.empty()
        rows.sort(key=lambda x: (SIG_RANK[x["Signal"]],
                                 CONV_RANK.get(x["Conviction"], 2), -x["IV %ile"]))
        st.session_state["scan_rows"] = rows
        st.session_state["scan_errors"] = errors

    if st.session_state.get("scan_rows"):
        st.dataframe(pd.DataFrame(st.session_state["scan_rows"]),
                     hide_index=True, width="stretch")
        pick_cols = st.columns([3, 2])
        pick = pick_cols[0].selectbox(
            "Load a scanned ticker into the engine",
            [r["Ticker"] for r in st.session_state["scan_rows"]],
        )
        if pick_cols[1].button(f"Load {pick} into inputs", width="stretch"):
            _seed_from_ibkr(_yf_fetch(pick))
            st.session_state["flash"] = f"Loaded {pick} from the scan."
            st.rerun()
    for err in st.session_state.get("scan_errors", []):
        st.caption(err)

st.markdown(
    '<div class="ve-foot">Volatility Engine · educational tool — not investment advice. '
    'Verify every option quote with your broker before trading.</div>',
    unsafe_allow_html=True,
)
