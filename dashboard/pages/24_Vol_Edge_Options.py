"""Page 24: Vol Edge — Options Lab (QuantLib-powered).

A monitoring + what-if panel for the Vol Edge short-straddle strategy.

- **Straddle pricer** — prices an ATM (or custom-strike) European straddle and
  its greeks using QuantLib's analytic Black-Scholes-Merton engine.  This is
  the canonical pricing path; the live bots use their own lightweight BS
  module (`Vol_Edge/core/options_pricer.py`) — numbers should agree closely.
- **Entry-gate simulator** — given DVOL and 30-day realised vol, shows whether
  the bot's entry gate (DVOL < 50 AND VRP > −5) would fire today.
- **Live positions** — surfaces open straddles once the bots persist them.
"""

from __future__ import annotations

import math
from datetime import date

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Vol Edge — Options Lab", page_icon="🎚️", layout="wide")
st.title("🎚️ Vol Edge — Options Lab")

try:
    import QuantLib as ql
    _QL_OK = True
except Exception:
    _QL_OK = False

if not _QL_OK:
    st.error(
        "QuantLib is not installed in this environment.\n\n"
        "Install it with `pip install QuantLib`, then reload this page."
    )
    st.stop()

st.caption(
    "Prices European straddles with QuantLib's analytic Black-Scholes-Merton engine "
    "and mirrors the Vol Edge entry gate. Vol Edge sells ATM straddles to harvest the "
    "variance risk premium — see **Strategy Explainer → Vol Edge** for the full thesis."
)


# ──────────────────────────────────────────────────────────────────────────────
# QuantLib pricing helper
# ──────────────────────────────────────────────────────────────────────────────
def price_european(S: float, K: float, T_years: float, r: float, sigma: float,
                   is_call: bool) -> dict:
    """Analytic BSM price + greeks for one European option via QuantLib."""
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    cal = ql.NullCalendar()

    expiry = today + max(1, int(round(T_years * 365)))
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, K)
    exercise = ql.EuropeanExercise(expiry)
    option = ql.VanillaOption(payoff, exercise)

    spot = ql.QuoteHandle(ql.SimpleQuote(S))
    rate_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc))
    div_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, dc))
    vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, cal, sigma, dc))
    process = ql.BlackScholesMertonProcess(spot, div_ts, rate_ts, vol_ts)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

    return {
        "price": option.NPV(),
        "delta": option.delta(),
        "gamma": option.gamma(),
        "vega": option.vega() / 100.0,       # per 1 vol-point (1%)
        "theta": option.thetaPerDay(),       # per calendar day
    }


# ──────────────────────────────────────────────────────────────────────────────
# Inputs
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Underlying & contract")
    asset = st.selectbox("Asset", ["BTC", "ETH"], index=0)
    default_spot = 95000.0 if asset == "BTC" else 2130.0
    spot = st.number_input("Spot price ($)", min_value=0.01, value=default_spot, step=1.0)
    atm = st.checkbox("ATM strike (= spot)", value=True)
    strike = spot if atm else st.number_input("Strike ($)", min_value=0.01, value=default_spot, step=1.0)
    dte_days = st.slider("Days to expiry", 1, 30, 7)
    iv_pct = st.slider("Implied vol — DVOL (%)", 5.0, 150.0, 50.0, step=0.5)
    rate_pct = st.number_input("Risk-free rate (%)", value=0.0, step=0.25)

T = dte_days / 365.0
r = rate_pct / 100.0
sigma = iv_pct / 100.0

call = price_european(spot, strike, T, r, sigma, is_call=True)
put = price_european(spot, strike, T, r, sigma, is_call=False)
straddle_premium = call["price"] + put["price"]

# Straddle greeks = sum of legs (long-straddle convention; the bot is SHORT, so
# its position greeks are the negatives of these).
s_delta = call["delta"] + put["delta"]
s_gamma = call["gamma"] + put["gamma"]
s_vega = call["vega"] + put["vega"]
s_theta = call["theta"] + put["theta"]


# ──────────────────────────────────────────────────────────────────────────────
# Straddle pricer output
# ──────────────────────────────────────────────────────────────────────────────
st.subheader(f"ATM Straddle — {asset}  ·  K=${strike:,.2f}  ·  {dte_days}d  ·  IV={iv_pct:.1f}%")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Straddle premium", f"${straddle_premium:,.2f}",
          help="Call + Put price. What the bot collects per unit when selling.")
m2.metric("Lower breakeven", f"${strike - straddle_premium:,.2f}")
m3.metric("Upper breakeven", f"${strike + straddle_premium:,.2f}")
m4.metric("Breakeven band", f"±{(straddle_premium / spot * 100):.2f}%",
          help="Move (either way) needed to wipe out the premium. Wider band = safer for the short seller.")

greeks_df = pd.DataFrame(
    {
        "Leg": ["Call", "Put", "Straddle (long)", "Bot position (short)"],
        "Price": [call["price"], put["price"], straddle_premium, -straddle_premium],
        "Delta": [call["delta"], put["delta"], s_delta, -s_delta],
        "Gamma": [call["gamma"], put["gamma"], s_gamma, -s_gamma],
        "Vega (per 1% IV)": [call["vega"], put["vega"], s_vega, -s_vega],
        "Theta (per day)": [call["theta"], put["theta"], s_theta, -s_theta],
    }
)
st.dataframe(
    greeks_df.round(4), use_container_width=True, hide_index=True,
    column_config={
        "Vega (per 1% IV)": st.column_config.NumberColumn(help="P&L per 1 vol-point move. The bot is short vega — it profits when IV falls."),
        "Theta (per day)": st.column_config.NumberColumn(help="Daily time-decay. The bot is short the straddle, so it COLLECTS theta (its position theta is positive)."),
    },
)
st.caption(
    "The bot sells the straddle, so its **position greeks are the negatives** of the "
    "long-straddle row: short vega (gains when IV falls), short gamma (loses on big moves), "
    "and **positive theta** (collects time-decay daily). The whole strategy is a bet that "
    "realised movement stays inside the breakeven band."
)


# ──────────────────────────────────────────────────────────────────────────────
# Entry-gate simulator
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Entry-gate simulator")
st.caption("Vol Edge only sells a straddle when implied vol is calm AND richer than realised. "
           "Enter today's DVOL and 30-day realised vol to see whether the gate fires.")

g1, g2 = st.columns(2)
with g1:
    dvol = st.number_input("DVOL (implied, %)", min_value=0.0, value=float(iv_pct), step=0.5,
                           help="Deribit 30-day implied-vol index for the asset.")
with g2:
    rv30 = st.number_input("Realised vol — RV30 (%)", min_value=0.0, value=45.0, step=0.5,
                           help="Trailing 30-day annualised realised volatility.")

vrp = dvol - rv30
DVOL_MAX, MIN_VRP = 50.0, -5.0
gate_dvol = dvol < DVOL_MAX
gate_vrp = vrp > MIN_VRP

c1, c2, c3 = st.columns(3)
c1.metric("VRP (DVOL − RV30)", f"{vrp:+.1f} vol-pts")
c2.metric(f"DVOL < {DVOL_MAX:.0f}?", "✅ pass" if gate_dvol else "❌ block")
c3.metric(f"VRP > {MIN_VRP:+.0f}?", "✅ pass" if gate_vrp else "❌ block")

if gate_dvol and gate_vrp:
    st.success(f"**WOULD ENTER** — calm IV regime ({dvol:.1f} < {DVOL_MAX:.0f}) and IV richer than "
               f"realised (VRP {vrp:+.1f} > {MIN_VRP:+.0f}). The bot would sell the ATM straddle above, "
               f"vega-sized to ≤ {'2' if asset == 'BTC' else '5'}% of equity.")
else:
    reasons = []
    if not gate_dvol:
        reasons.append(f"DVOL too high ({dvol:.1f} ≥ {DVOL_MAX:.0f}) — selling into a vol spike")
    if not gate_vrp:
        reasons.append(f"VRP too low ({vrp:+.1f} ≤ {MIN_VRP:+.0f}) — implied isn't rich enough vs realised")
    st.warning("**NO ENTRY** — " + "; ".join(reasons) + ".")


# ──────────────────────────────────────────────────────────────────────────────
# Live positions (placeholder until the straddle bots persist trades)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Live Vol Edge positions")
st.info(
    "No straddle trades have been recorded yet — the `straddle-btc` / `straddle-eth` bots are "
    "paper-trading and the entry gate (DVOL < 50 & VRP > −5) hasn't fired since deploy. Once "
    "they open a position, this section will show the live straddle, its greeks, and mark-to-market "
    "P&L (synced from `Vol_Edge/Straddle_V1/*_straddle.db`)."
)

st.caption("Pricing engine: QuantLib " + ql.__version__ + " · AnalyticEuropeanEngine (Black-Scholes-Merton).")
