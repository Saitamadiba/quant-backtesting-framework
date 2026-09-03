"""Page 24: Vol Edge — Options Lab (QuantLib + plain-English primer).

Covers BOTH volatility-selling strategies the fleet runs:
  • Vol Edge ATM straddle  (Vol_Edge/Straddle_V1/*)
  • Bull-Put credit spread (HyroTrader/bull_put_spread_bybit.py)

Sections:
  1. Primer        — what vol trading is, in plain English (weather + insurance metaphors).
  2. Side-by-side  — straddle vs bull-put, with payoff diagrams.
  3. Pricers       — QuantLib BSM straddle + BSM bull-put (tabs).
  4. Gate simulator — strategy selector drives the gate logic AND the explanations.
  5. Live state    — what's open on each bot today.

The gate simulator is the educational centrepiece: change DVOL / RV30 / cone
threshold and read the verdict + per-gate plain-English reasoning to learn
what each strategy looks for in a vol regime.
"""

from __future__ import annotations

import math
from datetime import date

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Vol Edge — Options Lab", page_icon="🎚️", layout="wide")
st.title("🎚️ Vol Edge — Options Lab")
st.caption(
    "Two strategies, one thesis: **sell overpriced insurance, manage the tail**. "
    "Vol Edge sells ATM straddles; the Bull-Put bot sells defined-risk credit spreads. "
    "Live option seats today: `straddle-btc/eth`, `bullput-btc/eth-demo`, `ironfly-btc/eth-bybit` "
    "(plus a calendar book) — their realized $ is on 🛰️ Live Fleet under **Options**. "
    "Below: a plain-English primer, payoff diagrams, both pricers, and a gate simulator "
    "that switches between strategies so you can see *why* each one waits for the regime it does."
)

try:
    import QuantLib as ql
    _QL_OK = True
except Exception:
    _QL_OK = False

if not _QL_OK:
    st.error(
        "QuantLib is not installed in this environment.\n\n"
        "Install with `pip install QuantLib`, then reload."
    )
    st.stop()

try:
    import plotly.graph_objects as go
    _PLOTLY_OK = True
except Exception:
    _PLOTLY_OK = False


# ──────────────────────────────────────────────────────────────────────────────
# QuantLib pricing helper (shared by both strategies)
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
        "theta": option.thetaPerDay(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 1. PRIMER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("🌦️ What is vol trading? (in plain English)")

st.markdown(
    """
Picture yourself running an **insurance company** that sells *storm insurance on Bitcoin*.

- Customers pay you a **premium** today, in exchange for a payout if a big move happens before the policy expires.
- If the next month is calm, you keep every dollar.
- If a real storm hits, you owe the difference.

That's the whole business of **selling volatility**. The crypto-specific name for the premium people pay for option insurance is *implied volatility* — and the question that drives every trade is the same one any insurer asks: **am I charging more than the storms I actually expect?**
"""
)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("### 🔮 DVOL")
    st.markdown(
        "**What it is:** The market's *implied* 30-day volatility for the asset (from Deribit option prices).\n\n"
        "**Metaphor:** The **weather forecast** — what option buyers *think* the next month's storms will look like."
    )
with c2:
    st.markdown("### 📜 RV30")
    st.markdown(
        "**What it is:** *Realised* 30-day volatility from actual daily price moves.\n\n"
        "**Metaphor:** What the weather **actually did** last month — the record, not the forecast."
    )
with c3:
    st.markdown("### 💸 VRP")
    st.markdown(
        "**What it is:** `DVOL − RV30`. The variance risk premium.\n\n"
        "**Metaphor:** The **insurance margin** — how much extra customers are paying above recent reality. "
        "Positive VRP = insurance is rich; negative VRP = customers got a bargain at your expense."
    )

st.info(
    "**The whole game in one sentence:** sell when VRP is rich, manage the months when the storm actually arrives — "
    "and *don't* sell into a regime where storms have been hitting back-to-back, because volatility **clusters**. "
    "(That last clause is why both bots have gates that say 'no' even when VRP looks juicy.)"
)


# ──────────────────────────────────────────────────────────────────────────────
# 2. THE TWO STRATEGIES — side by side
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("⚖️ The two strategies, side by side")

s1, s2 = st.columns(2)
with s1:
    st.subheader("🎯 Vol Edge — ATM Straddle")
    st.markdown(
        """
**What the bot does:** Sells **both** a call and a put at today's price (an at-the-money straddle).

**The bet:** *"Price will stay calm — neither up nor down by more than X% by expiry."*

**Metaphor:** Like betting the **temperature stays normal** — no heatwave **and** no cold snap. The wider the calm band, the safer for the seller.

**Where it makes money:** Price drifts inside the breakeven band → both legs decay → bot keeps the premium minus a small intrinsic give-back.

**Where it hurts:** ⚠️ **Both tails are open** — a big rally **or** a big crash loses real money. A short straddle is *uncovered* — there's no parachute.

**Bot exits:** time-decay-to-expiry (default), or risk-guard halt on a vol spike.
"""
    )

with s2:
    st.subheader("🪂 Bull-Put — Credit Spread")
    st.markdown(
        """
**What the bot does:** Sells a put at strike A, **buys** a put at a lower strike B. The lower put is the **safety net**.

**The bet:** *"Price will stay above strike A through expiry."*

**Metaphor:** Like betting the **floor doesn't drop out**. You collect a premium for the calm-side outcome, and the long lower put is your **parachute** — your worst loss is capped at `(A − B) − credit`.

**Where it makes money:** Price ≥ short strike A at expiry → keep the whole credit. Or hit a profit-target early (the bot takes 50% of credit and runs).

**Where it hurts:** Price falls through A → losses accrue, but **capped** at the wing. A genuine bearish regime can string capped losses together though — defined-risk doesn't mean win-rate-proof.

**Bot exits:** 50% profit-take, stop on `spot ≤ short_strike` touch, or expiry settlement.
"""
    )


# ──────────────────────────────────────────────────────────────────────────────
# Payoff diagrams (one chart with both strategies overlaid)
# ──────────────────────────────────────────────────────────────────────────────
if _PLOTLY_OK:
    st.markdown("### 📉 Payoff at expiry — see the shapes")

    pp1, pp2 = st.columns(2)

    # --- Straddle payoff ---
    K = 100.0; P = 8.0   # toy: strike $100, premium collected $8
    xs = [K * (1 + d / 100) for d in range(-30, 31)]
    short_straddle = [P - abs(x - K) for x in xs]
    with pp1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=short_straddle, mode="lines", name="Short straddle P&L",
                                 line=dict(width=3)))
        fig.add_hline(y=0, line_dash="dash", line_color="#888")
        fig.add_vline(x=K - P, line_dash="dot", line_color="#2ca02c", annotation_text="lower BE")
        fig.add_vline(x=K + P, line_dash="dot", line_color="#2ca02c", annotation_text="upper BE")
        fig.add_vline(x=K, line_dash="dot", line_color="#888", annotation_text="strike")
        fig.update_layout(
            title="Short straddle — both tails open",
            xaxis_title="Spot at expiry (% of strike)",
            yaxis_title="P&L per unit", height=320, margin=dict(t=40, b=40, l=40, r=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Profit *only* inside the breakeven band. Outside, losses grow with the move size — "
                   "this is the **temperature-stays-normal** bet visualised.")

    # --- Bull-put spread payoff ---
    K_short, K_wing = 100.0, 90.0   # toy: short K=100, wing K=90
    credit = 2.5
    width = K_short - K_wing
    max_loss = width - credit
    bps = []
    for x in xs:
        intr = max(K_short - x, 0.0) - max(K_wing - x, 0.0)
        bps.append(credit - intr)
    with pp2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=bps, mode="lines", name="Bull-put P&L",
                                 line=dict(width=3, color="#9467bd")))
        fig.add_hline(y=0, line_dash="dash", line_color="#888")
        fig.add_hline(y=-max_loss, line_dash="dot", line_color="#d62728",
                      annotation_text=f"max loss capped at ${max_loss:.1f}")
        fig.add_vline(x=K_short, line_dash="dot", line_color="#888", annotation_text="short K")
        fig.add_vline(x=K_wing, line_dash="dot", line_color="#1f77b4", annotation_text="wing (parachute)")
        fig.update_layout(
            title="Bull-put spread — floor capped by the wing",
            xaxis_title="Spot at expiry (% of strike)",
            yaxis_title="P&L per unit", height=320, margin=dict(t=40, b=40, l=40, r=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Profit is the full **credit** above the short strike, and the loss **flatlines** "
                   "once price reaches the long wing — the parachute opens.")


# ──────────────────────────────────────────────────────────────────────────────
# 3. PRICERS (tabs: straddle + bull-put)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("🧮 Pricers (QuantLib analytic BSM)")

with st.sidebar:
    st.subheader("Common inputs")
    asset = st.selectbox("Asset", ["BTC", "ETH"], index=0)
    default_spot = 95000.0 if asset == "BTC" else 2130.0
    spot = st.number_input("Spot price ($)", min_value=0.01, value=default_spot, step=1.0)
    dte_days = st.slider("Days to expiry", 1, 30, 14 if asset == "ETH" else 7)
    iv_pct = st.slider("Implied vol — DVOL (%)", 5.0, 150.0, 50.0, step=0.5)
    rate_pct = st.number_input("Risk-free rate (%)", value=0.0, step=0.25)

T = dte_days / 365.0
r = rate_pct / 100.0
sigma = iv_pct / 100.0

tab_straddle, tab_bps = st.tabs(["🎯 Straddle pricer", "🪂 Bull-put pricer"])

# ----- Straddle tab -----
with tab_straddle:
    a1, a2 = st.columns([1, 3])
    with a1:
        atm = st.checkbox("ATM strike (= spot)", value=True, key="strad_atm")
        strike = spot if atm else st.number_input(
            "Strike ($)", min_value=0.01, value=default_spot, step=1.0, key="strad_K"
        )

    call = price_european(spot, strike, T, r, sigma, is_call=True)
    put = price_european(spot, strike, T, r, sigma, is_call=False)
    straddle_premium = call["price"] + put["price"]
    s_delta = call["delta"] + put["delta"]
    s_gamma = call["gamma"] + put["gamma"]
    s_vega = call["vega"] + put["vega"]
    s_theta = call["theta"] + put["theta"]

    st.markdown(
        f"**ATM Straddle — {asset}**  ·  K=`${strike:,.2f}`  ·  `{dte_days}d`  ·  IV `{iv_pct:.1f}%`"
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Premium collected", f"${straddle_premium:,.2f}",
              help="Call + Put price — what the bot collects per unit when selling.")
    m2.metric("Lower breakeven", f"${strike - straddle_premium:,.2f}")
    m3.metric("Upper breakeven", f"${strike + straddle_premium:,.2f}")
    m4.metric("Breakeven band", f"±{(straddle_premium / spot * 100):.2f}%",
              help="The move (either way) that wipes out the premium. **Wider = safer for the seller.**")

    greeks_df = pd.DataFrame(
        {
            "Leg": ["Call", "Put", "Long straddle (reference)", "**Bot position (SHORT)**"],
            "Price": [call["price"], put["price"], straddle_premium, -straddle_premium],
            "Delta": [call["delta"], put["delta"], s_delta, -s_delta],
            "Gamma": [call["gamma"], put["gamma"], s_gamma, -s_gamma],
            "Vega (per 1% IV)": [call["vega"], put["vega"], s_vega, -s_vega],
            "Theta (per day)": [call["theta"], put["theta"], s_theta, -s_theta],
        }
    )
    st.dataframe(greeks_df.round(4), use_container_width=True, hide_index=True)
    st.caption(
        "The bot is **short** the straddle, so its position greeks are the *negatives* of the long row: "
        "**short vega** (profits when IV falls — the forecast cools off), **short gamma** (loses on big realised moves), "
        "**positive theta** (collects time decay every day). The whole strategy is a wager that *realised movement stays "
        "inside the breakeven band*."
    )

# ----- Bull-put tab -----
with tab_bps:
    st.markdown(f"**Bull-Put Spread — {asset}**  ·  spot `${spot:,.2f}`  ·  `{dte_days}d`  ·  IV `{iv_pct:.1f}%`")

    b1, b2, b3 = st.columns([1, 1, 1])
    with b1:
        sig_short = st.slider("Short-put distance (σ)", 0.25, 2.5, 1.00, 0.25, key="bps_short_sig",
                              help="Bot default: 1.00σ. The short strike sits one standard-deviation move BELOW spot at the given DTE.")
    with b2:
        sig_wing = st.slider("Wing-put distance (σ)", 0.5, 4.0, 2.00, 0.25, key="bps_wing_sig",
                             help="Bot default: 2.00σ. The long (parachute) put sits two σ below spot.")
    with b3:
        step = 500.0 if asset == "BTC" else 50.0
        st.metric("Strike grid", f"${step:,.0f}",
                  help="The bot snaps strikes to a grid — $500 for BTC, $50 for ETH.")

    if sig_wing <= sig_short:
        st.error("Wing must sit BELOW the short (higher σ-distance). The parachute can't be above the leak.")
    else:
        s1unit = spot * sigma * math.sqrt(T)
        K_short = round((spot - sig_short * s1unit) / step) * step
        K_wing = round((spot - sig_wing * s1unit) / step) * step

        sp = price_european(spot, K_short, T, r, sigma, is_call=False)
        wp = price_european(spot, K_wing, T, r, sigma, is_call=False)
        credit = sp["price"] - wp["price"]
        width = K_short - K_wing
        max_loss = max(width - credit, 0.0)
        be = K_short - credit
        rr = (credit / max_loss) if max_loss > 0 else float("inf")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Short strike (sell put @)", f"${K_short:,.0f}")
        m2.metric("Wing strike (buy put @)", f"${K_wing:,.0f}",
                  help="The parachute. Caps the loss.")
        m3.metric("Credit collected", f"${credit:,.2f}",
                  help="What you get paid per unit upfront.")
        m4.metric("Max loss (per unit)", f"${max_loss:,.2f}",
                  help="If spot ends ≤ wing strike at expiry, you lose this much per unit. Defined, not naked.")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Breakeven", f"${be:,.0f}",
                  help="Spot at expiry where P&L = 0. Below this you start losing; above this (up to short strike) the credit shrinks but stays positive.")
        m6.metric("Credit / max-loss (R:R)", f"{rr:.2f}",
                  help="The risk-reward per unit. Typical short-vol shape: **collect small, risk larger**, win often.")
        m7.metric("Width", f"${width:,.0f}")
        m8.metric("Credit / width", f"{(credit / width * 100):.1f}%" if width > 0 else "n/a",
                  help="What fraction of the spread's max risk you collect upfront. Higher = richer premium.")

        # Position greeks for the SHORT bull-put spread: short the short, long the wing
        pos_delta = -sp["delta"] + wp["delta"]
        pos_gamma = -sp["gamma"] + wp["gamma"]
        pos_vega = -sp["vega"] + wp["vega"]
        pos_theta = -sp["theta"] + wp["theta"]
        spread_df = pd.DataFrame({
            "Leg": [f"SHORT Put @ ${K_short:,.0f}", f"LONG Put @ ${K_wing:,.0f}",
                    "**Bull-put position (net)**"],
            "Price (per unit)": [-sp["price"], wp["price"], -credit],
            "Delta": [-sp["delta"], wp["delta"], pos_delta],
            "Gamma": [-sp["gamma"], wp["gamma"], pos_gamma],
            "Vega (per 1% IV)": [-sp["vega"], wp["vega"], pos_vega],
            "Theta (per day)": [-sp["theta"], wp["theta"], pos_theta],
        })
        st.dataframe(spread_df.round(4), use_container_width=True, hide_index=True)
        st.caption(
            "Net position is **short vega** (profits when IV falls), **collects theta**, and has **bullish-ish delta** "
            "(loses if spot drops). Note the gamma + vega are *smaller in magnitude* than a naked short put — that's "
            "the wing earning its keep: it gives up some premium to truncate the tail."
        )


# ──────────────────────────────────────────────────────────────────────────────
# 4. ENTRY-GATE SIMULATOR (the educational centrepiece)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("🎚️ Entry-gate simulator — when does the bot actually trade?")

st.markdown(
    """
Neither bot fires every day. Each waits for a specific vol regime.
Use the controls below to **dial in different conditions** and see *what each gate looks at* and *why*.
Switch strategies in the dropdown to compare how they read the same regime differently.
"""
)

strat = st.selectbox(
    "Strategy",
    ["Vol Edge — ATM straddle", "Bull-Put credit spread"],
    index=1,
    help="Each strategy has its own set of gates. Switch to compare.",
)

g1, g2 = st.columns(2)
with g1:
    sim_dvol = st.number_input(
        "DVOL (implied vol, %)", min_value=0.0, value=49.1, step=0.5, key="sim_dvol",
        help="Today's implied vol from Deribit. The forecast.",
    )
with g2:
    sim_rv30 = st.number_input(
        "RV30 (realised vol, %)", min_value=0.0, value=29.3, step=0.5, key="sim_rv30",
        help="Trailing 30-day annualised realised vol. The recent reality.",
    )

vrp = sim_dvol - sim_rv30
st.markdown(f"**Derived → VRP = DVOL − RV30 = `{vrp:+.1f}` vol-points** "
            f"({'rich premium — IV above realised' if vrp > 0 else 'cheap premium — IV below realised'})")

# ----- Vol Edge straddle gate -----
if strat.startswith("Vol Edge"):
    st.markdown("#### 🎯 Vol Edge straddle gate")
    st.caption(
        "The straddle is **uncovered on both sides**. The gate is therefore brutally simple: only sell "
        "into a *calm* IV regime where the premium is at least roughly fair vs realised."
    )

    DVOL_MAX = st.slider("DVOL ceiling (%)", 20.0, 100.0, 50.0, 1.0, key="sim_ve_dvolmax",
                         help="Bot default: 50. Anything higher means IV is screaming — that's vol-spike territory and short straddles get run over.")
    MIN_VRP = st.slider("Minimum VRP (vol-points)", -20.0, 20.0, -5.0, 0.5, key="sim_ve_vrpmin",
                        help="Bot default: −5. Even a slightly negative VRP is tolerated, but cheap IV vs realised is a red flag.")

    gate_dvol = sim_dvol < DVOL_MAX
    gate_vrp = vrp > MIN_VRP

    e1, e2 = st.columns(2)
    with e1:
        st.metric(f"DVOL < {DVOL_MAX:.0f}?  (calm enough)",
                  "✅ pass" if gate_dvol else "❌ block")
        st.caption(
            "**Why:** A straddle has no parachute. Selling into a 70-IV regime means realised could easily print 80 and you eat the difference both ways. "
            "Keep the seat-belt buckled — only sell into calmer weather."
        )
    with e2:
        st.metric(f"VRP > {MIN_VRP:+.0f}?  (premium not laughable)",
                  "✅ pass" if gate_vrp else "❌ block")
        st.caption(
            "**Why:** If implied is *below* realised, you're charging less than the past month's storms — you're the customer here, not the insurer."
        )

    if gate_dvol and gate_vrp:
        st.success(
            f"✅ **WOULD ENTER.** Calm IV regime (`{sim_dvol:.1f} < {DVOL_MAX:.0f}`) and VRP at least at floor "
            f"(`{vrp:+.1f} > {MIN_VRP:+.0f}`). The bot would sell the ATM straddle from the pricer above, "
            f"vega-sized to ≤ {'2' if asset == 'BTC' else '5'}% of equity."
        )
    else:
        reasons = []
        if not gate_dvol:
            reasons.append(f"DVOL too high (`{sim_dvol:.1f} ≥ {DVOL_MAX:.0f}`) — that's a **vol spike**, naked tails would feast")
        if not gate_vrp:
            reasons.append(f"VRP too low (`{vrp:+.1f} ≤ {MIN_VRP:+.0f}`) — **insurance is cheaper than recent reality**, no edge")
        st.warning("⏸️ **NO ENTRY** — " + "; ".join(reasons) + ".")

# ----- Bull-put gate -----
else:
    st.markdown("#### 🪂 Bull-Put gate")
    st.caption(
        "The bull-put has a parachute (the long wing), so the gate is **looser on absolute IV level** "
        "but **stricter on regime context**. Four independent checks, all must pass; each tries to catch "
        "a different way the trade could blow up."
    )

    defaults = {"ETH": dict(cone=60.8, vrp_p75=11.0, dvol_max=80.0, har_min=2.0),
                "BTC": dict(cone=48.4, vrp_p75=9.2,  dvol_max=80.0, har_min=2.0)}[asset]

    h1, h2 = st.columns(2)
    with h1:
        rv_cone = st.number_input(
            "RV-cone threshold (% — P60/P75 of trailing realised vol)",
            min_value=0.0, value=defaults["cone"], step=0.5, key="sim_bps_cone",
            help=(
                "The 60th percentile of trailing-90d RV30 for ETH (75th over 252d for BTC). "
                "This is the bot's anti-clustering guard — only sell when implied actually exceeds the **typical** "
                "recent realised level, not just today's RV30. Default = current live value."
            ),
        )
        vrp_p75 = st.number_input(
            "VRP-richness floor (% — trailing P75 of VRP)",
            min_value=-10.0, value=defaults["vrp_p75"], step=0.5, key="sim_bps_vrpp75",
            help="Bot only sells when today's VRP is in the top quartile of its own history.",
        )
    with h2:
        dvol_ceiling = st.slider("DVOL ceiling (%)", 40.0, 150.0, defaults["dvol_max"], 1.0,
                                 key="sim_bps_dvolmax", help="Tail-risk guard. Bot default: 80.")
        har_min = st.number_input("HAR-premium minimum (vol-points)",
                                  min_value=-10.0, value=defaults["har_min"], step=0.5, key="sim_bps_har",
                                  help="Forward-looking premium estimate must exceed this. Approximated as VRP > 2 in the live bot.")

    g_dvol_max = sim_dvol < dvol_ceiling
    g_cone = sim_dvol > rv_cone
    g_vrp_p75 = vrp > vrp_p75
    g_har = vrp > har_min
    all_pass = g_dvol_max and g_cone and g_vrp_p75 and g_har

    gate_rows = [
        ("Tail guard",     f"DVOL < {dvol_ceiling:.0f}", g_dvol_max,
         "**Why:** even with a wing, a 100-IV regime is shorthand for 'something is happening' — sit it out."),
        ("Anti-clustering (RV-cone)",  f"DVOL > {rv_cone:.1f}", g_cone,
         "**Why:** *the* hard-won lesson. Volatility clusters — if recent realised has been HIGH, today's IV is 'cheap vs realised' and storms keep coming. Wait for IV to actually exceed the typical recent realised before selling."),
        ("Premium-richness",   f"VRP > {vrp_p75:.1f}", g_vrp_p75,
         "**Why:** today's VRP must be in the top quartile of its own history. Only sell when *you* think the premium is fat, not just barely positive."),
        ("HAR-premium",   f"VRP > {har_min:.1f}", g_har,
         "**Why:** belt-and-braces — even the forecast-vs-forecast spread should clear a small floor."),
    ]
    rows_df = pd.DataFrame(
        {"Check": [r[0] for r in gate_rows],
         "Condition": [r[1] for r in gate_rows],
         "Verdict": ["✅ pass" if r[2] else "❌ block" for r in gate_rows]}
    )
    st.dataframe(rows_df, use_container_width=True, hide_index=True)
    for _label, _cond, _ok, _why in gate_rows:
        with st.expander(f"{'✅' if _ok else '❌'}  {_label}  —  {_cond}"):
            st.markdown(_why)

    if all_pass:
        st.success(
            f"✅ **WOULD ENTER.** DVOL `{sim_dvol:.1f}` clears the cone (`{rv_cone:.1f}`), the tail guard, "
            f"the P75 richness (`{vrp_p75:.1f}`) and the HAR floor. The bot would open the bull-put spread from "
            f"the pricer above, sized to ≤ 2% max-loss of equity."
        )
    else:
        blockers = [r[0] for r in gate_rows if not r[2]]
        st.warning(
            "⏸️ **NO ENTRY** — blocked by: **" + ", ".join(blockers) + "**. "
            "Expand the rows above to see *why* each check exists."
        )

    st.info(
        "**Lesson from the offline replay** (`an internal VRP-gate replay study`, the historical replay set): "
        "the **RV-cone gate is the value-add** — trades that pass it earn ~5× the mean R (~+0.035–0.039) of trades "
        "that fail it (~+0.003–0.007 ≈ breakeven gross). The cone's caution looks conservative — and is correct. "
        "A naïve 'VRP > 0' gate would be a *downgrade*, not an upgrade."
    )


# ──────────────────────────────────────────────────────────────────────────────
# 5. LIVE STATE
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("📡 Live state")

ls1, ls2 = st.columns(2)
with ls1:
    st.subheader("Vol Edge straddle bots")
    st.info(
        "**`straddle-btc` / `straddle-eth`** — paper-trading on the VPS. The entry gate "
        "(DVOL < 50 & VRP > −5) hasn't fired since deploy. Once they open a position, this "
        "section will show the live straddle, its greeks, and MTM P&L "
        "(synced from `Vol_Edge/Straddle_V1/*_straddle.db`)."
    )
with ls2:
    st.subheader("Bull-put bots")
    st.info(
        "**`bullput-eth-paper` / `bullput-btc-paper`** — paper-only on the VPS (funded "
        "sub-accounts can't trade options). Currently **idle**: the RV-cone gate correctly "
        "blocks because trailing-window realised vol is still higher than today's IV. The cone "
        "will fall organically as low-realised days roll in — the daily refresh cron keeps it current. "
        "Records land in `HyroTrader/bullput_{eth,btc}_shadow.db` once entries land."
    )

st.caption(
    f"Pricing engine: QuantLib {ql.__version__} · AnalyticEuropeanEngine (Black–Scholes–Merton). "
    "See **Strategy Explainer → Vol Edge** for the broader thesis, and "
    "`an internal VRP-gate replay study` for the historical gate study."
)
