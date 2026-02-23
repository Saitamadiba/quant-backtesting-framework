"""KPI metric card rendering helpers."""

import streamlit as st
import pandas as pd


def kpi_row(df: pd.DataFrame, cols=None):
    """Render a row of top-level KPI metric cards from a trades DataFrame."""
    if df.empty:
        st.info("No trade data available.")
        return

    cols = cols or st.columns(4)

    total = len(df)
    wins = (df["pnl_usd"] > 0).sum()
    win_rate = wins / total if total > 0 else 0
    total_pnl = df["pnl_usd"].sum()
    avg_r = df["r_multiple"].mean() if "r_multiple" in df.columns else 0

    cols[0].metric("Total Trades", f"{total:,}",
                   help=f"Total number of completed trades in the current filter. Currently {total:,}. A higher sample size gives more statistical confidence in your other metrics.")
    cols[1].metric("Win Rate", f"{win_rate:.1%}",
                   help=f"Percentage of trades that closed in profit. Currently {win_rate:.1%}. Most trend-following strategies operate between 35-50%. Below 35% is concerning unless your winners are significantly larger than losers.")
    cols[2].metric("Total P&L", f"${total_pnl:,.2f}",
                   help=f"Sum of all trade profits and losses in USD. Currently ${total_pnl:,.2f}. This is the bottom line — positive means net profitable. Compare against drawdown to assess risk-adjusted performance.")
    cols[3].metric("Avg R-Multiple", f"{avg_r:.2f}R",
                   help=f"Average reward-to-risk ratio per trade. Currently {avg_r:.2f}R. Positive means winners outpace losers on average. Above 0.3R is healthy for most strategies. A high win rate with negative Avg R still loses money.")


def strategy_kpis(df: pd.DataFrame):
    """Render 6 strategy-specific KPIs."""
    if df.empty:
        return

    c = st.columns(6)
    total = len(df)
    wins = (df["pnl_usd"] > 0).sum()
    win_rate = wins / total if total else 0
    total_pnl = df["pnl_usd"].sum()

    gross_profit = df.loc[df["pnl_usd"] > 0, "pnl_usd"].sum()
    gross_loss = df.loc[df["pnl_usd"] < 0, "pnl_usd"].abs().sum()
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0

    avg_r = df["r_multiple"].mean() if "r_multiple" in df.columns else 0
    total_r = df["r_multiple"].sum() if "r_multiple" in df.columns else 0
    avg_dur = df["duration_minutes"].mean() if "duration_minutes" in df.columns else 0

    c[0].metric("Trades", f"{total:,}",
                help=f"Number of trades for this strategy. Currently {total:,}. Need at least 30+ trades for statistically meaningful metrics.")
    c[1].metric("Win Rate", f"{win_rate:.1%}",
                help=f"Percentage of profitable trades. Currently {win_rate:.1%}. Evaluate alongside Profit Factor — a 40% win rate can be very profitable if winners are 2-3x larger than losers.")
    c[2].metric("Total P&L", f"${total_pnl:,.2f}",
                help=f"Net dollar profit/loss for this strategy. Currently ${total_pnl:,.2f}. This is the absolute return — combine with Total R to understand risk-adjusted performance.")
    pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"
    c[3].metric("Profit Factor", pf_str,
                help=f"Gross profit divided by gross loss. Currently {pf_str}. Above 1.0 = profitable. Above 1.5 = strong edge. Above 2.0 = excellent. Below 1.0 means the strategy is losing money.")
    c[4].metric("Total R", f"{total_r:.1f}R",
                help=f"Sum of all R-multiples. Currently {total_r:.1f}R. This measures total return in units of risk. A strategy risking $100/trade with 10R total earned $1,000 risk-adjusted.")
    c[5].metric("Avg Duration", f"{avg_dur:.0f}m" if pd.notna(avg_dur) else "N/A",
                help=f"Average trade holding time in minutes. Currently {avg_dur:.0f}m. Shorter durations with positive R mean better capital efficiency. Very long trades may indicate missed exits or stale positions.")


def session_kpis(df: pd.DataFrame, session_name: str):
    """Render KPIs for a single session."""
    s = df[df["session"] == session_name]
    total = len(s)
    wins = (s["pnl_usd"] > 0).sum() if total else 0
    wr = wins / total if total else 0
    avg_r = s["r_multiple"].mean() if total and "r_multiple" in s.columns else 0
    total_r = s["r_multiple"].sum() if total and "r_multiple" in s.columns else 0

    st.metric(session_name, f"{total} trades",
              help=f"Total trades during the {session_name} session. Currently {total}. Compare across sessions to identify where your strategy is most active.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Win Rate", f"{wr:.1%}",
              help=f"Win rate during {session_name}. Currently {wr:.1%}. Session-specific win rates reveal whether market conditions during this window suit your strategy.")
    c2.metric("Avg R", f"{avg_r:.2f}",
              help=f"Average R-multiple during {session_name}. Currently {avg_r:.2f}. A negative Avg R means this session is a net drag — consider disabling your bot during this window.")
    c3.metric("Total R", f"{total_r:.1f}",
              help=f"Cumulative R-multiple for {session_name}. Currently {total_r:.1f}. The session with the highest Total R is where your edge is strongest.")
