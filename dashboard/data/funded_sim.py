"""HyroTrader funded-account simulator.

Re-runs a bot's real trade history as if it had been taken on a HyroTrader
challenge account, applying their Trial rules, and reports pass / breach /
in-progress. Pure pandas/numpy (no Streamlit) so the rule engine is unit-tested
independently of the page (see tests/test_funded_sim.py).

Sizing model (transparent + configurable):
    notional_frac = clip(risk_pct / sl_dist, min=min_notional, max=max_leverage)
    pnl_frac      = notional_frac * ret_frac      (fraction of the account)
    risk_frac     = notional_frac * sl_dist        (loss if the stop is hit)
where ret_frac/sl_dist come from PRICE geometry (entry/SL/exit/direction), not
the bots' heterogeneously-scaled logged r_multiple/pnl. The leverage cap tames a
handful of trades whose logged stop sat almost on top of the entry.

Everything is proportional to the account balance, so the rule VERDICT is the
same at every account size — only the dollar figures scale. That is itself the
useful answer: a strategy either clears the funded rules or it doesn't.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# HyroTrader account sizes the page offers.
ACCOUNT_SIZES = [5_000, 10_000, 25_000, 50_000, 100_000, 200_000, 1_000_000]


@dataclass(frozen=True)
class HyroRules:
    """HyroTrader Trial rule set (fractions of the initial balance)."""
    profit_target: float = 0.05        # +5% closed PnL to pass
    max_loss: float = 0.10             # equity must stay ≥ 90% of start
    daily_drawdown: float = 0.05       # intraday high-to-low ≤ 5%
    min_trading_days: int = 5          # ≥5 qualifying days
    max_risk_per_trade: float = 0.03   # stop must risk ≤ 3%
    min_notional: float = 0.05         # each position ≥ 5% of balance
    min_day_pnl: float = 0.01          # a day "counts" only if |PnL| ≥ 1%


HYRO_TRIAL = HyroRules()

_LONG = ("LONG", "BUY", "L")


def _is_long(direction) -> bool:
    return str(direction).strip().upper() in _LONG


def _reconstruct_exit(row) -> float:
    """Fill a missing exit price from the exit reason (SL→stop, TP→target)."""
    xp = row.get("exit_price")
    if pd.notna(xp):
        return float(xp)
    er = str(row.get("exit_reason", "")).lower()
    if any(k in er for k in ("stop", "loss", "sl")) and pd.notna(row.get("stop_loss")):
        return float(row["stop_loss"])
    if any(k in er for k in ("tp", "take", "profit", "win", "target")) and \
            pd.notna(row.get("take_profit")):
        return float(row["take_profit"])
    return np.nan


def prepare_trades(df: pd.DataFrame, r_floor: float = -2.0,
                   r_cap: float = 10.0, min_sl_dist: float = 0.0005) -> pd.DataFrame:
    """Add price-geometry columns and keep only simulatable trades.

    Output columns added: dir_sign, sl_dist, ret_frac, R, _clamped, ts
    (realization time), open_date. Rows without a usable SL distance or return
    are dropped.

    Data-quality guard: a stop-based trade can lose ~1R and win a handful of R;
    a few logged trades sit with the stop almost on the entry (or carry a bad
    exit price), yielding absurd ±50R that would otherwise dominate the account.
    We floor the stop distance at `min_sl_dist` and clamp the realized R to
    [`r_floor`, `r_cap`], flagging clamped rows so the page can disclose them.
    """
    if df.empty:
        return df.copy()
    d = df.copy()
    for c in ("entry_price", "stop_loss", "take_profit", "exit_price"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["exit_eff"] = d.apply(_reconstruct_exit, axis=1)
    d["dir_sign"] = np.where(d["direction"].map(_is_long), 1.0, -1.0)
    d["sl_dist"] = (d["entry_price"] - d["stop_loss"]).abs() / d["entry_price"]
    d["ret_frac"] = d["dir_sign"] * (d["exit_eff"] - d["entry_price"]) / d["entry_price"]
    et = pd.to_datetime(d.get("entry_time"), errors="coerce", utc=True)
    xt = pd.to_datetime(d.get("exit_time"), errors="coerce", utc=True)
    d["ts"] = xt.fillna(et)
    d["open_date"] = et.dt.date
    d = d.dropna(subset=["sl_dist", "ret_frac", "ts"])
    d = d[d["sl_dist"] > 0].copy()
    # guard: floor the stop distance, clamp R, recompute the (clean) return
    d["sl_dist"] = d["sl_dist"].clip(lower=min_sl_dist)
    R_raw = d["ret_frac"] / d["sl_dist"]
    d["R"] = R_raw.clip(r_floor, r_cap)
    d["_clamped"] = (R_raw - d["R"]).abs() > 1e-9
    d["ret_frac"] = d["R"] * d["sl_dist"]
    # optional per-trade round-trip fee (bps of notional) — used by the knife
    # maker/taker arms so the taker fee drag shows. Absent → fee-free (no change
    # to the price-only bots). Net return drives R downstream.
    if "fee_bps" in d.columns:
        d["ret_frac"] = d["ret_frac"] - pd.to_numeric(d["fee_bps"], errors="coerce")\
            .fillna(0.0) / 10000.0
    return d.sort_values("ts").reset_index(drop=True)


@dataclass
class SimResult:
    balance: float
    trades: pd.DataFrame          # per-trade ledger (sized, with equity)
    daily: pd.DataFrame           # per-day drawdown table
    rules: dict                   # rule_name -> {status, value, limit, ...}
    verdict: str                  # PASS / BREACH / IN PROGRESS
    terminal_event: str = ""
    terminal_date: object = None
    coverage: int = 0             # simulatable / total
    n_total: int = 0
    meta: dict = field(default_factory=dict)


def simulate(df_prepared: pd.DataFrame, balance: float,
             risk_pct: float = 0.01, max_leverage: float = 3.0,
             rules: HyroRules = HYRO_TRIAL) -> SimResult:
    """Run the funded simulation on already-prepared trades for one balance."""
    n_total = len(df_prepared)
    if df_prepared.empty:
        return SimResult(balance, df_prepared, pd.DataFrame(), {}, "NO DATA",
                         coverage=0, n_total=0)

    d = df_prepared.copy()
    # ── sizing ────────────────────────────────────────────────────────────
    d["notional_frac"] = np.clip(risk_pct / d["sl_dist"], rules.min_notional, max_leverage)
    d["risk_frac"] = d["notional_frac"] * d["sl_dist"]
    d["pnl_frac"] = d["notional_frac"] * d["ret_frac"]
    d["notional_$"] = d["notional_frac"] * balance
    d["risk_$"] = d["risk_frac"] * balance
    d["pnl_$"] = d["pnl_frac"] * balance
    d["cum_pnl_$"] = d["pnl_$"].cumsum()
    d["equity_$"] = balance + d["cum_pnl_$"]
    d["R"] = d["ret_frac"] / d["sl_dist"]

    target_usd = rules.profit_target * balance
    floor_usd = balance * (1 - rules.max_loss)
    dd_limit_usd = rules.daily_drawdown * balance

    # ── max loss (10%) — realized-equity floor ────────────────────────────
    below = d[d["equity_$"] < floor_usd]
    maxloss_breached = not below.empty
    maxloss_date = below["ts"].iloc[0] if maxloss_breached else None
    min_equity = float(d["equity_$"].min())

    # ── profit target (5%) ────────────────────────────────────────────────
    hit_target = d[d["cum_pnl_$"] >= target_usd]
    target_reached = not hit_target.empty
    target_date = hit_target["ts"].iloc[0] if target_reached else None
    max_profit = float(d["cum_pnl_$"].max())

    # ── daily drawdown (5%) — intraday high-to-low of realized equity ─────
    d["day"] = d["ts"].dt.date
    daily_rows = []
    prev_close = balance
    worst_dd_usd = 0.0
    dd_breach_date = None
    for day, g in d.groupby("day", sort=True):
        # path within the day = day-open equity, then equity after each trade
        path = np.concatenate([[prev_close], g["equity_$"].to_numpy()])
        hi, lo = float(path.max()), float(path.min())
        dd = hi - lo
        breached = dd > dd_limit_usd
        if breached and dd_breach_date is None:
            dd_breach_date = pd.Timestamp(day, tz="UTC")  # tz-aware to match trade ts
        worst_dd_usd = max(worst_dd_usd, dd)
        daily_rows.append({"date": day, "open_equity": prev_close,
                           "high": hi, "low": lo, "dd_$": dd,
                           "dd_pct": dd / balance, "trades": len(g),
                           "day_pnl_$": float(g["pnl_$"].sum()), "breached": breached})
        prev_close = float(g["equity_$"].iloc[-1])
    daily = pd.DataFrame(daily_rows)
    dd_breached = dd_breach_date is not None

    # ── min trading days (5) — qualifying days ────────────────────────────
    d["qualifies"] = (d["notional_frac"] >= rules.min_notional) & \
                     (d["pnl_frac"].abs() >= rules.min_day_pnl)
    qual_days = sorted(d.loc[d["qualifies"], "open_date"].dropna().unique())
    n_qual_days = len(qual_days)

    # ── stop-loss obligation (SL present + risk ≤ 3%) ─────────────────────
    sl_present = d.get("stop_loss")
    has_sl = sl_present.notna() if sl_present is not None else pd.Series(True, index=d.index)
    risk_ok = d["risk_frac"] <= rules.max_risk_per_trade + 1e-9
    sl_violations = int((~(has_sl & risk_ok)).sum())
    sl_compliant = sl_violations == 0

    # ── verdict (earliest of breach vs qualified-pass) ────────────────────
    breach_dates = [x for x in (maxloss_date, dd_breach_date) if x is not None]
    first_breach = min(pd.Timestamp(x) for x in breach_dates) if breach_dates else None
    # pass requires target AND ≥5 qual days AND sl compliance, before any breach
    pass_date = None
    if target_reached and n_qual_days >= rules.min_trading_days and sl_compliant:
        pass_date = pd.Timestamp(target_date)
    verdict, t_event, t_date = "IN PROGRESS", "", None
    if first_breach is not None and (pass_date is None or first_breach <= pass_date):
        verdict = "BREACH"
        t_event = ("Max loss (−10%)" if maxloss_date is not None and
                   pd.Timestamp(maxloss_date) == first_breach else "Daily drawdown (5%)")
        t_date = first_breach
    elif pass_date is not None:
        verdict = "PASS"
        t_event = "Profit target reached + min trading days met"
        t_date = pass_date

    rules_out = {
        "Profit target": {"status": target_reached, "value": max_profit / balance,
                          "limit": rules.profit_target, "date": target_date,
                          "fmt": "pct"},
        "Max loss": {"status": not maxloss_breached,
                     "value": (balance - min_equity) / balance,
                     "limit": rules.max_loss, "date": maxloss_date, "fmt": "pct",
                     "invert": True},
        "Daily drawdown": {"status": not dd_breached, "value": worst_dd_usd / balance,
                           "limit": rules.daily_drawdown, "date": dd_breach_date,
                           "fmt": "pct", "invert": True},
        "Min trading days": {"status": n_qual_days >= rules.min_trading_days,
                             "value": n_qual_days, "limit": rules.min_trading_days,
                             "fmt": "int"},
        "Stop-loss obligation": {"status": sl_compliant, "value": sl_violations,
                                 "limit": 0, "fmt": "int", "invert": True,
                                 "note": "positions missing SL or risking > 3%"},
    }

    return SimResult(
        balance=balance, trades=d, daily=daily, rules=rules_out,
        verdict=verdict, terminal_event=t_event, terminal_date=t_date,
        coverage=len(d), n_total=n_total,
        meta={"target_usd": target_usd, "floor_usd": floor_usd,
              "dd_limit_usd": dd_limit_usd,
              "final_equity_usd": float(d["equity_$"].iloc[-1]),
              "total_pnl_usd": float(d["cum_pnl_$"].iloc[-1]),
              "n_qual_days": n_qual_days, "risk_pct": risk_pct,
              "max_leverage": max_leverage},
    )


def compare_sizes(df_prepared: pd.DataFrame, risk_pct: float = 0.01,
                  max_leverage: float = 3.0, rules: HyroRules = HYRO_TRIAL,
                  sizes=ACCOUNT_SIZES) -> pd.DataFrame:
    """Run the sim at every account size → one row each ($ scale, same verdict)."""
    rows = []
    for bal in sizes:
        r = simulate(df_prepared, bal, risk_pct, max_leverage, rules)
        if not r.trades.empty:
            rows.append({
                "Account": f"${bal:,.0f}",
                "Final equity": r.meta["final_equity_usd"],
                "Net PnL $": r.meta["total_pnl_usd"],
                "Return %": r.meta["total_pnl_usd"] / bal,
                "Profit target $": r.meta["target_usd"],
                "Max loss limit $": bal * rules.max_loss,
                "Worst daily DD %": r.rules["Daily drawdown"]["value"],
                "Trading days": r.meta["n_qual_days"],
                "Verdict": r.verdict,
            })
    return pd.DataFrame(rows)
