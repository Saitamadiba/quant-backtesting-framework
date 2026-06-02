"""
XRP × BTC regime study — tag XRP LR trades with BTC's new5 regime at entry
(strict no-lookahead) and compare per-BTC-regime stats vs XRP's own regime.

Produces the gate hypotheses for the dual-shadow forward test (task #5):
  - identifies BTC-regime bucket(s) where XRP's edge collapses → candidate gate
  - tests whether BTC-regime adds INCREMENTAL info over XRP's own regime
    (the financial-modeler check: don't double-count what XRP's regime already
    captures)

Pipeline mirrors wfo_regime_retag.py: default-params Faithful LR signals +
TradeSimulator on the full 2018-2026 XRP history (matches the validated WFO
pipeline used in run_lr_xrp_wfo.py). No XRP-specific param tuning — the
study characterises the edge by regime, not optimises parameters.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from backtrader_framework.optimization.strategy_adapters.lr_faithful_filters import (
    FaithfulLiquidityRaidAdapter,
)
from backtrader_framework.optimization.wfo_engine import (
    IndicatorEngine, TradeSimulator, TransactionCosts,
)
from regime_classifier import compute_features, classify_rule_based

OHLCV_DB = _ROOT / "duckdb_data" / "trading_data.duckdb"
OUT = _ROOT / "reports" / "xrp_btc_regime_study"
OUT.mkdir(parents=True, exist_ok=True)
MIN_N_TRUST = 30  # min trades per regime bucket to call a verdict


def load_ohlcv_capitalized(symbol: str, timeframe: str = "15m") -> pd.DataFrame:
    con = duckdb.connect(str(OHLCV_DB), read_only=True)
    df = con.execute(f"""
        SELECT timestamp, open AS Open, high AS High, low AS Low,
               close AS Close, volume AS Volume
        FROM ohlcv_data
        WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'
        ORDER BY timestamp
    """).fetchdf()
    con.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp")


def new5_regime_series(df_capitalized: pd.DataFrame) -> pd.Series:
    """Return new5 regime label per 15m bar (regime_classifier API)."""
    lc = df_capitalized.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    feats = compute_features(lc[["open", "high", "low", "close", "volume"]])
    return classify_rule_based(feats)


def asof_tag(timestamps: pd.DatetimeIndex, regime: pd.Series) -> list:
    """Strict no-lookahead asof: regime at-or-just-before each timestamp."""
    regime = regime.dropna()
    idx = regime.index.searchsorted(timestamps, side="right") - 1
    return [regime.iloc[i] if i >= 0 else pd.NA for i in idx]


def generate_xrp_trades() -> pd.DataFrame:
    print("[1/4] Loading XRP 15m OHLCV…")
    df = load_ohlcv_capitalized("XRP", "15m")
    print(f"      {len(df):,} bars  ({df.index.min().date()} → {df.index.max().date()})")

    print("[2/4] Computing indicators…")
    df = IndicatorEngine.calculate(df)

    print("[3/4] Generating Faithful LR signals + simulating trades (default params)…")
    adapter = FaithfulLiquidityRaidAdapter(symbol="XRP", apply_ml=False)
    params = adapter.get_default_params()
    signals = adapter.generate_signals(df, params, 0, len(df))
    print(f"      {len(signals):,} signals")

    costs = TransactionCosts.for_asset("XRP")
    highs = df["High"].values; lows = df["Low"].values
    closes = df["Close"].values; atrs = df["ATR"].values
    rows = []
    for sig in signals:
        tr = TradeSimulator.simulate(sig.to_dict(), df, costs, max_bars=2688,
                                     _highs=highs, _lows=lows,
                                     _closes=closes, _atrs=atrs)
        if tr is None:
            continue
        rows.append({
            "entry_ts": sig.time, "direction": sig.direction,
            "entry": sig.entry_price, "sl": sig.stop_loss,
            "outcome": tr.outcome, "r": tr.r_multiple_after_costs,
            "bars_held": tr.bars_held,
        })
    trades = pd.DataFrame(rows).set_index("entry_ts")
    print(f"      {len(trades):,} simulated trades")
    return trades, df


def aggregate(trades: pd.DataFrame, col: str) -> pd.DataFrame:
    g = (trades.dropna(subset=[col])
                .groupby(col)
                .agg(n=("r", "count"),
                     mean_R=("r", "mean"),
                     median_R=("r", "median"),
                     sum_R=("r", "sum"),
                     win_rate=("r", lambda x: (x > 0).mean()))
                .round({"mean_R": 3, "median_R": 3, "sum_R": 2, "win_rate": 3})
                .sort_values("mean_R", ascending=False))
    return g


def crosstab_mean_r(trades: pd.DataFrame) -> pd.DataFrame:
    df = trades.dropna(subset=["btc_regime", "xrp_regime"])
    return (df.groupby(["btc_regime", "xrp_regime"])["r"]
              .agg(["count", "mean"])
              .round(3)
              .unstack("xrp_regime"))


def main():
    print(f"\n{'='*78}\n  XRP × BTC regime study (cross-asset gate hypothesis)\n{'='*78}")

    # 1. XRP trades.
    xrp_trades, xrp_df = generate_xrp_trades()

    # 2. Regimes on BTC and on XRP.
    print("[4/4] Computing new5 regime on BTC and XRP (strict no-lookahead tag)…")
    btc_df = load_ohlcv_capitalized("BTC", "15m")
    btc_df = IndicatorEngine.calculate(btc_df)
    btc_regime = new5_regime_series(btc_df)
    xrp_regime = new5_regime_series(xrp_df)
    print(f"      BTC regime series: {len(btc_regime):,} bars; "
          f"XRP regime series: {len(xrp_regime):,} bars")

    # 3. Tag — strict asof (no lookahead).
    xrp_trades["btc_regime"] = asof_tag(xrp_trades.index, btc_regime)
    xrp_trades["xrp_regime"] = asof_tag(xrp_trades.index, xrp_regime)

    n_total = len(xrp_trades)
    print(f"\nTotal XRP trades: {n_total:,}  "
          f"(net mean R: {xrp_trades['r'].mean():+.3f}, "
          f"WR: {(xrp_trades['r']>0).mean():.1%})")

    # 4. The two views.
    print("\n=== XRP-trade R bucketed by BTC's new5 regime (the cross-asset gate axis) ===")
    btc_agg = aggregate(xrp_trades, "btc_regime")
    print(btc_agg.to_string())

    print("\n=== XRP-trade R bucketed by XRP's own new5 regime (control / baseline) ===")
    xrp_agg = aggregate(xrp_trades, "xrp_regime")
    print(xrp_agg.to_string())

    # 5. Gate hypothesis.
    print("\n=== GATE HYPOTHESES (BTC-regime buckets with mean_R < 0 and n ≥ %d) ===" % MIN_N_TRUST)
    losers = btc_agg[(btc_agg["mean_R"] < 0) & (btc_agg["n"] >= MIN_N_TRUST)]
    if len(losers):
        print(losers.to_string())
        print(f"\n→ CANDIDATE BTC-regime gate: BLOCK XRP entries when BTC is in {list(losers.index)}")
    else:
        print("(no BTC-regime bucket shows negative mR at n ≥ trust threshold — "
              "no gate signal; XRP's edge is regime-uniform under BTC's frame)")

    # 6. Joint cross-tab (incremental info check).
    print("\n=== Joint BTC × XRP regime — mean R (and n) per cell ===")
    ct = crosstab_mean_r(xrp_trades)
    print(ct.to_string())

    # 7. Incremental info — for each BTC regime, is the conditional XRP-edge
    # distinct from the marginal XRP-regime stat?
    print("\n=== Incremental-info read: does BTC-regime *add* over XRP-own-regime? ===")
    print("(BTC-regime bucket mean_R vs XRP's marginal mR per bucket; large gap = "
          "BTC adds info, small gap = XRP-own subsumes it)")
    marginal = xrp_trades["r"].mean()
    inc = btc_agg[["n", "mean_R"]].copy()
    inc["vs_marginal"] = (inc["mean_R"] - marginal).round(3)
    print(inc.to_string())

    # 8. Persist.
    out_csv = OUT / "xrp_trades_with_regimes.csv"
    xrp_trades.reset_index().to_csv(out_csv, index=False)
    btc_agg.to_csv(OUT / "by_btc_regime.csv")
    xrp_agg.to_csv(OUT / "by_xrp_regime.csv")
    print(f"\nSaved: {out_csv.relative_to(_ROOT)}  + by_btc_regime.csv + by_xrp_regime.csv")


if __name__ == "__main__":
    main()
