"""
ALT × BTC regime study — generalised version of xrp_btc_regime_study.py.

Tag <SYMBOL> LR trades with BTC's new5 regime at entry (strict no-lookahead)
and compare per-BTC-regime stats vs the asset's own regime. Produces the per-
asset gate hypothesis for internal forward validation.

Usage:
    .venv/bin/python3 alt_btc_regime_study.py BNB
    .venv/bin/python3 alt_btc_regime_study.py DOGE
    ...

Pipeline matches xrp_btc_regime_study.py / wfo_regime_retag.py: default-params
Faithful LR signals + TradeSimulator (TP1 + ATR trail) on the full history.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
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
OUT_ROOT = _ROOT / "reports" / "btc_regime_study"
MIN_N_TRUST = 30


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
    lc = df_capitalized.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    feats = compute_features(lc[["open", "high", "low", "close", "volume"]])
    return classify_rule_based(feats)


def asof_tag(timestamps: pd.DatetimeIndex, regime: pd.Series) -> list:
    regime = regime.dropna()
    idx = regime.index.searchsorted(timestamps, side="right") - 1
    return [regime.iloc[i] if i >= 0 else pd.NA for i in idx]


def generate_trades(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"[1/4] Loading {symbol} 15m OHLCV…")
    df = load_ohlcv_capitalized(symbol, "15m")
    print(f"      {len(df):,} bars  ({df.index.min().date()} → {df.index.max().date()})")

    print("[2/4] Computing indicators…")
    df = IndicatorEngine.calculate(df)

    print(f"[3/4] Generating Faithful LR signals + simulating trades (default params)…")
    adapter = FaithfulLiquidityRaidAdapter(symbol=symbol, apply_ml=False)
    params = adapter.get_default_params()
    signals = adapter.generate_signals(df, params, 0, len(df))
    print(f"      {len(signals):,} signals")

    costs = TransactionCosts.for_asset(symbol)
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
    return (trades.dropna(subset=[col])
                  .groupby(col)
                  .agg(n=("r", "count"),
                       mean_R=("r", "mean"),
                       median_R=("r", "median"),
                       sum_R=("r", "sum"),
                       win_rate=("r", lambda x: (x > 0).mean()))
                  .round({"mean_R": 3, "median_R": 3, "sum_R": 2, "win_rate": 3})
                  .sort_values("mean_R", ascending=False))


def main():
    sym = (sys.argv[1] if len(sys.argv) > 1 else "XRP").upper()
    out = OUT_ROOT / sym.lower()
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*78}\n  {sym} × BTC regime study (cross-asset gate hypothesis)\n{'='*78}")

    trades, asset_df = generate_trades(sym)

    print("[4/4] Computing new5 regime on BTC + asset (strict no-lookahead asof)…")
    btc_df = IndicatorEngine.calculate(load_ohlcv_capitalized("BTC", "15m"))
    btc_regime = new5_regime_series(btc_df)
    asset_regime = new5_regime_series(asset_df)
    print(f"      BTC regime: {len(btc_regime):,} bars; {sym} regime: {len(asset_regime):,} bars")

    trades["btc_regime"] = asof_tag(trades.index, btc_regime)
    trades["asset_regime"] = asof_tag(trades.index, asset_regime)

    n_total = len(trades)
    net_mr = trades["r"].mean()
    net_wr = (trades["r"] > 0).mean()
    print(f"\nTotal {sym} trades: {n_total:,}  (net mR: {net_mr:+.3f}, WR: {net_wr:.1%})")

    print(f"\n=== {sym} R bucketed by BTC's new5 regime ===")
    btc_agg = aggregate(trades, "btc_regime")
    print(btc_agg.to_string())

    print(f"\n=== {sym} R bucketed by {sym}'s own new5 regime (control) ===")
    asset_agg = aggregate(trades, "asset_regime")
    print(asset_agg.to_string())

    print(f"\n=== GATE HYPOTHESIS — BTC-regime buckets with mean_R < 0 and n ≥ {MIN_N_TRUST} ===")
    losers = btc_agg[(btc_agg["mean_R"] < 0) & (btc_agg["n"] >= MIN_N_TRUST)]
    if len(losers):
        print(losers.to_string())
        print(f"→ CANDIDATE BTC-regime gate for {sym}: BLOCK when BTC ∈ {list(losers.index)}")
    else:
        print(f"(no negative-mR BTC-regime bucket at n ≥ {MIN_N_TRUST} — no gate signal for {sym})")

    # Incremental info vs marginal
    print(f"\n=== Incremental info — BTC-regime mean_R vs {sym} marginal ({net_mr:+.3f}) ===")
    inc = btc_agg[["n", "mean_R"]].copy()
    inc["vs_marginal"] = (inc["mean_R"] - net_mr).round(3)
    print(inc.to_string())

    # Persist
    trades.reset_index().to_csv(out / f"{sym.lower()}_trades_with_regimes.csv", index=False)
    btc_agg.to_csv(out / "by_btc_regime.csv")
    asset_agg.to_csv(out / f"by_{sym.lower()}_regime.csv")
    # Also write a one-row summary for the cross-asset aggregator
    pd.DataFrame([{
        "symbol": sym,
        "n_total": n_total,
        "net_mR": round(net_mr, 4),
        "net_WR": round(net_wr, 4),
        "candidate_blocked": ",".join(losers.index) if len(losers) else "",
        "n_blocked_trades": int(losers["n"].sum()) if len(losers) else 0,
        "sum_R_blocked": round(losers["sum_R"].sum(), 2) if len(losers) else 0.0,
    }]).to_csv(out / "summary.csv", index=False)
    print(f"\nSaved: {out.relative_to(_ROOT)}/")


if __name__ == "__main__":
    main()
