#!/usr/bin/env python3
"""Microstructure gate study: do the continuously-recorded knife-research
feeds improve the SMC 1h→5m book?

PREREGISTERED PROTOCOL (fixed before results):
  * 14 microstructure features, each with a stated mechanism, built
    causally (feature availability = bar close; staleness caps: 2h for
    5m/15m feeds, 9h funding, 36h DVOL). Signed features multiply by
    trade side (+long/−short) so ">0 = the flow agrees with our trade".
  * Battery: per-feature quintile scan (Δ Q5−Q1, monotonicity, halves,
    AUC) on the overlap slice. Bonferroni mindset: 14 features → a
    lone p≈0.05 pattern is noise.
  * Walk-forward ridge (expanding yearly folds, gate on predicted net,
    keep 30/50% set on train): BASELINE (engine+regime features, the
    IC=+0.015 set) vs AUGMENTED (+microstructure). The OOS delta of the
    kept cohorts is the answer.
  * Interactions: only for singles passing |Δ|≥0.05 with both halves —
    top-3 × regime5, Bonferroni-corrected.
  * SUCCESS BAR: augmented keep-30 OOS net_mm > 0 AND > baseline's,
    halves same sign, ≥7/10 symbols. Context: at 5m stops the mm toll is
    ~0.9R vs +0.2R base gross — the gate needs ~5× concentration; state
    the result honestly either way.

Feeds: metrics_binance_5m (OI, top-trader LSR, taker buy/sell ratio,
2021-12→2026-06), klines_binance_15m (taker delta/CVD, 2020→),
spot_klines_bybit_15m (spot-perp basis), open_interest_5min_bybit,
funding (BTC, market-wide), dvol_vrp_BTC (market-wide).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)

from ny4h_range_reversal.analyze import feature_scan, tstat  # noqa: E402
from smc_mtf.analyze import CATS, NUM_FEATURES, add_nets  # noqa: E402

AUX = os.path.join(_BASE, "flow_aux_data")
OUT = os.path.join(_BASE, "reports", "smc_mtf")

MICRO = ["d_oi_1h_pct", "d_oi_4h_pct", "taker_bsr_sgn", "taker_bsr_1h_sgn",
         "top_lsr_sgn", "d_top_lsr_4h_sgn", "cvd_1h_sgn", "cvd_4h_sgn",
         "futvol_burst_1h", "basis_bp_sgn", "d_basis_4h_sgn",
         "d_oi_bybit_1h_pct", "btc_funding_sgn", "btc_dvol_pctile"]


def _ts(df, col, unit=None):
    s = df[col]
    if s.dtype.kind in "iu":
        return pd.to_datetime(s, unit=unit or "ms", utc=True)
    out = pd.to_datetime(s)
    return out.dt.tz_localize("UTC") if out.dt.tz is None else out.dt.tz_convert("UTC")


def build_symbol_features(sym: str) -> dict[str, pd.DataFrame]:
    """Per-symbol causal feature frames keyed by availability time."""
    out = {}
    p = f"{AUX}/metrics_binance_5m/{sym}USDT.parquet"
    if os.path.exists(p):
        m = pd.read_parquet(p)
        m["avail"] = _ts(m, "ts_utc") + pd.Timedelta(minutes=5)
        m = m.sort_values("avail")
        oi = m["sum_open_interest"].astype(float)
        m["d_oi_1h_pct"] = oi.pct_change(12)
        m["d_oi_4h_pct"] = oi.pct_change(48)
        tl = np.log(m["sum_toptrader_long_short_ratio"].astype(float).clip(0.05, 20))
        m["top_lsr_log"] = tl
        m["d_top_lsr_4h"] = tl.diff(48)
        tk = np.log(m["sum_taker_long_short_vol_ratio"].astype(float).clip(0.05, 20))
        m["taker_log"] = tk
        m["taker_log_1h"] = tk.rolling(12).mean()
        out["met"] = m[["avail", "d_oi_1h_pct", "d_oi_4h_pct", "top_lsr_log",
                        "d_top_lsr_4h", "taker_log", "taker_log_1h"]]
    p = f"{AUX}/klines_binance_15m/{sym}USDT.parquet"
    if os.path.exists(p):
        k = pd.read_parquet(p)
        k["avail"] = _ts(k, "ts_utc") + pd.Timedelta(minutes=15)
        k = k.sort_values("avail")
        v = k["volume"].astype(float)
        d = 2 * k["taker_buy_volume"].astype(float) - v
        k["cvd_1h"] = d.rolling(4).sum() / v.rolling(4).sum().replace(0, np.nan)
        k["cvd_4h"] = d.rolling(16).sum() / v.rolling(16).sum().replace(0, np.nan)
        k["futvol_burst_1h"] = (v.rolling(4).mean()
                                / v.rolling(96).mean().replace(0, np.nan))
        k["fut_close"] = k["close"].astype(float)
        out["kl"] = k[["avail", "cvd_1h", "cvd_4h", "futvol_burst_1h", "fut_close"]]
    p = f"{AUX}/spot_klines_bybit_15m/{sym}USDT.parquet"
    if os.path.exists(p) and "kl" in out:
        s = pd.read_parquet(p)
        s["avail"] = _ts(s, "ts_utc") + pd.Timedelta(minutes=15)
        s = s.sort_values("avail")[["avail", "close"]].rename(columns={"close": "spot_close"})
        b = pd.merge_asof(out["kl"], s, on="avail",
                          tolerance=pd.Timedelta("30min"), direction="backward")
        b["basis_bp"] = (b["fut_close"] / b["spot_close"].astype(float) - 1) * 1e4
        b["d_basis_4h"] = b["basis_bp"].diff(16)
        out["kl"] = b[["avail", "cvd_1h", "cvd_4h", "futvol_burst_1h",
                       "basis_bp", "d_basis_4h"]]
    p = f"{AUX}/open_interest_5min_bybit/{sym}USDT.parquet"
    if os.path.exists(p):
        ob = pd.read_parquet(p)
        ob["avail"] = _ts(ob, "ts_utc") + pd.Timedelta(minutes=5)
        ob = ob.sort_values("avail")
        ob["d_oi_bybit_1h_pct"] = ob["open_interest"].astype(float).pct_change(12)
        out["oib"] = ob[["avail", "d_oi_bybit_1h_pct"]]
    return out


def market_features() -> dict[str, pd.DataFrame]:
    out = {}
    f = pd.read_parquet("/tmp/fund_btc.parquet")
    f["avail"] = _ts(f, "funding_time_ms")
    out["fund"] = (f.sort_values("avail")
                   [["avail", "funding_rate"]].astype({"funding_rate": float}))
    d = pd.read_parquet("/tmp/dvol_btc.parquet")
    d["avail"] = _ts(d, "date") + pd.Timedelta(days=1)
    d = d.sort_values("avail")
    d["btc_dvol_pctile"] = d["dvol"].rolling(500, min_periods=100).rank(pct=True)
    out["dvol"] = d[["avail", "btc_dvol_pctile"]]
    return out


def join_features(t: pd.DataFrame) -> pd.DataFrame:
    mkt = market_features()
    parts = []
    for sym, g in t.groupby("symbol"):
        g = g.sort_values("entry_time").copy()
        fx = build_symbol_features(sym)
        def asof(frame, tol):
            return pd.merge_asof(g[["entry_time"]], frame,
                                 left_on="entry_time", right_on="avail",
                                 tolerance=pd.Timedelta(tol),
                                 direction="backward")
        if "met" in fx:
            m = asof(fx["met"], "2h")
            for c in ["d_oi_1h_pct", "d_oi_4h_pct", "top_lsr_log",
                      "d_top_lsr_4h", "taker_log", "taker_log_1h"]:
                g[c] = m[c].to_numpy()
        if "kl" in fx:
            k = asof(fx["kl"], "2h")
            for c in ["cvd_1h", "cvd_4h", "futvol_burst_1h", "basis_bp", "d_basis_4h"]:
                if c in k:
                    g[c] = k[c].to_numpy()
        if "oib" in fx:
            o = asof(fx["oib"], "2h")
            g["d_oi_bybit_1h_pct"] = o["d_oi_bybit_1h_pct"].to_numpy()
        fu = asof(mkt["fund"], "9h")
        g["funding_rate"] = fu["funding_rate"].to_numpy()
        dv = asof(mkt["dvol"], "36h")
        g["btc_dvol_pctile"] = dv["btc_dvol_pctile"].to_numpy()
        parts.append(g)
    t = pd.concat(parts, ignore_index=True)
    sgn = np.where(t["side"] == "long", 1.0, -1.0)
    t["taker_bsr_sgn"] = t.get("taker_log", np.nan) * sgn
    t["taker_bsr_1h_sgn"] = t.get("taker_log_1h", np.nan) * sgn
    t["top_lsr_sgn"] = t.get("top_lsr_log", np.nan) * sgn
    t["d_top_lsr_4h_sgn"] = t.get("d_top_lsr_4h", np.nan) * sgn
    t["cvd_1h_sgn"] = t.get("cvd_1h", np.nan) * sgn
    t["cvd_4h_sgn"] = t.get("cvd_4h", np.nan) * sgn
    t["basis_bp_sgn"] = t.get("basis_bp", np.nan) * sgn
    t["d_basis_4h_sgn"] = t.get("d_basis_4h", np.nan) * sgn
    t["btc_funding_sgn"] = t.get("funding_rate", np.nan) * sgn
    for f in MICRO:
        if f in t.columns:
            t[f] = t[f].replace([np.inf, -np.inf], np.nan)
            lo, hi = t[f].quantile(0.001), t[f].quantile(0.999)
            t[f] = t[f].clip(lo, hi)
    return t


def wf_gate(t: pd.DataFrame, feats: list[str], label: str) -> dict:
    from sklearn.linear_model import Ridge
    from scipy.stats import spearmanr

    t = t.copy()
    t["year"] = pd.to_datetime(t.entry_time).dt.year
    X = t[feats].astype(float).replace([np.inf, -np.inf], np.nan).copy()
    for c, cats in CATS.items():
        if c == "combo":
            continue
        s = t[c].astype(str)
        for cat in cats:
            X[f"{c}_{cat}"] = (s == cat).astype(float)
    years = sorted(t.year.unique())
    oos = np.zeros(len(t), bool)
    pred = np.full(len(t), np.nan)
    keeps = {k: np.zeros(len(t), bool) for k in (0.3, 0.5)}
    for yi in range(2, len(years)):
        y = years[yi]
        tr = (t.year < y).to_numpy(); te = (t.year == y).to_numpy()
        if tr.sum() < 1500 or te.sum() < 100:
            continue
        med = X[tr].median()
        Xtr, Xte = X[tr].fillna(med), X[te].fillna(med)
        mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
        m = Ridge(alpha=10.0).fit((Xtr - mu) / sd, t.gross_r.to_numpy()[tr])
        ptr = m.predict((Xtr - mu) / sd) - t.fee_r.to_numpy()[tr]
        pte = m.predict((Xte - mu) / sd) - t.fee_r.to_numpy()[te]
        oos |= te; pred[te] = pte
        for k in keeps:
            keeps[k][te] = pte >= np.quantile(ptr, 1 - k)
    g = t[oos].copy(); g["pred"] = pred[oos]
    ic = spearmanr(g.pred, g.gross_r, nan_policy="omit").statistic
    half = pd.to_datetime(g.entry_time).quantile(0.5)
    res = {"label": label, "oos_n": len(g), "ic": ic, "rows": []}
    for name, mask in [("all OOS", np.ones(len(g), bool)),
                       ("keep30", keeps[0.3][oos]), ("keep50", keeps[0.5][oos])]:
        s = g[mask]
        if len(s) < 50:
            continue
        h1 = s[pd.to_datetime(s.entry_time) <= half]
        h2 = s[pd.to_datetime(s.entry_time) > half]
        sym = s.groupby("symbol").net_mm.mean()
        res["rows"].append(
            f"| {label} {name} | {len(s)} | {s.gross_r.mean():+.3f} "
            f"| {tstat(s.gross_r):+.1f} | {s.net_r.mean():+.3f} "
            f"| {s.net_mm.mean():+.3f} | {h1.net_mm.mean():+.3f}/{h2.net_mm.mean():+.3f} "
            f"| {(sym > 0).sum()}/{len(sym)} |")
    return res


def main() -> None:
    t = add_nets(pd.read_parquet(os.path.join(OUT, "trades_all.parquet")))
    t = t[(t.symbol != "NQ") & (t.combo == "1h->5m")].copy()
    t["entry_time"] = pd.to_datetime(t.entry_time)
    t = join_features(t)
    cov = {f: float(t[f].notna().mean()) for f in MICRO}
    ov = t[t["taker_bsr_sgn"].notna()]

    lines = ["# SMC 1h→5m × recorded microstructure feeds — gate study (2026-07-17)",
             "",
             f"Book n={len(t)}; overlap with metrics feed n={len(ov)} "
             f"({ov.entry_time.min():%Y-%m-%d} → {ov.entry_time.max():%Y-%m-%d}). "
             "Protocol preregistered in script header.", "",
             "## Feature coverage", "",
             "| feature | coverage |", "|---|---|"]
    lines += [f"| {f} | {cov[f]:.0%} |" for f in MICRO]

    scan = feature_scan(ov, "gross_r", features=MICRO)
    scan.to_csv(os.path.join(OUT, "micro_feature_scan.csv"), index=False)
    lines += ["", "## Single-feature battery (overlap slice, gross basis)", "",
              "| feature | n | Q1 | Q5 | Δ | mono ρ | AUC | Δ H1 | Δ H2 | both |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in scan.iterrows():
        lines.append(f"| {r.feature} | {r.n} | {r.q1:+.3f} | {r.q5:+.3f} "
                     f"| {r.delta_q5_q1:+.3f} | {r.monotone_rho:+.2f} | {r.auc_win:.3f} "
                     f"| {r.delta_h1:+.3f} | {r.delta_h2:+.3f} "
                     f"| {'YES' if r.both_halves_same_sign else 'no'} |")

    hdr = ("| cohort | n | gross | t | net taker | net mm | mm halves | syms mm+ |\n"
           "|---|---|---|---|---|---|---|---|")
    base_feats = [f for f in NUM_FEATURES if f in ov.columns]
    lines += ["", "## Walk-forward gate: baseline vs augmented (overlap slice)", "", hdr]
    for label, feats in [("BASE", base_feats), ("AUG", base_feats + MICRO)]:
        r = wf_gate(ov, feats, label)
        lines.insert(len(lines) - 0, "")
        lines += r["rows"]
        lines += [f"", f"{label}: OOS n={r['oos_n']} IC={r['ic']:+.3f}", ""]

    # interactions only for battery survivors
    surv = scan[(scan.delta_q5_q1.abs() >= 0.05) & scan.both_halves_same_sign]
    lines += ["## Interaction check (battery survivors × regime5)", ""]
    if surv.empty:
        lines.append("No single feature passed |Δ|≥0.05 with both halves — "
                     "no interactions tested (prereg rule).")
    else:
        for f in surv.feature.head(3):
            top = ov[ov[f] >= ov[f].quantile(0.8)]
            lines += [f"### {f} top-quintile × regime5", "",
                      "| regime | n | gross | net mm |", "|---|---|---|---|"]
            for reg, gg in top.groupby("regime5"):
                lines.append(f"| {reg} | {len(gg)} | {gg.gross_r.mean():+.3f} "
                             f"| {gg.net_mm.mean():+.3f} |")
            lines.append("")

    with open(os.path.join(OUT, "micro_gate_report.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("wrote", os.path.join(OUT, "micro_gate_report.md"))


if __name__ == "__main__":
    main()
