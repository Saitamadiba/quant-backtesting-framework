"""Page 23: Shadow Trades — practice vs theory, per gate.

Every signal the LR live path rejects via a gate is recorded as a paper
trade in a per-bot ``<sym>_shadow_trades.db``.  Each shadow carries a
``block_reason`` so we can confront live R-multiples per gate against the
WFO claim that justified it.

Hybrid paper-bot extension (2026-05-31): the 7 no-Deribit-data LR paper bots
(XRP + BNB/DOGE/AVAX/LINK/DOT/BCH) route off-window signals here too, with
``block_reason='OFF_WINDOW'``, plus ``BTC_REGIME_BLOCK`` when the BTC-regime
gate fires.  This page now supports per-bot filtering, BTC-regime asof tagging
(via duckdb_data/trading_data.duckdb), per-session / per-regime / per-MTF
breakdowns, and a what-if RR analysis (replay each trade against klines at
reward 1.5R / 2R / 3R, fixed 1R risk).

Gates currently shadowed:

* ``OFF_WINDOW`` — signal generated outside the live 12-16 ET window.
* ``BTC_REGIME_BLOCK`` — BTC was in a losing regime for the asset.
* ``IV_GATE_{LOW|MED|HIGH}`` — DVOL bucket on the IV-block list (mostly LR ETH).
* ``LONDON_SHORTS_BAN`` — London-High SHORT (hardcoded 0%-WR rule on BTC/SOL).
* ``REGIME_BLOCK_<regime>`` — strategy disabled in regime (e.g. NQ ranging).
* ``COUNTER_TREND_LONG_IN_TRENDING_DOWN`` / ``..._SHORT_IN_TRENDING_UP``.
* ``SUPPRESSION_SHORT`` — SHORT in gamma SUPPRESSION.

Shadow-fleet gap closure (2026-07-06): the page now also carries every OTHER
shadow/paper vehicle running on the VPS that keeps a per-row trade book —
the depth/LRR/OFCS paper-execution books, the depth exit-policy counterfactual
book, the Momentum-4H / iFVG / Asia-basket / MM-partial-exit signal shadows,
the bull-put options paper books and the FVG funded-context sims.  Schema
bridges live in ``data/shadow_normalisers.SHADOW_DB_SPECS``; the "Shadow
fleet registry" section documents each vehicle's parameters and the logic
behind them.  Knife arms + knife counterfactual shadows stay on page 28.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    SHADOW_DB_STRATEGY_MAP,
    VPS_CACHE_DIR,
    VPS_SHADOW_DB_FILES,
    DUCKDB_PATH,
)
from data.vps_sync import sync_single_file
from data.shadow_normalisers import (
    table_for as _table_for,
    normaliser_for as _normaliser_for,
)


st.set_page_config(page_title="Shadow Trades", page_icon="🌑", layout="wide")
st.title("🌑 Shadow Trades — practice vs theory")

st.markdown(
    "Every signal the LR live path **rejects** via a gate is tracked here as a "
    "paper trade.  Compare the **live R-multiple per gate** to the WFO claim "
    "that put the gate there.  When a row's `total R` disagrees with the cited "
    "WFO finding for more than a couple of weeks, the gate is the suspect."
)


# ══════════════════════════════════════════════════════════════════════════════
#  Shadow fleet registry — every shadow/paper vehicle running on the VPS,
#  its parameters, and the logic behind them (gap closure 2026-07-06).
#  Vehicles with a per-row trade book also appear in the tabs/KPIs below;
#  "registry-only" entries (episode counterfactuals, recorders, monitors)
#  are documented here and live on their own pages/CLIs.
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("🛰️ Shadow fleet registry — what runs, with what dials, and why")
st.caption(
    "One entry per shadow/paper vehicle on the VPS.  Each pairs the exact "
    "parameters with the reasoning that put them there — the number first, "
    "the picture as a chaser.  Vehicles marked **[tab]** feed the trade "
    "tables below; **[page 28]** = knife family, documented on the Knife "
    "Bots page; **[registry-only]** = no per-trade book to tabulate here."
)

_FLEET = {
    "Paper-execution books (fee-modeled forward tests)": [
        ("Depth Paper Bot — `depth_paper_book.db` [tab]",
         "**Runs:** trader cron :06/:21/:36/:51 (`depth_paper_bot.py`). "
         "**Params:** fee haircut **0.10R** round-trip (`r_net = r_gross − 0.10`); "
         "risk **0.5%** of a **$10k** paper equity per trade; per-key open cap "
         "**999** (deliberately uncapped); allow-list **MM×12 / LR×11 (ex-LTC) / "
         "LRR×9 (ex-DOT/ADA/XRP) = 32 keys**; watermark start **2026-07-02** "
         "(no backfill — forward evidence only); source feed stale > **6h** → "
         "fail-closed (halts, doesn't trade).\n\n"
         "**Why:** the un-gated depth-cohort detectors print ~+0.24R on "
         "idealized fills — this book replays the same signals minus a "
         "realistic fee toll, forward-only.  It's the dress rehearsal before "
         "any funded routing: idealized-minus-fee is a *proof step*, not "
         "validated alpha, so nothing graduates until the fee-net edge holds "
         "at n≥200 over 2–4 weeks.  The cap is off because we're measuring "
         "the edge, not managing drawdown."),
        ("LRR Paper Bot — `lrr_paper_book.db` [tab]",
         "**Runs:** trader cron :04/:19/:34/:49 (`lrr_paper_bot.py`). "
         "**Params:** assets **BTC/ETH/SOL** only; **REQUIRE_ML=1** — consumes "
         "only `ml_pass=1` signals from `lrr_shadow_trades.db` (read-only "
         "feed); fee **0.10R**; risk **0.5% / $10k**; caps **3 open total, 1 "
         "per asset**; watermark **signal id 217** (no backfill); source "
         "stale > 3h → fail-closed.\n\n"
         "**Why:** the 2026-07-01 replay showed the LRR ML gate roughly "
         "doubles mean R (+0.087 → +0.209) and is the difference between "
         "fee-dead and fee-survivable — but the bootstrap CI still straddles "
         "zero and 3 of 10 symbols invert.  So the gate earns a paper book, "
         "not a live switch: let the forward tape vote before believing the "
         "backtest.  BTC/ETH/SOL because those are where the gate was robust; "
         "the caps mimic how a funded arm would actually run."),
        ("OFCS Paper Bot — `ofcs_paper_book.db` [tab]",
         "**Runs:** `ofcs_shadow` timer/cron (`ofcs_shadow/paper_bot.py`). "
         "**Params:** order-flow-conditioned sizing — per-signal risk "
         "(`ofcs_risk_pct`) scaled by **absorption tier × cross-count × ML-p** "
         "multipliers, booked side-by-side with a **flat-risk control** "
         "(`flat_risk_pct`) on the same trades; exits TP/SL/TIME.\n\n"
         "**Why:** tests sizing-as-a-dimmer — same trades, different dose — "
         "against the null that a light switch (flat size) is all you need.  "
         "The 2026-07-02 fleet review scored the overlay **Δ−0.041R vs flat "
         "on every arm** → NOT promoted; the book keeps accruing as the "
         "kill-confirmation.  Keeping a refuted idea's meter running is "
         "cheap insurance against re-litigating it from memory."),
        ("Depth Demo Executors (maker → taker) — `depth_{maker,taker}_book.db` [registry-only]",
         "**Runs:** trader cron :11/:26/:41/:56 (`depth_maker_bot.py` under "
         "`depth_taker.env` — TAKER mode, fresh book 2026-07-06); the MAKER "
         "arm is **RETIRED** (cron commented 2026-07-06), its book kept as "
         "the adverse-selection record.  **Params (taker redesign):** "
         "fixed-**$10k** book sizing via a composite guard (**$50/trade**, "
         "was $812 off the $161k demo equity); **LRR dropped** from the "
         "allow-list; demo orders on ByBit, not paper walks (table=orders, "
         "so no tab here — it's an executor, like the knife arms on p28).\n\n"
         "**Why:** the 2026-07-05 maker-vs-taker review found the maker arm "
         "was adverse selection in action — the limit orders that DIDN'T "
         "fill were the +1.46R winners, the fills averaged −0.17R.  A resting "
         "limit order is a free option you hand the market: it gets exercised "
         "against you exactly when you're wrong.  The taker redesign pays "
         "the crossing fee to keep the whole signal population."),
        ("Depth Policy Shadow — `depth_policy_book.db` [tab]",
         "**Runs:** trader cron :12/:27/:42/:57 (`depth_policy_shadow.py`). "
         "**Params:** hard stop **−0.40R**; trail arms at **+1.50R**, distance "
         "**0.30R**; TP **+2.00R**; **no re-entry**; fee **0.10R**; families "
         "**MM + LR only** (LRR excluded — the WFO said the policy *hurts* "
         "it); pre-registration freeze **2026-07-04 13:00Z**; 24h walk cap "
         "per trade.  Scores each booked paper trade counterfactually: "
         "`native_net` vs `policy_net`.\n\n"
         "**Why:** the exit-policy WFO cleared 3/3 time windows and 12/12 "
         "assets at fee-adjusted **Δ+0.35/+0.34R**, cutting −0.9R tails "
         "16→0.  Tighter-stop-plus-wider-TP is the grid's verdict (winners "
         "run 8–9R median MFE; the old symmetric exits sold them early).  "
         "But a WFO win is theory — this book is the practice leg, frozen "
         "before it started so the forward window can't be cherry-picked.  "
         "The paper bot itself stays untouched: policy is scored *beside* "
         "the book, never inside it."),
    ],
    "Signal shadows (record-only detectors — structurally cannot order)": [
        ("Momentum 4H — SOL & LTC shadows — `momentum_4h_{sol,ltc}_shadow.db` [tab]",
         "**Runs:** systemd `momentum-4h-sol-shadow` / `-ltc-shadow`. "
         "**Params:** EMA **50/200** trend bias, **ADX(14) ≥ 20**, 20-bar "
         "volume average, breakout entry, SL **1.5×ATR**, TP **5.0×ATR** "
         "(Phase 6.5 + WFO re-validation moved TP from 3.0); public "
         "kline-fetch only — the modules do not import an order client "
         "(AST-tested).\n\n"
         "**Why:** both cleared plain significance but not the family-wise "
         "Bonferroni bar (p̄=0.9985 across 33 cells): SOL is the strongest "
         "unproven cell (net **+0.744R, MC P=0.964, n=12**), LTC sits at "
         "**P=0.979, n=30**.  A ~4-fires/year detector simply hasn't had "
         "enough at-bats, so we let the season play out — record-only until "
         "n grows and MC P clears the harder curve.  Promotion is an edit "
         "to the registry file, never to the shadow."),
        ("Momentum XRP 1H-cascade shadow — `momentum_4h_xrp_shadow.db` [tab]",
         "**Runs:** systemd `momentum-4h-xrp-shadow`. **Params:** detects on "
         "**1H** bars, shadows a signal only when its direction agrees with "
         "the **4H EMA50/200 trend bias** (the validated bias-gated cascade); "
         "same SL 1.5×ATR / TP 5.0×ATR geometry; read-only klines.\n\n"
         "**Why:** the strongest non-4H candidate — **+0.345R net, MC "
         "P=0.998, 11/16 windows positive, n=92** — missing the 0.9985 "
         "Bonferroni bar *by a hair*.  It fires ~3× as often as the 4H "
         "cells, so the shadow reaches a verdict-quality sample fastest.  "
         "Uncertified until the larger-n MC clears the bar."),
        ("iFVG Sweep Shadow (NQ signal_gap) — `ifvg_sweep_shadow.db` [tab]",
         "**Runs:** systemd `ifvg-sweep-shadow` (`ifvg_sweep_shadow.py`), plus "
         "the earlier NQ writer's `ifvg_nq_signal_shadow.db`. **Params:** "
         "**NQ only**, `signal_gap` mode — the FVG **inversion itself is the "
         "trigger** (MSS + sweep prerequisite gates dropped), stop anchored "
         "to the gap, **0.5-ATR** minimum strength filter; cells **1h/4h/6h** "
         "with hold caps 672/168/168 bars; yfinance NQ=F, no order client.\n\n"
         "**Why:** the full 13-asset × 7-timeframe × 4-mode WFO+MC study "
         "found the iFVG edge is NQ-shaped: crypto never clears Bonferroni "
         "at any timeframe, while NQ signal_gap is robust across 1h–1d "
         "(**MC P=1.0000, WR 66–68%**, peaking ~4–6h).  The shadow forward- "
         "tests exactly the surviving cells and nothing else — promotion is "
         "decided by the scorecard script, never inside the bot."),
        ("iFVG Construction-A (BTC/ETH) — `ifvg_shadow/state.jsonl` [registry-only]",
         "**Runs:** trader cron 00:30 (`python3 -m ifvg_shadow.run`). "
         "**Params:** 15m BTC + ETH; fade a tap INTO an unfilled FVG zone; "
         "only entries whose bar is in the **vol_expansion** regime; "
         "TP/SL/**24h timeout**; R recorded net of cost; state is one "
         "JSONL, not a SQLite book (hence no tab).\n\n"
         "**Why:** the regime study's one surviving crypto variant, worth "
         "a cheap daily logger vs its **+0.057R** backtest expectation — "
         "a pilot light, not a burner: if the forward scorecard can't beat "
         "a number that small, the variant dies quietly."),
        ("Asia-Basket Shadow — `asia_basket_shadow.db` [tab]",
         "**Runs:** systemd `asia-basket-shadow` (`asia_basket_shadow.py`, "
         "poll 15 min). **Params:** **1.0% total basket risk per Asia "
         "night** → night return = 1% × mean(per-leg R); legs read "
         "READ-ONLY from the 7 LR paper bots' shadow DBs; **BTC 5-night "
         "trailing-bleed regime gate** (basket stands down when BTC's "
         "trailing Asia performance ≤ 0); DST-safe night keys; no order "
         "surface (AST-tested).\n\n"
         "**Why:** the pivot from risking the eval account — same bounded- "
         "basket math as the demo executor, run as an observer at funded-"
         "$10k scale.  One row = one night, because the 1% cap makes the "
         "night, not the leg, the unit of risk.  It accumulates the "
         "n≥100-nights evidence the WFO discipline demands before any real "
         "execution."),
        ("MM BTC Partial-Exit Shadow — `btc_momentum_mastery_v2_shadow.db` [tab]",
         "**Runs:** systemd `mm-btc-shadow` (full MM v2 bot in shadow). "
         "**Params:** identical to live MM BTC except **PARTIAL_EXIT_ENABLED="
         "True**: exit **half at 50% of the TP distance**, move SL to "
         "break-even on the runner.\n\n"
         "**Why:** the partial-exit replay says taking half early cuts tail "
         "losses at modest mean-R cost — but replays assume fills.  Running "
         "the whole bot twice (live config vs partial config) on the same "
         "tape is the cleanest A/B: same signals, same data, one dial "
         "changed.  Flip the live `PARTIAL_EXIT_ENABLED` only if the shadow "
         "book wins on the forward tape."),
    ],
    "Options paper + funded-context sims": [
        ("Bull-Put Spread Paper (BTC & ETH) — `bullput_{btc,eth}_shadow.db` [tab]",
         "**Runs:** systemd `bullput-btc-paper` / `bullput-eth-paper` "
         "(`bull_put_spread_bybit.py`, PAPER mode — no ByBit options "
         "account).  **Params:** **14-DTE** spreads; short strike at "
         "**1.0σ** below spot, wing at **2.0σ**; profit-take at **50% of "
         "credit**; max loss **2%** per spread; R defined as "
         "`realized_pnl / max_loss`.\n\n"
         "**Why:** the Vol-Edge income leg — selling the put wing when VRP "
         "is rich is collecting rent on insurance others overpay for, with "
         "the wing capping the flood risk.  Paper because there's no "
         "options account wired yet; the book builds the DVOL/VRP-"
         "conditioned evidence before any capital sees an options venue."),
        ("FVG Funded-Context Sims (BTC & NQ) — `fvg_{btc,nq}_funded_shadow.db` [tab]",
         "**Runs:** sidecar sims fed by the FVG bots' signals. **Params:** "
         "replays each FVG trade under the **funded-challenge risk state "
         "machine** — halt states, `is_qualifying`, consecutive-loss "
         "counter, daily-PnL limits, cycle-day — with commission booked "
         "separately; R = `realized_pnl / risk_amount`.\n\n"
         "**Why:** a strategy that's +EV standalone can still fail a "
         "challenge whose drawdown rules cut it off mid-recovery.  This is "
         "the dress rehearsal with the examiner's stopwatch running: it "
         "measures FVG *under the rules*, not in the void, before an FVG "
         "arm is ever pointed at a funded account."),
    ],
    "Knife family — see page 28 🔪 (documented here for completeness)": [
        ("Knife arms + counterfactual shadows [page 28]",
         "**Live arms:** forward-shadow detector (`knife_detector_shadow`), "
         "MAKER challenge / $100k / maker2 arms + TAKER L2-absorption arm.  "
         "**Counterfactual shadows (cron, episode-level):** "
         "`knife_mae_shadow` (**−0.5R MAE stop**; promotion trigger n≥100, "
         "Δ≥+0.10R, tail-cut ≥70% — a survival tool, meanR-neutral under "
         "realistic slip), `knife_winstate_shadow` (**first-120s Tier-A "
         "shape**, 2-state hold-vs-cut manage loop, freeze 06-30), "
         "`knife_touches_shadow` (**tch24 over-tested-wall gate** — the "
         "10+-touches cohort is negative inside +EV books, 23/23 symbols), "
         "`knife_widestop_shadow`, `knife_oi_skip_shadow` (OI gate skip "
         "ledger), `knife_continuation_shadow`, `knife_ofcs_shadow`, "
         "`knife_crossvenue_shadow` (Binance-led / lockstep / ahead break "
         "tags), `knife_trail_lock_tracker` (0.9R trail pilot), "
         "`knife_vol_floor_shadow` + `knife_floor_forward` (weekly vol-floor "
         "scoreboards).\n\n"
         "**Why:** the knife entry is structurally −EV on the firehose "
         "(−0.43R ride), so every one of these is an *exit/armor* "
         "experiment, not an alpha hunt — each isolates ONE lever (MAE "
         "stop, trail, touches gate, OI skip…) and scores it against the "
         "same episodes.  They are armor being fitted to a fighter who "
         "keeps losing on points — documented and compared on the Knife "
         "Bots page, kept out of these tabs so the per-gate LR/MM view "
         "stays clean."),
    ],
    "Recorders & monitors (no hypothetical trades — data collection)": [
        ("Depth-Touches Recorder — `depth_touches.db` [registry-only]",
         "**Runs:** trader cron 03:20Z (`depth_touches_recorder.py`, "
         "record-only sidecar — the paper bot is untouched). **Params:** "
         "computes `lvl_touches` per booked paper trade via the canonical "
         "helper; scores **skip-if-touches≥10** on FEE-MODELED `r_net`; "
         "families **MM + LR** (LRR research-null 0.495); freeze "
         "**2026-07-05**; promotion bar: pooled n≥**300**, skipped n≥**20**, "
         "kept meanR ≥ **+0.05** AND skipped ≤ **−0.05**, same-sign per "
         "family.\n\n"
         "**Why:** the fresh-extreme touches gate is the strongest effect "
         "in the whole program — a level hit 10+ times is a sea-wall the "
         "tide has already breached; fading it stops working.  This is "
         "rung 1 of the two-surface ladder: record on the paper surface "
         "first, gate only after both surfaces agree."),
        ("Market-data recorders [registry-only]",
         "`depth-logger` (Bybit L2 + tape, no creds), `oi_1m_recorder` "
         "(1-min open interest), `binance_alt_recorder` (cross-venue "
         "bookTicker), `liq_binance_recorder` (forceOrder liquidations), "
         "`skew_logger` / Deribit vol collector (DVOL + 25Δ skew), spot "
         "tape logger.  **Why:** every refuted pre-fill hypothesis taught "
         "us the ceiling is data, not modeling — these keep the raw feeds "
         "flowing so future pre-registered probes replay history instead "
         "of waiting for it.  Cameras on the intersection, not traffic "
         "cops."),
        ("Monitors [registry-only]",
         "`ct_short_shadow_monitor` (daily 06:30 — watches the LR counter-"
         "trend SHORT carve-out cohort), `lr_funnel_heartbeat` (daily 00:20 "
         "— did every LR stage fire yesterday?), `knife_recap` daily/weekly "
         "Telegram recaps, `displacement-btc` Phase-2 shadow + "
         "`displacement-bybit` $100k-demo executor (Displacement momentum "
         "family, own DBs).  **Why:** read-only tripwires — they change "
         "nothing, they just make silence loud."),
    ],
}

with st.expander("📇 Open the fleet registry (every vehicle, params + why)",
                 expanded=False):
    _n_vehicles = sum(len(v) for v in _FLEET.values())
    st.caption(f"{_n_vehicles} entries across {len(_FLEET)} groups. "
               "Last reconciled against the live VPS crontab + systemd "
               "unit list: **2026-07-06**.")
    for _group, _entries in _FLEET.items():
        st.markdown(f"#### {_group}")
        for _title, _body in _entries:
            with st.container(border=True):
                st.markdown(f"**{_title}**")
                st.markdown(_body)


# ── Sync controls (sidebar) ──────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Sync")
    if st.button("⟳ Sync shadow DBs from VPS", use_container_width=True):
        results: dict[str, dict] = {}
        with st.spinner("Syncing shadow DBs..."):
            for local_name, remote_path in VPS_SHADOW_DB_FILES.items():
                results[local_name] = sync_single_file(local_name, remote_path)
        ok = sum(1 for r in results.values() if r.get("status") == "ok")
        st.success(f"Synced {ok}/{len(results)}")
        for name, r in results.items():
            if r.get("status") != "ok":
                st.write(f"❌ {name}: {r.get('status')} {r.get('error','')}")
    st.caption(
        "Shadow DBs live next to each bot's trade DB on the VPS.  "
        "Sync pulls the latest snapshot into `dashboard/databases/`."
    )


# ── Loaders ──────────────────────────────────────────────────────────────────
# Normalisers live in `data/shadow_normalisers.py` so the schema-bridge
# contracts can be regression-tested independently of the page.

def _load_one(local_name: str) -> pd.DataFrame:
    p = VPS_CACHE_DIR / local_name
    if not p.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(p) as conn:
            # Table + schema bridge resolved per-file via the dispatch map
            # in data/shadow_normalisers.SHADOW_DB_SPECS.
            df = pd.read_sql_query(f"SELECT * FROM {_table_for(local_name)}",
                                   conn)
            _norm = _normaliser_for(local_name)
            if _norm is not None:
                df = _norm(df)
    except Exception as e:
        st.warning(f"Could not read {local_name}: {e}")
        return pd.DataFrame()
    if df.empty:
        return df
    strat, sym = SHADOW_DB_STRATEGY_MAP.get(local_name, ("?", "?"))
    df["strategy"] = strat
    _abbr = "".join(w[0] for w in strat.split()) if strat != "?" else "?"
    if sym == "MULTI" and "family" in df.columns and "asset" in df.columns:
        # Multi-asset book that ALSO tags which detector family produced
        # each row (depth paper/policy books: MM/LR/LRR) — fold the family
        # into the bot label so "DP MM BTC" and "DP LR BTC" stay separate.
        df["symbol"] = df["asset"].astype(str)
        df["bot"] = df.apply(
            lambda r: f"{_abbr} {r['family']} {r['asset']}", axis=1)
    elif sym == "MULTI" and "asset" in df.columns:
        # Multi-asset DB: per-row symbol. For ``manual_trades`` the bot
        # label also reflects the strategy_tag the sync auto-tagger chose
        # (e.g. "M FVG BTC", "M LR BTC", "M ad-hoc BTC") so the bot filter
        # in the sidebar can drill down to the source strategy or to
        # truly ad-hoc trades.
        df["symbol"] = df["asset"].astype(str)
        if local_name == "manual_trades.db" and "strategy_tag" in df.columns:
            def _mk_bot(row: pd.Series) -> str:
                tag = str(row.get("strategy_tag") or "ad-hoc")
                # Strip trailing asset tail from the tag if present
                # (e.g. "FVG BTC" → "FVG"); keeps the bot label compact.
                tag_short = tag.replace(f" {row['asset']}", "").strip()
                return f"{_abbr} {tag_short} {row['asset']}".strip()
            df["bot"] = df.apply(_mk_bot, axis=1)
        else:
            df["bot"] = df["symbol"].apply(lambda a: f"{_abbr} {a}")
    else:
        df["symbol"] = sym
        df["bot"] = f"{_abbr} {sym}"
    for c in ("opened_at_utc", "closed_at_utc"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    return df


def _collapse_setups(df: pd.DataFrame, gap_min: float = 60.0) -> pd.DataFrame:
    """Collapse re-detections of one swept level into a single DISTINCT SETUP.

    The live LR path re-emits the same unconsumed sweep on every 15m bar (and
    from scratch after each bot restart), so historically one setup wrote 4-12
    near-identical rows — inflating every per-gate ``n`` 4-12x and, where a
    setup won, dragging mean-R toward that one outcome counted many times (BTC
    raw +0.217 → +-0.217 once deduped).  This keeps one row per real setup so
    R / WR / n reflect distinct trade ideas, not poll cadence.

    Resolution order per LR row:
      1. ``setup_id`` (written by the dedup_shadow_setups backfill) when present.
      2. else chain ``(symbol, block_reason, sweep_type, direction)`` on time —
         consecutive rows whose gap <= ``gap_min`` are the same setup.
    Rows the post-fix tracker writes are already one-per-setup → no-op.  LRR
    rows (own UNIQUE-constraint dedup, no ``block_reason``) pass straight through.
    """
    if df.empty or "opened_at_utc" not in df.columns:
        return df
    df = df.copy()
    has_reason = "block_reason" in df.columns
    is_lr = df["block_reason"].notna() if has_reason else pd.Series(False, index=df.index)
    lr, rest = df[is_lr].copy(), df[~is_lr]
    if lr.empty:
        return df
    lr = lr.sort_values("opened_at_utc")
    keys = [k for k in ("symbol", "block_reason", "sweep_type", "direction")
            if k in lr.columns]
    # 1) backfilled setup_id (per-DB int) — exact agreement with the VPS DBs.
    if "setup_id" in lr.columns and lr["setup_id"].notna().any():
        lr["_setup"] = lr["setup_id"]
        # rows the backfill missed (NULL — e.g. written post-backfill) fall back
        # to a per-row unique id so each counts as its own setup.
        _miss = lr["_setup"].isna()
        if _miss.any() and "shadow_id" in lr.columns:
            lr.loc[_miss, "_setup"] = "sid:" + lr.loc[_miss, "shadow_id"].astype(str)
        grp = ["symbol", "_setup"] if "symbol" in lr.columns else ["_setup"]
    else:
        # 2) gap-cluster fallback (stale local copies that pre-date the backfill).
        # Sort by key then time so same-key rows are contiguous; a new setup
        # starts whenever the key changes OR the gap to the prior same-key row
        # exceeds gap_min.  cumsum yields a globally-unique setup id.
        lr = lr.sort_values(keys + ["opened_at_utc"])
        _same_key = (lr[keys] == lr[keys].shift()).all(axis=1)
        _dt = lr["opened_at_utc"].diff().dt.total_seconds() / 60.0
        lr["_setup"] = (~_same_key | _dt.isna() | (_dt > gap_min)).cumsum()
        grp = ["_setup"]
    sort_col = "shadow_id" if "shadow_id" in lr.columns else "opened_at_utc"
    lr = (lr.sort_values(sort_col)
            .drop_duplicates(subset=grp, keep="first")
            .drop(columns=["_setup"]))
    return pd.concat([lr, rest], ignore_index=True)


frames = [_load_one(name) for name in VPS_SHADOW_DB_FILES]
frames = [f for f in frames if not f.empty]
all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# ── Distinct-setup collapse (default ON) ─────────────────────────────────────
# One swept level == one setup.  Off shows every raw re-detection (inflated n).
with st.sidebar:
    st.divider()
    _collapse = st.checkbox(
        "Collapse re-detections → 1 row per setup", value=True,
        help="The live LR path re-logs the same unconsumed sweep every 15m bar "
             "and after every restart. ON counts each setup once (trustworthy "
             "R / WR / n); OFF shows the raw, duplicate-inflated rows.",
    )
if _collapse and not all_df.empty:
    _n_raw = len(all_df)
    all_df = _collapse_setups(all_df)
    _n_dropped = _n_raw - len(all_df)
    if _n_dropped > 0:
        st.sidebar.caption(
            f"Collapsed {_n_raw} raw rows → {len(all_df)} setups "
            f"({_n_dropped} re-detections hidden)."
        )


# ── BTC regime asof tagging (cached) ─────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _btc_regime_series_from_duckdb() -> pd.Series:
    """Return a pd.Series indexed by UTC timestamp with the BTC new5 regime
    label per 15m bar.  Uses the shared regime_classifier so the labels match
    the bots' BTCRegimeMonitor exactly."""
    try:
        import duckdb
        con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        df = con.execute("""
            SELECT timestamp, open AS Open, high AS High, low AS Low,
                   close AS Close, volume AS Volume
            FROM ohlcv_data
            WHERE symbol = 'BTC' AND timeframe = '15m'
            ORDER BY timestamp
        """).fetchdf()
        con.close()
    except Exception as e:
        st.warning(f"BTC regime tagging unavailable (duckdb read failed: {e})")
        return pd.Series(dtype="object")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    try:
        import sys, pathlib
        # repo root sits two levels up from dashboard/pages/
        _root = pathlib.Path(__file__).resolve().parents[2]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        from regime_classifier import compute_features, classify_rule_based
        lc = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                "Close": "close", "Volume": "volume"})
        feats = compute_features(lc[["open", "high", "low", "close", "volume"]])
        labels = classify_rule_based(feats).dropna()
        return labels
    except Exception as e:
        st.warning(f"regime_classifier not available: {e}")
        return pd.Series(dtype="object")


def _asof_tag_btc_regime(ts: pd.Series) -> pd.Series:
    btc = _btc_regime_series_from_duckdb()
    if btc.empty:
        return pd.Series([pd.NA] * len(ts), index=ts.index, dtype="object")
    # ensure tz-aware UTC for both sides
    ts_utc = pd.to_datetime(ts, utc=True, errors="coerce")
    idx = btc.index.searchsorted(ts_utc.fillna(btc.index.min()), side="right") - 1
    out = pd.Series(
        np.where(idx >= 0, btc.iloc[np.clip(idx, 0, None)].to_numpy(), pd.NA),
        index=ts.index, dtype="object",
    )
    out[ts_utc.isna()] = pd.NA
    return out


if not all_df.empty and "opened_at_utc" in all_df.columns:
    all_df["btc_regime"] = _asof_tag_btc_regime(all_df["opened_at_utc"])


# ── Sidebar bot filter ───────────────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.subheader("Filter")
    bot_options = ["All bots"] + (
        sorted(all_df["bot"].unique().tolist()) if not all_df.empty else []
    )
    sel_bot = st.selectbox("Bot", bot_options, index=0)


view_df = all_df if sel_bot == "All bots" else all_df[all_df["bot"] == sel_bot]


# ── Cache status row ─────────────────────────────────────────────────────────
st.subheader("Cache status")
status_cols = st.columns(min(len(VPS_SHADOW_DB_FILES), 8))
for i, (name, (strat, sym)) in enumerate(SHADOW_DB_STRATEGY_MAP.items()):
    p = VPS_CACHE_DIR / name
    _abbr = "".join(w[0] for w in strat.split()) if strat != "?" else "?"
    # Sentinel "MULTI" = single DB holds multiple assets. Render as "(multi)"
    # in the cache-status badge; per-asset rows in the body still split correctly.
    _sym_label = "(multi)" if sym == "MULTI" else sym
    col = status_cols[i % len(status_cols)]
    if p.exists():
        size_kb = round(p.stat().st_size / 1024, 1)
        age_min = int((datetime.now().timestamp() - p.stat().st_mtime) / 60)
        col.success(f"{_abbr} {_sym_label} · {size_kb} KB · {age_min} min", icon="🟢")
    else:
        col.error(f"{_abbr} {_sym_label} · not synced", icon="🔴")


if view_df.empty:
    st.info(
        f"No shadow data for **{sel_bot}**.  Click **⟳ Sync shadow DBs from VPS** "
        "in the sidebar, or pick a different bot."
    )
    st.stop()


# ── Data quality panel ───────────────────────────────────────────────────────
# Per-DB row count + column population %. Shows whether None cells are due
# to (a) the DB being empty (sync failure / bot silent), (b) the column not
# existing for that schema, or (c) historical rows pre-dating the column.

with st.expander("🔍 Data quality — per-source row counts + column population"):
    _qcols = ["block_reason", "session", "regime_gate", "mtf_score",
              "exit_reason", "r_multiple", "pnl_at_1pct"]
    _q_rows = []
    for _name, (_strat, _sym) in SHADOW_DB_STRATEGY_MAP.items():
        _p = VPS_CACHE_DIR / _name
        _row = {"source": _name, "strategy": _strat,
                "synced": "yes" if _p.exists() else "no",
                "rows": 0}
        if _p.exists():
            try:
                with sqlite3.connect(_p) as _c:
                    _tbl = _table_for(_name)
                    _n = _c.execute(f"SELECT COUNT(*) FROM {_tbl}").fetchone()[0]
                    _row["rows"] = _n
                    _info = {r[1] for r in
                             _c.execute(f"PRAGMA table_info({_tbl})").fetchall()}
                    for _col in _qcols:
                        if _col not in _info:
                            _row[_col] = "n/a (column absent)"
                        elif _n == 0:
                            _row[_col] = "—"
                        else:
                            _filled = _c.execute(
                                f"SELECT COUNT({_col}) FROM {_tbl}"
                            ).fetchone()[0]
                            _row[_col] = f"{int(100*_filled/_n)}%"
            except Exception as _e:
                _row["rows"] = f"read error: {_e}"
        _q_rows.append(_row)
    _qdf = pd.DataFrame(_q_rows)
    st.dataframe(_qdf, use_container_width=True, hide_index=True)
    st.caption(
        "**Reading the panel:** `n/a (column absent)` = the column doesn't "
        "exist in that schema (e.g. LRR has `regime` not `regime_gate`, "
        "manual_trades has none of the gate-related fields). A percentage "
        "= column exists but only that fraction of rows have it filled "
        "(pre-2026-05-23 LR rows pre-date the regime-gate logger and show "
        "NaN by design). `—` = source is empty (0 rows). `synced=no` = "
        "click ⟳ Sync in the sidebar."
    )


# ── Summary helper ───────────────────────────────────────────────────────────
def _summarise(df: pd.DataFrame, group_col: Optional[str] = None) -> pd.DataFrame:
    if df.empty:
        return df
    closed_mask = df["exit_reason"].isin(["TP", "SL", "TIME_EXIT"])
    closed = df[closed_mask]
    open_df = df[df["exit_reason"] == "OPEN"]
    rows = []
    # group_col may be missing entirely (e.g. LRR-only filter has no
    # block_reason) — surface that to the caller rather than silently
    # returning a malformed frame.
    if group_col is not None and group_col not in df.columns:
        return pd.DataFrame()
    if group_col is None:
        iter_ = [("All", closed, open_df)]
    else:
        keys = sorted(set(df[group_col].dropna().astype(str).unique()))
        iter_ = [
            (k, closed[closed[group_col].astype(str) == k],
                 open_df[open_df[group_col].astype(str) == k])
            for k in keys
        ]
    for k, g, og in iter_:
        n = len(g)
        tp = int((g["exit_reason"] == "TP").sum())
        sl = int((g["exit_reason"] == "SL").sum())
        te = int((g["exit_reason"] == "TIME_EXIT").sum())
        wr_tp = round(tp / n * 100, 1) if n else 0.0
        # Use NaN (rendered as "—" by Streamlit) instead of 0 when there are
        # no closed rows — avoids the misleading "0.000 / 0.00" cells.
        _has_r = n and "r_multiple" in g.columns and g["r_multiple"].notna().any()
        avg_r = round(g["r_multiple"].mean(), 3) if _has_r else float("nan")
        tot_r = round(g["r_multiple"].sum(), 2) if _has_r else float("nan")
        _has_pnl = (n and "pnl_at_1pct" in g.columns
                    and g["pnl_at_1pct"].notna().any())
        tot_pnl_1pct = (round(g["pnl_at_1pct"].sum(), 2)
                        if _has_pnl else float("nan"))
        rows.append({
            (group_col or "scope"): k,
            "open": len(og),
            "closed": n,
            "TP": tp,
            "SL": sl,
            "TIME_EXIT": te,
            "TP %": wr_tp,
            "avg R": avg_r,
            "total R": tot_r,
            "total $@1%": tot_pnl_1pct,
        })
    return pd.DataFrame(rows)


# ── Headline KPIs (respects bot filter) ──────────────────────────────────────
overall = _summarise(view_df)
k1, k2, k3, k4 = st.columns(4)
ov = overall.iloc[0]
k1.metric("Open shadows", int(ov["open"]))
k2.metric("Closed shadows", int(ov["closed"]))
k3.metric("Avg R", f"{ov['avg R']:+.3f}" if ov["avg R"] is not None else "—")
k4.metric("Total R", f"{ov['total R']:+.2f}")

st.markdown("---")


# ── Per-bot (only meaningful when not already filtered to one) ──────────────
if sel_bot == "All bots":
    st.subheader("Per bot")
    per_bot = _summarise(view_df, group_col="bot").rename(columns={"bot": "Bot"})
    st.dataframe(per_bot, use_container_width=True, hide_index=True)


# ── Per gate (the headline practice-vs-theory view) ──────────────────────────
st.subheader("By gate (`block_reason`) — *practice vs theory*")
st.caption(
    "Each row is a gate's cumulative live shadow performance.  Compare to "
    "the WFO claim that justified the gate in the strategy code."
)
per_reason = (
    _summarise(view_df, group_col="block_reason")
    .rename(columns={"block_reason": "Block reason"})
    .sort_values("closed", ascending=False)
)
st.dataframe(per_reason, use_container_width=True, hide_index=True)


# ── Cumulative R per gate (chart) ────────────────────────────────────────────
closed_chart = view_df[view_df["exit_reason"].isin(["TP", "SL", "TIME_EXIT"])].copy()
if not closed_chart.empty and "r_multiple" in closed_chart.columns:
    closed_chart = closed_chart.dropna(subset=["r_multiple", "closed_at_utc"])
    if not closed_chart.empty:
        closed_chart = closed_chart.sort_values("closed_at_utc")
        closed_chart["cum_R"] = (
            closed_chart.groupby("block_reason")["r_multiple"].cumsum()
        )
        fig = px.line(
            closed_chart, x="closed_at_utc", y="cum_R",
            color="block_reason",
            title=f"Cumulative R per gate — {sel_bot}",
            labels={"closed_at_utc": "closed at (UTC)",
                    "cum_R": "cumulative R", "block_reason": "gate"},
        )
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)


# ── Breakdown tabs (session / asset regime / BTC regime / MTF band) ──────────
st.subheader("Breakdowns")
st.caption(
    "Slice the shadow book by entry session, asset's own new5 regime, BTC's "
    "new5 regime at entry (asof from duckdb), and MTF score band.  Each row "
    "reports closed-trade stats only."
)

# Pre-compute MTF band
if "mtf_score" in view_df.columns:
    view_df = view_df.copy()
    view_df["mtf_band"] = pd.cut(
        view_df["mtf_score"],
        bins=[-0.001, 40, 55, 70, 85, 100],
        labels=["<40", "40-55", "55-70", "70-85", "85-100"],
    ).astype(str)

tab_sess, tab_arg, tab_btc, tab_mtf = st.tabs([
    "Session", "Asset regime", "BTC regime", "MTF band",
])
with tab_sess:
    if "session" in view_df.columns:
        st.dataframe(
            _summarise(view_df, group_col="session")
            .rename(columns={"session": "Session"})
            .sort_values("closed", ascending=False),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No `session` column in this shadow DB.")
with tab_arg:
    if "regime_gate" in view_df.columns:
        st.dataframe(
            _summarise(view_df, group_col="regime_gate")
            .rename(columns={"regime_gate": "Asset regime"})
            .sort_values("closed", ascending=False),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No `regime_gate` column in this shadow DB.")
with tab_btc:
    if "btc_regime" in view_df.columns:
        st.dataframe(
            _summarise(view_df, group_col="btc_regime")
            .rename(columns={"btc_regime": "BTC regime"})
            .sort_values("closed", ascending=False),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "BTC regime computed via `regime_classifier.classify_rule_based` on "
            "BTC 15m from `duckdb_data/trading_data.duckdb`, asof-tagged to each "
            "trade's `opened_at_utc` (strict no-lookahead)."
        )
    else:
        st.info("BTC regime tagging is unavailable (duckdb not loadable).")
with tab_mtf:
    if "mtf_band" in view_df.columns:
        st.dataframe(
            _summarise(view_df, group_col="mtf_band")
            .rename(columns={"mtf_band": "MTF band"})
            .sort_values("closed", ascending=False),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No `mtf_score` column in this shadow DB.")


# (internal forward-shadow tracker; sanitised from public commit history)

# ── RR What-If (replay against klines, fixed 1R risk, vary reward) ──────────
st.markdown("---")
st.subheader("RR what-if — same SL, different rewards")
st.caption(
    "For each closed shadow trade, replay forward against the asset's 15m "
    "klines (duckdb) with **fixed 1R risk** (the original SL) and three "
    "candidate rewards (**1.5R / 2R / 3R**).  Whichever the price path hits "
    "first wins — SL → −1R, target → +reward, neither within 2688 bars "
    "(~28 days) → TIME_EXIT at the last close.  Lets you see whether your "
    "current TP setting is leaving money on the table for each bucket."
)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_klines(symbol: str) -> pd.DataFrame:
    try:
        import duckdb
        con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        df = con.execute(f"""
            SELECT timestamp, high, low, close
            FROM ohlcv_data
            WHERE symbol = '{symbol}' AND timeframe = '15m'
            ORDER BY timestamp
        """).fetchdf()
        con.close()
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp")


def _replay_one(klines: pd.DataFrame, opened_at, entry: float, sl: float,
                direction: str, target_r: float, max_bars: int = 2688):
    """Return (outcome_r, outcome_reason) for a hypothetical trade with the
    given SL and target_r (multiple of risk).  outcome_reason ∈ {'TP','SL','TIME_EXIT'}."""
    if klines.empty or pd.isna(opened_at) or pd.isna(entry) or pd.isna(sl):
        return None, None
    risk = abs(entry - sl)
    if risk <= 0:
        return None, None
    is_long = str(direction).upper() == "BUY"
    target = entry + target_r * risk if is_long else entry - target_r * risk
    # walk forward from the first bar STRICTLY AFTER opened_at
    idx = klines.index.searchsorted(opened_at, side="right")
    end = min(idx + max_bars, len(klines))
    if idx >= end:
        return None, None
    sub = klines.iloc[idx:end]
    highs = sub["high"].to_numpy()
    lows = sub["low"].to_numpy()
    for i in range(len(sub)):
        if is_long:
            tp_hit = highs[i] >= target
            sl_hit = lows[i] <= sl
        else:
            tp_hit = lows[i] <= target
            sl_hit = highs[i] >= sl
        # Conservative tie-break: if BOTH hit on the same bar, assume SL first
        # (worst case for the trader — matches the WFO simulator convention).
        if sl_hit and not tp_hit:
            return -1.0, "SL"
        if tp_hit and not sl_hit:
            return float(target_r), "TP"
        if tp_hit and sl_hit:
            return -1.0, "SL"
    return 0.0, "TIME_EXIT"  # didn't hit either within max_bars


@st.cache_data(ttl=1800, show_spinner=False)
def _rr_whatif(view_records: tuple, target_rs: tuple, max_bars: int = 2688) -> pd.DataFrame:
    """Compute outcomes per reward level for the supplied trades.

    view_records: tuple of (bot, symbol, opened_at_utc, direction, entry_price,
                            stop_loss, exit_reason) — hashable for caching.
    """
    rows = []
    # Lazy klines per symbol
    klines_cache: dict[str, pd.DataFrame] = {}
    for bot, sym, opened_at, direction, entry, sl, exit_reason in view_records:
        if exit_reason not in ("TP", "SL", "TIME_EXIT"):
            continue
        if sym not in klines_cache:
            klines_cache[sym] = _load_klines(sym)
        kl = klines_cache[sym]
        if kl.empty:
            continue
        opened_at_dt = pd.to_datetime(opened_at, utc=True)
        for tr in target_rs:
            r_val, reason = _replay_one(kl, opened_at_dt, entry, sl, direction, tr, max_bars)
            if r_val is None:
                continue
            rows.append({
                "bot": bot, "target_R": tr,
                "outcome_R": r_val, "outcome": reason,
            })
    return pd.DataFrame(rows)


target_rs = (1.5, 2.0, 3.0)
closed = view_df[view_df["exit_reason"].isin(["TP", "SL", "TIME_EXIT"])].copy()
records = []
for _, row in closed.iterrows():
    records.append((
        row.get("bot"), row.get("symbol"), row.get("opened_at_utc"),
        row.get("direction"), row.get("entry_price"),
        row.get("stop_loss"), row.get("exit_reason"),
    ))

if not records:
    st.info("No closed shadow trades yet for the selected bot — nothing to replay.")
else:
    with st.spinner(f"Replaying {len(records)} trades against duckdb klines…"):
        whatif = _rr_whatif(tuple(records), target_rs)

    if whatif.empty:
        st.warning(
            "Replay produced no rows.  Likely cause: the asset's klines aren't "
            "in `trading_data.duckdb` (only BTC/ETH/SOL/NQ/XRP/LTC/ADA + 6 majors are)."
        )
    else:
        # Aggregate per target_R
        agg = (
            whatif.groupby("target_R")
            .agg(
                n=("outcome_R", "count"),
                wins=("outcome", lambda x: (x == "TP").sum()),
                losses=("outcome", lambda x: (x == "SL").sum()),
                time_exits=("outcome", lambda x: (x == "TIME_EXIT").sum()),
                mean_R=("outcome_R", "mean"),
                total_R=("outcome_R", "sum"),
            )
            .round({"mean_R": 3, "total_R": 2})
        )
        agg["WR %"] = (agg["wins"] / agg["n"] * 100).round(1)
        agg = agg.reset_index()
        agg["target_R"] = agg["target_R"].map(lambda v: f"{v:g}R")
        agg = agg.rename(columns={"target_R": "Reward (R)"})
        # Also include the bot's ACTUAL outcome as a reference row
        actual = pd.DataFrame([{
            "Reward (R)": "actual",
            "n": int(closed["r_multiple"].notna().sum()),
            "wins": int((closed["exit_reason"] == "TP").sum()),
            "losses": int((closed["exit_reason"] == "SL").sum()),
            "time_exits": int((closed["exit_reason"] == "TIME_EXIT").sum()),
            "mean_R": round(closed["r_multiple"].mean(), 3) if not closed.empty else None,
            "total_R": round(closed["r_multiple"].sum(), 2) if not closed.empty else 0.0,
            "WR %": round((closed["exit_reason"] == "TP").mean() * 100, 1) if not closed.empty else 0.0,
        }])
        combined = pd.concat([actual, agg], ignore_index=True)[
            ["Reward (R)", "n", "wins", "losses", "time_exits", "WR %", "mean_R", "total_R"]
        ]
        st.dataframe(combined, use_container_width=True, hide_index=True)
        st.caption(
            "**Reading the table:** the **actual** row is what each bot actually "
            "booked.  The 1.5/2/3R rows are the simulated outcomes if you'd kept "
            "the same SL but moved TP — same trades, same entries, just a "
            "different target.  Higher reward earns more per win but trades "
            "fewer wins for more time-exits."
        )


# ── Shadow trades browser — full pagination, per-column filters, reorder ────
st.subheader("Shadow trades — full browser")
st.caption(
    "Every row in the synced shadow DBs, grouped by strategy. **All rows** "
    "are shown (no row cap). Use the **🔍 Column filters** expander to "
    "narrow down by `bot` / `direction` / `exit_reason` / `block_reason` / "
    "anything categorical. Use the **📋 Columns to show** picker to control "
    "which columns appear AND the order (selection order = display order). "
    "Click any column header for native sort."
)

# Column lists per strategy — derived from the actual writer schema.
# Each list is ordered by importance for that strategy's reader.
_COMMON_HEAD = ["bot", "opened_at_utc", "direction", "exit_reason",
                "r_multiple"]
_COMMON_TAIL = ["entry_price", "stop_loss", "take_profit", "exit_price",
                "bars_held", "btc_regime"]

_STRATEGY_COLS = {
    "Liquidity Raid": _COMMON_HEAD + [
        "block_reason", "session", "regime_gate", "mtf_score",
        "pnl_at_1pct",
    ] + _COMMON_TAIL,
    "LR Paper":       _COMMON_HEAD + [
        "block_reason", "session", "regime_gate", "mtf_score",
        "pnl_at_1pct",
    ] + _COMMON_TAIL,
    "Momentum Mastery": _COMMON_HEAD + [
        "block_reason", "session", "regime_gate", "mtf_score",
        "pnl_at_1pct",
    ] + _COMMON_TAIL,
    "LRR Shadow": _COMMON_HEAD + [
        "vol_ratio", "wick_ratio", "prior_move_pct", "hour_et",
        "is_best_combo", "mtf_score", "regime",
        "dvol_band", "vrp_bucket", "htf_agree", "htf_trend",
        "liquidity_class", "cross_count", "ml_p", "ml_pass",
        "entry_price", "sl", "tp", "exit_price", "bars_held",
        "btc_regime",
    ],
    "Manual":     _COMMON_HEAD + [
        "symbol", "strategy_tag",
        "entry_price", "exit_price", "btc_regime",
    ],
    # ── Shadow-fleet gap closure (2026-07-06) ────────────────────────────────
    "Depth Paper": _COMMON_HEAD + [
        "family", "r_gross", "r_net", "pnl_usd", "risk_usd",
        "equity_before",
    ] + _COMMON_TAIL,
    "LRR Paper": _COMMON_HEAD + [
        "ml_p", "r_gross", "r_net", "pnl_usd", "risk_usd",
        "equity_before",
    ] + _COMMON_TAIL,
    "OFCS Paper": _COMMON_HEAD + [
        "regime_gate", "absorption_tier", "cell", "conditioning",
        "ml_p", "cross_count", "ofcs_risk_pct", "flat_risk_pct",
        "gate_authorised",
    ] + _COMMON_TAIL,
    "Depth Exit Policy": _COMMON_HEAD + [
        "family", "policy_exit", "policy_r", "policy_net",
        "native_reason", "native_r", "native_net", "policy_taker",
        "entry_price", "stop_loss", "take_profit", "btc_regime",
    ],
    "Momentum 4H": _COMMON_HEAD + [
        "sl_dist", "tp_atr",
    ] + _COMMON_TAIL,
    "iFVG Shadow": _COMMON_HEAD + [
        "symbol", "timeframe", "mode", "rr_target", "sl_anchor",
        "session_gate", "et_hour",
    ] + _COMMON_TAIL,
    "Asia Basket": [
        "bot", "opened_at_utc", "exit_reason", "r_multiple",
        "n_legs", "n_assets", "closed_legs", "night_ret_pct",
        "worst_leg_r", "regime_on", "btc_trailing_pct", "btc_regime",
    ],
    "MM Partial-Exit": _COMMON_HEAD + [
        "partial_tp", "partial_exit_done", "killzone", "sweep_type",
        "rr_ratio", "confidence", "regime_gate", "realized_pnl",
    ] + _COMMON_TAIL,
    "Bull-Put Paper": [
        "bot", "opened_at_utc", "exit_reason", "r_multiple",
        "realized_pnl", "short_strike", "wing_strike", "credit",
        "max_loss", "spot_entry", "n_units", "dvol", "vrp",
        "live_gate_passed", "btc_regime",
    ],
    "FVG Funded Shadow": _COMMON_HEAD + [
        "realized_pnl", "risk_amount", "halt_state", "is_qualifying",
        "cycle_day", "consec_losses_after", "daily_pnl_after",
        "entry_price", "stop_loss", "take_profit", "exit_price",
        "btc_regime",
    ],
}

# Display friendly hyphens for NaN / None in dataframe cells.
def _render(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # Use object dtype + fillna() so visible cells render as "—" rather than
    # NaN/None. Numeric columns keep their numeric dtype for the rendered
    # frame because Streamlit handles NaN -> blank gracefully in the live UI;
    # we only stringify text-like columns here.
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].where(out[c].notna(), "—")
    return out


_strategies_present = (view_df["strategy"].dropna().unique().tolist()
                       if "strategy" in view_df.columns else [])
_tab_labels = [s for s in
               ("Liquidity Raid", "LR Paper", "Momentum Mastery",
                "LRR Shadow", "Manual",
                "Depth Paper", "LRR Paper", "OFCS Paper", "Depth Exit Policy",
                "Momentum 4H", "iFVG Shadow", "Asia Basket",
                "MM Partial-Exit", "Bull-Put Paper", "FVG Funded Shadow")
               if s in _strategies_present]


from dashboard.data.shadow_normalisers import candidate_filter_columns as \
    _candidate_filter_columns  # noqa: E402  (defined here to keep tab scope)


def _render_strategy_table(strat: str, sub: pd.DataFrame) -> None:
    """Render the strategy's tab content: filters + column picker + table."""
    if sub.empty:
        st.info(f"No rows for {strat}.")
        return

    # Strategy-specific column list, intersected with what's actually present
    # AND that has at least one non-NaN value (honest-NaN policy from
    # 2026-06-06 cleanup — see internal note).
    base_cols = [c for c in _STRATEGY_COLS.get(strat, _COMMON_HEAD)
                 if c in sub.columns and sub[c].notna().any()]
    if not base_cols:
        st.info(f"All columns are 100% empty for {strat}.")
        return

    # ── 1. Per-column filters (expander, default collapsed) ────────────────
    filter_cols = _candidate_filter_columns(sub, base_cols)
    _filt_key_prefix = f"sb_v2_{strat.replace(' ', '_')}"
    with st.expander(f"🔍 Column filters ({len(filter_cols)} available)",
                     expanded=False):
        if not filter_cols:
            st.caption("No filterable columns in this strategy's view.")
        else:
            st.caption(
                "Each filter starts with **all values selected** (no filter "
                "applied). Deselect values to narrow down. "
                "Filters are AND-combined across columns. Click "
                "**Reset filters** to start over."
            )
            # Reset button — clears every per-column session_state key for
            # this tab. Streamlit re-runs the script after a button click so
            # the multiselects below pick up the cleared keys.
            if st.button("Reset filters", key=f"{_filt_key_prefix}_reset"):
                for c in filter_cols:
                    _k = f"{_filt_key_prefix}_filt_{c}"
                    if _k in st.session_state:
                        del st.session_state[_k]
            # Layout filters in a 3-column grid so a strategy with many
            # categorical columns (LRR has ~10) stays scannable.
            ncols = 3
            for i in range(0, len(filter_cols), ncols):
                row = st.columns(ncols)
                for j, col in enumerate(filter_cols[i:i+ncols]):
                    with row[j]:
                        # Use string conversion so multiselect can deal with
                        # mixed-dtype columns (NaN + int + str).
                        opts = sorted(
                            sub[col].dropna().astype(str).unique().tolist()
                        )
                        sel = st.multiselect(
                            col, opts, default=opts,
                            key=f"{_filt_key_prefix}_filt_{col}",
                        )
                        if sel and set(sel) != set(opts):
                            sub = sub[sub[col].astype(str).isin(sel)]

    if sub.empty:
        st.warning("No rows match the current filter combination.")
        return

    # ── 2. Column picker — controls visibility AND order ────────────────────
    # Streamlit's multiselect returns options in the order the user selects
    # them. We treat that as the column display order. Default = base_cols
    # which is the curated importance order from _STRATEGY_COLS.
    st.markdown(
        "**📋 Columns to show** — selection order = display order. "
        "Deselect any column to hide it; reorder by deselecting then re-selecting."
    )
    show_cols = st.multiselect(
        "Columns to show",
        options=base_cols,
        default=base_cols,
        key=f"{_filt_key_prefix}_cols",
        label_visibility="collapsed",
    )
    if not show_cols:
        st.warning(
            "Select at least one column to display "
            "(or use the legend above to add columns)."
        )
        return

    # ── 3. Sort key — default by opened_at_utc desc; user picks otherwise ──
    sort_options = ["(no sort — preserve insertion order)"] + show_cols
    _default_sort = ("opened_at_utc" if "opened_at_utc" in show_cols
                     else sort_options[0])
    sort_col = st.selectbox(
        "Sort by", sort_options,
        index=sort_options.index(_default_sort),
        key=f"{_filt_key_prefix}_sortcol",
    )
    sort_asc = st.checkbox(
        "Ascending", value=False,
        key=f"{_filt_key_prefix}_sortasc",
        help="(Or click the column header in the table for native sort.)",
    )

    # ── 4. Final assembly — ALL ROWS (no head() cap) ────────────────────────
    table = sub.copy()
    if sort_col != "(no sort — preserve insertion order)":
        table = table.sort_values(sort_col, ascending=sort_asc,
                                   na_position="last")
    table = table.loc[:, show_cols]

    st.caption(
        f"**Showing {len(table):,} rows.** "
        f"Click column headers for native sort. "
        f"Drag column edges to resize. Sort/filter/reorder is persisted "
        f"in the Streamlit session — refresh resets to defaults."
    )
    # Tall height so the user can scroll through hundreds of rows without
    # the page jumping. Streamlit handles internal virtualisation.
    st.dataframe(
        _render(table),
        use_container_width=True,
        hide_index=True,
        height=600,
    )

    # ── 5. Strategy-specific caption — honest-NaN explainer ─────────────────
    if strat == "LRR Shadow":
        st.caption(
            "LRR rows have no `block_reason` / `session` / `pnl_at_1pct` "
            "columns — the LRR scanner doesn't apply a gate, doesn't "
            "track a session label, and the DB stores R-only outcomes."
        )
    elif strat == "Manual":
        st.caption(
            "Manual trades have no `block_reason` (nothing rejected "
            "them), no `session` (not enforced), no `r_multiple` "
            "(manual trades don't carry an SL/TP in the broker DB)."
        )
    elif strat in ("Liquidity Raid", "LR Paper"):
        _rg_pct = int(100 * sub["regime_gate"].notna().mean()) \
                  if "regime_gate" in sub.columns else 0
        st.caption(
            f"`regime_gate` is populated on {_rg_pct}% of {strat} rows. "
            "Pre-2026-05-23 rows pre-date the regime-gate logger "
            "(written by `the internal strategy core`) and "
            "show NaN by design."
        )
    elif strat in ("Depth Paper", "LRR Paper"):
        st.caption(
            "`r_multiple` = **r_net** (gross R minus the 0.10R round-trip "
            "fee haircut) — the promotion decision runs on the net number. "
            "`family` = which detector family produced the signal "
            "(MM/LR/LRR). See the fleet registry above for parameters."
        )
    elif strat == "Depth Exit Policy":
        st.caption(
            "Counterfactual book: every row is a paper trade re-scored "
            "under the candidate exit policy (hard −0.40R / trail@1.5R "
            "d0.3 / TP 2.0R). `r_multiple` = **policy_net**; compare to "
            "`native_net` on the same row. `exit_reason` maps HARD→SL, "
            "TRAIL→TIME_EXIT — the native `policy_exit` column keeps the "
            "honest split."
        )
    elif strat == "Asia Basket":
        st.caption(
            "One row = one Asia **night** of the bounded basket, not one "
            "trade. `r_multiple` = the night's mean per-leg R; "
            "`night_ret_pct` = 1% cap × that mean. Nights close by the "
            "clock, so resolved nights read TIME_EXIT by construction."
        )
    elif strat == "Bull-Put Paper":
        st.caption(
            "Options credit spreads: `r_multiple` = realized_pnl / "
            "max_loss (risking 1 unit = the spread's max loss). "
            "TP = the 50%-of-credit profit-take; SL = short strike touched."
        )
    elif strat == "FVG Funded Shadow":
        st.caption(
            "Funded-context sim: FVG trades replayed under the challenge "
            "risk state machine (halt states, daily-PnL, consec-loss "
            "counter). `r_multiple` = realized_pnl / risk_amount, net of "
            "commission."
        )
    elif strat == "MM Partial-Exit":
        st.caption(
            "Full MM v2 bot in shadow with PARTIAL_EXIT on (half off at "
            "50% of TP distance, SL→BE). `r_multiple` is reconstructed as "
            "realized_pnl / (|entry−stop| × size) — the blended-trade R, "
            "since partial exits split the position."
        )


if _tab_labels:
    _tabs = st.tabs([f"{s} ({len(view_df[view_df['strategy']==s])})"
                    for s in _tab_labels])
    for tab, strat in zip(_tabs, _tab_labels):
        with tab:
            _render_strategy_table(strat,
                                    view_df[view_df["strategy"] == strat])
else:
    st.info("No shadow rows for the current filter.")

st.caption(
    "Sources: `Liquidity_Raid/<sym>_V2/<sym>_shadow_trades.db` (LR/LR Paper) — "
    "`Momentum_Mastery/<sym>/<sym>_shadow_trades.db` (MM) — "
    "`HyroTrader/lrr_shadow_trades.db` (LRR) — "
    "`HyroTrader/manual_trades.db` (Manual) — "
    "`HyroTrader/depth_paper_book.db` + `lrr_paper_book.db` + "
    "`depth_policy_book.db` (paper-execution books) — "
    "`ofcs_shadow/ofcs_paper_book.db` (OFCS) — "
    "`HyroTrader/momentum_4h_*_shadow.db`, `ifvg_*_shadow.db`, "
    "`asia_basket_shadow.db`, `bullput_*_shadow.db`, "
    "`fvg_*_funded_shadow.db` — "
    "`Momentum_Mastery/BTC/btc_momentum_mastery_v2_shadow.db` "
    "(MM partial-exit). "
    "All synced into `dashboard/databases/` on **⟳ Sync** above. "
    "Knife DBs live on page 28."
)
