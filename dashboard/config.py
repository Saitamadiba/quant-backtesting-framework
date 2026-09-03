"""Dashboard configuration: paths, VPS config, strategy registry, constants."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── Base Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent          # /Backtesting
DASHBOARD_DIR = BASE_DIR / "dashboard"
VPS_CACHE_DIR = DASHBOARD_DIR / "databases"
BACKTEST_RESULTS_DIR = DASHBOARD_DIR / "backtest_results"
DUCKDB_PATH = BASE_DIR / "duckdb_data" / "trading_data.duckdb"

# ── VPS Connection (from environment) ────────────────────────────────────────
VPS_HOST = os.getenv("VPS_HOST", "")
VPS_PORT = int(os.getenv("VPS_PORT", "22"))
VPS_USER = os.getenv("VPS_USER", "trader")
VPS_REMOTE_BASE = os.getenv("VPS_REMOTE_BASE", "/home/trader/trading_bots")

# ── Remote DB Mapping ─────────────────────────────────────────────────────────
# key = local cache filename, value = full remote path
VPS_DB_FILES = {
    "fvg_btc.db": f"{VPS_REMOTE_BASE}/FVG_Strategy/BTC/btc-usd_enhanced_v3_trades.db",
    "fvg_eth.db": f"{VPS_REMOTE_BASE}/FVG_Strategy/ETH/eth-usd_enhanced_v3_trades.db",
    "fvg_nq.db":  f"{VPS_REMOTE_BASE}/FVG_Strategy/NQ/nq-usd_enhanced_v3_trades.db",
    # LR BTC/ETH/SOL point at the HyroTrader $10k FUNDED ByBit sub-account
    # (real fills) to match the funded recap; NQ has no ByBit bot so stays paper.
    "lr_btc.db":  f"{VPS_REMOTE_BASE}/HyroTrader/lr_bybit_funded_10k.db",
    "lr_eth.db":  f"{VPS_REMOTE_BASE}/HyroTrader/lr_eth_bybit_funded_10k.db",
    "lr_nq.db":   f"{VPS_REMOTE_BASE}/Liquidity_Raid/NQ_V2/nq_liquidity_raid_v2.db",
    "lr_sol.db":  f"{VPS_REMOTE_BASE}/HyroTrader/lr_sol_bybit_funded_10k.db",
    "mm_btc.db":  f"{VPS_REMOTE_BASE}/Momentum_Mastery/BTC/btc_momentum_mastery_v2.db",
    "mm_eth.db":  f"{VPS_REMOTE_BASE}/Momentum_Mastery/ETH/eth_momentum_mastery_v2.db",
    "mm_nq.db":   f"{VPS_REMOTE_BASE}/Momentum_Mastery/NQ/nq_momentum_mastery_v2.db",
    "sbs.db":     f"{VPS_REMOTE_BASE}/SBS/bots/core/ml_training_data.db",
}

# ── LR Shadow DBs (gate-rejected signals tracked as paper trades) ─────────────
# Populated by the internal strategy core — every signal a gate
# rejects (IV gate, London-shorts ban, regime block, counter-trend, etc.)
# becomes a shadow paper trade with TP/SL/TIME_EXIT tracking + a block_reason
# field so practice can be confronted against the WFO claim that justified
# each gate.  See dashboard page 23_Shadow_Trades.
VPS_SHADOW_DB_FILES = {
    "lr_btc_shadow.db": f"{VPS_REMOTE_BASE}/Liquidity_Raid/BTC_V2/btc_shadow_trades.db",
    "lr_eth_shadow.db": f"{VPS_REMOTE_BASE}/Liquidity_Raid/ETH_V2/eth_shadow_trades.db",
    "lr_nq_shadow.db":  f"{VPS_REMOTE_BASE}/Liquidity_Raid/NQ_V2/nq_shadow_trades.db",
    "lr_sol_shadow.db": f"{VPS_REMOTE_BASE}/Liquidity_Raid/SOL_V2/sol_shadow_trades.db",
    "mm_btc_shadow.db": f"{VPS_REMOTE_BASE}/Momentum_Mastery/BTC/btc_shadow_trades.db",
    "mm_eth_shadow.db": f"{VPS_REMOTE_BASE}/Momentum_Mastery/ETH/eth_shadow_trades.db",
    "mm_nq_shadow.db":  f"{VPS_REMOTE_BASE}/Momentum_Mastery/NQ/nq_shadow_trades.db",
    # ── Hybrid paper-bot shadow DBs (2026-05-31) — collect off-window + BTC-regime-block
    #     signals from the 7 no-Deribit-data LR paper bots (XRP + 6 majors).
    "lr_xrp_paper_shadow.db":  f"{VPS_REMOTE_BASE}/Liquidity_Raid/XRP_V2/xrp_shadow_trades.db",
    "lr_bnb_paper_shadow.db":  f"{VPS_REMOTE_BASE}/Liquidity_Raid/BNB_V2/bnb_shadow_trades.db",
    "lr_doge_paper_shadow.db": f"{VPS_REMOTE_BASE}/Liquidity_Raid/DOGE_V2/doge_shadow_trades.db",
    "lr_avax_paper_shadow.db": f"{VPS_REMOTE_BASE}/Liquidity_Raid/AVAX_V2/avax_shadow_trades.db",
    "lr_link_paper_shadow.db": f"{VPS_REMOTE_BASE}/Liquidity_Raid/LINK_V2/link_shadow_trades.db",
    "lr_dot_paper_shadow.db":  f"{VPS_REMOTE_BASE}/Liquidity_Raid/DOT_V2/dot_shadow_trades.db",
    "lr_bch_paper_shadow.db":  f"{VPS_REMOTE_BASE}/Liquidity_Raid/BCH_V2/bch_shadow_trades.db",
    # ── LRR shadow scanner (2026-06-02) — single multi-asset DB. Different
    #    schema (table=lrr_signals, asset column per-row); the page's
    #    _load_one detects the filename and applies the schema-bridge.
    "lrr_shadow_trades.db":    f"{VPS_REMOTE_BASE}/HyroTrader/lrr_shadow_trades.db",
    # ── MM 15m forward-shadow (2026-07-15) — Tier-3 record-only arm from
    #    MM_15M_SHADOW_SPEC.md. Deploy is gated on build-gate B1; until the
    #    bot ships, the remote file is absent and sync/load fail soft.
    "mm_15m_shadow.db":        f"{VPS_REMOTE_BASE}/HyroTrader/mm_15m_shadow.db",
    # ── MM 5m maker-scalper forward-shadow (2026-07-15) — touch-gated limit
    #    fills, Tier-3 record-only (MM_5M_SHADOW_SPEC.md).
    "mm_5m_shadow.db":         f"{VPS_REMOTE_BASE}/HyroTrader/mm_5m_shadow.db",
    # ── Manual trades synced from ByBit funded sub-account (2026-06-04) —
    #    every trade NOT placed by a bot, auto-tagged with the nearest
    #    strategy signal (FVG/LR/MM ±30min) and labelled "Manual <TAG> SYM"
    #    in the dashboard. Schema-bridge in _load_one converts table name
    #    + a few column renames so the existing KPI/RR/regime machinery
    #    applies unchanged.
    "manual_trades.db":        f"{VPS_REMOTE_BASE}/HyroTrader/manual_trades.db",
    # ── Shadow-fleet gap closure (2026-07-06) — every shadow/paper vehicle
    #    running on the VPS with a per-row trade book now syncs here too.
    #    Schema bridges live in data/shadow_normalisers.SHADOW_DB_SPECS.
    #    (Knife arms + knife counterfactual shadows stay on page 28.)
    # Paper-EXECUTION books (fee-modeled forward tests):
    "depth_paper_book.db":     f"{VPS_REMOTE_BASE}/HyroTrader/depth_paper_book.db",
    "lrr_paper_book.db":       f"{VPS_REMOTE_BASE}/HyroTrader/lrr_paper_book.db",
    "ofcs_paper_book.db":      f"{VPS_REMOTE_BASE}/ofcs_shadow/ofcs_paper_book.db",
    "depth_policy_book.db":    f"{VPS_REMOTE_BASE}/HyroTrader/depth_policy_book.db",
    # Signal shadows (record-only detectors, structurally no orders):
    "momentum_4h_sol_shadow.db": f"{VPS_REMOTE_BASE}/HyroTrader/momentum_4h_sol_shadow.db",
    "momentum_4h_xrp_shadow.db": f"{VPS_REMOTE_BASE}/HyroTrader/momentum_4h_xrp_shadow.db",
    "momentum_4h_ltc_shadow.db": f"{VPS_REMOTE_BASE}/HyroTrader/momentum_4h_ltc_shadow.db",
    "ifvg_sweep_shadow.db":      f"{VPS_REMOTE_BASE}/HyroTrader/ifvg_sweep_shadow.db",
    "ifvg_nq_signal_shadow.db":  f"{VPS_REMOTE_BASE}/HyroTrader/ifvg_nq_signal_shadow.db",
    "asia_basket_shadow.db":     f"{VPS_REMOTE_BASE}/HyroTrader/asia_basket_shadow.db",
    "mm_btc_partial_shadow.db":  f"{VPS_REMOTE_BASE}/Momentum_Mastery/BTC/btc_momentum_mastery_v2_shadow.db",
    # Options paper + funded-context sims:
    "bullput_btc_shadow.db":     f"{VPS_REMOTE_BASE}/HyroTrader/bullput_btc_shadow.db",
    "bullput_eth_shadow.db":     f"{VPS_REMOTE_BASE}/HyroTrader/bullput_eth_shadow.db",
    "fvg_btc_funded_shadow.db":  f"{VPS_REMOTE_BASE}/HyroTrader/fvg_btc_funded_shadow.db",
    "fvg_nq_funded_shadow.db":   f"{VPS_REMOTE_BASE}/HyroTrader/fvg_nq_funded_shadow.db",
    # ── Shadow books deployed 2026-07-17 → 09-02 (registered 2026-09-03) ────
    #    Schema bridges in data/shadow_normalisers.py; era-scoped where the
    #    recorder was re-based (NEVER pool eras).
    "antiknife_shadow.db":       f"{VPS_REMOTE_BASE}/HyroTrader/antiknife_shadow.db",
    "crossvenue_shadow.db":      f"{VPS_REMOTE_BASE}/HyroTrader/crossvenue_shadow.db",
    "gated_lr_shadow.db":        f"{VPS_REMOTE_BASE}/HyroTrader/gated_lr_shadow.db",
    "wide_rr_shadow.db":         f"{VPS_REMOTE_BASE}/HyroTrader/wide_rr_shadow.db",
    "halt_shadow_book.db":       f"{VPS_REMOTE_BASE}/HyroTrader/halt_shadow_book.db",
    "sweep_engine.db":           f"{VPS_REMOTE_BASE}/sweep_engine/sweep_engine.db",
    "fib618_shadow.db":          f"{VPS_REMOTE_BASE}/fib618_shadow/fib618_shadow.db",
    "fvg_alts_shadow.db":        f"{VPS_REMOTE_BASE}/HyroTrader/fvg_alts_shadow.db",
    "lr_signal_shadow.db":       f"{VPS_REMOTE_BASE}/HyroTrader/lr_shadow_trades.db",
    "depth_policy_paper_book.db": f"{VPS_REMOTE_BASE}/HyroTrader/depth_policy_paper_book.db",
}

# The LR/MM regime + flow GATE shadow fleet: one canonical `shadow_trades` book
# per (bot, gate) under shadow_books/. Generated so a new gate book only needs
# adding to this list. Filenames double as the local cache names.
GATE_SHADOW_BOOKS = [
    "lr_avaxusdt_flow_gate", "lr_avaxusdt_regime_gate", "lr_bchusdt_flow_gate",
    "lr_bchusdt_regime_gate", "lr_bnbusdt_flow_gate", "lr_bnbusdt_regime_gate",
    "lr_btcusdt_flow_gate", "lr_btcusdt_regime_gate", "lr_dogeusdt_flow_gate",
    "lr_dogeusdt_regime_gate", "lr_dotusdt_flow_gate", "lr_dotusdt_regime_gate",
    "lr_linkusdt_flow_gate", "lr_linkusdt_regime_gate", "lr_solusdt_flow_gate",
    "lr_solusdt_regime_gate", "lr_xrpusdt_flow_gate", "lr_xrpusdt_regime_gate",
    "mm_btcusdt_flow_gate", "mm_ethusdt_flow_gate",
    "fvg_btcusdt_funding_gate", "fvg_ethusdt_funding_gate",
]
for _g in GATE_SHADOW_BOOKS:
    VPS_SHADOW_DB_FILES[f"{_g}.db"] = f"{VPS_REMOTE_BASE}/shadow_books/{_g}.db"

# ── Knife bot DBs (forward-shadow detector + the funded/demo arms) ───────────
# The "knife" catches over-extended liquidity sweeps (8-bar extreme break) and
# fades them back. Six order-placing arms plus the read-only detector write
# separate DBs on the VPS (all `funded_trades`, entry_mode maker|taker):
#   • knife_shadow.db          — ORIGINAL read-only forward-shadow detector (table=episodes)
#   • knife_funded_maker.db    — original maker arm (post-only limit), 2026-06
#   • knife_funded_taker.db    — taker arm (market at the break)
#   • knife_funded_100k.db     — maker arm on the $100k demo seat
#   • knife_funded_10k.db      — $10k challenge seat (key dead since 2026-09, see p.30)
#   • knife_funded_maker2.db   — maker v2 arm (key dead since 2026-09, see p.30)
#   • knife_funded_ethmstop.db — ETH maker arm with the managed stop
# Not synced by default (no local copy until the page's Sync button runs).
VPS_KNIFE_DB_FILES = {
    "knife_shadow.db":          f"{VPS_REMOTE_BASE}/HyroTrader/knife_shadow.db",
    "knife_funded_maker.db":    f"{VPS_REMOTE_BASE}/HyroTrader/knife_bybit_funded.db",
    "knife_funded_taker.db":    f"{VPS_REMOTE_BASE}/HyroTrader/knife_bybit_funded_taker.db",
    "knife_funded_100k.db":     f"{VPS_REMOTE_BASE}/HyroTrader/knife_bybit_funded_100k.db",
    "knife_funded_10k.db":      f"{VPS_REMOTE_BASE}/HyroTrader/knife_bybit_funded_challenge.db",
    "knife_funded_maker2.db":   f"{VPS_REMOTE_BASE}/HyroTrader/knife_bybit_funded_maker2.db",
    "knife_funded_ethmstop.db": f"{VPS_REMOTE_BASE}/HyroTrader/knife_bybit_funded_ethmakerstop.db",
}

# Frozen knife-detector v1 spec (features, K-window, folds, score cut, dev AUC).
KNIFE_SPEC_PATH = BASE_DIR / "feature_lab" / "_ml_imports" / "knife_detector_v1_spec.json"
# Offline research corpora the page reads when the VPS isn't synced.
#   • SCOUT — BTC/ETH/SOL only, the recent quote-feature forward window (§2 card).
#   • EPISODES — ALL 12 assets, the full knife detector population scored by the
#     frozen model (built by feature_lab/score_knife_episodes.py). Carries
#     is_holdout so the browser can separate dev (in-sample, optimistic) from
#     holdout (honest out-of-sample, break_time ≥ 2025-07-01).
KNIFE_SCOUT_PARQUET = (
    BASE_DIR / "feature_lab" / "reports"
    / "knife_v2_scout_table_20260612_082412.parquet"
)
KNIFE_EPISODES_PARQUET = (
    BASE_DIR / "feature_lab" / "reports"
    / "knife_episodes_scored_allassets.parquet"
)
KNIFE_FROZEN_SCORE_CUT = 0.6289843838166522

SHADOW_DB_STRATEGY_MAP = {
    "lr_btc_shadow.db": ("Liquidity Raid", "BTC"),
    "lr_eth_shadow.db": ("Liquidity Raid", "ETH"),
    "lr_nq_shadow.db":  ("Liquidity Raid", "NQ"),
    "lr_sol_shadow.db": ("Liquidity Raid", "SOL"),
    "mm_btc_shadow.db": ("Momentum Mastery", "BTC"),
    "mm_eth_shadow.db": ("Momentum Mastery", "ETH"),
    "mm_nq_shadow.db":  ("Momentum Mastery", "NQ"),
    # Hybrid paper-bot shadows — labelled "LR Paper" to keep them visually
    # distinct from the live-bot shadows.
    "lr_xrp_paper_shadow.db":  ("LR Paper", "XRP"),
    "lr_bnb_paper_shadow.db":  ("LR Paper", "BNB"),
    "lr_doge_paper_shadow.db": ("LR Paper", "DOGE"),
    "lr_avax_paper_shadow.db": ("LR Paper", "AVAX"),
    "lr_link_paper_shadow.db": ("LR Paper", "LINK"),
    "lr_dot_paper_shadow.db":  ("LR Paper", "DOT"),
    "lr_bch_paper_shadow.db":  ("LR Paper", "BCH"),
    # LRR scanner is single-DB multi-asset. The sentinel symbol "MULTI" tells
    # the page's loader that the asset is in the row, not the file mapping.
    "lrr_shadow_trades.db":    ("LRR Shadow", "MULTI"),
    # Manual trades — single multi-asset DB; per-row strategy_tag (set by
    # the sync script's auto-tagger) becomes the bot label downstream.
    "manual_trades.db":        ("Manual", "MULTI"),
    # ── Shadow-fleet gap closure (2026-07-06). MULTI = asset in the row.
    # Paper-execution books (fee-modeled: r_multiple = r_net of a 0.10R toll):
    "depth_paper_book.db":     ("Depth Paper", "MULTI"),
    "lrr_paper_book.db":       ("LRR Paper", "MULTI"),
    "ofcs_paper_book.db":      ("OFCS Paper", "MULTI"),
    # "Depth Exit Policy" (not "Depth Policy") so its bot abbreviation is
    # DEP — "Depth Paper" already owns DP and the labels would collide.
    "depth_policy_book.db":    ("Depth Exit Policy", "MULTI"),
    # Signal shadows:
    "momentum_4h_sol_shadow.db": ("Momentum 4H", "SOL"),
    "momentum_4h_xrp_shadow.db": ("Momentum 4H", "XRP"),
    "momentum_4h_ltc_shadow.db": ("Momentum 4H", "LTC"),
    "ifvg_sweep_shadow.db":      ("iFVG Shadow", "MULTI"),
    "ifvg_nq_signal_shadow.db":  ("iFVG Shadow", "MULTI"),
    "asia_basket_shadow.db":     ("Asia Basket", "BASKET"),
    "mm_btc_partial_shadow.db":  ("MM Partial-Exit", "BTC"),
    # Options paper + funded-context sims:
    "bullput_btc_shadow.db":     ("Bull-Put Paper", "BTC"),
    "bullput_eth_shadow.db":     ("Bull-Put Paper", "ETH"),
    "fvg_btc_funded_shadow.db":  ("FVG Funded Shadow", "BTC"),
    "fvg_nq_funded_shadow.db":   ("FVG Funded Shadow", "NQ"),
    # MM 15m / 5m forward-shadows (synced since 2026-07-16, labelled 2026-09-03)
    "mm_15m_shadow.db":          ("MM 15m Shadow", "MULTI"),
    "mm_5m_shadow.db":           ("MM 5m Shadow", "MULTI"),
    # ── Registered 2026-09-03 (deployed 07-17 → 09-02). MULTI = asset in the row.
    "antiknife_shadow.db":       ("Anti-Knife Shadow", "MULTI"),
    "crossvenue_shadow.db":      ("Knife Cross-Venue", "MULTI"),
    "gated_lr_shadow.db":        ("Gated LR Shadow", "MULTI"),
    "wide_rr_shadow.db":         ("Wide-RR Shadow", "MULTI"),
    # era-2 only — era-1 is the CLOSED phantom-fill book (memory 2026-08-24).
    "halt_shadow_book.db":       ("Halt Shadow e2", "MULTI"),
    # GROSS R by construction: the fee/slip toll is not deducted.
    "sweep_engine.db":           ("Sweep Engine gross", "MULTI"),
    "fib618_shadow.db":          ("Fib618 Shadow", "MULTI"),
    # current era only (era-1/2 were invalidated; the loader keeps the latest).
    "fvg_alts_shadow.db":        ("FVG Alts Shadow", "MULTI"),
    # era-2 only: the fee-charged, fill-adjudicated LR signal shadow.
    "lr_signal_shadow.db":       ("LR Signal Shadow e2", "BTC"),
    "depth_policy_paper_book.db": ("Depth Policy Paper", "MULTI"),
}
for _g in GATE_SHADOW_BOOKS:
    _fam, _sym, *_gate = _g.split("_")
    SHADOW_DB_STRATEGY_MAP[f"{_g}.db"] = (
        f"{_fam.upper()} {' '.join(_gate).title()} Shadow",
        _sym.upper().replace("USDT", ""))

# ── Remote ML Training DB Mapping ────────────────────────────────────────────
VPS_ML_FILES = {
    "fvg_btc_ml_training.db":  f"{VPS_REMOTE_BASE}/FVG_Strategy/BTC/ml_training_data.db",
    "fvg_eth_ml_training.db":  f"{VPS_REMOTE_BASE}/FVG_Strategy/ETH/ml_training_data.db",
    "fvg_nq_ml_training.db":   f"{VPS_REMOTE_BASE}/FVG_Strategy/NQ/ml_training_data.db",
    "lr_btc_ml_training.db":   f"{VPS_REMOTE_BASE}/Liquidity_Raid/BTC_V2/ml_training_data.db",
    "lr_eth_ml_training.db":   f"{VPS_REMOTE_BASE}/Liquidity_Raid/ETH_V2/ml_training_data.db",
    "lr_nq_ml_training.db":    f"{VPS_REMOTE_BASE}/Liquidity_Raid/NQ_V2/ml_training_data.db",
    "lr_sol_ml_training.db":   f"{VPS_REMOTE_BASE}/Liquidity_Raid/SOL_V2/ml_training_data.db",
    "mm_btc_ml_training.db":   f"{VPS_REMOTE_BASE}/Momentum_Mastery/BTC/ml_training_data.db",
    "mm_eth_ml_training.db":   f"{VPS_REMOTE_BASE}/Momentum_Mastery/ETH/ml_training_data.db",
    "mm_nq_ml_training.db":    f"{VPS_REMOTE_BASE}/Momentum_Mastery/NQ/ml_training_data.db",
    "root_ml_training.db":     f"{VPS_REMOTE_BASE}/ml_training_data.db",
}

# ── Strategy Registry ─────────────────────────────────────────────────────────
# RESEARCH_STRATEGIES are the WFO-backed families the Explainer / Deep-Dive /
# WFO / Shadow-Backtest pages have hand-written content and adapters for.
# STRATEGIES is the full palette: those five plus every family the live fleet
# has deployed since (their books come in via data/fleet_registry.py and show
# on the Overview / Journal / Equity / Session / Monthly pages and page 30).
RESEARCH_STRATEGIES = {
    "FVG": {"color": "#2196F3", "symbols": ["BTC", "ETH", "NQ"]},
    "Liquidity Raid": {"color": "#FF9800", "symbols": ["BTC", "ETH", "NQ", "SOL"]},
    "Momentum Mastery": {"color": "#4CAF50", "symbols": ["BTC", "ETH", "NQ"]},
    "Vol Edge": {"color": "#00BCD4", "symbols": ["BTC", "ETH"]},
    "SBS": {"color": "#9C27B0", "symbols": ["BTC", "ETH"]},
}

# Fleet families (2026-07 → 09), one line each — the mechanism in plain words.
# Colours are distinct from the research five so mixed charts stay legible.
FLEET_FAMILIES = {
    "Knife":         {"color": "#E53935", "symbols": ["ADA", "AVAX", "BCH", "BNB", "BTC", "DOGE", "DOT", "LINK", "LTC", "SOL", "XRP"],
                      "desc": "fade an over-extended liquidity sweep (8-bar extreme break) back to the level"},
    "Depth":         {"color": "#6D4C41", "desc": "L2-absorption entries at a level, taker and maker arms, policy exits"},
    "Desk":          {"color": "#8E24AA", "desc": "resting-limit seats at pre-computed levels, one claim per level"},
    "Retest":        {"color": "#5E35B1", "desc": "level retest seats with a managed backstop"},
    "SMC":           {"color": "#3949AB", "desc": "smart-money-concept setups (12h/4h and 1d/1h structure)"},
    "OFCS":          {"color": "#1E88E5", "desc": "order-flow conditioned sweeps; challenge rule-set since 2026-09-01"},
    "London Raid":   {"color": "#039BE5", "desc": "fade the London raid of the Asia range (maker + taker arms)"},
    "LRR":           {"color": "#00ACC1", "desc": "liquidity-raid reversal, short-only demo + paper book"},
    "Ferryman":      {"color": "#00897B", "desc": "600s markout scalp on the knife break — killed 2026-08, still recording"},
    "FVG Alts":      {"color": "#43A047", "desc": "FVG continuation on alts, 1-min scanner, demo twin"},
    "Funding Carry": {"color": "#7CB342", "desc": "delta-neutral cash-and-carry harvesting funding; $ book, no stop R"},
    "Momentum 4H":   {"color": "#C0CA33", "desc": "4H trend momentum on ADA (live) with SOL/XRP/LTC shadows"},
    "Displacement":  {"color": "#FDD835", "desc": "displacement-candle continuation on BTC"},
    "Phantom":       {"color": "#FFB300", "desc": "demo executor of the meta-conductor's ACTIVE tags"},
    "Options":       {"color": "#FB8C00", "symbols": ["BTC", "ETH"], "desc": "bull-put spreads, iron flies and calendars on BTC/ETH options"},
    "Analyst Drift": {"color": "#F4511E", "desc": "US equities: MOO→MOC drift after out-of-hours analyst news (Alpaca paper)"},
    "Level Seats":   {"color": "#757575", "desc": "generic level-seat shape (desk/retest family)"},
}

STRATEGIES = {
    **RESEARCH_STRATEGIES,
    **{k: {"color": v["color"], "symbols": list(v.get("symbols", []))} for k, v in FLEET_FAMILIES.items()},
}
try:  # symbols per family come from the fleet registry's literal symbol columns
    from data.fleet_registry import FAMILIES as _FLEET_FAMS, family_symbols as _fam_syms
    for _f in _FLEET_FAMS:
        STRATEGIES.setdefault(_f, {"color": "#888888", "symbols": []})
        STRATEGIES[_f]["symbols"] = _fam_syms(_f) or STRATEGIES[_f]["symbols"]
except Exception:  # registry import must never break the dashboard config
    pass

STRATEGY_COLORS = {s: v["color"] for s, v in STRATEGIES.items()}

# local DB file -> (strategy, symbol) — the LEGACY unified-universe books.
# Fleet books (Tier 1 + 2) join the same universe through FLEET_CACHE_DIR +
# data/fleet_registry.py; see data/schema_normalizer.load_all_live_trades.
DB_STRATEGY_MAP = {
    "fvg_btc.db": ("FVG", "BTC"),
    "fvg_eth.db": ("FVG", "ETH"),
    "fvg_nq.db":  ("FVG", "NQ"),
    "lr_btc.db":  ("Liquidity Raid", "BTC"),
    "lr_eth.db":  ("Liquidity Raid", "ETH"),
    "lr_nq.db":   ("Liquidity Raid", "NQ"),
    "lr_sol.db":  ("Liquidity Raid", "SOL"),
    "mm_btc.db":  ("Momentum Mastery", "BTC"),
    "mm_eth.db":  ("Momentum Mastery", "ETH"),
    "mm_nq.db":   ("Momentum Mastery", "NQ"),
    "sbs.db":     ("SBS", "ALL"),
}

# ── Fleet books (Tier 1 + Tier 2) synced as ONE rsync batch ──────────────────
# Local copies keep the VPS-relative path under databases/fleet/, so the same
# registry that drives page 30 drives the unified trade pages offline.
FLEET_CACHE_DIR = VPS_CACHE_DIR / "fleet"


def fleet_db_relpaths() -> list:
    """VPS-relative paths of every Tier-1/2 fleet book (deduped, globs skipped)."""
    try:
        from data.fleet_registry import BOOKS as _BOOKS
    except Exception:
        return []
    seen, out = set(), []
    for _b in _BOOKS:
        if _b.tier > 2 or any(c in _b.db for c in "*?["):
            continue
        if _b.db not in seen:
            seen.add(_b.db)
            out.append(_b.db)
    return out


# ── VPS Systemd Services / timers / cron seats ───────────────────────────────
# unit  → {strategy, symbol, kind, log}. `kind` is "service" (systemctl
# is-active on the .service), "timer" (a oneshot fired by a .timer — status is
# the timer's), or "cron" (no unit; health = log freshness). `log` is the file
# the seat actually writes (None = journal-only, which the trader login cannot
# read — an operator-side `StandardOutput=append:` fix). Rebuilt 2026-09-03
# from `systemctl list-units` + crontab; lr-btc / lr-sol / sbs-* were removed
# because those units no longer exist.
_L = f"{VPS_REMOTE_BASE}/logs"
BOT_SERVICES = {
    # ── Tier 1 · knife funded/demo arms ──
    "knife_bybit_funded_100k.service":        {"strategy": "Knife", "symbol": "$100k maker", "kind": "service", "log": "/var/log/knife_funded_100k/knife-funded-100k.log"},
    "knife_bybit_funded_challenge.service":   {"strategy": "Knife", "symbol": "$10k challenge", "kind": "service", "log": None},
    "knife_bybit_funded_taker.service":       {"strategy": "Knife", "symbol": "taker", "kind": "service", "log": None},
    "knife_bybit_funded_maker2.service":      {"strategy": "Knife", "symbol": "maker2", "kind": "service", "log": None},
    "knife_bybit_funded_ethmakerstop.service": {"strategy": "Knife", "symbol": "ETH mstop", "kind": "service", "log": None},
    # ── Tier 1 · Liquidity Raid funded + ByBit demo ──
    "lr-bybit-funded@10000.service":          {"strategy": "Liquidity Raid", "symbol": "BTC funded", "kind": "service", "log": f"{_L}/lr_bybit_funded_10k.log"},
    "lr-eth-bybit-funded@10000.service":      {"strategy": "Liquidity Raid", "symbol": "ETH funded", "kind": "service", "log": f"{_L}/lr_eth_bybit_funded_10k.log"},
    "lr-sol-bybit-funded@10000.service":      {"strategy": "Liquidity Raid", "symbol": "SOL funded", "kind": "service", "log": f"{_L}/lr_sol_bybit_funded_10k.log"},
    "lr-eth-bybit.service":                   {"strategy": "Liquidity Raid", "symbol": "ETH demo", "kind": "service", "log": f"{_L}/lr_eth_bybit.log"},
    "lr-avax-bybit.service":                  {"strategy": "Liquidity Raid", "symbol": "AVAX demo", "kind": "service", "log": f"{_L}/lr_avax_bybit.log"},
    "lr-doge-bybit.service":                  {"strategy": "Liquidity Raid", "symbol": "DOGE demo", "kind": "service", "log": f"{_L}/lr_doge_bybit.log"},
    "lr-dot-bybit.service":                   {"strategy": "Liquidity Raid", "symbol": "DOT demo", "kind": "service", "log": f"{_L}/lr_dot_bybit.log"},
    "lr-link-bybit.service":                  {"strategy": "Liquidity Raid", "symbol": "LINK demo", "kind": "service", "log": f"{_L}/lr_link_bybit.log"},
    # ── Tier 1 · other order-placing seats ──
    "momentum-4h-ada.service":                {"strategy": "Momentum 4H", "symbol": "ADA", "kind": "service", "log": "/var/log/momentum_4h/ada.log"},
    "displacement-bybit.service":             {"strategy": "Displacement", "symbol": "BTC demo", "kind": "service", "log": f"{_L}/displacement_bybit.log"},
    "retest_demo.service":                    {"strategy": "Retest", "symbol": "daemon", "kind": "service", "log": f"{_L}/retest_demo_daemon.log"},
    "retest_demo.bot":                        {"strategy": "Retest", "symbol": "cycle", "kind": "cron", "log": f"{_L}/retest_demo.log"},
    "desk_demo.bot":                          {"strategy": "Desk", "symbol": "cycle", "kind": "cron", "log": f"{_L}/desk_demo.log"},
    "lr_wide_demo.bot":                       {"strategy": "Liquidity Raid", "symbol": "wide demo", "kind": "cron", "log": f"{_L}/lr_wide_demo.log"},
    "lrr_short_demo.bot":                     {"strategy": "LRR", "symbol": "short demo", "kind": "cron", "log": f"{_L}/lrr_short_demo.log"},
    "smc_demo.bot":                           {"strategy": "SMC", "symbol": "1d/1h demo", "kind": "cron", "log": f"{_L}/smc_demo.log"},
    "smc12h4h_demo.bot":                      {"strategy": "SMC", "symbol": "12h/4h demo", "kind": "cron", "log": f"{_L}/smc12h4h_demo.log"},
    "depth_policy_taker":                     {"strategy": "Depth", "symbol": "policy taker", "kind": "cron", "log": f"{_L}/depth_policy_taker.log"},
    "depth_policy_maker":                     {"strategy": "Depth", "symbol": "policy maker", "kind": "cron", "log": f"{_L}/depth_policy_maker.log"},
    "depth_taker_v1":                         {"strategy": "Depth", "symbol": "taker v1", "kind": "cron", "log": f"{_L}/depth_taker_bot.log"},
    "depth_maker_v1":                         {"strategy": "Depth", "symbol": "maker v1", "kind": "cron", "log": f"{_L}/depth_maker_bot.log"},
    "ofcs-demo.timer":                        {"strategy": "OFCS", "symbol": "demo/challenge", "kind": "timer", "log": "/var/log/ofcs_demo/ofcs-demo.log"},
    "ofcs-demo-manage.timer":                 {"strategy": "OFCS", "symbol": "1-min manage", "kind": "timer", "log": "/var/log/ofcs_demo/ofcs-demo.log"},
    "london-raid-demo.timer":                 {"strategy": "London Raid", "symbol": "maker demo", "kind": "timer", "log": "/var/log/london_raid_demo/london-raid-demo.log"},
    "ferryman-demo.service":                  {"strategy": "Ferryman", "symbol": "demo", "kind": "service", "log": f"{_L}/ferryman_bot.log"},
    "fvg_alts_demo":                          {"strategy": "FVG Alts", "symbol": "demo twin", "kind": "cron", "log": f"{_L}/fvg_alts_demo.log"},
    "funding_carry_demo.bot":                 {"strategy": "Funding Carry", "symbol": "demo", "kind": "cron", "log": f"{_L}/funding_carry_demo.log"},
    "bullput-btc-demo.service":               {"strategy": "Options", "symbol": "BTC bull-put", "kind": "service", "log": f"{_L}/bullput_btc_bybit.log"},
    "bullput-eth-demo.service":               {"strategy": "Options", "symbol": "ETH bull-put", "kind": "service", "log": f"{_L}/bullput_eth_bybit.log"},
    "ironfly-btc-bybit.service":              {"strategy": "Options", "symbol": "BTC iron fly", "kind": "service", "log": f"{_L}/ironfly_btc_bybit.log"},
    "ironfly-eth-bybit.service":              {"strategy": "Options", "symbol": "ETH iron fly", "kind": "service", "log": f"{_L}/ironfly_eth_bybit.log"},
    "straddle-btc.service":                   {"strategy": "Vol Edge", "symbol": "BTC", "kind": "service", "log": f"{_L}/straddle_btc.log"},
    "straddle-eth.service":                   {"strategy": "Vol Edge", "symbol": "ETH", "kind": "service", "log": f"{_L}/straddle_eth.log"},
    "asia-basket-tactical.service":           {"strategy": "Asia Basket", "symbol": "tactical", "kind": "service", "log": "/var/log/asia_basket/asia_basket_tactical.log"},
    # ── Tier 1/2 · the research families' live + paper bots ──
    "fvg-btc.service":                        {"strategy": "FVG", "symbol": "BTC", "kind": "service", "log": f"{_L}/fvg_btc.log"},
    "fvg-eth.service":                        {"strategy": "FVG", "symbol": "ETH", "kind": "service", "log": f"{_L}/fvg_eth.log"},
    "fvg-nq.service":                         {"strategy": "FVG", "symbol": "NQ", "kind": "service", "log": f"{_L}/fvg_nq.log"},
    "lr-eth.service":                         {"strategy": "Liquidity Raid", "symbol": "ETH paper", "kind": "service", "log": f"{_L}/lr_eth.log"},
    "lr-nq.service":                          {"strategy": "Liquidity Raid", "symbol": "NQ paper", "kind": "service", "log": f"{_L}/lr_nq.log"},
    **{f"lr-{a}-paper.service": {"strategy": "Liquidity Raid", "symbol": f"{a.upper()} paper", "kind": "service", "log": None}
       for a in ("avax", "bch", "bnb", "doge", "dot", "link", "xrp")},
    "mm-btc.service":                         {"strategy": "Momentum Mastery", "symbol": "BTC (stopped)", "kind": "service", "log": f"{_L}/mm_btc.log"},
    "mm-eth.service":                         {"strategy": "Momentum Mastery", "symbol": "ETH", "kind": "service", "log": f"{_L}/mm_eth.log"},
    "mm-nq.service":                          {"strategy": "Momentum Mastery", "symbol": "NQ", "kind": "service", "log": f"{_L}/mm_nq.log"},
    "mm-btc-shadow.service":                  {"strategy": "Momentum Mastery", "symbol": "BTC shadow", "kind": "service", "log": f"{_L}/mm_btc_shadow.log"},
    "displacement-btc.service":               {"strategy": "Displacement", "symbol": "BTC signal", "kind": "service", "log": f"{_L}/displacement_btc.log"},
    "analyst_drift_paper.bot":                {"strategy": "Analyst Drift", "symbol": "Alpaca paper", "kind": "cron", "log": f"{_L}/analyst_drift_paper.log"},
    "lrr_paper_bot":                          {"strategy": "LRR", "symbol": "paper book", "kind": "cron", "log": f"{_L}/lrr_paper_bot.log"},
    "depth_paper_bot":                        {"strategy": "Depth", "symbol": "paper book", "kind": "cron", "log": f"{_L}/depth_paper_bot.log"},
    "depth_policy_paper_bot":                 {"strategy": "Depth", "symbol": "policy paper", "kind": "cron", "log": f"{_L}/depth_policy_paper_bot.log"},
    "ofcs-paper-bot.timer":                   {"strategy": "OFCS", "symbol": "paper book", "kind": "timer", "log": "/var/log/ofcs_shadow/ofcs-paper-bot.log"},
    "capital_ladder":                         {"strategy": "Fleet", "symbol": "capital ladder", "kind": "cron", "log": f"{_L}/capital_ladder.log"},
    "meta_conductor":                         {"strategy": "Fleet", "symbol": "meta-conductor", "kind": "cron", "log": None},
    # ── Tier 3 · shadow recorders ──
    "knife_detector_shadow.service":          {"strategy": "Knife", "symbol": "shadow", "kind": "service", "log": "/var/log/knife_shadow/knife-shadow.log"},
    "knife_crossvenue_shadow.service":        {"strategy": "Knife", "symbol": "cross-venue", "kind": "service", "log": f"{VPS_REMOTE_BASE}/HyroTrader/knife_crossvenue_shadow.log"},
    "antiknife_shadow":                       {"strategy": "Knife", "symbol": "anti-knife", "kind": "cron", "log": None},
    "asia-basket-shadow.service":             {"strategy": "Asia Basket", "symbol": "shadow", "kind": "service", "log": "/var/log/asia_basket/asia_basket_shadow.log"},
    "ifvg-sweep-shadow.service":              {"strategy": "iFVG", "symbol": "shadow", "kind": "service", "log": "/var/log/ifvg_sweep/shadow.log"},
    "momentum-4h-sol-shadow.service":         {"strategy": "Momentum 4H", "symbol": "SOL shadow", "kind": "service", "log": "/var/log/momentum_4h/sol-shadow.log"},
    "momentum-4h-xrp-shadow.service":         {"strategy": "Momentum 4H", "symbol": "XRP shadow", "kind": "service", "log": "/var/log/momentum_4h/xrp-shadow.log"},
    "momentum-4h-ltc-shadow.service":         {"strategy": "Momentum 4H", "symbol": "LTC shadow", "kind": "service", "log": "/var/log/momentum_4h/ltc-shadow.log"},
    "ofcs-shadow.timer":                      {"strategy": "OFCS", "symbol": "shadow", "kind": "timer", "log": "/var/log/ofcs_shadow/ofcs-shadow.log"},
    "fvg-funded-shadow.timer":                {"strategy": "FVG", "symbol": "funded shadow", "kind": "timer", "log": f"{_L}/fvg_funded_shadow.log"},
    "gated_lr_shadow":                        {"strategy": "Liquidity Raid", "symbol": "gated shadow", "kind": "cron", "log": f"{_L}/gated_lr_shadow.log"},
    "lrr_shadow_scanner":                     {"strategy": "LRR", "symbol": "shadow scanner", "kind": "cron", "log": f"{_L}/lrr_shadow_scanner.log"},
    "wide_rr_shadow":                         {"strategy": "Wide RR", "symbol": "shadow", "kind": "cron", "log": None},
    "halt_shadow":                            {"strategy": "Fleet", "symbol": "halt shadow", "kind": "cron", "log": f"{_L}/halt_shadow.log"},
    "sweep_engine.shadow":                    {"strategy": "Sweep Engine", "symbol": "shadow", "kind": "cron", "log": f"{_L}/sweep_engine.log"},
    "fib618_shadow.run":                      {"strategy": "Fib618", "symbol": "shadow", "kind": "cron", "log": f"{_L}/fib618_shadow.log"},
    "mm_5m_shadow":                           {"strategy": "Momentum Mastery", "symbol": "5m shadow", "kind": "cron", "log": f"{_L}/mm_5m_shadow.log"},
    "mm_15m_shadow":                          {"strategy": "Momentum Mastery", "symbol": "15m shadow", "kind": "cron", "log": f"{_L}/mm_15m_shadow.log"},
    "fvg_alts_shadow":                        {"strategy": "FVG Alts", "symbol": "1-min shadow", "kind": "cron", "log": f"{_L}/fvg_alts_shadow.log"},
    "erl_irl_shadow":                         {"strategy": "ERL/IRL", "symbol": "shadow", "kind": "cron", "log": f"{_L}/erl_irl_shadow.log"},
    "news_shadow.recorder":                   {"strategy": "News", "symbol": "PIT recorder", "kind": "cron", "log": f"{_L}/news_shadow.log"},
    "fomc_shadow.shadow":                     {"strategy": "FOMC", "symbol": "shadow", "kind": "cron", "log": f"{_L}/fomc_shadow.log"},
    # ── legacy / failing (kept visible so nobody forgets it is looping) ──
    "eth-enhanced-raid-bot.service":          {"strategy": "Legacy", "symbol": "ETH raid (auto-restart loop)", "kind": "service", "log": None},
}

SERVICE_WORK_DIRS = {
    "fvg-btc.service": f"{VPS_REMOTE_BASE}/FVG_Strategy/BTC",
    "fvg-eth.service": f"{VPS_REMOTE_BASE}/FVG_Strategy/ETH",
    "fvg-nq.service":  f"{VPS_REMOTE_BASE}/FVG_Strategy/NQ",
    "lr-eth.service":  f"{VPS_REMOTE_BASE}/Liquidity_Raid/ETH_V2",
    "lr-nq.service":   f"{VPS_REMOTE_BASE}/Liquidity_Raid/NQ_V2",
    "mm-btc.service":  f"{VPS_REMOTE_BASE}/Momentum_Mastery/BTC",
    "mm-eth.service":  f"{VPS_REMOTE_BASE}/Momentum_Mastery/ETH",
    "mm-nq.service":   f"{VPS_REMOTE_BASE}/Momentum_Mastery/NQ",
    "mm-btc-shadow.service": f"{VPS_REMOTE_BASE}/Momentum_Mastery/BTC",
    "straddle-btc.service":  f"{VPS_REMOTE_BASE}/Vol_Edge/Straddle_V1",
    "straddle-eth.service":  f"{VPS_REMOTE_BASE}/Vol_Edge/Straddle_V1",
    "displacement-btc.service": f"{VPS_REMOTE_BASE}/Displacement/BTC",
    **{u: f"{VPS_REMOTE_BASE}/HyroTrader" for u in BOT_SERVICES
       if u.startswith(("knife_", "lr-", "momentum-4h", "displacement-bybit", "ferryman",
                        "bullput", "ironfly", "asia-basket", "ifvg", "manual"))
       and u.endswith(".service") and u not in ("lr-eth.service", "lr-nq.service")
       and "paper" not in u},
    **{f"lr-{a}-paper.service": f"{VPS_REMOTE_BASE}/Liquidity_Raid/{a.upper()}_V2"
       for a in ("avax", "bch", "bnb", "doge", "dot", "link", "xrp")},
}

# ── VPS Log Files (read directly — journalctl needs the systemd-journal group) ─
# None means the unit logs ONLY to the journal, which the trader login cannot
# read; the Live Logs page says so instead of failing on a missing file.
SERVICE_LOG_FILES = {svc: info.get("log") for svc, info in BOT_SERVICES.items()}

VPS_BACKUP_SCRIPT = f"{VPS_REMOTE_BASE}/backup_dbs.sh"

# ── Session Time Ranges (Eastern Time — America/New_York, DST-aware) ─────────
# Hours are in ET.  Callers must convert UTC timestamps to ET via
# zoneinfo.ZoneInfo("America/New_York") before comparing — do NOT use a
# hardcoded UTC-5 offset (ET is UTC-4 during daylight saving time).
SESSIONS = {
    "Asian":    {"start": 19, "end": 3},
    "London":   {"start": 3,  "end": 8},
    "New York": {"start": 8,  "end": 16},
}

# ── SBS Local Backtest Data ───────────────────────────────────────────────────
SBS_TRAINING_CSV = BASE_DIR / "SBS" / "data" / "training" / "ml_training_data.csv"
SBS_RESULTS_DIR = BASE_DIR / "SBS" / "data" / "results"

# ── ML Paths ──────────────────────────────────────────────────────────────────
ML_TRAINING_SCRIPT = BASE_DIR / "SBS" / "research" / "ml" / "train_ml_model.py"
ML_TRAINING_DATA = BASE_DIR / "SBS" / "data" / "training" / "ml_training_data.csv"
ML_MODELS_DIR = BASE_DIR / "SBS" / "research" / "ml" / "models"
ML_TRAINING_DB = BASE_DIR / "ml_training_data.db"
ML_ROOT_TRAINING_SCRIPT = BASE_DIR / "ml_model_training.py"
ML_PREDICTIONS_DB = VPS_CACHE_DIR / "ml_predictions.db"
ML_PERFORMANCE_SCORER = BASE_DIR / "ml_performance_scorer.py"

# ── Monte Carlo ───────────────────────────────────────────────────────────────
MC_ANALYSIS_SCRIPT = (
    BASE_DIR / "SBS" / "research" / "analysis" / "monte_carlo"
    / "enhanced_monte_carlo_analysis.py"
)

# ── Binance API (public endpoints, no auth required) ─────────────────────────
BINANCE_REST_BASE = "https://api.binance.us/api/v3"
BINANCE_SYMBOL_MAP = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
}

# ── Manual Trading ───────────────────────────────────────────────────────────
MANUAL_TRADES_DB = DASHBOARD_DIR / "databases" / "manual_trades.db"

TRADING_PAIRS = {
    "BTC": {"source": "binance", "binance_symbol": "BTCUSDT",
            "tv_symbol": "BINANCEUS:BTCUSDT",
            "base": "BTC", "quote": "USDT", "price_decimals": 2, "qty_decimals": 5},
    "ETH": {"source": "binance", "binance_symbol": "ETHUSDT",
            "tv_symbol": "BINANCEUS:ETHUSDT",
            "base": "ETH", "quote": "USDT", "price_decimals": 2, "qty_decimals": 4},
    "NQ":  {"source": "yahoo", "yahoo_ticker": "NQ=F",
            "tv_symbol": "CME_MINI:NQ1!",
            "price_decimals": 2},
}

TV_INTERVAL_MAP = {
    "1m": "1", "5m": "5", "15m": "15", "30m": "30",
    "1h": "60", "4h": "240", "1D": "D",
}

TRADING_REFRESH_OPTIONS = {"Off": 0, "5s": 5_000, "10s": 10_000, "30s": 30_000}

CHART_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1D"]

MTF_MAP = {
    "1m":  ["5m", "15m"],
    "5m":  ["15m", "1h"],
    "15m": ["1h", "4h"],
    "30m": ["1h", "4h"],
    "1h":  ["4h", "1D"],
    "4h":  ["1D"],
    "1D":  [],
}

SESSION_COLORS = {
    "Asian":    "rgba(156, 39, 176, 0.06)",
    "London":   "rgba(76, 175, 80, 0.06)",
    "New York": "rgba(255, 152, 0, 0.06)",
}

# ── Bot Deploy Mapping (local file -> VPS remote path) ───────────────────────
DEPLOY_BOT_FILES = {
    "FVG BTC": (
        BASE_DIR / "FVG_Strategy" / "BTC" / "fvg_btc.py",
        f"{VPS_REMOTE_BASE}/FVG_Strategy/BTC/fvg_btc.py",
    ),
    "FVG ETH": (
        BASE_DIR / "FVG_Strategy" / "ETH" / "fvg_eth.py",
        f"{VPS_REMOTE_BASE}/the internal FVG live config",
    ),
    "FVG NQ": (
        BASE_DIR / "FVG_Strategy" / "NQ" / "fvg_nq.py",
        f"{VPS_REMOTE_BASE}/FVG_Strategy/NQ/fvg_nq.py",
    ),
    "LR BTC": (
        BASE_DIR / "Liquidity_Raid" / "BTC_V2" / "lr_btc.py",
        f"{VPS_REMOTE_BASE}/Liquidity_Raid/BTC_V2/lr_btc.py",
    ),
    "LR ETH": (
        BASE_DIR / "Liquidity_Raid" / "ETH_V2" / "lr_eth.py",
        f"{VPS_REMOTE_BASE}/Liquidity_Raid/ETH_V2/lr_eth.py",
    ),
    "MM BTC": (
        BASE_DIR / "Momentum_Mastery" / "BTC" / "btc_momentum_mastery_v2.py",
        f"{VPS_REMOTE_BASE}/Momentum_Mastery/BTC/btc_momentum_mastery_v2.py",
    ),
    "MM ETH": (
        BASE_DIR / "Momentum_Mastery" / "ETH" / "eth_momentum_mastery_v2.py",
        f"{VPS_REMOTE_BASE}/Momentum_Mastery/ETH/eth_momentum_mastery_v2.py",
    ),
    "MM NQ": (
        BASE_DIR / "Momentum_Mastery" / "NQ" / "nq_momentum_mastery_v2.py",
        f"{VPS_REMOTE_BASE}/Momentum_Mastery/NQ/nq_momentum_mastery_v2.py",
    ),
    "LR NQ": (
        BASE_DIR / "Liquidity_Raid" / "NQ_V2" / "lr_nq.py",
        f"{VPS_REMOTE_BASE}/Liquidity_Raid/NQ_V2/lr_nq.py",
    ),
    "LR SOL": (
        BASE_DIR / "Liquidity_Raid" / "SOL_V2" / "lr_sol.py",
        f"{VPS_REMOTE_BASE}/Liquidity_Raid/SOL_V2/lr_sol.py",
    ),
    "Straddle BTC": (
        BASE_DIR / "Vol_Edge" / "Straddle_V1" / "btc_straddle.py",
        f"{VPS_REMOTE_BASE}/Vol_Edge/Straddle_V1/btc_straddle.py",
    ),
    "Straddle ETH": (
        BASE_DIR / "Vol_Edge" / "Straddle_V1" / "eth_straddle.py",
        f"{VPS_REMOTE_BASE}/Vol_Edge/Straddle_V1/eth_straddle.py",
    ),
    "SBS BTC": (
        BASE_DIR / "SBS" / "bots" / "btc" / "sbs_btc.py",
        f"{VPS_REMOTE_BASE}/SBS/bots/btc/sbs_btc.py",
    ),
    "SBS ETH": (
        BASE_DIR / "SBS" / "bots" / "eth" / "sbs_eth.py",
        f"{VPS_REMOTE_BASE}/SBS/bots/eth/sbs_eth.py",
    ),
}

DEPLOY_SERVICE_MAP = {
    "FVG BTC":      "fvg-btc",
    "FVG ETH":      "fvg-eth",
    "FVG NQ":       "fvg-nq",
    "LR BTC":       "lr-btc",
    "LR ETH":       "lr-eth",
    "LR NQ":        "lr-nq",
    "LR SOL":       "lr-sol",
    "MM BTC":       "mm-btc",
    "MM ETH":       "mm-eth",
    "MM NQ":        "mm-nq",
    "Straddle BTC": "straddle-btc",
    "Straddle ETH": "straddle-eth",
    "SBS BTC":      "sbs-btc",
    "SBS ETH":      "sbs-eth",
}

# ── Macro Data (Macro Context page) ──────────────────────────────────────────
# FRED needs a free API key (https://fred.stlouisfed.org/docs/api/api_key.html).
# Put it in dashboard/.env as FRED_API_KEY=... — never hardcode it here.
# World Bank + the yfinance market-macro series are keyless.
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# ── Broker — Alpaca (Broker panel, READ-ONLY paper) ──────────────────────────
# Put paper-trading keys in dashboard/.env as ALPACA_API_KEY / ALPACA_SECRET_KEY
# — never hardcode them.  The dashboard panel is read-only; it never places
# orders.  ALPACA_PAPER stays True until you deliberately go live.
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() in ("1", "true", "yes")
ALPACA_BASE_URL = (
    "https://paper-api.alpaca.markets" if ALPACA_PAPER
    else "https://api.alpaca.markets"
)

# ── Account Settings ─────────────────────────────────────────────────────────
INITIAL_BALANCE = int(os.getenv("INITIAL_BALANCE", "10000"))

# ── Strategy Change Log ──────────────────────────────────────────────────────
STRATEGY_CHANGELOG = [
    # Example entry:
    # {
    #     "date": "2026-02-11",
    #     "label": "Strategy Update",
    #     "strategies": ["FVG"],
    #     "color": "#FFEB3B",
    #     "description": "Description of what changed",
    # },
]

# ── Auto-Refresh Options (seconds) ───────────────────────────────────────────
REFRESH_OPTIONS = {"Off": 0, "30s": 30_000, "1m": 60_000, "5m": 300_000}

# ── Unified Trade Schema Columns ──────────────────────────────────────────────
TRADE_SCHEMA_COLS = [
    "trade_id", "strategy", "symbol", "timeframe", "source",
    "direction", "entry_time", "exit_time",
    "entry_price", "exit_price", "stop_loss", "take_profit",
    "pnl_usd", "pnl_pct", "r_multiple",
    "session", "exit_reason", "duration_minutes",
    "running_balance", "mfe", "mae", "is_open",
]
