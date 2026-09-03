"""Fleet book registry — one row per live/paper/shadow trade book on the VPS.

This is the map the **Live Fleet** page reads by: for every bot it names the
SQLite book, the closed-trade columns (timestamp, R, $, prices) and, where the
book tracks it, the *running* leg — the position that is filled and not yet
closed, plus any order still working at the exchange.

It is the same roster the session recap prints (`session_pnl_snapshot.sh`'s
`ROWS` table), widened to the seats the recap doesn't carry (ofcs, london raid,
ferryman, fvg alts, funding carry, phantom, options, the paper/shadow books).
Think of the recap as the wall clock in the hallway and this as the movement
behind it: same time, more hands.

Every expression here lands inside a **SELECT** — `build_*_sql()` is the only
place SQL is assembled, and the remote collector refuses anything that is not a
read. Nothing in this module writes, and nothing here holds a secret.

R conventions, per the house standard:
  * Tier 1 (order-placing seats): realized $ **and** R where the book defines it.
  * Tier 2 (paper / virtual book): net R on a virtual $100k; $ is simulated.
  * Tier 3 (shadow / record-only): dimensionless R by design — no $ column.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# Tier labels — the money line, the rehearsal, and the instrument panel.
TIER_NAMES = {
    1: "Tier 1 · live / funded / demo orders",
    2: "Tier 2 · paper / virtual book",
    3: "Tier 3 · shadow / record-only",
}

# Every path is relative to the VPS `~/trading_bots` root.
REMOTE_ROOT = "~/trading_bots"


@dataclass(frozen=True)
class Book:
    """One tradable book: where it lives, and how to read its trades."""

    key: str                      # stable id (also the query id sent to the VPS)
    label: str                    # what the page shows
    tier: int
    db: str                       # path under ~/trading_bots (may contain a glob)
    table: str

    # ── closed trades ────────────────────────────────────────────────────────
    ts: str                       # close timestamp expression (sqlite datetime-able)
    closed_filter: str = "1=1"
    r: Optional[str] = None       # net R expression (None → book has no R)
    pnl: Optional[str] = None     # realized $ expression (None → dimensionless)
    symbol: Optional[str] = None
    side: Optional[str] = None
    entry: Optional[str] = None
    exit: Optional[str] = None

    # ── running leg: filled, not yet closed ──────────────────────────────────
    open_filter: Optional[str] = None
    open_ts: Optional[str] = None
    open_entry: Optional[str] = None
    open_sl: Optional[str] = None
    open_tp: Optional[str] = None
    open_qty: Optional[str] = None
    open_risk: Optional[str] = None

    # ── working leg: order placed at the exchange, not yet filled ────────────
    # Bounded by age: several seats re-arm a resting limit every scan and leave
    # the unfilled row behind, so an unbounded "working" filter counts months of
    # cancelled arms as live orders — a car park of ghosts. The exchange's own
    # open-order list is the truth; this is the book's recent claim.
    working_filter: Optional[str] = None
    working_ts: Optional[str] = None
    working_max_age_h: int = 6

    # Strategy FAMILY for the unified trade universe (Overview / Journal / Equity /
    # Session pages): several books roll up under one family ("Knife" has six
    # arms, "Liquidity Raid" a dozen seats). Colours and filters key off this.
    family: str = ""
    # Full-history columns (closed rows) — used LOCALLY on a synced copy, so
    # they carry the fields the unified schema wants and the window dump does
    # not: when the trade opened, its stop, and why it closed. Each falls back
    # to the running-leg column of the same meaning when unset.
    entry_ts: Optional[str] = None
    exit_reason: Optional[str] = None

    # Columns the BOT ITSELF recorded at the fill, split by when they were
    # knowable. `recorded_pit` = pre-fill (regime at the prior bar, MTF score,
    # funding, OI delta before entry…); `recorded_post` = post-fill (max
    # favourable excursion, absorption measured over the fill window…). The
    # fleet feature spine reads both and labels them; the dashboard never shows
    # a post-fill column as a gate. Canonical name → SQL expression.
    recorded_pit: dict = field(default_factory=dict)
    recorded_post: dict = field(default_factory=dict)
    # A stable per-row identity for "tag a fill once" semantics. Defaults to the
    # SQLite rowid, which is stable for tables without INTEGER PRIMARY KEY
    # reuse; books with a natural key name it here.
    row_key: Optional[str] = None

    # The env-file seat whose ByBit account this book trades ("desk_demo/desk_demo").
    # Only set where it is KNOWN — reconciliation treats an unknown seat as
    # "cannot check", never as "orphan".
    seat: Optional[str] = None
    # False where the balance read cannot see this book's positions — the option
    # seats trade ByBit's `option` category and the snapshot reads `linear`, so
    # an option leg absent from the position list proves nothing.
    reconcilable: bool = True

    note: str = ""
    # True for the books the SessionStart recap already prints, so the page can
    # be reconciled against it line by line.
    in_recap: bool = False

    @property
    def dollars(self) -> bool:
        return self.pnl is not None


# ── shared expressions ────────────────────────────────────────────────────────
# LR books store $ and the risk legs, not R: R = realized $ / $ risked.
_LR_R = "realized_pnl/(ABS(entry_price-stop_loss)*position_size)"
_LR_F = ("realized_pnl IS NOT NULL AND position_size>0 AND stop_loss IS NOT NULL "
         "AND entry_price<>stop_loss")

_KNIFE = dict(
    table="funded_trades", ts="closed_at_utc", closed_filter="r_multiple IS NOT NULL",
    r="r_multiple", pnl="pnl_usd", symbol="symbol", side="direction",
    entry="level", exit="exit_price",
    open_filter="filled_at_utc IS NOT NULL AND closed_at_utc IS NULL",
    open_ts="filled_at_utc", open_entry="level", open_sl="sl", open_tp="tp",
    open_qty="qty", open_risk="risk_usd",
    working_filter="placed_at_utc IS NOT NULL AND filled_at_utc IS NULL "
                   "AND closed_at_utc IS NULL AND COALESCE(exit_reason,'')=''",
    working_ts="placed_at_utc",
    family="Knife", exit_reason="exit_reason",
    recorded_pit={
        "regime5": "regime5", "mtf_score": "mtf_score", "daily_bias": "daily_bias",
        "h4_structure": "h4_structure", "atr_pct": "atr_pct", "er20": "er20",
        "funding_rate": "funding_rate", "oi_delta_pct": "oi_delta_pct",
        "oi_delta_5": "oi_delta_5", "oi_delta_10": "oi_delta_10",
        "oi_delta_240": "oi_delta_240", "cvd_imb": "cvd_imb",
        "self_ret_15m": "self_ret_15m", "btc_ret_15m": "btc_ret_15m",
        "htf_4h_bias": "htf_4h_bias", "lvl_touches_24h": "lvl_touches_24h",
        "favored": "favored", "entry_mode": "entry_mode",
    },
    recorded_post={
        "max_fav_r": "max_fav_r", "absorb_size": "absorb_size",
        "absorb_opp": "absorb_opp", "absorb_imb": "absorb_imb",
        "of_break_aggr_30": "of_break_aggr_30", "of_vel_60": "of_vel_60",
    },
)

_SEATS = dict(  # desk / retest / lr_wide / lrr_short share the `seats` shape
    table="seats", ts="closed_at", closed_filter="realized_r IS NOT NULL",
    r="realized_r", pnl="realized_pnl", symbol="symbol", side="side",
    entry="fill_px",
    open_filter="fill_time IS NOT NULL AND closed_at IS NULL",
    open_ts="fill_time", open_entry="fill_px", open_sl="sl", open_tp="tp",
    open_qty="qty", open_risk="risk_usd", exit_reason="close_reason",
    recorded_pit={"regime5": "regime5", "ltype": "ltype", "dist_atr": "dist_atr",
                  "mass": "mass", "tf": "tf"},
)

_SMC = dict(
    table="setups", ts="closed_at", closed_filter="realized_r IS NOT NULL",
    r="realized_r", pnl="realized_pnl", symbol="symbol", side="side",
    entry="real_fill_price",
    open_filter="real_fill_time IS NOT NULL AND closed_at IS NULL",
    open_ts="real_fill_time", open_entry="real_fill_price", open_sl="stop",
    open_tp="tp", open_qty="qty", family="SMC", exit_reason="close_reason",
    recorded_pit={"regime5": "regime5", "fv_node_density": "fv_node_density",
                  "fv_dist_ppoc_atr": "fv_dist_ppoc_atr"},
)

_DEPTH_ORDERS = dict(
    table="orders", ts="closed_at_utc",
    closed_filter="status='CLOSED' AND r_net IS NOT NULL",
    r="r_net", pnl="realized_pnl", symbol="symbol", side="direction",
    entry="COALESCE(avg_entry,entry)", exit="avg_exit",
    open_filter="filled_at_utc IS NOT NULL AND closed_at_utc IS NULL",
    open_ts="filled_at_utc", open_entry="COALESCE(avg_entry,entry)",
    open_sl="COALESCE(trail_sl,sl)", open_tp="tp", open_qty="qty", open_risk="risk_usd",
    working_filter="status IN ('OPEN','PLACED','NEW') AND filled_at_utc IS NULL",
    working_ts="opened_at_utc", family="Depth", exit_reason="note",
)

_PAPER = dict(  # lrr / depth virtual books
    table="paper_trades", ts="closed_at_utc", r="r_net", pnl="pnl_usd",
    symbol="asset", side="direction", entry="entry_price",
    open_filter="status='OPEN'", open_ts="opened_at_utc", open_entry="entry_price",
    open_sl="sl", open_tp="tp", open_risk="risk_usd", exit_reason="exit_reason",
)

_LR_TRADES = dict(
    table="trades", ts="exit_timestamp", closed_filter=_LR_F, r=_LR_R,
    pnl="realized_pnl", side="signal_type", entry="entry_price", exit="exit_price",
    open_filter="status='open'", open_ts="timestamp", open_entry="entry_price",
    open_sl="stop_loss", open_tp="take_profit", open_qty="position_size",
    family="Liquidity Raid", exit_reason="COALESCE(exit_reason, reason)",
)

_OPT = dict(  # option seats book $ only — no R (the risk leg is the spread)
    table="trades", ts="exit_timestamp", closed_filter="realized_pnl IS NOT NULL",
    pnl="realized_pnl", side="status",
    open_filter="status='open'", open_ts="timestamp",
    family="Options", exit_reason="exit_reason",
)


def _knife(key: str, label: str, db: str, recap: bool = True,
           seat: Optional[str] = None) -> Book:
    return Book(key=key, label=label, tier=1, db=f"HyroTrader/{db}",
                seat=seat, in_recap=recap, **_KNIFE)


def _seats(key: str, label: str, db: str, recap: bool = False,
           seat: Optional[str] = None, family: str = "Level Seats", **kw) -> Book:
    return Book(key=key, label=label, tier=1, db=db, seat=seat, in_recap=recap,
                family=family, **{**_SEATS, **kw})


def _lr(key: str, label: str, db: str, recap: bool = False) -> Book:
    # Every LR seat authenticates with HyroTrader/.env — one key, many symbols.
    return Book(key=key, label=label, tier=1, db=f"HyroTrader/{db}",
                symbol=f"'{label.split('-')[-1].upper()}'", seat="HyroTrader/HyroTrader",
                in_recap=recap, **_LR_TRADES)


def _opt(key: str, label: str, db: str, recap: bool = False,
         seat: Optional[str] = None) -> Book:
    # The option books hold one structure per row and name the underlying only in
    # the file, so the symbol comes from the seat label.
    under = label.split("-")[1].split("(")[0].upper()
    return Book(key=key, label=label, tier=1, db=f"HyroTrader/{db}",
                symbol=f"'{under}'", seat=seat, reconcilable=False,
                in_recap=recap, **_OPT)


# ══════════════════════════════════════════════════════════════════════════════
#  The roster
# ══════════════════════════════════════════════════════════════════════════════
BOOKS: list[Book] = [
    # ── Tier 1 · knife funded arms ───────────────────────────────────────────
    _knife("knife_100k", "knife-funded-100k", "knife_bybit_funded_100k.db",
            seat="HyroTrader/knife_funded_100k"),
    _knife("knife_10k", "knife-funded-10k", "knife_bybit_funded_challenge.db",
            seat="HyroTrader/knife_funded_challenge"),
    _knife("knife_taker", "knife-funded-taker", "knife_bybit_funded_taker.db",
            seat="HyroTrader/knife_funded_taker"),
    _knife("knife_maker2", "knife-funded-maker2", "knife_bybit_funded_maker2.db",
            seat="HyroTrader/knife_funded_maker2"),
    _knife("knife_ethmstop", "knife-funded-ethmstop", "knife_bybit_funded_ethmakerstop.db",
            seat="HyroTrader/knife_funded_ethmakerstop"),
    _knife("knife_maker1", "knife-funded-maker1", "knife_bybit_funded.db", recap=False),

    # ── Tier 1 · Liquidity Raid funded + demo seats ──────────────────────────
    _lr("lr_funded_btc", "lr-funded-btc", "lr_bybit_funded_10k.db", recap=True),
    _lr("lr_funded_eth", "lr-funded-eth", "lr_eth_bybit_funded_10k.db", recap=True),
    _lr("lr_funded_sol", "lr-funded-sol", "lr_sol_bybit_funded_10k.db", recap=True),
    _lr("lr_funded_avax", "lr-funded-avax", "lr_avax_bybit_funded_10k.db"),
    _lr("lr_funded_doge", "lr-funded-doge", "lr_doge_bybit_funded_10k.db"),
    _lr("lr_funded_dot", "lr-funded-dot", "lr_dot_bybit_funded_10k.db"),
    _lr("lr_funded_link", "lr-funded-link", "lr_link_bybit_funded_10k.db"),
    _lr("lr_demo_btc", "lr-demo-btc", "lr_bybit_btc.db", recap=True),
    _lr("lr_demo_eth", "lr-demo-eth", "lr_bybit_eth.db"),
    _lr("lr_demo_sol", "lr-demo-sol", "lr_bybit_sol.db", recap=True),
    _lr("lr_demo_avax", "lr-demo-avax", "lr_bybit_avax.db"),
    _lr("lr_demo_doge", "lr-demo-doge", "lr_bybit_doge.db"),
    _lr("lr_demo_dot", "lr-demo-dot", "lr_bybit_dot.db"),
    _lr("lr_demo_link", "lr-demo-link", "lr_bybit_link.db"),

    # ── Tier 1 · other order-placing seats ───────────────────────────────────
    Book(key="momentum_ada", label="momentum-4h-ada", tier=1, in_recap=True,
         db="HyroTrader/momentum_4h_ada.db", table="trades", ts="closed_at_utc",
         closed_filter="r_multiple IS NOT NULL", r="r_multiple", pnl="pnl_usd",
         symbol="'ADA'", side="direction", entry="entry", exit="exit_price",
         open_filter="opened_at_utc IS NOT NULL AND closed_at_utc IS NULL",
         open_ts="opened_at_utc", open_entry="entry", open_sl="sl", open_tp="tp",
         open_qty="qty", open_risk="risk_usd", family="Momentum 4H",
         exit_reason="exit_reason"),
    Book(key="mm_demo_btc", label="mm-demo-btc", tier=1,
         db="HyroTrader/mm_bybit_btc.db", **{**_LR_TRADES, "family": "Momentum Mastery"},
         symbol="'BTC'", seat="HyroTrader/HyroTrader"),
    Book(key="displacement_btc", label="displacement-btc", tier=1,
         db="HyroTrader/displacement_bybit_btc.db", table="displacement_trades",
         ts="closed_at_utc", closed_filter="r_net IS NOT NULL", r="r_net",
         symbol="symbol", side="direction", entry="entry_price", exit="exit_price",
         open_filter="closed_at_utc IS NULL", open_ts="opened_at_utc",
         open_entry="entry_price", open_sl="stop_price", open_tp="target_price",
         open_qty="bybit_qty", family="Displacement", exit_reason="exit_reason",
         seat="HyroTrader/HyroTrader"),
    _seats("desk_demo", "desk-demo", "desk_demo/desk_demo.db", recap=True,
           seat="desk_demo/desk_demo", family="Desk"),
    _seats("retest_demo", "retest-demo", "retest_demo/retest_demo.db", recap=True,
           seat="retest_demo/retest_demo", family="Retest"),
    _seats("lr_wide_demo", "lr-wide-demo", "lr_wide_demo/lr_wide_demo.db",
           seat="lr_wide_demo/lr_wide_demo", family="Liquidity Raid",
           open_sl="sl", open_tp="tp", entry="fill_px"),
    _seats("lrr_short_demo", "lrr-short-demo", "lrr_short_demo/lrr_short_demo.db",
           seat="lrr_short_demo/lrr_short_demo", family="LRR"),
    Book(key="smc_demo", label="smc-demo", tier=1, in_recap=True,
         db="smc_demo/smc_demo.db", seat="smc_demo/smc_demo", **_SMC),
    Book(key="smc12h4h_demo", label="smc-12h4h-demo", tier=1,
         db="smc12h4h_demo/smc12h4h_demo.db", seat="smc12h4h_demo/smc12h4h_demo",
         **_SMC),
    Book(key="depth_policy_taker", label="depth-policy-taker", tier=1, in_recap=True,
         db="HyroTrader/depth_policy_taker_book.db",
         seat="HyroTrader/depth_policy_taker", **_DEPTH_ORDERS),
    Book(key="depth_policy_maker", label="depth-policy-maker", tier=1, in_recap=True,
         db="HyroTrader/depth_policy_maker_book.db",
         seat="HyroTrader/depth_policy_maker", **_DEPTH_ORDERS),
    Book(key="depth_taker", label="depth-taker(v1)", tier=1,
         db="HyroTrader/depth_taker_book.db", seat="HyroTrader/depth_taker",
         **_DEPTH_ORDERS),
    Book(key="depth_maker", label="depth-maker(v1)", tier=1,
         db="HyroTrader/depth_maker_book.db", seat="HyroTrader/depth_demo",
         **_DEPTH_ORDERS),
    Book(key="ofcs_demo", label="ofcs-demo/challenge", tier=1,
         seat="ofcs_demo/ofcs_demo",
         db="ofcs_demo/ofcs_demo.db", table="trades", ts="exit_ts",
         closed_filter="realized_r IS NOT NULL", r="realized_r", pnl="realized_pnl",
         symbol="symbol", side="direction", entry="avg_fill_price", exit="exit_price",
         open_filter="status='filled'", open_ts="fill_ts",
         open_entry="avg_fill_price", open_sl="sl", open_tp="tp", open_qty="final_qty",
         working_filter="status='placed' AND fill_ts IS NULL", working_ts="placed_at",
         family="OFCS", exit_reason="close_reason",
         recorded_pit={"regime5": "regime5", "htf_dir": "htf_dir", "cell": "cell",
                       "absorption_proxy": "absorption_proxy", "cross_count": "cross_count",
                       "ml_p": "ml_p", "era": "era"},
         note="one book, era-scoped: challenge rules from 2026-09-01"),
    Book(key="london_raid", label="london-raid-demo", tier=1,
         seat="london_raid_demo/london_raid_demo",
         db="london_raid_demo/london_raid_demo.db", table="orders", ts="exit_ts",
         closed_filter="realized_r IS NOT NULL", r="realized_r", pnl="realized_pnl",
         symbol="symbol", side="direction", entry="avg_fill_price", exit="exit_price",
         open_filter="avg_fill_price IS NOT NULL AND exit_ts IS NULL",
         open_ts="fill_ts", open_entry="avg_fill_price", open_sl="sl", open_tp="tp",
         open_qty="qty", open_risk="risk_usd",
         working_filter="order_id IS NOT NULL AND fill_ts IS NULL AND status NOT IN "
                        "('cancelled','closed','orphaned')", working_ts="placed_at",
         family="London Raid", exit_reason="note",
         recorded_pit={"asia_high": "asia_high", "asia_low": "asia_low",
                       "asia_bars": "asia_bars", "sl_dist": "sl_dist", "atr": "atr"}),
    Book(key="london_raid_taker", label="london-raid-taker", tier=1,
         db="london_raid_taker_demo/london_raid_taker_demo.db", table="trades",
         ts="exit_ts", closed_filter="realized_r IS NOT NULL", r="realized_r",
         pnl="realized_pnl", symbol="symbol", side="direction",
         entry="avg_fill_price", exit="exit_price",
         open_filter="avg_fill_price IS NOT NULL AND exit_ts IS NULL",
         open_ts="fill_ts", open_entry="avg_fill_price", open_sl="sl", open_tp="tp",
         open_qty="qty", open_risk="risk_usd", family="London Raid", exit_reason="note"),
    Book(key="ferryman", label="ferryman-demo", tier=1,
         seat="HyroTrader/ferryman_demo",
         db="HyroTrader/ferryman_demo.db", table="orders", ts="exit_at",
         closed_filter="status='CLOSED' AND r_mult IS NOT NULL", r="r_mult", pnl="pnl",
         symbol="symbol", side="side", entry="fill_px", exit="exit_px",
         open_filter="fill_at IS NOT NULL AND exit_at IS NULL", open_ts="fill_at",
         open_entry="fill_px", open_sl="sl", open_qty="qty", open_risk="risk_usd",
         working_filter="order_id IS NOT NULL AND fill_at IS NULL AND status NOT IN "
                        "('EXPIRED','REJECTED','CLOSED','SKIPPED_CAP','SKIPPED_BUSY')",
         working_ts="placed_at", family="Ferryman", exit_reason="exit_reason"),
    Book(key="fvg_alts", label="fvg-alts-demo", tier=1, seat="fvgalt",
         db="HyroTrader/fvg_alts_demo.db", table="trades", ts="exit_ts",
         closed_filter="realized_r IS NOT NULL", r="realized_r", pnl="realized_pnl",
         symbol="symbol", side="direction", entry="avg_fill_price", exit="exit_price",
         open_filter="fill_ts IS NOT NULL AND exit_ts IS NULL", open_ts="fill_ts",
         open_entry="avg_fill_price", open_sl="sl", open_tp="tp1",
         open_qty="filled_qty",
         working_filter="status='placed' AND fill_ts IS NULL", working_ts="placed_at",
         family="FVG Alts", exit_reason="close_reason"),
    Book(key="funding_carry", label="funding-carry-demo", tier=1,
         seat="funding_carry_demo/funding_carry_demo",
         db="funding_carry_demo/funding_carry_demo.db", table="holds", ts="closed_at",
         closed_filter="status='closed' AND realized_pnl IS NOT NULL",
         pnl="realized_pnl", symbol="symbol", entry="perp_fill_px",
         open_filter="status='open'", open_ts="opened_at", open_entry="perp_fill_px",
         open_qty="q", family="Funding Carry", exit_reason="close_reason",
         note="cash-and-carry: $ only, no stop-defined R"),
    Book(key="phantom", label="phantom-conductor", tier=1,
         db="phantom_conductor/phantom_conductor.db", table="mirrors", ts="closed_at",
         closed_filter="r_geom IS NOT NULL", r="r_geom", pnl="pnl_usd",
         symbol="symbol", side="direction", entry="entry_fill", exit="exit_fill",
         open_filter="status='open'", open_ts="opened_at", open_entry="entry_fill",
         open_sl="sl", open_tp="tp", open_qty="qty", open_risk="risk_usd",
         family="Phantom", exit_reason="close_note"),
    _opt("bullput_btc", "bullput-btc(opt)", "bullput_btc_bybit.db", recap=True,
         seat="bullput"),
    _opt("bullput_eth", "bullput-eth(opt)", "bullput_eth_bybit.db", recap=True,
         seat="bullput"),
    _opt("ironfly_btc", "ironfly-btc(opt)", "ironfly_btc_bybit.db", recap=True,
         seat="ironfly"),
    _opt("ironfly_eth", "ironfly-eth(opt)", "ironfly_eth_bybit.db", recap=True,
         seat="ironfly"),
    _opt("calendar_btc", "calendar-btc(opt)", "calendar_btc_bybit.db", recap=True),
    _opt("bullput_btc_10k", "bullput-btc-funded", "bullput_btc_funded_10k.db"),
    _opt("bullput_eth_10k", "bullput-eth-funded", "bullput_eth_funded_10k.db"),

    # ── Tier 2 · LR paper signal books (Liquidity_Raid/<SYM>_V2, no exchange) ──
    # NQ is deliberately absent: the legacy dashboard map already reads it as
    # `lr_nq.db`, and a book counted twice is a book counted wrong.
    *[Book(key=f"lr_paper_{s_.lower()}", label=f"lr-paper-{s_.lower()}", tier=2,
           db=f"Liquidity_Raid/{s_}_V2/{s_.lower()}_liquidity_raid_v2.db",
           symbol=f"'{s_}'", **_LR_TRADES)
      for s_ in ("BTC", "ETH", "SOL", "AVAX", "BCH", "BNB", "DOGE", "DOT", "LINK", "XRP")],

    # ── Tier 2 · paper / virtual books ───────────────────────────────────────
    Book(key="lrr_paper", label="lrr-paper", tier=2, in_recap=True,
         db="HyroTrader/lrr_paper_book.db", closed_filter="r_net IS NOT NULL",
         family="LRR", **_PAPER),
    Book(key="depth_paper", label="depth-paper(fresh)", tier=2, in_recap=True,
         db="HyroTrader/depth_paper_book.db",
         closed_filter="r_net IS NOT NULL AND COALESCE(stale_signal,0)=0 AND "
                       "COALESCE(exit_reason,'') NOT IN ('POLICY_UNSCORABLE','SOURCE_GONE')",
         family="Depth", **_PAPER),
    Book(key="depth_policy_paper", label="depth-policy-paper(fresh)", tier=2, in_recap=True,
         db="HyroTrader/depth_policy_paper_book.db",
         closed_filter="r_net IS NOT NULL AND COALESCE(stale_signal,0)=0 AND "
                       "COALESCE(exit_reason,'') NOT IN ('POLICY_UNSCORABLE','SOURCE_GONE')",
         family="Depth", **_PAPER),
    Book(key="ofcs_paper_e1", label="ofcs-paper(gross,era1)", tier=2, in_recap=True,
         db="ofcs_shadow/ofcs_paper_book.db", table="ofcs_paper_trades",
         ts="replace(resolved_at,' UTC','')",
         closed_filter="realized_r IS NOT NULL AND COALESCE(era,1)<2",
         r="realized_r", symbol="asset", side="direction", entry="entry",
         exit="exit_price", family="OFCS", entry_ts="entry_ts", exit_reason="exit_reason"),
    Book(key="ofcs_paper_e2", label="ofcs-paper(net,era2)", tier=2, in_recap=True,
         db="ofcs_shadow/ofcs_paper_book.db", table="ofcs_paper_trades",
         ts="replace(resolved_at,' UTC','')",
         closed_filter="realized_r IS NOT NULL AND era=2",
         r="realized_r", symbol="asset", side="direction", entry="entry",
         exit="exit_price",
         open_filter="realized_r IS NULL AND era=2 AND status NOT IN ('rejected','skipped')",
         open_ts="entry_ts", open_entry="COALESCE(effective_entry,entry)",
         open_sl="sl", open_tp="tp", open_qty="qty", family="OFCS",
         exit_reason="exit_reason"),
    Book(key="analyst_drift", label="analyst-drift(alpaca)", tier=2,
         seat="analyst_drift_paper", reconcilable=False,
         db="analyst_drift_paper/analyst_paper.db", table="events", ts="resolved_at",
         closed_filter="status='closed' AND pnl_usd IS NOT NULL", pnl="pnl_usd",
         symbol="symbol", entry="entry_fill", exit="exit_fill",
         open_filter="status='slated'", open_ts="created_utc", open_qty="qty",
         family="Analyst Drift", exit_reason="reason",
         recorded_pit={"sent": "sent", "pit_rank": "pit_rank", "pit_quarter": "pit_quarter",
                       "beta": "beta", "oc_spy": "oc_spy", "era": "era", "eff_date": "eff_date"},
         recorded_post={"m_gross": "m_gross", "m_fill": "m_fill",
                        "slip_open_bps": "slip_open_bps", "slip_close_bps": "slip_close_bps"},
         note="US equities, Alpaca paper — $ book, no stop-defined R"),

    # ── Tier 3 · shadow / record-only (dimensionless R) ──────────────────────
    Book(key="knife_shadow", label="knife-shadow", tier=3, in_recap=True,
         db="HyroTrader/knife_shadow.db", table="episodes", ts="resolved_at_utc",
         closed_filter="r_net IS NOT NULL", r="r_net", symbol="symbol",
         side="direction", entry="fill_price", exit="exit_price",
         open_filter="fill_ts IS NOT NULL AND resolved_at_utc IS NULL",
         open_ts="fill_ts", open_entry="fill_price", open_sl="sl", open_tp="tp"),
    Book(key="antiknife", label="antiknife-shadow", tier=3, in_recap=True,
         db="HyroTrader/antiknife_shadow.db", table="anti", ts="booked_at",
         closed_filter="r_net IS NOT NULL", r="r_net", symbol="symbol",
         side="direction", entry="fill_price"),
    Book(key="crossvenue", label="knife-crossvenue", tier=3, in_recap=True,
         db="HyroTrader/crossvenue_shadow.db", table="cv", ts="tagged_at",
         closed_filter="r_net IS NOT NULL", r="r_net", symbol="symbol",
         side="direction", entry="level"),
    Book(key="mm_5m_shadow", label="mm-5m-shadow", tier=3, in_recap=True,
         db="HyroTrader/mm_5m_shadow.db", table="signals", ts="closed_at_utc",
         closed_filter="r_net IS NOT NULL", r="r_net", symbol="symbol",
         side="direction", entry="limit_price", exit="exit_price",
         open_filter="fill_bar_utc IS NOT NULL AND closed_at_utc IS NULL",
         open_ts="fill_bar_utc", open_entry="limit_price",
         open_sl="stop_loss", open_tp="tp1"),
    Book(key="mm_15m_shadow", label="mm-15m-shadow", tier=3, in_recap=True,
         db="HyroTrader/mm_15m_shadow.db", table="signals", ts="closed_at_utc",
         closed_filter="r_net IS NOT NULL", r="r_net", symbol="symbol",
         side="direction", entry="effective_entry", exit="exit_price",
         open_filter="fill_touched_utc IS NOT NULL AND closed_at_utc IS NULL",
         open_ts="fill_touched_utc", open_entry="effective_entry",
         open_sl="stop_loss", open_tp="tp1"),
    Book(key="sweep_engine", label="sweep-engine(gross)", tier=3, in_recap=True,
         db="sweep_engine/sweep_engine.db", table="events", ts="event_time",
         closed_filter="r_gross IS NOT NULL AND COALESCE(is_alias,0)=0 AND "
                       "COALESCE(feed,'binance_us')='bybit'",
         r="r_gross", symbol="symbol",
         # this book stores side as ±1; spell it out so the ledger reads like
         # every other row (and so the column stays one type)
         side="CASE WHEN side>0 THEN 'LONG' WHEN side<0 THEN 'SHORT' ELSE '' END",
         entry="entry", note="GROSS R — the fee/slip toll is not deducted"),
    Book(key="fib618", label="fib618-shadow(taker)", tier=3, in_recap=True,
         db="fib618_shadow/fib618_shadow.db", table="trades", ts="exit_time",
         closed_filter="net_taker IS NOT NULL", r="net_taker", symbol="symbol",
         side="side", entry="entry"),
    Book(key="gated_lr", label="gated-lr-shadow", tier=3, in_recap=True,
         db="HyroTrader/gated_lr_shadow.db", table="gated_signals", ts="closed_at_utc",
         closed_filter="r_net IS NOT NULL", r="r_net", symbol="sym", side="direction"),
    Book(key="lr_signal_e1", label="lr-signal(gross,era1)", tier=3, in_recap=True,
         db="HyroTrader/lr_shadow_trades.db", table="shadow_trades", ts="closed_at_utc",
         closed_filter="r_multiple IS NOT NULL AND COALESCE(era,1)<2", r="r_multiple",
         side="direction", entry="entry_price", exit="exit_price"),
    Book(key="lr_signal_e2", label="lr-signal(net,era2)", tier=3, in_recap=True,
         db="HyroTrader/lr_shadow_trades.db", table="shadow_trades", ts="closed_at_utc",
         closed_filter="r_net IS NOT NULL AND era=2", r="r_net",
         side="direction", entry="entry_price", exit="exit_price",
         open_filter="era=2 AND closed_at_utc IS NULL AND entry_fill_at IS NOT NULL",
         open_ts="entry_fill_at", open_entry="entry_price", open_sl="stop_loss",
         open_tp="take_profit"),
    Book(key="asia_basket", label="asia-basket(gross)", tier=3, in_recap=True,
         db="HyroTrader/asia_basket_shadow.db", table="basket_nights", ts="night",
         closed_filter="mean_r IS NOT NULL", r="mean_r",
         note="one row = one night's basket, not one trade"),
    Book(key="ifvg_sweep", label="ifvg-sweep-shadow", tier=3, in_recap=True,
         db="HyroTrader/ifvg_sweep_shadow.db", table="shadow_trades", ts="closed_at_utc",
         closed_filter="r_multiple IS NOT NULL", r="r_multiple"),
    Book(key="wide_rr", label="wide-rr-shadow", tier=3,
         db="HyroTrader/wide_rr_shadow.db", table="wide_rr", ts="resolved_at_utc",
         closed_filter="live_r_net IS NOT NULL", r="live_r_net", symbol="symbol",
         side="direction", entry="entry",
         open_filter="resolved_at_utc IS NULL AND entry IS NOT NULL",
         open_ts="opened_at_utc", open_entry="entry", open_sl="sl", open_tp="live_tp"),
    # ERA-2 ONLY. The era-1 rows of this book are a CLOSED phantom: 96% of them
    # were booked at prices the market had already left (+6,603R on fills that
    # never existed — a scorecard filled in after the whistle). Only the 08-29
    # fill-adjudicating resolver (era=2) is honest, so era-1 is not surfaced.
    Book(key="halt_shadow", label="halt-shadow(era2)", tier=3,
         db="HyroTrader/halt_shadow_book.db", table="halt_shadows",
         ts="closed_at_utc", closed_filter="r_net IS NOT NULL AND era=2", r="r_net",
         symbol="symbol", side="direction", entry="entry", exit="exit_price",
         open_filter="era=2 AND closed_at_utc IS NULL AND status='OPEN'",
         open_ts="recorded_at_utc", open_entry="entry", open_sl="sl", open_tp="tp",
         note="what a HALTED seat would have taken; era-1 excluded (phantom fills)"),
    Book(key="lr_gate_fleet", label="lr-gate-shadow(fleet)", tier=3, in_recap=True,
         db="shadow_books/*.db", table="shadow_trades", ts="closed_at_utc",
         closed_filter="realized_r IS NOT NULL", r="realized_r",
         side="direction", entry="entry_price", exit="exit_price",
         note="20 gate books summed into one line"),
]

BOOKS_BY_KEY = {b.key: b for b in BOOKS}


# ══════════════════════════════════════════════════════════════════════════════
#  SQL builders — the ONLY place a statement is assembled. All read-only.
# ══════════════════════════════════════════════════════════════════════════════
def _expr(e: Optional[str]) -> str:
    return e if e else "NULL"


def _within(ts: str, days: int) -> str:
    return f"datetime({ts})>=datetime('now','-{int(days)} day')"


def build_agg_sql(b: Book, days: int = 7) -> str:
    """Lifetime + rolling-window counts, ΣR, mean R, wins and Σ$ in one row."""
    r, pnl, ts = _expr(b.r), _expr(b.pnl), b.ts
    win = f"({r})>0" if b.r else (f"({pnl})>0" if b.pnl else "0")
    w = _within(ts, days)
    return (
        "SELECT COUNT(*) AS n,"
        f" AVG({r}) AS mean_r,"
        f" SUM({r}) AS sum_r,"
        f" SUM(CASE WHEN {win} THEN 1 ELSE 0 END) AS wins,"
        f" SUM({pnl}) AS sum_pnl,"
        f" SUM(CASE WHEN {w} THEN 1 ELSE 0 END) AS n_w,"
        f" SUM(CASE WHEN {w} THEN ({r}) ELSE 0 END) AS sum_r_w,"
        f" SUM(CASE WHEN {w} AND {win} THEN 1 ELSE 0 END) AS wins_w,"
        f" SUM(CASE WHEN {w} THEN ({pnl}) ELSE 0 END) AS sum_pnl_w"
        f" FROM {b.table} WHERE {b.closed_filter}"
    )


def build_trades_sql(b: Book, days: int = 7, limit: int = 500) -> str:
    """Every trade this book closed inside the window, newest first."""
    return (
        f"SELECT {b.ts} AS ts, {_expr(b.symbol)} AS symbol, {_expr(b.side)} AS side,"
        f" {_expr(b.entry)} AS entry, {_expr(b.exit)} AS exit_px,"
        f" {_expr(b.r)} AS r, {_expr(b.pnl)} AS pnl"
        f" FROM {b.table} WHERE {b.closed_filter} AND {_within(b.ts, days)}"
        f" ORDER BY datetime({b.ts}) DESC LIMIT {int(limit)}"
    )


_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _recorded_block(b: Book, available_cols=None) -> str:
    """The `pit__` / `post__` projection, limited to columns the book HAS.

    Books of one shape drift apart over time (an older knife arm predates the
    OI columns; lr_wide's seats never had `ltype`). A recorded column is a
    bonus, never a requirement: when `available_cols` is given, a bare-name
    expression that is not in it is dropped rather than breaking the read.
    """
    def keep(expr: str) -> bool:
        if available_cols is None:
            return True
        return (not _IDENT.match(expr)) or (expr in available_cols)
    extra = "".join(f", {v} AS pit__{k}" for k, v in b.recorded_pit.items() if keep(v))
    extra += "".join(f", {v} AS post__{k}" for k, v in b.recorded_post.items() if keep(v))
    return extra


def build_history_sql(b: Book, available_cols=None) -> str:
    """Every closed row with the unified-schema fields — no window, no cap.

    Meant for a LOCAL synced copy (the unified trade pages), never for the SSH
    round trip: it is the whole ledger, not this week's page of it.
    `available_cols` (the table's PRAGMA column set) makes the recorded
    pit/post block tolerant of books that lack some of those columns.
    """
    entry_ts = b.entry_ts or b.open_ts
    extra = _recorded_block(b, available_cols)
    return (
        f"SELECT {_expr(entry_ts)} AS entry_ts, {b.ts} AS exit_ts,"
        f" {_expr(b.symbol)} AS symbol, {_expr(b.side)} AS side,"
        f" {_expr(b.entry)} AS entry, {_expr(b.exit)} AS exit_px,"
        f" {_expr(b.open_sl)} AS sl, {_expr(b.open_tp)} AS tp,"
        f" {_expr(b.open_qty)} AS qty, {_expr(b.open_risk)} AS risk_usd,"
        f" {_expr(b.r)} AS r, {_expr(b.pnl)} AS pnl,"
        f" {_expr(b.exit_reason)} AS exit_reason{extra}"
        f" FROM {b.table} WHERE {b.closed_filter}"
        f" ORDER BY datetime({b.ts})"
    )


def build_fills_sql(b: Book, since_ts: Optional[str] = None, limit: int = 5000,
                    available_cols=None) -> str:
    """Every FILLED row (open or closed) newer than a watermark — the fleet
    feature spine's intake query. Same projection as the history query so the
    two never disagree about a column, plus a stable per-row key."""
    entry_ts = b.entry_ts or b.open_ts
    key = b.row_key or "rowid"
    extra = _recorded_block(b, available_cols)
    where = f"{_expr(entry_ts)} IS NOT NULL"
    if b.open_filter or b.closed_filter != "1=1":
        parts = [f"({b.closed_filter})"] if b.closed_filter != "1=1" else []
        if b.open_filter:
            parts.append(f"({b.open_filter})")
        where += " AND (" + " OR ".join(parts) + ")"
    if since_ts:
        where += f" AND datetime({_expr(entry_ts)}) > datetime('{since_ts}')"
    return (
        f"SELECT {key} AS row_key, {_expr(entry_ts)} AS entry_ts, {b.ts} AS exit_ts,"
        f" {_expr(b.symbol)} AS symbol, {_expr(b.side)} AS side,"
        f" {_expr(b.entry)} AS entry, {_expr(b.exit)} AS exit_px,"
        f" {_expr(b.open_sl)} AS sl, {_expr(b.open_tp)} AS tp,"
        f" {_expr(b.open_qty)} AS qty, {_expr(b.open_risk)} AS risk_usd,"
        f" {_expr(b.r)} AS r, {_expr(b.pnl)} AS pnl,"
        f" {_expr(b.exit_reason)} AS exit_reason{extra}"
        f" FROM {b.table} WHERE {where}"
        f" ORDER BY datetime({_expr(entry_ts)}) LIMIT {int(limit)}"
    )


def build_bucket_sql(b: Book, days: int = 7) -> str:
    """Per-HOUR totals inside the window — aggregated ON the VPS.

    The trade dump is capped per book so a chatty shadow recorder cannot flood
    the wire; charting off that cap would quietly under-count it. This query
    counts every row and ships one line per hour — the till roll, not the
    receipts. Hourly is the finest bucket the page offers; coarser ones (4h, day,
    week) are rolled up from these locally, so changing the zoom costs no extra
    round trip.
    """
    r, pnl = _expr(b.r), _expr(b.pnl)
    bucket = f"strftime('%Y-%m-%dT%H:00', {b.ts})"
    return (
        f"SELECT {bucket} AS bucket, COUNT(*) AS n,"
        f" SUM({r}) AS sum_r, SUM({pnl}) AS sum_pnl"
        f" FROM {b.table} WHERE {b.closed_filter} AND {_within(b.ts, days)}"
        f" GROUP BY {bucket} ORDER BY bucket"
    )


def build_open_sql(b: Book, limit: int = 200) -> Optional[str]:
    """Running entries: filled-and-live, plus any order still working."""
    parts = []
    if b.open_filter:
        parts.append(
            f"SELECT 'FILLED' AS state, {_expr(b.open_ts)} AS since,"
            f" {_expr(b.symbol)} AS symbol, {_expr(b.side)} AS side,"
            f" {_expr(b.open_entry)} AS entry, {_expr(b.open_sl)} AS sl,"
            f" {_expr(b.open_tp)} AS tp, {_expr(b.open_qty)} AS qty,"
            f" {_expr(b.open_risk)} AS risk_usd"
            f" FROM {b.table} WHERE {b.open_filter}"
        )
    if b.working_filter:
        age = (f" AND datetime({b.working_ts})>=datetime('now','-{int(b.working_max_age_h)} hour')"
               if b.working_ts and b.working_max_age_h else "")
        parts.append(
            f"SELECT 'WORKING' AS state, {_expr(b.working_ts)} AS since,"
            f" {_expr(b.symbol)} AS symbol, {_expr(b.side)} AS side,"
            f" {_expr(b.open_entry)} AS entry, {_expr(b.open_sl)} AS sl,"
            f" {_expr(b.open_tp)} AS tp, {_expr(b.open_qty)} AS qty,"
            f" {_expr(b.open_risk)} AS risk_usd"
            f" FROM {b.table} WHERE {b.working_filter}{age}"
        )
    if not parts:
        return None
    return " UNION ALL ".join(parts) + f" ORDER BY since DESC LIMIT {int(limit)}"


def build_spec(days: int = 7, trade_limit: int = 500,
               open_limit: int = 200) -> list[dict]:
    """The query plan handed to the remote collector — SELECTs only."""
    plan = []
    for b in BOOKS:
        plan.append({"id": f"{b.key}::agg", "db": b.db, "sql": build_agg_sql(b, days)})
        plan.append({"id": f"{b.key}::trades", "db": b.db,
                     "sql": build_trades_sql(b, days, trade_limit)})
        plan.append({"id": f"{b.key}::buckets", "db": b.db,
                     "sql": build_bucket_sql(b, days)})
        osql = build_open_sql(b, open_limit)
        if osql:
            plan.append({"id": f"{b.key}::open", "db": b.db, "sql": osql})
    return plan


# Families present in the fleet, in roster order — the unified universe's
# strategy list (Tier 1 + Tier 2 only; shadows are dimensionless by design).
FAMILIES: list[str] = []
for _b in BOOKS:
    if _b.tier <= 2 and _b.family and _b.family not in FAMILIES:
        FAMILIES.append(_b.family)


def family_symbols(family: str) -> list[str]:
    """Symbols a family trades, read off its books' literal symbol columns."""
    out: list[str] = []
    for _b in BOOKS:
        if _b.family == family and _b.symbol and _b.symbol.startswith("'"):
            sym = _b.symbol.strip("'")
            if sym not in out:
                out.append(sym)
    return out
