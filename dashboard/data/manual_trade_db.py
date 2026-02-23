"""SQLite CRUD for the separate manual-trades database."""

import sqlite3
import uuid
from datetime import datetime, timezone

from config import MANUAL_TRADES_DB

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS manual_trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id        TEXT    UNIQUE NOT NULL,
    symbol          TEXT    NOT NULL,
    direction       TEXT    NOT NULL,
    status          TEXT    NOT NULL DEFAULT 'open',
    entry_time      TEXT,
    exit_time       TEXT,
    entry_price     REAL,
    exit_price      REAL,
    quantity        REAL    NOT NULL,
    stop_loss       REAL,
    take_profit     REAL,
    entry_order_id  TEXT,
    exit_order_id   TEXT,
    sl_order_id     TEXT,
    tp_order_id     TEXT,
    pnl_usd         REAL,
    pnl_pct         REAL,
    fees_usd        REAL    DEFAULT 0.0,
    exit_reason     TEXT,
    notes           TEXT,
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL
);
"""


def _conn() -> sqlite3.Connection:
    MANUAL_TRADES_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(MANUAL_TRADES_DB))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the manual_trades table if it does not exist."""
    with _conn() as conn:
        conn.execute(_CREATE_TABLE)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def insert_trade(
    symbol: str,
    direction: str,
    quantity: float,
    entry_price: float | None = None,
    entry_time: str | None = None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    entry_order_id: str | None = None,
    fees_usd: float = 0.0,
    notes: str | None = None,
) -> str:
    """Insert a new trade. Returns the generated trade_id."""
    trade_id = uuid.uuid4().hex[:12]
    now = _now()
    with _conn() as conn:
        conn.execute(
            """INSERT INTO manual_trades
               (trade_id, symbol, direction, status, entry_time, entry_price,
                quantity, stop_loss, take_profit, entry_order_id, fees_usd,
                notes, created_at, updated_at)
               VALUES (?, ?, ?, 'open', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (trade_id, symbol, direction, entry_time or now, entry_price,
             quantity, stop_loss, take_profit, entry_order_id, fees_usd,
             notes, now, now),
        )
    return trade_id


def update_trade(trade_id: str, **fields) -> None:
    """Update arbitrary fields on a trade."""
    if not fields:
        return
    fields["updated_at"] = _now()
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [trade_id]
    with _conn() as conn:
        conn.execute(
            f"UPDATE manual_trades SET {set_clause} WHERE trade_id = ?",
            values,
        )


def close_trade(
    trade_id: str,
    exit_price: float,
    exit_reason: str,
    exit_order_id: str | None = None,
    fees_usd: float = 0.0,
) -> None:
    """Close an open trade: calculate P&L and set status to 'closed'."""
    trade = get_trade_by_id(trade_id)
    if trade is None:
        return

    entry_price = trade["entry_price"]
    quantity = trade["quantity"]
    direction = trade["direction"]

    if entry_price and quantity:
        if direction == "Long":
            pnl_usd = (exit_price - entry_price) * quantity
        else:
            pnl_usd = (entry_price - exit_price) * quantity
        pnl_pct = (pnl_usd / (entry_price * quantity)) * 100 if entry_price else 0.0
    else:
        pnl_usd = 0.0
        pnl_pct = 0.0

    total_fees = (trade["fees_usd"] or 0.0) + fees_usd

    update_trade(
        trade_id,
        status="closed",
        exit_time=_now(),
        exit_price=exit_price,
        exit_order_id=exit_order_id,
        pnl_usd=round(pnl_usd, 4),
        pnl_pct=round(pnl_pct, 4),
        fees_usd=round(total_fees, 4),
        exit_reason=exit_reason,
    )


def get_trade_by_id(trade_id: str) -> dict | None:
    """Return a single trade record or None."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM manual_trades WHERE trade_id = ?", (trade_id,)
        ).fetchone()
    return dict(row) if row else None


def get_open_trades(symbol: str | None = None) -> list[dict]:
    """Return all trades with status='open'."""
    query = "SELECT * FROM manual_trades WHERE status = 'open'"
    params: list = []
    if symbol:
        query += " AND symbol = ?"
        params.append(symbol)
    query += " ORDER BY entry_time DESC"
    with _conn() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def get_closed_trades(symbol: str | None = None, limit: int = 50) -> list[dict]:
    """Return recent closed trades, newest first."""
    query = "SELECT * FROM manual_trades WHERE status = 'closed'"
    params: list = []
    if symbol:
        query += " AND symbol = ?"
        params.append(symbol)
    query += " ORDER BY exit_time DESC LIMIT ?"
    params.append(limit)
    with _conn() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def get_all_trades(limit: int = 200) -> list[dict]:
    """Return all trades (open + closed), newest first."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM manual_trades ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_trade_stats() -> dict:
    """Aggregate stats across all closed trades."""
    with _conn() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*)                                        AS total,
                SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END)  AS wins,
                SUM(pnl_usd)                                    AS total_pnl,
                SUM(fees_usd)                                   AS total_fees,
                MAX(pnl_usd)                                    AS best_trade,
                MIN(pnl_usd)                                    AS worst_trade
            FROM manual_trades WHERE status = 'closed'
        """).fetchone()
    d = dict(row)
    total = d["total"] or 0
    d["win_rate"] = round((d["wins"] or 0) / total * 100, 1) if total else 0.0
    return d


# Auto-create the table on import
init_db()
