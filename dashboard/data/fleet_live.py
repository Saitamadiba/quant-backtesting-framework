"""Live fleet reader — one read-only SSH round trip, every bot's book.

The dashboard does NOT keep 40 databases in sync to answer "how is the fleet
doing right now". It pipes `fleet_collector_remote.py` to the VPS python, the
VPS runs the SELECT plan (and, if asked, reads each seat's ByBit equity), and
one JSON line comes back. One phone call to the warehouse instead of couriering
forty ledgers home.

Guarantees, matching `session_pnl_snapshot.sh`:
  * connection details come from the environment (`dashboard/.env` → config.py);
    nothing is hardcoded and no key is held locally;
  * every statement is a SELECT and every database is opened `mode=ro` — this
    path can read the fleet and cannot change it;
  * it fails CLOSED and quiet: an unreachable VPS returns `ok=False` with a
    message, never a partial-looking success.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from config import VPS_HOST, VPS_PORT, VPS_USER, VPS_REMOTE_BASE
from data.fleet_registry import BOOKS, BOOKS_BY_KEY, TIER_NAMES, build_spec

_COLLECTOR = Path(__file__).resolve().parent / "fleet_collector_remote.py"

# Where seats keep their ByBit credentials on the VPS. Only the ByBit fields are
# read, and only a hash of the key ever leaves the box.
ENV_GLOBS = ["*.env", "HyroTrader/*.env", "*/*.env", "HyroTrader/.env"]
ENV_SKIP = ["*template*", "*.bak.*", "*.example", "*example*"]

_REMOTE_PY = "./venv/bin/python3"


def _ssh_cmd(timeout: int) -> list[str]:
    return [
        "ssh", "-p", str(VPS_PORT),
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={min(15, timeout)}",
        "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=4",
        f"{VPS_USER}@{VPS_HOST}",
        f"cd {VPS_REMOTE_BASE} && {_REMOTE_PY} -",
    ]


def _payload(spec: dict) -> str:
    """The collector's own source plus the call that runs it — nothing is
    deployed to the VPS; the program lives here and is streamed in on stdin."""
    src = _COLLECTOR.read_text()
    return f"{src}\n\nmain(json.loads({json.dumps(json.dumps(spec))}))\n"


def fetch_raw(days: int = 7, with_balances: bool = True, trade_limit: int = 500,
              timeout: int = 90) -> dict:
    """Run the collector on the VPS and return its parsed JSON (fails closed)."""
    if not VPS_HOST:
        return {"ok": False, "error": "VPS_HOST not configured (dashboard/.env)."}
    spec = {
        "plan": build_spec(days=days, trade_limit=trade_limit),
        "balances": {"enabled": bool(with_balances), "env_globs": ENV_GLOBS,
                     "skip": ENV_SKIP, "workers": 6},
    }
    try:
        res = subprocess.run(_ssh_cmd(timeout), input=_payload(spec),
                             capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"VPS collector timed out after {timeout}s."}
    except Exception as e:                              # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    out = (res.stdout or "").strip()
    if res.returncode != 0 and not out:
        return {"ok": False,
                "error": f"ssh exit {res.returncode}: {(res.stderr or '').strip()[:300]}"}
    for line in reversed(out.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {"ok": False,
            "error": f"unparseable collector output: {out[:200]!r} "
                     f"{(res.stderr or '')[:200]!r}"}


# ══════════════════════════════════════════════════════════════════════════════
#  Shaping
# ══════════════════════════════════════════════════════════════════════════════
def _rows(res: dict, qid: str) -> tuple[list[str], list[list], Optional[str]]:
    r = (res or {}).get(qid)
    if not r:
        return [], [], "not run"
    if not r.get("ok"):
        return [], [], r.get("error", "error")
    return r.get("cols", []), r.get("rows", []), None


def _to_dt(s: pd.Series) -> pd.Series:
    """Parse a timestamp column that mixes formats across books.

    The fleet writes time five different ways — `2026-09-02 12:01:00`, ISO with
    a `T` and microseconds, some with an offset. Pandas infers ONE format per
    column and coerces the rest to NaT, which silently drops whole bots from the
    running list. `format="mixed"` parses each value on its own terms.
    """
    try:
        return pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
    except (ValueError, TypeError):
        return pd.to_datetime(s, errors="coerce", utc=True)


def _num(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return v if v == v else 0.0          # NaN → 0


def books_frame(raw: dict) -> pd.DataFrame:
    """One row per book: lifetime and rolling-window n / meanR / ΣR / WR / Σ$."""
    res = raw.get("results", {})
    out = []
    for b in BOOKS:
        cols, rows, err = _rows(res, f"{b.key}::agg")
        rec = {
            "key": b.key, "bot": b.label, "tier": b.tier,
            "tier_name": TIER_NAMES[b.tier], "db": b.db, "note": b.note,
            "in_recap": b.in_recap, "status": "ok" if not err else err,
        }
        # A glob book (shadow_books/*.db) answers once per file — fold them.
        n = sum(_num(r[cols.index("n")]) for r in rows) if rows else 0.0
        sum_r = sum(_num(r[cols.index("sum_r")]) for r in rows) if rows else 0.0
        wins = sum(_num(r[cols.index("wins")]) for r in rows) if rows else 0.0
        sum_pnl = sum(_num(r[cols.index("sum_pnl")]) for r in rows) if rows else 0.0
        n_w = sum(_num(r[cols.index("n_w")]) for r in rows) if rows else 0.0
        sum_r_w = sum(_num(r[cols.index("sum_r_w")]) for r in rows) if rows else 0.0
        wins_w = sum(_num(r[cols.index("wins_w")]) for r in rows) if rows else 0.0
        sum_pnl_w = sum(_num(r[cols.index("sum_pnl_w")]) for r in rows) if rows else 0.0
        rec.update({
            "n": int(n), "sum_r": sum_r if b.r else None,
            "mean_r": (sum_r / n) if (b.r and n) else None,
            "win_rate": (wins / n) if n else None,
            "pnl_usd": sum_pnl if b.pnl else None,
            "n_7d": int(n_w), "sum_r_7d": sum_r_w if b.r else None,
            "mean_r_7d": (sum_r_w / n_w) if (b.r and n_w) else None,
            "win_rate_7d": (wins_w / n_w) if n_w else None,
            "pnl_usd_7d": sum_pnl_w if b.pnl else None,
            "has_r": b.r is not None, "has_usd": b.pnl is not None,
        })
        # running legs, counted here so the table can show them beside the PnL
        ocols, orows, oerr = _rows(res, f"{b.key}::open")
        rec["open_n"] = 0 if oerr else sum(
            1 for r in orows if r[ocols.index("state")] == "FILLED")
        rec["working_n"] = 0 if oerr else sum(
            1 for r in orows if r[ocols.index("state")] == "WORKING")
        out.append(rec)
    return pd.DataFrame(out)


def trades_frame(raw: dict, days: int = 7) -> pd.DataFrame:
    """Every trade every book closed in the window — one flat, sortable ledger."""
    res = raw.get("results", {})
    frames = []
    for b in BOOKS:
        cols, rows, err = _rows(res, f"{b.key}::trades")
        if err or not rows:
            continue
        df = pd.DataFrame(rows, columns=cols)
        df.insert(0, "bot", b.label)
        df.insert(1, "tier", b.tier)
        df["has_usd"] = b.pnl is not None
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["bot", "tier", "ts", "symbol", "side",
                                     "entry", "exit_px", "r", "pnl", "has_usd"])
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = _to_dt(df["ts"])
    for c in ("entry", "exit_px", "r", "pnl"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.loc[~df["has_usd"].astype(bool), "pnl"] = pd.NA
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    df = df[df["ts"].notna() & (df["ts"] >= cutoff)]
    return df.sort_values("ts", ascending=False).reset_index(drop=True)


def daily_frame(raw: dict) -> pd.DataFrame:
    """One row per (book, UTC day): n, ΣR, Σ$ — counted on the VPS, uncapped.

    The charts read this, never the capped trade dump, so a shadow book that
    closes 900 episodes a day is not silently rounded down to its newest 500.
    """
    res = raw.get("results", {})
    frames = []
    for b in BOOKS:
        cols, rows, err = _rows(res, f"{b.key}::daily")
        if err or not rows:
            continue
        df = pd.DataFrame(rows, columns=cols)
        df["bot"], df["tier"] = b.label, b.tier
        if b.pnl is None:
            df["sum_pnl"] = pd.NA
        if b.r is None:
            df["sum_r"] = pd.NA
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["day", "n", "sum_r", "sum_pnl", "bot", "tier"])
    df = pd.concat(frames, ignore_index=True)
    df["day"] = pd.to_datetime(df["day"], errors="coerce").dt.date
    for c in ("n", "sum_r", "sum_pnl"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[df["day"].notna()].reset_index(drop=True)


def open_frame(raw: dict) -> pd.DataFrame:
    """The running book: filled-and-live positions, plus orders still working."""
    res = raw.get("results", {})
    frames = []
    for b in BOOKS:
        cols, rows, err = _rows(res, f"{b.key}::open")
        if err or not rows:
            continue
        df = pd.DataFrame(rows, columns=cols)
        df.insert(0, "bot", b.label)
        df.insert(1, "tier", b.tier)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["bot", "tier", "state", "since", "symbol",
                                     "side", "entry", "sl", "tp", "qty", "risk_usd"])
    df = pd.concat(frames, ignore_index=True)
    df["since"] = _to_dt(df["since"])
    for c in ("entry", "sl", "tp", "qty", "risk_usd"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    now = datetime.now(timezone.utc)
    df["age_h"] = (now - df["since"]).dt.total_seconds() / 3600.0
    return df.sort_values("since", ascending=False, na_position="last").reset_index(drop=True)


def accounts_frame(raw: dict) -> pd.DataFrame:
    """One row per distinct exchange account — equity, uPnL, and who trades it."""
    bal = raw.get("balances") or {}
    rows = []
    for a in bal.get("accounts", []):
        w = a.get("wallet", {}) or {}
        rows.append({
            "uid": a.get("uid", ""),
            "seats": ", ".join(a.get("seats", [])),
            "n_seats": len(a.get("seats", [])),
            "env": a.get("bybit_env", "?"),
            "equity": _num(w.get("equity")),
            "wallet_balance": _num(w.get("wallet_balance")),
            "available": _num(w.get("available")),
            "upnl": _num(w.get("upnl")),
            "positions": len(a.get("positions", [])),
            "status": "ok" if a.get("ok") else (a.get("error") or "unreachable"),
        })
    return pd.DataFrame(rows)


def exchange_positions_frame(raw: dict) -> pd.DataFrame:
    """Positions as the EXCHANGE sees them — the ground truth for money at risk."""
    bal = raw.get("balances") or {}
    rows = []
    for a in bal.get("accounts", []):
        label = ", ".join(a.get("seats", [])) or a.get("uid", "?")
        for p in a.get("positions", []):
            rows.append({
                "account": label, "uid": a.get("uid", ""),
                "symbol": p.get("symbol"), "side": p.get("side"),
                "size": _num(p.get("size")), "avg_price": _num(p.get("avg_price")),
                "mark_price": _num(p.get("mark_price")), "upnl": _num(p.get("upnl")),
                "value": _num(p.get("value")), "leverage": _num(p.get("leverage")),
                "sl": _num(p.get("sl")), "tp": _num(p.get("tp")),
            })
    return pd.DataFrame(rows)


def exchange_orders_frame(raw: dict) -> pd.DataFrame:
    """Orders actually resting at the exchange — the authoritative working list.

    A bot's book can still hold a row for an arm it cancelled two months ago;
    this list only contains orders the venue is holding right now.
    """
    bal = raw.get("balances") or {}
    rows = []
    for a in bal.get("accounts", []):
        label = ", ".join(a.get("seats", [])) or a.get("uid", "?")
        for o in a.get("orders", []):
            created = o.get("created") or ""
            ts = pd.NaT
            if str(created).isdigit():
                ts = pd.to_datetime(int(created), unit="ms", utc=True)
            rows.append({
                "account": label, "uid": a.get("uid", ""),
                "symbol": o.get("symbol"), "side": o.get("side"),
                "order_type": o.get("order_type"), "qty": _num(o.get("qty")),
                "price": _num(o.get("price")), "status": o.get("status"),
                "sl": _num(o.get("sl")), "tp": _num(o.get("tp")),
                "reduce_only": bool(o.get("reduce_only")), "placed": ts,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        now = datetime.now(timezone.utc)
        df["age_h"] = (now - df["placed"]).dt.total_seconds() / 3600.0
        df = df.sort_values("placed", ascending=False, na_position="last")
    return df.reset_index(drop=True)


def _base(sym) -> str:
    """`BCHUSDT`, `BCH`, `bch` → `BCH`. Books and the venue spell symbols differently."""
    t = str(sym or "").upper().strip()
    for suf in ("USDT", "USDC", "PERP", "USD"):
        if t.endswith(suf) and len(t) > len(suf):
            t = t[: -len(suf)]
    return t


def reconcile(raw: dict, running: pd.DataFrame) -> pd.DataFrame:
    """Tier-1 book legs the seat's OWN exchange account does not confirm.

    A seat's book says it is in a trade; the venue says that account holds
    nothing in that symbol. Either the book never caught the close (a stale row
    the risk guard still counts against the seat) or it was closed by hand.

    Deliberately conservative — a leg is only ever flagged when all three hold:
    the book's seat is known, that seat's account actually answered, and the
    account holds nothing in the base asset. An unchecked seat is reported as
    unchecked, never as an orphan; matching is on the base asset because a book
    may write `BCH` where the venue writes `BCHUSDT`.
    """
    empty = running.iloc[0:0] if not running.empty else pd.DataFrame(
        columns=["bot", "tier", "state", "since", "symbol", "side", "entry",
                 "sl", "tp", "qty", "risk_usd", "age_h"])
    if running.empty:
        return empty
    legs = running[(running["tier"] == 1) & (running["state"] == "FILLED")].copy()
    if legs.empty:
        return empty

    # seat → (answered?, {base assets held}) straight from the balance read
    held: dict[str, set] = {}
    for a in (raw.get("balances") or {}).get("accounts", []):
        if not a.get("ok"):
            continue
        bases = {_base(p.get("symbol")) for p in a.get("positions", [])}
        for seat in a.get("seats", []):
            held[seat] = bases

    seat_of = {b.label: b.seat for b in BOOKS if b.reconcilable}
    legs["seat"] = legs["bot"].map(seat_of)
    legs["base"] = legs["symbol"].map(_base)
    checkable = legs["seat"].notna() & legs["seat"].isin(held.keys())
    confirmed = legs.apply(
        lambda r: r["base"] in held.get(r["seat"] or "", set()), axis=1)
    orphans = legs[checkable & ~confirmed].copy()
    return orphans.drop(columns=["base"])


def seat_status_frame(raw: dict) -> pd.DataFrame:
    """Every key the fleet holds: which seats share it, and whether it answers."""
    bal = raw.get("balances") or {}
    rows = []
    for s in bal.get("seats", []):
        rows.append({
            "seats": ", ".join(s.get("seats", [])),
            "key_digest": s.get("key_digest", ""),
            "env": s.get("bybit_env", "?"),
            "uid": s.get("uid", ""),
            "status": "ok" if s.get("ok") else (s.get("error") or "no answer"),
        })
    return pd.DataFrame(rows).sort_values(
        ["status", "seats"], key=lambda c: c.str.lower() if c.name == "seats" else c
    ) if rows else pd.DataFrame(columns=["seats", "key_digest", "env", "uid", "status"])
