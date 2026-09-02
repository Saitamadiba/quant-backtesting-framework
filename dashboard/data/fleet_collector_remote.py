#!/usr/bin/env python3
"""READ-ONLY fleet collector — runs ON THE VPS, streamed in over SSH stdin.

The dashboard's **Live Fleet** page pipes this file to the VPS python and reads
one JSON line back. It is deliberately a dumb executor: the dashboard sends a
plan of SELECT statements, this refuses anything that is not a read, opens every
database with SQLite's `mode=ro` URI, and returns rows. Nothing here writes,
restarts, deploys, or places an order — a librarian who fetches books and cannot
hold a pen.

Two halves:
  1. `run_queries()` — the SELECT plan over the bot books (stdlib only).
  2. `collect_balances()` — per-seat ByBit account state (equity / positions),
     read with each seat's own key through the existing validated `bybit_client`.
     GET endpoints only: wallet balance, positions, and the key's own identity.
  3. `collect_alpaca()` — the same for the US-equities seat, read with the very
     key the bot itself uses and refusing anything that is not a PAPER key.

Secrets never leave the VPS: a seat is identified by its env-file path and the
account UID the exchange reports; API keys are only ever hashed (8 hex chars) so
two seats sharing one key can be spotted without the key being shown anywhere.

Invoked as:  cd ~/trading_bots && venv/bin/python3 - <<< "<this file + call>"
"""
from __future__ import annotations

import fnmatch
import glob
import hashlib
import io
import json
import logging
import os
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# Keep every library log line off stdout — stdout carries exactly one JSON line.
logging.basicConfig(level=logging.CRITICAL, stream=sys.stderr)

MAX_ROWS = 5000
SQL_FORBIDDEN = (";", "--", "/*", "pragma ", "attach ")
# Matched on word boundaries so an innocent column (`created_utc`, `updated_at`)
# is not mistaken for a statement.
SQL_BANNED_WORDS = ("insert", "update", "delete", "drop", "alter", "vacuum",
                    "reindex", "commit", "rollback", "begin", "into")
SQL_BANNED_RE = re.compile(r"\b(?:%s)\b" % "|".join(SQL_BANNED_WORDS))


# ══════════════════════════════════════════════════════════════════════════════
#  1. The SELECT plan
# ══════════════════════════════════════════════════════════════════════════════
def _assert_readonly(sql: str) -> None:
    """Refuse anything that is not a single bare SELECT. Fails CLOSED."""
    s = sql.strip()
    low = s.lower()
    if not low.startswith("select"):
        raise ValueError("not a SELECT")
    for tok in SQL_FORBIDDEN:
        if tok in low:
            raise ValueError(f"forbidden token {tok!r}")
    m = SQL_BANNED_RE.search(low)
    if m:
        raise ValueError(f"forbidden keyword {m.group(0)!r}")
    if re.search(r"\bcreate\s+(table|index|view|trigger)\b", low):
        raise ValueError("forbidden keyword 'create'")


def _connect_ro(path: str) -> sqlite3.Connection:
    """Open a database read-only — the connection itself cannot write."""
    uri = "file:" + path.replace("?", "%3f").replace("#", "%23") + "?mode=ro"
    con = sqlite3.connect(uri, uri=True, timeout=5.0)
    con.execute("PRAGMA query_only=ON")
    return con


def _expand(db: str) -> list[str]:
    return sorted(glob.glob(db)) if any(c in db for c in "*?[") else [db]


def run_queries(plan: list[dict]) -> dict:
    """Execute each {id, db, sql}; per-query failures are reported, never raised."""
    out: dict[str, dict] = {}
    cache: dict[str, sqlite3.Connection] = {}
    for item in plan:
        qid, db, sql = item["id"], item["db"], item["sql"]
        try:
            _assert_readonly(sql)
        except ValueError as e:
            out[qid] = {"ok": False, "error": f"blocked: {e}"}
            continue
        files = _expand(db)
        if not files:
            out[qid] = {"ok": False, "error": "db missing", "missing": True}
            continue
        cols: list[str] = []
        rows: list[list] = []
        errs: list[str] = []
        found = 0
        for path in files:
            if not os.path.exists(path):
                continue
            found += 1
            try:
                con = cache.get(path)
                if con is None:
                    con = cache[path] = _connect_ro(path)
                cur = con.execute(sql)
                cols = [d[0] for d in cur.description]
                rows.extend([list(r) for r in cur.fetchmany(MAX_ROWS)])
            except Exception as e:                      # noqa: BLE001
                errs.append(f"{os.path.basename(path)}: {e}")
        if not found:
            out[qid] = {"ok": False, "error": "db missing", "missing": True}
        elif errs and not rows:
            out[qid] = {"ok": False, "error": "; ".join(errs[:3])}
        else:
            out[qid] = {"ok": True, "cols": cols, "rows": rows[:MAX_ROWS],
                        "files": found, "truncated": len(rows) > MAX_ROWS,
                        "warn": "; ".join(errs[:3]) or None}
    for con in cache.values():
        try:
            con.close()
        except Exception:                               # noqa: BLE001
            pass
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  2. Per-seat ByBit balances (GET-only)
# ══════════════════════════════════════════════════════════════════════════════
_WANTED = ("BYBIT_API_KEY", "BYBIT_API_SECRET", "BYBIT_ENV", "BYBIT_TESTNET")


def _read_env(path: str) -> dict:
    """Pull only the four ByBit fields out of an env file. Nothing else is read."""
    vals = {}
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                if k in _WANTED:
                    vals[k] = v.strip().strip('"').strip("'")
    except Exception:                                   # noqa: BLE001
        return {}
    return vals


def _seat_name(path: str) -> str:
    base = os.path.basename(path)
    name = base[:-4] if base.endswith(".env") else base
    d = os.path.dirname(path)
    if name == "" or name == ".env":
        name = os.path.basename(d) or "root"
    return f"{d}/{name}" if d and d != "." else name


def discover_seats(patterns: list[str], skip: list[str]) -> list[dict]:
    """Every env file that carries a ByBit key, keyed by a hash of that key."""
    seen, seats = set(), []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            if path in seen:
                continue
            seen.add(path)
            if any(fnmatch.fnmatch(os.path.basename(path), s) for s in skip):
                continue
            env = _read_env(path)
            key, sec = env.get("BYBIT_API_KEY", ""), env.get("BYBIT_API_SECRET", "")
            if not key or not sec:
                continue
            seats.append({
                "seat": _seat_name(path),
                "env_file": path,
                "bybit_env": env.get("BYBIT_ENV") or (
                    "testnet" if env.get("BYBIT_TESTNET", "").lower()
                    in ("1", "true", "yes", "on") else "mainnet"),
                "key_digest": hashlib.sha1(key.encode()).hexdigest()[:8],
                "_key": key, "_secret": sec,
            })
    return seats


def _client_for(seat: dict):
    """A BybitClient bound to this seat's credentials. Constructed serially."""
    from bybit_client import BybitClient
    os.environ["BYBIT_API_KEY"] = seat["_key"]
    os.environ["BYBIT_API_SECRET"] = seat["_secret"]
    os.environ["BYBIT_ENV"] = seat["bybit_env"]
    return BybitClient()


def _f(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def _uid_of(client) -> tuple[str, str]:
    """Ask the exchange who this key is. Returns (uid, error)."""
    try:
        r = client._auth_get("/v5/user/query-api", {})
    except Exception as e:                              # noqa: BLE001
        return "", f"{type(e).__name__}: {e}"
    if str(r.get("retCode")) != "0":
        return "", f"retCode {r.get('retCode')}: {r.get('retMsg')}"
    res = r.get("result") or {}
    for k in ("userID", "uid", "userId"):
        if res.get(k) not in (None, ""):
            return str(res[k]), ""
    return "", "no uid in response"


def _wallet_and_positions(client) -> dict:
    out = {"ok": False}
    try:
        w = client.wallet_balance()
    except Exception as e:                              # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    if str(w.get("retCode")) != "0":
        out["error"] = f"wallet retCode {w.get('retCode')}: {w.get('retMsg')}"
        return out
    lst = (w.get("result") or {}).get("list") or [{}]
    a = lst[0] if lst else {}
    out["wallet"] = {
        "equity": _f(a.get("totalEquity")),
        "wallet_balance": _f(a.get("totalWalletBalance")),
        "available": _f(a.get("totalAvailableBalance")),
        "upnl": _f(a.get("totalPerpUPL")),
        "im_rate": _f(a.get("accountIMRate")),
        "mm_rate": _f(a.get("accountMMRate")),
    }
    try:
        p = client.positions()
        plist = (p.get("result") or {}).get("list") or [] if str(p.get("retCode")) == "0" else []
    except Exception:                                   # noqa: BLE001
        plist = []
    out["positions"] = [
        {"symbol": x.get("symbol"), "side": x.get("side"), "size": _f(x.get("size")),
         "avg_price": _f(x.get("avgPrice")), "mark_price": _f(x.get("markPrice")),
         "upnl": _f(x.get("unrealisedPnl")), "leverage": _f(x.get("leverage")),
         "value": _f(x.get("positionValue")),
         "sl": _f(x.get("stopLoss")), "tp": _f(x.get("takeProfit")),
         "opened": x.get("createdTime") or ""}
        for x in plist if _f(x.get("size")) != 0
    ]
    # Orders resting at the exchange — the authoritative "working" list. A bot
    # book can hold a row for an arm it already cancelled; this cannot.
    try:
        o = client._auth_get("/v5/order/realtime",
                             {"category": "linear", "settleCoin": "USDT", "limit": 50})
        olist = (o.get("result") or {}).get("list") or [] if str(o.get("retCode")) == "0" else []
    except Exception:                                   # noqa: BLE001
        olist = []
    out["orders"] = [
        {"symbol": x.get("symbol"), "side": x.get("side"),
         "order_type": x.get("orderType"), "qty": _f(x.get("qty")),
         "price": _f(x.get("price")), "status": x.get("orderStatus"),
         "sl": _f(x.get("stopLoss")), "tp": _f(x.get("takeProfit")),
         "reduce_only": bool(x.get("reduceOnly")),
         "created": x.get("createdTime") or ""}
        for x in olist
    ]
    out["ok"] = True
    return out


def collect_balances(patterns: list[str], skip: list[str], workers: int = 6) -> dict:
    """Equity + live positions for every distinct account the fleet trades.

    Two passes: identify each key's account (cheap), then read wallet+positions
    once per *account* — several seats commonly share one sub-account, and
    asking once per seat would both waste calls and double-count the equity.
    """
    t0 = time.time()
    seats = discover_seats(patterns, skip)
    if not seats:
        return {"ok": True, "seats": [], "accounts": [], "note": "no ByBit env files found"}

    # Dedupe by key first; construct clients serially (BybitClient reads os.environ).
    by_key: dict[str, dict] = {}
    for s in seats:
        by_key.setdefault(s["key_digest"], {"seats": [], "seat": s})["seats"].append(s["seat"])
    clients = {}
    for kd, grp in by_key.items():
        try:
            clients[kd] = _client_for(grp["seat"])
        except Exception as e:                          # noqa: BLE001
            grp["error"] = f"client init failed: {type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=workers) as ex:
        uids = {kd: ex.submit(_uid_of, c) for kd, c in clients.items()}
        for kd, fut in uids.items():
            uid, err = fut.result()
            by_key[kd]["uid"] = uid
            if err:
                by_key[kd]["error"] = err

    # One wallet read per distinct account.
    per_uid: dict[str, str] = {}
    for kd, grp in by_key.items():
        uid = grp.get("uid")
        if uid and uid not in per_uid:
            per_uid[uid] = kd
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {uid: ex.submit(_wallet_and_positions, clients[kd])
                for uid, kd in per_uid.items() if kd in clients}
        acct = {uid: f.result() for uid, f in futs.items()}

    seat_rows = []
    for kd, grp in by_key.items():
        uid = grp.get("uid") or ""
        a = acct.get(uid, {})
        seat_rows.append({
            "seats": sorted(grp["seats"]),
            "key_digest": kd,
            "bybit_env": grp["seat"]["bybit_env"],
            "uid": uid,
            "ok": bool(a.get("ok")),
            "error": grp.get("error") or a.get("error") or "",
        })

    accounts = []
    for uid, a in acct.items():
        labels = sorted({s for kd, g in by_key.items() if g.get("uid") == uid
                         for s in g["seats"]})
        accounts.append({
            "uid": uid, "seats": labels, "ok": bool(a.get("ok")),
            "error": a.get("error", ""),
            "wallet": a.get("wallet", {}), "positions": a.get("positions", []),
            "orders": a.get("orders", []),
            "bybit_env": next((g["seat"]["bybit_env"] for g in by_key.values()
                               if g.get("uid") == uid), "?"),
        })
    return {"ok": True, "seats": seat_rows, "accounts": accounts,
            "elapsed_s": round(time.time() - t0, 2)}


# ══════════════════════════════════════════════════════════════════════════════
#  3. The Alpaca equities seat (GET-only, PAPER only)
# ══════════════════════════════════════════════════════════════════════════════
def _alpaca_env(candidates: list) -> tuple:
    """The very credentials the analyst-drift bot loads — same file, same order."""
    for c in candidates:
        if not c or not os.path.isfile(c):
            continue
        kv = {}
        try:
            with open(c, "r", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        if k.startswith("export "):
                            k = k[len("export "):].strip()
                        if k in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
                            kv[k] = v.strip().strip('"').strip("'")
        except Exception:                               # noqa: BLE001
            continue
        if kv.get("ALPACA_API_KEY") and kv.get("ALPACA_SECRET_KEY"):
            return kv["ALPACA_API_KEY"], kv["ALPACA_SECRET_KEY"], c
    return "", "", ""


def collect_alpaca(cfg: dict) -> dict:
    """Equity, positions and resting orders for the US-equities paper seat.

    Fails CLOSED on a non-paper key: the bot itself refuses anything without the
    `PK` prefix, and a read-only panel is no reason to relax that — the one
    account this may ever touch is the paper one.
    """
    import urllib.request

    seat = cfg.get("seat", "analyst_drift_paper")
    base = cfg.get("base", "https://paper-api.alpaca.markets")
    key, sec, src = _alpaca_env(cfg.get("env_candidates", []))
    out = {"venue": "alpaca-paper", "seats": [seat], "ok": False, "uid": "",
           "bybit_env": "paper", "wallet": {}, "positions": [], "orders": []}
    if not key:
        out["error"] = "no Alpaca credentials found on the VPS"
        return out
    out["key_digest"] = hashlib.sha1(key.encode()).hexdigest()[:8]
    if not key.startswith("PK"):
        out["error"] = "refused: key is not a PAPER key (no PK prefix)"
        return out

    hdrs = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}

    def get(path):
        req = urllib.request.Request(f"{base}/v2/{path}", headers=hdrs)
        with urllib.request.urlopen(req, timeout=int(cfg.get("timeout", 20))) as fh:
            return json.load(fh)

    try:
        a = get("account")
    except Exception as e:                              # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    acct_no = str(a.get("account_number") or "")
    out["uid"] = ("…" + acct_no[-4:]) if acct_no else ""      # masked, never whole
    eq, last_eq = _f(a.get("equity")), _f(a.get("last_equity"))
    out["wallet"] = {
        "equity": eq,
        "wallet_balance": _f(a.get("cash")),
        "available": _f(a.get("buying_power")),
        "upnl": 0.0,                                   # filled from positions below
        "day_change": (eq - last_eq) if last_eq else 0.0,
        "status": a.get("status", ""),
        "blocked": bool(a.get("trading_blocked")),
    }
    try:
        poss = get("positions")
    except Exception:                                   # noqa: BLE001
        poss = []
    out["positions"] = [
        {"symbol": x.get("symbol"), "side": (x.get("side") or "").title(),
         "size": _f(x.get("qty")), "avg_price": _f(x.get("avg_entry_price")),
         "mark_price": _f(x.get("current_price")), "upnl": _f(x.get("unrealized_pl")),
         "leverage": 1.0, "value": _f(x.get("market_value")), "sl": 0.0, "tp": 0.0,
         "opened": ""}
        for x in poss
    ]
    out["wallet"]["upnl"] = sum(p["upnl"] for p in out["positions"])
    try:
        orders = get("orders?status=open&limit=50")
    except Exception:                                   # noqa: BLE001
        orders = []
    out["orders"] = [
        {"symbol": x.get("symbol"), "side": (x.get("side") or "").title(),
         "order_type": x.get("type"), "qty": _f(x.get("qty")),
         "price": _f(x.get("limit_price")), "status": x.get("status"),
         "sl": 0.0, "tp": 0.0, "reduce_only": False,
         "created": x.get("submitted_at") or ""}
        for x in orders
    ]
    out["ok"] = True
    out["env_file"] = src
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  entry point
# ══════════════════════════════════════════════════════════════════════════════
def main(spec: dict) -> None:
    real_stdout = sys.stdout
    sys.stdout = io.StringIO()          # anything a library prints is discarded
    out = {"ok": True, "server_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
           "cwd": os.getcwd()}
    try:
        out["results"] = run_queries(spec.get("plan", []))
    except Exception as e:                              # noqa: BLE001
        out["ok"] = False
        out["error"] = f"queries failed: {type(e).__name__}: {e}"
        out["results"] = {}

    bal = spec.get("balances") or {}
    if bal.get("enabled"):
        sys.path.insert(0, os.path.join(os.getcwd(), "HyroTrader"))
        try:
            out["balances"] = collect_balances(
                bal.get("env_globs", []), bal.get("skip", []),
                int(bal.get("workers", 6)))
        except Exception as e:                          # noqa: BLE001
            out["balances"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    else:
        out["balances"] = {"ok": True, "skipped": True, "accounts": [], "seats": []}

    alp = spec.get("alpaca") or {}
    if bal.get("enabled") and alp.get("enabled"):
        try:
            acct = collect_alpaca(alp)
        except Exception as e:                          # noqa: BLE001
            acct = {"venue": "alpaca-paper", "seats": [alp.get("seat", "alpaca")],
                    "ok": False, "error": f"{type(e).__name__}: {e}",
                    "wallet": {}, "positions": [], "orders": [], "uid": ""}
        out["balances"].setdefault("accounts", []).append(acct)
        out["balances"].setdefault("seats", []).append({
            "seats": acct.get("seats", []), "key_digest": acct.get("key_digest", ""),
            "bybit_env": "alpaca-paper", "uid": acct.get("uid", ""),
            "ok": bool(acct.get("ok")), "error": acct.get("error", ""),
        })

    sys.stdout = real_stdout
    print(json.dumps(out, default=str))
