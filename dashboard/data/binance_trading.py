"""Authenticated Binance.US REST API client for manual trading."""

import hashlib
import hmac
import os
import time

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from config import BINANCE_REST_BASE

# ── Credentials ──────────────────────────────────────────────────────────────
API_KEY = os.environ.get("BINANCE_API_KEY", "")
API_SECRET = os.environ.get("BINANCE_API_SECRET", "")


def is_configured() -> bool:
    """Check whether API credentials are present."""
    return bool(API_KEY and API_SECRET)


# ── Request Helpers ──────────────────────────────────────────────────────────

def _sign(params: dict) -> dict:
    """Add timestamp and HMAC-SHA256 signature to *params*."""
    params["timestamp"] = int(time.time() * 1000)
    query = "&".join(f"{k}={v}" for k, v in params.items())
    sig = hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    params["signature"] = sig
    return params


def _headers() -> dict:
    return {"X-MBX-APIKEY": API_KEY}


def _get(endpoint: str, params: dict | None = None, signed: bool = True) -> dict:
    """Signed (or public) GET request."""
    params = dict(params or {})
    if signed:
        params = _sign(params)
    resp = requests.get(
        f"{BINANCE_REST_BASE}/{endpoint}",
        params=params,
        headers=_headers(),
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def _post(endpoint: str, params: dict) -> dict:
    """Signed POST request."""
    params = _sign(dict(params))
    resp = requests.post(
        f"{BINANCE_REST_BASE}/{endpoint}",
        params=params,
        headers=_headers(),
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def _delete(endpoint: str, params: dict) -> dict:
    """Signed DELETE request."""
    params = _sign(dict(params))
    resp = requests.delete(
        f"{BINANCE_REST_BASE}/{endpoint}",
        params=params,
        headers=_headers(),
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


# ── Account ──────────────────────────────────────────────────────────────────

def get_account_info() -> dict:
    """GET /api/v3/account — balances and permissions."""
    return _get("account")


def get_asset_balance(asset: str) -> float:
    """Return the free balance for *asset* (e.g. 'USDT', 'BTC')."""
    info = get_account_info()
    for b in info.get("balances", []):
        if b["asset"] == asset:
            return float(b["free"])
    return 0.0


# ── Price ────────────────────────────────────────────────────────────────────

def get_ticker_price(symbol: str) -> float:
    """GET /api/v3/ticker/price — current price (public, fast)."""
    data = _get("ticker/price", {"symbol": symbol}, signed=False)
    return float(data["price"])


def get_ticker_24h(symbol: str) -> dict:
    """GET /api/v3/ticker/24hr — 24-hour stats (public)."""
    return _get("ticker/24hr", {"symbol": symbol}, signed=False)


# ── Exchange Info ────────────────────────────────────────────────────────────

def get_exchange_info(symbol: str) -> dict:
    """GET /api/v3/exchangeInfo — filters for *symbol*.

    Returns dict with keys: lot_step, lot_min, lot_max, tick_size, min_notional.
    """
    data = _get("exchangeInfo", {"symbol": symbol}, signed=False)
    result = {}
    for s in data.get("symbols", []):
        if s["symbol"] == symbol:
            for f in s.get("filters", []):
                if f["filterType"] == "LOT_SIZE":
                    result["lot_step"] = float(f["stepSize"])
                    result["lot_min"] = float(f["minQty"])
                    result["lot_max"] = float(f["maxQty"])
                elif f["filterType"] == "PRICE_FILTER":
                    result["tick_size"] = float(f["tickSize"])
                elif f["filterType"] == "MIN_NOTIONAL":
                    result["min_notional"] = float(f.get("minNotional", 0))
                elif f["filterType"] == "NOTIONAL":
                    result["min_notional"] = float(f.get("minNotional", 0))
            break
    return result


def round_quantity(qty: float, step: float) -> float:
    """Round *qty* down to the nearest valid lot step."""
    if step <= 0:
        return qty
    precision = len(str(step).rstrip("0").split(".")[-1]) if "." in str(step) else 0
    return round(qty - (qty % step), precision)


def round_price(price: float, tick: float) -> float:
    """Round *price* to the nearest valid tick size."""
    if tick <= 0:
        return price
    precision = len(str(tick).rstrip("0").split(".")[-1]) if "." in str(tick) else 0
    return round(price - (price % tick), precision)


# ── Orders ───────────────────────────────────────────────────────────────────

def place_market_order(symbol: str, side: str, quantity: float) -> dict:
    """Place a MARKET order. *side* is 'BUY' or 'SELL'."""
    return _post("order", {
        "symbol": symbol,
        "side": side.upper(),
        "type": "MARKET",
        "quantity": quantity,
    })


def place_limit_order(
    symbol: str, side: str, quantity: float, price: float,
    time_in_force: str = "GTC",
) -> dict:
    """Place a LIMIT order."""
    return _post("order", {
        "symbol": symbol,
        "side": side.upper(),
        "type": "LIMIT",
        "timeInForce": time_in_force,
        "quantity": quantity,
        "price": price,
    })


def place_stop_limit_order(
    symbol: str, side: str, quantity: float,
    stop_price: float, limit_price: float,
    time_in_force: str = "GTC",
) -> dict:
    """Place a STOP_LOSS_LIMIT order."""
    return _post("order", {
        "symbol": symbol,
        "side": side.upper(),
        "type": "STOP_LOSS_LIMIT",
        "timeInForce": time_in_force,
        "quantity": quantity,
        "stopPrice": stop_price,
        "price": limit_price,
    })


def place_stop_market_order(
    symbol: str, side: str, quantity: float, stop_price: float,
) -> dict:
    """Place a STOP_LOSS order (market execution at stop)."""
    return _post("order", {
        "symbol": symbol,
        "side": side.upper(),
        "type": "STOP_LOSS",
        "quantity": quantity,
        "stopPrice": stop_price,
    })


# ── Order Queries ────────────────────────────────────────────────────────────

def get_open_orders(symbol: str | None = None) -> list:
    """GET /api/v3/openOrders — all open orders, optionally for *symbol*."""
    params = {}
    if symbol:
        params["symbol"] = symbol
    return _get("openOrders", params)


def get_order_status(symbol: str, order_id: int) -> dict:
    """GET /api/v3/order — status of a specific order."""
    return _get("order", {"symbol": symbol, "orderId": order_id})


def get_recent_fills(symbol: str, limit: int = 50) -> list:
    """GET /api/v3/myTrades — recent fills."""
    return _get("myTrades", {"symbol": symbol, "limit": limit})


# ── Cancellation ─────────────────────────────────────────────────────────────

def cancel_order(symbol: str, order_id: int) -> dict:
    """DELETE /api/v3/order — cancel a specific open order."""
    return _delete("order", {"symbol": symbol, "orderId": order_id})


def cancel_all_orders(symbol: str) -> dict:
    """DELETE /api/v3/openOrders — cancel all open orders for *symbol*."""
    return _delete("openOrders", {"symbol": symbol})
