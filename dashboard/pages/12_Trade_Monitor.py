"""Trade Monitor — lightweight-charts with ICT indicators baked in + manual orders."""

import streamlit as st

st.set_page_config(page_title="Trade Monitor", page_icon="", layout="wide")

import numpy as np
import pandas as pd
from streamlit_lightweight_charts import renderLightweightCharts

from config import (
    TRADING_PAIRS, TRADING_REFRESH_OPTIONS, CHART_TIMEFRAMES,
)
from data.data_loader import get_live_trades, get_open_positions
from data.binance_helpers import (
    fetch_candles, calculate_indicators,
    detect_order_blocks, detect_fvgs, detect_fvgs_mtf,
    compute_session_levels,
)

# ── Constants ──────────────────────────────────────────────────────────────────
TF_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "4h": 14400, "1D": 86400,
}
SESS_COLORS = {"Asian": "#9c27b0", "London": "#4caf50", "New York": "#ff9800"}


# ── Helpers ────────────────────────────────────────────────────────────────────
def _to_unix(ts) -> int:
    """Convert any timestamp to Unix seconds (UTC)."""
    if isinstance(ts, pd.Timestamp):
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return int(ts.timestamp())
    return int(pd.Timestamp(ts).timestamp())


def _snap_to_candle(ts_unix: int, candle_times: list[int]) -> int:
    """Snap a timestamp to the nearest candle time so markers display."""
    if not candle_times:
        return ts_unix
    idx = int(np.searchsorted(candle_times, ts_unix))
    if idx == 0:
        return candle_times[0]
    if idx >= len(candle_times):
        return candle_times[-1]
    if abs(candle_times[idx] - ts_unix) < abs(candle_times[idx - 1] - ts_unix):
        return candle_times[idx]
    return candle_times[idx - 1]


def _zone_line(start_t: int, end_t: int, value: float, color: str,
               width: int = 1, style: int = 0) -> dict:
    """Build a Line series dict for a horizontal zone boundary."""
    return {
        "type": "Line",
        "data": [
            {"time": start_t, "value": round(value, 2)},
            {"time": end_t, "value": round(value, 2)},
        ],
        "options": {
            "color": color, "lineWidth": width, "lineStyle": style,
            "lastValueVisible": False, "priceLineVisible": False,
            "crosshairMarkerVisible": False,
        },
    }


def _place_manual_order(order: dict, pair_cfg: dict) -> None:
    """Execute a manual order on Binance and log to DB."""
    from data import binance_trading as api
    from data import manual_trade_db as db

    side = "BUY" if order["direction"] == "Long" else "SELL"
    sym = order["symbol"]
    qty = order["quantity"]

    try:
        if order["order_type"] == "Market":
            result = api.place_market_order(sym, side, qty)
        elif order["order_type"] == "Limit":
            result = api.place_limit_order(sym, side, qty, order["limit_price"])
        elif order["order_type"] == "Stop Market":
            result = api.place_stop_market_order(sym, side, qty, order["stop_price"])
        elif order["order_type"] == "Stop Limit":
            result = api.place_stop_limit_order(
                sym, side, qty, order["stop_price"], order["limit_price"]
            )
        else:
            st.error(f"Unknown order type: {order['order_type']}")
            return

        order_id = str(result.get("orderId", ""))
        status = result.get("status", "")
        fill_price = None
        fees = 0.0

        if status == "FILLED":
            fills = result.get("fills", [])
            if fills:
                total_qty = sum(float(f["qty"]) for f in fills)
                fill_price = (
                    sum(float(f["price"]) * float(f["qty"]) for f in fills)
                    / total_qty if total_qty else 0
                )
                fees = sum(float(f["commission"]) for f in fills)

        trade_id = db.insert_trade(
            symbol=sym, direction=order["direction"], quantity=qty,
            entry_price=fill_price, entry_order_id=order_id,
            stop_loss=order["sl"], take_profit=order["tp"], fees_usd=fees,
        )

        if status == "FILLED" and fill_price:
            exit_side = "SELL" if order["direction"] == "Long" else "BUY"
            if order["sl"]:
                try:
                    sl_r = api.place_stop_market_order(sym, exit_side, qty, order["sl"])
                    db.update_trade(trade_id, sl_order_id=str(sl_r.get("orderId", "")))
                except Exception as e:
                    st.warning(f"SL order failed: {e}")
            if order["tp"]:
                try:
                    tp_r = api.place_limit_order(sym, exit_side, qty, order["tp"])
                    db.update_trade(trade_id, tp_order_id=str(tp_r.get("orderId", "")))
                except Exception as e:
                    st.warning(f"TP order failed: {e}")

        st.success(f"Order placed (ID: {order_id}, status: {status})")
    except Exception as e:
        st.error(f"Order failed: {e}")


# ── Cached data loaders ────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def _fetch_chart_data(sym: str, tf: str, days: int):
    df = fetch_candles(sym, tf, days)
    if df.empty:
        return df
    return calculate_indicators(df)


@st.cache_data(ttl=120)
def _fetch_htf_fvgs(sym: str, tf: str, days: int):
    return detect_fvgs_mtf(sym, tf, days)


# ── Auto-Refresh ───────────────────────────────────────────────────────────────
try:
    from streamlit_autorefresh import st_autorefresh

    refresh_label = st.sidebar.selectbox(
        "Data Refresh", list(TRADING_REFRESH_OPTIONS.keys()), index=2, key="mon_refresh"
    )
    interval = TRADING_REFRESH_OPTIONS[refresh_label]
    if interval > 0:
        st_autorefresh(interval=interval, key="monitor_refresh")
except ImportError:
    st.sidebar.info("Install streamlit-autorefresh for auto-refresh.")

st.title("Trade Monitor")

# ── Symbol / Timeframe / History selectors ─────────────────────────────────────
scols = st.columns([1, 1, 1, 3])
sym_key = scols[0].selectbox("Symbol", list(TRADING_PAIRS.keys()), key="mon_sym")
timeframe = scols[1].selectbox("Timeframe", CHART_TIMEFRAMES, index=2, key="mon_tf")
history_days = scols[2].number_input("Days", min_value=1, max_value=30, value=3, key="mon_days")
pair = TRADING_PAIRS[sym_key]

# Indicator toggles
tcols = st.columns(6)
show_emas = tcols[0].checkbox("EMAs", True, key="t_ema")
show_vol = tcols[1].checkbox("Volume", True, key="t_vol")
show_obs = tcols[2].checkbox("Order Blocks", True, key="t_ob")
show_fvgs = tcols[3].checkbox("FVGs", True, key="t_fvg")
show_htf = tcols[4].checkbox("HTF FVGs", False, key="t_htf")
show_sess = tcols[5].checkbox("Session H/L", True, key="t_sess")

# ═══════════════════════════════════════════════════════════════════════════════
# FETCH DATA
# ═══════════════════════════════════════════════════════════════════════════════
with st.spinner("Loading chart data..."):
    df = _fetch_chart_data(sym_key, timeframe, history_days)

if df is None or df.empty:
    st.warning(f"No data available for {sym_key} {timeframe}.")
    st.stop()

tf_sec = TF_SECONDS.get(timeframe, 900)
last_unix = _to_unix(df.index[-1])
extend_unix = last_unix + 30 * tf_sec

# ═══════════════════════════════════════════════════════════════════════════════
# BUILD CHART SERIES
# ═══════════════════════════════════════════════════════════════════════════════
series = []

# Pre-compute Unix times for all candles
times_unix = [_to_unix(ts) for ts in df.index]

# ── 1. Candlestick + trade markers ────────────────────────────────────────────
candle_data = [
    {"time": t, "open": round(float(o), 2), "high": round(float(h), 2),
     "low": round(float(l), 2), "close": round(float(c), 2)}
    for t, o, h, l, c in zip(
        times_unix, df["Open"], df["High"], df["Low"], df["Close"]
    )
]

# Overlay trade markers from VPS bot data
markers = []
all_trades = get_live_trades()
sym_trades = (
    all_trades[all_trades["symbol"] == sym_key]
    if not all_trades.empty else pd.DataFrame()
)
open_pos = get_open_positions(sym_key)

if not sym_trades.empty:
    for _, trade in sym_trades.iterrows():
        if pd.notna(trade.get("entry_time")):
            entry_unix = _snap_to_candle(
                _to_unix(pd.Timestamp(trade["entry_time"])), times_unix
            )
            is_long = str(trade.get("direction", "")).lower() == "long"
            markers.append({
                "time": entry_unix,
                "position": "belowBar" if is_long else "aboveBar",
                "color": "#26a69a" if is_long else "#ef5350",
                "shape": "arrowUp" if is_long else "arrowDown",
                "text": f"{'L' if is_long else 'S'} {str(trade.get('strategy', ''))[:3]}",
            })
        if pd.notna(trade.get("exit_time")):
            exit_unix = _snap_to_candle(
                _to_unix(pd.Timestamp(trade["exit_time"])), times_unix
            )
            markers.append({
                "time": exit_unix,
                "position": "aboveBar",
                "color": "#ffeb3b",
                "shape": "circle",
                "text": "X",
            })

markers.sort(key=lambda m: m["time"])

candle_series = {
    "type": "Candlestick",
    "data": candle_data,
    "options": {
        "upColor": "#26a69a", "downColor": "#ef5350",
        "borderVisible": False,
        "wickUpColor": "#26a69a", "wickDownColor": "#ef5350",
    },
}
if markers:
    candle_series["markers"] = markers
series.append(candle_series)

# ── 2. Volume ──────────────────────────────────────────────────────────────────
if show_vol:
    vol_colors = np.where(
        df["Close"].values >= df["Open"].values,
        "rgba(38,166,154,0.3)", "rgba(239,83,80,0.3)",
    )
    vol_data = [
        {"time": t, "value": float(v), "color": str(c)}
        for t, v, c in zip(times_unix, df["Volume"].values, vol_colors)
    ]
    series.append({
        "type": "Histogram",
        "data": vol_data,
        "options": {
            "priceFormat": {"type": "volume"},
            "priceScaleId": "",
        },
        "priceScale": {"scaleMargins": {"top": 0.8, "bottom": 0}},
    })

# ── 3. EMAs ────────────────────────────────────────────────────────────────────
if show_emas:
    for col, color in [("EMA50", "#FFA726"), ("EMA200", "#42A5F5")]:
        ema_vals = df[col].dropna()
        if not ema_vals.empty:
            ema_data = [
                {"time": _to_unix(ts), "value": round(float(v), 2)}
                for ts, v in ema_vals.items()
            ]
            series.append({
                "type": "Line",
                "data": ema_data,
                "options": {
                    "color": color, "lineWidth": 1,
                    "lastValueVisible": False, "priceLineVisible": False,
                    "crosshairMarkerVisible": False,
                },
            })

# ── 4. Order Blocks ───────────────────────────────────────────────────────────
if show_obs:
    obs = detect_order_blocks(df, lookback=20)
    for ob in obs[-8:]:
        ob_t = _to_unix(ob["time"])
        is_bull = ob["type"] == "bullish"
        color = "rgba(38,166,154,0.7)" if is_bull else "rgba(239,83,80,0.7)"
        series.append(_zone_line(ob_t, extend_unix, ob["top"], color, width=1, style=2))
        series.append(_zone_line(ob_t, extend_unix, ob["bottom"], color, width=1, style=2))

# ── 5. FVGs (current timeframe) ───────────────────────────────────────────────
if show_fvgs:
    fvgs = detect_fvgs(df)
    for fvg in fvgs[-10:]:
        fvg_t = _to_unix(fvg["time"])
        is_bull = fvg["type"] == "bullish"
        color = "rgba(0,188,212,0.6)" if is_bull else "rgba(233,30,99,0.6)"
        series.append(_zone_line(fvg_t, extend_unix, fvg["top"], color))
        series.append(_zone_line(fvg_t, extend_unix, fvg["bottom"], color))

# ── 6. HTF FVGs ────────────────────────────────────────────────────────────────
if show_htf:
    htf_fvgs = _fetch_htf_fvgs(sym_key, timeframe, history_days)
    for fvg in htf_fvgs[-8:]:
        fvg_t = _to_unix(fvg["time"])
        is_bull = fvg["type"] == "bullish"
        color = "rgba(0,188,212,0.9)" if is_bull else "rgba(233,30,99,0.9)"
        series.append(_zone_line(fvg_t, extend_unix, fvg["top"], color, width=2))
        series.append(_zone_line(fvg_t, extend_unix, fvg["bottom"], color, width=2))

# ── 7. Session H/L ─────────────────────────────────────────────────────────────
if show_sess:
    levels = compute_session_levels(df)
    # Keep last 2 completed sessions per type
    seen: dict[str, int] = {}
    recent_levels = []
    for lvl in reversed(levels):
        k = lvl["session"]
        seen[k] = seen.get(k, 0) + 1
        if seen[k] <= 2:
            recent_levels.append(lvl)
    for lvl in recent_levels:
        color = SESS_COLORS.get(lvl["session"], "#ffffff")
        start = _to_unix(lvl["start_time"])
        # High = solid, Low = dotted
        h_line = _zone_line(start, extend_unix, lvl["high"], color, style=0)
        l_line = _zone_line(start, extend_unix, lvl["low"], color, style=1)
        # Show value on price scale for session levels
        h_line["options"]["lastValueVisible"] = True
        l_line["options"]["lastValueVisible"] = True
        series.append(h_line)
        series.append(l_line)

# ── 8. Open position SL/TP lines ──────────────────────────────────────────────
if not open_pos.empty:
    for _, pos in open_pos.iterrows():
        entry_t = (
            _to_unix(pd.Timestamp(pos["entry_time"]))
            if pd.notna(pos.get("entry_time"))
            else last_unix - 10 * tf_sec
        )
        if pd.notna(pos.get("stop_loss")) and float(pos["stop_loss"]) > 0:
            sl_line = _zone_line(entry_t, extend_unix, float(pos["stop_loss"]),
                                 "#ef5350", width=2, style=2)
            sl_line["options"]["lastValueVisible"] = True
            series.append(sl_line)
        if pd.notna(pos.get("take_profit")) and float(pos["take_profit"]) > 0:
            tp_line = _zone_line(entry_t, extend_unix, float(pos["take_profit"]),
                                 "#26a69a", width=2, style=2)
            tp_line["options"]["lastValueVisible"] = True
            series.append(tp_line)

# ═══════════════════════════════════════════════════════════════════════════════
# RENDER CHART
# ═══════════════════════════════════════════════════════════════════════════════
chart_options = {
    "height": 600,
    "layout": {
        "background": {"type": "solid", "color": "#131722"},
        "textColor": "#d1d4dc",
    },
    "grid": {
        "vertLines": {"color": "rgba(42, 46, 57, 0)"},
        "horzLines": {"color": "rgba(42, 46, 57, 0.6)"},
    },
    "rightPriceScale": {
        "borderColor": "rgba(197, 203, 206, 0.4)",
    },
    "timeScale": {
        "borderColor": "rgba(197, 203, 206, 0.4)",
        "timeVisible": True,
        "secondsVisible": False,
    },
    "crosshair": {"mode": 0},
    "watermark": {
        "visible": True,
        "fontSize": 48,
        "horzAlign": "center",
        "vertAlign": "center",
        "color": "rgba(171, 71, 188, 0.15)",
        "text": f"{sym_key} {timeframe}",
    },
}

renderLightweightCharts(
    [{"chart": chart_options, "series": series}],
    key=f"chart_{sym_key}_{timeframe}_{history_days}",
)

# Legend
st.caption(
    '<span style="color:#FFA726">&mdash;</span> EMA50 &nbsp; '
    '<span style="color:#42A5F5">&mdash;</span> EMA200 &nbsp; '
    '<span style="color:#26a69a">- -</span> Bull OB &nbsp; '
    '<span style="color:#ef5350">- -</span> Bear OB &nbsp; '
    '<span style="color:#00bcd4">&mdash;</span> Bull FVG &nbsp; '
    '<span style="color:#e91e63">&mdash;</span> Bear FVG &nbsp; '
    '<span style="color:#9c27b0">&mdash;</span> Asian H/L &nbsp; '
    '<span style="color:#4caf50">&mdash;</span> London H/L &nbsp; '
    '<span style="color:#ff9800">&mdash;</span> NY H/L',
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD TRADE DATA (for tables below)
# ═══════════════════════════════════════════════════════════════════════════════
closed_trades = (
    sym_trades[sym_trades.get("is_open", False) == False]  # noqa: E712
    if not sym_trades.empty else pd.DataFrame()
)

# ═══════════════════════════════════════════════════════════════════════════════
# MANUAL ORDER ENTRY
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("Manual Order Entry", expanded=False):
    if pair.get("source") != "binance":
        st.info(f"Manual trading not available for {sym_key} (not a Binance pair).")
    else:
        try:
            from data import binance_trading as api
            from data import manual_trade_db as db
            from data.binance_helpers import fetch_candles as _fc  # noqa: F811

            if not api.is_configured():
                st.warning(
                    "Binance API keys not set. "
                    "Add BINANCE_API_KEY and BINANCE_API_SECRET to `dashboard/.env`."
                )
            else:
                binance_sym = pair["binance_symbol"]

                @st.cache_data(ttl=3600)
                def _ex_info(s):
                    return api.get_exchange_info(s)

                ex_info = _ex_info(binance_sym)
                lot_step = ex_info.get("lot_step", 0.00001)
                tick_size = ex_info.get("tick_size", 0.01)

                try:
                    cur_price = api.get_ticker_price(binance_sym)
                except Exception:
                    cur_price = 0.0

                ocols = st.columns([1, 1, 1, 1])
                direction = ocols[0].radio(
                    "Direction", ["Long", "Short"], horizontal=True, key="m_dir"
                )
                order_type = ocols[1].selectbox(
                    "Type", ["Market", "Limit", "Stop Market", "Stop Limit"], key="m_otype"
                )
                qty = ocols[2].number_input(
                    f"Qty ({pair['base']})", min_value=lot_step, step=lot_step,
                    format=f"%.{pair['qty_decimals']}f", key="m_qty",
                )
                ocols[3].metric("Price", f"${cur_price:,.2f}")

                pcols = st.columns(4)
                limit_price = None
                stop_price = None

                if order_type in ("Limit", "Stop Limit"):
                    limit_price = pcols[0].number_input(
                        "Limit Price", min_value=0.0, step=tick_size, value=cur_price,
                        format=f"%.{pair['price_decimals']}f", key="m_lpx",
                    )
                if order_type in ("Stop Market", "Stop Limit"):
                    stop_price = pcols[1].number_input(
                        "Stop Price", min_value=0.0, step=tick_size, value=cur_price,
                        format=f"%.{pair['price_decimals']}f", key="m_spx",
                    )

                sl_price = pcols[2].number_input(
                    "Stop Loss", min_value=0.0, step=tick_size, value=0.0,
                    format=f"%.{pair['price_decimals']}f", key="m_sl",
                )
                tp_price = pcols[3].number_input(
                    "Take Profit", min_value=0.0, step=tick_size, value=0.0,
                    format=f"%.{pair['price_decimals']}f", key="m_tp",
                )

                # 2-step confirmation
                if "pending_order" not in st.session_state:
                    st.session_state.pending_order = None

                if st.button("Place Order", type="primary"):
                    st.session_state.pending_order = {
                        "symbol": binance_sym, "direction": direction,
                        "order_type": order_type,
                        "quantity": api.round_quantity(qty, lot_step),
                        "limit_price": (
                            api.round_price(limit_price, tick_size) if limit_price else None
                        ),
                        "stop_price": (
                            api.round_price(stop_price, tick_size) if stop_price else None
                        ),
                        "sl": sl_price if sl_price > 0 else None,
                        "tp": tp_price if tp_price > 0 else None,
                    }
                    st.rerun()

                if st.session_state.pending_order:
                    po = st.session_state.pending_order
                    st.warning(
                        f"**Confirm: {po['direction']} {po['quantity']} {pair['base']} "
                        f"via {po['order_type']}**"
                    )
                    if po.get("sl"):
                        st.write(f"SL: ${po['sl']:,.2f}")
                    if po.get("tp"):
                        st.write(f"TP: ${po['tp']:,.2f}")
                    ccols = st.columns(2)
                    if ccols[0].button("Confirm", type="primary"):
                        _place_manual_order(po, pair)
                        st.session_state.pending_order = None
                        st.rerun()
                    if ccols[1].button("Cancel"):
                        st.session_state.pending_order = None
                        st.rerun()

        except ImportError:
            st.info("Manual trading modules not available.")

# ═══════════════════════════════════════════════════════════════════════════════
# BOTTOM TABS
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
tab_pos, tab_bot, tab_history = st.tabs(
    ["Open Positions", "Bot Trades", "Trade History"]
)

# ── Open Positions ─────────────────────────────────────────────────────────────
with tab_pos:
    if not open_pos.empty:
        st.subheader("Bot Positions")
        display_cols = [
            "strategy", "symbol", "direction", "entry_time", "entry_price",
            "stop_loss", "take_profit",
        ]
        available = [c for c in display_cols if c in open_pos.columns]
        st.dataframe(open_pos[available], use_container_width=True, hide_index=True)
    else:
        st.info(f"No open bot positions for {sym_key}.")

    try:
        from data import manual_trade_db as mdb
        manual_open = mdb.get_open_trades()
        if manual_open:
            st.subheader("Manual Positions")
            mdf = pd.DataFrame(manual_open)
            mcols = [
                "trade_id", "symbol", "direction", "entry_price", "quantity",
                "stop_loss", "take_profit", "entry_time",
            ]
            st.dataframe(
                mdf[[c for c in mcols if c in mdf.columns]],
                use_container_width=True, hide_index=True,
            )
            for trade in manual_open:
                if st.button(
                    f"Close {trade['trade_id'][:8]}...",
                    key=f"close_m_{trade['trade_id']}",
                ):
                    try:
                        from data import binance_trading as bapi
                        exit_side = "SELL" if trade["direction"] == "Long" else "BUY"
                        result = bapi.place_market_order(
                            trade["symbol"], exit_side, trade["quantity"]
                        )
                        fills = result.get("fills", [])
                        total_qty = sum(float(f["qty"]) for f in fills) if fills else 0
                        exit_px = (
                            sum(float(f["price"]) * float(f["qty"]) for f in fills)
                            / total_qty if total_qty else 0
                        )
                        fees = (
                            sum(float(f["commission"]) for f in fills) if fills else 0
                        )
                        mdb.close_trade(
                            trade["trade_id"], exit_px, "manual",
                            str(result.get("orderId", "")), fees,
                        )
                        st.success(f"Closed at ${exit_px:,.2f}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Close failed: {e}")
        elif open_pos.empty:
            pass
    except ImportError:
        pass

# ── Bot Trades ─────────────────────────────────────────────────────────────────
with tab_bot:
    if closed_trades.empty:
        st.info(f"No closed bot trades for {sym_key}.")
    else:
        display_cols = [
            "strategy", "direction", "entry_time", "exit_time",
            "entry_price", "exit_price", "pnl_usd", "r_multiple",
            "session", "exit_reason",
        ]
        available = [c for c in display_cols if c in closed_trades.columns]
        recent = closed_trades.head(30)
        st.dataframe(
            recent[available].style.applymap(
                lambda v: "color: #26a69a" if isinstance(v, (int, float)) and v > 0
                else "color: #ef5350" if isinstance(v, (int, float)) and v < 0
                else "",
                subset=["pnl_usd"] if "pnl_usd" in available else [],
            ),
            use_container_width=True, hide_index=True,
        )

# ── Trade History (manual) ─────────────────────────────────────────────────────
with tab_history:
    try:
        from data import manual_trade_db as mdb2
        closed = mdb2.get_closed_trades(limit=50)
        if not closed:
            st.info("No closed manual trades.")
        else:
            stats = mdb2.get_trade_stats()
            hcols = st.columns(4)
            hcols[0].metric("Trades", stats["total"])
            hcols[1].metric("Win Rate", f"{stats['win_rate']}%")
            hcols[2].metric("Total P&L", f"${stats['total_pnl'] or 0:,.2f}")
            hcols[3].metric("Best", f"${stats['best_trade'] or 0:,.2f}")

            tdf = pd.DataFrame(closed)
            tcols_list = [
                "trade_id", "symbol", "direction", "entry_price", "exit_price",
                "pnl_usd", "exit_reason", "entry_time", "exit_time",
            ]
            st.dataframe(
                tdf[[c for c in tcols_list if c in tdf.columns]],
                use_container_width=True, hide_index=True,
            )
    except ImportError:
        st.info("Manual trading modules not available.")
