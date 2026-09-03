"""Page 3b: Live bot logs streamed from VPS via SSH."""

import re
from datetime import datetime

import streamlit as st

st.set_page_config(page_title="Live Logs", page_icon="\U0001F4DC", layout="wide")
st.title("\U0001F4DC Live Bot Logs")

from config import BOT_SERVICES, SERVICE_LOG_FILES, TRADING_REFRESH_OPTIONS
from data.vps_sync import run_ssh_command, get_bot_service_status, check_vps_reachable

# ── Auto-Refresh ─────────────────────────────────────────────────────────────
try:
    from streamlit_autorefresh import st_autorefresh

    refresh_label = st.sidebar.selectbox(
        "Auto-Refresh",
        list(TRADING_REFRESH_OPTIONS.keys()),
        index=0,
        key="logs_refresh_sel",
        help="Automatically re-fetch logs at this interval.",
    )
    interval = TRADING_REFRESH_OPTIONS[refresh_label]
    if interval > 0:
        st_autorefresh(interval=interval, key="logs_autorefresh")
except ImportError:
    st.sidebar.info("Install `streamlit-autorefresh` for auto-refresh.")

# ── Sidebar Controls ─────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("Bot Selection")

# Build display labels from BOT_SERVICES — "Family · seat" so the 80-odd seats
# sort by family in the picker. Journal-only units are marked: their output is
# not on disk where the trader login can read it.
service_labels = {
    svc: f"{info['strategy']} · {info['symbol']}" + ("" if info.get("log") else "  (journal-only)")
    for svc, info in sorted(BOT_SERVICES.items(), key=lambda kv: (kv[1]["strategy"], kv[1]["symbol"]))
}
selected_label = st.sidebar.selectbox(
    "Bot",
    list(service_labels.values()),
    key="logs_bot",
    help="Select which bot's logs to display.",
)
# Reverse-lookup service name from label
selected_service = next(
    svc for svc, label in service_labels.items() if label == selected_label
)

st.sidebar.markdown("---")
st.sidebar.subheader("Log Settings")

num_lines = st.sidebar.slider(
    "Lines to fetch",
    min_value=25,
    max_value=500,
    value=100,
    step=25,
    key="logs_lines",
    help="Number of most recent log lines to retrieve from the VPS.",
)

grep_pattern = st.sidebar.text_input(
    "Search / grep filter",
    value="",
    key="logs_grep",
    help="Filter log lines on the server side (case-insensitive). "
    "Examples: 'signal', 'error', 'price'.",
)

level_filter = st.sidebar.multiselect(
    "Log level filter",
    ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    default=["INFO", "WARNING", "ERROR", "CRITICAL"],
    key="logs_level",
    help="Show only lines matching these log levels (client-side filter).",
)

# ── VPS Reachability Check ───────────────────────────────────────────────────
st.caption(
    "Reads the bot's log file from the VPS via SSH. "
    "Requires SSH connectivity."
)

vps_ok = check_vps_reachable()
if not vps_ok:
    st.warning(
        "VPS is unreachable. Check your SSH configuration and network connection. "
        "Logs cannot be fetched while the VPS is offline.",
        icon="\u26A0\uFE0F",
    )
    st.stop()

# ── Service Status ───────────────────────────────────────────────────────────
svc_status = get_bot_service_status(selected_service)
status_icons = {
    "active": ("\U0001F7E2", "Active"),
    "inactive": ("\U0001F534", "Inactive"),
    "failed": ("\U0001F534", "Failed"),
    "activating": ("\U0001F7E1", "Starting..."),
    "deactivating": ("\U0001F7E1", "Stopping..."),
}
icon, label = status_icons.get(svc_status, ("\U0001F7E1", svc_status.title()))

_kind = BOT_SERVICES[selected_service].get("kind", "service")
c1, c2, c3 = st.columns([2, 2, 3])
c1.metric("Bot", selected_label.replace("  (journal-only)", ""))
c2.metric("Status" if _kind != "cron" else "Log freshness", f"{icon} {label}")
c3.metric("Unit" if _kind != "cron" else "Cron seat",
          f"`{selected_service}`" if _kind != "cron" else "no systemd unit")

st.markdown("---")

# ── Fetch Logs ───────────────────────────────────────────────────────────────
# Read the log file directly via tail (journalctl requires systemd-journal group)
log_file = SERVICE_LOG_FILES.get(selected_service)
if not log_file:
    st.warning(
        f"**`{selected_service}` logs only to the systemd journal**, which the read-only "
        "`trader` login cannot open (that needs the `adm`/`systemd-journal` group). "
        "Its trades are still on **🛰️ Live Fleet**. To make its log readable here, the "
        "operator adds `StandardOutput=append:/home/trader/trading_bots/logs/<name>.log` "
        "to the unit (as `admin`) and reloads — a small plumbing job, not a code change."
    )
    st.stop()
cmd = f"tail -n {num_lines} {log_file}"

# Server-side grep for performance (fewer bytes over SSH)
if grep_pattern.strip():
    # Sanitise: remove shell-dangerous characters
    safe_pattern = re.sub(r"[;&|`$(){}\\\"']", "", grep_pattern.strip())
    if safe_pattern:
        cmd += f" | grep -i -- {safe_pattern!r}"

with st.spinner("Fetching logs from VPS..."):
    result = run_ssh_command(cmd)

fetch_time = datetime.now().strftime("%H:%M:%S")

if result["returncode"] != 0 and not result["stdout"]:
    # Log file might not exist or tail failed
    stderr = result["stderr"] or "No output"
    if "No such file" in stderr:
        st.error(
            f"Log file not found on VPS: `{log_file}`\n\n"
            f"The bot may not have started yet or uses a different log path."
        )
    else:
        st.error(
            f"Failed to fetch logs.\n\n"
            f"**stderr:** {stderr}\n\n"
            f"**Command:** `{cmd}`"
        )
    st.stop()

raw_logs = result["stdout"]

# ── Client-Side Log Level Filter ─────────────────────────────────────────────
if raw_logs and level_filter:
    # Build a regex that matches common log level patterns
    # e.g., "- INFO -", "- WARNING -", "INFO:", "ERROR:", "[INFO]", etc.
    level_pattern = "|".join(level_filter)
    level_re = re.compile(
        rf"(?:^|-\s*|:\s*|\[\s*)(?:{level_pattern})(?:\s*-|\s*:|\s*\]|\s)",
        re.IGNORECASE,
    )
    lines = raw_logs.split("\n")
    filtered = [ln for ln in lines if level_re.search(ln)]
    display_logs = "\n".join(filtered)
    filtered_count = len(lines) - len(filtered)
else:
    display_logs = raw_logs
    filtered_count = 0

# ── Display Logs ─────────────────────────────────────────────────────────────
total_lines = display_logs.count("\n") + 1 if display_logs else 0
info_parts = [f"**{total_lines}** lines displayed"]
if filtered_count > 0:
    info_parts.append(f"{filtered_count} hidden by level filter")
if grep_pattern.strip():
    info_parts.append(f"grep: `{grep_pattern.strip()}`")
info_parts.append(f"fetched at {fetch_time}")

st.caption(" | ".join(info_parts))

if display_logs.strip():
    st.code(display_logs, language="log", line_numbers=True)
else:
    if grep_pattern.strip():
        st.info(
            f"No log lines match the filter '{grep_pattern}'. "
            "Try a different search term or clear the filter."
        )
    elif svc_status in ("inactive", "failed"):
        st.info(
            f"No logs available. The service is **{svc_status}**. "
            "Start the bot from the Deploy Bots page to generate logs."
        )
    else:
        st.info("No log lines returned. The bot may have just started.")

# ── Manual Refresh ───────────────────────────────────────────────────────────
st.markdown("---")
c1, c2 = st.columns([1, 5])
if c1.button("Refresh", type="primary", key="logs_manual_refresh"):
    st.rerun()
c2.caption(f"Last fetch: {fetch_time}")
