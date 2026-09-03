"""Page 9: Deploy bot code to VPS."""

from datetime import datetime
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Deploy Bots", page_icon="🚀", layout="wide")
st.title("🚀 Deploy Bots to VPS")

from config import (
    DEPLOY_BOT_FILES, DEPLOY_SERVICE_MAP,
    VPS_HOST, VPS_PORT, VPS_USER,
)
from data.vps_sync import deploy_file_to_vps, get_bot_service_status, manage_bot_service

st.caption(f"VPS: {VPS_USER}@{VPS_HOST}:{VPS_PORT}")
st.markdown(
    "Upload updated **research-strategy** bot scripts (FVG / LR / MM / Straddle / SBS) "
    "to the VPS. Two house rules are enforced here: every upload passes the **diff "
    "guard** (it is refused if the VPS copy is newer or longer than yours — another "
    "session's finished work is never overwritten from a stale checkout), and the "
    "**restart is the operator's**: the `trader` login has no sudo, so the button hands "
    "you the exact `sudo systemctl` line for an `admin` session. The fleet seats "
    "deployed since July (knife, depth, OFCS, …) ship through their own `deploy_*.sh` "
    "scripts, which carry the same guard."
)

st.markdown("---")

# ── Track deploy results in session_state so restart buttons persist ──────────
if "deploy_results" not in st.session_state:
    st.session_state["deploy_results"] = {}  # {label: result_dict}
if "bulk_deploy_results" not in st.session_state:
    st.session_state["bulk_deploy_results"] = {}

# ── Per-bot deploy cards ──────────────────────────────────────────────────────
st.caption("Each card shows a bot's local file status and its systemd service status on the VPS. Deploy pushes your local version to the server via SCP. Always check the service status after deploying — a restart is needed for changes to take effect.")
cols_per_row = 3
items = list(DEPLOY_BOT_FILES.items())

for i in range(0, len(items), cols_per_row):
    cols = st.columns(cols_per_row)
    for j, (label, (local_path, remote_path)) in enumerate(items[i:i + cols_per_row]):
        with cols[j]:
            local_p = Path(local_path)
            exists = local_p.exists()
            size_kb = round(local_p.stat().st_size / 1024, 1) if exists else 0
            mtime = datetime.fromtimestamp(local_p.stat().st_mtime).strftime("%Y-%m-%d %H:%M") if exists else "N/A"

            # Service status
            svc = DEPLOY_SERVICE_MAP.get(label)
            svc_status = get_bot_service_status(svc) if svc else "unknown"
            svc_icon = "🟢" if svc_status == "active" else "🔴" if svc_status in ("inactive", "dead") else "🟡"

            st.markdown(f"### {label}")
            st.caption(f"`{local_p.name}` — {size_kb}KB — modified {mtime}")
            st.caption(f"Service `{svc}` {svc_icon} {svc_status}")

            if not exists:
                st.error("Local file not found")
                continue

            # Deploy button
            if st.button(f"Deploy {label}", key=f"deploy_{label}", type="primary", help="Upload this bot's script to the VPS via SCP. Service must be restarted afterward for changes to take effect."):
                with st.spinner(f"Uploading {local_p.name}..."):
                    result = deploy_file_to_vps(str(local_path), remote_path)
                st.session_state["deploy_results"][label] = result

            # Show result and restart button (persists across reruns)
            deploy_result = st.session_state["deploy_results"].get(label)
            if deploy_result:
                if deploy_result["status"] == "ok":
                    st.success(f"Deployed ({deploy_result.get('size_kb', '?')}KB)")
                    if svc:
                        if st.button(f"Restart {svc}", key=f"restart_after_{label}", help="Back the seat's DBs up and get the operator's restart command."):
                            r = manage_bot_service(svc, "restart")
                            st.code(r.get("operator_command") or "", language="bash")
                            st.caption(r.get("stderr", ""))
                elif deploy_result["status"] == "blocked":
                    st.warning(f"Deploy **blocked by the diff guard**: {deploy_result.get('error')}")
                else:
                    st.error(f"Deploy failed: {deploy_result.get('error', deploy_result['status'])}")

st.markdown("---")

# ── Deploy All ────────────────────────────────────────────────────────────────
st.subheader("Bulk Deploy")
st.caption("Deploy all bot scripts in one click. Useful after a batch update (e.g., new ML model integration or parameter changes). After bulk deploy, use 'Restart All Bots' to apply the changes across all services.")

if st.button("Deploy All Bots", type="primary", help="Deploy all bot scripts to VPS in parallel. Use after batch code updates."):
    results = {}
    progress = st.progress(0, text="Deploying...")
    total = len(DEPLOY_BOT_FILES)

    for idx, (label, (local_path, remote_path)) in enumerate(DEPLOY_BOT_FILES.items()):
        progress.progress((idx + 1) / total, text=f"Deploying {label}...")
        results[label] = deploy_file_to_vps(str(local_path), remote_path)

    progress.progress(1.0, text="Done")
    st.session_state["bulk_deploy_results"] = results

# Show bulk results and restart button (persists across reruns)
bulk_results = st.session_state.get("bulk_deploy_results", {})
if bulk_results:
    ok = sum(1 for r in bulk_results.values() if r["status"] == "ok")
    fail = len(bulk_results) - ok

    if fail == 0:
        st.success(f"All {ok} bots deployed successfully.")
    else:
        st.warning(f"{ok} deployed, {fail} failed.")

    for label, result in bulk_results.items():
        icon = "✅" if result["status"] == "ok" else "🛡️" if result["status"] == "blocked" else "❌"
        detail = (f"{result.get('size_kb', '')}KB" if result["status"] == "ok"
                  else result.get("error", ""))
        st.write(f"{icon} **{label}** — {detail}")

    st.markdown("---")
    if st.button("Restart commands for the deployed bots", key="restart_all_after_deploy",
                 help="Backs each seat's DBs up and lists the operator's sudo commands — nothing is restarted from here."):
        cmds = []
        for label, result in bulk_results.items():
            svc = DEPLOY_SERVICE_MAP.get(label)
            if svc and result["status"] == "ok" and not result.get("noop"):
                r = manage_bot_service(svc, "restart")
                if r.get("operator_command"):
                    cmds.append(r["operator_command"])
        if cmds:
            st.code("\n".join(cmds), language="bash")
            st.caption("Paste in an `admin` session on the VPS, then verify each with `systemctl status <unit>`.")
        else:
            st.info("Nothing changed on the VPS, so nothing needs a restart.")
        st.session_state["bulk_deploy_results"] = {}
