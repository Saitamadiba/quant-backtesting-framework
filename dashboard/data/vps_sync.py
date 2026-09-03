"""Sync SQLite databases from VPS via rsync/scp over SSH."""

import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import streamlit as st

from config import (
    VPS_HOST, VPS_PORT, VPS_USER, VPS_DB_FILES, VPS_ML_FILES, VPS_CACHE_DIR,
    SERVICE_WORK_DIRS, VPS_BACKUP_SCRIPT, VPS_REMOTE_BASE, FLEET_CACHE_DIR,
    BOT_SERVICES, fleet_db_relpaths,
)


def _ssh_opts() -> list:
    """Common SSH options shared by ssh, scp, and rsync."""
    return ["-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10"]


def _ssh_args() -> list:
    """Args for ssh (uses lowercase -p for port)."""
    return _ssh_opts() + ["-p", str(VPS_PORT)]


def _scp_args() -> list:
    """Args for scp (uses uppercase -P for port)."""
    return _ssh_opts() + ["-P", str(VPS_PORT)]


def sync_single_file(local_name: str, remote_path: str) -> Dict:
    """Rsync a single DB file from VPS. Returns status dict."""
    VPS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_path = VPS_CACHE_DIR / local_name

    # Use --checksum to force content comparison (not just size/mtime)
    # so that unchanged files still get verified and the local mtime
    # is updated to reflect a successful sync check.
    remote = f"{VPS_USER}@{VPS_HOST}:{remote_path}"
    cmd = [
        "rsync", "-az", "--checksum", "--timeout=15",
        "-e", f"ssh {' '.join(_ssh_args())}",
        remote, str(local_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            # Touch local file so dashboard sees it as freshly synced
            local_path.touch()
            size_kb = round(local_path.stat().st_size / 1024, 1)
            return {
                "file": local_name, "status": "ok",
                "time": datetime.now().isoformat(), "size_kb": size_kb,
            }
        return {"file": local_name, "status": "error", "error": result.stderr.strip()}
    except subprocess.TimeoutExpired:
        return {"file": local_name, "status": "timeout"}
    except FileNotFoundError:
        # rsync not installed – fall back to scp
        return _scp_fallback(local_name, remote_path)
    except Exception as e:
        return {"file": local_name, "status": "error", "error": str(e)}


def _scp_fallback(local_name: str, remote_path: str) -> Dict:
    """Fallback to scp if rsync unavailable."""
    local_path = VPS_CACHE_DIR / local_name
    remote = f"{VPS_USER}@{VPS_HOST}:{remote_path}"
    cmd = ["scp"] + _scp_args() + [remote, str(local_path)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return {"file": local_name, "status": "ok", "time": datetime.now().isoformat()}
        return {"file": local_name, "status": "error", "error": result.stderr.strip()}
    except Exception as e:
        return {"file": local_name, "status": "error", "error": str(e)}


def sync_vps_databases() -> Dict[str, Dict]:
    """Sync all VPS trade databases. Returns {filename: status_dict}."""
    results = {}
    for local_name, remote_path in VPS_DB_FILES.items():
        results[local_name] = sync_single_file(local_name, remote_path)
    return results


def sync_vps_ml_data() -> Dict[str, Dict]:
    """Sync all ML training databases from VPS. Returns {filename: status_dict}."""
    results = {}
    for local_name, remote_path in VPS_ML_FILES.items():
        results[local_name] = sync_single_file(local_name, remote_path)
    return results


def sync_fleet_books(rel_paths=None) -> Dict[str, Dict]:
    """Pull every Tier-1/2 fleet book in ONE rsync (one SSH handshake), keeping
    the VPS-relative path under databases/fleet/ so the fleet registry's paths
    resolve locally exactly as they do on the box."""
    rel_paths = list(rel_paths if rel_paths is not None else fleet_db_relpaths())
    if not rel_paths:
        return {}
    FLEET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    listing = "\n".join(rel_paths) + "\n"
    cmd = [
        "rsync", "-az", "--checksum", "--timeout=60", "--files-from=-",
        "-e", f"ssh {' '.join(_ssh_args())}",
        f"{VPS_USER}@{VPS_HOST}:{VPS_REMOTE_BASE}/", str(FLEET_CACHE_DIR) + "/",
    ]
    try:
        res = subprocess.run(cmd, input=listing, capture_output=True, text=True,
                             timeout=240)
    except subprocess.TimeoutExpired:
        return {rp: {"file": rp, "status": "timeout"} for rp in rel_paths}
    except FileNotFoundError:
        return {rp: sync_single_file(rp, f"{VPS_REMOTE_BASE}/{rp}") for rp in rel_paths}
    out = {}
    for rp in rel_paths:
        lp = FLEET_CACHE_DIR / rp
        if lp.exists():
            out[rp] = {"file": rp, "status": "ok", "time": datetime.now().isoformat(),
                       "size_kb": round(lp.stat().st_size / 1024, 1)}
        else:
            out[rp] = {"file": rp, "status": "missing",
                       "error": (res.stderr or "").strip()[:200] if res.returncode else
                                "absent on VPS"}
    return out


def sync_all_vps_data() -> Dict[str, Dict]:
    """Sync the legacy trade DBs, the ML training DBs and every fleet book.
    Returns {filename: status_dict}."""
    results = sync_vps_databases()
    results.update(sync_vps_ml_data())
    results.update(sync_fleet_books())
    return results


def check_vps_reachable() -> bool:
    """Quick check if VPS is reachable via SSH."""
    cmd = ["ssh"] + _ssh_args() + [f"{VPS_USER}@{VPS_HOST}", "echo ok"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.returncode == 0 and "ok" in result.stdout
    except Exception:
        return False


def get_cached_db_status() -> Dict[str, Dict]:
    """Check which DB files exist in local cache and their age."""
    status = {}
    all_files = {**VPS_DB_FILES, **VPS_ML_FILES}
    for local_name in all_files:
        path = VPS_CACHE_DIR / local_name
        if path.exists():
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            age_min = (datetime.now() - mtime).total_seconds() / 60
            status[local_name] = {
                "exists": True,
                "modified": mtime.isoformat(),
                "age_minutes": round(age_min, 1),
                "size_kb": round(path.stat().st_size / 1024, 1),
            }
        else:
            status[local_name] = {"exists": False}
    return status


def run_ssh_command(command: str) -> Dict:
    """Run a command on VPS via SSH. Returns stdout/stderr/returncode."""
    cmd = ["ssh"] + _ssh_args() + [f"{VPS_USER}@{VPS_HOST}", command]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return {
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "SSH command timed out", "returncode": -1}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "returncode": -1}


def get_bot_service_status(service_name: str) -> str:
    """Health of one fleet seat as the VPS reports it.

    A systemd unit (``x.service`` / ``x.timer``) answers with ``is-active``. A
    cron seat has no unit, so its health is the age of its log: written in the
    last 3 hours → "active", older → "stale", no log → "cron". Bare names get
    ``.service`` appended for the legacy callers.
    """
    info = BOT_SERVICES.get(service_name, {})
    kind = info.get("kind", "service")
    if kind == "cron":
        log = info.get("log")
        if not log:
            return "cron"
        r = run_ssh_command(f"stat -c %Y {log} 2>/dev/null || echo 0")
        try:
            age = datetime.now().timestamp() - float(r["stdout"].strip().split()[-1])
        except (ValueError, IndexError):
            return "unreachable" if r["returncode"] != 0 else "cron"
        return "active" if age < 3 * 3600 else "stale"
    unit = service_name if service_name.endswith((".service", ".timer")) else f"{service_name}.service"
    result = run_ssh_command(f"systemctl is-active {unit} 2>/dev/null || echo unknown")
    return parse_is_active(result["stdout"]) if result["returncode"] == 0 else "unreachable"


def parse_is_active(stdout: str) -> str:
    """`systemctl is-active` prints the state AND exits non-zero for anything but
    active, so the `|| echo unknown` fallback appends a second line. The state is
    the FIRST line; "unknown" only when nothing else was printed. (The old
    last-line read turned every stopped unit into "unknown".)"""
    lines = [ln.strip() for ln in (stdout or "").split("\n") if ln.strip()]
    return lines[0] if lines else "unknown"


PROTECTED_EXTENSIONS = {".db", ".sqlite", ".sqlite3"}


def remote_file_stat(remote_path: str) -> Dict:
    """mtime / size / md5 of a VPS file, or {"exists": False}. Read-only."""
    r = run_ssh_command(
        f"if [ -f {remote_path!r} ]; then stat -c '%Y %s' {remote_path!r}; "
        f"md5sum {remote_path!r} | cut -d' ' -f1; else echo ABSENT; fi")
    out = (r.get("stdout") or "").strip().split("\n")
    if r.get("returncode") != 0 or not out or out[0] == "ABSENT":
        return {"exists": False, "reachable": r.get("returncode") == 0}
    try:
        mtime, size = out[0].split()
        return {"exists": True, "reachable": True, "mtime": float(mtime),
                "size": int(size), "md5": out[1] if len(out) > 1 else ""}
    except (ValueError, IndexError):
        return {"exists": False, "reachable": False}


def deploy_guard(local_stat: Dict, remote: Dict) -> Dict:
    """The diff-first rule as a decision: may this local file overwrite the VPS copy?

    Several sessions deploy to the same box, so the VPS copy — not the local
    checkout — is the source of truth. Refuse when the remote copy is NEWER or
    LONGER than ours (someone else's finished work would be thrown away), pass
    when it is identical (nothing to do) or older-and-not-longer. Never
    force from here; the override is a human's, on the command line.
    """
    if not remote.get("reachable", True):
        return {"ok": False, "reason": "VPS unreachable — cannot compare, will not deploy blind"}
    if not remote.get("exists"):
        return {"ok": True, "reason": "no VPS copy yet"}
    if remote.get("md5") and remote["md5"] == local_stat.get("md5"):
        return {"ok": True, "reason": "identical — nothing to deploy", "noop": True}
    if remote["mtime"] > local_stat["mtime"] + 1:
        return {"ok": False,
                "reason": "VPS copy is NEWER than local — another session changed it; "
                          "pull and merge first"}
    if remote["size"] > local_stat["size"]:
        return {"ok": False,
                "reason": "VPS copy is LONGER than local — it likely carries work you "
                          "don't have; pull and merge first"}
    return {"ok": True, "reason": "local is newer and not shorter"}


def deploy_file_to_vps(local_path: str, remote_path: str) -> Dict:
    """Upload a local file to the VPS via scp — behind the diff guard.

    Returns status "blocked" (with the reason) instead of overwriting when the
    VPS copy is newer or longer; the fix is to reconcile, never to force.
    """
    import hashlib
    local = Path(local_path)
    if not local.exists():
        return {"status": "error", "error": f"Local file not found: {local}"}

    if local.suffix.lower() in PROTECTED_EXTENSIONS:
        return {
            "status": "error",
            "error": f"BLOCKED: refusing to deploy {local.name} — "
                     f"database files must not be overwritten on VPS",
        }

    lstat = local.stat()
    local_stat = {"mtime": lstat.st_mtime, "size": lstat.st_size,
                  "md5": hashlib.md5(local.read_bytes()).hexdigest()}
    verdict = deploy_guard(local_stat, remote_file_stat(remote_path))
    if not verdict["ok"]:
        return {"status": "blocked", "error": verdict["reason"]}
    if verdict.get("noop"):
        return {"status": "ok", "noop": True, "time": datetime.now().isoformat(),
                "size_kb": round(lstat.st_size / 1024, 1), "note": verdict["reason"]}

    # Back up remote DBs in the target directory before deploying
    remote_dir = str(Path(remote_path).parent)
    run_ssh_command(f"bash {VPS_BACKUP_SCRIPT} {remote_dir}")

    remote = f"{VPS_USER}@{VPS_HOST}:{remote_path}"
    cmd = ["scp"] + _scp_args() + [str(local), remote]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return {
                "status": "ok",
                "time": datetime.now().isoformat(),
                "size_kb": round(local.stat().st_size / 1024, 1),
            }
        return {"status": "error", "error": result.stderr.strip()}
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def backup_bot_dbs(service_name: str) -> Dict:
    """Back up .db files for a bot on the VPS before restart/deploy."""
    work_dir = SERVICE_WORK_DIRS.get(service_name)
    if not work_dir:
        return {"success": False, "error": f"No work dir for service: {service_name}"}
    result = run_ssh_command(f"bash {VPS_BACKUP_SCRIPT} {work_dir}")
    return {
        "success": result["returncode"] == 0,
        "stdout": result["stdout"],
        "stderr": result["stderr"],
    }


def manage_bot_service(service_name: str, action: str) -> Dict:
    """Prepare a start/stop/restart — the dashboard never runs sudo.

    The `trader` login the dashboard uses has no sudo (by design: a compromised
    bot must not be able to escalate), and a service restart is the operator's
    action as `admin`. So this backs the seat's databases up (non-sudo) and
    returns the EXACT command for the operator to paste in an admin session;
    it does not attempt to run it. `success` is always False — nothing was
    restarted — and `operator_command` is the deliverable.
    """
    if action not in ("start", "stop", "restart"):
        return {"success": False, "error": f"Invalid action: {action}"}
    unit = service_name if service_name.endswith((".service", ".timer")) else f"{service_name}.service"
    if BOT_SERVICES.get(service_name, {}).get("kind") == "cron":
        return {"success": False, "operator_command": None,
                "stderr": f"{service_name} is a cron seat — there is no unit to {action}; "
                          "edit the crontab as the operator instead."}
    backup_result = None
    if action in ("restart", "stop"):
        backup_result = backup_bot_dbs(service_name)
    resp = {
        "success": False,
        "operator_command": f"sudo systemctl {action} {unit}",
        "stdout": "",
        "stderr": f"Not run from the dashboard: `trader` has no sudo. Paste the command "
                  f"in an `admin` session on the VPS, then confirm with `systemctl status {unit}`.",
    }
    if backup_result:
        resp["backup"] = backup_result
    return resp
