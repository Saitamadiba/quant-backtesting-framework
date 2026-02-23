"""Sync SQLite databases from VPS via rsync/scp over SSH."""

import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import streamlit as st

from config import (
    VPS_HOST, VPS_PORT, VPS_USER, VPS_DB_FILES, VPS_ML_FILES, VPS_CACHE_DIR,
    SERVICE_WORK_DIRS, VPS_BACKUP_SCRIPT,
)


def _ssh_opts() -> list:
    """Common SSH options shared by ssh, scp, and rsync."""
    return ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10"]


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

    remote = f"{VPS_USER}@{VPS_HOST}:{remote_path}"
    cmd = [
        "rsync", "-az", "--timeout=15",
        "-e", f"ssh {' '.join(_ssh_args())}",
        remote, str(local_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return {"file": local_name, "status": "ok", "time": datetime.now().isoformat()}
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


def sync_all_vps_data() -> Dict[str, Dict]:
    """Sync both trade databases and ML training data. Returns {filename: status_dict}."""
    results = sync_vps_databases()
    results.update(sync_vps_ml_data())
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
    """Get systemd service status for a bot on the VPS."""
    result = run_ssh_command(f"systemctl is-active {service_name}.service 2>/dev/null || echo unknown")
    status = result["stdout"].strip().split("\n")[-1] if result["returncode"] == 0 else "unreachable"
    return status


PROTECTED_EXTENSIONS = {".db", ".sqlite", ".sqlite3"}


def deploy_file_to_vps(local_path: str, remote_path: str) -> Dict:
    """Upload a local file to VPS via scp. Returns status dict."""
    local = Path(local_path)
    if not local.exists():
        return {"status": "error", "error": f"Local file not found: {local}"}

    if local.suffix.lower() in PROTECTED_EXTENSIONS:
        return {
            "status": "error",
            "error": f"BLOCKED: refusing to deploy {local.name} — "
                     f"database files must not be overwritten on VPS",
        }

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
    """Start/stop/restart a bot service on VPS. action: start|stop|restart"""
    if action not in ("start", "stop", "restart"):
        return {"success": False, "error": f"Invalid action: {action}"}

    # Back up databases before restart or stop (safety net)
    backup_result = None
    if action in ("restart", "stop"):
        backup_result = backup_bot_dbs(service_name)

    result = run_ssh_command(f"sudo systemctl {action} {service_name}.service")
    resp = {
        "success": result["returncode"] == 0,
        "stdout": result["stdout"],
        "stderr": result["stderr"],
    }
    if backup_result:
        resp["backup"] = backup_result
    return resp
