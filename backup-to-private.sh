#!/usr/bin/env bash
# backup-to-private.sh — snapshot the PROPRIETARY code from this working tree into
# the private backup repo, then commit. The working tree (this dir) stays the SINGLE
# source of truth; the private repo is a versioned, secret-free mirror.
#
# Excludes (airtight, belt-and-suspenders with the private repo's own .gitignore):
#   secrets  *.env (keeps *.env.example), *token*, *risk_state*
#   models   *.pkl *.joblib
#   data     *.db* *.duckdb *.parquet *.log + the multi-GB duckdb_data/ tick_data/
#
# Usage:  ./backup-to-private.sh            # mirror + commit
#         QUANT_PRIVATE_REPO=/path ./backup-to-private.sh
#         ./backup-to-private.sh --no-commit
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
PRIV="${QUANT_PRIVATE_REPO:-$HOME/Quant-Backtesting-private}"
[ -d "$PRIV/.git" ] || { echo "ERROR: private repo not found at $PRIV (set QUANT_PRIVATE_REPO)"; exit 1; }

RX=(--exclude='*.env' --exclude='telegram.json' --exclude='*.pkl' --exclude='*.joblib' --exclude='*.db'
    --exclude='*.db-wal' --exclude='*.db-shm' --exclude='*.duckdb' --exclude='*.log'
    --exclude='*.parquet' --exclude='*risk_state*' --exclude='__pycache__/'
    --exclude='*.pyc' --exclude='*.bak' --exclude='*.bak.*' --exclude='*.bak_*'
    --exclude='*.localbak.*'
    # vendored Python environments / installed packages (NOT our code)
    --exclude='venv/' --exclude='.venv/' --exclude='*_env/'
    --exclude='site-packages/' --exclude='*.so' --exclude='*.pyi'
    --exclude='*.dist-info/' --exclude='*.egg-info/' --exclude='node_modules/'
    # bulky non-code media / course assets
    --exclude='Course_Materials/' --exclude='*.mp4' --exclude='*.mov'
    --exclude='*.mkv' --exclude='*.webm' --exclude='*.zip' --exclude='*.tar'
    --exclude='*.tar.gz' --exclude='*.tgz')

cd "$SRC"
# 2026-07-27: Liquidity_Raid_Reversal ADDED. It was untracked in the public repo
# AND absent from this mirror, so the LRR/LR/MM detector core (core/session_sweep.py,
# core/detector.py — the signal logic depth_logger.py and the LRR scanner both import)
# existed ONLY in the local working tree with no backup anywhere.
rsync -a --delete "${RX[@]}" HyroTrader Displacement \
      Liquidity_Raid Liquidity_Raid_Reversal Momentum_Mastery SBS FVG_Strategy shared \
      Vol_Edge Momentum_4H_Trend ofcs_shadow ofcs_demo ifvg_shadow fib618_shadow smc_demo smc12h4h_demo sweep_engine "$PRIV/"
# full feature_lab (188 research scripts + tests + md); RX excludes the heavy
# reports/ parquets/dbs/logs so only the code + notes are mirrored. tests/ holds the
# top-level proprietary bot tests (knife_*, etc.) — mirrored too per CLAUDE.md.
rsync -a --delete "${RX[@]}" feature_lab books_indicator_battery liquidity_surf desk_demo retest_demo lrr_short_demo lr_wide_demo tests "$PRIV/"
# 2026-08-05: research_output/ ADDED — replay CSVs + the scripts that produced
# them (e.g. the OFCS skipped-trade tick reconstruction). Gitignored in public,
# and until now mirrored nowhere, so a study lived only in the working tree.
[ -d research_output ] && rsync -a --delete "${RX[@]}" research_output "$PRIV/"
# root-level proprietary scripts (no subdir deletion semantics needed)
cp -p replay_*.py deploy_*.sh session_pnl_snapshot.sh backup-to-private.sh backup-secrets.sh "$PRIV/" 2>/dev/null || true
# standing research docs: preregistrations, runbooks, specs, standards (secret-free
# bar documents — 2026-07-28: PATTERN_GATE_PREREG etc. were previously mirrored NOWHERE)
cp -p *_PREREG.md *_RUNBOOK.md *_SPEC.md *_STANDARD.md *_MAP.md *_PLAN.md *_DESIGN.md *_STATUS.md *_RISK.md "$PRIV/" 2>/dev/null || true

# SAFETY: never let a real secret env-file into the backup
if git -C "$PRIV" status --porcelain | grep -qE '\.env$'; then
  echo "ABORT: a real .env would be committed — check the excludes"; exit 1
fi

git -C "$PRIV" add -A
if git -C "$PRIV" diff --cached --quiet; then echo "no changes to back up."; exit 0; fi
[ "${1:-}" = "--no-commit" ] && { echo "staged (‑‑no-commit); review with: git -C $PRIV diff --cached"; exit 0; }
git -C "$PRIV" commit -q -m "backup: proprietary code snapshot $(date -u +%Y-%m-%dT%H:%MZ)"
echo "committed. push (your step): git -C $PRIV push"
