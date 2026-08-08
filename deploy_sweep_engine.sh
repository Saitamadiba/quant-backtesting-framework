#!/usr/bin/env bash
# deploy_sweep_engine.sh — deploy the Tier-3 SweepEngine shadow to the VPS.
# Record-only; NO order path. Diff-guarded, sudo-free, never ships *.db.
# Usage: ./deploy_sweep_engine.sh [--dry] [--force]
set -euo pipefail
set -a; . ~/.config/quant/deploy.env; set +a
PORT="${VPS_PORT:-22}"; DEST="${VPS_USER}@${VPS_HOST}"
RDIR="/home/trader/trading_bots/sweep_engine"
LDIR="$(cd "$(dirname "$0")/sweep_engine" && pwd)"
FILES=(__init__.py _vendor.py levels.py engine.py shadow.py compare.py SPEC.md)
CRON="1,16,31,46 * * * * cd /home/trader/trading_bots && /usr/bin/python3 -m sweep_engine.shadow >> /home/trader/trading_bots/logs/sweep_engine.log 2>&1"
# Detection and booking are two phases: an event resolves its lifecycle now but
# its max_hold horizon only elapses 12h later, so a separate pass books it.
CRON_RESOLVE="7,37 * * * * cd /home/trader/trading_bots && /usr/bin/python3 -m sweep_engine.shadow --resolve >> /home/trader/trading_bots/logs/sweep_engine.log 2>&1"
DRY=0; FORCE=0
for a in "$@"; do case "$a" in --dry) DRY=1;; --force) FORCE=1;; esac; done
ssh -p "$PORT" -o ConnectTimeout=12 "$DEST" "mkdir -p $RDIR /home/trader/trading_bots/logs"
echo "== diff-guard =="
BLOCKED=0
for f in "${FILES[@]}"; do
  [ -f "$LDIR/$f" ] || { echo "ABORT: local $f missing"; exit 1; }
  lsize=$(stat -f %z "$LDIR/$f" 2>/dev/null || stat -c %s "$LDIR/$f")
  lmtime=$(stat -f %m "$LDIR/$f" 2>/dev/null || stat -c %Y "$LDIR/$f")
  read -r rsize rmtime < <(ssh -p "$PORT" "$DEST" "stat -c '%s %Y' $RDIR/$f 2>/dev/null || echo '0 0'")
  lmd5=$(md5 -q "$LDIR/$f" 2>/dev/null || md5sum "$LDIR/$f" | cut -d' ' -f1)
  rmd5=$(ssh -p "$PORT" "$DEST" "md5sum $RDIR/$f 2>/dev/null | cut -d' ' -f1")
  # Content first. A deploy always leaves the VPS mtime NEWER than local, so an
  # mtime-only test flags every UNCHANGED file on the next run — and a guard
  # that cries wolf is how a --force reflex gets trained. Identical bytes carry
  # no overwrite risk; only a file that genuinely DIFFERS and is newer/longer
  # means another session got there first.
  if [ "$rsize" != "0" ] && [ "$lmd5" = "$rmd5" ]; then
    echo "  ok: $f (identical)"
  elif [ "$rsize" != "0" ] && { [ "$rmtime" -gt "$lmtime" ] || [ "$rsize" -gt "$lsize" ]; }; then
    echo "  GUARD TRIP: $f (VPS copy DIFFERS and is newer/longer — reconcile)"
    BLOCKED=1
  else echo "  ok: $f"; fi
done
if [ "$BLOCKED" = "1" ]; then
  if [ "$FORCE" = "1" ] && [ -t 0 ]; then
    read -r -p "Overwrite newer VPS copy? [y/N] " a; [ "$a" = "y" ] || exit 1
  else echo "ABORT: reconcile first (--force interactive-only)."; exit 1; fi
fi
[ "$DRY" = "1" ] && { echo "--dry: stop."; exit 0; }
echo "== upload =="
for f in "${FILES[@]}"; do scp -P "$PORT" -q "$LDIR/$f" "$DEST:$RDIR/$f"; done
for f in "${FILES[@]}"; do
  l=$(md5 -q "$LDIR/$f" 2>/dev/null || md5sum "$LDIR/$f"|cut -d' ' -f1)
  r=$(ssh -p "$PORT" "$DEST" "md5sum $RDIR/$f|cut -d' ' -f1")
  [ "$l" = "$r" ] || { echo "MISMATCH $f"; exit 1; }
done
echo "  md5 ok"
echo "== cron =="
ssh -p "$PORT" "$DEST" "crontab -l 2>/dev/null | grep -qF 'sweep_engine.shadow >>' || (crontab -l 2>/dev/null; echo '$CRON') | crontab -"
ssh -p "$PORT" "$DEST" "crontab -l 2>/dev/null | grep -qF 'shadow --resolve' || (crontab -l 2>/dev/null; echo '$CRON_RESOLVE') | crontab -"
ssh -p "$PORT" "$DEST" "crontab -l | grep sweep_engine"
echo "== migrate (idempotent: additive columns, alias flagging; deletes nothing) =="
ssh -p "$PORT" "$DEST" "cd /home/trader/trading_bots && python3 -m sweep_engine.shadow --migrate"
echo "== smoke =="
ssh -p "$PORT" "$DEST" "cd /home/trader/trading_bots && timeout 300 python3 -m sweep_engine.shadow && timeout 300 python3 -m sweep_engine.shadow --resolve && python3 -m sweep_engine.shadow --stats"
echo "deploy complete (Tier-3 record-only, no order path)."
