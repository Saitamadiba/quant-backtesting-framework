#!/usr/bin/env bash
# Deploy desk_demo — pre-positioned maker desk, ByBit DEMO executor (2026-07-18).
# Succeeds the retired depth_maker arm (07-06, adverse selection at sweep-event
# design): same demo account, maker-first, ahead-of-time 1h/4h levels + regime
# dimmer. See desk_demo/SPEC.md (pre-registered config + intake review).
#
# Sudo-free by design: uploads as trader; cron line is trader crontab (printed,
# added with --cron). Diff-guard: refuses to overwrite a VPS copy that is newer
# or longer than local (another session's work) — reconcile first, never --force
# as a reflex.
set -euo pipefail

ENVF="${HOME}/.config/quant/deploy.env"
[ -r "$ENVF" ] || { echo "FATAL: missing $ENVF"; exit 1; }
set -a; . "$ENVF"; set +a
PORT="${VPS_PORT:-22}"; RUSER="${VPS_USER}"; RHOST="${VPS_HOST}"
RDIR="/home/trader/trading_bots"
LOCAL="$(cd "$(dirname "$0")" && pwd)"
VENV="$RDIR/venv/bin/python3"

ssh_() { ssh -p "$PORT" -o ConnectTimeout=15 "${RUSER}@${RHOST}" "$@"; }

FILES=(desk_demo/__init__.py desk_demo/engine.py desk_demo/bot.py \
       desk_demo/thresholds_frozen.json desk_demo/SPEC.md \
       desk_demo/desk_demo.env.example tests/test_desk_demo.py)

echo "== diff-guard =="
for f in desk_demo/engine.py desk_demo/bot.py; do
  if ssh_ "test -f '$RDIR/$f'"; then
    LOC=$(md5 -q "$LOCAL/$f" 2>/dev/null || md5sum "$LOCAL/$f" | cut -d' ' -f1)
    VPS=$(ssh_ "md5sum '$RDIR/$f'" | cut -d' ' -f1)
    if [ "$LOC" = "$VPS" ]; then
      continue                      # identical — idempotent re-run
    fi
    VL=$(ssh_ "wc -l < '$RDIR/$f'"); LL=$(wc -l < "$LOCAL/$f")
    if [ "$VL" -gt "$LL" ]; then
      echo "REFUSING: VPS $f is longer than local (another session?) — reconcile first."
      exit 2
    fi
    ssh_ "cp '$RDIR/$f' '$RDIR/$f.bak.$(date -u +%Y%m%dT%H%M%SZ)'"
  fi
done

echo "== upload (sudo-free, as trader) =="
ssh_ "mkdir -p '$RDIR/desk_demo' '$RDIR/tests' '$RDIR/logs'"
for f in "${FILES[@]}"; do
  scp -q -P "$PORT" "$LOCAL/$f" "${RUSER}@${RHOST}:$RDIR/$f"
done

echo "== md5 =="
for f in desk_demo/engine.py desk_demo/bot.py desk_demo/thresholds_frozen.json; do
  LOC=$(md5 -q "$LOCAL/$f" 2>/dev/null || md5sum "$LOCAL/$f" | cut -d' ' -f1)
  VPS=$(ssh_ "md5sum '$RDIR/$f'" | cut -d' ' -f1)
  [ "$LOC" = "$VPS" ] || { echo "FATAL: md5 mismatch on $f"; exit 1; }
done
echo "  md5 OK"

echo "== VPS tests + compile =="
ssh_ "cd '$RDIR' && '$VENV' -m pytest tests/test_desk_demo.py -q 2>&1 | tail -2"
ssh_ "cd '$RDIR' && '$VENV' -c 'import ast; ast.parse(open(\"desk_demo/bot.py\").read()); print(\"compile OK\")'"

echo
echo "== env file (server-side; secrets never leave the VPS) =="
echo "  If $RDIR/desk_demo/desk_demo.env is missing, create it AS TRADER from the"
echo "  existing demo credentials (same demo account as smc_demo), mode 0600."
echo
echo "== cron (trader, no sudo) — add with: $0 --cron =="
echo '  3,18,33,48 * * * * cd /home/trader/trading_bots && venv/bin/python3 -m desk_demo.bot >> logs/desk_demo.log 2>&1'

if [ "${1:-}" = "--cron" ]; then
  ssh_ 'crontab -l 2>/dev/null | grep -q "desk_demo.bot" || \
    (crontab -l 2>/dev/null; echo "3,18,33,48 * * * * cd /home/trader/trading_bots && venv/bin/python3 -m desk_demo.bot >> logs/desk_demo.log 2>&1") | crontab -'
  echo "cron installed."
fi
echo "DONE"
