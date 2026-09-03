# Backtesting — project instructions

## Communication style (ALWAYS ON in this directory)

When explaining ANY work here — summaries, verdicts, design rationale, trade-offs,
risk caveats, why-this-not-that — **always pair the precise quant/trading
statement with a plain-English picture or metaphor a non-quant could repeat.**
The metaphor rides *alongside* the exact numbers (p-values, R-multiples, file
paths, window counts), never replaces them. Lead with the figure; the image is
the chaser.

Hold back where it would hurt: never bury the number, keep code/identifiers/
commit messages literal, one metaphor per idea (don't mix them), and drop any
analogy that distorts the mechanism. The reader is a sharp colleague over coffee,
not a child — simplify the language, never the claim.

Full house style, the "say it twice" rule, and a grounded metaphor vocabulary
(auction / sea-wall absorption / tiring-sprinter exhaustion / dimmer-not-switch
sizing / dress-rehearsal forward-shadow / harder-curve Bonferroni / confound-in-
disguise): see the **`plain-english-metaphors`** skill.

## Session-start fleet recap (ALWAYS ON in this directory)

**Open every new session with a two-part recap before anything else** — a cockpit
pre-flight: read the instrument panel and the last mechanic's note before you touch
the controls. A `SessionStart` hook (in `.claude/settings.local.json`) runs
`session_pnl_snapshot.sh --hook`, which does two things at once: it shows the live,
read-only PnL scoreboard to the user as a `systemMessage` the moment the session opens
(banner `=== FLEET SESSION RECAP — live PnL ===`, a **lifetime** column plus a rolling
**7-day** column), and it injects the same panel into your context as
`additionalContext`. So the user sees the numbers immediately; you add the research
half. Lead your first reply with both:

1. **What research just landed, and what it did to the bots.** Summarize the ~5–8
   most-recent `MEMORY.md` entries (already in context) in the house style — the exact
   result *and* a plain-English picture — and state each one's **impact**: which bot it
   deployed / killed / re-tuned / left record-only. (E.g. the 2026-07-20 knife
   taker-orphan booking fix: the closed-PnL matcher priced fills at the *intended* level
   not the *actual* one, so any taker entry that slipped >1% into the raid never booked
   and the risk guard was blind to the loss — 28.7% of taker fills orphaned vs 3.7% for
   maker; the picture is a till that only rings up the sales it expected. Fixed +
   deployed; 5 knife arms pending an operator restart.)

2. **Live fleet PnL, by tier**, straight from the injected snapshot:
   - **Tier 1 — Bybit live / funded / demo-order:** realized **$ PnL + R** — real orders
     on funded/demo seats (knife arms, LR/MM/OFCS funded, options, and the desk / retest
     / smc demo executors). This is the money line.
   - **Tier 2 — paper / virtual-book:** net **R** on a virtual $100k — "would this pass a
     challenge." Dollars here are simulated and uncapped, so read them as R, not a bank
     balance.
   - **Tier 3 — shadow / record-only:** dimensionless **cumulative R** by design (no
     sizing) — edge-measuring instruments, not P&L.

**Read the numbers honestly.** R is net of the fee/slip toll unless the label says gross
(e.g. `sweep-engine(gross)`). A deep-negative shadow (knife ≈ −1660R, antiknife ≈ −1290R)
is a *confirmed-dead detector still faithfully recording*, not a bug to chase — that is
the closure library doing its job, the black box on a plane that already landed. Keep the
recap **quick**: a compact digest, never a wall. If the session opens with a specific
narrow task, compress to a one-liner ("fleet nominal; knife arms still bleeding by
design; desk/retest the only green demo seats") and get to the task.

**Mechanism (for maintenance).** The scoreboard is `session_pnl_snapshot.sh` (repo root,
gitignored → private-routed like the deploy scripts): read-only `sqlite3 -readonly` over a
single SSH round-trip, connection from `~/.config/quant/deploy.env`, **fails CLOSED** to a
one-line notice if the VPS is unreachable — it never writes, restarts, or blocks a session.
Run it bare (`bash session_pnl_snapshot.sh`) for plain text by hand; the hook uses
`--hook`, which wraps the same panel as SessionStart JSON (`systemMessage` +
`additionalContext`). To add/rename a bot, edit the `ROWS` table in the script (one
`kind|path|table|R-expr|$-expr|filter|7d-timestamp-col|label` line per bot; `kind` A = has
R, B = dollars only). If the recap banner is absent (VPS down, or the hook not yet
reloaded — a brand-new session registers it; open `/hooks` once or restart if not), run the
script by hand or say PnL is unavailable this session.

## Connecting to the VPS (ALWAYS ON in this directory)

Routine, **read-only** interaction with the live VPS is normal and expected — you do not
need to ask before running NON-SUDO commands over SSH: tail a log, grep a config, list
listeners (`ss -tlnp`), check a service with `systemctl status` (status is non-sudo),
pull a DB or file to `/tmp` to inspect/diff. Treat the VPS like a filing cabinet you may
open and read freely; you just may not change the locks.

**Connection details come from the gitignored env-file — NEVER hardcode them here.** This
file lives in the PUBLIC repo, so host/IP/port live ONLY in `~/.config/quant/deploy.env`
(mode 0600), and the authorized SSH key (`~/.ssh/id_ed25519`, the `trader` user) is picked
up automatically — no `-i` needed. Reusable recipe:

```bash
set -a; . ~/.config/quant/deploy.env; set +a
ssh -p "${VPS_PORT:-22}" -o ConnectTimeout=12 "${VPS_USER}@${VPS_HOST}" '<read-only command>'
# pull a file to inspect/diff — into /tmp, NEVER into the repo working tree:
scp -P "${VPS_PORT:-22}" "${VPS_USER}@${VPS_HOST}:<remote-path>" /tmp/
```

**Two identities on the VPS — keep them straight (2026-06-19 hardening).** The bots run
as, and you connect/read as, the **`trader`** user (uid 1002), which is now **NON-sudo**
— it was deliberately removed from the `sudo` group so a compromised bot cannot escalate
to root. All sudo/root power lives with a SEPARATE human admin user, **`admin`** (uid
1003, in the `sudo` group). So: `trader` = the bots + your read-only SSH; `admin` = the
operator's privileged hand. Never expect `trader` to sudo — that is by design, not a
bug to route around.

**Sudo is the hard line — and it is the operator's (as `admin`), not yours.** You CANNOT
run sudo on the VPS (the auto-mode classifier blocks it, by design; and your `trader`
login has no sudo rights anyway) and must never try to work around it. Any root action —
`systemctl restart/stop/start`, package install, editing a root-owned file, truncating a
root log — is mine to run **as the `admin` user**. For those, print the EXACT command(s)
for me to paste **in an `admin` session** on the VPS, then wait and verify after I confirm
they ran; never assume a restart happened. Reading is the open door (as `trader`);
changing the machine is the door you knock on and hand me — `admin` — the key for.

This composes with the deploy rules below: reads are free, but before you EDIT or DEPLOY
any VPS file, the diff-first rule still applies — pull and diff the live copy first.

## Deploying to the live VPS (ALWAYS ON in this directory)

**Multiple instances of you run in parallel, and several may be deploying bot
updates to the same VPS at the same time.** Treat the VPS as the shared source of
truth, NOT your local checkout — another session may have pushed newer code
minutes ago. The live bots (e.g. `HyroTrader/`, `Displacement/`) are gitignored
and rsync-deployed, so git will NOT warn you about divergence; diffing against the
VPS is your only version control. Silently overwriting another session's work is
the failure we most need to prevent — it sends real, finished work to waste.

Before editing or deploying ANY file that runs on the VPS:

1. **Pull + diff the current VPS copy FIRST.** Sync the live file down (or run the
   deploy script's `--dry`/diff mode) and read it. If the VPS copy has changes you
   don't have locally, STOP and reconcile — another session made them. Never edit
   blind against a stale local copy.
2. **Never bypass the diff guard.** Every `deploy_*.sh` MUST refuse to overwrite a
   VPS copy that is newer or longer than local. If the guard trips, do NOT
   `--force` — investigate the divergence and merge it in first. `--force` is a
   deliberate, human-confirmed last resort, never a reflex.
3. **You cannot run sudo on the VPS** (the auto-mode classifier blocks it, and your
   `trader` login has no sudo rights — see the two-identity note above). Every
   `deploy_*.sh` MUST: (a) do all uploads sudo-free **as `trader`**; (b) gate any
   sudo step (systemctl restart/install) behind an interactive `[y/N]` prompt; and
   (c) skip sudo entirely in a non-interactive shell, printing the command for the
   operator to run **as `admin`** instead. The restart is ALWAYS the operator's
   action (as `admin`) — never assume it ran; verify after they confirm.
4. **After deploying, validate + record.** Confirm the new code/config actually
   landed (grep the VPS) and the services are healthy, then leave a memory note of
   what you deployed and when — so the next parallel session can see it and not
   undo it.

When in doubt, prefer the **`patch-deploy`** skill (core-vs-target separation,
dry-run, restart, validate) over a hand-rolled rsync. `deploy_knife_funded.sh` is
the reference implementation of the diff guard + sudo prompt above.

## Change backup, commit & sanitization (ALWAYS ON in this directory)

Nothing important may live ONLY on the VPS, and nothing sensitive may EVER reach
a repo. Every change is mirrored locally, committed at the right privacy level,
and sanitized before it lands in any repo.

1. **Mirror every VPS change back to local — keep it byte-identical.** Anything you
   change or deploy on the VPS (code, env, config) MUST also be saved in the local
   checkout and kept in sync. The VPS `trading_bots/` dir is NOT version-controlled,
   so the local copy is the only backup and the only version control. Apply the
   change locally and deploy FROM local (never hand-edit a VPS file that has no
   local twin); after deploying, local and VPS stay identical.

2. **Commit each settled change at its privacy/security level — never leave it
   uncommitted.** Route by sensitivity:
   - Non-sensitive code, docs, tests, generic tooling → the **public** repo.
   - Proprietary strategy logic, bot internals, live/funded configs, research
     artifacts, scorers → the **private** repo (the `.gitignore` already routes
     `HyroTrader/`, `Displacement/`, research outputs, etc. there).
   When unsure which repo, treat it as private.

3. **Sanitize anything bound for a PRIVATE repo — thoroughly, every time.** Before
   committing to the private repo, strip every secret and live identifier: API
   keys/secrets, bot tokens, chat IDs, account IDs/labels, credentials, host/IP/
   port, sub-account names, and any PII. Secrets live ONLY in gitignored env-files
   and are committed to NO repo, public or private. When unsure whether a value is
   sensitive, redact it — a sanitized placeholder in a private repo is always
   correct; a leaked token never is.

**Proprietary backup repo.** The gitignored proprietary code (`HyroTrader/`, tests,
`replay_*.py`, deploy scripts) is mirrored, secret-free and one-way, to a private repo
at `~/Quant-Backtesting-private` via `./backup-to-private.sh` (the working tree stays
the source of truth; `*.env`/`*.pkl`/`*.db`/data are excluded, config shape kept in
redacted `*.env.example`). Real secrets go to a gpg-AES256 bundle via
`./backup-secrets.sh` (your passphrase), never to git. Run both after settling changes.

## Never wipe live data (ALWAYS ON in this directory)

**Never delete, truncate, drop, or overwrite-with-empty a database or live data file
unless the user EXPLICITLY asks for that exact action.** The `*.db`/`*.duckdb` files hold
irreplaceable state and history — shadow books (`knife_shadow.db` episodes), funded trades,
feature stores, the tick re-resolution. None of it lives in any repo (DBs are gitignored), so
a wipe is permanent loss, not a recoverable commit. Concretely, never — as a side effect of any
task — run `> file.db`, `rm *.db*`, `DROP TABLE`, `TRUNCATE`, a `DELETE` without a `WHERE`, or
`open(path, "w")` over a populated data file. If a rebuild genuinely seems needed: STOP, back the
file up first (`cp x.db x.db.bak.$(date -u +%Y%m%dT%H%M%SZ)`), confirm with the user, and prefer
a guarded regenerate that **fails CLOSED** — refuses to overwrite good data with empty/decimated
output (reference: the `reresolve_lock.py` row-count guard). On the shared VPS, another session's
data is not yours to reset. Cautionary tale (2026-06-20): `knife_shadow.db` (671 episodes) was
truncated to 0 bytes as a side effect and had to be rebuilt — the loss is what this rule prevents.

## Security review of every change (ALWAYS ON in this directory)

The crown jewels here are **money and the edge** — exchange API keys that move real
funds, funded-challenge accounts, Telegram tokens, the VPS, and the proprietary
strategy logic. A leak or a compromised host is real capital and irreplaceable IP, not
a backtest miss. So security is not a one-off audit — it is a gate on every addition.

**Every new bot, feature, or integration is reviewed under cyber-security scrutiny
BEFORE it goes live — no exceptions.** Run the **`infra-security-audit`** skill's
*intake review* on the addition and clear it (PASS / PASS-WITH-FIXES, never BLOCK on an
unfixed CRITICAL) before deploying. The intake checks, at minimum: any new secret loads
from a gitignored env-file only (never a repo, public or private); new exchange access
is order-only / withdrawal-disabled / IP-allowlisted / key-isolated with the funded
interlock OFF by default; no new world-reachable port or network egress; external input
is sender-allowlisted and never `eval`/`shell`'d; the addition fails CLOSED (halts, not
trades) on missing/corrupt config; its `deploy_*.sh` carries the diff-guard + sudo-gate;
and it logs no secrets. Use the same skill's *full audit* mode for periodic stack-wide
reviews and after any incident.

This rule composes with the three above — it does not replace the sanitize-before-commit
or diff-first-deploy rules, it sits on top of them as the security lens.

## HyroTrader risk standard on every bot (ALWAYS ON in this directory)

**One risk ruleset, applied at the tier where it belongs — for every bot we deploy.**
The full policy is `HYROTRADER_RISK_STANDARD.md`; the short version, non-negotiable:

- **Tier 1 — anything that places ByBit orders** (`place_order`/`bybit_open`, funded OR
  demo) MUST drive the canonical `HyroTrader/risk_guard.py :: HyroTraderRiskGuard` — not a
  bespoke sizing class. "Drive" means actually CALL it: `enforce()` every cycle (it
  flattens + halts on a daily-DD / max-loss / consec-loss breach), size via
  `size_for_trade`, gate via `allow_new_trade_risk`, and feed closes to
  `record_trade_outcome`. **Instantiating the guard but never calling `enforce()` is NOT
  compliant** — that is sizing with no circuit breaker (the exact gap that fails a seat).
  Native SL on every order; notional ≤2×; aggregate ≤4%; 1% sizing / 2% ceiling; interlock
  OFF by default; isolated order-only IP-allowlisted key; fail CLOSED.
  **Halt ≠ dark:** a halted bot MUST keep accruing data — it still reacts to entry signals
  but takes them in SHADOW (no order) via the shared `HyroTrader/halt_shadow.py` (`record()`
  in place of the order; the `--resolve` cron books R-multiples off public OHLC). Shadow
  outcomes are NEVER fed to `record_trade_outcome`. A bot paired with an always-on shadow
  detector (knife + `knife_detector_shadow`) satisfies this via that detector.
- **Tier 2 — paper / virtual-book bots** apply the SAME rules on a virtual challenge book
  (1% sizing, native SL+RR, notional cap, DD/consec halts on virtual equity) so the paper
  P&L answers "would this pass a $100k challenge." Keep the raw R recorded too; a
  deliberately-permissive raw-edge test may disable the halts only behind an explicit,
  status-visible env flag.
- **Tier 3 — shadow / record-only detectors stay DIMENSIONLESS by design.** They record
  R-multiples with a defined SL+RR; do NOT bolt dollar-sizing or DD-halts onto them — it
  corrupts the sizing-agnostic edge measurement. The only requirement is that every
  recorded signal has a defined native SL + RR. A funded-viability *replay* overlay is
  optional and must never alter the raw R.

- **Sizing floor — the qualifying rule (2026-08-07, fleet-normalized).** The HyroTrader
  ruleset counts a trading day ONLY if the trade's PnL can reach **±1% of the INITIAL
  balance** — so any bot that places ByBit orders sizes at **1% minimum** (target 1%,
  ceiling 2%, min-notional 5% of initial; all inside Hyro's 3% hard cap). Sub-1% dials
  (0.25–0.5% "conservative" settings, Kelly blends, or dimmer/confidence/vol multipliers
  stacking below the floor) are structurally NON-compliant — the 2026-08-07 audit found
  five seats broken this way. Dimmers modulate trade SELECTION or size within [1%, 2%],
  never below the floor: **fewer trades, not smaller ones.** The LR core enforces this
  mechanically (`clamp_to_risk_band` in `Liquidity_Raid/core/position_manager.py`);
  guard-driven bots via `HYRO_RISK_PCT>=0.01` (env files: NO inline comments — systemd
  keeps them and the value stops parsing).
- **Telegram notification standard (ALWAYS for new bots).** Every new bot's alerts use
  the knife/trial3maker layout via the seat's designated bot (iQuant for research seats):
  sectioned ENTRY message (Direction / Entry Price / Position Size, `🛡️ Risk Management`
  with SL/TP/Risk Amount, `🎯 Setup Analysis`, `💼 Account Status`, `🚀 <strategy label>`,
  `🕒 <ts> UTC`) and `🏁 TRADE CLOSED ✅💰/❌💸` close with **ByBit-sourced money figures —
  never recomputed** (entry/exit = avg fill prices, PnL = closedPnl; never publish
  zeros/placeholders — retry until authoritative numbers exist). One alert per
  (trade, action). Reference: `knife_bybit_funded.py`; compact template:
  `ferryman_bot.py::fmt_entry/fmt_close`.

This is checked at `infra-security-audit` intake for every new bot, and composes with the
diff-first-deploy + sanitize + security rules above.

## Reuse the knife-program research in every new backtest (ALWAYS ON in this directory)

The knife program (2026-06 → 07) left behind two assets that every NEW strategy backtest
MUST use: a library of **closed questions** (so we never re-run a refuted design blind)
and a live **instrument panel** of continuously-recorded metrics (so new studies start
from real recorded features, not ad-hoc reinvention).

**1. Check the closures FIRST.** Before scaffolding any backtest, read `MEMORY.md` for
the relevant refutation memories. Standing results that pre-answer whole design spaces:
fade/reversal at visible levels is CLOSED at every scale and design (knife, LRR, NY4H,
structure scalper, PD-level, fib BOS); break-continuation (joiner) is closed BOTH
directions; inverting a net-negative book is toll-dead (anti-knife: the ~2.9bps
first-passage asymmetry < one spread crossing); confirmation-then-enter costs ~0.22–0.25R
paired vs a resting limit; the regime5 quiet→vol_expansion ordering is universal across
families. A new backtest that touches one of these spaces must state up front which
closure it is challenging and what NEW mechanism justifies re-opening it.

**2. Run new candidates through the recorded instrument panel, not fresh ad-hoc metrics.**
The stack continuously records, per signal/episode: tick microstructure k-window features
(`knife_shadow.db episodes.features_json`: into/opp volume+rate, imbalance, trade sizes,
large_share, pen_depth_atr, pen_speed, stall_secs, retrace_frac, absorb_ratio), plus
regime5, mtf_score, daily_bias, h4_structure, er20, funding_z, atr_pct, hour_et/dow,
lvl_touches_24h, dvol, vpin; cross-venue dislocation tags (`crossvenue_shadow.db`); the
anti-knife mirror (`antiknife_shadow.db`); markout at fill (`markout_monitor`); and the
raw tick/depth stores mapped in `VPS_DATA_BACKTEST_MAP.md`. A new strategy's evaluation
battery reuses this panel by default: winner-vs-loser AUC scan across the recorded
features, regime5 split, gross-vs-toll decomposition, and markout of its fills.

**3. Apply the methodological standards the program hardened.** Non-negotiable in any
backtest report: (a) label every feature PRE-fill vs POST-fill — the k120 features, the
frozen `score`, and `favored` are POST-fill [fill, fill+120s] and must never be presented
as entry gates; (b) decompose gross vs fee/slip toll before any verdict — an entry must
earn GROSS above the toll (~0.15–0.35R at our tiers) or execution can't save it; (c) any
multi-cell scan carries a family-wise bar (permutation max-|t| or Bonferroni) and any
survivor needs a half-split stability check; (d) audit intrabar fills at levels derived
from running extremes for same-bar look-ahead, and never sort outcome deciles by
(value, outcome) tuples; (e) pre-register the expectation + kill/keep bar BEFORE running
a forward shadow, and read clustered books with an effective-n haircut (simultaneous
same-direction episodes are one bet wearing several name tags); (f) bracket asymmetry on
a driftless path (first-passage) is NOT alpha — check whether "edge" survives at the
mirrored bracket before believing it.
