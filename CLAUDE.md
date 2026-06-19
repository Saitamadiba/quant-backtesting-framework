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
