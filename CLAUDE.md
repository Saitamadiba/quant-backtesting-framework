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
3. **You cannot run sudo on the VPS** (the auto-mode classifier blocks it, by
   design). Every `deploy_*.sh` MUST: (a) do all uploads sudo-free; (b) gate any
   sudo step (systemctl restart/install) behind an interactive `[y/N]` prompt; and
   (c) skip sudo entirely in a non-interactive shell, printing the command for the
   operator instead. The restart is ALWAYS the operator's action — never assume it
   ran; verify after they confirm.
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
