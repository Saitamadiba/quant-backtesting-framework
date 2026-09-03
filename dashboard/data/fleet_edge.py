"""WS1 — Bayesian edge estimation over the live fleet.

Replaces "this book has ΣR = +5.0" with a posterior: a mean-R interval, the
probability the book's *mean* clears the fee/slip toll, and an effective-n that
refuses to count ten simultaneous same-direction fills as ten independent bets.

*Picture: a witness statement with an error bar. Ten trades is a hunch; two
hundred is testimony — and ten witnesses who all heard it from one person are
still one witness.*

Design, and the three things it deliberately does NOT do:

* **The bar is the toll, not zero.** A book whose R is recorded NET of fees has
  already paid the crossing charge, so its bar is 0.0R. A book that records
  GROSS R (the registry labels them ``(gross…)``) still owes it, so its bar is
  ``+toll`` — it has to earn the toll back before it is worth anything. The
  posterior reports both P(mean R > 0) and P(mean R > bar); the second is the
  number that decides.
* **Effective n — by aggregation, not re-weighting.** Fills sharing (family,
  direction, 15-minute bucket) are one bet wearing several name tags, and so is
  a resting level re-stamped by a scanner every cycle. The posterior is fitted
  on ONE R per bet (the cluster's mean), so a replicated setup can neither
  narrow the interval nor drag the point estimate: ``halt_shadow`` era 2 reads
  +0.672R across 1,258 rows and −0.139R across the 134 levels behind them. The
  time-bucket key is byte-identical to the fleet feature spine's
  (``fleet_features.features.cluster_id``); ``test_fleet_edge`` enforces that.
* **Eras are never pooled.** Era-split books are already separate registry
  entries (``ofcs-paper(gross,era1)`` vs ``(net,era2)``); this module reads the
  registry's own ``closed_filter`` and adds nothing, so a recorder-era break
  cannot be silently averaged over.

This is a *reading* instrument. It has no kill bar and gates no trade: it
changes how a number is displayed, never what the fleet does.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── the toll, and the cluster bucket ─────────────────────────────────────────
# The fee/slip crossing charge measured across the fleet is 0.15–0.35R at our
# tiers (knife maker/taker audits, ferryman 600s replay, depth execution audit);
# 0.25R is the midpoint the plan pre-registered and the prior is centred on.
DEFAULT_TOLL_R = 0.25
BAR_MS = 15 * 60 * 1000
MIN_TRADES = 5          # below this the prior is the whole answer; still shown

_ROOT = Path(__file__).resolve().parents[2]


def cluster_id(family: str, side: Optional[str], fill_ms: int) -> str:
    """Simultaneous same-direction fills of one family are one bet: bucket to
    the 15-minute bar. Byte-identical to the feature spine's key."""
    d = (side or "?").upper()
    d = "L" if d in ("LONG", "BUY", "L") else "S" if d in ("SHORT", "SELL", "S") else "?"
    return f"{family}|{d}|{(fill_ms // BAR_MS) * BAR_MS}"


def is_gross(book) -> bool:
    """Does this book record R BEFORE the fee/slip toll? The registry says so in
    the label the scoreboard already prints — ``ofcs-paper(gross,era1)``."""
    return "gross" in str(getattr(book, "label", "")).lower()


def bar_for(book, toll: float = DEFAULT_TOLL_R) -> float:
    """The R level this book's posterior mean has to clear to be worth running."""
    return float(toll) if is_gross(book) else 0.0


# ── timestamps ───────────────────────────────────────────────────────────────

def to_utc(values) -> pd.Series:
    """Coerce a fleet book's timestamp column to tz-aware UTC.

    The books stamp time three ways: ISO strings, sqlite ``YYYY-MM-DD HH:MM:SS``,
    and raw epoch integers (``ferryman_demo.orders`` stores SECONDS). pandas
    reads a bare integer as NANOSECONDS, so an epoch-seconds column parses to
    January 1970 — the row survives and the clock is silently wrong, which
    collapses every fill into one 15-minute cluster bucket (ferryman: n_eff 2
    instead of 71). Pick the unit from magnitude instead of trusting the dtype.
    """
    s = pd.Series(values)
    if s.empty:
        return pd.to_datetime(s, utc=True, errors="coerce")
    num = pd.to_numeric(s, errors="coerce")
    numeric_frac = float(num.notna().mean())
    if numeric_frac >= 0.9:                       # an epoch column, whatever its dtype
        mag = float(np.nanmedian(np.abs(num.to_numpy(dtype=float))))
        unit = "ns" if mag >= 1e17 else "us" if mag >= 1e14 else "ms" if mag >= 1e11 else "s"
        return pd.to_datetime(num, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(s, errors="coerce", utc=True, format="mixed")


# ── reading one book's R series ──────────────────────────────────────────────

@dataclass
class BookSeries:
    """One book's closed R-multiples, with the clustering already applied."""
    key: str
    label: str
    family: str
    tier: int
    gross: bool
    r: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    clusters: List[str] = field(default_factory=list)
    first_ts: Optional[pd.Timestamp] = None
    last_ts: Optional[pd.Timestamp] = None
    error: str = ""

    @property
    def n(self) -> int:
        return int(len(self.r))

    @property
    def n_eff(self) -> int:
        return int(len(set(self.clusters))) if self.clusters else self.n

    def bet_r(self) -> np.ndarray:
        """One R per BET: the mean of the fills sharing a cluster.

        Weighting rows instead would let the most-replicated setup carry the
        book — ``halt_shadow`` era 2 reads +0.672R over 1,258 rows and −0.139R
        over the 134 distinct levels behind them, because one winning price path
        was re-stamped forty times. A bet contributes its outcome once.
        """
        if not self.clusters or len(self.clusters) != len(self.r):
            return self.r
        return (pd.Series(self.r, index=pd.Index(self.clusters, name="c"))
                .groupby(level=0, sort=False).mean().to_numpy(dtype=float))


def _book_path(book, fleet_dir: Path, legacy_dir: Path) -> Optional[Path]:
    """Where this book's DB actually sits locally — the fleet rsync tree first,
    then the legacy flat cache. Globbed books (one family, many files) are not
    read here; the scoreboard aggregates those server-side."""
    if any(c in book.db for c in "*?["):
        return None
    p = fleet_dir / book.db
    if p.exists():
        return p
    p2 = legacy_dir / os.path.basename(book.db)
    return p2 if p2.exists() else None


def load_book_series(book, db_path: Path) -> BookSeries:
    """Read one book's closed rows into an R series + cluster keys.

    Read-only (``mode=ro``) — this module never writes a fleet database.
    """
    from data.fleet_registry import build_history_sql

    out = BookSeries(key=book.key, label=book.label, family=book.family or book.label,
                     tier=int(book.tier), gross=is_gross(book))
    if not book.r:
        out.error = "book records no R"
        return out
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            cols = {row[1] for row in conn.execute(f"PRAGMA table_info({book.table})").fetchall()}
            raw = pd.read_sql_query(build_history_sql(book, available_cols=cols), conn)
    except Exception as e:                                   # noqa: BLE001
        out.error = f"unreadable: {e}"
        return out
    if raw.empty:
        out.error = "no closed rows"
        return out

    r = pd.to_numeric(raw.get("r"), errors="coerce")
    ts = to_utc(raw.get("entry_ts"))
    exit_ts = to_utc(raw.get("exit_ts"))
    ts = ts.where(ts.notna(), exit_ts)          # a book with no entry stamp clusters on its exit
    side = raw.get("side")
    keep = r.notna()
    if not keep.any():
        out.error = "no numeric R"
        return out

    r = r[keep].to_numpy(dtype=float)
    ts_k = ts[keep]
    side_k = side[keep] if side is not None else pd.Series([None] * len(r))
    ms = (ts_k.astype("int64") // 1_000_000).where(ts_k.notna(), -1).to_numpy()
    fam = out.family
    out.r = r
    keys = [cluster_id(fam, s, int(m)) if m >= 0 else f"{fam}|?|{i}"
            for i, (s, m) in enumerate(zip(side_k.tolist(), ms.tolist()))]
    out.clusters = _collapse_repeated_setups(keys, raw, keep)
    if ts_k.notna().any():
        out.first_ts = ts_k.min()
        out.last_ts = ts_k.max()
    return out



def _collapse_repeated_setups(keys: List[str], raw: pd.DataFrame, keep) -> List[str]:
    """Fold a re-recorded resting setup back into ONE bet.

    Several shadow recorders re-stamp the SAME resting order every scan, so one
    setup arrives as dozens of rows minutes apart — different 15-minute buckets,
    so the time-bucket cluster key cannot see them. The signature is the resting
    LEVEL (symbol, side, entry) and deliberately NOT the bracket: the recorders
    recompute an ATR-derived SL/TP every cycle, so the stop jitters in the eighth
    decimal while the level stays pinned, and any key including `sl` fails to
    collapse anything. Measured on ``halt_shadow`` era 2: 1,258 rows are 323
    (symbol, side, entry, sl) groups but only **134 distinct levels** — and the
    book reads +0.672R raw, +0.312R deduped on entry+sl, and −0.139R deduped on
    the level. One winning price path booked forty times is not forty wins.

    Rows with a unique signature keep their time bucket untouched, so a book that
    never re-records is unaffected. Where the rule errs it errs toward FEWER
    independent bets and a wider interval, which is the safe direction for an
    instrument whose whole job is to resist a flattering number.
    """
    if "entry" not in raw.columns:
        return keys
    entry = pd.to_numeric(raw["entry"], errors="coerce")[keep].round(8)
    sym = raw["symbol"].astype(str)[keep] if "symbol" in raw.columns else pd.Series([""] * len(keys))
    side = raw["side"].astype(str)[keep] if "side" in raw.columns else pd.Series([""] * len(keys))
    sig = pd.Series([f"{a}|{b}|{c}" for a, b, c in
                     zip(sym.tolist(), side.tolist(), entry.tolist())])
    valid = entry.notna().to_numpy()
    counts = sig[valid].value_counts() if valid.any() else pd.Series(dtype=int)
    repeated = set(counts[counts > 1].index)
    if not repeated:
        return keys
    sig_l = sig.tolist()
    return [f"setup::{sig_l[i]}" if (valid[i] and sig_l[i] in repeated) else k
            for i, k in enumerate(keys)]


# ── the posterior ────────────────────────────────────────────────────────────

def posterior(series: BookSeries, toll: float = DEFAULT_TOLL_R,
              prior: str = "toll", ci: float = 0.90) -> Dict[str, Any]:
    """Fit the Bayesian edge estimator to one series and flatten the summary.

    Returns a row-shaped dict; ``status`` is ``'ok'``, ``'thin'`` (fewer than
    two trades — no posterior is possible) or the read error.
    """
    row: Dict[str, Any] = {
        "key": series.key, "label": series.label, "family": series.family,
        "tier": series.tier, "gross": series.gross, "n": series.n,
        "n_eff": series.n_eff, "bar_r": toll if series.gross else 0.0,
        "sum_r": float(np.sum(series.r)) if series.n else 0.0,
        "raw_mean_r": float(np.mean(series.r)) if series.n else float("nan"),
        "first_ts": series.first_ts, "last_ts": series.last_ts,
        "status": series.error or "ok",
    }
    if series.error or series.n < 2:
        row["status"] = series.error or "thin"
        return row

    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from backtrader_framework.optimization.bayesian_edge import BayesianEdgeEstimator

    bets = series.bet_r()
    if len(bets) < 2:
        row["status"] = "thin"
        return row
    row["bet_mean_r"] = float(np.mean(bets))
    est = BayesianEdgeEstimator()
    est.fit(bets, prior=prior)
    bar = row["bar_r"]
    s = est.summary(threshold=bar)
    mr = s["mean_r"]
    dist = est._r_dist()                     # noqa: SLF001 — the CI width is the point
    lo, hi = (1.0 - ci) / 2.0, 1.0 - (1.0 - ci) / 2.0
    row.update({
        "post_mean_r": float(mr["posterior_mean"]),
        "ci_lo": float(dist.ppf(lo)),
        "ci_hi": float(dist.ppf(hi)),
        "p_above_0": float(mr["p_positive"]),
        "p_above_bar": float(mr.get("p_above_threshold", mr["p_positive"])),
        "post_win_rate": float(s["win_rate"]["posterior_mean"]),
        "wr_ci_lo": float(s["win_rate"]["credible_interval_95"][0]),
        "wr_ci_hi": float(s["win_rate"]["credible_interval_95"][1]),
        "expectancy": float(s["expectancy"]["posterior_mean"]),
        "r_shrinkage": float(s["shrinkage"]["r_shrinkage"]),
        "trades_for_half_width": s["sample_size_assessment"]["trades_for_half_width"],
        "variance_model": s["variance_model"],
    })
    return row


# ── the fleet sweep ──────────────────────────────────────────────────────────

def fleet_posteriors(books: Optional[Iterable] = None,
                     toll: float = DEFAULT_TOLL_R,
                     prior: str = "toll",
                     fleet_dir: Optional[Path] = None,
                     legacy_dir: Optional[Path] = None,
                     tiers: Sequence[int] = (1, 2, 3)) -> pd.DataFrame:
    """A posterior row per readable book, sorted by P(mean R > bar), worst last.

    Books whose DB is not synced locally come back with ``status`` set, never
    dropped — an absent book must be visible as absent, not as zero.
    """
    from data.fleet_registry import BOOKS
    try:
        from config import FLEET_CACHE_DIR, VPS_CACHE_DIR
        fdir = fleet_dir or Path(FLEET_CACHE_DIR)
        ldir = legacy_dir or Path(VPS_CACHE_DIR)
    except Exception:                                        # noqa: BLE001
        fdir = fleet_dir or (_ROOT / "dashboard" / "databases" / "fleet")
        ldir = legacy_dir or (_ROOT / "dashboard" / "databases")

    rows = []
    for b in (books if books is not None else BOOKS):
        if int(b.tier) not in tiers:
            continue
        p = _book_path(b, fdir, ldir)
        if p is None:
            rows.append(posterior(BookSeries(key=b.key, label=b.label,
                                             family=b.family or b.label, tier=int(b.tier),
                                             gross=is_gross(b),
                                             error="db not synced locally"), toll, prior))
            continue
        rows.append(posterior(load_book_series(b, p), toll, prior))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["status", "p_above_bar"], ascending=[True, False],
                        na_position="last").reset_index(drop=True)
    return df


def family_posteriors(books: Optional[Iterable] = None,
                      toll: float = DEFAULT_TOLL_R,
                      prior: str = "toll",
                      fleet_dir: Optional[Path] = None,
                      legacy_dir: Optional[Path] = None,
                      tiers: Sequence[int] = (1, 2, 3)) -> pd.DataFrame:
    """Same posterior, one row per FAMILY — the family's books pooled.

    Pooling is only honest inside a tier and a toll convention: a Tier-1 funded
    seat and a Tier-3 gross recorder are not the same instrument, so families
    are keyed by (family, tier, gross) and never merged across those.
    """
    from data.fleet_registry import BOOKS
    try:
        from config import FLEET_CACHE_DIR, VPS_CACHE_DIR
        fdir = fleet_dir or Path(FLEET_CACHE_DIR)
        ldir = legacy_dir or Path(VPS_CACHE_DIR)
    except Exception:                                        # noqa: BLE001
        fdir = fleet_dir or (_ROOT / "dashboard" / "databases" / "fleet")
        ldir = legacy_dir or (_ROOT / "dashboard" / "databases")

    pooled: Dict[tuple, BookSeries] = {}
    for b in (books if books is not None else BOOKS):
        if int(b.tier) not in tiers:
            continue
        p = _book_path(b, fdir, ldir)
        if p is None:
            continue
        s = load_book_series(b, p)
        if s.error or s.n == 0:
            continue
        k = (s.family, s.tier, s.gross)
        cur = pooled.get(k)
        if cur is None:
            pooled[k] = BookSeries(key=f"{s.family}|T{s.tier}", label=s.family,
                                   family=s.family, tier=s.tier, gross=s.gross,
                                   r=s.r, clusters=list(s.clusters),
                                   first_ts=s.first_ts, last_ts=s.last_ts)
        else:
            cur.r = np.concatenate([cur.r, s.r])
            cur.clusters.extend(s.clusters)
            if s.first_ts is not None:
                cur.first_ts = s.first_ts if cur.first_ts is None else min(cur.first_ts, s.first_ts)
            if s.last_ts is not None:
                cur.last_ts = s.last_ts if cur.last_ts is None else max(cur.last_ts, s.last_ts)
    rows = [posterior(s, toll, prior) for s in pooled.values()]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["status", "p_above_bar"], ascending=[True, False],
                          na_position="last").reset_index(drop=True)


def verdict(row: Dict[str, Any] | pd.Series, strong: float = 0.90,
            dead: float = 0.10) -> str:
    """A one-word reading of a posterior row — never a trading instruction.

    ``evidence for`` / ``evidence against`` are statements about the POSTERIOR,
    not a deploy or kill decision; a nominee still owes a pre-registered
    forward shadow before anything changes.
    """
    if row.get("status") != "ok":
        return "no read"
    n_eff = float(row.get("n_eff") or 0)
    p = float(row.get("p_above_bar") or 0.0)
    if n_eff < MIN_TRADES:
        return "prior dominates"
    if p >= strong:
        return "evidence for"
    if p <= dead:
        return "evidence against"
    return "undecided"
