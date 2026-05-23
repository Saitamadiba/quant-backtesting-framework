"""Live-faithful filter stack for the Liquidity Raid WFO adapter.

Lifted from ``the internal live-faithful study`` (internal audit)
into the framework so that the *main* `run_lr_*_wfo.py` runners can opt
into the same filter stack the live bot uses — without needing to copy
the helper code into every notebook or script.

What this module provides

* :func:`compute_daily_mtf` / :func:`compute_4h_structure` /
  :func:`mtf_score` — the live MTF score (daily EMA50/200 bias + 4H
  swing structure), expressed as a single 0-100 score per signal.
* :class:`MLPredictor` — wraps the live SweepQualityPredictor pickle and
  scores the same feature set the live bot computes per trade.
* :func:`load_dvol_series` / :func:`dvol_band` — load the Deribit DVOL
  parquet for an asset and bucketise into ``LOW`` (<45) / ``MED`` /
  ``HIGH`` (≥65) / ``n/a``, matching the live ``IV_REGIME_FILTER``.
* :func:`compute_regimes` — the rule-based new5 regime classifier
  (volatile / trending_up / trending_down / ranging).
* :func:`tag_signals` — convenience helper that takes the raw adapter
  output and returns a list of *enriched* signal dicts containing
  ``mtf_score``, ``ml_p``, ``dvol``, ``band``, ``regime``,
  ``counter_trend``.

Why this matters

The bare adapter generates a *signal universe* that is negative-EV when
read directly (negative-EV on raw signal over the historical sample).
The live bot's edge comes from gating that universe with MTF + ML +
IV-band + regime. The internal audit
showed the conclusion depends on the live filter stack. Running a WFO without them is **directionally wrong**, not
just imprecise.

This module exists so the main pipeline can call

>>> enriched = LiquidityRaidAdapter().tag_signals_with_live_filters(
...     signals, df, symbol="BTC")

and get the same trade list the live bot would have seen.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Iterable, List, Optional

import numpy as np
import pandas as pd


# ── Live-equivalent thresholds (mirrors LR core/config_base) ─────────
LOW_HI = 45.0       # DVOL < this  → LOW band
HIGH_LO = 65.0      # DVOL ≥ this  → HIGH band   (between = MED, the live block)
MIN_MTF_SCORE = 50  # live MTF floor used to gate signals
ML_THRESHOLD_LIVE = 0.73  # current live (shadow_mode) ML threshold


# ── MTF: daily bias + 4H structure ───────────────────────────────────

def compute_daily_mtf(df_15m: pd.DataFrame) -> pd.DataFrame:
    """Daily EMA50/EMA200 bias and a 0-1 strength (mirrors live MTFAnalysis).

    Returned frame is indexed by daily close, with columns
    ``bias`` ∈ {LONG, SHORT, NEUTRAL} and ``strength`` ∈ [0, 1].
    """
    d = df_15m["Close"].resample("1D").last().to_frame("close").dropna()
    d["ema_50"] = d["close"].ewm(span=50, adjust=False).mean()
    d["ema_200"] = d["close"].ewm(span=200, adjust=False).mean()
    d["ema_diff_pct"] = (d["ema_50"] - d["ema_200"]) / d["ema_200"] * 100
    d["strength"] = (d["ema_diff_pct"].abs() / 5).clip(0, 1)
    bull = (d["ema_50"] > d["ema_200"]) & (d["close"] > d["ema_50"])
    bear = (d["ema_50"] < d["ema_200"]) & (d["close"] < d["ema_50"])
    d["bias"] = np.where(bull, "LONG", np.where(bear, "SHORT", "NEUTRAL"))
    d.loc[d["bias"] == "NEUTRAL", "strength"] = 0.0
    return d[["bias", "strength"]]


def compute_4h_structure(df_15m: pd.DataFrame, swing_n: int = 3) -> pd.DataFrame:
    """4H higher-timeframe swing structure (HH/HL → BULLISH, LH/LL → BEARISH)."""
    h4 = df_15m[["High", "Low", "Close"]].resample("4h").agg(
        {"High": "max", "Low": "min", "Close": "last"}).dropna()
    highs = h4["High"].rolling(swing_n)
    lows = h4["Low"].rolling(swing_n)

    def _trend_dir(arr):
        if np.isnan(arr).any():
            return np.nan
        if (np.diff(arr) > 0).all(): return 1.0
        if (np.diff(arr) < 0).all(): return -1.0
        return 0.0

    h_dir = highs.apply(_trend_dir, raw=True)
    l_dir = lows.apply(_trend_dir, raw=True)
    structure = np.where((h_dir == 1) & (l_dir == 1), "BULLISH",
                np.where((h_dir == -1) & (l_dir == -1), "BEARISH",
                np.where(l_dir == 1, "BULLISH",
                np.where(h_dir == -1, "BEARISH", "MIXED"))))
    strength = np.where((h_dir == 1) & (l_dir == 1), 1.0,
                np.where((h_dir == -1) & (l_dir == -1), 1.0,
                np.where((l_dir == 1) | (h_dir == -1), 0.5, 0.0)))
    return pd.DataFrame({"h4_structure": structure, "h4_strength": strength},
                        index=h4.index)


def mtf_score(direction: str, daily_bias: str, daily_strength: float,
              h4_structure: str, h4_strength: float) -> float:
    """Composite MTF score (0-100): 20 base + 50·daily-strength (if aligned) + 30·h4."""
    base = 20.0
    daily_aligned = daily_bias == direction
    h4_aligned = ((h4_structure == "BULLISH" and direction == "LONG") or
                  (h4_structure == "BEARISH" and direction == "SHORT"))
    return float(base
                 + (50.0 * float(daily_strength) if daily_aligned else 0.0)
                 + (30.0 * float(h4_strength) if h4_aligned else 0.0))


# ── ML: SweepQualityPredictor wrapper ────────────────────────────────

class MLPredictor:
    """Loads the live SweepQualityPredictor pickle and scores a signal's P(win).

    Paths default to ``the internal ML imports path`` for backwards
    compatibility with the existing faithful scripts.
    """

    def __init__(self, model_path: Optional[Path] = None,
                 meta_path: Optional[Path] = None):
        if model_path is None or meta_path is None:
            # Default location used by the lr_*_faithful*.py scripts.
            _repo = Path(__file__).resolve().parents[3]
            ml_dir = _repo / "feature_lab" / "_ml_imports"
            model_path = model_path or ml_dir / "sweep_quality_model.pkl"
            meta_path = meta_path or ml_dir / "sweep_quality_metadata.json"
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(meta_path) as f:
            self.metadata = json.load(f)
        self.features = self.metadata["features"]

    def score(self, signal_dir: str, sig_confidence: float, meta: dict,
              symbol: str = "BTC") -> float:
        """Return P(win) for one signal, given the live feature set."""
        d = {
            "depth_atr": meta.get("depth_atr"),
            "body_ratio": meta.get("body_ratio"),
            "volume_ratio": meta.get("volume_ratio"),
            "candle_range_atr": meta.get("candle_range_atr"),
            "close_position_in_range": meta.get("close_position_in_range"),
            "is_bullish_bar": int(meta.get("is_bullish_bar", False)),
            "hour_et": meta.get("hour_et"),
            "structure_bias_val": meta.get("structure_bias_val"),
            "structure_conf_val": meta.get("structure_conf_val"),
            "htf_bullish": int(meta.get("htf_bullish", False)),
            "htf_bearish": int(meta.get("htf_bearish", False)),
            "adx_val": meta.get("adx_val"),
            "rsi_val": meta.get("rsi_val"),
            "atr_pctile20": meta.get("atr_pctile20"),
            "sl_vol_mult": meta.get("sl_vol_mult"),
            "rr_scale": meta.get("rr_scale"),
            "min_depth_threshold": meta.get("min_depth_threshold"),
            "confidence": sig_confidence,
            "direction_num": 1 if signal_dir == "LONG" else 0,
            "sweep_type_num": {"asia_low": 0, "asia_high": 1,
                               "london_low": 2}.get(meta.get("sweep_type"), -1),
            "session_num": 1 if meta.get("session") == "ny" else 0,
            "symbol_num": {"BTC": 0, "ETH": 1, "NQ": 2, "SOL": 3}.get(symbol, -1),
        }
        # Features that exist in the v2 schema but aren't computed here are
        # filled NaN; the model imputes (matches the faithful behavior).
        for f in ["atr_percentile_20", "atr_percentile_100", "realized_vol_20",
                  "vol_of_vol", "atr_ratio", "adx_slope_5", "rsi_divergence",
                  "candle_streak", "close_vs_range", "momentum_5",
                  "relative_volume", "volume_trend_5", "volume_price_confirm",
                  "dist_from_high_20", "dist_from_low_20", "ema_alignment",
                  "price_vs_ema200", "range_position",
                  "btc_eth_corr_20", "btc_eth_divergence"]:
            d[f] = np.nan
        depth = d["depth_atr"] or 0
        thresh = d["min_depth_threshold"] or 0
        d["depth_excess"] = depth - thresh
        h = d["hour_et"] or 0
        d["hour_sin"] = np.sin(2 * np.pi * h / 24)
        d["hour_cos"] = np.cos(2 * np.pi * h / 24)
        d["volume_zscore"] = 0.0
        br = d["body_ratio"] or 0
        d["body_depth_interaction"] = br * depth
        rsi = d["rsi_val"]
        d["rsi_extremity"] = abs((rsi or 50) - 50)
        X = np.array([[float(d.get(f, np.nan)) if d.get(f) is not None
                       else np.nan for f in self.features]])
        return float(self.model.predict_proba(X)[0, 1])


# ── DVOL band + per-asset OHLC attach ────────────────────────────────

def load_dvol_series(symbol: str, dvol_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load the per-asset Deribit DVOL parquet (``dvol_vrp_<SYMBOL>.parquet``)."""
    if dvol_dir is None:
        dvol_dir = Path(__file__).resolve().parents[3] / "flow_aux_data" / "dvol"
    p = Path(dvol_dir) / f"dvol_vrp_{symbol}.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["date", "dvol"])
    dv = pd.read_parquet(p).sort_values("date").reset_index(drop=True)
    dv["date"] = pd.to_datetime(dv["date"], utc=True)
    return dv


def attach_dvol(df_15m: pd.DataFrame, symbol: str,
                dvol_dir: Optional[Path] = None) -> pd.DataFrame:
    """Add a forward-filled ``DVOL`` column to a 15m OHLC frame for ``symbol``."""
    out = df_15m.copy()
    dv = load_dvol_series(symbol, dvol_dir)
    if dv.empty:
        out["DVOL"] = np.nan
        return out
    bar_s = (pd.DatetimeIndex(out.index).tz_convert("UTC").tz_localize(None)
             .astype("datetime64[s]").astype("int64").to_numpy())
    dv_s = (dv["date"].dt.tz_convert("UTC").dt.tz_localize(None)
            .astype("datetime64[s]").astype("int64").to_numpy())
    idx = np.searchsorted(dv_s, bar_s, side="right") - 1
    out["DVOL"] = np.where(idx >= 0,
                           dv["dvol"].to_numpy()[np.clip(idx, 0, None)],
                           np.nan)
    return out


def dvol_band(d: float) -> str:
    """Bucketise a DVOL value: ``LOW`` <45, ``MED`` 45-64, ``HIGH`` ≥65, ``n/a`` NaN."""
    if pd.isna(d):
        return "n/a"
    if d < LOW_HI:
        return "LOW"
    if d < HIGH_LO:
        return "MED"
    return "HIGH"


# ── new5 regime classifier (mirrors live regime_classifier.classify_rule_based) ─

def compute_regimes(df: pd.DataFrame) -> np.ndarray:
    """Return a per-bar regime label: volatile / trending_up / trending_down / ranging.

    Mirrors the rule-based path in the live ``regime_classifier`` module
    (Choppiness/ATR-percentile flavour). Inputs needed on ``df``: ``ATR``,
    ``Close``, ``EMA50``, and optionally ``ADX`` (defaults to no-trend if
    missing).
    """
    lookback = 50
    atr = df["ATR"].values
    close = df["Close"].values
    ema50 = df["EMA50"].values
    adx = df["ADX"].values if "ADX" in df.columns else None
    atr_pct = atr / np.where(close > 0, close, np.nan)
    avg_atr_pct = pd.Series(atr_pct).rolling(lookback).mean().values
    out = np.full(len(df), "unknown", dtype=object)
    valid = ~np.isnan(avg_atr_pct) & (close > 0) & (atr > 0)
    volatile = valid & (atr_pct > avg_atr_pct * 1.8)
    if adx is not None:
        trending = valid & (~volatile) & (adx > 30.0)
        trend_up = trending & (close > ema50)
        trend_dn = trending & (close <= ema50)
    else:
        trend_up = np.zeros(len(df), dtype=bool)
        trend_dn = np.zeros(len(df), dtype=bool)
    ranging = valid & (~volatile) & (~trend_up) & (~trend_dn)
    out[volatile] = "volatile"
    out[trend_up] = "trending_up"
    out[trend_dn] = "trending_down"
    out[ranging] = "ranging"
    return out


# ── Top-level: tag a list of adapter signals with the live filter stack ─

def tag_signals(signals: Iterable[Any], df: pd.DataFrame, symbol: str,
                ml: Optional[MLPredictor] = None,
                dvol_dir: Optional[Path] = None,
                regime_arr: Optional[np.ndarray] = None,
                ) -> List[dict]:
    """Enrich each adapter signal with ``mtf_score``, ``ml_p``, ``dvol``, ``band``,
    ``regime``, ``counter_trend`` — same columns the faithful parquets carry.

    ``df`` must be the 15m frame the adapter ran on. ``symbol`` selects
    the per-asset DVOL series + ML symbol embedding. ``regime_arr`` can
    be precomputed (saves work in loops); ``ml`` likewise.

    The signal objects are expected to expose ``.idx``, ``.time``,
    ``.direction``, ``.confidence`` and ``.metadata`` (the adapter's
    ``Signal`` dataclass shape).
    """
    if ml is None:
        ml = MLPredictor()
    if regime_arr is None:
        regime_arr = compute_regimes(df)
    if "DVOL" not in df.columns:
        df = attach_dvol(df, symbol, dvol_dir)
    daily_mtf = compute_daily_mtf(df)
    h4_struct = compute_4h_structure(df)
    out: List[dict] = []
    for sig in signals:
        md = getattr(sig, "metadata", None) or {}
        ets = pd.Timestamp(getattr(sig, "time"))
        if ets.tzinfo is None:
            ets = ets.tz_localize("UTC")
        try:
            dvol_at = float(df.loc[df.index <= ets, "DVOL"].iloc[-1])
        except (IndexError, KeyError):
            dvol_at = float("nan")
        idx = getattr(sig, "idx", -1)
        regime_at = (regime_arr[idx]
                     if 0 <= idx < len(regime_arr) else "n/a")
        direction = ("LONG" if getattr(sig, "direction", None)
                     in (1, "LONG", "BUY") else "SHORT")
        bar_date = ets.normalize()
        try:
            r0 = daily_mtf.loc[daily_mtf.index <= bar_date].iloc[-1]
            d_bias, d_strength = r0["bias"], float(r0["strength"])
        except (IndexError, KeyError):
            d_bias, d_strength = "NEUTRAL", 0.0
        try:
            r1 = h4_struct.loc[h4_struct.index <= ets].iloc[-1]
            h4_v, h4_s = str(r1["h4_structure"]), float(r1["h4_strength"])
        except (IndexError, KeyError):
            h4_v, h4_s = "MIXED", 0.0
        score = mtf_score(direction, d_bias, d_strength, h4_v, h4_s)
        conf = getattr(sig, "confidence", 0.5)
        ml_p = ml.score(direction,
                        float(conf) if conf is not None else 0.5,
                        md, symbol=symbol)
        counter_trend = ((regime_at == "trending_up" and direction == "SHORT")
                         or (regime_at == "trending_down" and direction == "LONG"))
        out.append({
            "signal": sig,
            "idx": idx,
            "time": ets,
            "direction": direction,
            "regime": regime_at,
            "dvol": dvol_at,
            "band": dvol_band(dvol_at),
            "mtf_score": score,
            "ml_p": ml_p,
            "counter_trend": counter_trend,
        })
    return out


# ── Wrapper adapter: drop-in replacement that returns only live-passing signals ─

class FaithfulLiquidityRaidAdapter:
    """Drop-in replacement for :class:`LiquidityRaidAdapter` that returns only the
    signals the live LR bot would actually take.

    Composition (delegation) rather than subclassing — :class:`WFOEngine` only
    needs the public adapter surface (``name``, ``default_timeframes``,
    ``get_param_space``, ``generate_signals``). Inheritance would couple us
    to private mutable state of the parent adapter (research knobs like
    ``enable_sweep_quality_gate``); the WFO knows nothing about either.

    Heavy artefacts (MTF/h4 frames, regime array, DVOL series, ML predictor)
    are cached by the ``id`` of the input ``df`` so a single WFO sweep over
    many param combos pays the construction cost once.

    Live current state: ML is in ``shadow_mode`` and does NOT
    gate live trades — default ``apply_ml=False`` to match. Flip to True
    once the June 22 ML decision lands.
    """

    def __init__(self, symbol: str = "BTC",
                 base: Optional[Any] = None,
                 *, apply_ml: bool = False,
                 min_mtf: float = MIN_MTF_SCORE,
                 ml_threshold: float = ML_THRESHOLD_LIVE,
                 block_bands: Iterable[str] = ("MED",),
                 block_counter_trend: bool = True):
        if base is None:
            # Local import avoids a circular ref at module-load.
            from .liquidity_raid_adapter import LiquidityRaidAdapter as _Base
            base = _Base()
        self.base = base
        self.symbol = symbol
        self.apply_ml = apply_ml
        self.min_mtf = float(min_mtf)
        self.ml_threshold = float(ml_threshold)
        self.block_bands = tuple(block_bands)
        self.block_counter_trend = bool(block_counter_trend)
        # Caches keyed by id(df).
        self._cache: dict = {}
        self._ml: Optional[MLPredictor] = None

    # ── Adapter surface delegation ──────────────────────────────────
    # `name` and `default_timeframes` are *properties* on the base adapter
    # (see backtrader_framework/.../base_adapter.py); preserve that shape so
    # the WFO engine's introspection (which reads, not calls) keeps working.
    @property
    def name(self) -> str:
        return f"{self.base.name}_faithful"

    @property
    def default_timeframes(self):
        return self.base.default_timeframes

    def get_param_space(self):
        return self.base.get_param_space()

    # ── Cache + filter ───────────────────────────────────────────────
    def _ctx_for(self, df: pd.DataFrame):
        key = id(df)
        ctx = self._cache.get(key)
        if ctx is not None:
            return ctx
        # Attach DVOL non-destructively (don't mutate caller's df).
        # NB: IndicatorEngine.calculate emits a DVOL column but fills it
        # with NaN (placeholder for a later pipeline step). Probe for actual
        # data, not just column presence, otherwise the IV-MED filter
        # becomes a no-op — every signal ends up band='n/a'.
        if "DVOL" in df.columns and df["DVOL"].notna().any():
            df_dv = df
        else:
            df_dv = attach_dvol(df.drop(columns=["DVOL"], errors="ignore"),
                                self.symbol)
        ctx = {
            "df": df_dv,
            "regime_arr": compute_regimes(df_dv),
            "daily_mtf": compute_daily_mtf(df_dv),
            "h4_struct": compute_4h_structure(df_dv),
        }
        if self.apply_ml and self._ml is None:
            self._ml = MLPredictor()
        self._cache[key] = ctx
        return ctx

    def generate_signals(self, df: pd.DataFrame, params: dict,
                         scan_start_idx: int, scan_end_idx: int):
        signals = self.base.generate_signals(df, params, scan_start_idx, scan_end_idx)
        if not signals:
            return signals
        ctx = self._ctx_for(df)
        df_dv = ctx["df"]
        # Re-implement tag_signals inline so we can reuse cached MTF frames.
        daily_mtf = ctx["daily_mtf"]
        h4_struct = ctx["h4_struct"]
        regime_arr = ctx["regime_arr"]
        survivors = []
        for sig in signals:
            md = getattr(sig, "metadata", None) or {}
            idx = getattr(sig, "idx", -1)
            regime_at = (regime_arr[idx] if 0 <= idx < len(regime_arr) else "n/a")
            direction = ("LONG" if getattr(sig, "direction", None)
                         in (1, "LONG", "BUY") else "SHORT")
            # Counter-trend filter (cheapest — drop first).
            if self.block_counter_trend and (
                (regime_at == "trending_up" and direction == "SHORT")
                or (regime_at == "trending_down" and direction == "LONG")
            ):
                continue
            # IV band filter.
            ets = pd.Timestamp(getattr(sig, "time"))
            if ets.tzinfo is None: ets = ets.tz_localize("UTC")
            try:
                dvol_at = float(df_dv.loc[df_dv.index <= ets, "DVOL"].iloc[-1])
            except (IndexError, KeyError):
                dvol_at = float("nan")
            if dvol_band(dvol_at) in self.block_bands:
                continue
            # MTF score filter.
            bar_date = ets.normalize()
            try:
                r0 = daily_mtf.loc[daily_mtf.index <= bar_date].iloc[-1]
                d_bias, d_strength = r0["bias"], float(r0["strength"])
            except (IndexError, KeyError):
                d_bias, d_strength = "NEUTRAL", 0.0
            try:
                r1 = h4_struct.loc[h4_struct.index <= ets].iloc[-1]
                h4_v, h4_s = str(r1["h4_structure"]), float(r1["h4_strength"])
            except (IndexError, KeyError):
                h4_v, h4_s = "MIXED", 0.0
            score = mtf_score(direction, d_bias, d_strength, h4_v, h4_s)
            if score < self.min_mtf:
                continue
            # ML filter (optional — live currently shadow_mode).
            if self.apply_ml:
                conf = getattr(sig, "confidence", 0.5)
                p = self._ml.score(direction,
                                   float(conf) if conf is not None else 0.5,
                                   md, symbol=self.symbol)
                if p < self.ml_threshold:
                    continue
            # Stash the filter context onto the signal for downstream telemetry.
            sig.metadata = dict(md, mtf_score=score, dvol=dvol_at,
                                band=dvol_band(dvol_at), regime=regime_at)
            survivors.append(sig)
        return survivors


def filter_live_stack(enriched: List[dict],
                      *,
                      min_mtf: float = MIN_MTF_SCORE,
                      min_ml: Optional[float] = None,
                      block_bands: Iterable[str] = ("MED",),
                      block_counter_trend: bool = True,
                      ) -> List[dict]:
    """Apply the live LR gate stack to enriched signals.

    Defaults reproduce the live ETH stack (MTF ≥ 50, IV-MED blocked,
    counter-trend blocked). Pass ``min_ml=0.73`` to also apply the live
    (shadow-mode) ML threshold once it's promoted.
    """
    block_bands = set(block_bands)
    out = []
    for e in enriched:
        if block_counter_trend and e.get("counter_trend"):
            continue
        if e.get("band") in block_bands:
            continue
        if e.get("mtf_score", 0) < min_mtf:
            continue
        if min_ml is not None and (e.get("ml_p") or 0) < min_ml:
            continue
        out.append(e)
    return out
