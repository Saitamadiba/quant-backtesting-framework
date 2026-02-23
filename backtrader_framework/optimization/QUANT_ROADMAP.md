# Quant Dev Roadmap — ML Training & Strategy Optimization

## Context
With 5 years of OHLCV data (BTC, ETH, NQ — 15m/1h/4h) and a working WFO engine, these are the next steps a quant dev would take to improve the trading bots, ranked by practical impact.

---

## Tier 1 — High Impact, Direct Edge Improvement

### 1. Regime-Adaptive Parameter Switching
The WFO already detects regimes but uses one param set across all conditions. A smarter approach: train separate optimal params per regime (trending, ranging, volatile), then at runtime detect the current regime and apply the matching params. This is the single biggest lift most quant shops get from historical data.

### 2. ML Feature Engineering Pipeline
The existing ML trade filter uses basic features (hour, day, regime, RSI). With 5 years of OHLCV, you can engineer much richer features:
- Volatility regime features (realized vol, vol-of-vol, ATR percentile rank)
- Cross-timeframe alignment (is 4h trend aligned with 1h signal?)
- Volume profile (relative volume vs 20-day, session volume skew)
- Momentum quality (ADX slope, RSI divergence, consecutive candle direction)
- Calendar effects (day-of-week, month-of-year, session killzones)
- Cross-asset correlation (BTC-ETH rolling correlation, NQ-BTC divergence)

### 3. Monte Carlo Confidence Intervals on WFO
Take the OOS trade sequence from WFO and run 10,000 bootstrap resamples. This gives you probability distributions for: future drawdown, Sharpe ratio, time-to-recovery. Instead of "PF = 1.2", you get "PF is between 0.8-1.6 with 95% confidence" — much more honest.

---

## Tier 2 — Portfolio & Risk Level

### 4. Portfolio-Level Optimization
Right now each strategy/symbol is optimized independently. A portfolio approach would:
- Measure correlation between strategy equity curves
- Allocate capital using risk parity or mean-variance
- Size positions using Kelly criterion (with fractional Kelly for safety)
- Show: "running SBS+FVG on BTC+ETH+NQ together, what's the combined Sharpe?"

### 5. Drawdown Regime Analysis & Auto-Disable Rules
Analyze historical drawdown patterns: how deep, how long, what triggers recovery. Build data-driven auto-disable thresholds: "if strategy hits -8R in 20 trades, pause for 48h" — calibrated from the actual distribution rather than gut feel.

### 6. Execution Timing Optimization
Analyze by hour/session: when do entries get the best fills? What's the actual slippage distribution? Some hours have wider spreads and worse execution — the data can quantify this and feed it back into the signal filter.

---

## Tier 3 — Advanced ML

### 7. Meta-Strategy Selector (Ensemble)
Train a classifier that takes current market state (volatility, trend strength, correlation regime, recent performance) and predicts which strategy will perform best in the next N bars. Deploy only the predicted winner, or weight allocation accordingly.

### 8. SHAP/Feature Importance Analysis
Use SHAP values on the ML trade filter to understand *why* it accepts or rejects trades. This feeds back into strategy development — if SHAP shows "RSI between 40-50 is the strongest predictor of winning LONG trades", that's actionable intelligence.

### 9. Bayesian Hyperparameter Tuning (Optuna)
Instead of grid search for the ML model itself, use Bayesian optimization to tune: number of trees, max depth, feature selection, class weights, confidence threshold. Much more efficient than manual tuning.

---

## Tier 4 — Robustness & Stress Testing

### 10. Synthetic Stress Scenarios
Generate synthetic price paths that mimic historical crashes (March 2020, May 2021, Nov 2022) but with different magnitudes. Test: does the strategy survive a 2x worse crash than anything in history?

### 11. Cross-Asset Robustness
Test SBS on ETH and NQ (not just BTC). If it works on all three, the edge is more likely structural. If it only works on BTC, it might be asset-specific and fragile.

---

## Implementation Order
| # | Item | Status |
|---|------|--------|
| 1 | Regime-Adaptive Parameter Switching | **Done** |
| 2 | ML Feature Engineering Pipeline | **Done** |
| 3 | Monte Carlo on WFO Results | **Done** |
| 4 | Portfolio-Level Optimization | **Done** |
| 5 | Drawdown Analysis & Auto-Disable | **Done** |
| 6 | Execution Timing Optimization | **Done** |
| 7 | Meta-Strategy Selector | **Done** |
| 8 | SHAP/Feature Importance | **Done** |
| 9 | Bayesian Hyperparameter Tuning | **Done** |
| 10 | Synthetic Stress Scenarios | **Done** |
| 11 | Cross-Asset Robustness | **Done** |
