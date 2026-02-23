"""
SHAP Feature Importance Analysis.

Computes SHAP (SHapley Additive exPlanations) values for trained ML models
to explain *why* predictions are made. Produces global importance rankings,
per-feature dependence data, and actionable natural-language insights.

Uses TreeExplainer for RF/GradientBoosting (exact, fast).
All SHAP imports are lazy — module loads without the shap package installed.

Usage:
    from shap_analysis import SHAPAnalyzer
    result = SHAPAnalyzer.analyze(model, X_test, feature_names, class_names)
    print(result['insights'])
"""

import numpy as np
from typing import Dict, List, Optional


# Feature categories for color-coding in dashboard
FEATURE_CATEGORIES = {
    'atr_percentile_20': 'Volatility', 'atr_percentile_100': 'Volatility',
    'realized_vol_20': 'Volatility', 'vol_of_vol': 'Volatility', 'atr_ratio': 'Volatility',
    'adx_slope_5': 'Momentum', 'rsi_divergence': 'Momentum', 'candle_streak': 'Momentum',
    'close_vs_range': 'Momentum', 'momentum_5': 'Momentum',
    'relative_volume': 'Volume', 'volume_trend_5': 'Volume', 'volume_price_confirm': 'Volume',
    'dist_from_high_20': 'Price Structure', 'dist_from_low_20': 'Price Structure',
    'ema_alignment': 'Price Structure', 'price_vs_ema200': 'Price Structure',
    'range_position': 'Price Structure',
    'btc_eth_corr_20': 'Cross-Asset', 'btc_eth_divergence': 'Cross-Asset',
    'regime_ranging': 'Regime', 'regime_trending_up': 'Regime',
    'regime_trending_down': 'Regime', 'regime_volatile': 'Regime',
}

CATEGORY_COLORS = {
    'Volatility': '#2196F3',
    'Momentum': '#4CAF50',
    'Volume': '#FF9800',
    'Price Structure': '#9C27B0',
    'Cross-Asset': '#00BCD4',
    'Regime': '#F44336',
    'Other': '#9E9E9E',
}


class SHAPAnalyzer:
    """Compute SHAP values and generate actionable feature importance insights."""

    MAX_SAMPLES = 500  # Cap samples for SHAP computation speed
    TOP_DEPENDENCE_FEATURES = 8  # Store dependence data for top N features

    @staticmethod
    def analyze(
        model,
        X: np.ndarray,
        feature_names: List[str],
        class_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Compute SHAP values for a trained model.

        Args:
            model: Trained sklearn model (RF, GradientBoosting, etc.).
            X: Feature matrix (n_samples × n_features), numpy array.
            feature_names: List of feature name strings.
            class_names: For multiclass, list of class label strings.

        Returns:
            Dict with global_importance, feature_summary, dependence,
            insights, and raw SHAP value samples.
        """
        import shap

        n_samples, n_features = X.shape

        # Subsample for speed
        if n_samples > SHAPAnalyzer.MAX_SAMPLES:
            rng = np.random.RandomState(42)
            idx = rng.choice(n_samples, SHAPAnalyzer.MAX_SAMPLES, replace=False)
            idx.sort()
            X_shap = X[idx]
        else:
            X_shap = X

        # Compute SHAP values
        is_tree = hasattr(model, 'estimators_') or hasattr(model, 'tree_')
        if is_tree:
            explainer = shap.TreeExplainer(model)
            explainer_type = 'TreeExplainer'
        else:
            background = shap.kmeans(X_shap, min(50, len(X_shap)))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            explainer_type = 'KernelExplainer'

        shap_values_raw = explainer.shap_values(X_shap)

        # Normalize SHAP values shape.
        # SHAP >=0.50 returns 3D ndarray (n_samples, n_features, n_classes) for multiclass.
        # Older versions return list of K arrays, each (n_samples, n_features).
        # Binary/regression returns 2D (n_samples, n_features).
        sv = np.asarray(shap_values_raw)
        is_multiclass = sv.ndim == 3

        if isinstance(shap_values_raw, list):
            # Old SHAP format: list of arrays → stack to 3D
            shap_3d = np.stack(shap_values_raw, axis=-1)
            is_multiclass = True
        elif sv.ndim == 3:
            shap_3d = sv  # Already (n_samples, n_features, n_classes)
        else:
            shap_3d = None

        if is_multiclass and shap_3d is not None:
            n_classes = shap_3d.shape[2]
            # Aggregate: mean absolute across classes
            shap_agg = np.mean(np.abs(shap_3d), axis=-1)  # (n_samples, n_features)
            # For dependence, use the class with highest variance
            shap_var_per_class = [
                np.var(shap_3d[:, :, ci], axis=0).sum() for ci in range(n_classes)
            ]
            primary_class_idx = int(np.argmax(shap_var_per_class))
            shap_primary = shap_3d[:, :, primary_class_idx]  # (n_samples, n_features)
            # Build list-of-arrays for per-class analysis
            shap_values_list = [shap_3d[:, :, ci] for ci in range(n_classes)]
        else:
            shap_agg = np.abs(sv)
            shap_primary = sv
            shap_values_list = None

        # ── Global importance ────────────────────────────────────────
        mean_abs_shap = np.mean(shap_agg, axis=0)  # (n_features,)
        ranked_indices = np.argsort(mean_abs_shap)[::-1]

        global_importance = []
        for rank, idx in enumerate(ranked_indices):
            idx = int(idx)
            global_importance.append({
                'feature': feature_names[idx],
                'importance': round(float(mean_abs_shap[idx]), 6),
                'rank': rank + 1,
                'category': FEATURE_CATEGORIES.get(feature_names[idx], 'Other'),
            })

        # ── Feature summary ──────────────────────────────────────────
        feature_summary = {}
        for i, fname in enumerate(feature_names):
            shap_col = shap_primary[:, i]
            feature_summary[fname] = {
                'mean_abs_shap': round(float(np.mean(np.abs(shap_col))), 6),
                'std_shap': round(float(np.std(shap_col)), 6),
                'positive_effect_frac': round(float(np.mean(shap_col > 0)), 4),
                'category': FEATURE_CATEGORIES.get(fname, 'Other'),
            }

        # ── Dependence data (top features) ───────────────────────────
        top_n = min(SHAPAnalyzer.TOP_DEPENDENCE_FEATURES, n_features)
        top_indices = ranked_indices[:top_n]

        dependence = {}
        for fi in top_indices:
            fi = int(fi)
            fname = feature_names[fi]
            feat_vals = X_shap[:, fi].tolist()
            shap_vals = shap_primary[:, fi].tolist()

            # Find strongest interacting feature (highest correlation of |SHAP residual|)
            interaction_feat = None
            best_corr = 0.0
            shap_col = shap_primary[:, fi]
            for j in range(n_features):
                if j == fi:
                    continue
                if np.std(X_shap[:, j]) < 1e-10:
                    continue
                corr = abs(float(np.corrcoef(np.abs(shap_col), X_shap[:, j])[0, 1]))
                if not np.isnan(corr) and corr > best_corr:
                    best_corr = corr
                    interaction_feat = feature_names[j]

            dependence[fname] = {
                'values': [round(v, 6) for v in feat_vals],
                'shap_values': [round(v, 6) for v in shap_vals],
                'interaction_feature': interaction_feat,
                'interaction_strength': round(best_corr, 4),
            }

        # ── Per-class breakdown (multiclass) ─────────────────────────
        per_class = None
        if is_multiclass and class_names and shap_values_list:
            per_class = {}
            for ci, cname in enumerate(class_names):
                if ci >= len(shap_values_list):
                    break
                class_shap = shap_values_list[ci]  # (n_samples, n_features)
                class_mean = np.mean(class_shap, axis=0)
                class_abs_mean = np.mean(np.abs(class_shap), axis=0)
                top5_idx = np.argsort(class_abs_mean)[::-1][:5]
                per_class[cname] = [
                    {
                        'feature': feature_names[int(idx)],
                        'mean_shap': round(float(class_mean[int(idx)]), 6),
                        'abs_importance': round(float(class_abs_mean[int(idx)]), 6),
                        'direction': 'positive' if class_mean[int(idx)] > 0 else 'negative',
                    }
                    for idx in top5_idx
                ]

        # ── Insights ─────────────────────────────────────────────────
        insights = SHAPAnalyzer._generate_insights(
            shap_primary, X_shap, feature_names, class_names,
        )

        # ── Build result ─────────────────────────────────────────────
        # Subsample raw values for dashboard (cap at 200 for JSON size)
        sample_n = min(200, len(X_shap))
        sample_idx = np.linspace(0, len(X_shap) - 1, sample_n, dtype=int)

        result = {
            'valid': True,
            'n_samples': len(X_shap),
            'n_features': n_features,
            'explainer_type': explainer_type,
            'global_importance': global_importance,
            'feature_summary': feature_summary,
            'dependence': dependence,
            'insights': insights,
            'per_class': per_class,
            'shap_values_sample': shap_primary[sample_idx].tolist(),
            'X_sample': X_shap[sample_idx].tolist(),
            'feature_names': feature_names,
            'class_names': class_names,
        }

        return result

    # ── Insight generation ────────────────────────────────────────

    @staticmethod
    def _generate_insights(
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: List[str],
        class_names: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Generate actionable natural-language insights from SHAP values."""
        insights = []
        n_samples, n_features = shap_values.shape

        for i, fname in enumerate(feature_names):
            feat_vals = X[:, i]
            shap_vals = shap_values[:, i]
            abs_mean = float(np.mean(np.abs(shap_vals)))

            # Skip weak features
            if abs_mean < 0.005 or n_samples < 20:
                continue

            # ── Monotonic relationship ────────────────────────────
            if np.std(feat_vals) > 1e-10:
                corr = float(np.corrcoef(feat_vals, shap_vals)[0, 1])
                if not np.isnan(corr) and abs(corr) > 0.3:
                    direction = 'higher' if corr > 0 else 'lower'
                    effect = 'positive' if corr > 0 else 'negative'
                    insights.append({
                        'feature': fname,
                        'type': 'monotonic',
                        'insight': (
                            f"{direction.title()} {fname.replace('_', ' ')} has a "
                            f"{effect} effect on predictions "
                            f"(correlation: {corr:.2f})"
                        ),
                        'strength': round(abs_mean, 4),
                        'correlation': round(corr, 4),
                    })
                    continue

            # ── Quartile analysis ─────────────────────────────────
            try:
                q25, q50, q75 = np.percentile(feat_vals, [25, 50, 75])
            except Exception:
                continue

            if q25 == q75:
                continue

            bins = [
                ('Q1 (low)', feat_vals <= q25),
                ('Q2', (feat_vals > q25) & (feat_vals <= q50)),
                ('Q3', (feat_vals > q50) & (feat_vals <= q75)),
                ('Q4 (high)', feat_vals > q75),
            ]

            bin_means = []
            for bin_name, mask in bins:
                if mask.sum() >= 5:
                    bin_means.append((bin_name, float(np.mean(shap_vals[mask])), mask.sum()))
                else:
                    bin_means.append((bin_name, None, 0))

            valid_bins = [(n, m, c) for n, m, c in bin_means if m is not None]
            if len(valid_bins) < 2:
                continue

            # Find the bin with highest mean SHAP
            best_bin = max(valid_bins, key=lambda x: x[1])
            worst_bin = min(valid_bins, key=lambda x: x[1])

            spread = best_bin[1] - worst_bin[1]
            if spread > 0.01:
                # Map bin to actual value range
                if best_bin[0] == 'Q1 (low)':
                    range_str = f"below {q25:.3f}"
                elif best_bin[0] == 'Q2':
                    range_str = f"between {q25:.3f} and {q50:.3f}"
                elif best_bin[0] == 'Q3':
                    range_str = f"between {q50:.3f} and {q75:.3f}"
                else:
                    range_str = f"above {q75:.3f}"

                insights.append({
                    'feature': fname,
                    'type': 'range_effect',
                    'insight': (
                        f"{fname.replace('_', ' ')} {range_str} has the strongest "
                        f"positive effect (mean SHAP: {best_bin[1]:+.4f}, n={best_bin[2]})"
                    ),
                    'strength': round(abs_mean, 4),
                    'best_range': range_str,
                    'best_shap': round(best_bin[1], 4),
                })

            # ── Threshold detection ───────────────────────────────
            # Sort by feature value, find where SHAP crosses zero
            sort_idx = np.argsort(feat_vals)
            sorted_shap = shap_vals[sort_idx]
            sorted_feat = feat_vals[sort_idx]

            # Smooth with rolling mean
            window = max(5, n_samples // 20)
            if len(sorted_shap) > window:
                smoothed = np.convolve(sorted_shap, np.ones(window)/window, mode='valid')
                smoothed_feat = sorted_feat[window//2:window//2+len(smoothed)]

                # Check for sign change
                sign_changes = np.where(np.diff(np.sign(smoothed)))[0]
                if len(sign_changes) == 1:
                    threshold_val = float(smoothed_feat[sign_changes[0]])
                    above_mean = float(np.mean(shap_vals[feat_vals > threshold_val]))
                    below_mean = float(np.mean(shap_vals[feat_vals <= threshold_val]))

                    if abs(above_mean - below_mean) > 0.01:
                        insights.append({
                            'feature': fname,
                            'type': 'threshold',
                            'insight': (
                                f"{fname.replace('_', ' ')} crosses impact threshold at "
                                f"{threshold_val:.3f} "
                                f"(below: {below_mean:+.4f}, above: {above_mean:+.4f} avg SHAP)"
                            ),
                            'strength': round(abs_mean, 4),
                            'threshold': round(threshold_val, 4),
                        })

        # Sort by strength descending
        insights.sort(key=lambda x: x['strength'], reverse=True)

        return insights
