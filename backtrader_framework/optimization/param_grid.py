"""Parameter grid generation for WFO optimization."""

import itertools
import math
from typing import Dict, List, Any

import numpy as np
from scipy.stats.qmc import Sobol

from .strategy_adapters.base_adapter import ParamSpec


def generate_grid(param_specs: List[ParamSpec], max_combos: int = 1000) -> List[Dict[str, Any]]:
    """
    Generate full parameter grid from specs.

    If total combos exceed max_combos, automatically coarsen steps
    (double step sizes) until grid fits within budget.
    """
    specs = [_copy_spec(s) for s in param_specs]

    # Iteratively coarsen until grid fits
    for _ in range(10):
        axes = []
        for s in specs:
            vals = _axis_values(s)
            axes.append(vals)

        total = 1
        for a in axes:
            total *= len(a)

        if total <= max_combos:
            break

        # Double the step of the param with the most values
        longest_idx = max(range(len(axes)), key=lambda i: len(axes[i]))
        specs[longest_idx].step *= 2
    else:
        # Still too large — fall back to random
        return generate_random_grid(param_specs, n_samples=max_combos)

    # Build grid
    grid = []
    for combo in itertools.product(*axes):
        params = {}
        for spec, val in zip(specs, combo):
            if spec.param_type == 'int':
                params[spec.name] = int(round(val))
            else:
                params[spec.name] = round(val, 6)
        grid.append(params)

    return grid


def generate_random_grid(
    param_specs: List[ParamSpec], n_samples: int = 500, seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Quasi-random sampling using Sobol sequences for better space coverage.

    Sobol sequences provide more uniform coverage of the parameter space
    compared to pseudo-random sampling, reducing the risk of missing
    important regions. Supports log-scale parameters via ParamSpec.log_scale.

    Always includes the default parameter set as the first sample.
    """
    grid = []

    # Always include defaults
    defaults = {s.name: s.default for s in param_specs}
    grid.append(defaults)

    n_dims = len(param_specs)
    if n_dims == 0 or n_samples <= 1:
        return grid

    # Sobol requires power-of-2 samples
    m = int(np.ceil(np.log2(max(n_samples - 1, 2))))
    sampler = Sobol(d=n_dims, scramble=True, seed=seed)
    unit_samples = sampler.random_base2(m)[: n_samples - 1]  # trim to requested size

    # Scale from [0,1] to parameter ranges
    for sample in unit_samples:
        params = {}
        for i, spec in enumerate(param_specs):
            if spec.log_scale and spec.min_val > 0:
                # Sample in log space for parameters spanning orders of magnitude
                log_min = np.log(spec.min_val)
                log_max = np.log(spec.max_val)
                val = np.exp(log_min + sample[i] * (log_max - log_min))
            else:
                val = spec.min_val + sample[i] * (spec.max_val - spec.min_val)

            if spec.step and spec.step > 0:
                val = round(val / spec.step) * spec.step
            val = max(spec.min_val, min(val, spec.max_val))

            if spec.param_type == 'int':
                params[spec.name] = int(round(val))
            else:
                params[spec.name] = round(val, 6)
        grid.append(params)

    return grid


def _axis_values(spec: ParamSpec) -> List[float]:
    """Generate axis values for one parameter."""
    if spec.step <= 0:
        return [spec.default]
    n_steps = int(round((spec.max_val - spec.min_val) / spec.step))
    vals = [spec.min_val + i * spec.step for i in range(n_steps + 1)]
    vals = [v for v in vals if v <= spec.max_val + 1e-9]
    if not vals:
        vals = [spec.default]
    return vals


def _copy_spec(s: ParamSpec) -> ParamSpec:
    return ParamSpec(
        name=s.name, default=s.default,
        min_val=s.min_val, max_val=s.max_val,
        step=s.step, param_type=s.param_type,
        log_scale=s.log_scale,
    )


def get_param_neighbors(
    params: Dict[str, Any],
    param_specs: List[ParamSpec],
    steps: int = 1,
) -> List[Dict[str, Any]]:
    """Generate neighbor parameter sets by varying each param by ±steps.

    For each parameter, generates up to 2 neighbors (one step up, one step down)
    keeping all other params fixed.  Clips to [min_val, max_val].
    Returns list of neighbor param dicts (excludes the center point).
    """
    neighbors = []
    spec_map = {s.name: s for s in param_specs}

    for name, center_val in params.items():
        spec = spec_map.get(name)
        if spec is None or spec.step <= 0:
            continue
        for direction in (-steps, steps):
            new_val = center_val + direction * spec.step
            new_val = max(spec.min_val, min(new_val, spec.max_val))
            if abs(new_val - center_val) < 1e-9:
                continue
            neighbor = dict(params)
            if spec.param_type == 'int':
                neighbor[name] = int(round(new_val))
            else:
                neighbor[name] = round(new_val, 6)
            neighbors.append(neighbor)

    return neighbors


def compute_stability_ratio(
    best_score: float,
    combo_scores: list,
    param_specs: List[ParamSpec],
) -> Dict[str, Any]:
    """Compute parameter stability from the score landscape.

    stability_ratio = mean(neighbor_scores) / best_score.
    ~1.0 = flat/stable optimum (robust).  <0.5 = spike (fragile/overfit).
    """
    if best_score <= 0 or not combo_scores:
        return {'stability_ratio': 0.0, 'n_neighbors_found': 0, 'neighbor_mean_score': None}

    best_params = max(combo_scores, key=lambda x: x[1])[0]
    neighbors = get_param_neighbors(best_params, param_specs)

    def _key(p):
        return tuple(sorted(p.items()))

    score_map = {_key(p): s for p, s in combo_scores}

    neighbor_scores = []
    for nb in neighbors:
        k = _key(nb)
        if k in score_map:
            neighbor_scores.append(score_map[k])

    if not neighbor_scores:
        return {'stability_ratio': 0.0, 'n_neighbors_found': 0, 'neighbor_mean_score': None}

    neighbor_mean = float(np.mean(neighbor_scores))
    stability_ratio = neighbor_mean / best_score if best_score > 0 else 0.0

    return {
        'stability_ratio': round(stability_ratio, 4),
        'n_neighbors_found': len(neighbor_scores),
        'neighbor_mean_score': round(neighbor_mean, 6),
    }


def suggest_from_param_spec(trial, spec: ParamSpec):
    """Convert a ParamSpec into an Optuna trial suggestion."""
    if spec.param_type == 'int':
        return trial.suggest_int(
            spec.name, int(spec.min_val), int(spec.max_val),
            step=max(1, int(spec.step)),
        )
    else:
        if spec.log_scale and spec.min_val > 0:
            return trial.suggest_float(
                spec.name, spec.min_val, spec.max_val, log=True,
            )
        elif spec.step > 0:
            return trial.suggest_float(
                spec.name, spec.min_val, spec.max_val, step=spec.step,
            )
        else:
            return trial.suggest_float(spec.name, spec.min_val, spec.max_val)
