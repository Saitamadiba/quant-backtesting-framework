"""Parameter grid generation for WFO optimization."""

import itertools
import math
from typing import Dict, List, Any

import numpy as np

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
    Random sampling when full grid is too large.

    Uses stratified random sampling across each parameter dimension.
    Always includes the default parameter set as the first sample.
    """
    rng = np.random.RandomState(seed)
    grid = []

    # Always include defaults
    defaults = {s.name: s.default for s in param_specs}
    grid.append(defaults)

    for _ in range(n_samples - 1):
        params = {}
        for s in param_specs:
            if s.param_type == 'int':
                val = rng.randint(int(s.min_val), int(s.max_val) + 1)
                params[s.name] = val
            else:
                val = rng.uniform(s.min_val, s.max_val)
                # Snap to step
                steps = round((val - s.min_val) / s.step) if s.step > 0 else 0
                val = s.min_val + steps * s.step
                val = min(val, s.max_val)
                params[s.name] = round(val, 6)
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
    )


def suggest_from_param_spec(trial, spec: ParamSpec):
    """Convert a ParamSpec into an Optuna trial suggestion."""
    if spec.param_type == 'int':
        return trial.suggest_int(
            spec.name, int(spec.min_val), int(spec.max_val),
            step=max(1, int(spec.step)),
        )
    else:
        if spec.step > 0:
            return trial.suggest_float(
                spec.name, spec.min_val, spec.max_val, step=spec.step,
            )
        else:
            return trial.suggest_float(spec.name, spec.min_val, spec.max_val)
