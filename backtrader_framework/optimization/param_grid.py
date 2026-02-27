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
