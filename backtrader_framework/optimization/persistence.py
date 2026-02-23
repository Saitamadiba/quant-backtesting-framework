"""Persistence layer for WFO results — JSON save/load."""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional


_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def _ensure_results_dir():
    os.makedirs(_RESULTS_DIR, exist_ok=True)


def save_wfo_result(result: Dict) -> str:
    """
    Save WFO result dict to JSON. Returns the file path.

    Filename: wfo_{strategy}_{symbol}_{timeframe}_{timestamp}.json
    """
    _ensure_results_dir()

    strategy = result.get('strategy_name', 'unknown')
    symbol = result.get('symbol', 'unknown')
    timeframe = result.get('timeframe', 'unknown')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    filename = f"wfo_{strategy}_{symbol}_{timeframe}_{ts}.json"
    filepath = os.path.join(_RESULTS_DIR, filename)

    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    return filepath


def load_wfo_result(filepath: str) -> Dict:
    """Load a single WFO result from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


def list_wfo_results() -> List[Dict]:
    """
    List all saved WFO results with metadata.

    Returns list of dicts with 'filepath', 'filename', 'strategy', 'symbol',
    'timeframe', 'timestamp' sorted newest-first.
    """
    _ensure_results_dir()
    results = []

    for fname in os.listdir(_RESULTS_DIR):
        if not fname.startswith('wfo_') or not fname.endswith('.json'):
            continue

        parts = fname.replace('.json', '').split('_')
        # wfo_{strategy}_{symbol}_{timeframe}_{date}_{time}
        if len(parts) >= 6:
            results.append({
                'filepath': os.path.join(_RESULTS_DIR, fname),
                'filename': fname,
                'strategy': parts[1],
                'symbol': parts[2],
                'timeframe': parts[3],
                'timestamp': f"{parts[4]}_{parts[5]}",
            })

    results.sort(key=lambda x: x['timestamp'], reverse=True)
    return results


def load_latest_wfo(strategy: str = None, symbol: str = None) -> Optional[Dict]:
    """Load the most recent WFO result, optionally filtered."""
    all_results = list_wfo_results()
    for r in all_results:
        if strategy and r['strategy'] != strategy:
            continue
        if symbol and r['symbol'] != symbol:
            continue
        return load_wfo_result(r['filepath'])
    return None
