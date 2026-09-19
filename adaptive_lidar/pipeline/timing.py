"""
Pipeline timing utilities.
Every stage must wrap its body with StageTimer.
"""
import time
from contextlib import contextmanager
from typing import Dict


@contextmanager
def stage_timer(name: str, timing_dict: Dict[str, float]):
    """Context manager that measures duration_ms and stores into timing_dict."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timing_dict[name] = (time.perf_counter() - t0) * 1000.0


class TimingStats:
    """Accumulated stage timing across frames."""

    def __init__(self):
        self._data: Dict[str, list] = {}

    def record(self, stage: str, ms: float):
        self._data.setdefault(stage, []).append(ms)

    def p50(self, stage: str) -> float:
        import numpy as np
        vals = self._data.get(stage, [0.0])
        return float(np.percentile(vals, 50))

    def p95(self, stage: str) -> float:
        import numpy as np
        vals = self._data.get(stage, [0.0])
        return float(np.percentile(vals, 95))

    def latest(self, stage: str) -> float:
        vals = self._data.get(stage, [0.0])
        return vals[-1] if vals else 0.0

    def total_latest(self) -> float:
        return sum(self.latest(s) for s in self._data)
