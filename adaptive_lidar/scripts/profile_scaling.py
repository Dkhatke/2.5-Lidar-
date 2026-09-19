"""
profile_scaling.py — per-stage latency vs point count.

Runs the full S0-S9 pipeline on synthetic clouds of increasing size and
records per-stage wall time.  Used to produce docs/perf_baseline.csv
(Phase 0) and docs/perf_phase1.csv (Phase 1).

Usage:
    python scripts/profile_scaling.py --out docs/perf_baseline.csv
    python scripts/profile_scaling.py --out docs/perf_phase1.csv --frames 5
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
_PARENT = os.path.dirname(_PKG)
for p in (_PARENT, _PKG):
    if p not in sys.path:
        sys.path.insert(0, p)

STAGES = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"]
TARGETS = [8_000, 30_000, 120_000]


def _resample(cloud: np.ndarray, n_target: int, rng: np.random.Generator) -> np.ndarray:
    """Grow/shrink a cloud to n_target points by resampling with small jitter.

    Only used for the *scaling* profile: we care about how each stage's cost
    grows with N, not about the scene being physically re-simulated.
    """
    n = len(cloud)
    if n_target <= n:
        idx = rng.choice(n, n_target, replace=False)
        return cloud[idx]
    reps = int(np.ceil(n_target / n))
    out = np.tile(cloud, (reps, 1))[:n_target].copy()
    out[:, :3] += rng.normal(0.0, 0.01, (n_target, 3)).astype(out.dtype)
    return out


def _make_cloud(n_target: int, rng: np.random.Generator) -> np.ndarray:
    """Build a synthetic cloud of approximately n_target points."""
    try:
        from adaptive_lidar.data.synthetic_scene import generate_scenario
        scene = generate_scenario("mixed_urban", frame_idx=0)
        cloud = np.column_stack([scene["points"], scene["intensity"]]).astype(np.float32)
    except Exception:
        from adaptive_lidar.data.synthetic_scene import generate_scene
        cloud = generate_scene(0, 8)
    return _resample(cloud, n_target, rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/perf_baseline.csv")
    ap.add_argument("--frames", type=int, default=3,
                    help="frames per size (first frame discarded as warm-up)")
    ap.add_argument("--budget", type=float, default=0.8)
    ap.add_argument("--backend", default="auto")
    args = ap.parse_args()

    from adaptive_lidar.utils.config import load_config
    from adaptive_lidar.pipeline.pipeline import Pipeline

    config = load_config()
    rng = np.random.default_rng(0)
    rows = []

    for n_target in TARGETS:
        cloud = _make_cloud(n_target, rng)
        pipe = Pipeline(config)
        pipe.build_stages(backend=args.backend)
        pipe.set_budget(args.budget)

        per_stage = {s: [] for s in STAGES}
        totals = []
        n_actual = 0
        for f in range(args.frames + 1):          # +1 warm-up
            t0 = time.perf_counter()
            frame = pipe.run(cloud_np=cloud, frame_id=f, timestamp=f * 0.1)
            wall = (time.perf_counter() - t0) * 1000.0
            if f == 0:
                continue                          # discard warm-up
            n_actual = len(frame.points)
            for s in STAGES:
                v = frame.timing.get(s)
                if isinstance(v, float):
                    per_stage[s].append(v)
            totals.append(wall)

        row = {"n_points": n_actual}
        for s in STAGES:
            row[s] = round(float(np.mean(per_stage[s])), 2) if per_stage[s] else 0.0
        row["stage_sum_ms"] = round(sum(row[s] for s in STAGES), 2)
        row["wall_ms"] = round(float(np.mean(totals)), 2)
        row["fps"] = round(1000.0 / max(row["wall_ms"], 1e-6), 2)
        rows.append(row)
        print(f"  N={n_actual:>7,}  wall={row['wall_ms']:>8.1f} ms  "
              + "  ".join(f"{s}:{row[s]:.1f}" for s in STAGES))

    out_path = os.path.join(_PKG, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out_path}")

    # Markdown echo for PROGRESS.md
    cols = list(rows[0].keys())
    print("\n| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")


if __name__ == "__main__":
    main()
