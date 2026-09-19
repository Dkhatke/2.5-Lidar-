"""
run_baselines.py — the central experiment.

Runs every allocation policy across a range of budgets and writes one tidy CSV
row per (policy, budget, frame).  Every number in the final report traces back
to a row in this file.

THE COMPARISON IS EQUAL MEMORY, NOT EQUAL RESOLUTION
----------------------------------------------------
Comparing an adaptive map against a uniform 5 cm map is rigged, and an
evaluator will see it: of course the adaptive one is smaller, it was told to
be.  Fix the memory instead and ask the question that matters — what is the
best map obtainable for this budget?  Uniform spends it evenly; ours spends it
where the value function says it matters.  Which one still contains the
pedestrian?

Usage:
    python scripts/run_baselines.py
    python scripts/run_baselines.py --frames 12 --scenario mixed_urban
    python scripts/run_baselines.py --backend oracle     # map quality only
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
for p in (os.path.dirname(_PKG), _PKG):
    if p not in sys.path:
        sys.path.insert(0, p)

from adaptive_lidar.data.loader import get_dataset                  # noqa: E402
from adaptive_lidar.evaluation.metrics import (                     # noqa: E402
    BAND_NAMES,
    boundary_consistency,
    elevation_error,
    latency_percentiles,
    map_completeness,
    object_retention_rate,
    range_stratified_iou,
)
from adaptive_lidar.evaluation.reference_map import build_reference_map  # noqa: E402
from adaptive_lidar.mapping.adaptive_map import UniformReference    # noqa: E402
from adaptive_lidar.mapping.allocation import AllocationController  # noqa: E402
from adaptive_lidar.pipeline.pipeline import Pipeline               # noqa: E402
from adaptive_lidar.pipeline.types import CLASS_NAMES, NUM_CLASSES  # noqa: E402
from adaptive_lidar.utils.config import load_config                 # noqa: E402

DEFAULT_BUDGETS = (1.0, 0.75, 0.5, 0.25, 0.1)


def run_one(cfg, frames, policy, budget, backend, reference):
    pipe = Pipeline(cfg)
    pipe.build_stages(backend=backend, policy=policy)
    pipe.set_budget(budget)

    pred, true, rng = [], [], []
    t0 = time.perf_counter()
    for k, f in enumerate(frames):
        fr = pipe.run(cloud_np=f, frame_id=f["frame_id"], timestamp=f["timestamp"])
        if k == 0:
            # Discard the first frame's timings: it pays import, allocator and
            # BLAS warm-up, and has no previous frame for the motion residual,
            # so it is not a steady-state measurement. Its MAP contribution is
            # kept - the map is meant to accumulate from frame zero.
            pipe.context.cumulative_timing.clear()
        if fr.gt_label is not None:
            pred.append(np.asarray(fr.sem_class))
            true.append(np.asarray(fr.gt_label))
            rng.append(fr._range)
    wall = time.perf_counter() - t0

    amap = pipe.context.amap
    ctx = pipe.context
    ego = frames[-1].get("pose")
    ego_xy = (0.0, 0.0) if ego is None else (ego[0, 3], ego[1, 3])

    sem = (range_stratified_iou(np.concatenate(pred), np.concatenate(true),
                                np.concatenate(rng)) if pred else None)
    elev = elevation_error(amap, reference, ego_xy)
    comp = map_completeness(amap, reference)
    orr = object_retention_rate(amap, frames)
    bnd = boundary_consistency(amap)
    lat = latency_percentiles(ctx)

    uni = ctx.uniform_ref
    return {
        "pipe": pipe, "sem": sem, "elev": elev, "comp": comp,
        "orr": orr, "bnd": bnd, "lat": lat, "wall_s": wall,
        "map_cells": amap.n_cells, "map_bytes": amap.nbytes(),
        "cells_per_level": amap.cell_counts(),
        "uniform5_cells": uni.n_cells if uni else 0,
        "uniform5_bytes": uni.nbytes() if uni else 0,
    }


def flatten(policy, budget, backend, scenario, r) -> dict:
    row = {
        "scenario": scenario, "policy": policy, "budget": budget,
        "backend": backend,
        "map_cells": r["map_cells"], "map_mb": r["map_bytes"] / 1e6,
        "uniform5_cells": r["uniform5_cells"],
        "uniform5_mb": r["uniform5_bytes"] / 1e6,
        "memory_reduction_pct": (
            100.0 * (1 - r["map_bytes"] / r["uniform5_bytes"])
            if r["uniform5_bytes"] else float("nan")),
        "wall_s": r["wall_s"],
    }
    for lvl in range(5):
        row[f"cells_L{lvl}"] = r["cells_per_level"].get(lvl, 0)

    lat = r["lat"].get("TOTAL", {})
    row["latency_p50_ms"] = lat.get("p50", float("nan"))
    row["latency_p95_ms"] = lat.get("p95", float("nan"))
    row["latency_p99_ms"] = lat.get("p99", float("nan"))
    row["fps_p50"] = lat.get("fps_p50", float("nan"))
    for s, d in r["lat"].items():
        if s != "TOTAL":
            row[f"{s}_p50_ms"] = d["p50"]
            row[f"{s}_p99_ms"] = d["p99"]

    if r["sem"]:
        row["miou"] = r["sem"]["overall"]["miou"]
        row["accuracy"] = r["sem"]["overall"]["accuracy"]
        for c in range(NUM_CLASSES):
            row[f"iou_{CLASS_NAMES[c]}"] = r["sem"]["overall"]["iou"][c]
        for name in BAND_NAMES:
            b = r["sem"]["bands"].get(name)
            row[f"miou_{name}"] = b["miou"] if b else float("nan")
            row[f"acc_{name}"] = b["accuracy"] if b else float("nan")

    ov = r["elev"].get("overall")
    row["elev_rmse_m"] = ov["rmse_m"] if ov else float("nan")
    row["elev_bias_m"] = ov["bias_m"] if ov else float("nan")
    for name in BAND_NAMES:
        b = r["elev"]["bands"].get(name)
        row[f"elev_rmse_{name}"] = b["rmse_m"] if b else float("nan")
        row[f"elev_bias_{name}"] = b["bias_m"] if b else float("nan")

    row["completeness"] = r["comp"]["completeness"]
    row["spurious_rate"] = r["comp"]["spurious_rate"]

    row["mean_orr"] = r["orr"]["mean_orr"]
    for c in range(NUM_CLASSES):
        d = r["orr"]["per_class"][c]
        row[f"orr_{CLASS_NAMES[c]}"] = d["orr"]
        row[f"orr_n_{CLASS_NAMES[c]}"] = d["observed"]

    t, i = r["bnd"].get("transition"), r["bnd"].get("interior")
    row["boundary_p95_transition_m"] = t["p95_m"] if t else float("nan")
    row["boundary_p95_interior_m"] = i["p95_m"] if i else float("nan")
    row["boundary_p95_ratio"] = r["bnd"].get("p95_ratio", float("nan"))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--scenario", default="mixed_urban")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--policies", default=None,
                    help="comma-separated; default = all")
    ap.add_argument("--budgets", default=None,
                    help="comma-separated fractions; default = 1,.75,.5,.25,.1")
    ap.add_argument("--out", default="docs/baselines.csv")
    ap.add_argument("--input", default=None)
    args = ap.parse_args()

    cfg = load_config()
    policies = (args.policies.split(",") if args.policies
                else list(AllocationController.POLICIES))
    budgets = ([float(b) for b in args.budgets.split(",")] if args.budgets
               else list(DEFAULT_BUDGETS))

    ds = get_dataset("synthetic" if args.input is None else "auto",
                     args.input, args.frames, scenario=args.scenario)
    frames = list(ds)

    print(f"\nBuilding the reference map from {len(frames)} frames "
          f"(GT-static points, uniform 5 cm)...")
    reference = build_reference_map(frames, source=ds.source)
    print(f"  {len(reference):,} reference cells, "
          f"{reference.n_moving_removed:,} moving points removed")
    for lim in reference.limitations():
        print(f"    limitation: {lim}")

    rows = []
    total = len(policies) * len(budgets)
    k = 0
    print(f"\nRunning {total} configurations x {len(frames)} frames...\n")
    print(f"  {'policy':<40}{'budget':>8}{'cells':>10}{'MB':>8}"
          f"{'mIoU':>8}{'ORR vru':>9}{'RMSE':>8}{'p99 ms':>9}")
    for policy in policies:
        for b in budgets:
            k += 1
            r = run_one(cfg, frames, policy, b, args.backend, reference)
            row = flatten(policy, b, r["pipe"].context.semantic_backend_name,
                          args.scenario, r)
            rows.append(row)
            print(f"  {policy:<40}{b:>8.2f}{row['map_cells']:>10,}"
                  f"{row['map_mb']:>8.2f}{row.get('miou', float('nan')):>8.3f}"
                  f"{row.get('orr_vru', float('nan')):>9.3f}"
                  f"{row['elev_rmse_m']:>8.3f}{row['latency_p99_ms']:>9.1f}")

    out = os.path.join(_PKG, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    ordered = ([c for c in ("scenario", "policy", "budget", "backend",
                            "map_cells", "map_mb", "uniform5_cells",
                            "uniform5_mb", "memory_reduction_pct")
                if c in keys]
               + [c for c in keys if c not in
                  ("scenario", "policy", "budget", "backend", "map_cells",
                   "map_mb", "uniform5_cells", "uniform5_mb",
                   "memory_reduction_pct")])
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ordered)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}  ({len(rows)} rows, {len(ordered)} columns)")

    meta = {
        "scenario": args.scenario,
        "data_source": ds.name,
        "n_frames": len(frames),
        "n_points_per_frame": int(np.mean([len(f["points"]) for f in frames])),
        "backend_requested": args.backend,
        "backend_active": rows[0]["backend"] if rows else None,
        "policies": policies,
        "budgets": budgets,
        "reference_cells": len(reference),
        "reference_frames": reference.n_frames,
        "reference_moving_removed": reference.n_moving_removed,
        "reference_limitations": reference.limitations(),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    mp = os.path.join(_PKG, "docs", "baselines_meta.json")
    with open(mp, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Wrote {mp}")


if __name__ == "__main__":
    main()
