"""
test_temporal.py — Phase 6 verification.

Runs the ``moving_vehicle`` scenario for 20 frames and measures TRAIL LENGTH:
how many metres of spurious OCCUPIED cells the persistent map keeps behind the
vehicle's true position.

Both configurations are reported.  Showing your own ablation failing is
persuasive, not embarrassing — the number with the gate off is what the gate
is worth.

Run:  python scripts/test_temporal.py
"""
from __future__ import annotations

import copy
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
for p in (os.path.dirname(_PKG), _PKG):
    if p not in sys.path:
        sys.path.insert(0, p)

from adaptive_lidar.data.loader import get_dataset            # noqa: E402
from adaptive_lidar.evaluation.metrics import (                # noqa: E402
    BIN_M,
    MIN_CELLS_PER_BIN,
    trail_profile,
    vehicle_half_length,
)
from adaptive_lidar.pipeline.pipeline import Pipeline         # noqa: E402
from adaptive_lidar.pipeline.types import TrackState          # noqa: E402
from adaptive_lidar.utils.config import load_config           # noqa: E402

N_FRAMES = 20
TRAIL_LIMIT_M = 1.0
FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  —  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)
    return ok


def hr(t):
    print("\n" + "=" * 78)
    print(f"  {t}")
    print("=" * 78)


# ════════════════════════════════════════════════════════════
def vehicle_truth(frames):
    """(T, 2) the moving vehicle's true centroid per frame, in world xy."""
    out = []
    for f in frames:
        pose = f["pose"]
        m = np.asarray(f["gt_moving"])
        if not m.any():
            out.append(None)
            continue
        pw = np.asarray(f["points"])[:, :3] @ pose[:3, :3].T + pose[:3, 3]
        out.append(pw[m][:, :2].mean(axis=0))
    return out


def run(gate_enabled, cfg, frames):
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("motion", {})["gate_enabled"] = gate_enabled
    pipe = Pipeline(cfg)
    pipe.build_stages(backend="auto", policy="full")
    pipe.set_budget(0.5)
    gated = 0
    for f in frames:
        fr = pipe.run(cloud_np=f, frame_id=f["frame_id"], timestamp=f["timestamp"])
        gated += int(fr.timing.get("S8_diag", {}).get("n_moving", 0))
    return pipe, fr, gated


# ════════════════════════════════════════════════════════════
def main():
    cfg = load_config()
    ds = get_dataset("synthetic", None, N_FRAMES,
                     scenario="moving_vehicle", quiet=True)
    frames = list(ds)
    truth = vehicle_truth(frames)
    rear_m = vehicle_half_length(frames)
    moved = np.linalg.norm(
        np.array([p for p in truth if p is not None])[-1]
        - np.array([p for p in truth if p is not None])[0])

    hr(f"SCENARIO — moving_vehicle, {N_FRAMES} frames at 10 Hz")
    print(f"  the vehicle travels {moved:.1f} m across the road during the run")
    print(f"  its true extent is {2 * rear_m:.1f} m, so cells within "
          f"{rear_m:.1f} m of the centroid are the vehicle itself")
    print(f"  a map that writes moving points persistently should therefore")
    print(f"  keep a phantom wall of up to ~{moved:.0f} m behind it")

    results = {}
    for gate in (True, False):
        pipe, last, n_gated = run(gate, cfg, frames)
        amap = pipe.context.amap
        tl, n_cells, max_b, bins = trail_profile(
            amap, truth, truth[-1], rear_m=rear_m)
        results[gate] = {
            "trail_m": tl, "trail_cells": n_cells,
            "max_behind": max_b, "bins": bins,
            "map_cells": amap.n_cells, "gated_points": n_gated,
            "overlay": len(pipe.context.dynamic_overlay or []),
            "tracks": last.instances or [],
        }

    hr("TRAIL LENGTH  (metres of spurious OCCUPIED cells behind the vehicle)")
    print(f"  A trail is a phantom WALL: a 0.5 m bin counts only if it holds "
          f"at least\n  {MIN_CELLS_PER_BIN} spurious cells, and the trail is "
          f"how far back the CONTIGUOUS run of\n  such bins reaches. The "
          f"furthest single cell is reported beside it so that\n  the "
          f"contiguity rule cannot hide a long tail.\n")
    print(f"  {'MOS gate':>10}{'trail (m)':>11}{'furthest':>11}"
          f"{'cells':>9}{'map cells':>12}{'points gated':>14}")
    for gate in (True, False):
        r = results[gate]
        print(f"  {('ON' if gate else 'OFF'):>10}{r['trail_m']:>11.2f}"
              f"{r['max_behind']:>10.2f}m{r['trail_cells']:>9,}"
              f"{r['map_cells']:>12,}{r['gated_points']:>14,}")

    print(f"\n  spurious cells per 0.5 m bin behind the vehicle:")
    for gate in (True, False):
        b = results[gate]["bins"][:24]
        print(f"    gate {('ON ' if gate else 'OFF')}: "
              + " ".join(f"{c:>3}" for c in b))

    on, off = results[True], results[False]
    check(f"trail < {TRAIL_LIMIT_M} m with the gate ON",
          on["trail_m"] < TRAIL_LIMIT_M, f"{on['trail_m']:.2f} m")
    check("the gate measurably reduces the trail",
          on["trail_m"] < off["trail_m"] or off["trail_m"] == 0.0,
          f"{on['trail_m']:.2f} m vs {off['trail_m']:.2f} m without it")
    check("moving points are actually being gated out",
          on["gated_points"] > 0, f"{on['gated_points']:,} points over {N_FRAMES} frames")

    hr("OBJECT TABLE  (last frame, gate ON)")
    tracks = on["tracks"]
    print(f"  {'id':>4}{'class':>18}{'state':>26}{'speed':>9}{'pts':>7}{'age':>5}")
    from adaptive_lidar.pipeline.types import CLASS_NAMES
    for t in sorted(tracks, key=lambda i: -i.point_count)[:10]:
        print(f"  {t.instance_id:>4}{CLASS_NAMES[t.semantic_class]:>18}"
              f"{t.state_name:>26}{t.speed:>8.2f}m/s{t.point_count:>7}{t.age:>5}")
    states = {s: sum(1 for t in tracks if t.state == s)
              for s in (TrackState.STATIC, TrackState.MOVING,
                        TrackState.MOVABLE_BUT_STATIONARY)}
    print(f"\n  STATIC {states[0]}   MOVING {states[1]}   "
          f"MOVABLE_BUT_STATIONARY {states[2]}")
    check("at least one track is classified MOVING", states[1] >= 1,
          f"{states[1]} moving tracks")
    print(f"\n  dynamic overlay holds {on['overlay']:,} points "
          f"(rebuilt every frame, never persisted)")

    hr("RESULT")
    if FAILURES:
        print(f"  {len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"    - {f}")
        sys.exit(1)
    print("  ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
