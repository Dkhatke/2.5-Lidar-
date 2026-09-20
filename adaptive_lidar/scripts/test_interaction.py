"""
Click-to-inspect, verified against a REAL precomputed run.

The unit tests in ``tests/test_selection.py`` build cell dictionaries by
hand, which proves the lookup logic but not that it agrees with the map the
pipeline actually produces. This script runs the real thing:

    python scripts/test_interaction.py

and asserts four claims that would each silently give a wrong-but-plausible
answer if broken:

  1. the inverse projection round-trips every camera and both view frames;
  2. a cell picked out of a real cached frame, projected to a pixel and
     clicked back, resolves to THAT cell;
  3. the cell found is the finest one covering the point;
  4. clicking a tracked object selects the object, not the ground under it.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from adaptive_lidar.pipeline.types import ResolutionLevel          # noqa: E402
from adaptive_lidar.utils.config import load_config                # noqa: E402
from adaptive_lidar.visualization import camera as CAM             # noqa: E402
from adaptive_lidar.visualization import interactive_map as IM     # noqa: E402
from adaptive_lidar.visualization import selection as SEL          # noqa: E402
from adaptive_lidar.visualization.coordinate_transform import (    # noqa: E402
    round_trip_error)

N_FRAMES = 6
SCENARIO = "mixed_urban"


def _run():
    from adaptive_lidar.data.loader import get_dataset
    from adaptive_lidar.visualization.playback import precompute
    ds = get_dataset("synthetic", None, N_FRAMES, scenario=SCENARIO,
                     quiet=True)
    return precompute(load_config(), list(ds), SCENARIO, "full", 0.5,
                      "auto", True)


def main() -> int:
    print(f"precomputing {SCENARIO}, {N_FRAMES} frames...")
    run = _run()
    snap = run.frames[len(run) // 2]
    cells = snap.cells
    n = len(cells["cx"])
    print(f"frame {snap.frame_id}: {n:,} cached cells, "
          f"{len(snap.objects)} tracked objects, ego at "
          f"({snap.ego_xy[0]:.1f}, {snap.ego_xy[1]:.1f})")
    levels, counts = np.unique(cells["level"], return_counts=True)
    print("  levels present: " + ", ".join(
        f"{ResolutionLevel.name(int(l))}x{int(c):,}"
        for l, c in zip(levels, counts)))

    failures = []

    # ── 1. the inverse projection ────────────────────────────
    print("\n1. inverse projection round-trip")
    worst_overall = 0.0
    for preset in CAM.CAMERA_PRESETS:
        for frame_ref in CAM.FRAME_CHOICES:
            view = IM.build_view(preset, frame_ref, snap.ego_xy,
                                 snap.heading)
            hw = IM.image_size_for(view)
            t = IM.transform_for(view, frame_ref, snap.ego_xy, snap.heading,
                                 hw)
            rng = np.random.default_rng(0)
            pts = np.array([t.screen_to_world(x, y) for x, y in
                            zip(rng.uniform(0, hw[1], 150),
                                rng.uniform(0, hw[0], 150))])
            err = round_trip_error(t, pts)
            worst_overall = max(worst_overall, err)
            ok = err < 0.01 * t.metres_per_pixel()
            print(f"   {preset:<12} {frame_ref:<8} max error "
                  f"{err * 1000:7.4f} mm   {'ok' if ok else 'FAIL'}")
            if not ok:
                failures.append(f"round trip {preset}/{frame_ref}")
    print(f"   worst across every camera: {worst_overall * 1000:.4f} mm "
          f"(a pixel is {1000 / 7:.0f} mm at the default zoom)")

    # ── 2. click a real cell, get that cell back ─────────────
    print("\n2. click a real cell -> the same cell")
    for preset, frame_ref in (("top-follow", "World"), ("chase", "World"),
                              ("chase", "Vehicle")):
        view = IM.build_view(preset, frame_ref, snap.ego_xy, snap.heading)
        hw = IM.image_size_for(view)
        t = IM.transform_for(view, frame_ref, snap.ego_xy, snap.heading, hw)

        # Only cells whose own pixel is on screen can be clicked at all.
        rng = np.random.default_rng(11)
        trial = rng.choice(n, size=min(400, n), replace=False)
        hit = miss = offscreen = 0
        for row in trial:
            cx = float(cells["cx"][row])
            cy = float(cells["cy"][row])
            px, py = t.world_to_screen(cx, cy)
            if not t.contains_display_px(px, py):
                offscreen += 1
                continue
            got = SEL.find_cell_row(cells, *t.screen_to_world(px, py))
            if got is not None and got[0] == int(row):
                hit += 1
            else:
                # A coarse cell can legitimately lose its own centre to a
                # finer cell drawn on top of it; that is claim 3, not a bug.
                got_lvl = -1 if got is None else got[1]
                if got_lvl >= 0 and got_lvl < int(cells["level"][row]):
                    hit += 1
                else:
                    miss += 1
        tested = hit + miss
        rate = hit / max(tested, 1)
        print(f"   {preset:<12} {frame_ref:<8} {hit}/{tested} exact "
              f"({100 * rate:.1f}%), {offscreen} off screen   "
              f"{'ok' if rate == 1.0 else 'FAIL'}")
        if rate != 1.0:
            failures.append(f"click->cell {preset}/{frame_ref}")

    # ── 3. the finest containing cell wins ───────────────────
    print("\n3. the finest cell covering the point is the one returned")
    rng = np.random.default_rng(5)
    checked = worse = 0
    for row in rng.choice(n, size=min(600, n), replace=False):
        cx, cy = float(cells["cx"][row]), float(cells["cy"][row])
        got = SEL.find_cell_row(cells, cx, cy)
        if got is None:
            continue
        got_row, got_lvl = got
        # No cell of a FINER level may also contain this point.
        for lvl in range(got_lvl):
            s = ResolutionLevel.size(lvl)
            sel = np.flatnonzero(cells["level"] == lvl)
            if sel.size == 0:
                continue
            ix = np.floor(np.asarray(cells["cx"][sel], np.float64) / s)
            iy = np.floor(np.asarray(cells["cy"][sel], np.float64) / s)
            if np.any((ix == np.floor(cx / s)) & (iy == np.floor(cy / s))):
                worse += 1
                break
        checked += 1
    print(f"   {checked} points checked, {worse} returned a coarser cell "
          f"than one that existed   {'ok' if worse == 0 else 'FAIL'}")
    if worse:
        failures.append("finest-cell rule")

    # ── 4. objects beat the ground under them ────────────────
    print("\n4. clicking a tracked object selects the object")
    view = IM.build_view("top-follow", "World", snap.ego_xy, snap.heading)
    hw = IM.image_size_for(view)
    t = IM.transform_for(view, "World", snap.ego_xy, snap.heading, hw)
    tested = ok_n = 0
    for o in snap.objects:
        if o["points"] < 12:
            continue
        c = o["centroid_world"]
        px, py = t.world_to_screen(float(c[0]), float(c[1]))
        if not t.contains_display_px(px, py):
            continue
        sel = SEL.select_at(px, py, t, snap, 0)
        tested += 1
        same = sel.kind == "object" and sel.obj["id"] == o["id"]
        ok_n += int(same)
        print(f"   #{o['id']:<3} {o['state_name']:<24} -> {sel.kind:<7} "
              f"{'ok' if same else 'FAIL'}")
    if tested == 0:
        print("   no on-screen objects with >= 12 points in this frame")
    elif ok_n != tested:
        failures.append("object hit priority")

    print("\n" + "=" * 60)
    if failures:
        print("FAILED: " + "; ".join(failures))
        return 1
    print("all interaction claims hold on a real precomputed run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
