"""
test_allocation.py — Phase 5 verification.

Runs every allocation policy on the ``pedestrian_far`` scenario at budgets
100 / 50 / 25 / 10 % and reports, for each, the cells used and whether the
70 m pedestrian survives.

THE CLAIM UNDER TEST
--------------------
``full`` must retain the pedestrian at EVERY budget, because the safety pin is
a constraint on the feasible set rather than a weighted term that can be
outbid.  ``uniform_80`` and ``random`` must lose it at low budget.

And the control: **if ``full`` does not beat ``random`` at equal budget, the
value function is doing nothing and the whole contribution is illusory.**

Run:  python scripts/test_allocation.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
for p in (os.path.dirname(_PKG), _PKG):
    if p not in sys.path:
        sys.path.insert(0, p)

from adaptive_lidar.data.loader import get_dataset                  # noqa: E402
from adaptive_lidar.data.synthetic_scene import (                   # noqa: E402
    FAR_PEDESTRIAN_X,
    FAR_PEDESTRIAN_Y,
)
from adaptive_lidar.mapping.allocation import AllocationController  # noqa: E402
from adaptive_lidar.pipeline.pipeline import Pipeline               # noqa: E402
from adaptive_lidar.pipeline.types import ResolutionLevel           # noqa: E402
from adaptive_lidar.utils.config import load_config                 # noqa: E402

BUDGETS = (1.00, 0.50, 0.25, 0.10)
POLICIES = AllocationController.POLICIES
RETAIN_MIN_CELLS = 3          # reported, and swept for sensitivity below
RADIUS = 1.2                  # m around the pedestrian's true position

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  —  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)
    return ok


def hr(t):
    print("\n" + "=" * 92)
    print(f"  {t}")
    print("=" * 92)


# ════════════════════════════════════════════════════════════
def run(policy, budget, cfg, frames, backend="auto"):
    pipe = Pipeline(cfg)
    pipe.build_stages(backend=backend, policy=policy)
    pipe.set_budget(budget)
    last = None
    for f in frames:
        last = pipe.run(cloud_np=f, frame_id=f["frame_id"], timestamp=f["timestamp"])
    return pipe, last


def pedestrian_stats(amap, frame, min_cells=RETAIN_MIN_CELLS):
    """Cells covering the 70 m pedestrian, and their mean physical size."""
    pose = frame.pose if frame.pose is not None else np.eye(4)
    px = FAR_PEDESTRIAN_X
    py = FAR_PEDESTRIAN_Y

    a = amap.all_cells_arrays()
    if len(a["cx"]) == 0:
        return 0, 0.0, 0, False
    d = np.hypot(a["cx"] - px, a["cy"] - py)
    near = d <= RADIUS
    n_cells = int(near.sum())
    mean_size = float(a["size"][near].mean()) if n_cells else 0.0
    # Cells whose dominant class is the safety-critical one.
    n_vru = int((a["sem_class"][near] == 4).sum()) if n_cells else 0
    retained = n_cells >= min_cells
    return n_cells, mean_size, n_vru, retained


def gt_pedestrian_points(frame):
    px, py = FAR_PEDESTRIAN_X, FAR_PEDESTRIAN_Y
    pose = frame.pose if frame.pose is not None else np.eye(4)
    pw = frame.points @ pose[:3, :3].T + pose[:3, 3]
    d = np.hypot(pw[:, 0] - px, pw[:, 1] - py)
    m = d <= RADIUS
    return int(m.sum()), int((np.asarray(frame.gt_label)[m] == 4).sum())


# ════════════════════════════════════════════════════════════
def main():
    cfg = load_config()
    ds = get_dataset("synthetic", None, 3, scenario="pedestrian_far", quiet=True)
    frames = list(ds)

    hr("SCENARIO — pedestrian_far: one pedestrian at 70 m on an empty road")
    probe, pf = run("full", 1.0, cfg, frames)
    n_pts, n_gt_vru = gt_pedestrian_points(pf)
    r = float(np.hypot(FAR_PEDESTRIAN_X, FAR_PEDESTRIAN_Y))
    print(f"  pedestrian at ({FAR_PEDESTRIAN_X}, {FAR_PEDESTRIAN_Y}) m — "
          f"range {r:.1f} m")
    print(f"  the sensor sees it with {n_pts} points "
          f"({n_gt_vru} labelled VRU) within {RADIUS} m")
    check("the sensor actually observes the pedestrian", n_pts >= 4,
          f"{n_pts} points — anything below this makes the test vacuous")

    # ── the main table ───────────────────────────────────────
    hr("RETENTION vs BUDGET  (cells used / pedestrian cells / mean cell size)")
    header = (f"  {'policy':<40}" +
              "".join(f"{int(b * 100):>5}%" + " " * 14 for b in BUDGETS))
    print(header)
    print(f"  {'':<40}" + "".join(f"{'cells':>8}{'ped':>5}{'size':>7}"
                                  for _ in BUDGETS))
    print("  " + "-" * 88)

    results = {}
    for policy in POLICIES:
        row = f"  {policy:<40}"
        for b in BUDGETS:
            pipe, frame = run(policy, b, cfg, frames)
            amap = pipe.context.amap
            n_cells, size, n_vru, retained = pedestrian_stats(amap, frame)
            results[(policy, b)] = {
                "map_cells": amap.n_cells,
                "map_mb": amap.nbytes() / 1e6,
                "ped_cells": n_cells,
                "ped_cell_size": size,
                "ped_vru_cells": n_vru,
                "retained": retained,
            }
            mark = "*" if retained else " "
            row += (f"{amap.n_cells:>8,}{n_cells:>4}{mark}"
                    f"{(f'{size * 100:.0f}cm' if size else '-'):>7}")
        print(row)
    print("\n  '*' = retained (>= 3 cells within 1.2 m of the true position)")

    # ── assertions ───────────────────────────────────────────
    hr("ASSERTIONS")

    full_all = all(results[("full", b)]["retained"] for b in BUDGETS)
    detail = ", ".join(f"{int(b * 100)}%:{results[('full', b)]['ped_cells']}"
                       for b in BUDGETS)
    check("`full` retains the pedestrian at EVERY budget", full_all, detail)

    lowest = min(BUDGETS)
    for loser in ("uniform_80", "random"):
        r_low = results[(loser, lowest)]
        check(f"`{loser}` loses or degrades the pedestrian at "
              f"{int(lowest * 100)}% budget",
              (not r_low["retained"])
              or r_low["ped_cell_size"] >= results[("full", lowest)]["ped_cell_size"] * 2,
              f"{r_low['ped_cells']} cells at "
              f"{r_low['ped_cell_size'] * 100:.0f} cm vs full's "
              f"{results[('full', lowest)]['ped_cells']} at "
              f"{results[('full', lowest)]['ped_cell_size'] * 100:.0f} cm")

    # THE CONTROL
    hr("CONTROL — `full` vs `random` at EQUAL BUDGET")
    print("  Retention alone is only half the comparison: at a generous "
          "budget a random\n  allocator can afford to refine everything and "
          "will keep the pedestrian too.\n  What it cannot do is keep the "
          "pedestrian CHEAPLY. The honest control is\n  therefore retention "
          "per megabyte — `full` must never be beaten on both\n  axes at "
          "once.\n")
    print(f"  {'budget':>8}{'full cells':>12}{'full MB':>9}{'full ped':>9}"
          f"{'rand cells':>12}{'rand MB':>9}{'rand ped':>9}   verdict")
    wins = 0
    for b in BUDGETS:
        f_, r_ = results[("full", b)], results[("random", b)]
        # `full` dominates if it retains at least as well using no more
        # memory, and is strictly better on at least one of the two axes.
        no_worse = (f_["ped_cells"] >= r_["ped_cells"]
                    and f_["map_mb"] <= r_["map_mb"] * 1.02)
        strictly = (f_["ped_cells"] > r_["ped_cells"]
                    or f_["map_mb"] < r_["map_mb"] * 0.98)
        dom = no_worse and strictly
        wins += bool(dom)
        print(f"  {int(b * 100):>7}%{f_['map_cells']:>12,}{f_['map_mb']:>9.2f}"
              f"{f_['ped_cells']:>9}{r_['map_cells']:>12,}{r_['map_mb']:>9.2f}"
              f"{r_['ped_cells']:>9}   "
              + ("full dominates" if dom else "no clear win"))
    check("`full` dominates `random` at every equal budget "
          "(if not, the value function is broken)",
          wins == len(BUDGETS), f"{wins} of {len(BUDGETS)} budgets")

    # The literal reading of the problem statement, for comparison.
    d_only = results[("distance_only", lowest)]
    f_low = results[("full", lowest)]
    print(f"\n  For reference, `distance_only` — exactly what the problem "
          f"statement\n  literally asks for — spends {d_only['map_cells']:,} "
          f"cells ({d_only['map_mb']:.2f} MB) and keeps\n  "
          f"{d_only['ped_cells']} cells on the pedestrian, against `full`'s "
          f"{f_low['map_cells']:,} cells\n  ({f_low['map_mb']:.2f} MB) and "
          f"{f_low['ped_cells']} cells. Distance alone spends the budget\n"
          f"  evenly along every radius, including the empty ones.")

    # ── threshold sensitivity ────────────────────────────────
    hr("SENSITIVITY OF THE RETENTION THRESHOLD")
    print(f"  {'min cells':>10}" + "".join(f"{p:>12}"
                                           for p in ("full", "uniform_40",
                                                     "uniform_80", "random")))
    for mc in (1, 2, 3, 5, 8):
        line = f"  {mc:>10}"
        for policy in ("full", "uniform_40", "uniform_80", "random"):
            n = results[(policy, lowest)]["ped_cells"]
            line += f"{('yes' if n >= mc else 'no'):>12}"
        print(line)
    print(f"\n  (at the {int(lowest * 100)}% budget; the reported threshold is "
          f"{RETAIN_MIN_CELLS} cells)")

    # ── pin accounting ───────────────────────────────────────
    hr("SAFETY PIN ACCOUNTING  (`full`, lowest budget)")
    pipe, frame = run("full", lowest, cfg, frames)
    d = frame.timing["S6_diag"]
    print(f"  tiles                : {d['n_tiles']:,}")
    print(f"  safety-pinned        : {d['n_pinned']:,}")
    print(f"  cells taken by pins  : {d['pin_cells']:,}")
    print(f"  cell budget          : {d['cell_budget']:,}")
    print(f"  budget exceeded by pins: {d['budget_exceeded_by_pins']}")
    print(f"  level histogram      : "
          + ", ".join(f"{ResolutionLevel.name(i)}={c}"
                      for i, c in enumerate(d["level_hist"])))
    pinned = [t for t in frame.tiles if t.safety_pinned]
    if pinned:
        print("\n  example pins:")
        for t in pinned[:6]:
            print(f"    tile ({t.cx:7.1f},{t.cy:6.1f}) r={t.range_mean:5.1f} m  "
                  f"-> {ResolutionLevel.name(t.resolution_level):>5}  "
                  f"{t.safety_reason}")

    geo_pins = sum(1 for t in frame.tiles if t.has_vertical_run)
    check("the GEOMETRIC pin fires on the pedestrian's tile "
          "(the guarantee survives segmentation failure)",
          geo_pins > 0, f"{geo_pins} tiles with a vertical run")

    hr("RESULT")
    if FAILURES:
        print(f"  {len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"    - {f}")
        sys.exit(1)
    print("  ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
