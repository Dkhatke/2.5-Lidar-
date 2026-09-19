"""
Metrics.

Every number the project reports comes from here, and every one is measured
rather than derived from a cell count or a nominal byte budget.

Three choices worth stating:

* **Signed bias alongside RMSE.**  A planner can absorb variance; a systematic
  10 cm underestimate of a kerb height is what drives it into the kerb.  Bias
  is rarely reported and is the more dangerous of the two.

* **p99 alongside p50 and p95.**  For a real-time system the tail is the
  requirement — the mean hides exactly the frames that would drop.

* **tracemalloc, never cells x bytes.**  Cell-count arithmetic omits array
  overhead, per-level padding and key storage, all of which are real memory.
"""
from __future__ import annotations

import tracemalloc
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from adaptive_lidar.pipeline.types import NUM_CLASSES, Occupancy

#: The PS asks for "accuracy across varying distances", which means these.
RANGE_BANDS: Tuple[Tuple[float, float], ...] = (
    (0.0, 10.0), (10.0, 30.0), (30.0, 60.0), (60.0, 100.0))
BAND_NAMES = tuple(f"{int(a)}-{int(b)}m" for a, b in RANGE_BANDS)


def band_of(r: np.ndarray) -> np.ndarray:
    """(N,) index into RANGE_BANDS, -1 outside them all."""
    out = np.full(len(r), -1, np.int8)
    for i, (lo, hi) in enumerate(RANGE_BANDS):
        out[(r >= lo) & (r < hi)] = i
    return out


# ════════════════════════════════════════════════════════════
# Semantic accuracy, stratified by range  (M6)
# ════════════════════════════════════════════════════════════
def iou_from_counts(inter: np.ndarray, union: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, inter / np.maximum(union, 1), np.nan)


def range_stratified_iou(
    pred: np.ndarray,
    true: np.ndarray,
    ranges: np.ndarray,
) -> Dict[str, Any]:
    """Per-class IoU and mIoU, overall and per range band.

    Points with ``true < 0`` (unlabelled / outside the taxonomy) are excluded
    from every denominator — defaulting them to a real class would inflate
    that class's score with points nobody annotated.
    """
    pred = np.asarray(pred).astype(np.int64)
    true = np.asarray(true).astype(np.int64)
    valid = true >= 0
    bands = band_of(np.asarray(ranges))

    def block(mask) -> Dict[str, Any]:
        p, t = pred[mask], true[mask]
        inter = np.zeros(NUM_CLASSES)
        union = np.zeros(NUM_CLASSES)
        support = np.zeros(NUM_CLASSES, np.int64)
        for c in range(NUM_CLASSES):
            pc, tc = p == c, t == c
            inter[c] = (pc & tc).sum()
            union[c] = (pc | tc).sum()
            support[c] = tc.sum()
        iou = iou_from_counts(inter, union)
        seen = support > 0
        return {
            "iou": iou.tolist(),
            "support": support.tolist(),
            "miou": float(np.nanmean(iou[seen])) if seen.any() else float("nan"),
            "accuracy": float((p == t).mean()) if len(p) else float("nan"),
            "n": int(len(p)),
        }

    out = {"overall": block(valid), "bands": {}}
    for i, name in enumerate(BAND_NAMES):
        m = valid & (bands == i)
        out["bands"][name] = block(m) if m.any() else None
    return out


# ════════════════════════════════════════════════════════════
# Elevation accuracy against the reference map
# ════════════════════════════════════════════════════════════
def elevation_error(amap, reference, ego_xy=(0.0, 0.0)) -> Dict[str, Any]:
    """Elevation RMSE and SIGNED BIAS, by range band and by resolution level."""
    a = amap.all_cells_arrays()
    if len(a["cx"]) == 0 or len(reference) == 0:
        return {"overall": None, "bands": {}, "levels": {}}

    rows = reference.lookup(a["cx"], a["cy"])
    hit = rows >= 0
    if not hit.any():
        return {"overall": None, "bands": {}, "levels": {}}

    err = a["ground_z"][hit] - reference.ground_z[rows[hit]]
    r = np.hypot(a["cx"][hit] - ego_xy[0], a["cy"][hit] - ego_xy[1])
    lvl = a["level"][hit]

    def stats(m) -> Optional[Dict[str, float]]:
        if not np.any(m):
            return None
        e = err[m]
        return {
            "n": int(m.sum()),
            "rmse_m": float(np.sqrt(np.mean(e ** 2))),
            "bias_m": float(np.mean(e)),
            "mae_m": float(np.mean(np.abs(e))),
            "p95_abs_m": float(np.percentile(np.abs(e), 95)),
        }

    bands = band_of(r)
    return {
        "overall": stats(np.ones(len(err), bool)),
        "coverage": float(hit.mean()),
        "bands": {n: stats(bands == i) for i, n in enumerate(BAND_NAMES)},
        "levels": {int(l): stats(lvl == l) for l in np.unique(lvl)},
    }


def map_completeness(amap, reference) -> Dict[str, float]:
    """How much of the reference the map covers, and how much it invents."""
    a = amap.all_cells_arrays()
    if len(reference) == 0:
        return {"completeness": float("nan"), "spurious_rate": float("nan")}
    if len(a["cx"]) == 0:
        return {"completeness": 0.0, "spurious_rate": 0.0}

    # Completeness: reference cells covered by SOME map cell, at that cell's
    # own size — a coarse cell legitimately covers many reference cells.
    rx, ry = reference.centres()
    covered = np.zeros(len(reference), bool)
    for lvl, store in enumerate(amap.levels):
        if len(store) == 0:
            continue
        size = store.size
        from adaptive_lidar.utils.grouping import morton_2d
        ix = np.floor(rx / size).astype(np.int64)
        iy = np.floor(ry / size).astype(np.int64)
        code = morton_2d(ix, iy, level=lvl)
        pos = np.clip(np.searchsorted(store.keys, code), 0, len(store.keys) - 1)
        covered |= store.keys[pos] == code

    rows = reference.lookup(a["cx"], a["cy"])
    # A map cell is spurious if nothing in the reference lies under it.
    spurious = rows < 0
    return {
        "completeness": float(covered.mean()),
        "spurious_rate": float(spurious.mean()),
        "n_reference_cells": int(len(reference)),
        "n_map_cells": int(len(a["cx"])),
    }


# ════════════════════════════════════════════════════════════
# Object Retention Rate  (the headline safety metric)
# ════════════════════════════════════════════════════════════
def object_retention_rate(
    amap,
    frames: Sequence[Dict[str, Any]],
    min_cells: int = 3,
    min_points: int = 10,
    radius_pad: float = 0.4,
) -> Dict[str, Any]:
    """ORR per class.

        ORR_c(B) = (# GT objects of class c represented by >= min_cells cells
                    whose dominant class is c)
                 / (# GT objects of class c OBSERVED with >= min_points points)

    Note the denominator: only objects the sensor actually saw.  Including
    objects hidden behind a wall would make this a measure of occlusion rather
    than of allocation.
    """
    a = amap.all_cells_arrays()
    objects: Dict[int, Dict[str, Any]] = {}

    for f in frames:
        gi = f.get("gt_instance")
        gl = f.get("gt_label")
        if gi is None or gl is None:
            continue
        pose = f.get("pose")
        pose = np.eye(4) if pose is None else np.asarray(pose)
        pw = np.asarray(f["points"])[:, :3] @ pose[:3, :3].T + pose[:3, 3]
        gi = np.asarray(gi)
        gl = np.asarray(gl)
        for oid in np.unique(gi):
            if oid <= 0:
                continue
            m = gi == oid
            cls = int(np.bincount(np.clip(gl[m], 0, NUM_CLASSES - 1),
                                  minlength=NUM_CLASSES).argmax())
            rec = objects.setdefault(int(oid), {
                "cls": cls, "n_points": 0,
                "lo": pw[m][:, :2].min(axis=0), "hi": pw[m][:, :2].max(axis=0)})
            rec["n_points"] += int(m.sum())
            rec["lo"] = np.minimum(rec["lo"], pw[m][:, :2].min(axis=0))
            rec["hi"] = np.maximum(rec["hi"], pw[m][:, :2].max(axis=0))

    per_class = {c: {"observed": 0, "retained": 0} for c in range(NUM_CLASSES)}
    details = []
    cx, cy, scls = a["cx"], a["cy"], a["sem_class"]

    for oid, rec in objects.items():
        if rec["n_points"] < min_points:
            continue
        c = rec["cls"]
        per_class[c]["observed"] += 1
        lo = rec["lo"] - radius_pad
        hi = rec["hi"] + radius_pad
        inside = ((cx >= lo[0]) & (cx <= hi[0]) & (cy >= lo[1]) & (cy <= hi[1]))
        n_in = int(inside.sum())
        n_right = int((scls[inside] == c).sum()) if n_in else 0
        ok = n_right >= min_cells
        per_class[c]["retained"] += int(ok)
        details.append({"id": oid, "cls": c, "points": rec["n_points"],
                        "cells": n_in, "cells_correct_class": n_right,
                        "retained": ok})

    out = {"per_class": {}, "thresholds": {"min_cells": min_cells,
                                           "min_points": min_points},
           "objects": details}
    for c in range(NUM_CLASSES):
        o, r = per_class[c]["observed"], per_class[c]["retained"]
        out["per_class"][c] = {
            "observed": o, "retained": r,
            "orr": (r / o) if o else float("nan")}
    seen = [v["orr"] for v in out["per_class"].values() if v["observed"]]
    out["mean_orr"] = float(np.mean(seen)) if seen else float("nan")
    return out


# ════════════════════════════════════════════════════════════
# Boundary consistency  (M4 — the alignment artefact test)
# ════════════════════════════════════════════════════════════
def boundary_consistency(amap) -> Dict[str, Any]:
    """Ground-height discrepancy across LEVEL-TRANSITION edges vs interior ones.

    If cells of different sizes meet cleanly, the elevation step across a
    transition edge looks like the step across an ordinary interior edge.  If
    the two distributions differ, the hierarchy is producing a seam — which is
    exactly the "alignment error" the problem statement warns about, observed
    at map scale rather than argued about.
    """
    a = amap.all_cells_arrays()
    n = len(a["cx"])
    if n < 10:
        return {"n_transition": 0, "n_interior": 0}

    # Rasterise onto a common coarse lattice so cells of different sizes can
    # be compared to their neighbours at all.
    res = 0.8
    ix = np.floor(a["cx"] / res).astype(np.int64)
    iy = np.floor(a["cy"] / res).astype(np.int64)
    x0, y0 = ix.min(), iy.min()
    w, h = int(ix.max() - x0) + 1, int(iy.max() - y0) + 1
    if w * h > 8_000_000:
        return {"n_transition": 0, "n_interior": 0}

    gz = np.full((h, w), np.nan, np.float32)
    lv = np.full((h, w), -1, np.int8)
    gz[iy - y0, ix - x0] = a["ground_z"]
    lv[iy - y0, ix - x0] = a["level"]

    diffs_t, diffs_i = [], []
    for dy, dx in ((0, 1), (1, 0)):
        g1 = gz[:h - dy, :w - dx]
        g2 = gz[dy:, dx:]
        l1 = lv[:h - dy, :w - dx]
        l2 = lv[dy:, dx:]
        ok = np.isfinite(g1) & np.isfinite(g2) & (l1 >= 0) & (l2 >= 0)
        d = np.abs(g1 - g2)
        trans = ok & (l1 != l2)
        inter = ok & (l1 == l2)
        diffs_t.append(d[trans])
        diffs_i.append(d[inter])

    t = np.concatenate(diffs_t) if diffs_t else np.zeros(0)
    i = np.concatenate(diffs_i) if diffs_i else np.zeros(0)

    def d(v):
        if v.size == 0:
            return None
        return {"n": int(v.size), "mean_m": float(v.mean()),
                "p50_m": float(np.percentile(v, 50)),
                "p95_m": float(np.percentile(v, 95))}

    res_t, res_i = d(t), d(i)
    ratio = (res_t["p95_m"] / res_i["p95_m"]
             if res_t and res_i and res_i["p95_m"] > 1e-9 else float("nan"))
    return {"transition": res_t, "interior": res_i,
            "p95_ratio": ratio,
            "n_transition": int(t.size), "n_interior": int(i.size)}


# ════════════════════════════════════════════════════════════
# Trail length — the MOS gate ablation
# ════════════════════════════════════════════════════════════
#: A trail is a phantom WALL, so a bin counts as trailing only if it holds at
#: least this many spurious cells. One isolated cell 8 m back is a speck, not a
#: wall, and a metric that lets a single straggler define the answer measures
#: the worst outlier rather than the artefact.
BIN_M = 0.5
MIN_CELLS_PER_BIN = 3


def moving_object_path(loader_frames) -> List[Optional[np.ndarray]]:
    out = []
    for f in loader_frames:
        m = np.asarray(f.get("gt_moving")) if f.get("gt_moving") is not None else None
        if m is None or not m.any():
            out.append(None)
            continue
        pose = f.get("pose")
        pose = np.eye(4) if pose is None else np.asarray(pose)
        pw = np.asarray(f["points"])[:, :3] @ pose[:3, :3].T + pose[:3, 3]
        out.append(pw[m][:, :2].mean(axis=0))
    return out


def vehicle_half_length(frames) -> float:
    """Half the largest moving OBJECT's extent, not of all movers together.

    Taking the bounding box of every moving point at once is wrong the moment
    a scene has two of them: in `convoy` an oncoming car and one travelling
    alongside are fifty metres apart, and their combined box reported a 26 m
    "half length". Grouping by ground-truth instance first keeps it a property
    of an object.
    """
    best = 0.0
    for f in frames:
        m = f.get("gt_moving")
        if m is None:
            continue
        m = np.asarray(m)
        if m.sum() < 20:
            continue
        pts = np.asarray(f["points"])[m][:, :2]
        inst = f.get("gt_instance")
        if inst is None:
            best = max(best, float(np.ptp(pts, axis=0).max()))
            continue
        ids = np.asarray(inst)[m]
        for oid in np.unique(ids):
            sel = ids == oid
            if sel.sum() < 20:
                continue
            best = max(best, float(np.ptp(pts[sel], axis=0).max()))
    return max(best / 2.0, 1.0)


def trail_profile(amap, truth_path, current_xy, corridor=2.5, rear_m=2.2):
    """Spurious-obstacle profile behind the vehicle's true position.

    Returns (trail_m, n_cells, max_behind_m, bins) where

      trail_m      how far back the CONTIGUOUS phantom extends, in metres:
                   walking backwards from the vehicle in 0.5 m bins, the
                   distance at which the first bin holding fewer than
                   MIN_CELLS_PER_BIN spurious cells is reached.
      n_cells      every spurious cell behind the vehicle, contiguous or not.
      max_behind_m the furthest single spurious cell — reported alongside so
                   the contiguity rule cannot hide a long tail.

    A cell is spurious if the map still ASSERTS something standing above the
    ground there. Cells the map has already retracted by free-space carving
    are not phantoms; they are the system correcting itself.
    """
    from adaptive_lidar.pipeline.types import Occupancy

    a = amap.all_cells_arrays()
    if len(a["cx"]) == 0 or current_xy is None:
        return 0.0, 0, 0.0, []

    obst = (((a["z_max"] - a["ground_z"]) > 0.35)
            & (a["occupancy_state"] != Occupancy.FREE))
    if not obst.any():
        return 0.0, 0, 0.0, []

    pts = np.array([p for p in truth_path if p is not None])
    if len(pts) < 2:
        return 0.0, 0, 0.0, []

    cx, cy = a["cx"][obst], a["cy"][obst]
    d = np.min(np.hypot(cx[:, None] - pts[None, :, 0],
                        cy[:, None] - pts[None, :, 1]), axis=1)
    on_path = d <= corridor

    travel = pts[-1] - pts[0]
    L = float(np.linalg.norm(travel))
    if L < 1e-6:
        return 0.0, 0, 0.0, []
    u = travel / L
    s_cell = (cx[on_path] - pts[0][0]) * u[0] + (cy[on_path] - pts[0][1]) * u[1]
    s_now = (current_xy[0] - pts[0][0]) * u[0] + (current_xy[1] - pts[0][1]) * u[1]

    # "Behind the vehicle" means behind its REAR EXTENT, not behind its
    # centroid. The exclusion is therefore half the vehicle's true length,
    # measured from the ground truth rather than assumed: with a hardcoded
    # 1.5 m against a 4.4 m car, the car's own rear half was being counted as
    # its trail.
    behind_m = s_now - s_cell
    keep = behind_m > rear_m
    behind_m = behind_m[keep]
    if behind_m.size == 0:
        return 0.0, 0, 0.0, []

    n_bins = int(np.ceil(behind_m.max() / BIN_M)) + 1
    counts = np.bincount((behind_m / BIN_M).astype(np.int64), minlength=n_bins)

    # The trail is the EXTENT of the phantom: from the nearest qualifying bin
    # to the furthest one. Requiring the run to start immediately behind the
    # vehicle would report zero whenever there is a gap between the car and the
    # wall it left, which is exactly the case where the phantom is worst.
    lo = int(rear_m / BIN_M)
    qual = np.flatnonzero(counts[lo:] >= MIN_CELLS_PER_BIN)
    trail = 0.0 if qual.size == 0 else float((qual.max() - qual.min() + 1) * BIN_M)
    return trail, int(behind_m.size), float(behind_m.max()), counts.tolist()


# ════════════════════════════════════════════════════════════
# Memory and latency
# ════════════════════════════════════════════════════════════
def measure_memory(build_fn) -> Tuple[Any, int]:
    """Run ``build_fn`` under tracemalloc and return (result, bytes)."""
    tracemalloc.start()
    base = tracemalloc.take_snapshot()
    result = build_fn()
    snap = tracemalloc.take_snapshot()
    used = sum(s.size_diff for s in snap.compare_to(base, "lineno"))
    tracemalloc.stop()
    return result, int(used)


def latency_percentiles(ctx, stages=("S0", "S1", "S2", "S3", "S4",
                                     "S5", "S6", "S7", "S8", "S9")
                        ) -> Dict[str, Dict[str, float]]:
    """p50 / p95 / p99 per stage, plus the total of the p50s.

    p99 is the one that matters for a real-time claim: the mean hides exactly
    the frames that would miss their deadline.
    """
    out: Dict[str, Dict[str, float]] = {}
    for s in stages:
        vals = ctx.cumulative_timing.get(s)
        if not vals:
            continue
        v = np.asarray(vals, float)
        out[s] = {"p50": float(np.percentile(v, 50)),
                  "p95": float(np.percentile(v, 95)),
                  "p99": float(np.percentile(v, 99)),
                  "mean": float(v.mean()), "max": float(v.max()),
                  "n": int(v.size)}
    tot50 = sum(d["p50"] for d in out.values())
    tot99 = sum(d["p99"] for d in out.values())
    out["TOTAL"] = {"p50": tot50, "p95": sum(d["p95"] for d in out.values()),
                    "p99": tot99, "mean": sum(d["mean"] for d in out.values()),
                    "max": sum(d["max"] for d in out.values()),
                    "n": max((d["n"] for d in out.values()), default=0)}
    out["TOTAL"]["fps_p50"] = 1000.0 / tot50 if tot50 else 0.0
    out["TOTAL"]["fps_p99"] = 1000.0 / tot99 if tot99 else 0.0
    return out
