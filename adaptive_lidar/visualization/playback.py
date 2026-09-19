"""
Playback: pre-compute a scenario once, then scrub it instantly.

The rule this module exists to enforce is that **the pipeline runs once per
frame, ever**.  Camera movement, layer changes, frame-selector changes and
scrubbing all read from the cache; none of them re-runs perception or
re-touches the map.  Animating by re-running the pipeline per displayed frame
would make the replay rate look like a latency measurement, which it is not.

Three things are accumulated alongside the frames because the map itself
cannot answer them:

* **The swept corridor.** The map slides a window and evicts cells behind the
  vehicle, so "every cell ever resolved finely" is not recoverable from it.
  A separate raster records, per location, the FIRST frame at which it was
  resolved at level 0 or 1. One int16 array serves every frame — the corridor
  as of frame k is just ``raster >= 0 and raster <= k``.

* **Wall thickness.** A correctness check: after driving past a flat vertical
  surface for a whole run, accumulated observations of it should still be one
  cell thick. If the pose transform were wrong they would smear.

* **Trail length.** Imported from ``evaluation.metrics`` so the dashboard and
  ``scripts/test_temporal.py`` report the same number from the same code.

Nothing here reads ground truth. The trail metric needs to know where the
vehicle actually was, and that read lives in ``evaluation/metrics.py`` with
the rest of the evaluation code — ``tests/test_gt_label_isolation.py``
enforces the boundary, and it caught this module the first time round.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from adaptive_lidar.evaluation.metrics import (
    moving_object_path,
    trail_profile,
    vehicle_half_length,
)
from adaptive_lidar.pipeline.types import ResolutionLevel

#: Cells kept per frame, as a box around the ego. The camera can never show
#: more than this, and keeping the whole world per frame would put hundreds of
#: megabytes into the session.
KEEP_AHEAD, KEEP_BEHIND, KEEP_SIDE = 72.0, 36.0, 40.0

#: Resolution of the swept-corridor raster. Coarser than the finest cell on
#: purpose: the corridor is about WHERE detail was spent, not about the detail.
CORRIDOR_RES = 0.4

#: Levels that count as "resolved finely" for the corridor.
CORRIDOR_LEVELS = (0, 1)

#: Stored per cached frame, in the narrowest dtype that still draws
#: correctly. `size` is omitted because it is a pure function of `level`, and
#: the difference between 45 and 20 bytes per cell is the difference between
#: a 100 MB session and a 25 MB one once both MOS variants are cached.
CELL_FIELDS = ("cx", "cy", "level", "sem_class", "ground_z", "z_max",
               "n_points", "entropy", "occupancy_state", "dynamic_prob",
               "intensity_mean", "last_seen")

_CELL_DTYPE = {
    "cx": np.float32, "cy": np.float32, "level": np.int8,
    "sem_class": np.int8, "ground_z": np.float16, "z_max": np.float16,
    "n_points": np.uint16, "occupancy_state": np.int8, "last_seen": np.uint16,
}
#: These three are 0..1 and only ever feed a colour ramp, so a byte is plenty.
_CELL_U8 = ("entropy", "dynamic_prob", "intensity_mean")


def decode_cells(cells: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Undo the byte packing applied on the way into the cache."""
    out = dict(cells)
    for k in _CELL_U8:
        if k in out and out[k].dtype == np.uint8:
            out[k] = out[k].astype(np.float32) / 255.0
    for k in ("ground_z", "z_max"):
        if k in out:
            out[k] = out[k].astype(np.float32)
    return out


@dataclass
class FrameSnapshot:
    """Everything the renderer needs for one frame, and nothing more."""
    frame_id: int
    timestamp: float
    pose: np.ndarray
    heading: float
    cells: Dict[str, np.ndarray]
    objects: List[Dict[str, Any]]
    overlay: np.ndarray                 # moving points, world frame
    n_points: int
    telemetry: Dict[str, Any]

    @property
    def ego_xy(self) -> Tuple[float, float]:
        return float(self.pose[0, 3]), float(self.pose[1, 3])


@dataclass
class Corridor:
    """Per-location first frame at which it was resolved at level 0 or 1."""
    first_frame: np.ndarray             # (H, W) int16, -1 = never
    x0: float
    y0: float
    res: float

    def mask_upto(self, frame_idx: int) -> np.ndarray:
        return (self.first_frame >= 0) & (self.first_frame <= frame_idx)

    def area_upto(self, frame_idx: int) -> float:
        return float(self.mask_upto(frame_idx).sum()) * self.res * self.res

    def cells_upto(self, frame_idx: int) -> np.ndarray:
        """(K, 2) world xy of every corridor cell resolved by ``frame_idx``."""
        iy, ix = np.nonzero(self.mask_upto(frame_idx))
        return np.stack([self.x0 + (ix + 0.5) * self.res,
                         self.y0 + (iy + 0.5) * self.res], axis=1)


@dataclass
class PlaybackRun:
    scenario: str
    policy: str
    budget: float
    backend: str
    gate_enabled: bool
    frames: List[FrameSnapshot] = field(default_factory=list)
    corridor: Optional[Corridor] = None
    wall: Optional[Dict[str, Any]] = None
    trail: Optional[Dict[str, Any]] = None
    truth_path: List[Optional[np.ndarray]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def duration_s(self) -> float:
        return self.frames[-1].timestamp if self.frames else 0.0


# ════════════════════════════════════════════════════════════
def _clip_cells(a: Dict[str, np.ndarray], ego: Tuple[float, float]
                ) -> Dict[str, np.ndarray]:
    """Keep only the cells any camera could show this frame."""
    n = len(a["cx"])
    if n == 0:
        return {k: a[k][:0].copy() for k in CELL_FIELDS if k in a}
    keep = ((a["cx"] > ego[0] - KEEP_BEHIND) & (a["cx"] < ego[0] + KEEP_AHEAD)
            & (np.abs(a["cy"] - ego[1]) < KEEP_SIDE))
    out = {}
    for k in CELL_FIELDS:
        if k not in a:
            continue
        v = a[k][keep]
        if k in _CELL_U8:
            out[k] = np.clip(np.asarray(v, np.float32) * 255.0,
                             0, 255).astype(np.uint8)
        else:
            out[k] = v.astype(_CELL_DTYPE.get(k, v.dtype), copy=False)
    # `size` is reconstructed from `level` at draw time.
    out["size"] = ResolutionLevel.sizes_array()[
        np.clip(out["level"].astype(np.int64), 0, 4)]
    return out


def _objects(frame, pose) -> List[Dict[str, Any]]:
    R, t = pose[:3, :3], pose[:3, 3]
    out = []
    for i in (frame.instances or []):
        cw = R @ np.asarray(i.centroid, float) + t
        vw = R @ np.asarray(i.velocity if i.velocity is not None
                            else np.zeros(3), float)
        out.append({
            "id": int(i.instance_id),
            "cls": int(i.semantic_class),
            "state": int(i.state),
            "state_name": i.state_name,
            "speed_world": float(np.linalg.norm(vw)),
            "vel_world": vw.astype(np.float32),
            "centroid_world": cw.astype(np.float32),
            "extent": np.asarray(i.extent, np.float32),
            "points": int(i.point_count),
            "age": int(i.age),
        })
    return out


def _update_corridor(corridor_acc: Dict[Tuple[int, int], int],
                     cells: Dict[str, np.ndarray], frame_idx: int):
    """Record the first frame at which each location became finely resolved."""
    if len(cells.get("cx", ())) == 0:
        return
    fine = np.isin(cells["level"], CORRIDOR_LEVELS)
    if not fine.any():
        return
    ix = np.floor(cells["cx"][fine] / CORRIDOR_RES).astype(np.int64)
    iy = np.floor(cells["cy"][fine] / CORRIDOR_RES).astype(np.int64)
    for kx, ky in zip(ix.tolist(), iy.tolist()):
        corridor_acc.setdefault((kx, ky), frame_idx)


def _build_corridor(acc: Dict[Tuple[int, int], int]) -> Optional[Corridor]:
    if not acc:
        return None
    keys = np.array(list(acc.keys()), dtype=np.int64)
    vals = np.array(list(acc.values()), dtype=np.int16)
    x_lo, y_lo = keys[:, 0].min(), keys[:, 1].min()
    w = int(keys[:, 0].max() - x_lo) + 1
    h = int(keys[:, 1].max() - y_lo) + 1
    if w * h > 30_000_000:
        return None
    grid = np.full((h, w), -1, np.int16)
    grid[keys[:, 1] - y_lo, keys[:, 0] - x_lo] = vals
    return Corridor(grid, float(x_lo) * CORRIDOR_RES,
                    float(y_lo) * CORRIDOR_RES, CORRIDOR_RES)


# ════════════════════════════════════════════════════════════
# G — the correctness proof
# ════════════════════════════════════════════════════════════
def wall_thickness(amap, y_face: float, x_range: Tuple[float, float],
                   min_height: float = 1.5, search: float = 1.0
                   ) -> Dict[str, Any]:
    """How many cells thick a flat vertical surface is after a whole run.

    If the pose transform were wrong, accumulating the same wall over sixty
    metres of driving would smear it across several cells. Measuring the
    perpendicular spread of the cells that fall on it turns that into a
    number, and it costs one reduction.
    """
    a = amap.all_cells_arrays()
    if len(a["cx"]) == 0:
        return {"ok": False, "reason": "no cells"}
    tall = (a["z_max"] - a["ground_z"]) > min_height
    band = (tall & (np.abs(a["cy"] - y_face) < search)
            & (a["cx"] > x_range[0]) & (a["cx"] < x_range[1]))
    n = int(band.sum())
    if n < 20:
        return {"ok": False, "reason": f"only {n} cells on the face"}

    y = a["cy"][band]
    sizes = a["size"][band]
    median_cell = float(np.median(sizes))

    # Two spreads, because they answer different questions. The 5-95 range
    # includes the tail of cells seen at grazing incidence, where a 2 cm range
    # error maps to a large LATERAL error and the facade legitimately looks
    # thicker. The interquartile range describes the bulk of the wall, which
    # is what "has the pose transform smeared it?" is actually asking.
    lo, hi = np.percentile(y, [5, 95])
    q1, q3 = np.percentile(y, [25, 75])
    spread = float(hi - lo)
    iqr = float(q3 - q1)
    return {
        "ok": True,
        "n_cells": n,
        "spread_m": spread,
        "iqr_m": iqr,
        "median_cell_m": median_cell,
        "thickness_cells": spread / max(median_cell, 1e-6),
        "thickness_cells_iqr": iqr / max(median_cell, 1e-6),
        "std_m": float(np.std(y)),
        "y_face": y_face,
        "x_range": x_range,
    }


#: Where the flat building face is in each scenario, and over what stretch
#: of it to measure. Chosen once so the readout is reproducible.
#: The buildings are 6 m deep boxes, so their centre line is 3 m behind the
#: surface the beams actually hit. Probing the centre found four cells;
#: probing the near face finds the wall.
#: (y of the near face, x stretch to measure over). The x stretch avoids
#: tree canopies, which overhang the pavement and would otherwise be measured
#: as part of the wall — the probe has to look at the wall, not near it.
WALL_PROBE = {
    "mixed_urban": (8.5, (0.0, 22.0)),     # facade between the trees at x=26+
    "convoy": (10.0, (-10.0, 80.0)),       # unbroken facade, no trees
}


# ════════════════════════════════════════════════════════════
def precompute(config, loader_frames, scenario: str, policy: str,
               budget: float, backend: str, gate_enabled: bool,
               overlay_stride: int = 3) -> PlaybackRun:
    """Run the pipeline ONCE over the scenario and cache every frame."""
    import copy

    from adaptive_lidar.pipeline.pipeline import Pipeline

    cfg = copy.deepcopy(config)
    cfg.setdefault("motion", {})["gate_enabled"] = gate_enabled

    pipe = Pipeline(cfg)
    pipe.build_stages(backend=backend, policy=policy)
    pipe.set_budget(budget)

    run = PlaybackRun(scenario, policy, budget, backend, gate_enabled)
    corridor_acc: Dict[Tuple[int, int], int] = {}

    for k, f in enumerate(loader_frames):
        fr = pipe.run(cloud_np=f, frame_id=f["frame_id"],
                      timestamp=f["timestamp"])
        if k == 0:
            pipe.context.cumulative_timing.clear()   # warm-up, as everywhere

        pose = fr.pose if fr.pose is not None else np.eye(4)
        ego = (float(pose[0, 3]), float(pose[1, 3]))
        a = pipe.context.amap.all_cells_arrays()

        _update_corridor(corridor_acc, a, k)

        overlay = np.zeros((0, 3), np.float32)
        ov = pipe.context.dynamic_overlay
        if ov is not None and len(ov):
            overlay = np.asarray(ov.points[::overlay_stride], np.float32)

        run.frames.append(FrameSnapshot(
            frame_id=int(fr.frame_id),
            timestamp=float(fr.timestamp),
            pose=np.asarray(pose, np.float64).copy(),
            heading=float(np.arctan2(pose[1, 0], pose[0, 0])),
            cells=_clip_cells(a, ego),
            objects=_objects(fr, pose),
            overlay=overlay,
            n_points=int(len(fr.points)),
            telemetry=dict(fr.timing.get("telemetry", {})),
        ))

    run.corridor = _build_corridor(corridor_acc)

    # ── the correctness readout ──────────────────────────────
    probe = WALL_PROBE.get(scenario)
    if probe is not None:
        run.wall = wall_thickness(pipe.context.amap, probe[0], probe[1])

    # ── the trail, from the same code the Phase 6 test uses ──
    truth = moving_object_path(loader_frames)
    run.truth_path = truth
    pts = [p for p in truth if p is not None]
    if len(pts) >= 2:
        rear = vehicle_half_length(loader_frames)
        tl, n_cells, max_b, bins = trail_profile(
            pipe.context.amap, truth, truth[-1], rear_m=rear)
        run.trail = {"trail_m": tl, "cells": n_cells, "max_behind_m": max_b,
                     "bins": bins, "rear_m": rear,
                     "travelled_m": float(np.linalg.norm(pts[-1] - pts[0]))}
    return run
