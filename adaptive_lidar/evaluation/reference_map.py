"""
The reference map — the closest thing to ground truth a 2.5D map can have.

No dataset ships ground-truth 2.5D maps, so one is constructed:

  1. accumulate N consecutive frames using their poses,
  2. remove every point the ground truth marks as moving,
  3. build a UNIFORM 5 cm map from the accumulation.

LIMITATIONS — read these before quoting any number against it
-------------------------------------------------------------
* **Pose drift.**  On synthetic data the poses are exact, so the reference is
  exact.  On SemanticKITTI the poses come from a SLAM solution and accumulate
  drift, which smears surfaces by a few centimetres over 50 frames — the same
  order as the finest cell.  Elevation RMSE against this reference therefore
  has a floor set by the pose, not by the map.

* **Residual dynamics.**  Step 2 removes points whose *label* says moving.
  A car that is stationary for these 50 frames and drives away later is in the
  reference as permanent structure; a car that moves only slightly may keep
  points that the label calls static.

* **Occlusion gaps.**  A surface no beam ever reached is absent from the
  reference, so "the map has no cell here" cannot be distinguished from "the
  reference has no cell here either".  Completeness is therefore reported only
  over cells the reference actually has.

* **It is a 5 cm map, not the world.**  Comparing a 5 cm adaptive cell against
  it is fair; comparing an 80 cm cell against it charges the coarse cell for
  the quantisation the budget explicitly asked for.  That is the intended
  reading — it is what "accuracy across varying distances" measures — but it
  should not be mistaken for sensor error.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from adaptive_lidar.mapping.adaptive_map import BASE, AdaptiveMap
from adaptive_lidar.pipeline.types import NUM_CLASSES
from adaptive_lidar.utils.grouping import (
    group_by_key,
    group_sizes,
    morton_2d,
    morton_2d_inverse,
    segment_bincount,
    segment_reduce,
)


@dataclass
class ReferenceMap:
    """A uniform 5 cm elevation + semantic map built from GT-static points."""

    keys: np.ndarray          # (M,) int64 level-0 Morton, sorted
    ground_z: np.ndarray      # (M,) float32
    z_max: np.ndarray         # (M,) float32
    n_points: np.ndarray      # (M,) int32
    gt_class: np.ndarray      # (M,) int8 — majority GT class
    cell: float
    n_frames: int
    n_moving_removed: int
    source: str

    def __len__(self) -> int:
        return len(self.keys)

    def centres(self):
        ix, iy = morton_2d_inverse(self.keys, 0)
        return ((ix + 0.5) * self.cell).astype(np.float32), \
               ((iy + 0.5) * self.cell).astype(np.float32)

    def lookup(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Row index of the reference cell under each (x, y); -1 if absent."""
        ix = np.floor(np.asarray(x) / self.cell).astype(np.int64)
        iy = np.floor(np.asarray(y) / self.cell).astype(np.int64)
        k = morton_2d(ix, iy)
        if len(self.keys) == 0:
            return np.full(k.shape, -1, np.int64)
        pos = np.clip(np.searchsorted(self.keys, k), 0, len(self.keys) - 1)
        return np.where(self.keys[pos] == k, pos, -1)

    def limitations(self) -> List[str]:
        return [
            "poses are exact on synthetic data, SLAM-derived (and drifting) "
            "on SemanticKITTI",
            "only label-declared moving points are removed; a car parked for "
            "the whole window is permanent structure here",
            "surfaces no beam reached are absent, so completeness is scored "
            "only where the reference has a cell",
            "it is itself a 5 cm map, so a coarse adaptive cell is charged "
            "for the quantisation the budget asked for",
        ]


def build_reference_map(
    frames: List[Dict[str, Any]],
    cell: float = BASE,
    max_frames: int = 50,
    source: str = "synthetic",
) -> ReferenceMap:
    """Accumulate GT-static points from ``frames`` into a uniform map.

    ``frames`` are loader dicts (with ``points``, ``pose``, ``gt_label``,
    ``gt_moving``), not pipeline Frames — the reference must not depend on
    anything the pipeline computed.
    """
    xs, ys, zs, cs = [], [], [], []
    n_removed = 0
    used = 0

    for f in frames[:max_frames]:
        pts = np.asarray(f["points"])[:, :3]
        pose = f.get("pose")
        pose = np.eye(4) if pose is None else np.asarray(pose)
        mov = f.get("gt_moving")
        keep = np.ones(len(pts), bool) if mov is None else ~np.asarray(mov)
        n_removed += int((~keep).sum())

        pw = pts[keep] @ pose[:3, :3].T + pose[:3, 3]
        xs.append(pw[:, 0])
        ys.append(pw[:, 1])
        zs.append(pw[:, 2])
        gl = f.get("gt_label")
        cs.append(np.full(int(keep.sum()), -1, np.int8) if gl is None
                  else np.asarray(gl)[keep].astype(np.int8))
        used += 1

    if not xs:
        e = np.zeros(0)
        return ReferenceMap(e.astype(np.int64), e.astype(np.float32),
                            e.astype(np.float32), e.astype(np.int32),
                            e.astype(np.int8), cell, 0, 0, source)

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    z = np.concatenate(zs).astype(np.float32)
    c = np.concatenate(cs)

    ix = np.floor(x / cell).astype(np.int64)
    iy = np.floor(y / cell).astype(np.int64)
    uniq, starts, order = group_by_key(morton_2d(ix, iy))

    zo = z[order]
    from adaptive_lidar.utils.grouping import (
        segment_percentile_sorted,
        segment_sort,
    )
    zs_sorted = segment_sort(zo, starts)
    gz = segment_percentile_sorted(zs_sorted, starts, 10)
    zmax = segment_reduce(zo, starts, "max")
    npts = group_sizes(starts).astype(np.int32)

    co = np.clip(c[order].astype(np.int64), 0, NUM_CLASSES - 1)
    hist = segment_bincount(co, starts, NUM_CLASSES)
    dom = np.argmax(hist, axis=1).astype(np.int8)
    dom = np.where(hist.sum(axis=1) > 0, dom, np.int8(-1))

    return ReferenceMap(uniq, gz.astype(np.float32), zmax.astype(np.float32),
                        npts, dom, cell, used, n_removed, source)
