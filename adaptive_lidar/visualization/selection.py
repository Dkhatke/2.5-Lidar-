"""
Hit testing: turning a click into a selection.

The priority is object -> cell -> empty space, because an object box is a
deliberate target a few pixels across while a cell is whatever happens to be
underneath, and a user who clicks a car means the car.

CELL LOOKUP RUNS AGAINST THE CACHED FRAME, NOT THE LIVE MAP
-----------------------------------------------------------
``AdaptiveMap.cell_at`` only exists for the map's current state, which is the
last frame of the run.  A click while scrubbing at frame 4 has to answer with
frame 4's values, so lookup goes through the per-frame cache instead.  The
logic is the same as ``cell_at``: try the finest level first, because if a
5 cm cell exists at that point it is the more informative answer than the
40 cm cell that also contains it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from adaptive_lidar.pipeline.types import CLASS_NAMES, ResolutionLevel
from adaptive_lidar.visualization.playback import cell_record

#: How close, in pixels, a click must be to an object box to select it
#: instead of the cell underneath.
OBJECT_HIT_PX = 18.0


@dataclass
class Selection:
    """What the user picked. ``kind`` is "cell", "object" or "empty"."""
    kind: str
    world_xy: Tuple[float, float]
    frame_idx: int
    cell: Optional[Dict[str, Any]] = None
    obj: Optional[Dict[str, Any]] = None
    #: How far the reported cell is from the point actually asked about.
    #: Non-zero only for an object, whose centroid is a centre of MASS while
    #: the sensor only ever observes SURFACES — so the centroid of a car
    #: usually lands in the unobserved gap inside its own outline.
    cell_offset_m: float = 0.0

    @property
    def is_empty(self) -> bool:
        return self.kind == "empty"

    def label(self) -> str:
        if self.kind == "object" and self.obj:
            return (f"object #{self.obj['id']} "
                    f"({CLASS_NAMES[self.obj['cls']]})")
        if self.kind == "cell" and self.cell:
            return (f"cell at ({self.cell['cx']:.2f}, {self.cell['cy']:.2f}) "
                    f"· {self.cell['resolution'] * 100:.0f} cm")
        return "nothing"


# ════════════════════════════════════════════════════════════
# Cell lookup against a cached frame
# ════════════════════════════════════════════════════════════
#: Index origin offset, so a negative cell index still packs into a
#: monotonically ordered unsigned key. 2^20 cells at 80 cm is 840 km.
_IDX_OFF = 1 << 20


def _pack(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    return ((ix + _IDX_OFF) << 32) | (iy + _IDX_OFF)


def _level_index(cells: Dict[str, np.ndarray], level: int
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """(sorted packed keys, row for each key) for one level of one frame.

    A sorted array plus ``searchsorted`` rather than a Python dict: building
    the dict for 118k cells costs more than the whole render, and this is
    built on the first click of a frame the user may click many times.
    Memoised on the cells dict itself, which is the cached frame, so
    scrubbing back and forth does not rebuild it.
    """
    cache = cells.setdefault("_lookup", {})
    if level in cache:
        return cache[level]
    size = ResolutionLevel.size(level)
    sel = np.flatnonzero(cells["level"] == level).astype(np.int64)
    if sel.size == 0:
        out = (np.empty(0, np.int64), sel)
    else:
        # Centres are (i + 0.5) * size, so the index comes back exactly;
        # rounding rather than flooring keeps float32 from landing a hair
        # below the boundary and picking the neighbour.
        ix = np.rint(np.asarray(cells["cx"][sel], np.float64) / size
                     - 0.5).astype(np.int64)
        iy = np.rint(np.asarray(cells["cy"][sel], np.float64) / size
                     - 0.5).astype(np.int64)
        keys = _pack(ix, iy)
        order = np.argsort(keys, kind="stable")
        out = (keys[order], sel[order])
    cache[level] = out
    return out


def find_cell_row(cells: Dict[str, np.ndarray], x: float, y: float
                  ) -> Optional[Tuple[int, int]]:
    """(row, level) of the FINEST cached cell containing (x, y), or None."""
    if cells is None or len(cells.get("cx", ())) == 0:
        return None
    present = set(np.unique(cells["level"]).tolist())
    for level in range(ResolutionLevel.N_LEVELS):
        if level not in present:
            continue
        size = ResolutionLevel.size(level)
        want = _pack(np.int64(np.floor(x / size)),
                     np.int64(np.floor(y / size)))
        keys, rows = _level_index(cells, level)
        if keys.size == 0:
            continue
        j = int(np.searchsorted(keys, want))
        if j < keys.size and keys[j] == want:
            return int(rows[j]), int(level)
    return None


def nearest_cell(cells: Dict[str, np.ndarray], x: float, y: float,
                 max_m: float = 1.5
                 ) -> Tuple[Optional[Dict[str, Any]], float]:
    """The closest cell within ``max_m``, and how far away it is.

    Used for the cell "under" an object. An exact lookup at the centroid
    usually fails — 18 of 30 tracked objects on frame 0 of `mixed_urban` —
    not because the object is missing from the map but because a LiDAR sees
    the surfaces of a car and not the middle of it. Returning nothing there
    would read as "the map lost this object", which is the opposite of true.
    """
    exact = cell_at_world(cells, x, y)
    if exact is not None:
        return exact, 0.0
    if cells is None or len(cells.get("cx", ())) == 0:
        return None, float("inf")
    d = np.hypot(np.asarray(cells["cx"], np.float32) - x,
                 np.asarray(cells["cy"], np.float32) - y)
    j = int(np.argmin(d))
    if float(d[j]) > max_m:
        return None, float(d[j])
    rec = cell_record(cells, j)
    rec["row"] = j
    return rec, float(d[j])


def cell_at_world(cells: Dict[str, np.ndarray], x: float, y: float
                  ) -> Optional[Dict[str, Any]]:
    """The finest cached cell containing (x, y), decoded for the inspector."""
    hit = find_cell_row(cells, x, y)
    if hit is None:
        return None
    row, _level = hit
    rec = cell_record(cells, row)
    rec["row"] = row
    return rec


# ════════════════════════════════════════════════════════════
# Object lookup
# ════════════════════════════════════════════════════════════
def find_object(objects: List[Dict[str, Any]], transform,
                display_x: float, display_y: float,
                max_px: float = OBJECT_HIT_PX,
                min_points: int = 12) -> Optional[Dict[str, Any]]:
    """The nearest drawn object within ``max_px`` of the click, or None.

    Distance is measured in SCREEN pixels rather than metres: an object 60 m
    away is a few pixels across, and a metre-based radius would make it
    impossible to hit while making nearby objects greedy.
    """
    best, best_d = None, max_px
    for o in objects or []:
        if o.get("points", 0) < min_points:
            continue
        c = o["centroid_world"]
        sx, sy = transform.world_to_screen(float(c[0]), float(c[1]))
        d = float(np.hypot(sx - display_x, sy - display_y))
        if d < best_d:
            best, best_d = o, d
    return best


# ════════════════════════════════════════════════════════════
# The one entry point the UI calls
# ════════════════════════════════════════════════════════════
def select_at(display_x: float, display_y: float, transform, snapshot,
              frame_idx: int, *, allow_objects: bool = True) -> Selection:
    """Resolve a click into a Selection, object first, then cell, then empty."""
    wx, wy = transform.screen_to_world(display_x, display_y)

    if allow_objects:
        obj = find_object(snapshot.objects, transform, display_x, display_y)
        if obj is not None:
            c = obj["centroid_world"]
            cell, off = nearest_cell(snapshot.cells, float(c[0]), float(c[1]))
            return Selection("object", (float(c[0]), float(c[1])), frame_idx,
                             cell=cell, obj=obj, cell_offset_m=off)

    cell = cell_at_world(snapshot.cells, wx, wy)
    if cell is not None:
        return Selection("cell", (wx, wy), frame_idx, cell=cell)
    return Selection("empty", (wx, wy), frame_idx)


def reselect_on_frame(sel: Optional[Selection], snapshot, frame_idx: int
                      ) -> Optional[Selection]:
    """Re-resolve an existing selection against a different frame.

    Scrubbing should keep the inspector pointed at the same PLACE, showing
    what the map knew there at the new time — not silently keep showing stale
    values from the frame the click happened on.
    """
    if sel is None or sel.is_empty:
        return sel
    if sel.kind == "object" and sel.obj is not None:
        oid = sel.obj["id"]
        match = next((o for o in (snapshot.objects or [])
                      if o["id"] == oid), None)
        if match is not None:
            c = match["centroid_world"]
            cell, off = nearest_cell(snapshot.cells, float(c[0]), float(c[1]))
            return Selection("object", (float(c[0]), float(c[1])), frame_idx,
                             cell=cell, obj=match, cell_offset_m=off)
        # The track is gone this frame; fall back to the place it was.
    wx, wy = sel.world_xy
    cell = cell_at_world(snapshot.cells, wx, wy)
    if cell is None:
        return Selection("empty", (wx, wy), frame_idx)
    return Selection("cell", (wx, wy), frame_idx, cell=cell)
