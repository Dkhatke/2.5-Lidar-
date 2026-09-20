"""
Pipeline data -> scene geometry.

The one rule this module exists to enforce: **no pipeline logic in
JavaScript.** Everything the renderer draws is decided here, in Python,
from data the pipeline already produced — cached map cells, tracked
instances, the ego pose. The browser receives geometry and colours, and its
only job is to put them on screen and report back what was clicked.

    frame snapshot  ->  build_scene_data()  ->  JSON  ->  three.js

WHAT IS MEASURED AND WHAT IS DRAWN
----------------------------------
Cells are measured. A cell's footprint is its real resolution, its base is
``ground_z``, its top is ``z_max``; nothing is scaled up to look better.

Object *boxes* are not entirely measured, and the difference is recorded
rather than hidden. A tracked instance's ``extent`` is the bounding box of
the points the sensor actually returned, and a LiDAR sees one side of a car,
so that box is routinely too small to read as a car. Where a dimension is
enlarged to a class floor for visibility, the object carries
``geom_source: "padded"`` and the floor it was padded to — and the scene
panel says so. Never ``"measured"`` for a number the sensor did not measure.

PACKING
-------
Cell arrays go over the wire as base64 of the raw little-endian buffers,
16 bytes per cell. The scene is re-sent on every displayed frame, so the
alternative — JSON numbers, roughly 60 bytes a cell — would be the cost of
playback rather than a detail of it.
"""
from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from adaptive_lidar.evaluation.metrics import RANGE_BANDS
from adaptive_lidar.pipeline.types import (CLASS_NAMES, ResolutionLevel,
                                           Traversability, TrackState,
                                           VehicleProfile)
from adaptive_lidar.visualization import inspector as INS
from adaptive_lidar.visualization.render import (CLASS_COLOURS, LEVEL_COLOURS,
                                                 TRAV_COLOURS)

#: How a cell's colour is chosen. Every palette comes from `render.py`, so
#: the scene and the 2D map cannot drift into different colour languages.
COLOUR_MODES = ("semantic", "traversability", "resolution level", "height")

#: Height exaggeration per display mode. "subtle" compresses tall structure
#: so a 12 m facade does not hide the road behind it; it is a display
#: choice and the scene panel states the factor in use.
HEIGHT_MODES: Dict[str, float] = {"off": 0.0, "subtle": 0.45, "extruded": 1.0}

#: Cap on cells sent to the browser, and the radius they are taken from.
#: Beyond either, the farthest are dropped — never the nearest, and never
#: silently: the count is reported under the canvas.
#:
#: 18,000 cells is 288 kB packed and about 384 kB once base64'd, which is
#: re-sent on every displayed frame. The full cached frame is ~78,000 cells
#: and would be four times that, for detail that is beyond the far clip of
#: any useful camera anyway.
DEFAULT_MAX_CELLS = 18000
DEFAULT_RADIUS_M = 55.0

#: Cell heights are quantised to centimetres on the wire. Cells are at
#: least 5 cm across and the renderer is drawing boxes, so a centimetre is
#: far below what any camera resolves; the inspector still reads the stored
#: float. Centres stay float32 because the click path round-trips through
#: them into the real cell lookup, where a centimetre of drift could pick
#: the neighbouring 5 cm cell.
_Z_QUANT = 100.0

#: Minimum drawn size per class, in metres, when the measured extent is
#: smaller. Visualisation geometry, flagged as such on every object it
#: touches. A pedestrian that returns four points still has to be findable.
CLASS_FLOOR: Dict[int, Tuple[float, float, float]] = {
    2: (0.35, 0.35, 0.80),      # static_obstacle
    3: (3.80, 1.70, 1.40),      # vehicle
    4: (0.50, 0.50, 1.65),      # vru
    5: (0.60, 0.60, 1.20),      # vegetation
}
_DEFAULT_FLOOR = (0.40, 0.40, 0.60)

#: A static obstacle narrower than this and taller than POLE_MIN_H is drawn
#: as a cylinder rather than a box. It is a drawing decision, not a class.
POLE_MAX_FOOTPRINT = 0.80
POLE_MIN_H = 1.50

#: Which classes get an object box in the scene.
#:
#: Ground classes are excluded, and it is worth saying why. The tracker
#: does produce instances the classifier called ground_drivable — typically
#: an 8 m x 1 m x 0.16 m slab of road surface. In the 2D map that is a thin
#: outline and harmless. Here it becomes a solid slab lying across the
#: road, which (a) reads as an object that is not there and (b) sits
#: between the cursor and the cells, so it swallows the cell clicks this
#: tab exists for. They remain visible in the Live demo's object table and
#: 2D overlay; nothing is hidden, it is drawn where it makes sense.
SCENE_OBJECT_CLASSES = (2, 3, 4, 5)


# ════════════════════════════════════════════════════════════
# Packing
# ════════════════════════════════════════════════════════════
def _b64(a: np.ndarray, dtype) -> str:
    return base64.b64encode(
        np.ascontiguousarray(a, dtype=dtype).tobytes()).decode("ascii")


def _hex_to_rgb01(h: str) -> List[float]:
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


def palettes() -> Dict[str, Any]:
    """Every colour the renderer may use, taken from ``render.py``."""
    return {
        "semantic": [_hex_to_rgb01(CLASS_COLOURS[i])
                     for i in range(len(CLASS_NAMES))],
        "traversability": [_hex_to_rgb01(TRAV_COLOURS[i]) for i in range(3)],
        "level": [_hex_to_rgb01(LEVEL_COLOURS[i])
                  for i in range(ResolutionLevel.N_LEVELS)],
        "classNames": list(CLASS_NAMES),
        "levelNames": [ResolutionLevel.name(i)
                       for i in range(ResolutionLevel.N_LEVELS)],
        "travNames": [Traversability.NAMES[i] for i in range(3)],
        "stateNames": [TrackState.NAMES[i] for i in range(3)],
    }


# ════════════════════════════════════════════════════════════
# Cells
# ════════════════════════════════════════════════════════════
def select_cells(cells: Dict[str, np.ndarray], ego_xy: Tuple[float, float],
                 *, radius_m: float, max_cells: int
                 ) -> Tuple[np.ndarray, int]:
    """Row indices to draw, plus how many were dropped.

    Nearest-first, so what is cut is always the far field. The caller is
    expected to report the drop rather than let the scene quietly thin out.
    """
    n = len(cells.get("cx", ()))
    if n == 0:
        return np.zeros(0, np.int64), 0
    d2 = ((np.asarray(cells["cx"], np.float32) - ego_xy[0]) ** 2
          + (np.asarray(cells["cy"], np.float32) - ego_xy[1]) ** 2)
    keep = np.flatnonzero(d2 <= radius_m * radius_m)
    dropped = int(n - keep.size)
    if keep.size > max_cells:
        order = np.argsort(d2[keep], kind="stable")[:max_cells]
        dropped += int(keep.size - max_cells)
        keep = np.sort(keep[order])
    return keep.astype(np.int64), dropped


def cell_geometry(cells: Dict[str, np.ndarray], keep: np.ndarray,
                  *, profile: Optional[VehicleProfile] = None,
                  need_traversability: bool = True) -> Dict[str, Any]:
    """The packed per-cell buffers the renderer instances from.

    One entry per cell: centre, footprint (its true resolution), base
    (``ground_z``), top (``z_max``), level, class, traversability verdict,
    and a normalised return count for the height/opacity ramps.
    """
    if keep.size == 0:
        return {"n": 0}

    cx = np.asarray(cells["cx"], np.float32)[keep]
    cy = np.asarray(cells["cy"], np.float32)[keep]
    lvl = np.asarray(cells["level"], np.int64)[keep]
    gz = np.asarray(cells["ground_z"], np.float32)[keep]
    zmax = np.asarray(cells["z_max"], np.float32)[keep]
    # z_max is the true max INSIDE the clearance band, so on a sparsely
    # observed cell it can sit below the ground estimate. Clamp for drawing;
    # the inspector still shows both stored numbers unclamped.
    top = np.maximum(zmax, gz)

    # Only computed when something is going to use it. The verdict needs a
    # neighbour lookup per cell, and it was being computed for all ~78,000
    # cached cells on every redraw to colour 18,000 of them — in a mode
    # that usually is not even selected.
    if need_traversability:
        verdict = INS.traversability_arrays(cells, profile, keep)[keep]
    else:
        verdict = np.zeros(keep.size, np.int8)
    npts = np.asarray(cells["n_points"], np.float32)[keep]
    zq = np.clip(np.stack([gz, top], 1).ravel() * _Z_QUANT,
                 -32768, 32767)

    return {
        "n": int(keep.size),
        # Footprint is not sent: it is ResolutionLevel.size(level), and the
        # renderer already has that table.
        "xy": _b64(np.stack([cx, cy], 1).ravel(), np.float32),
        "zcm": _b64(zq, np.int16),
        "level": _b64(lvl, np.uint8),
        "cls": _b64(np.asarray(cells["sem_class"], np.int64)[keep].clip(0, 5),
                    np.uint8),
        "trav": _b64(verdict, np.uint8),
        "returns": _b64(np.clip(npts, 0, 255), np.uint8),
        "sizes": [float(ResolutionLevel.size(i))
                  for i in range(ResolutionLevel.N_LEVELS)],
        "zRange": [float(gz.min()), float(top.max())],
    }


# ════════════════════════════════════════════════════════════
# Objects
# ════════════════════════════════════════════════════════════
def _primitive(cls: int, ext: Sequence[float]) -> str:
    """Which shape reads as this thing. A drawing choice, not a class."""
    if cls == 4:
        return "person"
    if cls == 3:
        return "vehicle"
    if cls == 2 and max(ext[0], ext[1]) < POLE_MAX_FOOTPRINT \
            and ext[2] >= POLE_MIN_H:
        return "pole"
    return "box"


def object_to_scene_object(obj: Dict[str, Any], ego_xy) -> Dict[str, Any]:
    """One tracked instance as drawable geometry, with its provenance.

    ``geom_source`` is the honest part: "measured" means every dimension is
    the tracker's own bounding box; "padded" means at least one was raised
    to a class floor so the object is visible at all, and ``floor`` says to
    what.
    """
    c = np.asarray(obj["centroid_world"], float)
    measured = np.asarray(obj["extent"], float)
    cls = int(obj["cls"])
    floor = np.asarray(CLASS_FLOOR.get(cls, _DEFAULT_FLOOR), float)
    drawn = np.maximum(measured, floor)
    padded = bool(np.any(drawn > measured + 1e-6))

    v = np.asarray(obj["vel_world"], float)
    speed = float(obj["speed_world"])
    yaw = float(np.arctan2(v[1], v[0])) if speed > 0.4 else 0.0

    return {
        "id": int(obj["id"]),
        "cls": cls,
        "className": CLASS_NAMES[cls],
        "state": int(obj["state"]),
        "stateName": obj["state_name"],
        "primitive": _primitive(cls, drawn),
        # Base sits at the bottom of the drawn box, so a padded object grows
        # upward from where the points were rather than sinking into the road.
        "centre": [float(c[0]), float(c[1]),
                   float(c[2] - measured[2] / 2.0 + drawn[2] / 2.0)],
        "size": [float(drawn[0]), float(drawn[1]), float(drawn[2])],
        "measured": [float(measured[0]), float(measured[1]),
                     float(measured[2])],
        "geomSource": "padded" if padded else "measured",
        "floor": [float(f) for f in floor] if padded else None,
        "yaw": yaw,
        "speed": speed,
        "vel": [float(v[0]), float(v[1])],
        "points": int(obj["points"]),
        "age": int(obj["age"]),
        "range": float(np.hypot(c[0] - ego_xy[0], c[1] - ego_xy[1])),
        "label": f"{int(obj['id']):02d} {CLASS_NAMES[cls]}",
    }


def ego_to_scene_object(snapshot) -> Dict[str, Any]:
    """The ego box and the sensor marker, from the stored pose."""
    ex, ey = snapshot.ego_xy
    return {
        "xy": [float(ex), float(ey)],
        "heading": float(snapshot.heading),
        # The vehicle outline the 2D overlay already draws, given depth.
        "size": [4.6, 2.0, 1.5],
        # Sensor height is the config's mount height, not a measurement of
        # anything in the scene.
        "sensorZ": 1.8,
    }


# ════════════════════════════════════════════════════════════
# The scene
# ════════════════════════════════════════════════════════════
def distance_rings() -> List[float]:
    """The evaluation's own range-band edges, so the rings mean something.

    Not a decorative scale: these are the boundaries every accuracy table in
    ``docs/RESULTS.md`` is stratified by.
    """
    edges = sorted({float(b) for band in RANGE_BANDS for b in band})
    return [e for e in edges if e > 0.0]


def build_scene_data(snapshot, *, frame_idx: int, n_frames: int,
                     colour_by: str = "semantic",
                     height_mode: str = "subtle",
                     max_cells: int = DEFAULT_MAX_CELLS,
                     radius_m: float = DEFAULT_RADIUS_M,
                     show_points: bool = False,
                     max_points: int = 20000,
                     min_object_points: int = 12,
                     profile: Optional[VehicleProfile] = None,
                     selected: Optional[Dict[str, Any]] = None,
                     ) -> Dict[str, Any]:
    """Everything the renderer needs for one frame, and nothing more."""
    ego = snapshot.ego_xy
    keep, dropped = select_cells(snapshot.cells, ego, radius_m=radius_m,
                                 max_cells=max_cells)
    cells = cell_geometry(snapshot.cells, keep, profile=profile,
                          need_traversability=(colour_by == "traversability"))

    objects = [object_to_scene_object(o, ego)
               for o in (snapshot.objects or [])
               if o.get("points", 0) >= min_object_points
               and int(o["cls"]) in SCENE_OBJECT_CLASSES]
    ground_tracks = sum(1 for o in (snapshot.objects or [])
                        if o.get("points", 0) >= min_object_points
                        and int(o["cls"]) not in SCENE_OBJECT_CLASSES)

    points: Dict[str, Any] = {"n": 0}
    if show_points and snapshot.overlay is not None \
            and len(snapshot.overlay):
        p = np.asarray(snapshot.overlay, np.float32)[:, :3]
        if len(p) > max_points:
            step = int(np.ceil(len(p) / max_points))
            p = p[::step]
        points = {"n": int(len(p)), "xyz": _b64(p.ravel(), np.float32)}

    return {
        "frame": {"index": int(frame_idx), "count": int(n_frames),
                  "id": int(snapshot.frame_id),
                  "time": float(snapshot.timestamp),
                  "points": int(snapshot.n_points)},
        "ego": ego_to_scene_object(snapshot),
        "cells": cells,
        "cellsDropped": int(dropped),
        "objects": objects,
        "groundTracks": int(ground_tracks),
        "points": points,
        "rings": distance_rings(),
        "palettes": palettes(),
        "colourBy": colour_by,
        "heightScale": HEIGHT_MODES.get(height_mode, 0.45),
        "heightMode": height_mode,
        "selected": selected or {},
    }


def selection_marker(sel) -> Dict[str, Any]:
    """What the renderer should highlight, from an existing Selection."""
    if sel is None or getattr(sel, "is_empty", True):
        return {}
    if sel.kind == "object" and sel.obj is not None:
        return {"kind": "object", "id": int(sel.obj["id"]),
                "xy": [float(sel.world_xy[0]), float(sel.world_xy[1])]}
    if sel.kind == "cell" and sel.cell is not None:
        return {"kind": "cell",
                "xy": [float(sel.cell["cx"]), float(sel.cell["cy"])],
                "size": float(sel.cell["resolution"]),
                "z": [float(sel.cell["ground_z"]),
                      float(max(sel.cell["z_max"], sel.cell["ground_z"]))]}
    return {}


def scene_summary(data: Dict[str, Any]) -> Dict[str, str]:
    """The short readout under the canvas. All counts, no claims."""
    n = data["cells"].get("n", 0)
    movers = sum(1 for o in data["objects"] if o["stateName"] == "MOVING")
    padded = sum(1 for o in data["objects"] if o["geomSource"] == "padded")
    out = {
        "cells drawn": f"{n:,}",
        "objects": f"{len(data['objects'])}",
        "moving": f"{movers}",
    }
    if data["cellsDropped"]:
        out["cells beyond the draw radius"] = f"{data['cellsDropped']:,}"
    if padded:
        out["boxes padded for visibility"] = f"{padded}"
    if data["points"].get("n"):
        out["moving points"] = f"{data['points']['n']:,}"
    if data.get("groundTracks"):
        out["ground-class tracks (not boxed)"] = f"{data['groundTracks']}"
    return out
