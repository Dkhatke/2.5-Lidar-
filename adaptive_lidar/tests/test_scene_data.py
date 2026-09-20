"""
The scene adapter: does the drawing agree with the map?

Every number the renderer receives is decided here, in Python, so these
tests are where the 3D view is held to the 2.5D record. Three things would
be invisible if they broke — a cell drawn at the wrong footprint, a height
taken from the wrong field, and a box presented as measured when it was
padded — so each gets a test of its own.
"""
from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from adaptive_lidar.pipeline.types import (CLASS_NAMES, ResolutionLevel,
                                           Traversability, VehicleProfile)
from adaptive_lidar.visualization import inspector as INS
from adaptive_lidar.visualization import playback as PB
from adaptive_lidar.visualization import scene_data as SD
from adaptive_lidar.visualization import selection as SEL


# ════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════
def _cells(entries, **overrides):
    """A cache-format cells dict from [(cx, cy, level), ...]."""
    n = len(entries)
    out = {
        "cx": np.array([e[0] for e in entries], np.float32),
        "cy": np.array([e[1] for e in entries], np.float32),
        "level": np.array([e[2] for e in entries], np.int8),
    }
    for k in PB.CELL_FIELDS:
        if k in out:
            continue
        out[k] = (np.full(n, 128, np.uint8) if k in PB._CELL_U8
                  else np.zeros(n, PB._CELL_DTYPE.get(k, np.float32)))
    out[PB.EVIDENCE_FIELD] = np.full((n, 6), 42, np.uint8)
    out["size"] = ResolutionLevel.sizes_array()[
        np.clip(out["level"].astype(np.int64), 0, 4)]
    for k, v in overrides.items():
        out[k][:] = v
    return out


def _centre(ix, iy, level):
    s = ResolutionLevel.size(level)
    return ((ix + 0.5) * s, (iy + 0.5) * s)


def _obj(oid=1, xy=(10.0, 2.0), cls=3, ext=(4.2, 1.8, 1.5), state=1,
         points=200, vel=(3.0, 0.0, 0.0)):
    return {"id": oid, "cls": cls, "state": state,
            "state_name": ["STATIC", "MOVING", "MOVABLE_BUT_STATIONARY"][state],
            "speed_world": float(np.hypot(vel[0], vel[1])),
            "vel_world": np.array(vel, np.float32),
            "centroid_world": np.array([xy[0], xy[1], ext[2] / 2], np.float32),
            "extent": np.array(ext, np.float32),
            "points": points, "age": 4}


class _Snap:
    def __init__(self, cells, objects=(), overlay=None, ego=(0.0, 0.0)):
        self.cells = cells
        self.objects = list(objects)
        self.overlay = (np.zeros((0, 3), np.float32) if overlay is None
                        else overlay)
        self.ego_xy = ego
        self.heading = 0.3
        self.frame_id = 7
        self.timestamp = 0.7
        self.n_points = 120000


def _unpack(d, key, dtype):
    return np.frombuffer(base64.b64decode(d[key]), dtype=dtype)


# ════════════════════════════════════════════════════════════
# Cell geometry
# ════════════════════════════════════════════════════════════
def test_cell_centres_survive_the_wire_exactly():
    """The click path round-trips through these, into the real lookup.

    A quantised centre could resolve to the neighbouring 5 cm cell, which
    is why the centres stay float32 while the heights do not.
    """
    # Deliberately NOT nested: a coarse cell containing a fine one is a
    # real case, but then the finest wins and this test would be asserting
    # the wrong thing. Each level gets its own patch of world.
    pts = [_centre(401 + 200 * lvl, -97 - 200 * lvl, lvl) + (lvl,)
           for lvl in range(ResolutionLevel.N_LEVELS)]
    cells = _cells(pts)
    keep = np.arange(len(pts))
    g = SD.cell_geometry(cells, keep)
    xy = _unpack(g, "xy", np.float32).reshape(-1, 2)
    for i, (cx, cy, _lvl) in enumerate(pts):
        assert xy[i, 0] == pytest.approx(cx, abs=1e-6)
        assert xy[i, 1] == pytest.approx(cy, abs=1e-6)
        # ...and the point still finds its own cell through the real lookup.
        assert SEL.find_cell_row(cells, float(xy[i, 0]),
                                 float(xy[i, 1])) == (i, i)


@pytest.mark.parametrize("level", range(ResolutionLevel.N_LEVELS))
def test_footprint_is_the_cells_real_resolution(level):
    """Not sent per cell — derived from the level, so it cannot drift."""
    cells = _cells([_centre(30, 8, level) + (level,)])
    g = SD.cell_geometry(cells, np.arange(1))
    lvl = _unpack(g, "level", np.uint8)
    assert int(lvl[0]) == level
    assert g["sizes"][level] == pytest.approx(ResolutionLevel.size(level))


def test_height_comes_from_ground_z_and_z_max():
    cells = _cells([_centre(30, 8, 1) + (1,)], ground_z=0.35, z_max=2.85)
    g = SD.cell_geometry(cells, np.arange(1))
    z = _unpack(g, "zcm", np.int16).astype(np.float32) / 100.0
    assert z[0] == pytest.approx(0.35, abs=0.01)
    assert z[1] == pytest.approx(2.85, abs=0.01)


def test_a_top_below_ground_is_clamped_for_drawing_only():
    """z_max is the max INSIDE the clearance band, so it can sit low.

    An inverted box would render as a hole in the road. The stored numbers
    are untouched — the inspector still shows both.
    """
    cells = _cells([_centre(30, 8, 1) + (1,)], ground_z=1.20, z_max=0.40)
    g = SD.cell_geometry(cells, np.arange(1))
    z = _unpack(g, "zcm", np.int16).astype(np.float32) / 100.0
    assert z[1] >= z[0]
    # float16 in the cache, so 0.40 comes back as 0.3999; the point is that
    # it was not rewritten, not that it is exact.
    assert float(cells["z_max"][0]) == pytest.approx(0.40, abs=1e-3)


def test_traversability_sent_is_the_maps_own_verdict():
    cells = _cells([_centre(30, 8, 1) + (1,)], sem_class=4, n_points=30,
                   overhead_clearance=9.0)
    g = SD.cell_geometry(cells, np.arange(1))
    trav = _unpack(g, "trav", np.uint8)
    assert int(trav[0]) == Traversability.BLOCKED


def test_the_vectorised_verdict_matches_the_per_cell_one():
    """One rule, two shapes. They must not drift.

    ``derive`` writes the inspector's reason strings and
    ``traversability_arrays`` colours 18,000 cells; if they disagreed the
    scene would contradict the panel describing it.
    """
    rng = np.random.default_rng(4)
    entries, n = [], 60
    for i in range(n):
        lvl = int(rng.integers(0, 3))
        entries.append(_centre(int(rng.integers(0, 40)),
                               int(rng.integers(0, 40)), lvl) + (lvl,))
    cells = _cells(entries)
    cells["ground_z"][:] = rng.normal(0, 0.25, n).astype(np.float32)
    cells["z_max"][:] = cells["ground_z"] + rng.gamma(1.0, 0.5, n)
    cells["z_var"][:] = rng.gamma(1.0, 0.02, n)
    cells["overhead_clearance"][:] = rng.choice([1.0, 9.0], n)
    cells["sem_class"][:] = rng.integers(0, 6, n)
    cells["n_points"][:] = rng.integers(0, 40, n)

    vec = INS.traversability_arrays(cells, VehicleProfile.wheeled())
    for row in range(n):
        rec = SEL.cell_at_world(cells, float(cells["cx"][row]),
                                float(cells["cy"][row]))
        if rec is None or rec["row"] != row:
            continue        # a finer cell covers this centre; not this row
        one = INS.derive(rec, cells, VehicleProfile.wheeled())
        assert one.verdict == vec[row], (row, one.reasons)


def test_nearest_cells_are_kept_and_the_drop_is_counted():
    entries = [_centre(i, 0, 2) + (2,) for i in range(200)]
    cells = _cells(entries)
    keep, dropped = SD.select_cells(cells, (0.0, 0.0), radius_m=1000.0,
                                    max_cells=20)
    assert keep.size == 20 and dropped == 180
    d = np.hypot(cells["cx"][keep], cells["cy"][keep])
    assert d.max() <= np.hypot(cells["cx"], cells["cy"]).max()
    # the ones kept are the nearest 20, in any order
    order = np.argsort(np.hypot(cells["cx"], cells["cy"]))[:20]
    assert set(keep.tolist()) == set(order.tolist())


def test_the_radius_also_drops_and_counts():
    entries = [_centre(i * 40, 0, 2) + (2,) for i in range(20)]
    cells = _cells(entries)
    keep, dropped = SD.select_cells(cells, (0.0, 0.0), radius_m=25.0,
                                    max_cells=10000)
    assert keep.size + dropped == len(entries)
    assert np.hypot(cells["cx"][keep], cells["cy"][keep]).max() <= 25.0


def test_an_empty_frame_does_not_crash():
    g = SD.cell_geometry(_cells([]), np.zeros(0, np.int64))
    assert g == {"n": 0}


# ════════════════════════════════════════════════════════════
# Objects
# ════════════════════════════════════════════════════════════
def test_a_well_measured_box_is_not_padded():
    o = SD.object_to_scene_object(_obj(ext=(4.6, 1.9, 1.6)), (0.0, 0.0))
    assert o["geomSource"] == "measured"
    assert o["size"] == pytest.approx(o["measured"])
    assert o["floor"] is None


def test_an_under_measured_box_is_padded_and_says_so():
    """A LiDAR sees one side of a car, so this is the normal case."""
    o = SD.object_to_scene_object(_obj(ext=(0.9, 0.4, 0.5)), (0.0, 0.0))
    assert o["geomSource"] == "padded"
    assert o["size"][0] > o["measured"][0]
    assert o["floor"] == list(SD.CLASS_FLOOR[3])
    # the measured numbers are still carried, unmodified
    assert o["measured"] == pytest.approx([0.9, 0.4, 0.5])


def test_padding_grows_upward_from_where_the_points_were():
    """The base must not sink into the road when a box is enlarged."""
    ext = (0.9, 0.4, 0.5)
    o = SD.object_to_scene_object(_obj(ext=ext), (0.0, 0.0))
    measured_base = ext[2] / 2 - ext[2] / 2          # centroid z - h/2 = 0
    drawn_base = o["centre"][2] - o["size"][2] / 2
    assert drawn_base == pytest.approx(measured_base, abs=1e-6)


@pytest.mark.parametrize("cls,ext,want", [
    (4, (0.5, 0.5, 1.7), "person"),
    (3, (4.2, 1.8, 1.5), "vehicle"),
    (2, (0.25, 0.25, 3.0), "pole"),
    (2, (6.0, 3.0, 2.5), "box"),
    (5, (1.2, 1.2, 2.0), "box"),
])
def test_primitive_choice(cls, ext, want):
    o = SD.object_to_scene_object(_obj(cls=cls, ext=ext), (0.0, 0.0))
    assert o["primitive"] == want


def test_object_state_comes_from_the_tracker_not_the_class():
    """A parked car is a vehicle and is not moving. Both must survive."""
    o = SD.object_to_scene_object(_obj(cls=3, state=2, vel=(0.0, 0.0, 0.0)),
                                  (0.0, 0.0))
    assert o["className"] == "vehicle"
    assert o["stateName"] == "MOVABLE_BUT_STATIONARY"
    assert o["speed"] == pytest.approx(0.0)


def test_object_range_is_measured_from_the_ego():
    o = SD.object_to_scene_object(_obj(xy=(30.0, 40.0)), (0.0, 0.0))
    assert o["range"] == pytest.approx(50.0, abs=1e-3)


def test_yaw_is_only_taken_from_a_real_velocity():
    slow = SD.object_to_scene_object(_obj(vel=(0.1, 0.1, 0.0)), (0.0, 0.0))
    fast = SD.object_to_scene_object(_obj(vel=(0.0, 5.0, 0.0)), (0.0, 0.0))
    assert slow["yaw"] == 0.0
    assert fast["yaw"] == pytest.approx(np.pi / 2, abs=1e-6)


# ════════════════════════════════════════════════════════════
# Ego and world anchoring
# ════════════════════════════════════════════════════════════
def test_ego_uses_the_stored_pose():
    snap = _Snap(_cells([]), ego=(12.5, -3.25))
    e = SD.ego_to_scene_object(snap)
    assert e["xy"] == [12.5, -3.25]
    assert e["heading"] == pytest.approx(snap.heading)


def test_static_geometry_does_not_move_with_the_ego():
    """World anchoring, asserted rather than assumed.

    The camera may follow the vehicle; the world may not be dragged along
    with it. Same cells, two ego poses, identical drawn coordinates.
    """
    cells = _cells([_centre(200, 40, 1) + (1,)])
    a = SD.build_scene_data(_Snap(cells, ego=(0.0, 0.0)),
                            frame_idx=0, n_frames=2, radius_m=1e4)
    b = SD.build_scene_data(_Snap(cells, ego=(25.0, 4.0)),
                            frame_idx=1, n_frames=2, radius_m=1e4)
    assert a["cells"]["xy"] == b["cells"]["xy"]
    assert a["cells"]["zcm"] == b["cells"]["zcm"]
    assert a["ego"]["xy"] != b["ego"]["xy"]


# ════════════════════════════════════════════════════════════
# The payload
# ════════════════════════════════════════════════════════════
def test_the_payload_is_json_serialisable():
    """It crosses a websocket; a numpy scalar in it is a runtime failure."""
    snap = _Snap(_cells([_centre(5, 5, 1) + (1,)]), [_obj()])
    json.dumps(SD.build_scene_data(snap, frame_idx=0, n_frames=3))


def test_rings_are_the_evaluations_own_range_bands():
    """Not a decorative scale: these are the bands every table uses."""
    from adaptive_lidar.evaluation.metrics import RANGE_BANDS
    assert SD.distance_rings() == [10.0, 30.0, 60.0, 100.0]
    assert SD.distance_rings()[-1] == RANGE_BANDS[-1][1]


def test_palettes_come_from_the_2d_renderer():
    from adaptive_lidar.visualization.render import CLASS_COLOURS
    p = SD.palettes()
    assert p["classNames"] == list(CLASS_NAMES)
    first = CLASS_COLOURS[0].lstrip("#")
    assert p["semantic"][0] == pytest.approx(
        [int(first[i:i + 2], 16) / 255 for i in (0, 2, 4)])


@pytest.mark.parametrize("mode,scale", list(SD.HEIGHT_MODES.items()))
def test_height_mode_only_scales_the_drawing(mode, scale):
    snap = _Snap(_cells([_centre(5, 5, 1) + (1,)], ground_z=0.0, z_max=3.0))
    d = SD.build_scene_data(snap, frame_idx=0, n_frames=1, height_mode=mode)
    assert d["heightScale"] == scale
    # The stored extent goes over the wire unscaled; the renderer applies
    # the factor, so the inspector and the scene never disagree in metres.
    z = _unpack(d["cells"], "zcm", np.int16).astype(np.float32) / 100.0
    assert z[1] - z[0] == pytest.approx(3.0, abs=0.01)


def test_tiny_clusters_are_left_out_of_the_scene():
    snap = _Snap(_cells([_centre(5, 5, 1) + (1,)]), [_obj(points=3)])
    d = SD.build_scene_data(snap, frame_idx=0, n_frames=1)
    assert d["objects"] == []


def test_points_are_only_the_real_dynamic_overlay():
    pts = np.random.default_rng(0).normal(0, 5, (500, 3)).astype(np.float32)
    snap = _Snap(_cells([_centre(5, 5, 1) + (1,)]), overlay=pts)
    off = SD.build_scene_data(snap, frame_idx=0, n_frames=1,
                              show_points=False)
    on = SD.build_scene_data(snap, frame_idx=0, n_frames=1, show_points=True)
    assert off["points"]["n"] == 0
    assert on["points"]["n"] == 500
    back = _unpack(on["points"], "xyz", np.float32).reshape(-1, 3)
    assert back == pytest.approx(pts, abs=1e-6)


def test_the_summary_reports_what_was_dropped():
    entries = [_centre(i, 0, 2) + (2,) for i in range(400)]
    snap = _Snap(_cells(entries))
    d = SD.build_scene_data(snap, frame_idx=0, n_frames=1, max_cells=50)
    s = SD.scene_summary(d)
    assert s["cells drawn"] == "50"
    assert "cells beyond the draw radius" in s


def test_the_summary_reports_padded_boxes():
    snap = _Snap(_cells([_centre(5, 5, 1) + (1,)]),
                 [_obj(ext=(0.5, 0.3, 0.4))])
    s = SD.scene_summary(SD.build_scene_data(snap, frame_idx=0, n_frames=1))
    assert s["boxes padded for visibility"] == "1"


# ════════════════════════════════════════════════════════════
# Selection marker
# ════════════════════════════════════════════════════════════
def test_the_marker_follows_a_cell_selection():
    cells = _cells([_centre(60, 20, 2) + (2,)], ground_z=0.2, z_max=1.4)
    cx, cy = _centre(60, 20, 2)
    sel = SEL.Selection("cell", (cx, cy), 0,
                        cell=SEL.cell_at_world(cells, cx, cy))
    m = SD.selection_marker(sel)
    assert m["kind"] == "cell"
    assert m["xy"] == pytest.approx([cx, cy], abs=1e-3)
    assert m["size"] == pytest.approx(ResolutionLevel.size(2))


def test_the_marker_follows_an_object_selection():
    sel = SEL.Selection("object", (10.0, 2.0), 0, obj=_obj(oid=42))
    assert SD.selection_marker(sel) == {"kind": "object", "id": 42,
                                        "xy": [10.0, 2.0]}


def test_no_selection_is_an_empty_marker():
    assert SD.selection_marker(None) == {}
    assert SD.selection_marker(SEL.Selection("empty", (1.0, 1.0), 0)) == {}


# ════════════════════════════════════════════════════════════
# What gets a box, and what does not
# ════════════════════════════════════════════════════════════
def test_ground_class_tracks_get_no_box_but_are_counted():
    """A tracked instance the classifier called road is not an object.

    The tracker really does emit these — an 8 x 1 x 0.16 m slab of road
    surface. Drawn as a solid, it reads as something that is not there AND
    sits between the cursor and the cells, swallowing the cell clicks this
    view exists for. They stay in the Live demo's object table; here they
    are counted, not boxed.
    """
    snap = _Snap(_cells([_centre(5, 5, 1) + (1,)]),
                 [_obj(oid=1, cls=0, ext=(8.3, 1.0, 0.16)),
                  _obj(oid=2, cls=1, ext=(4.0, 1.0, 0.20)),
                  _obj(oid=3, cls=3, ext=(4.2, 1.8, 1.5))])
    d = SD.build_scene_data(snap, frame_idx=0, n_frames=1)
    assert [o["id"] for o in d["objects"]] == [3]
    assert d["groundTracks"] == 2
    assert SD.scene_summary(d)["ground-class tracks (not boxed)"] == "2"


@pytest.mark.parametrize("cls", SD.SCENE_OBJECT_CLASSES)
def test_every_physical_class_does_get_a_box(cls):
    snap = _Snap(_cells([_centre(5, 5, 1) + (1,)]), [_obj(cls=cls)])
    assert len(SD.build_scene_data(snap, frame_idx=0, n_frames=1)
               ["objects"]) == 1
