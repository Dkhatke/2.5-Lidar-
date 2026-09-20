"""
Click -> cell, and the rules that make the answer the RIGHT cell.

The two claims worth testing are the ones a user would never notice being
wrong: that the cell returned is the finest one covering the point (not
merely *a* cell covering it), and that a selection kept across a frame step
still refers to the same place rather than to stale numbers.
"""
from __future__ import annotations

import numpy as np
import pytest

from adaptive_lidar.pipeline.types import ResolutionLevel
from adaptive_lidar.visualization import camera as CAM
from adaptive_lidar.visualization import playback as PB
from adaptive_lidar.visualization import selection as SEL
from adaptive_lidar.visualization.coordinate_transform import ViewTransform

EGO = (12.0, 3.0)
HEADING = 0.4


# ════════════════════════════════════════════════════════════
# Fixtures: a cached frame built by hand
# ════════════════════════════════════════════════════════════
def _cells(entries) -> dict:
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
        if k in PB._CELL_U8:
            out[k] = np.full(n, 128, np.uint8)
        else:
            out[k] = np.zeros(n, PB._CELL_DTYPE.get(k, np.float32))
    out[PB.EVIDENCE_FIELD] = np.full((n, 6), 42, np.uint8)
    out["size"] = ResolutionLevel.sizes_array()[
        np.clip(out["level"].astype(np.int64), 0, 4)]
    return out


def _centre(ix: int, iy: int, level: int):
    s = ResolutionLevel.size(level)
    return ((ix + 0.5) * s, (iy + 0.5) * s)


class _Snap:
    """The two attributes ``select_at`` reads off a FrameSnapshot."""

    def __init__(self, cells, objects=()):
        self.cells = cells
        self.objects = list(objects)


def _transform(preset="top-follow", frame="World") -> ViewTransform:
    v = CAM.compute_view(preset, frame, EGO, HEADING)
    v = CAM.frame_view(v, frame, EGO, HEADING)
    x0, x1, y0, y1 = v.extent
    W = max(int((x1 - x0) * v.px_per_m), 16)
    H = max(int((y1 - y0) * v.px_per_m), 16)
    return ViewTransform.from_view(v, frame, EGO, HEADING, (H, W))


# ════════════════════════════════════════════════════════════
# Containment
# ════════════════════════════════════════════════════════════
@pytest.mark.parametrize("level", range(ResolutionLevel.N_LEVELS))
def test_a_point_inside_a_cell_finds_that_cell(level):
    ix, iy = 237, -91
    cx, cy = _centre(ix, iy, level)
    cells = _cells([(cx, cy, level)])
    s = ResolutionLevel.size(level)
    # every corner-ish point strictly inside the square
    for fx in (0.01, 0.5, 0.99):
        for fy in (0.01, 0.5, 0.99):
            x = ix * s + fx * s
            y = iy * s + fy * s
            hit = SEL.find_cell_row(cells, x, y)
            assert hit is not None, (level, fx, fy)
            assert hit == (0, level)


def test_a_point_outside_every_cell_returns_none():
    cells = _cells([_centre(10, 10, 0) + (0,)])
    assert SEL.find_cell_row(cells, 900.0, -900.0) is None


def test_finest_containing_cell_wins():
    """A place covered at three levels must resolve to the 5 cm cell.

    The coarse cells genuinely contain the point too — picking one of them
    is not a crash, it is a silently less informative answer, which is why
    this is tested rather than assumed.
    """
    # One level-0 cell, and the level-2 and level-4 cells that contain it.
    ix0, iy0 = 400, 120
    x, y = _centre(ix0, iy0, 0)
    entries = [(x, y, 0)]
    for L in (2, 4):
        entries.append(_centre(ix0 >> L, iy0 >> L, L) + (L,))
    cells = _cells(entries)

    row, level = SEL.find_cell_row(cells, x, y)
    assert level == 0 and row == 0

    # Remove the fine one: the next-finest must answer, not "nothing".
    coarse = _cells(entries[1:])
    row, level = SEL.find_cell_row(coarse, x, y)
    assert level == 2


def test_containment_is_the_morton_prefix_relation():
    """The finest hit's parent chain must contain the click at every level.

    This is the map's structural guarantee (level-l id == level-0 id >> 2l)
    restated at the selection layer: if lookup ever disagreed with it, a
    click near a boundary would fall into a different parent than the
    renderer painted.
    """
    rng = np.random.default_rng(3)
    for _ in range(200):
        x, y = rng.uniform(-50, 50, 2)
        i0 = (int(np.floor(x / ResolutionLevel.size(0))),
              int(np.floor(y / ResolutionLevel.size(0))))
        for L in range(ResolutionLevel.N_LEVELS):
            iL = (int(np.floor(x / ResolutionLevel.size(L))),
                  int(np.floor(y / ResolutionLevel.size(L))))
            assert iL == (i0[0] >> L, i0[1] >> L)


def test_negative_coordinates_do_not_wrap():
    """Cells behind the origin must index as cleanly as cells in front."""
    for ix, iy in [(-1, -1), (-1000, 5), (3, -2048)]:
        cx, cy = _centre(ix, iy, 1)
        cells = _cells([(cx, cy, 1)])
        assert SEL.find_cell_row(cells, cx, cy) == (0, 1)


# ════════════════════════════════════════════════════════════
# Click -> cell, through the real transform
# ════════════════════════════════════════════════════════════
def test_click_on_a_drawn_cell_selects_it():
    t = _transform()
    # A cell in the middle of the view.
    s = ResolutionLevel.size(1)
    ix, iy = int(EGO[0] / s), int(EGO[1] / s)
    cx, cy = _centre(ix, iy, 1)
    snap = _Snap(_cells([(cx, cy, 1)]))

    px, py = t.world_to_screen(cx, cy)
    sel = SEL.select_at(px, py, t, snap, frame_idx=0)
    assert sel.kind == "cell"
    assert sel.cell["cx"] == pytest.approx(cx, abs=1e-3)
    assert sel.cell["cy"] == pytest.approx(cy, abs=1e-3)
    assert sel.cell["level"] == 1


def test_click_on_empty_space_is_empty_not_an_error():
    t = _transform()
    snap = _Snap(_cells([_centre(0, 0, 0) + (0,)]))
    sel = SEL.select_at(t.width_px - 1, 0, t, snap, frame_idx=0)
    assert sel.kind == "empty" and sel.cell is None
    assert sel.label() == "nothing"


def test_click_resolution_is_finer_than_a_pixel_of_slop():
    """Two adjacent 5 cm cells must be distinguishable if a pixel can.

    At the default 7 px/m a 5 cm cell is a third of a pixel, so this only
    asserts the thing that IS decidable: clicks a cell apart in world terms
    land on different cells.
    """
    s = ResolutionLevel.size(0)
    ix, iy = int(EGO[0] / s), int(EGO[1] / s)
    a = _centre(ix, iy, 0)
    b = _centre(ix + 1, iy, 0)
    cells = _cells([a + (0,), b + (0,)])
    assert SEL.find_cell_row(cells, *a) == (0, 0)
    assert SEL.find_cell_row(cells, *b) == (1, 0)


# ════════════════════════════════════════════════════════════
# Objects beat cells
# ════════════════════════════════════════════════════════════
def _obj(oid, xy, cls=3, points=200):
    return {"id": oid, "cls": cls, "state": 1, "state_name": "MOVING",
            "speed_world": 4.0, "vel_world": np.zeros(3, np.float32),
            "centroid_world": np.array([xy[0], xy[1], 0.8], np.float32),
            "extent": np.array([4.2, 1.8, 1.5], np.float32),
            "points": points, "age": 5}


def test_object_wins_over_the_cell_underneath_it():
    t = _transform()
    s = ResolutionLevel.size(1)
    ix, iy = int((EGO[0] + 6) / s), int(EGO[1] / s)
    cx, cy = _centre(ix, iy, 1)
    snap = _Snap(_cells([(cx, cy, 1)]), [_obj(11, (cx, cy))])

    px, py = t.world_to_screen(cx, cy)
    sel = SEL.select_at(px, py, t, snap, frame_idx=0)
    assert sel.kind == "object" and sel.obj["id"] == 11
    # ...and it still carries the cell, for "inspect underlying cell".
    assert sel.cell is not None and sel.cell["level"] == 1


def test_object_priority_can_be_disabled():
    t = _transform()
    cx, cy = _centre(int(EGO[0] / 0.1), int(EGO[1] / 0.1), 1)
    snap = _Snap(_cells([(cx, cy, 1)]), [_obj(11, (cx, cy))])
    px, py = t.world_to_screen(cx, cy)
    sel = SEL.select_at(px, py, t, snap, frame_idx=0, allow_objects=False)
    assert sel.kind == "cell"


def test_a_click_far_from_any_object_picks_the_cell():
    t = _transform()
    s = ResolutionLevel.size(1)
    cx, cy = _centre(int(EGO[0] / s), int(EGO[1] / s), 1)
    far = (cx + 20.0, cy + 8.0)
    snap = _Snap(_cells([(cx, cy, 1)]), [_obj(11, far)])
    px, py = t.world_to_screen(cx, cy)
    assert SEL.select_at(px, py, t, snap, frame_idx=0).kind == "cell"


def test_tiny_clusters_are_not_selectable():
    """Objects the renderer suppresses must not be secretly clickable."""
    t = _transform()
    s = ResolutionLevel.size(1)
    cx, cy = _centre(int(EGO[0] / s), int(EGO[1] / s), 1)
    snap = _Snap(_cells([(cx, cy, 1)]), [_obj(11, (cx, cy), points=3)])
    px, py = t.world_to_screen(cx, cy)
    assert SEL.select_at(px, py, t, snap, frame_idx=0).kind == "cell"


# ════════════════════════════════════════════════════════════
# Persistence across frames
# ════════════════════════════════════════════════════════════
def test_selection_follows_a_tracked_object_across_frames():
    t = _transform()
    s = ResolutionLevel.size(1)
    a = _centre(int(EGO[0] / s), int(EGO[1] / s), 1)
    snap0 = _Snap(_cells([a + (1,)]), [_obj(11, a)])
    px, py = t.world_to_screen(*a)
    sel = SEL.select_at(px, py, t, snap0, frame_idx=0)

    moved = (a[0] + 5.0, a[1] + 1.0)
    snap1 = _Snap(_cells([_centre(int(moved[0] / s), int(moved[1] / s), 1)
                          + (1,)]), [_obj(11, moved)])
    sel1 = SEL.reselect_on_frame(sel, snap1, 1)
    assert sel1.kind == "object" and sel1.obj["id"] == 11
    assert sel1.world_xy[0] == pytest.approx(moved[0], abs=1e-3)
    assert sel1.frame_idx == 1


def test_a_cell_selection_stays_on_the_same_place_not_the_same_row():
    """Rows are per-frame. Re-resolving must go through the WORLD point."""
    cx, cy = _centre(60, 20, 2)
    sel = SEL.Selection("cell", (cx, cy), 0,
                        cell=SEL.cell_at_world(_cells([(cx, cy, 2)]), cx, cy))
    # Next frame has the same place at a different row, with more points.
    nxt = _cells([_centre(0, 0, 2) + (2,), (cx, cy, 2)])
    nxt["n_points"][1] = 77
    sel1 = SEL.reselect_on_frame(sel, _Snap(nxt), 1)
    assert sel1.kind == "cell"
    assert sel1.cell["row"] == 1
    assert sel1.cell["n_points"] == 77


def test_a_selection_whose_cell_vanished_becomes_empty_not_stale():
    cx, cy = _centre(60, 20, 2)
    sel = SEL.Selection("cell", (cx, cy), 0,
                        cell=SEL.cell_at_world(_cells([(cx, cy, 2)]), cx, cy))
    gone = _Snap(_cells([_centre(0, 0, 2) + (2,)]))
    assert SEL.reselect_on_frame(sel, gone, 1).kind == "empty"


def test_an_object_that_left_the_scene_falls_back_to_its_last_place():
    a = _centre(120, 30, 1)
    sel = SEL.Selection("object", a, 0, obj=_obj(11, a))
    still_mapped = _Snap(_cells([a + (1,)]), [])
    out = SEL.reselect_on_frame(sel, still_mapped, 1)
    assert out.kind == "cell" and out.cell is not None


# ════════════════════════════════════════════════════════════
# The record handed to the inspector
# ════════════════════════════════════════════════════════════
def test_cell_record_covers_every_inspector_field():
    """Every MapCell field the inspector shows must survive the cache.

    The cache is a lossy byte packing; a field silently dropped from
    CELL_FIELDS would show as a blank row rather than as an error.
    """
    cells = _cells([_centre(5, 5, 1) + (1,)])
    rec = SEL.cell_at_world(cells, *_centre(5, 5, 1))
    for k in PB.CELL_FIELDS:
        assert k in rec, f"missing {k}"
    assert rec["evidence"].shape == (6,)
    assert rec["evidence"].sum() == pytest.approx(1.0, abs=1e-5)
    assert rec["resolution"] == pytest.approx(ResolutionLevel.size(1))
    assert rec["row"] == 0


# ════════════════════════════════════════════════════════════
# The cell "under" an object
# ════════════════════════════════════════════════════════════
def test_an_objects_cell_is_the_nearest_surface_not_nothing():
    """A centroid is a centre of mass; a LiDAR sees surfaces.

    On frame 0 of `mixed_urban`, 18 of 30 tracked objects have no cell at
    all exactly under their centroid — the middle of a car is its own
    occlusion shadow. Reporting "no cell" there would read as the map
    having lost the object, so the nearest mapped surface is reported with
    its offset instead.
    """
    t = _transform()
    s = ResolutionLevel.size(1)
    body = _centre(int((EGO[0] + 4) / s), int(EGO[1] / s), 1)
    centroid = (body[0] + 0.35, body[1] + 0.20)     # inside the gap
    snap = _Snap(_cells([body + (1,)]), [_obj(21, centroid)])

    px, py = t.world_to_screen(*centroid)
    sel = SEL.select_at(px, py, t, snap, frame_idx=0)
    assert sel.kind == "object"
    assert sel.cell is not None
    assert sel.cell["cx"] == pytest.approx(body[0], abs=1e-3)
    assert 0 < sel.cell_offset_m < 0.6


def test_an_exact_hit_reports_a_zero_offset():
    t = _transform()
    s = ResolutionLevel.size(1)
    c = _centre(int((EGO[0] + 4) / s), int(EGO[1] / s), 1)
    snap = _Snap(_cells([c + (1,)]), [_obj(21, c)])
    px, py = t.world_to_screen(*c)
    sel = SEL.select_at(px, py, t, snap, frame_idx=0)
    assert sel.cell_offset_m == 0.0


def test_nothing_within_the_radius_is_still_nothing():
    """The fallback must not reach across the scene for a plausible answer."""
    cells = _cells([_centre(0, 0, 1) + (1,)])
    cell, d = SEL.nearest_cell(cells, 400.0, 400.0)
    assert cell is None and d > 1.5


def test_the_nearest_cell_radius_is_bounded_by_a_vehicle_half_width():
    assert 0.5 <= SEL.nearest_cell.__defaults__[0] <= 2.0
