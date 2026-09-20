"""
Things drawn on top of the map: the vehicle, tracked objects, the swept
corridor, and the sensor's own view.

All of it is pure NumPy rasterisation into the image the map renderer
produced, so nothing here needs a plotting library or a second canvas.

The object boxes use a distinct colour AND a distinct line style per track
state, so the three states stay separable in greyscale, in a projector's
washed-out colours, and for a colour-blind viewer.  Collapsing
MOVABLE_BUT_STATIONARY into either neighbour is exactly the mistake the
`convoy` scenario is built to expose, so it gets its own treatment here too.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from adaptive_lidar.pipeline.types import TrackState

#: Colour and dash pattern per track state. The dash pattern is what carries
#: the distinction when colour does not.
STATE_STYLE = {
    TrackState.STATIC: {
        "rgb": (110, 116, 128), "dash": 0, "name": "STATIC",
        "note": "not movable — structure"},
    TrackState.MOVABLE_BUT_STATIONARY: {
        "rgb": (34, 122, 200), "dash": 6, "name": "MOVABLE_BUT_STATIONARY",
        "note": "could move, is not moving — a parked car"},
    TrackState.MOVING: {
        "rgb": (214, 40, 40), "dash": 2, "name": "MOVING",
        "note": "moving in the WORLD frame"},
}

EGO_RGB = (20, 24, 32)
EGO_NOSE_RGB = (250, 190, 40)
CORRIDOR_RGB = (255, 196, 120)
PATH_RGB = (150, 120, 220)


# ════════════════════════════════════════════════════════════
# world → pixel, honouring the view's rotation
# ════════════════════════════════════════════════════════════
def make_projector(view, frame_of_reference, ego_xy, heading):
    """Return ``f(xy) -> (px, py)`` for the active view and frame."""
    from adaptive_lidar.visualization.camera import to_frame

    x0, x1, y0, y1 = view.extent
    ppm = view.px_per_m
    cx, cy = view.centre
    rot = view.rotation
    c, s = np.cos(rot), np.sin(rot)

    def project(xy: np.ndarray):
        xy = np.asarray(xy, np.float32).reshape(-1, 2)
        if len(xy) == 0:
            return np.zeros(0, np.int32), np.zeros(0, np.int32)
        q = to_frame(xy, frame_of_reference, ego_xy, heading)
        if abs(rot) > 1e-6:
            d = q - np.array([cx, cy], np.float32)
            q = np.stack([d[:, 0] * c - d[:, 1] * s,
                          d[:, 0] * s + d[:, 1] * c], axis=1) \
                + np.array([cx, cy], np.float32)
        return (((q[:, 0] - x0) * ppm).astype(np.int32),
                ((q[:, 1] - y0) * ppm).astype(np.int32))

    return project


def _plot(img, px, py, rgb, size=1):
    H, W = img.shape[:2]
    for dy in range(-(size // 2), size // 2 + 1):
        for dx in range(-(size // 2), size // 2 + 1):
            gx, gy = px + dx, py + dy
            ok = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
            if np.any(ok):
                img[gy[ok], gx[ok]] = rgb


def _line(img, p0, p1, rgb, width=1, dash=0):
    """Bresenham-free line: sample it densely enough that gaps cannot appear."""
    n = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]))) + 1
    if n <= 1:
        return
    t = np.linspace(0.0, 1.0, n)
    px = (p0[0] + (p1[0] - p0[0]) * t).astype(np.int32)
    py = (p0[1] + (p1[1] - p0[1]) * t).astype(np.int32)
    if dash:
        keep = (np.arange(n) // dash) % 2 == 0
        px, py = px[keep], py[keep]
    _plot(img, px, py, rgb, width)


def _polyline(img, pts, rgb, width=1, dash=0, close=True):
    m = len(pts)
    rng = range(m) if close else range(m - 1)
    for i in rng:
        _line(img, pts[i], pts[(i + 1) % m], rgb, width, dash)


def _rect_corners(cx, cy, lx, ly, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    hx, hy = lx / 2.0, ly / 2.0
    local = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], np.float32)
    return np.stack([cx + local[:, 0] * c - local[:, 1] * s,
                     cy + local[:, 0] * s + local[:, 1] * c], axis=1)


# ════════════════════════════════════════════════════════════
# I — the ego vehicle
# ════════════════════════════════════════════════════════════
def draw_ego(img, project, ego_xy, heading, length=4.6, width=2.0):
    """A box at the sensor pose with a forward indicator.

    Nobody needs a car model; they need to know where the sensor is and which
    way it is pointing.
    """
    corners = _rect_corners(ego_xy[0], ego_xy[1], length, width, heading)
    px, py = project(corners)
    pts = list(zip(px.tolist(), py.tolist()))
    if len(pts) == 4:
        _polyline(img, pts, EGO_RGB, width=2)
        # Forward indicator: a spike from the centre out through the nose.
        nose = np.array([[ego_xy[0] + (length * 0.95) * np.cos(heading),
                          ego_xy[1] + (length * 0.95) * np.sin(heading)],
                         [ego_xy[0], ego_xy[1]]], np.float32)
        nx, ny = project(nose)
        _line(img, (nx[1], ny[1]), (nx[0], ny[0]), EGO_NOSE_RGB, width=2)
        _plot(img, np.array([nx[0]]), np.array([ny[0]]), EGO_NOSE_RGB, 5)
    return img


# ════════════════════════════════════════════════════════════
# I — tracked objects
# ════════════════════════════════════════════════════════════
def draw_objects(img, project, objects, *, show_velocity=True,
                 show_prediction=False, predict_s=2.0, min_points=12):
    """Wireframe boxes coloured AND dashed by track state.

    Returns the objects actually drawn, so the caller can list exactly what
    is on screen rather than a different set.
    """
    drawn = []
    for o in objects:
        if o["points"] < min_points:
            continue
        c = o["centroid_world"]
        ext = o["extent"]
        lx, ly = float(max(ext[0], 0.6)), float(max(ext[1], 0.6))
        v = o["vel_world"]
        yaw = float(np.arctan2(v[1], v[0])) if o["speed_world"] > 0.4 else 0.0

        style = STATE_STYLE[o["state"]]
        px, py = project(_rect_corners(c[0], c[1], lx, ly, yaw))
        pts = list(zip(px.tolist(), py.tolist()))
        if len(pts) == 4:
            _polyline(img, pts, style["rgb"], width=2, dash=style["dash"])

        if show_velocity and o["speed_world"] > 0.4:
            # Arrow length proportional to speed, 1 m per m/s.
            tip = np.array([[c[0] + v[0], c[1] + v[1]]], np.float32)
            ax, ay = project(np.array([[c[0], c[1]]], np.float32))
            bx, by = project(tip)
            _line(img, (ax[0], ay[0]), (bx[0], by[0]), style["rgb"], width=2)
            _plot(img, bx, by, style["rgb"], 5)

        if show_prediction and o["state"] == TrackState.MOVING \
                and o["speed_world"] > 0.4:
            end = np.array([[c[0] + v[0] * predict_s,
                             c[1] + v[1] * predict_s]], np.float32)
            ax, ay = project(np.array([[c[0], c[1]]], np.float32))
            ex, ey = project(end)
            _line(img, (ax[0], ay[0]), (ex[0], ey[0]), PATH_RGB,
                  width=1, dash=3)
        drawn.append(o)
    return img, drawn


def draw_overlay_points(img, project, pts, rgb=(235, 30, 70), radius=1):
    """The dynamic overlay — moving points, rebuilt every frame.

    Goes through the projector rather than a raw extent mapping, so it lands
    correctly under a rotated camera and in the vehicle frame. The version in
    render.py assumes an axis-aligned world window and cannot do either.
    """
    if pts is None or len(pts) == 0:
        return img
    px, py = project(np.asarray(pts, np.float32)[:, :2])
    _plot(img, px, py, rgb, 1 + 2 * radius)
    return img


# ════════════════════════════════════════════════════════════
# Selection highlight
# ════════════════════════════════════════════════════════════
SELECT_RGB = (255, 214, 10)
SELECT_DARK = (40, 34, 0)


def draw_cell_highlight(img, project, cx, cy, size, *, rgb=SELECT_RGB,
                        min_px=9):
    """Outline the selected cell, at its TRUE size where that is visible.

    A 5 cm cell at 7 px/m is a third of a pixel, so an outline drawn at true
    size would be invisible and the user would think the click missed. Below
    ``min_px`` the marker is grown to a fixed size and the inspector states
    the real dimension — enlarging the drawn box is a readability choice, and
    the number next to it stays honest.
    """
    half = float(size) / 2.0
    corners = np.array([[cx - half, cy - half], [cx + half, cy - half],
                        [cx + half, cy + half], [cx - half, cy + half]],
                       np.float32)
    px, py = project(corners)
    w = max(px.max() - px.min(), py.max() - py.min())
    if w < min_px:
        c0x, c0y = project(np.array([[cx, cy]], np.float32))
        h = min_px // 2
        px = np.array([c0x[0] - h, c0x[0] + h, c0x[0] + h, c0x[0] - h])
        py = np.array([c0y[0] - h, c0y[0] - h, c0y[0] + h, c0y[0] + h])
    pts = list(zip(px.tolist(), py.tolist()))
    # Dark underlay first, so the marker reads on both pale and bright cells.
    _polyline(img, [(a + 1, b + 1) for a, b in pts], SELECT_DARK, width=3)
    _polyline(img, pts, rgb, width=2)
    return img


def draw_crosshair(img, project, x, y, *, rgb=SELECT_RGB, arm=11, gap=4):
    """Where the click landed, as distinct from what it selected.

    Worth drawing separately: when a click near an object selects the object
    rather than the cell under the cursor, seeing both marks is what makes
    the hit-priority rule legible instead of surprising.
    """
    cx, cy = project(np.array([[x, y]], np.float32))
    cx, cy = int(cx[0]), int(cy[0])
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        _line(img, (cx + dx * gap, cy + dy * gap),
              (cx + dx * arm, cy + dy * arm), rgb, width=1)
    return img


def draw_object_highlight(img, project, obj, *, rgb=SELECT_RGB):
    """A halo around the selected object's box."""
    c = obj["centroid_world"]
    ext = obj["extent"]
    lx = float(max(ext[0], 0.6)) + 0.9
    ly = float(max(ext[1], 0.6)) + 0.9
    v = obj["vel_world"]
    yaw = float(np.arctan2(v[1], v[0])) if obj["speed_world"] > 0.4 else 0.0
    px, py = project(_rect_corners(c[0], c[1], lx, ly, yaw))
    pts = list(zip(px.tolist(), py.tolist()))
    if len(pts) == 4:
        _polyline(img, pts, SELECT_DARK, width=3)
        _polyline(img, pts, rgb, width=2, dash=4)
    return img


# ════════════════════════════════════════════════════════════
# D — the swept corridor
# ════════════════════════════════════════════════════════════
def draw_corridor(img, project, corridor, upto_frame, alpha=0.45):
    """Every location ever resolved at level 0 or 1, blended over the map.

    This cannot come from the map: the sliding window has already evicted the
    cells behind the vehicle. It comes from the separate accumulator in
    ``playback.py``, which is why it survives eviction.
    """
    if corridor is None:
        return img, 0.0
    xy = corridor.cells_upto(upto_frame)
    if len(xy) == 0:
        return img, 0.0
    px, py = project(xy)
    H, W = img.shape[:2]
    ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    if not np.any(ok):
        return img, corridor.area_upto(upto_frame)
    gx, gy = px[ok], py[ok]
    base = img[gy, gx].astype(np.float32)
    img[gy, gx] = (base * (1 - alpha)
                   + np.array(CORRIDOR_RGB, np.float32) * alpha).astype(np.uint8)
    return img, corridor.area_upto(upto_frame)


# ════════════════════════════════════════════════════════════
# C — cell-age tint
# ════════════════════════════════════════════════════════════
def apply_age_tint(colours: np.ndarray, last_seen: np.ndarray,
                   frame_idx: int, window: int = 3,
                   boost: float = 0.45) -> np.ndarray:
    """Brighten cells refined within the last ``window`` frames.

    Without it the refinement front is inferable but not visible: every fine
    cell looks the same whether it was resolved this frame or twenty frames
    ago. With it, the leading edge of the island is a bright rim that moves
    with the vehicle.
    """
    if len(colours) == 0:
        return colours
    age = (int(frame_idx) - np.asarray(last_seen, np.int32)) % 65536
    fresh = np.clip(1.0 - age / max(window, 1), 0.0, 1.0)[:, None]
    out = colours.astype(np.float32)
    out = out + (255.0 - out) * fresh * boost
    return np.clip(out, 0, 255).astype(np.uint8)


# ════════════════════════════════════════════════════════════
# B — the sensor's own view
# ════════════════════════════════════════════════════════════
def render_range_image(snapshot_points: Optional[np.ndarray],
                       ri_index: Optional[np.ndarray],
                       ri_valid: Optional[np.ndarray],
                       ri_range: Optional[np.ndarray],
                       max_range: float = 80.0) -> np.ndarray:
    """The range image as an image — the sensor's actual view.

    There is no 3D scene view in this repository, so "first person from the
    LiDAR origin" is served by the projection the sensor really produces and
    every geometric feature is really computed on. That is a more honest
    answer than a synthesised perspective render would be.
    """
    if ri_range is None:
        return np.full((64, 512, 3), 240, np.uint8)
    r = np.asarray(ri_range, np.float32)
    v = np.asarray(ri_valid, bool) if ri_valid is not None else r > 0
    t = np.clip(r / max_range, 0.0, 1.0)
    # Near = warm, far = cool, empty = pale.
    img = np.zeros(r.shape + (3,), np.uint8)
    img[..., 0] = (255 * (1.0 - t)).astype(np.uint8)
    img[..., 1] = (140 * (1.0 - np.abs(t - 0.5) * 2)).astype(np.uint8)
    img[..., 2] = (255 * t).astype(np.uint8)
    img[~v] = (238, 240, 244)
    return np.flipud(img)


def legend_states_html() -> str:
    rows = []
    for st in (TrackState.MOVING, TrackState.MOVABLE_BUT_STATIONARY,
               TrackState.STATIC):
        d = STATE_STYLE[st]
        dash = ("solid" if d["dash"] == 0
                else ("dashed" if d["dash"] > 4 else "dotted"))
        rows.append(
            f'<span style="display:inline-flex;align-items:center;'
            f'margin-right:16px;font-size:12px">'
            f'<span style="width:22px;height:0;border-top:3px {dash} '
            f'rgb{d["rgb"]};margin-right:6px"></span>'
            f'<b>{d["name"]}</b>&nbsp;<span style="opacity:.65">'
            f'{d["note"]}</span></span>')
    return "<div>" + "".join(rows) + "</div>"
